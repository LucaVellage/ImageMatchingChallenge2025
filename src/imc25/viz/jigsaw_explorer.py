"""Dash app to visualize IMC clustering as a “jigsaw puzzle” graph."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

try:
    from dash import Dash, Input, Output, State, dcc, html
except Exception:  # pragma: no cover
    Dash = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    nodes: pd.DataFrame
    edges_embed: pd.DataFrame
    edges_geom: pd.DataFrame | None
    layout_pos: dict[str, tuple[float, float]]
    communities: dict[str, str]
    stats: dict[str, Any]


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def _safe_read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def list_datasets(cache_root: Path) -> list[str]:
    out: list[str] = []
    for p in cache_root.iterdir():
        if p.is_dir() and (p / "meta.json").exists():
            out.append(p.name)
    return sorted(out)


def load_clusters_map(clusters_csv: Path | None, dataset: str) -> dict[str, str]:
    if clusters_csv is None or not clusters_csv.exists():
        return {}
    df = pd.read_csv(clusters_csv)
    if not {"dataset", "image_id", "scene"}.issubset(df.columns):
        return {}
    sub = df[df["dataset"] == dataset]
    return dict(zip(sub["image_id"].astype(str), sub["scene"].astype(str), strict=False))


def load_meta(cache_dir: Path) -> tuple[list[str], list[Path]]:
    meta = _safe_read_json(cache_dir / "meta.json")
    image_ids = [str(x) for x in meta.get("image_ids", [])]
    paths = [Path(str(x)) for x in meta.get("paths", [])]
    if len(image_ids) != len(paths):
        raise ValueError(f"Bad meta.json in {cache_dir}: image_ids and paths length mismatch")
    return image_ids, paths


def load_embedding_edges(cache_dir: Path, image_ids: list[str], embeddings: np.ndarray) -> pd.DataFrame:
    pairs_path = cache_dir / "pairs_topk_K30.npz"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Missing pairs file: {pairs_path}")
    pairs = np.load(pairs_path)
    a = pairs["a"].astype(np.int64, copy=False)
    b = pairs["b"].astype(np.int64, copy=False)
    if a.shape != b.shape:
        raise ValueError(f"Bad pairs file: {pairs_path}")

    emb = _normalize_rows(embeddings.astype(np.float32, copy=False))
    sim = (emb[a] * emb[b]).sum(axis=1)
    src = [image_ids[i] for i in a.tolist()]
    dst = [image_ids[i] for i in b.tolist()]

    df = pd.DataFrame({"src": src, "dst": dst, "similarity": sim.astype(np.float32)})
    # Deduplicate undirected edges.
    u = np.minimum(df["src"].to_numpy(), df["dst"].to_numpy())
    v = np.maximum(df["src"].to_numpy(), df["dst"].to_numpy())
    df["u"] = u
    df["v"] = v
    df = df.groupby(["u", "v"], as_index=False)["similarity"].max()
    df = df.rename(columns={"u": "src", "v": "dst"})
    return df


def load_geometric_edges(cache_dir: Path, image_ids: list[str]) -> pd.DataFrame | None:
    stats_path = cache_dir / "pair_stats.csv"
    if not stats_path.exists():
        return None
    df = pd.read_csv(stats_path)
    required = {"i", "j", "num_inliers", "inlier_ratio"}
    if not required.issubset(df.columns):
        return None
    src = [image_ids[int(i)] for i in df["i"].tolist()]
    dst = [image_ids[int(j)] for j in df["j"].tolist()]
    out = df.copy()
    out.insert(0, "src", src)
    out.insert(1, "dst", dst)
    return out


def compute_communities(
    nodes: list[str],
    edges: pd.DataFrame,
    *,
    weight_col: str,
    min_weight: float,
) -> dict[str, str]:
    g = nx.Graph()
    g.add_nodes_from(nodes)
    for row in edges.itertuples(index=False):
        w = float(getattr(row, weight_col))
        if w >= min_weight:
            g.add_edge(row.src, row.dst, weight=w)

    if g.number_of_edges() == 0:
        return {n: "singleton" for n in nodes}

    comms = list(nx.algorithms.community.greedy_modularity_communities(g, weight="weight"))
    comms = sorted(comms, key=len, reverse=True)
    mapping: dict[str, str] = {}
    for idx, c in enumerate(comms, start=1):
        label = f"comm_{idx:04d}"
        for n in c:
            mapping[str(n)] = label
    for n in nodes:
        mapping.setdefault(n, "singleton")
    return mapping


def compute_layout(
    nodes: list[str],
    edges: pd.DataFrame,
    *,
    weight_col: str,
    min_weight: float,
    seed: int = 0,
    iterations: int = 80,
) -> dict[str, tuple[float, float]]:
    g = nx.Graph()
    g.add_nodes_from(nodes)
    for row in edges.itertuples(index=False):
        w = float(getattr(row, weight_col))
        if w >= min_weight:
            g.add_edge(row.src, row.dst, weight=w)

    if g.number_of_edges() == 0:
        pos = nx.circular_layout(g, scale=1.0)
    else:
        pos = nx.spring_layout(g, seed=seed, weight="weight", iterations=iterations)
    return {str(k): (float(v[0]), float(v[1])) for k, v in pos.items()}


def _palette(n: int) -> list[str]:
    base = [
        "#4CC9F0",
        "#F72585",
        "#B5179E",
        "#7209B7",
        "#3A0CA3",
        "#4361EE",
        "#4895EF",
        "#560BAD",
        "#F77F00",
        "#F9C74F",
        "#90BE6D",
        "#43AA8B",
    ]
    if n <= len(base):
        return base[:n]
    out = []
    for i in range(n):
        out.append(base[i % len(base)])
    return out


def _thumbnail_data_url(path: Path, *, max_side: int = 420) -> str | None:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = max(w, h) / float(max_side)
            if scale > 1.0:
                im = im.resize((int(w / scale), int(h / scale)), resample=Image.BICUBIC)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=86, optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None


@lru_cache(maxsize=512)
def _thumb_cached(path_str: str) -> str | None:
    return _thumbnail_data_url(Path(path_str))


def _node_strength(nodes: list[str], edges: pd.DataFrame, weight_col: str) -> dict[str, float]:
    strength = {n: 0.0 for n in nodes}
    for row in edges.itertuples(index=False):
        w = float(getattr(row, weight_col))
        strength[str(row.src)] += w
        strength[str(row.dst)] += w
    return strength


def make_graph_figure(
    bundle: DatasetBundle,
    *,
    edge_source: str,
    min_inliers: int,
    min_ratio: float,
    min_similarity: float,
    color_mode: str,
    show_edges: bool,
    selected_node: str | None,
) -> go.Figure:
    nodes_df = bundle.nodes
    pos = bundle.layout_pos

    if edge_source == "geom" and bundle.edges_geom is not None:
        edges = bundle.edges_geom
        edges = edges[(edges["num_inliers"] >= min_inliers) & (edges["inlier_ratio"] >= min_ratio)]
        edge_weight = edges["num_inliers"].astype(float).to_numpy()
        edge_label = "geom"
    else:
        edges = bundle.edges_embed
        edges = edges[edges["similarity"] >= min_similarity]
        edge_weight = edges["similarity"].astype(float).to_numpy()
        edge_label = "embed"

    node_ids = nodes_df["image_id"].astype(str).tolist()

    if color_mode == "clusters" and "cluster" in nodes_df.columns:
        labels = nodes_df["cluster"].astype(str).tolist()
    else:
        labels = [bundle.communities.get(i, "singleton") for i in node_ids]

    uniq = sorted(set(labels))
    colors = _palette(len(uniq))
    color_map = {k: colors[i] for i, k in enumerate(uniq)}
    node_color = [color_map[l] for l in labels]

    x = np.array([pos[i][0] for i in node_ids], dtype=np.float32)
    y = np.array([pos[i][1] for i in node_ids], dtype=np.float32)

    fig_data: list[go.BaseTraceType] = []

    if show_edges and len(edges):
        xs: list[float] = []
        ys: list[float] = []
        for row, w in zip(edges.itertuples(index=False), edge_weight, strict=False):
            a = str(row.src)
            b = str(row.dst)
            if a not in pos or b not in pos:
                continue
            xs.extend([pos[a][0], pos[b][0], math.nan])
            ys.extend([pos[a][1], pos[b][1], math.nan])
        fig_data.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="rgba(255,255,255,0.18)", width=1.0),
                hoverinfo="skip",
                name=f"edges ({edge_label})",
            )
        )

    size_strength = _node_strength(node_ids, bundle.edges_embed, "similarity")
    sizes = np.array([size_strength[i] for i in node_ids], dtype=np.float32)
    if len(sizes):
        sizes = 10.0 + 18.0 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-6)
    else:
        sizes = np.full((len(node_ids),), 10.0, dtype=np.float32)

    symbols = np.array(["circle"] * len(node_ids), dtype=object)
    if "cluster" in nodes_df.columns:
        outlier_mask = nodes_df["cluster"].astype(str).str.lower().eq("outliers").to_numpy()
        symbols[outlier_mask] = "x"

    fig_data.append(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=sizes,
                color=node_color,
                symbol=symbols,
                line=dict(color="rgba(255,255,255,0.30)", width=1),
                opacity=0.95,
            ),
            text=node_ids,
            hovertext=labels,
            customdata=node_ids,
            hovertemplate="<b>%{text}</b><br>label=%{hovertext}<extra></extra>",
            name="images",
        )
    )

    if selected_node is not None and selected_node in pos:
        sx, sy = pos[selected_node]
        fig_data.append(
            go.Scatter(
                x=[sx],
                y=[sy],
                mode="markers",
                marker=dict(
                    size=28,
                    color="rgba(0,0,0,0)",
                    symbol="circle",
                    line=dict(color="rgba(255,255,255,0.95)", width=3),
                ),
                hoverinfo="skip",
                showlegend=False,
                name="selected",
            )
        )

    fig = go.Figure(data=fig_data)
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"Jigsaw Graph — {bundle.name} ({len(node_ids)} images)",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def load_bundle(cache_root: Path, dataset: str, clusters_csv: Path | None) -> DatasetBundle:
    cache_dir = cache_root / dataset
    image_ids, paths = load_meta(cache_dir)
    if not image_ids:
        raise ValueError(f"No images in {cache_dir}/meta.json")

    embeddings = np.load(cache_dir / "embeddings.npy")
    edges_embed = load_embedding_edges(cache_dir, image_ids, embeddings)
    edges_geom = load_geometric_edges(cache_dir, image_ids)

    clusters_map = load_clusters_map(clusters_csv, dataset)
    nodes = pd.DataFrame(
        {
            "image_id": image_ids,
            "path": [str(p) for p in paths],
        }
    )
    if clusters_map:
        nodes["cluster"] = [clusters_map.get(i, "unlabeled") for i in image_ids]

    communities = compute_communities(image_ids, edges_embed, weight_col="similarity", min_weight=0.28)
    layout_pos = compute_layout(image_ids, edges_embed, weight_col="similarity", min_weight=0.25, seed=0)

    stats: dict[str, Any] = {
        "num_nodes": len(image_ids),
        "num_edges_embed": int(len(edges_embed)),
        "num_edges_geom": int(len(edges_geom)) if edges_geom is not None else 0,
    }
    return DatasetBundle(
        name=dataset,
        nodes=nodes,
        edges_embed=edges_embed,
        edges_geom=edges_geom,
        layout_pos=layout_pos,
        communities=communities,
        stats=stats,
    )


def run_dash(cache_root: Path, clusters_csv: Path | None, host: str, port: int) -> None:
    if Dash is None:
        raise RuntimeError("dash is not installed; cannot start the web app")

    datasets = list_datasets(cache_root)
    if not datasets:
        raise SystemExit(f"No dataset caches found in {cache_root}")

    bundles = {d: load_bundle(cache_root, d, clusters_csv) for d in datasets}
    initial = datasets[0]

    app = Dash(__name__)
    app.title = "IMC25 Jigsaw Explorer"

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H2("IMC25: Jigsaw Explorer (Clustering + Outliers)"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Dataset"),
                                    dcc.Dropdown(
                                        id="ds",
                                        options=[{"label": d, "value": d} for d in datasets],
                                        value=initial,
                                        clearable=False,
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("Edge source"),
                                    dcc.Dropdown(
                                        id="edge-src",
                                        options=[
                                            {"label": "Embedding similarity", "value": "embed"},
                                            {"label": "Geometric inliers (if available)", "value": "geom"},
                                        ],
                                        value="geom" if bundles[initial].edges_geom is not None else "embed",
                                        clearable=False,
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("Color by"),
                                    dcc.Dropdown(
                                        id="color-mode",
                                        options=[
                                            {"label": "Clusters (cache/clusters.csv)", "value": "clusters"},
                                            {"label": "Communities (graph modularity)", "value": "communities"},
                                        ],
                                        value="clusters"
                                        if "cluster" in bundles[initial].nodes.columns
                                        else "communities",
                                        clearable=False,
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("Show edges"),
                                    dcc.Checklist(
                                        id="show-edges",
                                        options=[{"label": "on", "value": "on"}],
                                        value=["on"],
                                        style={"marginTop": "6px"},
                                    ),
                                ],
                                style={"flex": "0.7"},
                            ),
                        ],
                        style={"display": "flex", "gap": "14px", "alignItems": "end"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Min similarity"),
                                    dcc.Slider(
                                        id="min-sim",
                                        min=0.0,
                                        max=1.0,
                                        step=0.02,
                                        value=0.35,
                                        marks={0.0: "0.0", 0.3: "0.3", 0.5: "0.5", 0.7: "0.7", 0.9: "0.9"},
                                    ),
                                ],
                                style={"flex": "1"},
                                id="min-sim-wrap",
                            ),
                            html.Div(
                                [
                                    html.Label("Min inliers"),
                                    dcc.Slider(
                                        id="min-inliers",
                                        min=0,
                                        max=250,
                                        step=1,
                                        value=15,
                                        marks={0: "0", 15: "15", 50: "50", 100: "100", 200: "200"},
                                    ),
                                ],
                                style={"flex": "1"},
                                id="min-inliers-wrap",
                            ),
                            html.Div(
                                [
                                    html.Label("Min inlier ratio"),
                                    dcc.Slider(
                                        id="min-ratio",
                                        min=0.0,
                                        max=1.0,
                                        step=0.02,
                                        value=0.5,
                                        marks={0.0: "0.0", 0.5: "0.5", 0.7: "0.7", 0.9: "0.9"},
                                    ),
                                ],
                                style={"flex": "1"},
                                id="min-ratio-wrap",
                            ),
                        ],
                        style={"display": "flex", "gap": "14px", "marginTop": "10px"},
                    ),
                ],
                style={"padding": "14px 18px 10px 18px"},
            ),
            html.Div(
                [
                    html.Div(
                        [dcc.Graph(id="graph", style={"height": "80vh"})],
                        style={"flex": "2.2", "minWidth": "560px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4("Selected image"),
                                    html.Img(
                                        id="img",
                                        style={
                                            "width": "100%",
                                            "borderRadius": "10px",
                                            "border": "1px solid rgba(255,255,255,0.12)",
                                            "background": "rgba(255,255,255,0.02)",
                                        },
                                    ),
                                    html.Pre(
                                        id="meta",
                                        style={
                                            "whiteSpace": "pre-wrap",
                                            "background": "rgba(255,255,255,0.06)",
                                            "padding": "10px",
                                            "borderRadius": "10px",
                                            "border": "1px solid rgba(255,255,255,0.08)",
                                            "fontSize": "12px",
                                            "marginTop": "10px",
                                        },
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.H4("Top neighbors"),
                                    dcc.Graph(id="nbr-chart", style={"height": "240px"}),
                                    html.Div(id="nbr-thumbs", style={"display": "flex", "flexWrap": "wrap", "gap": "8px"}),
                                ],
                                style={"marginTop": "12px"},
                            ),
                        ],
                        style={"flex": "1", "minWidth": "360px", "paddingRight": "14px"},
                    ),
                ],
                style={"display": "flex", "gap": "14px", "paddingLeft": "14px"},
            ),
        ],
        style={"fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"},
    )

    @app.callback(
        Output("edge-src", "options"),
        Output("edge-src", "value"),
        Output("color-mode", "options"),
        Output("color-mode", "value"),
        Input("ds", "value"),
        State("edge-src", "value"),
        State("color-mode", "value"),
    )
    def _sync_controls(ds: str, edge_src: str, color_mode: str):
        bundle = bundles[str(ds)]

        edge_opts = [
            {"label": "Embedding similarity", "value": "embed"},
            {
                "label": "Geometric inliers (pair_stats.csv)",
                "value": "geom",
                "disabled": bundle.edges_geom is None,
            },
        ]
        if bundle.edges_geom is None and edge_src == "geom":
            edge_src = "embed"

        has_clusters = "cluster" in bundle.nodes.columns
        color_opts = [
            {"label": "Communities (graph modularity)", "value": "communities"},
            {"label": "Clusters (cache/clusters.csv)", "value": "clusters", "disabled": not has_clusters},
        ]
        if not has_clusters and color_mode == "clusters":
            color_mode = "communities"

        return edge_opts, edge_src, color_opts, color_mode

    @app.callback(
        Output("min-sim-wrap", "style"),
        Output("min-inliers-wrap", "style"),
        Output("min-ratio-wrap", "style"),
        Input("edge-src", "value"),
    )
    def _toggle_threshold_controls(edge_src: str):
        show = {"flex": "1"}
        hide = {"flex": "1", "display": "none"}
        if edge_src == "geom":
            return hide, show, show
        return show, hide, hide

    @app.callback(
        Output("graph", "figure"),
        Output("img", "src"),
        Output("meta", "children"),
        Output("nbr-chart", "figure"),
        Output("nbr-thumbs", "children"),
        Input("ds", "value"),
        Input("edge-src", "value"),
        Input("min-sim", "value"),
        Input("min-inliers", "value"),
        Input("min-ratio", "value"),
        Input("color-mode", "value"),
        Input("show-edges", "value"),
        Input("graph", "clickData"),
    )
    def _update(ds, edge_src, min_sim, min_inliers, min_ratio, color_mode, show_edges_value, click_data):
        bundle = bundles[str(ds)]
        show_edges = bool(show_edges_value and "on" in show_edges_value)

        selected = None
        if click_data and click_data.get("points"):
            selected = click_data["points"][0].get("customdata")

        fig = make_graph_figure(
            bundle,
            edge_source=str(edge_src),
            min_inliers=int(min_inliers or 0),
            min_ratio=float(min_ratio or 0.0),
            min_similarity=float(min_sim or 0.0),
            color_mode=str(color_mode),
            show_edges=show_edges,
            selected_node=selected,
        )

        img_src = None
        meta_text = "Click a node to inspect it."
        nbr_fig = go.Figure()
        nbr_fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10))
        thumbs: list[Any] = []

        if selected is None or selected not in bundle.nodes["image_id"].values:
            return fig, img_src, meta_text, nbr_fig, thumbs

        row = bundle.nodes[bundle.nodes["image_id"] == selected].iloc[0]
        path = Path(str(row["path"]))
        img_src = _thumb_cached(str(path))

        label = None
        if str(color_mode) == "clusters" and "cluster" in bundle.nodes.columns:
            label = str(row.get("cluster"))
        else:
            label = bundle.communities.get(str(selected), "singleton")

        meta_text = f"image_id: {selected}\npath: {path}\nlabel: {label}\n"

        if edge_src == "geom" and bundle.edges_geom is not None:
            e = bundle.edges_geom
            mask = (e["src"] == selected) | (e["dst"] == selected)
            nbrs = e[mask].copy()
            nbrs["other"] = np.where(nbrs["src"] == selected, nbrs["dst"], nbrs["src"])
            nbrs = nbrs.sort_values("num_inliers", ascending=False).head(10)
            nbr_fig = go.Figure(
                data=[
                    go.Bar(
                        x=nbrs["other"],
                        y=nbrs["num_inliers"],
                        marker_color="#4CC9F0",
                        hovertemplate="%{x}<br>inliers=%{y}<extra></extra>",
                    )
                ]
            )
            nbr_fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=10, b=40), xaxis_tickangle=-35)
            for other in nbrs["other"].tolist()[:8]:
                other_row = bundle.nodes[bundle.nodes["image_id"] == other]
                if len(other_row):
                    p = Path(str(other_row.iloc[0]["path"]))
                    src = _thumb_cached(str(p))
                    if src is not None:
                        thumbs.append(
                            html.Img(
                                src=src,
                                title=str(other),
                                style={
                                    "width": "48%",
                                    "borderRadius": "8px",
                                    "border": "1px solid rgba(255,255,255,0.12)",
                                },
                            )
                        )
        else:
            e = bundle.edges_embed
            mask = (e["src"] == selected) | (e["dst"] == selected)
            nbrs = e[mask].copy()
            nbrs["other"] = np.where(nbrs["src"] == selected, nbrs["dst"], nbrs["src"])
            nbrs = nbrs.sort_values("similarity", ascending=False).head(10)
            nbr_fig = go.Figure(
                data=[
                    go.Bar(
                        x=nbrs["other"],
                        y=nbrs["similarity"],
                        marker_color="#F72585",
                        hovertemplate="%{x}<br>sim=%{y:.3f}<extra></extra>",
                    )
                ]
            )
            nbr_fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=10, b=40), xaxis_tickangle=-35)
            for other in nbrs["other"].tolist()[:8]:
                other_row = bundle.nodes[bundle.nodes["image_id"] == other]
                if len(other_row):
                    p = Path(str(other_row.iloc[0]["path"]))
                    src = _thumb_cached(str(p))
                    if src is not None:
                        thumbs.append(
                            html.Img(
                                src=src,
                                title=str(other),
                                style={
                                    "width": "48%",
                                    "borderRadius": "8px",
                                    "border": "1px solid rgba(255,255,255,0.12)",
                                },
                            )
                        )

        return fig, img_src, meta_text, nbr_fig, thumbs

    # Dash 3+: run_server -> run
    if hasattr(app, "run"):
        app.run(host=host, port=port, debug=False)
    else:  # pragma: no cover
        # Older Dash versions
        app.run_server(host=host, port=port, debug=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive IMC-style jigsaw explorer (graph + clusters + outliers)")
    parser.add_argument("--cache-root", type=Path, default=Path("cache"), help="Cache root containing dataset folders")
    parser.add_argument(
        "--clusters-csv",
        type=Path,
        default=Path("cache/clusters.csv"),
        help="Optional clusters CSV (dataset,image_id,scene) to color nodes",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Dash server host")
    parser.add_argument("--port", type=int, default=8060, help="Dash server port")
    args = parser.parse_args()

    clusters_csv = args.clusters_csv if args.clusters_csv.exists() else None
    run_dash(args.cache_root, clusters_csv, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
