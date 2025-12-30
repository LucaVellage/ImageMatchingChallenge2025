from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def _knn_pairs(emb: np.ndarray, *, k: int, mutual: bool) -> list[tuple[int, int, float]]:
    emb = _normalize_rows(emb)
    n = int(emb.shape[0])
    if n <= 1:
        return []
    kk = int(min(k + 1, n))
    nn = NearestNeighbors(n_neighbors=kk, metric="cosine", algorithm="auto")
    nn.fit(emb)
    dist, ind = nn.kneighbors(emb, return_distance=True)
    neigh = ind[:, 1:]
    d = dist[:, 1:]
    sim = 1.0 - d

    directed: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j, s in zip(neigh[i].tolist(), sim[i].tolist(), strict=False):
            if i == int(j):
                continue
            directed[(int(i), int(j))] = float(s)

    edges: list[tuple[int, int, float]] = []
    if mutual:
        for (i, j), w in directed.items():
            if (j, i) in directed and i < j:
                edges.append((i, j, max(float(w), float(directed[(j, i)]))))
        edges.sort()
        return edges

    for (i, j), w in directed.items():
        a, b = (i, j) if i < j else (j, i)
        edges.append((a, b, float(w)))
    edges.sort()
    return edges

def _knn_pairs_scored(
    emb: np.ndarray,
    *,
    k: int,
    scorer: str,
    model,
    topm: int | None,
) -> list[tuple[int, int, float]]:
    emb = _normalize_rows(emb)
    n = emb.shape[0]
    if n <= 1:
        return []

    kk = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=kk, metric="cosine")
    nn.fit(emb)
    _, ind = nn.kneighbors(emb)
    neigh = ind[:, 1:]

    edges = []
    for i in range(n):
        js = neigh[i]
        zi = emb[i]
        zj = emb[js]

        if scorer == "mlp":
            with torch.no_grad():
                scores = model(
                    torch.from_numpy(np.repeat(zi[None, :], len(js), axis=0)),
                    torch.from_numpy(zj),
                ).numpy()

        elif scorer == "gbdt":
            cos = (zi * zj).sum(axis=1)
            X = np.stack([cos], axis=1)
            scores = model.predict_proba(X)[:, 1]

        else:
            raise ValueError(scorer)

        order = np.argsort(-scores)
        if topm is not None:
            order = order[: int(topm)]

        for o in order:
            j = int(js[o])
            if i < j:
                edges.append((i, j, float(scores[o])))

    edges.sort()
    return edges


def _cluster_graph(
    image_ids: list[str],
    edges: list[tuple[int, int, float]],
    *,
    min_sim: float,
    method: str,
    seed: int,
) -> list[list[str]]:
    g = nx.Graph()
    g.add_nodes_from(image_ids)
    for i, j, w in edges:
        if float(w) >= float(min_sim):
            g.add_edge(image_ids[int(i)], image_ids[int(j)], weight=float(w))

    if g.number_of_edges() == 0:
        return [[n] for n in image_ids]

    method = method.lower().strip()
    if method == "louvain":
        comms = list(nx.algorithms.community.louvain_communities(g, weight="weight", seed=int(seed)))
    elif method in {"greedy", "greedy_modularity"}:
        comms = list(nx.algorithms.community.greedy_modularity_communities(g, weight="weight"))
    elif method in {"components", "cc"}:
        comms = [set(c) for c in nx.connected_components(g)]
    else:
        raise ValueError(f"Unknown method: {method}")

    comms = [sorted([str(x) for x in c]) for c in comms]
    comms.sort(key=lambda c: (-len(c), c[0] if c else ""))
    return comms


def _label_clusters(dataset: str, image_ids: list[str], clusters: list[list[str]], *, min_cluster_size: int) -> pd.DataFrame:
    scene_of: dict[str, str] = {}
    idx = 1
    for c in clusters:
        if len(c) < int(min_cluster_size):
            continue
        scene = f"cluster_{idx:04d}"
        idx += 1
        for im in c:
            scene_of[str(im)] = scene
    for im in image_ids:
        scene_of.setdefault(str(im), "outliers")
    df = pd.DataFrame({"dataset": dataset, "image_id": image_ids})
    df["scene"] = df["image_id"].map(scene_of)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster images per dataset from a retrieval embeddings cache")
    parser.add_argument("--cache-root", type=Path, default=Path("cache_retrieval"))
    parser.add_argument("--out-csv", type=Path, default=Path("cache/clusters_retrieval.csv"))
    parser.add_argument("--dataset", action="append", default=None)

    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--mutual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-sim", type=float, default=0.35)
    parser.add_argument("--method", default="louvain", choices=["louvain", "greedy", "components"])
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge-scorer", default="cosine", choices=["cosine", "mlp", "gbdt"], help="Edge scoring method for candidate pairs.")
    parser.add_argument("--edge-model", type=Path, default=None, help="Path to trained edge model (required for mlp/gbdt).")
    parser.add_argument("--edge-topm", type=int, default=None, help="Keep only top-M neighbors per image after scoring.")
    args = parser.parse_args()


    #Load trained MLP Model for edge scoring
    edge_scorer = args.edge_scorer.lower()
    edge_model = None

    if edge_scorer in {"mlp", "gbdt"}:
        if args.edge_model is None:
            raise SystemExit(f"--edge-scorer {edge_scorer} requires --edge-model")

        if edge_scorer == "mlp":
            import torch
            from imc25.models.edge_mlp import EdgeMLP

            ckpt = torch.load(args.edge_model, map_location="cpu")
            edge_model = EdgeMLP(ckpt["dim"])
            edge_model.load_state_dict(ckpt["state_dict"])
            edge_model.eval()  # inference mode only

        
        elif edge_scorer == "gbdt":
            import joblib
            edge_model = joblib.load(args.edge_model)


    cache_root = args.cache_root
    if not cache_root.exists():
        raise SystemExit(f"cache-root not found: {cache_root}")

    datasets_filter = set(args.dataset) if args.dataset else None
    out_rows: list[pd.DataFrame] = []
    for ds_dir in sorted([p for p in cache_root.iterdir() if p.is_dir() and (p / "meta.json").exists()]):
        dataset = ds_dir.name
        if datasets_filter and dataset not in datasets_filter:
            continue
        emb_path = ds_dir / "embeddings.npy"
        meta_path = ds_dir / "meta.json"
        if not emb_path.exists() or not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        image_ids = [str(x) for x in meta.get("image_ids", [])]
        emb = np.load(emb_path)
        print("Inference embeddings shape:", emb.shape)

        if len(image_ids) != int(emb.shape[0]):
            raise ValueError(f"{dataset}: meta/image_ids mismatch embeddings rows")

        if edge_scorer == "cosine":
            edges = _knn_pairs(emb, k=int(args.topk), mutual=bool(args.mutual))
        else:
            edges = _knn_pairs_scored(
                emb,
                k=int(args.topk),
                scorer=edge_scorer,
                model=edge_model,
                topm=args.edge_topm,
            )

        clusters = _cluster_graph(image_ids, edges, min_sim=float(args.min_sim), method=str(args.method), seed=int(args.seed))
        df = _label_clusters(dataset, image_ids, clusters, min_cluster_size=int(args.min_cluster_size))
        out_rows.append(df)
        n_clusters = int(df.loc[df["scene"] != "outliers", "scene"].nunique())
        n_out = int((df["scene"] == "outliers").sum())
        print(f"[ok] {dataset}: clusters={n_clusters} outliers={n_out} images={len(df)}")

    if not out_rows:
        raise SystemExit("No datasets found in cache-root (missing meta.json/embeddings.npy)")

    out = pd.concat(out_rows, ignore_index=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[ok] wrote {args.out_csv} rows={len(out)}")


if __name__ == "__main__":
    main()
