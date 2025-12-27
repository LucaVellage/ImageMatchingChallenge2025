"""Dash app to explore camera poses from an IMC-style `submission.csv`."""

from __future__ import annotations

import argparse
import base64
import io
import math
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from dash import Dash, Input, Output, dcc, html
except Exception:  # pragma: no cover
    Dash = None  # type: ignore[assignment]

from PIL import Image


_SEP_RE = re.compile(r"[;\s]+")


class PointCloud(NamedTuple):
    points: np.ndarray  # (N, 3)
    colors: np.ndarray | None  # (N, 3) in [0,1]


def _parse_floats(value: str, expected: int, *, label: str) -> np.ndarray:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError(f"Missing {label}")
    text = str(value).strip().strip("[]()")
    parts = [p for p in _SEP_RE.split(text) if p]
    if len(parts) != expected:
        raise ValueError(f"Expected {expected} floats for {label}, got {len(parts)}: {text!r}")
    return np.asarray([float(p) for p in parts], dtype=np.float64)


def parse_rotation_matrix(value: str) -> np.ndarray:
    mat = _parse_floats(value, 9, label="rotation_matrix").reshape(3, 3)
    return mat


def parse_translation_vector(value: str) -> np.ndarray:
    vec = _parse_floats(value, 3, label="translation_vector")
    return vec


def camera_center_world(R_cw: np.ndarray, t_cw: np.ndarray) -> np.ndarray:
    # COLMAP-style extrinsics: x_cam = R_cw * x_world + t_cw
    # Camera center in world: C = -R^T t
    return -R_cw.T @ t_cw


def camera_forward_world(R_cw: np.ndarray) -> np.ndarray:
    # Camera looks along +Z in camera coordinates -> R^T * [0,0,1] in world.
    return R_cw.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)


def load_submission(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"image_id", "dataset", "scene", "image", "rotation_matrix", "translation_vector"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    centers = []
    forwards = []
    rotations = []
    translations = []
    ok_rows = []
    for idx, row in df.iterrows():
        try:
            R = parse_rotation_matrix(row["rotation_matrix"])
            t = parse_translation_vector(row["translation_vector"])
            centers.append(camera_center_world(R, t))
            forwards.append(camera_forward_world(R))
            rotations.append(R)
            translations.append(t)
            ok_rows.append(True)
        except Exception:
            centers.append(np.array([np.nan, np.nan, np.nan], dtype=np.float64))
            forwards.append(np.array([np.nan, np.nan, np.nan], dtype=np.float64))
            rotations.append(None)
            translations.append(None)
            ok_rows.append(False)

    df = df.copy()
    centers_arr = np.stack(centers, axis=0)
    forwards_arr = np.stack(forwards, axis=0)
    df["center_x"] = centers_arr[:, 0]
    df["center_y"] = centers_arr[:, 1]
    df["center_z"] = centers_arr[:, 2]
    df["fwd_x"] = forwards_arr[:, 0]
    df["fwd_y"] = forwards_arr[:, 1]
    df["fwd_z"] = forwards_arr[:, 2]
    df["R_cw"] = rotations
    df["t_cw"] = translations
    df["pose_ok"] = ok_rows
    return df[df["pose_ok"]].reset_index(drop=True)


def index_images(images_root: Path) -> dict[str, Path]:
    if not images_root.exists():
        raise FileNotFoundError(f"images root not found: {images_root}")
    image_index: dict[str, Path] = {}
    for path in images_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            image_index.setdefault(path.name, path)
    return image_index


def load_pointcloud(path: Path, *, max_points: int = 150_000) -> PointCloud:
    try:
        import warnings

        warnings.filterwarnings(
            "ignore",
            message=r".*joblib will operate in serial mode.*",
            category=UserWarning,
        )
        import open3d as o3d  # type: ignore
    except Exception as e:
        raise RuntimeError("open3d is not installed; cannot load pointcloud") from e
    if not path.exists():
        raise FileNotFoundError(f"pointcloud not found: {path}")
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    cols = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
    if max_points and pts.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
        cols = cols[idx] if cols is not None else None
    return PointCloud(points=pts, colors=cols)


def _image_aspect(path: Path) -> float | None:
    try:
        with Image.open(path) as im:
            w, h = im.size
        if h <= 0:
            return None
        return float(w) / float(h)
    except Exception:
        return None


def _thumbnail_data_url(path: Path, *, max_side: int = 720) -> str | None:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = max(w, h) / float(max_side)
            if scale > 1.0:
                im = im.resize((int(w / scale), int(h / scale)), resample=Image.BICUBIC)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=88, optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None


def _frustum_lines_world(
    *,
    R_cw: np.ndarray,
    C_w: np.ndarray,
    aspect: float,
    fov_deg: float,
    scale: float,
) -> np.ndarray:
    depth = float(scale)
    half_w = depth * math.tan(math.radians(fov_deg) / 2.0)
    half_h = half_w / float(aspect if aspect > 0 else 1.0)

    corners_c = np.array(
        [
            [-half_w, -half_h, depth],
            [half_w, -half_h, depth],
            [half_w, half_h, depth],
            [-half_w, half_h, depth],
        ],
        dtype=np.float64,
    )
    corners_w = (R_cw.T @ corners_c.T).T + C_w[None, :]
    origin_w = C_w.reshape(1, 3)

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
    ]
    segments = []
    for c in range(4):
        segments.append(np.vstack([origin_w[0], corners_w[c]]))
    for a, b in edges:
        segments.append(np.vstack([corners_w[a], corners_w[b]]))

    # Build a polyline format for Plotly: p0, p1, None, p0, p1, None...
    lines = []
    for seg in segments:
        lines.append(seg[0])
        lines.append(seg[1])
        lines.append([np.nan, np.nan, np.nan])
    return np.asarray(lines, dtype=np.float64)


def make_pose_figure(
    df: pd.DataFrame,
    *,
    pointcloud: PointCloud | None = None,
    selected_row: int | None = None,
    show_pointcloud: bool = True,
    show_frustum: bool = True,
    frustum_scale: float = 1.0,
    fov_deg: float = 60.0,
    title: str = "Camera Pose Explorer",
) -> go.Figure:
    traces: list[go.BaseTraceType] = []

    if pointcloud is not None and show_pointcloud:
        pc = pointcloud.points
        traces.append(
            go.Scatter3d(
                x=pc[:, 0],
                y=pc[:, 1],
                z=pc[:, 2],
                mode="markers",
                marker=dict(size=1, color="rgba(180,180,180,0.55)"),
                name="pointcloud",
                hoverinfo="skip",
            )
        )

    traces.append(
        go.Scatter3d(
            x=df["center_x"],
            y=df["center_y"],
            z=df["center_z"],
            mode="markers",
            marker=dict(size=5, color="#4CC9F0"),
            name="cameras",
            customdata=df.index.to_numpy(),
            hovertemplate="<b>%{text}</b><br>x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<extra></extra>",
            text=df["image_id"],
        )
    )

    if selected_row is not None and 0 <= selected_row < len(df):
        row = df.iloc[selected_row]
        C = np.array([row["center_x"], row["center_y"], row["center_z"]], dtype=np.float64)
        fwd = np.array([row["fwd_x"], row["fwd_y"], row["fwd_z"]], dtype=np.float64)
        fwd = fwd / (np.linalg.norm(fwd) + 1e-12)

        traces.append(
            go.Scatter3d(
                x=[C[0]],
                y=[C[1]],
                z=[C[2]],
                mode="markers",
                marker=dict(size=8, color="#F72585"),
                name="selected",
                hoverinfo="skip",
            )
        )

        arrow_len = float(frustum_scale) * 1.25
        arrow_pts = np.vstack([C, C + fwd * arrow_len, [np.nan, np.nan, np.nan]])
        traces.append(
            go.Scatter3d(
                x=arrow_pts[:, 0],
                y=arrow_pts[:, 1],
                z=arrow_pts[:, 2],
                mode="lines",
                line=dict(color="#F72585", width=6),
                name="forward",
                hoverinfo="skip",
            )
        )

        if show_frustum and "R_cw" in df.columns and "t_cw" in df.columns:
            R = row["R_cw"]
            t = row["t_cw"]
            if isinstance(R, np.ndarray) and isinstance(t, np.ndarray):
                aspect = row.get("aspect", 1.0)
                lines = _frustum_lines_world(
                    R_cw=R,
                    C_w=C,
                    aspect=float(aspect) if aspect else 1.0,
                    fov_deg=float(fov_deg),
                    scale=float(frustum_scale),
                )
                traces.append(
                    go.Scatter3d(
                        x=lines[:, 0],
                        y=lines[:, 1],
                        z=lines[:, 2],
                        mode="lines",
                        line=dict(color="#F72585", width=3),
                        name="frustum",
                        hoverinfo="skip",
                    )
                )

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        title=title,
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(aspectmode="data"),
        legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="right", x=0.98),
    )
    return fig


def _add_image_aspect(df: pd.DataFrame, image_index: dict[str, Path]) -> pd.DataFrame:
    aspect_list = []
    for _, row in df.iterrows():
        aspect = None
        img_path = image_index.get(str(row["image"]))
        if img_path is not None:
            aspect = _image_aspect(img_path)
        aspect_list.append(aspect if aspect else 1.0)
    out = df.copy()
    out["aspect"] = aspect_list
    return out


def export_html(fig: go.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)


def run_dash_app(
    df_all: pd.DataFrame,
    *,
    image_index: dict[str, Path] | None,
    pointcloud: PointCloud | None,
    host: str,
    port: int,
) -> None:
    if Dash is None:
        raise RuntimeError("dash is not installed; cannot start the web app")

    datasets = sorted(df_all["dataset"].unique().tolist())
    initial_dataset = datasets[0] if datasets else None
    initial_scene = None
    if initial_dataset is not None:
        scenes = sorted(df_all[df_all["dataset"] == initial_dataset]["scene"].unique().tolist())
        initial_scene = scenes[0] if scenes else None

    app = Dash(__name__)
    app.title = "IMC25 Pose Explorer"

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H2("IMC25: Camera Pose Explorer"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Dataset"),
                                    dcc.Dropdown(
                                        id="dataset-dd",
                                        options=[{"label": d, "value": d} for d in datasets],
                                        value=initial_dataset,
                                        clearable=False,
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("Scene"),
                                    dcc.Dropdown(id="scene-dd", clearable=False, value=initial_scene),
                                ],
                                style={"flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("Pointcloud"),
                                    dcc.Checklist(
                                        id="pc-toggle",
                                        options=[{"label": "show", "value": "show"}],
                                        value=["show"],
                                        style={"marginTop": "6px"},
                                    ),
                                ],
                                style={"flex": "0.6"},
                            ),
                            html.Div(
                                [
                                    html.Label("Frustum scale"),
                                    dcc.Slider(
                                        id="frustum-scale",
                                        min=0.2,
                                        max=8.0,
                                        step=0.1,
                                        value=1.0,
                                        marks={0.2: "0.2", 1.0: "1", 3.0: "3", 6.0: "6", 8.0: "8"},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                    ),
                                ],
                                style={"flex": "1.2"},
                            ),
                        ],
                        style={"display": "flex", "gap": "14px", "alignItems": "end"},
                    ),
                ],
                style={"padding": "14px 18px 10px 18px"},
            ),
            html.Div(
                [
                    html.Div(
                        [dcc.Graph(id="pose-graph", style={"height": "78vh"})],
                        style={"flex": "2.2", "minWidth": "520px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4("Selected frame"),
                                    html.Img(
                                        id="frame-img",
                                        style={
                                            "width": "100%",
                                            "borderRadius": "10px",
                                            "border": "1px solid rgba(255,255,255,0.12)",
                                        },
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.H4("Pose"),
                                    html.Pre(
                                        id="pose-pre",
                                        style={
                                            "whiteSpace": "pre-wrap",
                                            "background": "rgba(255,255,255,0.06)",
                                            "padding": "10px",
                                            "borderRadius": "10px",
                                            "border": "1px solid rgba(255,255,255,0.08)",
                                            "fontSize": "12px",
                                        },
                                    ),
                                ],
                                style={"marginTop": "12px"},
                            ),
                        ],
                        style={"flex": "1", "minWidth": "340px", "paddingRight": "14px"},
                    ),
                ],
                style={"display": "flex", "gap": "14px", "paddingLeft": "14px"},
            ),
        ],
        style={"fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"},
    )

    @app.callback(Output("scene-dd", "options"), Output("scene-dd", "value"), Input("dataset-dd", "value"))
    def _update_scenes(dataset: str):
        if not dataset:
            return [], None
        scenes = sorted(df_all[df_all["dataset"] == dataset]["scene"].unique().tolist())
        opts = [{"label": s, "value": s} for s in scenes]
        return opts, (scenes[0] if scenes else None)

    @app.callback(
        Output("pose-graph", "figure"),
        Output("frame-img", "src"),
        Output("pose-pre", "children"),
        Input("dataset-dd", "value"),
        Input("scene-dd", "value"),
        Input("pose-graph", "clickData"),
        Input("pc-toggle", "value"),
        Input("frustum-scale", "value"),
    )
    def _update_view(dataset: str, scene: str, click_data, pc_value, frustum_scale):
        df = df_all
        if dataset:
            df = df[df["dataset"] == dataset]
        if scene:
            df = df[df["scene"] == scene]
        df = df.reset_index(drop=True)

        selected_row: int | None = 0 if len(df) else None
        if click_data and click_data.get("points"):
            cd = click_data["points"][0].get("customdata")
            if cd is not None:
                try:
                    selected_row = int(cd)
                except Exception:
                    selected_row = selected_row

        if selected_row is not None and (selected_row < 0 or selected_row >= len(df)):
            selected_row = 0 if len(df) else None

        show_pc = pointcloud is not None and pc_value and "show" in pc_value
        fig = make_pose_figure(
            df,
            pointcloud=pointcloud,
            selected_row=selected_row,
            show_pointcloud=show_pc,
            frustum_scale=float(frustum_scale or 1.0),
            title=f"{dataset}/{scene} ({len(df)} frames)" if dataset and scene else f"{len(df)} frames",
        )

        img_src = None
        pose_text = "No data"
        if selected_row is not None and len(df):
            row = df.iloc[selected_row]
            pose_text = (
                f"image_id: {row['image_id']}\n"
                f"image: {row['image']}\n"
                f"dataset: {row['dataset']}  scene: {row['scene']}\n\n"
                f"center (world): [{row['center_x']:.4f}, {row['center_y']:.4f}, {row['center_z']:.4f}]\n"
                f"forward (world): [{row['fwd_x']:.4f}, {row['fwd_y']:.4f}, {row['fwd_z']:.4f}]\n\n"
                f"R_cw:\n{parse_rotation_matrix(row['rotation_matrix'])}\n\n"
                f"t_cw:\n{parse_translation_vector(row['translation_vector'])}\n"
            )
            if image_index is not None:
                img_path = image_index.get(str(row["image"]))
                if img_path is not None:
                    img_src = _thumbnail_data_url(img_path)

        return fig, img_src, pose_text

    # Dash 3+: run_server -> run
    if hasattr(app, "run"):
        app.run(host=host, port=port, debug=False)
    else:  # pragma: no cover
        # Older Dash versions
        app.run_server(host=host, port=port, debug=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive 3D camera pose explorer for IMC-style submission.csv")
    parser.add_argument("--csv", type=Path, default=Path("submission.csv"), help="Path to submission-like CSV")
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="Folder containing images (used to show previews + correct frustum aspect).",
    )
    parser.add_argument("--pointcloud", type=Path, default=None, help="Optional .ply pointcloud to overlay")
    parser.add_argument("--max-points", type=int, default=150_000, help="Max pointcloud points to plot")
    parser.add_argument("--export-html", type=Path, default=None, help="Export a standalone HTML plot to this path")
    parser.add_argument("--no-server", action="store_true", help="Only export HTML (requires --export-html)")
    parser.add_argument("--host", default="127.0.0.1", help="Dash server host")
    parser.add_argument("--port", type=int, default=8050, help="Dash server port")
    args = parser.parse_args()

    df = load_submission(args.csv)
    image_index = index_images(args.images_root) if args.images_root else None
    pc = load_pointcloud(args.pointcloud, max_points=args.max_points) if args.pointcloud else None
    if image_index is not None:
        df = _add_image_aspect(df, image_index)

    if args.export_html is not None:
        fig = make_pose_figure(
            df,
            pointcloud=pc,
            selected_row=0 if len(df) else None,
            show_pointcloud=pc is not None,
            title=f"{args.csv.name} ({len(df)} frames)",
        )
        export_html(fig, args.export_html)

    if args.no_server:
        if args.export_html is None:
            raise SystemExit("--no-server requires --export-html")
        return

    run_dash_app(df, image_index=image_index, pointcloud=pc, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
