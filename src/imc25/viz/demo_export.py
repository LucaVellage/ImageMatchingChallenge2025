from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from imc25.viz.pose_explorer import export_html, index_images, load_pointcloud, load_submission, make_pose_figure


def _ply_vertex_count(path: Path) -> int | None:
    try:
        with path.open("rb") as f:
            for _ in range(200):
                line = f.readline()
                if not line:
                    return None
                if line.startswith(b"element vertex "):
                    return int(line.split()[-1])
                if line.strip() == b"end_header":
                    break
        return None
    except Exception:
        return None


def export_demo_html(
    *,
    submission_csv: Path,
    recon_root: Path,
    out_dir: Path,
    outliers_label: str = "outliers",
    max_points: int = 150_000,
    include_pointcloud: bool = True,
) -> list[Path]:
    df = pd.read_csv(submission_csv)
    required = {"dataset", "scene", "image", "rotation_matrix", "translation_vector"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{submission_csv} missing columns: {sorted(missing)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    rows: list[dict[str, str]] = []
    for (dataset, scene), grp in df.groupby(["dataset", "scene"], sort=True):
        if str(scene).lower() == str(outliers_label).lower():
            continue

        cluster_dir = recon_root / f"{dataset}_{scene}"
        images_root = cluster_dir / "images"
        pointcloud_path = cluster_dir / "dense_points.ply"

        cluster_csv = out_dir / f"{dataset}_{scene}.csv"
        grp.to_csv(cluster_csv, index=False)

        try:
            df_pose = load_submission(cluster_csv)
        except Exception:
            continue
        if len(df_pose) == 0:
            continue

        image_index = None
        if images_root.exists():
            try:
                image_index = index_images(images_root)
            except Exception:
                image_index = None

        if image_index is not None:
            # Optional enhancement: frustum aspect ratio per image.
            from imc25.viz.pose_explorer import _add_image_aspect

            df_pose = _add_image_aspect(df_pose, image_index)

        pc = None
        if include_pointcloud and pointcloud_path.exists():
            try:
                pc = load_pointcloud(pointcloud_path, max_points=max_points)
            except Exception:
                pc = None

        out_html = out_dir / f"{dataset}_{scene}.html"
        fig = make_pose_figure(
            df_pose,
            pointcloud=pc,
            selected_row=0 if len(df_pose) else None,
            show_pointcloud=pc is not None,
            title=f"{dataset}/{scene} (poses={len(df_pose)})",
        )
        export_html(fig, out_html)
        written.append(out_html)

        rows.append(
            {
                "dataset": str(dataset),
                "scene": str(scene),
                "poses": str(int(len(df_pose))),
                "dense_points": str(_ply_vertex_count(pointcloud_path) or 0) if pointcloud_path.exists() else "0",
                "html": out_html.name,
            }
        )

    # Write index.html
    index_html = out_dir / "index.html"
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>IMC25 Demo</title>",
        "<style>body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px}table{border-collapse:collapse}th,td{border:1px solid #ddd;padding:6px 10px}th{background:#f6f6f6;text-align:left}</style>",
        "</head><body>",
        f"<h1>IMC25 Demo</h1><p>Generated from <code>{submission_csv}</code></p>",
        "<table><thead><tr><th>dataset</th><th>scene</th><th>poses</th><th>dense_points</th><th>html</th></tr></thead><tbody>",
    ]
    for r in rows:
        lines.append(
            "<tr>"
            f"<td>{r['dataset']}</td>"
            f"<td>{r['scene']}</td>"
            f"<td>{r['poses']}</td>"
            f"<td>{r['dense_points']}</td>"
            f"<td><a href='{r['html']}'>{r['html']}</a></td>"
            "</tr>"
        )
    lines += ["</tbody></table>", "</body></html>"]
    index_html.write_text("\n".join(lines), encoding="utf-8")
    written.insert(0, index_html)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export presentation-ready HTML demos per cluster")
    parser.add_argument("--submission-csv", type=Path, default=Path("submission.csv"))
    parser.add_argument("--recon-root", type=Path, default=Path("outputs_retrieval_test"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--outliers-label", default="outliers")
    parser.add_argument("--max-points", type=int, default=150_000)
    parser.add_argument("--include-pointcloud", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    written = export_demo_html(
        submission_csv=args.submission_csv,
        recon_root=args.recon_root,
        out_dir=args.out_dir,
        outliers_label=str(args.outliers_label),
        max_points=int(args.max_points),
        include_pointcloud=bool(args.include_pointcloud),
    )
    print(f"[ok] wrote demo HTML: {written[0]}", flush=True)


if __name__ == "__main__":
    main()

