from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from imc25.matching.colmap_db import image_pair_id, insert_matches, insert_two_view_geometry
from imc25.matching.match_utils import ransac_fundamental


@dataclass(frozen=True)
class FusionConfig:
    """
    Controls how deep-match correspondences are merged into an existing COLMAP (SIFT) database.

    - `merge_px`: if a deep keypoint is within this distance of an existing SIFT keypoint,
      we reuse the SIFT keypoint index; otherwise we append a new keypoint.
    - `min_inliers`: minimum verified inliers to insert/replace a two-view geometry.
    - `ransac_thresh_px`: RANSAC reprojection threshold for the final fused geometry.
    - `skip_if_sift_inliers_ge`: skip deep matching for pairs that already have at least this many SIFT inliers.
    """

    merge_px: float = 2.0
    min_inliers: int = 8
    ransac_thresh_px: float = 2.5
    skip_if_sift_inliers_ge: int = 25


def _read_keypoints_xy(conn: sqlite3.Connection, *, image_id: int) -> np.ndarray:
    row = conn.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (int(image_id),)).fetchone()
    if not row:
        return np.zeros((0, 2), dtype=np.float32)
    rows, cols, data = row
    arr = np.frombuffer(data, dtype=np.float32)
    kp = arr.reshape(int(rows), int(cols))
    return kp[:, :2].astype(np.float32, copy=False)


def _read_keypoints_full(conn: sqlite3.Connection, *, image_id: int) -> np.ndarray:
    row = conn.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (int(image_id),)).fetchone()
    if not row:
        return np.zeros((0, 6), dtype=np.float32)
    rows, cols, data = row
    arr = np.frombuffer(data, dtype=np.float32)
    kp = arr.reshape(int(rows), int(cols))
    if kp.shape[1] != 6:
        # Keep 6 columns for compatibility with COLMAP's keypoint schema.
        out = np.zeros((kp.shape[0], 6), dtype=np.float32)
        out[:, : min(2, kp.shape[1])] = kp[:, : min(2, kp.shape[1])]
        out[:, 2] = 1.0
        out[:, 5] = 1.0
        return out
    return kp.astype(np.float32, copy=False)


def _read_descriptors(conn: sqlite3.Connection, *, image_id: int) -> np.ndarray:
    row = conn.execute("SELECT rows, cols, data FROM descriptors WHERE image_id=?", (int(image_id),)).fetchone()
    if not row:
        return np.zeros((0, 128), dtype=np.uint8)
    rows, cols, data = row
    arr = np.frombuffer(data, dtype=np.uint8)
    return arr.reshape(int(rows), int(cols)).astype(np.uint8, copy=False)


def _write_keypoints(conn: sqlite3.Connection, *, image_id: int, keypoints: np.ndarray) -> None:
    kp = np.asarray(keypoints, dtype=np.float32)
    if kp.ndim != 2 or kp.shape[1] != 6:
        raise ValueError(f"Expected keypoints (N,6), got {kp.shape}")
    conn.execute(
        "UPDATE keypoints SET rows=?, cols=?, data=? WHERE image_id=?",
        (int(kp.shape[0]), int(kp.shape[1]), kp.tobytes(order="C"), int(image_id)),
    )


def _write_descriptors(conn: sqlite3.Connection, *, image_id: int, descriptors: np.ndarray) -> None:
    desc = np.asarray(descriptors, dtype=np.uint8)
    if desc.ndim != 2:
        raise ValueError(f"Expected descriptors (N,D), got {desc.shape}")
    conn.execute(
        "UPDATE descriptors SET rows=?, cols=?, data=? WHERE image_id=?",
        (int(desc.shape[0]), int(desc.shape[1]), desc.tobytes(order="C"), int(image_id)),
    )


def _pair_inliers(conn: sqlite3.Connection, *, image_id1: int, image_id2: int) -> int:
    pid = image_pair_id(int(image_id1), int(image_id2))
    row = conn.execute("SELECT rows FROM two_view_geometries WHERE pair_id=?", (int(pid),)).fetchone()
    return int(row[0]) if row else 0


def _read_two_view_matches(conn: sqlite3.Connection, *, image_id1: int, image_id2: int) -> np.ndarray:
    pid = image_pair_id(int(image_id1), int(image_id2))
    row = conn.execute("SELECT rows, cols, data FROM two_view_geometries WHERE pair_id=?", (int(pid),)).fetchone()
    if not row:
        return np.zeros((0, 2), dtype=np.uint32)
    rows, cols, data = row
    arr = np.frombuffer(data, dtype=np.uint32)
    m = arr.reshape(int(rows), int(cols))
    return m.astype(np.uint32, copy=False)


def _unique_matches(matches: np.ndarray) -> np.ndarray:
    if matches.size == 0:
        return matches.astype(np.uint32, copy=False)
    m = matches.astype(np.uint32, copy=False)
    # Use a packed uint64 key for uniqueness.
    key = (m[:, 0].astype(np.uint64) << 32) | m[:, 1].astype(np.uint64)
    order = np.argsort(key)
    key = key[order]
    m = m[order]
    keep = np.ones((m.shape[0],), dtype=bool)
    keep[1:] = key[1:] != key[:-1]
    return m[keep]


def _load_retrieval_pairs(cache_root: Path, dataset: str) -> list[tuple[str, str]]:
    meta_path = cache_root / dataset / "meta.json"
    if not meta_path.exists():
        return []
    meta = json.loads(meta_path.read_text())
    image_ids = [str(x) for x in meta.get("image_ids", [])]
    topk = int(meta.get("topk", 0) or 0)
    if topk <= 0:
        return []
    pairs_path = cache_root / dataset / f"pairs_topk_K{topk}.npz"
    if not pairs_path.exists():
        # Fallback: pick any compatible pairs file if the exact K file is missing.
        cand = sorted((cache_root / dataset).glob("pairs_topk_K*.npz"))
        if not cand:
            return []
        pairs_path = cand[0]
    data = np.load(str(pairs_path))
    a = np.asarray(data["a"], dtype=np.int64)
    b = np.asarray(data["b"], dtype=np.int64)
    out: list[tuple[str, str]] = []
    for ia, ib in zip(a.tolist(), b.tolist(), strict=False):
        if ia < 0 or ib < 0 or ia >= len(image_ids) or ib >= len(image_ids) or ia == ib:
            continue
        out.append((image_ids[int(ia)], image_ids[int(ib)]))
    return out


def fuse_sift_dino_diffusion(
    *,
    db_path: Path,
    images_dir: Path,
    cache_root: Path,
    dataset: str,
    cluster_image_ids: list[str],
    dino: dict | None,
    diffusion: dict | None,
    cfg: FusionConfig = FusionConfig(),
    verbose: bool = True,
) -> None:
    """
    Augment an existing (SIFT) COLMAP db with additional verified correspondences from DINO and diffusion matchers.

    The DB is modified in-place:
    - new keypoints are appended per image (if needed),
    - matches / two_view_geometries are inserted/replaced if they improve inlier count.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"COLMAP DB not found: {db_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")

    # Map stem -> actual filename in cluster images/ dir.
    stem_to_name: dict[str, str] = {}
    for p in images_dir.iterdir():
        if not (p.is_file() or p.is_symlink()):
            continue
        stem_to_name.setdefault(p.stem, p.name)

    cluster_ids = {str(x) for x in cluster_image_ids}
    candidate_pairs = _load_retrieval_pairs(cache_root, dataset)
    if not candidate_pairs:
        if verbose:
            print(f"[fusion] {dataset}: no retrieval pairs found; skipping fusion", flush=True)
        return

    # Keep only pairs inside the current cluster and that map to existing filenames.
    pairs: list[tuple[str, str]] = []
    for ia, ib in candidate_pairs:
        if ia not in cluster_ids or ib not in cluster_ids:
            continue
        na = stem_to_name.get(str(ia))
        nb = stem_to_name.get(str(ib))
        if not na or not nb or na == nb:
            continue
        a, b = (na, nb) if na < nb else (nb, na)
        pairs.append((a, b))
    pairs = sorted(set(pairs))
    if not pairs:
        if verbose:
            print(f"[fusion] {dataset}: no within-cluster pairs after filtering; skipping fusion", flush=True)
        return

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")

        # Image name -> image_id.
        rows = conn.execute("SELECT image_id, name FROM images").fetchall()
        image_id_by_name: dict[str, int] = {str(name): int(iid) for iid, name in rows}

        # Pre-filter pairs that already have strong SIFT inliers.
        todo_pairs: list[tuple[str, str]] = []
        for na, nb in pairs:
            ida = image_id_by_name.get(na)
            idb = image_id_by_name.get(nb)
            if ida is None or idb is None:
                continue
            if _pair_inliers(conn, image_id1=ida, image_id2=idb) >= int(cfg.skip_if_sift_inliers_ge):
                continue
            todo_pairs.append((na, nb))

    if not todo_pairs:
        if verbose:
            print(f"[fusion] {dataset}: all candidate pairs already strong under SIFT; skipping fusion", flush=True)
        return

    if verbose:
        print(
            f"[fusion] {dataset}: augmenting db={db_path.name} with deep matches on {len(todo_pairs)}/{len(pairs)} pairs",
            flush=True,
        )

    # Lazy imports so fusion remains usable even if optional deps are missing.
    dino_extract = None
    dino_match = None
    diffusion_extract = None
    diffusion_match = None

    if dino is not None:
        try:
            from imc25.matching.dino_matcher import extract_image_features as _dino_extract
            from imc25.matching.dino_matcher import match_pair as _dino_match
        except Exception:
            dino = None
        else:
            dino_extract, dino_match = _dino_extract, _dino_match

    if diffusion is not None:
        try:
            from imc25.matching.diffusion_matcher import extract_image_features as _diff_extract
            from imc25.matching.diffusion_matcher import match_pair as _diff_match
        except Exception:
            diffusion = None
        else:
            diffusion_extract, diffusion_match = _diff_extract, _diff_match

    if dino is None and diffusion is None:
        if verbose:
            print("[fusion] deep matchers unavailable; leaving SIFT db unchanged", flush=True)
        return

    # Collect per-pair coordinate correspondences from deep matchers.
    coords_by_image: dict[str, list[np.ndarray]] = {}
    pair_corrs: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}

    # Feature caches (per image name).
    dino_feats: dict[str, object] = {}
    diff_feats: dict[str, object] = {}

    def _get_dino_feats(name: str):
        if name in dino_feats:
            return dino_feats[name]
        p = images_dir / name
        feats = dino_extract(
            p,
            extractor=dino["extractor"],
            keypoints=dino["kp_cfg"],
            cache_dir=Path(dino["cache_dir"]) if dino.get("cache_dir") else None,
            verbose=False,
        )
        dino_feats[name] = feats
        return feats

    def _get_diff_feats(name: str):
        if name in diff_feats:
            return diff_feats[name]
        p = images_dir / name
        feats = diffusion_extract(
            p,
            extractor=diffusion["extractor"],
            keypoints=diffusion["kp_cfg"],
            cache_dir=Path(diffusion["cache_dir"]) if diffusion.get("cache_dir") else None,
            verbose=False,
        )
        diff_feats[name] = feats
        return feats

    for na, nb in todo_pairs:
        # Prefer DINO first (cheaper); only run diffusion if DINO has too few inliers.
        got_any = False

        if dino is not None and dino_extract is not None and dino_match is not None:
            fa = _get_dino_feats(na)
            fb = _get_dino_feats(nb)
            r = dino_match(fa, fb, match_cfg=dino["match_cfg"], device=dino["extractor"].device)
            if r.F is not None and int(r.inlier_matches.shape[0]) >= int(cfg.min_inliers):
                kpa = np.asarray(fa.keypoints_xy, dtype=np.float32)[r.inlier_matches[:, 0].astype(np.int64)]
                kpb = np.asarray(fb.keypoints_xy, dtype=np.float32)[r.inlier_matches[:, 1].astype(np.int64)]
                coords_by_image.setdefault(na, []).append(kpa)
                coords_by_image.setdefault(nb, []).append(kpb)
                pair_corrs.setdefault((na, nb), []).append((kpa, kpb))
                got_any = True

        if (not got_any) and diffusion is not None and diffusion_extract is not None and diffusion_match is not None:
            fa = _get_diff_feats(na)
            fb = _get_diff_feats(nb)
            r = diffusion_match(fa, fb, match_cfg=diffusion["match_cfg"], device=diffusion["extractor"].device)
            if r.F is not None and int(r.inlier_matches.shape[0]) >= int(cfg.min_inliers):
                kpa = np.asarray(fa.keypoints_xy, dtype=np.float32)[r.inlier_matches[:, 0].astype(np.int64)]
                kpb = np.asarray(fb.keypoints_xy, dtype=np.float32)[r.inlier_matches[:, 1].astype(np.int64)]
                coords_by_image.setdefault(na, []).append(kpa)
                coords_by_image.setdefault(nb, []).append(kpb)
                pair_corrs.setdefault((na, nb), []).append((kpa, kpb))

    if not pair_corrs:
        if verbose:
            print(f"[fusion] {dataset}: no deep correspondences passed min_inliers; db unchanged", flush=True)
        return

    # Compute per-image keypoint augmentation and mapping (coord -> keypoint index).
    coord_to_idx: dict[str, dict[tuple[int, int], int]] = {}
    added_per_image: dict[str, int] = {}

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")

        # Image name -> image_id.
        rows = conn.execute("SELECT image_id, name FROM images").fetchall()
        image_id_by_name = {str(name): int(iid) for iid, name in rows}

        for name, coord_list in coords_by_image.items():
            image_id = image_id_by_name.get(name)
            if image_id is None:
                continue
            coords = np.concatenate(coord_list, axis=0) if coord_list else np.zeros((0, 2), dtype=np.float32)
            if coords.shape[0] == 0:
                continue

            # Quantize for stable hashing (0.5px grid).
            q = np.round(coords * 2.0).astype(np.int32)
            uq = np.unique(q, axis=0)

            kp_full = _read_keypoints_full(conn, image_id=image_id)
            kp_xy = kp_full[:, :2].astype(np.float32, copy=False)
            nn = None
            if kp_xy.shape[0] > 0:
                nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
                nn.fit(kp_xy)

            mapping: dict[tuple[int, int], int] = {}
            new_pts: list[tuple[float, float]] = []
            for qi in uq.tolist():
                key = (int(qi[0]), int(qi[1]))
                x = float(qi[0]) / 2.0
                y = float(qi[1]) / 2.0
                idx = None
                if nn is not None:
                    dist, ind = nn.kneighbors(np.array([[x, y]], dtype=np.float32), return_distance=True)
                    if float(dist[0, 0]) <= float(cfg.merge_px):
                        idx = int(ind[0, 0])
                if idx is None:
                    idx = int(kp_full.shape[0] + len(new_pts))
                    new_pts.append((x, y))
                mapping[key] = idx

            if new_pts:
                add = np.zeros((len(new_pts), 6), dtype=np.float32)
                add[:, 0] = np.array([p[0] for p in new_pts], dtype=np.float32)
                add[:, 1] = np.array([p[1] for p in new_pts], dtype=np.float32)
                add[:, 2] = 1.0
                add[:, 5] = 1.0
                kp_full_new = np.concatenate([kp_full, add], axis=0)

                desc = _read_descriptors(conn, image_id=image_id)
                if desc.shape[0] != kp_full.shape[0]:
                    # If descriptors are missing/mismatched, regenerate dummy descriptors for all keypoints.
                    desc = np.zeros((kp_full.shape[0], 128), dtype=np.uint8)
                desc_add = np.zeros((add.shape[0], desc.shape[1]), dtype=np.uint8)
                desc_new = np.concatenate([desc, desc_add], axis=0)

                _write_keypoints(conn, image_id=image_id, keypoints=kp_full_new)
                _write_descriptors(conn, image_id=image_id, descriptors=desc_new)
                added_per_image[name] = int(add.shape[0])

            coord_to_idx[name] = mapping

        if verbose:
            total_added = int(sum(added_per_image.values()))
            if total_added > 0:
                print(
                    f"[fusion] {dataset}: appended {total_added} keypoints across {len(added_per_image)} images",
                    flush=True,
                )

        # Insert/replace improved pair geometries.
        updated = 0
        for (na, nb), corr_list in pair_corrs.items():
            ida = image_id_by_name.get(na)
            idb = image_id_by_name.get(nb)
            if ida is None or idb is None:
                continue

            existing = _pair_inliers(conn, image_id1=ida, image_id2=idb)
            if existing >= int(cfg.skip_if_sift_inliers_ge):
                continue

            # Existing inliers (SIFT) + deep matches mapped to (idx_a, idx_b).
            base_inliers = _read_two_view_matches(conn, image_id1=ida, image_id2=idb)
            extra: list[np.ndarray] = []
            map_a = coord_to_idx.get(na, {})
            map_b = coord_to_idx.get(nb, {})

            for kpa, kpb in corr_list:
                qa = np.round(np.asarray(kpa, dtype=np.float32) * 2.0).astype(np.int32)
                qb = np.round(np.asarray(kpb, dtype=np.float32) * 2.0).astype(np.int32)
                ia = np.array([map_a.get((int(x), int(y)), -1) for x, y in qa.tolist()], dtype=np.int64)
                ib = np.array([map_b.get((int(x), int(y)), -1) for x, y in qb.tolist()], dtype=np.int64)
                good = (ia >= 0) & (ib >= 0)
                if not np.any(good):
                    continue
                m = np.stack([ia[good], ib[good]], axis=1).astype(np.uint32, copy=False)
                extra.append(m)

            if not extra and base_inliers.shape[0] == 0:
                continue

            fused = base_inliers
            if extra:
                fused = np.concatenate([base_inliers, *extra], axis=0) if base_inliers.size else np.concatenate(extra, axis=0)
            fused = _unique_matches(fused)
            if fused.shape[0] < 8:
                continue

            kp_a = _read_keypoints_xy(conn, image_id=ida)
            kp_b = _read_keypoints_xy(conn, image_id=idb)
            r = ransac_fundamental(
                kp_a,
                kp_b,
                fused,
                thresh_px=float(cfg.ransac_thresh_px),
                confidence=0.999,
                max_iters=10000,
            )
            if r.F is None or int(r.inlier_matches.shape[0]) < int(cfg.min_inliers):
                continue
            if int(r.inlier_matches.shape[0]) <= int(existing):
                continue

            insert_matches(conn, image_id1=ida, image_id2=idb, matches=r.inlier_matches)
            insert_two_view_geometry(
                conn,
                image_id1=ida,
                image_id2=idb,
                inlier_matches=r.inlier_matches,
                config=3,
                F_mat=r.F,
                E_mat=None,
                H_mat=None,
                qvec=None,
                tvec=None,
            )
            updated += 1

        if verbose:
            print(f"[fusion] {dataset}: updated {updated} two-view geometries", flush=True)
