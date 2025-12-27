from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAX_IMAGE_ID = 2147483647

CAMERA_MODEL_IDS: dict[str, int] = {
    # Common COLMAP camera model IDs.
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
    "FULL_OPENCV": 6,
    "FOV": 7,
    "SIMPLE_RADIAL_FISHEYE": 8,
    "RADIAL_FISHEYE": 9,
    "THIN_PRISM_FISHEYE": 10,
}


def image_pair_id(image_id1: int, image_id2: int) -> int:
    if image_id1 == image_id2:
        raise ValueError("image_id1 must differ from image_id2")
    a, b = (image_id1, image_id2) if image_id1 < image_id2 else (image_id2, image_id1)
    return a * MAX_IMAGE_ID + b


def _blob_from_array(arr: np.ndarray) -> bytes:
    return arr.tobytes(order="C")


def create_empty_colmap_db(db_path: Path, *, overwrite: bool) -> None:
    if overwrite and db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS rigs (
                rig_id               INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,
                ref_sensor_id        INTEGER                             NOT NULL,
                ref_sensor_type      INTEGER                             NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS rig_ref_sensor_assignment
                ON rigs(ref_sensor_id, ref_sensor_type);
            CREATE TABLE IF NOT EXISTS rig_sensors (
                rig_id               INTEGER                             NOT NULL,
                sensor_id            INTEGER                             NOT NULL,
                sensor_type          INTEGER                             NOT NULL,
                sensor_from_rig      BLOB,
                FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE
            );
            CREATE UNIQUE INDEX IF NOT EXISTS rig_sensor_assignment
                ON rig_sensors(sensor_id, sensor_type);

            CREATE TABLE IF NOT EXISTS cameras (
                camera_id            INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,
                model                INTEGER                             NOT NULL,
                width                INTEGER                             NOT NULL,
                height               INTEGER                             NOT NULL,
                params               BLOB,
                prior_focal_length   INTEGER                             NOT NULL
            );

            CREATE TABLE IF NOT EXISTS frames (
                frame_id             INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,
                rig_id               INTEGER                             NOT NULL,
                FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS frame_data (
                frame_id             INTEGER                             NOT NULL,
                data_id              INTEGER                             NOT NULL,
                sensor_id            INTEGER                             NOT NULL,
                sensor_type          INTEGER                             NOT NULL,
                FOREIGN KEY(frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE
            );
            CREATE UNIQUE INDEX IF NOT EXISTS frame_sensor_assignment
                ON frame_data(data_id, sensor_type);

            CREATE TABLE IF NOT EXISTS images (
                image_id   INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,
                name       TEXT                                NOT NULL UNIQUE,
                camera_id  INTEGER                             NOT NULL,
                prior_qw   REAL,
                prior_qx   REAL,
                prior_qy   REAL,
                prior_qz   REAL,
                prior_tx   REAL,
                prior_ty   REAL,
                prior_tz   REAL,
                CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 2147483647),
                FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name);

            CREATE TABLE IF NOT EXISTS pose_priors (
                image_id                   INTEGER  PRIMARY KEY  NOT NULL,
                position                   BLOB,
                coordinate_system          INTEGER               NOT NULL,
                position_covariance        BLOB,
                FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS keypoints (
                image_id  INTEGER  PRIMARY KEY  NOT NULL,
                rows      INTEGER               NOT NULL,
                cols      INTEGER               NOT NULL,
                data      BLOB,
                FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS descriptors (
                image_id  INTEGER  PRIMARY KEY  NOT NULL,
                rows      INTEGER               NOT NULL,
                cols      INTEGER               NOT NULL,
                data      BLOB,
                FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS matches (
                pair_id  INTEGER  PRIMARY KEY  NOT NULL,
                rows     INTEGER               NOT NULL,
                cols     INTEGER               NOT NULL,
                data     BLOB
            );
            CREATE TABLE IF NOT EXISTS two_view_geometries (
                pair_id  INTEGER  PRIMARY KEY  NOT NULL,
                rows     INTEGER               NOT NULL,
                cols     INTEGER               NOT NULL,
                data     BLOB,
                config   INTEGER               NOT NULL,
                F        BLOB,
                E        BLOB,
                H        BLOB,
                qvec     BLOB,
                tvec     BLOB
            );
            """
        )


@dataclass(frozen=True)
class CameraSpec:
    model: str
    width: int
    height: int
    params: np.ndarray  # float64
    prior_focal_length: int = 0


def guess_simple_radial(width: int, height: int) -> CameraSpec:
    f = 1.2 * float(max(width, height))
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    k = 0.0
    params = np.array([f, cx, cy, k], dtype=np.float64)
    return CameraSpec(model="SIMPLE_RADIAL", width=width, height=height, params=params, prior_focal_length=0)


def insert_camera(conn: sqlite3.Connection, camera: CameraSpec) -> int:
    model_id = CAMERA_MODEL_IDS.get(camera.model)
    if model_id is None:
        raise ValueError(f"Unknown camera model: {camera.model}")
    params_blob = struct.pack("<" + "d" * int(camera.params.size), *camera.params.tolist())
    cur = conn.execute(
        "INSERT INTO cameras(model,width,height,params,prior_focal_length) VALUES (?,?,?,?,?)",
        (model_id, int(camera.width), int(camera.height), params_blob, int(camera.prior_focal_length)),
    )
    return int(cur.lastrowid)


def insert_image(conn: sqlite3.Connection, *, name: str, camera_id: int) -> int:
    cur = conn.execute("INSERT INTO images(name,camera_id) VALUES (?,?)", (name, int(camera_id)))
    return int(cur.lastrowid)


def insert_keypoints(conn: sqlite3.Connection, *, image_id: int, keypoints_xy: np.ndarray) -> None:
    if keypoints_xy.ndim != 2 or keypoints_xy.shape[1] != 2:
        raise ValueError(f"Expected keypoints (N,2), got {keypoints_xy.shape}")
    n = int(keypoints_xy.shape[0])
    kp = np.zeros((n, 6), dtype=np.float32)
    kp[:, 0:2] = keypoints_xy.astype(np.float32, copy=False)
    kp[:, 2] = 1.0
    kp[:, 5] = 1.0
    conn.execute(
        "INSERT INTO keypoints(image_id,rows,cols,data) VALUES (?,?,?,?)",
        (int(image_id), n, 6, _blob_from_array(kp)),
    )


def insert_dummy_descriptors(conn: sqlite3.Connection, *, image_id: int, num_keypoints: int) -> None:
    n = int(num_keypoints)
    desc = np.zeros((n, 128), dtype=np.uint8)
    conn.execute(
        "INSERT INTO descriptors(image_id,rows,cols,data) VALUES (?,?,?,?)",
        (int(image_id), n, 128, _blob_from_array(desc)),
    )


def insert_matches(
    conn: sqlite3.Connection,
    *,
    image_id1: int,
    image_id2: int,
    matches: np.ndarray,
) -> None:
    if matches.ndim != 2 or matches.shape[1] != 2:
        raise ValueError(f"Expected matches (N,2), got {matches.shape}")
    pair = image_pair_id(int(image_id1), int(image_id2))
    m = matches.astype(np.uint32, copy=False)
    conn.execute(
        "INSERT OR REPLACE INTO matches(pair_id,rows,cols,data) VALUES (?,?,?,?)",
        (int(pair), int(m.shape[0]), 2, _blob_from_array(m)),
    )


def insert_two_view_geometry(
    conn: sqlite3.Connection,
    *,
    image_id1: int,
    image_id2: int,
    inlier_matches: np.ndarray,
    config: int,
    F_mat: np.ndarray | None,
    E_mat: np.ndarray | None,
    H_mat: np.ndarray | None,
    qvec: np.ndarray | None,
    tvec: np.ndarray | None,
) -> None:
    if inlier_matches.ndim != 2 or inlier_matches.shape[1] != 2:
        raise ValueError(f"Expected inlier_matches (N,2), got {inlier_matches.shape}")
    pair = image_pair_id(int(image_id1), int(image_id2))
    m = inlier_matches.astype(np.uint32, copy=False)

    def mat_blob(mat: np.ndarray | None) -> bytes:
        if mat is None:
            mat = np.zeros((3, 3), dtype=np.float64)
        mat = np.asarray(mat, dtype=np.float64).reshape(3, 3)
        return _blob_from_array(mat)

    def vec_blob(vec: np.ndarray | None, n: int) -> bytes:
        if vec is None:
            vec = np.zeros((n,), dtype=np.float64)
        vec = np.asarray(vec, dtype=np.float64).reshape(n)
        return _blob_from_array(vec)

    conn.execute(
        "INSERT OR REPLACE INTO two_view_geometries(pair_id,rows,cols,data,config,F,E,H,qvec,tvec) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            int(pair),
            int(m.shape[0]),
            2,
            _blob_from_array(m),
            int(config),
            mat_blob(F_mat),
            mat_blob(E_mat),
            mat_blob(H_mat),
            vec_blob(qvec, 4),
            vec_blob(tvec, 3),
        ),
    )
