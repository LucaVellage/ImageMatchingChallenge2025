#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/home/shamika/documents/prog_workspace/ml_imc25/ImageMatchingChallenge2025}
OUT=$BASE/outputs/test2

print_step(){
  printf '\n==== %s ====%n' "$1"
}

# Sanity checks
if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found" >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "docker daemon not accessible" >&2
  exit 1
fi

print_step "Clean previous outputs"
mkdir -p "$OUT" "$OUT/sparse"
rm -f "$OUT/colmap.db" "$OUT/dense_points.ply"
rm -rf "$OUT/sparse/0" "$OUT/dense"
mkdir -p "$OUT"
touch "$OUT/colmap.db"

docker_run(){
  docker run --rm --gpus all -v "$BASE":/workspace -w /workspace colmap/colmap:latest "$@"
}

print_step "Feature extraction"
docker_run colmap feature_extractor \
  --database_path outputs/test2/colmap.db \
  --image_path outputs/test2/images \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model SIMPLE_RADIAL

print_step "Exhaustive matching"
docker_run colmap exhaustive_matcher \
  --database_path outputs/test2/colmap.db

print_step "Sparse reconstruction"
docker_run colmap mapper \
  --database_path outputs/test2/colmap.db \
  --image_path outputs/test2/images \
  --output_path outputs/test2/sparse \
  --Mapper.ba_refine_principal_point 0

print_step "Undistort for dense"
docker_run colmap image_undistorter \
  --image_path outputs/test2/images \
  --input_path outputs/test2/sparse/0 \
  --output_path outputs/test2/dense \
  --output_type COLMAP

print_step "PatchMatch stereo"
docker_run colmap patch_match_stereo \
  --workspace_path outputs/test2/dense \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true \
  --PatchMatchStereo.gpu_index 0

print_step "Stereo fusion to dense point cloud"
docker_run colmap stereo_fusion \
  --workspace_path outputs/test2/dense \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path outputs/test2/dense_points.ply

print_step "Dense point cloud info"
ls -l "$OUT/dense_points.ply"
