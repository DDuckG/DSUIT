#!/usr/bin/env bash
set -euo pipefail

VIDEO_DIR=${1:-../data/videos/}
FPS=${2:-2}
OUT_DIR=${3:-../datasets/images/raw_frames}

echo "Videos dir: $VIDEO_DIR"
echo "FPS: $FPS"
echo "Output dir: $OUT_DIR"

mkdir -p "$OUT_DIR"

shopt -s nullglob
mp4s=("$VIDEO_DIR"/*.mp4)

if [ ${#mp4s[@]} -eq 0 ]; then
    echo "No .mp4 files found in $VIDEO_DIR"
    exit 0
fi

count=0
for vid in "${mp4s[@]}"; do
    if [[ ! -f "$vid" ]]; then
        continue
    fi
    count=$((count++))
    base=$(basename "$vid")
    name="${base%.*}"
    safe_name=$(echo "$name" | sed -E 's/[^A-Za-z0-9]/_/g')

    echo "Processing [$count]: $vid -> frames prefix: ${safe_name}_"

    existing_pattern="$OUT_DIR/${safe_name}_*.jpg"
    if compgen -G "$OUT_DIR/${safe_name}_*.jpg" > /dev/null; then
        echo " - Removing existing frames matching $OUT_DIR/${safe_name}_*.jpg"
        rm -f "$OUT_DIR/${safe_name}_"*.jpg
    fi

    out_pattern="$OUT_DIR/${safe_name}_%05d.jpg"
    echo " - Extracting frames to $out_pattern"
    if ! ffmpeg -hide_banner -loglevel error -i "$vid" -r "$FPS" "$out_pattern"; then
        echo " - ERROR: ffmpeg failed for $vid (continuing to next)"
        continue
    fi
    echo " - Done extracting for $vid"

done

echo "Processed $count .mp4 videos(). Frames saved to $OUT_DIR"