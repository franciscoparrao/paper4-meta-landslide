#!/usr/bin/env bash
# Paper 4 — Sequential per-band Sentinel-2 composite for Taltal via Earth Search.
# Earth Search avoids SAS expiration that breaks long-running PC composites.

set -uo pipefail

SURTGIS="/home/franciscoparrao/proyectos/surtgis/target/release/surtgis"
ROOT="/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer"
DEM="$ROOT/factors/taltal/dem_30m.tif"
OUTDIR="$ROOT/factors/taltal/imagery"
LOGDIR="$ROOT/factors/taltal/imagery/logs"
BBOX="-70.735574,-26.058971,-68.994233,-23.702685"
DATETIME="2023-01-01/2023-12-31"

mkdir -p "$OUTDIR" "$LOGDIR"

BANDS=(green blue nir swir16 swir22)
FAILED=()

for band in "${BANDS[@]}"; do
    out="$OUTDIR/s2_${band}.tif"
    log="$LOGDIR/s2_${band}.log"
    if [[ -f "$out" ]]; then
        echo "[skip] $out exists"
        continue
    fi
    echo "=== $(date -Iseconds) starting band: $band ==="
    if ! "$SURTGIS" --compress --streaming --max-memory 6G stac composite \
            --catalog es \
            --collection sentinel-2-l2a \
            --asset "$band" \
            --bbox="$BBOX" \
            --datetime="$DATETIME" \
            --max-scenes 12 \
            --align-to "$DEM" \
            --band-chunk-size 1 \
            --strip-rows 256 \
            --cache \
            "$out" > "$log" 2>&1; then
        echo "[FAIL] $band — see $log"
        FAILED+=("$band")
        rm -f "$out"
    else
        echo "[done] $band ($(du -h "$out" | cut -f1))"
    fi
done

echo ""
echo "=== $(date -Iseconds) all bands done ==="
ls -lah "$OUTDIR"/*.tif 2>/dev/null
if (( ${#FAILED[@]} > 0 )); then
    echo "FAILED: ${FAILED[*]}" >&2
    exit 1
fi
