#!/usr/bin/env bash
# Paper 4 — compute additional SurtGis features for target basins.
#
# Reads DEM + precomputed rasters from paper1_susceptibilidad/factors/,
# writes new features to paper4_meta_learning_transfer/factors_extra/.
#
# Idempotent: each output is skipped if it already exists.

set -euo pipefail

SURTGIS="/home/franciscoparrao/proyectos/surtgis/target/release/surtgis"
PAPER1_FACTORS="/home/franciscoparrao/proyectos/postdoc/papers/paper1_susceptibilidad/factors"
PAPER4_ROOT="/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer"
OUT_BASE="$PAPER4_ROOT/factors_extra"

ALL_BASINS=(02_costeras_tilviche_loa 03_costeras_loa_caracoles 04_rio_salado 05_rio_copiapo 06_rio_huasco 07_rio_elqui 08_rio_limari 09_rio_maipo 10_rio_rapel 11_rio_maule 14_costeras_bueno_puelo 15_costeras_magallanes taltal choapa)
# Approximate centroid latitudes (degrees, negative = southern hemisphere)
ALL_LATS=(-21.0 -22.5 -26.3 -27.5 -28.5 -30.0 -31.0 -33.5 -34.5 -35.5 -41.0 -53.0 -25.4 -31.7)

# Per-basin override for source factors directory.
# Defaults to $PAPER1_FACTORS/$basin if not set.
declare -A SRC_OVERRIDE=(
    [taltal]="$PAPER4_ROOT/factors/taltal"
    [choapa]="$PAPER4_ROOT/factors/choapa"
)

# Optional CLI filter: basin name(s) to run. Empty = run all.
if (( $# > 0 )); then
    BASINS=("$@")
    LATS=()
    for target in "$@"; do
        for j in "${!ALL_BASINS[@]}"; do
            if [[ "${ALL_BASINS[$j]}" == "$target" ]]; then
                LATS+=("${ALL_LATS[$j]}")
                continue 2
            fi
        done
        echo "ERROR: unknown basin '$target'. Valid: ${ALL_BASINS[*]}" >&2
        exit 1
    done
else
    BASINS=("${ALL_BASINS[@]}")
    LATS=("${ALL_LATS[@]}")
fi

CELL_SIZE=30

sg() { "$SURTGIS" --compress "$@"; }

FAILED=()

run_if_missing() {
    local out="$1"; shift
    if [[ -f "$out" ]]; then
        echo "  [skip] ${out#$OUT_BASE/}"
        return 0
    fi
    mkdir -p "$(dirname "$out")"
    echo "  [run ] ${out#$OUT_BASE/}"
    if ! "$@"; then
        echo "  [FAIL] ${out#$OUT_BASE/}" >&2
        FAILED+=("${out#$OUT_BASE/}")
        rm -f "$out"
        return 0
    fi
}

for i in "${!BASINS[@]}"; do
    basin="${BASINS[$i]}"
    lat="${LATS[$i]}"

    src="${SRC_OVERRIDE[$basin]:-$PAPER1_FACTORS/$basin}"
    dem="$src/dem_30m.tif"
    slope_deg="$src/terrain/slope.tif"
    hand="$src/hydrology/hand.tif"
    flow_acc="$src/hydrology/flow_accumulation.tif"
    flow_dir="$src/hydrology/flow_direction_d8.tif"
    stream="$src/hydrology/stream_network.tif"

    out="$OUT_BASE/$basin"
    echo ""
    echo "========== $basin (lat=$lat) =========="

    # --- helper: slope in radians (needed by LS-factor, sediment-connectivity)
    slope_rad="$out/terrain/slope_rad.tif"
    run_if_missing "$slope_rad" sg terrain slope -u radians "$dem" "$slope_rad"

    # --- Terrain: multi-scale TPI
    for r in 3 9 21; do
        run_if_missing "$out/terrain/tpi_r${r}.tif" \
            sg terrain tpi -r "$r" "$dem" "$out/terrain/tpi_r${r}.tif"
    done

    # --- Terrain: Weiss-style landform classification
    run_if_missing "$out/terrain/landform.tif" \
        sg terrain landform "$dem" "$out/terrain/landform.tif"

    # --- Terrain: valley depth (needed by relative-slope-position)
    valley_depth="$out/terrain/valley_depth.tif"
    run_if_missing "$valley_depth" sg terrain valley-depth "$dem" "$valley_depth"

    # --- Terrain: relative slope position
    run_if_missing "$out/terrain/relative_slope_position.tif" \
        sg terrain relative-slope-position \
            --hand "$hand" --valley-depth "$valley_depth" \
            "$out/terrain/relative_slope_position.tif"

    # --- Terrain: LS-factor (RUSLE)
    run_if_missing "$out/terrain/ls_factor.tif" \
        sg terrain ls-factor \
            --slope "$slope_rad" --flow-acc "$flow_acc" --cell-size "$CELL_SIZE" \
            "$out/terrain/ls_factor.tif"

    # Skipped: terrain/solar-radiation-annual (too slow per basin, partially
    # captured by aspect/eastness/northness/hillshade/svf already in Paper 1).

    # --- Terrain: curvedness
    run_if_missing "$out/terrain/curvedness.tif" \
        sg terrain curvedness "$dem" "$out/terrain/curvedness.tif"

    # --- Terrain: shape index
    run_if_missing "$out/terrain/shape_index.tif" \
        sg terrain shape-index "$dem" "$out/terrain/shape_index.tif"

    # --- Terrain: surface area ratio
    run_if_missing "$out/terrain/surface_area_ratio.tif" \
        sg terrain surface-area-ratio "$dem" "$out/terrain/surface_area_ratio.tif"

    # --- Hydrology: drainage density (needs stream network)
    run_if_missing "$out/hydrology/drainage_density.tif" \
        sg hydrology drainage-density --cell-size "$CELL_SIZE" \
            "$stream" "$out/hydrology/drainage_density.tif"

    # --- Hydrology: sediment connectivity (Borselli)
    run_if_missing "$out/hydrology/sediment_connectivity.tif" \
        sg hydrology sediment-connectivity \
            --slope "$slope_rad" --flow-acc "$flow_acc" --flow-dir "$flow_dir" \
            "$out/hydrology/sediment_connectivity.tif"

    # --- Texture: GLCM on DEM (6 measures at default radius=3, levels=32)
    for tex in energy contrast homogeneity correlation entropy dissimilarity; do
        run_if_missing "$out/texture/glcm_${tex}_dem.tif" \
            sg texture glcm -t "$tex" "$dem" "$out/texture/glcm_${tex}_dem.tif"
    done

    # --- Focal stats: DEM at three scales
    for r in 1 4 10; do
        run_if_missing "$out/focal_stats/dem_std_r${r}.tif" \
            sg statistics focal -s std -r "$r" "$dem" "$out/focal_stats/dem_std_r${r}.tif"
    done
    run_if_missing "$out/focal_stats/dem_range_r4.tif" \
        sg statistics focal -s range -r 4 "$dem" "$out/focal_stats/dem_range_r4.tif"

    # --- Focal stats: slope at medium scale
    run_if_missing "$out/focal_stats/slope_std_r4.tif" \
        sg statistics focal -s std -r 4 "$slope_deg" "$out/focal_stats/slope_std_r4.tif"
    run_if_missing "$out/focal_stats/slope_mean_r4.tif" \
        sg statistics focal -s mean -r 4 "$slope_deg" "$out/focal_stats/slope_mean_r4.tif"
done

echo ""
echo "========== Done =========="
if (( ${#FAILED[@]} > 0 )); then
    echo "Failed outputs (${#FAILED[@]}):" >&2
    printf '  %s\n' "${FAILED[@]}" >&2
    exit 1
fi
