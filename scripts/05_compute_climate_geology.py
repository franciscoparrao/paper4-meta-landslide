#!/usr/bin/env python3
"""Paper 4 — generate climate + geology rasters for Taltal & Choapa.

Brings these basins to feature parity with Paper 1 basins:
- climate/  : 5 WorldClim bioclim variables (bio_01, bio_12, bio_13, bio_14, bio_15)
              clipped to basin DEM grid + UTM 19S 30m
- geology/  : 3 categorical rasters (lithology_class, rock_type, geological_age)
              rasterized from SERNAGEOMIN 1:1M map at DEM grid
              uses encoding.json from Paper 1 for class consistency
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
PAPER1 = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper1_susceptibilidad/factors")
WORLDCLIM = PAPER1 / "_worldclim"
GEOLOGY_SHP = Path("/home/franciscoparrao/proyectos/Agentes/Remociones (2)/"
                    "Shape_Mapa_Geol_gico_Chile_1_1.000.000/"
                    "SNGM - Mapa Geologico Chile 1 millon-20191205T144710Z-001/"
                    "SNGM - Mapa Geologico Chile 1 millon/"
                    "geolchile_region.shp")
ENCODING_JSON = PAPER1 / "04_rio_salado/geology/encoding.json"

# WorldClim files use no leading zero: bio_1, bio_12, bio_13, bio_14, bio_15
CLIMATE_VARS = [("bio_01", "bio_1"), ("bio_12", "bio_12"),
                ("bio_13", "bio_13"), ("bio_14", "bio_14"), ("bio_15", "bio_15")]

BASINS = ["taltal", "choapa"]


def reproject_clim_to_dem(src_path: Path, dem_path: Path, out_path: Path) -> None:
    """Reproject + resample WorldClim raster to match DEM grid."""
    NODATA = -9999.0
    with rasterio.open(dem_path) as dst_ref, rasterio.open(src_path) as src:
        out = np.full((dst_ref.height, dst_ref.width), NODATA, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_ref.transform,
            dst_crs=dst_ref.crs,
            dst_nodata=NODATA,
            resampling=Resampling.bilinear,
        )
        profile = dst_ref.profile.copy()
        profile.update(dtype="float32", count=1, compress="deflate", nodata=NODATA)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out, 1)


def encode_lithology(resumen, encoding):
    return encoding["lithology_classes"].get(str(resumen).upper().strip(), 0)


def encode_rock(roca1, encoding):
    if roca1 is None or str(roca1).lower() == "nan":
        return 0
    # encoding stores 8-char prefixed names; match accordingly
    rt = str(roca1).lower().strip()
    rt8 = rt[:8]
    if rt in encoding["rock_types"]:
        return encoding["rock_types"][rt]
    if rt8 in encoding["rock_types"]:
        return encoding["rock_types"][rt8]
    return 0


def encode_age(edad, encoding):
    if edad is None or str(edad).lower() == "nan":
        return 0
    return encoding["geological_ages"].get(str(edad), 0)


def rasterize_geology(basin: str, dem_path: Path, geology_gdf: gpd.GeoDataFrame,
                       encoding: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dem_path) as dem:
        bounds = dem.bounds
        from shapely.geometry import box
        bbox_geom = box(*bounds)
        clip = geology_gdf[geology_gdf.intersects(bbox_geom)].copy()
        print(f"  [{basin}] geology polygons in bbox: {len(clip)}")
        if len(clip) == 0:
            return

        clip["lith_id"] = clip["RESUMEN"].apply(lambda v: encode_lithology(v, encoding))
        clip["rock_id"] = clip["ROCA1"].apply(lambda v: encode_rock(v, encoding))
        clip["age_id"]  = clip["EDAD"].apply(lambda v: encode_age(v, encoding))

        for field, fname in [
            ("lith_id", "lithology_class.tif"),
            ("rock_id", "rock_type.tif"),
            ("age_id",  "geological_age.tif"),
        ]:
            shapes = [(g, int(v)) for g, v in zip(clip.geometry, clip[field])
                      if g is not None and not g.is_empty]
            arr = rasterize(
                shapes,
                out_shape=(dem.height, dem.width),
                transform=dem.transform,
                fill=0,
                dtype="uint8",
            )
            profile = dem.profile.copy()
            profile.update(dtype="uint8", count=1, compress="deflate", nodata=255)
            out_path = out_dir / fname
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(arr, 1)
            n_classes = len(np.unique(arr[arr > 0]))
            print(f"    [{basin}] {fname}: {n_classes} unique classes")


def main():
    encoding = json.loads(ENCODING_JSON.read_text())
    print("Loading SERNAGEOMIN polygons...")
    geology = gpd.read_file(GEOLOGY_SHP)
    if geology.crs != "EPSG:32719":
        geology = geology.to_crs("EPSG:32719")
    print(f"  Total polygons: {len(geology)}")

    for basin in BASINS:
        dem_path = ROOT / f"factors/{basin}/dem_30m.tif"
        if not dem_path.exists():
            print(f"[skip] {basin}: DEM missing")
            continue
        print(f"\n=== {basin} ===")

        # CLIMATE
        clim_dir = ROOT / f"factors/{basin}/climate"
        for out_name, src_var in CLIMATE_VARS:
            src = WORLDCLIM / f"wc2.1_2.5m_{src_var}.tif"
            out = clim_dir / f"{out_name}.tif"
            if out.exists():
                print(f"  [skip] {out.name}")
                continue
            if not src.exists():
                print(f"  [WARN] missing source {src.name}")
                continue
            reproject_clim_to_dem(src, dem_path, out)
            print(f"  [done] climate/{out.name}")

        # GEOLOGY
        geo_dir = ROOT / f"factors/{basin}/geology"
        rasterize_geology(basin, dem_path, geology, encoding, geo_dir)
        # copy encoding.json for reference
        shutil.copy(ENCODING_JSON, geo_dir / "encoding.json")
        print(f"  [done] geology/encoding.json")


if __name__ == "__main__":
    main()
