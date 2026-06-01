#!/usr/bin/env python3
"""Paper 4 — unify per-basin landslide inventories to common schema.

Output schema (EPSG:4326):
    geometry  - Point or Polygon
    label     - 1 (positive landslide) or 0 (negative)
    source    - origin file/dataset
    basin     - basin slug
    geom_type - 'point' or 'polygon'

Saves to data/inventarios/{basin}.gpkg
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
PAPER1_INV = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper1_susceptibilidad/basin_inventory")
LANDSLIDE_DS = Path("/home/franciscoparrao/proyectos/landslide_dataset")
BNA = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper1_susceptibilidad/Cuencas_BNA/Cuencas_BNA.shp")
OUT_DIR = ROOT / "data" / "inventarios"
OUT_DIR.mkdir(parents=True, exist_ok=True)

bna = gpd.read_file(BNA).to_crs("EPSG:4326")


def to_wgs84_records(gdf, label, source, basin):
    g = gdf.to_crs("EPSG:4326").copy()
    g["label"] = int(label)
    g["source"] = source
    g["basin"] = basin
    g["geom_type"] = g.geometry.geom_type.str.lower()
    g["geom_type"] = g["geom_type"].replace({"multipolygon": "polygon", "multipoint": "point"})
    return g[["geometry", "label", "source", "basin", "geom_type"]]


def write_basin(records, basin):
    if not records:
        print(f"  [skip] {basin}: no records")
        return
    out = pd.concat(records, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")
    # Force object dtype on string columns for fiona compatibility (StringDtype breaks fiona)
    for col in ("source", "basin", "geom_type"):
        out[col] = pd.Series([str(v) for v in out[col]], dtype="O")
    out_path = OUT_DIR / f"{basin}.gpkg"
    out.to_file(out_path, driver="GPKG")
    counts = out.groupby(["label", "geom_type"]).size().to_dict()
    print(f"  [done] {basin}: N={len(out)} {counts} → {out_path.name}")


def chanaral():
    g = gpd.read_file(ROOT / "data/chañaral/sustainability-2726284-supplementary.gpkg",
                      layer="puntos_utm")
    pos = g[g["REMOCION"] == 1]
    neg = g[g["REMOCION"] == 0]
    src = "sustainability_2023"
    return [
        to_wgs84_records(pos, label=1, source=src, basin="chanaral"),
        to_wgs84_records(neg, label=0, source=src, basin="chanaral"),
    ]


def taltal():
    poly = gpd.read_file(ROOT / "data/taltal/poligonos_remociones.gpkg")
    neg = gpd.read_file(ROOT / "data/taltal/puntos_no_remociones_poligonos.shp")
    return [
        to_wgs84_records(poly, label=1, source="tesis_taltal_poligonos", basin="taltal"),
        to_wgs84_records(neg[["geometry"]], label=0,
                         source="tesis_taltal_neg_puntos", basin="taltal"),
    ]


def maule():
    base = LANDSLIDE_DS / ("landslide_earthquake_2010/Comprehensive earthquake-induced "
                           "landslide inventory dataset of the 2010 Chile megathrust earthquake")
    shps = [
        ("Maule2010_Earthquake-induced_DisruptedSlides_ UTM19S_1059.shp", "disrupted"),
        ("Maule2010_Earthquake-induced_Flows_ UTM19S_110.shp", "flows"),
        ("Maule2010_Earthquake-induced_LateralSpreads_UTM19S_49.shp", "lateral_spreads"),
        ("Maule2010_Earthquake-induced_CoherentSlides_ UTM19S_9.shp", "coherent"),
    ]
    poly_maule = bna[bna["COD_CUEN"] == "073"].geometry.unary_union
    out = []
    for fname, typology in shps:
        g = gpd.read_file(base / fname).to_crs("EPSG:4326")
        inside = g[g.within(poly_maule)]
        if len(inside) == 0:
            continue
        out.append(to_wgs84_records(
            inside[["geometry"]],
            label=1, source=f"maule2010_{typology}", basin="maule",
        ))
    return out


def choapa():
    cat = gpd.read_file(LANDSLIDE_DS / "catastro_remociones/catastro.geojson")
    poly_choapa = bna[bna["COD_CUEN"] == "047"].geometry.unary_union
    inside = cat[cat.within(poly_choapa)]
    return [to_wgs84_records(inside[["geometry"]], label=1,
                             source="catastro_sernageomin", basin="choapa")]


def paper1_csv(basin_slug, csv_name):
    df = pd.read_csv(PAPER1_INV / csv_name)
    g = gpd.GeoDataFrame(df,
                         geometry=gpd.points_from_xy(df["lon"], df["lat"]),
                         crs="EPSG:4326")
    return [to_wgs84_records(g[["geometry"]], label=1,
                             source=f"paper1_{csv_name}", basin=basin_slug)]


basins = {
    # Source basins (current 4 + new 6 from Paper 1)
    "chanaral":   chanaral,
    "taltal":     taltal,
    "maule":      maule,
    "choapa":     choapa,
    "tilviche":   lambda: paper1_csv("tilviche",        "02_costeras_tilviche_loa.csv"),
    "caracoles":  lambda: paper1_csv("caracoles",       "03_costeras_loa_caracoles.csv"),
    "maipo":      lambda: paper1_csv("maipo",           "09_rio_maipo.csv"),
    "rapel":      lambda: paper1_csv("rapel",           "10_rio_rapel.csv"),
    "bueno_puelo": lambda: paper1_csv("bueno_puelo",    "14_costeras_bueno_puelo.csv"),
    "magallanes": lambda: paper1_csv("magallanes",      "15_costeras_magallanes.csv"),
    # Target basins
    "copiapo":    lambda: paper1_csv("copiapo",  "05_rio_copiapo.csv"),
    "huasco":     lambda: paper1_csv("huasco",   "06_rio_huasco.csv"),
    "elqui":      lambda: paper1_csv("elqui",    "07_rio_elqui.csv"),
    "limari":     lambda: paper1_csv("limari",   "08_rio_limari.csv"),
}

for slug, builder in basins.items():
    print(f"=== {slug} ===")
    write_basin(builder(), slug)

print()
print("=== summary ===")
for slug in basins:
    p = OUT_DIR / f"{slug}.gpkg"
    if not p.exists():
        continue
    g = gpd.read_file(p)
    pos_pt  = ((g["label"] == 1) & (g["geom_type"] == "point")).sum()
    pos_pol = ((g["label"] == 1) & (g["geom_type"] == "polygon")).sum()
    neg_pt  = ((g["label"] == 0) & (g["geom_type"] == "point")).sum()
    print(f"{slug:<10} pos_pt={pos_pt:<5} pos_pol={pos_pol:<4} neg_pt={neg_pt:<5}  total={len(g)}")
