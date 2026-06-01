"""Figure 1 (main): Study area map — 14 Chilean basins along an extreme
hyper-arid (Atacama) to hyper-humid (Patagonia) climate gradient.

Markers placed at true polygon centroids (no hardcoded coordinates).
Map extent auto-fit to basin geometries. Latitude-corrected aspect.
Numbered N->S to avoid label overlap. Inset shows location in South America.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LogNorm
from rasterio.mask import mask as rio_mask

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from style import COLOR_WONG, JOURNAL_WIDTH, save_pub, setup_style

ROOT = Path(__file__).parent.parent.parent
BNA = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper1_susceptibilidad/Cuencas_BNA/Cuencas_BNA.shp")
WORLDCLIM = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper1_susceptibilidad/factors/_worldclim/wc2.1_2.5m_bio_12.tif")
OUT = ROOT / "figures_pub" / "out" / "fig01_study_area"

# (slug, BNA cod_cuen, label, role) — numbered N->S automatically after centroid lookup
BASINS = [
    ("tilviche",    "022", "Tilviche-Loa",  "source"),
    ("chanaral",    "027", "Chañaral",      "source"),
    ("caracoles",   "023", "Loa-Caracoles", "source"),
    ("taltal",      "029", "Taltal",        "source"),
    ("copiapo",     "030", "Copiapó",       "target"),
    ("huasco",      "032", "Huasco",        "target"),
    ("elqui",       "043", "Elqui",         "target"),
    ("limari",      "045", "Limarí",        "target"),
    ("choapa",      "047", "Choapa",        "source"),
    ("maipo",       "060", "Maipo",         "source"),
    ("rapel",       "061", "Rapel",         "source"),
    ("bueno_puelo", "072", "Bueno-Puelo",   "source"),
    ("maule",       "073", "Maule",         "source"),
    ("magallanes",  "120", "Magallanes",    "source"),
]


def basin_mean_precip(geom, p):
    with rasterio.open(p) as src:
        try:
            arr, _ = rio_mask(src, [geom], crop=True, nodata=src.nodata, filled=True)
            arr = arr[0]
            valid = arr[(arr != src.nodata) & np.isfinite(arr)]
            return float(valid.mean()) if len(valid) else float("nan")
        except Exception:
            return float("nan")


def main():
    setup_style(journal="rse", base_fontsize=8)

    bna = gpd.read_file(BNA).to_crs("EPSG:4326")
    rows = []
    for slug, cod, label, role in BASINS:
        poly = bna[bna["COD_CUEN"] == cod]
        if len(poly) == 0:
            print(f"  [warn] BNA {cod} ({label}) not found")
            continue
        geom = poly.geometry.union_all() if hasattr(poly.geometry, "union_all") else poly.geometry.unary_union
        c = geom.centroid
        rows.append({
            "slug": slug, "label": label, "role": role, "geometry": geom,
            "precip_mm": basin_mean_precip(geom, WORLDCLIM),
            "lat": c.y, "lon": c.x,
        })
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    # Number N -> S
    gdf = gdf.sort_values("lat", ascending=False).reset_index(drop=True)
    gdf["num"] = gdf.index + 1

    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        chile_outline = world[world["name"] == "Chile"]
        sam_outline = world[world["continent"] == "South America"]
    except Exception:
        world = None; chile_outline = None; sam_outline = None

    vmax = max(float(np.nanmax(gdf["precip_mm"].values)), 2700)
    norm = LogNorm(vmin=1, vmax=vmax)
    cmap = plt.cm.YlGnBu

    # Single full-Chile panel. No set_aspect (avoids box-shrink marker misalignment);
    # axes box fills the gridspec cell, so longitude is mildly stretched (fine for a locator map).
    minx, miny, maxx, maxy = gdf.total_bounds
    xlim = (minx - 1.5, maxx + 1.5)
    ylim = (miny - 1.2, maxy + 1.2)

    w = JOURNAL_WIDTH["rse"]["single"]
    fig = plt.figure(figsize=(w * 1.65, w * 2.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.25], wspace=0.04)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_main.set_facecolor("#D6E4EC")
    ax_main.set_xlim(*xlim); ax_main.set_ylim(*ylim)

    if chile_outline is not None:
        other = world[world["continent"] == "South America"]
        other = other[other["name"] != "Chile"]
        other.plot(ax=ax_main, color="#EEE", edgecolor="#888", linewidth=0.4, zorder=0)
        chile_outline.plot(ax=ax_main, color="#F5F2EC", edgecolor="#444",
                           linewidth=0.6, zorder=1)
    for _, r in gdf.iterrows():
        gpd.GeoDataFrame([r], geometry="geometry", crs="EPSG:4326").plot(
            ax=ax_main, color=cmap(norm(max(r["precip_mm"], 1))),
            edgecolor="#222", linewidth=0.5, zorder=3)
    for _, r in gdf.iterrows():
        edge = COLOR_WONG["red"] if r["role"] == "source" else COLOR_WONG["blue"]
        ax_main.scatter(r["lon"], r["lat"], marker="s" if r["role"] == "source" else "o",
                        s=60, facecolor="white", edgecolor=edge, linewidth=1.2, zorder=5)
        ax_main.text(r["lon"], r["lat"], str(r["num"]), fontsize=6,
                     fontweight="bold", color=edge, ha="center", va="center", zorder=6)
    ax_main.grid(True, alpha=0.25, linewidth=0.3, linestyle=":", color="#888", zorder=1)
    ax_main.tick_params(labelsize=7)
    ax_main.set_xlabel("Longitude (°W)", fontsize=8)
    ax_main.set_ylabel("Latitude (°S)", fontsize=8)

    # Horizontal colorbar below the map
    cbar_ax = fig.add_axes([0.13, 0.055, 0.38, 0.008])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Annual precipitation (mm/yr)", fontsize=7, labelpad=1)
    cbar.ax.tick_params(labelsize=6, pad=1)

    # ---- Sidebar: numbered basin list + role legend + inset ----
    ax_side = fig.add_subplot(gs[0, 1])
    ax_side.axis("off")

    y = 0.97
    ax_side.text(0.02, y, "Basins (N$\\rightarrow$S)", fontsize=8.5,
                 fontweight="bold", transform=ax_side.transAxes)
    y -= 0.035
    for _, r in gdf.iterrows():
        role_color = COLOR_WONG["red"] if r["role"] == "source" else COLOR_WONG["blue"]
        ax_side.text(0.02, y, f"{r['num']:>2d}  {r['label']}",
                     fontsize=7, family="monospace",
                     transform=ax_side.transAxes, va="top")
        ax_side.text(0.97, y, "■" if r["role"] == "source" else "●",
                     fontsize=8, color=role_color,
                     transform=ax_side.transAxes, va="top", ha="right")
        y -= 0.032

    y -= 0.025
    ax_side.text(0.05, y, "■ Source (meta-train)", fontsize=6.5,
                 color=COLOR_WONG["red"], transform=ax_side.transAxes, va="top")
    y -= 0.028
    ax_side.text(0.05, y, "● Target (K-shot)", fontsize=6.5,
                 color=COLOR_WONG["blue"], transform=ax_side.transAxes, va="top")

    # Inset: South America
    inset_ax = fig.add_axes([0.70, 0.14, 0.22, 0.16])
    inset_ax.set_facecolor("#D6E4EC")
    inset_ax.set_xlim(-82, -33); inset_ax.set_ylim(-56, 14)
    inset_ax.set_aspect(1.0 / math.cos(math.radians(25)), adjustable="box")
    if sam_outline is not None:
        sam_outline.plot(ax=inset_ax, color="#EEE", edgecolor="#666", linewidth=0.3)
        if chile_outline is not None:
            chile_outline.plot(ax=inset_ax, color="#FBE3CC", edgecolor="#444", linewidth=0.5)
    inset_ax.add_patch(plt.Rectangle((xlim[0], ylim[0]),
                                     xlim[1] - xlim[0], ylim[1] - ylim[0],
                                     fill=False, edgecolor=COLOR_WONG["red"],
                                     linewidth=1.0, zorder=3))
    inset_ax.set_xticks([]); inset_ax.set_yticks([])
    for s in ["top", "right", "bottom", "left"]:
        inset_ax.spines[s].set_linewidth(0.4)

    save_pub(fig, OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
