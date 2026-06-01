#!/usr/bin/env python3
"""Paper 4 — Figure 1: study area map (8 Andean basins, climate gradient).

- 8 basin polygons from BNA shapefile + paper4 extracts (Taltal, Choapa).
- Color-coded by mean annual precipitation (bio_12) as climate proxy.
- Source vs target distinguished by hatch / outline.
- South America inset for geographic context.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
BNA = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper1_susceptibilidad/Cuencas_BNA/Cuencas_BNA.shp")
WORLDCLIM_BIO12 = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper1_susceptibilidad/factors/_worldclim/wc2.1_2.5m_bio_12.tif")

# Basin metadata: (slug, BNA cod_cuen, label, role, lat_centroid, lon_centroid)
BASINS = [
    ("chanaral", "027", "Chañaral",  "source", -26.3, -69.6),
    ("taltal",   "029", "Taltal",     "source", -25.4, -69.9),
    ("copiapo",  "030", "Copiapó",    "target", -27.5, -69.7),
    ("huasco",   "032", "Huasco",     "target", -28.7, -70.2),
    ("elqui",    "043", "Elqui",      "target", -30.0, -70.4),
    ("limari",   "045", "Limarí",     "target", -30.8, -70.6),
    ("choapa",   "047", "Choapa",     "source", -31.6, -70.8),
    ("maule",    "073", "Maule",      "source", -35.5, -71.5),
]

OUT = ROOT / "figures" / "study_area.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


def basin_mean_precip(geom, bio12_path):
    """Compute basin-mean annual precipitation from WorldClim bio_12."""
    with rasterio.open(bio12_path) as src:
        try:
            arr, _ = rio_mask(src, [geom], crop=True, nodata=src.nodata,
                              filled=True)
            arr = arr[0]
            valid = arr[(arr != src.nodata) & np.isfinite(arr)]
            if len(valid) == 0:
                return float("nan")
            return float(valid.mean())
        except Exception:
            return float("nan")


def main():
    bna = gpd.read_file(BNA).to_crs("EPSG:4326")

    rows = []
    for slug, cod, label, role, lat, lon in BASINS:
        poly = bna[bna["COD_CUEN"] == cod]
        if len(poly) == 0:
            print(f"  WARN: {slug} (cod={cod}) not found in BNA")
            continue
        geom = poly.geometry.unary_union
        precip = basin_mean_precip(geom, WORLDCLIM_BIO12)
        rows.append({
            "slug": slug, "label": label, "role": role,
            "geometry": geom, "precip_mm": precip,
            "lat": lat, "lon": lon,
        })
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    print("\nBasin mean annual precipitation (mm/yr):")
    for _, r in gdf.iterrows():
        print(f"  {r['label']:<12} role={r['role']:<7} precip≈{r['precip_mm']:6.1f} mm")

    # Map figure
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 250})
    fig, (ax_main, ax_inset) = plt.subplots(
        1, 2, figsize=(12, 8),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    # Main map: color by precipitation (log scale for arid → humid)
    vmax = float(np.nanmax(gdf["precip_mm"].values))
    norm = plt.matplotlib.colors.LogNorm(vmin=10, vmax=max(vmax, 1500))
    cmap = plt.cm.YlGnBu
    gdf.plot(ax=ax_main, column="precip_mm", cmap=cmap, norm=norm,
             edgecolor="black", linewidth=0.6,
             legend=True,
             legend_kwds={"label": "Mean annual precipitation (mm/yr)",
                          "shrink": 0.6, "orientation": "horizontal", "pad": 0.06})

    # Source / target markers
    for _, r in gdf.iterrows():
        marker = "s" if r["role"] == "source" else "o"
        edge = "red" if r["role"] == "source" else "blue"
        size = 130 if r["role"] == "source" else 110
        ax_main.scatter(r["lon"], r["lat"], marker=marker, s=size,
                        facecolor="white", edgecolor=edge, linewidth=2.0, zorder=5)
        ax_main.text(r["lon"] + 0.3, r["lat"] + 0.05, r["label"],
                     fontsize=10, fontweight="bold",
                     ha="left", va="center", zorder=6,
                     bbox=dict(facecolor="white", alpha=0.7,
                               edgecolor="none", pad=2))

    ax_main.set_xlim(-72.5, -68.5)
    ax_main.set_ylim(-36.5, -24)
    ax_main.set_xlabel("Longitude (°W)")
    ax_main.set_ylabel("Latitude (°S)")
    ax_main.set_title("Study basins — Andean Chile")
    ax_main.grid(True, alpha=0.3, linestyle=":")

    # Legend for source/target
    leg_handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="white",
                   markeredgecolor="red", markersize=12, markeredgewidth=2,
                   label="Source basin (meta-train)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                   markeredgecolor="blue", markersize=11, markeredgewidth=2,
                   label="Target basin (K-shot eval)"),
    ]
    ax_main.legend(handles=leg_handles, loc="lower right", frameon=True,
                   fontsize=9, framealpha=0.9)

    # Inset: South America with rectangle of zoom area
    ax_inset.set_xlim(-82, -33)
    ax_inset.set_ylim(-58, 14)
    # Country outlines via natural earth (low res via geopandas built-in)
    try:
        # Use cached natural earth countries shipped with geopandas
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        sam = world[world["continent"] == "South America"]
        sam.plot(ax=ax_inset, color="lightgray", edgecolor="black", linewidth=0.5)
        chile = world[world["name"] == "Chile"]
        chile.plot(ax=ax_inset, color="#fdbb84", edgecolor="black", linewidth=0.7)
    except Exception:
        # Fallback: just draw a box for Chile
        ax_inset.add_patch(plt.Rectangle((-75, -55), 7, 38,
                                         fill=True, color="#fdbb84",
                                         edgecolor="black", linewidth=0.7))
    # Rectangle for zoom region
    ax_inset.add_patch(plt.Rectangle(
        (-72.5, -36.5), -68.5 - (-72.5), -24 - (-36.5),
        fill=False, edgecolor="red", linewidth=1.8))
    ax_inset.set_xticks([]); ax_inset.set_yticks([])
    ax_inset.set_aspect("equal")
    ax_inset.set_title("Location", fontsize=10)

    fig.suptitle("Eight Andean basins span a hyper-arid (north) to semi-humid (south) gradient",
                 y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
