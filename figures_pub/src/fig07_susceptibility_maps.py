"""Figure 7 (main): full-basin susceptibility maps from K=1 adapted models.

2x2 panels: rows = target basins (Copiapo, Elqui), cols = Independent K=1 vs
Reptile K=1. Each map is the mean sigmoid over 3 seeds x 20 episodes
(scripts/25_susceptibility_maps.py). Inventory positives overlaid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from style import JOURNAL_WIDTH, add_panel_label, save_pub, setup_style

ROOT = Path(__file__).parent.parent.parent
MAPS = ROOT / "results" / "susceptibility_maps"
ML = ROOT / "data" / "ml_ready"
SPATIAL_CSV = ROOT / "results" / "spatial_benchmark" / "summary.csv"
BNA = Path("/mnt/kingston/proyectos/postdoc/papers/paper1_susceptibilidad/Cuencas_BNA/Cuencas_BNA.shp")
OUT = ROOT / "figures_pub" / "out" / "fig07_susceptibility_maps"

BASINS = [("copiapo", "Copiapó", "030"), ("elqui", "Elqui", "043")]
METHODS = [("independent", "Independent"), ("reptile", "Reptile (meta-learned)")]
CMAP = "magma"
MAX_PX = 1400  # decimated read width for plotting


def read_map(basin: str, method: str):
    """Read decimated map and convert to within-basin percentile (0-100).

    Raw sigmoid outputs of K=1-adapted models are uncalibrated (the Reptile
    mean map has a median near 0 in Copiapo while ranking correctly), so maps
    are displayed as within-basin percentiles - the standard presentation for
    susceptibility classes - keeping the two methods visually comparable.
    """
    p = MAPS / f"{basin}_{method}_k1_mean.tif"
    with rasterio.open(p) as src:
        f = max(1, int(np.ceil(max(src.shape) / MAX_PX)))
        arr = src.read(1, out_shape=(src.height // f, src.width // f)).astype(np.float32)
        b = src.bounds
    extent = (b.left / 1e3, b.right / 1e3, b.bottom / 1e3, b.top / 1e3)  # km
    valid = np.isfinite(arr)
    pct = np.full(arr.shape, np.nan, dtype=np.float32)
    v = arr[valid]
    order = v.argsort().argsort()  # ranks
    pct[valid] = 100.0 * order / max(len(v) - 1, 1)
    return pct, extent


def inventory_xy(basin: str):
    with h5py.File(ML / f"{basin}.h5") as f:
        lab = f["label"][:]
        x, y = f["x_utm"][:], f["y_utm"][:]
    return x[lab == 1] / 1e3, y[lab == 1] / 1e3


def f1_at_k1(df, basin, method):
    r = df[(df.target == basin) & (df.K == 1) & (df.method == method)]
    return float(r.f1_mean.iloc[0]) if len(r) else np.nan


def add_scalebar(ax, extent, km=50):
    x0 = extent[0] + 0.06 * (extent[1] - extent[0])
    y0 = extent[2] + 0.06 * (extent[3] - extent[2])
    ax.plot([x0, x0 + km], [y0, y0], color="white", lw=2.2, solid_capstyle="butt", zorder=5)
    ax.plot([x0, x0 + km], [y0, y0], color="black", lw=0.6, solid_capstyle="butt", zorder=6)
    ax.text(x0 + km / 2, y0 + 0.018 * (extent[3] - extent[2]), f"{km} km",
            ha="center", va="bottom", fontsize=6.5, color="white", zorder=6,
            path_effects=None,
            bbox=dict(facecolor="black", alpha=0.45, pad=1.0, edgecolor="none"))


def main():
    setup_style(journal="rse", base_fontsize=9)
    df = pd.read_csv(SPATIAL_CSV)
    bna = gpd.read_file(BNA).to_crs("EPSG:32719")

    w = JOURNAL_WIDTH["rse"]["double"]
    # Row aspect from basin bounding boxes; Copiapo wider than tall, Elqui similar
    fig, axes = plt.subplots(2, 2, figsize=(w, w * 0.86))
    panel = "abcd"
    im = None

    for i, (basin, blabel, cod) in enumerate(BASINS):
        geom = bna[bna["COD_CUEN"].astype(str).str.zfill(3) == cod]
        inv_x, inv_y = inventory_xy(basin)
        for j, (method, mlabel) in enumerate(METHODS):
            ax = axes[i, j]
            arr, extent = read_map(basin, method)
            im = ax.imshow(arr, extent=extent, origin="upper", cmap=CMAP,
                           vmin=0, vmax=100, interpolation="nearest", rasterized=True)
            # basin outline in km (imshow extent is in km)
            for g in geom.geometry:
                parts = g.geoms if g.geom_type == "MultiPolygon" else [g]
                for part in parts:
                    xs, ys = part.exterior.xy
                    ax.plot(np.asarray(xs) / 1e3, np.asarray(ys) / 1e3,
                            color="0.25", lw=0.5, zorder=4)
            ax.scatter(inv_x, inv_y, s=6, facecolor="#56B4E9", edgecolor="black",
                       linewidth=0.3, zorder=5,
                       label=f"Inventory (N={len(inv_x)})" if j == 0 else None)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_linewidth(0.4); s.set_color("0.4")
            f1 = f1_at_k1(df, basin, method)
            ax.set_title(f"{blabel} — {mlabel}", fontsize=8.5, pad=10)
            ax.text(0.975, 0.035, f"$F_1$ (spatial CV, $K$=1) = {f1:.3f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.45, pad=1.4, edgecolor="none"),
                    zorder=6)
            if j == 0:
                add_scalebar(ax, extent, km=50)
                ax.legend(loc="upper left", fontsize=6.5, frameon=True,
                          framealpha=0.85, borderpad=0.4, handletextpad=0.3)
            add_panel_label(ax, panel[i * 2 + j], x=-0.045, y=1.13)

    fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.10,
                        wspace=0.06, hspace=0.20)
    cax = fig.add_axes([0.30, 0.055, 0.40, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Susceptibility percentile within basin", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.4)

    save_pub(fig, OUT)


if __name__ == "__main__":
    main()
