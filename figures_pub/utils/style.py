"""Publication-quality matplotlib style for Paper 4.

Usage at the top of every figure script:
    from style import setup_style, COLOR_WONG, COLOR_TOL, JOURNAL_WIDTH
    setup_style(journal="rse")   # "rse" | "isprs_j" | "rsase"

Designed to eliminate matplotlib-default telltale signs:
- Wong palette (colorblind-safe) for categorical
- Tol Bright for highlights
- Diverging RdBu_r for symmetric metrics
- Sequential cmocean.thermal for ordered metrics
- No spines top/right
- Subtle grid (alpha=0.25, linewidth=0.4)
- 300 DPI raster, vector PDF preferred
- Font sizes tuned to journal column widths
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Wong (2011, Nature Methods) — colorblind-safe categorical (8 colors)
COLOR_WONG = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "red":     "#D55E00",
    "pink":    "#CC79A7",
}

# Paul Tol Bright — 7 colors, colorblind-safe
COLOR_TOL = {
    "blue":   "#4477AA",
    "cyan":   "#66CCEE",
    "green":  "#228833",
    "yellow": "#CCBB44",
    "red":    "#EE6677",
    "purple": "#AA3377",
    "grey":   "#BBBBBB",
}

# Semantic mapping for methods comparison (consistent across all figures)
METHOD_COLORS = {
    "independent":   COLOR_WONG["black"],   # baseline reference
    "finetune":      COLOR_WONG["skyblue"], # transfer learning
    "reptile":       COLOR_WONG["orange"],  # meta-learning #1
    "fomaml":        COLOR_WONG["pink"],    # meta-learning #2 (highlight)
    "dann":          COLOR_WONG["green"],   # domain-adversarial
    # Extended baselines (added for ISPRS JPRS R1)
    "protonet":      COLOR_WONG["blue"],    # meta-learning #3 (metric-based)
    "meta_baseline": COLOR_TOL["purple"],   # meta-learning #4 (frozen encoder)
    "cdan":          COLOR_WONG["red"],     # domain-adversarial #2 (conditional)
}

# Method display labels
METHOD_LABELS = {
    "independent":   "Independent",
    "finetune":      "Fine-tune",
    "reptile":       "Reptile",
    "fomaml":        "FOMAML",
    "dann":          "DANN",
    "protonet":      "ProtoNet",
    "meta_baseline": "Meta-Baseline",
    "cdan":          "CDAN",
}

# Journal column widths in inches
JOURNAL_WIDTH = {
    "rse":         {"single": 88/25.4,  "double": 180/25.4},  # Elsevier
    "isprs_j":     {"single": 90/25.4,  "double": 190/25.4},  # Elsevier
    "rsase":       {"single": 90/25.4,  "double": 190/25.4},  # Elsevier
    "ldd":         {"single": 80/25.4,  "double": 168/25.4},  # Wiley
    "geomorpho":   {"single": 90/25.4,  "double": 190/25.4},  # Elsevier
    "nhess":       {"single": 84/25.4,  "double": 175/25.4},  # Copernicus
    "default":     {"single": 88/25.4,  "double": 180/25.4},  # Generic ~3.46 / 7.09 in
}


def setup_style(journal: str = "rse", base_fontsize: int = 9) -> None:
    """Configure matplotlib rcParams for publication-quality output.

    Args:
        journal: target journal key in JOURNAL_WIDTH
        base_fontsize: base font size in points (9 typical for Elsevier)
    """
    mpl.rcdefaults()

    # Font
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":       base_fontsize,
        "axes.titlesize":  base_fontsize + 1,
        "axes.labelsize":  base_fontsize,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "legend.fontsize": base_fontsize - 1,
        "figure.titlesize": base_fontsize + 2,
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
    })

    # Spines (only bottom + left)
    mpl.rcParams.update({
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.bottom": True,
        "axes.spines.left":   True,
        "axes.linewidth":     0.8,
        "axes.edgecolor":     "#222222",
    })

    # Ticks (inside, narrow)
    mpl.rcParams.update({
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
    })

    # Grid (subtle, only y by default — many figures don't need x grid)
    mpl.rcParams.update({
        "axes.grid":      False,
        "grid.alpha":     0.25,
        "grid.linewidth": 0.4,
        "grid.linestyle": "-",
        "grid.color":     "#888888",
    })

    # Lines and markers
    mpl.rcParams.update({
        "lines.linewidth":   1.6,
        "lines.markersize":  5,
        "lines.markeredgewidth": 0.5,
    })

    # Legend
    mpl.rcParams.update({
        "legend.frameon":   False,
        "legend.handlelength": 1.5,
        "legend.borderpad": 0.4,
    })

    # Figure / save
    mpl.rcParams.update({
        "figure.dpi":     120,
        "savefig.dpi":    300,
        "savefig.bbox":   "tight",
        "savefig.format": "pdf",
        "pdf.fonttype":   42,    # TrueType, editable in Illustrator
        "ps.fonttype":    42,
    })


def add_panel_label(ax, label, x=-0.12, y=1.05, fontsize=11):
    """Add bold panel label (a, b, c, ...) at top-left of axis."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize, fontweight="bold",
        va="top", ha="left",
    )


def style_ax(ax, *, x_grid=False, y_grid=True):
    """Apply consistent axis styling (call after plotting)."""
    if y_grid:
        ax.grid(axis="y", alpha=0.25, linewidth=0.4, linestyle="-", color="#888888")
    if x_grid:
        ax.grid(axis="x", alpha=0.25, linewidth=0.4, linestyle="-", color="#888888")
    ax.set_axisbelow(True)


def save_pub(fig, path, formats=("pdf", "png")):
    """Save figure in vector PDF (preferred) + 300 DPI PNG (for previews)."""
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    stem = p.with_suffix("")
    for fmt in formats:
        out = stem.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=300 if fmt == "png" else None, bbox_inches="tight")
        print(f"  → {out}")
