# Publication-quality figures pipeline — Paper 4

Pipeline para regenerar las figuras del paper con calidad publication-ready,
eliminando los telltale signs de matplotlib-default.

## Estructura

```
figures_pub/
├── utils/
│   └── style.py          # rcParams global, Wong palette, helpers
├── src/
│   ├── fig01_study_area.py
│   ├── fig02_f1_vs_k_panel.py
│   ├── fig03_spatial_lift_vs_k.py    ← killer figure (done)
│   ├── fig04_adaptation_curves.py
│   ├── fig05_spatial_vs_random.py
│   └── figS*_*.py
├── out/                  # generated PDF + PNG
└── Makefile
```

## Workflow

```bash
# Setup once
pip install --user matplotlib scienceplots cmocean

# Generate all figures
cd figures_pub
make all

# Specific figure
make out/fig03_spatial_lift_vs_k.pdf

# Clean
make clean
```

## Style guide aplicado

### Palette
- **Wong** (colorblind-safe, 8 colors) para métodos categóricos
- **Tol Bright** para highlights
- **RdBu_r** divergente para AUC/correlation
- **cmocean.thermal** para sequential ordered

### Method colors (consistent across all figs)
| Method | Color |
|---|---|
| Independent | Black `#000000` |
| Fine-tune | Sky blue `#56B4E9` |
| Reptile | Orange `#E69F00` |
| FOMAML | Pink `#CC79A7` (highlight) |
| DANN | Green `#009E73` |

### Typography
- **Helvetica** / Arial sans-serif (matches RSE body text)
- Hierarchy: panel labels 11pt bold, axes 9pt, ticks 8pt, annotations 7pt
- No `ax.set_title()` — use `\caption{...}` in LaTeX

### Layout
- **Single column RSE**: width = 88mm = 3.46 in
- **Double column RSE**: width = 180mm = 7.09 in
- Spines: bottom + left only
- Grid: subtle y-only `alpha=0.25`
- Legend: manual positioning, no `loc="best"`

### Annotations
Cada figura main paper tiene ≥1 annotation que apunta a feature específico:
- Fig 3: "+5.5 pp" callout at FOMAML K=1 peak + "advantage vanishes" arrow at K=10
- Fig 4: "F1=0.78 at step 0" callout for FOMAML
- Fig 5: drop magnitude annotations per basin

### Output
- **PDF vector** (preferred for LaTeX `\includegraphics`)
- **PNG 300 DPI** (preview, backup, online media)
- `pdf.fonttype=42` → editable in Illustrator
- `savefig.bbox='tight'` → no whitespace

## Checklist pre-submission

- [ ] Figuras en PDF vector
- [ ] Wong/Tol palettes (no `tab10` defaults)
- [ ] Helvetica/Arial typography
- [ ] Spines bottom+left only
- [ ] Anotaciones manuales pointing to findings
- [ ] Panel labels (a), (b), (c) bold top-left
- [ ] No `ax.set_title()` (use LaTeX caption)
- [ ] Single-column width consistent
- [ ] Aspect ratios consistent across all figs
- [ ] Colorblind preview validated (Color Oracle or coblis)
- [ ] All references to figures match in main text

## Estado

| Fig | Status | Mejora vs original |
|---|---|---|
| **fig03_spatial_lift_vs_k** | ✅ Done | Wong palette + annotations + single-col width |
| fig01_study_area | TODO | Cartopy LCC + numbered basins + inset |
| fig02_f1_vs_k_panel | TODO | 2×2 grid Wong + panel labels + manual legend |
| fig04_adaptation_curves | TODO | 2×2 grid + annotation "F1=0.78 at step 0" |
| fig05_spatial_vs_random | TODO | Subtle dashed/solid distinction |
| Sup figs | TODO | Same style.py applied |

## Notas

- Skill source: `/home/franciscoparrao/.claude/skills/paper-figures`
- Style based on: Wong 2011 Nature Methods, Tol color schemes, SciencePlots
- Reference papers replicated style: Rußwurm 2020 RSE, Reichenbach 2018 ESR
