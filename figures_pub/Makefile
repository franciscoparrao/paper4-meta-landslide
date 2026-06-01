.PHONY: all clean figs main supp

PYTHON = python3
SRC = src
OUT = out

# Main paper figures (5)
MAIN_FIGS = \
	$(OUT)/fig01_study_area.pdf \
	$(OUT)/fig02_f1_vs_k_panel.pdf \
	$(OUT)/fig03_spatial_lift_vs_k.pdf \
	$(OUT)/fig04_adaptation_curves.pdf \
	$(OUT)/fig05_spatial_vs_random.pdf

# Supplementary figures (7+)
SUPP_FIGS = \
	$(OUT)/figS1_auc_vs_k_panel.pdf \
	$(OUT)/figS2_spatial_f1_vs_k_panel.pdf \
	$(OUT)/figS3_adaptation_curves_K10.pdf \
	$(OUT)/figS4_ablations.pdf \
	$(OUT)/figS5_dann_comparison.pdf \
	$(OUT)/figS6_summary_heatmap.pdf \
	$(OUT)/figS7_classical_vs_meta.pdf \
	$(OUT)/figS8_climate_distance.pdf \
	$(OUT)/figS9_elqui_stratification.pdf

all: main supp

main: $(MAIN_FIGS)

supp: $(SUPP_FIGS)

$(OUT)/%.pdf: $(SRC)/%.py utils/style.py
	$(PYTHON) $<

clean:
	rm -rf $(OUT)/*.pdf $(OUT)/*.png
