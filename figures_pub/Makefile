.PHONY: all main supp clean

PYTHON = python3
SRC = src
OUT = out

# Script stems and output PDF names differ for some figures (e.g.
# fig03_lift_vs_k.py -> fig03_spatial_lift_vs_k.pdf), so figures are built by
# running every script rather than via a pattern rule.

MAIN_SCRIPTS = $(wildcard $(SRC)/fig[0-9]*.py)
SUPP_SCRIPTS = $(wildcard $(SRC)/figS*.py)

all: main supp

main:
	@for f in $(MAIN_SCRIPTS); do echo "== $$f"; $(PYTHON) $$f || exit 1; done

supp:
	@for f in $(SUPP_SCRIPTS); do echo "== $$f"; $(PYTHON) $$f || exit 1; done

clean:
	rm -rf $(OUT)/*.pdf $(OUT)/*.png
