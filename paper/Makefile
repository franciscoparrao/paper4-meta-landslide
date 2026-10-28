.PHONY: all paper figures clean

MAIN = paper4_main
LATEX = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX = bibtex

all: figures paper

figures:
	$(MAKE) -C ../figures_pub all

paper: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex references.bib
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).spl $(MAIN).toc $(MAIN).nav $(MAIN).snm \
	      sections/*.aux

cleanall: clean
	rm -f $(MAIN).pdf
