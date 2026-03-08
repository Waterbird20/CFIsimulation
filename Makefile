STEM = circuit_full_ps

.PHONY: all clean

all: $(STEM)_quantikz.svg

$(STEM).pdf: $(STEM).tex
	pdflatex -interaction=nonstopmode $(STEM).tex

$(STEM)_quantikz.svg: $(STEM).pdf
	pdftocairo -svg $(STEM).pdf $(STEM)_quantikz.svg

clean:
	rm -f $(STEM).aux $(STEM).log $(STEM).fls $(STEM).fdb_latexmk \
	       $(STEM).pdf $(STEM)_quantikz.svg
