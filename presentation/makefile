file = slides

all : $(file:=.tex)
	pdflatex $(file:=.tex)
	# @bibtex $(file:=.aux)
	pdflatex $(file:=.tex)
	pdflatex $(file:=.tex)

	@echo "\n\nCompleted compiling successfully!\n\n"
	ls -lgh

quiet : $(file:=.tex)
	pdflatex $(file:=.tex) 1> tmp.tmp
	# @bibtex $(file:=.aux) 1> tmp.tmp
	pdflatex $(file:=.tex) 1> tmp.tmp
	pdflatex $(file:=.tex) 1> tmp.tmp

	rm tmp.tmp

	@echo "\n\nCompleted compiling successfully!\n\n"
	ls -lgh

pdf : all
	evince $(file).pdf &
	
clean :
	rm $(file:=.aux) $(file:=.bbl) $(file:=.blg) || true
	rm $(file:=.ilg) $(file:=.lof) $(file:=.log) || true
	rm $(file:=.nav) $(file:=.lot) $(file:=.nlo) || true
	rm $(file:=.nls) $(file:=.out) $(file:=.toc) || true
	rm $(file:=.pdf) $(file:=.snm) || true

	ls -lgh

.PHONY: clean pdf all quiet

$(V).SILENT:
