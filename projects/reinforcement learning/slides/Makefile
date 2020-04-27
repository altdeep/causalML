.PHONY: all clean

SOURCES = $(wildcard *.tex)
TARGETS = $(SOURCES:%.tex=%.png)

all: $(TARGETS)

clean:
	rm -f *.aux *.log *.pdf *.png

%.png: %.tex
	pdflatex -shell-escape $<
