#!/usr/bin/env bash
# Paper 4 (R1) — build the Editorial Manager package for the RSASE revision.
# Output: revision_R1/em_package/ (PDFs, LaTeX source zip, highlights, README)
set -euo pipefail
ROOT=/mnt/kingston/proyectos/postdoc/papers/paper4_meta_learning_transfer
PKG=$ROOT/revision_R1/em_package
SRC=$PKG/_latex_source
rm -rf "$PKG"; mkdir -p "$SRC/sections" "$SRC/figures"

latex_build () {   # $1 = dir, $2 = main stem
  ( cd "$1" && pdflatex -interaction=nonstopmode -halt-on-error "$2" >/dev/null \
      && bibtex "$2" >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error "$2" >/dev/null \
      && pdflatex -interaction=nonstopmode -halt-on-error "$2" >/dev/null )
}

# 1. highlighted manuscript (changes in blue): compile the working copy
latex_build "$ROOT/paper" paper4_main
cp "$ROOT/paper/paper4_main.pdf" "$PKG/02_revised_manuscript_marked_changes.pdf"

# 2. clean LaTeX source: same text, \revtext switched off, local figures/ path
sed -e 's|\\graphicspath{{../figures_pub/out/}}|\\graphicspath{{figures/}}|' \
    -e 's|^\\newcommand{\\revtextoff}{\\renewcommand{\\revtext}\[1\]{##1}}|&\n\\revtextoff|' \
    "$ROOT/paper/paper4_main.tex" > "$SRC/paper4_main.tex"
grep -q '^\\revtextoff$' "$SRC/paper4_main.tex" || { echo "revtextoff not inserted"; exit 1; }
cp "$ROOT"/paper/sections/*.tex "$SRC/sections/"
cp "$ROOT/paper/references.bib" "$ROOT/paper/elsarticle.cls" "$SRC/"
cp "$(kpsewhich elsarticle-harv.bst)" "$SRC/"
for f in $(grep -o 'includegraphics\[[^]]*\]{[^}]*}' "$ROOT/paper/paper4_main.tex" | sed 's/.*{//; s/}//' | sort -u); do
  cp "$ROOT/figures_pub/out/$f" "$SRC/figures/"
done
latex_build "$SRC" paper4_main
cp "$SRC/paper4_main.pdf" "$PKG/03_revised_manuscript_clean.pdf"
( cd "$SRC" && rm -f *.aux *.log *.out *.blg *.spl paper4_main.pdf && zip -qr "$PKG/04_latex_source_R1.zip" . )
rm -rf "$SRC"

# 3. response to reviewers and cover letter
cp "$ROOT/revision_R1/response_to_reviewers.pdf" "$PKG/01_response_to_reviewers.pdf"
( cd "$ROOT/revision_R1" && pandoc cover_letter_R1.md -o "$PKG/06_cover_letter_R1.pdf" --pdf-engine=xelatex \
    -V geometry:margin=2.5cm -V fontsize=11pt -V mainfont="DejaVu Serif" )

# 4. highlights as plain text (EM item "Highlights")
python3 - "$ROOT/paper/paper4_main.tex" "$PKG/05_highlights.txt" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
block = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", s, re.S).group(1)
items = [re.sub(r"\\revtext\{(.*)\}$", r"\1", l.strip()[len("\\item"):].strip()) for l in block.splitlines() if l.strip().startswith("\\item")]
items = [i.replace("--", "–") for i in items]
open(sys.argv[2], "w").write("\n".join("• " + i for i in items) + "\n")
for i in items: print(len(i), i)
PY

# 5. declaration of interest: NOT copied. The R1 declaration must be generated with Elsevier's online
#    Declaration of Interests form (new title; SurtGIS disclosure, same text as the manuscript section).
ls -l "$PKG"
