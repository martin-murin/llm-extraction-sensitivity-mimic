"""Build the paper: substitute claims, concatenate sources, convert to LaTeX.

Reads paper/sources/*.md, substitutes {{claim_key}} or {{claim_key:format}}
placeholders with values from paper/claims/claims.json, and produces:

  paper/build/paper.md   — concatenated, substituted markdown (for review)
  paper/build/paper.tex  — LaTeX produced via pandoc (for arXiv submission)

PDF compilation is NOT done here. To get a PDF locally, either:
  - Drop paper/build/paper.tex into Overleaf
  - Run pdflatex manually: cd paper/build && pdflatex paper.tex

Usage:
  python paper/build.py
  python paper/build.py --supplement   # build only the supplement
  python paper/build.py --combined     # build main + supplement in one PDF
  python paper/build.py --list-claims  # print claim keys and exit
  python paper/build.py --no-pandoc    # produce paper.md only
  python paper/build.py --run-latex    # run pdflatex/bibtex/pdflatex/pdflatex
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
SOURCES_DIR = PAPER_DIR / "sources"
BUILD_DIR = PAPER_DIR / "build"
CLAIMS_PATH = PAPER_DIR / "claims" / "claims.json"
REFERENCES_PATH = PAPER_DIR / "references.bib"
TEMPLATE_PATH = PAPER_DIR / "templates" / "article-arxiv.tex"
TABLE_FILTER_PATH = PAPER_DIR / "templates" / "wide-tables.lua"

PLACEHOLDER = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)(?::([^}]+))?\}\}")
REF_NOBREAK_CMD = re.compile(
    r"\b(Figure|Figures|Section|Sections|Table|Tables|Appendix|Appendices|"
    r"Eq\.|Eqs\.|Equation|Equations)~(?=\\(?:ref|eqref|autoref)\{)"
)
REF_NOBREAK_NUM = re.compile(
    r"\b(Figure|Figures|Section|Sections|Table|Tables|Appendix|Appendices|"
    r"Eq\.|Eqs\.|Equation|Equations)~(?=(?:S)?\d+(?:\.\d+)*)"
)
REF_TILDE_ESCAPED_TEX = re.compile(
    r"\b(Figure|Figures|Section|Sections|Table|Tables|Appendix|Appendices|"
    r"Eq\\.|Eqs\\.|Equation|Equations)"
    r"\\textasciitilde\{\}(?=(?:\\ref\{|\\eqref\{|\\autoref\{|(?:S)?\d))"
)


def load_claims() -> dict:
    if not CLAIMS_PATH.exists():
        print(f"No claims.json at {CLAIMS_PATH}. Run paper/claims/recompute_all.py first.",
              file=sys.stderr)
        sys.exit(1)
    return json.loads(CLAIMS_PATH.read_text())


def substitute(text: str, claims: dict) -> tuple[str, list[str]]:
    """Substitute {{key}} placeholders. Returns (substituted, missing_keys)."""
    missing = []

    def replace(match: re.Match) -> str:
        key = match.group(1)
        format_spec = match.group(2)
        if key not in claims:
            missing.append(key)
            return f"!!UNDEFINED_CLAIM:{key}!!"
        value = claims[key]["value"]
        if format_spec is None:
            format_spec = claims[key].get("format_default", "")
        if format_spec:
            try:
                return format(value, format_spec)
            except (ValueError, TypeError):
                return str(value)
        return str(value)

    result = PLACEHOLDER.sub(replace, text)
    return result, sorted(set(missing))


def assemble(directory: Path) -> str:
    """Concatenate all .md files in directory in alphabetical order."""
    files = sorted(p for p in directory.glob("*.md") if not p.name.startswith("_"))
    return "\n\n".join(f.read_text() for f in files)


def assemble_with_pagebreaks(directory: Path) -> str:
    """Concatenate markdown files and force each file to start on a new page."""
    files = sorted(p for p in directory.glob("*.md") if not p.name.startswith("_"))
    chunks: list[str] = []
    for idx, fpath in enumerate(files):
        text = fpath.read_text().strip()
        if idx > 0:
            chunks.append("\\clearpage")
        chunks.append(text)
    return "\n\n".join(chunks)


def assemble_combined() -> str:
    """Assemble one document: main text + bibliography + supplement."""
    main_md = assemble(SOURCES_DIR)
    supplement_md = assemble_with_pagebreaks(SOURCES_DIR / "supplement")
    bibliography_block = (
        "\\clearpage\n"
        "\\bibliography{../references}\n"
    )
    supplement_transition = (
        "\\clearpage\n"
        "\\section*{Supplementary Material}\n"
        "\\appendix\n"
        "\\setcounter{section}{0}\n"
        "\\renewcommand{\\thesection}{\\Alph{section}}\n"
        "\\renewcommand{\\thesubsection}{\\Alph{section}.\\arabic{subsection}}\n"
        "\\setcounter{figure}{0}\n"
        "\\renewcommand{\\thefigure}{S\\arabic{figure}}\n"
        "\\setcounter{table}{0}\n"
        "\\renewcommand{\\thetable}{S\\arabic{table}}\n"
        "\\floatplacement{figure}{H}\n"
    )
    return (
        f"{main_md}\n\n"
        f"{bibliography_block}\n\n"
        f"{supplement_transition}\n\n"
        f"{supplement_md}\n"
    )


def enforce_nonbreaking_references(text: str) -> str:
    """Convert markdown tilde spacing in references to LaTeX-safe nonbreak space.

    Pandoc escapes literal '~' to '\\textasciitilde{}' when mixed with raw LaTeX
    commands (e.g. 'Figure~\\ref{...}'), which renders visibly in PDF. We rewrite
    only common reference patterns to '\\nobreakspace{}' before pandoc.
    """
    text = REF_NOBREAK_CMD.sub(r"\1\\nobreakspace{}", text)
    text = REF_NOBREAK_NUM.sub(r"\1\\nobreakspace{}", text)
    return text


def apply_box_boundaries_to_tables(tex: str) -> str:
    """Convert pandoc booktabs-style tables to boxed-grid style.

    Applies a conservative post-process to generated LaTeX so all tables have:
    - vertical boundaries between columns and around table edges
    - horizontal row separators via \\hline
    """
    tex = tex.replace("\\toprule\\noalign{}", "\\hline")
    tex = tex.replace("\\midrule\\noalign{}", "\\hline")
    tex = tex.replace("\\bottomrule\\noalign{}", "\\hline")

    longtable_spec_pattern = re.compile(
        r"(\\begin\{longtable\}\[\]\{)@\{\}(.*?)@\{\}\}",
        flags=re.DOTALL,
    )

    def _rewrite_spec(match: re.Match[str]) -> str:
        begin = match.group(1)
        spec_block = match.group(2)
        columns = [line.strip() for line in spec_block.splitlines() if line.strip()]
        if not columns:
            return match.group(0)
        return f"{begin}|{'|'.join(columns)}|}}"

    tex = longtable_spec_pattern.sub(_rewrite_spec, tex)
    tex = tex.replace("\\tabularnewline", "\\tabularnewline\\hline")
    return tex


def normalize_tex_reference_tildes(tex: str) -> str:
    """Restore LaTeX nonbreaking-space marker in common reference patterns."""
    return REF_TILDE_ESCAPED_TEX.sub(r"\1~", tex)


def to_latex(markdown: str, out_tex: Path, *, include_bibliography: bool = True) -> bool:
    """Convert markdown to LaTeX via pandoc."""
    cmd = [
        "pandoc",
        "--from=markdown+tex_math_dollars+raw_tex",
        "--to=latex",
        "--standalone",
        "--number-sections",
        "--natbib",
        "--metadata=natbiboptions:numbers",
        "--metadata=biblio-style:unsrt",
    ]
    if TEMPLATE_PATH.exists():
        cmd += ["--template", str(TEMPLATE_PATH)]
    if TABLE_FILTER_PATH.exists():
        cmd += ["--lua-filter", str(TABLE_FILTER_PATH)]
    if include_bibliography and REFERENCES_PATH.exists():
        cmd += ["--bibliography", str(REFERENCES_PATH)]
    cmd += ["-o", str(out_tex)]
    try:
        result = subprocess.run(cmd, input=markdown, capture_output=True, text=True)
    except FileNotFoundError:
        print("pandoc not installed. Skipping LaTeX generation.", file=sys.stderr)
        print("  install pandoc to enable .tex output.", file=sys.stderr)
        return False
    if result.returncode != 0:
        print("pandoc failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return False
    # Pandoc passes raw \cite{} through, but may omit \bibliography metadata.
    # Ensure BibTeX has \bibstyle/\bibdata in .aux by appending a bibliography
    # block when references exist and cites are present.
    tex = out_tex.read_text(encoding="utf-8")
    tex = apply_box_boundaries_to_tables(tex)
    tex = normalize_tex_reference_tildes(tex)
    has_cite = "\\cite{" in tex
    has_bibliography_block = "\\bibliography{" in tex or "\\printbibliography" in tex
    if include_bibliography and REFERENCES_PATH.exists() and has_cite and not has_bibliography_block:
        bib_block = "\\bibliographystyle{unsrt}\n\\bibliography{../references}\n"
        if "\\end{document}" in tex:
            tex = tex.replace("\\end{document}", f"{bib_block}\\end{{document}}")
    out_tex.write_text(tex, encoding="utf-8")
    return True


def run_latex_sequence(target: str) -> bool:
    """Run pdflatex; bibtex (if cited); pdflatex; pdflatex in paper/build."""
    tex_name = f"{target}.tex"
    base_name = target
    for ext in ("aux", "bbl", "blg"):
        stale = BUILD_DIR / f"{base_name}.{ext}"
        if stale.exists():
            stale.unlink()
    commands: list[list[str]] = [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_name],
    ]
    build_ok = True
    for cmd in commands:
        result = subprocess.run(cmd, cwd=BUILD_DIR, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"{' '.join(cmd)} failed:", file=sys.stderr)
            print(result.stdout, file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return False

    aux_path = BUILD_DIR / f"{base_name}.aux"
    aux_text = aux_path.read_text(encoding="utf-8") if aux_path.exists() else ""
    has_citations = "\\citation{" in aux_text
    if has_citations:
        bib = subprocess.run(
            ["bibtex", base_name],
            cwd=BUILD_DIR,
            capture_output=True,
            text=True,
        )
        if bib.returncode != 0:
            print("bibtex failed:", file=sys.stderr)
            print(bib.stdout, file=sys.stderr)
            print(bib.stderr, file=sys.stderr)
            return False

    for _ in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_name],
            cwd=BUILD_DIR,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("pdflatex failed:", file=sys.stderr)
            print(result.stdout, file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            build_ok = False
            break
    return build_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplement", action="store_true",
                        help="Build supplement instead of main paper")
    parser.add_argument("--combined", action="store_true",
                        help="Build single combined document (main + supplement)")
    parser.add_argument("--list-claims", action="store_true",
                        help="Print claim keys and exit")
    parser.add_argument("--no-pandoc", action="store_true",
                        help="Produce paper.md only; skip LaTeX")
    parser.add_argument(
        "--run-latex",
        action="store_true",
        help="After pandoc, run pdflatex; bibtex (if needed); pdflatex; pdflatex",
    )
    args = parser.parse_args()

    if args.list_claims:
        claims = load_claims()
        for key in sorted(claims.keys()):
            value = claims[key]["value"]
            print(f"  {key:50s} = {value}")
        return 0

    BUILD_DIR.mkdir(exist_ok=True)
    claims = load_claims()

    if args.supplement and args.combined:
        print("--supplement and --combined are mutually exclusive.", file=sys.stderr)
        return 1

    if args.combined:
        raw = assemble_combined()
        target = "paper"
        include_bibliography = False
    elif args.supplement:
        source_dir = SOURCES_DIR / "supplement"
        target = "supplement"
        include_bibliography = True
    else:
        source_dir = SOURCES_DIR
        target = "main"
        include_bibliography = True

    if not args.combined:
        if not source_dir.is_dir():
            print(f"Source directory does not exist: {source_dir}", file=sys.stderr)
            return 1
        raw = assemble(source_dir)
    substituted, missing = substitute(raw, claims)
    substituted = enforce_nonbreaking_references(substituted)

    if missing:
        print(f"WARNING: {len(missing)} undefined claim(s):")
        for key in missing:
            print(f"  - {key}")
        print("Substituted with placeholder strings.")
        print()

    md_path = BUILD_DIR / f"{target}.md"
    md_path.write_text(substituted)
    print(f"Wrote {md_path} ({len(substituted)} chars)")

    if not args.no_pandoc:
        tex_path = BUILD_DIR / f"{target}.tex"
        if to_latex(substituted, tex_path, include_bibliography=include_bibliography):
            print(f"Wrote {tex_path}")
            if args.run_latex:
                ok = run_latex_sequence(target)
                if ok:
                    print(f"Built {BUILD_DIR / f'{target}.pdf'}")
                else:
                    return 1
            else:
                print(
                    "To compile PDF with citations: "
                    f"cd {BUILD_DIR} && pdflatex {target}.tex && bibtex {target} "
                    f"&& pdflatex {target}.tex && pdflatex {target}.tex"
                )

    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
