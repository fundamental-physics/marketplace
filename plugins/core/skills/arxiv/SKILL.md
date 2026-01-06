---
name: arxiv
description: Use when the user mentions 'arxiv' or asks to find, search for, or download arxiv papers (by arxiv ID, author name, or topic), get paper source code, or retrieve LaTeX files from arxiv. Can search for papers to find arxiv IDs when only author/topic is provided, then download them. Prefer source format over PDF.
allowed-tools: []
---

# ArXiv Search and Download Skill

Download and view arXiv paper sources using the `arxiv` command-line tool.

## ArXiv ID Format

- Modern: `YYMM.NNNNN` (e.g., `2301.07041`)
- Old format: `category/YYMMNNN` (e.g., `astro-ph/0701123`)

## Downloading Papers

Use the `arxiv` CLI tool bundled with this skill:

```bash
# View all LaTeX source files (.tex, .bib, .bbl) with syntax highlighting
python $SKILL_DIR/scripts/arxiv.py 2301.07041

# Save entire source to a directory
python $SKILL_DIR/scripts/arxiv.py 2301.07041 --save

# List files in the archive
python $SKILL_DIR/scripts/arxiv.py 2301.07041 --list

# View only specific file types
python $SKILL_DIR/scripts/arxiv.py 2301.07041 --tex   # Only .tex files
python $SKILL_DIR/scripts/arxiv.py 2301.07041 --bib   # Only .bib files
python $SKILL_DIR/scripts/arxiv.py 2301.07041 --bbl   # Only .bbl files
```

The default output (no flags) wraps each file in markdown code blocks with appropriate language tags (latex/bibtex).

## Searching ArXiv

```bash
# Search by author
python $SKILL_DIR/scripts/arxiv.py --search "au:Handley"

# Search by title
python $SKILL_DIR/scripts/arxiv.py --search "ti:cosmology"

# Search by category
python $SKILL_DIR/scripts/arxiv.py --search "cat:astro-ph.CO"

# Combined search
python $SKILL_DIR/scripts/arxiv.py --search "ti:cosmology+AND+au:Planck"

# Limit results (default 10)
python $SKILL_DIR/scripts/arxiv.py --search "au:Hinton" -n 5
```

### Search Query Prefixes

| Prefix | Field |
|--------|-------|
| `ti:` | Title |
| `au:` | Author |
| `abs:` | Abstract |
| `cat:` | Category |
| `all:` | All fields |

## Typical Workflow

1. Search for papers: `python $SKILL_DIR/scripts/arxiv.py --search "au:Author"`
2. Pick an arxiv ID from results
3. View the source: `python $SKILL_DIR/scripts/arxiv.py <id>`
4. Use `--save` to extract files locally if needed

## Integration with LLM Analysis

After saving source with `--save`:

```bash
python $SKILL_DIR/scripts/arxiv.py 2301.07041 --save
code2prompt 2301.07041 --include "*.tex" --output-file /tmp/paper.md
```

Then pass to external LLMs for analysis.
