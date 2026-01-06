---
name: inspire
description: Use when the user mentions 'INSPIRE', 'INSPIRE-HEP', 'inspirehep', or asks to search high-energy physics literature, find HEP papers, get citation counts, or retrieve BibTeX from INSPIRE. Specializes in particle physics, quantum field theory, and related fields.
---

# INSPIRE-HEP Search and Retrieval Skill

Search and retrieve papers from INSPIRE-HEP, the high-energy physics literature database.

## Record Identifiers

INSPIRE accepts multiple identifier types:
- **INSPIRE recid**: `451647` (internal ID)
- **arXiv ID**: `1207.7214` or `hep-th/9711200`
- **DOI**: `10.1103/PhysRevLett.19.1264`

## Basic Usage

```bash
# Get a record by INSPIRE ID
python scripts/inspire.py 451647

# Get by arXiv ID
python scripts/inspire.py 1207.7214

# Get by DOI
python scripts/inspire.py 10.1103/PhysRevLett.19.1264

# Get BibTeX
python scripts/inspire.py 1207.7214 --format bibtex
```

## Searching

```bash
# Search by author (SPIRES syntax)
python scripts/inspire.py --search "a E.Witten.1"

# Search by title
python scripts/inspire.py --search "t dark matter"

# Search by arXiv category
python scripts/inspire.py --search "arXiv:hep-th"

# Combined search
python scripts/inspire.py --search "a Maldacena and t AdS/CFT"

# Most cited results
python scripts/inspire.py --search "a Hawking" --sort mostcited

# Limit results
python scripts/inspire.py --search "t supersymmetry" -n 5
```

### Search Query Syntax (SPIRES-compatible)

| Prefix | Field | Example |
|--------|-------|---------|
| `a` | Author | `a E.Witten.1` |
| `t` | Title | `t black hole` |
| `k` | Keywords | `k inflation` |
| `j` | Journal | `j Phys.Rev.Lett.` |
| `eprint` | arXiv ID | `eprint 1207.7214` |
| `topcite` | Citation count | `topcite 1000+` |

Boolean operators: `and`, `or`, `not`

## Citations

```bash
# Get papers citing a record
python scripts/inspire.py 1207.7214 --citations

# Top 20 citing papers
python scripts/inspire.py 1207.7214 --citations -n 20
```

## Output Formats

```bash
# JSON (default - shows metadata)
python scripts/inspire.py 1207.7214

# BibTeX
python scripts/inspire.py 1207.7214 --format bibtex

# LaTeX (European style)
python scripts/inspire.py 1207.7214 --format latex-eu

# LaTeX (US style)
python scripts/inspire.py 1207.7214 --format latex-us
```

## Typical Workflow

1. Search for papers: `python scripts/inspire.py --search "a Author"`
2. Note the INSPIRE recid or arXiv ID
3. Get full details: `python scripts/inspire.py <id>`
4. Get BibTeX for citation: `python scripts/inspire.py <id> --format bibtex`
5. Check citations: `python scripts/inspire.py <id> --citations`

## Rate Limits

INSPIRE allows 15 requests per 5-second window per IP address.

## Comparison with arXiv Skill

- **INSPIRE**: Citation counts, publication info, SPIRES search syntax, HEP-focused
- **arXiv**: Paper source code (LaTeX), all physics categories, preprint access

Use both together: search on INSPIRE for citation data, then use arXiv skill to get the paper source.
