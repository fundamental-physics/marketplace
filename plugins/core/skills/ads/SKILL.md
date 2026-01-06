---
name: ads
description: Use when the user mentions 'ADS', 'NASA ADS', 'Astrophysics Data System', or asks to search astronomy/astrophysics literature, find papers by bibcode, get citation counts from ADS, or retrieve BibTeX from the ADS database.
---

# NASA ADS Search and Retrieval Skill

Search and retrieve papers from NASA's Astrophysics Data System (ADS).

## Setup

ADS requires an API token for authentication.

### Getting a Token

1. Go to https://ui.adsabs.harvard.edu
2. Sign in (or create an account)
3. Click your name in the top-right → "Account settings"
4. Select "API Token" from the left sidebar
5. Click "Generate a new key"
6. Copy the token string

### Storing the Token

Save your token in one of these locations:

**Option 1: Config file (recommended)**
```bash
mkdir -p ~/.ads
echo "YOUR_TOKEN_HERE" > ~/.ads/dev_key
chmod 600 ~/.ads/dev_key
```

**Option 2: Environment variable**
```bash
export ADS_DEV_KEY="YOUR_TOKEN_HERE"
```

## Bibcode Format

ADS uses bibcodes as unique identifiers:
- Format: `YYYYJJJJJVVVVMPPPPA`
- Example: `2019ApJ...882L..12P` (ApJ Letters, 2019)

## Basic Usage

```bash
# Get a record by bibcode
python scripts/ads.py 2019ApJ...882L..12P

# Get BibTeX
python scripts/ads.py 2019ApJ...882L..12P --format bibtex
```

## Searching

```bash
# Search by author
python scripts/ads.py --search "author:Einstein"

# Search by first author only
python scripts/ads.py --search "author:^Hawking"

# Search by title
python scripts/ads.py --search 'title:"dark energy"'

# Search by abstract
python scripts/ads.py --search 'abs:"gravitational waves"'

# Search by year range
python scripts/ads.py --search "author:Penrose year:2015-2020"

# Search by journal
python scripts/ads.py --search "bibstem:ApJ year:2023"

# Combined search
python scripts/ads.py --search 'author:Planck title:"cosmological parameters"'

# Limit results
python scripts/ads.py --search "author:Witten" -n 5
```

### Search Query Syntax

| Field | Syntax | Example |
|-------|--------|---------|
| Author | `author:"Last, First"` | `author:"Hawking, S"` |
| First Author | `author:^Name` | `author:^Einstein` |
| Title | `title:"phrase"` | `title:"black hole"` |
| Abstract | `abs:"phrase"` | `abs:"dark matter"` |
| Year | `year:YYYY` or `year:YYYY-YYYY` | `year:2020-2023` |
| Journal | `bibstem:abbrev` | `bibstem:MNRAS` |
| arXiv ID | `arXiv:id` | `arXiv:1207.7214` |
| DOI | `doi:value` | `doi:10.1086/345794` |
| Object | `object:name` | `object:M31` |

Boolean operators: `AND` (default), `OR`, `NOT`, `-` (negation)

### Common Journal Abbreviations

| Abbreviation | Journal |
|--------------|---------|
| `ApJ` | Astrophysical Journal |
| `ApJL` | ApJ Letters |
| `MNRAS` | Monthly Notices of the RAS |
| `A&A` | Astronomy & Astrophysics |
| `AJ` | Astronomical Journal |
| `PhRvD` | Physical Review D |
| `PhRvL` | Physical Review Letters |
| `JCAP` | Journal of Cosmology and Astroparticle Physics |

## Citations

```bash
# Get papers citing a record
python scripts/ads.py 2019ApJ...882L..12P --citations

# Top 20 citing papers
python scripts/ads.py 2019ApJ...882L..12P --citations -n 20
```

## Output Formats

```bash
# Default (shows metadata)
python scripts/ads.py 2019ApJ...882L..12P

# BibTeX
python scripts/ads.py 2019ApJ...882L..12P --format bibtex
```

## Typical Workflow

1. Search for papers: `python scripts/ads.py --search "author:Name"`
2. Note the bibcode
3. Get full details: `python scripts/ads.py <bibcode>`
4. Get BibTeX: `python scripts/ads.py <bibcode> --format bibtex`
5. Check citations: `python scripts/ads.py <bibcode> --citations`

## Rate Limits

ADS allows 5000 requests per day per API token.

## Comparison with Other Skills

- **ADS**: Astronomy/astrophysics focus, published papers, bibcodes
- **INSPIRE**: High-energy physics focus, SPIRES syntax, recids
- **arXiv**: Preprint source code (LaTeX), all physics categories

Use together: search on ADS for published papers with citation data, use arXiv skill to get source.
