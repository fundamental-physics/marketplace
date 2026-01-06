---
name: zenodo
description: Use when the user mentions 'Zenodo' or asks to find, search, or download datasets, software, or research artifacts with DOIs. Zenodo hosts scientific datasets, code releases, and supplementary materials from publications.
---

# Zenodo Search and Download Skill

Search and download research data, software, and publications from Zenodo.

**Requires**: `requests` (`pip install requests`)

## What is Zenodo?

Zenodo is a general-purpose open repository for research outputs:
- Datasets and data products
- Software releases and code
- Publications and preprints
- Presentations and posters
- Supplementary materials

All records get DOIs for citation.

## Basic Usage

```bash
# Get record details
python scripts/zenodo.py 1234567

# Get BibTeX citation
python scripts/zenodo.py 1234567 --format bibtex

# List files in a record
python scripts/zenodo.py 1234567 --files

# Download a file
python scripts/zenodo.py 1234567 --download filename.csv

# Download first file
python scripts/zenodo.py 1234567 --download
```

## Searching

```bash
# Basic search
python scripts/zenodo.py --search "dark matter simulation"

# Search for datasets only
python scripts/zenodo.py --search "cosmology" --type dataset

# Search for software
python scripts/zenodo.py --search "MCMC sampler" --type software

# Most recent results
python scripts/zenodo.py --search "gravitational waves" --sort mostrecent

# Limit results
python scripts/zenodo.py --search "Planck" -n 5
```

### Search Query Syntax

Zenodo uses Elasticsearch query syntax:
- `title:keyword` - Search in title
- `creators.name:Author` - Search by creator
- `doi:10.5281/zenodo.XXX` - Search by DOI
- `keywords:physics` - Search in keywords
- `communities:astronomy` - Search in community

Combine with `AND`, `OR`, `NOT`.

### Record Types

Filter with `--type`:
- `dataset` - Research data
- `software` - Code and software
- `publication` - Papers, reports
- `poster` - Conference posters
- `presentation` - Slides
- `image` - Figures, plots
- `video` - Videos

## Downloading Files

```bash
# List available files first
python scripts/zenodo.py 1234567 --files

# Download specific file
python scripts/zenodo.py 1234567 --download data.hdf5

# Download to specific directory
python scripts/zenodo.py 1234567 --download data.hdf5 -o ./data/
```

## Output Formats

```bash
# Default (JSON metadata)
python scripts/zenodo.py 1234567

# BibTeX for citation
python scripts/zenodo.py 1234567 --format bibtex

# DataCite XML
python scripts/zenodo.py 1234567 --format datacite
```

## Typical Workflow

1. Search for datasets: `python scripts/zenodo.py --search "query" --type dataset`
2. Note the record ID
3. Check files: `python scripts/zenodo.py <id> --files`
4. Download: `python scripts/zenodo.py <id> --download <filename>`
5. Get citation: `python scripts/zenodo.py <id> --format bibtex`

## Finding Data for Papers

Many papers deposit supplementary data on Zenodo. Search by:
- Paper title or keywords
- Author names
- Related arXiv ID in description

## API Notes

- No authentication required for public records
- Rate limits apply for heavy usage
- Large files may take time to download
