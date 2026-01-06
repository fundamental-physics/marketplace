---
name: hepdata
description: Use when the user mentions 'HEPData' or asks to find experimental data, download data tables from HEP papers, get digitized plots, or retrieve cross-section measurements. HEPData contains data points behind figures in high-energy physics publications.
---

# HEPData Search and Download Skill

Search and download experimental data tables from HEPData, the high-energy physics data repository.

**Requires**: `requests` (`pip install requests`)

## What is HEPData?

HEPData stores the actual data points behind plots and tables in HEP publications:
- Cross-section measurements
- Exclusion limits
- Differential distributions
- Digitized figures

Data is linked to papers via INSPIRE IDs and available in multiple formats.

## Basic Usage

```bash
# Get record by HEPData ID
python scripts/hepdata.py ins1234567

# Get record by INSPIRE ID
python scripts/hepdata.py 1234567 --inspire

# Get record by arXiv ID
python scripts/hepdata.py 1907.12345 --arxiv

# List tables in a record
python scripts/hepdata.py 1234567 --inspire --tables
```

## Searching

```bash
# Search by keywords
python scripts/hepdata.py --search "Higgs cross section"

# Search by reaction
python scripts/hepdata.py --search 'reactions:"P P --> TOP TOPBAR"'

# Search by collaboration
python scripts/hepdata.py --search "collaboration:ATLAS"

# Search by observable
python scripts/hepdata.py --search "observables:SIG"

# Limit results
python scripts/hepdata.py --search "dark matter" -n 5
```

### Search Query Syntax

- `reactions:"P P --> X"` - Search by reaction
- `collaboration:NAME` - Filter by collaboration (ATLAS, CMS, LHCb, etc.)
- `observables:TYPE` - Filter by observable type
- `cmenergies:13000` - Center-of-mass energy in GeV
- `keywords:term` - Search keywords

## Downloading Data

```bash
# List available tables first
python scripts/hepdata.py 1234567 --inspire --tables

# Download specific table as CSV
python scripts/hepdata.py 1234567 --inspire --download "Table 1" --format csv

# Download all tables as YAML
python scripts/hepdata.py 1234567 --inspire --download --format yaml

# Download to specific directory
python scripts/hepdata.py 1234567 --inspire --download "Table 1" -o ./data/
```

### Download Formats

- `csv` - Comma-separated values (default)
- `yaml` - YAML format (HEPData native)
- `json` - JSON format
- `root` - ROOT file
- `yoda` - YODA format (for Rivet)

## Identifier Types

HEPData accepts three types of identifiers:

| Flag | Type | Example |
|------|------|---------|
| (none) | HEPData ID | `ins1234567` |
| `--inspire` | INSPIRE recid | `1234567` |
| `--arxiv` | arXiv ID | `1907.12345` |

## Typical Workflow

1. Find paper on INSPIRE: get the INSPIRE ID
2. Check for HEPData: `python scripts/hepdata.py <inspire_id> --inspire`
3. List tables: `python scripts/hepdata.py <id> --inspire --tables`
4. Download data: `python scripts/hepdata.py <id> --inspire --download "Table 1" -f csv`
5. Use in analysis

## Common Use Cases

### Get Exclusion Limits

```bash
# Find ATLAS SUSY limits
python scripts/hepdata.py --search "collaboration:ATLAS SUSY exclusion"

# Download the limit data
python scripts/hepdata.py 1234567 --inspire --download "Exclusion contour" -f csv
```

### Get Cross-Section Measurements

```bash
# Find Higgs measurements
python scripts/hepdata.py --search "Higgs cross section 13 TeV"

# Download measurement table
python scripts/hepdata.py 1234567 --inspire --download "Cross section" -f csv
```

### Overlay Theory on Data

Download data points to compare with your theoretical predictions:

```python
import pandas as pd
data = pd.read_csv("Table_1.csv")
# Plot data vs your theory
```

## Integration with INSPIRE Skill

Use together with the `inspire` skill:
1. Search on INSPIRE for papers
2. Get the INSPIRE recid
3. Fetch data from HEPData using that ID

## API Notes

- No authentication required
- Data is CC0 licensed
- Some older records may have limited format support
