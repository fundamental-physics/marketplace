---
name: gwosc
description: Use when the user mentions 'GWOSC', 'gravitational waves', 'LIGO', 'Virgo', or asks to find gravitational wave events, get event parameters, or download strain data. GWOSC provides open data from gravitational wave detectors.
---

# GWOSC Search and Download Skill

Search and download gravitational wave data from the Gravitational Wave Open Science Center (GWOSC).

**Requires**: `requests` (`pip install requests`)

## What is GWOSC?

GWOSC provides open access to gravitational wave data from LIGO, Virgo, and KAGRA:
- Detected event catalogs (GWTC)
- Event parameters (masses, spins, distances)
- Strain time series data
- Calibrated detector data

## Basic Usage

```bash
# List available catalogs
python scripts/gwosc.py --catalogs

# List events from GWTC
python scripts/gwosc.py --list

# List events from a specific catalog
python scripts/gwosc.py --list O3_Discovery_Papers

# Get event details
python scripts/gwosc.py GW150914

# Get specific version
python scripts/gwosc.py GW150914 --version 3
```

## Searching Events

```bash
# Search by primary mass range
python scripts/gwosc.py --min-mass1 20 --max-mass1 50

# Search for nearby events
python scripts/gwosc.py --max-distance 500

# Search by SNR threshold
python scripts/gwosc.py --min-snr 20

# Combined search
python scripts/gwosc.py --min-mass1 30 --max-distance 1000 --min-snr 15

# Limit results
python scripts/gwosc.py --min-mass1 10 -n 5
```

### Search Parameters

| Parameter | Description |
|-----------|-------------|
| `--min-mass1`, `--max-mass1` | Primary mass range (M_sun) |
| `--min-mass2`, `--max-mass2` | Secondary mass range (M_sun) |
| `--min-distance`, `--max-distance` | Luminosity distance range (Mpc) |
| `--min-snr` | Minimum network SNR |

## Strain Data

```bash
# List available strain files for an event
python scripts/gwosc.py GW150914 --strain

# Download 32s strain from Hanford
python scripts/gwosc.py GW150914 --download H1

# Download 4096s strain from Livingston in GWF format
python scripts/gwosc.py GW150914 --download L1 --duration 4096 --format gwf

# Download to specific directory
python scripts/gwosc.py GW150914 --download H1 -o ./data/
```

### Detectors

| Code | Detector |
|------|----------|
| H1 | LIGO Hanford |
| L1 | LIGO Livingston |
| V1 | Virgo |

### Strain Formats

| Format | Description |
|--------|-------------|
| `hdf5` | HDF5 format (default) |
| `gwf` | Gravitational Wave Frame format |
| `txt` | Gzip-compressed text |

### Strain Durations

- `32` - 32 seconds around the event (default)
- `4096` - 4096 seconds for longer analysis

## Event Parameters

When viewing an event, key parameters include:

- **GPS Time**: Event time in GPS seconds
- **Mass 1/2**: Component masses in solar masses
- **Chirp Mass**: Key mass parameter for waveform
- **Effective Spin**: Mass-weighted spin projection
- **Luminosity Distance**: Distance in Mpc
- **Redshift**: Cosmological redshift
- **Network SNR**: Combined signal-to-noise ratio
- **False Alarm Rate**: Statistical significance

## Typical Workflow

1. Browse catalogs: `python scripts/gwosc.py --catalogs`
2. List events: `python scripts/gwosc.py --list`
3. Find events by criteria: `python scripts/gwosc.py --min-mass1 30`
4. Get event details: `python scripts/gwosc.py GW190521`
5. Check strain files: `python scripts/gwosc.py GW190521 --strain`
6. Download data: `python scripts/gwosc.py GW190521 --download H1`

## Common Use Cases

### Find Binary Black Hole Mergers

```bash
# High-mass BBH (stellar mass)
python scripts/gwosc.py --min-mass1 20 --min-mass2 10

# Intermediate mass
python scripts/gwosc.py --min-mass1 50
```

### Find Nearby Events

```bash
# Events within 500 Mpc
python scripts/gwosc.py --max-distance 500
```

### Get Data for Analysis

```bash
# Download strain for parameter estimation
python scripts/gwosc.py GW150914 --download H1 --duration 4096 -o ./strain/
python scripts/gwosc.py GW150914 --download L1 --duration 4096 -o ./strain/
```

## API Notes

- No authentication required
- Data is openly available under CC-BY license
- Strain files can be large (especially 4096s)

## See Also

- Python package: `pip install gwosc` for programmatic access
- PyCBC, Bilby, LALSuite for GW data analysis
