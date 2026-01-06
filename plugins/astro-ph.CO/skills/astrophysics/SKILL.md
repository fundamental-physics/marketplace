---
name: Astrophysics Analysis
description: This skill should be used when the user asks to "analyze astronomical data", "process FITS files", "create sky maps", "work with cosmological simulations", "fit spectral data", "calculate redshifts", "analyze light curves", "work with astronomical catalogs", "perform astrometry", "process images from telescopes", or mentions astronomical instruments, surveys, or astrophysical phenomena. Provides guidance for scientific astrophysics workflows and best practices.
version: 1.0.0
---

# Astrophysics Analysis Skill

## Overview

This skill provides specialized guidance for astrophysics analysis workflows, astronomical data processing, and scientific computing tasks common in astronomy research. It covers observational astronomy, cosmological analysis, and astrophysical simulations.

## Core Capabilities

### Astronomical Data Formats

Handle standard astronomy file formats:

- **FITS files**: Read and write Flexible Image Transport System files using astropy.io.fits
- **HDF5**: Process large simulation datasets with h5py
- **VOTable**: Work with Virtual Observatory tables using astropy.io.votable
- **ASCII catalogs**: Parse space-separated and CSV astronomical catalogs

### Observational Analysis

Perform common observational astronomy tasks:

- **Photometry**: Aperture and PSF photometry using photutils
- **Spectroscopy**: Spectral extraction, calibration, and line fitting with specutils
- **Astrometry**: World coordinate system handling and coordinate transformations
- **Time series**: Light curve analysis and period finding

### Cosmological Computations

Execute cosmological calculations:

- **Distance measures**: Luminosity distance, angular diameter distance, comoving distance
- **Cosmological parameters**: Work with standard cosmologies (Planck, WMAP)
- **Power spectra**: Matter and CMB power spectrum analysis
- **Redshift calculations**: Photometric and spectroscopic redshift handling

## Recommended Python Stack

Use these established astrophysics libraries:

```python
# Core astronomy
import astropy
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.cosmology import Planck18

# Visualization
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize

# Data analysis
import numpy as np
from scipy import optimize, interpolate

# Specialized tools
import photutils  # Photometry
import specutils  # Spectroscopy
import healpy     # HEALPix sky maps
```

## Common Workflows

### Loading FITS Data

```python
from astropy.io import fits
from astropy.wcs import WCS

# Open FITS file
with fits.open('image.fits') as hdul:
    data = hdul[0].data
    header = hdul[0].header
    wcs = WCS(header)
```

### Coordinate Transformations

```python
from astropy.coordinates import SkyCoord
from astropy import units as u

# Create sky coordinate
coord = SkyCoord(ra=10.5*u.deg, dec=-30.2*u.deg, frame='icrs')

# Transform to galactic
galactic = coord.galactic
print(f"l={galactic.l:.2f}, b={galactic.b:.2f}")
```

### Cosmological Distances

```python
from astropy.cosmology import Planck18
import astropy.units as u

z = 0.5  # redshift
d_L = Planck18.luminosity_distance(z)
d_A = Planck18.angular_diameter_distance(z)
d_C = Planck18.comoving_distance(z)
```

### Spectral Line Fitting

```python
from specutils import Spectrum1D
from specutils.fitting import fit_lines
from astropy.modeling import models

# Create Gaussian model for emission line
g_init = models.Gaussian1D(amplitude=1*u.Jy, mean=6563*u.AA, stddev=2*u.AA)
g_fit = fit_lines(spectrum, g_init)
```

## Best Practices

### Units and Quantities

Always use astropy units for physical quantities:

```python
from astropy import units as u

# Attach units
wavelength = 5000 * u.AA
flux = 1e-17 * u.erg / u.s / u.cm**2 / u.AA

# Convert between units
wavelength_nm = wavelength.to(u.nm)
```

### Error Handling

Propagate uncertainties through calculations:

```python
from astropy.nddata import StdDevUncertainty
from uncertainties import ufloat

# Using uncertainties package
flux = ufloat(1.5e-17, 0.2e-17)  # value +/- error
```

### Reproducibility

Document analysis parameters and random seeds:

```python
import numpy as np
np.random.seed(42)  # For reproducible results

# Log parameters
params = {
    'aperture_radius': 5.0,  # arcsec
    'background_annulus': (10.0, 15.0),
    'sigma_clip': 3.0
}
```

### Memory Management

Handle large datasets efficiently:

```python
# Use memmap for large FITS files
with fits.open('large_image.fits', memmap=True) as hdul:
    # Process in chunks
    chunk = hdul[0].data[1000:2000, 1000:2000]
```

## Data Sources

Access astronomical archives:

- **MAST**: Hubble, JWST, TESS data via astroquery.mast
- **ESO Archive**: VLT, ALMA data via astroquery.eso
- **SDSS**: Sloan Digital Sky Survey via astroquery.sdss
- **Vizier**: Catalog access via astroquery.vizier
- **Simbad**: Object information via astroquery.simbad

### Example Archive Query

```python
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

# Query Simbad for object
result = Simbad.query_object("M31")

# Query Vizier catalog
v = Vizier(columns=['*'])
catalogs = v.query_region("M31", radius=1*u.deg, catalog="II/246")
```

## Visualization Standards

Create publication-quality figures:

```python
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize

# Astronomical image display
norm = ImageNormalize(data, interval=ZScaleInterval())
plt.imshow(data, norm=norm, cmap='gray', origin='lower')
plt.colorbar(label='Counts')
```

## Performance Considerations

Optimize computation for large datasets:

- Use NumPy vectorized operations instead of loops
- Consider Dask for out-of-core computation
- Use JAX for GPU-accelerated numerical work
- Profile code with cProfile before optimization

## When to Use This Skill

Activate this skill for tasks involving:

- Processing telescope observations
- Analyzing galaxy spectra or photometry
- Working with cosmological simulations
- Calculating astronomical quantities with proper units
- Querying astronomical databases and catalogs
- Creating sky maps or coordinate transformations
- Fitting models to astrophysical data
