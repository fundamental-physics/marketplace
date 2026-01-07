---
name: jax-bandflux
description: JAX-accelerated supernova bandflux calculation for light curve modeling. Use when the user asks to "compute bandflux", "calculate supernova flux", "fit light curves with JAX", "use SALT3 model", "create custom SED model", "set up supernova likelihood", or mentions jax-bandflux, SALT3Source, TimeSeriesSource, or supernova photometry. Provides installation guidance, synthetic data generation, and likelihood templates for SALT3 and custom SED models.
---

# JAX-bandflux

## Overview

JAX-bandflux provides JAX-accelerated supernova light curve modeling with two source models:

- **SALT3Source**: Type Ia supernova standardization using SALT3-NIR model
  - Parameters: `x0` (amplitude), `x1` (stretch), `c` (color)
  - Model range: phase -20 to +50 days, wavelength 2000-20000 Angstroms

- **TimeSeriesSource**: Custom SED models from arbitrary spectral time series
  - Parameters: `amplitude` (scaling factor)
  - Works with any 2D flux grid (phase x wavelength)

## Workflow Position

This skill provides component **(2) Science** in the physics workflow:

```
(1) Research/Brainstorm → (2) Science → (3) Inference → (4) Visualization
```

**Inputs:** Photometric observations (times, fluxes, errors, bands)
**Outputs:** Model bandflux values, likelihood functions for sampling

Bandflux likelihoods connect to inference tools (e.g., nested sampling via blackjax-ns).

## Installation

### pip (CPU)

```bash
pip install jax-bandflux
pip install --upgrade "jax[cpu]"
```

### pip (GPU/CUDA 12)

```bash
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install jax-bandflux
```

### uv (recommended for speed)

```bash
# CPU
uv pip install jax-bandflux "jax[cpu]"

# GPU/CUDA 12
uv pip install "jax[cuda12]" --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
uv pip install jax-bandflux
```

### conda

```bash
conda create -n jax-bandflux python=3.11
conda activate jax-bandflux
conda install -c conda-forge jax
pip install jax-bandflux
```

### Verify Installation

```python
import jax
jax.config.update("jax_enable_x64", True)

from jax_supernovae import SALT3Source, TimeSeriesSource
print(f"JAX devices: {jax.devices()}")
print("Installation verified!")
```

## Quick Start 1: SALT3 Likelihood with Synthetic Data

This example demonstrates computing a SALT3 likelihood using synthetic photometric data.

### Step 1: Generate Synthetic Data

```python
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_supernovae import SALT3Source
from jax_supernovae.bandpasses import get_bandpass
from jax_supernovae.salt3 import precompute_bandflux_bridge

# True parameters for synthetic data
TRUE_PARAMS = {'x0': 1.0e-4, 'x1': 0.5, 'c': 0.05}
TRUE_T0 = 0.0        # Peak time (MJD offset)
TRUE_Z = 0.05        # Redshift

# Observation configuration
BANDS = ['bessellb', 'bessellv', 'bessellr']
OBS_TIMES = np.array([-10, -5, 0, 5, 10, 15, 20, 25, 30])  # Days from t0
NOISE_LEVEL = 0.05   # 5% flux uncertainty

# Generate synthetic observations
np.random.seed(42)
source = SALT3Source()

# Convert observer times to rest-frame phases
phases = (OBS_TIMES - TRUE_T0) / (1.0 + TRUE_Z)

# Generate observations for each band
obs_times, obs_fluxes, obs_errors, obs_bands = [], [], [], []

for band in BANDS:
    true_flux = np.array(source.bandflux(TRUE_PARAMS, band, phases, zp=27.5, zpsys='ab'))
    flux_err = np.abs(true_flux) * NOISE_LEVEL
    noisy_flux = true_flux + np.random.normal(0, flux_err)

    obs_times.extend(OBS_TIMES)
    obs_fluxes.extend(noisy_flux)
    obs_errors.extend(flux_err)
    obs_bands.extend([band] * len(OBS_TIMES))

# Convert to JAX arrays
times = jnp.array(obs_times)
fluxes = jnp.array(obs_fluxes)
fluxerrs = jnp.array(obs_errors)
zps = jnp.full(len(times), 27.5)

print(f"Generated {len(times)} observations in {len(BANDS)} bands")
```

### Step 2: Set Up Optimized Mode (Bridges)

Pre-compute bridges for ~100x speedup in likelihood evaluation:

```python
# Identify unique bands and create bridges
unique_bands = list(set(obs_bands))
bridges = tuple(precompute_bandflux_bridge(get_bandpass(b)) for b in unique_bands)

# Map observations to band indices
band_to_idx = {b: i for i, b in enumerate(unique_bands)}
band_indices = jnp.array([band_to_idx[b] for b in obs_bands])

print(f"Pre-computed bridges for: {unique_bands}")
```

### Step 3: Define Likelihood Function

```python
FIXED_Z = TRUE_Z  # Fixed redshift (known from spectroscopy)

@jax.jit
def loglikelihood(params):
    """Gaussian log-likelihood for SALT3 model.

    Args:
        params: dict with keys 't0', 'x0', 'x1', 'c'

    Returns:
        Log-likelihood value (scalar)
    """
    t0 = params['t0']
    x0 = params['x0']
    x1 = params['x1']
    c = params['c']

    # Convert to rest-frame phases
    phases = (times - t0) / (1.0 + FIXED_Z)

    # Compute model fluxes (optimized mode)
    bandflux_params = {'x0': x0, 'x1': x1, 'c': c}
    model_fluxes = source.bandflux(
        bandflux_params,
        bands=None,  # Use band_indices instead
        phases=phases,
        zp=zps,
        zpsys='ab',
        band_indices=band_indices,
        bridges=bridges,
        unique_bands=unique_bands,
    )

    # Gaussian log-likelihood
    chi2 = jnp.sum(((fluxes - model_fluxes) / fluxerrs) ** 2)
    return -0.5 * chi2

# Test at true parameters
test_params = {'t0': TRUE_T0, 'x0': TRUE_PARAMS['x0'],
               'x1': TRUE_PARAMS['x1'], 'c': TRUE_PARAMS['c']}
logL = loglikelihood(test_params)
print(f"Log-likelihood at true params: {logL:.2f}")
```

### Step 4: Evaluate at Different Parameters

```python
# Grid search over x1 (stretch parameter)
x1_values = jnp.linspace(-1.0, 2.0, 20)
logL_values = []

for x1_test in x1_values:
    test_p = {'t0': TRUE_T0, 'x0': TRUE_PARAMS['x0'],
              'x1': float(x1_test), 'c': TRUE_PARAMS['c']}
    logL_values.append(float(loglikelihood(test_p)))

# Find best-fit x1
best_idx = np.argmax(logL_values)
print(f"Best-fit x1: {x1_values[best_idx]:.3f} (true: {TRUE_PARAMS['x1']})")
```

**Output:** SALT3 parameters (t0, x0, x1, c) and likelihood values.

## Quick Start 2: Custom SED Model

This example shows fitting custom spectral energy distributions using TimeSeriesSource.

### Step 1: Create Custom SED Model

```python
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_supernovae import TimeSeriesSource
from jax_supernovae.bandpasses import get_bandpass
from jax_supernovae.salt3 import precompute_bandflux_bridge

# Define phase and wavelength grids
phase = np.linspace(-20, 50, 100)   # Days relative to peak
wave = np.linspace(3000, 9000, 200) # Angstroms

# Create 2D flux grid (simple Gaussian model for demo)
p_grid, w_grid = np.meshgrid(phase, wave, indexing='ij')

# Gaussian time profile (peaked at phase=0)
time_profile = np.exp(-0.5 * (p_grid / 12.0)**2)

# Gaussian wavelength profile (peaked at 5500 Angstroms)
wave_profile = np.exp(-0.5 * ((w_grid - 5500.0) / 1200.0)**2)

# Combined flux (erg/s/cm^2/Angstrom)
flux_grid = time_profile * wave_profile * 1e-15

# Create TimeSeriesSource
source = TimeSeriesSource(
    phase, wave, flux_grid,
    zero_before=True,        # Zero flux before minphase
    time_spline_degree=3,    # Cubic interpolation
    name='gaussian_sn',
    version='1.0'
)

print(f"Created TimeSeriesSource with parameters: {source.param_names}")
```

### Step 2: Generate Synthetic Observations

```python
# True amplitude for synthetic data
TRUE_AMPLITUDE = 2.5

# Observation times and bands
obs_phases = np.array([-10, -5, 0, 5, 10, 15, 20, 25])
obs_bands = ['bessellb', 'bessellv', 'bessellr', 'bessellv',
             'bessellb', 'bessellr', 'bessellv', 'bessellr']

# Generate true fluxes
params_true = {'amplitude': TRUE_AMPLITUDE}
true_fluxes = []
for ph, band in zip(obs_phases, obs_bands):
    flux = source.bandflux(params_true, band, ph, zp=25.0, zpsys='ab')
    true_fluxes.append(float(flux))

# Add 5% Gaussian noise
np.random.seed(123)
flux_errors = np.abs(true_fluxes) * 0.05
observed_fluxes = np.array(true_fluxes) + np.random.normal(0, flux_errors)

# Convert to JAX arrays
phases_jax = jnp.array(obs_phases)
fluxes_jax = jnp.array(observed_fluxes)
errors_jax = jnp.array(flux_errors)
zps_jax = jnp.full(len(obs_phases), 25.0)

print(f"Generated {len(obs_phases)} observations")
```

### Step 3: Set Up Optimized Mode and Likelihood

```python
# Pre-compute bridges for unique bands
unique_bands = ['bessellb', 'bessellv', 'bessellr']
bridges = tuple(precompute_bandflux_bridge(get_bandpass(b)) for b in unique_bands)

# Create band indices
band_to_idx = {b: i for i, b in enumerate(unique_bands)}
band_indices = jnp.array([band_to_idx[b] for b in obs_bands])

@jax.jit
def loglikelihood(amplitude):
    """Log-likelihood for amplitude parameter."""
    params = {'amplitude': amplitude}

    model_fluxes = source.bandflux(
        params, None, phases_jax,
        zp=zps_jax, zpsys='ab',
        band_indices=band_indices,
        bridges=bridges,
        unique_bands=unique_bands
    )

    chi2 = jnp.sum(((fluxes_jax - model_fluxes) / errors_jax)**2)
    return -0.5 * chi2

# Grid search for best amplitude
test_amplitudes = jnp.linspace(1.0, 4.0, 50)
logL_values = jax.vmap(loglikelihood)(test_amplitudes)

best_idx = jnp.argmax(logL_values)
best_amplitude = test_amplitudes[best_idx]

print(f"Best-fit amplitude: {best_amplitude:.3f}")
print(f"True amplitude: {TRUE_AMPLITUDE}")
```

## API Reference

### SALT3Source

```python
from jax_supernovae import SALT3Source

source = SALT3Source()

# Parameters (passed as dict):
# - x0: amplitude (>0, typically ~1e-4 to 1e-2)
# - x1: stretch (-3 to 3)
# - c: color (-0.3 to 0.3)

# Simple mode (convenient, slower)
flux = source.bandflux({'x0': 1e-4, 'x1': 0.5, 'c': 0.1},
                       'bessellb', phase=0.0, zp=27.5, zpsys='ab')

# Optimized mode (~100x faster, use for sampling)
flux = source.bandflux(params, None, phases,
                       band_indices=indices, bridges=bridges,
                       unique_bands=bands, zp=zps, zpsys='ab')
```

### TimeSeriesSource

```python
from jax_supernovae import TimeSeriesSource

source = TimeSeriesSource(phase, wave, flux_grid,
                          zero_before=True, time_spline_degree=3)

# Parameters: amplitude (scaling factor)
flux = source.bandflux({'amplitude': 1.0}, 'bessellv', phase=0.0,
                       zp=25.0, zpsys='ab')
```

### Bridge Pattern (Critical for Performance)

```python
from jax_supernovae.bandpasses import get_bandpass
from jax_supernovae.salt3 import precompute_bandflux_bridge

# Compute once, reuse in JIT-compiled functions
bridges = tuple(precompute_bandflux_bridge(get_bandpass(b))
                for b in unique_bands)
band_indices = jnp.array([band_to_idx[b] for b in obs_bands])
```

## Performance: Simple vs Optimized Mode

| Mode | Use Case | Speed |
|------|----------|-------|
| Simple | One-off calculations, exploration | 1x |
| Optimized | MCMC, nested sampling, optimization | ~100x |

**Always use optimized mode** (precomputed bridges) for likelihood evaluation in sampling loops.

## Connecting to Inference

The likelihood functions above are ready for use with JAX-based samplers.

### Nested Sampling with blackjax-ns

```python
# Example: Using likelihood with blackjax nested sampling
# (requires: pip install git+https://github.com/handley-lab/blackjax@proposal)

from blackjax.ns.utils import uniform_prior

# Define prior bounds
PRIOR_BOUNDS = {
    "t0": (-10.0, 10.0),
    "log_x0": (-5.0, -2.6),
    "x1": (-3.0, 3.0),
    "c": (-0.3, 0.3),
}

# Initialize nested sampling
rng_key = jax.random.PRNGKey(0)
particles, logprior_fn = uniform_prior(rng_key, n_live=125, bounds=PRIOR_BOUNDS)

# The loglikelihood function from Quick Start 1 can be used directly
# with adjustment for log_x0 -> x0 transformation
```

For complete nested sampling examples, see the JAX-bandflux repository:
https://github.com/samleeney/JAX-bandflux/blob/main/examples/ns.py

## Bundled Scripts

This skill includes helper scripts for common operations:

- **generate_synthetic_data.py**: Generate synthetic SALT3 or custom SED data
  ```bash
  python generate_synthetic_data.py --model salt3 --output data.npz
  python generate_synthetic_data.py --model custom --output custom_data.npz
  ```

- **salt3_likelihood.py**: Template for SALT3 likelihood setup
  ```python
  from salt3_likelihood import setup_likelihood
  logL, param_names = setup_likelihood(times, fluxes, fluxerrs, bands, zps, fixed_z)
  ```

- **custom_sed_likelihood.py**: Template for TimeSeriesSource likelihood
  ```python
  from custom_sed_likelihood import setup_likelihood
  logL, source = setup_likelihood(phase_grid, wave_grid, flux_grid, ...)
  ```
