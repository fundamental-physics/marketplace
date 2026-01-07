#!/usr/bin/env python3
"""SALT3 likelihood function template for nested sampling.

This script provides a JIT-compiled likelihood function for SALT3 model fitting.
Designed for use with blackjax nested sampling or other JAX-based samplers.

Usage:
    1. Load your photometric data (times, fluxes, fluxerrs, bands)
    2. Call setup_likelihood() to create bridges and likelihood function
    3. Use loglikelihood() in your sampler

Example:
    from salt3_likelihood import setup_likelihood
    logL, param_names = setup_likelihood(times, fluxes, fluxerrs, bands, zps, fixed_z)
    result = logL({'t0': 0.0, 'x0': 1e-4, 'x1': 0.5, 'c': 0.05})
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_supernovae import SALT3Source
from jax_supernovae.bandpasses import get_bandpass
from jax_supernovae.salt3 import precompute_bandflux_bridge


def setup_likelihood(times, fluxes, fluxerrs, bands, zps, fixed_z):
    """Set up SALT3 likelihood with precomputed bridges.

    Args:
        times: Observation times (MJD), array-like
        fluxes: Observed flux values, array-like
        fluxerrs: Flux uncertainties, array-like
        bands: List/array of band names per observation
        zps: Zero points per observation, array-like
        fixed_z: Fixed redshift value (float)

    Returns:
        loglikelihood: JIT-compiled likelihood function that takes a dict
                      with keys 't0', 'x0', 'x1', 'c'
        param_names: List of parameter names ['t0', 'x0', 'x1', 'c']
    """
    # Convert to JAX arrays
    times_jax = jnp.array(times)
    fluxes_jax = jnp.array(fluxes)
    fluxerrs_jax = jnp.array(fluxerrs)
    zps_jax = jnp.array(zps)

    # Convert bands to list of strings
    if isinstance(bands, np.ndarray):
        # Handle both string and byte arrays
        bands = [b.decode('utf-8') if isinstance(b, bytes) else str(b) for b in bands]
    else:
        bands = list(bands)

    # Create source instance (this also registers common bandpasses)
    source = SALT3Source()

    # Set up bridges for optimized evaluation
    unique_bands = list(set(bands))
    bridges = tuple(precompute_bandflux_bridge(get_bandpass(b)) for b in unique_bands)
    band_to_idx = {b: i for i, b in enumerate(unique_bands)}
    band_indices = jnp.array([band_to_idx[b] for b in bands])

    @jax.jit
    def loglikelihood(params):
        """Gaussian log-likelihood for SALT3 model.

        Args:
            params: Dictionary with keys 't0', 'x0', 'x1', 'c'

        Returns:
            Log-likelihood value (scalar JAX array)
        """
        # Convert observer times to rest-frame phases
        phases = (times_jax - params['t0']) / (1.0 + fixed_z)

        # Compute model fluxes using optimized mode
        bandflux_params = {'x0': params['x0'], 'x1': params['x1'], 'c': params['c']}
        model_fluxes = source.bandflux(
            bandflux_params,
            bands=None,
            phases=phases,
            zp=zps_jax,
            zpsys='ab',
            band_indices=band_indices,
            bridges=bridges,
            unique_bands=unique_bands
        )

        # Gaussian log-likelihood
        chi2 = jnp.sum(((fluxes_jax - model_fluxes) / fluxerrs_jax) ** 2)
        log_det = jnp.sum(jnp.log(2.0 * jnp.pi * fluxerrs_jax**2))
        return -0.5 * (chi2 + log_det)

    return loglikelihood, ['t0', 'x0', 'x1', 'c']


def setup_likelihood_with_sigma(times, fluxes, fluxerrs, bands, zps, fixed_z):
    """Set up SALT3 likelihood with error scaling hyperparameter.

    This version includes log_sigma to rescale flux errors, useful when
    the reported uncertainties may be under/overestimated.

    Args:
        times: Observation times (MJD)
        fluxes: Observed flux values
        fluxerrs: Flux uncertainties
        bands: List of band names per observation
        zps: Zero points per observation
        fixed_z: Fixed redshift value

    Returns:
        loglikelihood: JIT-compiled likelihood function that takes a dict
                      with keys 't0', 'log_x0', 'x1', 'c', 'log_sigma'
        param_names: List of parameter names
    """
    # Convert to JAX arrays
    times_jax = jnp.array(times)
    fluxes_jax = jnp.array(fluxes)
    fluxerrs_jax = jnp.array(fluxerrs)
    zps_jax = jnp.array(zps)

    if isinstance(bands, np.ndarray):
        bands = [b.decode('utf-8') if isinstance(b, bytes) else str(b) for b in bands]
    else:
        bands = list(bands)

    # Create source instance (this also registers common bandpasses)
    source = SALT3Source()

    unique_bands = list(set(bands))
    bridges = tuple(precompute_bandflux_bridge(get_bandpass(b)) for b in unique_bands)
    band_to_idx = {b: i for i, b in enumerate(unique_bands)}
    band_indices = jnp.array([band_to_idx[b] for b in bands])

    @jax.jit
    def loglikelihood(params):
        """Log-likelihood with error scaling.

        Args:
            params: Dictionary with 't0', 'log_x0', 'x1', 'c', 'log_sigma'

        Returns:
            Log-likelihood value
        """
        # Transform from log-space
        x0 = 10.0 ** params['log_x0']
        sigma = 10.0 ** params['log_sigma']

        phases = (times_jax - params['t0']) / (1.0 + fixed_z)

        bandflux_params = {'x0': x0, 'x1': params['x1'], 'c': params['c']}
        model_fluxes = source.bandflux(
            bandflux_params, None, phases,
            zp=zps_jax, zpsys='ab',
            band_indices=band_indices,
            bridges=bridges,
            unique_bands=unique_bands
        )

        # Scale errors by sigma hyperparameter
        eff_fluxerrs = sigma * fluxerrs_jax
        chi2 = jnp.sum(((fluxes_jax - model_fluxes) / eff_fluxerrs) ** 2)
        log_det = jnp.sum(jnp.log(2.0 * jnp.pi * eff_fluxerrs**2))
        return -0.5 * (chi2 + log_det)

    return loglikelihood, ['t0', 'log_x0', 'x1', 'c', 'log_sigma']


# Example usage
if __name__ == "__main__":
    import sys

    # Check for data file argument
    if len(sys.argv) < 2:
        print("Usage: python salt3_likelihood.py <data.npz>")
        print("\nGenerating test data...")

        # Generate simple test data
        from generate_synthetic_data import generate_salt3_data
        data = generate_salt3_data()
    else:
        data = np.load(sys.argv[1], allow_pickle=True)

    # Extract data
    times = data['times']
    fluxes = data['fluxes']
    fluxerrs = data['fluxerrs']
    bands = data['bands']
    zps = data['zps']

    # Get true parameters if available
    if 'true_params' in data:
        true_p = data['true_params']
        if isinstance(true_p, np.ndarray):
            true_p = true_p.item()
        fixed_z = true_p.get('z', 0.05)
    else:
        fixed_z = 0.05

    print(f"Loaded {len(times)} observations")
    print(f"Fixed redshift: {fixed_z}")

    # Set up likelihood
    logL, param_names = setup_likelihood(times, fluxes, fluxerrs, bands, zps, fixed_z)
    print(f"Parameters: {param_names}")

    # Test at true parameters if available
    if 'true_params' in data:
        test_params = {
            't0': true_p.get('t0', 0.0),
            'x0': true_p['x0'],
            'x1': true_p['x1'],
            'c': true_p['c']
        }
        logL_value = logL(test_params)
        print(f"\nLog-likelihood at true params: {logL_value:.2f}")
        print(f"  t0={test_params['t0']}, x0={test_params['x0']:.2e}, "
              f"x1={test_params['x1']}, c={test_params['c']}")
