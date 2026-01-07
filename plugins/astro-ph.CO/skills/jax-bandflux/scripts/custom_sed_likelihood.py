#!/usr/bin/env python3
"""TimeSeriesSource likelihood function template.

For fitting custom SED models to photometric observations.

Usage:
    1. Create your flux grid (phase x wavelength)
    2. Call setup_likelihood() with grid and observations
    3. Use loglikelihood() for amplitude fitting or full sampling

Example:
    from custom_sed_likelihood import setup_likelihood
    logL, source = setup_likelihood(phase_grid, wave_grid, flux_grid,
                                    obs_phases, obs_fluxes, obs_errors,
                                    obs_bands, obs_zps)
    result = logL(2.5)  # Evaluate at amplitude=2.5
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_supernovae import SALT3Source, TimeSeriesSource
from jax_supernovae.bandpasses import get_bandpass
from jax_supernovae.salt3 import precompute_bandflux_bridge

# Register common bandpasses by instantiating SALT3Source once
_SALT3_INIT = SALT3Source()


def setup_likelihood(phase_grid, wave_grid, flux_grid,
                     obs_phases, obs_fluxes, obs_errors, obs_bands, obs_zps,
                     zero_before=True, time_spline_degree=3):
    """Set up TimeSeriesSource likelihood.

    Args:
        phase_grid: 1D array of phase values (days)
        wave_grid: 1D array of wavelength values (Angstroms)
        flux_grid: 2D flux array, shape (n_phase, n_wave) in erg/s/cm^2/A
        obs_phases: Observed phases (days)
        obs_fluxes: Observed flux values
        obs_errors: Flux uncertainties
        obs_bands: Band names per observation
        obs_zps: Zero points per observation
        zero_before: Zero flux before min phase (default True)
        time_spline_degree: Interpolation degree (1=linear, 3=cubic)

    Returns:
        loglikelihood: JIT-compiled likelihood function(amplitude)
        source: TimeSeriesSource instance
    """
    # Create TimeSeriesSource from grids
    source = TimeSeriesSource(
        phase_grid, wave_grid, flux_grid,
        zero_before=zero_before,
        time_spline_degree=time_spline_degree
    )

    # Convert to JAX arrays
    phases_jax = jnp.array(obs_phases)
    fluxes_jax = jnp.array(obs_fluxes)
    errors_jax = jnp.array(obs_errors)
    zps_jax = jnp.array(obs_zps)

    # Convert bands to list of strings
    if isinstance(obs_bands, np.ndarray):
        obs_bands = [b.decode('utf-8') if isinstance(b, bytes) else str(b) for b in obs_bands]
    else:
        obs_bands = list(obs_bands)

    # Set up bridges for optimized evaluation
    unique_bands = list(set(obs_bands))
    bridges = tuple(precompute_bandflux_bridge(get_bandpass(b)) for b in unique_bands)
    band_to_idx = {b: i for i, b in enumerate(unique_bands)}
    band_indices = jnp.array([band_to_idx[b] for b in obs_bands])

    @jax.jit
    def loglikelihood(amplitude):
        """Log-likelihood for amplitude parameter.

        Args:
            amplitude: Scaling factor for the SED model

        Returns:
            Log-likelihood value (scalar JAX array)
        """
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

    return loglikelihood, source


def setup_likelihood_dict(phase_grid, wave_grid, flux_grid,
                          obs_phases, obs_fluxes, obs_errors, obs_bands, obs_zps,
                          zero_before=True, time_spline_degree=3):
    """Set up TimeSeriesSource likelihood with dict-based parameters.

    Same as setup_likelihood but takes parameters as a dictionary,
    which is useful for samplers that use structured parameter dicts.

    Returns:
        loglikelihood: Function that takes {'amplitude': value}
        source: TimeSeriesSource instance
    """
    source = TimeSeriesSource(
        phase_grid, wave_grid, flux_grid,
        zero_before=zero_before,
        time_spline_degree=time_spline_degree
    )

    phases_jax = jnp.array(obs_phases)
    fluxes_jax = jnp.array(obs_fluxes)
    errors_jax = jnp.array(obs_errors)
    zps_jax = jnp.array(obs_zps)

    if isinstance(obs_bands, np.ndarray):
        obs_bands = [b.decode('utf-8') if isinstance(b, bytes) else str(b) for b in obs_bands]
    else:
        obs_bands = list(obs_bands)

    unique_bands = list(set(obs_bands))
    bridges = tuple(precompute_bandflux_bridge(get_bandpass(b)) for b in unique_bands)
    band_to_idx = {b: i for i, b in enumerate(unique_bands)}
    band_indices = jnp.array([band_to_idx[b] for b in obs_bands])

    @jax.jit
    def loglikelihood(params):
        """Log-likelihood taking dict with 'amplitude' key."""
        model_fluxes = source.bandflux(
            params, None, phases_jax,
            zp=zps_jax, zpsys='ab',
            band_indices=band_indices,
            bridges=bridges,
            unique_bands=unique_bands
        )

        chi2 = jnp.sum(((fluxes_jax - model_fluxes) / errors_jax)**2)
        return -0.5 * chi2

    return loglikelihood, source


def create_gaussian_sed(phase_range=(-20, 50), wave_range=(3000, 9000),
                        n_phase=100, n_wave=200,
                        peak_phase=0.0, phase_width=12.0,
                        peak_wave=5500.0, wave_width=1200.0,
                        flux_scale=1e-15):
    """Create a simple Gaussian SED model for testing.

    Args:
        phase_range: (min, max) phase in days
        wave_range: (min, max) wavelength in Angstroms
        n_phase: Number of phase grid points
        n_wave: Number of wavelength grid points
        peak_phase: Phase of peak brightness (days)
        phase_width: Gaussian width in phase (days)
        peak_wave: Wavelength of peak flux (Angstroms)
        wave_width: Gaussian width in wavelength (Angstroms)
        flux_scale: Overall flux scaling (erg/s/cm^2/A)

    Returns:
        phase: 1D array of phases
        wave: 1D array of wavelengths
        flux: 2D flux grid (n_phase, n_wave)
    """
    phase = np.linspace(phase_range[0], phase_range[1], n_phase)
    wave = np.linspace(wave_range[0], wave_range[1], n_wave)

    p_grid, w_grid = np.meshgrid(phase, wave, indexing='ij')

    # Gaussian time profile
    time_profile = np.exp(-0.5 * ((p_grid - peak_phase) / phase_width)**2)

    # Gaussian wavelength profile
    wave_profile = np.exp(-0.5 * ((w_grid - peak_wave) / wave_width)**2)

    # Combined flux
    flux = time_profile * wave_profile * flux_scale

    return phase, wave, flux


# Example usage
if __name__ == "__main__":
    import sys

    # Check for data file argument
    if len(sys.argv) < 2:
        print("Usage: python custom_sed_likelihood.py <data.npz>")
        print("\nGenerating test data with Gaussian SED...")

        from generate_synthetic_data import generate_custom_sed_data
        data = generate_custom_sed_data()

        # Extract grids from generated data
        grids = data['model_grids']
        if isinstance(grids, np.ndarray):
            grids = grids.item()
        phase_grid = grids['phase']
        wave_grid = grids['wave']
        flux_grid = grids['flux']
    else:
        data = np.load(sys.argv[1], allow_pickle=True)

        # Try to get model grids
        if 'model_grids' in data:
            grids = data['model_grids']
            if isinstance(grids, np.ndarray):
                grids = grids.item()
            phase_grid = grids['phase']
            wave_grid = grids['wave']
            flux_grid = grids['flux']
        else:
            print("No model_grids in data file, creating default Gaussian SED")
            phase_grid, wave_grid, flux_grid = create_gaussian_sed()

    # Extract observations
    obs_phases = data['phases']
    obs_fluxes = data['fluxes']
    obs_errors = data['fluxerrs']
    obs_bands = data['bands']
    obs_zps = data['zps']

    print(f"Loaded {len(obs_phases)} observations")
    print(f"SED grid: {len(phase_grid)} phases x {len(wave_grid)} wavelengths")

    # Set up likelihood
    logL, source = setup_likelihood(
        phase_grid, wave_grid, flux_grid,
        obs_phases, obs_fluxes, obs_errors, obs_bands, obs_zps
    )

    print(f"Source parameters: {source.param_names}")

    # Get true amplitude if available
    if 'true_params' in data:
        true_p = data['true_params']
        if isinstance(true_p, np.ndarray):
            true_p = true_p.item()
        true_amp = true_p.get('amplitude', 1.0)
    else:
        true_amp = 1.0

    # Evaluate at true amplitude
    logL_true = logL(true_amp)
    print(f"\nLog-likelihood at true amplitude ({true_amp}): {logL_true:.2f}")

    # Grid search for best amplitude
    print("\nGrid search for best amplitude...")
    test_amps = jnp.linspace(0.5, 5.0, 50)
    logL_values = jax.vmap(logL)(test_amps)

    best_idx = jnp.argmax(logL_values)
    best_amp = float(test_amps[best_idx])

    print(f"Best-fit amplitude: {best_amp:.3f}")
    print(f"True amplitude: {true_amp}")
    print(f"Difference: {abs(best_amp - true_amp):.4f}")
