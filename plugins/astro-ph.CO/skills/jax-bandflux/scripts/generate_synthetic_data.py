#!/usr/bin/env python3
"""Generate synthetic supernova photometric data with Gaussian noise.

Creates test data for SALT3 or TimeSeriesSource likelihood fitting.
Output: NumPy arrays ready for likelihood computation.

Usage:
    python generate_synthetic_data.py --model salt3 --output data.npz
    python generate_synthetic_data.py --model custom --output custom_data.npz
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax_supernovae import SALT3Source, TimeSeriesSource


def generate_salt3_data(
    x0=1.0e-4, x1=0.5, c=0.05, t0=0.0, z=0.05,
    bands=('bessellb', 'bessellv', 'bessellr'),
    obs_times=(-10, -5, 0, 5, 10, 15, 20, 25, 30),
    noise_level=0.05, zp=27.5, seed=42
):
    """Generate synthetic SALT3 observations.

    Args:
        x0: Amplitude parameter (default 1e-4)
        x1: Stretch parameter (default 0.5)
        c: Color parameter (default 0.05)
        t0: Peak time MJD offset (default 0.0)
        z: Redshift (default 0.05)
        bands: Tuple of band names (default bessell BVR)
        obs_times: Observation times relative to t0 (default -10 to +30 days)
        noise_level: Fractional flux uncertainty (default 0.05 = 5%)
        zp: Zero point (default 27.5)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with arrays: times, fluxes, fluxerrs, bands, zps, true_params
    """
    np.random.seed(seed)
    source = SALT3Source()
    params = {'x0': x0, 'x1': x1, 'c': c}

    # Convert observer times to rest-frame phases
    phases = (np.array(obs_times) - t0) / (1.0 + z)

    times, fluxes, errors, band_list = [], [], [], []

    for band in bands:
        true_flux = np.array(source.bandflux(params, band, phases, zp=zp, zpsys='ab'))
        flux_err = np.abs(true_flux) * noise_level
        noisy_flux = true_flux + np.random.normal(0, flux_err)

        times.extend(obs_times)
        fluxes.extend(noisy_flux)
        errors.extend(flux_err)
        band_list.extend([band] * len(obs_times))

    return {
        'times': np.array(times),
        'fluxes': np.array(fluxes),
        'fluxerrs': np.array(errors),
        'bands': np.array(band_list),
        'zps': np.full(len(times), zp),
        'true_params': {'t0': t0, 'x0': x0, 'x1': x1, 'c': c, 'z': z}
    }


def generate_custom_sed_data(
    amplitude=2.5,
    obs_phases=(-10, -5, 0, 5, 10, 15, 20, 25),
    bands=('bessellb', 'bessellv', 'bessellr'),
    noise_level=0.05, zp=25.0, seed=123
):
    """Generate synthetic TimeSeriesSource observations.

    Creates a Gaussian SED model and generates noisy observations.

    Args:
        amplitude: True amplitude scaling (default 2.5)
        obs_phases: Observation phases in days (default -10 to +25)
        bands: Tuple of band names (default bessell BVR)
        noise_level: Fractional flux uncertainty (default 0.05 = 5%)
        zp: Zero point (default 25.0)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with arrays: phases, fluxes, fluxerrs, bands, zps,
                               true_params, model_grids
    """
    np.random.seed(seed)

    # Create Gaussian SED model
    phase = np.linspace(-20, 50, 100)
    wave = np.linspace(3000, 9000, 200)
    p_grid, w_grid = np.meshgrid(phase, wave, indexing='ij')

    # Gaussian time profile (peaked at phase=0)
    time_profile = np.exp(-0.5 * (p_grid / 12.0)**2)

    # Gaussian wavelength profile (peaked at 5500 Angstroms)
    wave_profile = np.exp(-0.5 * ((w_grid - 5500.0) / 1200.0)**2)

    # Combined flux (erg/s/cm^2/Angstrom)
    flux_grid = time_profile * wave_profile * 1e-15

    source = TimeSeriesSource(phase, wave, flux_grid, zero_before=True)
    params = {'amplitude': amplitude}

    # Cycle through bands
    n_obs = len(obs_phases)
    band_cycle = [bands[i % len(bands)] for i in range(n_obs)]

    fluxes, errors = [], []
    for ph, band in zip(obs_phases, band_cycle):
        true_flux = float(source.bandflux(params, band, ph, zp=zp, zpsys='ab'))
        flux_err = abs(true_flux) * noise_level
        noisy_flux = true_flux + np.random.normal(0, flux_err)
        fluxes.append(noisy_flux)
        errors.append(flux_err)

    return {
        'phases': np.array(obs_phases),
        'fluxes': np.array(fluxes),
        'fluxerrs': np.array(errors),
        'bands': np.array(band_cycle),
        'zps': np.full(n_obs, zp),
        'true_params': {'amplitude': amplitude},
        'model_grids': {'phase': phase, 'wave': wave, 'flux': flux_grid}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic supernova photometric data"
    )
    parser.add_argument(
        "--model",
        choices=["salt3", "custom"],
        default="salt3",
        help="Model type: salt3 (SALT3Source) or custom (TimeSeriesSource)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="synthetic_data.npz",
        help="Output file path (.npz format)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.05,
        help="Fractional noise level (default 0.05 = 5%%)"
    )
    args = parser.parse_args()

    print(f"Generating {args.model} synthetic data...")

    if args.model == "salt3":
        data = generate_salt3_data(seed=args.seed, noise_level=args.noise)
        print(f"  True params: x0={data['true_params']['x0']:.2e}, "
              f"x1={data['true_params']['x1']}, c={data['true_params']['c']}")
    else:
        data = generate_custom_sed_data(seed=args.seed, noise_level=args.noise)
        print(f"  True amplitude: {data['true_params']['amplitude']}")

    print(f"  Generated {len(data['fluxes'])} observations")

    # Save with allow_pickle for dict storage
    np.savez(args.output, **{k: v for k, v in data.items()})
    print(f"Saved to {args.output}")
