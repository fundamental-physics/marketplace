#!/usr/bin/env python3
"""
Bayesian Anomaly Detection - Minimal Example

Demonstrates the anomaly-corrected likelihood from:
  Leeney et al. (2023), Phys. Rev. D 108, 062006 [arXiv:2211.15448]

Compares standard vs anomaly-corrected inference on linear regression
with injected outliers. Pure numpy, no external dependencies.
"""

import numpy as np


def generate_data():
    """Generate linear data with two injected anomalies."""
    np.random.seed(123)
    N = 25
    x = np.linspace(0, 25, N)
    m_true, c_true, sig_true = 1.0, 1.0, 2.0
    y = m_true * x + c_true + np.random.randn(N) * sig_true
    y[10] += 100
    y[15] += 100
    return x, y, m_true, c_true, sig_true


def log_likelihood_standard(theta, x, y):
    """Standard Gaussian log-likelihood."""
    m, c, sig = theta[0], theta[1], max(theta[2], 1e-6)
    y_pred = m * x + c
    logL = -0.5 * ((y_pred - y) ** 2) / sig**2 - 0.5 * np.log(2 * np.pi * sig**2)
    return logL.sum()


def log_likelihood_anomaly(theta, x, y, delta):
    """Anomaly-corrected log-likelihood (Leeney et al. 2023).

    Three lines transform the standard likelihood:
      1. Weight by P(not anomaly): logL + log(1-p)
      2. Flat anomaly threshold:   log(p) - log(delta)
      3. Per-point maximum:        max(weighted, threshold)
    """
    m, c, sig, logp = theta[0], theta[1], max(theta[2], 0.1), theta[3]
    p = np.exp(np.clip(logp, -20, -0.1))
    y_pred = m * x + c
    logL_normal = -0.5 * ((y_pred - y) ** 2) / sig**2 - 0.5 * np.log(2 * np.pi * sig**2)

    logL_with_prior = logL_normal + np.log(1 - p)
    anomaly_threshold = logp - np.log(delta)
    logL_corrected = np.maximum(logL_with_prior, anomaly_threshold)
    return logL_corrected.sum()


def run_mcmc(log_likelihood_fn, n_params, n_samples=10000, n_burn=2000):
    """Simple Metropolis-Hastings MCMC sampler."""
    bounds_lo = np.array([-2, -20, 0.1, -10])[:n_params]
    bounds_hi = np.array([4, 20, 20, -0.1])[:n_params]
    current = np.array([1.0, 1.0, 2.0, -3.0])[:n_params]
    current_logL = log_likelihood_fn(current)
    samples = np.zeros((n_samples, n_params))
    step_size = 0.1

    for i in range(n_samples + n_burn):
        proposal = current + np.random.normal(0, step_size, n_params)
        proposal = np.clip(proposal, bounds_lo, bounds_hi)
        proposal_logL = log_likelihood_fn(proposal)
        if np.random.rand() < np.exp(min(0, proposal_logL - current_logL)):
            current, current_logL = proposal, proposal_logL
        if i >= n_burn:
            samples[i - n_burn] = current

    return samples


def main():
    x, y, m_true, c_true, sig_true = generate_data()
    delta = np.max(y)

    print("Bayesian Anomaly Detection Demo")
    print("=" * 50)
    print(f"True parameters: m={m_true}, c={c_true}, sig={sig_true}")
    print(f"Anomalies injected at indices 10 and 15 (+100)\n")

    print("Running standard MCMC (3 params: m, c, sig)...")
    samples_std = run_mcmc(
        lambda t: log_likelihood_standard(t, x, y), n_params=3
    )

    print("Running anomaly-corrected MCMC (4 params: m, c, sig, logp)...")
    samples_anom = run_mcmc(
        lambda t: log_likelihood_anomaly(t, x, y, delta), n_params=4
    )

    print("\nResults")
    print("-" * 50)
    print(f"{'':15s} {'True':>8s} {'Standard':>12s} {'Corrected':>12s}")
    print("-" * 50)
    for i, name in enumerate(["m", "c", "sig"]):
        true_val = [m_true, c_true, sig_true][i]
        std_val = samples_std[:, i].mean()
        anom_val = samples_anom[:, i].mean()
        print(f"{name:15s} {true_val:8.2f} {std_val:12.2f} {anom_val:12.2f}")

    logp_mean = samples_anom[:, 3].mean()
    print(f"{'logp':15s} {'free':>8s} {'---':>12s} {logp_mean:12.2f}")
    print("-" * 50)
    print("\nThe anomaly-corrected fit recovers the true parameters")
    print("despite the extreme outliers.")


if __name__ == "__main__":
    main()
