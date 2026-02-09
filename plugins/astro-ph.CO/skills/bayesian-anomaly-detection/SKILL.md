---
name: bayesian-anomaly-detection
description: Flag anomalies and outliers in data using Bayesian inference. Use when the user asks to "flag the anomalies in this data", "which points are outliers", "find the anomalies", "flag outliers", "clean my data of outliers", "detect anomalies", "robust fit ignoring outliers", "mitigate RFI", "handle contaminated data", "anomaly detection", "outlier detection", or mentions flagging bad data points, anomaly-corrected likelihood, or epsilon mask.
---

# Bayesian Anomaly Detection

## Overview

Identify anomalies in any dataset by jointly inferring model parameters and per-point anomaly probabilities. Rather than pre-processing to remove outliers, this method modifies your likelihood function so that the sampler **automatically determines how likely each data point is to be anomalous** and returns robust parameter estimates.

**What you get:** For each data point, a continuous probability of being anomalous (e.g. 0.02 = almost certainly clean, 0.95 = almost certainly anomalous). These are not binary classifications -- they are posterior probabilities from the weighted mean over the epsilon mask, capturing genuine uncertainty about borderline points. For your model parameters, unbiased estimates even when the data contains outliers.

Key features:
- **Continuous anomaly probabilities**: Each data point gets a probability of being anomalous, not a hard yes/no
- **Model-agnostic**: Works with any likelihood-based inference (linear regression, spectral fitting, cosmological parameter estimation, etc.)
- **Minimal code changes**: Only 3 lines modify a standard likelihood function
- **No manual thresholds**: The data and model jointly determine anomaly probabilities

## Workflow Position

This skill provides components **(2) Science** and **(3) Inference** in the physics workflow:

```
(1) Research/Brainstorm -> (2) Science -> (3) Inference -> (4) Visualization
```

**Inputs:** Any dataset with potential outliers/anomalies and a parametric model
**Outputs:** Per-point anomaly probabilities and robust parameter estimates

Connects to any sampler (MCMC, nested sampling, optimisation).

## Mathematical Framework

### Binary Anomaly Mask

Introduce a binary anomaly mask $\varepsilon_i$ for each data point $i$:

$$\varepsilon_i = \begin{cases} 0 & \text{if data point } i \text{ is expected} \\ 1 & \text{if data point } i \text{ is anomalous} \end{cases}$$

### Bernoulli Prior

Each mask element follows a Bernoulli prior:

$$P(\varepsilon_i) = p^{\varepsilon_i} (1-p)^{1-\varepsilon_i}$$

where $p$ is the prior probability that any given data point is anomalous. This can be fixed or treated as a free parameter (recommended).

### Piecewise Likelihood

The joint likelihood over data and anomaly masks:

$$P(D, \varepsilon | \theta) = \prod_{i=1}^N \left[ \mathcal{L}_i(\theta)(1-p) \right]^{1-\varepsilon_i} \left[ \frac{p}{\Delta} \right]^{\varepsilon_i}$$

where:
- $\mathcal{L}_i(\theta)$ is the standard likelihood for data point $i$
- $\Delta$ is a normalisation constant (data range scale)
- Anomalous points contribute a flat likelihood $p/\Delta$

### Max-Approximation for Marginalisation

Marginalising over all $2^N$ possible mask configurations is intractable. The key insight is to approximate by taking the **most likely mask** for each parameter set $\theta$. This reduces to a simple per-point maximum:

$$\log P(D|\theta) \approx \sum_{i=1}^N \max\!\Big(\log \mathcal{L}_i + \log(1-p),\;\; \log p - \log \Delta\Big)$$

This is the **anomaly-corrected log-likelihood**. Each data point either contributes its normal likelihood (if consistent with the model) or a flat anomaly penalty (if it is an outlier). The transition is automatic and determined by the data.

## Quick Start: Linear Regression with Outliers

### 1. Generate Data with Anomalies

```python
import numpy as np

np.random.seed(123)
N = 25
x = np.linspace(0, 25, N)
m_true, c_true, sig_true = 1.0, 1.0, 2.0
y = m_true * x + c_true + np.random.randn(N) * sig_true

# Inject two anomalies
y[10] += 100
y[15] += 100
```

### 2. Standard Likelihood (No Anomaly Correction)

```python
def log_likelihood_standard(theta, x, y):
    m, c, sig = theta[0], theta[1], theta[2]
    sig = max(sig, 1e-6)
    y_pred = m * x + c
    logL = -0.5 * ((y_pred - y)**2) / sig**2 - 0.5 * np.log(2 * np.pi * sig**2)
    return logL.sum()
```

### 3. Anomaly-Corrected Likelihood (The Key Modification)

Only three lines differ from the standard likelihood:

```python
def log_likelihood_anomaly(theta, x, y, delta):
    m, c, sig, logp = theta[0], theta[1], theta[2], theta[3]
    sig = max(sig, 0.1)
    p = np.exp(np.clip(logp, -20, -0.1))
    y_pred = m * x + c

    # Standard per-point log-likelihood
    logL_normal = -0.5 * ((y_pred - y)**2) / sig**2 - 0.5 * np.log(2 * np.pi * sig**2)

    # --- Three key lines ---
    logL_with_prior = logL_normal + np.log(1 - p)          # weight by P(not anomaly)
    anomaly_threshold = logp - np.log(delta)                # flat anomaly likelihood
    logL_corrected = np.maximum(logL_with_prior, anomaly_threshold)  # per-point max

    return logL_corrected.sum()
```

Set `delta = np.max(y)` (data range scale) and sample `logp` as a free parameter with prior $\log p \in [-10, -0.1]$.

### 4. MCMC Sampling

```python
def run_mcmc(log_likelihood, n_params, n_samples=5000, n_burn=1000):
    samples = np.zeros((n_samples, n_params))
    current = np.array([1.0, 1.0, 2.0, -3.0])[:n_params]
    current_logL = log_likelihood(current)
    step_size = 0.1
    bounds_lo = np.array([-2, -20, 0.1, -10])[:n_params]
    bounds_hi = np.array([4, 20, 20, -0.1])[:n_params]

    for i in range(n_samples + n_burn):
        proposal = current + np.random.normal(0, step_size, n_params)
        proposal = np.clip(proposal, bounds_lo, bounds_hi)
        proposal_logL = log_likelihood(proposal)

        if np.random.rand() < np.exp(min(0, proposal_logL - current_logL)):
            current, current_logL = proposal, proposal_logL

        if i >= n_burn:
            samples[i - n_burn] = current

    return samples

delta = np.max(y)
samples_std = run_mcmc(lambda t: log_likelihood_standard(t, x, y), 3)
samples_anom = run_mcmc(lambda t: log_likelihood_anomaly(t, x, y, delta), 4)

print(f"Standard:  m={samples_std[:,0].mean():.2f}, c={samples_std[:,1].mean():.2f}, "
      f"sig={samples_std[:,2].mean():.2f}")
print(f"Corrected: m={samples_anom[:,0].mean():.2f}, c={samples_anom[:,1].mean():.2f}, "
      f"sig={samples_anom[:,2].mean():.2f}")
```

The anomaly-corrected fit recovers the true parameters ($m=1$, $c=1$, $\sigma=2$) even with extreme outliers, while the standard fit is severely biased.

### 5. Posterior Anomaly Probabilities

Track the epsilon mask across posterior samples to get a continuous probability that each data point is anomalous:

```python
def log_likelihood_with_mask(theta, x, y, delta):
    m, c, sig, logp = theta[0], theta[1], theta[2], theta[3]
    sig = max(sig, 0.1)
    p = np.exp(np.clip(logp, -20, -0.1))
    y_pred = m * x + c
    logL_normal = -0.5 * ((y_pred - y)**2) / sig**2 - 0.5 * np.log(2 * np.pi * sig**2)
    logL_with_prior = logL_normal + np.log(1 - p)
    anomaly_threshold = logp - np.log(delta)
    logL_corrected = np.maximum(logL_with_prior, anomaly_threshold)
    epsilon_mask = (logL_with_prior < anomaly_threshold).astype(int)
    return logL_corrected.sum(), epsilon_mask

# Collect masks during sampling, then compute weighted mean:
posterior_fraction = np.average(epsilon_masks, weights=weights, axis=0)
# For equally-weighted MCMC samples, weights = np.ones(n_samples)
# For nested sampling, use the posterior weights from the sampler
```

Points 10 and 15 will have high anomaly probabilities (near 1.0), while clean data points will have low probabilities (near 0.0). Borderline points will have intermediate values, reflecting genuine uncertainty.

## How to Get Anomaly Probabilities for Your Data

The recipe for any model:

1. **Start with your existing likelihood** that computes per-point `logL_normal[i]`
2. **Set `delta`** to the scale of your data (e.g., `np.max(np.abs(y))`)
3. **Add `logp` as a free parameter** with prior $\log p \in [-10, -0.1]$
4. **Add three lines** to your likelihood:
   ```python
   logL_with_prior = logL_normal + np.log(1 - p)
   anomaly_threshold = logp - np.log(delta)
   logL_corrected = np.maximum(logL_with_prior, anomaly_threshold)
   ```
5. **Run your sampler** (MCMC, nested sampling, etc.) and collect the epsilon mask at each sample
6. **Compute the posterior anomaly probability** for each data point:
   ```python
   anomaly_prob = np.average(epsilon_masks, weights=weights, axis=0)
   # anomaly_prob[i] near 1 → probably anomalous
   # anomaly_prob[i] near 0 → probably clean
   # intermediate values → genuine uncertainty
   ```

This works for any model: spectral fitting, radio frequency interference mitigation, cosmological parameter estimation, time series analysis, or any parametric model with potential outliers.

For large datasets with many anomalies, see Anstey & Leeney (2023) for likelihood reweighting methods achieving ~25x speedup.

## Citation

If you use this method, please cite:

```bibtex
@article{Leeney2023,
    author = {Leeney, Samuel K. and Handley, William J. and Sherrill, Bradley and Sherrill, Neil},
    title = "{Bayesian approach to radio frequency interference mitigation}",
    journal = {Physical Review D},
    volume = {108},
    number = {6},
    pages = {062006},
    year = {2023},
    doi = {10.1103/PhysRevD.108.062006},
    eprint = {2211.15448},
    archivePrefix = {arXiv},
    primaryClass = {astro-ph.CO}
}

@article{AnsteyLeeney2023,
    author = {Anstey, Dominic and Leeney, Samuel K.},
    title = "{Enhanced Bayesian RFI Mitigation and Transient Flagging Using Likelihood Reweighting}",
    year = {2023},
    eprint = {2310.02146},
    archivePrefix = {arXiv},
    primaryClass = {astro-ph.IM}
}
```

## Bundled Scripts

- **bayesian_anomaly_detection_example.py**: Self-contained demo comparing standard vs anomaly-corrected inference on linear regression with injected outliers. Pure numpy, no external dependencies.
  ```bash
  python bayesian_anomaly_detection_example.py
  ```
