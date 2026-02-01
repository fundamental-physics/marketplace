# SBI Methods Reference

## Table of Contents

1. [Method Families Overview](#method-families-overview)
2. [Neural Posterior Estimation (NPE)](#neural-posterior-estimation-npe)
3. [Neural Likelihood Estimation (NLE)](#neural-likelihood-estimation-nle)
4. [Neural Ratio Estimation (NRE)](#neural-ratio-estimation-nre)
5. [Method Selection Guide](#method-selection-guide)
6. [Density Estimator Configuration](#density-estimator-configuration)

---

## Method Families Overview

SBI implements three fundamental approaches to simulation-based inference:

| Family | Target | Sampling | Amortization | Key Trade-off |
|--------|--------|----------|--------------|---------------|
| NPE | p(θ\|x) | Direct | Full | Fast sampling, may struggle with sharp posteriors |
| NLE | p(x\|θ) | MCMC | Partial | Interpretable likelihood, slower sampling |
| NRE | r(x,θ) = p(x\|θ)/p(x) | MCMC | Partial | Stable training, requires good classifier |

---

## Neural Posterior Estimation (NPE)

Directly learns the posterior distribution p(θ|x) using normalizing flows or mixture density networks.

### Variants

| Class | Description | Use Case |
|-------|-------------|----------|
| `NPE` | Amortized NPE (alias for NPE_C) | Default choice |
| `SNPE` | Sequential NPE with proposal correction | Single observation, simulation-efficient |
| `SNPE_A` | Original sequential NPE | Legacy |
| `SNPE_C` | APT (Automatic Posterior Transformation) | Most robust sequential variant |
| `FMPE` | Flow Matching Posterior Estimation | Alternative to normalizing flows |
| `TSNPE` | Truncated Sequential NPE | Bounded parameter spaces |

### Basic Usage

```python
from sbi.inference import NPE

inference = NPE(prior=prior, density_estimator="nsf")
inference.append_simulations(theta, x).train()
posterior = inference.build_posterior()

# Direct sampling - no MCMC needed
samples = posterior.sample((10_000,), x=x_observed)
```

### When to Use NPE

- **Default choice** for most problems
- Fast posterior sampling required
- Analyzing multiple observations (amortization)
- Moderate-dimensional parameter spaces (< 20 params)

### Limitations

- May struggle with very sharp/multimodal posteriors
- Density estimation in high dimensions is challenging
- Sequential variants lose amortization

---

## Neural Likelihood Estimation (NLE)

Learns the likelihood p(x|θ) as a conditional density estimator. Requires MCMC for posterior sampling.

### Variants

| Class | Description | Use Case |
|-------|-------------|----------|
| `NLE` | Amortized NLE (alias for NLE_A) | Default NLE |
| `SNLE` | Sequential NLE | Single observation |
| `MNLE` | Mixed NLE | Decision-making applications |

### Basic Usage

```python
from sbi.inference import NLE

inference = NLE(prior=prior, density_estimator="nsf")
inference.append_simulations(theta, x).train()
posterior = inference.build_posterior()

# Uses MCMC internally
samples = posterior.sample((10_000,), x=x_observed)
```

### When to Use NLE

- Need interpretable likelihood function
- Want to combine learned likelihood with analytic prior
- Likelihood structure is important for the science
- Parameter space is low-dimensional (MCMC scales poorly)

### Limitations

- Slower sampling due to MCMC
- MCMC tuning may be required
- Less sample-efficient than NPE for simple problems

---

## Neural Ratio Estimation (NRE)

Learns the likelihood-to-evidence ratio r(x,θ) = p(x|θ)/p(x) via classification.

### Variants

| Class | Description | Use Case |
|-------|-------------|----------|
| `NRE` | Amortized NRE (alias for NRE_A) | Default NRE |
| `NRE_A` | AALR (Amortized Approximate Likelihood Ratio) | Original method |
| `NRE_B` | SRE (Sequential Ratio Estimation) | Alternative formulation |
| `NRE_C` | Contrastive NRE | Most recent variant |
| `BNRE` | Balanced NRE | Improved calibration |
| `SNRE` | Sequential NRE | Single observation |

### Basic Usage

```python
from sbi.inference import NRE

inference = NRE(prior=prior, classifier="resnet")
inference.append_simulations(theta, x).train()
posterior = inference.build_posterior()

# Uses MCMC internally
samples = posterior.sample((10_000,), x=x_observed)
```

### When to Use NRE

- Classification is easier than density estimation for your problem
- High-dimensional observations (classifier can be more stable)
- Want ratio for model comparison (Bayes factors)

### Limitations

- Requires MCMC for sampling
- Classifier quality directly affects posterior quality
- May need careful network architecture tuning

---

## Method Selection Guide

### Decision Tree

```
Start here
    │
    ├─ Need fast posterior sampling?
    │   ├─ Yes → NPE
    │   └─ No → Continue
    │
    ├─ Need interpretable likelihood?
    │   ├─ Yes → NLE
    │   └─ No → Continue
    │
    ├─ High-dimensional observations (images, time series)?
    │   ├─ Yes → NRE (classifier often more stable) or NPE with embedding
    │   └─ No → NPE
    │
    └─ Default → NPE
```

### Amortized vs Sequential

| Scenario | Recommendation |
|----------|----------------|
| Many observations to analyze | Amortized (NPE, NLE, NRE) |
| Single observation, simulations expensive | Sequential (SNPE, SNLE, SNRE) |
| Exploring different observations interactively | Amortized |
| Publication-quality inference on one dataset | Sequential (more simulation-efficient) |

---

## Density Estimator Configuration

### String Shortcuts

```python
# Normalizing flows
inference = NPE(prior=prior, density_estimator="nsf")  # Neural Spline Flow (recommended)
inference = NPE(prior=prior, density_estimator="maf")  # Masked Autoregressive Flow

# Mixture models
inference = NPE(prior=prior, density_estimator="mdn")  # Mixture Density Network
inference = NPE(prior=prior, density_estimator="made") # MADE
```

### Utility Functions

```python
from sbi.neural_nets import posterior_nn, likelihood_nn, classifier_nn

# NPE configuration
estimator = posterior_nn(
    model="nsf",
    hidden_features=128,      # Width of hidden layers
    num_transforms=8,         # Number of flow layers
    num_bins=10,              # Spline bins (NSF only)
    embedding_net=None,       # Optional observation embedding
)
inference = NPE(prior=prior, density_estimator=estimator)

# NLE configuration
estimator = likelihood_nn(
    model="maf",
    hidden_features=64,
    num_transforms=5,
)
inference = NLE(prior=prior, density_estimator=estimator)

# NRE configuration
classifier = classifier_nn(
    model="resnet",
    hidden_features=64,
)
inference = NRE(prior=prior, classifier=classifier)
```

### Flow Architecture Recommendations

| Problem Complexity | Recommended Configuration |
|-------------------|--------------------------|
| Simple (< 5 params, unimodal) | `"maf"` or `"nsf"` with defaults |
| Moderate (5-15 params) | `posterior_nn("nsf", num_transforms=8)` |
| Complex (multimodal, correlations) | `posterior_nn("nsf", num_transforms=12, hidden_features=128)` |
| Very high-dimensional | Consider NRE or use embedding networks |
