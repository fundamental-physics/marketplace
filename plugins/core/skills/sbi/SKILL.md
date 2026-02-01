---
name: sbi
description: Use this skill when the user asks about "simulation-based inference", "likelihood-free inference", "neural posterior estimation", "NPE", "NLE", "NRE", "SNPE", "SNLE", "SNRE", "amortized inference", or needs to estimate posterior distributions when the likelihood is intractable but a simulator is available. Provides guidance for training neural density estimators, sampling posteriors, and validating inference quality with sbi (the Python package).
---

# SBI (Simulation-Based Inference)

## Overview

SBI is a PyTorch-based toolkit for simulation-based inference (also called likelihood-free inference). It enables Bayesian parameter estimation when traditional likelihood calculations are intractable—you only need a simulator that generates synthetic observations given parameters.

**Key capabilities:**
- Train neural networks to approximate posteriors, likelihoods, or likelihood ratios
- Amortized inference: train once, condition on many observations
- Sequential/multi-round inference for simulation efficiency
- Embedding networks for high-dimensional observations
- Diagnostics: simulation-based calibration (SBC), TARP, expected coverage

**Workflow position:** This skill provides component **(3) Inference** in the physics workflow.

## Installation

```bash
uv pip install sbi
```

Requires Python 3.10+. GPU optional but recommended for large networks.

## Quick Start

The minimal sbi workflow requires three components: a **prior**, a **simulator**, and **observed data**.

```python
import torch
from sbi.inference import NPE
from sbi.utils import BoxUniform

# 1. Define prior over parameters
prior = BoxUniform(low=torch.zeros(3), high=torch.ones(3))

# 2. Define simulator: theta -> x
def simulator(theta):
    # Replace with your physics simulator
    return theta + 0.1 * torch.randn_like(theta)

# 3. Generate training data
theta = prior.sample((10_000,))
x = simulator(theta)

# 4. Train neural posterior estimator
inference = NPE(prior=prior)
inference.append_simulations(theta, x).train()
posterior = inference.build_posterior()

# 5. Sample posterior given observation
x_observed = torch.tensor([0.5, 0.3, 0.7])
samples = posterior.sample((10_000,), x=x_observed)
```

**Important:** The simulator need not be differentiable—sbi learns from input-output pairs only.

## Method Selection

SBI offers three method families. Choose based on your use case:

| Method | Learns | Sampling | Best for |
|--------|--------|----------|----------|
| **NPE** | p(θ\|x) directly | Direct sampling | Most use cases, fast posterior access |
| **NLE** | p(x\|θ) | MCMC required | When likelihood interpretability matters |
| **NRE** | p(x\|θ)/p(x) ratio | MCMC required | Classifier-based, stable training |

**Start with NPE** unless you have specific reasons for NLE/NRE. NPE provides direct sampling without MCMC.

For detailed method comparison and variants (SNPE, FMPE, TSNPE, etc.), see `references/methods.md`.

## Core Patterns

### Amortized Inference (Default)

Train once, condition on any observation:

```python
inference = NPE(prior=prior)
inference.append_simulations(theta, x).train()
posterior = inference.build_posterior()

# Reuse for multiple observations
for x_obs in observations:
    samples = posterior.sample((5000,), x=x_obs)
```

### Sequential/Multi-Round Inference

More simulation-efficient but loses amortization. Use when simulations are expensive and you have one specific observation:

```python
from sbi.inference import SNPE

inference = SNPE(prior=prior)
proposal = prior

for round_idx in range(3):
    theta = proposal.sample((1000,))
    x = simulator(theta)
    inference.append_simulations(theta, x, proposal=proposal).train()
    posterior = inference.build_posterior()
    proposal = posterior.set_default_x(x_observed)

samples = posterior.sample((10_000,), x=x_observed)
```

See `references/multi-round.md` for details.

### High-Dimensional Observations

For images, time series, or high-dimensional data, use embedding networks:

```python
from sbi.neural_nets import posterior_nn
from torch import nn

# CNN embedding for 2D data
embedding = nn.Sequential(
    nn.Conv2d(1, 32, 3), nn.ReLU(), nn.Flatten(),
    nn.Linear(32 * 26 * 26, 64)
)

density_estimator = posterior_nn(
    model="nsf",
    embedding_net=embedding
)

inference = NPE(prior=prior, density_estimator=density_estimator)
```

See `references/embedding-networks.md` for CNN/RNN patterns.

### Density Estimator Configuration

Three ways to configure the neural network:

```python
# 1. String shortcut (simplest)
inference = NPE(prior=prior, density_estimator="nsf")

# 2. Utility function (more control)
from sbi.neural_nets import posterior_nn
estimator = posterior_nn(model="nsf", hidden_features=128, num_transforms=8)
inference = NPE(prior=prior, density_estimator=estimator)

# 3. Custom architecture (full control)
# See references/methods.md
```

Available models: `"maf"`, `"nsf"`, `"mdn"`, `"made"`. NSF (Neural Spline Flows) recommended for most cases.

## Posterior Analysis

### Sampling and Evaluation

```python
# Sample from posterior
samples = posterior.sample((10_000,), x=x_observed)

# Evaluate log probability
log_prob = posterior.log_prob(theta_test, x=x_observed)

# MAP estimate
map_estimate = posterior.map(x=x_observed)
```

### Visualization with anesthetic

```python
from anesthetic import make_2d_axes
import matplotlib.pyplot as plt

# Convert to numpy for anesthetic
samples_np = samples.numpy()

fig, axes = make_2d_axes(['theta0', 'theta1', 'theta2'])
# Create anesthetic Samples object from numpy array
from anesthetic import Samples
posterior_samples = Samples(samples_np, columns=['theta0', 'theta1', 'theta2'])
posterior_samples.plot_2d(axes)
fig.savefig('posterior_corner.png')
```

### Posterior Predictive Checks

```python
# Draw parameters from posterior
theta_posterior = posterior.sample((1000,), x=x_observed)

# Simulate observations from posterior parameters
x_predictive = simulator(theta_posterior)

# Compare x_predictive distribution to x_observed
```

## Validation and Diagnostics

Always validate your posterior before reporting results. SBI provides three diagnostic tools:

| Diagnostic | Purpose | Simulations Needed |
|------------|---------|-------------------|
| **SBC** | Check calibration | ~200 |
| **Expected Coverage** | Credible interval accuracy | ~100-500 |
| **TARP** | Necessary & sufficient accuracy test | ~200 |

```python
from sbi.diagnostics import run_sbc, check_sbc

# Run simulation-based calibration
thetas, xs, ranks, _ = run_sbc(
    posterior, prior, simulator, num_simulations=200
)
check_sbc(ranks, thetas, prior.sample((1000,)))
```

See `references/diagnostics.md` for detailed guidance on interpreting results.

## Common Pitfalls

1. **Shape mismatches**: Ensure `x_observed` has same shape as simulator output. Flatten multi-dimensional outputs or use embedding networks.

2. **Prior bounds**: Parameters sampled outside prior bounds cause errors. Use `BoxUniform` or properly bounded distributions.

3. **Insufficient training data**: Start with 10k simulations for simple problems, scale up for complex posteriors.

4. **Overconfident posteriors**: Always run diagnostics. Narrow posteriors may indicate poor calibration.

5. **NLE/NRE without MCMC**: These methods require MCMC sampling. Use `posterior.sample()` which handles this automatically, but expect slower sampling than NPE.

## Reference Documentation

Detailed documentation by topic:

- **Method selection**: See `references/methods.md` for NPE/NLE/NRE variants and when to use each
- **Multi-round inference**: See `references/multi-round.md` for sequential refinement
- **Embedding networks**: See `references/embedding-networks.md` for high-dimensional observations
- **Diagnostics**: See `references/diagnostics.md` for SBC, TARP, and coverage tests
