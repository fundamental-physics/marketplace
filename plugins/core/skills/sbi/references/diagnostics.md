# Posterior Diagnostics and Validation

## Table of Contents

1. [Why Validate?](#why-validate)
2. [Simulation-Based Calibration (SBC)](#simulation-based-calibration-sbc)
3. [Expected Coverage](#expected-coverage)
4. [TARP](#tarp)
5. [Local C2ST](#local-c2st)
6. [Posterior Predictive Checks](#posterior-predictive-checks)
7. [Diagnostic Selection Guide](#diagnostic-selection-guide)

---

## Why Validate?

Neural posterior estimation can fail silently. The network may:
- Be overconfident (posteriors too narrow)
- Be underconfident (posteriors too wide)
- Be biased (posteriors systematically offset)
- Miss modes (multimodal posteriors approximated as unimodal)

**Always validate before reporting results.** Diagnostics require additional simulations but are essential for trustworthy inference.

---

## Simulation-Based Calibration (SBC)

SBC checks if the posterior is well-calibrated by verifying that posterior ranks follow a uniform distribution.

### Concept

For a well-calibrated posterior:
1. Draw θ* from prior
2. Simulate x* = simulator(θ*)
3. Draw samples from posterior p(θ|x*)
4. Compute rank of θ* among posterior samples

If calibrated, ranks should be uniformly distributed.

### Implementation

```python
from sbi.diagnostics import run_sbc, check_sbc

# Run SBC
thetas, xs, ranks, dap_samples = run_sbc(
    posterior=posterior,
    prior=prior,
    simulator=simulator,
    num_simulations=200,  # ~200 is usually sufficient
    num_posterior_samples=1000,
)

# Check calibration
fig, axes = check_sbc(
    ranks=ranks,
    prior_samples=prior.sample((1000,)),
    num_posterior_samples=1000,
)
fig.savefig('sbc_ranks.png')
```

### Interpreting Results

**Histogram of ranks should be uniform:**

| Pattern | Meaning | Action |
|---------|---------|--------|
| Uniform | Well-calibrated | Good! |
| U-shaped | Overconfident (too narrow) | Increase training data, wider architecture |
| Inverted U | Underconfident (too wide) | More training, narrower architecture |
| Skewed | Biased | Check simulator, prior, training |

### SBC Plots

```python
# Rank histogram per parameter
fig, axes = check_sbc(ranks, prior_samples, num_posterior_samples)

# Look for:
# - All histograms approximately flat
# - 95% of bars within expected range (shown as gray band)
```

---

## Expected Coverage

Checks if credible intervals contain the true parameter at the expected rate.

### Concept

A 90% credible interval should contain the true parameter 90% of the time across many simulations.

### Implementation

```python
from sbi.diagnostics import check_coverage

# Run coverage check
coverage_results = check_coverage(
    posterior=posterior,
    prior=prior,
    simulator=simulator,
    num_simulations=500,
    credible_levels=[0.5, 0.75, 0.9, 0.95],
)

# Plot coverage
import matplotlib.pyplot as plt

levels = [0.5, 0.75, 0.9, 0.95]
fig, ax = plt.subplots()
ax.plot(levels, coverage_results, 'o-', label='Observed')
ax.plot([0, 1], [0, 1], 'k--', label='Ideal')
ax.set_xlabel('Credible Level')
ax.set_ylabel('Observed Coverage')
ax.legend()
fig.savefig('coverage.png')
```

### Interpreting Results

| Pattern | Meaning |
|---------|---------|
| Points on diagonal | Well-calibrated |
| Points above diagonal | Conservative (overconfident intervals) |
| Points below diagonal | Anti-conservative (underconfident intervals) |

---

## TARP

TARP (Tests of Accuracy with Random Points) provides a necessary and sufficient condition for posterior accuracy.

### Concept

TARP tests whether the learned posterior matches the true posterior by comparing expected coverage to observed coverage using randomly sampled reference points.

### Implementation

```python
from sbi.diagnostics import run_tarp, check_tarp

# Run TARP
ecp, alpha = run_tarp(
    posterior=posterior,
    prior=prior,
    simulator=simulator,
    num_simulations=200,
    num_posterior_samples=1000,
)

# Check results
fig, ax = check_tarp(ecp, alpha)
fig.savefig('tarp.png')
```

### Interpreting Results

```python
# TARP produces expected coverage probability (ECP) vs. credibility level (alpha)
# Plot should follow diagonal for well-calibrated posterior

# Deviations indicate:
# - Above diagonal: posterior too narrow (overconfident)
# - Below diagonal: posterior too wide (underconfident)
```

### TARP vs SBC

| Aspect | SBC | TARP |
|--------|-----|------|
| Tests | Necessary condition | Necessary & sufficient |
| Hyperparameters | None | Requires tuning |
| Interpretation | Simple (uniform ranks) | Coverage plot |
| Recommendation | Start with SBC | Use TARP for thorough validation |

---

## Local C2ST

Classifier Two-Sample Test compares learned posterior to reference samples.

### Concept

Train a classifier to distinguish between:
- Samples from learned posterior
- Samples from reference distribution

If classifier accuracy ≈ 0.5, distributions are indistinguishable (good).

### Implementation

```python
from sbi.diagnostics import c2st

# Compare posterior samples to prior (basic sanity check)
posterior_samples = posterior.sample((1000,), x=x_observed)
prior_samples = prior.sample((1000,))

accuracy = c2st(posterior_samples, prior_samples)
print(f"C2ST accuracy: {accuracy:.3f}")

# Interpretation:
# ~0.5: Can't distinguish (if comparing to prior, posterior should differ!)
# ~1.0: Completely distinguishable
```

### When to Use

- Comparing posteriors from different methods
- Checking if posterior differs from prior (it should, unless data uninformative)
- Validating against ground truth when available

---

## Posterior Predictive Checks

Verify that data simulated from the posterior looks like the observed data.

### Implementation

```python
import torch
import matplotlib.pyplot as plt

# Sample parameters from posterior
theta_samples = posterior.sample((100,), x=x_observed)

# Simulate observations from posterior samples
x_predictive = torch.stack([simulator(theta) for theta in theta_samples])

# Compare distributions
fig, axes = plt.subplots(1, x_observed.shape[0], figsize=(12, 3))
for i, ax in enumerate(axes):
    ax.hist(x_predictive[:, i].numpy(), bins=30, alpha=0.7, label='Posterior predictive')
    ax.axvline(x_observed[i].item(), color='red', linewidth=2, label='Observed')
    ax.set_xlabel(f'x_{i}')
    ax.legend()
fig.tight_layout()
fig.savefig('posterior_predictive.png')
```

### Interpretation

- Observed data should fall within posterior predictive distribution
- Systematic mismatch suggests model misspecification
- Very narrow predictive distribution may indicate overfitting

---

## Diagnostic Selection Guide

### Recommended Workflow

```
1. Quick check: Posterior predictive
   └─ Does simulated data look like observation?

2. Calibration: SBC (always run this)
   └─ Are ranks uniform?

3. Thorough validation: TARP
   └─ Necessary and sufficient accuracy test

4. If comparing methods: C2ST
   └─ Are posteriors different?
```

### Minimum Validation Requirements

| Publication Stage | Recommended Diagnostics |
|------------------|------------------------|
| Exploration | Posterior predictive |
| Draft | SBC + Posterior predictive |
| Submission | SBC + TARP + Posterior predictive |
| Response to reviewers | Full suite including C2ST comparisons |

### Simulation Budget

| Diagnostic | Simulations Needed | Notes |
|------------|-------------------|-------|
| Posterior predictive | 100-500 | Uses existing posterior |
| SBC | ~200 | Independent simulations |
| Expected coverage | 100-500 | Independent simulations |
| TARP | ~200 | Independent simulations |
| C2ST | 0 | Uses existing samples |

**Total for thorough validation:** ~500-1000 additional simulations beyond training.
