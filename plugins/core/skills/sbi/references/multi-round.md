# Multi-Round (Sequential) Inference

## Table of Contents

1. [Overview](#overview)
2. [When to Use Sequential Inference](#when-to-use-sequential-inference)
3. [SNPE Workflow](#snpe-workflow)
4. [SNLE and SNRE](#snle-and-snre)
5. [Proposal Correction](#proposal-correction)
6. [Practical Considerations](#practical-considerations)

---

## Overview

Sequential (multi-round) inference iteratively refines the posterior by focusing simulations in high-probability regions. Each round:

1. Sample parameters from a proposal distribution
2. Run simulator to generate observations
3. Train/update the neural network
4. Use the learned posterior as the next proposal

This is more simulation-efficient than amortized inference but produces a posterior specific to one observation.

**Trade-off:**
- Amortized: ~10k simulations, works for any x_o
- Sequential: ~1-3k simulations, specific to one x_o

---

## When to Use Sequential Inference

**Use sequential when:**
- Simulations are computationally expensive
- You have one specific observation to analyze
- Simulation budget is limited
- Prior is very broad relative to posterior

**Use amortized when:**
- Simulations are cheap
- You need to analyze multiple observations
- Interactive exploration of different x_o values
- Building a reusable inference tool

---

## SNPE Workflow

### Basic Multi-Round Loop

```python
import torch
from sbi.inference import SNPE
from sbi.utils import BoxUniform

prior = BoxUniform(low=torch.zeros(3), high=torch.ones(3))
x_observed = torch.tensor([0.5, 0.3, 0.7])

inference = SNPE(prior=prior)
proposal = prior

num_rounds = 3
simulations_per_round = 1000

for round_idx in range(num_rounds):
    # Sample from current proposal
    theta = proposal.sample((simulations_per_round,))
    x = simulator(theta)

    # Train with proposal information
    inference.append_simulations(theta, x, proposal=proposal)
    density_estimator = inference.train()

    # Build posterior and set as next proposal
    posterior = inference.build_posterior(density_estimator)
    proposal = posterior.set_default_x(x_observed)

    print(f"Round {round_idx + 1}: trained on {len(theta)} simulations")

# Final posterior
samples = posterior.sample((10_000,), x=x_observed)
```

### Key Points

1. **Pass `proposal` to `append_simulations`**: This enables importance-weighted training
2. **Use `set_default_x`**: Creates a proposal that samples from p(θ|x_o)
3. **First round uses prior**: `proposal = prior` initially
4. **Accumulate simulations**: Each round adds to the training set

---

## SNLE and SNRE

Sequential variants exist for all method families:

### SNLE (Sequential Neural Likelihood Estimation)

```python
from sbi.inference import SNLE

inference = SNLE(prior=prior)
proposal = prior

for round_idx in range(num_rounds):
    theta = proposal.sample((simulations_per_round,))
    x = simulator(theta)

    inference.append_simulations(theta, x, proposal=proposal)
    likelihood_estimator = inference.train()

    posterior = inference.build_posterior(likelihood_estimator)
    proposal = posterior.set_default_x(x_observed)

samples = posterior.sample((10_000,), x=x_observed)
```

### SNRE (Sequential Neural Ratio Estimation)

```python
from sbi.inference import SNRE

inference = SNRE(prior=prior)
proposal = prior

for round_idx in range(num_rounds):
    theta = proposal.sample((simulations_per_round,))
    x = simulator(theta)

    inference.append_simulations(theta, x, proposal=proposal)
    classifier = inference.train()

    posterior = inference.build_posterior(classifier)
    proposal = posterior.set_default_x(x_observed)

samples = posterior.sample((10_000,), x=x_observed)
```

---

## Proposal Correction

When training on samples from a proposal q(θ) ≠ p(θ), the network must correct for this bias.

### Automatic Correction (SNPE-C / APT)

SNPE-C (the default SNPE) uses Automatic Posterior Transformation to handle proposal mismatch:

```python
from sbi.inference import SNPE

# SNPE automatically applies APT correction
inference = SNPE(prior=prior)  # Uses SNPE_C internally
```

### Manual Importance Weighting

For custom workflows, you can compute importance weights:

```python
# Weights for samples from proposal q when targeting prior p
log_weights = prior.log_prob(theta) - proposal.log_prob(theta)
weights = torch.exp(log_weights - log_weights.max())
weights = weights / weights.sum()
```

---

## Practical Considerations

### Number of Rounds

| Scenario | Recommended Rounds |
|----------|-------------------|
| Simple posterior, cheap simulations | 1-2 |
| Moderate complexity | 2-3 |
| Complex posterior, expensive simulations | 3-5 |
| Diminishing returns typically after | 4-5 |

### Simulations Per Round

```python
# Typical allocation
round_1 = 2000  # More for initial exploration
round_2 = 1000  # Refine
round_3 = 1000  # Final polish

# Or equal allocation
simulations_per_round = total_budget // num_rounds
```

### Convergence Monitoring

```python
# Track posterior entropy or spread across rounds
for round_idx in range(num_rounds):
    # ... training ...

    samples = posterior.sample((1000,), x=x_observed)
    spread = samples.std(dim=0).mean()
    print(f"Round {round_idx + 1}: posterior spread = {spread:.4f}")
```

### Avoiding Proposal Collapse

If the proposal becomes too narrow, the posterior may miss important regions:

```python
# Option 1: Mix proposal with prior
from torch.distributions import MixtureSameFamily, Categorical

def make_mixed_proposal(posterior, prior, x_o, prior_weight=0.1):
    """Mix learned posterior with prior to prevent collapse."""
    # Sample from mixture: 90% posterior, 10% prior
    n_samples = 1000
    n_prior = int(n_samples * prior_weight)
    n_posterior = n_samples - n_prior

    theta_prior = prior.sample((n_prior,))
    theta_posterior = posterior.sample((n_posterior,), x=x_o)
    return torch.cat([theta_prior, theta_posterior])

# Option 2: Truncate to prior support (TSNPE)
from sbi.inference import TSNPE
inference = TSNPE(prior=prior)  # Handles truncation automatically
```

### When Sequential Fails

Watch for these warning signs:

1. **Posterior collapses to point mass**: Increase simulations or reduce rounds
2. **Posterior doesn't change across rounds**: May have converged, or proposal correction failing
3. **Posterior excludes true value (in simulation studies)**: Check proposal correction, increase prior mixing
