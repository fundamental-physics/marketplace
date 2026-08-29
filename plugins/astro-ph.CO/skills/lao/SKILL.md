---
name: lao
description: Use when the user mentions 'LAO', 'Laplace approximation optimization', asks about marginalizing nuisance parameters for evidence calculation, or needs to compute Bayesian evidence in high-dimensional (30+D) parameter spaces where some parameters can be assumed Gaussian.
---

# LAO (Laplace Approximation Optimization)

Marginalize nuisance parameters via optimization + Laplace approximation, preserving nested sampling for parameters of interest.

## When to Use

- Total dimensionality >30 where full nested sampling is impractical
- Clear split between parameters of interest (θ) and nuisance parameters (φ)
- Nuisance parameters approximately Gaussian given θ
- Need Bayesian evidence for model comparison
- Differentiable likelihood

## Core Equation

```
L_marg(θ) ≈ L(θ, φ̂) · (2π)^(dim(φ)/2) · |H|^(-1/2)
```

Where:
- φ̂(θ) = MAP estimate from optimizing log L + log π over φ
- H = Hessian of negative log-posterior at φ̂

## Implementation Pattern

```python
@jax.jit
def laplace_marginal(theta, data):
    # 1. Pre-compute θ-dependent quantities outside optimization
    theory = compute_theory(theta)

    def neg_log_posterior(phi):
        return -(log_likelihood(theory, phi, data) + log_prior_phi(phi, theta))

    # 2. Find MAP
    phi_hat = optimize(neg_log_posterior, phi_init)

    # 3. Hessian and Laplace approximation
    H = jax.hessian(neg_log_posterior)(phi_hat)
    _, logdet_H = jnp.linalg.slogdet(H)

    log_post_map = -neg_log_posterior(phi_hat)
    return log_post_map - 0.5 * logdet_H + 0.5 * dim_phi * jnp.log(2 * jnp.pi)
```

## Key Techniques

### Pre-whitening

Improves optimizer conditioning when parameter scales differ:

```python
L = jnp.linalg.cholesky(H_fiducial)
L_inv = jnp.linalg.inv(L)

def objective_whitened(phi_white):
    return objective(L_inv.T @ phi_white)

phi_hat = L_inv.T @ optimize(objective_whitened, L.T @ phi_init)
```

### Hierarchical Models

For independent objects, use `jax.vmap` and exploit block-diagonal Hessian:

```python
marginals = jax.vmap(single_object_laplace, in_axes=(None, 0))(theta, object_data)
total = jnp.sum(marginals)
```

### Pre-computation

Separate expensive θ-dependent calculations from φ optimization:

```python
theory = compute_expensive_theory(theta)  # Once per θ
phi_hat = optimize(lambda phi: likelihood(theory, phi), phi_init)  # Many iterations
```

## Scaling

- Full nested sampling: O((dim θ + dim φ)³)
- LAO: O(dim(θ)³ × dim(φ))

## Limitations

- Assumes p(φ|θ,d) is approximately Gaussian
- Multimodal or heavy-tailed φ distributions may bias evidence
- Requires differentiable likelihood for automatic Hessian
