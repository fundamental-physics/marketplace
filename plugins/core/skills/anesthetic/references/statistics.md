# Bayesian Statistics

## Overview

For `NestedSamples`, anesthetic computes Bayesian statistics:

- **logZ**: Log-evidence (marginal likelihood)
- **D_KL**: Kullback-Leibler divergence from prior to posterior
- **logL_P**: Posterior-averaged log-likelihood
- **d_G**: Gaussian model dimensionality

These satisfy Occam's razor equation: `logZ = logL_P - D_KL`

## Computing Statistics

### Point Estimates

```python
from anesthetic import read_chains

samples = read_chains('chains/run')  # Must be NestedSamples

# Individual statistics
log_evidence = samples.logZ()
kl_divergence = samples.D_KL()
avg_logL = samples.logL_P()
dimensionality = samples.d_G()

print(f"log(Z) = {log_evidence:.2f}")
print(f"D_KL = {kl_divergence:.2f}")
print(f"<log L> = {avg_logL:.2f}")
print(f"d_G = {dimensionality:.2f}")
```

### All Statistics with Uncertainties

```python
# Get all stats with sampling uncertainty
stats = samples.stats(nsamples=1000)

print(stats)
#        logZ    D_KL   logL_P    d_G
# 0    -12.34   5.67   -6.67    3.21
# 1    -12.28   5.71   -6.57    3.18
# ...

# Mean and std
print(f"logZ = {stats['logZ'].mean():.2f} +/- {stats['logZ'].std():.2f}")
```

### Stats as Series (Point Estimates)

```python
# Without nsamples, returns a Series of mean values
stats = samples.stats()
print(stats)
# logZ     -12.31
# D_KL       5.69
# logL_P    -6.62
# d_G        3.20
```

## Model Comparison

Compare evidence between models:

```python
samples1 = read_chains('chains/model1')
samples2 = read_chains('chains/model2')

logZ1 = samples1.logZ()
logZ2 = samples2.logZ()

# Log Bayes factor
log_B = logZ1 - logZ2
print(f"ln(B_12) = {log_B:.2f}")

# Interpretation (Jeffreys scale)
if log_B > 5:
    print("Strong evidence for Model 1")
elif log_B > 2.5:
    print("Moderate evidence for Model 1")
elif log_B > 1:
    print("Weak evidence for Model 1")
elif log_B > -1:
    print("Inconclusive")
```

### With Uncertainties

```python
stats1 = samples1.stats(nsamples=1000)
stats2 = samples2.stats(nsamples=1000)

# Compute difference with proper error propagation
delta_logZ = stats1['logZ'] - stats2['logZ']
print(f"Delta logZ = {delta_logZ.mean():.2f} +/- {delta_logZ.std():.2f}")
```

### Normalized Statistics

```python
# Compare to a reference model
stats = samples.stats(nsamples=1000, norm=reference_stats)

# This adds Delta_logZ, Delta_D_KL, Delta_logL_P, Delta_d_G columns
print(stats[['Delta_logZ', 'Delta_D_KL']])
```

## Plotting Statistics

The `stats()` method returns a `Samples` object, so you can use anesthetic's `plot_2d` to visualize correlations between Bayesian statistics:

```python
from anesthetic import read_chains, make_2d_axes

samples = read_chains('chains/run')

# Get stats as a Samples object (nsamples draws from the distribution)
stats = samples.stats(nsamples=1000)

# Plot correlations between statistics using anesthetic
fig, axes = make_2d_axes(['logZ', 'D_KL', 'd_G'])
stats.plot_2d(axes)
fig.savefig('stats_corner.png')
```

This creates a corner plot showing the correlations and uncertainties between log-evidence, KL divergence, and model dimensionality.

You can also include `logL_P` (posterior-averaged log-likelihood):

```python
fig, axes = make_2d_axes(['logZ', 'D_KL', 'logL_P', 'd_G'])
stats.plot_2d(axes, kind='kde')
fig.savefig('full_stats.png')
```

### Comparing Statistics Between Models

Compare Bayesian statistics from multiple runs/models:

```python
from anesthetic import read_chains, make_2d_axes

samples1 = read_chains('chains/model1')
samples2 = read_chains('chains/model2')

stats1 = samples1.stats(nsamples=2000)
stats2 = samples2.stats(nsamples=2000)

params = ['logZ', 'D_KL', 'logL_P', 'd_G']
fig, axes = make_2d_axes(params, figsize=(6, 6), facecolor='w', upper=False)
stats1.plot_2d(axes, label='Model 1')
stats2.plot_2d(axes, label='Model 2')
axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes), len(axes)), loc='upper right')
fig.savefig('model_comparison_stats.png')
```

This visualizes how the evidence, KL divergence, and other statistics differ between models, including their uncertainties and correlations.

## Derived Parameters

Add computed parameters from existing ones:

```python
# Create derived parameter
samples['omega_m'] = samples['omegamh2'] / (samples['H0']/100)**2

# Set label for plotting
samples.set_label('omega_m', r'$\Omega_m$')

# Now plot includes derived parameter
samples.plot_2d(['omega_m', 'sigma8', 'H0'])
```

### Function-Based Derived Parameters

```python
import numpy as np

# More complex derived quantities
samples['log_mass'] = np.log10(samples['mass'])
samples['ratio'] = samples['param1'] / samples['param2']

samples.set_label('log_mass', r'$\log_{10}(M/M_\odot)$')
samples.set_label('ratio', r'$p_1/p_2$')
```

## MCMC Diagnostics

For `MCMCSamples`:

### Remove Burn-In

```python
from anesthetic import read_chains

mcmc = read_chains('mcmc_chains/run')  # MCMCSamples

# Remove first 30% of samples
mcmc_clean = mcmc.remove_burn_in(0.3)

# Remove first 1000 samples
mcmc_clean = mcmc.remove_burn_in(1000)

# Keep last 50%
mcmc_clean = mcmc.remove_burn_in(-0.5)

# Different burn-in per chain
mcmc_clean = mcmc.remove_burn_in([0.2, 0.3, 0.25])  # For 3 chains
```

### Gelman-Rubin Convergence

```python
# Total R-1 statistic (should be < 0.01 for good convergence)
R_minus_1 = mcmc_clean.Gelman_Rubin()
print(f"R-1 = {R_minus_1:.4f}")

# Per-parameter R-1
R_minus_1_total, R_minus_1_per_param = mcmc_clean.Gelman_Rubin(per_param=True)
print(R_minus_1_per_param)

# Only check specific parameters
R_minus_1 = mcmc_clean.Gelman_Rubin(params=['x0', 'x1'])
```

## Temperature Scaling

For thermodynamic analysis:

```python
# Get samples at different temperatures
beta = 0.5  # Inverse temperature (beta=1 is posterior, beta=0 is prior)
samples_tempered = samples.set_beta(beta)

# Compute statistics at multiple temperatures
import numpy as np
betas = np.linspace(0, 1, 100)
stats = samples.stats(beta=betas)

# Plot logZ vs temperature
import matplotlib.pyplot as plt
plt.plot(betas, stats['logZ'])
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\ln Z(\beta)$')
```
