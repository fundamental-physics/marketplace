---
name: anesthetic
description: Use this skill when the user asks to "create corner plots", "visualize posteriors", "plot chains", "analyze nested sampling output", "plot marginal distributions", "compare prior and posterior", "make triangle plots", or work with posterior samples from PolyChord, MultiNest, UltraNest, Cobaya, or other Bayesian inference tools. Provides complete guidance for creating publication-quality posterior visualizations from nested sampling or MCMC chains.
---

# Anesthetic

## Overview

Anesthetic is a Python package for visualizing and analyzing posterior samples from Bayesian inference. It creates publication-quality corner plots, 1D/2D marginal distributions, and computes Bayesian statistics (evidence, KL divergence) from nested sampling or MCMC chains.

**Key capabilities:**
- Load chains from PolyChord, MultiNest, UltraNest, Cobaya, GetDist formats
- Create corner/triangle plots with KDE, histogram, or scatter representations
- Compare prior vs posterior distributions
- Compute log-evidence, KL divergence, model dimensionality
- Built on pandas DataFrames for easy parameter manipulation

**Workflow position:** This skill provides component **(4) Visualization** in the physics workflow.

## Installation

First, ensure anesthetic is installed in your environment:

```bash
# Using uv (recommended)
uv pip install anesthetic

# Or using pip
pip install anesthetic

# Or using conda
conda install -c conda-forge anesthetic
```

Optional for faster KDE: `uv pip install fastkde`

## Quick Start

```python
from anesthetic import read_chains, make_2d_axes
import matplotlib.pyplot as plt

# Load posterior chains (auto-detects format)
samples = read_chains('path/to/chains_root')

# Create corner plot (two-step process)
fig, axes = make_2d_axes(['x0', 'x1', 'x2'])  # Create axes first
samples.plot_2d(axes)                          # Then plot onto them
fig.savefig('corner.png')
plt.close(fig)
```

**Important:** Always create axes first with `make_2d_axes()` or `make_1d_axes()`, then pass them to `plot_2d()` or `plot_1d()`. The plot methods return axes, not (fig, axes) tuples.

## Core Workflow

1. **Load chains** - `read_chains()` auto-detects format, or use format-specific readers
2. **Select parameters** - Choose which parameters to plot
3. **Create axes** - `make_1d_axes()` or `make_2d_axes()`
4. **Plot** - `samples.plot_1d()` or `samples.plot_2d()`
5. **Customize** - Labels, colors, reference lines

## Reference Documentation

Detailed documentation is split by topic:

- **Loading data**: See `references/loading-chains.md` for all supported formats and creating samples from arrays
- **Plotting**: See `references/plotting-guide.md` for 1D/2D plots, plot kinds, and multi-chain comparisons
- **Customization**: See `references/customization.md` for labels, colors, axes, legends, and reference values
- **Statistics**: See `references/statistics.md` for Bayesian evidence, KL divergence, and MCMC diagnostics
- **Full API**: See `references/api-reference.md` for complete function signatures and options

## Common Patterns

### Basic Corner Plot

```python
from anesthetic import read_chains, make_2d_axes

samples = read_chains('chains/run')
fig, axes = make_2d_axes(['param1', 'param2', 'param3'])
samples.plot_2d(axes, kind='kde')
fig.savefig('corner.png')
```

### Prior vs Posterior Comparison

```python
samples = read_chains('chains/run')  # NestedSamples
prior = samples.prior()

fig, axes = make_2d_axes(['x0', 'x1', 'x2'])
prior.plot_2d(axes, kind='kde', label='Prior', alpha=0.5)
samples.plot_2d(axes, kind='kde', label='Posterior')
axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)), loc='lower center')
```

### Multiple Chain Comparison

```python
samples1 = read_chains('chains/model1')
samples2 = read_chains('chains/model2')

fig, axes = make_2d_axes(['x0', 'x1'])
samples1.plot_2d(axes, kind='kde', label='Model 1')
samples2.plot_2d(axes, kind='kde', label='Model 2')
axes.iloc[-1, 0].legend()
```

### Compute Bayesian Evidence

```python
samples = read_chains('chains/run')  # Must be NestedSamples
print(f"log(Z) = {samples.logZ():.2f}")
print(f"D_KL = {samples.D_KL():.2f}")

# With uncertainties
stats = samples.stats(nsamples=1000)
print(f"log(Z) = {stats['logZ'].mean():.2f} +/- {stats['logZ'].std():.2f}")
```

### Plot Bayesian Statistics

```python
# stats is a Samples object, so use anesthetic's plot_2d
stats = samples.stats(nsamples=1000)
fig, axes = make_2d_axes(['logZ', 'D_KL', 'd_G'])
stats.plot_2d(axes)
fig.savefig('stats_corner.png')
```

### Add Truth Values

```python
fig, axes = make_2d_axes(['x0', 'x1', 'x2'])
samples.plot_2d(axes)
truth = {'x0': 0.3, 'x1': 0.8, 'x2': 0.7}
axes.axlines(truth, color='red', linestyle='--')
axes.scatter(truth, marker='*', s=100, c='red')
```

## Runnable Example

A complete example script is available at `scripts/example_plots.py`. Run it with:

```bash
python scripts/example_plots.py path/to/chains_root
```

This creates `corner_plot.png`, `1d_marginals.png`, and `prior_posterior.png`.
