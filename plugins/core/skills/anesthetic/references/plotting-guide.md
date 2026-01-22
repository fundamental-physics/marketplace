# Plotting Guide

**Important:** Always create axes first with `make_1d_axes()` or `make_2d_axes()`, then pass them to `plot_1d()` or `plot_2d()`. The plot methods return axes only, not (fig, axes) tuples.

## 1D Marginal Plots

### Basic 1D Plot

```python
from anesthetic import read_chains, make_1d_axes

samples = read_chains('chains/run')

# Create axes first, then plot
fig, axes = make_1d_axes(['x0', 'x1', 'x2'])
samples.plot_1d(axes)
```

### 1D Plot Kinds

```python
# KDE (default) - smooth kernel density estimate
samples.plot_1d(axes, kind='kde_1d')

# Histogram
samples.plot_1d(axes, kind='hist_1d')

# Fast KDE (requires fastkde package)
samples.plot_1d(axes, kind='fastkde_1d')
```

### Multiple Parameters

```python
# Plot specific parameters
fig, axes = make_1d_axes(['param1', 'param2', 'param3'])
samples.plot_1d(axes)

# Control layout columns
fig, axes = make_1d_axes(['x0', 'x1', 'x2', 'x3'], ncol=2)
samples.plot_1d(axes)
```

## 2D Corner Plots (Triangle Plots)

### Basic Corner Plot

```python
from anesthetic import read_chains, make_2d_axes

samples = read_chains('chains/run')

# Create axes first, then plot
fig, axes = make_2d_axes(['x0', 'x1', 'x2'])
samples.plot_2d(axes)
```

### 2D Plot Kinds

The `kind` or `kinds` parameter controls plot types. Use a dict with keys 'diagonal', 'lower', 'upper':

```python
# Default: KDE on diagonal and lower, scatter on upper
samples.plot_2d(axes, kind='default')
# Equivalent to: kind={'diagonal': 'kde_1d', 'lower': 'kde_2d', 'upper': 'scatter_2d'}

# KDE only (no upper triangle)
samples.plot_2d(axes, kind='kde')
# Equivalent to: kind={'diagonal': 'kde_1d', 'lower': 'kde_2d'}

# Histogram
samples.plot_2d(axes, kind='hist')
# Equivalent to: kind={'diagonal': 'hist_1d', 'lower': 'hist_2d'}

# Scatter
samples.plot_2d(axes, kind='scatter')
# Equivalent to: kind={'diagonal': 'hist_1d', 'lower': 'scatter_2d'}

# Fast KDE
samples.plot_2d(axes, kind='fastkde')
# Equivalent to: kind={'diagonal': 'fastkde_1d', 'lower': 'fastkde_2d'}

# Custom combination
samples.plot_2d(axes, kind={
    'diagonal': 'kde_1d',
    'lower': 'kde_2d',
    'upper': 'scatter_2d'
})
```

### Available Plot Types

**Diagonal (1D):**
- `kde_1d`: Kernel density estimate with filled regions
- `hist_1d`: Histogram
- `fastkde_1d`: Fast KDE (requires fastkde)

**Lower/Upper (2D):**
- `kde_2d`: 2D KDE contours (filled)
- `hist_2d`: 2D histogram
- `scatter_2d`: Scatter plot
- `fastkde_2d`: Fast 2D KDE (requires fastkde)

### Controlling Triangle Regions

```python
# Lower triangle only (most common)
fig, axes = make_2d_axes(['x0', 'x1', 'x2'], upper=False)
samples.plot_2d(axes, kind='kde')

# No diagonal
fig, axes = make_2d_axes(['x0', 'x1', 'x2'], diagonal=False)

# Upper only
fig, axes = make_2d_axes(['x0', 'x1', 'x2'], lower=False, upper=True)

# Rectangle plot (different x and y parameters)
fig, axes = make_2d_axes([['x0', 'x1'], ['y0', 'y1', 'y2']])
```

## Prior vs Posterior Comparison

For `NestedSamples`, compare posterior to prior:

```python
samples = read_chains('chains/run')  # NestedSamples

# Get prior distribution (reweighted to beta=0)
prior = samples.prior()

# Create plot with both
fig, axes = make_2d_axes(['x0', 'x1', 'x2'])
prior.plot_2d(axes, kind='kde', label='Prior', alpha=0.5)
samples.plot_2d(axes, kind='kde', label='Posterior')

# Add legend
axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)), loc='lower center')
```

## Multiple Chain Comparison

Overlay multiple chains on the same axes:

```python
samples1 = read_chains('chains/model1')
samples2 = read_chains('chains/model2')

fig, axes = make_2d_axes(['x0', 'x1', 'x2'])
samples1.plot_2d(axes, kind='kde', label='Model 1')
samples2.plot_2d(axes, kind='kde', label='Model 2')

# Add legend to lower-left corner plot
axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)), loc='lower center')
```

## Styling Multiple Distributions

```python
fig, axes = make_2d_axes(['x0', 'x1', 'x2'])

# Different colors
samples1.plot_2d(axes, kind='kde', label='Run 1',
                 facecolor='C0', edgecolor='C0')
samples2.plot_2d(axes, kind='kde', label='Run 2',
                 facecolor='C1', edgecolor='C1')

# Transparency for overlays
samples1.plot_2d(axes, alpha=0.7)
samples2.plot_2d(axes, alpha=0.7)
```

## Using Separate kwargs for Different Plot Regions

```python
samples.plot_2d(axes,
    kind={'diagonal': 'kde_1d', 'lower': 'kde_2d'},
    diagonal_kwargs={'color': 'blue'},
    lower_kwargs={'levels': [0.68, 0.95], 'alpha': 0.8}
)
```

## Saving Figures

```python
import matplotlib.pyplot as plt
from anesthetic import make_2d_axes

fig, axes = make_2d_axes(['x0', 'x1', 'x2'])
samples.plot_2d(axes)
fig.savefig('corner_plot.png', dpi=150, bbox_inches='tight')
fig.savefig('corner_plot.pdf', bbox_inches='tight')
plt.close(fig)
```
