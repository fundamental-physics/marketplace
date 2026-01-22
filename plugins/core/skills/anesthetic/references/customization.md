# Customization Guide

## Parameter Labels

### Setting Labels on Samples

```python
# Set individual label
samples.set_label('x0', r'$\Omega_m$')
samples.set_label('x1', r'$\sigma_8$')

# Get a label
label = samples.get_label('x0')  # Returns '$\Omega_m$'

# Get all labels as dict
labels_map = samples.get_labels_map()
```

### Labels During Axis Creation

```python
from anesthetic import make_2d_axes

fig, axes = make_2d_axes(
    ['x0', 'x1', 'x2'],
    labels={'x0': r'$\Omega_m$', 'x1': r'$\sigma_8$', 'x2': r'$h$'}
)
```

### Labels When Loading

```python
from anesthetic import NestedSamples

samples = NestedSamples(
    data=data,
    columns=['x0', 'x1'],
    logL=logL,
    logL_birth=logL_birth,
    labels={'x0': r'$\alpha$', 'x1': r'$\beta$'}
)
```

## Log-Scale Axes

```python
from anesthetic import make_1d_axes, make_2d_axes

# 1D plots with log-scale
fig, axes = make_1d_axes(['mass', 'radius'], logx=['mass'])
samples.plot_1d(axes)

# 2D plots with log-scale
fig, axes = make_2d_axes(['mass', 'luminosity', 'temp'],
                         logx=['mass', 'luminosity'],
                         logy=['mass', 'luminosity'])
samples.plot_2d(axes)
```

## Colors and Styling

### Face and Edge Colors

```python
# Solid fill color
samples.plot_2d(axes, facecolor='C0')

# Edge color for contours
samples.plot_2d(axes, edgecolor='black')

# Combined
samples.plot_2d(axes, facecolor='skyblue', edgecolor='navy')
```

### Transparency

```python
# Global transparency
samples.plot_2d(axes, alpha=0.7)

# Per-region transparency
samples.plot_2d(axes,
    diagonal_kwargs={'alpha': 1.0},
    lower_kwargs={'alpha': 0.6}
)
```

### Colormaps

```python
# For scatter plots
samples.plot_2d(axes, kind='scatter', cmap='viridis')

# For 2D histograms
samples.plot_2d(axes, kind='hist', cmap='Blues')
```

## Contour Levels

Default contour levels are 68% and 95% credible regions.

```python
# Custom probability levels
samples.plot_2d(axes, levels=[0.68, 0.95, 0.99])

# Single contour
samples.plot_2d(axes, levels=[0.95])
```

## Reference Values (Truth Markers)

### Scatter Points

Mark specific parameter values (e.g., true values, best-fit):

```python
# Add markers at specific parameter values
axes.scatter({'x0': 0.3, 'x1': 0.8}, marker='*', s=100, c='red', zorder=10)

# Multiple points
axes.scatter({'x0': 0.3, 'x1': 0.8}, marker='o', c='blue')
axes.scatter({'x0': 0.35, 'x1': 0.75}, marker='s', c='green')
```

### Vertical/Horizontal Lines

```python
# Add reference lines at specific values
axes.axlines({'x0': 0.3, 'x1': 0.8}, color='red', linestyle='--', alpha=0.7)
```

### Shaded Regions

```python
# Shade a parameter range
axes.axspans({'x0': (0.25, 0.35)}, color='gray', alpha=0.2)

# Multiple spans
axes.axspans({
    'x0': (0.25, 0.35),
    'x1': (0.7, 0.9)
}, color='yellow', alpha=0.3)
```

## Legend Placement

The legend is typically placed in the corner plot's unused space:

```python
fig, axes = make_2d_axes(['x0', 'x1', 'x2'])
samples1.plot_2d(axes, label='Model A')
samples2.plot_2d(axes, label='Model B')

# Place legend in upper triangle area
axes.iloc[-1, 0].legend(
    bbox_to_anchor=(len(axes)/2, len(axes)),
    loc='lower center',
    ncol=2
)
```

Alternative placements:

```python
# Outside the plot
axes.iloc[-1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# In a specific subplot
axes.iloc[1, 0].legend(loc='upper right')
```

## Axis Limits

```python
import matplotlib.pyplot as plt
from anesthetic import make_2d_axes

fig, axes = make_2d_axes(['x0', 'x1'])
samples.plot_2d(axes)

# Set limits on specific axes
axes.loc['x1', 'x0'].set_xlim(0, 1)
axes.loc['x1', 'x0'].set_ylim(0, 2)

# Access diagonal twin axis
axes.loc['x0', 'x0'].twin.set_xlim(0, 1)
```

## Figure Size

```python
from anesthetic import make_2d_axes
import matplotlib.pyplot as plt

# Control figure size
fig, axes = make_2d_axes(['x0', 'x1', 'x2'], figsize=(8, 8))

# Or adjust after creation
fig.set_size_inches(10, 10)
```

## Tick Parameters

```python
fig, axes = make_2d_axes(['x0', 'x1', 'x2'])
samples.plot_2d(axes)

# Modify ticks on all axes
axes.tick_params(labelsize=8, rotation=45)
```

## Complete Styling Example

```python
from anesthetic import read_chains, make_2d_axes
import matplotlib.pyplot as plt

# Load data
posterior = read_chains('chains/run')
prior = posterior.prior()

# Set labels
posterior.set_label('x0', r'$\Omega_m$')
posterior.set_label('x1', r'$\sigma_8$')
posterior.set_label('x2', r'$h$')

# Create figure
params = ['x0', 'x1', 'x2']
fig, axes = make_2d_axes(params, figsize=(8, 8))

# Plot prior (light, background)
prior.plot_2d(axes, kind='kde', label='Prior',
              facecolor='gray', alpha=0.3)

# Plot posterior (main)
posterior.plot_2d(axes, kind='kde', label='Posterior',
                  facecolor='C0', alpha=0.8)

# Add truth values
truth = {'x0': 0.3, 'x1': 0.8, 'x2': 0.7}
axes.axlines(truth, color='red', linestyle='--', lw=1.5)
axes.scatter(truth, marker='*', s=100, c='red', zorder=10)

# Legend
axes.iloc[-1, 0].legend(
    bbox_to_anchor=(len(axes)/2, len(axes)),
    loc='lower center'
)

# Save
fig.savefig('publication_plot.pdf', bbox_inches='tight')
```
