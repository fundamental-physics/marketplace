# API Reference

## Top-Level Imports

```python
from anesthetic import (
    # Classes
    Samples,
    MCMCSamples,
    NestedSamples,
    # Axis creation
    make_1d_axes,
    make_2d_axes,
    # Loading functions
    read_chains,
    read_csv,
    read_hdf,
)
```

## Loading Functions

### read_chains

```python
read_chains(root, *args, **kwargs) -> NestedSamples | MCMCSamples
```
Auto-detect chain format and load. Tries: PolyChord, MultiNest, Cobaya, UltraNest, NestedFit, GetDist, CSV.

**Parameters:**
- `root`: str or Path - Root name for chain files

**Returns:** `NestedSamples` or `MCMCSamples` depending on detected format

### read_csv

```python
read_csv(filename) -> Samples
```
Load samples from CSV file saved with `samples.to_csv()`.

### read_hdf

```python
read_hdf(filename, key) -> Samples
```
Load samples from HDF5 file saved with `samples.to_hdf()`.

## Axis Creation Functions

### make_1d_axes

```python
make_1d_axes(
    params,           # list[str] - Parameter names
    ncol=None,        # int - Number of columns in grid
    labels=None,      # dict - {param: latex_label}
    logx=None,        # list[str] - Parameters to plot on log scale
    **fig_kw          # Passed to plt.subplots()
) -> tuple[Figure, AxesSeries]
```

### make_2d_axes

```python
make_2d_axes(
    params,           # list[str] or [list[str], list[str]] - Parameters
    labels=None,      # dict - {param: latex_label}
    upper=True,       # bool - Include upper triangle
    lower=True,       # bool - Include lower triangle
    diagonal=True,    # bool - Include diagonal
    logx=None,        # list[str] - Log scale for x-axis
    logy=None,        # list[str] - Log scale for y-axis
    **fig_kw          # Passed to plt.subplots()
) -> tuple[Figure, AxesDataFrame]
```

## Sample Classes

### Samples (Base Class)

```python
Samples(
    data=None,        # array-like - Shape (nsamples, nparams)
    columns=None,     # list[str] - Parameter names
    weights=None,     # array-like - Sample weights
    logL=None,        # array-like - Log-likelihoods
    labels=None,      # dict - {param: latex_label}
    label=None,       # str - Legend label
    logzero=-1e30     # float - Threshold for log(0)
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `plot_1d(axes=None, kind='kde_1d', logx=None, label=None, **kwargs)` | Create 1D marginal plots |
| `plot_2d(axes=None, kind='default', logx=None, logy=None, label=None, **kwargs)` | Create 2D corner plot |
| `set_label(param, label)` | Set LaTeX label for parameter |
| `get_label(param)` | Get label for parameter |
| `get_labels_map()` | Get dict of all labels |
| `to_csv(filename)` | Save to CSV |
| `to_hdf(filename, key)` | Save to HDF5 |
| `importance_sample(logL_new, action='add', inplace=False)` | Re-weight samples |

### MCMCSamples

Inherits from `Samples`. Additional methods:

| Method | Description |
|--------|-------------|
| `remove_burn_in(burn_in, reset_index=False, inplace=False)` | Remove burn-in samples |
| `Gelman_Rubin(params=None, per_param=False)` | Compute R-1 convergence statistic |

**remove_burn_in `burn_in` values:**
- `0 < burn_in < 1`: Remove first fraction
- `burn_in > 1`: Remove first N samples
- `-1 < burn_in < 0`: Keep last fraction
- `burn_in < -1`: Keep last N samples
- `list`: Different burn-in per chain

### NestedSamples

Inherits from `Samples`. Additional parameters:

```python
NestedSamples(
    ...,
    logL_birth=None,  # array-like or int - Birth log-likelihoods or nlive
    beta=1.0          # float - Inverse temperature
)
```

**Properties:**
- `beta`: Thermodynamic inverse temperature

**Methods:**

| Method | Description |
|--------|-------------|
| `prior(inplace=False)` | Get prior distribution (beta=0) |
| `set_beta(beta, inplace=False)` | Change inverse temperature |
| `logZ(nsamples=None, beta=None)` | Log-evidence |
| `D_KL(nsamples=None, beta=None)` | KL divergence |
| `logL_P(nsamples=None, beta=None)` | Posterior avg log-likelihood |
| `d_G(nsamples=None, beta=None)` | Model dimensionality |
| `stats(nsamples=None, beta=None, norm=None)` | All statistics |
| `gui()` | Launch interactive GUI |

## Axes Classes

### AxesSeries

pandas Series of matplotlib Axes for 1D plots.

| Method | Description |
|--------|-------------|
| `set_xlabels()` | Set x-axis labels from index |
| `tick_params(**kwargs)` | Modify tick parameters |

### AxesDataFrame

pandas DataFrame of matplotlib Axes for 2D plots.

| Method | Description |
|--------|-------------|
| `set_labels()` | Set axis labels |
| `scatter(points, **kwargs)` | Add scatter markers at parameter values |
| `axlines(values, **kwargs)` | Add reference lines |
| `axspans(ranges, **kwargs)` | Add shaded regions |
| `tick_params(**kwargs)` | Modify tick parameters |

**scatter/axlines/axspans parameter format:**
```python
{'param1': value1, 'param2': value2}  # For scatter/axlines
{'param1': (low, high), 'param2': (low, high)}  # For axspans
```

## Plot Kind Options

### 1D Kinds (for plot_1d and plot_2d diagonal)

| Kind | Description |
|------|-------------|
| `'kde_1d'` | Kernel density estimate (default) |
| `'hist_1d'` | Histogram |
| `'fastkde_1d'` | Fast KDE (requires fastkde) |

### 2D Kinds (for plot_2d lower/upper)

| Kind | Description |
|------|-------------|
| `'kde_2d'` | 2D KDE contours |
| `'hist_2d'` | 2D histogram |
| `'scatter_2d'` | Scatter plot |
| `'fastkde_2d'` | Fast 2D KDE (requires fastkde) |

### plot_2d Shortcuts

| Shortcut | Expands To |
|----------|-----------|
| `'default'` | `{'diagonal': 'kde_1d', 'lower': 'kde_2d', 'upper': 'scatter_2d'}` |
| `'kde'` | `{'diagonal': 'kde_1d', 'lower': 'kde_2d'}` |
| `'hist'` | `{'diagonal': 'hist_1d', 'lower': 'hist_2d'}` |
| `'scatter'` | `{'diagonal': 'hist_1d', 'lower': 'scatter_2d'}` |
| `'fastkde'` | `{'diagonal': 'fastkde_1d', 'lower': 'fastkde_2d'}` |

## Common kwargs

### Plotting kwargs

| Kwarg | Description | Default |
|-------|-------------|---------|
| `label` | Legend label | `samples.label` |
| `alpha` | Transparency | 1.0 |
| `facecolor` | Fill color | Auto |
| `edgecolor` | Line/contour color | Auto |
| `levels` | Contour probability levels | `[0.68, 0.95]` |
| `cmap` | Colormap (for scatter/hist) | 'viridis' |

### Sub-plot kwargs

For `plot_2d`, pass kwargs to specific regions:

```python
samples.plot_2d(axes,
    diagonal_kwargs={'color': 'blue'},
    lower_kwargs={'levels': [0.68, 0.95]},
    upper_kwargs={'s': 1}  # marker size for scatter
)
```
