# Loading Chains

## Auto-Detection with read_chains

The simplest way to load chains:

```python
from anesthetic import read_chains

samples = read_chains('path/to/chains_root')
```

`read_chains` auto-detects the format by trying readers in order: PolyChord, MultiNest, Cobaya, UltraNest, NestedFit, GetDist, CSV.

## Supported Formats

### PolyChord

Files required: `<root>_dead-birth.txt` and optionally `<root>_phys_live-birth.txt`

```python
from anesthetic.read.polychord import read_polychord

samples = read_polychord('chains/my_run')  # Returns NestedSamples
```

### MultiNest

New format: `<root>dead-birth.txt`, `<root>phys_live-birth.txt`
Old format: `<root>ev.dat`, `<root>phys_live.points`

```python
from anesthetic.read.multinest import read_multinest

samples = read_multinest('chains/mn')  # Returns NestedSamples
```

### UltraNest

Reads from UltraNest output directory containing `results/points.hdf5`.

```python
from anesthetic.read.ultranest import read_ultranest

samples = read_ultranest('ultranest_output/')  # Returns NestedSamples
```

### Cobaya (MCMC)

Files required: `<root>.*.txt` chain files and `<root>.updated.yaml`

For optimal label reading, install GetDist: `pip install getdist`

```python
from anesthetic.read.cobaya import read_cobaya

samples = read_cobaya('cobaya_chains/run')  # Returns MCMCSamples
```

### GetDist Format

Files required: `<root>_1.txt`, `<root>_2.txt`, etc. and `<root>.paramnames`

```python
from anesthetic.read.getdist import read_getdist

samples = read_getdist('getdist_chains/run')  # Returns MCMCSamples
```

### CSV (Anesthetic Export)

Load chains previously saved with `samples.to_csv()`:

```python
from anesthetic import read_csv

samples = read_csv('samples.csv')
```

### HDF5

Load chains previously saved with `samples.to_hdf()`:

```python
from anesthetic import read_hdf

samples = read_hdf('samples.h5', key='samples')
```

## Creating Samples from Arrays

When you have samples as numpy arrays:

### NestedSamples (from nested sampling)

```python
import numpy as np
from anesthetic import NestedSamples

# data: (nsamples, nparams) array of parameter values
# logL: (nsamples,) array of log-likelihoods
# logL_birth: (nsamples,) array of birth log-likelihoods

samples = NestedSamples(
    data=data,
    columns=['param1', 'param2', 'param3'],  # Parameter names
    logL=logL,
    logL_birth=logL_birth,
    labels={'param1': r'$\alpha$', 'param2': r'$\beta$'},  # Optional LaTeX labels
    label='My Run'  # Legend label
)
```

If you only have number of live points (not birth likelihoods):

```python
samples = NestedSamples(
    data=data,
    columns=['x0', 'x1', 'x2'],
    logL=logL,
    logL_birth=500  # Integer = number of live points
)
```

### MCMCSamples (from MCMC)

```python
from anesthetic import MCMCSamples

samples = MCMCSamples(
    data=data,
    columns=['param1', 'param2'],
    weights=weights,  # Optional sample weights
    logL=logL,  # Optional log-likelihoods
    labels={'param1': r'$\alpha$'},
    label='MCMC Run'
)
```

### Samples (generic weighted samples)

```python
from anesthetic import Samples

samples = Samples(
    data=data,
    columns=['x', 'y', 'z'],
    weights=weights,  # Optional
    labels={'x': r'$x$', 'y': r'$y$', 'z': r'$z$'}
)
```

## Parameter Names Files

Anesthetic reads `.paramnames` files in GetDist format:

```
# Example .paramnames file
x0    x_0
x1    x_1
x2    x_2
derived1    \chi^2
```

Format: `parameter_name    latex_label` (tab or space separated)

## Saving Chains

```python
# Save to CSV (can be read back with read_csv)
samples.to_csv('output.csv')

# Save to HDF5 (can be read back with read_hdf)
samples.to_hdf('output.h5', key='posterior')
```
