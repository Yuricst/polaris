# polaris
 polaris --- PythOnLibrary for AstRodynamIcS

<p align="center">
  <img src="./etc/polaris_logo.png" width="550" title="hover text">
</p>
polaris is a Python library for preliminary spacecraft trajectory design. 

PyPI: https://pypi.org/project/astro-polaris/




### Installation
Install via pip

```bash
pip install astro-polaris
```

or clone this repository with

```bash
$ git clone https://github.com/Yuricst/polaris.git
```



### Dependencies

Core dependencies are

- `numba`, `numpy`, `pandas`, `scipy`

Although not necessary to run polaris, the following packages are also used within the example scripts and Jupyter notebooks:
- `tqdm`, `plotly`

  

### Imports

Subpackages within `polaris`  are imported as

```python
import polaris.SolarSystemConstants as ssc
import polaris.Propagator as prop
import polaris.R3BP as r3bp
import polaris.Keplerian as kepl
import polaris.Coordinates as coord
```

For examples, go to ```./examples/``` to see Jupyter notebook tutorials. 
The full documentation is available at https://github.com/Yuricst/polaris/wiki. 



### Quick Example

Here is a quick example of constructing a halo orbit in the Earth-Moon system. We first import the module

```python
import numpy as np
import matplotlib.pyplot as plt

import polaris.Propagator as prop
import polaris.R3BP as r3bp
```

Define the CR3BP system parameters via

```python
param_earth_moon = r3bp.get_cr3bp_param('399','301')   # NAIF ID's '399': Earth, '301': Moon
param_earth_moon.mu
```

We construct an initial guess of a colinear halo orbit at Earth-Moon L2 with z-direction amplitude of 4000 km via the Lindstedt*–*Poincaré method

```python
haloinit = r3bp.get_halo_approx(mu=param_earth_moon.mu, lp=2, lstar=param_earth_moon.lstar, az_km=4000, family=1, phase=0.0)
```

We then apply differential correction on the initial guess

```python
p_conv, state_conv, flag_conv = r3bp.ssdc_periodic_xzplane(param_earth_moon.mu, haloinit["state_guess"],haloinit["period_guess"], fix="z", message=False)
```

We finally propagate the result

```python
prop0 = prop.propagate_cr3bp(param_earth_moon.mu, state_conv, p_conv)
```

and plot the result

```python
plt.rcParams["font.size"] = 20
fig, axs = plt.subplots(1, 3, figsize=(18, 8))
axs[0].plot(prop0["xs"], prop0["ys"])
axs[0].set(xlabel='x, canonical', ylabel='y, canonical')
axs[1].plot(prop0["xs"], prop0["zs"])
axs[1].set(xlabel='x, canonical', ylabel='z, canonical')
axs[2].plot(prop0["ys"], prop0["zs"])
axs[2].set(xlabel='y, canonical', ylabel='z, canonical')
for idx in range(3):
    axs[idx].grid(True)
    axs[idx].axis("equal")
plt.suptitle('L2 halo')
plt.tight_layout(rect=[0, 0.01, 1, 1.03])
plt.show()
```

