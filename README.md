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

### Usage

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
