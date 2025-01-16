# Epidemic Modeling with Social Structure

This project implements various epidemiological models (SIRM, SIRT, SIRV) that incorporate social structure and population heterogeneity. The models are implemented using JAX for efficient computation and parameter sweeps.

## Models

- **SIRM**: SIR model with heterogeneous susceptibility (mask wearing)
- **SIRT**: SIR model with heterogeneous testing
- **SIRV**: SIR model with heterogeneous vaccination

## Project Structure

```
src/
├── models/
│   ├── SIRM/
│   ├── SIRT/
│   └── SIRV/
└── utils/
    ├── Contact_Matrix.py
    ├── R0.py
    └── batch_sweep.py
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Fabio-FS/Polarization_and_Homophily_SIR.git

# Install dependencies
pip install -r requirements.txt
```

## Usage

Example usage of the SIRM model:

```python
from src.models.SIRM.dynamic import sim_SIRM_trajectory
from src.models.SIRM.sweep import sweep_pol_mean_SIRM

# Run a single simulation
params = {
    'recovery_rate': 0.1,
    'dT': 1.0,
    'homophilic_tendency': 0,
    'susceptibility_rates': (0.0, 0.6),
    'beta_params': (2.0, 2.0)
}
trajectory = sim_SIRM_trajectory(beta_params=(2.0, 2.0), params=params, n_steps=1000)
```

## Dependencies

- JAX
- NumPy
- Matplotlib (for notebooks)
- Jupyter (for running notebooks)
