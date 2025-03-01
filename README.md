# Socially Structured Epidemiological Models

## Project Overview

This project implements a framework for simulating and analyzing epidemiological models with socially structured populations. It explores how population heterogeneity, behavioral patterns, and public health interventions affect disease transmission dynamics.

### Key Features

- Multiple epidemiological model variants (SIR with masks, testing, vaccination)
- Population structuring with controllable polarization and homophily
- Efficient implementation using JAX for accelerated computation
- Comprehensive parameter sweeps and visualization tools

## Models

The framework implements several variations of compartmental epidemiological models:

### 1. Mask-SIR Model

A modified SIR model where individuals have different mask-wearing behaviors affecting their susceptibility to infection.

- **maskSIR**: Individuals interact with the entire population
- **maskSIRD**: Individuals only interact within their own compartment

### 2. SIRT Model

SIR model with testing, where testing rates vary across population groups.

### 3. SIRV Model

SIR model with vaccination, where vaccination rates vary across population groups.

## Population Structure

Population heterogeneity is modeled using several key parameters:

- **Polarization**: Controls how uniform or divided the population is in adopting behaviors
- **Homophily**: Controls the tendency of similar individuals to interact more frequently
- **Mean Behavior**: The average adoption rate of protective behaviors

## Installation

```bash
# Clone the repository
git clone https://github.com/Fabio-FS/Polarization_and_Homophily_SIR.git
cd Polarization_and_Homophily_SIR

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage

Basic example of running a simulation:

```python
from src.models.mask_SIR import sweep_pol_mask_maskSIR
from src.utils.distributions import homogeneous_distribution
import numpy as np
import matplotlib.pyplot as plt

# Define parameter ranges
mask_max_range = {"m": 0, "M": 1, "n": 50}  # Range for maximum mask-wearing
pol_range = {"m": 0, "M": 1, "n": 50}       # Range for polarization values

# Run parameter sweep
results = sweep_pol_mask_maskSIR(
    mask_max_range=mask_max_range,
    pol_range=pol_range,
    h=0,                      # Homophily parameter
    dT=0.1,                   # Time step
    T=1000,                   # Total simulation time
    beta_M=0.2,               # Maximum susceptibility
    batch_size=1000,
    N_COMPARTMENTS=100,
    SPB_exponent=1
)

# Process and visualize results
(S_final, I_final, R_final), R0, OH = results
# ... (visualization code)
```

## Key Findings

- If there is no homophily (h=0), polarization has a paradoxical effect:
  - Positive effect for measures reducing individual susceptibility (masks)
  - Negative effect for cumulative measures (vaccination, testing)
- With homophily (h≠0), heterophily (h<0) is generally beneficial for disease containment
- In homophilous societies (h>0), having a polarized population can paradoxically improve outcomes

## Project Structure

```
src/
├── models/                  # Model implementations
│   ├── mask_SIR/            # SIR model with mask-wearing
│   ├── mask_SIR_D/          # Decoupled SIR model with mask-wearing
│   ├── SIRT/                # SIR model with testing
│   └── SIRV/                # SIR model with vaccination
├── utils/                   # Utility functions
│   ├── batch_sweep.py       # Parameter sweeping functionality
│   ├── Contact_Matrix.py    # Social contact matrix creation
│   ├── distributions.py     # Population distribution functions
│   └── R0.py                # Reproduction number calculations
notebooks/                  # Analysis notebooks
├── Fig_0_test.ipynb
├── Fig_1.ipynb
├── Fig_2.ipynb
├── Fig_3.ipynb
└── ...
figures/                    # Generated figures
└── ...
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@article{sartori2025???,
  ???
}
```