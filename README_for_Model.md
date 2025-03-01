# Model Documentation

This document provides detailed explanations of the mathematical models and their implementations in this project.

## Population Structure

All models in this project use a similar approach to model population heterogeneity:

### Beta Distribution for Population Structure

The population is divided into N compartments, where each compartment represents individuals with similar behavioral characteristics. The distribution of the population across these compartments follows a beta distribution with parameters α and β, which control the shape of the distribution.

```python
def my_beta_asymmetric(a: float, b: float, n_groups: int, norm: float = 1.0) -> jnp.ndarray:
    """Generate population sizes based on asymmetric beta distribution"""
    x = jnp.linspace(1/n_groups/2, 1-1/n_groups/2, n_groups)
    
    from jax.scipy.stats import beta as jbeta
    y = jbeta.pdf(x, a, b)
    y = y / jnp.sum(y) * norm
    return y
```

### Polarization and Mean Parameters

Polarization (p) controls how concentrated or dispersed the population is in terms of adopting protective behaviors:

- Low polarization (p → 0): Most individuals have similar behavior
- High polarization (p → 1): Population is divided between extremes

For symmetric distributions, the polarization parameter is converted to the alpha parameter of a symmetric beta distribution:

```python
def pol_to_alpha(pol: jnp.ndarray) -> jnp.ndarray:
    """Convert polarization values to alpha values"""
    return 1/(pol*2) - 0.5
```

For asymmetric distributions, mean (m) and polarization (p) are converted to alpha and beta parameters using:

```python
def pol_mean_to_ab(pol: jnp.ndarray, m: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert polarization and mean to alpha, beta parameters of beta distribution
    
    Args:
        pol: polarization values
        m: Mean values
    
    Returns:
        Tuple of (alpha, beta) parameters
    """
    a = -m*(4*(m-1)*m+pol)/pol
    b = 4*m*(m-1)**2/pol + m - 1
    return a, b
```

The equations for calculating alpha and beta from polarization (p) and mean (m) are:

α = -m(4m(m-1) + p)/p

β = 4m(m-1)²/p + m - 1

These parameters define a beta distribution that has the desired mean value and polarization (which is related to the variance of the distribution). This allows for modeling populations with different average behaviors and different levels of behavioral heterogeneity.


### Contact Matrix

A contact matrix C is used to model the interaction between different population groups. The homophily parameter (h) controls the tendency of similar individuals to interact with each other:

- h > 0: Homophily (similar individuals interact more)
- h = 0: Random mixing
- h < 0: Heterophily (dissimilar individuals interact more)

```python
def create_contact_matrix(n_groups: int, homophilic_tendency: float, populations: jnp.ndarray) -> jnp.ndarray:
    """Create contact matrix where Cᵢⱼ is the probability of j contacting i."""
    positions = jnp.linspace(0, 1, n_groups)
    diffs = jnp.abs(positions[:, None] - positions[None, :])
    weights = jnp.exp(-homophilic_tendency * diffs)
    
    C = weights / jnp.sum(weights, axis=0)[None, :]
    C2, _, _ = matrix_scaling(C)
    C2 = C2 * n_groups * n_groups
    return C2
```

## 1. Mask-SIR Model

### Model Description

The Mask-SIR model extends the standard SIR model by incorporating varying mask-wearing behaviors across the population, which affects individual susceptibility to infection.

### Compartments

- **S**: Susceptible individuals
- **I**: Infected individuals
- **R**: Recovered individuals

### Parameters

- **β_M**: Maximum susceptibility rate
- **γ**: Recovery rate
- **μ_max**: Maximum mask-wearing effectiveness
- **SPB_exponent**: Non-linear scaling of behavior adoption

### Key Equations

The dynamics of the system are governed by:

```
dS_i/dt = -β_i * S_i * ∑_j C_ij * I_j
dI_i/dt = β_i * S_i * ∑_j C_ij * I_j - γ * I_i
dR_i/dt = γ * I_i
```

where:
- β_i = β_M * (1 - mask_wearing_i)
- mask_wearing_i = μ_max * (i/N)^SPB_exponent

### Implementation

The mask-wearing values are generated using:

```python
def generate_mask_wearing(mu_max: float, N_COMPARTMENTS: int, SPB_exponent: float = 1.0) -> jnp.ndarray:
    """Generate mask wearing values with non-linear interpolation (default linear)"""
    x = jnp.linspace(0, 1, num=N_COMPARTMENTS)
    mask = mu_max * jnp.power(x, SPB_exponent)
    return mask
```

Susceptibilities are calculated from mask-wearing values:

```python
def calculate_susceptibilities(beta_M: float, mask_wearing: jnp.ndarray) -> jnp.ndarray:
    """Calculate effective susceptibilities based on mask wearing behavior"""
    return beta_M * (1 - mask_wearing)
```

### Variants

#### maskSIR

In this variant, individuals interact with the entire population based on the contact matrix.

```python
def maskSIR_step(state: StateType, susceptibilities: jnp.ndarray, gamma: float, 
                contact_matrix: jnp.ndarray, use_contact_matrix: bool, dT: float = 1.0) -> StateType:
    # ... implementation ...
```

#### maskSIRD (Decoupled)

In this variant, each compartment only interacts with itself, representing isolated subpopulations.

```python
def maskSIRD_step(state: StateType, susceptibilities: jnp.ndarray, gamma: float, 
                 dT: float = 1.0) -> StateType:
    S, I, R = state
    N = S + I + R  # Population of each compartment
    
    new_infections = susceptibilities * S * (I / N) * dT
    new_recoveries = gamma * I * dT

    S_new = S - new_infections
    I_new = I + new_infections - new_recoveries
    R_new = R + new_recoveries

    return S_new, I_new, R_new
```

## 2. SIRT Model (SIR with Testing)

### Model Description

The SIRT model extends the SIR model by incorporating variable testing rates across the population, which accelerates the removal of infected individuals from circulation.

### Compartments

- **S**: Susceptible individuals
- **I**: Infected individuals
- **R**: Recovered individuals

### Parameters

- **β**: Susceptibility rate
- **γ**: Base recovery rate
- **τ_i**: Testing rate for compartment i

### Key Equations

```
dS_i/dt = -β * S_i * ∑_j C_ij * I_j
dI_i/dt = β * S_i * ∑_j C_ij * I_j - (γ + τ_i) * I_i
dR_i/dt = (γ + τ_i) * I_i
```

### Implementation

Testing rates are generated with higher testing rates for high-index compartments:

```python
def generate_testing_rates(test_range: Tuple[float, float], N_COMPARTMENTS: int, exponent: float = 1.0) -> jnp.ndarray:
    """Generate testing rate values with non-linear interpolation"""
    test_low, test_high = test_range
    return test_high + jnp.power(homogeneous_distribution(N_COMPARTMENTS,0,1), exponent) * (test_low - test_high)
```

The effective recovery rate is modified by testing:

```python
gammaS = params['recovery_rate'] + testing_rates
```

## 3. SIRV Model (SIR with Vaccination)

### Model Description

The SIRV model extends the SIR model by incorporating a vaccination process that moves individuals directly from the susceptible to the vaccinated compartment.

### Compartments

- **S**: Susceptible individuals
- **I**: Infected individuals
- **R**: Recovered individuals
- **V**: Vaccinated individuals

### Parameters

- **β**: Susceptibility rate
- **γ**: Recovery rate
- **ν_i**: Vaccination rate for compartment i

### Key Equations

```
dS_i/dt = -β * S_i * ∑_j C_ij * I_j - ν_i * S_i
dI_i/dt = β * S_i * ∑_j C_ij * I_j - γ * I_i
dR_i/dt = γ * I_i
dV_i/dt = ν_i * S_i
```

### Implementation

Vaccination rates are generated in a similar fashion to testing rates:

```python
def generate_vaccination_rates(vacc_range: Tuple[float, float], N_COMPARTMENTS: int, exponent: float = 1.0) -> jnp.ndarray:
    """Generate vaccination values with non-linear interpolation"""
    vacc_low, vacc_high = vacc_range
    return vacc_high + jnp.power(homogeneous_distribution(N_COMPARTMENTS,0,1), exponent) * (vacc_low - vacc_high)
```

The SIRV step function incorporates the vaccination process:

```python
def SIRV_step(state: StateType, vaccination_rates: jnp.ndarray, susceptibility: float, 
             gamma: float, contact_matrix: jnp.ndarray, use_contact_matrix: bool, dT: float = 1.0) -> StateType:
    S, I, R, V = state
    
    # Vaccination process
    new_vaccinations = vaccination_rates * S * dT
    
    # Infection process
    infection_force = ...
    new_infections = susceptibility * S * infection_force * dT
    new_recoveries = gamma * I * dT
    
    S_new = S - new_infections - new_vaccinations
    I_new = I + new_infections - new_recoveries
    R_new = R + new_recoveries
    V_new = V + new_vaccinations
    
    return S_new, I_new, R_new, V_new
```

## R₀ Calculation

The basic reproduction number R₀ is calculated for each model using the next-generation matrix approach. This involves:

1. Creating a next-generation matrix (NGM) based on model parameters
2. Finding the largest eigenvalue of this matrix

```python
def create_ngm_maskSIR(gamma: float, beta_M: float, mask_wearing: jnp.ndarray, 
                      populations: jnp.ndarray, C: jnp.ndarray) -> jnp.ndarray:
    """Create next generation matrix for maskSIR model"""
    pop_fractions = populations / jnp.sum(populations)
    susceptibilities = beta_M * (1 - mask_wearing)
    return (1/gamma) * jnp.diag(susceptibilities * pop_fractions) @ C

def R0_maskSIR(beta_M: float, mask_wearing: jnp.ndarray, gamma: float, 
               C: jnp.ndarray, populations: jnp.ndarray) -> float:
    """Calculate R0 for maskSIR model"""
    ngm = create_ngm_maskSIR(gamma, beta_M, mask_wearing, populations, C)
    return power_iteration(ngm)
```

## Parameter Sweeps

Parameter sweeps are implemented using JAX's vectorized map (vmap) functionality for efficient parallel computation:

```python
def batch_sweep(simulation_fn: Callable, param_ranges: jnp.ndarray, base_params: Dict[str, Any],
               n_steps: int, batch_size: int, param_updater: Callable, ...) -> Tuple[StateType, jnp.ndarray, jnp.ndarray]:
    """Run parameter sweeps in parallel batches"""
    # ...
    vectorized_sim = jax.vmap(run_single_sim, in_axes=0)
    # ...
```

This allows for efficient exploration of the parameter space to analyze how different combinations of parameters affect disease dynamics.