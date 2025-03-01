# Visualization Guide

This guide explains how to reproduce the key figures from the project and interpret their results.

## Setup

First, import the necessary packages and set up the plotting environment:

```python
%load_ext autoreload
%autoreload 2

from src.models.mask_SIR import sweep_pol_mask_maskSIR, sweep_pol_mean_maskSIR, sweep_hom_pol_maskSIR
from src.models.SIRT import sweep_pol_SPB_SIRT, sweep_pol_mean_SIRT, sweep_hom_pol_SIRT
from src.models.SIRV import sweep_pol_SPB_SIRV, sweep_pol_mean_SIRV, sweep_pol_hom_SIRV
from src.utils.distributions import pol_to_alpha, homogeneous_distribution
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directories for saving figures
path_Plot_with_labels = "../figures/with_labels/"
path_Plot_without_labels = "../figures/no_labels/"

if not os.path.exists(path_Plot_with_labels):
    os.makedirs(path_Plot_with_labels)

if not os.path.exists(path_Plot_without_labels):
    os.makedirs(path_Plot_without_labels)

# Define color palettes
CP1 = ['#fdbb84','#fc8d59','#ef6548','#d7301f','#990000']   # Colors for polarization
CP2 = ['#d9f0a3','#addd8e','#78c679','#31a354','#006837']   # Colors for mean behavior
CP3 = ['#d0d1e6','#a6bddb','#74a9cf','#2b8cbe','#045a8d']   # Colors for homophily

# Set up colormaps
import matplotlib as mpl
def discretize_cmaps(cmap_name, N):
    cmap = plt.cm.get_cmap(cmap_name, N)
    return cmap

my_hot_r = discretize_cmaps('hot_r', 12)
my_hot_r.set_bad('gray')

my_vir_r = discretize_cmaps('viridis_r', 12)
my_vir_r.set_bad('gray')

# Define figure dimensions
Lx, Ly = 4, 3  # Default figure size
```

## Figure 1: Polarization vs. Mask-Wearing (maskSIR Model)

This figure shows how the disease spread varies with different levels of polarization and mask-wearing effectiveness.

```python
# Run parameter sweep
NP = 100  # Number of polarization points
NS = 100  # Number of mask-wearing levels

mask_max_range = {"m": 0, "M": 1, "n": NS}  # Range for maximum mask-wearing
pol_range = {"m": 0, "M": 1, "n": NP}       # Range for polarization

results = sweep_pol_mask_maskSIR(
    mask_max_range=mask_max_range,
    pol_range=pol_range,
    h=0,                      # Homophily parameter
    dT=0.1,
    T=1000,
    beta_M=0.2,               # Maximum susceptibility
    batch_size=1000,
    N_COMPARTMENTS=100,
    SPB_exponent=1
)

# Process results
(S_final, I_final, R_final), R0, OH = results
S = np.sum(S_final, axis=1)
I = np.sum(I_final, axis=1)
R = np.sum(R_final, axis=1)

# Reshape arrays to match parameter grid
S = S.reshape(NP, NS).transpose()
I = I.reshape(NP, NS).transpose()
R = R.reshape(NP, NS).transpose()
R0 = R0.reshape(NP, NS).transpose()
OH = OH.reshape(NP, NS).transpose()

# Calculate total infected (R+I)
FIG_A_RI = R + I
FIG_A_R0 = R0

# Create plot
masks = homogeneous_distribution(NS, 0, 1)
pol = homogeneous_distribution(NP, 0, 1)

fig, ax = plt.subplots(figsize=(Lx, Ly))
cax = ax.imshow(FIG_A_RI, cmap=my_hot_r, aspect="auto", origin="lower", 
               extent=[0, 1, 0, 1], vmin=0, vmax=1)
cbar = fig.colorbar(cax, ax=ax)

# Add contour lines
CS = ax.contour(pol, masks, FIG_A_RI, levels=[0.25, 0.5, 0.75], 
                linewidths=1, colors="black", linestyles="dashed")

ax.set_xlabel("Polarization")
ax.set_ylabel("Maximum mask effectiveness")
ax.set_title("Total infected population")
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])

plt.tight_layout()
plt.savefig("figures/maskSIR_pol_vs_mask.png", dpi=300)
plt.show()
```

### Interpretation

- The heatmap shows the proportion of the population that becomes infected
- Darker colors represent lower infection levels (better outcomes)
- Key observation: With homophily h=0, higher polarization (x-axis) can paradoxically lead to better outcomes (lower infection rates) for mask-wearing interventions
- This happens because the benefit gained from the high-compliance portion of the population outweighs the cost from the low-compliance portion

## Figure 2: Polarization vs. Mean Behavior (maskSIR Model)

This figure explores how the disease spread varies with different combinations of polarization and mean behavior level.

```python
# Run parameter sweep
NP = 100  # Number of polarization points
NM = 100  # Number of mean behavior points

pol_range = {"m": 0, "M": 1, "n": NP}
mean_range = {"m": 0.0, "M": 1.0, "n": NM}

results = sweep_pol_mean_maskSIR(
    mu_max=1,                # Fixed maximum mask effectiveness
    pol_range=pol_range,
    mean_range=mean_range,
    dT=0.1,
    T=1000,
    beta_M=0.35,
    batch_size=1000,
    N_COMPARTMENTS=100,
    SPB_exponent=1
)

# Process results
(S_final, I_final, R_final), R0, OH = results
S = np.sum(S_final, axis=1)
I = np.sum(I_final, axis=1)
R = np.sum(R_final, axis=1)

# Reshape arrays
S = np.flipud(S.reshape(NP, NM))
I = np.flipud(I.reshape(NP, NM))
R = np.flipud(R.reshape(NP, NM))
R0 = np.flipud(R0.reshape(NP, NM))
OH = np.flipud(OH.reshape(NP, NM))

FIG_B_RI = np.flipud(R + I)

# Create plot
mean_vals = homogeneous_distribution(NM, 0, 1)
pol = homogeneous_distribution(NP, 0, 1)

fig, ax = plt.subplots(figsize=(Lx, Ly))
cax = ax.imshow(FIG_B_RI, cmap=my_hot_r, aspect="auto", origin="lower", 
               extent=[0, 1, 0, 1], vmin=0, vmax=1, interpolation="none")
cbar = fig.colorbar(cax, ax=ax)

# Add contour lines
CS = ax.contour(pol, mean_vals, FIG_B_RI, levels=[0.25, 0.5, 0.75], 
                linewidths=0.5, colors="black", linestyles="dashed")

ax.set_xlabel("Polarization")
ax.set_ylabel("Mean behavior")
ax.set_title("Total infected population")
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])

plt.tight_layout()
plt.savefig("figures/maskSIR_pol_vs_mean.png", dpi=300)
plt.show()
```

### Interpretation

- This heatmap shows how the infection levels vary with polarization (x-axis) and mean behavior level (y-axis)
- For a fixed mean behavior level (horizontal slice), increasing polarization tends to decrease the total infected population
- For fixed polarization (vertical slice), increasing mean behavior (higher average mask usage) decreases infection levels
- The non-linear relationship suggests that the distribution of behaviors can be as important as the average level

## Figure 3: Polarization vs. Homophily (maskSIR Model)

This figure examines how social structure (homophily) interacts with polarization to affect disease spread.

```python
# Run parameter sweep
NP = 100  # number of polarization points
NH = 100  # number of homophily points

h_range = {"m": -10.0, "M": 10.0, "n": NH}
pol_range = {"m": 0, "M": 1, "n": NP}

results = sweep_hom_pol_maskSIR(
    h_range=h_range,
    mu_max=1,                # Fixed maximum mask effectiveness
    pol_range=pol_range,
    dT=0.1,
    T=1000,
    beta_M=0.35,
    batch_size=1000,
    N_COMPARTMENTS=100,
    SPB_exponent=1
)

# Process results
(S_final, I_final, R_final), R0, OH = results
S = np.sum(S_final, axis=1)
I = np.sum(I_final, axis=1)
R = np.sum(R_final, axis=1)

# Reshape arrays
S = S.reshape(NP, NH)
I = I.reshape(NP, NH)
R = R.reshape(NP, NH)
R0 = R0.reshape(NP, NH)
OH = OH.reshape(NP, NH)

FIG_C_RI = R + I

# Create plot
h_vals = homogeneous_distribution(NH, h_range["m"], h_range["M"])
pol = homogeneous_distribution(NP, pol_range["m"], pol_range["M"])

fig, ax = plt.subplots(figsize=(Lx, Ly))
cax = ax.imshow(FIG_C_RI, cmap=my_hot_r, aspect="auto", origin="lower", 
               extent=[-10, 10, 0, 1], vmin=0, vmax=1, interpolation="none")
cbar = fig.colorbar(cax, ax=ax)

# Add contour lines
CS = ax.contour(h_vals, pol, FIG_C_RI, levels=[0.25, 0.5, 0.75], 
                linewidths=1, colors="black", linestyles="dashed")

ax.set_xlabel("Homophily (h)")
ax.set_ylabel("Polarization")
ax.set_title("Total infected population")
ax.set_xticks([-10, 0, 10])
ax.set_yticks([0, 0.5, 1])

plt.tight_layout()
plt.savefig("figures/maskSIR_pol_vs_homophily.png", dpi=300)
plt.show()
```

### Interpretation

- This figure shows how infection levels vary with homophily (x-axis) and polarization (y-axis)
- Negative homophily (heterophily, h<0) generally leads to better outcomes
- With positive homophily (h>0), which is common in real societies, higher polarization can improve outcomes
- This complex relationship demonstrates that social structure strongly influences the effect of behavioral heterogeneity

## Figure 4: Testing Effectiveness (SIRT Model)

This figure shows how testing interventions affect disease spread at different polarization levels.

```python
# Run parameter sweep
NP = 100
NS = 100

test_max_range = {"m": 0, "M": 0.3, "n": NS}
pol_range = {"m": 0, "M": 1, "n": NP}

results = sweep_pol_SPB_SIRT(
    test_max_range=test_max_range,
    pol_range=pol_range,
    h=0,
    dT=1,
    T=1000,
    batch_size=1000,
    susceptibility_rate=0.25,
    N_COMPARTMENTS=100
)

# Process results
(S_final, I_final, R_final), R0, OH = results
S = np.sum(S_final, axis=1)
I = np.sum(I_final, axis=1)
R = np.sum(R_final, axis=1)

# Reshape arrays
S = S.reshape(NP, NS).transpose()
I = I.reshape(NP, NS).transpose()
R = R.reshape(NP, NS).transpose()
R0 = R0.reshape(NP, NS).transpose()
OH = OH.reshape(NP, NS).transpose()

FIG_A_RI = R + I
FIG_A_R0 = R0

# Create plot
test_rates = homogeneous_distribution(NS, test_max_range["m"], test_max_range["M"])
pol = homogeneous_distribution(NP, pol_range["m"], pol_range["M"])

fig, ax = plt.subplots(figsize=(Lx, Ly))
cax = ax.imshow(FIG_A_RI, cmap=my_hot_r, aspect="auto", origin="lower", 
               extent=[0, 1, 0, 0.3], vmin=0, vmax=1)
cbar = fig.colorbar(cax, ax=ax)

# Add contour lines
CS = ax.contour(pol, test_rates, FIG_A_RI, levels=[0.25, 0.5, 0.75], 
                linewidths=1, colors="black", linestyles="dashed")

ax.set_xlabel("Polarization")
ax.set_ylabel("Maximum testing rate")
ax.set_title("Total infected population (SIRT model)")
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.15, 0.3])

plt.tight_layout()
plt.savefig("figures/SIRT_pol_vs_test.png", dpi=300)
plt.show()
```

### Interpretation

- The heatmap shows how testing effectiveness interacts with polarization
- In contrast to mask-wearing, higher polarization leads to worse outcomes for testing interventions
- This is because testing primarily works by removing infected individuals from circulation, which is a cumulative effect rather than an individual protection measure
- Note the inverse relationship compared to the mask-wearing case

## Figure 5: Vaccination Effectiveness (SIRV Model)

This figure explores how vaccination effectiveness varies with polarization.

```python
# Run parameter sweep
NP = 100
NS = 100

vacc_max_range = {"m": 0, "M": 0.1, "n": NS}
pol_range = {"m": 0, "M": 1, "n": NP}

results = sweep_pol_SPB_SIRV(
    vacc_max_range=vacc_max_range,
    pol_range=pol_range,
    h=0,
    dT=1,
    T=1000,
    batch_size=1000,
    susceptibility_rate=0.3,
    N_COMPARTMENTS=100
)

# Process results
(S_final, I_final, R_final, V_final), R0, OH = results
S = np.sum(S_final, axis=1)
I = np.sum(I_final, axis=1)
R = np.sum(R_final, axis=1)
V = np.sum(V_final, axis=1)

# Reshape arrays
S = S.reshape(NP, NS).transpose()
I = I.reshape(NP, NS).transpose()
R = R.reshape(NP, NS).transpose()
V = V.reshape(NP, NS).transpose()
R0 = R0.reshape(NP, NS).transpose()
OH = OH.reshape(NP, NS).transpose()

FIG_A_RI = R + I
FIG_A_V = V

# Create plot
vacc_rates = homogeneous_distribution(NS, vacc_max_range["m"], vacc_max_range["M"])
pol = homogeneous_distribution(NP, pol_range["m"], pol_range["M"])

fig, ax = plt.subplots(figsize=(Lx, Ly))
cax = ax.imshow(FIG_A_RI, cmap=my_hot_r, aspect="auto", origin="lower", 
               extent=[0, 1, 0, 0.1], vmin=0, vmax=1)
cbar = fig.colorbar(cax, ax=ax)

# Add contour lines
CS = ax.contour(pol, vacc_rates, FIG_A_RI, levels=[0.25, 0.5, 0.75], 
                linewidths=1, colors="black", linestyles="dashed")

ax.set_xlabel("Polarization")
ax.set_ylabel("Maximum vaccination rate")
ax.set_title("Total infected population (SIRV model)")
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.05, 0.1])

plt.tight_layout()
plt.savefig("figures/SIRV_pol_vs_vacc.png", dpi=300)
plt.show()
```

### Interpretation

- Similar to testing, vaccination shows the opposite relationship with polarization compared to mask-wearing
- Higher polarization leads to worse outcomes for vaccination campaigns
- This is because vaccination provides both individual protection and contributes to herd immunity, making uniform vaccination more effective
- The results highlight that different intervention types interact differently with population heterogeneity

## Figure 6: Line Plots for Fixed Values

Creating line plots for fixed values of one parameter can help visualize slices through the parameter space:

```python
# Example: Line plot of polarization effect at different homophily values
NP = 100  # number of polarization points
NH = 5    # number of homophily points to plot

h_range = {"m": -10.0, "M": 10.0, "n": NH}
pol_range = {"m": 0, "M": 1, "n": NP}

results = sweep_hom_pol_maskSIR(
    h_range=h_range,
    mu_max=1,
    pol_range=pol_range,
    dT=0.1,
    T=1000,
    beta_M=0.35,
    batch_size=1000,
    N_COMPARTMENTS=100,
    SPB_exponent=1
)

# Process results
(S_final, I_final, R_final), R0, OH = results
S = np.sum(S_final, axis=1)
I = np.sum(I_final, axis=1)
R = np.sum(R_final, axis=1)

# Reshape arrays
S = S.reshape(NP, NH)
I = I.reshape(NP, NH)
R = R.reshape(NP, NH)
R0 = R0.reshape(NP, NH)
OH = OH.reshape(NP, NH)

temp = R + I  # Total infected

# Create plot
homs = homogeneous_distribution(h_range["n"], h_range["m"], h_range["M"])
pols = homogeneous_distribution(pol_range["n"], pol_range["m"], pol_range["M"])

fig, ax = plt.subplots(figsize=(Lx, Ly))
for i in range(NH):
    ax.plot(pols, temp[:, i], color=CP3[i], label=f"h = {homs[i]:.1f}")

ax.legend()
ax.set_xlabel("Polarization")
ax.set_ylabel("Total infected proportion")
ax.set_title("Effect of polarization at different homophily values")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])

plt.tight_layout()
plt.savefig("figures/maskSIR_pol_lines.png", dpi=300)
plt.show()
```

### Interpretation

- Line plots help visualize how one parameter (polarization) affects outcomes at different fixed values of another parameter (homophily)
- This allows for more detailed comparisons than heatmaps alone
- In this example, we can see the complex interplay between polarization and homophily more clearly

## Figure 7: Relative Change Plots

Creating plots of the relative change compared to a baseline can highlight the effect size:

```python
# Example: Relative change in infected proportion compared to zero polarization
NP = 100  # Number of polarization points
NM = 5    # Number of mean behavior points to plot

pol_range = {"m": 0, "M": 1, "n": NP}
mean_range = {"m": 0.0, "M": 1.0, "n": NM}

results = sweep_pol_mean_maskSIR(
    mu_max=1,
    pol_range=pol_range,
    mean_range=mean_range,
    dT=0.1,
    T=1000,
    beta_M=0.35,
    batch_size=1000,
    N_COMPARTMENTS=100,
    SPB_exponent=1
)

# Process results
(S_final, I_final, R_final), R0, OH = results
S = np.sum(S_final, axis=1)
I = np.sum(I_final, axis=1)
R = np.sum(R_final, axis=1)

# Reshape arrays correctly
S = S.reshape(NM, NP).T  # Shape becomes (NP, NM)
I = I.reshape(NM, NP).T
R = R.reshape(NM, NP).T
R0 = R0.reshape(NM, NP).T
OH = OH.reshape(NM, NP).T

temp = R + I  # Total infected

# Create plot showing relative change
means = homogeneous_distribution(mean_range["n"], mean_range["m"], mean_range["M"])
pols = homogeneous_distribution(pol_range["n"], pol_range["m"], pol_range["M"])

fig, ax = plt.subplots(figsize=(Lx, Ly))
for i in range(NM):
    # Calculate percentage change relative to zero polarization
    baseline = temp[0, i]  # Value at polarization = 0
    relative_change = (temp[:, i] - baseline) / baseline * 100
    ax.plot(pols, relative_change, color=CP2[i], label=f"mean = {means[i]:.2f}")

# Add a horizontal line at 0
ax.axhline(0, color='black', lw=1, ls='--')

ax.legend()
ax.set_xlabel("Polarization")
ax.set_ylabel("% Change in infected population")
ax.set_title("Relative effect of polarization")
ax.set_xlim(0, 1)
ax.set_ylim(-50, 100)
ax.set_xticks([0, 0.5, 1])

plt.tight_layout()
plt.savefig("figures/maskSIR_pol_relative.png", dpi=300)
plt.show()
```

### Interpretation

- Relative change plots highlight the magnitude of the effect compared to a baseline condition
- In this example, we can see the percentage change in infection levels as polarization increases
- The zero line indicates no change from the baseline
- Values below zero show improvement (decreased infection), while values above zero show worse outcomes

## Comparing Models

You can create comparative visualizations to highlight differences between models:

```python
# Run all three models with the same parameters
NP = 50
NS = 50

# Parameter ranges for all models
pol_range = {"m": 0, "M": 1, "n": NP}
mask_max_range = {"m": 0, "M": 1, "n": NS}
test_max_range = {"m": 0, "M": 0.3, "n": NS}
vacc_max_range = {"m": 0, "M": 0.1, "n": NS}

# maskSIR
results_mask = sweep_pol_mask_maskSIR(
    mask_max_range=mask_max_range,
    pol_range=pol_range,
    h=0,
    dT=0.1,
    T=1000,
    beta_M=0.2,
    batch_size=1000,
    N_COMPARTMENTS=100,
    SPB_exponent=1
)

# SIRT
results_test = sweep_pol_SPB_SIRT(
    test_max_range=test_max_range,
    pol_range=pol_range,
    h=0,
    dT=1,
    T=1000,
    batch_size=1000,
    susceptibility_rate=0.25,
    N_COMPARTMENTS=100
)

# SIRV
results_vacc = sweep_pol_SPB_SIRV(
    vacc_max_range=vacc_max_range,
    pol_range=pol_range,
    h=0,
    dT=1,
    T=1000,
    batch_size=1000,
    susceptibility_rate=0.3,
    N_COMPARTMENTS=100
)

# Process and reshape results
# (Processing code omitted for brevity)

# Create comparative plot
fig, axes = plt.subplots(1, 3, figsize=(3*Lx, Ly))

# Plot maskSIR
im0 = axes[0].imshow(RI_mask, cmap=my_hot_r, aspect="auto", origin="lower", 
                    extent=[0, 1, 0, 1], vmin=0, vmax=1)
axes[0].set_title("Mask-wearing (SIR)")
axes[0].set_xlabel("Polarization")
axes[0].set_ylabel("Max mask effectiveness")

# Plot SIRT
im1 = axes[1].imshow(RI_test, cmap=my_hot_r, aspect="auto", origin="lower", 
                    extent=[0, 1, 0, 0.3], vmin=0, vmax=1)
axes[1].set_title("Testing (SIRT)")
axes[1].set_xlabel("Polarization")
axes[1].set_ylabel("Max testing rate")

# Plot SIRV
im2 = axes[2].imshow(RI_vacc, cmap=my_hot_r, aspect="auto", origin="lower", 
                    extent=[0, 1, 0, 0.1], vmin=0, vmax=1)
axes[2].set_title("Vaccination (SIRV)")
axes[2].set_xlabel("Polarization")
axes[2].set_ylabel("Max vaccination rate")

# Add colorbar
cbar = fig.colorbar(im0, ax=axes.ravel().tolist())
cbar.set_label("Total infected proportion")

plt.tight_layout()
plt.savefig("figures/model_comparison.png", dpi=300)
plt.show()
```

### Interpretation

- When comparing different models, the opposite effects of polarization become clear
- For mask-wearing (individual protection), higher polarization improves outcomes
- For testing and vaccination (population-level interventions), higher polarization worsens outcomes
- This fundamental difference highlights how the nature of the intervention interacts with population heterogeneity

## Tips for Effective Visualization

1. **Use consistent colormaps**: Use the same colormap for similar types of data to enable easy comparison between figures.

2. **Add contour lines**: Contour lines help highlight specific thresholds and make it easier to compare different regions of the parameter space.

3. **Create multiple visualization types**: Use both heatmaps for overall patterns and line plots for specific slices through the parameter space.

4. **Show relative changes**: Plots showing percentage changes from a baseline help highlight the magnitude and direction of effects.

5. **Use informative axis labels and titles**: Clearly label all axes and include units where appropriate.

6. **Standardize plot sizes and styles**: Use consistent figure dimensions and styling for a cohesive presentation.

7. **Save high-resolution figures**: Use dpi=300 or higher for publication-quality images.

8. **Consider accessibility**: Use colorblind-friendly palettes and include alternative representations (like contour lines) to make your visualizations accessible.

mask_max_range = {"m": 0, "M": 1, "n": NS}  # Range for maximum mask-wearing
pol_range = {"m": 0, "M": 1, "n": NP}       # Range for polarization

results = sweep_pol_mask_maskSIR(
    mask_max_range=mask_max_range,
    pol_range=pol_range,
    h=0,                      # Homophily parameter
    dT=0.1,
    T=1000,
    beta_M=0.2,               # Maximum susceptibility
    batch_size=1000,
    N_COMPARTMENTS=100,
    SPB_exponent=1
)

# Process results
(S_final, I_final, R_final), R0, OH = results
S = np.sum(S_final, axis=1)
I = np.sum(I_final, axis=1)
R = np.sum(R_final, axis=1)

# Reshape arrays to match parameter grid
S = S.reshape(NP, NS).transpose()
I = I.reshape(NP, NS).transpose()
R = R.reshape(NP, NS).transpose()
R0 = R0.reshape(NP, NS).transpose()
OH = OH.reshape(NP, NS).transpose()

# Calculate total infected (R+I)
FIG_A_RI = R + I
FIG_A_R0 = R0

# Create plot
masks = homogeneous_distribution(NS, 0, 1)
pol = homogeneous_distribution(NP, 0, 1)

fig, ax = plt.subplots(figsize=(Lx, Ly))
cax = ax.imshow(FIG_A_RI, cmap=my_hot_r, aspect="auto", origin="lower", 
               extent=[0, 1, 0, 1], vmin=0, vmax=1)
cbar = fig.colorbar(cax, ax=ax)

# Add contour lines
CS = ax.contour(pol, masks, FIG_A_RI, levels=[0.25, 0.5, 0.75], 
                linewidths=1, colors="black", linestyles="dashed")

ax.set_xlabel("Polarization")
ax.set_ylabel("Maximum mask effectiveness")
ax.set_title("Total infected population")
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])

plt.tight_layout()
plt.savefig("figures/maskSIR_pol_vs_mask.png", dpi=300)
plt.show()
```

### Interpretation

- The heatmap shows the proportion of the population that becomes infected
- Darker colors represent lower infection levels (better outcomes)
- Key observation: With homophily h=0, higher polarization (x-axis) can paradoxically lead to better outcomes (lower infection rates) for mask-wearing interventions
- This happens because the benefit gained from the high-compliance portion of the population outweighs the cost from the low-compliance portion

## Figure 2: Polarization vs. Mean Behavior (maskSIR Model)

This figure explores how the disease spread varies with different combinations of polarization and mean behavior level.

```python
# Run parameter sweep
NP = 100  # Number of polarization points
NM = 100  # Number of mean behavior points

pol_range = {"m": 0, "M": 1, "n": NP}
mean_range = {"m": 0.0, "M": 1.0, "n": NM}

results = sweep_pol_mean_maskSIR(
    mu_max=1,                # Fixed maximum mask effectiveness
    pol_range=pol_range,
    mean_range=mean_range,
    dT=0.1,
    T=1000,
    beta_M=0.35,
    batch_size=1000,
    N_COMPARTMENTS=100,
    SPB_exponent=1
)

# Process results
(S_final, I_final, R_final), R0, OH = results
S = np.sum(S_final, axis=1)
I = np.sum(I_final, axis=1)
R = np.sum(R_final, axis=1)

# Reshape arrays
S = np.flipud(S.reshape(NP, NM))
I = np.flipud(I.reshape(NP, NM))
R = np.flipud(R.reshape(NP, NM))
R0 = np.flipud(R0.reshape(NP, NM))
OH = np.flipud(OH.reshape(NP, NM))

FIG_B_RI = np.flipud(R + I)

# Create plot
mean_vals = homogeneous_distribution(NM, 0, 1)
pol = homogeneous_distribution(NP, 0, 1)

fig, ax = plt.subplots(figsize=(Lx, Ly))
cax = ax.imshow(FIG_B_RI, cmap=my_hot_r, aspect="auto", origin="lower", 
               extent=[0, 1, 0, 1], vmin=0, vmax=1, interpolation="none")
cbar = fig.colorbar(cax, ax=ax)

# Add contour lines
CS = ax.contour(pol, mean_vals, FIG_B_RI, levels=[0.25, 0.5, 0.75], 
                linewidths=0.5, colors="black", linestyles="dashed")

ax.set_xlabel("Polarization")
ax.set_ylabel("Mean behavior")
ax.set_title("Total infected population")
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])

plt.tight_layout()
plt.savefig("figures/maskSIR_pol_vs_mean.png", dpi=300)
plt.show()
```

### Interpretation

- This heatmap shows how the infection levels vary with polarization (x-axis) and mean behavior level (y-axis)
- For a fixed mean behavior level (horizontal slice), increasing polarization tends to decrease the total infected population
- For fixed polarization (vertical slice), increasing mean behavior (higher average mask usage) decreases infection levels
- The non-linear relationship suggests that the distribution of behaviors can be as important as the average level

## Figure 3: Polarization vs. Homophily (maskSIR Model)

This figure examines how social structure (homophily) interacts with polarization to affect disease spread.

```python
# Run parameter sweep
NP = 100  # number of polarization points
NH = 100  # number of homophily points

h_range = {"m": -10.0, "M": 10.0, "n": NH}
pol_range = {"m": 0, "M": 1, "n": NP}

results = sweep_hom_pol_maskSIR(
    h_range=h_range,
    mu_max=1,                # Fixed maximum mask effectiveness
    pol_range=pol_range,
    dT=0.1,
    T=1000,
    beta_M=0.35,
    batch_size=1000,
    N_COMPARTMENTS=100,
    SPB_exponent=1
)

# Process results
(S_final, I_final, R_final), R0, OH = results
S = np.sum(S_final, axis=1)
I = np.sum(I_final, axis=1)
R = np.sum(R_final, axis=1)

# Reshape arrays
S = S.reshape(NP, NH)
I = I.reshape(NP, NH)
R = R.reshape(NP, NH)
R0 = R0.reshape(NP, NH)
OH = OH.reshape(NP, NH)

FIG_C_RI = R + I

# Create plot
h_vals = homogeneous_distribution(NH, h_range["m"], h_range["M"])
pol = homogeneous_distribution(NP, pol_range["m"], pol_range["M"])

fig, ax = plt.subplots(figsize=(Lx, Ly))
cax = ax.imshow(FIG_C_RI, cmap=my_hot_r, aspect="auto", origin="lower", 
               extent=[-10, 10, 0, 1], vmin=0, vmax=1, interpolation="none")
cbar = fig.colorbar(cax, ax=ax)

# Add contour lines
CS = ax.contour(h_vals, pol, FIG_C_RI, levels=[0.25, 0.5, 0.75], 
                linewidths=1, colors="black", linestyles="dashed")

ax.set_xlabel("Homophily (h)")
ax.set_ylabel("Polarization")
ax.set_title("Total infected population")
ax.set_xticks([-10, 0, 10])
ax.set_yticks([0, 0.5, 1])

plt.tight_layout()
plt.savefig("figures/maskSIR_pol_vs_homophily.png", dpi=300)
plt.show()
```

### Interpretation

- This figure shows how infection levels vary with homophily (x-axis) and polarization (y-axis)
- Negative homophily (heterophily, h<0) generally leads to better outcomes
- With positive homophily (h>0), which is common in real societies, higher polarization can improve outcomes
- This complex relationship demonstrates that social structure strongly influences the effect of behavioral heterogeneity

# Visualization Guide (Part 2)

## Figure 4: Testing Effectiveness (SIRT Model)

This figure shows how testing interventions affect disease spread at different polarization levels.

```python
# Run parameter sweep
NP = 100
NS = 100

test_max_range = {"m": 0, "M": 0.3, "n": NS}
pol_range = {"m": 0, "M": 1, "n": NP}

results = sweep_pol_SPB_SIRT(
    test_max_range=test_max_range,
    pol_range=pol_range,
    h=0,
    dT=1,
    T=1000,
    batch_size=1000,
    susceptibility_rate=0.25,
    N_COMPARTMENTS=100
)

# Process results
(S_final, I_final, R_final), R0, OH = results
S = np.sum(S_final, axis=1)
I = np.sum(I_final, axis=1)
R = np.sum(R_final, axis=1)

# Reshape arrays
S = S.reshape(NP, NS).transpose()
I = I.reshape(NP, NS).transpose()
R = R.reshape(NP, NS).transpose()
R0 = R0.reshape(NP, NS).transpose()
OH = OH.reshape(NP, NS).transpose()

FIG_A_RI = R + I
FIG_A_R0 = R0

# Create plot
test_rates = homogeneous_distribution(NS, test_max_range["m"], test_max_range["M"])
pol = homogeneous_distribution(NP, pol_range["m"], pol_range["M"])

fig, ax = plt.subplots(figsize=(Lx, Ly))
cax = ax.imshow(FIG_A_RI, cmap=my_hot_r, aspect="auto", origin="lower", 
               extent=[0, 1, 0, 0.3], vmin=0, vmax=1)
cbar = fig.colorbar(cax, ax=ax)

# Add contour lines
CS = ax.contour(pol, test_rates, FIG_A_RI, levels=[0.25, 0.5, 0.75], 
                linewidths=1, colors="black", linestyles="dashed")

ax.set_xlabel("Polarization")
ax.set_ylabel("Maximum testing rate")
ax.set_title("Total infected population (SIRT model)")
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.15, 0.3])

plt.tight_layout()
plt.savefig("figures/SIRT_pol_vs_test.png", dpi=300)
plt.show()
```

### Interpretation

- The heatmap shows how testing effectiveness interacts with polarization
- In contrast to mask-wearing, higher polarization leads to worse outcomes for testing interventions
- This is because testing primarily works by removing infected individuals from circulation, which is a cumulative effect rather than an individual protection measure
- Note the inverse relationship compared to the mask-wearing case

## Figure 5: Vaccination Effectiveness (SIRV Model)

This figure explores how vaccination effectiveness varies with polarization.

```python
# Run parameter sweep
NP = 100
NS = 100

vacc_max_range = {"m": 0, "M": 0.1, "n": NS}
pol_range = {"m": 0, "M": 1, "n": NP}

results = sweep_pol_SPB_SIRV(
    vacc_max_range=vacc_max_range,
    pol_range=pol_range,
    h=0,
    dT=1,
    T=1000,
    batch_size=1000,
    susceptibility_rate=0.3,
    N_COMPARTMENTS=100
)

# Process results
(S_final, I_final, R_final, V_final), R0, OH = results
S = np.sum(S_final, axis=1)
I = np.sum(I_final, axis=1)
R = np.sum(R_final, axis=1)
V = np.sum(V_final, axis=1)

# Reshape arrays
S = S.reshape(NP, NS).transpose()
I = I.reshape(NP, NS).transpose()
R = R.reshape(NP, NS).transpose()
V = V.reshape(NP, NS).transpose()
R0 = R0.reshape(NP, NS).transpose()
OH = OH.reshape(NP, NS).transpose()

FIG_A_RI = R + I
FIG_A_V = V

# Create plot
vacc_rates = homogeneous_distribution(NS, vacc_max_range["m"], vacc_max_range["M"])
pol = homogeneous_distribution(NP, pol_range["m"], pol_range["M"])

fig, ax = plt.subplots(figsize=(Lx, Ly))
cax = ax.imshow(FIG_A_RI, cmap=my_hot_r, aspect="auto", origin="lower", 
               extent=[0, 1, 0, 0.1], vmin=0, vmax=1)
cbar = fig.colorbar(cax, ax=ax)

# Add contour lines
CS = ax.contour(pol, vacc_rates, FIG_A_RI, levels=[0.25, 0.5, 0.75], 
                linewidths=1, colors="black", linestyles="dashed")

ax.set_xlabel("Polarization")
ax.set_ylabel("Maximum vaccination rate")
ax.set_title("Total infected population (SIRV model)")
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.05, 0.1])

plt.tight_layout()
plt.savefig("figures/SIRV_pol_vs_vacc.png", dpi=300)
plt.show()
```

### Interpretation

- Similar to testing, vaccination shows the opposite relationship with polarization compared to mask-wearing
- Higher polarization leads to worse outcomes for vaccination campaigns
- This is because vaccination provides both individual protection and contributes to herd immunity, making uniform vaccination more effective
- The results highlight that different intervention types interact differently with population heterogeneity

## Figure 6: Line Plots for Fixed Values

Creating line plots for fixed values of one parameter can help visualize slices through the parameter space:

```python
# Example: Line plot of polarization effect at different homophily values
NP = 100  # number of polarization points
NH = 5    # number of homophily points to plot

h_range = {"m": -10.0, "M": 10.0, "n": NH}
pol_range = {"m": 0, "M": 1, "n": NP}

results = sweep_hom_pol_maskSIR(
    h_range=h_range,
    mu_max=1,
    pol_range=pol_range,
    dT=0.1,
    T=1000,
    beta_M=0.35,
    batch_size=1000,
    N_COMPARTMENTS=100,
    SPB_exponent=1
)

# Process results
(S_final, I_final, R_final), R0, OH = results
S = np.sum(S_final, axis=1)
I = np.sum(I_final, axis=1)
R = np.sum(R_final, axis=1)

# Reshape arrays
S = S.reshape(NP, NH)
I = I.reshape(NP, NH)
R = R.reshape(NP, NH)
R0 = R0.reshape(NP, NH)
OH = OH.reshape(NP, NH)

temp = R + I  # Total infected

# Create plot
homs = homogeneous_distribution(h_range["n"], h_range["m"], h_range["M"])
pols = homogeneous_distribution(pol_range["n"], pol_range["m"], pol_range["M"])

fig, ax = plt.subplots(figsize=(Lx, Ly))
for i in range(NH):
    ax.plot(pols, temp[:, i], color=CP3[i], label=f"h = {homs[i]:.1f}")

ax.legend()
ax.set_xlabel("Polarization")
ax.set_ylabel("Total infected proportion")
ax.set_title("Effect of polarization at different homophily values")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])

plt.tight_layout()
plt.savefig("figures/maskSIR_pol_lines.png", dpi=300)
plt.show()
```

### Interpretation

- Line plots help visualize how one parameter (polarization) affects outcomes at different fixed values of another parameter (homophily)
- This allows for more detailed comparisons than heatmaps alone
- In this example, we can see the complex interplay between polarization and homophily more clearly

## Figure 7: Relative Change Plots

Creating plots of the relative change compared to a baseline can highlight the effect size:

```python
# Example: Relative change in infected proportion compared to zero polarization
NP = 100  # Number of polarization points
NM = 5    # Number of mean behavior points to plot

pol_range = {"m": 0, "M": 1, "n": NP}
mean_range = {"m": 0.0, "M": 1.0, "n": NM}

results = sweep_pol_mean_maskSIR(
    mu_max=1,
    pol_range=pol_range,
    mean_range=mean_range,
    dT=0.1,
    T=1000,
    beta_M=0.35,
    batch_size=1000,
    N_COMPARTMENTS=100,
    SPB_exponent=1
)

# Process results
(S_final, I_final, R_final), R0, OH = results
S = np.sum(S_final, axis=1)
I = np.sum(I_final, axis=1)
R = np.sum(R_final, axis=1)

# Reshape arrays correctly
S = S.reshape(NM, NP).T  # Shape becomes (NP, NM)
I = I.reshape(NM, NP).T
R = R.reshape(NM, NP).T
R0 = R0.reshape(NM, NP).T
OH = OH.reshape(NM, NP).T

temp = R + I  # Total infected

# Create plot showing relative change
means = homogeneous_distribution(mean_range["n"], mean_range["m"], mean_range["M"])
pols = homogeneous_distribution(pol_range["n"], pol_range["m"], pol_range["M"])

fig, ax = plt.subplots(figsize=(Lx, Ly))
for i in range(NM):
    # Calculate percentage change relative to zero polarization
    baseline = temp[0, i]  # Value at polarization = 0
    relative_change = (temp[:, i] - baseline) / baseline * 100
    ax.plot(pols, relative_change, color=CP2[i], label=f"mean = {means[i]:.2f}")

# Add a horizontal line at 0
ax.axhline(0, color='black', lw=1, ls='--')

ax.legend()
ax.set_xlabel("Polarization")
ax.set_ylabel("% Change in infected population")
ax.set_title("Relative effect of polarization")
ax.set_xlim(0, 1)
ax.set_ylim(-50, 100)
ax.set_xticks([0, 0.5, 1])

plt.tight_layout()
plt.savefig("figures/maskSIR_pol_relative.png", dpi=300)
plt.show()
```

### Interpretation

- Relative change plots highlight the magnitude of the effect compared to a baseline condition
- In this example, we can see the percentage change in infection levels as polarization increases
- The zero line indicates no change from the baseline
- Values below zero show improvement (decreased infection), while values above zero show worse outcomes

## Comparing Models

You can create comparative visualizations to highlight differences between models:

```python
# Run all three models with the same parameters
NP = 50
NS = 50

# Parameter ranges for all models
pol_range = {"m": 0, "M": 1, "n": NP}
mask_max_range = {"m": 0, "M": 1, "n": NS}
test_max_range = {"m": 0, "M": 0.3, "n": NS}
vacc_max_range = {"m": 0, "M": 0.1, "n": NS}

# maskSIR
results_mask = sweep_pol_mask_maskSIR(
    mask_max_range=mask_max_range,
    pol_range=pol_range,
    h=0,
    dT=0.1,
    T=1000,
    beta_M=0.2,
    batch_size=1000,
    N_COMPARTMENTS=100,
    SPB_exponent=1
)

# SIRT
results_test = sweep_pol_SPB_SIRT(
    test_max_range=test_max_range,
    pol_range=pol_range,
    h=0,
    dT=1,
    T=1000,
    batch_size=1000,
    susceptibility_rate=0.25,
    N_COMPARTMENTS=100
)

# SIRV
results_vacc = sweep_pol_SPB_SIRV(
    vacc_max_range=vacc_max_range,
    pol_range=pol_range,
    h=0,
    dT=1,
    T=1000,
    batch_size=1000,
    susceptibility_rate=0.3,
    N_COMPARTMENTS=100
)

# Process and reshape results
# (Processing code omitted for brevity)

# Create comparative plot
fig, axes = plt.subplots(1, 3, figsize=(3*Lx, Ly))

# Plot maskSIR
im0 = axes[0].imshow(RI_mask, cmap=my_hot_r, aspect="auto", origin="lower", 
                    extent=[0, 1, 0, 1], vmin=0, vmax=1)
axes[0].set_title("Mask-wearing (SIR)")
axes[0].set_xlabel("Polarization")
axes[0].set_ylabel("Max mask effectiveness")

# Plot SIRT
im1 = axes[1].imshow(RI_test, cmap=my_hot_r, aspect="auto", origin="lower", 
                    extent=[0, 1, 0, 0.3], vmin=0, vmax=1)
axes[1].set_title("Testing (SIRT)")
axes[1].set_xlabel("Polarization")
axes[1].set_ylabel("Max testing rate")

# Plot SIRV
im2 = axes[2].imshow(RI_vacc, cmap=my_hot_r, aspect="auto", origin="lower", 
                    extent=[0, 1, 0, 0.1], vmin=0, vmax=1)
axes[2].set_title("Vaccination (SIRV)")
axes[2].set_xlabel("Polarization")
axes[2].set_ylabel("Max vaccination rate")

# Add colorbar
cbar = fig.colorbar(im0, ax=axes.ravel().tolist())
cbar.set_label("Total infected proportion")

plt.tight_layout()
plt.savefig("figures/model_comparison.png", dpi=300)
plt.show()
```

