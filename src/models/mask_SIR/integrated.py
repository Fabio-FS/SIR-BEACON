from ..consolidated_dynamics import sim_maskSIR_final
from ..consolidated_sweeps import sweep_pol_behavior, sweep_pol_mean, sweep_hom_pol

def sweep_pol_mask_maskSIR(*args, **kwargs):
    return sweep_pol_behavior("mask", *args, **kwargs)

def sweep_pol_mean_maskSIR(*args, **kwargs):
    return sweep_pol_mean("mask", *args, **kwargs)

def sweep_hom_pol_maskSIR(*args, **kwargs):
    return sweep_hom_pol("mask", *args, **kwargs)