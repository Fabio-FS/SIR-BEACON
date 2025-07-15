from ..consolidated_dynamics import sim_SIRT_final
from ..consolidated_sweeps import sweep_pol_behavior, sweep_pol_mean, sweep_hom_pol

def sweep_pol_SPB_SIRT(*args, **kwargs):
    return sweep_pol_behavior("test", *args, **kwargs)

def sweep_pol_mean_SIRT(*args, **kwargs):
    return sweep_pol_mean("test", *args, **kwargs)

def sweep_hom_pol_SIRT(*args, **kwargs):
    return sweep_hom_pol("test", *args, **kwargs)