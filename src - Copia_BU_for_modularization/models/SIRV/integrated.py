from ..consolidated_dynamics import sim_SIRV_final
from ..consolidated_sweeps import sweep_pol_behavior, sweep_pol_mean, sweep_hom_pol

def sweep_pol_SPB_SIRV(*args, **kwargs):
    return sweep_pol_behavior("vaccine", *args, **kwargs)

def sweep_pol_mean_SIRV(*args, **kwargs):
    return sweep_pol_mean("vaccine", *args, **kwargs)

def sweep_pol_hom_SIRV(*args, **kwargs):
    return sweep_hom_pol("vaccine", *args, **kwargs)