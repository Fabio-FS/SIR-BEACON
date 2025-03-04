# Import the integrated versions but expose them with the original names
from .integrated import (
    sweep_pol_SPB_SIRT,
    sweep_pol_mean_SIRT,
    sweep_hom_pol_SIRT
)

# Export all the functions with their original names
__all__ = [
    'sweep_pol_SPB_SIRT',
    'sweep_pol_mean_SIRT',
    'sweep_hom_pol_SIRT'
]