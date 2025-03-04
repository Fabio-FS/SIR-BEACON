# Import the integrated versions but expose them with the original names
from .integrated import (
    sweep_pol_SPB_SIRV,
    sweep_pol_mean_SIRV,
    sweep_pol_hom_SIRV
)

# Export all the functions with their original names
__all__ = [
    'sweep_pol_SPB_SIRV',
    'sweep_pol_mean_SIRV',
    'sweep_pol_hom_SIRV'
]