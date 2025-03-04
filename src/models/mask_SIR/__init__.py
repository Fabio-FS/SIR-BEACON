# Import the integrated versions but expose them with the original names
from .integrated import (
    sweep_pol_mask_maskSIR,
    sweep_pol_mean_maskSIR,
    sweep_hom_pol_maskSIR
)

# Export all the functions with their original names
__all__ = [
    'sweep_pol_mask_maskSIR',
    'sweep_pol_mean_maskSIR',
    'sweep_hom_pol_maskSIR'
]