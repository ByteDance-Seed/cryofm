from __future__ import annotations

import math

# Recommended protection factors assume per-particle translation errors are
# approximately 2D isotropic Gaussian. In that case the radial error magnitude
# follows a Rayleigh distribution, and the radius that covers a target fraction
# p of particles can be written as:
#
#   r_p = sigma * sqrt(-2 * log(1 - p))
#
# The scheduler expands the particle-mask radius by:
#
#   extra_radius_px = protection_radius_factor * trans_update_rms
#
# where trans_update_rms ~= sqrt(E[dx^2 + dy^2]) = sqrt(2) * sigma. Dividing
# the Rayleigh quantile by that RMS gives the dimensionless factor:
#
#   protection_radius_factor = sqrt(-log(1 - p))
#
# This lets users choose coverage-oriented presets without manually deriving the
# matching factor each time.


def particle_mask_protection_factor_for_coverage(coverage: float) -> float:
    """Return the radius factor that covers ``coverage`` under the Rayleigh model."""
    coverage = float(coverage)
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must be in (0, 1), got {coverage}")
    return math.sqrt(-math.log1p(-coverage))


PARTICLE_MASK_PROTECTION_COVERAGE_PRESETS: dict[str, float] = {
    "0.9": particle_mask_protection_factor_for_coverage(0.9),
    "0.99": particle_mask_protection_factor_for_coverage(0.99),
    "0.999": particle_mask_protection_factor_for_coverage(0.999),
    "0.9999": particle_mask_protection_factor_for_coverage(0.9999),
}

DEFAULT_PARTICLE_MASK_PROTECTION_COVERAGE = "0.999"
DEFAULT_PARTICLE_MASK_PROTECTION_RADIUS_FACTOR = (
    PARTICLE_MASK_PROTECTION_COVERAGE_PRESETS[
        DEFAULT_PARTICLE_MASK_PROTECTION_COVERAGE
    ]
)