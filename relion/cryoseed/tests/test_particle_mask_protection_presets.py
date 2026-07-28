from __future__ import annotations

import math

from cryoseed.cli import build_parser
from cryoseed.config import MainConfig
from cryoseed.utils.particle_mask import (
    DEFAULT_PARTICLE_MASK_PROTECTION_COVERAGE,
    DEFAULT_PARTICLE_MASK_PROTECTION_RADIUS_FACTOR,
    PARTICLE_MASK_PROTECTION_COVERAGE_PRESETS,
    particle_mask_protection_factor_for_coverage,
)


def test_particle_mask_protection_presets_match_rayleigh_derivation():
    for coverage_text, factor in PARTICLE_MASK_PROTECTION_COVERAGE_PRESETS.items():
        coverage = float(coverage_text)
        expected = particle_mask_protection_factor_for_coverage(coverage)
        assert math.isclose(factor, expected, rel_tol=0.0, abs_tol=1e-12)


def test_particle_mask_default_matches_0999_preset():
    cfg = MainConfig()
    assert DEFAULT_PARTICLE_MASK_PROTECTION_COVERAGE == "0.999"
    assert math.isclose(
        cfg.data.particle_mask.protection_radius_factor,
        DEFAULT_PARTICLE_MASK_PROTECTION_RADIUS_FACTOR,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_cli_particle_mask_protection_coverage_sets_factor():
    parser = build_parser()
    args = parser.parse_args(
        [
            "homorefine",
            "--particle-mask-protection-coverage",
            "0.99",
        ]
    )

    cfg = MainConfig.from_cli_args(args)
    assert math.isclose(
        cfg.data.particle_mask.protection_radius_factor,
        PARTICLE_MASK_PROTECTION_COVERAGE_PRESETS["0.99"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_cli_particle_mask_protection_factor_overrides_coverage_preset():
    parser = build_parser()
    args = parser.parse_args(
        [
            "homorefine",
            "--particle-mask-protection-coverage",
            "0.99",
            "--particle-mask-protection-radius-factor",
            "1.75",
        ]
    )

    cfg = MainConfig.from_cli_args(args)
    assert cfg.data.particle_mask.protection_radius_factor == 1.75