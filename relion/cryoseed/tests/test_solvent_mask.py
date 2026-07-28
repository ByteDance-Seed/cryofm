from __future__ import annotations

import torch

from cryoseed.config import MainConfig
from cryoseed.engines.homorefine import HomoRefineEngine
from cryoseed.modules.volume import VoxelGrid

def _make_engine(config: MainConfig) -> HomoRefineEngine:
    engine = HomoRefineEngine.__new__(HomoRefineEngine)
    torch.nn.Module.__init__(engine)
    engine.config = config
    engine.device = torch.device("cpu")
    return engine


def test_engine_apply_solvent_mask_uses_static_mask_for_both_references_and_preserves_accumulators():
    config = MainConfig()
    engine = _make_engine(config)
    engine.solvent_mask = torch.full((4, 4, 4), 0.5)
    engine.unmasked_volume_real_half0 = torch.ones((1, 4, 4, 4))
    engine.unmasked_volume_real_half1 = torch.ones((1, 4, 4, 4))

    engine.volume_half0 = VoxelGrid(grid_size=4, requires_accum=True)
    engine.volume_half1 = VoxelGrid(grid_size=4, requires_accum=True)
    engine.volume_real_half0 = torch.zeros((1, 4, 4, 4))
    engine.volume_real_half1 = torch.zeros((1, 4, 4, 4))
    engine.volume_half0.accum_numer.fill_(3.0)
    engine.volume_half1.accum_numer.fill_(5.0)
    numer0 = engine.volume_half0.accum_numer.clone()
    numer1 = engine.volume_half1.accum_numer.clone()

    selected_mask = engine.apply_solvent_mask()

    torch.testing.assert_close(selected_mask, torch.full((4, 4, 4), 0.5))
    torch.testing.assert_close(
        engine.volume_half0.volume_real,
        selected_mask.unsqueeze(0),
    )
    torch.testing.assert_close(
        engine.volume_half1.volume_real,
        selected_mask.unsqueeze(0),
    )
    torch.testing.assert_close(engine.volume_half0.accum_numer, numer0)
    torch.testing.assert_close(engine.volume_half1.accum_numer, numer1)


def test_engine_apply_solvent_mask_uses_unmasked_identity_when_disabled():
    config = MainConfig()
    engine = _make_engine(config)
    engine.solvent_mask = None
    engine.unmasked_volume_real_half0 = torch.full((1, 4, 4, 4), 2.0)
    engine.unmasked_volume_real_half1 = torch.full((1, 4, 4, 4), 3.0)
    engine.volume_real_half0 = torch.zeros((1, 4, 4, 4))
    engine.volume_real_half1 = torch.zeros((1, 4, 4, 4))

    selected_mask = engine.apply_solvent_mask()

    assert selected_mask is None
    torch.testing.assert_close(engine.volume_real_half0, engine.unmasked_volume_real_half0)
    torch.testing.assert_close(engine.volume_real_half1, engine.unmasked_volume_real_half1)