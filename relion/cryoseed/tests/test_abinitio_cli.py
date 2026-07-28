from __future__ import annotations

from pathlib import Path

import pytest

from cryoseed.cli import build_parser
from cryoseed.config import MainConfig
from cryoseed.state import OptimState


def test_abinitio_cli_loads_command_defaults():
    parser = build_parser()
    args = parser.parse_args(["abinitio"])

    cfg = MainConfig.from_cli_args(args)

    assert args.command == "abinitio"
    assert cfg.scheduler.schedule_check_interval_iters == 100
    assert cfg.scheduler.confidence_threshold == 0.5
    assert cfg.pose_search.oversampling_deduplicate is False
    assert cfg.homorefine.first_epoch_ncc is True
    assert cfg.abinitio.num_epochs == 1000
    assert cfg.abinitio.init_particles_per_volume == 100
    assert cfg.abinitio.target_side_length_resolution == 10.0
    assert cfg.abinitio.target_healpix_order is None
    assert cfg.abinitio.learning_rate_decay == 0.9995
    assert cfg.abinitio.pose_rotation_stability_factor == 1.0
    assert cfg.abinitio.pose_translation_stability_factor == 0.5
    assert cfg.abinitio.solvent_mask == "none"
    assert cfg.pose_search.init_trans_grid_extent is None


def test_init_trans_grid_extent_null_auto_derives_from_image_size():
    parser = build_parser()
    args = parser.parse_args(["abinitio", "--image-size", "200"])

    cfg = MainConfig.from_cli_args(args)
    state = OptimState.from_config(cfg)

    assert cfg.pose_search.init_trans_grid_extent is None
    assert state.schedule.trans_grid_extent == 100.0


def test_command_defaults_yaml_only_lists_supported_fields():
    root = Path(__file__).resolve().parents[1]
    homorefine_path = root / "src" / "cryoseed" / "config" / "defaults" / "homorefine.yaml"
    abinitio_path = root / "src" / "cryoseed" / "config" / "defaults" / "abinitio.yaml"

    homorefine_raw = MainConfig._load_file(str(homorefine_path))
    abinitio_raw = MainConfig._load_file(str(abinitio_path))

    assert "abinitio" not in homorefine_raw
    assert "use_cache" not in abinitio_raw["scheduler"]
    assert "increase_radius_aggressive_factor" not in abinitio_raw["scheduler"]
    assert "homorefine" not in abinitio_raw


@pytest.mark.parametrize(
    "argv",
    [
        ["abinitio", "--use-cache"],
        ["abinitio", "--increase-radius-aggressive-factor", "0.5"],
        ["abinitio", "--first-epoch-ncc"],
        ["homorefine", "--learning-rate", "0.5"],
    ],
)
def test_command_cli_rejects_hidden_flags(argv):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


@pytest.mark.parametrize(
    ("command", "config_text", "error_match"),
    [
        (
            "abinitio",
            "scheduler:\n  use_cache: true\n",
            "abinitio does not support scheduler.use_cache",
        ),
        (
            "abinitio",
            "scheduler:\n  increase_radius_aggressive_factor: 0.5\n",
            "abinitio does not support scheduler.increase_radius_aggressive_factor",
        ),
        (
            "abinitio",
            "reconstruction:\n  external_reconstruct: true\n",
            "abinitio does not support reconstruction.external_reconstruct",
        ),
    ],
)
def test_command_config_rejects_fixed_field_overrides(
    tmp_path,
    command,
    config_text,
    error_match,
):
    config_path = tmp_path / f"{command}.yaml"
    config_path.write_text(
        config_text,
        encoding="utf-8",
    )

    parser = build_parser()
    args = parser.parse_args([command, "--config", str(config_path)])

    with pytest.raises(
        ValueError,
        match=error_match,
    ):
        MainConfig.from_cli_args(args)


def test_full_config_loads():
    full_config_path = Path(__file__).resolve().parents[1] / "full_config.yaml"
    cfg = MainConfig.from_file(str(full_config_path))

    assert cfg.homorefine.first_epoch_ncc is True
    assert cfg.scheduler.schedule_check_interval_iters == 100
    assert cfg.abinitio.target_side_length_resolution == 10.0


def test_full_config_lists_full_schema():
    full_config_path = Path(__file__).resolve().parents[1] / "full_config.yaml"
    raw = MainConfig._load_file(str(full_config_path))

    assert raw == MainConfig().to_dict()

@pytest.mark.parametrize(
    ("command", "config_text", "error_match"),
    [
        (
            "abinitio",
            "scheduler:\n  use_cache: true\n",
            "abinitio does not support scheduler.use_cache",
        ),
        (
            "abinitio",
            "scheduler:\n  increase_radius_aggressive_factor: 0.5\n",
            "abinitio does not support scheduler.increase_radius_aggressive_factor",
        ),
        (
            "abinitio",
            "reconstruction:\n  external_reconstruct: true\n",
            "abinitio does not support reconstruction.external_reconstruct",
        ),
    ],
)
def test_from_file_rejects_fixed_field_overrides(tmp_path, command, config_text, error_match):
    config_path = tmp_path / f"{command}.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    with pytest.raises(ValueError, match=error_match):
        MainConfig.from_file(str(config_path), command=command)


def test_save_output_config_only_prints_supported_fields(tmp_path):
    cfg = MainConfig()
    cfg.io.output_path = str(tmp_path)

    homorefine_path, _ = cfg.save_output_config(command="homorefine", filename="homorefine.yml")
    abinitio_path, _ = cfg.save_output_config(command="abinitio", filename="abinitio.yml")

    homorefine_raw = MainConfig._load_file(str(homorefine_path))
    abinitio_raw = MainConfig._load_file(str(abinitio_path))

    assert "abinitio" not in homorefine_raw
    assert "use_cache" not in abinitio_raw["scheduler"]
    assert "increase_radius_aggressive_factor" not in abinitio_raw["scheduler"]
    assert "homorefine" not in abinitio_raw