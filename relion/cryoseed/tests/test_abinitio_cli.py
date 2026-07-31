from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from cryoseed.cli import build_parser
from cryoseed.config import MainConfig
from cryoseed.state import OptimState


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _assert_command_yaml_shape(raw: dict, *, command: str) -> None:
    other_command = "homorefine" if command == "abinitio" else "abinitio"
    assert command in raw
    assert other_command not in raw
    assert "modules" in raw
    if command == "abinitio":
        assert "use_cache" not in raw["abinitio"]["scheduler"]
        assert "increase_radius_aggressive_factor" not in raw["abinitio"]["scheduler"]
        assert "external_reconstruct" not in raw["abinitio"]["engine"]


def test_abinitio_cli_loads_command_defaults():
    parser = build_parser()
    args = parser.parse_args(["abinitio"])

    cfg = MainConfig.from_cli_args(args)

    assert args.command == "abinitio"
    assert cfg.abinitio.scheduler.schedule_check_interval_iters == 100
    assert cfg.abinitio.scheduler.confidence_threshold == 0.5
    assert cfg.abinitio.engine.num_epochs == 1000
    assert cfg.abinitio.scheduler.target_side_length_resolution == 10.0
    assert cfg.abinitio.engine.solvent_mask == "none"
    assert cfg.modules.search.init_trans_grid_extent is None


def test_abinitio_cli_help_only_exposes_leaf_config_flags():
    parser = build_parser()
    subparsers_action = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    abinitio_parser = subparsers_action.choices["abinitio"]
    option_strings = {
        option
        for action in abinitio_parser._actions
        for option in action.option_strings
    }

    assert "--num-volumes" in option_strings
    assert "--learning-rate" in option_strings
    assert "--volume" not in option_strings
    assert "--search" not in option_strings
    assert "--statistics" not in option_strings
    assert "--engine" not in option_strings
    assert "--solver" not in option_strings
    assert "--scheduler" not in option_strings


def test_init_trans_grid_extent_null_auto_derives_from_image_size():
    parser = build_parser()
    args = parser.parse_args(["abinitio", "--image-size", "200"])

    cfg = MainConfig.from_cli_args(args)
    state = OptimState.from_config(cfg, command="abinitio")

    assert cfg.modules.search.init_trans_grid_extent is None
    assert state.schedule.trans_grid_extent == 100.0


def test_command_defaults_yaml_only_lists_supported_fields():
    root = _repo_root()
    homorefine_path = root / "src" / "cryoseed" / "config" / "defaults" / "homorefine.yaml"
    abinitio_path = root / "src" / "cryoseed" / "config" / "defaults" / "abinitio.yaml"

    homorefine_raw = MainConfig._load_file(str(homorefine_path))
    abinitio_raw = MainConfig._load_file(str(abinitio_path))

    _assert_command_yaml_shape(homorefine_raw, command="homorefine")
    _assert_command_yaml_shape(abinitio_raw, command="abinitio")


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


def test_full_config_loads():
    full_config_path = _repo_root() / "full_config.yaml"
    cfg = MainConfig.from_file(str(full_config_path))

    assert cfg.homorefine.scheduler.first_epoch_ncc is True
    assert cfg.abinitio.scheduler.schedule_check_interval_iters == 100


def test_full_config_lists_full_schema():
    full_config_path = _repo_root() / "full_config.yaml"
    raw = MainConfig._load_file(str(full_config_path))

    assert raw == MainConfig().to_dict()

@pytest.mark.parametrize(
    ("command", "config_text", "error_match"),
    [
        (
            "abinitio",
            "scheduler:\n  use_cache: true\n",
            "Unknown config section `scheduler`.",
        ),
        (
            "abinitio",
            "scheduler:\n  increase_radius_aggressive_factor: 0.5\n",
            "Unknown config section `scheduler`.",
        ),
        (
            "abinitio",
            "reconstruction:\n  external_reconstruct: true\n",
            "Unknown config section `reconstruction`.",
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

    _assert_command_yaml_shape(homorefine_raw, command="homorefine")
    _assert_command_yaml_shape(abinitio_raw, command="abinitio")