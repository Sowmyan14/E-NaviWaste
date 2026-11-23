import copy

import pytest
import yaml

from electronics_classifier.cli import main
from electronics_classifier.config import (
    ConfigError,
    config_hash,
    deep_merge,
    load_config,
    normalize_vocabulary,
)


def write_yaml(path, data):
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def threshold(cfg):
    return cfg["pipeline"]["detector"]["params"]["raw_box_threshold"]


def test_default_config_loads_and_validates():
    cfg = load_config()
    assert cfg["pipeline"]["detector"]["name"] == "grounding_dino"
    assert threshold(cfg) == 0.15


def test_layer_order_default_preset_override_set(tmp_path):
    preset = write_yaml(
        tmp_path / "p.yaml", {"pipeline": {"detector": {"params": {"raw_box_threshold": 0.2}}}}
    )
    override = write_yaml(
        tmp_path / "o.yaml", {"pipeline": {"detector": {"params": {"raw_box_threshold": 0.3}}}}
    )
    assert threshold(load_config()) == 0.15
    assert threshold(load_config(presets=[str(preset)])) == 0.2
    assert threshold(load_config(presets=[str(preset)], overrides=[override])) == 0.3
    layered = load_config(
        presets=[str(preset)],
        overrides=[override],
        sets=["pipeline.detector.params.raw_box_threshold=0.4"],
    )
    assert threshold(layered) == 0.4


def test_dicts_are_deep_merged(tmp_path):
    preset = write_yaml(
        tmp_path / "p.yaml", {"pipeline": {"detector": {"params": {"raw_box_threshold": 0.2}}}}
    )
    cfg = load_config(presets=[str(preset)])
    params = cfg["pipeline"]["detector"]["params"]
    assert params["raw_box_threshold"] == 0.2
    assert params["model_id"] == "IDEA-Research/grounding-dino-tiny"
    assert cfg["pipeline"]["crop"]["padding_ratio"] == 0.12


def test_lists_are_replaced_whole(tmp_path):
    override = write_yaml(
        tmp_path / "o.yaml",
        {"pipeline": {"detection_filters": [{"name": "top_k", "params": {"k": 3}}]}},
    )
    cfg = load_config(overrides=[override])
    assert cfg["pipeline"]["detection_filters"] == [{"name": "top_k", "params": {"k": 3}}]


def test_component_rename_replaces_params_but_same_name_merges(tmp_path):
    renamed = write_yaml(tmp_path / "r.yaml", {"pipeline": {"detector": {"name": "fake"}}})
    cfg = load_config(overrides=[renamed])
    assert cfg["pipeline"]["detector"] == {"name": "fake", "params": {}}

    renamed_with_params = write_yaml(
        tmp_path / "rp.yaml", {"pipeline": {"detector": {"name": "fake", "params": {"a": 1}}}}
    )
    cfg = load_config(overrides=[renamed_with_params])
    assert cfg["pipeline"]["detector"] == {"name": "fake", "params": {"a": 1}}

    same = write_yaml(tmp_path / "s.yaml", {"pipeline": {"detector": {"name": "grounding_dino"}}})
    assert "model_id" in load_config(overrides=[same])["pipeline"]["detector"]["params"]


def test_set_name_resets_params_and_set_accepts_flow_values():
    cfg = load_config(sets=["pipeline.detector.name=fake"])
    assert cfg["pipeline"]["detector"] == {"name": "fake", "params": {}}

    cfg = load_config(sets=["pipeline.image_preprocess=[{name: ensure_rgb_uint8}]"])
    assert cfg["pipeline"]["image_preprocess"] == [{"name": "ensure_rgb_uint8", "params": {}}]

    assert load_config(sets=["run.seed=5"])["run"]["seed"] == 5


def test_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1}, "c": [1]}
    snapshot = copy.deepcopy(base)
    deep_merge(base, {"a": {"b": 2}, "c": [2]})
    assert base == snapshot


def test_unknown_top_level_key_fails(tmp_path):
    override = write_yaml(tmp_path / "o.yaml", {"surprise": 1})
    with pytest.raises(ConfigError, match="surprise"):
        load_config(overrides=[override])


def test_validation_errors_name_the_full_config_path(tmp_path):
    with pytest.raises(ConfigError, match=r"pipeline\.crop\.padding_ratio"):
        load_config(sets=["pipeline.crop.padding_ratio=1.0"])
    with pytest.raises(ConfigError, match=r"pipeline\.crop\.padding_ratio"):
        load_config(sets=["pipeline.crop.padding_ratio=wide"])
    with pytest.raises(ConfigError, match=r"pipeline\.crop\.padding_ratioo"):
        load_config(sets=["pipeline.crop.padding_ratioo=0.1"])
    with pytest.raises(ConfigError, match=r"rendering\.mask_alpha"):
        load_config(sets=["rendering.mask_alpha=2"])


def test_bad_set_assignments_are_config_errors():
    with pytest.raises(ConfigError, match="dotted.key=value"):
        load_config(sets=["pipeline.crop.padding_ratio"])
    with pytest.raises(ConfigError, match="unknown config path"):
        load_config(sets=["pipeline.nope.thing=1"])
    with pytest.raises(ConfigError, match="not a mapping"):
        load_config(sets=["pipeline.detection_filters.k=1"])


def test_missing_preset_and_override_files_are_config_errors(tmp_path):
    with pytest.raises(ConfigError, match="unknown preset 'does_not_exist'"):
        load_config(presets=["does_not_exist"])
    with pytest.raises(ConfigError, match="preset file not found"):
        load_config(presets=[str(tmp_path / "missing.yaml")])
    with pytest.raises(ConfigError, match="cannot read config file"):
        load_config(overrides=[tmp_path / "missing.yaml"])


def test_electronic_device_is_auto_appended():
    cfg = load_config(sets=["vocabulary.categories=[router, modem]"])
    assert cfg["vocabulary"]["categories"] == ["router", "modem", "electronic device"]
    assert load_config()["vocabulary"]["categories"].count("electronic device") == 1


def test_vocabulary_is_normalized():
    assert normalize_vocabulary(["  Router ", "router", "", "USB-C Hub.", "wi.fi  box"]) == [
        "router",
        "usb-c hub",
        "wi fi box",
        "electronic device",
    ]


def test_same_config_same_hash_and_changed_threshold_changes_it():
    assert config_hash(load_config()) == config_hash(load_config())
    changed = load_config(sets=["pipeline.detector.params.raw_box_threshold=0.2"])
    assert config_hash(changed) != config_hash(load_config())
    assert len(config_hash(changed)) == 12


def test_show_config_prints_the_merged_config(capsys):
    code = main(["show-config", "--set", "pipeline.detector.params.raw_box_threshold=0.2"])
    out = capsys.readouterr().out
    assert code == 0
    assert "raw_box_threshold: 0.2" in out
    assert "# config_hash:" in out


def test_show_config_bad_input_exits_2_with_one_line_error(capsys):
    code = main(["show-config", "--set", "pipeline.nope.thing=1"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.err.startswith("error: ")
    assert "Traceback" not in captured.err
