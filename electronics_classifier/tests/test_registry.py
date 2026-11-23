import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field

from electronics_classifier import registry
from electronics_classifier.config import ConfigError
from electronics_classifier.registry import KINDS, build, describe_components, register


@pytest.fixture(autouse=True)
def isolated_registry():
    snapshot = {kind: dict(names) for kind, names in registry._REGISTRY.items()}
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(snapshot)


class Dummy:
    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        threshold: float = Field(default=0.5, ge=0.0, description="A threshold.")

    def __init__(self, params, build):
        self.params = params


def test_builtins_register_with_documented_params():
    described = describe_components()
    assert set(described) == set(KINDS)
    for components in described.values():
        for name, rows in components.items():
            for field_name, _type, _default, description in rows:
                assert description, f"{name}.{field_name} is missing a Field(description=...)"


def test_registered_component_is_built_with_validated_params(build_ctx):
    register("detector", "dummy")(Dummy)
    component = build("detector", {"name": "dummy", "params": {"threshold": 0.7}}, build_ctx)
    assert isinstance(component, Dummy)
    assert component.params.threshold == 0.7


def test_unknown_name_lists_registered_names(build_ctx):
    register("detector", "dummy")(Dummy)
    with pytest.raises(ConfigError) as err:
        build("detector", {"name": "nope"}, build_ctx, path="pipeline.detector")
    message = str(err.value)
    assert "pipeline.detector.name" in message
    assert "nope" in message
    assert "dummy" in message


def test_dotted_path_plugin_loads_without_registration(build_ctx):
    component = build(
        "detector",
        {"name": "conftest:TinyDetector", "params": {"phrase": "modem"}},
        build_ctx,
    )
    assert component.params.phrase == "modem"


def test_bad_dotted_path_is_a_config_error(build_ctx):
    with pytest.raises(ConfigError, match="no_such_module_xyz"):
        build("detector", {"name": "no_such_module_xyz:Thing"}, build_ctx)
    with pytest.raises(ConfigError, match="no attribute"):
        build("detector", {"name": "conftest:DoesNotExist"}, build_ctx)


def test_bad_params_error_shows_full_config_path(build_ctx):
    register("detector", "dummy")(Dummy)
    with pytest.raises(ConfigError, match=r"pipeline\.detector\.params\.bogus"):
        build(
            "detector",
            {"name": "dummy", "params": {"bogus": 1}},
            build_ctx,
            path="pipeline.detector",
        )
    with pytest.raises(ConfigError, match=r"pipeline\.detector\.params\.threshold"):
        build(
            "detector",
            {"name": "dummy", "params": {"threshold": -1}},
            build_ctx,
            path="pipeline.detector",
        )


def test_duplicate_registration_with_a_different_class_is_rejected():
    register("detector", "dummy")(Dummy)

    class Other:
        Params = Dummy.Params

    with pytest.raises(ValueError, match="already registered"):
        register("detector", "dummy")(Other)


def test_register_rejects_unknown_kind_and_colon_names():
    with pytest.raises(ValueError, match="unknown component kind"):
        register("nonsense", "x")
    with pytest.raises(ValueError, match="':'"):
        register("detector", "a:b")


def test_import_and_load_builtin_pull_in_no_heavy_dependencies():
    code = (
        "import sys, electronics_classifier;"
        "from electronics_classifier.registry import load_builtin;"
        "load_builtin();"
        "heavy = {'torch', 'transformers', 'ultralytics', 'segment_anything', 'sam2'};"
        "loaded = heavy & set(sys.modules);"
        "assert not loaded, loaded"
    )
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "PYTHONPATH": src}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
