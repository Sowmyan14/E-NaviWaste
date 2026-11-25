"""The RAM++ + Grounding DINO detector (``ram_grounding_dino``), with no models loaded.

``StubRam`` swaps the two model calls (``_tag_image`` for RAM, ``_run`` for DINO) so the tag
cleaning, prompt building, chunking and label mapping run for real.
"""

import builtins
import os

import numpy as np
import pytest
from conftest import VALID_REPLY, make_image, make_pipeline

from electronics_classifier.components.detection_filters import sanitize_detections
from electronics_classifier.components.detectors.ram_grounding_dino import (
    DEFAULT_DROP_TAGS,
    RamGroundingDinoDetector,
    clean_tags,
)
from electronics_classifier.config import ConfigError
from electronics_classifier.interfaces import BuildContext
from electronics_classifier.registry import build
from electronics_classifier.schemas import RawDetection

VOCAB = ("router", "laptop", "electronic device")
IMAGE = make_image("solid", 200, 200)


class StubRam(RamGroundingDinoDetector):
    """Real prompt logic, fake models. ``tags`` may be a string or an Exception to raise."""

    def __init__(self, tags="", categories=VOCAB, **params):
        self.params = RamGroundingDinoDetector.Params(ram_checkpoint="unused.pth", **params)
        self.vocabulary = tuple(categories)
        self.device = "cpu"
        self._count_tokens = lambda text: len(text.split())
        self.tags = tags
        self.prompts: list[str] = []
        self.tag_calls = 0

    def _tag_image(self, image_rgb):
        self.tag_calls += 1
        if isinstance(self.tags, Exception):
            raise self.tags
        return self.tags

    def _run(self, image_rgb, prompt):
        self.prompts.append(prompt)
        phrases = prompt.removesuffix(" .").split(" . ")
        return [(p, 0.8, [10 + 5 * i, 10, 60 + 5 * i, 60]) for i, p in enumerate(phrases)]


def phrases_of(detector):
    return [d.phrase for d in detector.predict(IMAGE)]


# --- tag cleaning ----------------------------------------------------------------------


def test_clean_tags_splits_lowercases_and_removes_prompt_separators():
    raw = "Laptop Computer | kettle. | smart, speaker |  Router "
    assert clean_tags(raw) == ["laptop computer", "kettle", "smart speaker", "router"]


def test_clean_tags_drops_duplicates_stoplisted_and_overlong_tags():
    raw = "router | ROUTER | black | living room | a very long scene description tag | kettle"
    assert clean_tags(raw, drop=["black", "living room"]) == ["router", "kettle"]


def test_clean_tags_caps_the_count_and_survives_non_string_input():
    assert clean_tags("a | b | c | d", max_tags=2) == ["a", "b"]
    assert clean_tags(None) == [] and clean_tags(42) == [] and clean_tags("") == []
    assert clean_tags(" | || | ") == []


def test_the_default_stoplist_covers_scene_words_but_no_device():
    assert {"black", "floor", "living room", "table"} <= set(DEFAULT_DROP_TAGS)
    for device in ("laptop", "router", "television", "kettle", "toaster", "monitor"):
        assert device not in DEFAULT_DROP_TAGS


# --- prompt and detections -------------------------------------------------------------


def test_tags_come_first_then_the_vocabulary_without_repeats():
    detector = StubRam("kettle | laptop | floor")
    assert phrases_of(detector) == ["kettle", "laptop", "router", "electronic device"]
    assert detector.prompts == ["kettle . laptop . router . electronic device ."]


def test_phrases_outside_the_vocabulary_are_kept_and_the_detector_says_so():
    detector = StubRam("kettle")
    assert detector.open_vocabulary is True
    assert "kettle" in phrases_of(detector)


def test_vocabulary_can_be_left_out_of_the_prompt():
    detector = StubRam("kettle | toaster", include_vocabulary=False)
    assert phrases_of(detector) == ["kettle", "toaster"]


def test_no_tags_and_no_vocabulary_means_no_model_call_and_no_detections():
    detector = StubRam("floor | wall", include_vocabulary=False)
    assert detector.predict(IMAGE) == [] and detector.prompts == []


def test_empty_tags_fall_back_to_the_vocabulary_alone():
    assert phrases_of(StubRam("")) == list(VOCAB)


def test_a_failing_tagger_falls_back_to_the_vocabulary_instead_of_raising():
    assert phrases_of(StubRam(RuntimeError("mps out of memory"))) == list(VOCAB)


def test_a_long_tag_list_is_chunked_without_losing_a_tag():
    tags = [f"gadget{i}" for i in range(20)]
    detector = StubRam(
        " | ".join(tags), include_vocabulary=False, max_prompt_tokens=12, max_tags=20
    )
    found = phrases_of(detector)
    assert len(detector.prompts) > 1
    assert found == tags  # every tag, in order, exactly once


def test_confidence_is_clamped_and_boxes_are_integers():
    detector = StubRam("kettle", include_vocabulary=False)
    detector._run = lambda image, prompt: [("kettle", 1.7, [1.4, 2.6, 30.2, 40.9])]
    [found] = detector.predict(IMAGE)
    assert found.confidence == 1.0 and found.bbox_xyxy == (1, 3, 30, 41)


def test_a_label_that_matches_no_tag_is_dropped():
    detector = StubRam("kettle", include_vocabulary=False)
    detector._run = lambda image, prompt: [("something else", 0.9, [1, 1, 20, 20])]
    assert detector.predict(IMAGE) == []


# --- pipeline integration --------------------------------------------------------------


def test_sanitize_without_a_vocabulary_keeps_free_text_phrases():
    dets = [RawDetection(phrase="kettle", confidence=0.9, bbox_xyxy=(1, 1, 50, 50))]
    assert sanitize_detections(dets, 100, 100, VOCAB) == []
    assert [d.phrase for d in sanitize_detections(dets, 100, 100, None)] == ["kettle"]


def test_the_pipeline_keeps_ram_only_phrases_but_still_drops_them_for_a_closed_detector():
    class Closed(StubRam):
        open_vocabulary = False

    reply = VALID_REPLY.replace("router", "kettle")
    image = make_image("noise", 300, 300)
    for detector, expected in ((StubRam("kettle", include_vocabulary=False), 1),
                               (Closed("kettle", include_vocabulary=False), 0)):  # fmt: skip
        pipeline = make_pipeline(detector=detector, replies=[reply])
        _, result = pipeline.run(image)
        assert len(result.items) == expected
        if expected:
            assert result.items[0].detector_phrase == "kettle"


# --- construction ----------------------------------------------------------------------


def test_a_missing_ram_checkpoint_is_a_config_error_naming_the_path(build_ctx):
    spec = {"name": "ram_grounding_dino", "params": {"ram_checkpoint": "nowhere/ram.pth"}}
    with pytest.raises(ConfigError, match=r"pipeline\.detector\.params\.ram_checkpoint"):
        build("detector", spec, build_ctx, "pipeline.detector")


def test_the_checkpoint_is_required():
    with pytest.raises(ConfigError, match=r"pipeline\.detector\.params\.ram_checkpoint"):
        build(
            "detector",
            {"name": "ram_grounding_dino", "params": {}},
            BuildContext(VOCAB, "cpu", 0),
            "pipeline.detector",
        )


def test_missing_packages_give_an_install_hint(tmp_path, build_ctx, monkeypatch):
    checkpoint = tmp_path / "ram.pth"
    checkpoint.write_bytes(b"")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("torch", "ram", "transformers"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    spec = {"name": "ram_grounding_dino", "params": {"ram_checkpoint": str(checkpoint)}}
    with pytest.raises(ImportError, match=r"electronics-classifier\[grounded\]"):
        build("detector", spec, build_ctx)


@pytest.mark.parametrize(
    ("device", "fallback", "expected"),
    [("mps", True, "1"), ("mps", False, None), ("cpu", True, None)],
)
def test_the_mps_cpu_fallback_is_set_only_for_apple_gpus(
    tmp_path, monkeypatch, device, fallback, expected
):
    class NoModels(RamGroundingDinoDetector):
        def _load_ram(self):
            pass

        def _load_backend(self):
            pass

    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    checkpoint = tmp_path / "ram.pth"
    checkpoint.write_bytes(b"")
    params = NoModels.Params(ram_checkpoint=str(checkpoint), mps_fallback=fallback)
    NoModels(params, BuildContext(VOCAB, device, 0))
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == expected
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)


def test_the_detector_is_registered_and_listed_with_its_parameters():
    from electronics_classifier.registry import describe_components

    rows = describe_components()["detector"]["ram_grounding_dino"]
    by_name = {name: (default, text) for name, _type, default, text in rows}
    assert by_name["ram_checkpoint"][0] == "required"
    assert all(text for _default, text in by_name.values())  # every parameter is described
    assert np.isclose(float(by_name["raw_box_threshold"][0]), 0.25)
