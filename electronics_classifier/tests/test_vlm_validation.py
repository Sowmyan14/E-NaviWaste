"""The VLM as validator: ``is_electronic`` verdict, detector-phrase hint, drop or warn."""

import json

import numpy as np
import pytest
from conftest import VALID_REPLY, FakeDetector, FakeVLMClient, det, make_image

from electronics_classifier.extractor import PHRASE_HINT, USER_TEXT, parse_reply
from electronics_classifier.pipeline import ElectronicsClassificationPipeline
from electronics_classifier.system_prompt import load_template

IMAGE = make_image("noise", 480, 640)
TWO = [det("router", 0.9, (20, 20, 220, 220)), det("laptop", 0.8, (360, 200, 560, 400))]


class OpenFakeDetector(FakeDetector):
    open_vocabulary = True  # like ram_grounding_dino: its phrases are not limited to the vocabulary


class NoSegmenter:
    def segment(self, image_rgb, boxes):
        return [None] * len(boxes)


def default_pipeline(detections, replies, sets=(), client=None):
    """The NEW default config (validate prompt, drop, category_open) with fake models.

    OCR is off because the synthetic image has no text for it to confirm a model name against.
    """
    return ElectronicsClassificationPipeline(
        detector=OpenFakeDetector(detections),
        segmenter=NoSegmenter(),
        vlm_client=client or FakeVLMClient(replies, cycle=True),
        sets=["pipeline.attribute_postprocess=[{name: normalize_unknown}, {name: category_open}]",
              *sets],
    )  # fmt: skip


def reply(is_electronic=True, **fields):
    base = {"brand": "TP-Link", "model": "unknown", "category": "router", "color": "black"}
    return json.dumps({"is_electronic": is_electronic, **base, **fields})


# --- parsing ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), ("yes", True), ("No", False), ("TRUE", True), (" no ", False)],
)
def test_the_verdict_is_read_from_bools_and_yes_no_words(value, expected):
    assert parse_reply(reply(value))["is_electronic"] is expected


@pytest.mark.parametrize("value", [None, "maybe", 3, [], {"a": 1}, ""])
def test_an_unreadable_verdict_is_treated_as_no_verdict(value):
    attributes = parse_reply(reply(value))
    assert attributes is not None and "is_electronic" not in attributes
    assert attributes["category"] == "router"


def test_a_reply_without_the_key_keeps_the_four_key_shape():
    assert parse_reply(VALID_REPLY) == {
        "brand": "TP-Link", "model": "Archer", "category": "router", "color": "black",
    }  # fmt: skip


# --- the prompt ------------------------------------------------------------------------


def test_the_validate_prompt_loads_and_asks_for_the_verdict():
    template = load_template("config/prompts/vlm_validate_prompt.txt")
    assert "{categories}" in template and "is_electronic" in template


def default_prompt_text():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return (root / "config" / "prompts" / "vlm_validate_prompt.txt").read_text(encoding="utf-8")


def test_the_prompt_tells_the_vlm_to_infer_the_brand_from_features_and_context():
    text = default_prompt_text()
    brand_rules = [line for line in text.splitlines() if line.startswith('- "brand"')]
    assert len(brand_rules) == 1
    rule = brand_rules[0].lower()
    for clue in ("logo", "distinctive features", "kind of object", "context"):
        assert clue in rule
    assert "accessories" in rule and "nearby" in rule  # scene context examples


def test_the_prompt_makes_an_unknown_brand_the_exception_not_the_default():
    text = default_prompt_text()
    assert "Give your best-supported brand even when no logo is visible" in text
    assert '"unknown" is the exception, not the default' in text
    assert "Do not name a brand from a hunch" in text  # still no invented brands
    assert "Never guess a brand" not in text  # the old blanket ban is gone


def test_the_model_number_rule_stays_strict_and_the_example_does_not_teach_unknown_brand():
    text = default_prompt_text()
    assert "Never fabricate or infer a model number" in text
    assert 'Write a "model" only if its exact name or number is printed' in text
    example = [line for line in text.splitlines() if line.startswith('{"is_electronic"')]
    assert len(example) == 1 and '"brand": "unknown"' not in example[0]
    assert 'If "model" or "color" cannot be read' in text  # the generic unknown rule skips brand


def test_the_legacy_prompts_used_by_baseline_still_forbid_guessing_a_brand():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "config" / "prompts"
    assert "Never guess a brand" in (root / "vlm_system_prompt.txt").read_text(encoding="utf-8")
    assert "Never guess a brand" in (root / "vlm_open_vocab_prompt.txt").read_text(encoding="utf-8")


def test_the_builtin_prompt_loads_from_any_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert "is_electronic" in load_template("config/prompts/vlm_validate_prompt.txt")


def test_the_default_pipeline_uses_the_validate_prompt_and_sends_the_phrase_hint(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    client = FakeVLMClient([reply()], cycle=True)
    default_pipeline([det("router")], None, client=client).run(IMAGE)
    system_prompt, _images, user_text = client.calls[0]
    assert "is_electronic" in system_prompt
    assert user_text == USER_TEXT + PHRASE_HINT.format(phrase="router")


# --- drop or warn ----------------------------------------------------------------------


def run(replies, sets=(), detections=TWO, trace=False):
    return default_pipeline(detections, replies, sets).run(IMAGE, return_trace=trace)


def test_a_not_electronic_item_is_dropped_by_default_and_recorded_in_the_trace():
    _, result, trace = run([reply(True), reply(False, category="desk")], trace=True)
    assert [i.id for i in result.items] == ["obj_1"]
    assert trace.dropped_not_electronic == ["obj_2"]
    assert trace.vlm_replies["obj_2"]  # the raw reply is kept for debugging


def test_warn_mode_keeps_the_item_and_flags_it():
    _, result = run(
        [reply(True), reply(False, category="desk")],
        sets=["pipeline.vlm.on_not_electronic=warn"],
    )
    assert len(result.items) == 2
    assert "NOT_ELECTRONIC" in result.items[1].warnings
    assert "NOT_ELECTRONIC" not in result.items[0].warnings


def test_a_missing_verdict_never_drops_an_item():
    _, result = run([VALID_REPLY])
    assert len(result.items) == 2
    assert all("NOT_ELECTRONIC" not in i.warnings for i in result.items)


def test_when_every_item_is_not_electronic_the_result_is_empty_but_valid():
    annotated, result = run([reply(False, category="poster")])
    assert result.items == []
    assert annotated.shape == IMAGE.shape and np.array_equal(annotated, IMAGE)


def test_the_verdict_never_leaks_into_the_json_attributes():
    _, result = run([reply(True)])
    dumped = result.items[0].attributes.model_dump(by_alias=True)
    assert "is_electronic" not in dumped


# --- category_mismatch with free-text phrases ------------------------------------------


@pytest.mark.parametrize(
    ("phrase", "category", "flagged"),
    [
        ("laptop computer", "laptop", False),  # shares a word: agree
        ("power strip", "power strip", False),
        ("router", "laptop", True),  # two different specific categories
        ("router", "electronic device", False),
    ],
)
def test_category_mismatch_treats_a_shared_word_as_agreement(phrase, category, flagged):
    detections = [det(phrase, 0.9, (20, 20, 220, 220))]
    _, result = run([reply(True, category=category)], detections=detections)
    assert ("CATEGORY_MISMATCH" in result.items[0].warnings) is flagged
