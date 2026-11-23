import json

import pytest
from conftest import make_prepared_item

from electronics_classifier.assembler import (
    assemble_result_payload,
    evaluate_warning_rules,
    make_draft,
    result_to_json,
    unique_in_order,
)
from electronics_classifier.config import ConfigError
from electronics_classifier.interfaces import BuildContext, ItemContext
from electronics_classifier.registry import build
from electronics_classifier.schemas import ClassificationResult, RawDetection, RunMeta

KNOWN = {"brand": "TP-Link", "model": "Archer", "category": "router", "color": "black"}


def rule(build_ctx, name, **params):
    return build("warning_rule", {"name": name, "params": params}, build_ctx)


def evaluate(build_ctx, name, attrs=None, item=None, dets=None, index=0, **params):
    item = item or make_prepared_item()
    dets = dets if dets is not None else [item.detection]
    draft = make_draft(item, attrs or KNOWN)
    ctx = ItemContext(item, index, dets, 300, 200)
    return rule(build_ctx, name, **params).evaluate(draft, ctx)


# --- each rule triggers on its condition and not otherwise ---------------------------


def test_unknown_field_flags_each_unknown_field_separately(build_ctx):
    assert evaluate(build_ctx, "unknown_field") == []
    only_model = {**KNOWN, "model": "unknown"}
    assert evaluate(build_ctx, "unknown_field", only_model) == ["MODEL_UNKNOWN"]
    everything = {**KNOWN, "brand": "unknown", "model": "unknown", "color": "unknown"}
    assert evaluate(build_ctx, "unknown_field", everything) == [
        "BRAND_UNKNOWN", "MODEL_UNKNOWN", "COLOR_UNKNOWN",
    ]  # fmt: skip
    assert evaluate(build_ctx, "unknown_field", {**KNOWN, "brand": "Unknown Brand"}) == []


@pytest.mark.parametrize("confidence,expected", [(0.39, True), (0.40, False), (0.41, False)])
def test_low_detector_confidence_is_strictly_below_the_threshold(build_ctx, confidence, expected):
    item = make_prepared_item(confidence=confidence)
    codes = evaluate(build_ctx, "low_detector_confidence", item=item, threshold=0.40)
    assert (codes == ["LOW_DETECTOR_CONFIDENCE"]) is expected


def test_low_detector_confidence_default_threshold_is_0_40(build_ctx):
    low, fine = make_prepared_item(confidence=0.3), make_prepared_item(confidence=0.5)
    assert evaluate(build_ctx, "low_detector_confidence", item=low)
    assert not evaluate(build_ctx, "low_detector_confidence", item=fine)


@pytest.mark.parametrize(
    "padded,expected",
    [
        ((0, 0, 63, 200), True),
        ((0, 0, 200, 63), True),
        ((0, 0, 64, 64), False),
        ((0, 0, 90, 80), False),
    ],
)
def test_small_crop_uses_the_shorter_padded_side(build_ctx, padded, expected):
    item = make_prepared_item(box=(10, 10, 20, 20), padded=padded)
    codes = evaluate(build_ctx, "small_crop", item=item, min_side_px=64)
    assert (codes == ["SMALL_CROP"]) is expected


def test_overlap_rule_compares_with_the_other_detections(build_ctx):
    a = make_prepared_item(box=(0, 0, 10, 10), det_id="obj_1")
    b = RawDetection(detection_id="obj_2", phrase="modem", confidence=0.8, bbox_xyxy=(0, 0, 10, 20))
    far = RawDetection(
        detection_id="obj_3", phrase="cable", confidence=0.8, bbox_xyxy=(90, 90, 99, 99)
    )
    both = [a.detection, b]  # IoU 0.5
    assert evaluate(build_ctx, "overlap", item=a, dets=both, iou=0.5) == [
        "PARTIALLY_OCCLUDED_OR_OVERLAPPING"
    ]  # at the threshold counts
    assert evaluate(build_ctx, "overlap", item=a, dets=both, iou=0.51) == []
    assert evaluate(build_ctx, "overlap", item=a, dets=[a.detection, far], iou=0.15) == []
    assert evaluate(build_ctx, "overlap", item=a, dets=[a.detection], iou=0.0) == []  # alone


def test_overlap_rule_marks_both_overlapping_items(build_ctx):
    a = make_prepared_item(box=(0, 0, 10, 10), det_id="obj_1")
    b = make_prepared_item(box=(2, 2, 12, 12), det_id="obj_2")
    dets = [a.detection, b.detection]
    assert evaluate(build_ctx, "overlap", item=a, dets=dets, index=0)
    assert evaluate(build_ctx, "overlap", item=b, dets=dets, index=1)


def test_category_oov_rule_fires_only_for_out_of_vocabulary_categories(build_ctx):
    assert evaluate(build_ctx, "category_oov") == []
    assert evaluate(build_ctx, "category_oov", {**KNOWN, "category": "toaster"}) == [
        "CATEGORY_OUT_OF_VOCAB"
    ]
    assert evaluate(build_ctx, "category_oov", {**KNOWN, "category": "unknown"})


def test_rule_params_are_validated_at_build_time(build_ctx):
    for name, params, field in [
        ("low_detector_confidence", {"threshold": 1.5}, "threshold"),
        ("small_crop", {"min_side_px": -1}, "min_side_px"),
        ("overlap", {"iou": -0.1}, "iou"),
        ("unknown_field", {"x": 1}, "x"),
    ]:
        with pytest.raises(ConfigError, match=rf"params\.{field}"):
            rule(build_ctx, name, **params)


def test_rules_run_in_order_with_duplicates_removed(build_ctx):
    class Echo:
        def __init__(self, *codes):
            self.codes = list(codes)

        def evaluate(self, draft, item_ctx):
            return self.codes

    item = make_prepared_item()
    ctx = ItemContext(item, 0, [item.detection], 300, 200)
    codes = evaluate_warning_rules(
        [Echo("B", "A"), Echo("A", "C"), Echo("B")], make_draft(item, KNOWN), ctx
    )
    assert codes == ["B", "A", "C"]
    assert evaluate_warning_rules([], make_draft(item, KNOWN), ctx) == []
    assert unique_in_order(["x", "y"], ["y", "z"], []) == ["x", "y", "z"]


# --- assembler -----------------------------------------------------------------------


META = RunMeta(config_hash="3f9a1c0b7d2e", components={"detector": "fake", "vlm_client": "fake"})


def assemble(items, attrs=None, warnings=None, **kwargs):
    return assemble_result_payload(items, attrs or {}, warnings or {}, 300, 200, META, **kwargs)


def test_alias_keys_and_meta_appear_in_the_json():
    item = make_prepared_item(box=(12, 8, 46, 31), padded=(4, 2, 60, 44))
    result = assemble([item], {"obj_1": KNOWN}, {"obj_1": ["MODEL_UNKNOWN"]})
    payload = json.loads(result_to_json(result))
    assert payload["image_width"] == 300 and payload["image_height"] == 200
    [out] = payload["items"]
    assert out["attributes"]["@context"] == "https://schema.org"
    assert out["attributes"]["@type"] == "Product"
    assert "context" not in out["attributes"] and "type" not in out["attributes"]
    assert out["bbox_xyxy"] == [12, 8, 46, 31]  # the unpadded detector box
    assert out["id"] == "obj_1" and out["detector_phrase"] == "router"
    assert out["detector_confidence"] == 0.9
    assert out["warnings"] == ["MODEL_UNKNOWN"]
    assert payload["meta"] == {
        "config_hash": "3f9a1c0b7d2e",
        "components": {"detector": "fake", "vlm_client": "fake"},
    }
    assert payload["annotated_image_format"] == "png"


def test_the_plans_example_output_is_a_valid_result():
    example = {
        "image_width": 1280,
        "image_height": 960,
        "items": [
            {
                "id": "obj_1",
                "bbox_xyxy": [120, 85, 460, 310],
                "detector_phrase": "router",
                "detector_confidence": 0.81,
                "attributes": {
                    "@context": "https://schema.org",
                    "@type": "Product",
                    "brand": "TP-Link",
                    "model": "unknown",
                    "category": "router",
                    "color": "black",
                },
                "warnings": ["MODEL_UNKNOWN"],
            }
        ],
        "annotated_image_format": "png",
        "meta": {
            "config_hash": "3f9a1c0b7d2e",
            "components": {"detector": "grounding_dino", "vlm_client": "transformers"},
        },
    }
    result = ClassificationResult.model_validate(example)
    assert json.loads(result_to_json(result)) == example


def test_result_round_trips_through_json():
    result = assemble([make_prepared_item()], {"obj_1": KNOWN})
    assert ClassificationResult.model_validate_json(result_to_json(result)) == result


def test_a22_unicode_stays_unicode_in_the_json():
    brand = "Samsung 삼성 電子 🙂"
    result = assemble([make_prepared_item()], {"obj_1": {**KNOWN, "brand": brand}})
    text = result_to_json(result)
    assert brand in text and "\\u" not in text
    assert text.encode("utf-8").decode("utf-8") == text
    assert json.loads(text)["items"][0]["attributes"]["brand"] == brand


def test_items_keep_the_given_order_and_each_gets_its_own_data():
    a = make_prepared_item(box=(0, 0, 20, 20), det_id="obj_1")
    b = make_prepared_item(box=(30, 30, 60, 60), det_id="obj_2", phrase="modem", confidence=0.5)
    result = assemble(
        [a, b],
        {"obj_1": KNOWN, "obj_2": {**KNOWN, "brand": "Netgear", "category": "modem"}},
        {"obj_2": ["LOW_DETECTOR_CONFIDENCE"]},
    )
    assert [i.id for i in result.items] == ["obj_1", "obj_2"]
    assert [i.attributes.brand for i in result.items] == ["TP-Link", "Netgear"]
    assert [i.warnings for i in result.items] == [[], ["LOW_DETECTOR_CONFIDENCE"]]
    assert result.items[1].detector_phrase == "modem" and result.items[1].detector_confidence == 0.5


def test_warnings_are_deduplicated_in_stable_order():
    noisy = {"obj_1": ["B", "A", "B", "C", "A"]}
    result = assemble([make_prepared_item()], {"obj_1": KNOWN}, noisy)
    assert result.items[0].warnings == ["B", "A", "C"]


def test_missing_attributes_become_unknown_and_non_strings_are_cast():
    a = make_prepared_item(det_id="obj_1")
    b = make_prepared_item(det_id="obj_2", box=(60, 60, 90, 90))
    result = assemble([a, b], {"obj_2": {"brand": 7, "model": None}})
    unknown = result.items[0].attributes
    assert (unknown.brand, unknown.model, unknown.category, unknown.color) == ("unknown",) * 4
    assert result.items[1].attributes.brand == "7"


def test_no_items_gives_an_empty_result_and_format_is_configurable():
    result = assemble([], annotated_image_format="jpeg")
    assert result.items == [] and result.annotated_image_format == "jpeg"
    assert json.loads(result_to_json(result))["items"] == []


def test_assembling_needs_detection_ids():
    with pytest.raises(ValueError, match="ids must be assigned"):
        assemble([make_prepared_item(det_id=None)])
    with pytest.raises(ValueError, match="ids must be assigned"):
        make_draft(make_prepared_item(det_id=None), KNOWN)


def test_rules_see_the_whole_set_when_run_through_the_full_chain(build_ctx):
    """Post-process -> draft -> rules -> assemble, as the pipeline will chain them."""
    from electronics_classifier.extractor import apply_postprocessors

    items = [
        make_prepared_item(box=(0, 0, 100, 100), det_id="obj_1", confidence=0.3),
        make_prepared_item(box=(10, 10, 110, 110), det_id="obj_2"),
    ]
    dets = [i.detection for i in items]
    post = [
        build("attribute_postprocess", {"name": "normalize_unknown"}, build_ctx),
        build("attribute_postprocess", {"name": "category_vocab"}, build_ctx),
    ]
    rules = [
        build("warning_rule", {"name": n}, build_ctx)
        for n in ("unknown_field", "low_detector_confidence", "overlap")
    ]
    raw = {"brand": "N/A", "model": "X1", "category": "Toaster", "color": "black"}
    attrs_by_id, warnings_by_id = {}, {}
    for idx, item in enumerate(items):
        ctx = ItemContext(item, idx, dets, 300, 200)
        attrs, extra, _ = apply_postprocessors(post, raw, ctx)
        warnings_by_id[item.detection.detection_id] = extra + evaluate_warning_rules(
            rules, make_draft(item, attrs), ctx
        )
        attrs_by_id[item.detection.detection_id] = attrs
    result = assemble(items, attrs_by_id, warnings_by_id)
    assert result.items[0].warnings == [
        "CATEGORY_OUT_OF_VOCAB", "BRAND_UNKNOWN", "LOW_DETECTOR_CONFIDENCE",
        "PARTIALLY_OCCLUDED_OR_OVERLAPPING",
    ]  # fmt: skip
    assert result.items[1].warnings == [
        "CATEGORY_OUT_OF_VOCAB", "BRAND_UNKNOWN", "PARTIALLY_OCCLUDED_OR_OVERLAPPING",
    ]  # fmt: skip
    assert result.items[0].attributes.category == "electronic device"
    assert BuildContext  # keep the import used by fixtures' type
