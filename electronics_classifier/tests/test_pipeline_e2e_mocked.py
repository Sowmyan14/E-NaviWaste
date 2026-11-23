import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import FakeVLMClient, det, make_image, make_pipeline

from electronics_classifier.assembler import result_to_json
from electronics_classifier.cli import main
from electronics_classifier.config import ConfigError, config_hash, load_config
from electronics_classifier.pipeline import ElectronicsClassificationPipeline
from electronics_classifier.registry import registered_names
from electronics_classifier.schemas import ClassificationResult
from electronics_classifier.trace import PipelineTrace

IMAGE = make_image("noise", 480, 640)
FAKE_BOX_QUARTER = (
    "pipeline.detector={name: fake, params: {detections: [{box_rel: [0.25, 0.25, 0.5, 0.5]}]}}"
)
TWO_FAKE_BOXES = (
    "pipeline.detector={name: fake, params: {detections: ["
    "{phrase: router, box_rel: [0.1, 0.1, 0.4, 0.4]}, "
    "{phrase: modem, box_rel: [0.6, 0.6, 0.9, 0.9]}]}}"
)
FAKE_ACME = (
    "pipeline.vlm.client={name: fake, params: "
    "{response: {brand: Acme, model: X1, category: router, color: red}}}"
)
FAKE_LAPTOP = (
    "pipeline.detector={name: fake, params: "
    "{detections: [{phrase: laptop, box_rel: [0.1, 0.1, 0.9, 0.9]}]}}"
)
SAMPLE = Path(__file__).parent / "fixtures" / "sample.jpg"


def two_devices():
    return [
        det("router", 0.9, (60, 80, 300, 260)),
        det("modem", 0.8, (360, 120, 590, 220)),
    ]


# --- full run ------------------------------------------------------------------------


def test_full_run_produces_matching_image_and_json():
    pipeline = make_pipeline(two_devices())
    annotated, result = pipeline.run(IMAGE)
    assert annotated.shape == IMAGE.shape and annotated.dtype == np.uint8
    assert (annotated != IMAGE).any()
    assert isinstance(result, ClassificationResult)
    assert (result.image_width, result.image_height) == (640, 480)
    assert [i.id for i in result.items] == ["obj_1", "obj_2"]  # ordered by (y1, x1)
    first = result.items[0]
    assert first.bbox_xyxy == [60, 80, 300, 260] and first.detector_phrase == "router"
    assert first.attributes.brand == "TP-Link" and first.attributes.category == "router"
    assert first.warnings == []
    assert result.meta.config_hash == pipeline.config_hash


def test_every_label_matches_one_json_item():
    from electronics_classifier.renderer import format_label

    pipeline = make_pipeline(two_devices(), ['{"brand": "A", "model": "1", "category": "router"}'])
    _, result = pipeline.run(IMAGE)
    template = pipeline.style.label_template
    banners = [format_label(item, template) for item in result.items]
    assert banners[0].startswith("! obj_1 router | A 1 | unknown")  # color unknown -> warning
    for item, banner in zip(result.items, banners, strict=True):
        assert item.id in banner and item.attributes.category in banner


def test_zero_detections_give_an_unmodified_copy_and_no_items():
    annotated, result = make_pipeline([]).run(IMAGE)
    assert result.items == [] and (annotated == IMAGE).all() and annotated is not IMAGE
    assert result.meta is not None


def test_invalid_input_raises_value_error():
    pipeline = make_pipeline(two_devices())
    for bad in (np.zeros((9, 9), np.uint8), np.zeros((9, 9, 4), np.uint8),
                np.zeros((9, 9, 3), np.float32), "nope", None):  # fmt: skip
        with pytest.raises(ValueError, match="RGB uint8"):
            pipeline.run(bad)


def test_runs_are_deterministic():
    pipeline = make_pipeline(two_devices())
    first = pipeline.run(IMAGE)
    second = pipeline.run(IMAGE)
    assert result_to_json(first[1]) == result_to_json(second[1])
    assert (first[0] == second[0]).all()


def test_boxes_are_in_original_pixels_after_a_resize():
    big = make_image("gradient", 2000, 3000)  # resized to 1280 wide before detection
    sets = [FAKE_BOX_QUARTER]
    pipeline = ElectronicsClassificationPipeline(
        sets=sets, vlm_client=FakeVLMClient([json_reply()], cycle=True)
    )
    _, result = pipeline.run(big)
    x1, y1, x2, y2 = result.items[0].bbox_xyxy
    assert abs(x1 - 750) <= 3 and abs(y1 - 500) <= 3 and abs(x2 - 1500) <= 3 and abs(y2 - 1000) <= 3


def json_reply(**overrides):
    return json.dumps(
        {"brand": "Acme", "model": "X1", "category": "router", "color": "black", **overrides}
    )


# --- the swap test: behaviour changes through config only ----------------------------


def test_swapping_detector_filters_preprocess_and_vlm_by_config_alone():
    base = [TWO_FAKE_BOXES, FAKE_ACME]
    baseline = ElectronicsClassificationPipeline(sets=base)
    _, result = baseline.run(IMAGE)
    assert len(result.items) == 2 and result.items[0].attributes.brand == "Acme"
    assert result.meta.components["detector"] == "fake"
    assert result.meta.components["vlm_client"] == "fake"

    # a different filter list: keep only one detection
    one = ElectronicsClassificationPipeline(
        sets=[*base, "pipeline.detection_filters=[{name: top_k, params: {k: 1}}]"]
    )
    assert len(one.run(IMAGE)[1].items) == 1

    # a different detector (new phrase) and a different VLM reply
    other = ElectronicsClassificationPipeline(
        sets=[
            FAKE_LAPTOP,
            "pipeline.vlm.client={name: fake, params: {response: {brand: Zed, category: laptop}}}",
        ]
    )
    [item] = other.run(IMAGE)[1].items
    assert (item.detector_phrase, item.attributes.brand) == ("laptop", "Zed")

    # a different pre-processing list
    no_resize = ElectronicsClassificationPipeline(
        sets=[*base, "pipeline.image_preprocess=[{name: ensure_rgb_uint8}]"]
    )
    assert no_resize.components["image_preprocess"] == ["ensure_rgb_uint8"]
    hashes = {baseline.config_hash, one.config_hash, other.config_hash, no_resize.config_hash}
    assert len(hashes) == 4


def test_override_files_and_presets_swap_components(tmp_path):
    override = tmp_path / "o.yaml"
    override.write_text(yaml.safe_dump({"pipeline": {"detection_filters": [
        {"name": "top_k", "params": {"k": 1}}]}}))  # fmt: skip
    sets = ["pipeline.detector.name=fake", "pipeline.vlm.client.name=fake"]
    pipeline = ElectronicsClassificationPipeline(overrides=[override], sets=sets)
    assert pipeline.components["detection_filter"] == ["top_k"]
    fast = ElectronicsClassificationPipeline(presets=["fast"], sets=sets)
    assert fast.config["pipeline"]["image_preprocess"][1]["params"]["max_side"] == 960
    assert fast.config["pipeline"]["detection_filters"][-1]["params"]["k"] == 10
    assert fast.config["pipeline"]["detector"]["name"] == "fake"


def test_fast_preset_values():
    pipe = load_config(presets=["fast"])["pipeline"]
    assert pipe["vlm"]["client"]["name"] == "transformers"
    assert pipe["vlm"]["client"]["params"] == {
        "model_name": "Qwen/Qwen2-VL-2B-Instruct",
        "temperature": 0.0,
        "max_new_tokens": 128,
    }
    assert pipe["segmenter"]["name"] == "none"
    assert pipe["detector"]["params"]["model_id"] == "IDEA-Research/grounding-dino-tiny"


# --- injection -----------------------------------------------------------------------


def test_every_component_kind_can_be_injected():
    class Recorder:
        def __init__(self):
            self.calls = 0

    class Pre(Recorder):
        def apply(self, image, state):
            self.calls += 1
            return image

        def map_boxes_back(self, boxes, state):
            return np.asarray(boxes, dtype=float)

    class Keep(Recorder):
        def apply(self, dets, ctx):
            self.calls += 1
            return dets

    class Crop(Recorder):
        def apply(self, crop):
            self.calls += 1
            return crop

    class Post(Recorder):
        def apply(self, attrs, ctx):
            self.calls += 1
            return {**attrs, "brand": "Injected"}, ["CUSTOM_POST"]

    class Rule(Recorder):
        def evaluate(self, draft, ctx):
            self.calls += 1
            return ["CUSTOM_RULE"]

    class Seg(Recorder):
        def segment(self, image, boxes):
            self.calls += 1
            return [None] * len(boxes)

    class Gate(Recorder):
        def should_segment(self, idx, dets):
            self.calls += 1
            return True

    parts = dict(
        image_preprocess=[Pre()], detection_filters=[Keep()], crop_transforms=[Crop()],
        attribute_postprocess=[Post()], warning_rules=[Rule()], segmenter=Seg(),
        segmentation_gate=Gate(),
    )  # fmt: skip
    pipeline = make_pipeline(two_devices(), **parts)
    _, result = pipeline.run(IMAGE)
    for group in parts.values():
        for part in group if isinstance(group, list) else [group]:
            assert part.calls > 0
    item = result.items[0]
    assert item.attributes.brand == "Injected"
    assert item.warnings[:3] == ["SEGMENTATION_FAILED", "CUSTOM_POST", "CUSTOM_RULE"]
    assert pipeline.components["detector"].startswith("injected:")
    assert pipeline.components["warning_rule"] == ["injected:Rule"]


def test_unknown_component_argument_is_rejected():
    with pytest.raises(TypeError, match="unknown component argument"):
        ElectronicsClassificationPipeline(detektor=object())


def test_from_config_accepts_a_resolved_config():
    resolved = load_config(sets=["rendering.output_format=jpeg"])
    pipeline = ElectronicsClassificationPipeline.from_config(
        resolved,
        detector=make_pipeline(two_devices()).detector,
        vlm_client=FakeVLMClient([json_reply()], cycle=True),
    )
    assert pipeline.run(IMAGE)[1].annotated_image_format == "jpeg"
    assert pipeline.config_hash == config_hash(resolved)


# --- trace ---------------------------------------------------------------------------


def test_trace_exposes_every_stage_and_each_filters_effect():
    pipeline = make_pipeline(two_devices() + [det("router", 0.1, (400, 300, 500, 400))])
    outputs = pipeline.run(IMAGE, return_trace=True)
    assert len(outputs) == 3
    trace = outputs[2]
    assert isinstance(trace, PipelineTrace)
    assert trace.preprocessed_image is not None
    assert len(trace.raw_detections) == 3 and len(trace.mapped_detections) == 3
    names = [name for name, _ in trace.filter_steps]
    assert names == ["score_threshold", "min_size", "min_area_ratio", "nms", "top_k"]
    counts = [len(dets) for _, dets in trace.filter_steps]
    assert counts == [2, 2, 2, 2, 2]  # score_threshold removed the 0.1 detection
    assert [d.detection_id for d in trace.detections] == ["obj_1", "obj_2"]
    assert len(trace.prepared_items) == 2
    assert trace.vlm_replies["obj_1"] and trace.attributes_before["obj_1"]["brand"] == "TP-Link"
    assert len(trace.attributes_after["obj_1"]) == 2  # one snapshot per post-processor


def test_no_trace_unless_requested():
    assert len(make_pipeline(two_devices()).run(IMAGE)) == 2


# --- build-time failures -------------------------------------------------------------


def test_bad_label_template_fails_at_build_time():
    with pytest.raises(ConfigError, match=r"rendering\.label_template"):
        make_pipeline(sets=["rendering.label_template={nope}"])


def test_missing_prompt_file_or_placeholder_fails_at_build_time(tmp_path):
    with pytest.raises(ConfigError, match=r"pipeline\.vlm\.system_prompt_file"):
        make_pipeline(sets=[f"pipeline.vlm.system_prompt_file={tmp_path / 'nope.txt'}"])
    bad = tmp_path / "bad.txt"
    bad.write_text("no placeholder")
    with pytest.raises(ConfigError, match=r"pipeline\.vlm\.system_prompt_file.*categories"):
        make_pipeline(sets=[f"pipeline.vlm.system_prompt_file={bad}"])


def test_custom_prompt_file_is_used(tmp_path):
    prompt = tmp_path / "p.txt"
    prompt.write_text("Pick from {categories}. Reply as JSON.")
    pipeline = make_pipeline(two_devices(), sets=[f"pipeline.vlm.system_prompt_file={prompt}"])
    pipeline.run(IMAGE)
    sent = pipeline.extractor.client.calls[0][0]
    assert sent.startswith('Pick from "router", "modem"') and sent.endswith("Reply as JSON.")


def test_accurate_sam_preset_fails_fast_without_the_checkpoint():
    with pytest.raises(ConfigError, match=r"pipeline\.segmenter\.params\.checkpoint"):
        make_pipeline(presets=["accurate_sam"])


def test_accurate_sam_preset_values():
    cfg = load_config(presets=["accurate_sam"])
    assert cfg["pipeline"]["segmenter"]["name"] == "sam"
    assert cfg["pipeline"]["segmenter"]["params"]["model_type"] == "vit_b"
    assert cfg["pipeline"]["crop"]["padding_ratio"] == 0.25
    assert cfg["pipeline"]["segmentation_gate"]["name"] == "overlap"


def test_segmentation_runs_through_the_pipeline_and_failures_are_flagged():
    class FailingSegmenter:
        def segment(self, image, boxes):
            return [None] * len(boxes)

    pipeline = make_pipeline(
        two_devices(),
        segmenter=FailingSegmenter(),
        segmentation_gate=make_pipeline().segmentation_gate,  # config-built overlap gate
    )
    _, result = pipeline.run(IMAGE)
    assert all("SEGMENTATION_FAILED" not in i.warnings for i in result.items)  # no overlaps

    # IoU about 0.19: above the overlap gate (0.15), below NMS (0.5), so both items survive
    overlapping = [det("router", 0.9, (60, 80, 300, 260)), det("modem", 0.8, (200, 120, 440, 300))]
    _, result = make_pipeline(
        overlapping, segmenter=FailingSegmenter(), segmentation_gate=pipeline.segmentation_gate
    ).run(IMAGE)
    assert [i.warnings[0] for i in result.items] == ["SEGMENTATION_FAILED"] * 2


def test_masks_from_a_segmenter_are_drawn_and_cropped():
    class BoxMask:
        def segment(self, image, boxes):
            out = []
            for x1, y1, x2, y2 in boxes.astype(int):
                mask = np.zeros(image.shape[:2], bool)
                mask[y1:y2, x1:x2] = True
                out.append(mask)
            return out

    gate = make_pipeline().segmentation_gate
    pipeline = make_pipeline(two_devices(), segmenter=BoxMask(), segmentation_gate=gate)
    # a gate that approves everything: swap by config instead
    pipeline = ElectronicsClassificationPipeline(
        sets=["pipeline.segmentation_gate={name: always}"],
        detector=pipeline.detector,
        vlm_client=FakeVLMClient([json_reply()], cycle=True),
        segmenter=BoxMask(),
    )
    flat = np.full((480, 640, 3), 100, dtype=np.uint8)
    annotated, result = pipeline.run(flat, return_trace=False)
    assert (annotated[150, 150] != 100).any()  # blended mask inside the first box
    assert all("SEGMENTATION_FAILED" not in i.warnings for i in result.items)


# --- CLI smoke tests -----------------------------------------------------------------


def test_sample_fixture_exists_and_is_a_jpeg():
    assert SAMPLE.is_file() and SAMPLE.read_bytes()[:2] == b"\xff\xd8"


def test_cli_smoke_run_with_fakes_writes_everything(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EC_USE_FAKES", "1")
    out = tmp_path / "out"
    code = main(["run", str(SAMPLE), "--out-dir", str(out), "--save-intermediates"])
    assert code == 0
    written = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    for expected in ("annotated.png", "result.json", "resolved_config.yaml",
                     "intermediates/00_preprocessed.png", "intermediates/01_raw_detections.png",
                     "intermediates/02_filter_4_nms.png", "intermediates/crop_obj_1.png",
                     "intermediates/vlm_obj_1.txt", "intermediates/attributes.json"):  # fmt: skip
        assert expected in written, expected
    payload = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert payload["meta"]["components"]["detector"] == "fake"
    assert payload["meta"]["config_hash"] == config_hash(
        yaml.safe_load((out / "resolved_config.yaml").read_text())
    )
    assert "closer photo" in capsys.readouterr().out  # brand/model unknown -> tip


def test_cli_run_without_intermediates_writes_only_the_three_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("EC_USE_FAKES", "1")
    out = tmp_path / "out"
    args = ["run", str(SAMPLE), "--out-dir", str(out), "--set", "rendering.output_format=jpeg"]
    assert main(args) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["annotated.jpeg", "resolved_config.yaml", "result.json"]


def test_cli_module_entry_point_runs(tmp_path):
    env = {**os.environ, "EC_USE_FAKES": "1", "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    done = subprocess.run(
        [sys.executable, "-m", "electronics_classifier.cli", "run", str(SAMPLE),
         "--out-dir", str(tmp_path / "o")],  # fmt: skip
        capture_output=True, text=True, env=env,
    )
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "o" / "result.json").is_file()


def test_list_components_lists_every_builtin(capsys):
    assert main(["list-components"]) == 0
    out = capsys.readouterr().out
    from electronics_classifier.registry import KINDS

    for kind in KINDS:
        assert kind in out
        for name in registered_names(kind):
            assert f"  {name}\n" in out
    for must in ("grounding_dino", "fake", "sam", "sam2", "transformers", "openai_compatible",
                 "category_oov", "category_mismatch"):
        assert must in out
