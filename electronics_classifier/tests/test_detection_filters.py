import logging
import time

import numpy as np
import pytest
from conftest import make_image

from electronics_classifier.components.detection_filters import (
    assign_ids,
    sanitize_detections,
)
from electronics_classifier.components.detectors.grounding_dino import (
    GroundingDinoDetector,
    build_prompt,
    chunk_vocabulary,
    map_label,
)
from electronics_classifier.config import normalize_vocabulary
from electronics_classifier.interfaces import BuildContext, FilterContext
from electronics_classifier.registry import build
from electronics_classifier.schemas import RawDetection

CTX = FilterContext(image_w=100, image_h=100)


def det(phrase="router", conf=0.9, box=(10, 10, 50, 50)):
    return RawDetection(phrase=phrase, confidence=conf, bbox_xyxy=box)


def make_filter(build_ctx, name, **params):
    return build("detection_filter", {"name": name, "params": params}, build_ctx)


def run_chain(build_ctx, specs, dets):
    for name, params in specs:
        dets = make_filter(build_ctx, name, **params).apply(dets, CTX)
    return dets


# --- each filter alone ---------------------------------------------------------------


def test_score_threshold(build_ctx):
    dets = [det(conf=0.2), det(conf=0.25, box=(0, 0, 40, 40)), det(conf=0.9, box=(60, 60, 90, 90))]
    kept = make_filter(build_ctx, "score_threshold", min_score=0.25).apply(dets, CTX)
    assert [d.confidence for d in kept] == [0.25, 0.9]


def test_min_size(build_ctx):
    dets = [det(box=(0, 0, 31, 100)), det(box=(0, 0, 100, 31)), det(box=(0, 0, 32, 32))]
    kept = make_filter(build_ctx, "min_size", min_width_px=32, min_height_px=32).apply(dets, CTX)
    assert [d.bbox_xyxy for d in kept] == [(0, 0, 32, 32)]


def test_min_area_ratio(build_ctx):
    dets = [det(box=(0, 0, 10, 10)), det(box=(0, 0, 20, 20))]  # 1% and 4% of 100x100
    kept = make_filter(build_ctx, "min_area_ratio", min_ratio=0.02).apply(dets, CTX)
    assert [d.bbox_xyxy for d in kept] == [(0, 0, 20, 20)]


def test_nms_class_agnostic_versus_per_class(build_ctx):
    pair = [det("router", 0.9, (0, 0, 50, 50)), det("modem", 0.8, (2, 2, 50, 50))]
    agnostic = make_filter(build_ctx, "nms", iou_threshold=0.5, class_agnostic=True)
    per_class = make_filter(build_ctx, "nms", iou_threshold=0.5, class_agnostic=False)
    assert [d.phrase for d in agnostic.apply(pair, CTX)] == ["router"]
    assert len(per_class.apply(pair, CTX)) == 2


def test_nms_keeps_distinct_boxes_and_the_more_confident_duplicate(build_ctx):
    far = det("router", 0.6, (60, 60, 95, 95))
    low, high = det(conf=0.5), det(conf=0.95)
    kept = make_filter(build_ctx, "nms").apply([low, far, high], CTX)
    assert sorted(d.confidence for d in kept) == [0.6, 0.95]


def test_top_k_keeps_the_most_confident(build_ctx):
    dets = [det(conf=c, box=(i * 10, 0, i * 10 + 9, 9)) for i, c in enumerate([0.3, 0.9, 0.5])]
    kept = make_filter(build_ctx, "top_k", k=2).apply(dets, CTX)
    assert [d.confidence for d in kept] == [0.9, 0.5]
    assert make_filter(build_ctx, "top_k", k=0).apply(dets, CTX) == []


def test_filters_handle_empty_and_single_inputs(build_ctx):
    for name in ("score_threshold", "min_size", "min_area_ratio", "nms", "top_k"):
        assert make_filter(build_ctx, name).apply([], CTX) == []
    assert len(make_filter(build_ctx, "nms").apply([det()], CTX)) == 1


def test_filters_never_mutate_inputs(build_ctx):
    dets = [det(conf=0.9), det(conf=0.8, box=(12, 12, 50, 50)), det(conf=0.1, box=(70, 70, 75, 75))]
    snapshot = [d.model_copy(deep=True) for d in dets]
    for name in ("score_threshold", "min_size", "min_area_ratio", "nms", "top_k"):
        make_filter(build_ctx, name).apply(dets, CTX)
        assert dets == snapshot


# --- ordering ------------------------------------------------------------------------


def test_filter_order_matters(build_ctx):
    # A is small but confident and overlaps the bigger B; NMS keeps A, min_size drops A.
    a = det("router", 0.9, (10, 10, 30, 30))
    b = det("router", 0.5, (10, 10, 35, 35))
    nms = ("nms", {"iou_threshold": 0.5})
    size = ("min_size", {"min_width_px": 22, "min_height_px": 22})
    assert run_chain(build_ctx, [nms, size], [a, b]) == []
    assert run_chain(build_ctx, [size, nms], [a, b]) == [b]


def test_top_k_before_versus_after_nms_differ(build_ctx):
    dets = [
        det("router", 0.9, (0, 0, 40, 40)),
        det("modem", 0.8, (1, 1, 40, 40)),  # duplicate of the first
        det("charger", 0.7, (60, 60, 95, 95)),
    ]
    nms = ("nms", {})
    top2 = ("top_k", {"k": 2})
    assert len(run_chain(build_ctx, [top2, nms], dets)) == 1
    assert len(run_chain(build_ctx, [nms, top2], dets)) == 2


def test_nms_before_score_threshold_runs_and_gives_the_same_set(build_ctx):
    # Greedy NMS only lets a box suppress boxes with lower confidence, so for these two
    # filters either order keeps the same boxes. The configured order is still honoured.
    dets = [det(conf=0.9), det(conf=0.3, box=(11, 11, 50, 50)), det(conf=0.4, box=(60, 60, 90, 90))]
    nms = ("nms", {})
    score = ("score_threshold", {"min_score": 0.35})
    assert run_chain(build_ctx, [nms, score], dets) == run_chain(build_ctx, [score, nms], dets)


def test_result_does_not_depend_on_input_order(build_ctx):
    rng = np.random.default_rng(0)
    scores, xs, ys = rng.random(30), rng.integers(0, 55, 30), rng.integers(0, 55, 30)
    dets = [
        det("router", round(float(c), 2), (int(x), int(y), int(x) + 40, int(y) + 40))
        for c, x, y in zip(scores, xs, ys, strict=True)
    ]
    specs = [("nms", {"iou_threshold": 0.4}), ("top_k", {"k": 5})]
    expected = run_chain(build_ctx, specs, dets)
    for seed in range(5):
        shuffled = list(dets)
        np.random.default_rng(seed).shuffle(shuffled)
        assert run_chain(build_ctx, specs, shuffled) == expected


# --- ids -----------------------------------------------------------------------------


def test_ids_are_unique_and_ordered_by_y_then_x():
    dets = [
        det(box=(50, 40, 60, 50)),
        det(box=(10, 40, 20, 50)),
        det(box=(80, 5, 90, 15)),
    ]
    out = assign_ids(dets)
    assert [d.detection_id for d in out] == ["obj_1", "obj_2", "obj_3"]
    assert [d.bbox_xyxy[:2] for d in out] == [(80, 5), (10, 40), (50, 40)]
    assert all(d.detection_id is None for d in dets)  # input untouched


def test_ids_identical_for_shuffled_input():
    dets = [det(box=(i % 5 * 10, i // 5 * 10, i % 5 * 10 + 9, i // 5 * 10 + 9)) for i in range(15)]
    expected = assign_ids(dets)
    for seed in range(5):
        shuffled = list(dets)
        np.random.default_rng(seed).shuffle(shuffled)
        assert assign_ids(shuffled) == expected


# --- sanitize ------------------------------------------------------------------------


def test_sanitize_clamps_and_drops_invalid_detections():
    nan = float("nan")

    def raw(box, conf=0.9):
        return RawDetection.model_construct(phrase="router", confidence=conf, bbox_xyxy=box)

    hostile = [
        det("router", 0.9, (10, 10, 50, 50)),
        raw((-20, -5, 40, 30)),  # negative coordinates: clamped
        raw((90, 90, 300, 300)),  # partly outside: clamped
        raw((200, 200, 300, 300)),  # entirely outside: empty after clamping
        raw((50, 10, 20, 40)),  # x2 <= x1
        raw((30, 30, 30, 60)),  # zero area
        raw((nan, 0, 10, 10)),
        raw((0, 0, 10, 10), conf=nan),
        raw((0, 0, 10, 10), conf=1.7),
        raw((0, 0, 10)),  # wrong arity
    ]
    out = sanitize_detections(hostile, 100, 100)
    assert [d.bbox_xyxy for d in out] == [(10, 10, 50, 50), (0, 0, 40, 30), (90, 90, 100, 100)]
    assert all(type(v) is int for d in out for v in d.bbox_xyxy)


def test_sanitize_drops_phrases_outside_the_vocabulary_with_a_warning(caplog):
    dets = [det("router"), det("toaster", box=(60, 60, 90, 90))]
    with caplog.at_level(logging.WARNING):
        out = sanitize_detections(dets, 100, 100, vocabulary=["router", "electronic device"])
    assert [d.phrase for d in out] == ["router"]
    assert any("toaster" in r.message and r.levelno == logging.WARNING for r in caplog.records)


def test_sanitize_keeps_a_one_pixel_wide_box_at_the_border_for_the_filters_to_judge(build_ctx):
    border = det("router", 0.9, (99, 10, 100, 50))
    out = sanitize_detections([border], 100, 100)
    assert [d.bbox_xyxy for d in out] == [(99, 10, 100, 50)]
    assert make_filter(build_ctx, "min_size").apply(out, CTX) == []


def test_1000_boxes_go_through_the_default_filters_quickly(build_ctx, resolved_config):
    rng = np.random.default_rng(0)
    x1 = rng.integers(0, 80, 1000)
    y1 = rng.integers(0, 80, 1000)
    dets = [
        det("router", float(c), (int(x), int(y), int(x) + 15, int(y) + 15))
        for c, x, y in zip(rng.random(1000), x1, y1, strict=True)
    ]
    filters = [
        build("detection_filter", spec, build_ctx)
        for spec in resolved_config["pipeline"]["detection_filters"]
    ]
    start = time.perf_counter()
    out = dets
    for f in filters:
        out = f.apply(out, CTX)
    assert time.perf_counter() - start < 1.0
    assert len(out) <= 20


# --- vocabulary, prompts and the Grounding DINO label logic --------------------------


def test_vocabulary_normalization_gives_a_clean_prompt():
    vocab = normalize_vocabulary(["Router", " router ", "", "USB-C Hub.", "power strip", "."])
    prompt = build_prompt(vocab)
    assert vocab == ["router", "usb-c hub", "power strip", "electronic device"]
    assert prompt == "router . usb-c hub . power strip . electronic device ."
    assert ". ." not in prompt and "  " not in prompt


def test_chunk_vocabulary_keeps_whole_phrases_within_the_limit():
    count = lambda text: len(text.split())  # noqa: E731
    phrases = [f"device{i}" for i in range(12)] + ["smart speaker"]
    chunks = chunk_vocabulary(phrases, 10, count)
    assert len(chunks) > 1
    assert [p for chunk in chunks for p in chunk] == phrases  # nothing lost, nothing truncated
    assert all(count(build_prompt(chunk)) <= 10 for chunk in chunks)


def test_chunk_vocabulary_gives_an_oversized_phrase_its_own_prompt(caplog):
    count = lambda text: len(text.split())  # noqa: E731
    with caplog.at_level(logging.WARNING):
        chunks = chunk_vocabulary(["a", "one two three four five six", "b"], 4, count)
    assert chunks == [["a"], ["one two three four five six"], ["b"]]
    assert any("alone exceeds" in r.message for r in caplog.records)


def test_map_label_uses_the_longest_vocabulary_match():
    vocab = ["router", "smart speaker", "speaker", "power strip"]
    assert map_label("smart speaker", vocab) == "smart speaker"
    assert map_label("a wifi router.", vocab) == "router"
    assert map_label("toaster", vocab) is None
    assert map_label("", vocab) is None


class StubDino(GroundingDinoDetector):
    """Grounding DINO with the model replaced by a scripted ``_run``."""

    def _load_backend(self):
        self._count_tokens = lambda text: len(text.split())
        self.prompts = []

    def _run(self, image_rgb, prompt):
        self.prompts.append(prompt)
        first = prompt.split(" . ")[0]
        return [(first, 0.8, [10.4, 10.6, 50.2, 60.0]), ("toaster", 0.9, [1, 1, 9, 9])]


def test_dino_splits_a_long_vocabulary_and_merges_the_results(caplog):
    vocab = tuple(f"device{i}" for i in range(12)) + ("electronic device",)
    ctx = BuildContext(categories=vocab, device="cpu", seed=0)
    detector = StubDino(StubDino.Params(max_prompt_tokens=10), ctx)
    with caplog.at_level(logging.WARNING):
        found = detector.predict(make_image("solid", 80, 80))
    assert len(detector.prompts) == len(detector.chunks) > 1
    assert [d.phrase for d in found] == [chunk[0] for chunk in detector.chunks]  # one per prompt
    assert found[0].bbox_xyxy == (10, 11, 50, 60)
    assert any("toaster" in r.message for r in caplog.records)  # unmapped label dropped


def test_dino_returns_an_empty_list_when_nothing_is_found():
    class Empty(StubDino):
        def _run(self, image_rgb, prompt):
            return []

    ctx = BuildContext(categories=("router", "electronic device"), device="cpu", seed=0)
    assert Empty(Empty.Params(), ctx).predict(make_image("solid", 40, 40)) == []


def test_fake_detector_places_boxes_relative_to_the_image(build_ctx):
    detector = build("detector", {"name": "fake"}, build_ctx)
    [found] = detector.predict(make_image("solid", 200, 400))
    assert (found.phrase, found.bbox_xyxy) == ("router", (100, 50, 300, 150))
    custom = build(
        "detector",
        {"name": "fake", "params": {"detections": [{"phrase": "modem", "box_rel": [0, 0, 1, 1]}]}},
        build_ctx,
    )
    assert custom.predict(make_image("solid", 10, 20))[0].bbox_xyxy == (0, 0, 20, 10)


def test_dino_import_error_carries_an_install_hint(build_ctx, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("torch", "transformers"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"electronics-classifier\[dino\]"):
        build("detector", {"name": "grounding_dino"}, build_ctx)
