import builtins
import logging

import numpy as np
import pytest
from conftest import make_image

from electronics_classifier.components.detection_filters import assign_ids
from electronics_classifier.components.segmenters import Sam2Segmenter, SamSegmenter
from electronics_classifier.config import ConfigError, CropConfig
from electronics_classifier.crop_builder import (
    MASK_FILL,
    build_prepared_items,
    downscale_to_max_side,
    pad_and_clamp,
    segment_approved,
    segmentation_warnings,
)
from electronics_classifier.interfaces import BuildContext
from electronics_classifier.registry import build
from electronics_classifier.schemas import RawDetection, WarningCode


def det(box, phrase="router", conf=0.9):
    return RawDetection(phrase=phrase, confidence=conf, bbox_xyxy=box)


def crop_cfg(**overrides):
    values = {"padding_ratio": 0.12, "include_full_image_context": True, "max_image_side_px": 1024}
    return CropConfig(**{**values, **overrides})


def make(build_ctx, kind, name, **params):
    return build(kind, {"name": name, "params": params}, build_ctx)


# --- padding and clamping ------------------------------------------------------------


def test_padding_expands_each_side_by_a_fraction_of_the_box():
    assert pad_and_clamp((40, 40, 60, 60), 0.5, 100, 100) == (30, 30, 70, 70)
    assert pad_and_clamp((40, 40, 60, 60), 0.0, 100, 100) == (40, 40, 60, 60)
    assert pad_and_clamp((20, 10, 60, 30), 0.25, 200, 200) == (10, 5, 70, 35)  # w and h differ


@pytest.mark.parametrize(
    "box,expected",
    [
        ((0, 40, 20, 60), (0, 30, 30, 70)),  # left edge
        ((80, 40, 100, 60), (70, 30, 100, 70)),  # right edge
        ((40, 0, 60, 20), (30, 0, 70, 30)),  # top edge
        ((40, 80, 60, 100), (30, 70, 70, 100)),  # bottom edge
        ((0, 0, 100, 100), (0, 0, 100, 100)),  # whole image
    ],
)
def test_padding_clamps_at_all_four_edges(box, expected):
    assert pad_and_clamp(box, 0.5, 100, 100) == expected


@pytest.mark.parametrize(
    "box", [(0, 0, 1, 1), (99, 99, 100, 100), (99, 0, 100, 1), (0, 99, 1, 100), (50, 50, 51, 52)]
)
def test_crop_is_never_empty_even_for_tiny_boxes_at_the_border(box):
    x1, y1, x2, y2 = pad_and_clamp(box, 0.12, 100, 100)
    assert 0 <= x1 < x2 <= 100 and 0 <= y1 < y2 <= 100
    items = build_prepared_items(make_image("noise", 100, 100), [det(box)], [None], crop_cfg())
    assert items[0].crop_rgb.shape[0] >= 1 and items[0].crop_rgb.shape[1] >= 1


def test_padding_degenerate_box_still_yields_one_pixel():
    assert pad_and_clamp((100, 100, 100, 100), 0.12, 100, 100) == (99, 99, 100, 100)


# --- crops and context ---------------------------------------------------------------


def test_prepared_item_fields_and_untouched_input():
    image = make_image("noise", 120, 160)
    snapshot = image.copy()
    d = det((40, 30, 100, 90))
    [item] = build_prepared_items(image, [d], [None], crop_cfg(padding_ratio=0.1))
    assert item.detection == d
    assert item.padded_bbox_xyxy == (34, 24, 106, 96)
    assert (item.crop_rgb == image[24:96, 34:106]).all()
    assert item.mask is None
    item.crop_rgb[...] = 0  # the crop is a copy
    assert (image == snapshot).all()


def test_read_only_and_non_contiguous_images_are_accepted():
    base = make_image("noise", 60, 80)
    flipped = base[:, ::-1]
    read_only = base.copy()
    read_only.setflags(write=False)
    for image in (flipped, read_only):
        snapshot = image.copy()
        [item] = build_prepared_items(image, [det((10, 10, 40, 40))], [None], crop_cfg())
        assert item.padded_bbox_xyxy == (6, 6, 44, 44)
        assert (item.crop_rgb == image[6:44, 6:44]).all()
        assert (image == snapshot).all()


def test_context_is_the_downscaled_full_image_or_absent():
    image = make_image("gradient", 300, 600)
    [with_ctx] = build_prepared_items(
        image, [det((10, 10, 60, 60))], [None], crop_cfg(max_image_side_px=200)
    )
    assert with_ctx.context_rgb.shape == (100, 200, 3)
    assert not with_ctx.context_rgb.flags.writeable
    [without] = build_prepared_items(
        image, [det((10, 10, 60, 60))], [None], crop_cfg(include_full_image_context=False)
    )
    assert without.context_rgb is None
    assert image.flags.writeable  # the caller's array is not locked


def test_small_images_are_not_upscaled_and_context_is_a_copy():
    image = make_image("noise", 50, 60)
    [item] = build_prepared_items(image, [det((5, 5, 30, 30))], [None], crop_cfg())
    assert item.context_rgb.shape == image.shape
    assert not np.shares_memory(item.context_rgb, image)


def test_crops_and_context_respect_max_image_side():
    image = make_image("noise", 3000, 4000)
    [item] = build_prepared_items(
        image, [det((0, 0, 4000, 3000))], [None], crop_cfg(max_image_side_px=512)
    )
    assert max(item.crop_rgb.shape[:2]) <= 512
    assert max(item.context_rgb.shape[:2]) <= 512
    assert downscale_to_max_side(make_image("solid", 10, 20), 512).shape == (10, 20, 3)


# --- crop transforms -----------------------------------------------------------------


class Marker:
    """Records the order of application and stamps a value into the crop."""

    def __init__(self, log, label, value):
        self.log, self.label, self.value = log, label, value

    def apply(self, crop):
        self.log.append(self.label)
        out = crop.copy()
        out[0, 0] = self.value
        return out


def test_crop_transforms_run_in_the_configured_order():
    log = []
    image = make_image("solid", 60, 60)
    a, b = Marker(log, "a", 11), Marker(log, "b", 22)
    [item] = build_prepared_items(image, [det((10, 10, 50, 50))], [None], crop_cfg(), [a, b])
    assert log == ["a", "b"]
    assert item.crop_rgb[0, 0].tolist() == [22, 22, 22]  # last transform wins
    log.clear()
    [item] = build_prepared_items(image, [det((10, 10, 50, 50))], [None], crop_cfg(), [b, a])
    assert log == ["b", "a"]
    assert item.crop_rgb[0, 0].tolist() == [11, 11, 11]


def test_transforms_run_on_the_masked_crop_and_before_the_final_downscale(build_ctx):
    upscale = make(build_ctx, "crop_transform", "upscale_min_side", min_side=400)
    image = make_image("noise", 200, 200)
    [item] = build_prepared_items(
        image, [det((50, 50, 90, 90))], [None], crop_cfg(max_image_side_px=300), [upscale]
    )
    # 48px crop is upscaled to 400 and then brought back under max_image_side_px
    assert max(item.crop_rgb.shape[:2]) == 300


def test_builtin_crop_transforms(build_ctx):
    crop = make_image("checkerboard", 40, 60)
    identity = make(build_ctx, "crop_transform", "identity")
    assert identity.apply(crop) is crop

    upscale = make(build_ctx, "crop_transform", "upscale_min_side", min_side=80)
    assert upscale.apply(crop).shape == (80, 120, 3)
    big = make_image("solid", 100, 120)
    assert upscale.apply(big) is big  # never downscales

    soft = make_image("blurred", 40, 60)
    sharp = make(build_ctx, "crop_transform", "sharpen", amount=1.5).apply(soft)
    assert sharp.shape == soft.shape and sharp.dtype == np.uint8
    assert sharp.astype(int).std() > soft.astype(int).std()
    assert (make(build_ctx, "crop_transform", "sharpen", amount=0.0).apply(soft) == soft).all()

    eq = make(build_ctx, "crop_transform", "clahe").apply(make_image("gradient", 64, 64))
    assert eq.shape == (64, 64, 3) and eq.dtype == np.uint8


def test_transform_params_are_validated(build_ctx):
    with pytest.raises(ConfigError, match=r"params\.min_side"):
        make(build_ctx, "crop_transform", "upscale_min_side")  # required
    with pytest.raises(ConfigError, match=r"params\.amount"):
        make(build_ctx, "crop_transform", "sharpen", amount=-1)


# --- gate + segmentation -------------------------------------------------------------


class RecordingSegmenter:
    def __init__(self):
        self.calls = []

    def segment(self, image_rgb, boxes_xyxy):
        self.calls.append(np.asarray(boxes_xyxy).tolist())
        masks = []
        for x1, y1, x2, y2 in np.asarray(boxes_xyxy).astype(int):
            mask = np.zeros(image_rgb.shape[:2], dtype=bool)
            mask[y1:y2, x1:x2] = True
            masks.append(mask)
        return masks


class ExplodingGate:
    def should_segment(self, item_idx, dets):
        raise AssertionError("gate must not be consulted without a real segmenter")


DETS = [
    det((10, 10, 60, 60)),  # overlaps the next one
    det((20, 20, 70, 70)),
    det((200, 200, 240, 240)),  # alone
]


def test_overlap_gate_triggers_only_on_overlapping_items(build_ctx):
    gate = make(build_ctx, "segmentation_gate", "overlap", iou=0.15)
    assert [gate.should_segment(i, DETS) for i in range(3)] == [True, True, False]
    strict = make(build_ctx, "segmentation_gate", "overlap", iou=0.99)
    assert [strict.should_segment(i, DETS) for i in range(3)] == [False, False, False]
    assert gate.should_segment(0, DETS[:1]) is False  # a lone detection never overlaps


def test_always_gate_approves_everything(build_ctx):
    gate = make(build_ctx, "segmentation_gate", "always")
    assert all(gate.should_segment(i, DETS) for i in range(3))


def test_segmenter_only_receives_items_the_gate_approved(build_ctx):
    segmenter = RecordingSegmenter()
    gate = make(build_ctx, "segmentation_gate", "overlap", iou=0.15)
    outcome = segment_approved(segmenter, gate, make_image("solid", 300, 300), DETS)
    masks = outcome.masks
    assert len(segmenter.calls) == 1 and len(segmenter.calls[0]) == 2
    assert masks[0] is not None and masks[1] is not None and masks[2] is None
    assert outcome.approved == (0, 1)


def test_gate_is_not_consulted_when_the_segmenter_is_none(build_ctx):
    none = make(build_ctx, "segmenter", "none")
    outcome = segment_approved(none, ExplodingGate(), make_image("solid", 300, 300), DETS)
    assert outcome.masks == [None, None, None]
    assert outcome.approved == ()
    assert none.segment(make_image("solid", 10, 10), np.zeros((4, 4))) == [None] * 4


def test_nothing_is_segmented_when_no_item_is_approved(build_ctx):
    segmenter = RecordingSegmenter()
    gate = make(build_ctx, "segmentation_gate", "overlap", iou=0.99)
    outcome = segment_approved(segmenter, gate, make_image("solid", 300, 300), DETS)
    assert outcome.masks == [None] * 3 and outcome.approved == ()
    assert segmenter.calls == []


# --- masks ---------------------------------------------------------------------------


def test_masked_crop_fills_outside_pixels_gray_without_eroding_edges():
    image = make_image("noise", 100, 100)
    box = (20, 20, 80, 80)
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True
    [item] = build_prepared_items(image, [det(box)], [mask], crop_cfg(padding_ratio=0.0))
    crop = item.crop_rgb
    inside = mask[20:80, 20:80]
    assert (crop[~inside] == MASK_FILL).all()
    assert (crop[inside] == image[20:80, 20:80][inside]).all()  # mask edge pixels kept as is
    assert item.mask is mask


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((100, 100), dtype=bool),  # all False
        np.ones((50, 50), dtype=bool),  # wrong shape
        np.ones((100, 100, 3), dtype=bool),  # wrong ndim
        "not a mask",
    ],
)
def test_unusable_masks_fall_back_to_the_unmasked_crop(bad, caplog):
    image = make_image("noise", 100, 100)
    [plain] = build_prepared_items(image, [det((20, 20, 60, 60))], [None], crop_cfg())
    with caplog.at_level(logging.WARNING):
        [item] = build_prepared_items(image, [det((20, 20, 60, 60))], [bad], crop_cfg())
    assert item.mask is None
    assert (item.crop_rgb == plain.crop_rgb).all()
    assert caplog.records


def test_mask_that_misses_the_crop_is_ignored():
    image = make_image("noise", 100, 100)
    elsewhere = np.zeros((100, 100), dtype=bool)
    elsewhere[80:100, 80:100] = True
    [item] = build_prepared_items(image, [det((5, 5, 30, 30))], [elsewhere], crop_cfg())
    assert item.mask is None


def test_missing_mask_entries_mean_no_mask():
    image = make_image("noise", 100, 100)
    dets = [det((5, 5, 30, 30)), det((50, 50, 90, 90))]
    items = build_prepared_items(image, dets, [], crop_cfg())
    assert [i.mask for i in items] == [None, None]


# --- SAM / SAM2 ----------------------------------------------------------------------


def test_sam_with_a_missing_checkpoint_is_a_config_error_naming_the_path(build_ctx, tmp_path):
    for name, extra in (("sam", {}), ("sam2", {"model_cfg": "cfg.yaml"})):
        with pytest.raises(ConfigError) as err:
            build(
                "segmenter",
                {"name": name, "params": {"checkpoint": str(tmp_path / "nope.pth"), **extra}},
                build_ctx,
                path="pipeline.segmenter",
            )
        assert "pipeline.segmenter.params.checkpoint" in str(err.value)
        assert "nope.pth" in str(err.value)


def test_sam_requires_a_checkpoint_param(build_ctx):
    with pytest.raises(ConfigError, match=r"pipeline\.segmenter\.params\.checkpoint"):
        build("segmenter", {"name": "sam"}, build_ctx, path="pipeline.segmenter")


def test_sam_missing_dependency_carries_an_install_hint(build_ctx, tmp_path, monkeypatch):
    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"x")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] in ("segment_anything", "sam2"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"electronics-classifier\[sam\]"):
        build("segmenter", {"name": "sam", "params": {"checkpoint": str(checkpoint)}}, build_ctx)
    with pytest.raises(ImportError, match=r"electronics-classifier\[sam2\]"):
        build(
            "segmenter",
            {"name": "sam2", "params": {"checkpoint": str(checkpoint), "model_cfg": "c.yaml"}},
            build_ctx,
        )


class FakePredictor:
    """Stands in for SamPredictor. ``fail_on`` is the 1-based predict call that raises."""

    def __init__(self, fail_on=None, fail_set_image=False):
        self.image = None
        self.boxes = []
        self.fail_on = fail_on
        self.fail_set_image = fail_set_image

    def set_image(self, image):
        if self.fail_set_image:
            raise RuntimeError("out of memory")
        self.image = image

    def predict(self, box, multimask_output):
        assert multimask_output is False
        self.boxes.append(box.tolist())
        if self.fail_on is not None and len(self.boxes) == self.fail_on:
            raise RuntimeError("boom")
        h, w = self.image.shape[:2]
        mask = np.zeros((1, h, w), dtype=bool)
        x1, y1, x2, y2 = box.astype(int)
        mask[0, y1:y2, x1:x2] = True
        return mask, np.array([0.9]), np.zeros((1, 4, 4))


@pytest.mark.parametrize("cls", [SamSegmenter, Sam2Segmenter])
def test_box_prompt_segmenters_return_one_original_frame_mask_per_box(cls, tmp_path):
    checkpoint = tmp_path / "w.pth"
    checkpoint.write_bytes(b"x")

    class Stubbed(cls):
        def _load_predictor(self):
            return FakePredictor(fail_on=2)

    extra = {"model_cfg": "c.yaml"} if cls is Sam2Segmenter else {}
    params = cls.Params(checkpoint=str(checkpoint), **extra)
    segmenter = Stubbed(params, BuildContext(("router",), "cpu", 0))
    image = make_image("solid", 60, 80)
    masks = segmenter.segment(image, np.array([[5, 5, 20, 20], [30, 10, 50, 40], [1, 1, 9, 9]]))
    assert masks[0].shape == (60, 80) and masks[0].dtype == bool and masks[0].sum() == 225
    assert masks[1] is None  # the failing box does not lose the others
    assert masks[2] is not None
    assert segmenter.segment(image, np.zeros((0, 4))) == []


# --- forced SAM failures: degrade to the plain crop and flag the item ---------------------


def sam_with(predictor, tmp_path, cls=SamSegmenter):
    """A real SamSegmenter/Sam2Segmenter whose model is replaced by ``predictor``."""
    checkpoint = tmp_path / "w.pth"
    checkpoint.write_bytes(b"x")

    class Stubbed(cls):
        def _load_predictor(self):
            return predictor

    extra = {"model_cfg": "c.yaml"} if cls is Sam2Segmenter else {}
    params = cls.Params(checkpoint=str(checkpoint), **extra)
    return Stubbed(params, BuildContext(("router",), "cpu", 0))


def run_segmented(segmenter, build_ctx, gate="always"):
    """Gate -> segmenter -> crops -> warnings, as the pipeline will chain them."""
    image = make_image("noise", 300, 300)
    dets = assign_ids(DETS)  # three devices: ids obj_1..obj_3, ordered by (y1, x1)
    outcome = segment_approved(segmenter, make(build_ctx, "segmentation_gate", gate), image, dets)
    items = build_prepared_items(image, dets, outcome.masks, crop_cfg(padding_ratio=0.0))
    return image, outcome, items, segmentation_warnings(outcome, items)


def test_one_failing_box_gives_a_plain_crop_and_a_segmentation_warning(build_ctx, tmp_path, caplog):
    predictor = FakePredictor(fail_on=2)  # SAM raises for the second box
    with caplog.at_level(logging.WARNING):
        image, outcome, items, warnings = run_segmented(sam_with(predictor, tmp_path), build_ctx)
    assert [m is None for m in outcome.masks] == [False, True, False]
    assert warnings == {"obj_2": [WarningCode.SEGMENTATION_FAILED.value]}
    assert any("segmentation failed" in r.message for r in caplog.records)

    failed = items[1]
    assert failed.mask is None
    x1, y1, x2, y2 = failed.padded_bbox_xyxy
    assert (failed.crop_rgb == image[y1:y2, x1:x2]).all()  # the plain rectangular crop
    for ok in (items[0], items[2]):  # the other items keep their masked crops
        assert ok.mask is not None
        assert (ok.crop_rgb == MASK_FILL).any()


def test_failing_set_image_flags_every_approved_item_and_does_not_abort(
    build_ctx, tmp_path, caplog
):
    predictor = FakePredictor(fail_set_image=True)
    with caplog.at_level(logging.WARNING):
        image, outcome, items, warnings = run_segmented(sam_with(predictor, tmp_path), build_ctx)
    assert outcome.masks == [None, None, None]
    assert predictor.boxes == []  # never got as far as prompting
    assert warnings == {
        f"obj_{i}": [WarningCode.SEGMENTATION_FAILED.value] for i in (1, 2, 3)
    }
    assert any("segmentation setup failed" in r.message for r in caplog.records)
    for item in items:
        x1, y1, x2, y2 = item.padded_bbox_xyxy
        assert (item.crop_rgb == image[y1:y2, x1:x2]).all()


@pytest.mark.parametrize("cls", [SamSegmenter, Sam2Segmenter])
def test_sam_and_sam2_fail_the_same_way(build_ctx, tmp_path, cls):
    for predictor, expected in (
        (FakePredictor(fail_set_image=True), {"obj_1", "obj_2", "obj_3"}),
        (FakePredictor(fail_on=3), {"obj_3"}),
    ):
        *_, warnings = run_segmented(sam_with(predictor, tmp_path, cls), build_ctx)
        assert set(warnings) == expected


def test_only_gate_approved_items_are_flagged(build_ctx, tmp_path):
    # obj_1 and obj_2 overlap and are approved; obj_3 is alone, never segmented, never flagged.
    predictor = FakePredictor(fail_set_image=True)
    *_, warnings = run_segmented(sam_with(predictor, tmp_path), build_ctx, gate="overlap")
    assert set(warnings) == {"obj_1", "obj_2"}


def test_no_warning_when_segmentation_succeeds_or_is_off(build_ctx, tmp_path):
    *_, ok_warnings = run_segmented(sam_with(FakePredictor(), tmp_path), build_ctx)
    assert ok_warnings == {}
    image = make_image("noise", 300, 300)
    dets = assign_ids(DETS)
    outcome = segment_approved(
        make(build_ctx, "segmenter", "none"), ExplodingGate(), image, dets
    )
    items = build_prepared_items(image, dets, outcome.masks, crop_cfg())
    assert segmentation_warnings(outcome, items) == {}


def test_an_unusable_mask_from_the_segmenter_also_counts_as_a_failure(build_ctx):
    class BadMasks:
        def segment(self, image_rgb, boxes_xyxy):
            return [np.zeros(image_rgb.shape[:2], dtype=bool), np.ones((5, 5), dtype=bool), None]

    class Real(BadMasks):  # segment_approved only skips the gate for NoSegmenter
        pass

    *_, warnings = run_segmented(Real(), build_ctx)
    assert set(warnings) == {"obj_1", "obj_2", "obj_3"}


def test_segmentation_warnings_need_assigned_ids(build_ctx):
    image = make_image("noise", 300, 300)
    segmenter = RecordingSegmenter()
    gate = make(build_ctx, "segmentation_gate", "always")
    outcome = segment_approved(segmenter, gate, image, DETS)
    outcome.masks[0] = None  # simulate a failure on an item whose id was never assigned
    items = build_prepared_items(image, DETS, outcome.masks, crop_cfg())
    with pytest.raises(ValueError, match="ids must be assigned"):
        segmentation_warnings(outcome, items)


def test_segmentation_failed_is_a_builtin_warning_code():
    assert WarningCode.SEGMENTATION_FAILED.value == "SEGMENTATION_FAILED"
