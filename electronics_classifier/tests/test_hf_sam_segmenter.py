"""The Hugging Face SAM segmenter (``hf_sam``), with the model call stubbed."""

import builtins
import os

import numpy as np
import pytest
from conftest import VALID_REPLY, det, make_image, make_pipeline

from electronics_classifier.components.segmenters import HfSamSegmenter
from electronics_classifier.interfaces import BuildContext
from electronics_classifier.registry import build, registered_names

H, W = 120, 200
IMAGE = np.zeros((H, W, 3), dtype=np.uint8)
BOXES = np.array([[10, 10, 80, 90], [100, 20, 190, 110], [5, 5, 30, 30]])


def box_mask(box):
    mask = np.zeros((H, W), dtype=bool)
    x1, y1, x2, y2 = (int(v) for v in box)
    mask[y1:y2, x1:x2] = True
    return mask


class StubSam(HfSamSegmenter):
    """Real batching and retry logic; ``_masks`` answers from a script."""

    def __init__(self, fail_batch=False, bad_boxes=(), short=False):
        self.device = "cpu"
        self.fail_batch, self.bad_boxes, self.short = fail_batch, set(bad_boxes), short
        self.calls: list[int] = []  # number of boxes in each model call

    def _masks(self, image_rgb, boxes):
        self.calls.append(len(boxes))
        if self.fail_batch and len(boxes) > 1:
            raise RuntimeError("batch failed")
        if any(int(b[0]) in self.bad_boxes for b in boxes):
            raise RuntimeError("bad box")
        masks = [box_mask(b) for b in boxes]
        return masks[:-1] if self.short else masks


def test_every_box_gets_a_bool_mask_in_the_original_frame_from_one_call():
    sam = StubSam()
    masks = sam.segment(IMAGE, BOXES)
    assert sam.calls == [3]  # all boxes in a single model call
    assert [m.shape for m in masks] == [(H, W)] * 3 and all(m.dtype == bool for m in masks)
    assert masks[0][50, 40] and not masks[0][50, 150]


def test_no_boxes_means_no_model_call():
    sam = StubSam()
    assert sam.segment(IMAGE, np.zeros((0, 4))) == [] and sam.calls == []


def test_a_failed_batch_is_retried_box_by_box():
    sam = StubSam(fail_batch=True)
    masks = sam.segment(IMAGE, BOXES)
    assert sam.calls == [3, 1, 1, 1] and all(m is not None for m in masks)


def test_one_bad_box_loses_only_its_own_mask():
    sam = StubSam(fail_batch=True, bad_boxes={100})
    masks = sam.segment(IMAGE, BOXES)
    assert [m is None for m in masks] == [False, True, False]


def test_a_wrong_mask_count_is_retried_instead_of_misaligning_masks_and_boxes():
    sam = StubSam(short=True)
    masks = sam.segment(IMAGE, BOXES)
    assert len(masks) == 3 and sam.calls[0] == 3 and sam.calls[1:] == [1, 1, 1]


def test_the_segmenter_is_registered_with_a_described_model_id():
    assert "hf_sam" in registered_names("segmenter")
    assert HfSamSegmenter.Params().model_id == "facebook/sam-vit-base"


def test_missing_packages_give_an_install_hint(build_ctx, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("torch", "transformers"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"electronics-classifier\[grounded\]"):
        build("segmenter", {"name": "hf_sam"}, build_ctx)


@pytest.mark.parametrize(("device", "expected"), [("mps", "1"), ("cpu", None)])
def test_the_mps_cpu_fallback_is_set_only_on_apple_gpus(monkeypatch, device, expected):
    class NoModel(HfSamSegmenter):
        def _load_backend(self):
            pass

    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    NoModel(NoModel.Params(), BuildContext(("router", "electronic device"), device, 0))
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == expected
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)


# --- inside the pipeline ---------------------------------------------------------------


def test_masks_reach_the_crop_and_a_missing_mask_becomes_a_warning():
    class FirstOnly(StubSam):
        def _masks(self, image_rgb, boxes):
            if len(boxes) > 1:
                raise RuntimeError("batch")
            return [box_mask(b) for b in boxes] if int(boxes[0][0]) == 100 else [None]

    detections = [det("router", 0.9, (100, 20, 190, 110)), det("laptop", 0.8, (10, 10, 80, 90))]
    pipeline = make_pipeline(
        detections, [VALID_REPLY],
        sets=["pipeline.segmentation_gate.name=always"], segmenter=FirstOnly(),
    )  # fmt: skip
    annotated, result = pipeline.run(make_image("noise", H, W))
    by_phrase = {i.detector_phrase: i for i in result.items}
    assert "SEGMENTATION_FAILED" not in by_phrase["router"].warnings
    assert "SEGMENTATION_FAILED" in by_phrase["laptop"].warnings
    assert annotated.shape == (H, W, 3)
