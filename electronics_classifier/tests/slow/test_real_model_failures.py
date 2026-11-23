"""Tier C: real-model failures (Section 10.1). Informational, not a gate.

Needs real weights (`pip install -e '.[dino,vlm]'`) and the photos listed in
tests/fixtures/failure_images/README.md. Run with: pytest tests/slow -m slow -s

Every test states the IDEAL behaviour and is xfail(strict=False): a failure is printed as an
expected failure instead of breaking CI. A test is skipped when its photo is missing.
"""

import io
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.slow
PHOTOS = Path(__file__).resolve().parents[1] / "fixtures" / "failure_images"
informational = pytest.mark.xfail(strict=False, reason="Tier C: real-model behaviour")


def load_photo(name: str) -> np.ndarray:
    path = PHOTOS / name
    if not path.is_file():
        pytest.skip(f"missing photo {path.relative_to(PHOTOS.parents[1])} (see its README.md)")
    return np.array(Image.open(path).convert("RGB"))


_PIPELINE = {}


def get_pipeline():
    """Built on first use, after the photo check, so a missing photo is reported first."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if "default" not in _PIPELINE:
        from electronics_classifier.pipeline import ElectronicsClassificationPipeline

        _PIPELINE["default"] = ElectronicsClassificationPipeline()
    return _PIPELINE["default"]


def classify(image, pipe=None):
    _, result = (pipe or get_pipeline()).run(image)
    for item in result.items:
        a = item.attributes
        print(f"  {item.id} {a.category} | {a.brand} {a.model} | {item.warnings}")
    return result


def categories(result):
    return [i.attributes.category for i in result.items]


@informational
def test_c01_kitchen_without_vocabulary_devices_gives_no_confident_false_positive():
    result = classify(load_photo("kitchen_toaster_kettle.jpg"))
    assert all(
        i.attributes.category == "electronic device" and "LOW_DETECTOR_CONFIDENCE" in i.warnings
        for i in result.items
    )


@informational
def test_c02_a_tv_screen_showing_a_laptop_is_one_television():
    result = classify(load_photo("tv_showing_laptop.jpg"))
    assert categories(result) == ["television"]


@informational
def test_c03_a_poster_of_a_smartphone_is_not_a_device():
    assert classify(load_photo("poster_smartphone.jpg")).items == []


@informational
def test_c04_six_chargers_on_a_power_strip():
    result = classify(load_photo("power_strip_six_chargers.jpg"))
    assert categories(result).count("power strip") == 1
    assert categories(result).count("charger") <= 6


@informational
def test_c05_a_cable_tangle_is_a_handful_of_cables_not_one_giant_box():
    image = load_photo("cable_tangle.jpg")
    result = classify(image)
    area = image.shape[0] * image.shape[1]
    boxes = [
        (i.bbox_xyxy[2] - i.bbox_xyxy[0]) * (i.bbox_xyxy[3] - i.bbox_xyxy[1])
        for i in result.items
    ]
    assert 2 <= len(result.items) <= 12 and max(boxes) / area < 0.5


@informational
def test_c06_a_small_remote_in_a_large_photo_is_found_despite_downscaling():
    from electronics_classifier.pipeline import ElectronicsClassificationPipeline

    image = load_photo("remote_small_in_large_4000x3000.jpg")
    default = classify(image)
    full_res = ElectronicsClassificationPipeline(
        sets=["pipeline.image_preprocess=[{name: ensure_rgb_uint8}]"]
    )
    print("  without resize:", categories(classify(image, full_res)))
    assert "remote control" in categories(default)


@informational
def test_c07_a_dark_router_is_detected():
    path = PHOTOS / "router_black_on_black.jpg"
    image = load_photo("router_black_on_black.jpg" if path.is_file() else "router_clear_label.jpg")
    if not path.is_file():
        image = (image * 0.15).astype(np.uint8)  # synthetic darkening of the sample photo
    assert classify(image).items


@informational
def test_c08_motion_blur_and_heavy_jpeg_leave_the_brand_unknown():
    image = load_photo("router_clear_label.jpg")
    blurred = cv2.blur(image, (41, 1))  # horizontal motion blur
    buffer = io.BytesIO()
    Image.fromarray(blurred).save(buffer, format="JPEG", quality=5)
    degraded = np.array(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))
    result = classify(degraded)
    assert result.items and all(i.attributes.brand == "unknown" for i in result.items)


@informational
def test_c09_a_blank_label_leaves_the_model_unknown():
    result = classify(load_photo("router_blank_label.jpg"))
    assert result.items and all(i.attributes.model == "unknown" for i in result.items)


@informational
def test_c10_a_rotated_logo_is_read_correctly_or_left_unknown():
    result = classify(load_photo("router_logo_rotated.jpg"))
    assert result.items and all(i.attributes.brand in ("TP-Link", "unknown") for i in result.items)


@informational
def test_c11_a_non_latin_brand_is_read_or_left_unknown():
    result = classify(load_photo("device_non_latin_brand.jpg"))
    assert result.items
    assert all(i.attributes.brand in ("삼성", "Samsung", "unknown") for i in result.items)


@informational
def test_c12_two_identical_routers_side_by_side_are_two_items():
    assert len(classify(load_photo("two_routers_overlapping.jpg")).items) == 2


@informational
def test_c13_a_glare_photo_still_finds_the_device_and_admits_unreadable_fields():
    result = classify(load_photo("router_glare.jpg"))
    assert result.items
    assert all("unknown" in (i.attributes.brand, i.attributes.model) for i in result.items)


@informational
def test_c14_a_crowded_shelf_respects_top_k_and_finishes():
    image = load_photo("store_shelf_30_devices.jpg")
    start = time.perf_counter()
    result = classify(image)
    print(f"  runtime: {time.perf_counter() - start:.1f}s for {len(result.items)} items")
    assert 0 < len(result.items) <= 20
