import numpy as np
import pytest
from conftest import make_image

from electronics_classifier.components.image_preprocess import (
    apply_preprocess,
    map_boxes_back,
    validate_image,
)
from electronics_classifier.registry import build


def step(build_ctx, name, **params):
    return build("image_preprocess", {"name": name, "params": params}, build_ctx)


def test_resize_then_map_back_round_trips_within_one_pixel(build_ctx):
    image = make_image("gradient", 400, 800)
    steps = [step(build_ctx, "ensure_rgb_uint8"), step(build_ctx, "resize_max_side", max_side=400)]
    resized, state = apply_preprocess(steps, image)
    assert resized.shape[:2] == (200, 400)  # resized to half

    box_on_resized = np.array([[50, 40, 150, 120]])  # a known box found on the resized image
    mapped = map_boxes_back(steps, box_on_resized, state)
    assert np.abs(mapped - np.array([[100, 80, 300, 240]])).max() <= 1


def test_two_geometric_steps_are_undone_in_reverse_order(build_ctx):
    image = make_image("noise", 800, 1200)
    steps = [
        step(build_ctx, "resize_max_side", max_side=600),
        step(build_ctx, "resize_max_side", max_side=300),
    ]
    resized, state = apply_preprocess(steps, image)
    assert max(resized.shape[:2]) == 300
    assert state.scale_x == pytest.approx(0.25, abs=0.01)
    mapped = map_boxes_back(steps, np.array([[10, 20, 100, 150]]), state)
    assert np.abs(mapped - np.array([[40, 80, 400, 600]])).max() <= 4  # <= 1 px of the 300 px frame


def test_resize_never_upscales_and_keeps_aspect_ratio(build_ctx):
    small = make_image("solid", 100, 200)
    steps = [step(build_ctx, "resize_max_side", max_side=1280)]
    out, state = apply_preprocess(steps, small)
    assert out.shape == small.shape
    assert map_boxes_back(steps, np.array([[1, 2, 3, 4]]), state).tolist() == [[1, 2, 3, 4]]

    big = make_image("solid", 3000, 4000)
    out, _ = apply_preprocess([step(build_ctx, "resize_max_side", max_side=1000)], big)
    assert out.shape[:2] == (750, 1000)


def test_extreme_aspect_ratio_never_produces_an_empty_image(build_ctx):
    sliver = make_image("solid", 4, 5000)
    out, _ = apply_preprocess([step(build_ctx, "resize_max_side", max_side=100)], sliver)
    assert out.shape[0] >= 1 and out.shape[1] == 100


@pytest.mark.parametrize(
    "name,params", [("ensure_rgb_uint8", {}), ("clahe", {}), ("gamma", {"gamma": 2.0})]
)
def test_non_geometric_steps_leave_box_coordinates_unchanged(build_ctx, name, params):
    image = make_image("gradient", 120, 160)
    steps = [step(build_ctx, name, **params)]
    out, state = apply_preprocess(steps, image)
    assert out.shape == image.shape and out.dtype == np.uint8
    boxes = np.array([[5, 6, 70, 80], [0, 0, 160, 120]])
    assert map_boxes_back(steps, boxes, state).tolist() == boxes.tolist()


def test_gamma_brightens_above_one_and_is_identity_at_one(build_ctx):
    image = make_image("gradient", 32, 32)
    assert (apply_preprocess([step(build_ctx, "gamma", gamma=1.0)], image)[0] == image).all()
    brighter = apply_preprocess([step(build_ctx, "gamma", gamma=2.0)], image)[0]
    assert brighter.mean() > image.mean()


@pytest.mark.parametrize(
    "bad,needle",
    [
        (np.zeros((10, 10), dtype=np.uint8), "(H, W, 3)"),  # grayscale
        (np.zeros((10, 10, 4), dtype=np.uint8), "(H, W, 3)"),  # RGBA
        (np.zeros((10, 10, 3), dtype=np.float32), "uint8"),
        (np.zeros((10, 10, 3), dtype=np.uint16), "uint8"),
        (np.zeros((0, 0, 3), dtype=np.uint8), "empty"),
        ("not an array", "(H, W, 3)"),
    ],
)
def test_validate_image_rejects_bad_input_with_a_helpful_message(bad, needle):
    with pytest.raises(ValueError, match="RGB uint8|empty") as err:
        validate_image(bad)
    assert needle in str(err.value)


def test_ensure_rgb_uint8_handles_non_contiguous_and_read_only_input(build_ctx):
    base = make_image("noise", 30, 40)
    flipped = base[:, ::-1]
    assert not flipped.flags["C_CONTIGUOUS"]
    read_only = make_image("noise", 30, 40, seed=1)
    read_only.setflags(write=False)
    snapshots = [flipped.copy(), read_only.copy()]
    for image, snapshot in zip([flipped, read_only], snapshots, strict=True):
        steps = [
            step(build_ctx, "ensure_rgb_uint8"),
            step(build_ctx, "resize_max_side", max_side=20),
            step(build_ctx, "clahe"),
            step(build_ctx, "gamma", gamma=1.5),
        ]
        out, _ = apply_preprocess(steps, image)
        assert out.flags["C_CONTIGUOUS"]
        assert (image == snapshot).all()


def test_all_zero_and_all_255_images_run_cleanly(build_ctx):
    for image in (np.zeros((40, 40, 3), np.uint8), np.full((40, 40, 3), 255, np.uint8)):
        steps = [step(build_ctx, "clahe"), step(build_ctx, "gamma", gamma=2.0)]
        with np.errstate(all="raise"):
            out, _ = apply_preprocess(steps, image)
        assert out.shape == image.shape


def test_bad_step_params_are_config_errors(build_ctx):
    from electronics_classifier.config import ConfigError

    with pytest.raises(ConfigError, match=r"params\.max_side"):
        step(build_ctx, "resize_max_side", max_side=0)
    with pytest.raises(ConfigError, match=r"params\.gamma"):
        step(build_ctx, "gamma", gamma=-1)
    with pytest.raises(ConfigError, match=r"params\.surprise"):
        step(build_ctx, "clahe", surprise=1)
