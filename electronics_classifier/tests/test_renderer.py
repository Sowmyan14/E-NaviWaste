import json
import logging

import numpy as np
import pytest
from conftest import make_image

from electronics_classifier.assembler import result_to_json
from electronics_classifier.config import ConfigError
from electronics_classifier.renderer import (
    MAX_LABEL_CHARS,
    PALETTE,
    RenderStyle,
    format_label,
    plan_banners,
    render_annotated_image,
    validate_label_template,
)
from electronics_classifier.schemas import (
    ClassificationResult,
    ClassifiedItem,
    SchemaOrgProductAttributes,
)

TEMPLATE = "{id} {category} | {brand} {model} | {color} | {confidence:.2f}"
STYLE = RenderStyle(label_template=TEMPLATE)


def make_item(item_id="obj_1", box=(100, 100, 300, 300), warnings=(), confidence=0.81, **attrs):
    fields = {"brand": "TP-Link", "model": "Archer", "category": "router", "color": "black"}
    return ClassifiedItem(
        id=item_id,
        bbox_xyxy=list(box),
        detector_phrase="router",
        detector_confidence=confidence,
        attributes=SchemaOrgProductAttributes(**{**fields, **attrs}),
        warnings=list(warnings),
    )


def rgb(index):
    return list(PALETTE.by_idx(index).as_rgb())


# --- label text ----------------------------------------------------------------------


def test_format_label_uses_the_template_and_a_warning_prefix():
    assert format_label(make_item(), TEMPLATE) == "obj_1 router | TP-Link Archer | black | 0.81"
    flagged = make_item(warnings=["MODEL_UNKNOWN"])
    assert format_label(flagged, TEMPLATE).startswith("! obj_1 router")
    assert format_label(make_item(), "{brand}") == "TP-Link"


def test_label_text_matches_the_json_fields_of_the_same_item():
    result = ClassificationResult(
        image_width=640, image_height=480, items=[make_item(model="XR-5000", color="dark blue")]
    )
    payload = json.loads(result_to_json(result))["items"][0]
    expected = TEMPLATE.format(
        id=payload["id"],
        category=payload["attributes"]["category"],
        brand=payload["attributes"]["brand"],
        model=payload["attributes"]["model"],
        color=payload["attributes"]["color"],
        confidence=payload["detector_confidence"],
    )
    assert format_label(result.items[0], TEMPLATE) == expected
    [banner] = plan_banners(result.items, 480, 640, STYLE)
    assert banner.text == expected and banner.item_id == payload["id"]


def test_a22_non_ascii_becomes_question_marks_but_the_data_is_untouched():
    item = make_item(brand="Samsung 삼성 電子 🙂")
    label = format_label(item, TEMPLATE)
    assert label == "obj_1 router | Samsung ?? ?? ? Archer | black | 0.81"
    assert item.attributes.brand == "Samsung 삼성 電子 🙂"


def test_a23_braces_in_attribute_text_stay_literal():
    item = make_item(brand="{id}{0}", model="{}", color="{confidence:.2f}")
    assert format_label(item, TEMPLATE) == "obj_1 router | {id}{0} {} | {confidence:.2f} | 0.81"


def test_control_characters_become_question_marks_and_labels_are_truncated():
    nasty = make_item(brand="a\nb\x00c\x1bd")
    assert format_label(nasty, "{brand}") == "a?b?c?d"
    long = make_item(brand="x" * 500)
    label = format_label(long, "{brand}")
    assert len(label) == MAX_LABEL_CHARS and label.endswith("...")
    flagged = format_label(make_item(brand="x" * 500, warnings=["W"]), "{brand}")
    assert len(flagged) == MAX_LABEL_CHARS and flagged.startswith("! ")


def test_format_label_never_raises_even_for_a_bad_template(caplog):
    with caplog.at_level(logging.WARNING):
        assert format_label(make_item(), "{nope}") == "obj_1"
        assert format_label(make_item(), "{confidence:zz}") == "obj_1"
        assert format_label(make_item(), "{") == "obj_1"
    assert caplog.records


@pytest.mark.parametrize(
    "bad",
    ["{nope}", "{}", "{0}", "{brand.upper}", "{id[0]}", "{confidence:zz}", "{unclosed", "}"],
)
def test_a30_bad_label_templates_are_rejected_at_build_time(bad):
    with pytest.raises(ConfigError, match=r"rendering\.label_template"):
        validate_label_template(bad)


def test_valid_label_templates_are_accepted():
    for good in (TEMPLATE, "{id}", "no placeholders", "{confidence:.1f} {{literal}}"):
        assert validate_label_template(good) == good


# --- drawing -------------------------------------------------------------------------


def test_render_returns_a_new_array_of_the_same_shape_and_leaves_the_input_alone():
    image = make_image("noise", 400, 600)
    snapshot = image.copy()
    out = render_annotated_image(image, [make_item()], {}, STYLE)
    assert out.shape == image.shape and out.dtype == np.uint8
    assert out is not image and not np.shares_memory(out, image)
    assert (image == snapshot).all()
    assert (out != image).any()


def test_read_only_and_non_contiguous_inputs_are_accepted():
    base = make_image("noise", 300, 300)
    for image in (base[:, ::-1], np.require(base, requirements="C")):
        image.setflags(write=False)
        out = render_annotated_image(image, [make_item(box=(50, 80, 200, 250))], {}, STYLE)
        assert out.shape == image.shape


def test_no_items_gives_an_unmodified_copy():
    image = make_image("noise", 100, 100)
    out = render_annotated_image(image, [], {}, STYLE)
    assert (out == image).all() and out is not image


def test_boxes_use_the_palette_colour_in_rgb_order():
    image = np.full((400, 400, 3), 100, dtype=np.uint8)
    out = render_annotated_image(image, [make_item(box=(100, 150, 300, 350))], {}, STYLE)
    column = out[250, 298:302]  # the vertical box edge at x = 300
    assert (column == rgb(0)).all(axis=1).any()
    assert (out[250, 200] == 100).all()  # the inside is untouched without a mask


def test_each_item_gets_its_own_deterministic_colour():
    image = np.full((400, 400, 3), 100, dtype=np.uint8)
    items = [
        make_item("obj_1", box=(20, 150, 120, 350)),
        make_item("obj_2", box=(200, 150, 300, 350)),
    ]
    first = render_annotated_image(image, items, {}, STYLE)
    assert (first[250, 118:122] == rgb(0)).all(axis=1).any()
    assert (first[250, 298:302] == rgb(1)).all(axis=1).any()
    assert (render_annotated_image(image, items, {}, STYLE) == first).all()


def test_masks_are_blended_at_mask_alpha_and_only_where_the_mask_is():
    image = np.full((400, 400, 3), 100, dtype=np.uint8)
    mask = np.zeros((400, 400), dtype=bool)
    mask[150:250, 150:250] = True
    item = make_item(box=(100, 100, 300, 300))
    style = RenderStyle(TEMPLATE, mask_alpha=0.5)
    out = render_annotated_image(image, [item], {"obj_1": mask}, style)
    expected = (0.5 * 100 + 0.5 * np.array(rgb(0))).round()
    assert np.abs(out[200, 200].astype(int) - expected).max() <= 2
    assert (out[275, 275] == 100).all()  # inside the box, outside the mask

    invisible = RenderStyle(TEMPLATE, mask_alpha=0)
    none = render_annotated_image(image, [item], {"obj_1": mask}, invisible)
    assert (none[200, 200] == 100).all()


@pytest.mark.parametrize(
    "bad",
    [np.zeros((400, 400), dtype=bool), np.ones((10, 10), dtype=bool), None, "mask"],
)
def test_unusable_masks_are_ignored(bad):
    image = np.full((400, 400, 3), 100, dtype=np.uint8)
    out = render_annotated_image(image, [make_item()], {"obj_1": bad}, STYLE)
    assert (out[200, 200] == 100).all()


def test_banner_is_drawn_in_the_item_colour_above_the_box():
    image = np.full((400, 400, 3), 100, dtype=np.uint8)
    item = make_item(box=(100, 150, 300, 350))
    out = render_annotated_image(image, [item], {}, STYLE)
    [banner] = plan_banners([item], 400, 400, STYLE)
    assert banner.y + banner.h <= 150  # above the box
    assert (out[banner.y + 1, banner.x + 1] == rgb(0)).all()


# --- banner placement ----------------------------------------------------------------


def inside(b, image_h, image_w):
    return 0 <= b.x and 0 <= b.y and b.x + b.w <= image_w and b.y + b.h <= image_h


def test_banner_near_the_top_goes_inside_the_box_and_near_the_right_edge_is_clamped():
    top = make_item("obj_1", box=(50, 0, 200, 100))
    right = make_item("obj_2", box=(590, 200, 640, 260))
    banners = plan_banners([top, right], 480, 640, STYLE)
    assert banners[0].y == 0  # no room above: inside the box top
    assert banners[1].x + banners[1].w <= 640 and banners[1].x < 590  # pulled back from the edge
    assert all(inside(b, 480, 640) for b in banners)


def test_banner_at_the_bottom_edge_stays_in_frame():
    low = make_item(box=(10, 470, 100, 480))
    [banner] = plan_banners([low], 480, 640, STYLE)
    assert inside(banner, 480, 640)


def test_overlapping_banners_are_staggered():
    twins = [make_item(f"obj_{i}", box=(100, 200, 300, 400)) for i in (1, 2, 3)]
    banners = plan_banners(twins, 500, 800, STYLE)
    for i, a in enumerate(banners):
        for b in banners[i + 1 :]:
            separated = a.x + a.w <= b.x or b.x + b.w <= a.x or a.y + a.h <= b.y or b.y + b.h <= a.y
            assert separated


def test_a28_fifty_overlapping_detections_and_corner_items_keep_every_banner_in_frame():
    rng = np.random.default_rng(0)
    xs, ys = rng.integers(0, 520, 46), rng.integers(0, 420, 46)
    items = [
        make_item(
            f"obj_{i}",
            box=(int(x), int(y), int(x) + 80, int(y) + 60),
            warnings=["W"] if i % 2 else [],
        )
        for i, (x, y) in enumerate(zip(xs, ys, strict=True))
    ]
    corners = [(0, 0, 40, 40), (600, 0, 640, 40), (0, 440, 40, 480), (600, 440, 640, 480)]
    items += [make_item(f"obj_c{i}", box=c) for i, c in enumerate(corners)]
    image = make_image("noise", 480, 640)
    assert len(items) == 50
    banners = plan_banners(items, 480, 640, STYLE)
    assert len(banners) == 50 and all(inside(b, 480, 640) for b in banners)
    out = render_annotated_image(image, items, {}, STYLE)
    assert out.shape == image.shape


def test_a29_a_label_wider_than_the_image_is_truncated_and_clamped():
    item = make_item(box=(120, 30, 190, 90), brand="VeryLongBrandName" * 3, model="M" * 40)
    [banner] = plan_banners([item], 120, 200, STYLE)
    assert inside(banner, 120, 200) and banner.w <= 200
    assert banner.text.endswith("...") and len(banner.text) < len(format_label(item, TEMPLATE))
    out = render_annotated_image(make_image("noise", 120, 200), [item], {}, STYLE)
    assert out.shape == (120, 200, 3)


@pytest.mark.parametrize("size", [(1, 1), (5, 5), (20, 3), (3, 20)])
def test_tiny_images_never_raise(size):
    h, w = size
    item = make_item(box=(0, 0, max(w, 1), max(h, 1)))
    out = render_annotated_image(make_image("solid", h, w), [item], {}, STYLE)
    assert out.shape == (h, w, 3)
    assert all(inside(b, h, w) for b in plan_banners([item], h, w, STYLE))


def test_stroke_and_font_scale_follow_the_config_and_image_size():
    item = make_item(box=(100, 150, 300, 350))
    small = plan_banners([item], 400, 400, RenderStyle(TEMPLATE, font_scale=1.0))[0]
    large = plan_banners([item], 400, 400, RenderStyle(TEMPLATE, font_scale=2.0))[0]
    assert large.h > small.h and large.w >= small.w
    big_image = plan_banners([item], 2000, 2000, STYLE)[0]
    assert big_image.h > small.h  # text grows with the image


def test_render_style_from_config():
    style = RenderStyle.from_config(
        {"label_template": "{id}", "stroke_scale": 2, "font_scale": 1.5, "mask_alpha": 0.2}
    )
    assert style == RenderStyle("{id}", 2.0, 1.5, 0.2)
