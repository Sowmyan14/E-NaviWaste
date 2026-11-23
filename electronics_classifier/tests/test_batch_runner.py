import csv
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml
from conftest import make_image
from PIL import Image

from electronics_classifier.schemas import (
    ClassificationResult,
    ClassifiedItem,
    SchemaOrgProductAttributes,
    WarningCode,
)

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "run_batch.py"
spec = importlib.util.spec_from_file_location("run_batch", SCRIPT)
run_batch = importlib.util.module_from_spec(spec)
sys.modules["run_batch"] = run_batch
spec.loader.exec_module(run_batch)


@pytest.fixture
def batch(tmp_path, monkeypatch):
    monkeypatch.setenv("EC_USE_FAKES", "1")
    images = tmp_path / "imgs"
    images.mkdir()
    Image.fromarray(make_image("noise", 240, 320)).save(images / "first.jpg")
    Image.fromarray(make_image("gradient", 200, 300)).save(images / "second.png")
    (images / "notes.txt").write_text("not an image")  # ignored: not an image suffix
    configs = []
    for name, min_score in (("baseline", 0.25), ("strict", 0.95)):
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump({"pipeline": {"detection_filters": [
            {"name": "score_threshold", "params": {"min_score": min_score}}]}}))  # fmt: skip
        configs.append(path)
    return images, configs, tmp_path / "out"


def read_summary(out):
    with (out / "summary.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_two_images_under_two_configs_give_four_result_folders_and_a_summary(batch):
    images, configs, out = batch
    args = [str(images), "--configs", *map(str, configs), "--out-dir", str(out)]
    assert run_batch.main(args) == 0
    for config in ("baseline", "strict"):
        for image in ("first", "second"):
            folder = out / config / image
            assert {p.name for p in folder.iterdir()} == {
                "annotated.png", "result.json", "resolved_config.yaml",
            }  # fmt: skip
    rows = read_summary(out)
    assert [(r["image"], r["config"]) for r in rows] == [
        ("first", "baseline"), ("second", "baseline"), ("first", "strict"), ("second", "strict"),
    ]  # fmt: skip


def test_summary_columns_and_values(batch):
    images, configs, out = batch
    run_batch.main([str(images), "--configs", *map(str, configs), "--out-dir", str(out)])
    with (out / "summary.csv").open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert header[:4] == ["image", "config", "items", "mean_detector_confidence"]
    assert header[4:-1] == [code.value for code in WarningCode] and header[-1] == "other_warnings"
    baseline, strict = read_summary(out)[0], read_summary(out)[2]
    assert baseline["items"] == "1" and baseline["mean_detector_confidence"] == "0.9000"
    assert baseline["BRAND_UNKNOWN"] == "1" and baseline["MODEL_UNKNOWN"] == "1"
    assert strict["items"] == "0" and strict["mean_detector_confidence"] == ""  # nothing survived


def make_item(item_id, confidence, warnings):
    return ClassifiedItem(
        id=item_id,
        bbox_xyxy=[0, 0, 10, 10],
        detector_phrase="router",
        detector_confidence=confidence,
        attributes=SchemaOrgProductAttributes(),
        warnings=warnings,
    )


def test_summary_row_counts_items_per_code_and_groups_custom_codes():
    result = ClassificationResult(
        image_width=10,
        image_height=10,
        items=[
            make_item("obj_1", 0.9, ["BRAND_UNKNOWN", "CUSTOM_A"]),
            make_item("obj_2", 0.5, ["BRAND_UNKNOWN", "SEGMENTATION_FAILED", "CUSTOM_B"]),
        ],
    )
    row = dict(zip(run_batch.HEADER, run_batch.summary_row("img", "cfg", result), strict=True))
    assert row["items"] == 2 and row["mean_detector_confidence"] == "0.7000"
    assert row["BRAND_UNKNOWN"] == 2 and row["SEGMENTATION_FAILED"] == 1
    assert row["MODEL_UNKNOWN"] == 0 and row["other_warnings"] == 2


def test_images_sharing_a_stem_get_distinct_result_folders(batch):
    images, configs, out = batch
    # first.jpg already exists, so this makes the stem "first" ambiguous
    Image.fromarray(make_image("noise", 100, 100)).save(images / "first.png")
    run_batch.main([str(images), "--configs", str(configs[0]), "--out-dir", str(out)])
    folders = {p.name for p in (out / "baseline").iterdir()}
    assert folders == {"first.jpg", "first.png", "second"}


def test_an_unreadable_image_is_skipped_with_a_message_and_the_rest_still_run(batch, capsys):
    images, configs, out = batch
    (images / "broken.jpg").write_bytes(b"not a jpeg")
    assert run_batch.main([str(images), "--configs", str(configs[0]), "--out-dir", str(out)]) == 0
    assert "skipped broken.jpg" in capsys.readouterr().err
    assert sorted(r["image"] for r in read_summary(out)) == ["first", "second"]


def test_bad_inputs_exit_2_with_a_one_line_error_before_any_image_is_processed(
    batch, tmp_path, capsys
):
    images, configs, out = batch
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"pipeline": {"nope": 1}}))
    empty = tmp_path / "empty"
    empty.mkdir()
    cases = [
        [str(tmp_path / "missing"), "--configs", str(configs[0])],  # not a folder
        [str(empty), "--configs", str(configs[0])],  # no images
        [str(images), "--configs", str(bad)],  # invalid config
        [str(images), "--configs", str(configs[0]), str(configs[0])],  # duplicate config names
    ]
    for args in cases:
        assert run_batch.main([*args, "--out-dir", str(out)]) == 2, args
        err = capsys.readouterr().err
        assert err.startswith("error: ") and err.count("\n") == 1
    assert not out.exists()  # nothing was written for any failed batch
