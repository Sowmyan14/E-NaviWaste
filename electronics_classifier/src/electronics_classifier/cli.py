"""Command line interface: ``run``, ``show-config``, ``list-components``.

Exit codes: 0 on success, 2 on bad input or configuration (one-line error, no traceback).
Set ``EC_USE_FAKES=1`` to swap the detector and the VLM for fakes (offline smoke tests).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from PIL import Image, ImageOps

from .assembler import result_to_json
from .config import ConfigError, config_hash, dump_yaml, load_config
from .pipeline import ElectronicsClassificationPipeline
from .registry import describe_components
from .renderer import ascii_only
from .schemas import ClassificationResult, RawDetection
from .trace import PipelineTrace

FAKE_SETS = ("pipeline.detector.name=fake", "pipeline.vlm.client.name=fake")


class InputError(Exception):
    """The input image could not be read."""


def _one_line(text: object) -> str:
    return " ".join(str(text).split())


# --- image loading -------------------------------------------------------------------


def _to_rgb_uint8(image: Image.Image) -> np.ndarray:
    if image.mode.startswith("I"):  # 16/32-bit grey: scale down instead of clipping to white
        values = np.asarray(image, dtype=np.float64)
        scale = 256.0 if image.mode.startswith("I;16") else max(values.max() / 255.0, 1.0)
        image = Image.fromarray(np.clip(values / scale, 0, 255).astype(np.uint8))
    return np.array(image.convert("RGB"))


def load_image(path: str | Path) -> np.ndarray:
    """Pillow, upright via EXIF, converted to ``(H, W, 3) uint8`` RGB."""
    try:
        with Image.open(path) as opened:
            return _to_rgb_uint8(ImageOps.exif_transpose(opened))
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InputError(f"cannot read image {path}: {_one_line(exc)}") from None


# --- output --------------------------------------------------------------------------


def _overlay(image_rgb: np.ndarray, detections: Sequence[RawDetection], path: Path) -> None:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # supervision draws on BGR
    if detections:
        found = sv.Detections(
            xyxy=np.array([d.bbox_xyxy for d in detections], dtype=np.float64),
            class_id=np.arange(len(detections)),
        )
        labels = [ascii_only(f"{d.phrase} {d.confidence:.2f}") for d in detections]
        bgr = sv.BoxAnnotator().annotate(bgr, found)
        bgr = sv.LabelAnnotator().annotate(bgr, found, labels=labels)
    cv2.imwrite(str(path), bgr)


def save_trace(trace: PipelineTrace, image_rgb: np.ndarray, out_dir: Path) -> Path:
    """Write the intermediates: overlays per stage, crops, raw VLM text and attributes."""
    folder = out_dir / "intermediates"
    folder.mkdir(parents=True, exist_ok=True)
    if trace.preprocessed_image is not None:
        Image.fromarray(trace.preprocessed_image).save(folder / "00_preprocessed.png")
    _overlay(image_rgb, trace.mapped_detections, folder / "01_raw_detections.png")
    for number, (name, detections) in enumerate(trace.filter_steps, 1):
        _overlay(image_rgb, detections, folder / f"02_filter_{number}_{name}.png")
    for item in trace.prepared_items:
        item_id = item.detection.detection_id
        Image.fromarray(item.crop_rgb).save(folder / f"crop_{item_id}.png")
        replies = "\n--- retry ---\n".join(trace.vlm_replies.get(item_id, []))
        (folder / f"vlm_{item_id}.txt").write_text(replies, encoding="utf-8")
    summary = {
        "attributes_before": trace.attributes_before,
        "attributes_after_each_postprocessor": trace.attributes_after,
    }
    (folder / "attributes.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return folder


def write_outputs(
    out_dir: Path,
    annotated: np.ndarray,
    result: ClassificationResult,
    resolved: dict,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / f"annotated.{result.annotated_image_format}"
    Image.fromarray(annotated).save(image_path)
    json_path = out_dir / "result.json"
    json_path.write_text(result_to_json(result), encoding="utf-8")
    config_path = out_dir / "resolved_config.yaml"
    config_path.write_text(dump_yaml(resolved) + f"# config_hash: {config_hash(resolved)}\n")
    return [image_path, json_path, config_path]


# --- commands ------------------------------------------------------------------------


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", action="append", default=[], metavar="NAME",
                        help="preset name from config/presets/ or a YAML path (repeatable)")
    parser.add_argument("--override", action="append", default=[], metavar="FILE",
                        help="override YAML file (repeatable)")
    parser.add_argument("--set", dest="sets", action="append", default=[], metavar="K=V",
                        help="dotted.key=value, value parsed as YAML (repeatable)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="electronics_classifier.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="classify the electronics in one image")
    run.add_argument("input", metavar="INPUT", help="path to the image")
    run.add_argument("--out-dir", default="out", metavar="OUT", help="output folder (default: out)")
    _add_config_args(run)
    run.add_argument("--save-intermediates", action="store_true",
                     help="also write per-stage overlays, crops and raw VLM text")

    show = sub.add_parser("show-config", help="print the resolved config without running anything")
    _add_config_args(show)

    sub.add_parser("list-components", help="list every component kind, name and its parameters")
    return parser


def _show_config(args: argparse.Namespace) -> int:
    resolved = load_config(args.preset, args.override, args.sets)
    sys.stdout.write(dump_yaml(resolved))
    sys.stdout.write(f"# config_hash: {config_hash(resolved)}\n")
    return 0


def _list_components() -> int:
    for kind, components in describe_components().items():
        print(kind)
        if not components:
            print("  (none registered)")
        for name, params in components.items():
            print(f"  {name}")
            for field_name, type_name, default, description in params:
                print(f"    {field_name}: {type_name} = {default}  {description}".rstrip())
    return 0


def _run(args: argparse.Namespace) -> int:
    sets = list(args.sets)
    if os.environ.get("EC_USE_FAKES") == "1":
        sets += FAKE_SETS  # last, so they win over the config
    image = load_image(args.input)  # cheap check first: a bad path must not wait for model loads
    pipeline = ElectronicsClassificationPipeline(
        presets=args.preset, overrides=args.override, sets=sets
    )
    outputs = pipeline.run(image, return_trace=args.save_intermediates)
    annotated, result = outputs[0], outputs[1]

    out_dir = Path(args.out_dir)
    written = write_outputs(out_dir, annotated, result, pipeline.config)
    if args.save_intermediates:
        written.append(save_trace(outputs[2], image, out_dir))
    print(f"{len(result.items)} item(s) found. Wrote: " + ", ".join(str(p) for p in written))
    flagged = {"BRAND_UNKNOWN", "MODEL_UNKNOWN"}
    if any(flagged & set(item.warnings) for item in result.items):
        print("Tip: brand or model could not be read; try a closer photo of the label side.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "show-config":
            return _show_config(args)
        if args.command == "list-components":
            return _list_components()
        if args.command == "run":
            return _run(args)
    except (ConfigError, InputError, ImportError, OSError) as exc:
        print(f"error: {_one_line(exc)}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
