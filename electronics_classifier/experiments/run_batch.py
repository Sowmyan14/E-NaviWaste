"""Run every image in a folder under every override config and compare the results.

    python experiments/run_batch.py IMAGES_DIR --configs a.yaml b.yaml --out-dir OUT

Each ``--configs`` file is an override layered on the built-in config (the same files you would
pass to ``cli run --override``). Output:

    OUT/<config_name>/<image_name>/annotated.png, result.json, resolved_config.yaml
    OUT/summary.csv   one row per (image, config): item count, mean detector confidence and
                      the number of items carrying each warning code

``EC_USE_FAKES=1`` swaps the detector and VLM for fakes (offline check of the whole batch).
Each config's pipeline is built once and reused for all images.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from electronics_classifier.cli import FAKE_SETS, InputError, load_image, write_outputs
from electronics_classifier.config import ConfigError
from electronics_classifier.pipeline import ElectronicsClassificationPipeline
from electronics_classifier.schemas import ClassificationResult, WarningCode

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
WARNING_COLUMNS = [code.value for code in WarningCode]
OTHER_WARNINGS = "other_warnings"  # any code that is not a built-in WarningCode
HEADER = ["image", "config", "items", "mean_detector_confidence", *WARNING_COLUMNS, OTHER_WARNINGS]


def find_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def image_names(images: Sequence[Path]) -> dict[Path, str]:
    """The stem, or the full file name when two images share a stem (a.jpg and a.png)."""
    stems = Counter(p.stem for p in images)
    return {p: p.stem if stems[p.stem] == 1 else p.name for p in images}


def summary_row(image: str, config: str, result: ClassificationResult) -> list[object]:
    confidences = [item.detector_confidence for item in result.items]
    mean = f"{sum(confidences) / len(confidences):.4f}" if confidences else ""
    counts: Counter[str] = Counter()
    for item in result.items:
        counts.update(set(item.warnings))  # items carrying the code, not occurrences
    other = sum(n for code, n in counts.items() if code not in WARNING_COLUMNS)
    return [image, config, len(result.items), mean, *(counts[c] for c in WARNING_COLUMNS), other]


def run_batch(images_dir: Path, configs: Sequence[Path], out_dir: Path) -> list[list[object]]:
    images = find_images(images_dir)
    if not images:
        raise InputError(f"no images found in {images_dir}")
    names = image_names(images)
    sets = list(FAKE_SETS) if os.environ.get("EC_USE_FAKES") == "1" else []

    # build every pipeline first, so a bad config fails before any image is processed
    pipelines = {
        config: ElectronicsClassificationPipeline(overrides=[config], sets=sets)
        for config in configs
    }
    rows: list[list[object]] = []
    for config, pipeline in pipelines.items():
        for path in images:
            try:
                image = load_image(path)
            except InputError as exc:
                print(f"warning: skipped {path.name}: {exc}", file=sys.stderr)
                continue
            annotated, result = pipeline.run(image)
            target = out_dir / config.stem / names[path]
            write_outputs(target, annotated, result, pipeline.config)
            rows.append(summary_row(names[path], config.stem, result))

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run every image under every override config.")
    parser.add_argument("images_dir", metavar="IMAGES_DIR", type=Path)
    parser.add_argument("--configs", nargs="+", required=True, type=Path, metavar="FILE",
                        help="override YAML files; each becomes one result folder")
    parser.add_argument("--out-dir", required=True, type=Path, metavar="OUT")
    args = parser.parse_args(argv)
    try:
        if not args.images_dir.is_dir():
            raise InputError(f"not a folder: {args.images_dir}")
        if len({c.stem for c in args.configs}) != len(args.configs):
            raise ConfigError("config files must have distinct names (stems)")
        rows = run_batch(args.images_dir, args.configs, args.out_dir)
    except (ConfigError, InputError, ImportError, OSError) as exc:
        print(f"error: {' '.join(str(exc).split())}", file=sys.stderr)
        return 2
    print(f"{len(rows)} run(s). Wrote {args.out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
