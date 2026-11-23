# electronics_classifier

An image-only household electronics classifier. Give it **one RGB photo**; it returns

1. an **annotated image** (a box, an optional mask and a label banner per item), and
2. a **structured JSON** list of the devices found, with Schema.org `Product` attributes
   (`brand`, `model`, `category`, `color`), the detector's confidence, and warnings.

Both outputs are built from the same records, so a label can never disagree with the JSON.

Every stage that a researcher might want to change (models, pre-processing, filters,
thresholds, prompts, post-processing, warning logic) is a **named, config-selected
component**. Nothing is hardcoded.

```text
photo -> preprocess -> detector -> sanitize -> filters -> ids
      -> segmenter+gate -> crop builder -> crop transforms
      -> VLM + extractor -> post-processors -> warning rules -> assembler -> JSON
                                                              \-> renderer -> annotated image
```

## Install

Python 3.10 to 3.12.

```bash
pip install -e .            # core: numpy, opencv, Pillow, pydantic, pyyaml, supervision
pip install -e '.[dino]'    # default detector (Grounding DINO): transformers, torch
pip install -e '.[vlm]'     # default VLM client: transformers, torch, qwen-vl-utils
pip install -e '.[sam]'     # optional SAM masks (segment-anything)
pip install -e '.[sam2]'    # optional SAM 2 masks (alternative to `sam`)
pip install -e '.[dev]'     # pytest, ruff
```

Heavy packages are imported lazily, so `import electronics_classifier`, the CLI help and the
whole unit-test suite work with **no GPU, no torch and no weights**. Weights are downloaded by
the libraries on first use and are never committed. The `config/` folder is found next to
`src/`, so install with `pip install -e` (or run from a source checkout).

## Run

```bash
python -m electronics_classifier.cli run photo.jpg --out-dir out
python -m electronics_classifier.cli run photo.jpg --preset fast --set rendering.output_format=jpeg
python -m electronics_classifier.cli run photo.jpg --save-intermediates   # per-stage overlays
python -m electronics_classifier.cli show-config --preset fast            # resolved config
python -m electronics_classifier.cli list-components                      # every component
```

`run` writes `annotated.png` (or `.jpeg`), `result.json` and `resolved_config.yaml` to the
output folder (default `out`). With `--save-intermediates` it also writes an `intermediates/`
folder: the pre-processed image, a detection overlay after every filter, each crop, the raw
VLM text, and the attributes before and after each post-processor. Exit code is `0` on
success and `2` on a bad image or bad configuration (one line, no traceback). When brand or
model could not be read, the CLI prints a tip to retake a closer photo of the label side.

Python API:

```python
from electronics_classifier import ElectronicsClassificationPipeline

pipeline = ElectronicsClassificationPipeline(presets=["fast"], sets=["run.seed=1"])
annotated_rgb, result = pipeline.run(image_rgb)          # (H, W, 3) uint8, RGB
annotated_rgb, result, trace = pipeline.run(image_rgb, return_trace=True)
print(result.model_dump_json(by_alias=True, indent=2))
```

The Python API is strict: anything but a non-empty `(H, W, 3) uint8` array raises
`ValueError`. The CLI loads files with Pillow, applies the EXIF rotation and converts to RGB
first. Ready-made components can be injected by kind
(`ElectronicsClassificationPipeline(detector=..., vlm_client=...)`).

`EC_USE_FAKES=1` swaps the detector and the VLM for fakes so you can smoke-test offline. The
result's `meta.components` then says `fake`, so it cannot be mistaken for a real run.

## Configuration reference

One YAML file describes the whole run: `config/pipeline.yaml`. Its sections mirror the stages:

| Section | What it controls |
|---|---|
| `run` | `device` (`auto` picks cuda, then mps, then cpu) and `seed` |
| `vocabulary.categories` | The one category list used by the detector, the VLM prompt, the post-processors and the rules. `electronic device` is always appended. Entries are lower-cased, stripped, de-duplicated; dots are removed because `.` separates Grounding DINO phrases |
| `pipeline.image_preprocess` | Ordered steps before detection: `ensure_rgb_uint8`, `resize_max_side`, `clahe`, `gamma` |
| `pipeline.detector` | `grounding_dino` (default) or `fake`; returns raw detections above a low floor score |
| `pipeline.detection_filters` | Ordered: `score_threshold`, `min_size`, `min_area_ratio`, `nms`, `top_k` |
| `pipeline.segmenter` / `segmentation_gate` | `none` (default), `sam`, `sam2`; the gate (`always`, `overlap`) picks which items get a mask |
| `pipeline.crop` | `padding_ratio`, `include_full_image_context`, `max_image_side_px` |
| `pipeline.crop_transforms` | Ordered: `identity`, `upscale_min_side`, `sharpen`, `clahe` |
| `pipeline.vlm` | `client` (`transformers`, `fake`), `max_retries`, `system_prompt_file` (a template containing `{categories}`) |
| `pipeline.attribute_postprocess` | Ordered: `normalize_unknown`, `category_vocab`, `brand_aliases` |
| `pipeline.warning_rules` | Ordered: `unknown_field`, `low_detector_confidence`, `small_crop`, `overlap`, `category_oov` |
| `rendering` | `output_format`, `label_template`, `stroke_scale`, `font_scale`, `mask_alpha` |

`list-components` prints every component with its parameters, defaults and descriptions.
Detection thresholds live **only** in the filters: the detector just returns everything above
its low floor, so you can tune filters without re-running the model.

### Layering rules

Settings are layered; later layers win:

1. built-in `config/pipeline.yaml`
2. each `--preset NAME` (from `config/presets/`, or a path)
3. each `--override FILE.yaml`
4. each `--set dotted.key=value` (the value is parsed as YAML)

How layers merge:

- **Dicts are deep-merged. Lists are replaced whole**, which keeps ordered step lists
  unambiguous. To change one filter, restate the whole `detection_filters` list (the `fast`
  preset does this). `--set` can replace a list with a YAML flow value, for example
  `--set 'pipeline.detection_filters=[{name: top_k, params: {k: 5}}]'`.
- **Swapping a component replaces its whole block.** If a layer sets a component `name` that
  differs from the layer below, that component's `{name, params}` is replaced instead of
  merged, so the old component's parameters never leak into the new one. (Setting the same
  name only changes the keys you give.)
- The merged config is validated with strict models (unknown keys are errors). Component
  parameters are validated when the pipeline is built. Every error names the full config path
  (for example `pipeline.detector.params.raw_box_threshold`) and is raised **at build time**,
  never in the middle of inference.

Every run writes `resolved_config.yaml` and records `meta.config_hash` (the first 12 hex
characters of a SHA-256 of the canonical resolved config) plus `meta.components` (kind to
component names) in `result.json`, so any output can be reproduced and compared.

### Presets

| Preset | What it does |
|---|---|
| `fast` | Smaller VLM (`Qwen/Qwen2-VL-2B-Instruct`, 128 new tokens), `resize_max_side` 960, `top_k` 10, no SAM |
| `accurate_sam` | SAM `vit_b` masks for overlapping items and `padding_ratio` 0.25. You must download the checkpoint to `checkpoints/sam_vit_b_01ec64.pth` (or override the path); building fails with a clear `ConfigError` until it exists |

## How to add or swap a component

1. Write a class implementing the interface for its kind, with an inner `Params(BaseModel)` (`extra="forbid"`, every field has a `description`).
2. Register it: `@register("detector", "my_detector")` inside the package, **or** reference it in config by dotted path: `name: "my_pkg.my_module:MyDetector"`.
3. Select it in config (or a preset/override file): `pipeline.detector: {name: my_detector, params: {...}}`.
4. Run `cli list-components` to confirm it appears; run `cli show-config` to confirm the resolved config.
5. Compare against the baseline by comparing `result.json` files; each carries `meta.config_hash`.

**Swapping the detection model** (e.g. Grounding DINO base) is a one-line config change:
`pipeline.detector.params.model_id`. Switching the whole recipe is a preset (`--preset fast`).

Interfaces live in `src/electronics_classifier/interfaces.py`. A component is constructed as
`Component(params, build_ctx)`; the build context carries the shared vocabulary, the resolved
device, the seed and the read-only resolved config.

## Output example

```json
{
  "image_width": 1280,
  "image_height": 960,
  "items": [
    {
      "id": "obj_1",
      "bbox_xyxy": [120, 85, 460, 310],
      "detector_phrase": "router",
      "detector_confidence": 0.81,
      "attributes": {
        "@context": "https://schema.org",
        "@type": "Product",
        "brand": "TP-Link",
        "model": "unknown",
        "category": "router",
        "color": "black"
      },
      "warnings": ["MODEL_UNKNOWN"]
    }
  ],
  "annotated_image_format": "png",
  "meta": {
    "config_hash": "3f9a1c0b7d2e",
    "components": {"detector": "grounding_dino", "vlm_client": "transformers"}
  }
}
```

- `bbox_xyxy` is the detector's box in **original-image pixels** (not the padded crop).
- Only the detector's score is reported as confidence. The VLM gives no calibrated score.
- Items are numbered `obj_1, obj_2, ...` ordered by `(y1, x1)`, so runs are deterministic.
- The JSON keeps the original Unicode (UTF-8); only the on-image label replaces characters
  OpenCV cannot draw with `?`.

Warning codes: `BRAND_UNKNOWN`, `MODEL_UNKNOWN`, `COLOR_UNKNOWN`, `LOW_DETECTOR_CONFIDENCE`,
`CATEGORY_OUT_OF_VOCAB`, `VLM_PARSE_FAILED`, `SMALL_CROP`, `PARTIALLY_OCCLUDED_OR_OVERLAPPING`
and `SEGMENTATION_FAILED` (an item the gate chose for a mask ended up without one, so its crop
is the plain rectangle). A single item's trouble never aborts a run: a failed VLM call or an
unparseable reply gives all-`unknown` attributes plus `VLM_PARSE_FAILED`.

## Photo tips

- Use good, even lighting; avoid glare and strong shadows on the device.
- Show the **label side**: brand and model are read only from visible text and logos.
- If a result says `BRAND_UNKNOWN` or `MODEL_UNKNOWN`, take a **close-up** of the label.
- Keep the device sharp (no motion blur) and large enough in the frame. Very small objects in
  very large photos can be missed after `resize_max_side` shrinks the detector input; compare
  with `--set 'pipeline.image_preprocess=[{name: ensure_rgb_uint8}]'`.
- A photo of a screen or a printed picture of a device can be mistaken for a real device.

## Tests

```bash
pytest -q                         # everything except Tier C: all offline, no GPU, no weights
pytest tests/test_failure_modes.py -q
pytest tests/slow -m slow -s      # Tier C: real models and real photos (optional)
ruff check src tests
```

- **Tier A** (`test_failure_modes.py`, A01 to A34): hostile input or output the system must
  survive. All pass.
- **Tier B** (B01 to B08, same file): known design weaknesses. Each asserts the *ideal*
  behaviour and is `xfail(strict=True)`; the reason names the suggested fix. When a weakness
  is fixed the test becomes an XPASS and fails the run on purpose, so the marker gets removed.
  Expected result: **8 xfailed**.
- **Tier C** (`tests/slow/`, C01 to C14): real-model failures, informational only. Photos
  cannot be generated, so `tests/fixtures/failure_images/README.md` lists every file needed;
  a test whose photo is missing is skipped with the file name.

A note on A32: greedy NMS only lets a box suppress boxes with *lower* confidence, and
`score_threshold` only removes low-confidence boxes, so those two keep the same set in either
order. The order-matters test therefore uses `nms` against `min_size` (and `top_k`).

## Design notes and choices

These points were not fixed by the plan and were decided with the maintainer:

- Crops, the context image, SAM masks and the annotated output come from the **original**
  image; pre-processing only changes what the detector sees.
- Out-of-range detector confidence (outside 0 to 1) drops the detection; phrases outside the
  vocabulary are dropped both by the detector backend and by the always-on sanitize step.
- `gamma` is `out = in^(1/gamma)`: above 1 brightens.
- Only unparseable VLM output is retried (`max_retries`); a VLM exception goes straight to
  fallback attributes.
- `SEGMENTATION_FAILED` is a built-in warning code beyond the original list.
- `category_oov` is available as a rule but is not in the default config, because the
  `category_vocab` post-processor already fixes categories. If you remove `category_vocab`,
  add `category_oov` or out-of-vocabulary categories will pass through without a warning.
- `brand_aliases` reads a YAML or JSON mapping file; aliases match case-insensitively and
  ignore hyphens, underscores and extra spaces.
- Both `sam` and `sam2` are real segmenters behind one interface (`pip install -e '.[sam]'`
  or `'.[sam2]'`), but they, and the real `transformers` VLM client, are verified in this
  repository only against stub models, not against real weights.
- The repository's own license is **MIT**.

## Licensing

- This repository: **MIT** (see `LICENSE`).
- Default stack: Grounding DINO (Apache-2.0), `supervision` (MIT) and `transformers`
  (Apache-2.0).
- **YOLO-World / Ultralytics is deliberately not included**: Ultralytics is AGPL-3.0 and would
  impose AGPL obligations on a combined work. Nothing here imports it.
- Model weights have their own licenses. **Verify the license of the VLM weights you choose
  (for example Qwen2.5-VL or Qwen2-VL) and of the SAM weights before distributing anything.**
