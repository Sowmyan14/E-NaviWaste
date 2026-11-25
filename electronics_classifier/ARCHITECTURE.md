# Architecture

This document explains how `electronics_classifier` is built and what happens during one
forward pass (one photo in, annotated image and JSON out). Read it before changing the
pipeline. For install, CLI flags and the full config reference, see `README.md`.

## 1. The idea in one paragraph

The classifier is a **cascade of models** with a thin layer of rules around it (the default is
**Grounded-SAM + RAM++**, after IDEA-Research/Grounded-Segment-Anything):

1. A **detector** finds *where* devices are. RAM++ (Recognize Anything) tags the photo, those
   tags become the prompt for Grounding DINO (zero-shot), and DINO returns boxes with a phrase
   such as `router` or `kettle`. The phrases come from the image, not from a fixed list.
2. A **segmenter** (SAM) turns each surviving box into a mask, so the crop can hide neighbours.
3. A **vision-language model (VLM)** looks at the masked crop of each box. It is told the
   detector's phrase, **validates** it (is this really an electronic device? what is it called?),
   and reads `brand`, `model`, `category`, `color`.
4. Everything else is deterministic code: **filters** clean up the boxes, **post-processors**
   clean up the VLM's answer, **warning rules** flag doubtful results, and an **assembler** and
   **renderer** produce the final JSON and annotated image from the same records.

The previous pipeline (Grounding DINO prompted with a fixed vocabulary, no masks, vocabulary-locked
categories) is the `baseline` preset (`--preset baseline`). Where this document says "default" it
means Grounded-SAM + RAM++; where it names `grounding_dino`, that is the baseline detector.

The key design principle: **every stage a researcher might want to change is a named,
config-selected component.** The pipeline class itself contains no model logic. It only wires
components together in a fixed order.

## 2. The big picture

```text
                      one RGB photo (H, W, 3) uint8
                                  |
 [A] image_preprocess  --->  detector's input (e.g. resized)        (steps, ordered)
 [B] detector          --->  raw boxes + phrase + score             (one component)
        map boxes back to original pixels, sanitize
 [C] detection_filters --->  clean boxes, then ids obj_1..obj_N     (steps, ordered)
                                  |
       for the whole image                       for each item (box)
 [D] segmenter + gate  --->  optional masks
 [E] crop builder      --->  padded crop + full-scene context
 [F] crop_transforms   --->  crop tweaked for the VLM               (steps, ordered)
                                  |
 [G] VLM + extractor   --->  {brand, model, category, color}        (one component + parser)
 [H] attribute_postprocess -> cleaned attributes                    (steps, ordered)
 [I] warning_rules     --->  warning codes                          (steps, ordered)
                                  |
 [J] assembler         --->  ClassificationResult  (-> result.json)
 [K] renderer          --->  annotated image       (-> annotated.png)
```

The letters match the comments in `pipeline.py::run`, so you can search for `[G]` and land on
the right code.

## 3. Code map

All paths are relative to `src/electronics_classifier/`.

| File | Role |
|---|---|
| `pipeline.py` | `ElectronicsClassificationPipeline`: builds every component once, then `run()` chains stages A to K. **Start reading here.** |
| `interfaces.py` | The `Protocol` for each component kind (`Detector`, `DetectionFilter`, `VLMClient`, ...) and the context objects (`BuildContext`, `ItemContext`, ...) |
| `schemas.py` | The data types that flow between stages: `RawDetection`, `PreparedItem`, `ClassifiedItem`, `ClassificationResult`, `WarningCode` |
| `registry.py` | `@register(kind, name)` and `build(kind, spec, ctx)`: turns a `{name, params}` config entry into an object |
| `config.py` | Config layering (base, presets, overrides, `--set`), strict validation, `config_hash` |
| `crop_builder.py` | Segmentation orchestration, padding, cropping, the context image (fixed logic) |
| `extractor.py` | Builds the VLM request, calls the client, parses JSON, retries, falls back (fixed logic) |
| `system_prompt.py` | The VLM system prompt template; `{categories}` is filled from the vocabulary |
| `assembler.py` | Combines items, attributes and warnings into the result (fixed logic) |
| `renderer.py` | Draws boxes, masks and label banners onto the **original** image |
| `trace.py` | `PipelineTrace`: optional record of every stage's output, for debugging |
| `cli.py` | Command-line wrapper (`run`, `show-config`, `list-components`) |
| `components/` | All swappable components, one module per kind (see section 6) |
| `../../config/` | `pipeline.yaml` (defaults), `presets/`, `prompts/` |

Two categories of code, and the difference matters when you iterate:

- **Components** (`components/`) are the parts you swap by editing config. They implement an
  interface, declare a `Params` model, and register a name.
- **Fixed logic** (`crop_builder.py`, `extractor.py`, `assembler.py`, `renderer.py`,
  `pipeline.py`) is the glue. It is configured by values (for example `crop.padding_ratio`) but
  you change it by editing code.

## 4. One forward pass, step by step

Take a photo of a desk with a router and a charger. Here is what `pipeline.run(image)` does.
Coordinates are always **original-image pixels** unless stated otherwise.

### Step 0: Validate the input
`validate_image` accepts only a non-empty `(H, W, 3)` `uint8` array and raises `ValueError`
otherwise. (The CLI handles file loading, EXIF rotation and RGB conversion before this.)

### [A] Pre-process (`image_preprocess`, ordered list)
Default: `ensure_rgb_uint8`, then `resize_max_side` (longest side at most 1280 px). Other
steps: `clahe`, `gamma`. These steps change **only what the detector sees**. Geometric steps
record their scale and offset in a `PreprocessState` so boxes can be mapped back.

### [B] Detect (`detector`, one component)
Default: `ram_grounding_dino` (RAM++ then `IDEA-Research/grounding-dino-tiny`):

1. **Tag.** RAM++ looks at the photo (resized to 384 px) and returns tags such as
   `"kettle | countertop | black"`.
2. **Clean.** Tags are lower-cased, de-duplicated, and stripped of dots and commas (DINO's
   separator). `drop_tags` removes scene words (colours, rooms, furniture, verbs) and tags longer
   than four words are dropped. The first `max_tags` are kept.
3. **Prompt.** The cleaned tags, then (`include_vocabulary`) the shared vocabulary categories,
   are joined like `"kettle . laptop . router ."`. If the list is too long for one prompt it is
   split into chunks of at most `max_prompt_tokens` and DINO runs once per chunk.

If RAM++ fails on an image, the prompt falls back to the vocabulary for that image. The detector
sets `open_vocabulary = True`, because its phrases are not limited to the vocabulary.

The baseline detector `grounding_dino` skips steps 1 and 2: its prompt is the vocabulary alone.

Each returned label is mapped back to the longest prompt phrase it contains; unmappable labels
are dropped. The result is a list of `RawDetection(phrase, confidence, bbox_xyxy)`.

The detector applies only a **low floor score** (0.25 box, 0.2 text for `ram_grounding_dino`, the
Grounded-SAM demo values; 0.15 for the baseline). Real thresholding is done later by
filters, so you can tune thresholds without re-running the model.

### Map back and sanitize
Boxes are mapped from the pre-processed frame to original pixels (`_map_back`, which undoes the
geometric steps in reverse order). Then `sanitize_detections` runs always, and is not
configurable: it clamps boxes to the image and drops non-finite values, out-of-range
confidences, empty boxes and, for a closed-vocabulary detector, phrases outside the vocabulary.
A detector that sets `open_vocabulary` (like `ram_grounding_dino`) skips that last check.

### [C] Filter (`detection_filters`, ordered list)
Each filter takes a list of detections and returns a smaller list. **Order matters.** Defaults:

| Filter | What it removes |
|---|---|
| `score_threshold` (0.25) | Low-confidence boxes |
| `min_size` (32x32 px) | Tiny boxes |
| `min_area_ratio` (2%) | Slivers and background scraps |
| `full_frame_guard` (90%) | Whole-photo boxes, when smaller boxes exist (otherwise `containment` would later treat them as containers and drop every real box inside) |
| `nms` (IoU 0.5, class-agnostic) | Duplicate boxes on the same object |
| `containment` (90%) | Boxes lying inside a larger box: parts like a lens or label |
| `top_k` (20) | Everything beyond the 20 most confident |

Then `assign_ids` sorts the survivors by `(y1, x1)` and names them `obj_1 ... obj_N`. The
sort makes ids deterministic across runs.

**Early exit:** if nothing survives, `run` returns the original image and an empty `items`
list immediately.

### [D] Segment (`segmenter` + `segmentation_gate`)
Default segmenter is `hf_sam` (Segment Anything from Hugging Face `transformers`, no checkpoint
file). The **gate** (`always` by default) decides which items get a mask, and the segmenter is
prompted with those boxes, all in one model call. If that call fails each box is retried alone,
so one bad box loses only its own mask. With `segmenter: none` (the baseline) this stage does
nothing and the gate is never consulted; `sam` and `sam2` are the checkpoint-file alternatives. Masks exist to blank out a neighbouring device
that overlaps the crop. If an approved item ends up with no usable mask, the item gets a
`SEGMENTATION_FAILED` warning and a plain rectangular crop.

### [E] Build crops (`crop_builder.build_prepared_items`)
For each detection, from the **original** image:

1. Pad the box by `padding_ratio` (12% per side) and clamp to the image.
2. Cut the crop. If a mask exists, pixels outside it become neutral gray (128).
3. Run the crop transforms ([F]).
4. Downscale so the longer side is at most `max_image_side_px` (1024).
5. Attach a **context image**: the whole photo, downscaled, shared by all items. It is
   skipped when the crop already covers 90% or more of the photo.

The result is a `PreparedItem`: the detection, the padded box, the crop, the context and the
mask.

### [F] Crop transforms (`crop_transforms`, ordered list)
Applied to each crop before the VLM sees it. Default: `identity`. Others: `upscale_min_side`,
`sharpen`, `clahe`. Use these to help the VLM read small label text.

### [G] VLM extraction (`vlm_client` + `AttributeExtractor`)
This runs **once per item, sequentially**. The extractor:

1. Sends the system prompt (the preferred categories are filled in from the vocabulary), the
   user text, and the images: **crop first, context second**. The user text tells the VLM to
   describe only Image 1 and names the detector's phrase as **a hint to check, not a fact**.
2. Expects a JSON object with `brand`, `model`, `category`, `color` and, with the default prompt
   (`config/prompts/vlm_validate_prompt.txt`), `is_electronic`. A missing or unreadable verdict is
   "no verdict" and never drops an item. The parser tolerates
   prose and code fences around the JSON, ignores extra keys, and turns missing keys and
   `null` into `"unknown"`.
3. If the reply is not parseable, it retries up to `max_retries` times with a reminder to
   return JSON only. A client **exception** (timeout, out of memory) is not retried.
4. If everything fails, the item gets all-`unknown` attributes, category `electronic device`,
   and a `VLM_PARSE_FAILED` warning. One bad item never aborts the run.

**Validation.** When the VLM answers `is_electronic: false`, `vlm.on_not_electronic` decides: `drop`
(default) removes the item from the result and the picture and records its id in
`trace.dropped_not_electronic`; `warn` keeps it with a `NOT_ELECTRONIC` warning. This is what
removes RAM scene tags that DINO boxed anyway (a table, a wall, a poster of a phone).

VLM clients: `openai_compatible` (default in `pipeline.yaml`: a local Ollama server running
`qwen3-vl:8b-instruct`), `transformers` (in-process Hugging Face model) and `fake`. They all
implement one method, `generate(system_prompt, images, user_text) -> str`, so prompts,
parsing and retries are identical whichever you use.

### [H] Post-process attributes (`attribute_postprocess`, ordered list)
Each step takes the attribute dict and returns `(new_dict, warnings)`. Defaults, in order:

1. `normalize_unknown`: strips control characters, collapses whitespace, caps length, and maps
   empty or "n/a"-like values to `unknown`.
2. `ocr_verify_model`: if the VLM claimed a model name, RapidOCR reads the crop and the model
   is kept only if the text is really there. Otherwise it becomes `unknown` plus
   `MODEL_UNVERIFIED`. This is the guard against hallucinated model numbers. It only runs
   for items where a model was claimed.
3. `category_open` (default): a category outside the vocabulary is kept, cleaned, and flagged
   `CATEGORY_NOVEL`.

Alternatives: `category_vocab` (the baseline: a category outside the vocabulary becomes
`electronic device` plus `CATEGORY_OUT_OF_VOCAB`) and `brand_aliases` (canonicalise brand
spellings from a mapping file).

### [I] Warning rules (`warning_rules`, ordered list)
Each rule inspects a draft of the finished item and returns warning codes. Defaults:
`unknown_field` (`BRAND/MODEL/COLOR_UNKNOWN`), `low_detector_confidence` (below 0.40),
`small_crop` (shorter side under 64 px), `overlap` (IoU of 0.15 or more with another box) and
`category_mismatch` (the detector phrase and the VLM category are two different *specific*
categories that share no word). Rules never change attributes; they only add warnings.

An item's final warnings are the union, de-duplicated in first-seen order, of: extractor
warnings, segmentation warnings, post-processor warnings and rule warnings.

### [J] Assemble
`assemble_result_payload` builds one `ClassifiedItem` per prepared item, in id order. Its
`bbox_xyxy` is the **detector's unpadded box** in original pixels. Its `attributes` are
Schema.org `Product` fields. The result carries `meta.config_hash` and `meta.components`, so
any output can be traced to the config that made it.

### [K] Render
`render_annotated_image` draws a box, an optional mask overlay and a label banner (from
`rendering.label_template`) per item on the **original** image. It reads the same
`ClassifiedItem` records as the JSON, so the picture and the JSON cannot disagree.

### Return value
`run()` returns `(annotated_rgb, ClassificationResult)`, or with `return_trace=True`,
`(annotated_rgb, result, PipelineTrace)`.

## 5. Data flow: what each stage hands to the next

```text
image ndarray
  -> list[RawDetection]              detector, filters   (phrase, confidence, bbox)
  -> list[RawDetection + obj_N ids]  assign_ids
  -> list[PreparedItem]              crop builder        (adds crop, context, mask)
  -> Attributes dict                 extractor, post-processors   (brand/model/category/color)
  -> ClassifiedItem                  assembler           (adds warnings)
  -> ClassificationResult + image    renderer
```

Types are defined in `schemas.py`. Only the **detector's score** is reported as confidence.
The VLM gives no calibrated score, so there is none for brand, model or color.

## 6. The component system

### The nine kinds

| Kind (in code) | Config key | Cardinality | Interface method |
|---|---|---|---|
| `image_preprocess` | `pipeline.image_preprocess` | ordered list | `apply(image, state)`, `map_boxes_back(boxes, state)` |
| `detector` | `pipeline.detector` | one | `predict(image) -> list[RawDetection]` |
| `detection_filter` | `pipeline.detection_filters` | ordered list | `apply(dets, ctx) -> dets` |
| `segmenter` | `pipeline.segmenter` | one | `segment(image, boxes) -> masks` |
| `segmentation_gate` | `pipeline.segmentation_gate` | one | `should_segment(idx, dets) -> bool` |
| `crop_transform` | `pipeline.crop_transforms` | ordered list | `apply(crop) -> crop` |
| `vlm_client` | `pipeline.vlm.client` | one | `generate(system, images, user_text) -> str` |
| `attribute_postprocess` | `pipeline.attribute_postprocess` | ordered list | `apply(attrs, item_ctx) -> (attrs, warnings)` |
| `warning_rule` | `pipeline.warning_rules` | ordered list | `evaluate(draft, item_ctx) -> codes` |

Protocols for all of these are in `interfaces.py`.

### How a config entry becomes an object
1. Config entry: `{name: nms, params: {iou_threshold: 0.5}}`.
2. `registry.build("detection_filter", spec, build_ctx, path)` looks up the class registered
   under that name (`@register("detection_filter", "nms")`), validates `params` against the
   class's inner `Params` model (`extra="forbid"`), then calls `cls(params, build_ctx)`.
3. `build_ctx` (a `BuildContext`) carries the shared vocabulary, the resolved device, the seed,
   and a read-only copy of the config.

Everything is built **once, in `__init__`**, and every config problem surfaces there with the
full config path in the message (for example `pipeline.detector.params.raw_box_threshold`),
never in the middle of a run. Heavy imports (torch, transformers, SAM, RapidOCR) are lazy, so
unit tests run with no GPU and no weights.

### The shared vocabulary
`vocabulary.categories` is the single source of truth for what the project knows by name. It feeds
the detector prompt (added to the RAM tags with `include_vocabulary`, or the whole prompt in the
baseline), the VLM system prompt, `category_vocab`, `category_open` and the `category_*` rules.
Change it in one place and all of them follow. `electronic device` is always appended as the fallback category.

### Configuration layering
Later layers win: `config/pipeline.yaml`, then each `--preset`, then each `--override FILE`,
then each `--set dotted.key=value`. Dicts deep-merge; **lists are replaced whole** (to change
one filter you restate the entire `detection_filters` list). If a layer swaps a component's
`name`, its whole `{name, params}` block is replaced so stale params never leak. The final
config is hashed (`meta.config_hash`) and saved as `resolved_config.yaml` on each CLI run.

## 7. Failure handling philosophy

- **Configuration errors fail early** (at build time, naming the config path).
- **Input errors fail loudly** (`ValueError` from the API; one-line message and exit code 2
  from the CLI).
- **Per-item model trouble degrades quietly**: a bad VLM reply gives fallback attributes and a
  warning, never a crash. Missing masks give a warning and a plain crop.
- **Doubt becomes a warning**, not a silent guess: unknown fields, unverified models, low
  confidence, small crops, overlaps, category disagreement.

## 8. How to iterate on the architecture

Find your goal below, then touch only what is listed.

| I want to... | Change |
|---|---|
| Find more or fewer devices | `detector.params.max_tags` and `drop_tags`, `vocabulary.categories`, the thresholds in `detection_filters`, `detector.params.model_id` |
| Compare with the old pipeline | `--preset baseline` (Grounding DINO + vocabulary prompt, no masks) |
| Stop scene objects (table, wall) becoming items | `drop_tags`, and `vlm.on_not_electronic: drop` |
| Stop parts (lens, label) showing up as devices | `containment` filter (see its `containers` param) |
| Help the detector with small objects in big photos | Raise or remove `resize_max_side` in `image_preprocess` |
| Help the VLM read labels | `crop.padding_ratio`, `crop.max_image_side_px`, add `upscale_min_side` or `sharpen` to `crop_transforms` |
| Isolate overlapping devices | The default `hf_sam` segmenter already masks every item; the `accurate_sam` preset uses a SAM checkpoint file instead |
| Try a different VLM | `pipeline.vlm.client` (name and params) |
| Change what the VLM is told | `pipeline.vlm.system_prompt_file` (must contain `{categories}`); see `config/prompts/` |
| Restrict categories to the vocabulary | `--preset baseline`, or swap `category_open` for `category_vocab` |
| Add a new check on results | Write a `warning_rule` |
| Clean or correct VLM output | Write an `attribute_postprocess` step |
| Run a different recipe | Add a preset under `config/presets/` |
| Add a stage that does not exist yet | Edit `pipeline.py::run`, add a kind to `registry.KINDS` and `interfaces.py`, and a field to `PipelineConfig` in `config.py` |

### Adding a component (the common case)
1. Write a class with an inner `Params(BaseModel)` (`extra="forbid"`, each field with a
   `description`), an `__init__(self, params, build)`, and the method from the table in
   section 6.
2. Register it with `@register("<kind>", "<name>")` in a module under `components/` and import
   that module to `_BUILTIN_MODULES` in `components/__init__.py` (a new file in `components/`
   is not found automatically). Or skip registering and refer to
   it from config by dotted path: `name: "my_pkg.my_module:MyClass"`.
3. Select it in a config file, preset or `--set`.
4. Check it with `python -m electronics_classifier.cli list-components` and `show-config`.
5. Compare results across configs with `experiments/run_batch.py`, which writes a
   `summary.csv` with one row per (image, config).

### Debugging a wrong result
- CLI: `--save-intermediates` writes the pre-processed image, a detection overlay after every
  filter, each crop, the raw VLM text, and the attributes after each post-processor.
- Python: `pipeline.run(img, return_trace=True)` returns a `PipelineTrace` with the same
  information.
- Work backwards: wrong box means detector or filters; right box but wrong brand or model
  means the crop, the prompt or the VLM; right VLM text but wrong final value means a
  post-processor.
- Offline smoke test with no models: `EC_USE_FAKES=1` (read by the CLI). The result's `meta.components` says
  `fake`.

## 9. Design decisions worth knowing before you change things

- **Crops, context, masks and the annotated image all come from the original image.**
  Pre-processing affects only the detector's input.
- **Detection thresholds live only in the filters.** The detector is deliberately permissive.
- **Order is configuration.** Reordering filters or post-processors changes behaviour (for
  example `nms` before `min_size` gives different results than after).
- **Items are processed independently and sequentially.** There is no batching across crops;
  that is the first place to look if VLM latency matters.
- **The VLM is not trusted for model numbers.** It is told to answer `unknown` unless the
  exact name is printed, and `ocr_verify_model` then checks the claim against OCR. Brands are
  not OCR-checked because they can come from logos.
- **Tests mirror the structure.** `tests/test_failure_modes.py` holds hostile-input cases
  (Tier A) and known weaknesses as strict `xfail` tests (Tier B); `tests/slow/` needs real
  models. Run `pytest -q` after any change; it needs no GPU.

## 10. Known limitations

- Detection is only as good as the RAM++ tag list (about 4,500 tags). A device RAM++ does not
  tag is still found if its category is in the vocabulary (`include_vocabulary`), otherwise it is
  missed.
- RAM tags are noisy. The stoplist and the VLM verdict catch most scene objects, but an unusual
  one can reach the VLM, which costs a model call.
- `category_mismatch` treats phrases that share a word as agreeing (`usb hub` vs `usb charger`).
- RAM++ on Apple GPUs relies on CPU fallback for unsupported ops and has not been verified on
  every Mac; `ram_on_cpu` is the escape hatch. `recognize-anything` pins old `timm` and `fairscale`.
- `containment` can drop a real device lying fully inside another's box (a charger on a power
  strip's box).
- Very small objects in very large photos can be missed after `resize_max_side` shrinks the
  detector input.
- The RAM++ detector, the SAM segmenters and the `transformers` VLM client are tested in this
  repository only against stub models, not real weights (`tests/slow/` C15-C17 cover the default
  pipeline when the weights and photos are present).
