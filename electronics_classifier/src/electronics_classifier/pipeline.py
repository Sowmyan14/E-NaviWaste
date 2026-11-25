"""The pipeline: chains the stages. All behaviour lives in the components it wires together."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .assembler import (
    assemble_result_payload,
    evaluate_warning_rules,
    make_draft,
    unique_in_order,
)
from .components.detection_filters import assign_ids, sanitize_detections
from .components.image_preprocess import apply_preprocess, map_boxes_back, validate_image
from .config import ConfigError, CropConfig, config_hash, load_config, resolve_config
from .crop_builder import build_prepared_items, segment_approved, segmentation_warnings
from .extractor import AttributeExtractor, apply_postprocessors
from .interfaces import BuildContext, FilterContext, ItemContext, PreprocessState
from .registry import build
from .renderer import RenderStyle, render_annotated_image, validate_label_template
from .schemas import ClassificationResult, RawDetection, RunMeta, WarningCode
from .system_prompt import load_template
from .trace import PipelineTrace

# Ready-made components can be injected by these keyword names (tests, notebooks).
COMPONENT_KWARGS = (
    "image_preprocess",
    "detector",
    "detection_filters",
    "segmenter",
    "segmentation_gate",
    "crop_transforms",
    "vlm_client",
    "attribute_postprocess",
    "warning_rules",
)


RunOutput = (
    tuple[np.ndarray, ClassificationResult]
    | tuple[np.ndarray, ClassificationResult, PipelineTrace]
)


def _name(component: object, spec_name: str | None) -> str:
    return spec_name if spec_name is not None else f"injected:{type(component).__name__}"


class ElectronicsClassificationPipeline:
    """Classify the electronics in one RGB image: ``run(image) -> (annotated, result)``.

    Everything that can be wrong with the configuration is detected here, at build time.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        presets: Sequence[str] = (),
        overrides: Sequence[str | Path] = (),
        sets: Sequence[str] = (),
        *,
        resolved_config: Mapping[str, Any] | None = None,
        **components: Any,
    ) -> None:
        unknown = sorted(set(components) - set(COMPONENT_KWARGS))
        if unknown:
            raise TypeError(
                f"unknown component argument(s) {unknown}; allowed: {', '.join(COMPONENT_KWARGS)}"
            )
        if resolved_config is not None:
            self.config = resolve_config(resolved_config)
        else:
            self.config = load_config(presets, overrides, sets, base_path=config_path)
        self.config_hash = config_hash(self.config)
        self._build_components(components)

    @classmethod
    def from_config(
        cls, resolved_config: Mapping[str, Any], **components: Any
    ) -> ElectronicsClassificationPipeline:
        return cls(resolved_config=resolved_config, **components)

    # --- build ---------------------------------------------------------------------

    def _build_components(self, injected: Mapping[str, Any]) -> None:
        cfg = self.config
        pipe = cfg["pipeline"]
        self.ctx = BuildContext.from_config(cfg)
        self.style = RenderStyle.from_config(cfg["rendering"])
        validate_label_template(self.style.label_template)
        self.output_format = cfg["rendering"]["output_format"]
        self.crop_cfg = CropConfig(**pipe["crop"])

        names: dict[str, str | list[str]] = {}

        def one(kind: str, key: str, spec: Mapping[str, Any], path: str) -> Any:
            component = injected.get(key)
            if component is None:
                component = build(kind, spec, self.ctx, path)
            names[kind] = _name(component, None if injected.get(key) is not None else spec["name"])
            return component

        def many(kind: str, key: str, specs: Sequence[Mapping[str, Any]], path: str) -> list[Any]:
            if injected.get(key) is not None:
                components = list(injected[key])
                names[kind] = [_name(c, None) for c in components]
                return components
            components = [build(kind, s, self.ctx, f"{path}.{i}") for i, s in enumerate(specs)]
            names[kind] = [s["name"] for s in specs]
            return components

        self.image_preprocess = many(
            "image_preprocess", "image_preprocess", pipe["image_preprocess"],
            "pipeline.image_preprocess",
        )  # fmt: skip
        self.detector = one("detector", "detector", pipe["detector"], "pipeline.detector")
        self.detection_filters = many(
            "detection_filter", "detection_filters", pipe["detection_filters"],
            "pipeline.detection_filters",
        )  # fmt: skip
        self.segmenter = one("segmenter", "segmenter", pipe["segmenter"], "pipeline.segmenter")
        self.segmentation_gate = one(
            "segmentation_gate", "segmentation_gate", pipe["segmentation_gate"],
            "pipeline.segmentation_gate",
        )  # fmt: skip
        self.crop_transforms = many(
            "crop_transform", "crop_transforms", pipe["crop_transforms"],
            "pipeline.crop_transforms",
        )  # fmt: skip
        client = one("vlm_client", "vlm_client", pipe["vlm"]["client"], "pipeline.vlm.client")
        self.attribute_postprocess = many(
            "attribute_postprocess", "attribute_postprocess", pipe["attribute_postprocess"],
            "pipeline.attribute_postprocess",
        )  # fmt: skip
        self.warning_rules = many(
            "warning_rule", "warning_rules", pipe["warning_rules"], "pipeline.warning_rules"
        )

        template = None
        template_file = pipe["vlm"]["system_prompt_file"]
        if template_file:
            try:
                template = load_template(template_file)
            except ConfigError as exc:
                raise ConfigError(f"pipeline.vlm.system_prompt_file: {exc}") from None
        self.on_not_electronic = pipe["vlm"]["on_not_electronic"]
        self.extractor = AttributeExtractor(
            client, self.ctx.categories, template, pipe["vlm"]["max_retries"]
        )

        self._filter_names = (
            names["detection_filter"] if isinstance(names["detection_filter"], list) else []
        )
        self.components = names
        self.meta = RunMeta(config_hash=self.config_hash, components=names)

    # --- run -----------------------------------------------------------------------

    def run(
        self, image_rgb: np.ndarray, return_trace: bool = False
    ) -> RunOutput:
        image = validate_image(image_rgb)  # ValueError for anything but (H, W, 3) uint8
        trace = PipelineTrace() if return_trace else None
        height, width = image.shape[:2]

        # [A] pre-process, [B] detect, then back to original pixels
        pre, state = apply_preprocess(self.image_preprocess, image)
        raw = self.detector.predict(pre)
        mapped = self._map_back(raw, state)
        # An open-vocabulary detector (RAM tags) makes its own phrases: do not drop them
        open_vocab = getattr(self.detector, "open_vocabulary", False)
        vocabulary = None if open_vocab else self.ctx.categories
        dets = sanitize_detections(mapped, width, height, vocabulary)
        if trace is not None:
            trace.preprocessed_image = pre
            trace.raw_detections = list(raw)
            trace.mapped_detections = list(dets)

        # [C] filters in configured order, then stable ids
        filter_ctx = FilterContext(width, height)
        for index, step in enumerate(self.detection_filters):
            dets = step.apply(dets, filter_ctx)
            if trace is not None:
                name = self._filter_names[index] if index < len(self._filter_names) else str(index)
                trace.filter_steps.append((name, list(dets)))
        dets = assign_ids(dets)
        if trace is not None:
            trace.detections = list(dets)

        if not dets:
            result = assemble_result_payload(
                [], {}, {}, width, height, self.meta, self.output_format
            )
            return self._finish(image.copy(), result, trace)

        # [D] segmentation, [E] crops, [F] crop transforms
        outcome = segment_approved(self.segmenter, self.segmentation_gate, image, dets)
        items = build_prepared_items(
            image, dets, outcome.masks, self.crop_cfg, self.crop_transforms
        )
        segmentation_failed = segmentation_warnings(outcome, items)
        if trace is not None:
            trace.prepared_items = list(items)

        # [G] VLM, [H] post-process, [I] warning rules, per item
        attributes_by_id: dict[str, dict] = {}
        warnings_by_id: dict[str, list[str]] = {}
        kept_items = []
        for index, item in enumerate(items):
            item_id = item.detection.detection_id
            item_ctx = ItemContext(item, index, dets, width, height)
            extraction = self.extractor.extract(item)
            not_electronic = extraction.attributes.get("is_electronic") is False
            if not_electronic and self.on_not_electronic == "drop":
                if trace is not None:
                    trace.vlm_replies[item_id] = extraction.raw_replies
                    trace.dropped_not_electronic.append(item_id)
                continue
            kept_items.append(item)
            attributes, post_warnings, snapshots = apply_postprocessors(
                self.attribute_postprocess, extraction.attributes, item_ctx
            )
            rule_warnings = evaluate_warning_rules(
                self.warning_rules, make_draft(item, attributes), item_ctx
            )
            attributes_by_id[item_id] = attributes
            verdict_warnings = [WarningCode.NOT_ELECTRONIC.value] if not_electronic else []
            warnings_by_id[item_id] = unique_in_order(
                extraction.warnings, segmentation_failed.get(item_id, []), post_warnings,
                rule_warnings, verdict_warnings,
            )  # fmt: skip
            if trace is not None:
                trace.vlm_replies[item_id] = extraction.raw_replies
                trace.attributes_before[item_id] = extraction.attributes
                trace.attributes_after[item_id] = snapshots

        # [J] assemble, [K] render
        result = assemble_result_payload(
            kept_items, attributes_by_id, warnings_by_id, width, height, self.meta,
            self.output_format,
        )  # fmt: skip
        masks = {i.detection.detection_id: i.mask for i in kept_items if i.mask is not None}
        annotated = render_annotated_image(image, result.items, masks, self.style)
        return self._finish(annotated, result, trace)

    @staticmethod
    def _finish(annotated: np.ndarray, result: ClassificationResult, trace: PipelineTrace | None):
        return (annotated, result) if trace is None else (annotated, result, trace)

    def _map_back(self, dets: Sequence[RawDetection], state: PreprocessState) -> list[RawDetection]:
        """Boxes from the pre-processed frame to original pixels. Malformed boxes pass through
        untouched; ``sanitize_detections`` drops them next."""
        mapped = list(dets)
        indices: list[int] = []
        boxes: list[np.ndarray] = []
        for i, det in enumerate(dets):
            try:
                box = np.asarray(det.bbox_xyxy, dtype=np.float64)
            except (TypeError, ValueError, AttributeError):
                continue
            if box.shape == (4,):
                indices.append(i)
                boxes.append(box)
        if boxes:
            back = map_boxes_back(self.image_preprocess, np.stack(boxes), state)
            for i, box in zip(indices, back, strict=True):
                mapped[i] = dets[i].model_copy(update={"bbox_xyxy": tuple(float(v) for v in box)})
        return mapped
