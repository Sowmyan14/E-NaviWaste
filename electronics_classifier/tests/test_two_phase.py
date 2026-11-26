"""``detect()`` then ``classify()``: the same result as ``run()``, with models freed in between."""

import numpy as np
import pytest
from conftest import VALID_REPLY, FakeDetector, FakeVLMClient, det, make_image, make_pipeline

IMAGES = [make_image("noise", 480, 640, seed=1), make_image("noise", 300, 400, seed=2)]
DETECTIONS = [det("router", 0.9, (20, 20, 220, 220)), det("laptop", 0.8, (300, 200, 600, 420))]


def pipeline_with(detections=DETECTIONS, replies=None):
    return make_pipeline(detections, replies or [VALID_REPLY])


def test_detect_then_classify_equals_run_for_every_image():
    expected = [pipeline_with().run(image) for image in IMAGES]
    split = pipeline_with()
    detected = [split.detect(image) for image in IMAGES]  # all detection first
    got = [split.classify(d) for d in detected]  # then all VLM work
    for (annotated, result), (want_annotated, want_result) in zip(got, expected, strict=True):
        assert result.model_dump() == want_result.model_dump()
        assert np.array_equal(annotated, want_annotated)


def test_detect_never_calls_the_vlm_and_classify_never_calls_the_detector():
    detector = FakeDetector(DETECTIONS)
    client = FakeVLMClient([VALID_REPLY], cycle=True)
    pipeline = make_pipeline(detector=detector, vlm_client=client)
    detected = pipeline.detect(IMAGES[0])
    assert len(detector.calls) == 1 and client.calls == []
    pipeline.classify(detected)
    assert len(detector.calls) == 1 and len(client.calls) == len(DETECTIONS)


def test_detected_image_carries_ids_crops_and_the_original_photo():
    detected = pipeline_with().detect(IMAGES[0])
    assert [d.detection_id for d in detected.detections] == ["obj_1", "obj_2"]
    assert len(detected.items) == 2 and detected.image.shape == IMAGES[0].shape
    assert all(item.crop_rgb.size for item in detected.items)


def test_classify_still_works_after_the_detection_models_are_released():
    pipeline = pipeline_with()
    detected = [pipeline.detect(image) for image in IMAGES]
    pipeline.release_detection_models()
    assert pipeline.detector is None and pipeline.segmenter is None
    for d in detected:
        _, result = pipeline.classify(d)
        assert len(result.items) == 2


def test_detecting_after_release_is_a_clear_error_not_a_nonetype_crash():
    pipeline = pipeline_with()
    pipeline.release_detection_models()
    for call in (pipeline.detect, pipeline.run):
        with pytest.raises(RuntimeError, match="released"):
            call(IMAGES[0])


def test_nothing_detected_gives_the_original_photo_and_no_items_in_both_ways():
    pipeline = pipeline_with(detections=[])
    detected = pipeline.detect(IMAGES[0])
    assert detected.items == [] and detected.detections == []
    annotated, result = pipeline.classify(detected)
    run_annotated, run_result = pipeline.run(IMAGES[0])
    assert result.items == [] and result.model_dump() == run_result.model_dump()
    assert np.array_equal(annotated, IMAGES[0]) and np.array_equal(run_annotated, IMAGES[0])
    assert annotated is not detected.image  # a copy, so callers cannot change the input


def test_a_shared_trace_matches_run_with_trace():
    _, _, want = pipeline_with().run(IMAGES[0], return_trace=True)
    split = pipeline_with()
    from electronics_classifier.trace import PipelineTrace

    trace = PipelineTrace()
    split.classify(split.detect(IMAGES[0], trace), trace)
    assert [d.detection_id for d in trace.detections] == [d.detection_id for d in want.detections]
    assert len(trace.prepared_items) == len(want.prepared_items)
    assert trace.vlm_replies.keys() == want.vlm_replies.keys()
    assert [n for n, _ in trace.filter_steps] == [n for n, _ in want.filter_steps]


def test_invalid_input_is_rejected_by_detect_with_the_usual_value_error():
    with pytest.raises(ValueError):
        pipeline_with().detect(np.zeros((10, 10), dtype=np.uint8))
