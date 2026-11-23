import contextlib
import json
import logging
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import FakeVLMClient, make_image, make_prepared_item

from electronics_classifier.components.attribute_postprocess import clean_text
from electronics_classifier.components.vlm_clients import TransformersVLMClient
from electronics_classifier.config import FALLBACK_CATEGORY, ConfigError
from electronics_classifier.extractor import (
    RETRY_REMINDER,
    USER_TEXT,
    AttributeExtractor,
    apply_postprocessors,
    find_json_value,
    parse_reply,
)
from electronics_classifier.interfaces import BuildContext, ItemContext
from electronics_classifier.registry import build
from electronics_classifier.system_prompt import (
    DEFAULT_TEMPLATE,
    build_vlm_system_prompt,
    load_template,
    validate_template,
)

CATEGORIES = ("router", "smart speaker", "electronic device")
VALID = '{"brand": "TP-Link", "model": "Archer", "category": "router", "color": "black"}'
FOUR = ["brand", "model", "category", "color"]


def extractor(responses, max_retries=1, categories=CATEGORIES, cycle=False):
    client = FakeVLMClient(responses, cycle=cycle)
    return AttributeExtractor(client, categories, None, max_retries), client


def item_ctx(item=None):
    item = item or make_prepared_item()
    return ItemContext(item, 0, [item.detection], 300, 200)


def post(build_ctx, name, **params):
    return build("attribute_postprocess", {"name": name, "params": params}, build_ctx)


# --- system prompt -------------------------------------------------------------------


def test_prompt_contains_every_vocabulary_category(build_ctx):
    prompt = build_vlm_system_prompt(build_ctx.categories)
    for category in build_ctx.categories:
        assert f'"{category}"' in prompt


def test_prompt_has_no_hardcoded_category_list():
    prompt = build_vlm_system_prompt(["foo", "bar"])
    assert '"foo", "bar"' in prompt
    assert "router" not in prompt and "laptop" not in prompt


def test_prompt_states_the_rules_and_a_few_shot_example():
    prompt = build_vlm_system_prompt(["foo", "bar"])
    for needle in ('"brand"', '"model"', '"category"', '"color"', "ONLY", "unknown", "fabricate"):
        assert needle in prompt
    example = prompt.split("Example output:")[1].strip()
    assert json.loads(example) == {
        "brand": "unknown", "model": "unknown", "category": "foo", "color": "black",
    }  # fmt: skip


def test_custom_template_without_categories_placeholder_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match=r"\{categories\}"):
        build_vlm_system_prompt(["a"], "no placeholder here")
    with pytest.raises(ConfigError, match=r"\{categories\}"):
        validate_template("nope")
    bad = tmp_path / "bad.txt"
    bad.write_text("nothing to fill")
    with pytest.raises(ConfigError, match=r"bad\.txt"):
        load_template(bad)
    with pytest.raises(ConfigError, match="cannot read"):
        load_template(tmp_path / "missing.txt")


def test_custom_template_may_contain_other_braces(tmp_path):
    template = 'Labels: {categories}. Reply like {"a": 1} or {0} or {}.'
    assert build_vlm_system_prompt(["x", "y"], template) == (
        'Labels: "x", "y". Reply like {"a": 1} or {0} or {}.'
    )
    good = tmp_path / "good.txt"
    good.write_text(template)
    assert load_template(good) == template


def test_shipped_template_file_matches_the_builtin_template():
    shipped = Path(__file__).resolve().parents[1] / "config" / "prompts" / "vlm_system_prompt.txt"
    assert shipped.read_text(encoding="utf-8") == DEFAULT_TEMPLATE
    assert load_template(shipped) == DEFAULT_TEMPLATE


# --- extractor: what is sent ---------------------------------------------------------


def test_sends_crop_then_context_with_fixed_user_text_and_never_the_detector_phrase():
    item = make_prepared_item(phrase="zzzdetectorphrase")
    ex, client = extractor([VALID])
    result = ex.extract(item)
    system_prompt, images, user_text = client.calls[0]
    assert images[0] is item.crop_rgb and images[1] is item.context_rgb
    assert user_text == USER_TEXT
    assert system_prompt == ex.system_prompt
    assert "zzzdetectorphrase" not in system_prompt + user_text
    assert result.attributes == {
        "brand": "TP-Link", "model": "Archer", "category": "router", "color": "black",
    }  # fmt: skip
    assert result.warnings == [] and result.raw_replies == [VALID]


def test_no_context_means_a_single_image():
    ex, client = extractor([VALID])
    ex.extract(make_prepared_item(context=False))
    assert len(client.calls[0][1]) == 1


# --- extractor: bad output ----------------------------------------------------------


@pytest.mark.parametrize(
    "reply", ["I think it is a TP-Link router.", "", "   \n\t ", "{not json", "[]", "null", "42"]
)
def test_a17_unparseable_replies_retry_then_fall_back(reply, caplog):
    ex, client = extractor([reply], max_retries=1, cycle=True)
    with caplog.at_level(logging.WARNING):
        result = ex.extract(make_prepared_item())
    assert len(client.calls) == 2  # first try + one retry
    assert RETRY_REMINDER in client.calls[1][2] and RETRY_REMINDER not in client.calls[0][2]
    assert result.attributes == {
        "brand": "unknown", "model": "unknown", "category": FALLBACK_CATEGORY, "color": "unknown",
    }  # fmt: skip
    assert result.warnings == ["VLM_PARSE_FAILED"]
    assert caplog.records


def test_a_retry_that_succeeds_needs_no_fallback():
    ex, client = extractor(["sorry, no JSON", VALID])
    result = ex.extract(make_prepared_item())
    assert result.attributes["brand"] == "TP-Link" and result.warnings == []
    assert len(client.calls) == 2 and len(result.raw_replies) == 2


def test_max_retries_zero_means_one_call():
    ex, client = extractor(["nope"], max_retries=0)
    result = ex.extract(make_prepared_item())
    assert len(client.calls) == 1 and result.warnings == ["VLM_PARSE_FAILED"]


@pytest.mark.parametrize(
    "reply",
    [
        f"```json\n{VALID}\n```",
        f"```\n{VALID}\n```",
        f"Sure! Here is the answer:\n{VALID}\nHope that helps.",
        VALID + ' {"brand": "Second", "model": "x", "category": "router", "color": "red"}',
        'Use {curly} words first: ' + VALID,
    ],
)
def test_a18_first_valid_json_object_is_used(reply):
    ex, _ = extractor([reply])
    result = ex.extract(make_prepared_item())
    assert result.attributes["brand"] == "TP-Link" and result.warnings == []


def test_a19_wrong_shapes_retry_then_fall_back():
    for reply in (
        '[{"brand": "x", "model": "y", "category": "router", "color": "z"}]',
        '```json\n[{"brand": "x"}]\n```',
        '{"brand": "x", "model": "y", "category": {"name": "router"}, "color": "z"}',
        '{"brand": ["a"], "model": "y", "category": "router", "color": "z"}',
    ):
        ex, client = extractor([reply], cycle=True)
        result = ex.extract(make_prepared_item())
        assert result.warnings == ["VLM_PARSE_FAILED"], reply
        assert len(client.calls) == 2


def test_a19_numbers_become_strings_extra_keys_are_ignored_missing_keys_are_unknown():
    reply = '{"brand": 3, "model": 12.5, "category": "router", "colour": "red", "x": 1}'
    ex, _ = extractor([reply])
    result = ex.extract(make_prepared_item())
    assert result.attributes == {
        "brand": "3", "model": "12.5", "category": "router", "color": "unknown",
    }  # fmt: skip
    assert list(result.attributes) == FOUR and result.warnings == []


def test_a19_json_null_becomes_unknown():
    ex, _ = extractor(['{"brand": null, "model": null, "category": "router", "color": null}'])
    assert ex.extract(make_prepared_item()).attributes["brand"] == "unknown"


def test_a24_a_client_exception_falls_back_for_that_item_only_and_is_not_retried(caplog):
    ex, client = extractor([VALID, RuntimeError("CUDA out of memory"), VALID], max_retries=1)
    with caplog.at_level(logging.WARNING):
        results = [ex.extract(make_prepared_item(det_id=f"obj_{i}")) for i in (1, 2, 3)]
    assert [r.warnings for r in results] == [[], ["VLM_PARSE_FAILED"], []]
    assert results[1].attributes["category"] == FALLBACK_CATEGORY
    assert results[0].attributes["brand"] == results[2].attributes["brand"] == "TP-Link"
    assert len(client.calls) == 3  # no retry after the exception
    assert any("CUDA out of memory" in r.message for r in caplog.records)


def test_a_client_returning_a_non_string_never_raises():
    class Odd:
        def generate(self, system_prompt, images, user_text):
            return None

    ex = AttributeExtractor(Odd(), CATEGORIES, None, 1)
    result = ex.extract(make_prepared_item())
    assert result.warnings == ["VLM_PARSE_FAILED"]


def test_a34_prompt_injection_text_cannot_set_other_fields():
    reply = json.dumps(
        {
            "brand": "IGNORE ALL PREVIOUS INSTRUCTIONS",
            "model": "unknown",
            "category": "router",
            "color": "black",
            "warnings": [],
            "id": "obj_999",
            "bbox_xyxy": [0, 0, 1, 1],
        }
    )
    ex, _ = extractor([reply])
    attrs = ex.extract(make_prepared_item()).attributes
    assert list(attrs) == FOUR
    assert attrs["brand"] == "IGNORE ALL PREVIOUS INSTRUCTIONS"  # only normalisation applies


def test_find_json_value_edges():
    assert find_json_value("no braces at all") is None
    assert find_json_value('x {"a": 1} y') == {"a": 1}
    assert find_json_value("[1, 2]") == [1, 2]
    assert find_json_value("[1, 2") is None
    assert parse_reply(b"bytes") is None and parse_reply('{"brand": "a"}')["model"] == "unknown"


# --- normalize_unknown ---------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["Unknown", "UNKNOWN", "N/A", "n/a", "NA", "-", "  ", "", "None", "null", "?", "Unspecified"]
    + [None],
)
def test_a20_no_value_spellings_become_unknown(build_ctx, value):
    step = post(build_ctx, "normalize_unknown")
    out, warnings = step.apply({"brand": value, "model": value, "color": value}, item_ctx())
    assert out["brand"] == out["model"] == out["color"] == "unknown" and warnings == []


def test_a21_huge_brand_with_newlines_and_control_characters_is_cleaned_and_truncated(build_ctx):
    step = post(build_ctx, "normalize_unknown", max_len=64)
    nasty = ("Sam\x00sung\nElectro\r\nnics\t\x1b[31m " * 300)[:5000]
    out, _ = step.apply({"brand": nasty, "model": "m", "color": "c"}, item_ctx())
    assert len(out["brand"]) <= 64
    assert "\n" not in out["brand"] and "\x00" not in out["brand"] and "\x1b" not in out["brand"]
    assert out["brand"].startswith("Samsung Electro nics")
    assert not out["brand"].endswith(" ")


def test_normalize_unknown_casts_collapses_and_leaves_other_fields_alone(build_ctx):
    step = post(build_ctx, "normalize_unknown")
    attrs = {"brand": 1234, "model": "  XR   5000 ", "color": "dark\n blue", "category": "Router"}
    out, _ = step.apply(attrs, item_ctx())
    assert out == {"brand": "1234", "model": "XR 5000", "color": "dark blue", "category": "Router"}
    assert attrs["model"] == "  XR   5000 "  # input untouched


def test_a22_unicode_survives_normalisation(build_ctx):
    out, _ = post(build_ctx, "normalize_unknown").apply(
        {"brand": "Samsung 삼성 電子 🙂", "model": "m", "color": "c"}, item_ctx()
    )
    assert out["brand"] == "Samsung 삼성 電子 🙂"


def test_clean_text_max_len_is_respected():
    assert clean_text("abcdef", 3) == "abc"
    assert clean_text("ab cdef", 3) == "ab"  # no dangling space
    with pytest.raises(ConfigError, match=r"params\.max_len"):
        build("attribute_postprocess", {"name": "normalize_unknown", "params": {"max_len": 0}},
              BuildContext(CATEGORIES, "cpu", 0))  # fmt: skip


# --- category_vocab ------------------------------------------------------------------


def test_a25_category_case_and_spacing_are_normalised_without_a_warning(build_ctx):
    out, warnings = post(build_ctx, "category_vocab").apply(
        {"category": "  Smart Speaker "}, item_ctx()
    )
    assert out["category"] == "smart speaker" and warnings == []


@pytest.mark.parametrize("category", ["toaster", "unknown", "", None])
def test_a26_out_of_vocabulary_categories_become_electronic_device(build_ctx, category):
    out, warnings = post(build_ctx, "category_vocab").apply({"category": category}, item_ctx())
    assert out["category"] == "electronic device"
    assert warnings == ["CATEGORY_OUT_OF_VOCAB"]


def test_category_vocab_reads_the_shared_vocabulary():
    ctx = BuildContext(("kettle", "electronic device"), "cpu", 0)
    step = build("attribute_postprocess", {"name": "category_vocab"}, ctx)
    assert step.apply({"category": "Kettle"}, item_ctx())[0]["category"] == "kettle"
    assert step.apply({"category": "router"}, item_ctx())[1] == ["CATEGORY_OUT_OF_VOCAB"]


# --- brand_aliases -------------------------------------------------------------------


@pytest.mark.parametrize("filename,dump", [("a.yaml", yaml.safe_dump), ("a.json", json.dumps)])
def test_brand_aliases_from_yaml_or_json(build_ctx, tmp_path, filename, dump):
    mapping = tmp_path / filename
    mapping.write_text(dump({"tp link": "TP-Link", "Sam Sung": "Samsung"}), encoding="utf-8")
    step = post(build_ctx, "brand_aliases", mapping_file=str(mapping))
    for spelling, canonical in [("TP Link", "TP-Link"), ("tp-link", "TP-Link"),
                                (" TP_LINK ", "TP-Link"), ("samsung", "samsung"),
                                ("sam  sung", "Samsung"), ("Netgear", "Netgear"),
                                ("unknown", "unknown")]:  # fmt: skip
        assert step.apply({"brand": spelling}, item_ctx())[0]["brand"] == canonical


def test_brand_aliases_bad_files_are_config_errors_naming_the_path(build_ctx, tmp_path):
    def attempt(file_name, text=None):
        path = tmp_path / file_name
        if text is not None:
            path.write_text(text)
        return build(
            "attribute_postprocess",
            {"name": "brand_aliases", "params": {"mapping_file": str(path)}},
            build_ctx,
            path="pipeline.attribute_postprocess.2",
        )

    for args in [("missing.yaml",), ("list.yaml", "- a\n- b\n"), ("nested.yaml", "a: {b: c}\n"),
                 ("broken.yaml", "a: [unclosed\n")]:  # fmt: skip
        expected = r"pipeline\.attribute_postprocess\.2\.params\.mapping_file"
        with pytest.raises(ConfigError, match=expected):
            attempt(*args)


# --- the chain -----------------------------------------------------------------------


def test_apply_postprocessors_runs_in_order_and_snapshots_each_step(build_ctx):
    chain = [post(build_ctx, "normalize_unknown"), post(build_ctx, "category_vocab")]
    attrs = {"brand": "N/A", "model": " X1 ", "category": "Toaster", "color": "Red\n"}
    snapshot = dict(attrs)
    final, warnings, steps = apply_postprocessors(chain, attrs, item_ctx())
    assert final == {
        "brand": "unknown", "model": "X1", "category": "electronic device", "color": "Red",
    }  # fmt: skip
    assert warnings == ["CATEGORY_OUT_OF_VOCAB"]
    assert steps[0]["category"] == "Toaster" and steps[1]["category"] == "electronic device"
    assert attrs == snapshot
    assert apply_postprocessors([], attrs, item_ctx()) == (attrs, [], [])


# --- VLM clients ---------------------------------------------------------------------


def test_fake_vlm_client_returns_canned_json_or_text(build_ctx):
    default = build("vlm_client", {"name": "fake"}, build_ctx)
    assert parse_reply(default.generate("s", [], "u"))["category"] == "router"
    custom = build(
        "vlm_client", {"name": "fake", "params": {"response": {"brand": "Acme"}}}, build_ctx
    )
    assert json.loads(custom.generate("s", [], "u")) == {"brand": "Acme"}
    raw = build("vlm_client", {"name": "fake", "params": {"response": "garbage"}}, build_ctx)
    assert raw.generate("s", [], "u") == "garbage"
    with pytest.raises(ConfigError, match=r"params\.surprise"):
        build("vlm_client", {"name": "fake", "params": {"surprise": 1}}, build_ctx)


class _Inputs(dict):
    def to(self, device):
        self.device = device
        return self


class _StubProcessor:
    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.kwargs = messages, kwargs
        return _Inputs(input_ids=np.zeros((1, 5), dtype=int))

    def batch_decode(self, tokens, skip_special_tokens):
        self.decoded = tokens
        return ['{"brand": "Acme"}']


class _StubModel:
    def generate(self, **kwargs):
        self.kwargs = kwargs
        return np.arange(8).reshape(1, 8)


class StubbedTransformers(TransformersVLMClient):
    def _load_backend(self):
        self._inference_mode = contextlib.nullcontext
        self._processor, self._model = _StubProcessor(), _StubModel()


def test_transformers_client_builds_a_chat_with_crop_first_and_decodes_only_new_tokens():
    ctx = BuildContext(CATEGORIES, "cpu", 0)
    client = StubbedTransformers(StubbedTransformers.Params(), ctx)
    crop, scene = make_image("noise", 20, 30), make_image("gradient", 40, 60)
    reply = client.generate("SYSTEM", [crop, scene], "USER")
    assert reply == '{"brand": "Acme"}'

    system, user = client._processor.messages
    assert system["role"] == "system" and system["content"][0]["text"] == "SYSTEM"
    kinds = [part["type"] for part in user["content"]]
    assert kinds == ["image", "image", "text"] and user["content"][2]["text"] == "USER"
    assert np.asarray(user["content"][0]["image"]).shape == crop.shape  # crop is image 1
    assert client._processor.kwargs["add_generation_prompt"] is True
    assert client._model.kwargs["do_sample"] is False  # temperature 0 -> greedy
    assert client._model.kwargs["max_new_tokens"] == 256
    assert client._processor.decoded.shape == (1, 3)  # only tokens after the 5 prompt tokens


def test_transformers_client_samples_only_when_temperature_is_positive():
    ctx = BuildContext(CATEGORIES, "cpu", 0)
    params = StubbedTransformers.Params(temperature=0.7, max_new_tokens=32)
    client = StubbedTransformers(params, ctx)
    client.generate("S", [make_image("noise", 10, 10)], "U")
    assert client._model.kwargs["do_sample"] is True
    assert client._model.kwargs["temperature"] == 0.7
    assert client._model.kwargs["max_new_tokens"] == 32
