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


# --- category_open -------------------------------------------------------------------


def test_category_open_keeps_vocabulary_categories_without_a_warning(build_ctx):
    out, warnings = post(build_ctx, "category_open").apply({"category": " Smart  Speaker"}, item_ctx())
    assert out["category"] == "smart speaker" and warnings == []


def test_category_open_keeps_a_novel_category_and_flags_it(build_ctx):
    out, warnings = post(build_ctx, "category_open").apply({"category": "Electric Kettle"}, item_ctx())
    assert out["category"] == "electric kettle"
    assert warnings == ["CATEGORY_NOVEL"]


@pytest.mark.parametrize(
    "category", ["unknown", "", None, "12", "a device that boils water for tea and coffee"]
)
def test_category_open_falls_back_for_empty_or_sentence_categories(build_ctx, category):
    out, warnings = post(build_ctx, "category_open").apply({"category": category}, item_ctx())
    assert out["category"] == "electronic device"
    assert warnings == ["CATEGORY_OUT_OF_VOCAB"]


def test_category_mismatch_ignores_novel_categories(build_ctx):
    from electronics_classifier.schemas import ClassifiedItem, SchemaOrgProductAttributes

    rule = build("warning_rule", {"name": "category_mismatch"}, build_ctx)
    item = item_ctx()
    draft = ClassifiedItem(
        id="obj_1", bbox_xyxy=[0, 0, 1, 1], detector_phrase="router", detector_confidence=0.9,
        attributes=SchemaOrgProductAttributes(category="kettle"),
    )  # fmt: skip
    assert rule.evaluate(draft, item) == []


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


# --- openai_compatible client: real SDK, real HTTP, against a tiny local mock server -------

import base64  # noqa: E402
import builtins  # noqa: E402
import io  # noqa: E402
import socket  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

from electronics_classifier.assembler import result_to_json  # noqa: E402,F401
from electronics_classifier.config import dump_yaml  # noqa: E402
from electronics_classifier.pipeline import ElectronicsClassificationPipeline  # noqa: E402

REPLY_JSON = '{"brand": "Acme", "model": "unknown", "category": "router", "color": "black"}'


class MockServer:
    """An OpenAI-compatible server (models list + chat completions) on 127.0.0.1."""

    def __init__(self):
        self.reply: str | None = REPLY_JSON
        self.models = ["mock-vlm"]
        self.fail_status = 0  # every chat request answers with this HTTP status
        self.fail_first = 0  # only the first N chat requests fail (with 429)
        self.chats: list[dict] = []  # recorded chat requests: {"headers": ..., "body": ...}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                data = [{"id": m, "object": "model", "created": 0, "owned_by": "mock"}
                        for m in outer.models]  # fmt: skip
                self._send(200, {"object": "list", "data": data})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                headers = {k.lower(): v for k, v in self.headers.items()}
                outer.chats.append({"headers": headers, "body": body})
                if outer.fail_status:
                    return self._send(outer.fail_status, {"error": {"message": "boom"}})
                if len(outer.chats) <= outer.fail_first:
                    return self._send(429, {"error": {"message": "slow down"}})
                message = {"role": "assistant", "content": outer.reply}
                self._send(200, {
                    "id": "x", "object": "chat.completion", "created": 0, "model": "mock-vlm",
                    "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                })  # fmt: skip

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self):
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def server():
    mock = MockServer()
    yield mock
    mock.close()


def make_client(url, **params):
    spec = {"name": "openai_compatible", "params": {"base_url": url, "model": "mock-vlm", **params}}
    return build("vlm_client", spec, BuildContext(CATEGORIES, "cpu", 0), path="pipeline.vlm.client")


def test_openai_client_sends_lossless_pngs_crop_first_and_returns_the_reply(server):
    client = make_client(server.url)
    crop, scene = make_image("noise", 40, 60), make_image("gradient", 30, 50)
    assert client.generate("SYSTEM TEXT", [crop, scene], "USER TEXT") == REPLY_JSON

    body = server.chats[0]["body"]
    assert (body["model"], body["temperature"], body["max_tokens"]) == ("mock-vlm", 0.0, 256)
    system, user = body["messages"]
    assert system == {"role": "system", "content": "SYSTEM TEXT"}
    assert [part["type"] for part in user["content"]] == ["image_url", "image_url", "text"]
    assert user["content"][2]["text"] == "USER TEXT"
    from PIL import Image

    for part, original in zip(user["content"][:2], (crop, scene), strict=True):
        prefix, payload = part["image_url"]["url"].split(",", 1)
        assert prefix == "data:image/png;base64"
        decoded = np.array(Image.open(io.BytesIO(base64.b64decode(payload))))
        assert (decoded == original).all()  # lossless: small label text survives the trip


def test_no_key_for_local_servers_and_a_key_from_the_environment_otherwise(server, monkeypatch):
    tiny = [make_image("noise", 8, 8)]
    make_client(server.url).generate("s", tiny, "u")
    assert server.chats[-1]["headers"]["authorization"] == "Bearer not-needed"
    monkeypatch.setenv("MY_VLM_KEY", "sekret-123")
    make_client(server.url, api_key_env="MY_VLM_KEY").generate("s", tiny, "u")
    assert server.chats[-1]["headers"]["authorization"] == "Bearer sekret-123"


def test_the_api_key_never_appears_in_params_the_resolved_config_or_errors(server, monkeypatch):
    monkeypatch.setenv("MY_VLM_KEY", "sekret-123")
    client = make_client(server.url, api_key_env="MY_VLM_KEY")
    assert "sekret-123" not in repr(client.params)
    assert "sekret-123" not in str(client.params.model_dump())

    pipeline = ElectronicsClassificationPipeline(
        sets=[
            "pipeline.detector.name=fake",
            "pipeline.vlm.client={name: openai_compatible, params: "
            f"{{base_url: '{server.url}', model: mock-vlm, api_key_env: MY_VLM_KEY}}}}",
        ]
    )
    assert "sekret-123" not in dump_yaml(pipeline.config)  # what resolved_config.yaml contains
    assert "MY_VLM_KEY" in dump_yaml(pipeline.config)  # only the variable NAME is recorded

    monkeypatch.delenv("MY_VLM_KEY")
    expected = r"pipeline\.vlm\.client\.params\.api_key_env.*MY_VLM_KEY"
    with pytest.raises(ConfigError, match=expected):
        make_client(server.url, api_key_env="MY_VLM_KEY")
    monkeypatch.setenv("MY_VLM_KEY", "")
    with pytest.raises(ConfigError, match="not set or empty"):
        make_client(server.url, api_key_env="MY_VLM_KEY")


def test_build_time_check_reports_an_unreachable_server_and_a_missing_model(server):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]  # nothing listens here once the socket closes
    with pytest.raises(ConfigError, match=r"params\.base_url: cannot reach a model server"):
        make_client(f"http://127.0.0.1:{dead_port}/v1")

    server.models = ["other-a", "other-b"]
    with pytest.raises(ConfigError, match=r"params\.model: 'mock-vlm' is not available.*other-a"):
        make_client(server.url)
    assert make_client(server.url, check_server=False)  # the escape hatch skips both checks
    server.models = []  # a server that lists nothing is accepted
    assert make_client(server.url)


def test_bad_urls_and_params_are_config_errors_naming_the_path():
    for bad_url in ("localhost:1234", "ftp://x/v1", "", "http://"):
        with pytest.raises(ConfigError, match=r"pipeline\.vlm\.client\.params\.base_url"):
            make_client(bad_url, check_server=False)
    invalid = (("temperature", -1), ("max_tokens", 0), ("timeout_s", 0), ("max_retries", -1))
    for field, value in invalid:
        with pytest.raises(ConfigError, match=rf"params\.{field}"):
            make_client("http://127.0.0.1:1/v1", check_server=False, **{field: value})
    spec = {"name": "openai_compatible", "params": {}}  # model is required
    with pytest.raises(ConfigError, match=r"params\.model"):
        build("vlm_client", spec, BuildContext(CATEGORIES, "cpu", 0), path="pipeline.vlm.client")


def test_a_warning_names_the_host_when_images_leave_this_machine(server, caplog):
    with caplog.at_level(logging.WARNING):
        make_client(server.url)  # 127.0.0.1: nothing leaves the machine
        assert not [r for r in caplog.records if "sends crops" in r.message]
        make_client("https://models.example.test/v1", check_server=False)
    warnings = [r.message for r in caplog.records if "sends crops" in r.message]
    assert warnings == ["openai_compatible sends crops and context images to models.example.test"]


def test_requests_are_retried_on_429_and_http_errors_become_a_parse_failed_fallback(server):
    server.fail_first = 1  # first chat request: HTTP 429, then success
    client = make_client(server.url, max_retries=2)
    assert client.generate("s", [make_image("noise", 8, 8)], "u") == REPLY_JSON
    assert len(server.chats) == 2

    server.chats.clear()
    server.fail_status = 500
    extractor = AttributeExtractor(make_client(server.url, max_retries=0), CATEGORIES, None, 1)
    result = extractor.extract(make_prepared_item())
    assert result.warnings == ["VLM_PARSE_FAILED"] and len(server.chats) == 1  # no retry on error
    assert result.attributes["category"] == FALLBACK_CATEGORY


def test_an_empty_reply_is_a_parse_failure_not_a_crash(server):
    server.reply = None  # the server answers with `content: null`
    extractor = AttributeExtractor(make_client(server.url), CATEGORIES, None, 0)
    assert extractor.extract(make_prepared_item()).warnings == ["VLM_PARSE_FAILED"]


def test_the_pipeline_runs_end_to_end_through_the_server(server):
    sets = [
        "pipeline.detector.name=fake",
        "pipeline.vlm.client={name: openai_compatible, params: "
        f"{{base_url: '{server.url}', model: mock-vlm}}}}",
    ]
    pipeline = ElectronicsClassificationPipeline(sets=sets)
    _, result = pipeline.run(make_image("noise", 480, 640))
    [item] = result.items
    assert item.attributes.brand == "Acme" and item.attributes.model == "unknown"
    assert "MODEL_UNKNOWN" in item.warnings and "CATEGORY_MISMATCH" not in item.warnings
    assert result.meta.components["vlm_client"] == "openai_compatible"
    [chat] = server.chats  # one request per item, with the crop and the scene
    assert sum(p["type"] == "image_url" for p in chat["body"]["messages"][1]["content"]) == 2
    system_prompt = chat["body"]["messages"][0]["content"]
    assert '"router"' in system_prompt and "model" in system_prompt  # the vocabulary prompt


def test_a_missing_openai_package_carries_an_install_hint(server, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openai" or name.startswith("openai."):
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"electronics-classifier\[server\]"):
        make_client(server.url)


# --- ocr_verify_model: a claimed model name must be visible in the crop ---------------------

import cv2  # noqa: E402

from electronics_classifier.components.attribute_postprocess import (  # noqa: E402
    OcrVerifyModel,
    RapidOcrReader,
    model_is_visible,
    normalize_for_match,
)

pytest.importorskip("rapidfuzz", reason="matching needs: pip install electronics-classifier[ocr]")


class FakeReader:
    def __init__(self, text):
        self.text = text
        self.reads = 0

    def read(self, image):
        self.reads += 1
        return self.text


def ocr_step(text="", available=True, **params):
    """An OcrVerifyModel whose OCR engine is a fake that 'reads' ``text``."""
    made = []

    class Step(OcrVerifyModel):
        def _make_reader(self):
            made.append(1)
            if not available:
                raise ImportError("No module named 'rapidocr'")
            return FakeReader(text)

    return Step(Step.Params(**params), BuildContext(CATEGORIES, "cpu", 0)), made


def verify(step, model):
    return step.apply({"brand": "Acme", "model": model, "category": "router"}, item_ctx())


def test_ocr_is_not_called_or_even_loaded_when_there_is_no_model_claim():
    step, made = ocr_step("anything")
    for claim in ("unknown", "N/A", "none", "  ", None):
        attrs, warnings = verify(step, claim)
        assert warnings == [] and attrs["brand"] == "Acme"
    assert made == []  # not needed, so the engine was never created


def test_a_visible_model_is_kept_and_the_engine_is_loaded_once_and_reused():
    step, made = ocr_step("TP-Link Archer AX50 Wi-Fi 6 Router")
    attrs, warnings = verify(step, "Archer AX50")
    assert (attrs["model"], warnings) == ("Archer AX50", [])
    verify(step, "AX50")
    assert len(made) == 1 and step._reader.reads == 2


@pytest.mark.parametrize(
    "claimed,seen",
    [
        ("XR-5000", "model xr 5000 made in china"),
        ("xr5000", "XR-5000"),
        ("Archer AX50", "ARCHER-AX50"),
    ],
)
def test_matching_ignores_case_spacing_and_punctuation(claimed, seen):
    assert model_is_visible(claimed, seen, 85.0)


def test_a_model_that_cannot_be_found_becomes_unknown_with_a_warning():
    step, _ = ocr_step("Made in China  Input 12V 1A")
    original = {"brand": "Apple", "model": "MacBook Pro", "category": "laptop"}
    attrs, warnings = step.apply(original, item_ctx())
    assert attrs == {"brand": "Apple", "model": "unknown", "category": "laptop"}
    assert warnings == ["MODEL_UNVERIFIED"] and original["model"] == "MacBook Pro"  # input intact


def test_blank_ocr_output_verifies_nothing():
    step, _ = ocr_step("")
    assert verify(step, "XR-5000") == (
        {"brand": "Acme", "model": "unknown", "category": "router"}, ["MODEL_UNVERIFIED"]
    )


def test_a_fragment_of_the_claim_does_not_verify_it():
    assert not model_is_visible("XR-5000", "XR5", 85.0)  # OCR saw only part of the claim
    assert model_is_visible("XR5", "XR-5000", 85.0)  # but a short claim inside longer text is fine


def test_short_names_must_match_exactly_but_long_names_tolerate_ocr_slips():
    assert model_is_visible("A1", "model A1", 85.0)
    assert not model_is_visible("A1", "model AI", 85.0)  # one wrong character: rejected
    assert model_is_visible("Archer AX50", "ARCHER AX5O", 85.0)  # 0 read as O: accepted
    assert not model_is_visible("Archer AX50", "ARCHER AX5O", 100.0)  # unless made strict
    assert normalize_for_match("xr-5000 !") == "XR5000"


def test_without_rapidocr_claims_are_reported_unverified_with_one_warning(caplog):
    step, made = ocr_step(available=False)
    with caplog.at_level(logging.WARNING):
        first = verify(step, "XR-5000")
        second = verify(step, "AX50")
    for attrs, warnings in (first, second):
        assert attrs["model"] == "unknown" and warnings == ["MODEL_UNVERIFIED"]
    assert len(made) == 1  # tried once, then remembered
    notices = [r for r in caplog.records if "RapidOCR is not installed" in r.message]
    assert len(notices) == 1 and "electronics-classifier[ocr]" in notices[0].message
    assert verify(step, "unknown")[1] == []  # nothing to verify, nothing flagged


def test_if_unavailable_error_fails_when_the_pipeline_is_built():
    with pytest.raises(ImportError, match=r"electronics-classifier\[ocr\]"):
        ocr_step(available=False, if_unavailable="error")


def test_ocr_params_are_validated(build_ctx):
    for params, field in (
        ({"min_confidence": 2}, "min_confidence"),
        ({"min_match": 101}, "min_match"),
        ({"if_unavailable": "ignore"}, "if_unavailable"),
        ({"surprise": 1}, "surprise"),
    ):
        with pytest.raises(ConfigError, match=rf"params\.{field}"):
            post(build_ctx, "ocr_verify_model", **params)


def test_the_reader_filters_low_confidence_lines_and_passes_opencv_colour_order():
    seen = {}

    class Engine:
        def __call__(self, image):
            seen["image"] = image
            return type("R", (), {"txts": ("GOOD", "noise", "OK"), "scores": (0.9, 0.2, 0.5)})()

    reader = RapidOcrReader(0.5, engine=Engine())
    rgb = np.zeros((2, 2, 3), np.uint8)
    rgb[..., 0] = 200  # pure red in RGB
    assert reader.read(rgb) == "GOOD OK"  # the 0.2 line is dropped, 0.5 is kept
    assert seen["image"][0, 0].tolist() == [0, 0, 200]  # the engine got BGR

    empty = type("R", (), {"txts": None, "scores": None})()
    assert RapidOcrReader(0.5, engine=lambda image: empty).read(rgb) == ""


def label_crop(*lines):
    image = np.full((90 * len(lines) + 30, 700, 3), 235, np.uint8)
    for row, text in enumerate(lines):
        origin = (20, 70 + 90 * row)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.8, (20, 20, 20), 3)
    return image


def test_the_real_rapidocr_engine_verifies_printed_text_and_rejects_the_rest(build_ctx):
    pytest.importorskip("rapidocr", reason="needs: pip install electronics-classifier[ocr]")
    step = post(build_ctx, "ocr_verify_model")
    printed = make_prepared_item()
    printed.crop_rgb[...] = 0  # replace the noise crop with a real label
    label = label_crop("TP-Link", "Archer AX50")
    printed = printed.model_copy(update={"crop_rgb": label})
    ctx = ItemContext(printed, 0, [printed.detection], 300, 200)

    assert step.apply({"model": "Archer AX50"}, ctx) == ({"model": "Archer AX50"}, [])
    assert step.apply({"model": "XR-5000"}, ctx) == ({"model": "unknown"}, ["MODEL_UNVERIFIED"])
    blank = printed.model_copy(update={"crop_rgb": np.full((200, 200, 3), 128, np.uint8)})
    blank_ctx = ItemContext(blank, 0, [blank.detection], 300, 200)
    expected = ({"model": "unknown"}, ["MODEL_UNVERIFIED"])
    assert step.apply({"model": "XR-5000"}, blank_ctx) == expected


def test_ocr_verification_is_on_by_default_and_only_loads_when_a_model_is_claimed(monkeypatch):
    from conftest import make_pipeline

    from electronics_classifier.config import load_config

    names = [step["name"] for step in load_config()["pipeline"]["attribute_postprocess"]]
    assert names == ["normalize_unknown", "ocr_verify_model", "category_vocab"]

    def must_not_load(self, *args, **kwargs):
        raise AssertionError("the OCR engine was loaded although no model was claimed")

    monkeypatch.setattr(RapidOcrReader, "__init__", must_not_load)
    no_model = '{"brand": "Acme", "model": "unknown", "category": "router", "color": "black"}'
    pipeline = make_pipeline([det_box()], [no_model], ocr=True)
    _, result = pipeline.run(make_image("noise", 480, 640))
    assert result.items[0].attributes.model == "unknown"
    assert "MODEL_UNVERIFIED" not in result.items[0].warnings


def det_box():
    from conftest import det

    return det("router", 0.9, (100, 100, 400, 300))


def test_the_default_pipeline_keeps_a_visible_model_and_rejects_an_invented_one():
    pytest.importorskip("rapidocr", reason="needs: pip install electronics-classifier[ocr]")
    from conftest import make_pipeline

    image = make_image("solid", 480, 640)
    image[100:300, 100:640] = label_crop("TP-Link", "Archer AX50")[:200, :540]
    real = '{"brand": "TP-Link", "model": "Archer AX50", "category": "router", "color": "black"}'
    fake = '{"brand": "TP-Link", "model": "XR-5000", "category": "router", "color": "black"}'
    det_ = [det_box()]

    _, kept = make_pipeline(det_, [real], ocr=True).run(image)
    assert kept.items[0].attributes.model == "Archer AX50"
    assert "MODEL_UNVERIFIED" not in kept.items[0].warnings

    _, rejected = make_pipeline(det_, [fake], ocr=True).run(image)
    assert rejected.items[0].attributes.model == "unknown"
    assert rejected.items[0].warnings[:2] == ["MODEL_UNVERIFIED", "MODEL_UNKNOWN"]
