"""``_ram_compat``: shims that let ``ram`` import on transformers 5.x. Needs transformers only."""

import sys
import types

import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")

from electronics_classifier.components.detectors import _ram_compat as compat  # noqa: E402

NAMES = ("apply_chunking_to_forward", "prune_linear_layer", "find_pruneable_heads_and_indices")


HELPERS = ("invert_attention_mask", "get_head_mask", "_convert_head_mask_to_5d")


@pytest.fixture
def modeling_utils(monkeypatch):
    from transformers import modeling_utils as mu

    # Start from "the names are gone" whatever the installed version; monkeypatch restores them.
    for name in NAMES:
        monkeypatch.delattr(mu, name, raising=False)
    monkeypatch.delattr(mu.PreTrainedModel, "all_tied_weights_keys", raising=False)
    for name in HELPERS:
        monkeypatch.delattr(mu.PreTrainedModel, name, raising=False)
    return mu


def test_the_moved_and_removed_names_are_back_after_patching(modeling_utils):
    compat.patch_transformers()
    for name in NAMES:
        assert callable(getattr(modeling_utils, name))
    assert modeling_utils.PreTrainedModel.all_tied_weights_keys == {}


def test_patching_twice_changes_nothing_the_second_time(modeling_utils):
    compat.patch_transformers()
    first = modeling_utils.find_pruneable_heads_and_indices
    compat.patch_transformers()
    assert modeling_utils.find_pruneable_heads_and_indices is first


def test_existing_names_are_never_overwritten(modeling_utils):
    sentinel = object()
    modeling_utils.apply_chunking_to_forward = sentinel
    compat.patch_transformers()
    assert modeling_utils.apply_chunking_to_forward is sentinel


def test_the_replacement_head_pruning_helper_matches_the_old_behaviour(modeling_utils):
    import torch

    compat.patch_transformers()
    heads, index = modeling_utils.find_pruneable_heads_and_indices({1}, 4, 2, set())
    assert heads == {1}
    assert torch.equal(index, torch.tensor([0, 1, 4, 5, 6, 7]))  # head 1 owns slots 2 and 3
    # head 0 was pruned earlier, so 3 heads remain and original head 2 now sits at position 1
    heads, index = modeling_utils.find_pruneable_heads_and_indices({2}, 3, 2, {0})
    assert heads == {2} and torch.equal(index, torch.tensor([0, 1, 4, 5]))


def test_the_attention_mask_helpers_are_restored_with_their_4x_behaviour(modeling_utils):
    import torch

    compat.patch_transformers()
    model = types.SimpleNamespace(dtype=torch.float32)
    convert = modeling_utils.PreTrainedModel._convert_head_mask_to_5d
    model._convert_head_mask_to_5d = lambda mask, n: convert(model, mask, n)
    invert = modeling_utils.PreTrainedModel.invert_attention_mask
    mask = invert(model, torch.tensor([[1, 0, 1]]))
    assert mask.shape == (1, 1, 1, 3)  # broadcasts over heads and query positions
    assert mask[0, 0, 0, 0] == 0 and mask[0, 0, 0, 1] == torch.finfo(torch.float32).min
    assert invert(model, torch.ones(2, 3, 3)).shape == (2, 1, 3, 3)
    with pytest.raises(ValueError):
        invert(model, torch.ones(3))

    get_head_mask = modeling_utils.PreTrainedModel.get_head_mask
    assert get_head_mask(model, None, 4) == [None] * 4
    assert get_head_mask(model, torch.ones(2), 3).shape == (3, 1, 2, 1, 1)


def test_the_tokenizer_patch_replaces_ram_init_tokenizer_only_where_it_exists(monkeypatch):
    from transformers import BertTokenizer

    monkeypatch.delattr(BertTokenizer, "additional_special_tokens_ids", raising=False)
    fake = types.ModuleType("ram.models.utils")
    fake.init_tokenizer = original = lambda name: None
    other = types.ModuleType("ram.models.ram")  # a module without the function is left alone
    monkeypatch.setitem(sys.modules, "ram.models.utils", fake)
    monkeypatch.setitem(sys.modules, "ram.models.ram", other)
    compat.patch_ram_tokenizer()
    assert fake.init_tokenizer is not original
    assert not hasattr(other, "init_tokenizer")
