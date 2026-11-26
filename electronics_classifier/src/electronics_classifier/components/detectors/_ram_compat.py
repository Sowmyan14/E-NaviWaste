"""Make ``recognize-anything`` (package ``ram``) importable on ``transformers`` 5.x.

``ram`` bundles a copy of the BERT code written for ``transformers`` 4.2x and was never updated.
Several things it relies on are gone in 5.x. Each shim below is applied only when the name is
missing, so on an older ``transformers`` this module changes nothing.

Remove it when ``ram`` supports ``transformers`` 5.x.
"""

from __future__ import annotations

import sys


def _invert_attention_mask(self, encoder_attention_mask):
    """1 for "attend", 0 for "ignore" becomes 0 and a large negative number (added to scores)."""
    import torch

    if encoder_attention_mask.dim() == 3:
        mask = encoder_attention_mask[:, None, :, :]
    elif encoder_attention_mask.dim() == 2:
        mask = encoder_attention_mask[:, None, None, :]
    else:
        dims = encoder_attention_mask.dim()
        raise ValueError(f"expected a 2-D or 3-D attention mask, got {dims}-D")
    mask = mask.to(dtype=self.dtype)
    return (1.0 - mask) * torch.finfo(self.dtype).min


def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    return head_mask.to(dtype=self.dtype)


def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
    if head_mask is None:
        return [None] * num_hidden_layers
    head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
    return head_mask.unsqueeze(-1) if is_attention_chunked else head_mask


def patch_transformers() -> None:
    """Call after ``import transformers`` and *before* ``import ram``."""
    from transformers import modeling_utils, pytorch_utils

    # Moved from ``modeling_utils`` to ``pytorch_utils``.
    for name in ("apply_chunking_to_forward", "prune_linear_layer"):
        if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))

    # Removed in 5.x. Same behaviour as the 4.x helper.
    if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            import torch

            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head -= sum(1 if pruned < head else 0 for pruned in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            return heads, torch.arange(len(mask))[mask].long()

        modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    # Base-class helpers dropped in 5.x that ram's BERT still calls (4.x behaviour).
    cls = modeling_utils.PreTrainedModel
    if not hasattr(cls, "invert_attention_mask"):
        cls.invert_attention_mask = _invert_attention_mask
    if not hasattr(cls, "get_head_mask"):
        cls.get_head_mask = _get_head_mask
        cls._convert_head_mask_to_5d = _convert_head_mask_to_5d

    # 5.x reads this while tying weights; models written for 4.x never set it.
    if not hasattr(modeling_utils.PreTrainedModel, "all_tied_weights_keys"):
        modeling_utils.PreTrainedModel.all_tied_weights_keys = {}


def patch_ram_tokenizer() -> None:
    """Call after ``import ram``. ``additional_special_tokens_ids`` no longer exists."""
    from transformers import BertTokenizer

    def init_tokenizer(text_encoder_type: str):
        tokenizer = BertTokenizer.from_pretrained(text_encoder_type)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids("[ENC]")
        return tokenizer

    if hasattr(BertTokenizer, "additional_special_tokens_ids"):
        return  # an older transformers: ram's own init_tokenizer works
    # ``ram.models.ram_plus`` the attribute is the function, so go through sys.modules.
    for name in ("ram.models.utils", "ram.models.ram_plus", "ram.models.ram"):
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "init_tokenizer"):
            module.init_tokenizer = init_tokenizer
