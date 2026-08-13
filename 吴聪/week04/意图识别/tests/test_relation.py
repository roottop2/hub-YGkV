# -*- coding: utf-8 -*-
"""R-BERT 关系抽取模块测试。"""
import torch
from transformers import AutoTokenizer

from ecommerce_qa import config
from ecommerce_qa.relation_extraction.dataset import (
    REL_SPECIAL_TOKENS, build_marked_text, collate, locate,
)
from ecommerce_qa.relation_extraction.model import RBERT


def test_locate_tail_inside_head():
    """tail 是 head 的子串时，应定位到 head 之后的那一个。"""
    # 文本：小(0)米(1)1(2)4(3) (4)是(5) (6)小(7)米(8)
    hs, he, ts, te = locate("小米14 是 小米 生产的", "小米14", "小米")
    assert hs == 0 and he == 4        # "小米14" 是 4 个字符，占 0..4
    assert ts == 7 and te == 9        # head 之后的"小米"在 7..9


def test_marked_text_head_first():
    """head 在 tail 前面时，标记顺序正确。"""
    marked = build_marked_text("订单 202608130001 包含 小米15", "202608130001", "小米15")
    assert "[E1]202608130001[/E1]" in marked
    assert "[E2]小米15[/E2]" in marked
    assert marked.index("[E1]") < marked.index("[E2]")


def test_marked_text_tail_first():
    """tail 在 head 前面时，标记顺序相反。"""
    marked = build_marked_text("小米15 属于 手机", "小米15", "手机")
    # head="小米15" 在前，所以 [E1] 仍在 [E2] 前面
    assert marked.index("[E1]") < marked.index("[E2]")
    assert "[E1]小米15[/E1]" in marked


def test_collate_padding():
    """collate 应把不定长序列填充成 (B, max_len)。"""
    batch = [([1, 2, 3], 1, 2, 0), ([4, 5], 1, 1, 1)]
    ids, masks, e1, e2, labels = collate(batch, pad_id=0)
    assert ids.shape == (2, 3)
    assert masks[1].tolist() == [1, 1, 0]
    assert e1.tolist() == [1, 1]
    assert labels.tolist() == [0, 1]


def test_model_forward():
    """R-BERT 前向传播输出形状正确。"""
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": REL_SPECIAL_TOKENS})
    model = RBERT(num_relations=3)
    model.bert.resize_token_embeddings(len(tokenizer))

    text = "[E1]耐克[/E1] 是 [E2]Air Max[/E2] 生产的"
    enc = tokenizer(text)
    ids = enc["input_ids"]
    e1_id = tokenizer.convert_tokens_to_ids("[E1]")
    e2_id = tokenizer.convert_tokens_to_ids("[E2]")

    logits = model(
        torch.tensor([ids]),
        torch.tensor([enc["attention_mask"]]),
        torch.tensor([ids.index(e1_id)]),
        torch.tensor([ids.index(e2_id)]),
    )
    assert logits.shape == (1, 3)
