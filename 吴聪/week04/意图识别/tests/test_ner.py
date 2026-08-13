# -*- coding: utf-8 -*-
"""NER 模块测试：CRF 算法、模型训练冒烟。"""
import torch
from torch.utils.data import DataLoader

from ecommerce_qa.ner.crf import CRF
from ecommerce_qa.ner.dataset import NerDataset, build_label_map, build_vocab, collate
from ecommerce_qa.ner.model import BiLSTMCRF


def test_crf_decode_shape():
    """Viterbi 解码应返回 (B, T) 且在合法标签范围内。"""
    crf = CRF(num_labels=3)
    T, B, L = 5, 2, 3
    emissions = torch.randn(T, B, L)
    mask = torch.ones(T, B, dtype=torch.long)
    paths = crf.decode(emissions, mask)
    assert paths.shape == (B, T)
    assert (paths >= 0).all() and (paths < L).all()


def test_crf_nll_finite():
    """负对数似然应是有限标量，且数学上必然 >= 0。"""
    crf = CRF(num_labels=3)
    T, B, L = 5, 2, 3
    emissions = torch.randn(T, B, L)
    mask = torch.ones(T, B, dtype=torch.long)
    tags = torch.randint(0, L, (T, B))
    loss = crf.nll(emissions, tags, mask)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    # logZ >= 任何单条路径分数，所以 NLL = logZ - 路径分数 >= 0
    # （负值说明 forward_alg 算错了，训练方向会反）
    assert loss.item() >= 0.0, f"NLL 应为非负，实际 {loss.item()}"


def test_crf_better_path_lower_nll():
    """发射分数更高的路径，其 NLL 应更低（训练信号方向正确）。"""
    crf = CRF(num_labels=2)
    T, B, L = 4, 1, 2
    # 位置 1~3 的标签 0 发射分数明显更高 -> 全 0 路径应更优
    emissions = torch.tensor([
        [[0.0, 0.0]],
        [[1.0, 0.0]],
        [[1.0, 0.0]],
        [[1.0, 0.0]],
    ])
    mask = torch.ones(T, B, dtype=torch.long)
    correct = torch.zeros(T, B, dtype=torch.long)   # 全 0
    wrong = torch.ones(T, B, dtype=torch.long)      # 全 1
    nll_correct = crf.nll(emissions, correct, mask)
    nll_wrong = crf.nll(emissions, wrong, mask)
    assert nll_correct < nll_wrong


def test_model_train_smoke():
    """重复同一条标注训练几个 epoch，损失应下降且解码正确。

    注意：这里用大量重复的固定标注（而非几条随机句子），
    保证信号足够强，避免小数据下损失波动导致的误报。
    """
    torch.manual_seed(42)
    records = (
        [{"tokens": list("耐克"), "tags": ["B-BRAND", "I-BRAND"]}] * 16
        + [{"tokens": list("手机"), "tags": ["O", "O"]}] * 16
    )
    vocab = build_vocab(records)
    label_map = build_label_map()
    dataset = NerDataset(records, vocab, label_map)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate)

    model = BiLSTMCRF(len(vocab), 16, 16, len(label_map))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch_losses = []
    for _ in range(6):
        total, n = 0.0, 0
        for tokens, tags, mask in loader:
            loss = model.loss(tokens, tags, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        epoch_losses.append(total / n)

    assert epoch_losses[-1] < epoch_losses[0], (
        f"损失没有下降: {epoch_losses[0]} -> {epoch_losses[-1]}"
    )

    # 解码验证：模型应学会 耐克->BRAND
    model.eval()
    chars = list("耐克")
    tok = torch.tensor([[vocab.get(c, vocab["<UNK>"]) for c in chars]])
    m = torch.ones(1, len(chars), dtype=torch.long)
    ids = model.decode(tok, m)[0].tolist()
    id2label = {v: k for k, v in label_map.items()}
    labels = [id2label[i] for i in ids]
    assert labels == ["B-BRAND", "I-BRAND"], f"解码结果错误: {labels}"
