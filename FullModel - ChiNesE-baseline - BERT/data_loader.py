# -*- coding: utf-8 -*-
"""
ChiNesE 数据载入器：兼容当前旧版 GlobalPointer 训练脚本。

支持格式：
1. ChiNesE / Mulco 常见 JSONL：
   {"id": 360, "text": "...", "label": {"Person": {"罗成": [[3, 4]]}}}
2. JSON 数组，每个元素格式同上；
3. 兼容 entities 列表格式：
   {"text": "...", "entities": [{"start_idx": 0, "end_idx": 1, "type": "Person"}]}

事实依据：
- 你给出的 ChiNesE 样例是 label -> 实体类型 -> 实体文本 -> [[start, end], ...]；
- 样例中的 span 为字符级闭区间，例如 [[3, 4]] 对应两个字符“罗成”；
- ChiNesE 中存在嵌套/重叠实体，所以同一句样本允许追加多个 span 标签。
"""
import json
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from config import MAX_LEN
except Exception:
    MAX_LEN = 256

max_len = MAX_LEN

# ChiNesE 10 类标签。保持原数据集大小写，避免和 CLUENER 小写标签混淆。
ent2id = {
    "Person": 0,
    "Location": 1,
    "Organization": 2,
    "Time": 3,
    "Work": 4,
    "Food": 5,
    "Product": 6,
    "Medicine": 7,
    "Event": 8,
    "Creature": 9,
}

# 兼容少数资料/转换脚本可能使用的同义类别名。
LABEL_ALIASES = {
    "Organism": "Creature",
    "Organisms": "Creature",
    "PER": "Person",
    "PERSON": "Person",
    "LOC": "Location",
    "LOCATION": "Location",
    "ORG": "Organization",
    "ORGANIZATION": "Organization",
}

id2ent = {v: k for k, v in ent2id.items()}


def _normalize_label(label: str) -> str:
    if label is None:
        return label
    label = str(label).strip()
    return LABEL_ALIASES.get(label, label)


def _read_records(path: str) -> List[Dict[str, Any]]:
    """兼容 JSON 数组和 JSONL。"""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    if not text:
        return []
    if text[0] == '[':
        return json.loads(text)

    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _append_entity(item: List, text: str, start: int, end: int, label: str) -> None:
    """
    加入一个闭区间字符级实体。
    ChiNesE 样例 span 使用闭区间：[start, end] 均包含实体字符。
    """
    label = _normalize_label(label)
    if label not in ent2id:
        return
    if start is None or end is None:
        return

    try:
        start = int(start)
        end = int(end)
    except Exception:
        return

    if 0 <= start <= end < len(text):
        item.append((start, end, ent2id[label]))


def load_data(path):
    """
    加载 ChiNesE 数据，输出格式：
        [[text, (char_start, char_end_closed, label_id), ...], ...]

    该内部格式与当前 GlobalPointer 训练脚本保持一致。
    """
    D = []
    records = _read_records(path)

    for d in records:
        text = d.get('text', '')
        item = [text]

        # ChiNesE 常见格式：label -> mention -> [[start, end], ...]
        label_dict = d.get('label')
        if isinstance(label_dict, dict):
            for raw_label, mention_map in label_dict.items():
                label = _normalize_label(raw_label)
                if label not in ent2id:
                    continue

                if isinstance(mention_map, dict):
                    for _, spans in mention_map.items():
                        for span in spans:
                            if isinstance(span, (list, tuple)) and len(span) >= 2:
                                _append_entity(item, text, span[0], span[1], label)
                elif isinstance(mention_map, list):
                    for span in mention_map:
                        if isinstance(span, (list, tuple)) and len(span) >= 2:
                            _append_entity(item, text, span[0], span[1], label)

        # 兼容通用 entities 列表格式。
        entities = d.get('entities')
        if isinstance(entities, list):
            for e in entities:
                if not isinstance(e, dict):
                    continue
                start = e.get('start_idx', e.get('start'))
                end = e.get('end_idx', e.get('end'))
                label = e.get('type', e.get('label'))
                # 本版本默认 end 已经是闭区间，不做 end - 1。
                _append_entity(item, text, start, end, label)

        D.append(item)

    return D


class EntDataset(Dataset):
    def __init__(self, data, tokenizer, istrain=True):
        self.data = data
        self.tokenizer = tokenizer
        self.istrain = istrain

    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        if self.istrain:
            text = item[0]
            encoder_txt = self.tokenizer.encode_plus(
                text,
                max_length=max_len,
                truncation=True,
                return_offsets_mapping=True,
            )
            input_ids = encoder_txt["input_ids"]
            token_type_ids = encoder_txt.get("token_type_ids", [0] * len(input_ids))
            attention_mask = encoder_txt["attention_mask"]
            token2char_span_mapping = encoder_txt["offset_mapping"]
            return text, token2char_span_mapping, input_ids, token_type_ids, attention_mask
        raise NotImplementedError('EntDataset 当前仅用于训练/验证。预测请使用 predict_CME.py。')

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        """将序列 padding 到同一长度。"""
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)
        return np.array(outputs)

    def collate(self, examples):
        raw_text_list, batch_input_ids, batch_attention_mask, batch_labels, batch_segment_ids = [], [], [], [], []

        for item in examples:
            raw_text, token2char_span_mapping, input_ids, token_type_ids, attention_mask = self.encoder(item)

            labels = np.zeros((len(ent2id), max_len, max_len))
            for start, end, label in item[1:]:
                start_token_idx, end_token_idx = -1, -1

                for idx, (token_start, token_end) in enumerate(token2char_span_mapping):
                    if token_start <= start < token_end:
                        start_token_idx = idx
                        break

                for idx, (token_start, token_end) in enumerate(token2char_span_mapping):
                    if token_start <= end < token_end:
                        end_token_idx = idx
                        break

                if start_token_idx != -1 and end_token_idx != -1 and start_token_idx <= end_token_idx:
                    labels[label, start_token_idx, end_token_idx] = 1

            raw_text_list.append(raw_text)
            batch_input_ids.append(input_ids)
            batch_segment_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels[:, :len(input_ids), :len(input_ids)])

        batch_inputids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_segmentids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
        batch_attentionmask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()
        batch_labels = torch.tensor(self.sequence_padding(batch_labels, seq_dims=3)).long()

        return raw_text_list, batch_inputids, batch_attentionmask, batch_segmentids, batch_labels

    def __getitem__(self, index):
        return self.data[index]
