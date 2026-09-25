# -*- coding: utf-8 -*-
"""
CLUENER 数据载入器：兼容当前旧版 GlobalPointer 训练脚本。

支持格式：
1. CLUENER 官方常见 JSONL：
   {"text": "...", "label": {"name": {"张三": [[0, 1]]}}}
2. JSON 数组，每个元素格式同上；
3. 兼容 entities 列表格式：
   {"text": "...", "entities": [{"start_idx": 0, "end_idx": 1, "type": "name"}]}

注意：CLUENER 的 span 通常是闭区间，即 [start, end] 均包含实体字符。
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

ent2id = {
    "address": 0,
    "book": 1,
    "company": 2,
    "game": 3,
    "government": 4,
    "movie": 5,
    "name": 6,
    "organization": 7,
    "position": 8,
    "scene": 9,
}
id2ent = {v: k for k, v in ent2id.items()}


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
    """加入一个闭区间字符级实体。"""
    if label not in ent2id:
        return
    if start is None or end is None:
        return
    start = int(start)
    end = int(end)
    if 0 <= start <= end < len(text):
        item.append((start, end, ent2id[label]))


def load_data(path):
    """
    加载 CLUENER 数据，输出格式：
        [[text, (char_start, char_end_closed, label_id), ...], ...]
    """
    D = []
    records = _read_records(path)
    for d in records:
        text = d.get('text', '')
        item = [text]

        # CLUENER 官方常见格式：label -> mention -> [[start, end], ...]
        label_dict = d.get('label')
        if isinstance(label_dict, dict):
            for label, mention_map in label_dict.items():
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

        # 兼容你之前 CMeEE 风格/通用转换后的 entities 列表格式。
        entities = d.get('entities')
        if isinstance(entities, list):
            for e in entities:
                start = e.get('start_idx', e.get('start'))
                end = e.get('end_idx', e.get('end'))
                label = e.get('type', e.get('label'))
                # 对 CLUENER，本版本默认 end 已经是闭区间，不再 end - 1。
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
