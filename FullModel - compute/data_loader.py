# -*- coding: utf-8 -*-
"""
实体识别数据载入器。

Span-Type Semantic Alignment 版本：
- 保留 GlobalPointer span 标签；
- 兼容旧训练脚本的 batch 返回格式，仍返回 start/end boundary label，
  但默认 Boundary-Aware 已关闭，新训练脚本不会使用它们；
- 支持 CMeEE-V2 的 entities 列表格式，也支持 CLUENER 常见 JSONL label 字典格式。
"""
import json
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from config import (
        CLUENER_LABEL_DESCRIPTIONS,
        CME_LABEL_DESCRIPTIONS,
        CUSTOM_ENT2ID,
        DATASET_NAME,
        END_INDEX_IS_EXCLUSIVE,
        MAX_LEN,
    )
except Exception:
    MAX_LEN = 256
    DATASET_NAME = 'CMeEE-V2'
    END_INDEX_IS_EXCLUSIVE = True
    CUSTOM_ENT2ID = None
    CME_LABEL_DESCRIPTIONS = {}
    CLUENER_LABEL_DESCRIPTIONS = {}

max_len = MAX_LEN

CME_ENT2ID = {"bod": 0, "dis": 1, "sym": 2, "mic": 3, "pro": 4, "ite": 5, "dep": 6, "dru": 7, "equ": 8}
CLUENER_ENT2ID = {
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


def _select_ent2id() -> Dict[str, int]:
    if CUSTOM_ENT2ID is not None:
        return dict(CUSTOM_ENT2ID)
    name = str(DATASET_NAME).lower()
    if 'cluener' in name:
        return CLUENER_ENT2ID.copy()
    return CME_ENT2ID.copy()


ent2id = _select_ent2id()
id2ent = {v: k for k, v in ent2id.items()}


def get_label_descriptions() -> Dict[str, str]:
    name = str(DATASET_NAME).lower()
    if 'cluener' in name:
        return {k: CLUENER_LABEL_DESCRIPTIONS.get(k, k) for k in ent2id.keys()}
    return {k: CME_LABEL_DESCRIPTIONS.get(k, k) for k in ent2id.keys()}


def _read_records(path: str) -> List[Dict[str, Any]]:
    """兼容 JSON 数组和 JSONL 两种格式。"""
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


def _append_entity(target: List, text: str, start: int, end: int, label: str) -> None:
    """将字符级实体加入样本。end 使用闭区间。"""
    if label not in ent2id:
        return
    if start is None or end is None:
        return
    start = int(start)
    end = int(end)
    effective_exclusive = bool(END_INDEX_IS_EXCLUSIVE)
    if 'cluener' in str(DATASET_NAME).lower():
        # CLUENER 官方常见标注为闭区间，避免用户切换 DATASET_NAME 后忘记同步修改配置。
        effective_exclusive = False
    if effective_exclusive:
        end = end - 1
    if 0 <= start <= end < len(text):
        target.append((start, end, ent2id[label]))


def load_data(path: str):
    """
    加载训练/验证数据，输出格式：
        [[text, (char_start, char_end_closed, label_id), ...], ...]

    支持：
    1) CMeEE-V2: {'text': str, 'entities': [{'start_idx','end_idx','type'}, ...]}
    2) CLUENER:  {'text': str, 'label': {type: {mention: [[start,end], ...]}}}
    3) 通用 entities: start/end/type 或 start_idx/end_idx/type 字段。
    """
    D = []
    records = _read_records(path)
    for d in records:
        text = d.get('text', '')
        item = [text]

        if 'entities' in d and isinstance(d['entities'], list):
            for e in d['entities']:
                start = e.get('start_idx', e.get('start'))
                end = e.get('end_idx', e.get('end'))
                label = e.get('type', e.get('label'))
                _append_entity(item, text, start, end, label)

        elif 'label' in d and isinstance(d['label'], dict):
            # CLUENER 常见格式：label -> mention -> [[start, end], ...]
            for label, mention_map in d['label'].items():
                if label not in ent2id:
                    continue
                if isinstance(mention_map, dict):
                    for _, spans in mention_map.items():
                        for span in spans:
                            if len(span) >= 2:
                                start, end = span[0], span[1]
                                # CLUENER span 通常是闭区间，若 END_INDEX_IS_EXCLUSIVE=False 不会减 1。
                                _append_entity(item, text, start, end, label)
                elif isinstance(mention_map, list):
                    for span in mention_map:
                        if len(span) >= 2:
                            _append_entity(item, text, span[0], span[1], label)

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

            input_ids = encoder_txt['input_ids']
            token_type_ids = encoder_txt.get('token_type_ids', [0] * len(input_ids))
            attention_mask = encoder_txt['attention_mask']
            token2char_span_mapping = encoder_txt['offset_mapping']
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
        raw_text_list = []
        batch_input_ids = []
        batch_attention_mask = []
        batch_segment_ids = []
        batch_labels = []
        batch_start_boundary_labels = []
        batch_end_boundary_labels = []

        for item in examples:
            raw_text, token2char_span_mapping, input_ids, token_type_ids, attention_mask = self.encoder(item)

            labels = np.zeros((len(ent2id), max_len, max_len), dtype=np.int64)
            # 兼容旧代码；当前 Span-Type 训练默认不使用这两个标签。
            start_boundary_labels = np.zeros(max_len, dtype=np.float32)
            end_boundary_labels = np.zeros(max_len, dtype=np.float32)

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
                    start_boundary_labels[start_token_idx] = 1.0
                    end_boundary_labels[end_token_idx] = 1.0

            seq_len = len(input_ids)
            raw_text_list.append(raw_text)
            batch_input_ids.append(input_ids)
            batch_segment_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels[:, :seq_len, :seq_len])
            batch_start_boundary_labels.append(start_boundary_labels[:seq_len])
            batch_end_boundary_labels.append(end_boundary_labels[:seq_len])

        batch_inputids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_segmentids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
        batch_attentionmask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()
        batch_labels = torch.tensor(self.sequence_padding(batch_labels, seq_dims=3)).long()
        batch_start_boundary_labels = torch.tensor(self.sequence_padding(batch_start_boundary_labels)).float()
        batch_end_boundary_labels = torch.tensor(self.sequence_padding(batch_end_boundary_labels)).float()

        return (
            raw_text_list,
            batch_inputids,
            batch_attentionmask,
            batch_segmentids,
            batch_labels,
            batch_start_boundary_labels,
            batch_end_boundary_labels,
        )

    def __getitem__(self, index):
        return self.data[index]
