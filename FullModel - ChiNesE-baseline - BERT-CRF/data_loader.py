# -*- coding: utf-8 -*-
"""
ChiNesE 数据载入器：BERT-CRF flat NER baseline 版本。

支持格式：
1. ChiNesE / Mulco 常见 JSONL：
   {"id": 360, "text": "...", "label": {"Person": {"罗成": [[3, 4]]}}}
2. JSON 数组，每个元素格式同上；
3. 兼容 entities 列表格式：
   {"text": "...", "entities": [{"start_idx": 0, "end_idx": 1, "type": "Person"}]}

重要事实：
- ChiNesE 是嵌套 NER 数据集；
- BERT-CRF 是 flat 序列标注模型，每个 token 只能有一个 BIO 标签；
- 因此训练标签需要从嵌套实体中选择一组不重叠 span，本代码支持 innermost / outermost 策略；
- 评估时保留完整 gold entities，不因 flat 训练策略删除嵌套实体。
"""
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from config import MAX_LEN, FLAT_NER_STRATEGY
except Exception:
    MAX_LEN = 256
    FLAT_NER_STRATEGY = 'innermost'

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
id2ent = {v: k for k, v in ent2id.items()}

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

# CRF 序列标签：O + B/I-实体类型
crf_label2id = {"O": 0}
for ent_name in ent2id.keys():
    crf_label2id[f"B-{ent_name}"] = len(crf_label2id)
    crf_label2id[f"I-{ent_name}"] = len(crf_label2id)
crf_id2label = {v: k for k, v in crf_label2id.items()}


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


def _append_entity(entities: List[Tuple[int, int, str]], text: str, start: int, end: int, label: str) -> None:
    """加入一个闭区间字符级实体：(start, end, label_name)。"""
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
        entities.append((start, end, label))


def _extract_entities_from_record(d: Dict[str, Any]) -> Tuple[str, List[Tuple[int, int, str]]]:
    """从 ChiNesE 记录中抽取完整 gold entities。"""
    text = d.get('text', '')
    entities: List[Tuple[int, int, str]] = []

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
                            _append_entity(entities, text, span[0], span[1], label)
            elif isinstance(mention_map, list):
                for span in mention_map:
                    if isinstance(span, (list, tuple)) and len(span) >= 2:
                        _append_entity(entities, text, span[0], span[1], label)

    # 兼容通用 entities 列表格式。
    raw_entities = d.get('entities')
    if isinstance(raw_entities, list):
        for e in raw_entities:
            if not isinstance(e, dict):
                continue
            start = e.get('start_idx', e.get('start'))
            end = e.get('end_idx', e.get('end'))
            label = e.get('type', e.get('label'))
            _append_entity(entities, text, start, end, label)

    # 去重，避免同一实体重复出现。
    seen = set()
    unique_entities = []
    for ent in entities:
        if ent not in seen:
            seen.add(ent)
            unique_entities.append(ent)

    return text, unique_entities


def load_data(path):
    """
    加载 ChiNesE 数据，输出格式：
        [{"text": str, "entities": [(char_start, char_end_closed, label_name), ...]}, ...]
    """
    D = []
    records = _read_records(path)
    for d in records:
        text, entities = _extract_entities_from_record(d)
        D.append({"text": text, "entities": entities})
    return D


def select_flat_entities(entities: List[Tuple[int, int, str]], text_len: int, strategy: str = 'innermost') -> List[Tuple[int, int, str]]:
    """
    从嵌套/重叠实体中选择一组不重叠实体，用于 BERT-CRF 的 flat BIO 训练。

    innermost：优先选择更短实体；
    outermost：优先选择更长实体。
    """
    strategy = (strategy or 'innermost').lower()
    if strategy not in {'innermost', 'outermost'}:
        raise ValueError(f"FLAT_NER_STRATEGY 只支持 'innermost' 或 'outermost'，当前为: {strategy}")

    dedup = []
    seen = set()
    for start, end, label in entities:
        key = (int(start), int(end), str(label))
        if key not in seen:
            seen.add(key)
            dedup.append((int(start), int(end), str(label)))

    if strategy == 'innermost':
        # 短 span 优先，保留内层实体。
        sorted_entities = sorted(dedup, key=lambda x: (x[1] - x[0] + 1, x[0], x[1], x[2]))
    else:
        # 长 span 优先，保留外层实体。
        sorted_entities = sorted(dedup, key=lambda x: (-(x[1] - x[0] + 1), x[0], x[1], x[2]))

    occupied = [False] * max(text_len, 0)
    selected: List[Tuple[int, int, str]] = []
    for start, end, label in sorted_entities:
        if start < 0 or end >= text_len or start > end:
            continue
        if any(occupied[start:end + 1]):
            continue
        for i in range(start, end + 1):
            occupied[i] = True
        selected.append((start, end, label))

    # 训练标签按文本顺序生成。
    selected.sort(key=lambda x: (x[0], x[1], x[2]))
    return selected


def _find_token_index(offset_mapping, char_pos: int) -> int:
    for idx, (token_start, token_end) in enumerate(offset_mapping):
        if token_start <= char_pos < token_end:
            return idx
    return -1


class EntDataset(Dataset):
    def __init__(self, data, tokenizer, istrain=True, flat_strategy: str = None):
        self.data = data
        self.tokenizer = tokenizer
        self.istrain = istrain
        self.flat_strategy = flat_strategy or FLAT_NER_STRATEGY

    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        text = item['text']
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

    def _build_bio_labels(self, text: str, token2char_span_mapping, entities: List[Tuple[int, int, str]], seq_len: int) -> List[int]:
        """构造 BERT-CRF 的单层 BIO 标签。"""
        labels = [crf_label2id['O']] * seq_len
        flat_entities = select_flat_entities(entities, len(text), self.flat_strategy)

        for char_start, char_end, ent_type in flat_entities:
            start_token_idx = _find_token_index(token2char_span_mapping, char_start)
            end_token_idx = _find_token_index(token2char_span_mapping, char_end)
            if start_token_idx == -1 or end_token_idx == -1 or start_token_idx > end_token_idx:
                continue
            if start_token_idx >= seq_len or end_token_idx >= seq_len:
                continue
            labels[start_token_idx] = crf_label2id[f'B-{ent_type}']
            for token_idx in range(start_token_idx + 1, end_token_idx + 1):
                labels[token_idx] = crf_label2id[f'I-{ent_type}']
        return labels

    def collate(self, examples):
        raw_text_list = []
        batch_input_ids = []
        batch_attention_mask = []
        batch_segment_ids = []
        batch_labels = []
        batch_offsets = []
        batch_gold_entities = []

        for item in examples:
            raw_text, offset_mapping, input_ids, token_type_ids, attention_mask = self.encoder(item)
            labels = self._build_bio_labels(raw_text, offset_mapping, item['entities'], len(input_ids))

            raw_text_list.append(raw_text)
            batch_input_ids.append(input_ids)
            batch_segment_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            batch_offsets.append(offset_mapping)
            batch_gold_entities.append(item['entities'])

        batch_inputids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_segmentids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
        batch_attentionmask = torch.tensor(self.sequence_padding(batch_attention_mask)).long()
        batch_labels = torch.tensor(self.sequence_padding(batch_labels)).long()

        return (
            raw_text_list,
            batch_inputids,
            batch_attentionmask,
            batch_segmentids,
            batch_labels,
            batch_offsets,
            batch_gold_entities,
        )

    def __getitem__(self, index):
        return self.data[index]
