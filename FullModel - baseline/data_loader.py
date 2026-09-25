# -*- coding: utf-8 -*-
"""
@Time: 2021/8/27 13:52
@Auth: Xhw
@Description: 实体识别的数据载入器
"""
import json
import torch
from torch.utils.data import Dataset
import numpy as np

max_len = 256
ent2id = {"bod": 0, "dis": 1, "sym": 2, "mic": 3, "pro": 4, "ite": 5, "dep": 6, "dru": 7, "equ": 8}
id2ent = {}
for k, v in ent2id.items(): id2ent[v] = k

def load_data(path):
    """
    加载训练数据
    
    核心修复：CMeEE-V2 数据集的 end_idx 是开区间格式（指向实体最后一个字符的后一位）
    例如："苯丙酮尿症" start_idx=22, end_idx=27（指向"症"后面的"筛"字位置）
    
    为了配合 GlobalPointer 的闭区间逻辑（end_idx 指向最后一个字符），必须减 1
    转换后：end_idx=26（指向"症"字本身）
    
    如果不进行此转换，模型会学习到错误的边界，导致预测时多包含一个字符
    """
    D = []
    for d in json.load(open(path, encoding='utf-8')):
        D.append([d['text']])
        for e in d['entities']:
            start, end, label = e['start_idx'], e['end_idx'], e['type']
            
            # 【关键修复】必须执行：将开区间转换为闭区间
            # CMeEE-V2 的 end_idx 指向实体最后一个字符的后一位（开区间）
            # GlobalPointer 需要闭区间（end_idx 指向最后一个字符）
            # 因此必须减 1
            end = end - 1
            
            # 验证：确保转换后的索引有效
            if start <= end:
                D[-1].append((start, end, ent2id[label]))
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
            # 【修复】只调用一次 encode_plus，同时获取 input_ids 和 offset_mapping
            # 这样可以确保 input_ids 和 offset_mapping 绝对对应，避免截断时的不一致
            encoder_txt = self.tokenizer.encode_plus(
                text, 
                max_length=max_len, 
                truncation=True,
                return_offsets_mapping=True  # 关键：一次性返回所有信息
            )
            
            # 从 encode_plus 的结果中直接获取所有需要的信息
            input_ids = encoder_txt["input_ids"]
            token_type_ids = encoder_txt["token_type_ids"]
            attention_mask = encoder_txt["attention_mask"]
            token2char_span_mapping = encoder_txt["offset_mapping"]  # 从结果中直接取

            return text, token2char_span_mapping, input_ids, token_type_ids, attention_mask
        else:
            #TODO 测试
            pass

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        """Numpy函数，将序列padding到同一长度
        """
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
        item = self.data[index]
        return item

