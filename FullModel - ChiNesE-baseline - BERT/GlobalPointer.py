# -*- coding: utf-8 -*-
"""
Reference code url: https://github.com/gaohongkui/GlobalPointer_pytorch/tree/main/models
"""
import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true):
        y_pred = y_pred.data.cpu().numpy()
        y_true = y_true.data.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)

        return X, Y, Z

class GlobalPointer(nn.Module):
    def __init__(
        self,
        encoder,
        ent_type_size,
        inner_dim,
        RoPE=True,
        rnn_hidden_size=256,
        rnn_layers=1,
        efficient=False,
        use_bigru=True,
    ):
        #encodr: RoBerta-Large as encoder
        #inner_dim: 64
        #ent_type_size: ent_cls_num
        #rnn_hidden_size: RNN 隐藏层大小（单向），双向 RNN 的输出维度将是 rnn_hidden_size * 2
        #rnn_layers: RNN 层数
        #efficient: 是否启用高效版本
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.efficient = efficient
        self.use_bigru = use_bigru
        
        if self.use_bigru:
            # RNN 层：使用单独的 BiGRU 进一步建模上下文信息
            # PyTorch GRU 的 dropout 仅在 num_layers>1 时作用于层间；单层时在输出侧使用 gru_dropout
            self.bigru = nn.GRU(
                input_size=self.hidden_size,
                hidden_size=self.rnn_hidden_size,
                num_layers=self.rnn_layers,
                bidirectional=True,
                batch_first=True,
                dropout=0.2 if self.rnn_layers > 1 else 0.0,
            )
            self.gru_dropout = nn.Dropout(0.2)
            rnn_output_dim = self.rnn_hidden_size * 2
        else:
            # 消融：不使用 BiGRU，直接使用 encoder 输出
            self.bigru = None
            self.gru_dropout = None
            rnn_output_dim = self.hidden_size

        if self.efficient:
            # 高效模式共享 q/k 投影，并使用轻量分类器
            self.q_dense = nn.Linear(rnn_output_dim, self.inner_dim)
            self.k_dense = nn.Linear(rnn_output_dim, self.inner_dim)
            self.classifier = nn.Linear(4 * self.inner_dim, self.ent_type_size)
        else:
            # 兼容原始实现
            self.dense = nn.Linear(rnn_output_dim, self.ent_type_size * self.inner_dim * 2)

        self.RoPE = RoPE


    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1]*len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        self.device = input_ids.device
        
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # 通过 BiGRU 进一步建模序列表征（或消融时直接使用 encoder 输出）
        if self.use_bigru:
            # rnn_output: (batch_size, seq_len, rnn_hidden_size * 2)
            # 注意：训练时 input_ids/attention_mask 会 padding 到 batch 内最长长度；
            # 若直接喂给 BiGRU 会让 padding 污染隐藏状态，因此需要 pack/unpack。
            seq_lengths = attention_mask.sum(dim=1).to(torch.long).cpu()  # (batch_size,)
            packed_input = pack_padded_sequence(
                last_hidden_state,
                seq_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.bigru(packed_input)
            rnn_output, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=seq_len,
            )
            rnn_output = self.gru_dropout(rnn_output)
        else:
            # rnn_output: (batch_size, seq_len, hidden_size)
            rnn_output = last_hidden_state

        if self.efficient:
            # 高效实现：共享 q/k + 轻量分类器
            qw = self.q_dense(rnn_output)  # (b, m, d)
            kw = self.k_dense(rnn_output)  # (b, n, d)
            if self.RoPE:
                pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
                cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
                sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
                qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1).reshape(qw.shape)
                kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1).reshape(kw.shape)
                qw = qw * cos_pos + qw2 * sin_pos
                kw = kw * cos_pos + kw2 * sin_pos

            extraction_scores = torch.einsum('bmd,bnd->bmn', qw, kw)  # (b, m, n)
            # 拼接首尾 token 表征
            qw_i = qw.unsqueeze(2).expand(batch_size, seq_len, seq_len, self.inner_dim)
            kw_i = kw.unsqueeze(2).expand(batch_size, seq_len, seq_len, self.inner_dim)
            qw_j = qw.unsqueeze(1).expand(batch_size, seq_len, seq_len, self.inner_dim)
            kw_j = kw.unsqueeze(1).expand(batch_size, seq_len, seq_len, self.inner_dim)
            concat_features = torch.cat([qw_i, kw_i, qw_j, kw_j], dim=-1)  # (b, m, n, 4d)
            classification_scores = self.classifier(concat_features).permute(0, 3, 1, 2)  # (b, h, m, n)
            extraction_scores = extraction_scores.unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
            logits = extraction_scores + classification_scores
        else:
            # 原始实现
            outputs = self.dense(rnn_output)
            outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
            outputs = torch.stack(outputs, dim=-2)
            qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
            if self.RoPE:
                pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
                cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
                sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
                qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
                qw2 = qw2.reshape(qw.shape)
                qw = qw * cos_pos + qw2 * sin_pos
                kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
                kw2 = kw2.reshape(kw.shape)
                kw = kw * cos_pos + kw2 * sin_pos
            logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask：同时屏蔽无效 start（行）与 end（列）
        mask_start = attention_mask.unsqueeze(1).unsqueeze(3)  # (batch, 1, seq_len, 1)
        mask_end = attention_mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, seq_len)
        pad_mask = mask_start * mask_end
        pad_mask = pad_mask.expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 显式屏蔽 [CLS] 和 [SEP] 位置，避免在特殊符号上预测实体
        # 屏蔽 [CLS] 位置（索引 0）
        logits[:, :, 0, :] -= 1e12
        logits[:, :, :, 0] -= 1e12
        
        # 使用 attention_mask 动态找到每个样本的 [SEP] 位置（最后一个有效token的位置）
        # attention_mask: (batch_size, seq_len)，1表示有效token，0表示padding
        # 找到每个样本最后一个1的位置，即 [SEP] 的位置
        sep_positions = (attention_mask.sum(dim=1) - 1).long()  # (batch_size,)，每个样本最后一个有效token的索引，转换为long类型
        
        # 创建 [SEP] 位置的掩码（向量化操作）
        # sep_mask: (batch_size, seq_len)，在 [SEP] 位置为 True
        batch_indices = torch.arange(batch_size, device=self.device)
        # 确保 [SEP] 位置有效（大于0，不是 [CLS]）
        valid_sep_mask = sep_positions > 0
        sep_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
        sep_mask[batch_indices[valid_sep_mask], sep_positions[valid_sep_mask]] = True
        
        # 扩展维度以匹配 logits 的形状: (batch_size, ent_type_size, seq_len, seq_len)
        sep_mask_expanded = sep_mask.unsqueeze(1).unsqueeze(2).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        sep_mask_expanded_T = sep_mask.unsqueeze(1).unsqueeze(3).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        
        # 屏蔽 [SEP] 位置
        logits = logits - sep_mask_expanded * 1e12  # 屏蔽行（start position）
        logits = logits - sep_mask_expanded_T * 1e12  # 屏蔽列（end position）

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits/self.inner_dim**0.5

