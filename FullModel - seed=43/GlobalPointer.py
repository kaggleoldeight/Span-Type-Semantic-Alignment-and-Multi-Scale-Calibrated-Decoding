# -*- coding: utf-8 -*-
"""
Span-Type + Layer-wise Multi-Scale Fusion GlobalPointer

核心改动：
1. 保留原 GlobalPointer 主分支，不改写 RoBERTa->BiGRU->GlobalPointer 主链路；
2. 保留 Span-Type 语义对齐与 Span-Type Fusion；
3. 新增 RoBERTa 多层 hidden states 的轻量辅助 GlobalPointer 分支；
4. 实验A新增 gated residual 主分支融合：多尺度表示以 alpha*gate 的形式进入 rnn_output；
5. 验证/预测阶段仍通过 beta 融合 multiscale logits，beta=0 可回退到原强基线。
"""
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def apply_boundary_soft_rerank(span_logits, *args, **kwargs):
    """兼容旧 import。当前版本默认不使用 Boundary-Aware，直接返回原 span_logits。"""
    return span_logits


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
        use_boundary_aware=False,
        use_span_type_contrastive=True,
        span_type_proj_dim=256,
        span_type_hidden_size=256,
        span_width_emb_size=32,
        span_type_max_width=64,
        span_type_dropout=0.2,
        use_multiscale_fusion=False,
        multiscale_layers=None,
        multiscale_dropout=0.1,
        multiscale_detach_encoder=True,
        use_gated_multiscale_main_fusion=False,
        gated_multiscale_detach_encoder=True,
        gated_multiscale_init_alpha=0.0,
        gated_multiscale_max_alpha=0.10,
        gated_multiscale_dropout=0.10,
    ):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.efficient = efficient
        self.use_bigru = use_bigru
        self.use_boundary_aware = use_boundary_aware
        self.use_span_type_contrastive = use_span_type_contrastive
        self.span_type_proj_dim = span_type_proj_dim
        self.span_type_max_width = span_type_max_width
        self.use_multiscale_fusion = bool(use_multiscale_fusion)
        self.use_gated_multiscale_main_fusion = bool(use_gated_multiscale_main_fusion)
        # gated main fusion 依赖 RoBERTa 多层 hidden states；如果开启 gated，则强制开启 hidden-state 抽取。
        self.need_multiscale_hidden_states = bool(self.use_multiscale_fusion or self.use_gated_multiscale_main_fusion)
        self.multiscale_layers = list(multiscale_layers or [-1, -2, -4, -6])
        self.multiscale_detach_encoder = bool(multiscale_detach_encoder)
        self.gated_multiscale_detach_encoder = bool(gated_multiscale_detach_encoder)
        self.gated_multiscale_max_alpha = float(gated_multiscale_max_alpha)

        if self.use_bigru:
            self.bigru = nn.GRU(
                input_size=self.hidden_size,
                hidden_size=self.rnn_hidden_size,
                num_layers=self.rnn_layers,
                bidirectional=True,
                batch_first=True,
                dropout=0.2 if self.rnn_layers > 1 else 0.0,
            )
            self.gru_dropout = nn.Dropout(0.2)
            self.rnn_output_dim = self.rnn_hidden_size * 2
        else:
            self.bigru = None
            self.gru_dropout = None
            self.rnn_output_dim = self.hidden_size

        if self.efficient:
            self.q_dense = nn.Linear(self.rnn_output_dim, self.inner_dim)
            self.k_dense = nn.Linear(self.rnn_output_dim, self.inner_dim)
            self.classifier = nn.Linear(4 * self.inner_dim, self.ent_type_size)
        else:
            self.dense = nn.Linear(self.rnn_output_dim, self.ent_type_size * self.inner_dim * 2)

        # Layer-wise Multi-Scale auxiliary GlobalPointer branch.
        # 旧分支：产生辅助 span logits，不替换主分支 rnn_output。
        # 实验A新增：同一组 layer scalar-mix 可被 gated main fusion 复用。
        if self.need_multiscale_hidden_states:
            self.multiscale_layer_weights = nn.Parameter(torch.zeros(len(self.multiscale_layers)))
        else:
            self.multiscale_layer_weights = None

        if self.use_multiscale_fusion:
            self.multiscale_dropout = nn.Dropout(multiscale_dropout)
            self.multiscale_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.rnn_output_dim),
                nn.GELU(),
                nn.Dropout(multiscale_dropout),
            )
            self.multiscale_dense = nn.Linear(self.rnn_output_dim, self.ent_type_size * self.inner_dim * 2)
        else:
            self.multiscale_dropout = None
            self.multiscale_proj = None
            self.multiscale_dense = None

        # Gated Multi-Scale Main Fusion（实验A）：
        # rnn_output <- rnn_output + alpha * gate([rnn_output; ms_repr]) * ms_repr
        # alpha 初始化为 0，保证训练起点等价于原强基线。
        if self.use_gated_multiscale_main_fusion:
            self.gated_multiscale_dropout = nn.Dropout(gated_multiscale_dropout)
            self.gated_multiscale_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.rnn_output_dim),
                nn.GELU(),
                nn.Dropout(gated_multiscale_dropout),
            )
            self.gated_multiscale_gate = nn.Sequential(
                nn.Linear(self.rnn_output_dim * 2, self.rnn_output_dim),
                nn.Sigmoid(),
            )
            self.gated_multiscale_alpha = nn.Parameter(torch.tensor(float(gated_multiscale_init_alpha)))
        else:
            self.gated_multiscale_dropout = None
            self.gated_multiscale_proj = None
            self.gated_multiscale_gate = None
            self.gated_multiscale_alpha = None

        # 兼容旧 Boundary-Aware 参数；默认关闭，不参与训练或预测。
        self.boundary_dropout = None
        self.start_boundary_head = None
        self.end_boundary_head = None

        # Span-Type semantic alignment modules.
        if self.use_span_type_contrastive:
            self.width_embedding = nn.Embedding(self.span_type_max_width + 1, span_width_emb_size)
            span_input_dim = self.rnn_output_dim * 3 + span_width_emb_size
            self.span_type_projector = nn.Sequential(
                nn.Linear(span_input_dim, span_type_hidden_size),
                nn.GELU(),
                nn.Dropout(span_type_dropout),
                nn.Linear(span_type_hidden_size, span_type_proj_dim),
            )
            self.type_desc_projector = nn.Linear(self.hidden_size, span_type_proj_dim)
            self.type_embeddings = nn.Parameter(torch.empty(ent_type_size, span_type_proj_dim))
            nn.init.normal_(self.type_embeddings, mean=0.0, std=0.02)
        else:
            self.width_embedding = None
            self.span_type_projector = None
            self.type_desc_projector = None
            self.type_embeddings = None

        self.RoPE = RoPE
        self.device = None

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def _encode_sequence(self, input_ids, attention_mask, token_type_ids):
        self.device = input_ids.device
        if self.need_multiscale_hidden_states:
            context_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
            )
        else:
            context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)

        last_hidden_state = context_outputs[0]
        selected_hidden_states = None
        if self.need_multiscale_hidden_states:
            all_hidden_states = getattr(context_outputs, 'hidden_states', None)
            if all_hidden_states is None and len(context_outputs) > 2:
                all_hidden_states = context_outputs[2]
            if all_hidden_states is not None:
                selected_hidden_states = []
                valid_weights = []
                total_layers = len(all_hidden_states)
                for weight_idx, layer_idx in enumerate(self.multiscale_layers):
                    real_idx = layer_idx if layer_idx >= 0 else total_layers + layer_idx
                    if 0 <= real_idx < total_layers:
                        h = all_hidden_states[real_idx]
                        selected_hidden_states.append(h)
                        valid_weights.append(weight_idx)
                if selected_hidden_states:
                    selected_hidden_states = (selected_hidden_states, valid_weights)
                else:
                    selected_hidden_states = None

        batch_size = last_hidden_state.size(0)
        seq_len = last_hidden_state.size(1)

        if self.use_bigru:
            seq_lengths = attention_mask.sum(dim=1).to(torch.long).cpu()
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
            rnn_output = last_hidden_state
        return rnn_output, batch_size, seq_len, selected_hidden_states

    def _compute_span_logits(self, rnn_output, batch_size, seq_len):
        if self.efficient:
            qw = self.q_dense(rnn_output)
            kw = self.k_dense(rnn_output)
            if self.RoPE:
                pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
                cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
                sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
                qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1).reshape(qw.shape)
                kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1).reshape(kw.shape)
                qw = qw * cos_pos + qw2 * sin_pos
                kw = kw * cos_pos + kw2 * sin_pos

            extraction_scores = torch.einsum('bmd,bnd->bmn', qw, kw)
            qw_i = qw.unsqueeze(2).expand(batch_size, seq_len, seq_len, self.inner_dim)
            kw_i = kw.unsqueeze(2).expand(batch_size, seq_len, seq_len, self.inner_dim)
            qw_j = qw.unsqueeze(1).expand(batch_size, seq_len, seq_len, self.inner_dim)
            kw_j = kw.unsqueeze(1).expand(batch_size, seq_len, seq_len, self.inner_dim)
            concat_features = torch.cat([qw_i, kw_i, qw_j, kw_j], dim=-1)
            classification_scores = self.classifier(concat_features).permute(0, 3, 1, 2)
            extraction_scores = extraction_scores.unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
            logits = extraction_scores + classification_scores
        else:
            outputs = self.dense(rnn_output)
            outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
            outputs = torch.stack(outputs, dim=-2)
            qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
            if self.RoPE:
                pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
                cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
                sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
                qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1).reshape(qw.shape)
                kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1).reshape(kw.shape)
                qw = qw * cos_pos + qw2 * sin_pos
                kw = kw * cos_pos + kw2 * sin_pos
            logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        return logits

    def _compute_logits_from_dense(self, sequence_output, dense_layer, batch_size, seq_len):
        """用指定 dense 层计算普通 GlobalPointer logits，供辅助分支复用。"""
        outputs = dense_layer(sequence_output)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
        if self.RoPE:
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1).reshape(qw.shape)
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1).reshape(kw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw = kw * cos_pos + kw2 * sin_pos
        return torch.einsum('bmhd,bnhd->bhmn', qw, kw)

    def _mix_multiscale_hidden(self, selected_hidden_states, detach_encoder: bool = True):
        """对 RoBERTa 多层 hidden states 做可学习 scalar mix。"""
        if self.multiscale_layer_weights is None or selected_hidden_states is None:
            return None
        hidden_list, valid_weight_indices = selected_hidden_states
        if not hidden_list:
            return None
        weight_indices = torch.tensor(valid_weight_indices, dtype=torch.long, device=hidden_list[0].device)
        layer_weights = torch.softmax(self.multiscale_layer_weights.index_select(0, weight_indices), dim=0)
        mixed = 0.0
        for w, h in zip(layer_weights, hidden_list):
            if detach_encoder:
                h = h.detach()
            mixed = mixed + w * h
        return mixed

    def compute_multiscale_main_repr(self, selected_hidden_states, batch_size, seq_len):
        """计算 gated main fusion 使用的多尺度表示，形状与 rnn_output 一致。"""
        if not self.use_gated_multiscale_main_fusion or self.gated_multiscale_proj is None:
            return None
        mixed = self._mix_multiscale_hidden(
            selected_hidden_states,
            detach_encoder=self.gated_multiscale_detach_encoder,
        )
        if mixed is None:
            return None
        mixed = self.gated_multiscale_dropout(mixed)
        return self.gated_multiscale_proj(mixed)

    def apply_gated_multiscale_fusion(self, rnn_output, multiscale_main_repr):
        """安全残差门控融合。alpha 从 0 起步，并限制最大值。"""
        if multiscale_main_repr is None or self.gated_multiscale_gate is None:
            return rnn_output
        alpha = torch.clamp(self.gated_multiscale_alpha, min=0.0, max=self.gated_multiscale_max_alpha)
        gate = self.gated_multiscale_gate(torch.cat([rnn_output, multiscale_main_repr], dim=-1))
        return rnn_output + alpha * gate * multiscale_main_repr

    def compute_multiscale_logits(self, selected_hidden_states, attention_mask, batch_size, seq_len):
        """
        RoBERTa 多层 hidden states 的辅助 GlobalPointer logits。
        注意：该分支不替换主分支，只用于 auxiliary loss 和验证/预测阶段 beta fusion。
        """
        if not self.use_multiscale_fusion or self.multiscale_dense is None:
            return None
        mixed = self._mix_multiscale_hidden(
            selected_hidden_states,
            detach_encoder=self.multiscale_detach_encoder,
        )
        if mixed is None:
            return None
        mixed = self.multiscale_dropout(mixed)
        multiscale_repr = self.multiscale_proj(mixed)
        logits = self._compute_logits_from_dense(multiscale_repr, self.multiscale_dense, batch_size, seq_len)
        logits = self._apply_span_masks(logits, attention_mask, batch_size, seq_len)
        logits = logits / self.inner_dim ** 0.5
        return logits

    def get_multiscale_parameters(self):
        """返回 Multi-Scale 辅助 logits 分支参数，便于训练脚本单独设置学习率。"""
        if not self.use_multiscale_fusion:
            return []
        params = []
        if self.multiscale_layer_weights is not None:
            params.append(self.multiscale_layer_weights)
        for module in [self.multiscale_proj, self.multiscale_dense]:
            if module is not None:
                params += list(module.parameters())
        return params

    def get_gated_multiscale_parameters(self):
        """返回 gated main fusion 分支参数。实验A中该分支单独使用较高学习率。"""
        if not self.use_gated_multiscale_main_fusion:
            return []
        params = []
        # layer_weights 同时服务 auxiliary logits 和 gated main fusion，避免重复加入优化器。
        for module in [self.gated_multiscale_proj, self.gated_multiscale_gate]:
            if module is not None:
                params += list(module.parameters())
        if self.gated_multiscale_alpha is not None:
            params.append(self.gated_multiscale_alpha)
        return params

    def _apply_span_masks(self, logits, attention_mask, batch_size, seq_len):
        mask_start = attention_mask.unsqueeze(1).unsqueeze(3)
        mask_end = attention_mask.unsqueeze(1).unsqueeze(1)
        pad_mask = mask_start * mask_end
        pad_mask = pad_mask.expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 屏蔽 [CLS]
        logits[:, :, 0, :] -= 1e12
        logits[:, :, :, 0] -= 1e12

        # 屏蔽每个样本的 [SEP]
        sep_positions = (attention_mask.sum(dim=1) - 1).long()
        batch_indices = torch.arange(batch_size, device=self.device)
        valid_sep_mask = sep_positions > 0
        sep_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
        sep_mask[batch_indices[valid_sep_mask], sep_positions[valid_sep_mask]] = True

        sep_mask_row = sep_mask.unsqueeze(1).unsqueeze(2).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        sep_mask_col = sep_mask.unsqueeze(1).unsqueeze(3).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits - sep_mask_row * 1e12
        logits = logits - sep_mask_col * 1e12

        # 排除下三角，保证 start <= end
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12
        return logits

    @torch.no_grad()
    def initialize_type_embeddings(self, tokenizer, label_descriptions: List[str], device=None, max_length: int = 32):
        """
        使用同一个预训练 encoder 对实体类型描述做均值池化，并初始化 type_embeddings。
        这一步只在训练开始前调用一次；之后 type_embeddings 作为可训练参数更新。
        """
        if not self.use_span_type_contrastive or self.type_embeddings is None:
            return
        if len(label_descriptions) != self.ent_type_size:
            raise ValueError(f'label_descriptions 数量({len(label_descriptions)})必须等于 ent_type_size({self.ent_type_size})')

        device = device or next(self.parameters()).device
        was_training = self.training
        self.eval()
        encoded = tokenizer(
            label_descriptions,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        token_type_ids = encoded.get('token_type_ids')
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_ids = token_type_ids.to(device)

        outputs = self.encoder(input_ids, attention_mask, token_type_ids)[0]
        # 排除 [CLS]/[SEP]/padding 做 mean pooling。
        valid_mask = attention_mask.float()
        if valid_mask.size(1) > 0:
            valid_mask[:, 0] = 0.0
            sep_pos = (attention_mask.sum(dim=1) - 1).long()
            rows = torch.arange(attention_mask.size(0), device=device)
            valid_mask[rows, sep_pos] = 0.0
        denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        desc_repr = (outputs * valid_mask.unsqueeze(-1)).sum(dim=1) / denom
        type_init = self.type_desc_projector(desc_repr)
        type_init = F.normalize(type_init, dim=-1)
        self.type_embeddings.data.copy_(type_init)
        if was_training:
            self.train()

    def get_span_type_parameters(self):
        """返回 Span-Type 新增模块参数，便于训练脚本单独设置学习率。"""
        if not self.use_span_type_contrastive:
            return []
        params = []
        for module in [self.width_embedding, self.span_type_projector, self.type_desc_projector]:
            if module is not None:
                params += list(module.parameters())
        if self.type_embeddings is not None:
            params.append(self.type_embeddings)
        return params

    def get_span_representations(self, sequence_output: torch.Tensor, span_indices: torch.Tensor) -> torch.Tensor:
        """
        根据 span_indices 抽取 span 表示。
        span_indices: (N, 4)，列为 [batch_idx, label_idx, start_idx, end_idx]。
        返回: (N, span_type_proj_dim)
        """
        if span_indices.numel() == 0:
            return sequence_output.new_zeros((0, self.span_type_proj_dim))
        if not self.use_span_type_contrastive:
            raise RuntimeError('use_span_type_contrastive=False，不能抽取 span type 表示。')

        batch_idx = span_indices[:, 0].long()
        start_idx = span_indices[:, 2].long()
        end_idx = span_indices[:, 3].long()

        start_repr = sequence_output[batch_idx, start_idx]
        end_repr = sequence_output[batch_idx, end_idx]

        # O(N) cumsum mean pooling，避免构造完整 span 内部张量。
        zero = sequence_output.new_zeros(sequence_output.size(0), 1, sequence_output.size(-1))
        cumsum = torch.cumsum(torch.cat([zero, sequence_output], dim=1), dim=1)
        span_sum = cumsum[batch_idx, end_idx + 1] - cumsum[batch_idx, start_idx]
        span_len = (end_idx - start_idx + 1).clamp_min(1).unsqueeze(-1).float()
        mean_repr = span_sum / span_len

        width_ids = (end_idx - start_idx).clamp(min=0, max=self.span_type_max_width)
        width_repr = self.width_embedding(width_ids)

        span_input = torch.cat([start_repr, end_repr, mean_repr, width_repr], dim=-1)
        span_repr = self.span_type_projector(span_input)
        span_repr = F.normalize(span_repr, dim=-1)
        return span_repr

    def get_type_representations(self) -> torch.Tensor:
        if not self.use_span_type_contrastive:
            raise RuntimeError('use_span_type_contrastive=False，不能获取 type 表示。')
        return F.normalize(self.type_embeddings, dim=-1)


    @torch.no_grad()
    def build_valid_span_indices(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        构造有效 span 索引，列为 [batch_idx, dummy_label_idx, start_idx, end_idx]。
        dummy_label_idx 只为复用 get_span_representations 的索引格式，实际不参与计算。
        有效范围会排除 [CLS]、[SEP]、padding，并保证 start <= end。
        """
        device = attention_mask.device
        batch_size, seq_len = attention_mask.shape
        all_indices = []
        for b in range(batch_size):
            valid_len = int(attention_mask[b].sum().item())
            # token 0 是 [CLS]，valid_len - 1 是 [SEP]。
            start_min = 1
            end_max = valid_len - 2
            if end_max < start_min:
                continue
            positions = torch.arange(start_min, end_max + 1, device=device)
            starts, ends = torch.meshgrid(positions, positions, indexing='ij')
            keep = starts <= ends
            starts = starts[keep]
            ends = ends[keep]
            batch_col = torch.full_like(starts, b)
            label_col = torch.zeros_like(starts)
            all_indices.append(torch.stack([batch_col, label_col, starts, ends], dim=1))
        if not all_indices:
            return torch.zeros((0, 4), dtype=torch.long, device=device)
        return torch.cat(all_indices, dim=0).long()

    @torch.no_grad()
    def compute_span_type_similarity_logits(
        self,
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_size: int = 8192,
    ) -> torch.Tensor:
        """
        计算所有有效 span 与各实体类型的语义相似度，输出形状与 GlobalPointer logits 一致：
        (batch, ent_type, seq_len, seq_len)。

        该函数使用 chunk 逐块计算，避免一次性构造 (batch*seq*seq) 的巨大 span 表示。
        返回值仅用于验证/预测阶段的 soft fusion，不参与反向传播。
        """
        batch_size, seq_len, _ = sequence_output.shape
        sim_logits = sequence_output.new_zeros((batch_size, self.ent_type_size, seq_len, seq_len))
        if not self.use_span_type_contrastive:
            return sim_logits

        span_indices = self.build_valid_span_indices(attention_mask)
        if span_indices.numel() == 0:
            return sim_logits

        type_repr = self.get_type_representations()
        chunk_size = max(1, int(chunk_size))
        for start in range(0, span_indices.size(0), chunk_size):
            cur_indices = span_indices[start:start + chunk_size]
            span_repr = self.get_span_representations(sequence_output, cur_indices)
            sim = torch.matmul(span_repr, type_repr.t())  # (N, ent_type), cosine similarity
            b_idx = cur_indices[:, 0].long()
            s_idx = cur_indices[:, 2].long()
            e_idx = cur_indices[:, 3].long()
            sim_logits[b_idx, :, s_idx, e_idx] = sim
        return sim_logits

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        return_boundary=False,
        return_span_type=False,
        return_multiscale=False,
    ):
        rnn_output, batch_size, seq_len, selected_hidden_states = self._encode_sequence(input_ids, attention_mask, token_type_ids)
        multiscale_main_repr = self.compute_multiscale_main_repr(selected_hidden_states, batch_size, seq_len)
        rnn_output = self.apply_gated_multiscale_fusion(rnn_output, multiscale_main_repr)
        logits = self._compute_span_logits(rnn_output, batch_size, seq_len)
        logits = self._apply_span_masks(logits, attention_mask, batch_size, seq_len)
        logits = logits / self.inner_dim ** 0.5

        if return_span_type or return_multiscale:
            outputs = {
                'span_logits': logits,
                'sequence_output': rnn_output,
            }
            if return_multiscale:
                multiscale_logits = self.compute_multiscale_logits(
                    selected_hidden_states,
                    attention_mask,
                    batch_size,
                    seq_len,
                )
                if multiscale_logits is not None:
                    outputs['multiscale_logits'] = multiscale_logits
            return outputs

        if return_boundary:
            # 兼容旧训练/预测脚本的返回格式。
            return logits, None, None
        return logits
