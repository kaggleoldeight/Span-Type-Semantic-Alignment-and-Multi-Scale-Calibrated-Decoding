# -*- coding: utf-8 -*-
"""
BERT-CRF 模型。

本文件不依赖 torchcrf，内置一个简洁 CRF 实现，避免服务器环境缺少额外依赖导致无法运行。
"""
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class CRF(nn.Module):
    """线性链条件随机场，batch_first=True。"""

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """返回 log-likelihood。"""
        if mask.dtype != torch.bool:
            mask = mask.bool()

        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        raise ValueError(f'Unsupported reduction: {reduction}')

    def _compute_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = emissions.shape

        # 约定第一个有效 token 为 [CLS]，attention_mask 为 True。
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for i in range(1, seq_length):
            transition_score = self.transitions[tags[:, i - 1], tags[:, i]]
            emission_score = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)
            score += (transition_score + emission_score) * mask[:, i]

        seq_ends = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, seq_ends.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        return score

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_length):
            # (batch, from_tag, to_tag)
            next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, i].unsqueeze(1)
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)

        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """Viterbi 解码。返回每个样本的最佳标签序列，长度等于有效 token 数。"""
        if mask.dtype != torch.bool:
            mask = mask.bool()

        batch_size, seq_length, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history = []

        for i in range(1, seq_length):
            next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_score, best_path = next_score.max(dim=1)
            best_score = best_score + emissions[:, i]
            history.append(best_path)
            score = torch.where(mask[:, i].unsqueeze(1), best_score, score)

        score += self.end_transitions
        best_last_score, best_last_tag = score.max(dim=1)

        best_paths: List[List[int]] = []
        seq_ends = mask.long().sum(dim=1) - 1
        for batch_idx in range(batch_size):
            seq_len = int(seq_ends[batch_idx].item()) + 1
            best_tag = int(best_last_tag[batch_idx].item())
            best_path = [best_tag]

            # history[t] 对应位置 t+1 的回溯指针。
            for hist in reversed(history[:seq_len - 1]):
                best_tag = int(hist[batch_idx][best_tag].item())
                best_path.append(best_tag)

            best_path.reverse()
            best_paths.append(best_path)
        return best_paths


class BertCRF(nn.Module):
    def __init__(self, pretrained_model_dir: str, num_labels: int, dropout_prob: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_dir)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = self.dropout(outputs[0])
        emissions = self.classifier(sequence_output)

        mask = attention_mask.bool()
        loss = None
        if labels is not None:
            log_likelihood = self.crf(emissions, labels, mask=mask, reduction='mean')
            loss = -log_likelihood

        decoded = self.crf.decode(emissions, mask=mask)
        return loss, emissions, decoded
