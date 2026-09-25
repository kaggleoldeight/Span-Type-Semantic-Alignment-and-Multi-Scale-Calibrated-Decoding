"""
对抗训练模块
实现FGM (Fast Gradient Method) 对抗训练
"""
import torch
import torch.nn as nn
from typing import Dict, Any


class FGM:
    """Fast Gradient Method 对抗训练类"""
    
    def __init__(self, model: nn.Module):
        """
        初始化FGM对抗训练器
        
        Args:
            model: 要应用对抗训练的模型
        """
        self.model = model
        self.backup = {}  # 用于备份原始梯度
        
    def attack(self, epsilon: float = 1.0, emb_name: str = 'encoder.embeddings.word_embeddings'):
        """
        执行对抗攻击
        
        Args:
            epsilon: 扰动大小
            emb_name: 词向量层名称
        """
        # 遍历模型的所有参数
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # 备份原始参数权重
                self.backup[name] = param.data.clone()
                
                # 检查梯度是否存在
                if param.grad is None:
                    print(f"⚠️ 警告: {name} 的梯度为None，跳过对抗攻击")
                    continue
                
                # 计算梯度范数
                grad_norm = torch.norm(param.grad)
                
                # 避免除零错误
                if grad_norm != 0:
                    # 计算扰动: r = epsilon * grad / ||grad||
                    r_at = epsilon * param.grad / grad_norm
                    
                    # 将扰动加到词向量参数上
                    param.data.add_(r_at)
                else:
                    # 如果梯度为零，不添加扰动
                    print(f"⚠️ 警告: {name} 的梯度为零，跳过对抗攻击")
    
    def restore(self, emb_name: str = 'encoder.embeddings.word_embeddings'):
        """
        恢复原始词向量
        
        Args:
            emb_name: 词向量层名称
        """
        # 原位恢复备份参数，保持 param.data 的内存引用不变，避免优化器状态错位
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and name in self.backup:
                param.data.copy_(self.backup[name])
        
        # 清空备份
        self.backup.clear()


class PGD:
    """Projected Gradient Descent 对抗训练类（已修复逐词范数计算）"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.emb_backup = {}
        
    def attack(self, epsilon: float = 0.3, alpha: float = 0.1, 
               emb_name: str = 'encoder.embeddings.word_embeddings', 
               is_first_attack: bool = False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                
                if param.grad is None:
                    continue
                
                # 【核心修复1】按行（dim=-1）计算每个词向量的梯度范数
                # param.grad 形状: (vocab_size, hidden_size) -> grad_norm: (vocab_size, 1)
                grad_norm = torch.norm(param.grad, p=2, dim=-1, keepdim=True)
                grad_norm = torch.clamp(grad_norm, min=1e-8) # 防止除零
                
                # 计算步长并叠加扰动
                r_at = alpha * param.grad / grad_norm
                param.data.add_(r_at)
                
                # 【核心修复2】调用修复后的投影方法
                projected_data = self._project(param.data, self.emb_backup[name], epsilon)
                param.data.copy_(projected_data) # 必须用 copy_ 原位替换，防止优化器断连
    
    def restore(self, emb_name: str = 'encoder.embeddings.word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and name in self.emb_backup:
                param.data.copy_(self.emb_backup[name])
        self.emb_backup.clear()
    
    def _project(self, param, original_param, epsilon):
        delta = param - original_param
        # 【核心修复3】按行计算当前扰动的范数
        delta_norm = torch.norm(delta, p=2, dim=-1, keepdim=True)
        # 【核心修复4】计算缩放系数：超过 epsilon 的行才进行缩放
        scale = torch.clamp(epsilon / (delta_norm + 1e-8), max=1.0)
        return original_param + delta * scale


class AdversarialTrainer:
    """对抗训练管理器"""
    
    def __init__(self, model: nn.Module, method: str = 'FGM', emb_name: str = 'encoder.embeddings.word_embeddings', **kwargs):
        """
        初始化对抗训练管理器
        
        Args:
            model: 要应用对抗训练的模型
            method: 对抗训练方法 ('FGM' 或 'PGD')
            emb_name: embedding层名称
            **kwargs: 其他参数
        """
        self.model = model
        self.method = method.upper()
        self.emb_name = emb_name
        
        if self.method == 'FGM':
            self.adversarial = FGM(model)
        elif self.method == 'PGD':
            self.adversarial = PGD(model)
        else:
            raise ValueError(f"不支持的对抗训练方法: {method}")
        
        self.kwargs = kwargs
    
    def attack(self, **kwargs):
        """执行对抗攻击"""
        attack_kwargs = {**self.kwargs, **kwargs}
        
        # 确保使用正确的emb_name
        attack_kwargs['emb_name'] = self.emb_name
        
        # 过滤掉PGD不接受的参数
        if self.method == 'PGD':
            # PGD只接受这些参数
            pgd_params = ['epsilon', 'alpha', 'emb_name', 'is_first_attack']
            attack_kwargs = {k: v for k, v in attack_kwargs.items() if k in pgd_params}
        elif self.method == 'FGM':
            # FGM只接受这些参数
            fgm_params = ['epsilon', 'emb_name']
            attack_kwargs = {k: v for k, v in attack_kwargs.items() if k in fgm_params}
        
        self.adversarial.attack(**attack_kwargs)
    
    def restore(self, **kwargs):
        """恢复原始参数"""
        restore_kwargs = {**self.kwargs, **kwargs}
        
        # 确保使用正确的emb_name
        restore_kwargs['emb_name'] = self.emb_name
        
        # 过滤掉不接受的参数
        if self.method == 'PGD':
            # PGD只接受这些参数
            pgd_params = ['emb_name']
            restore_kwargs = {k: v for k, v in restore_kwargs.items() if k in pgd_params}
        elif self.method == 'FGM':
            # FGM只接受这些参数
            fgm_params = ['emb_name']
            restore_kwargs = {k: v for k, v in restore_kwargs.items() if k in fgm_params}
        
        self.adversarial.restore(**restore_kwargs)
