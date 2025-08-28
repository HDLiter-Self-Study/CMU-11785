"""
Optimizers Module

统一的优化器架构，支持不同任务类型：
- BaseOptimizer: 通用优化器基类
- ClassificationOptimizer: 分类任务优化器
- VerificationOptimizer: 验证任务优化器

Design Pattern: 策略模式 + 适配器模式
- 使用适配器模式将不同任务的特定逻辑封装到adapter中
- 使用策略模式在运行时选择不同的优化策略
"""

from .base_optimizer import BaseOptimizer
from .classification_optimizer import ClassificationOptimizer
from .verification_optimizer import VerificationOptimizer

__all__ = ["BaseOptimizer", "ClassificationOptimizer", "VerificationOptimizer"]
