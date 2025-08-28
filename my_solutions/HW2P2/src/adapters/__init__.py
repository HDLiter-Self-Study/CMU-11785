"""
Adapters Module - 任务特定逻辑适配器

Adapter Pattern（适配器模式）实现说明：

为什么叫 "Adapter"？
==================

1. 设计模式来源:
   - Adapter Pattern 是经典的结构型设计模式
   - 作用：让原本接口不兼容的类能够协作
   - 在这里：让不同任务的特定逻辑能够统一在BaseOptimizer中使用

2. 实际应用场景:
   - Classification 和 Verification 任务有不同的：
     * 数据加载方式
     * 模型结构
     * 损失函数
     * 评估指标
     * 优化策略
   - 但都需要统一的优化流程（参数搜索、训练、评估）

3. Adapter 的作用:
   - 封装任务特定逻辑
   - 提供统一接口给 BaseOptimizer
   - 实现代码复用和解耦

4. 类比理解:
   - 就像电源适配器：不同设备（任务）需要不同电压（逻辑），
     但都通过统一的插口（接口）连接到电源（优化器）

Architecture:
============
BaseAdapter (抽象基类)
├── ClassificationAdapter (分类任务适配器)
└── VerificationAdapter (验证任务适配器)

每个 Adapter 实现:
- prepare_data(): 准备任务特定数据
- create_model(): 创建任务特定模型
- define_objective(): 定义任务特定目标函数
- evaluate(): 实现任务特定评估逻辑
"""

from .base_adapter import TaskAdapter
from .classification_adapter import ClassificationAdapter
from .verification_adapter import VerificationAdapter

__all__ = ["TaskAdapter", "ClassificationAdapter", "VerificationAdapter"]  # 基类  # 分类任务适配器  # 验证任务适配器
