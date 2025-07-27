# 高级深度学习框架挖空练习 - 进阶版

## 🎯 升级目标
基于你的RNN分类器需求，创建更有挑战性的练习，涵盖：
- 自定义梯度计算和反向传播
- 多层RNN的时间展开
- 复杂的张量操作和维度管理
- 高级优化器实现
- 自注意力机制和Transformer组件

## 🔥 高难度挖空策略

### 1. 完全挖空实现 (仅保留接口)
```python
class MultiLayerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cell_type='RNN'):
        super().__init__()
        # TODO: 完全从零实现多层RNN
        # 不提供任何实现提示，需要理解:
        # - 层间连接方式
        # - 参数初始化策略  
        # - 梯度流动路径
        pass
    
    def forward(self, x, h_0=None):
        # TODO: 实现多时间步、多层的前向传播
        # 需要处理: 序列展开、隐状态传递、层间连接
        pass
    
    def backward(self, grad_output):
        # TODO: 完全自定义的BPTT实现
        # 需要理解: 时间维度梯度、层间梯度传播
        pass
```

### 2. 算法级别挖空 (核心算法实现)
```python
class CustomAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # TODO: 实现多头自注意力机制
        # 不使用nn.MultiheadAttention，从头实现
        pass
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # TODO: 实现注意力核心算法
        # 包括: 缩放、掩码、softmax、加权求和
        pass
```

### 3. 数学公式挖空 (纯数学实现)
```python
def custom_layer_norm(x, gamma, beta, eps=1e-5):
    # TODO: 不使用nn.LayerNorm，手动实现层归一化
    # 数学公式: y = γ * (x - μ) / √(σ² + ε) + β
    # 需要处理: 均值计算、方差计算、反向传播
    pass

def gelu_activation(x):
    # TODO: 手动实现GELU激活函数
    # 数学公式: GELU(x) = x * Φ(x)，其中Φ是标准正态分布的CDF
    # 需要实现: 高斯误差函数的近似
    pass
```

## 📚 进阶练习模块设计

### Phase 1: 核心算法实现 (⭐⭐⭐⭐⭐)

#### 1. 自定义RNN Cell族
```python
# rnn_cells_exercise.py
class RNNCell:      # 基础RNN单元 - 完全从零实现
class LSTMCell:     # LSTM单元 - 包含门控机制  
class GRUCell:      # GRU单元 - 简化的门控
class CustomCell:   # 自定义变体 - 创新设计
```

#### 2. 高级注意力机制
```python
# attention_mechanisms_exercise.py  
class ScaledDotProductAttention:    # 点积注意力核心
class MultiHeadAttention:           # 多头注意力完整实现
class CrossAttention:               # 交叉注意力机制
class SparseAttention:              # 稀疏注意力优化
```

#### 3. 自定义优化器
```python
# optimizers_exercise.py
class AdamOptimizer:                # Adam优化器手动实现
class AdamWOptimizer:               # AdamW变体
class CosineScheduler:              # 余弦退火调度
class WarmupScheduler:              # 预热调度策略
```

### Phase 2: 复杂架构组件 (⭐⭐⭐⭐⭐)

#### 4. Transformer Block
```python
# transformer_exercise.py
class TransformerEncoderLayer:
    def __init__(self, d_model, nhead, dim_feedforward):
        # TODO: 完全自定义实现，不使用nn.TransformerEncoderLayer
        # 包括: 多头注意力、前馈网络、残差连接、层归一化
        pass
    
    def forward(self, src, src_mask=None):
        # TODO: 实现完整的编码器层前向传播
        # 需要处理: 注意力计算、残差连接、归一化顺序
        pass
```

#### 5. 位置编码和嵌入
```python
# embeddings_exercise.py  
class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        # TODO: 实现正弦余弦位置编码
        # 数学公式需要手动实现，不使用预设函数
        pass

class LearnablePositionalEmbedding:
    # TODO: 实现可学习的位置嵌入
    pass
```

### Phase 3: 端到端系统 (⭐⭐⭐⭐⭐)

#### 6. 自定义训练循环
```python
# training_loop_exercise.py
class CustomTrainer:
    def __init__(self, model, optimizer, scheduler=None):
        # TODO: 实现完整的训练管理系统
        # 包括: 梯度累积、梯度裁剪、混合精度、检查点
        pass
    
    def train_epoch(self, dataloader):
        # TODO: 实现训练epoch逻辑
        # 需要处理: 批次处理、损失计算、梯度更新、指标跟踪
        pass
    
    def validate(self, dataloader):
        # TODO: 实现验证逻辑
        pass
```

#### 7. 高级损失函数
```python
# losses_exercise.py
class FocalLoss:
    # TODO: 实现Focal Loss，处理类别不平衡
    pass

class LabelSmoothingCrossEntropy:
    # TODO: 实现标签平滑交叉熵
    pass

class ContrastiveLoss:
    # TODO: 实现对比学习损失
    pass
```

## 🔥 超高难度挑战

### 8. 从零实现Transformer
```python
# full_transformer_exercise.py
class CustomTransformer:
    """完全自定义的Transformer实现，不使用任何PyTorch内置组件"""
    
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        # TODO: 实现完整Transformer
        # 挑战: 所有组件(注意力、FFN、归一化、位置编码)都需自己实现
        pass
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # TODO: 实现编码器-解码器架构
        # 最高难度: 处理掩码、交叉注意力、因果建模
        pass
```

### 9. 自动微分引擎
```python
# autograd_exercise.py
class Tensor:
    """自定义张量类，实现自动微分"""
    def __init__(self, data, requires_grad=False):
        # TODO: 实现张量的梯度追踪机制
        pass
    
    def backward(self, grad=None):
        # TODO: 实现自动微分的反向传播
        # 超高难度: 构建计算图、梯度传播
        pass

# 支持的操作需要全部手动实现梯度计算
def matmul(a, b):     # 矩阵乘法 + 梯度
def softmax(x):       # Softmax + 梯度  
def cross_entropy():  # 交叉熵 + 梯度
```

## 🧪 高级验证系统

### 数值稳定性测试
```python
def test_numerical_stability():
    # 测试梯度爆炸/消失
    # 测试数值精度
    # 测试边界条件
    pass

def test_gradient_correctness():
    # 有限差分验证梯度
    # 对比自动微分结果
    pass

def benchmark_performance():
    # 性能对比测试
    # 内存使用分析
    pass
```

### 理论验证测试
```python
def test_mathematical_correctness():
    # 验证注意力权重和为1
    # 验证LayerNorm的均值和方差
    # 验证梯度的数学正确性
    pass
```

## 📈 难度对比

| 模块 | 基础版难度 | 进阶版难度 | 挑战内容 |
|------|------------|------------|----------|
| 注意力机制 | ⭐⭐☆ | ⭐⭐⭐⭐⭐ | 手动实现缩放点积、多头分离 |
| RNN单元 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | LSTM门控、梯度流动、BPTT |
| 优化器 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 动量、自适应学习率、二阶优化 |
| Transformer | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 完整编码器-解码器、掩码处理 |
| 自动微分 | N/A | ⭐⭐⭐⭐⭐ | 计算图构建、动态梯度计算 |

## 🎯 学习成果

完成这些练习后，你将能够：
1. **深度理解**现代深度学习的数学原理
2. **手动实现**主流架构的核心组件
3. **调试和优化**复杂的神经网络
4. **设计创新**的网络架构
5. **构建**完整的深度学习框架

---

**这个进阶方案基于你的RNN代码复杂度设计，难度显著提升。你觉得这个挑战级别如何？需要我开始实现具体的挖空文件吗？**
