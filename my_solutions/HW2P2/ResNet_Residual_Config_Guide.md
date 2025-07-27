# ResNet 残差连接配置详解

## 概述
ResNet架构现在支持丰富的残差连接配置选项，允许用户精细控制shortcut连接的行为。这些配置在 `config/search_spaces/architectures/resnet/main.yaml` 中的 `residual_params` 部分定义。

## 配置参数详解

### 1. projection_type - 投影类型
控制当输入输出维度不匹配时，shortcut连接的投影方式。

#### 可选值：
- **"auto"** (推荐默认值)
  - 自动选择投影方式
  - 当通道数不匹配时使用conv投影
  - 当仅步长改变时使用avg_pool投影
  - 平衡计算效率和表达能力

- **"conv"** (标准方式)
  - 使用1×1卷积进行投影
  - 最强的表达能力，但参数量最多
  - 适合对精度要求高的场景

- **"avg_pool"** (轻量化)
  - 使用平均池化+可选的1×1卷积
  - 参数量少，计算高效
  - 适合资源受限的场景

- **"max_pool"** (实验性)
  - 使用最大池化+可选的1×1卷积
  - 保留更强的特征信号
  - 适合特定的实验场景

#### 实现原理：
```python
# Auto模式的选择逻辑
if projection_type == "auto":
    if in_channels != out_channels:
        projection_type = "conv"  # 通道不匹配用conv
    else:
        projection_type = "avg_pool"  # 仅步长不匹配用池化
```

### 2. projection_norm - 投影后归一化
控制是否在投影层后应用归一化。

#### 可选值：
- **true** (推荐默认值)
  - 在投影后应用BatchNorm等归一化
  - 有助于训练稳定性
  - 与主路径的归一化保持一致

- **false**
  - 不应用归一化
  - 减少计算量和参数量
  - 适合某些特定实验设置

### 3. residual_scale - 残差缩放
控制残差连接的缩放因子。

#### 数值范围：0.1 - 2.0
- **1.0** (标准ResNet)
  - 不进行缩放，保持原始ResNet行为
  - 平衡的梯度流

- **< 1.0** (如0.8, 0.9)
  - 减弱残差信号
  - 让主路径学习更多特征
  - 适合过拟合严重的场景

- **> 1.0** (如1.2, 1.5)
  - 增强残差信号
  - 促进梯度流动
  - 适合深层网络或训练困难的场景

#### 数学公式：
```
output = main_path + shortcut * residual_scale
```

### 4. residual_dropout - 残差Dropout
在残差路径应用dropout正则化。

#### 数值范围：0.0 - 0.3
- **0.0** (默认值)
  - 不应用dropout
  - 保持标准ResNet行为

- **0.05 - 0.1** (轻度正则化)
  - 轻微的正则化效果
  - 适合轻微过拟合的场景

- **0.1 - 0.3** (强正则化)
  - 更强的正则化效果
  - 适合严重过拟合或大型模型

## 配置示例

### 标准ResNet配置
```yaml
residual_params:
  projection_type: "auto"
  projection_norm: true
  residual_scale: 1.0
  residual_dropout: 0.0
```

### 轻量化配置
```yaml
residual_params:
  projection_type: "avg_pool"
  projection_norm: false
  residual_scale: 0.8
  residual_dropout: 0.0
```

### 强正则化配置
```yaml
residual_params:
  projection_type: "conv"
  projection_norm: true
  residual_scale: 0.9
  residual_dropout: 0.1
```

### 深层网络配置
```yaml
residual_params:
  projection_type: "conv"
  projection_norm: true
  residual_scale: 1.2
  residual_dropout: 0.05
```

## 与其他参数的协同

### 与SE模块结合
```yaml
se_params:
  use_se: true
  se_reduction: 16
  se_activation: "swish"

residual_params:
  projection_type: "conv"
  projection_norm: true
  residual_scale: 0.9  # 因为SE增强了特征，适当减小残差
  residual_dropout: 0.1
```

### 与不同Block类型结合
- **BasicBlock**: 适合所有投影类型
- **BottleneckBlock**: 建议使用"conv"投影以匹配4x扩展

## 性能影响分析

### 参数量影响
- **Conv投影**: 参数量最多，每个投影层增加 `in_channels × out_channels` 参数
- **Pool投影**: 参数量最少，仅在通道不匹配时增加少量参数
- **归一化**: 每层增加 `2 × out_channels` 参数

### 计算量影响
- **Conv投影**: FLOPs最多
- **Pool投影**: FLOPs最少
- **残差dropout**: 训练时增加随机掩码计算

### 精度影响
- **Conv投影**: 通常精度最高
- **Pool投影**: 精度略低但效率高
- **适度残差缩放**: 通常有助于性能
- **残差dropout**: 有助于泛化但可能影响训练精度

## 实验建议

1. **首次实验**: 使用默认的"auto"配置
2. **资源受限**: 尝试"avg_pool" + projection_norm=false
3. **精度优先**: 使用"conv" + projection_norm=true
4. **过拟合**: 增加residual_dropout到0.1-0.15
5. **训练困难**: 尝试residual_scale=1.1-1.2

## 代码实现要点

新的残差连接配置在以下文件中实现：
- `src/models/architectures/resnet/blocks/basic_block.py`
- `src/models/architectures/resnet/blocks/bottleneck_block.py`

关键方法：
- `_build_shortcut()`: 构建可配置的shortcut连接
- `forward()`: 应用残差缩放和dropout

所有配置都向后兼容，未指定的参数将使用合理的默认值。
