# 深度学习框架实现练习指南

## 🎯 练习目标

通过填写挖空的代码，深入理解现代深度学习框架的设计原理和实现细节。

## 📚 学习路径

### Phase 1: 基础组件 (推荐开始顺序)

#### 1. SEModule (注意力机制)
**文件**: `src/models/common_blocks/attention/se_module.py`

**需要填写的TODO项目**:
- `reduced_channels = None # TODO` → 计算降维后的通道数
- `self.global_pool = None # TODO` → 全局平均池化层
- `self.fc = None # TODO` → 全连接层序列（降维→激活→升维→Sigmoid）
- `y = None # TODO` (Squeeze操作) → 全局平均池化
- `y = None # TODO` (Excitation操作) → 全连接层处理
- `return None # TODO` (Scale操作) → 应用注意力权重
- `return None # TODO` (from_config) → 根据配置创建SE模块

**核心概念**: Squeeze-Excitation, 通道注意力, 权重重标定

#### 2. ConvolutionBlock (基础卷积块)
**文件**: `src/models/common_blocks/convolution_block.py`

**需要填写的TODO项目**:
- `self.layers = None # TODO` → 卷积-归一化-激活的Sequential
- `return None # TODO` → 前向传播

**核心概念**: 标准卷积流程, 批归一化, 激活函数

### Phase 2: 残差网络组件

#### 3. BasicBlock (基础残差块)
**文件**: `src/models/architectures/resnet/blocks/basic_block.py`

**需要填写的TODO项目**:
- `self.conv1 = None # TODO` → 第一个3x3卷积
- `self.bn1 = None # TODO` → 第一个批归一化
- `self.conv2 = None # TODO` → 第二个3x3卷积  
- `self.bn2 = None # TODO` → 第二个批归一化
- `out = None # TODO` (5个) → 前向传播的每一步

**核心概念**: 残差连接, 跳跃连接, 梯度流动

#### 4. BottleneckBlock (瓶颈残差块)
**文件**: `src/models/architectures/resnet/blocks/bottleneck_block.py`

**需要填写的TODO项目**:
- `self.conv1 = None # TODO` → 1x1卷积（降维）
- `self.bn1 = None # TODO` → 批归一化
- `self.conv2 = None # TODO` → 3x3卷积（主体）
- `self.bn2 = None # TODO` → 批归一化
- `self.conv3 = None # TODO` → 1x1卷积（升维）
- `self.bn3 = None # TODO` → 批归一化
- `out = None # TODO` (8个) → 前向传播的每一步

**核心概念**: 瓶颈设计, 参数效率, 计算优化

### Phase 3: 现代架构组件

#### 5. ConvNeXtBlock (现代卷积块)
**文件**: `src/models/architectures/convnext/blocks/convnext_block.py`

**需要填写的TODO项目**:
- `self.dwconv = None # TODO` → 深度可分离卷积
- `self.norm = None # TODO` → LayerNorm
- `self.pwconv1 = None # TODO` → 第一个逐点卷积
- `self.act = None # TODO` → GELU激活
- `self.pwconv2 = None # TODO` → 第二个逐点卷积
- `self.gamma = None # TODO` → 层缩放参数
- `x = None # TODO` (5个) → 前向传播的每一步

**核心概念**: 深度可分离卷积, LayerNorm, GELU, 层缩放

### Phase 4: 完整网络架构

#### 6. ResNet (残差网络)
**文件**: `src/models/architectures/resnet/resnet.py`

**需要填写的TODO项目**:
- `self.stem = None # TODO` → 网络起始层
- `block = None # TODO` → 残差块创建
- `self.global_pool = None # TODO` → 全局平均池化
- `self.flatten = None # TODO` → 特征展平
- `self.classifier = None # TODO` → 分类器
- `x = None # TODO` (4个) → 前向传播的每一步

**核心概念**: 分阶段设计, 特征图尺寸变化, 分类头

#### 7. ConvNeXt (现代卷积网络)
**文件**: `src/models/architectures/convnext/convnext.py`

**需要填写的TODO项目**:
- `self.stem = None # TODO` → 网络起始层
- `stage_blocks.append(None) # TODO` → ConvNeXt块创建
- `self.global_pool = None # TODO` → 全局平均池化
- `self.flatten = None # TODO` → 特征展平
- `self.classifier = None # TODO` → 分类器
- `x = None # TODO` (4个) → 前向传播的每一步

**核心概念**: 现代架构设计, DropPath, 分阶段处理

### Phase 5: 系统设计

#### 8. ArchitectureFactory (工厂模式)
**文件**: `src/models/architecture_factory.py`

**需要填写的TODO项目**:
- `self.builders = None # TODO` → 架构构建器字典
- `arch_type = None # TODO` → 从配置获取架构类型
- `return None # TODO` (3个) → 返回相应的模型实例

**核心概念**: 工厂模式, 动态创建, 配置驱动

## 🧪 验证方法

运行验证脚本检查实现正确性：
```bash
python verify_implementation.py
```

**验证内容**:
- 模块基本功能测试
- 输入输出形状验证
- 端到端前向传播测试
- 参数数量合理性检查

## 💡 实现提示

### 通用原则
1. **张量形状追踪**: 确保每一步的输入输出形状正确
2. **参数初始化**: 注意bias=False的使用
3. **激活函数位置**: 理解何时应用激活函数
4. **残差连接**: 确保跳跃连接的实现正确

### 具体提示

**SEModule**:
```python
# Squeeze: 全局平均池化
y = self.global_pool(x).view(b, c)

# Excitation: 全连接处理
y = self.fc(y).view(b, c, 1, 1)

# Scale: 应用权重
return x * y.expand_as(x)
```

**残差连接**:
```python
# 基本模式
out = self.conv1(x)
out = self.bn1(out)  
out = self.activation(out)
# ... 更多层
out += self.shortcut(x)  # 残差连接
out = self.activation(out)
```

**ConvNeXt Block**:
```python
# 深度可分离卷积 + LayerNorm + FFN
x = self.dwconv(x)
x = x.permute(0, 2, 3, 1)  # 为LayerNorm转换维度
x = self.norm(x)
# ... FFN处理
x = x.permute(0, 3, 1, 2)  # 转换回来
```

## 🔍 常见错误

1. **形状不匹配**: 检查卷积的输入输出通道数
2. **激活函数遗漏**: 确保在正确位置应用激活
3. **残差连接错误**: 注意shortcut的实现
4. **参数初始化**: 检查bias参数的设置
5. **维度转换**: ConvNeXt中注意permute操作

## 📖 学习检查点

完成每个模块后，问自己：
1. 这个模块解决了什么问题？
2. 输入输出的张量形状如何变化？
3. 关键参数的作用是什么？
4. 如何与其他模块协作？
5. 为什么要这样设计？

## 🎓 进阶练习

完成基础实现后，可以尝试：
1. 添加新的激活函数支持
2. 实现不同的归一化方法
3. 添加新的架构变体
4. 优化计算效率
5. 添加可视化功能

---

**开始你的深度学习框架实现之旅吧！** 🚀

记住：理解比记忆更重要，每一行代码都有其存在的意义。通过实现这些核心组件，你将深入理解现代深度学习框架的设计哲学。
