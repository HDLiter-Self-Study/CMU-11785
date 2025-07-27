# 深度学习框架结构练习 - 挖空示例

## 📚 学习目标
通过填写挖空的代码，深入理解：
1. PyTorch模块的构造和前向传播
2. 残差连接的实现原理
3. 注意力机制(SE模块)的工作方式
4. 架构工厂模式的设计思路
5. 配置驱动的系统设计

## 🎯 挖空策略

### 1. 核心实现挖空 (保留结构，挖空逻辑)
```python
# 示例：BasicBlock挖空版本
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, se_module=None):
        super().__init__()
        
        # TODO: 实现第一个卷积层 (3x3, 带padding)
        self.conv1 = None  # 填写: nn.Conv2d(...)
        
        # TODO: 实现第一个BatchNorm
        self.bn1 = None    # 填写: nn.BatchNorm2d(...)
        
        # TODO: 实现激活函数
        self.relu = None   # 填写: nn.ReLU(...)
        
        # TODO: 实现第二个卷积层
        self.conv2 = None  # 填写: nn.Conv2d(...)
        
        # TODO: 实现第二个BatchNorm  
        self.bn2 = None    # 填写: nn.BatchNorm2d(...)
        
        # TODO: 实现SE模块集成
        self.se_module = None  # 填写: se_module if se_module else None
        
        # TODO: 实现下采样层(当stride!=1或通道数不匹配时)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = None  # 填写: nn.Sequential(...)

    def forward(self, x):
        # TODO: 保存残差连接的输入
        identity = None  # 填写: x
        
        # TODO: 实现前向传播路径
        out = None       # 填写: self.conv1(x)
        out = None       # 填写: self.bn1(out)  
        out = None       # 填写: self.relu(out)
        out = None       # 填写: self.conv2(out)
        out = None       # 填写: self.bn2(out)
        
        # TODO: 应用SE模块(如果存在)
        if self.se_module is not None:
            out = None   # 填写: self.se_module(out)
        
        # TODO: 处理下采样
        if self.downsample is not None:
            identity = None  # 填写: self.downsample(identity)
        
        # TODO: 实现残差连接
        out = None       # 填写: out + identity
        out = None       # 填写: self.relu(out)
        
        return out
```

### 2. 函数签名保留 (挖空实现体)
```python
# 示例：ArchitectureFactory挖空版本
class ArchitectureFactory:
    def __init__(self):
        # TODO: 初始化架构注册表
        self.architectures = {}
        # 提示: 需要注册resnet和convnext架构
        pass
    
    def register_architecture(self, name: str, architecture_class):
        """注册新架构"""
        # TODO: 实现架构注册逻辑
        # 提示: 将架构类存储到self.architectures字典中
        pass
    
    def create_model(self, config: Dict[str, Any]):
        """根据配置创建模型"""
        # TODO: 实现模型创建逻辑
        # 提示: 1. 从config获取architecture名称
        #      2. 检查是否已注册
        #      3. 创建并返回模型实例
        pass
```

### 3. 关键概念挖空 (保留提示)
```python
# 示例：SE模块挖空版本
class SEModule(nn.Module):
    """Squeeze-and-Excitation模块"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        # TODO: 实现全局平均池化
        # 提示: 将空间维度压缩为1x1
        self.global_avgpool = None
        
        # TODO: 实现通道降维层
        # 提示: 使用1x1卷积或全连接层，输出通道为channels//reduction
        self.fc1 = None
        
        # TODO: 实现激活函数
        # 提示: 通常使用ReLU
        self.relu = None
        
        # TODO: 实现通道升维层
        # 提示: 恢复到原始通道数
        self.fc2 = None
        
        # TODO: 实现Sigmoid激活
        # 提示: 生成0-1之间的注意力权重
        self.sigmoid = None

    def forward(self, x):
        # 保存原始输入
        batch_size, channels, _, _ = x.size()
        
        # TODO: 实现Squeeze操作
        # 提示: 全局平均池化 -> [B, C, H, W] -> [B, C, 1, 1]
        y = None
        
        # TODO: 压平为2D张量
        # 提示: [B, C, 1, 1] -> [B, C]
        y = None
        
        # TODO: 实现Excitation操作
        # 提示: fc1 -> relu -> fc2 -> sigmoid
        y = None  # fc1
        y = None  # relu  
        y = None  # fc2
        y = None  # sigmoid
        
        # TODO: 重塑为可广播的形状
        # 提示: [B, C] -> [B, C, 1, 1]
        y = None
        
        # TODO: 应用注意力权重
        # 提示: 元素级相乘
        return None
```

## 🧪 验证脚本示例

```python
# 示例：验证脚本结构
def test_basic_block():
    """测试BasicBlock实现"""
    print("🧪 测试BasicBlock...")
    
    # TODO: 创建测试输入
    x = torch.randn(2, 64, 32, 32)  # [batch, channels, height, width]
    
    # TODO: 创建BasicBlock实例
    block = BasicBlock(64, 64, stride=1)
    
    # TODO: 前向传播
    output = block(x)
    
    # TODO: 验证输出形状
    expected_shape = (2, 64, 32, 32)
    assert output.shape == expected_shape, f"期望形状{expected_shape}, 实际{output.shape}"
    
    # TODO: 验证梯度可以反向传播
    loss = output.sum()
    loss.backward()
    
    print("   ✅ BasicBlock测试通过")

def test_se_module():
    """测试SE模块实现"""
    print("🧪 测试SEModule...")
    
    # TODO: 测试用例
    # 1. 输入输出形状一致性
    # 2. 注意力权重范围[0,1]
    # 3. 参数数量正确性
    
    pass

# 运行所有测试
if __name__ == "__main__":
    test_basic_block()
    test_se_module()
    print("🎉 所有测试通过！")
```

## 📋 挖空计划

### Phase 1: 基础模块 (推荐开始)
- [ ] `SEModule` - 理解注意力机制
- [ ] `BasicBlock` - 理解残差连接
- [ ] `ConvolutionBlock` - 理解基础卷积块

### Phase 2: 架构实现
- [ ] `BottleneckBlock` - 理解瓶颈设计
- [ ] `ConvNeXtBlock` - 理解现代架构设计
- [ ] `ResNet` - 理解网络组装

### Phase 3: 系统设计
- [ ] `ArchitectureFactory` - 理解工厂模式
- [ ] `ConvNeXt` - 理解完整架构
- [ ] 配置系统集成

## ❓ 学习检查点

每完成一个模块，问自己：
1. 这个模块的核心功能是什么？
2. 输入输出的张量形状如何变化？
3. 参数是如何初始化和使用的？
4. 如何与其他模块协作？
5. 为什么要这样设计？

## 🎯 挖空原则

1. **保留结构** - 类定义、函数签名、导入语句
2. **挖空核心** - 具体实现逻辑、张量操作
3. **提供提示** - 关键概念、预期行为
4. **渐进难度** - 从简单到复杂
5. **完整验证** - 每个模块都有对应测试

## 📝 填写指导

- `None` 表示需要填写的地方
- `# TODO:` 说明需要实现的功能
- `# 提示:` 给出实现建议
- `# 填写:` 给出具体的代码模板

---

**请审阅这个挖空示例，确认学习目标和难度是否合适，然后我将开始对实际代码进行挖空处理。**
