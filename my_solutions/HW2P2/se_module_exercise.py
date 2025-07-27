"""
SE模块挖空示例 - 供学生填写练习
请填写所有标记为 # TODO 的部分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation 模块

    论文: "Squeeze-and-Excitation Networks" (https://arxiv.org/abs/1709.01507)

    核心思想:
    1. Squeeze: 通过全局平均池化压缩空间信息
    2. Excitation: 通过全连接层学习通道间的相互依赖关系
    3. Scale: 将学习到的权重应用到原始特征图上
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        初始化SE模块

        Args:
            channels: 输入特征图的通道数
            reduction: 通道降维比例，用于减少参数量
        """
        super().__init__()

        # TODO: 实现全局平均池化
        # 提示: 使用 nn.AdaptiveAvgPool2d 将空间维度压缩为 1x1
        # 目标: [B, C, H, W] -> [B, C, 1, 1]
        self.global_avgpool = None

        # TODO: 实现第一个全连接层(通道降维)
        # 提示: 输入通道=channels, 输出通道=channels//reduction
        # 注意: 可以使用 nn.Linear 或 nn.Conv2d(kernel_size=1)
        self.fc1 = None

        # TODO: 实现ReLU激活函数
        # 提示: 使用 nn.ReLU(inplace=True) 节省内存
        self.relu = None

        # TODO: 实现第二个全连接层(通道升维)
        # 提示: 输入通道=channels//reduction, 输出通道=channels
        self.fc2 = None

        # TODO: 实现Sigmoid激活函数
        # 提示: 使用 nn.Sigmoid() 生成 [0,1] 范围的权重
        self.sigmoid = None

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征图 [batch_size, channels, height, width]

        Returns:
            output: 经过SE模块调制的特征图，形状与输入相同
        """
        # 获取输入的形状信息
        batch_size, channels, height, width = x.size()

        # TODO: 实现 Squeeze 操作
        # 提示: 使用全局平均池化压缩空间维度
        # 目标: [B, C, H, W] -> [B, C, 1, 1]
        y = None

        # TODO: 重塑张量为2D，便于全连接层处理
        # 提示: 使用 .view() 或 .reshape()
        # 目标: [B, C, 1, 1] -> [B, C]
        y = None

        # TODO: 实现 Excitation 操作 - 第一个全连接层
        # 提示: 通过fc1降维
        y = None

        # TODO: 应用ReLU激活
        y = None

        # TODO: 实现 Excitation 操作 - 第二个全连接层
        # 提示: 通过fc2升维回原始通道数
        y = None

        # TODO: 应用Sigmoid激活，生成通道注意力权重
        # 注意: Sigmoid输出范围为[0,1]，表示每个通道的重要性
        y = None

        # TODO: 重塑权重张量为可广播的形状
        # 提示: [B, C] -> [B, C, 1, 1]，便于与原始特征图相乘
        y = None

        # TODO: 实现 Scale 操作
        # 提示: 将注意力权重应用到原始特征图
        # 这是逐元素相乘，每个通道会被对应的权重缩放
        return None


# ====================== 验证代码 ======================


def test_se_module():
    """测试SE模块的实现正确性"""
    print("🧪 开始测试 SEModule...")

    # 测试用例1: 基本功能测试
    print("   测试1: 基本功能...")
    channels = 64
    se_module = SEModule(channels, reduction=16)

    # 创建测试输入
    batch_size, height, width = 2, 32, 32
    x = torch.randn(batch_size, channels, height, width)

    # 前向传播
    output = se_module(x)

    # 验证输出形状
    expected_shape = (batch_size, channels, height, width)
    assert output.shape == expected_shape, f"输出形状错误: 期望{expected_shape}, 实际{output.shape}"
    print("      ✅ 输出形状正确")

    # 测试用例2: 参数数量验证
    print("   测试2: 参数数量...")
    total_params = sum(p.numel() for p in se_module.parameters())
    # SE模块参数 = fc1权重 + fc1偏置 + fc2权重 + fc2偏置
    # = (channels * channels//reduction) + channels//reduction + (channels//reduction * channels) + channels
    expected_params = channels * (channels // 16) + (channels // 16) + (channels // 16) * channels + channels
    expected_params = channels * (channels // 16) * 2 + (channels // 16) + channels
    print(f"      参数总量: {total_params}")
    print("      ✅ 参数数量合理")

    # 测试用例3: 梯度反向传播
    print("   测试3: 梯度反向传播...")
    loss = output.sum()
    loss.backward()

    # 检查所有参数都有梯度
    for name, param in se_module.named_parameters():
        assert param.grad is not None, f"参数 {name} 没有梯度"
    print("      ✅ 梯度反向传播正常")

    # 测试用例4: 注意力权重范围
    print("   测试4: 注意力机制...")
    with torch.no_grad():
        # 手动检查中间结果
        y = se_module.global_avgpool(x)
        y = y.view(y.size(0), -1)
        y = se_module.fc1(y)
        y = se_module.relu(y)
        y = se_module.fc2(y)
        weights = se_module.sigmoid(y)

        # 验证Sigmoid输出范围
        assert torch.all(weights >= 0) and torch.all(weights <= 1), "注意力权重不在[0,1]范围内"
        print("      ✅ 注意力权重范围正确")

    print("🎉 SEModule 所有测试通过！\n")


def test_se_integration():
    """测试SE模块与其他层的集成"""
    print("🧪 测试 SE模块集成...")

    # 创建一个简单的残差块，集成SE模块
    class SimpleResidualWithSE(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)
            self.se = SEModule(channels)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self.se(out)  # 应用SE模块
            out += identity  # 残差连接
            out = self.relu(out)
            return out

    # 测试集成效果
    channels = 64
    block = SimpleResidualWithSE(channels)
    x = torch.randn(2, channels, 32, 32)
    output = block(x)

    assert output.shape == x.shape, "集成测试失败"
    print("   ✅ SE模块集成测试通过")
    print()


if __name__ == "__main__":
    """
    运行测试前，请确保已正确实现 SEModule 中的所有 TODO 项
    """
    try:
        test_se_module()
        test_se_integration()
        print("🎊 恭喜！SE模块实现完全正确！")

        # 额外的学习检查
        print("\n📚 学习检查点:")
        print("   1. 你理解SE模块的三个核心步骤了吗？(Squeeze, Excitation, Scale)")
        print("   2. 为什么要使用全局平均池化而不是最大池化？")
        print("   3. reduction参数的作用是什么？")
        print("   4. SE模块如何改善网络的表征能力？")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("\n🔍 调试提示:")
        print("   1. 检查所有 TODO 项是否都已实现")
        print("   2. 确认张量形状变换是否正确")
        print("   3. 验证每一步的输入输出维度")
        print("   4. 可以添加 print() 语句查看中间结果")
