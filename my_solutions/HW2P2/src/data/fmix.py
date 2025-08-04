#!/usr/bin/env python3
"""
最终的FMix实现 - 纯PyTorch版本
基于官方实现但完全使用PyTorch，无需numpy和scipy依赖
"""

import math
import random
from typing import Tuple, Optional, Union

import torch
import torch.nn.functional as F


def fftfreqnd_torch(
    h: int, w: Optional[int] = None, z: Optional[int] = None, device: torch.device = None
) -> torch.Tensor:
    """
    PyTorch版本的频率bin计算

    Args:
        h: 第一维度大小
        w: 第二维度大小（可选）
        z: 第三维度大小（可选）
        device: 设备

    Returns:
        频率距离张量
    """
    if device is None:
        device = torch.device("cpu")

    # 计算频率
    fy = torch.fft.fftfreq(h, device=device)

    if w is not None:
        fy = fy.unsqueeze(-1)

        if w % 2 == 1:
            fx = torch.fft.fftfreq(w, device=device)[: w // 2 + 2]
        else:
            fx = torch.fft.fftfreq(w, device=device)[: w // 2 + 1]
    else:
        fx = torch.tensor(0.0, device=device)

    if z is not None:
        fy = fy.unsqueeze(-1)
        fz = torch.fft.fftfreq(z, device=device)[:, None]
    else:
        fz = torch.tensor(0.0, device=device)

    return torch.sqrt(fx**2 + fy**2 + fz**2)


def get_spectrum_torch(
    freqs: torch.Tensor, decay_power: float, ch: int, h: int, w: int = 0, z: int = 0
) -> torch.Tensor:
    """
    PyTorch版本的频谱生成

    Args:
        freqs: 频率bin张量
        decay_power: 衰减幂
        ch: 通道数
        h, w, z: 维度大小

    Returns:
        频谱张量
    """
    device = freqs.device

    # 计算衰减因子
    min_freq = 1.0 / max(w, h, z) if max(w, h, z) > 0 else 1.0
    scale = 1.0 / (torch.maximum(freqs, torch.tensor(min_freq, device=device)) ** decay_power)

    # 生成随机参数
    param_size = [ch] + list(freqs.shape) + [2]
    param = torch.randn(param_size, device=device)

    # 扩展scale维度
    scale = scale.unsqueeze(-1).unsqueeze(0)

    return scale * param


def make_low_freq_image_torch(
    decay_power: float, shape: Tuple[int, ...], ch: int = 1, device: torch.device = None
) -> torch.Tensor:
    """
    PyTorch版本的低频图像生成

    Args:
        decay_power: 衰减幂
        shape: 图像形状
        ch: 通道数
        device: 设备

    Returns:
        低频图像mask
    """
    if device is None:
        device = torch.device("cpu")

    # 计算频率
    freqs = fftfreqnd_torch(*shape, device=device)

    # 生成频谱
    spectrum = get_spectrum_torch(freqs, decay_power, ch, *shape)

    # 转换为复数
    spectrum_complex = spectrum[:, 0] + 1j * spectrum[:, 1]

    # 逆FFT
    if len(shape) == 1:
        mask = torch.fft.irfft(spectrum_complex, n=shape[0], dim=-1)
        mask = mask[:1, : shape[0]]
    elif len(shape) == 2:
        mask = torch.fft.irfft2(spectrum_complex, s=shape, dim=(-2, -1))
        mask = mask[:1, : shape[0], : shape[1]]
    elif len(shape) == 3:
        mask = torch.fft.irfftn(spectrum_complex, s=shape, dim=(-3, -2, -1))
        mask = mask[:1, : shape[0], : shape[1], : shape[2]]
    else:
        raise ValueError(f"Unsupported shape dimensions: {len(shape)}")

    # 归一化到[0,1]
    mask = mask - mask.min()
    mask = mask / mask.max()

    return mask


def sample_lam_torch(alpha: float, reformulate: bool = False, device: torch.device = None) -> float:
    """
    PyTorch版本的lambda采样（替代scipy.stats.beta）

    Args:
        alpha: beta分布参数
        reformulate: 是否使用重新表述
        device: 设备

    Returns:
        采样的lambda值
    """
    if device is None:
        device = torch.device("cpu")

    if reformulate:
        # Beta(alpha+1, alpha)
        beta_dist = torch.distributions.Beta(alpha + 1, alpha)
    else:
        # Beta(alpha, alpha)
        beta_dist = torch.distributions.Beta(alpha, alpha)

    return beta_dist.sample().item()


def binarise_mask_torch(
    mask: torch.Tensor, lam: float, in_shape: Tuple[int, ...], max_soft: float = 0.0
) -> torch.Tensor:
    """
    PyTorch版本的mask二值化

    Args:
        mask: 输入mask
        lam: 目标lambda值
        in_shape: 输入形状
        max_soft: 软化参数

    Returns:
        二值化的mask
    """
    device = mask.device

    # 展平mask
    mask_flat = mask.reshape(-1)

    # 排序获取索引（降序）
    _, idx = torch.sort(mask_flat, descending=True)

    # 计算需要设为1的像素数量
    total_pixels = mask_flat.numel()
    num = math.ceil(lam * total_pixels) if random.random() > 0.5 else math.floor(lam * total_pixels)

    # 计算软化范围
    eff_soft = max_soft
    if max_soft > lam or max_soft > (1 - lam):
        eff_soft = min(lam, 1 - lam)

    soft_pixels = int(total_pixels * eff_soft)
    num_low = max(0, num - soft_pixels)
    num_high = min(total_pixels, num + soft_pixels)

    # 创建新的mask
    new_mask = torch.zeros_like(mask_flat)

    # 设置高值区域为1
    if num_high > 0:
        new_mask[idx[:num_high]] = 1.0

    # 设置低值区域为0（已经是0）
    if num_low < total_pixels:
        new_mask[idx[num_low:]] = 0.0

    # 设置软化过渡区域
    if num_low < num_high:
        transition_indices = idx[num_low:num_high]
        if len(transition_indices) > 0:
            transition_values = torch.linspace(1.0, 0.0, len(transition_indices), device=device)
            new_mask[transition_indices] = transition_values

    # 重塑回原始形状
    new_mask = new_mask.reshape(1, *in_shape)

    return new_mask


def sample_mask_torch(
    alpha: float,
    decay_power: float,
    shape: Union[int, Tuple[int, ...]],
    max_soft: float = 0.0,
    reformulate: bool = False,
    device: torch.device = None,
) -> Tuple[float, torch.Tensor]:
    """
    PyTorch版本的mask采样

    Args:
        alpha: beta分布参数
        decay_power: 衰减幂
        shape: mask形状
        max_soft: 软化参数
        reformulate: 是否重新表述
        device: 设备

    Returns:
        (lambda值, mask张量)
    """
    if device is None:
        device = torch.device("cpu")

    if isinstance(shape, int):
        shape = (shape,)

    # 采样lambda
    lam = sample_lam_torch(alpha, reformulate, device)

    # 生成低频图像
    mask = make_low_freq_image_torch(decay_power, shape, ch=1, device=device)

    # 二值化
    mask = binarise_mask_torch(mask, lam, shape, max_soft)

    return lam, mask


class FMix:
    """
    最终的FMix实现 - 纯PyTorch版本
    基于官方实现但完全使用PyTorch，无需外部依赖

    Args:
        decay_power: 频率衰减的衰减幂 prop 1/f**d
        alpha: beta分布的alpha值，用于采样mask的均值
        size: 所需mask的形状，最多3维
        max_soft: 0到0.5之间的软化值，用于平滑mask中的硬边缘
        reformulate: 如果为True，使用重新表述的形式
        num_classes: 类别数量
        spatial_transform: 是否应用空间变换（保留以兼容旧接口）
    """

    def __init__(
        self,
        decay_power: float = 3.0,
        alpha: float = 1.0,
        size: Tuple[int, int] = (32, 32),
        max_soft: float = 0.0,
        reformulate: bool = False,
        num_classes: int = 10,
        spatial_transform: bool = True,
    ):
        self.decay_power = decay_power
        self.alpha = alpha
        self.size = size
        self.max_soft = max_soft
        self.reformulate = reformulate
        self.num_classes = num_classes
        self.spatial_transform = spatial_transform  # 为了兼容性保留

        # 用于调试的属性
        self._last_original_lambda = None
        self._last_effective_lambda = None

    def __call__(self, images: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用FMix增强

        Args:
            images: 输入图像张量 [B, C, H, W]
            targets: 目标标签张量 [B] 或 [B, num_classes]

        Returns:
            (混合后的图像, 混合后的标签)
        """
        batch_size = images.size(0)
        device = images.device

        # 使用PyTorch版本的算法生成mask
        lam, mask = sample_mask_torch(self.alpha, self.decay_power, self.size, self.max_soft, self.reformulate, device)

        # 扩展mask到批次维度
        if mask.dim() == 3:  # [1, H, W]
            mask = mask.expand(batch_size, -1, -1).unsqueeze(1)  # [B, 1, H, W]

        # 调整mask大小到图像大小
        if mask.shape[-2:] != images.shape[-2:]:
            mask = F.interpolate(mask, size=images.shape[-2:], mode="bilinear", align_corners=False)

        # 扩展到所有通道
        mask = mask.expand(-1, images.size(1), -1, -1)  # [B, C, H, W]

        # 生成随机排列
        index = torch.randperm(batch_size, device=device)

        # 混合图像
        mixed_images = images * mask + images[index] * (1 - mask)

        # 计算有效lambda（实际混合比例）
        effective_lam = mask.reshape(batch_size, -1).mean(dim=1)

        # 混合标签
        mixed_targets = self._mix_labels(targets, targets[index], effective_lam, device)

        # 保存lambda值用于调试
        self._last_original_lambda = torch.tensor([lam] * batch_size, device=device)
        self._last_effective_lambda = effective_lam

        return mixed_images, mixed_targets

    def _mix_labels(
        self, targets1: torch.Tensor, targets2: torch.Tensor, lam_batch: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        混合标签

        Args:
            targets1: 第一组标签
            targets2: 第二组标签
            lam_batch: 混合比例批次
            device: 设备

        Returns:
            混合后的标签
        """
        if targets1.dim() == 1:  # 类别索引
            targets1_onehot = F.one_hot(targets1, num_classes=self.num_classes).float()
            targets2_onehot = F.one_hot(targets2, num_classes=self.num_classes).float()
        else:  # 已经是one-hot
            targets1_onehot = targets1.float()
            targets2_onehot = targets2.float()

        # 混合标签
        lam_expanded = lam_batch.reshape(-1, 1)
        mixed_targets = lam_expanded * targets1_onehot + (1 - lam_expanded) * targets2_onehot

        return mixed_targets

    def get_last_lambdas(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        获取最后一次调用的lambda值（用于调试）

        Returns:
            (原始lambda, 有效lambda)
        """
        return self._last_original_lambda, self._last_effective_lambda

    def apply_spatial_transforms(self, mask: torch.Tensor) -> torch.Tensor:
        """
        应用空间变换（如果需要）

        Args:
            mask: 输入mask

        Returns:
            变换后的mask
        """
        if not self.spatial_transform:
            return mask

        # 简单的空间变换示例
        if random.random() > 0.5:
            # 随机水平翻转
            mask = torch.flip(mask, dims=[-1])

        if random.random() > 0.5:
            # 随机垂直翻转
            mask = torch.flip(mask, dims=[-2])

        return mask

    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"FMix(decay_power={self.decay_power}, alpha={self.alpha}, "
            f"size={self.size}, max_soft={self.max_soft}, num_classes={self.num_classes})"
        )
