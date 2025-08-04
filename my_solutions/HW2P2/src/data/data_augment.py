import torch
import torchvision.transforms.v2 as transforms
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageFilter
import random


# === 独立的自定义Transform类 ===


class GridMask:
    """GridMask数据增强实现"""

    def __init__(self, prob=0.6, d_ratio_range=(0.1, 0.4), ratio=0.6):
        self.prob = prob
        self.d_ratio_range = d_ratio_range
        self.ratio = ratio

    def __call__(self, img):
        if random.random() > self.prob:
            return img

        if isinstance(img, torch.Tensor):
            img_array = img.permute(1, 2, 0).numpy()
            img_array = (img_array * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_array)
        else:
            img_pil = img

        w, h = img_pil.size
        d_ratio = random.uniform(*self.d_ratio_range)
        d = int(min(w, h) * d_ratio)

        # Create mask
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(0, h, d):
            for j in range(0, w, d):
                mask[i : i + int(d * self.ratio), j : j + int(d * self.ratio)] = 0

        # Apply mask
        img_array = np.array(img_pil)
        for c in range(img_array.shape[2]):
            img_array[:, :, c] = img_array[:, :, c] * mask

        result = Image.fromarray(img_array)

        if isinstance(img, torch.Tensor):
            result = transforms.ToTensor()(result)

        return result


class SaltPepperNoise:
    """椒盐噪声数据增强实现"""

    def __init__(self, ratio=0.01, prob=0.5):
        self.ratio = ratio
        self.prob = prob

    def __call__(self, img):
        if random.random() > self.prob:
            return img

        if isinstance(img, torch.Tensor):
            img_copy = img.clone()
            mask = torch.rand_like(img_copy[0]) < self.ratio
            salt_pepper = torch.rand_like(img_copy[0]) > 0.5
            img_copy[:, mask & salt_pepper] = 1.0  # Salt
            img_copy[:, mask & ~salt_pepper] = 0.0  # Pepper
            return img_copy
        else:
            img_array = np.array(img)
            mask = np.random.random(img_array.shape[:2]) < self.ratio
            salt_pepper = np.random.random(img_array.shape[:2]) > 0.5
            img_array[mask & salt_pepper] = 255  # Salt
            img_array[mask & ~salt_pepper] = 0  # Pepper
            return Image.fromarray(img_array)


class AugmentationPipelineFactory:
    """优化的数据增强管道工厂，通过驼峰命名自动映射torchvision类"""

    def __init__(self, transforms_list: List[Tuple[str, Dict[str, Any]]]):
        """初始化增强管道

        Args:
            transforms_list: [(transform_name, params), ...] 形式的变换列表
        """
        self.transforms = []
        for name, params in transforms_list:
            transform = self._create_transform(name, params)
            if transform:
                self.transforms.append(transform)
            else:
                raise ValueError(f"Transform '{name}' not implemented")

    def get_pipeline(self) -> transforms.Compose:
        """获取组合的增强管道"""
        return transforms.Compose(self.transforms)

    def _snake_to_camel(self, snake_str: str) -> str:
        """将蛇形命名转换为驼峰命名

        Examples:
            random_erasing -> RandomErasing
            gaussian_blur -> GaussianBlur
            trivial_augment_wide -> TrivialAugmentWide
        """
        words = snake_str.split("_")
        return "".join(word.capitalize() for word in words)

    def _create_transform(self, name: str, params: Dict[str, Any]):
        """统一的变换创建方法

        策略：
        1. 首先检查是否为特殊情况（需要自定义实现）
        2. 然后尝试通过驼峰命名直接获取torchvision类
        3. 最后回退到手动方法映射
        """

        # 特殊情况：需要自定义实现的transforms
        if name.lower() == "gridmask":
            return self._get_gridmask(**params)
        elif name.lower() == "salt_pepper_noise":
            return self._get_salt_pepper_noise(**params)
        elif name.lower() == "jpeg_compression":
            # 使用torchvision的JPEG压缩，通过RandomApply添加概率控制
            jpeg_transform = transforms.JPEG(quality=params.get("quality", 75))
            prob = params.get("prob", 0.5)
            return transforms.RandomApply([jpeg_transform], p=prob)

        # 核心策略：驼峰命名直接映射
        camel_name = self._snake_to_camel(name)
        transform_cls = getattr(transforms, camel_name, None)

        if transform_cls:
            try:
                return transform_cls(**params)
            except Exception as e:
                print(f"Warning: Failed to create {camel_name} with params {params}: {e}")
                return None

        # 回退：检查原始名称（适用于已经是驼峰的情况）
        transform_cls = getattr(transforms, name, None)
        if transform_cls:
            try:
                return transform_cls(**params)
            except Exception as e:
                print(f"Warning: Failed to create {name} with params {params}: {e}")
                return None

        # 最后回退：手动映射（保留向后兼容性）
        if hasattr(self, f"_get_{name.lower()}"):
            return getattr(self, f"_get_{name.lower()}")(**params)

        return None

    # === 特殊情况的自定义实现（无法通过驼峰命名自动映射） ===

    def _get_gridmask(self, prob=0.6, d_ratio_range=(0.1, 0.4), ratio=0.6):
        """创建GridMask实例"""
        return GridMask(prob, d_ratio_range, ratio)

    def _get_salt_pepper_noise(self, ratio=0.01, prob=0.5):
        """创建SaltPepperNoise实例"""
        return SaltPepperNoise(ratio, prob)


# 注意：如果您的输入已经是 List[Tuple[str, Dict[str, Any]]] 格式，
# 可以直接使用 AugmentationPipelineFactory，无需额外的转换函数
