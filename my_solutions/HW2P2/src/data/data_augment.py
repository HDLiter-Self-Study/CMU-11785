import torchvision.transforms.v2 as transforms
from typing import Dict, List, Any, Optional, Union, Tuple


class AugmentationPipelineFactory:
    def __init__(self, transforms_list: List[Tuple[str, Dict[str, Any]]]):
        """Initialize the augmentation pipeline with a list of transforms.

        Args:
            transforms_list (List[Tuple[str, Dict[str, Any]]]): A list of tuples where each tuple contains
                the name of the transform and its parameters.
        """
        self.transforms = []
        for name, params in transforms_list:
            # First try to get the transform method from the class
            transform_method = getattr(self, f"_get_{name}", None)
            if transform_method:
                self.transforms.append(transform_method(**params))
                continue
            # If not found, try to get it from default methods
            transform = self._get_standard_transform(name, params)
            if transform:
                self.transforms.append(transform)
                continue

            raise ValueError(f"Transform '{name}' is not recognized or not implemented.")

    def get_pipeline(self) -> transforms.Compose:
        """Get the composed augmentation pipeline."""
        return transforms.Compose(self.transforms)

    def _get_standard_transform(self, name: str, params: Dict[str, Any]) -> Optional[transforms.Transform]:
        """Get a standard transform from torchvision.transforms.

        Args:
            name (str): The name of the transform.
            params (Dict[str, Any]): Parameters for the transform.

        Returns:
            Optional[transforms.Transform]: The transform if found, otherwise None.
        """
        # Get change the name format to match torchvision.transforms.v2
        name = name.replace("_", " ").title().replace(" ", "")
        transform_cls = getattr(transforms, name, None)
        if transform_cls:
            return transform_cls(**params)
