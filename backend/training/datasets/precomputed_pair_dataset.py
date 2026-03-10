"""
Pre-computed Pair Dataset

pairs CSV 파일에서 미리 생성된 페어를 로드하여
(img1, img2, label, is_same_dept) 4-tuple을 반환합니다.

기존 PairwiseDataset과 동일한 이미지 로딩/transform을 사용하되,
페어 생성 로직은 외부 스크립트(prepare_training_pairs.py)에 위임합니다.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PrecomputedPairDataset(Dataset):
    """
    미리 계산된 페어를 로드하는 Dataset.

    Args:
        pairs_csv: 페어 CSV 경로 (image_path_1, image_path_2, label, pair_type 컬럼)
        image_dir: 이미지 루트 디렉토리
        transform: 이미지 transform (None이면 기본 transform 사용)
        is_train: True면 augmentation 포함
    """

    DEFAULT_IMAGE_SIZE: int = 448
    IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __init__(
        self,
        pairs_csv: Union[str, Path],
        image_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        is_train: bool = True,
    ) -> None:
        self._image_dir = Path(image_dir)
        self._transform = transform or self._get_default_transform(is_train)

        # CSV 로드
        self._image_paths_1: list[str] = []
        self._image_paths_2: list[str] = []
        self._labels: list[int] = []
        self._is_same_dept: list[int] = []

        with open(pairs_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._image_paths_1.append(row["image_path_1"])
                self._image_paths_2.append(row["image_path_2"])
                self._labels.append(int(row["label"]))
                self._is_same_dept.append(
                    1 if row.get("pair_type", "") == "same_dept" else 0
                )

    def _get_default_transform(self, is_train: bool) -> Callable:
        if is_train:
            return transforms.Compose([
                transforms.Resize((self.DEFAULT_IMAGE_SIZE + 32, self.DEFAULT_IMAGE_SIZE + 32)),
                transforms.RandomCrop(self.DEFAULT_IMAGE_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.DEFAULT_IMAGE_SIZE, self.DEFAULT_IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Returns:
            (img1_tensor, img2_tensor, label, is_same_dept)
            - label: +1 or -1
            - is_same_dept: 1 if same department pair, 0 otherwise
        """
        img1 = self._load_image(self._image_paths_1[idx])
        img2 = self._load_image(self._image_paths_2[idx])

        img1_tensor = self._transform(img1)
        img2_tensor = self._transform(img2)

        return img1_tensor, img2_tensor, self._labels[idx], self._is_same_dept[idx]

    def _load_image(self, image_path: str) -> Image.Image:
        full_path = self._image_dir / image_path
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")
        return Image.open(full_path).convert("RGB")

    @property
    def num_same_dept(self) -> int:
        return sum(self._is_same_dept)

    @property
    def num_cross_dept(self) -> int:
        return len(self._is_same_dept) - self.num_same_dept

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_pairs={len(self)}, "
            f"same_dept={self.num_same_dept}, "
            f"cross_dept={self.num_cross_dept})"
        )
