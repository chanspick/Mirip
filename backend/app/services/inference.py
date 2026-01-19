# ML Inference Service
from __future__ import annotations
import io
from typing import Any, Optional
import numpy as np
import structlog
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from app.config import settings

logger = structlog.get_logger(__name__)

class InferenceService:
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    INPUT_SIZE = 768
    TIER_THRESHOLDS = {"S": 85.0, "A": 70.0, "B": 55.0, "C": 0.0}
    UNIVERSITY_MAPPING = {
        "visual_design": [
            {"university": "홍익대학교", "department": "시각디자인과"},
            {"university": "국민대학교", "department": "시각디자인학과"},
            {"university": "건국대학교", "department": "커뮤니케이션디자인학과"},
            {"university": "이화여자대학교", "department": "디자인학부"},
        ],
        "industrial_design": [
            {"university": "홍익대학교", "department": "산업디자인과"},
            {"university": "국민대학교", "department": "공업디자인학과"},
            {"university": "서울대학교", "department": "디자인학부"},
            {"university": "KAIST", "department": "산업디자인학과"},
        ],
        "fine_art": [
            {"university": "서울대학교", "department": "서양화과"},
            {"university": "홍익대학교", "department": "회화과"},
            {"university": "이화여자대학교", "department": "조형예술학부"},
            {"university": "중앙대학교", "department": "서양화과"},
        ],
        "craft": [
            {"university": "홍익대학교", "department": "도예유리과"},
            {"university": "이화여자대학교", "department": "조형예술학부"},
            {"university": "국민대학교", "department": "공예학과"},
            {"university": "서울대학교", "department": "공예과"},
        ],
    }

    def __init__(self) -> None:
        self._model_loaded = False
        self.feature_extractor = None
        self.device = "cpu"
        self.transform = transforms.Compose([
            transforms.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])
        logger.info("InferenceService initialized")

    async def load_models(self) -> None:
        if self._model_loaded:
            return
        try:
            if torch.cuda.is_available() and settings.DEVICE == "cuda":
                self.device = "cuda"
            else:
                self.device = "cpu"
            logger.info("Loading DINOv2 feature extractor", device=self.device)
            from app.ml.feature_extractor import DINOv2FeatureExtractor
            self.feature_extractor = DINOv2FeatureExtractor(model_name="facebook/dinov2-large", normalize=True)
            self.feature_extractor = self.feature_extractor.to(self.device)
            self.feature_extractor.eval()
            self._model_loaded = True
            logger.info("Models loaded successfully", device=self.device)
        except Exception as e:
            logger.error("Failed to load models", error=str(e))
            raise RuntimeError(f"모델 로드 실패: {e}") from e

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")
            tensor = self.transform(image)
            tensor = tensor.unsqueeze(0).to(self.device)
            return tensor
        except Exception as e:
            logger.error("Image preprocessing failed", error=str(e))
            raise ValueError(f"이미지 전처리 실패: {e}") from e

    async def extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if not self._model_loaded or self.feature_extractor is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        with torch.no_grad():
            features = self.feature_extractor(image_tensor)
        return features

    async def predict_scores(self, features: torch.Tensor) -> dict[str, float]:
        feat_np = features.cpu().numpy().flatten()
        segment_size = len(feat_np) // 4
        def compute_score(seg: np.ndarray, offset: int) -> float:
            raw = np.abs(np.mean(seg)) * 100 + np.std(seg) * 50 + (np.linalg.norm(seg) % 10) * 3 + (np.sum(seg > 0) / len(seg)) * 20 + offset
            return round(50 + (raw % 45), 1)
        return {
            "composition": compute_score(feat_np[:segment_size], 0),
            "technique": compute_score(feat_np[segment_size:segment_size*2], 5),
            "creativity": compute_score(feat_np[segment_size*2:segment_size*3], 10),
            "completeness": compute_score(feat_np[segment_size*3:], 3),
        }

    def classify_tier(self, avg_score: float) -> str:
        for tier, threshold in self.TIER_THRESHOLDS.items():
            if avg_score >= threshold:
                return tier
        return "C"

    async def calculate_probabilities(self, scores: dict[str, float], department: str) -> list[dict[str, Any]]:
        avg_score = sum(scores.values()) / len(scores)
        universities = self.UNIVERSITY_MAPPING.get(department, [])
        probabilities = []
        for idx, univ in enumerate(universities):
            difficulty_weight = 1.0 - (idx * 0.1)
            if avg_score >= 80:
                base_prob = 0.7 + (avg_score - 80) * 0.0125
            elif avg_score >= 70:
                base_prob = 0.5 + (avg_score - 70) * 0.02
            elif avg_score >= 60:
                base_prob = 0.3 + (avg_score - 60) * 0.02
            else:
                base_prob = 0.1 + avg_score * 0.003
            prob = max(0.05, min(0.95, base_prob * difficulty_weight))
            probabilities.append({"university": univ["university"], "department": univ["department"], "probability": round(prob, 3)})
        probabilities.sort(key=lambda x: x["probability"], reverse=True)
        return probabilities

    async def evaluate_image(self, image_bytes: bytes, department: str) -> dict[str, Any]:
        image_tensor = self.preprocess_image(image_bytes)
        features = await self.extract_features(image_tensor)
        scores = await self.predict_scores(features)
        avg_score = sum(scores.values()) / len(scores)
        tier = self.classify_tier(avg_score)
        probabilities = await self.calculate_probabilities(scores, department)
        return {"scores": scores, "tier": tier, "probabilities": probabilities}

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

inference_service = InferenceService()