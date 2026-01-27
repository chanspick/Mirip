# Feedback Generation Service
from typing import Any, Optional
import structlog
from app.config import settings

logger = structlog.get_logger(__name__)

# 점수 범위별 피드백 템플릿 (한국어)
FEEDBACK_TEMPLATES_KO = {
    "composition": {
        "high": ["구도 구성이 매우 안정적입니다.", "화면 구성의 균형감이 뛰어납니다.", "시각적 흐름이 자연스럽게 잘 연결됩니다."],
        "medium": ["구도가 전반적으로 안정적이나 일부 보완이 필요합니다.", "화면 배치에서 약간의 불균형이 보입니다."],
        "low": ["구도 구성에 개선이 필요합니다.", "화면의 균형감을 재검토해 보세요.", "시각적 중심이 불명확합니다."],
    },
    "technique": {
        "high": ["표현 기법이 능숙하고 세련되었습니다.", "재료 사용이 매우 효과적입니다.", "기술적 완성도가 높습니다."],
        "medium": ["기법 표현이 양호하나 더 연습이 필요합니다.", "재료 활용에서 발전 가능성이 보입니다."],
        "low": ["기본 기법 연습이 더 필요합니다.", "재료 특성에 대한 이해를 높여보세요.", "표현 방식의 다양성을 탐구해 보세요."],
    },
    "creativity": {
        "high": ["독창적인 아이디어가 돋보입니다.", "창의적 발상이 매우 뛰어납니다.", "개성 있는 표현이 인상적입니다."],
        "medium": ["창의성이 보이나 더 발전시킬 수 있습니다.", "아이디어를 더 과감하게 표현해 보세요."],
        "low": ["더 독창적인 접근을 시도해 보세요.", "자신만의 시각을 더 탐구해 보세요.", "다양한 참고자료를 통해 영감을 얻어보세요."],
    },
    "completeness": {
        "high": ["작품의 완성도가 매우 높습니다.", "세부 표현까지 꼼꼼하게 마무리되었습니다.", "전체적인 통일감이 뛰어납니다."],
        "medium": ["전반적으로 완성되었으나 세부 마무리가 필요합니다.", "일부 영역에서 추가 작업이 도움이 될 것 같습니다."],
        "low": ["작품 완성도를 높이기 위한 추가 작업이 필요합니다.", "세부 표현에 더 시간을 투자해 보세요.", "마무리 단계에 더 집중해 보세요."],
    },
}

# 등급별 종합 피드백 템플릿
TIER_OVERALL_TEMPLATES_KO = {
    "S": "전체적으로 매우 우수한 작품입니다. 구도, 기법, 창의성, 완성도 모든 면에서 높은 수준을 보여주고 있습니다. 현재의 방향성을 유지하면서 더욱 발전시켜 나가시길 바랍니다.",
    "A": "전반적으로 좋은 작품입니다. 대부분의 영역에서 안정적인 실력을 보여주고 있으며, 몇 가지 부분만 보완하면 더욱 완성도 높은 작품이 될 것입니다.",
    "B": "발전 가능성이 보이는 작품입니다. 기본기는 갖추어져 있으나, 일부 영역에서 추가적인 연습과 보완이 필요합니다. 꾸준한 연습을 통해 실력을 향상시켜 나가세요.",
    "C": "기초를 다지는 것이 필요한 단계입니다. 각 영역별로 기본기를 탄탄히 하는 것에 집중하시고, 다양한 연습을 통해 실력을 쌓아가시길 바랍니다.",
}


class FeedbackService:
    def __init__(self):
        self._client = None
        logger.info("FeedbackService initialized")

    def _get_score_level(self, score: float) -> str:
        if score >= 80:
            return "high"
        elif score >= 60:
            return "medium"
        else:
            return "low"

    def _select_feedback(self, templates: list[str], score: float, axis: str) -> str:
        import hashlib
        seed = int(hashlib.md5(f"{axis}{score}".encode()).hexdigest()[:8], 16)
        return templates[seed % len(templates)]

    async def generate_feedback(
        self,
        scores: dict[str, float],
        tier: str,
        department: str,
        theme: Optional[str] = None,
        language: str = "ko",
    ) -> Optional[dict[str, Any]]:
        strengths = []
        improvements = []
        for axis, score in scores.items():
            level = self._get_score_level(score)
            templates = FEEDBACK_TEMPLATES_KO.get(axis, {})
            if level == "high":
                feedback_text = self._select_feedback(templates.get("high", ["좋습니다."]), score, axis)
                strengths.append(feedback_text)
            elif level == "low":
                feedback_text = self._select_feedback(templates.get("low", ["개선이 필요합니다."]), score, axis)
                improvements.append(feedback_text)
            else:
                medium_templates = templates.get("medium", ["양호합니다."])
                feedback_text = self._select_feedback(medium_templates, score, axis)
                if score >= 70:
                    strengths.append(feedback_text)
                else:
                    improvements.append(feedback_text)
        if not strengths:
            strengths.append("전반적인 노력이 보입니다.")
        if not improvements:
            improvements.append("현재의 방향성을 유지하면서 발전시켜 나가세요.")
        overall = TIER_OVERALL_TEMPLATES_KO.get(tier, "피드백을 생성할 수 없습니다.")
        return {"strengths": strengths, "improvements": improvements, "overall": overall}

    async def generate_comparison_summary(
        self,
        results: list[dict[str, Any]],
        language: str = "ko",
    ) -> Optional[str]:
        if not results:
            return "비교할 작품이 없습니다."
        num_items = len(results)
        tiers = [r.get("tier", "C") for r in results]
        tier_order = {"S": 0, "A": 1, "B": 2, "C": 3}
        best_idx = min(range(num_items), key=lambda i: tier_order.get(tiers[i], 4))
        best_tier = tiers[best_idx]
        avg_scores = []
        for r in results:
            scores = r.get("scores", {})
            if scores:
                avg = sum(scores.values()) / len(scores)
                avg_scores.append(avg)
            else:
                avg_scores.append(0)
        summary_parts = [
            f"총 {num_items}개의 작품을 비교했습니다.",
            f"{best_idx + 1}번 작품이 {best_tier}등급으로 가장 우수합니다.",
        ]
        if num_items > 1:
            score_diff = max(avg_scores) - min(avg_scores)
            if score_diff > 20:
                summary_parts.append("작품 간 수준 차이가 큰 편입니다.")
            elif score_diff < 5:
                summary_parts.append("작품들이 비슷한 수준을 보이고 있습니다.")
        return " ".join(summary_parts)


feedback_service = FeedbackService()