# Evaluate Router
import uuid
from datetime import datetime
import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from app.models.request import DepartmentType, LanguageType
from app.models.response import ErrorResponse, EvaluateResponse, Feedback, Probability, RubricScores
from app.services.feedback import feedback_service
from app.services.inference import inference_service

logger = structlog.get_logger(__name__)
router = APIRouter()

@router.post(
    '/evaluate',
    response_model=EvaluateResponse,
    responses={400: {'model': ErrorResponse}, 500: {'model': ErrorResponse}, 503: {'model': ErrorResponse}},
    summary='Single image evaluation',
)
async def evaluate_image(
    image: UploadFile = File(...),
    department: DepartmentType = Form(...),
    theme: str | None = Form(None),
    include_feedback: bool = Form(True),
    language: LanguageType = Form('ko'),
) -> EvaluateResponse:
    evaluation_id = str(uuid.uuid4())
    logger.info('Evaluate request', evaluation_id=evaluation_id, department=department)
    if image.content_type not in ['image/jpeg', 'image/png']:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Unsupported image format.')
    if not inference_service.is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='Model not loaded.')
    try:
        image_bytes = await image.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Empty image.')
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Image too large.')
        result = await inference_service.evaluate_image(image_bytes, department)
        scores = RubricScores(
            composition=result['scores']['composition'],
            technique=result['scores']['technique'],
            creativity=result['scores']['creativity'],
            completeness=result['scores']['completeness'],
        )
        probabilities = [
            Probability(university=p['university'], department=p['department'], probability=p['probability'])
            for p in result['probabilities']
        ]
        feedback = None
        if include_feedback:
            feedback_result = await feedback_service.generate_feedback(
                scores=result['scores'], tier=result['tier'], department=department, theme=theme, language=language
            )
            if feedback_result:
                feedback = Feedback(
                    strengths=feedback_result['strengths'],
                    improvements=feedback_result['improvements'],
                    overall=feedback_result['overall'],
                )
        logger.info('Evaluation completed', evaluation_id=evaluation_id, tier=result['tier'])
        return EvaluateResponse(
            evaluation_id=evaluation_id, tier=result['tier'], scores=scores,
            probabilities=probabilities, feedback=feedback, created_at=datetime.now(),
        )
    except ValueError as e:
        logger.warning('Invalid image', error=str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error('Evaluation failed', error=str(e), exc_info=e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Evaluation error.')
