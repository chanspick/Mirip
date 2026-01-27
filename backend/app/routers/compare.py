# Compare Router
import uuid
from datetime import datetime
import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from app.models.request import DepartmentType
from app.models.response import CompareResponse, CompareItem, ErrorResponse, RubricScores
from app.services.inference import inference_service
from app.services.feedback import feedback_service

logger = structlog.get_logger(__name__)
router = APIRouter()

@router.post(
    '/compare',
    response_model=CompareResponse,
    responses={400: {'model': ErrorResponse}, 500: {'model': ErrorResponse}, 503: {'model': ErrorResponse}},
    summary='Multiple image comparison',
)
async def compare_images(
    images: list[UploadFile] = File(..., description='Images to compare (2-10)'),
    department: DepartmentType = Form(...),
) -> CompareResponse:
    comparison_id = str(uuid.uuid4())
    logger.info('Compare request', comparison_id=comparison_id, image_count=len(images))
    if len(images) < 2 or len(images) > 10:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='2-10 images required.')
    for idx, img in enumerate(images):
        if img.content_type not in ['image/jpeg', 'image/png']:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f'Image {idx+1}: unsupported format.')
    if not inference_service.is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='Model not loaded.')
    try:
        results = []
        for idx, img in enumerate(images):
            image_bytes = await img.read()
            if len(image_bytes) == 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f'Image {idx+1} is empty.')
            if len(image_bytes) > 10 * 1024 * 1024:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f'Image {idx+1} exceeds 10MB.')
            result = await inference_service.evaluate_image(image_bytes, department)
            results.append({'index': idx, 'result': result})
        for r in results:
            r['avg_score'] = sum(r['result']['scores'].values()) / 4
        results.sort(key=lambda x: x['avg_score'], reverse=True)
        items = []
        for rank, r in enumerate(results, 1):
            scores = RubricScores(
                composition=r['result']['scores']['composition'],
                technique=r['result']['scores']['technique'],
                creativity=r['result']['scores']['creativity'],
                completeness=r['result']['scores']['completeness'],
            )
            items.append(CompareItem(image_index=r['index'], tier=r['result']['tier'], scores=scores, rank=rank))
        summary_result = await feedback_service.generate_comparison_summary(
            [{'tier': r['result']['tier'], 'scores': r['result']['scores']} for r in results]
        )
        summary = summary_result if summary_result else f'Compared {len(images)} images.'
        logger.info('Comparison completed', comparison_id=comparison_id, item_count=len(items))
        return CompareResponse(comparison_id=comparison_id, items=items, summary=summary, created_at=datetime.now())
    except ValueError as e:
        logger.warning('Invalid image', error=str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error('Comparison failed', error=str(e), exc_info=e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Comparison error.')
