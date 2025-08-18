from fastapi import APIRouter

from schemas.common import CommonResponse
from schemas.learning import LearningFeignRes, LearningFeignReq
from services.learning_planner import build_learning_plan


router = APIRouter()


@router.post(
    "/generate-learning-plan",
    response_model=CommonResponse[LearningFeignRes]
)
def generate_monthly_learning_plan(req: LearningFeignReq):
    print(req.improvements)
    return build_learning_plan(req)
