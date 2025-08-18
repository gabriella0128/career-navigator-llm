from fastapi import APIRouter

from schemas.common import CommonResponse
from schemas.interview import InterviewQuestionFeignReq, InterviewQuestionFeignRes, InterviewAnswerFeignRes, \
    InterviewAnswerFeignReq
from services.answer_generator import generate_daily_question_answer
from services.question_generator import generate_daily_questions

router = APIRouter()


@router.post(
    "/generate-daily-questions",
    response_model=CommonResponse[InterviewQuestionFeignRes]
)
def generate_daily_questions_route(req: InterviewQuestionFeignReq):
    return generate_daily_questions(req)


@router.post(
    "/generate-daily-question-answer",
    response_model=CommonResponse[InterviewAnswerFeignRes]
)
def generate_daily_answer_route(req: InterviewAnswerFeignReq):
    return generate_daily_question_answer(req)
