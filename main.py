from fastapi import FastAPI

from api.routes.interview import router as interview_router
from api.routes.learning import router as learning_router

app = FastAPI(title="Career Navigator LLM", version="1.0.0")

app.include_router(interview_router, tags=["interview"])
app.include_router(learning_router, tags=["learning"])


@app.get("/health")
def health():
    return {"ok": True}