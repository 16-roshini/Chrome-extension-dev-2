"""
FastAPI application entry point.
Run with:  uvicorn server.main:app --reload --port 8000
"""

from fastapi import FastAPI
from server.ocr_pii.router import router

app = FastAPI(
    title="OCR + PII Detection API",
    description="Dev 2 module — SIH 2026 Problem Statement 26171 (ISRO)",
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
async def root():
    return {"message": "OCR + PII Detection API is running. Visit /docs for Swagger UI."}
