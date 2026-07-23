"""
FastAPI Application — Multimodal Depression Detection System
Entry point: uvicorn webapp.main:app --reload --port 8000
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# NumPy 2.x compatibility shim - Restores np.in1d for librosa
import webapp.numpy_compat  # noqa: F401

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from webapp.config import API_TITLE, API_VERSION, CORS_ORIGINS
from webapp.routes.analysis   import router as analysis_router
from webapp.routes.prediction import router as predict_router

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "Production-grade REST API for multimodal depression detection "
        "using Attention Fusion + Cross-Modal Transformer on DAIC-WOZ / AVEC-2017."
    ),
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Routers ────────────────────────────────────────────────────────────
app.include_router(analysis_router)
app.include_router(predict_router)

# ── Static frontend ────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": API_VERSION, "service": API_TITLE}
