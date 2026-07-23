"""
FastAPI routes — /api/analysis/*
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import APIRouter
from webapp.services.analysis_service import (get_eda_stats, get_regression_results,
                                               get_classification_results, get_best_models)

router = APIRouter(prefix="/api/analysis", tags=["Analysis"])


@router.get("/eda")
async def eda():
    """EDA stats: class distribution, PCA, modality stats, feature norms."""
    return get_eda_stats()


@router.get("/regression")
async def regression():
    """Train & evaluate all regression models; return metrics + best model."""
    return get_regression_results()


@router.get("/classification")
async def classification():
    """Train & evaluate all classifiers; return metrics + best model."""
    return get_classification_results()


@router.get("/best-model")
async def best_model():
    """Return the single best regression and classification model."""
    return get_best_models()
