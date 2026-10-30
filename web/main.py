"""
FastAPI app — bird species from attribute vectors (random forest).

Run: uvicorn main:app --reload --port 3031
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

_DEMO_ROOT = Path(__file__).resolve().parent.parent
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

from bird_species_core import (  # noqa: E402
    get_or_train_model,
    sample_demo_birds,
)

_model = None
_training = False
_train_error: str | None = None
_lock = threading.Lock()


def _train_in_background() -> None:
    global _model, _training, _train_error
    with _lock:
        if _model is not None or _training:
            return
        _training = True
        _train_error = None
    try:
        trained = get_or_train_model()
        with _lock:
            _model = trained
    except Exception as exc:  # noqa: BLE001
        with _lock:
            _train_error = str(exc)
    finally:
        with _lock:
            _training = False


def _ensure_training_started() -> None:
    threading.Thread(target=_train_in_background, daemon=True).start()


def _status_payload() -> dict[str, Any]:
    with _lock:
        if _model is None:
            return {
                "ready": False,
                "training": _training,
                "error": _train_error,
            }
        return {
            "ready": True,
            "training": False,
            "error": None,
            "holdout_accuracy": _model.holdout_accuracy,
            "cv_mean": _model.cv_mean,
            "cv_std": _model.cv_std,
            "train_size": _model.train_size,
            "test_size": _model.test_size,
            "species_count": len(_model.class_names),
            "samples": sample_demo_birds(_model),
        }


app = FastAPI(title="Bird Species Identifier")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


@app.on_event("startup")
async def startup() -> None:
    _ensure_training_started()


class PredictRequest(BaseModel):
    image_id: int = Field(..., ge=1)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.get("/help", response_class=HTMLResponse)
async def help_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "help.html")


@app.get("/api/status")
async def api_status() -> JSONResponse:
    return JSONResponse(_status_payload())


@app.post("/api/predict")
async def api_predict(body: PredictRequest) -> JSONResponse:
    with _lock:
        if _model is None:
            raise HTTPException(status_code=503, detail="Model not ready")
        model = _model
    try:
        result = model.predict_image(body.image_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(result)
