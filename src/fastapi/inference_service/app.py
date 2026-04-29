from __future__ import annotations

import contextlib
import datetime
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPARK_PIPELINES_DIR = PROJECT_ROOT / "src" / "spark-pipelines"
if SPARK_PIPELINES_DIR.exists() and str(SPARK_PIPELINES_DIR) not in sys.path:
    sys.path.insert(0, str(SPARK_PIPELINES_DIR))

from inference_artifacts import parse_positive_int_env, parse_synthesis_mode_env, resolve_inference_run_context
from inference_step_01_live_station import run_live_station_step
from inference_step_02_weather import run_weather_step
from inference_step_03_history import run_history_step
from inference_step_04_features import run_feature_rows_step
from inference_step_05_inference import run_output_step

app = FastAPI()
PIPELINE_LOCK = threading.Lock()


class InferenceRequestModel(BaseModel):
    station_id: str = Field(..., min_length=1)
    name: str | None = None
    lat: float | None = None
    lon: float | None = None
    request_timestamp: datetime.datetime | None = None

    @model_validator(mode="after")
    def validate_coordinates(self) -> "InferenceRequestModel":
        if (self.lat is None) != (self.lon is None):
            raise ValueError("lat and lon must be provided together")
        return self


def _format_request_timestamp(value: datetime.datetime) -> str:
    if value.tzinfo is None:
        return value.isoformat()
    return value.isoformat().replace("+00:00", "Z")


def _build_request_payload(request: InferenceRequestModel) -> dict[str, Any]:
    payload: dict[str, Any] = {"station_id": request.station_id}
    if request.name is not None:
        payload["name"] = request.name
    if request.lat is not None:
        payload["lat"] = request.lat
    if request.lon is not None:
        payload["lon"] = request.lon
    if request.request_timestamp is not None:
        payload["request_timestamp"] = _format_request_timestamp(request.request_timestamp)
    return payload


def _resolve_history_year_offset_days() -> int:
    raw_value = os.environ.get("INFERENCE_HISTORY_YEAR_OFFSET_DAYS")
    if raw_value is None or not raw_value.strip():
        return 365
    try:
        offset = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"INFERENCE_HISTORY_YEAR_OFFSET_DAYS must be an integer, got: {raw_value!r}"
        ) from exc
    if offset < 0:
        raise ValueError(f"INFERENCE_HISTORY_YEAR_OFFSET_DAYS must be >= 0, got: {offset}")
    return offset


@contextlib.contextmanager
def _temporary_env(updates: dict[str, str]):
    originals: dict[str, str | None] = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, previous in originals.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


@app.post("/inference/run")
def run_inference(request: InferenceRequestModel) -> dict[str, Any]:
    request_payload = _build_request_payload(request)
    request_json = json.dumps(request_payload, ensure_ascii=True)

    try:
        with PIPELINE_LOCK:
            with _temporary_env({"INFERENCE_REQUEST_JSON": request_json}):
                run_id, run_ts = resolve_inference_run_context()
                horizon_steps = parse_positive_int_env("INFERENCE_HORIZON_STEPS", default=6)
                history_lookback_hours = parse_positive_int_env(
                    "INFERENCE_HISTORY_LOOKBACK_HOURS",
                    default=168,
                )
                history_warmup_hours = parse_positive_int_env(
                    "INFERENCE_HISTORY_WARMUP_HOURS",
                    default=336,
                )
                history_synthesis_mode = parse_synthesis_mode_env(
                    env_key="INFERENCE_SYNTHESIS_MODE",
                    default="auto",
                )
                history_year_offset_days = _resolve_history_year_offset_days()
                historical_return_steps = parse_positive_int_env(
                    "HISTORICAL_RETURN_STEPS",
                    default=6,
                )

                run_live_station_step(
                    run_id=run_id,
                    run_ts=run_ts,
                )
                run_weather_step(
                    run_id=run_id,
                    run_ts=run_ts,
                    horizon_steps=horizon_steps,
                )
                run_history_step(
                    run_id=run_id,
                    run_ts=run_ts,
                    lookback_hours=history_lookback_hours,
                    warmup_hours=history_warmup_hours,
                    synthesis_mode=history_synthesis_mode,
                    history_year_offset_days=history_year_offset_days,
                )
                run_feature_rows_step(
                    run_id=run_id,
                    run_ts=run_ts,
                )
                output_payload, _ = run_output_step(
                    run_id=run_id,
                    run_ts=run_ts,
                    historical_return_steps=historical_return_steps,
                )
                return output_payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
