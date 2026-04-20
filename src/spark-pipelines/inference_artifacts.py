"""
Shared helpers for stage-10 inference step scripts.

This module centralizes:
- request parsing
- run_id/run_ts resolution
- per-step artifact IO under models/inference/run_*/<step>
- live station, weather, and synthetic history logic
"""

from __future__ import annotations

import datetime
import json
import math
import os
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo

import requests

from sparkutils import (
    apply_local_spark_defaults,
    get_spark,
    resolve_data_path,
    resolve_inference_artifacts_root,
)

# Ensure src.* imports resolve when running scripts directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stations.canonical_station_resolver import CanonicalStationResolver


GBFS_STATUS_URL = "https://gbfs.velobixi.com/gbfs/2-2/en/station_status.json"
GBFS_INFO_URL = "https://gbfs.velobixi.com/gbfs/2-2/fr/station_information.json"
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
STN_0001_SAMPLE_REQUEST_JSON = (
    '{"name":"du Mont-Royal / Clark","lat":45.51941,"lon":-73.58685,"station_id":"218"}'
)
MONTREAL_TZ = ZoneInfo("America/Montreal")
MODEL_INPUT_CATEGORICAL_COLUMNS = ["temp_bin"]
MODEL_INPUT_BOOLEAN_COLUMNS = ["is_weekday"]
MODEL_INPUT_EXCLUDED_COLUMNS = {
    "station_id",
    "ts_hour",
    "station_inflow",
    "station_outflow",
    "station_netflow",
}


@dataclass
class InferenceRequest:
    station_id: str | None = None
    name: str | None = None
    lat: float | None = None
    lon: float | None = None
    request_timestamp: datetime.datetime | None = None


@dataclass
class LiveStationSnapshot:
    station_id: str
    name: str
    lat: float
    lon: float
    capacity: int | None
    num_bikes_available: int | None
    num_docks_available: int | None


@dataclass
class WeatherHorizonResult:
    rows: list[dict[str, Any]]
    fallback_applied: bool
    missing_timestamps: list[str]


@dataclass
class SyntheticHistoryResult:
    rows: list[dict[str, Any]]
    mode_requested: str
    mode_used: str
    lookback_hours: int
    synthesized_count: int
    missing_before_fill: int
    window_start: str
    window_end: str


def to_serializable_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Unsupported object for dataclass serialization: {type(obj)}")


def normalize_station_name(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().lower().split())


def to_utc_hour(ts: datetime.datetime) -> datetime.datetime:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    return ts.replace(minute=0, second=0, microsecond=0)


def parse_request_timestamp(raw_value: str | None) -> datetime.datetime:
    if not raw_value:
        return to_utc_hour(datetime.datetime.now(timezone.utc))

    parsed = datetime.datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    return to_utc_hour(parsed)


def parse_bool_env(env_key: str, default: bool = False) -> bool:
    raw_value = os.environ.get(env_key)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_float_env(env_key: str) -> float | None:
    raw_value = os.environ.get(env_key)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{env_key} must be a float, got: {raw_value!r}") from exc


def parse_positive_int_env(env_key: str, default: int) -> int:
    raw_value = os.environ.get(env_key)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{env_key} must be an integer, got: {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{env_key} must be > 0, got: {value}")
    return value


def parse_synthesis_mode_env(
    env_key: str = "INFERENCE_SYNTHESIS_MODE",
    default: str = "auto",
) -> str:
    raw_value = os.environ.get(env_key, default)
    normalized = raw_value.strip().lower()
    allowed = {"auto", "iterative", "fallback"}
    if normalized not in allowed:
        raise ValueError(
            f"{env_key} must be one of {sorted(allowed)}, got: {raw_value!r}"
        )
    return normalized


def parse_single_station_id(raw_value: str | None) -> str | None:
    if raw_value is None or not raw_value.strip():
        return None
    station_ids = [token.strip() for token in raw_value.split(",") if token.strip()]
    if not station_ids:
        return None
    if len(station_ids) > 1:
        raise ValueError(
            "Provide exactly one station id in INFERENCE_STATION_ID (or PIPELINE_STATION_ID), "
            f"got: {station_ids}"
        )
    return station_ids[0]


def build_request_from_env() -> InferenceRequest:
    request_json = os.environ.get("INFERENCE_REQUEST_JSON")
    if request_json and request_json.strip():
        try:
            payload = json.loads(request_json)
        except json.JSONDecodeError as exc:
            raise ValueError("INFERENCE_REQUEST_JSON must be valid JSON") from exc
        return InferenceRequest(
            station_id=payload.get("station_id"),
            name=payload.get("name"),
            lat=payload.get("lat"),
            lon=payload.get("lon"),
            request_timestamp=parse_request_timestamp(payload.get("request_timestamp")),
        )

    station_id = parse_single_station_id(
        os.environ.get("INFERENCE_STATION_ID") or os.environ.get("PIPELINE_STATION_ID")
    )
    return InferenceRequest(
        station_id=station_id,
        name=os.environ.get("INFERENCE_NAME"),
        lat=parse_float_env("INFERENCE_LAT"),
        lon=parse_float_env("INFERENCE_LON"),
        request_timestamp=parse_request_timestamp(os.environ.get("INFERENCE_REQUEST_TIMESTAMP")),
    )


# ---------------------------------------------------------------------------
# Run context + artifact path helpers
# ---------------------------------------------------------------------------


def build_default_run_id() -> str:
    env_run_id = os.environ.get("INFERENCE_RUN_ID")
    if env_run_id and env_run_id.strip():
        return env_run_id.strip()

    pipeline_run_id = os.environ.get("PIPELINE_RUN_ID")
    if pipeline_run_id and pipeline_run_id.strip():
        return pipeline_run_id.strip()

    job_run_id = os.environ.get("PIPELINE_JOB_RUN_ID")
    if job_run_id and job_run_id.strip():
        repair_count = os.environ.get("PIPELINE_REPAIR_COUNT", "0").strip() or "0"
        return f"job_{job_run_id.strip()}_repair_{repair_count}"

    now_utc = datetime.datetime.now(timezone.utc)
    return f"run_{now_utc.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"


def build_default_run_ts() -> str:
    env_run_ts = os.environ.get("INFERENCE_RUN_TS")
    if env_run_ts and env_run_ts.strip():
        return env_run_ts.strip()

    pipeline_run_ts = os.environ.get("PIPELINE_RUN_TS")
    if pipeline_run_ts and pipeline_run_ts.strip():
        return pipeline_run_ts.strip()

    return datetime.datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_inference_run_context(
    run_id: str | None = None,
    run_ts: str | None = None,
) -> tuple[str, str]:
    resolved_run_id = run_id.strip() if run_id and run_id.strip() else build_default_run_id()
    resolved_run_ts = run_ts.strip() if run_ts and run_ts.strip() else build_default_run_ts()
    return resolved_run_id, resolved_run_ts


def normalize_run_folder_name(run_id: str) -> str:
    cleaned = run_id.strip()
    if cleaned.startswith("run_"):
        return cleaned
    return f"run_{cleaned}"


def get_inference_run_dir(base_path: str, run_id: str) -> Path:
    inference_root = resolve_inference_artifacts_root(base_data_path=base_path)
    return Path(inference_root) / normalize_run_folder_name(run_id)


def get_step_dir(base_path: str, run_id: str, step_name: str) -> Path:
    return get_inference_run_dir(base_path, run_id) / step_name


def write_step_artifact(
    base_path: str,
    run_id: str,
    step_name: str,
    payload: dict[str, Any],
    filename: str,
) -> str:
    step_dir = get_step_dir(base_path, run_id, step_name)
    step_dir.mkdir(parents=True, exist_ok=True)
    output_path = step_dir / filename
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(output_path)


def read_step_artifact(
    base_path: str,
    run_id: str,
    step_name: str,
    filename: str,
) -> dict[str, Any]:
    artifact_path = get_step_dir(base_path, run_id, step_name) / filename
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Missing artifact for step={step_name}, run_id={run_id}: {artifact_path}. "
            "Run previous step(s) with the same run_id first."
        )
    return json.loads(artifact_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Step computation helpers
# ---------------------------------------------------------------------------


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def fetch_station_status(timeout_sec: int = 20) -> dict[str, dict[str, Any]]:
    response = requests.get(GBFS_STATUS_URL, timeout=timeout_sec)
    response.raise_for_status()
    payload = response.json()
    stations = payload.get("data", {}).get("stations", [])
    return {
        str(row.get("station_id")): row
        for row in stations
        if row.get("station_id") is not None
    }


def fetch_station_information(timeout_sec: int = 20) -> dict[str, dict[str, Any]]:
    response = requests.get(GBFS_INFO_URL, timeout=timeout_sec)
    response.raise_for_status()
    payload = response.json()
    stations = payload.get("data", {}).get("stations", [])
    return {
        str(row.get("station_id")): row
        for row in stations
        if row.get("station_id") is not None
    }


def resolve_live_station_id(
    request: InferenceRequest,
    info_by_station_id: dict[str, dict[str, Any]],
) -> str:
    if request.station_id:
        station_id = str(request.station_id)
        if station_id in info_by_station_id:
            return station_id

    candidates = list(info_by_station_id.values())
    if not candidates:
        raise ValueError("station_information feed returned no stations")

    name_matches: list[dict[str, Any]] = []
    if request.name:
        target_name = normalize_station_name(request.name)
        name_matches = [
            row
            for row in candidates
            if normalize_station_name(row.get("name")) == target_name
        ]
        if len(name_matches) == 1:
            return str(name_matches[0]["station_id"])

    geo_candidates = name_matches if name_matches else candidates
    if request.lat is not None and request.lon is not None:
        nearest = min(
            geo_candidates,
            key=lambda row: haversine_km(
                float(request.lat),
                float(request.lon),
                float(row.get("lat", 0.0)),
                float(row.get("lon", 0.0)),
            ),
        )
        return str(nearest["station_id"])

    if name_matches:
        return str(name_matches[0]["station_id"])

    raise ValueError(
        "Unable to resolve station_id from mixed input. Provide station_id, or name with coordinates."
    )


def build_live_station_snapshot(request: InferenceRequest) -> LiveStationSnapshot:
    status_by_station_id = fetch_station_status()
    info_by_station_id = fetch_station_information()

    station_id = resolve_live_station_id(request, info_by_station_id)

    if station_id not in status_by_station_id:
        raise ValueError(f"station_id={station_id} not found in station_status feed")

    status_row = status_by_station_id[station_id]
    info_row = info_by_station_id.get(station_id, {})

    name = str(info_row.get("name") or request.name or "")
    lat_value = info_row.get("lat") if info_row.get("lat") is not None else request.lat
    lon_value = info_row.get("lon") if info_row.get("lon") is not None else request.lon

    if lat_value is None or lon_value is None:
        raise ValueError(f"Missing coordinates for station_id={station_id}")

    return LiveStationSnapshot(
        station_id=station_id,
        name=name,
        lat=float(lat_value),
        lon=float(lon_value),
        capacity=(
            int(info_row["capacity"])
            if info_row.get("capacity") is not None
            else None
        ),
        num_bikes_available=(
            int(status_row["num_bikes_available"])
            if status_row.get("num_bikes_available") is not None
            else None
        ),
        num_docks_available=(
            int(status_row["num_docks_available"])
            if status_row.get("num_docks_available") is not None
            else None
        ),
    )


def load_canonical_mapping_pandas(base_path: str):
    import pandas as pd

    parquet_dir = (
        Path(base_path)
        / "silver"
        / "station_cleaning"
        / "station_direct_match_mapping"
    )
    csv_dir = (
        Path(base_path)
        / "silver"
        / "station_cleaning"
        / "station_direct_match_mapping_csv"
    )

    if parquet_dir.exists():
        return pd.read_parquet(parquet_dir)

    csv_parts = sorted(csv_dir.glob("part-*.csv"))
    if csv_parts:
        return pd.read_csv(csv_parts[0])

    raise FileNotFoundError(
        "Missing station direct mapping artifact. Expected parquet or CSV under silver/station_cleaning."
    )


def resolve_canonical_station_id(
    base_path: str,
    snapshot: LiveStationSnapshot,
    request: InferenceRequest,
) -> dict[str, Any]:
    canonical_mapping_df = load_canonical_mapping_pandas(base_path)
    resolver = CanonicalStationResolver(canonical_mapping=canonical_mapping_df)

    resolved = resolver.resolve(
        raw_name=request.name or snapshot.name,
        lat=snapshot.lat,
        lon=snapshot.lon,
    )

    if resolved.get("canonical_station_id") is None:
        raise ValueError(
            "CanonicalStationResolver returned canonical_station_id=None. "
            f"station_id={snapshot.station_id}, name={snapshot.name}, lat={snapshot.lat}, lon={snapshot.lon}"
        )

    return resolved


def load_champion_row_for_station(
    spark,
    base_path: str,
    canonical_station_id: str,
) -> dict[str, Any]:
    champion_path = f"{base_path}/models/summary/champion/current"
    safe_station_id = canonical_station_id.replace("'", "\\'")
    champion_df = spark.read.parquet(champion_path).where(
        f"station_id = '{safe_station_id}'"
    )
    rows = champion_df.limit(2).collect()

    if not rows:
        raise ValueError(
            f"No champion row found for canonical_station_id={canonical_station_id}"
        )
    if len(rows) > 1:
        raise ValueError(
            f"Multiple champion rows found for canonical_station_id={canonical_station_id}"
        )

    return rows[0].asDict(recursive=True)


def ensure_model_artifact_path_exists(path_value: str | None, label: str) -> None:
    if path_value is None or not str(path_value).strip():
        raise ValueError(f"Champion row is missing {label}")

    path_text = str(path_value).strip()
    if path_text.startswith("dbfs:/Volumes/"):
        candidate_path = Path(path_text[len("dbfs:") :])
    elif path_text.startswith("dbfs:/"):
        candidate_path = Path("/dbfs") / path_text[len("dbfs:/") :]
    elif path_text.startswith("file://"):
        candidate_path = Path(path_text[len("file://") :])
    else:
        candidate_path = Path(path_text)

    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Champion {label} does not exist: {candidate_path}. "
            "Re-run stage 07/08/09 pipeline to refresh model artifacts and champion snapshot."
        )


def validate_champion_row_for_station(
    champion_row: dict[str, Any],
    canonical_station_id: str,
) -> None:
    required_fields = [
        "station_id",
        "run_id",
        "target_col",
        "inflow_model_name",
        "outflow_model_name",
        "inflow_model_path",
        "outflow_model_path",
    ]
    missing_fields = [
        field
        for field in required_fields
        if champion_row.get(field) is None
        or not str(champion_row.get(field)).strip()
    ]
    if missing_fields:
        raise ValueError(
            f"Champion row for canonical_station_id={canonical_station_id} is missing fields: {missing_fields}"
        )

    champion_station_id = str(champion_row["station_id"])
    if champion_station_id != canonical_station_id:
        raise ValueError(
            "Champion row station mismatch: "
            f"expected {canonical_station_id}, got {champion_station_id}"
        )

    ensure_model_artifact_path_exists(
        champion_row.get("inflow_model_path"),
        "inflow_model_path",
    )
    ensure_model_artifact_path_exists(
        champion_row.get("outflow_model_path"),
        "outflow_model_path",
    )


def load_and_validate_champion_row_for_station(
    spark,
    base_path: str,
    canonical_station_id: str,
) -> dict[str, Any]:
    champion_row = load_champion_row_for_station(
        spark=spark,
        base_path=base_path,
        canonical_station_id=canonical_station_id,
    )
    validate_champion_row_for_station(
        champion_row=champion_row,
        canonical_station_id=canonical_station_id,
    )
    return champion_row


def load_champion_station_ids(spark, base_path: str) -> list[str]:
    champion_path = f"{base_path}/models/summary/champion/current"
    rows = (
        spark.read.parquet(champion_path)
        .select("station_id")
        .dropna(subset=["station_id"])
        .dropDuplicates(["station_id"])
        .collect()
    )
    station_ids = sorted(
        str(row["station_id"])
        for row in rows
        if row["station_id"] is not None
    )
    if not station_ids:
        raise ValueError(
            "Champion snapshot is empty: models/summary/champion/current has no station_id values"
        )
    return station_ids


def validate_canonical_station_against_champion_scope(
    canonical_station_id: str,
    champion_station_ids: list[str],
) -> None:
    if canonical_station_id in champion_station_ids:
        return

    if len(champion_station_ids) == 1:
        supported_station_id = champion_station_ids[0]
        raise ValueError(
            f"Request maps to canonical_station_id={canonical_station_id}, "
            f"but current champion scope only supports {supported_station_id}. "
            f"Use a request JSON that resolves to {supported_station_id}, for example: "
            f"{STN_0001_SAMPLE_REQUEST_JSON}"
        )

    supported_ids = ", ".join(champion_station_ids)
    raise ValueError(
        f"Request maps to canonical_station_id={canonical_station_id}, "
        f"which is not present in champion scope. Supported station ids: {supported_ids}"
    )


def fetch_forecast_weather_rows(
    lat: float,
    lon: float,
    request_ts_utc: datetime.datetime,
    horizon_steps: int = 6,
    timeout_sec: int = 20,
) -> WeatherHorizonResult:
    request_ts_utc = to_utc_hour(request_ts_utc)
    horizon_end = request_ts_utc + timedelta(hours=horizon_steps)

    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "hourly": "temperature_2m,precipitation",
        "timezone": "UTC",
        "forecast_days": 2,
    }

    response = requests.get(OPENMETEO_FORECAST_URL, params=params, timeout=timeout_sec)
    response.raise_for_status()
    payload = response.json()

    times = payload.get("hourly", {}).get("time", [])
    temperatures = payload.get("hourly", {}).get("temperature_2m", [])
    precipitations = payload.get("hourly", {}).get("precipitation", [])

    available_by_ts: dict[datetime.datetime, dict[str, Any]] = {}
    for ts_raw, temp, precip in zip(times, temperatures, precipitations):
        ts = datetime.datetime.fromisoformat(ts_raw).replace(tzinfo=timezone.utc)
        if request_ts_utc <= ts < horizon_end:
            available_by_ts[ts] = {
                "timestamp": ts.isoformat().replace("+00:00", "Z"),
                "temp": float(temp) if temp is not None else None,
                "precip": float(precip) if precip is not None else None,
            }

    expected_ts_list = [
        request_ts_utc + timedelta(hours=i)
        for i in range(horizon_steps)
    ]
    missing_ts = [
        ts.isoformat().replace("+00:00", "Z")
        for ts in expected_ts_list
        if ts not in available_by_ts
    ]

    if not available_by_ts:
        raise ValueError(
            "Weather horizon unavailable: no forecast points were returned by Open-Meteo "
            f"for [{request_ts_utc.isoformat().replace('+00:00', 'Z')}, "
            f"{horizon_end.isoformat().replace('+00:00', 'Z')})"
        )

    raw_temp_series: list[float | None] = []
    raw_precip_series: list[float | None] = []
    for ts in expected_ts_list:
        row = available_by_ts.get(ts)
        raw_temp_series.append(None if row is None else row.get("temp"))
        raw_precip_series.append(None if row is None else row.get("precip"))

    if all(value is None for value in raw_temp_series) and all(
        value is None for value in raw_precip_series
    ):
        raise ValueError(
            "Weather horizon unavailable: all requested hourly points are missing from Open-Meteo"
        )

    def interpolate_and_fill(
        series: list[float | None],
        metric_name: str,
    ) -> list[float]:
        known_idx = [i for i, value in enumerate(series) if value is not None]
        if not known_idx:
            raise ValueError(
                f"Weather horizon unavailable: {metric_name} has no usable values in requested window"
            )

        filled: list[float | None] = [
            float(value) if value is not None else None
            for value in series
        ]

        first_idx = known_idx[0]
        first_val = filled[first_idx]
        assert first_val is not None
        for i in range(first_idx):
            filled[i] = first_val

        for left_idx, right_idx in zip(known_idx, known_idx[1:]):
            left_val = filled[left_idx]
            right_val = filled[right_idx]
            assert left_val is not None
            assert right_val is not None
            gap = right_idx - left_idx
            if gap > 1:
                step = (right_val - left_val) / gap
                for i in range(left_idx + 1, right_idx):
                    filled[i] = left_val + step * (i - left_idx)

        last_idx = known_idx[-1]
        last_val = filled[last_idx]
        assert last_val is not None
        for i in range(last_idx + 1, len(filled)):
            filled[i] = last_val

        if any(value is None for value in filled):
            raise ValueError(
                f"Weather horizon interpolation failed to fill {metric_name} for all requested hours"
            )

        return [float(value) for value in filled]

    filled_temp_series = interpolate_and_fill(raw_temp_series, "temperature_2m")
    filled_precip_series = interpolate_and_fill(raw_precip_series, "precipitation")

    fallback_rows: list[dict[str, Any]] = []
    for idx, ts in enumerate(expected_ts_list):
        fallback_rows.append(
            {
                "timestamp": ts.isoformat().replace("+00:00", "Z"),
                "temp": float(filled_temp_series[idx]),
                "precip": float(filled_precip_series[idx]),
            }
        )

    fallback_applied = bool(missing_ts) or any(
        value is None for value in raw_temp_series + raw_precip_series
    )

    return WeatherHorizonResult(
        rows=fallback_rows,
        fallback_applied=fallback_applied,
        missing_timestamps=missing_ts,
    )


def _to_iso_z(ts: datetime.datetime) -> str:
    return to_utc_hour(ts).isoformat().replace("+00:00", "Z")


def _hour_of_week(ts: datetime.datetime) -> int:
    return ts.weekday() * 24 + ts.hour


def _clamp_nonnegative_count(value: float | None) -> int:
    if value is None:
        return 0
    return max(0, int(round(float(value))))


def load_observed_station_flow_history(
    spark,
    base_path: str,
    canonical_station_id: str,
    request_ts_utc: datetime.datetime,
    lookback_hours: int,
    warmup_hours: int,
    history_year_offset_days: int = 0,
) -> dict[datetime.datetime, tuple[float, float]]:
    request_ts_utc = to_utc_hour(request_ts_utc)
    start_ts_utc = request_ts_utc - timedelta(hours=lookback_hours + warmup_hours)

    offset_days = int(history_year_offset_days)
    if offset_days < 0:
        raise ValueError(
            f"history_year_offset_days must be >= 0, got: {history_year_offset_days}"
        )
    history_year_offset = timedelta(days=offset_days)

    query_request_ts_utc = request_ts_utc - history_year_offset
    query_start_ts_utc = start_ts_utc - history_year_offset

    safe_station_id = canonical_station_id.replace("'", "\\'")
    start_text = query_start_ts_utc.strftime("%Y-%m-%d %H:%M:%S")
    end_text = query_request_ts_utc.strftime("%Y-%m-%d %H:%M:%S")

    flow_path = f"{base_path}/gold/station_flow"
    try:
        flow_df = spark.read.parquet(flow_path)
    except Exception as exc:
        message = str(exc).lower()
        if "path does not exist" in message or "no such file or directory" in message:
            return {}
        raise

    station_rows = (
        flow_df.where(f"station_id = '{safe_station_id}'")
        .where(f"ts_hour >= TIMESTAMP('{start_text}')")
        .where(f"ts_hour < TIMESTAMP('{end_text}')")
        .select("ts_hour", "station_inflow", "station_outflow")
        .collect()
    )

    observed_by_ts: dict[datetime.datetime, tuple[float, float]] = {}
    for row in station_rows:
        ts_raw = row["ts_hour"]
        if ts_raw is None:
            continue
        ts_utc = to_utc_hour(ts_raw) + history_year_offset
        inflow = float(row["station_inflow"] or 0.0)
        outflow = float(row["station_outflow"] or 0.0)
        observed_by_ts[ts_utc] = (inflow, outflow)

    return observed_by_ts


def build_hour_of_week_medians(values_by_ts: dict[datetime.datetime, float]) -> dict[int, float]:
    values_by_how: dict[int, list[float]] = {}
    for ts, value in values_by_ts.items():
        values_by_how.setdefault(_hour_of_week(ts), []).append(float(value))

    medians: dict[int, float] = {}
    for how, values in values_by_how.items():
        if values:
            medians[how] = float(median(values))
    return medians


def _predict_iterative_value(
    ts: datetime.datetime,
    state_by_ts: dict[datetime.datetime, float],
    profile_by_how: dict[int, float],
    global_median: float,
) -> float:
    weighted_values: list[tuple[float, float]] = []
    lag_specs = [(1, 0.50), (24, 0.30), (168, 0.15)]
    for lag_hours, weight in lag_specs:
        lag_ts = ts - timedelta(hours=lag_hours)
        if lag_ts in state_by_ts:
            weighted_values.append((state_by_ts[lag_ts], weight))

    profile_value = profile_by_how.get(_hour_of_week(ts))
    if profile_value is not None:
        weighted_values.append((profile_value, 0.05))

    if not weighted_values:
        return float(global_median)

    total_weight = sum(weight for _, weight in weighted_values)
    blended = sum(value * weight for value, weight in weighted_values) / total_weight
    return max(0.0, float(blended))


def _predict_fallback_value(
    ts: datetime.datetime,
    state_by_ts: dict[datetime.datetime, float],
    profile_by_how: dict[int, float],
    global_median: float,
) -> float:
    profile_value = profile_by_how.get(_hour_of_week(ts))
    if profile_value is not None:
        return max(0.0, float(profile_value))

    lag_ts = ts - timedelta(hours=1)
    if lag_ts in state_by_ts:
        return max(0.0, float(state_by_ts[lag_ts]))

    return max(0.0, float(global_median))


def synthesize_history_iterative(
    observed_by_ts: dict[datetime.datetime, tuple[float, float]],
    target_timestamps: list[datetime.datetime],
) -> tuple[list[dict[str, Any]], int]:
    inflow_series = {ts: values[0] for ts, values in observed_by_ts.items()}
    outflow_series = {ts: values[1] for ts, values in observed_by_ts.items()}

    inflow_profile = build_hour_of_week_medians(inflow_series)
    outflow_profile = build_hour_of_week_medians(outflow_series)
    inflow_global = float(median(inflow_series.values())) if inflow_series else 0.0
    outflow_global = float(median(outflow_series.values())) if outflow_series else 0.0

    synthesized_count = 0
    rows: list[dict[str, Any]] = []
    for ts in target_timestamps:
        observed = observed_by_ts.get(ts)
        if observed is not None:
            inflow_value = float(observed[0])
            outflow_value = float(observed[1])
            source = "observed"
        else:
            inflow_value = _predict_iterative_value(
                ts=ts,
                state_by_ts=inflow_series,
                profile_by_how=inflow_profile,
                global_median=inflow_global,
            )
            outflow_value = _predict_iterative_value(
                ts=ts,
                state_by_ts=outflow_series,
                profile_by_how=outflow_profile,
                global_median=outflow_global,
            )
            source = "synthetic_iterative"
            synthesized_count += 1

        inflow_series[ts] = inflow_value
        outflow_series[ts] = outflow_value
        rows.append(
            {
                "timestamp": _to_iso_z(ts),
                "station_inflow": _clamp_nonnegative_count(inflow_value),
                "station_outflow": _clamp_nonnegative_count(outflow_value),
                "source": source,
            }
        )

    return rows, synthesized_count


def synthesize_history_fallback(
    observed_by_ts: dict[datetime.datetime, tuple[float, float]],
    target_timestamps: list[datetime.datetime],
) -> tuple[list[dict[str, Any]], int]:
    inflow_series = {ts: values[0] for ts, values in observed_by_ts.items()}
    outflow_series = {ts: values[1] for ts, values in observed_by_ts.items()}

    inflow_profile = build_hour_of_week_medians(inflow_series)
    outflow_profile = build_hour_of_week_medians(outflow_series)
    inflow_global = float(median(inflow_series.values())) if inflow_series else 0.0
    outflow_global = float(median(outflow_series.values())) if outflow_series else 0.0

    synthesized_count = 0
    rows: list[dict[str, Any]] = []
    for ts in target_timestamps:
        observed = observed_by_ts.get(ts)
        if observed is not None:
            inflow_value = float(observed[0])
            outflow_value = float(observed[1])
            source = "observed"
        else:
            inflow_value = _predict_fallback_value(
                ts=ts,
                state_by_ts=inflow_series,
                profile_by_how=inflow_profile,
                global_median=inflow_global,
            )
            outflow_value = _predict_fallback_value(
                ts=ts,
                state_by_ts=outflow_series,
                profile_by_how=outflow_profile,
                global_median=outflow_global,
            )
            source = "synthetic_fallback"
            synthesized_count += 1

        inflow_series[ts] = inflow_value
        outflow_series[ts] = outflow_value
        rows.append(
            {
                "timestamp": _to_iso_z(ts),
                "station_inflow": _clamp_nonnegative_count(inflow_value),
                "station_outflow": _clamp_nonnegative_count(outflow_value),
                "source": source,
            }
        )

    return rows, synthesized_count


def synthesize_station_history(
    spark,
    base_path: str,
    canonical_station_id: str,
    request_ts_utc: datetime.datetime,
    lookback_hours: int = 168,
    requested_mode: str = "auto",
    warmup_hours: int = 336,
    history_year_offset_days: int = 0,
) -> SyntheticHistoryResult:
    request_ts_utc = to_utc_hour(request_ts_utc)
    window_start_ts = request_ts_utc - timedelta(hours=lookback_hours)
    target_timestamps = [
        window_start_ts + timedelta(hours=i)
        for i in range(lookback_hours)
    ]

    observed_by_ts = load_observed_station_flow_history(
        spark=spark,
        base_path=base_path,
        canonical_station_id=canonical_station_id,
        request_ts_utc=request_ts_utc,
        lookback_hours=lookback_hours,
        warmup_hours=warmup_hours,
        history_year_offset_days=history_year_offset_days,
    )

    missing_before_fill = sum(
        1
        for ts in target_timestamps
        if ts not in observed_by_ts
    )

    mode_used = requested_mode
    if requested_mode == "iterative":
        rows, synthesized_count = synthesize_history_iterative(
            observed_by_ts=observed_by_ts,
            target_timestamps=target_timestamps,
        )
    elif requested_mode == "fallback":
        rows, synthesized_count = synthesize_history_fallback(
            observed_by_ts=observed_by_ts,
            target_timestamps=target_timestamps,
        )
    else:
        try:
            rows, synthesized_count = synthesize_history_iterative(
                observed_by_ts=observed_by_ts,
                target_timestamps=target_timestamps,
            )
            mode_used = "iterative"
        except Exception:
            rows, synthesized_count = synthesize_history_fallback(
                observed_by_ts=observed_by_ts,
                target_timestamps=target_timestamps,
            )
            mode_used = "fallback"

    return SyntheticHistoryResult(
        rows=rows,
        mode_requested=requested_mode,
        mode_used=mode_used,
        lookback_hours=lookback_hours,
        synthesized_count=synthesized_count,
        missing_before_fill=missing_before_fill,
        window_start=_to_iso_z(window_start_ts),
        window_end=_to_iso_z(request_ts_utc),
    )


# ---------------------------------------------------------------------------
# Milestone 5: Stage-06-compatible feature rows for inference horizon.
# ---------------------------------------------------------------------------


def parse_iso_utc_timestamp(raw_value: str) -> datetime.datetime:
    parsed = datetime.datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    return to_utc_hour(parsed)


def _spark_dayofweek_from_local(local_ts: datetime.datetime) -> int:
    iso_day = local_ts.isoweekday()  # Monday=1 ... Sunday=7
    return 1 if iso_day == 7 else iso_day + 1  # Spark: Sunday=1 ... Saturday=7


def build_temporal_feature_values(ts_utc: datetime.datetime) -> dict[str, Any]:
    normalized_utc = to_utc_hour(ts_utc)
    local_ts = normalized_utc.astimezone(MONTREAL_TZ)

    dow = _spark_dayofweek_from_local(local_ts)
    hod = int(local_ts.hour)
    moy = int(local_ts.month)
    two_pi = 2.0 * math.pi

    return {
        "dow": int(dow),
        "is_weekday": bool(2 <= dow <= 6),
        "hod": hod,
        "moy": moy,
        "dow_cos": float(math.cos(float(dow) * two_pi / 7.0)),
        "hod_cos": float(math.cos(float(hod) * two_pi / 24.0)),
        "moy_cos": float(math.cos(float(moy) * two_pi / 12.0)),
    }


def compute_temp_bin(temp_value: float | None) -> str | None:
    if temp_value is None:
        return None

    temp = float(temp_value)
    if temp < -20:
        return "<-20"
    if temp < -10:
        return "-20:-10"
    if temp < 0:
        return "-10:0"
    if temp < 10:
        return "0:10"
    if temp < 15:
        return "10:15"
    if temp < 20:
        return "15:20"
    if temp < 25:
        return "20:25"
    if temp < 30:
        return "25:30"
    return "30+"


def derive_model_input_column_groups(gold_columns: list[str]) -> dict[str, list[str]]:
    categorical_columns = [
        column
        for column in MODEL_INPUT_CATEGORICAL_COLUMNS
        if column in gold_columns
    ]
    boolean_columns = [
        column
        for column in MODEL_INPUT_BOOLEAN_COLUMNS
        if column in gold_columns
    ]
    numeric_columns = [
        column
        for column in gold_columns
        if column not in MODEL_INPUT_EXCLUDED_COLUMNS
        and column not in categorical_columns
        and column not in boolean_columns
    ]

    return {
        "categorical_columns": categorical_columns,
        "boolean_columns": boolean_columns,
        "numeric_columns": numeric_columns,
        "model_input_columns": categorical_columns + numeric_columns,
    }


def _build_numeric_profiles_for_columns(
    row_dicts: list[dict[str, Any]],
    numeric_columns: list[str],
) -> tuple[dict[str, dict[int, float]], dict[str, float]]:
    values_by_column_and_how: dict[str, dict[int, list[float]]] = {
        column: {}
        for column in numeric_columns
    }
    values_by_column: dict[str, list[float]] = {
        column: []
        for column in numeric_columns
    }

    for row in row_dicts:
        ts_raw = row.get("ts_hour")
        if ts_raw is None:
            continue
        ts_utc = to_utc_hour(ts_raw)
        local_ts = ts_utc.astimezone(MONTREAL_TZ)
        hour_of_week = local_ts.weekday() * 24 + local_ts.hour

        for column in numeric_columns:
            value = row.get(column)
            if value is None:
                continue
            numeric_value = float(value)
            values_by_column.setdefault(column, []).append(numeric_value)
            values_by_column_and_how.setdefault(column, {}).setdefault(hour_of_week, []).append(numeric_value)

    profile_by_column: dict[str, dict[int, float]] = {}
    global_median_by_column: dict[str, float] = {}

    for column in numeric_columns:
        by_how = values_by_column_and_how.get(column, {})
        profile_by_column[column] = {
            how: float(median(values))
            for how, values in by_how.items()
            if values
        }

        column_values = values_by_column.get(column, [])
        if column_values:
            global_median_by_column[column] = float(median(column_values))

    return profile_by_column, global_median_by_column


def _resolve_profile_value(
    column: str,
    hour_of_week: int,
    profile_by_column: dict[str, dict[int, float]],
    global_median_by_column: dict[str, float],
    default_value: float = 0.0,
) -> float:
    column_profile = profile_by_column.get(column, {})
    if hour_of_week in column_profile:
        return float(column_profile[hour_of_week])

    if column in global_median_by_column:
        return float(global_median_by_column[column])

    return float(default_value)


def _build_synthetic_station_profiles(
    synthetic_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[int, float]], dict[str, float]]:
    inflow_by_ts: dict[datetime.datetime, float] = {}
    outflow_by_ts: dict[datetime.datetime, float] = {}
    netflow_by_ts: dict[datetime.datetime, float] = {}

    for row in synthetic_rows:
        ts_raw = row.get("timestamp")
        if ts_raw is None:
            continue
        ts_utc = parse_iso_utc_timestamp(str(ts_raw))

        inflow = float(row.get("station_inflow") or 0.0)
        outflow = float(row.get("station_outflow") or 0.0)

        inflow_by_ts[ts_utc] = inflow
        outflow_by_ts[ts_utc] = outflow
        netflow_by_ts[ts_utc] = inflow - outflow

    profile_by_column = {
        "station_inflow": build_hour_of_week_medians(inflow_by_ts),
        "station_outflow": build_hour_of_week_medians(outflow_by_ts),
        "station_netflow": build_hour_of_week_medians(netflow_by_ts),
    }
    global_median_by_column = {
        "station_inflow": float(median(inflow_by_ts.values())) if inflow_by_ts else 0.0,
        "station_outflow": float(median(outflow_by_ts.values())) if outflow_by_ts else 0.0,
        "station_netflow": float(median(netflow_by_ts.values())) if netflow_by_ts else 0.0,
    }
    return profile_by_column, global_median_by_column


def load_stage06_gold_station_df(
    spark,
    base_path: str,
    canonical_station_id: str,
):
    gold_path = f"{base_path}/gold/station_flow"
    safe_station_id = canonical_station_id.replace("'", "\\'")
    try:
        station_df = spark.read.parquet(gold_path).where(
            f"station_id = '{safe_station_id}'"
        )
    except Exception as exc:
        message = str(exc).lower()
        if "path does not exist" in message or "no such file or directory" in message:
            raise FileNotFoundError(
                f"Stage-06 gold output missing at {gold_path}. Run stage 06 before milestone 5 inference feature build."
            ) from exc
        raise

    if station_df.limit(1).count() == 0:
        raise ValueError(
            f"No Stage-06 gold rows found for station_id={canonical_station_id} in {gold_path}"
        )

    return station_df


def build_stage06_compatible_horizon_feature_dataframes(
    spark,
    base_path: str,
    canonical_station_id: str,
    weather_rows: list[dict[str, Any]],
    synthetic_history_rows: list[dict[str, Any]],
):
    from pyspark.sql import functions as F

    station_gold_df = load_stage06_gold_station_df(
        spark=spark,
        base_path=base_path,
        canonical_station_id=canonical_station_id,
    )

    gold_columns = station_gold_df.columns
    gold_schema = station_gold_df.schema

    station_gold_rows = [
        row.asDict(recursive=True)
        for row in station_gold_df.select(*gold_columns).collect()
    ]

    profiled_numeric_columns = [
        column
        for column in gold_columns
        if column not in {"station_id", "ts_hour", "temp_bin", "is_weekday"}
    ]
    stage06_profile_by_column, stage06_global_by_column = _build_numeric_profiles_for_columns(
        row_dicts=station_gold_rows,
        numeric_columns=profiled_numeric_columns,
    )

    synthetic_profile_by_column, synthetic_global_by_column = _build_synthetic_station_profiles(
        synthetic_rows=synthetic_history_rows,
    )

    weather_by_ts: dict[datetime.datetime, dict[str, Any]] = {}
    for row in weather_rows:
        ts_raw = row.get("timestamp")
        if ts_raw is None:
            continue
        ts_utc = parse_iso_utc_timestamp(str(ts_raw))
        weather_by_ts[ts_utc] = {
            "temp": (
                float(row.get("temp"))
                if row.get("temp") is not None
                else None
            ),
            "precip": (
                float(row.get("precip"))
                if row.get("precip") is not None
                else None
            ),
        }

    if not weather_by_ts:
        raise ValueError("Weather artifact has no rows to build horizon feature rows")

    horizon_timestamps = sorted(weather_by_ts.keys())
    feature_rows: list[dict[str, Any]] = []
    for ts_utc in horizon_timestamps:
        local_ts = ts_utc.astimezone(MONTREAL_TZ)
        hour_of_week = local_ts.weekday() * 24 + local_ts.hour

        row: dict[str, Any] = {}
        for column in profiled_numeric_columns:
            row[column] = _resolve_profile_value(
                column=column,
                hour_of_week=hour_of_week,
                profile_by_column=stage06_profile_by_column,
                global_median_by_column=stage06_global_by_column,
                default_value=0.0,
            )

        # Station flow columns can be better anchored by synthetic-history profiles.
        for station_flow_col in ["station_inflow", "station_outflow", "station_netflow"]:
            if station_flow_col in synthetic_profile_by_column:
                row[station_flow_col] = _resolve_profile_value(
                    column=station_flow_col,
                    hour_of_week=hour_of_week,
                    profile_by_column=synthetic_profile_by_column,
                    global_median_by_column=synthetic_global_by_column,
                    default_value=row.get(station_flow_col, 0.0),
                )

        weather_point = weather_by_ts[ts_utc]
        row["station_id"] = canonical_station_id
        row["ts_hour"] = ts_utc
        row["temp"] = weather_point.get("temp")
        row["precip"] = weather_point.get("precip")
        row["temp_bin"] = compute_temp_bin(row.get("temp"))
        row.update(build_temporal_feature_values(ts_utc))

        if "station_inflow" in row and "station_outflow" in row:
            row["station_netflow"] = float(row["station_inflow"]) - float(row["station_outflow"])

        feature_rows.append(row)

    if not feature_rows:
        raise ValueError("No feature rows were generated for inference horizon")

    feature_df = spark.createDataFrame(feature_rows)
    for field in gold_schema.fields:
        if field.name in feature_df.columns:
            feature_df = feature_df.withColumn(field.name, F.col(field.name).cast(field.dataType))
        else:
            feature_df = feature_df.withColumn(field.name, F.lit(None).cast(field.dataType))

    feature_df = feature_df.select(*gold_columns).orderBy("ts_hour")

    model_column_groups = derive_model_input_column_groups(gold_columns)
    model_input_columns = [
        "station_id",
        "ts_hour",
        *model_column_groups["categorical_columns"],
        *model_column_groups["numeric_columns"],
        *model_column_groups["boolean_columns"],
    ]
    model_input_df = feature_df.select(*model_input_columns).orderBy("ts_hour")

    missing_gold_columns = [
        column
        for column in gold_columns
        if column not in feature_df.columns
    ]
    missing_model_input_columns = [
        column
        for column in model_column_groups["model_input_columns"]
        if column not in feature_df.columns
    ]

    metadata = {
        "gold_columns": gold_columns,
        "model_column_groups": model_column_groups,
        "missing_gold_columns": missing_gold_columns,
        "missing_model_input_columns": missing_model_input_columns,
        "row_count": len(feature_rows),
        "horizon_start": _to_iso_z(horizon_timestamps[0]),
        "horizon_end_exclusive": _to_iso_z(horizon_timestamps[-1] + timedelta(hours=1)),
    }

    return feature_df, model_input_df, metadata


# ---------------------------------------------------------------------------
# Milestone 6: Champion model scoring on inference feature rows.
# ---------------------------------------------------------------------------


def _normalize_prediction(value: Any) -> float:
    if value is None:
        return 0.0
    numeric_value = float(value)
    return max(0.0, numeric_value)


def score_champion_models_from_feature_rows(
    spark,
    model_input_path: str,
    inflow_model_path: str,
    outflow_model_path: str,
) -> dict[str, Any]:
    from pyspark.ml import PipelineModel
    from pyspark.sql import functions as F

    model_input_df = spark.read.parquet(model_input_path).orderBy("ts_hour")
    if model_input_df.limit(1).count() == 0:
        raise ValueError(
            f"Milestone 6 scoring input is empty: {model_input_path}"
        )

    inflow_model = PipelineModel.load(inflow_model_path)
    outflow_model = PipelineModel.load(outflow_model_path)

    inflow_rows = (
        inflow_model.transform(model_input_df)
        .select("ts_hour", F.col("prediction").alias("predicted_inflow"))
        .orderBy("ts_hour")
        .collect()
    )
    outflow_rows = (
        outflow_model.transform(model_input_df)
        .select("ts_hour", F.col("prediction").alias("predicted_outflow"))
        .orderBy("ts_hour")
        .collect()
    )

    if len(inflow_rows) != len(outflow_rows):
        raise ValueError(
            "Milestone 6 scoring row mismatch between inflow and outflow predictions: "
            f"inflow={len(inflow_rows)} outflow={len(outflow_rows)}"
        )

    timestamps: list[str] = []
    model_inflow: list[float] = []
    model_outflow: list[float] = []
    for inflow_row, outflow_row in zip(inflow_rows, outflow_rows):
        inflow_ts = to_utc_hour(inflow_row["ts_hour"])
        outflow_ts = to_utc_hour(outflow_row["ts_hour"])
        if inflow_ts != outflow_ts:
            raise ValueError(
                "Milestone 6 scoring timestamp mismatch between inflow and outflow outputs: "
                f"{_to_iso_z(inflow_ts)} vs {_to_iso_z(outflow_ts)}"
            )

        timestamps.append(_to_iso_z(inflow_ts))
        model_inflow.append(_normalize_prediction(inflow_row["predicted_inflow"]))
        model_outflow.append(_normalize_prediction(outflow_row["predicted_outflow"]))

    return {
        "timestamps": timestamps,
        "model_inflow": model_inflow,
        "model_outflow": model_outflow,
        "row_count": len(model_inflow),
    }


def build_spark_session():
    spark = get_spark()
    apply_local_spark_defaults(spark)
    return spark


def get_base_path() -> str:
    return resolve_data_path()
