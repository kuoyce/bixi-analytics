"""
Stage 10 draft: modular inference orchestrator.

This script executes inference as four sequential steps that communicate via
run-scoped artifacts under:
- models/inference/run_*/live_station
- models/inference/run_*/weather
- models/inference/run_*/synthetic_history
- models/inference/run_*/feature_rows
- models/inference/run_*/output

The final returned payload contains only the required output columns.
"""

from __future__ import annotations

import json

from inference_artifacts import (
    parse_positive_int_env,
    parse_synthesis_mode_env,
    resolve_inference_run_context,
)
from inference_step_01_live_station import run_live_station_step
from inference_step_02_weather import run_weather_step
from inference_step_03_history import run_history_step
from inference_step_04_features import run_feature_rows_step
from inference_step_05_inference import run_output_step


def run_inference_pipeline(
    run_id: str | None = None,
    run_ts: str | None = None,
    horizon_steps: int = 6,
    history_lookback_hours: int = 168,
    history_warmup_hours: int = 336,
    history_synthesis_mode: str = "auto",
) -> tuple[dict, str]:
    resolved_run_id, resolved_run_ts = resolve_inference_run_context(run_id, run_ts)

    run_live_station_step(
        run_id=resolved_run_id,
        run_ts=resolved_run_ts,
    )
    run_weather_step(
        run_id=resolved_run_id,
        run_ts=resolved_run_ts,
        horizon_steps=horizon_steps,
    )
    run_history_step(
        run_id=resolved_run_id,
        run_ts=resolved_run_ts,
        lookback_hours=history_lookback_hours,
        warmup_hours=history_warmup_hours,
        synthesis_mode=history_synthesis_mode,
    )
    run_feature_rows_step(
        run_id=resolved_run_id,
        run_ts=resolved_run_ts,
    )
    output_payload, output_path = run_output_step(
        run_id=resolved_run_id,
        run_ts=resolved_run_ts,
    )
    return output_payload, output_path


def main() -> None:
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

    output_payload, _ = run_inference_pipeline(
        horizon_steps=horizon_steps,
        history_lookback_hours=history_lookback_hours,
        history_warmup_hours=history_warmup_hours,
        history_synthesis_mode=history_synthesis_mode,
    )

    print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
