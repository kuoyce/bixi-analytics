"""
Step 04: materialize stage-06-compatible feature rows for inference horizon.

Reads:
- live_station/live_station.json
- weather/weather.json
- synthetic_history/synthetic_history.json

Writes:
- models/inference/run_*/feature_rows/stage06_compatible
- models/inference/run_*/feature_rows/model_input
- models/inference/run_*/feature_rows/feature_rows.json
"""

from __future__ import annotations

import argparse
import json

from inference_artifacts import (
    build_spark_session,
    build_stage06_compatible_horizon_feature_dataframes,
    get_base_path,
    get_step_dir,
    read_step_artifact,
    resolve_inference_run_context,
    write_step_artifact,
)


def run_feature_rows_step(run_id: str, run_ts: str) -> tuple[dict, str]:
    base_path = get_base_path()

    live_payload = read_step_artifact(
        base_path=base_path,
        run_id=run_id,
        step_name="live_station",
        filename="live_station.json",
    )
    weather_payload = read_step_artifact(
        base_path=base_path,
        run_id=run_id,
        step_name="weather",
        filename="weather.json",
    )
    synthetic_history_payload = read_step_artifact(
        base_path=base_path,
        run_id=run_id,
        step_name="synthetic_history",
        filename="synthetic_history.json",
    )

    station_id = str(live_payload["live_station"]["station_id"])
    canonical_station_id = str(live_payload["canonical_station_id"])

    spark = build_spark_session()
    try:
        feature_df, model_input_df, metadata = build_stage06_compatible_horizon_feature_dataframes(
            spark=spark,
            base_path=base_path,
            canonical_station_id=canonical_station_id,
            weather_rows=weather_payload["rows"],
            synthetic_history_rows=synthetic_history_payload["rows"],
        )

        if metadata["missing_gold_columns"]:
            raise ValueError(
                "Milestone 5 feature build is missing Stage-06 columns: "
                f"{metadata['missing_gold_columns']}"
            )
        if metadata["missing_model_input_columns"]:
            raise ValueError(
                "Milestone 5 feature build is missing model input columns: "
                f"{metadata['missing_model_input_columns']}"
            )

        step_dir = get_step_dir(base_path, run_id, "feature_rows")
        step_dir.mkdir(parents=True, exist_ok=True)

        stage06_compatible_path = str(step_dir / "stage06_compatible")
        model_input_path = str(step_dir / "model_input")

        feature_df.write.mode("overwrite").parquet(stage06_compatible_path)
        model_input_df.write.mode("overwrite").parquet(model_input_path)
    finally:
        spark.stop()

    payload = {
        "run_id": run_id,
        "run_ts": run_ts,
        "request_timestamp": live_payload["request_timestamp"],
        "station_id": station_id,
        "canonical_station_id": canonical_station_id,
        "feature_rows_path": stage06_compatible_path,
        "model_input_path": model_input_path,
        "row_count": metadata["row_count"],
        "horizon_start": metadata["horizon_start"],
        "horizon_end_exclusive": metadata["horizon_end_exclusive"],
        "gold_columns": metadata["gold_columns"],
        "model_column_groups": metadata["model_column_groups"],
        "missing_gold_columns": metadata["missing_gold_columns"],
        "missing_model_input_columns": metadata["missing_model_input_columns"],
    }

    artifact_path = write_step_artifact(
        base_path=base_path,
        run_id=run_id,
        step_name="feature_rows",
        payload=payload,
        filename="feature_rows.json",
    )
    return payload, artifact_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference Step 04: stage-06-compatible feature rows")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-ts", default=None)
    args = parser.parse_args()

    run_id, run_ts = resolve_inference_run_context(args.run_id, args.run_ts)
    payload, artifact_path = run_feature_rows_step(run_id=run_id, run_ts=run_ts)

    print(
        json.dumps(
            {
                "step": "feature_rows",
                "run_id": run_id,
                "run_ts": run_ts,
                "artifact_path": artifact_path,
                "feature_rows_path": payload["feature_rows_path"],
                "model_input_path": payload["model_input_path"],
                "row_count": payload["row_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
