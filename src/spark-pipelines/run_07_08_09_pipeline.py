"""
Run stages 07 -> 08 -> 09 with a shared pipeline run context.

This mirrors Databricks job behavior by propagating one run_id and run_ts across
all three stages via environment variables.
"""

import argparse
import datetime
import os
import subprocess
import sys
import uuid
from pathlib import Path


DEFAULT_CUTOFF_DATE = "2025-08-01"


def parse_station_ids(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return sorted({token.strip() for token in raw_value.split(",") if token.strip()})


def build_default_run_id() -> str:
    env_run_id = os.environ.get("PIPELINE_RUN_ID")
    if env_run_id and env_run_id.strip():
        return env_run_id.strip()

    job_run_id = os.environ.get("PIPELINE_JOB_RUN_ID")
    if job_run_id and job_run_id.strip():
        repair_count = os.environ.get("PIPELINE_REPAIR_COUNT", "0").strip() or "0"
        return f"job_{job_run_id.strip()}_repair_{repair_count}"

    now_utc = datetime.datetime.now(datetime.UTC)
    return f"run_{now_utc.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"


def build_default_run_ts() -> str:
    env_run_ts = os.environ.get("PIPELINE_RUN_TS")
    if env_run_ts and env_run_ts.strip():
        return env_run_ts.strip()
    return datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")


def run_stage(
    python_executable: str,
    script_path: Path,
    env: dict[str, str],
    extra_args: list[str] | None = None,
) -> None:
    command = [python_executable, str(script_path)]
    if extra_args:
        command.extend(extra_args)
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage 07/08/09 with shared run metadata")
    parser.add_argument("--run-id", default=None, help="Optional explicit PIPELINE_RUN_ID")
    parser.add_argument("--run-ts", default=None, help="Optional explicit PIPELINE_RUN_TS (ISO UTC)")
    parser.add_argument(
        "--pipeline-station-id",
        default=os.environ.get("PIPELINE_STATION_ID"),
        help="Comma-separated station IDs for Stage 07 fan-out (maps to PIPELINE_STATION_ID)",
    )
    parser.add_argument(
        "--mode",
        default=os.environ.get("PIPELINE_MODE", "local"),
        help="Pipeline mode: local or production",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch stage scripts",
    )
    parser.add_argument(
        "--cutoff-date",
        default=os.environ.get("CUTOFF_DATE", DEFAULT_CUTOFF_DATE),
        help="Train/test split cutoff date shared by stages 07 and 08",
    )
    args = parser.parse_args()

    run_id = args.run_id.strip() if args.run_id and args.run_id.strip() else build_default_run_id()
    run_ts = args.run_ts.strip() if args.run_ts and args.run_ts.strip() else build_default_run_ts()

    script_dir = Path(__file__).resolve().parent
    stage07_script = script_dir / "07_build_models_from_gold_rides.py"
    stage08_script = script_dir / "08_combi_netflow_model.py"
    stage09_script = script_dir / "09_champ_netflow_model.py"
    stage_scripts = [stage07_script, stage08_script, stage09_script]

    for path in stage_scripts:
        if not path.exists():
            raise FileNotFoundError(f"Missing stage script: {path}")

    env = os.environ.copy()
    env["PIPELINE_MODE"] = str(args.mode)
    env["PIPELINE_RUN_ID"] = run_id
    env["PIPELINE_RUN_TS"] = run_ts
    env["CUTOFF_DATE"] = str(args.cutoff_date)
    if args.pipeline_station_id and args.pipeline_station_id.strip():
        env["PIPELINE_STATION_ID"] = args.pipeline_station_id.strip()

    # Compatibility aliases for legacy script options.
    env["STAGE7_RUN_ID"] = run_id
    env["CHAMP_RUN_ID"] = run_id

    station_ids = parse_station_ids(args.pipeline_station_id)
    if not station_ids:
        raise ValueError(
            "Provide --pipeline-station-id (or PIPELINE_STATION_ID) with at least one station id."
        )

    print("=== Local Runner 07 -> 08 -> 09 ===")
    print(f"PIPELINE_MODE: {env['PIPELINE_MODE']}")
    print(f"PIPELINE_RUN_ID: {run_id}")
    print(f"PIPELINE_RUN_TS: {run_ts}")
    print(f"CUTOFF_DATE: {env['CUTOFF_DATE']}")
    print(f"PIPELINE_STATION_ID: {','.join(station_ids)}")

    for station_id in station_ids:
        for target_col in ["station_inflow", "station_outflow"]:
            run_stage(
                python_executable=args.python,
                script_path=stage07_script,
                env=env,
                extra_args=[
                    "--pipeline-station-id",
                    station_id,
                    "--pipeline-target-col",
                    target_col,
                    "--cutoff-date",
                    env["CUTOFF_DATE"],
                ],
            )

    run_stage(
        python_executable=args.python,
        script_path=stage08_script,
        env=env,
        extra_args=[
            "--cutoff-date",
            env["CUTOFF_DATE"],
        ],
    )
    run_stage(
        python_executable=args.python,
        script_path=stage09_script,
        env=env,
    )

    print("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
