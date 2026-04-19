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


def run_stage(python_executable: str, script_path: Path, env: dict[str, str]) -> None:
    command = [python_executable, str(script_path)]
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage 07/08/09 with shared run metadata")
    parser.add_argument("--run-id", default=None, help="Optional explicit PIPELINE_RUN_ID")
    parser.add_argument("--run-ts", default=None, help="Optional explicit PIPELINE_RUN_TS (ISO UTC)")
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
    args = parser.parse_args()

    run_id = args.run_id.strip() if args.run_id and args.run_id.strip() else build_default_run_id()
    run_ts = args.run_ts.strip() if args.run_ts and args.run_ts.strip() else build_default_run_ts()

    script_dir = Path(__file__).resolve().parent
    stage_scripts = [
        script_dir / "07_build_models_from_gold_rides.py",
        script_dir / "08_combi_netflow_model.py",
        script_dir / "09_champ_netflow_model.py",
    ]

    for path in stage_scripts:
        if not path.exists():
            raise FileNotFoundError(f"Missing stage script: {path}")

    env = os.environ.copy()
    env["PIPELINE_MODE"] = str(args.mode)
    env["PIPELINE_RUN_ID"] = run_id
    env["PIPELINE_RUN_TS"] = run_ts

    # Compatibility aliases for legacy script options.
    env["STAGE7_RUN_ID"] = run_id
    env["CHAMP_RUN_ID"] = run_id

    print("=== Local Runner 07 -> 08 -> 09 ===")
    print(f"PIPELINE_MODE: {env['PIPELINE_MODE']}")
    print(f"PIPELINE_RUN_ID: {run_id}")
    print(f"PIPELINE_RUN_TS: {run_ts}")

    for stage_script in stage_scripts:
        stage_args = [args.python, str(stage_script)]
        print(f"Running: {' '.join(stage_args)}")
        subprocess.run(stage_args, env=env, check=True)

    print("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
