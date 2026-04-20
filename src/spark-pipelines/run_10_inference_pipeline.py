"""
Run stage-10 inference steps with a shared run context.

This mirrors run_07_08_09_pipeline.py behavior by propagating one run_id and run_ts
across all inference step scripts.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from inference_artifacts import build_default_run_id, build_default_run_ts


def run_step(
    python_executable: str,
    script_path: Path,
    env: dict[str, str],
    run_id: str,
    run_ts: str,
) -> None:
    command = [
        python_executable,
        str(script_path),
        "--run-id",
        run_id,
        "--run-ts",
        run_ts,
    ]
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage-10 step pipeline")
    parser.add_argument("--run-id", default=None, help="Optional explicit PIPELINE_RUN_ID")
    parser.add_argument("--run-ts", default=None, help="Optional explicit PIPELINE_RUN_TS")
    parser.add_argument(
        "--mode",
        default=os.environ.get("PIPELINE_MODE", "local"),
        help="Pipeline mode: local or production",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch step scripts",
    )
    args = parser.parse_args()

    run_id = args.run_id.strip() if args.run_id and args.run_id.strip() else build_default_run_id()
    run_ts = args.run_ts.strip() if args.run_ts and args.run_ts.strip() else build_default_run_ts()

    script_dir = Path(__file__).resolve().parent
    step_scripts = [
        script_dir / "inference_step_01_live_station.py",
        script_dir / "inference_step_02_weather.py",
        script_dir / "inference_step_03_history.py",
        script_dir / "inference_step_04_features.py",
        script_dir / "inference_step_05_inference.py",
    ]

    for path in step_scripts:
        if not path.exists():
            raise FileNotFoundError(f"Missing inference step script: {path}")

    env = os.environ.copy()
    env["PIPELINE_MODE"] = str(args.mode)
    env["PIPELINE_RUN_ID"] = run_id
    env["PIPELINE_RUN_TS"] = run_ts
    env["INFERENCE_RUN_ID"] = run_id
    env["INFERENCE_RUN_TS"] = run_ts

    print("=== Local Runner Stage 10 Steps ===")
    print(f"PIPELINE_MODE: {env['PIPELINE_MODE']}")
    print(f"PIPELINE_RUN_ID: {run_id}")
    print(f"PIPELINE_RUN_TS: {run_ts}")

    for step_script in step_scripts:
        run_step(
            python_executable=args.python,
            script_path=step_script,
            env=env,
            run_id=run_id,
            run_ts=run_ts,
        )

    print("=== Stage 10 step pipeline complete ===")


if __name__ == "__main__":
    main()
