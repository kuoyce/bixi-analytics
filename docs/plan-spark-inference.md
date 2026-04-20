# Spark Inference Draft Plan (Stage 10)

## Objective
Refactor stage-10 inference into modular, run-scoped steps where each step writes its own artifact folder, and the final step reads all prior artifacts by `run_id` to produce the inference output.

## Modular Step Architecture
Stage-10 is now split into these scripts under `src/spark-pipelines/`:
1. `inference_step_01_live_station.py`
   - Resolves live station snapshot, canonical station id, and champion context.
2. `inference_step_02_weather.py`
   - Fetches forecast weather horizon for the request timestamp.
3. `inference_step_03_history.py`
   - Builds/synthesizes one-week station inflow/outflow history.
4. `inference_step_04_features.py`
   - Builds stage-06-compatible horizon feature rows and validates both:
     - final transformed stage-06 gold columns
     - stage-07 model input column groups
   - Saves feature artifacts under run-scoped inference subdirectories (not gold).
5. `inference_step_05_inference.py`
   - Reads all prior step artifacts and writes final output payload.
   - Runs Milestone-6 champion scoring using:
     - champion model paths from `live_station/live_station.json`
     - model input parquet path from `feature_rows/feature_rows.json`
6. `run_10_inference_pipeline.py`
   - Runs steps 01 -> 02 -> 03 -> 04_features -> output with one shared `run_id` and `run_ts`.
7. `10_inference_draft.py`
   - Thin orchestrator that executes the same step flow in-process.

Shared helpers are centralized in:
- `inference_artifacts.py`

## Run Context Contract
To match cross-step behavior used in `run_07_08_09_pipeline.py`, stage-10 uses a shared run context:
1. `PIPELINE_RUN_ID` (or `INFERENCE_RUN_ID`) for cross-step artifact grouping.
2. `PIPELINE_RUN_TS` (or `INFERENCE_RUN_TS`) for run timestamp propagation.
3. If not provided, run id and run ts are auto-generated.

Run folder naming:
- Artifact root is resolved by `sparkutils.resolve_inference_artifacts_root(...)`:
   - local default: `<data_path>/models/inference`
   - Databricks default: `/Volumes/workspace/bixi-fs/models/inference`
   - optional override: `INFERENCE_ARTIFACTS_ROOT`
- Artifacts are stored under `<resolved_inference_root>/run_*/...`.
- If run id already starts with `run_`, it is used directly.
- Otherwise, it is normalized to `run_<run_id>`.

## Artifact Folder Layout (Per Run)
For run id `X`, artifacts are written to:
1. `data/models/inference/run_X/live_station/live_station.json`
2. `data/models/inference/run_X/weather/weather.json`
3. `data/models/inference/run_X/synthetic_history/synthetic_history.json`
4. `data/models/inference/run_X/feature_rows/stage06_compatible/`
5. `data/models/inference/run_X/feature_rows/model_input/`
6. `data/models/inference/run_X/feature_rows/feature_rows.json`
7. `data/models/inference/run_X/output/station_<station_id>.json`

## Input Contract
Primary request source:
1. `INFERENCE_REQUEST_JSON` (optional full JSON payload)

Field-level fallback (when `INFERENCE_REQUEST_JSON` is not provided):
1. `INFERENCE_STATION_ID`
2. `INFERENCE_NAME`
3. `INFERENCE_LAT`
4. `INFERENCE_LON`
5. `INFERENCE_REQUEST_TIMESTAMP`

Runtime controls:
1. `INFERENCE_HORIZON_STEPS` (default `6`)
2. `INFERENCE_HISTORY_LOOKBACK_HOURS` (default `168`)
3. `INFERENCE_HISTORY_WARMUP_HOURS` (default `336`)
4. `INFERENCE_SYNTHESIS_MODE` (`auto|iterative|fallback`, default `auto`)

Compatibility fallback:
1. `PIPELINE_STATION_ID` is accepted when `INFERENCE_STATION_ID` is not set.

## Final Output Contract
The final output file contains only:
```
{
  "request_timestamp": ...,
  "station_id": ...,
  "capacity": ...,
  "num_bikes_available": ...,
  "num_docks_available": ...,
  "canonical_station_id": ...,
  "model_inflow": [],
  "model_outflow": []
}
```

## Status Record
This section is the long-lived progress record for stage-10.
Implemented items should stay listed here (do not remove), while remaining work stays in the pending section.

### Completed (Record)
1. Designed and implemented a modular stage-10 step architecture.
2. Implemented shared inference helpers in `inference_artifacts.py`.
3. Implemented step 01 (`inference_step_01_live_station.py`) for live station + canonical + champion validation.
4. Implemented step 02 (`inference_step_02_weather.py`) for weather horizon retrieval with partial-gap fill behavior.
5. Implemented step 03 (`inference_step_03_history.py`) for one-week synthetic history generation (`iterative|fallback|auto`).
6. Implemented step 05 (`inference_step_05_inference.py`) to read prior step artifacts by `run_id` and emit final output.
7. Implemented `run_10_inference_pipeline.py` to propagate shared `run_id` and `run_ts` across stage-10 steps.
8. Refactored `10_inference_draft.py` into a thin in-process orchestrator that runs the same step flow.
9. Constrained final payload schema to exactly these 8 fields: `request_timestamp`, `station_id`, `capacity`, `num_bikes_available`, `num_docks_available`, `canonical_station_id`, `model_inflow`, and `model_outflow`.
10. Implemented Milestone 5 feature-row generation in `inference_step_04_features.py`:
   - Builds stage-06-compatible horizon rows.
   - Checks full transformed column coverage against stage-06 gold schema.
   - Checks model input coverage against stage-07 model input column groups.
   - Writes artifacts to `models/inference/run_*/feature_rows/...` instead of gold.
11. Implemented Milestone 6 scoring in `inference_step_05_inference.py`:
   - Loads champion inflow/outflow pipeline model paths from `live_station/live_station.json`.
   - Loads horizon model input rows from `feature_rows/model_input/`.
   - Produces non-negative prediction arrays for `model_inflow` and `model_outflow`.
12. Implemented Databricks-compatible inference artifact pathing:
   - Centralized inference artifact root resolution in `sparkutils.resolve_inference_artifacts_root(...)`.
   - Default Databricks artifact destination is `/Volumes/workspace/bixi-fs/models/inference`.
   - Updated `inference_artifacts.py` step-directory resolution to use the centralized helper.
   - Added `dbfs:/` and `dbfs:/Volumes/...` handling for champion model artifact path existence checks.

### Pending (Remaining Work)
1. No open milestone remains in the current Stage-10 scope.

## Validation Summary
Validated with sample request:
- `{"name":"du Mont-Royal / Clark","lat":45.51941,"lon":-73.58685,"station_id":"218"}`

Confirmed behavior:
1. All step artifacts are written under one run folder.
2. Milestone-5 feature artifacts are written under `.../feature_rows/` and include:
   - `stage06_compatible/`
   - `model_input/`
   - `feature_rows.json` with schema/input coverage checks
3. `feature_rows.json` reports no missing stage-06 transformed columns and no missing model input columns.
4. Final output is written to `.../output/station_218.json`.
5. Final payload includes exactly the required eight fields.
6. `model_inflow` and `model_outflow` are populated from champion model scoring (not placeholders).
