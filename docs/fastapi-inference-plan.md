# FastAPI Inference Endpoint Plan

## Objective
Expose a FastAPI endpoint that validates an inference request, runs the stage-10 inference steps, and returns only the output JSON. Provide a test script that compares API output to the run_10_inference_pipeline.py output for the same request.

## Steps
1. Create src/fastapi/inference_service/ and add a FastAPI app with a Pydantic request model (station_id required; name/lat/lon/request_timestamp optional).
2. Implement POST /inference/run to:
   - serialize the request into INFERENCE_REQUEST_JSON,
   - run the step pipeline in-process with a lock to avoid env collisions,
   - return only the output payload dict.
3. Add scripts/test_fastapi_inference.py to:
   - run run_10_inference_pipeline.py with a fixed run_id/run_ts and the same request JSON,
   - call the API endpoint,
   - compare outputs with numeric tolerance and report differences.
