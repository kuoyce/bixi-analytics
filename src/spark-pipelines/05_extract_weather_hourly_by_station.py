import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import openmeteo_requests
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from sparkutils import (
    apply_local_spark_defaults,
    get_spark,
    resolve_data_path,
    set_log_level_safe,
)


def parse_station_ids(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return sorted({token.strip() for token in raw_value.split(",") if token.strip()})


def parse_bootstrap_ts(raw_value: str) -> datetime:
    return datetime.strptime(raw_value, "%Y-%m-%d %H:%M:%S")


def empty_weather_df(spark: SparkSession):
    schema = T.StructType(
        [
            T.StructField("canonical_station_id", T.StringType(), False),
            T.StructField("ts_hour", T.TimestampType(), False),
            T.StructField("temp", T.DoubleType(), True),
            T.StructField("precip", T.DoubleType(), True),
        ]
    )
    return spark.createDataFrame([], schema)


def load_existing_station_max_ts(spark: SparkSession, output_path: str, station_ids: list[str]) -> dict[str, datetime]:
    if not os.path.exists(output_path):
        return {}

    try:
        existing_idx = (
            spark.read.parquet(output_path)
            .where(F.col("canonical_station_id").isin(station_ids))
            .select("canonical_station_id", "ts_hour")
        )
        return {
            r["canonical_station_id"]: r["max_ts"]
            for r in existing_idx.groupBy("canonical_station_id").agg(F.max("ts_hour").alias("max_ts")).collect()
        }
    except Exception:
        return {}


def fetch_station_weather_rows(
    client: openmeteo_requests.Client,
    url: str,
    timezone_name: str,
    local_tz: ZoneInfo,
    lat: float,
    lon: float,
    start_ts: datetime,
    end_ts: datetime,
) -> list[dict]:
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": start_ts.date().strftime("%Y-%m-%d"),
        "end_date": end_ts.date().strftime("%Y-%m-%d"),
        "hourly": ["temperature_2m", "precipitation"],
        "timezone": timezone_name,
    }

    response = client.weather_api(url, params=params)[0]
    hourly = response.Hourly()

    start_utc = datetime.fromtimestamp(hourly.Time(), tz=timezone.utc)
    end_utc = datetime.fromtimestamp(hourly.TimeEnd(), tz=timezone.utc)
    interval = timedelta(seconds=hourly.Interval())

    temp_values = list(hourly.Variables(0).ValuesAsNumpy())
    precip_values = list(hourly.Variables(1).ValuesAsNumpy())
    total_points = min(len(temp_values), len(precip_values))

    rows: list[dict] = []
    for i in range(total_points):
        ts_utc = start_utc + (interval * i)
        if ts_utc >= end_utc:
            break
        ts_local = ts_utc.astimezone(local_tz).replace(tzinfo=None)
        if start_ts <= ts_local <= end_ts:
            rows.append(
                {
                    "ts_hour": ts_local,
                    "temp": float(temp_values[i]) if temp_values[i] is not None else None,
                    "precip": float(precip_values[i]) if precip_values[i] is not None else None,
                }
            )

    return rows


def main():
    spark = get_spark()
    apply_local_spark_defaults(spark)
    set_log_level_safe(spark, "WARN")

    url = "https://archive-api.open-meteo.com/v1/archive"
    timezone_name = os.environ.get("WEATHER_TIMEZONE", "America/Montreal")
    local_tz = ZoneInfo(timezone_name)

    selected_station_ids = parse_station_ids(os.environ.get("WEATHER_STATION_IDS"))
    max_stations_per_run = int(os.environ.get("WEATHER_MAX_STATIONS_PER_RUN", "5"))
    bootstrap_start_ts = parse_bootstrap_ts(
        os.environ.get("WEATHER_BOOTSTRAP_START_TS", "2023-01-01 00:00:00")
    )

    base_path = resolve_data_path()
    station_summary_path = f"{base_path}/silver/station_cleaning/station_canonical_summary"
    output_path = f"{base_path}/silver/weather/hourly_by_station"
    summary_dir = Path(f"{base_path}/silver/weather/hourly_by_station_summary")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "weather_summary.json"

    client = openmeteo_requests.Client()

    stations_sdf = (
        spark.read.parquet(station_summary_path)
        .select(
            F.col("canonical_station_id"),
            F.col("canonical_lat").alias("lat"),
            F.col("canonical_lon").alias("lon"),
        )
        .dropna(subset=["canonical_station_id", "lat", "lon"])
        .dropDuplicates(["canonical_station_id"])
        .orderBy("canonical_station_id")
    )

    now_local = datetime.now(local_tz).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    request_end_ts = now_local

    if selected_station_ids:
        stations_sdf = stations_sdf.where(F.col("canonical_station_id").isin(selected_station_ids))
        station_rows = stations_sdf.orderBy("canonical_station_id").collect()
        if not station_rows:
            raise ValueError("No stations selected/found after applying station filters.")

        selected_ids = [r["canonical_station_id"] for r in station_rows]
        existing_last_by_station = load_existing_station_max_ts(spark, output_path, selected_ids)
        selection_mode = "explicit_station_ids"
    else:
        all_station_rows = stations_sdf.collect()
        if not all_station_rows:
            raise ValueError("No stations found in station_canonical_summary.")

        all_station_ids = [r["canonical_station_id"] for r in all_station_rows]
        existing_last_by_station = load_existing_station_max_ts(spark, output_path, all_station_ids)

        candidate_ids = []
        for sid in all_station_ids:
            last_ts = existing_last_by_station.get(sid)
            if last_ts is None or last_ts < request_end_ts:
                candidate_ids.append(sid)

        selected_ids = candidate_ids[:max_stations_per_run]
        if not selected_ids:
            selected_ids = all_station_ids[:max_stations_per_run]

        stations_by_id = {r["canonical_station_id"]: r for r in all_station_rows}
        station_rows = [stations_by_id[sid] for sid in selected_ids if sid in stations_by_id]
        selection_mode = "auto_up_to_date_aware"

    request_log: list[dict] = []
    fetched_records: list[dict] = []

    for row in station_rows:
        station_id = row["canonical_station_id"]
        lat = row["lat"]
        lon = row["lon"]

        last_ts = existing_last_by_station.get(station_id)
        if last_ts is not None and last_ts >= request_end_ts:
            request_log.append(
                {
                    "canonical_station_id": station_id,
                    "request_start": last_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "request_end": request_end_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "skipped_up_to_date",
                }
            )
            continue

        if last_ts is None:
            request_start_ts = bootstrap_start_ts
        else:
            # Keep a 1-hour overlap only when we actually need to fetch new data.
            request_start_ts = (last_ts - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

        try:
            rows = fetch_station_weather_rows(
                client,
                url,
                timezone_name,
                local_tz,
                lat,
                lon,
                request_start_ts,
                request_end_ts,
            )
            for rec in rows:
                rec["canonical_station_id"] = station_id
                fetched_records.append(rec)

            request_log.append(
                {
                    "canonical_station_id": station_id,
                    "request_start": request_start_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "request_end": request_end_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "fetched",
                }
            )
        except Exception as exc:
            request_log.append(
                {
                    "canonical_station_id": station_id,
                    "request_start": request_start_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "request_end": request_end_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    if fetched_records:
        fetched_sdf = spark.createDataFrame(fetched_records).select(
            "canonical_station_id", "ts_hour", "temp", "precip"
        )
        touched_station_ids = [
            r["canonical_station_id"]
            for r in fetched_sdf.select("canonical_station_id").distinct().collect()
        ]

        if os.path.exists(output_path):
            existing_touched = (
                spark.read.parquet(output_path)
                .where(F.col("canonical_station_id").isin(touched_station_ids))
                .select("canonical_station_id", "ts_hour", "temp", "precip")
            )
        else:
            existing_touched = empty_weather_df(spark)

        merged = existing_touched.withColumn("__source_rank", F.lit(1)).unionByName(
            fetched_sdf.withColumn("__source_rank", F.lit(2)),
            allowMissingColumns=True,
        )

        w = Window.partitionBy("canonical_station_id", "ts_hour").orderBy(F.col("__source_rank").desc())
        updated_touched = (
            merged.withColumn("__rn", F.row_number().over(w))
            .where(F.col("__rn") == 1)
            .drop("__rn", "__source_rank")
        )

        # Explicitly cast write columns so parquet write schema is stable.
        write_df = updated_touched.select(
            F.col("canonical_station_id").cast("string").alias("canonical_station_id"),
            F.col("ts_hour").cast("timestamp").alias("ts_hour"),
            F.col("temp").cast("double").alias("temp"),
            F.col("precip").cast("double").alias("precip"),
        )

        writer = write_df.write.mode("overwrite").partitionBy("canonical_station_id")
        if os.path.exists(output_path):
            writer = writer.option("partitionOverwriteMode", "dynamic")
        writer.parquet(output_path)
    else:
        touched_station_ids = []

    if os.path.exists(output_path):
        try:
            final_sdf = spark.read.parquet(output_path)
            metrics = final_sdf.agg(
                F.count("*").alias("row_count"),
                F.min("ts_hour").alias("data_start"),
                F.max("ts_hour").alias("data_end"),
            ).collect()[0]
            row_count = int(metrics["row_count"])
            data_start = metrics["data_start"].isoformat() if metrics["data_start"] else None
            data_end = metrics["data_end"].isoformat() if metrics["data_end"] else None
        except Exception:
            row_count = 0
            data_start = None
            data_end = None
    else:
        row_count = 0
        data_start = None
        data_end = None

    failed_count = sum(1 for r in request_log if r["status"] == "failed")
    fetched_count = sum(1 for r in request_log if r["status"] == "fetched")
    skipped_count = sum(1 for r in request_log if r["status"] == "skipped_up_to_date")

    summary = {
        "dataset": "weather_hourly_by_station",
        "timezone": timezone_name,
        "output_dir": output_path,
        "selected_station_count": len(selected_ids),
        "fetched_station_count": int(fetched_count),
        "skipped_station_count": int(skipped_count),
        "failed_station_count": int(failed_count),
        "touched_partition_count": len(touched_station_ids),
        "row_count": row_count,
        "data_start": data_start,
        "data_end": data_end,
        "last_refresh_utc": datetime.now(timezone.utc).isoformat(),
        "requests": request_log,
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Stage 5 complete: extract weather hourly by station ===")
    print(f"Selection mode: {selection_mode}")
    print(f"Output dir: {output_path}")
    print(f"Summary path: {summary_path}")
    print(f"Stations selected: {len(selected_ids)}")
    print(f"Stations fetched: {fetched_count}; skipped: {skipped_count}; failed: {failed_count}")
    print(f"Rows: {row_count}")
    print(f"Coverage: {data_start} -> {data_end}")

    spark.stop()


if __name__ == "__main__":
    main()
