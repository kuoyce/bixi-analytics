import os

from pyspark.sql.functions import (
    col,
    lower,
    trim,
    regexp_replace,
    format_string,
    round,
    concat_ws,
    broadcast,
)

from sparkutils import get_spark, resolve_data_path, read_table, write_table, apply_local_spark_defaults


def normalize_station_name(column):
    return lower(
        trim(
            regexp_replace(
                regexp_replace(
                    regexp_replace(column, r"\s+", " "),
                    r"\s*/\s*",
                    " / ",
                ),
                r"\s*-\s*",
                "-",
            )
        )
    )


def parse_target_years(raw_years: str):
    if not raw_years:
        return []
    years = []
    for token in raw_years.split(","):
        token = token.strip()
        if not token:
            continue
        years.append(int(token))
    return sorted(set(years))



def main():
    spark = get_spark()
    apply_local_spark_defaults(spark)
    # spark.conf.set("spark.sql.files.maxRecordsPerFile", "500000")
    MAXRECORDSPERFILE = int(os.environ.get("AUGMENT_MAX_RECORDS_PER_FILE", "100000"))
    write_strategy = os.environ.get("AUGMENT_WRITE_STRATEGY", "by_year").strip().lower()
    enable_full_metrics = os.environ.get("AUGMENT_ENABLE_FULL_METRICS", "0") == "1"
    target_years = parse_target_years(os.environ.get("AUGMENT_TARGET_YEARS", ""))

    base_path = resolve_data_path()
    # if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
    #     default_format = "delta"
    # else:
    #     default_format = "parquet"

    default_format = "parquet"
    read_format = default_format
    write_format = default_format

    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        write_format = "delta"
        table_location = "workspace.`bixi-fs`"

    # storage_format = os.environ.get("SILVER_STORAGE_FORMAT", default_format)
    
    rides_stage_path = f"{base_path}/silver/rides_stage"
    rides_final_path = f"{base_path}/silver/rides"
    if write_format == "delta":
        rides_final_path = f"{table_location}.`silver-rides`"
    station_cleaning_base = f"{base_path}/silver/station_cleaning"
    mapping_csv_path = f"{station_cleaning_base}/station_direct_match_mapping_csv"
    mapping_table_path = f"{station_cleaning_base}/station_direct_match_mapping"

    silver_rides_df = read_table(spark, rides_stage_path, read_format)

    # Prefer parquet table over CSV for better performance
    if os.path.exists(mapping_table_path):
        mapping_df = (
            read_table(spark, mapping_table_path, read_format)
            .select("station_key", "canonical_station_id", "canonical_lat", "canonical_lon", "normalized_name")
            .dropDuplicates(["station_key"])
        )
        print(f"Using mapping source (parquet): {mapping_table_path}")
    elif os.path.exists(mapping_csv_path):
        mapping_df = (
            spark.read.option("header", True)
            .csv(mapping_csv_path)
            .select("station_key", "canonical_station_id")
            .dropDuplicates(["station_key"])
        )
        print(f"Using mapping source (CSV fallback): {mapping_csv_path}")
    else:
        raise FileNotFoundError(f"Mapping not found at {mapping_table_path} or {mapping_csv_path}")

    keyed_rides_df = (
        silver_rides_df
        .withColumn("start_station_name_norm", normalize_station_name(col("start_station_name")))
        .withColumn("end_station_name_norm", normalize_station_name(col("end_station_name")))
        .withColumn(
            "start_coord_key",
            format_string(
                "%.6f,%.6f",
                round(col("start_station_latitude"), 6),
                round(col("start_station_longitude"), 6),
            ),
        )
        .withColumn(
            "end_coord_key",
            format_string(
                "%.6f,%.6f",
                round(col("end_station_latitude"), 6),
                round(col("end_station_longitude"), 6),
            ),
        )
        .withColumn("start_station_key", concat_ws("|", col("start_coord_key"), col("start_station_name_norm")))
        .withColumn("end_station_key", concat_ws("|", col("end_coord_key"), col("end_station_name_norm")))
    )

    start_map_df = mapping_df.select(
        col("station_key").alias("start_station_key"),
        col("canonical_station_id").alias("start_canonical_station_id"),
    )
    end_map_df = mapping_df.select(
        col("station_key").alias("end_station_key"),
        col("canonical_station_id").alias("end_canonical_station_id"),
    )

    # Mapping tables are small relative to rides; broadcast them to avoid large shuffle joins.
    augmented_df = keyed_rides_df.join(broadcast(start_map_df), on="start_station_key", how="left").join(
        broadcast(end_map_df), on="end_station_key", how="left"
    )

    # Do not force a high repartition by default; it can trigger OOM in local mode.
    target_repartition = int(os.environ.get("AUGMENT_REPARTITION_PARTITIONS", "0"))
    if target_repartition > 0:
        augmented_df = augmented_df.repartition(target_repartition, col("ride_year"))

    if write_strategy not in {"all", "by_year"}:
        raise ValueError("AUGMENT_WRITE_STRATEGY must be one of: all, by_year")

    if write_strategy == "all" or write_format == "delta":
        write_table(
            augmented_df,
            rides_final_path,
            write_format,
            partition_cols=["ride_year"],
            maxRecordsPerFile=MAXRECORDSPERFILE,
        )
        years_written = [r["ride_year"] for r in augmented_df.select("ride_year").distinct().collect()]
    else:
        # Dynamic partition overwrite updates only touched ride_year partitions.
        spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

        if target_years:
            years_written = target_years
        else:
            years_written = [r["ride_year"] for r in augmented_df.select("ride_year").distinct().collect()]

        years_written = sorted([y for y in years_written if y is not None])
        if not years_written:
            raise ValueError("No valid ride_year values found to write.")

        for ride_year in years_written:
            year_df = augmented_df.filter(col("ride_year") == ride_year)
            if target_repartition > 0:
                year_df = year_df.repartition(target_repartition)
            write_table(
                year_df,
                rides_final_path,
                write_format,
                partition_cols=["ride_year"],
                maxRecordsPerFile=MAXRECORDSPERFILE,
            )

    total_rows = None
    start_mapping_rate = None
    end_mapping_rate = None
    if enable_full_metrics:
        total_rows = augmented_df.count()
        start_unmapped_rows = augmented_df.filter(col("start_canonical_station_id").isNull()).count()
        end_unmapped_rows = augmented_df.filter(col("end_canonical_station_id").isNull()).count()

        start_mapped_rows = total_rows - start_unmapped_rows
        end_mapped_rows = total_rows - end_unmapped_rows

        start_mapping_rate = (start_mapped_rows / total_rows * 100.0) if total_rows else 0.0
        end_mapping_rate = (end_mapped_rows / total_rows * 100.0) if total_rows else 0.0

    print("=== Stage 4 complete: augment silver rides with station mapping ===")
    print(f"Storage format: {write_format}")
    print(f"spark.sql.shuffle.partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
    print(f"spark.sql.files.maxPartitionBytes: {spark.conf.get('spark.sql.files.maxPartitionBytes')}")
    print(f"Max records per file: {MAXRECORDSPERFILE}")
    print(f"Write strategy: {write_strategy}")
    print(f"Years written: {years_written}")
    print(f"Rides source path: {rides_stage_path}")
    print(f"Mapping source: {mapping_table_path} or {mapping_csv_path}")
    print(f"Rides final path: {rides_final_path}")
    if enable_full_metrics:
        print(f"Total rides checked: {total_rows:,}")
        print(f"Start mapping rate: {start_mapping_rate:.4f}%")
        print(f"End mapping rate:   {end_mapping_rate:.4f}%")
    else:
        print("Full metrics disabled (set AUGMENT_ENABLE_FULL_METRICS=1 to enable).")

    spark.stop()


if __name__ == "__main__":
    main()
