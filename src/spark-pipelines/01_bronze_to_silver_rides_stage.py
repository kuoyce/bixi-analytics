import os

from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import col, to_timestamp, from_unixtime, year

from sparkutils import get_spark, resolve_data_path, read_table, write_table

def main():
    spark = get_spark()
    # spark.conf.set("spark.sql.files.maxRecordsPerFile", "500000")
    MAXRECORDSPERFILE = 500000
    spark.conf.set("spark.sql.shuffle.partitions", "200")

    base_path = resolve_data_path()
    # if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
    #     default_format = "delta"
    # else:
    #     default_format = "parquet"
    default_format = "parquet"

    storage_format = os.environ.get("SILVER_STORAGE_FORMAT", default_format)

    bronze_path = f"{base_path}/bronze/"
    rides_stage_path = f"{base_path}/silver/rides_stage"

    french_to_english = {
        "STARTSTATIONNAME": "start_station_name",
        "STARTSTATIONARRONDISSEMENT": "start_station_district",
        "STARTSTATIONLATITUDE": "start_station_latitude",
        "STARTSTATIONLONGITUDE": "start_station_longitude",
        "ENDSTATIONNAME": "end_station_name",
        "ENDSTATIONARRONDISSEMENT": "end_station_district",
        "ENDSTATIONLATITUDE": "end_station_latitude",
        "ENDSTATIONLONGITUDE": "end_station_longitude",
        "STARTTIMEMS": "start_time_ms",
        "ENDTIMEMS": "end_time_ms",
    }

    schema = StructType([
        StructField("STARTSTATIONNAME", StringType(), True),
        StructField("STARTSTATIONARRONDISSEMENT", StringType(), True),
        StructField("STARTSTATIONLATITUDE", DoubleType(), True),
        StructField("STARTSTATIONLONGITUDE", DoubleType(), True),
        StructField("ENDSTATIONNAME", StringType(), True),
        StructField("ENDSTATIONARRONDISSEMENT", StringType(), True),
        StructField("ENDSTATIONLATITUDE", DoubleType(), True),
        StructField("ENDSTATIONLONGITUDE", DoubleType(), True),
        StructField("STARTTIMEMS", DoubleType(), True),
        StructField("ENDTIMEMS", DoubleType(), True),
    ])

    raw_df = (
        spark.read.option("pathGlobFilter", "*.csv")
        .option("header", True)
        .schema(schema)
        .option("recursiveFileLookup", "true")
        .csv(bronze_path)
    )

    rides_df = raw_df.select([col(c).alias(french_to_english.get(c, c)) for c in raw_df.columns])
    rides_df = rides_df.withColumn("start_time_ms", to_timestamp(from_unixtime(col("start_time_ms") / 1000)))
    rides_df = rides_df.withColumn("end_time_ms", to_timestamp(from_unixtime(col("end_time_ms") / 1000)))
    rides_df = rides_df.withColumn("ride_year", year(col("start_time_ms")))

    rides_df = rides_df.repartition(200, col("ride_year"))
    write_table(rides_df, rides_stage_path, storage_format, partition_cols=["ride_year"], maxRecordsPerFile=MAXRECORDSPERFILE)

    total_rows = rides_df.count()
    min_start_ms = rides_df.agg({"start_time_ms": "min"}).collect()
    max_end_ms = rides_df.agg({"end_time_ms": "max"}).collect()

    print("=== Stage 1 complete: bronze -> silver rides_stage ===")
    print(f"Storage format: {storage_format}")
    print(f"Rides stage path: {rides_stage_path}")
    print(f"Total rides written: {total_rows:,}")
    print(min_start_ms)
    print(max_end_ms)

    spark.stop()


if __name__ == "__main__":
    main()
