import os
import json
import re
from collections import defaultdict

from pyspark.sql import Window
from pyspark.sql import functions as F

from sparkutils import get_spark, resolve_data_path, read_table, write_table


"""
Plan (implemented step-by-step)
1) Build normalized station points from silver rides (start + end).
2) Aggregate historical station keys by (coord_key, normalized_name).
3) Build base canonical coordinates by traffic rank.
4) Optionally merge nearby coordinates (<= 0.05 km) for the same normalized name.
5) Build and save final direct-match mapping and canonical summary.
6) If complexity grows, persist intermediates for debugging and reproducibility.

Output
station_direct_match_mapping (parquet): station-level lookup keyed by station_key = coord_key|normalized_name, with canonical assignment (canonical_station_id) plus metadata (cluster_id, observed_trip_count, year range).
station_direct_match_mapping_csv (CSV): same mapping exported for quick inspection/sharing.
station_canonical_summary (parquet): one row per canonical station with rolled-up totals (canonical_total_trips, member coord/name counts, first/last year).
station_coord_cluster (parquet): coordinate-to-cluster mapping (coord_key -> canonical_coord_key) from proximity merging.
intermediate/* (optional): debug artifacts (historical_station_keys, coord_totals, proximity_edges) when SAVE_INTERMEDIATE=1.
"""


def normalize_station_name(column):
    return F.lower(
        F.trim(
            F.regexp_replace(
                F.regexp_replace(
                    F.regexp_replace(column, r"\s+", " "),
                    r"\s*/\s*",
                    " / ",
                ),
                r"\s*-\s*",
                "-",
            )
        )
    )


def normalize_station_name_py(text):
    if text is None:
        return ""
    normalized = str(text).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s*/\s*", " / ", normalized)
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    return normalized.strip()


def parse_manual_station_overrides(raw_json):
    if not raw_json:
        return []

    parsed = json.loads(raw_json)
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("MANUAL_STATION_OVERRIDES_JSON must be a JSON object or array")

    normalized_overrides = []
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Override at index {idx} must be an object")
        from_name = item.get("from_normalized_name")
        to_name = item.get("to_normalized_name")
        if not from_name or not to_name:
            raise ValueError(
                "Each override must include from_normalized_name and to_normalized_name"
            )

        normalized_overrides.append(
            {
                "from_coord_key": item.get("from_coord_key"),
                "from_normalized_name": normalize_station_name_py(from_name),
                "to_normalized_name": normalize_station_name_py(to_name),
                "reason": item.get("reason", "manual_override"),
            }
        )

    return normalized_overrides


def apply_manual_station_overrides(station_points_df, manual_overrides):
    if not manual_overrides:
        return station_points_df, 0

    points_df = station_points_df.withColumn("original_normalized_name", F.col("normalized_name"))
    new_name_col = F.col("normalized_name")

    for override in manual_overrides:
        condition = F.col("normalized_name") == F.lit(override["from_normalized_name"])
        from_coord_key = override.get("from_coord_key")
        if from_coord_key:
            condition = condition & (F.col("coord_key") == F.lit(from_coord_key))

        new_name_col = F.when(condition, F.lit(override["to_normalized_name"]))\
            .otherwise(new_name_col)

    points_df = points_df.withColumn("normalized_name", new_name_col)

    applied_rows = points_df.filter(
        F.col("normalized_name") != F.col("original_normalized_name")
    ).count()

    return points_df.drop("original_normalized_name"), applied_rows


def normalize_station_name_py(text):
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*/\s*", " / ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return text.strip()


def parse_manual_station_overrides(raw_json):
    if not raw_json:
        return []

    parsed = json.loads(raw_json)
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("MANUAL_STATION_OVERRIDES_JSON must be a JSON object or array")

    result = []
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Override at index {idx} must be a JSON object")

        from_name = item.get("from_normalized_name")
        to_name = item.get("to_normalized_name")
        if not from_name or not to_name:
            raise ValueError(
                "Each override must include from_normalized_name and to_normalized_name"
            )

        result.append(
            {
                "from_coord_key": item.get("from_coord_key"),
                "from_normalized_name": normalize_station_name_py(from_name),
                "to_normalized_name": normalize_station_name_py(to_name),
                "reason": item.get("reason", "manual_override"),
            }
        )

    return result


def apply_manual_station_overrides(station_points_df, manual_overrides):
    if not manual_overrides:
        return station_points_df, 0

    points_df = station_points_df.withColumn("_orig_name", F.col("normalized_name"))
    updated_name_col = F.col("normalized_name")

    for override in manual_overrides:
        cond = F.col("normalized_name") == F.lit(override["from_normalized_name"])
        from_coord_key = override.get("from_coord_key")
        if from_coord_key:
            cond = cond & (F.col("coord_key") == F.lit(from_coord_key))

        updated_name_col = F.when(cond, F.lit(override["to_normalized_name"]))\
            .otherwise(updated_name_col)

    points_df = points_df.withColumn("normalized_name", updated_name_col)
    applied_rows = points_df.filter(F.col("normalized_name") != F.col("_orig_name")).count()

    return points_df.drop("_orig_name"), applied_rows


def haversine_km_expr(lat1, lon1, lat2, lon2):
    r = F.lit(6371.0)
    dlat = F.radians(lat2 - lat1)
    dlon = F.radians(lon2 - lon1)
    a = (
        F.pow(F.sin(dlat / 2.0), 2)
        + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) * F.pow(F.sin(dlon / 2.0), 2)
    )
    return 2.0 * r * F.asin(F.sqrt(a))


def build_coord_cluster_mapping(coord_totals_df, proximity_edges_df, max_edge_collect):
    edge_count = proximity_edges_df.count()

    coord_rows = coord_totals_df.select("coord_key", "coord_total_trips").collect()
    coord_trips = {row["coord_key"]: row["coord_total_trips"] for row in coord_rows}

    if edge_count == 0:
        mapping_rows = [
            {
                "coord_key": coord_key,
                "cluster_id": f"cluster_{idx + 1:06d}",
                "canonical_coord_key": coord_key,
                "cluster_size": 1,
            }
            for idx, coord_key in enumerate(sorted(coord_trips.keys()))
        ]
        return mapping_rows, edge_count, False

    if edge_count > max_edge_collect:
        mapping_rows = [
            {
                "coord_key": coord_key,
                "cluster_id": f"cluster_{idx + 1:06d}",
                "canonical_coord_key": coord_key,
                "cluster_size": 1,
            }
            for idx, coord_key in enumerate(sorted(coord_trips.keys()))
        ]
        return mapping_rows, edge_count, True

    edges = proximity_edges_df.select("src_coord_key", "dst_coord_key").collect()

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            if root_a < root_b:
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

    for coord_key in coord_trips.keys():
        parent.setdefault(coord_key, coord_key)

    for row in edges:
        union(row["src_coord_key"], row["dst_coord_key"])

    groups = defaultdict(list)
    for coord_key in coord_trips.keys():
        groups[find(coord_key)].append(coord_key)

    mapping_rows = []
    ordered_group_roots = sorted(groups.keys())
    for group_idx, group_root in enumerate(ordered_group_roots, start=1):
        members = sorted(groups[group_root])
        canonical_coord_key = sorted(
            members,
            key=lambda c: (-coord_trips.get(c, 0), c),
        )[0]
        for coord_key in members:
            mapping_rows.append(
                {
                    "coord_key": coord_key,
                    "cluster_id": f"cluster_{group_idx:06d}",
                    "canonical_coord_key": canonical_coord_key,
                    "cluster_size": len(members),
                }
            )

    return mapping_rows, edge_count, False


def main():
    spark = get_spark()
    
    # Tuning for local mode: reduce shuffle partitions for small data, suppress expected warnings
    spark.conf.set("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "50"))
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    # Spark 4 may block runtime updates to some shuffle configs; keep this best-effort.
    try:
        spark.conf.set("spark.shuffle.sort.bypassMergeThreshold", "200")
    except Exception as exc:
        print(
            "Warning: could not set spark.shuffle.sort.bypassMergeThreshold at runtime "
            f"({exc}). Continuing with Spark default."
        )
    try:
        spark.conf.set("spark.shuffle.sort.bypassMergeThreshold", "200")
        bypass_merge_set = True
    except Exception:
        # Spark 4 may reject runtime changes for this config.
        bypass_merge_set = False

    base_path = resolve_data_path()
    default_format = "parquet"
    storage_format = os.environ.get("SILVER_STORAGE_FORMAT", default_format)

    proximity_km = float(os.environ.get("STATION_PROXIMITY_KM", "0.05"))
    enable_proximity_merge = os.environ.get("ENABLE_PROXIMITY_MERGE", "1") == "1"
    save_intermediate = os.environ.get("SAVE_INTERMEDIATE", "1") == "1"
    max_edge_collect = int(os.environ.get("MAX_EDGE_COLLECT", "500000"))
    enable_manual_station_overrides = os.environ.get("ENABLE_MANUAL_STATION_OVERRIDES", "1") == "1"

    overrides_from_env = parse_manual_station_overrides(
        os.environ.get("MANUAL_STATION_OVERRIDES_JSON", "")
    )

    # Built-in known rename/suffix cases; coord-scoped to avoid over-merge.
    default_manual_overrides = [
        {
            "from_coord_key": "45.532077,-73.575143",
            "from_normalized_name": normalize_station_name_py("marquette / du mont-royal (sud)"),
            "to_normalized_name": normalize_station_name_py("marquette / du mont-royal"),
            "reason": "known_station_suffix_variant_marquette_mont_royal",
        },
        {
            "from_normalized_name": normalize_station_name_py("rivard / du mont-royal"),
            "to_normalized_name": normalize_station_name_py("rivard / mont-royal"),
            "reason": "known_station_variant_rivard_mont_royal",
        },
        {
            "from_normalized_name": normalize_station_name_py("métro laurier (berri / laurier)"),
            "to_normalized_name": normalize_station_name_py("berri / laurier"),
            "reason": "known_station_variant_metro_laurier_berri_laurier",
        },
    ]

    manual_station_overrides = []
    if enable_manual_station_overrides:
        manual_station_overrides.extend(default_manual_overrides)
        manual_station_overrides.extend(overrides_from_env)

    rides_stage_path = f"{base_path}/silver/rides_stage"
    station_cleaning_base = f"{base_path}/silver/station_cleaning"

    mapping_table_path = f"{station_cleaning_base}/station_direct_match_mapping"
    mapping_csv_path = f"{station_cleaning_base}/station_direct_match_mapping_csv"
    canonical_table_path = f"{station_cleaning_base}/station_canonical_summary"
    cluster_table_path = f"{station_cleaning_base}/station_coord_cluster"
    intermediate_base = f"{station_cleaning_base}/intermediate"

    silver_rides_df = read_table(spark, rides_stage_path, storage_format)

    start_points_df = (
        silver_rides_df.filter(
            F.col("start_station_name").isNotNull()
            & F.col("start_station_latitude").isNotNull()
            & F.col("start_station_longitude").isNotNull()
            & F.col("start_time_ms").isNotNull()
        )
        .select(
            F.format_string(
                "%.6f,%.6f",
                F.round(F.col("start_station_latitude"), 6),
                F.round(F.col("start_station_longitude"), 6),
            ).alias("coord_key"),
            F.round(F.col("start_station_latitude"), 6).alias("lat"),
            F.round(F.col("start_station_longitude"), 6).alias("lon"),
            normalize_station_name(F.col("start_station_name")).alias("normalized_name"),
            F.year(F.col("start_time_ms")).alias("trip_year"),
        )
    )

    end_points_df = (
        silver_rides_df.filter(
            F.col("end_station_name").isNotNull()
            & F.col("end_station_latitude").isNotNull()
            & F.col("end_station_longitude").isNotNull()
            & F.col("end_time_ms").isNotNull()
        )
        .select(
            F.format_string(
                "%.6f,%.6f",
                F.round(F.col("end_station_latitude"), 6),
                F.round(F.col("end_station_longitude"), 6),
            ).alias("coord_key"),
            F.round(F.col("end_station_latitude"), 6).alias("lat"),
            F.round(F.col("end_station_longitude"), 6).alias("lon"),
            normalize_station_name(F.col("end_station_name")).alias("normalized_name"),
            F.year(F.col("end_time_ms")).alias("trip_year"),
        )
    )

    station_points_df = start_points_df.unionByName(end_points_df)
    station_points_df, manual_override_applied_rows = apply_manual_station_overrides(
        station_points_df=station_points_df,
        manual_overrides=manual_station_overrides,
    )

    historical_keys_df = (
        station_points_df.filter(F.col("normalized_name").isNotNull())
        .groupBy("coord_key", "lat", "lon", "normalized_name")
        .agg(
            F.count("normalized_name").alias("observed_trip_count"),
            F.min("trip_year").alias("first_year_seen"),
            F.max("trip_year").alias("last_year_seen"),
        )
    )

    coord_totals_df = (
        station_points_df.groupBy("coord_key", "lat", "lon")
        .count()
        .withColumnRenamed("count", "coord_total_trips")
    )

    if save_intermediate:
        write_table(historical_keys_df, f"{intermediate_base}/historical_station_keys", storage_format)
        write_table(coord_totals_df, f"{intermediate_base}/coord_totals", storage_format)

    lat_lon_margin = proximity_km / 111.32

    if enable_proximity_merge:
        left_df = historical_keys_df.select(
            F.col("coord_key").alias("src_coord_key"),
            F.col("lat").alias("src_lat"),
            F.col("lon").alias("src_lon"),
            F.col("normalized_name").alias("normalized_name"),
        )
        right_df = historical_keys_df.select(
            F.col("coord_key").alias("dst_coord_key"),
            F.col("lat").alias("dst_lat"),
            F.col("lon").alias("dst_lon"),
            F.col("normalized_name").alias("normalized_name"),
        )

        candidate_pairs_df = (
            left_df.join(right_df, on="normalized_name", how="inner")
            .where(F.col("src_coord_key") < F.col("dst_coord_key"))
            .where(F.abs(F.col("src_lat") - F.col("dst_lat")) <= F.lit(lat_lon_margin))
            .where(F.abs(F.col("src_lon") - F.col("dst_lon")) <= F.lit(lat_lon_margin))
            .withColumn(
                "distance_km",
                haversine_km_expr(
                    F.col("src_lat"),
                    F.col("src_lon"),
                    F.col("dst_lat"),
                    F.col("dst_lon"),
                ),
            )
            .where(F.col("distance_km") <= F.lit(proximity_km))
            .select("src_coord_key", "dst_coord_key", "normalized_name", "distance_km")
        )

        if save_intermediate:
            write_table(candidate_pairs_df, f"{intermediate_base}/proximity_edges", storage_format)

        coord_cluster_rows, edge_count, fallback_to_identity = build_coord_cluster_mapping(
            coord_totals_df=coord_totals_df,
            proximity_edges_df=candidate_pairs_df,
            max_edge_collect=max_edge_collect,
        )
    else:
        coord_keys = [r["coord_key"] for r in coord_totals_df.select("coord_key").collect()]
        coord_cluster_rows = [
            {
                "coord_key": coord_key,
                "cluster_id": f"cluster_{idx + 1:06d}",
                "canonical_coord_key": coord_key,
                "cluster_size": 1,
            }
            for idx, coord_key in enumerate(sorted(coord_keys))
        ]
        edge_count = 0
        fallback_to_identity = False

    coord_cluster_df = spark.createDataFrame(coord_cluster_rows)

    write_table(coord_cluster_df, cluster_table_path, storage_format)

    canonical_coord_df = (
        coord_cluster_df.join(coord_totals_df, on="coord_key", how="left")
        .groupBy("canonical_coord_key")
        .agg(
            F.sum("coord_total_trips").alias("coord_total_trips"),
            F.max("cluster_size").alias("cluster_size"),
        )
        .join(
            coord_totals_df.select(
                F.col("coord_key").alias("canonical_coord_key"),
                F.col("lat").alias("canonical_lat"),
                F.col("lon").alias("canonical_lon"),
            ),
            on="canonical_coord_key",
            how="left",
        )
    )

    canonical_rank_window = Window.orderBy(F.col("coord_total_trips").desc(), F.col("canonical_coord_key").asc())
    canonical_coord_df = (
        canonical_coord_df.withColumn("station_rank", F.row_number().over(canonical_rank_window))
        .withColumn("canonical_station_id", F.format_string("STN_%04d", F.col("station_rank")))
        .select(
            "canonical_coord_key",
            "canonical_station_id",
            "canonical_lat",
            "canonical_lon",
            "coord_total_trips",
            "cluster_size",
        )
    )

    direct_match_mapping_df = (
        historical_keys_df.join(coord_cluster_df.select("coord_key", "canonical_coord_key", "cluster_id"), on="coord_key", how="left")
        .join(canonical_coord_df, on="canonical_coord_key", how="left")
        .withColumn("station_key", F.concat_ws("|", F.col("coord_key"), F.col("normalized_name")))
        .select(
            "station_key",
            "coord_key",
            "normalized_name",
            "canonical_station_id",
            "canonical_coord_key",
            "canonical_lat",
            "canonical_lon",
            "cluster_id",
            "cluster_size",
            "observed_trip_count",
            "first_year_seen",
            "last_year_seen",
        )
        .dropDuplicates(["station_key"])
    )

    canonical_summary_df = (
        direct_match_mapping_df.groupBy(
            "canonical_station_id",
            "canonical_coord_key",
            "canonical_lat",
            "canonical_lon",
            "cluster_size",
        )
        .agg(
            F.sum("observed_trip_count").alias("canonical_total_trips"),
            F.countDistinct("coord_key").alias("member_coord_count"),
            F.countDistinct("normalized_name").alias("member_name_count"),
            F.min("first_year_seen").alias("first_year_seen"),
            F.max("last_year_seen").alias("last_year_seen"),
        )
    )

    write_table(direct_match_mapping_df, mapping_table_path, storage_format)
    write_table(canonical_summary_df, canonical_table_path, storage_format)
    direct_match_mapping_df.coalesce(1).write.mode("overwrite").option("header", True).csv(mapping_csv_path)

    total_station_keys = direct_match_mapping_df.count()
    unique_canonical = direct_match_mapping_df.select("canonical_station_id").distinct().count()
    unique_coord = coord_totals_df.select("coord_key").distinct().count()
    merged_coord = canonical_coord_df.select("canonical_coord_key").distinct().count()

    print("=== Stage 2 complete: notebook-inspired station direct mapping ===")
    print(f"Storage format: {storage_format}")
    print(f"Rides source path: {rides_stage_path}")
    print(f"Bypass merge threshold set at runtime: {bypass_merge_set}")
    print(f"Proximity merge enabled: {enable_proximity_merge}")
    print(f"Manual station overrides enabled: {enable_manual_station_overrides}")
    print(f"Manual station override rules loaded: {len(manual_station_overrides)}")
    print(f"Rows affected by manual overrides: {manual_override_applied_rows:,}")
    print(f"Proximity threshold (km): {proximity_km}")
    print(f"Proximity edge count: {edge_count:,}")
    print(f"Fallback to identity clustering: {fallback_to_identity}")
    print(f"Unique coordinates (raw): {unique_coord:,}")
    print(f"Unique canonical coordinates (after merge): {merged_coord:,}")
    print(f"Station key rows: {total_station_keys:,}")
    print(f"Unique canonical stations: {unique_canonical:,}")
    print(f"Direct mapping table path: {mapping_table_path}")
    print(f"Direct mapping CSV path: {mapping_csv_path}")
    print(f"Canonical summary path: {canonical_table_path}")
    print(f"Cluster mapping path: {cluster_table_path}")
    if save_intermediate:
        print(f"Intermediate base path: {intermediate_base}")

    validation_cases = [
        ("metro_mont_royal", "45.524420,-73.581663", "45.524353,-73.581432"),
        ("marquette_mont_royal", "45.532077,-73.575143", "45.532218,-73.575431"),
    ]

    print("Validation checks for known manual-merge cases:")
    for case_name, src_coord, dst_coord in validation_cases:
        case_df = (
            direct_match_mapping_df
            .filter(F.col("coord_key").isin([src_coord, dst_coord]))
            .select("coord_key", "normalized_name", "canonical_station_id", "cluster_id", "observed_trip_count")
        )

        rows = case_df.collect()
        if not rows:
            print(f"- {case_name}: no rows found for coords {src_coord}, {dst_coord}")
            continue

        coord_count = len({r["coord_key"] for r in rows})
        canonical_ids = sorted({r["canonical_station_id"] for r in rows if r["canonical_station_id"]})
        merged_ok = len(canonical_ids) == 1 and coord_count >= 2

        print(f"- {case_name}: coords={coord_count}, canonical_ids={canonical_ids}, merged={merged_ok}")
        case_df.orderBy("coord_key", "normalized_name").show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
