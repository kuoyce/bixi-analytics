# BIXI Analytics - Data Analysis Strategy
## Optimized for Parquet Format with DuckDB

---

## 📊 Data Import Strategy

### Current Data State
- **Format**: Parquet (3 files with optimized compression)
- **Column Names**: English-translated (e.g., `start_station_name`, `start_time_ms`)
- **Data Types**: 
  - String columns: `object` (station names, districts)
  - Coordinates: `float32` (reduced precision for 1m accuracy)
  - Timestamps: `int64` (milliseconds since epoch)
- **Total Size**: ~600-800 MB (versus ~5 GB CSV)
- **Rows**: ~27.5 million total

### Import Methods

#### Option A: Pandas (Simple Overview)
```python
import pandas as pd
from pathlib import Path

parquet_files = sorted(Path('data').rglob('*.parquet'))
df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
```
**Best for**: Quick exploration, small subsets, data quality checks

#### Option B: DuckDB (Full Dataset Analysis) ⭐ RECOMMENDED
```python
import duckdb

# Query directly from Parquet without loading to memory
result = duckdb.sql("""
    SELECT * FROM read_parquet('data/**/*.parquet')
""")
```
**Best for**: Aggregations, filtering, temporal analysis, geographic analysis

#### Option C: DuckDB + Pandas Hybrid (Recommended)
```python
import duckdb
import pandas as pd

# Use DuckDB for heavy lifting, convert to pandas for visualization
result_df = duckdb.sql("""
    SELECT * FROM read_parquet('data/**/*.parquet')
    WHERE condition
""").to_df()
```

---

## 🎯 Analysis Breakdown with DuckDB Benefits

### 1. **Data Overview** 
**DuckDB Value**: ⭐ Low | **Use**: Pandas
- Query schema directly from Parquet
- Check row counts with SQL `COUNT(*)`
- Minimal data movement needed

```python
# Quick overview with DuckDB
duckdb.sql("SELECT COUNT(*) as row_count FROM read_parquet('data/**/*.parquet')").show()
duckdb.sql("DESCRIBE read_parquet('data/**/*.parquet')").show()
```

---

### 2. **Data Quality Assessment**
**DuckDB Value**: ⭐⭐ Medium | **Use**: Pandas for aggregation
- Missing values: Easy with pandas `.isna()`
- Duplicates: Can use SQL or pandas
- Data type consistency: Check with pandas schema

```python
# Check duplicates with DuckDB (efficient)
duckdb.sql("""
    SELECT COUNT(*) as total_rows,
           COUNT(DISTINCT *) as unique_rows
    FROM read_parquet('data/**/*.parquet')
""").show()
```

#### ⚠️ **CRITICAL FINDING: Station Name Inconsistencies (Year-Based)**

**Discovery (updated from workbook)**: Station cleaning analysis found **179 coordinate locations with multiple station names** at identical GPS coordinates, affecting **3,281,706 trips**. The root cause remains **progressive data standardization across years (2024-2026)** plus naming convention drift.

**Examples**:
- **Location (45.520626, -73.575951)**:
  - 2024: "Duluth  / St-Denis" (extra space) - 48,541 trips
  - 2025: Mixed naming - 48,472 trips
  - 2026: "Duluth / St-Denis" (standardized) - 781 trips
  
- **Location (45.525513, -73.574242)**:
  - 2024: "du Parc-Lafontaine / Rachel" - 42,267 trips
  - 2025: Changed to "du Parc-La Fontaine / Rachel" - 35,464 trips
  - 2026: Continues new format - 539 trips

- **Parc Jean-Drapeau (Multiple locations)**:
  - "Chemin Macdonald" vs "Chemin MacDonald" (capitalization)
  - Plus "Chemin du Tour-de-l'Isle" at different GPS coordinates

**Impact on Analysis**:
- Top 50 routes may show self-loops (same station → same station) due to name variation
- Station popularity metrics will be fragmented across naming variants
- Network analysis must account for these variations

**Recommended Solution**:
Use GPS coordinates (latitude/longitude) as the canonical station identifier instead of station names. Group trips by coordinate pairs before analysis to consolidate all naming variants.

**SQL Example for Consolidation**:
```sql
-- View stations by coordinates (consolidates all name variations)
SELECT 
    start_station_latitude,
    start_station_longitude,
    COUNT(DISTINCT start_station_name) as num_name_variants,
    COUNT(*) as total_trips
FROM read_parquet('data/**/*.parquet')
GROUP BY start_station_latitude, start_station_longitude
HAVING COUNT(DISTINCT start_station_name) > 1
ORDER BY total_trips DESC
```

---

### 3. **Temporal Analysis** 🚀
**DuckDB Value**: ⭐⭐⭐⭐⭐ VERY HIGH | **Use**: DuckDB

**Why DuckDB Excels**:
- SQL's date functions are optimized for timestamp operations
- Extract hour, day of week, month without loading full data
- Aggregations (GROUP BY hour/day) are vectorized
- Result is small → easy to pandas for visualization

```python
# Extract temporal features in DuckDB
result = duckdb.sql("""
    SELECT 
        HOUR(to_timestamp(start_time_ms/1000.0)) as hour_of_day,
        DAYOFWEEK(to_timestamp(start_time_ms/1000.0)) as day_of_week,
        COUNT(*) as trip_count,
        AVG(end_time_ms - start_time_ms) as avg_trip_duration_ms
    FROM read_parquet('data/**/*.parquet')
    GROUP BY hour_of_day, day_of_week
    ORDER BY hour_of_day, day_of_week
""").to_df()

# Visualize in pandas
import matplotlib.pyplot as plt
result.pivot_table(index='hour_of_day', columns='day_of_week', values='trip_count').plot()
```

**Analysis Tasks**:
1. Distribution by hour of day → SQL GROUP BY hour
2. Distribution by day of week → SQL DAYOFWEEK extraction
3. Busiest times → SQL ORDER BY with LIMIT
4. Temporal trends → SQL window functions for trend calculation

---

### 4. **Geographic Analysis** 🚀
**DuckDB Value**: ⭐⭐⭐⭐⭐ VERY HIGH | **Use**: DuckDB

**Why DuckDB Excels**:
- Large GROUP BY operations on strings (stations, districts)
- Fast aggregations for popularity metrics
- CASE statements for intra/inter-district classification
- Result sets are small (distinct stations < 1000) → easy to pandas

```python
# Most popular stations (massive aggregation in DuckDB)
result = duckdb.sql("""
    SELECT 
        start_station_name,
        start_station_district,
        COUNT(*) as start_count
    FROM read_parquet('data/**/*.parquet')
    GROUP BY start_station_name, start_station_district
    ORDER BY start_count DESC
    LIMIT 50
""").to_df()

# Intra-district vs inter-district trips
result = duckdb.sql("""
    SELECT 
        CASE WHEN start_station_district = end_station_district 
             THEN 'intra_district' 
             ELSE 'inter_district' 
        END as trip_type,
        COUNT(*) as trip_count
    FROM read_parquet('data/**/*.parquet')
    GROUP BY trip_type
""").to_df()
```

**Analysis Tasks**:
1. Top 50 start/end stations → SQL with COUNT(*) GROUP BY
2. District distribution → SQL GROUP BY district
3. Intra vs inter-district → SQL CASE statement
4. Station activity heatmap → SQL pivot result

---

### 5. **Trip Duration Analysis**
**DuckDB Value**: ⭐⭐⭐ High | **Use**: DuckDB for aggregation, Pandas for outlier detection

**Trip Duration**: Calculated as `(end_time_ms - start_time_ms)` in milliseconds

```python
# Duration statistics with DuckDB
result = duckdb.sql("""
    SELECT 
        MIN(end_time_ms - start_time_ms) as min_duration_ms,
        MAX(end_time_ms - start_time_ms) as max_duration_ms,
        AVG(end_time_ms - start_time_ms) as avg_duration_ms,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY end_time_ms - start_time_ms) as median_duration_ms,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY end_time_ms - start_time_ms) as p75_duration_ms,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY end_time_ms - start_time_ms) as p95_duration_ms
    FROM read_parquet('data/**/*.parquet')
""").to_df()

# For distribution and outlier detection, bring to pandas
df_duration = duckdb.sql("""
    SELECT end_time_ms - start_time_ms as duration_ms
    FROM read_parquet('data/**/*.parquet')
""").to_df()

# Detect outliers (IQR method in pandas)
Q1 = df_duration['duration_ms'].quantile(0.25)
Q3 = df_duration['duration_ms'].quantile(0.75)
IQR = Q3 - Q1
outliers = df_duration[(df_duration['duration_ms'] < Q1 - 1.5*IQR) | 
                        (df_duration['duration_ms'] > Q3 + 1.5*IQR)]
```

**Analysis Tasks**:
1. Min/max/mean/median statistics → SQL aggregation functions
2. Percentiles (p75, p95) → SQL PERCENTILE_CONT
3. Distribution histogram → Bring subset to pandas
4. Outliers → Calculate thresholds in SQL, identify in pandas

---

### 6. **Station Analysis (Network)** 🚀
**DuckDB Value**: ⭐⭐⭐⭐⭐ VERY HIGH | **Use**: DuckDB

**Why DuckDB Excels**:
- Complex JOIN operations for route analysis
- Multiple aggregations in single query
- Popular routes involve two GROUP BY dimensions (start + end)

```python
# Most popular routes (start → end station pairs)
routes = duckdb.sql("""
    SELECT 
        start_station_name,
        end_station_name,
        COUNT(*) as route_count
    FROM read_parquet('data/**/*.parquet')
    GROUP BY start_station_name, end_station_name
    ORDER BY route_count DESC
    LIMIT 100
""").to_df()

# Station connectivity (how many unique destinations from each station)
connectivity = duckdb.sql("""
    SELECT 
        start_station_name,
        COUNT(DISTINCT end_station_name) as destinations,
        COUNT(*) as total_trips
    FROM read_parquet('data/**/*.parquet')
    GROUP BY start_station_name
    ORDER BY destinations DESC
""").to_df()

# Catchment areas by district
catchment = duckdb.sql("""
    SELECT 
        start_station_district,
        end_station_district,
        COUNT(*) as cross_district_trips
    FROM read_parquet('data/**/*.parquet')
    GROUP BY start_station_district, end_station_district
    ORDER BY cross_district_trips DESC
""").to_df()
```

**Analysis Tasks**:
1. Top 100 routes → SQL with two-column GROUP BY
2. Station popularity → SQL COUNT(*) with single column GROUP BY
3. Network connectivity → SQL COUNT(DISTINCT) + GROUP BY
4. District connections → SQL pivot/crosstab

---

### 7. **Distance Analysis**
**DuckDB Value**: ⭐⭐ Medium | **Use**: Python (pandas/numpy) for haversine calculation, DuckDB for aggregation

**Why Medium**:
- Haversine distance is mathematical, not SQL-native
- Can be done in DuckDB but more complex
- Better to compute in Python, then aggregate

```python
import numpy as np
import pandas as pd

# Fetch coordinates from Parquet
coords = duckdb.sql("""
    SELECT 
        start_station_latitude,
        start_station_longitude,
        end_station_latitude,
        end_station_longitude
    FROM read_parquet('data/**/*.parquet')
""").to_df()

# Haversine distance function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

coords['distance_km'] = haversine(
    coords['start_station_latitude'],
    coords['start_station_longitude'],
    coords['end_station_latitude'],
    coords['end_station_longitude']
)

# Distance statistics
print(coords['distance_km'].describe())

# Correlation with duration (from earlier analysis)
distance_duration = coords.merge(duration_df, left_index=True, right_index=True)
correlation = distance_duration['distance_km'].corr(distance_duration['duration_ms'])
```

**Analysis Tasks**:
1. Calculate distances → Python haversine
2. Avg distance by district → SQL GROUP BY after adding distance column
3. Distance vs duration correlation → Pandas/numpy
4. Outlier distances → Python numpy operations

---

### 8. **Summary Insights**
**DuckDB Value**: ⭐⭐⭐ High | **Use**: Mix of DuckDB and Pandas

```python
# Key metrics collection
summary = duckdb.sql("""
    SELECT 
        COUNT(*) as total_trips,
        COUNT(DISTINCT start_station_name) as num_start_stations,
        COUNT(DISTINCT end_station_name) as num_end_stations,
        COUNT(DISTINCT start_station_district) as num_districts,
        MIN(to_timestamp(start_time_ms/1000.0)) as earliest_trip,
        MAX(to_timestamp(end_time_ms/1000.0)) as latest_trip
    FROM read_parquet('data/**/*.parquet')
""").to_df()
```

---

## 📈 Recommended Analysis Workflow

### Phase 1: Quick Exploration (5 min)
```python
import duckdb

# Overview
duckdb.sql("SELECT COUNT(*) FROM read_parquet('data/**/*.parquet')").show()

# Sample data
duckdb.sql("SELECT * FROM read_parquet('data/**/*.parquet') LIMIT 5").show()
```

### Phase 2: DuckDB-Heavy Analysis (15-20 min)
1. Temporal analysis (hour/day distributions)
2. Geographic analysis (station popularity)
3. Station network analysis (routes, connectivity)
4. Duration statistics

### Phase 3: Historical Station Rider Patterns (15-20 min)
**Main goal**: Identify station-level historical demand patterns, velocity (in+out), and dead-zone periods.

1. **Top most frequently used stations**
    - Rank stations by total activity: `starts + ends`
    - Produce daily/weekly/monthly top-N leaderboards
    - Compare station rank stability over time (persistent hubs vs seasonal spikes)

2. **Highest velocity stations (in + out flow)**
    - Compute hourly `in_count`, `out_count`, and `velocity = in_count + out_count`
    - Surface peak-velocity windows by station and by district
    - Flag highly imbalanced stations (`abs(in_count - out_count)` high) for rebalancing signals

3. **Dead zones / dead periods detection**
    - Define dead period candidates as sudden drops to near-zero movement during normally active windows
    - Detect two scenarios:
      - probable full station: high inbound pressure then no arrivals/departures
      - probable empty station: high outbound demand historically then no departures
    - Quantify dead-zone risk score per station-hour (frequency and duration of events)

4. **Suggested exploratory items**
    - Station seasonality profile (month-over-month activity heatmap)
    - Weekday vs weekend behavior by station cluster
    - Rolling trend shifts (7-day / 30-day moving averages)
    - Weather/event sensitivity overlay (if external data is added)
    - Early warning candidates: stations with repeated high-velocity + dead-period combinations

### Phase 4: Python Computation (10-15 min)
1. Haversine distance calculations
2. Outlier detection
3. Correlation analysis

### Phase 5: Visualization (10 min)
- Bring results to pandas/matplotlib/plotly
- Create heatmaps, histograms, scatter plots

---

## 🚀 Performance Benefits

| Analysis | Time (CSV) | Time (Parquet + DuckDB) | Speedup |
|----------|-----------|------------------------|---------|
| Top 100 routes | ~45s | ~2s | **22.5x** |
| Hourly distribution | ~50s | ~1.5s | **33x** |
| Missing values check | ~30s | ~0.5s | **60x** |
| Duration percentiles | ~60s | ~1s | **60x** |
| Memory usage | ~10 GB | ~500 MB | **20x** |

---

## 💾 Data Location
- **Parquet files**: `./bixi-analytics/data/`
- **Column names**: All translated to English (see mapping in cell 5)
- **Ready to use**: Yes, conversion completed successfully

---

## 🔧 Setup Required
```bash
# Already installed
# pip install duckdb pandas numpy
```

---

## � Phase 2 Analysis Results (Completed)

### Key Findings from DuckDB-Heavy Analysis:

**Dataset Overview:**
- Total trips: **27,588,048** (2024-2026)
- Stations: ~100+ unique locations (across name variations)
- Temporal range: January 1, 2024 → February 1, 2026
- Data quality: **Zero NULL values** ✓

**Temporal Patterns:**
- **Peak hours**: 5-6 AM (450K-468K trips) - overnight/early morning commute
- **Evening peak**: 7-8 PM (~300K trips/hour)
- **Lowest activity**: 2-4 PM (~8K-14K trips/hour)
- **Weekly pattern**: Weekday commute peaks > weekend leisure trips

**Geographic Insights:**
- **76% intra-district trips** (same district start/end)
- **24% inter-district trips** (cross-district)
- **Top districts**: Ville-Marie, Le Plateau-Mont-Royal, Rosemont-La Petite-Patrie
- **Most connected hub**: Multiple hubs with 100+ unique destinations

**Trip Duration:**
- **Median**: 10.4 minutes
- **Mean**: 16.8 minutes
- **IQR**: 6.0-17.8 minutes
- **Outliers**: 5.1% of trips exceed 35.5 minutes (mostly legitimate longer trips)

**Trip Validity Analysis - Enhanced:**

*Short Trips (<2 minutes):*
- **Count**: 825,347 trips (3.0% of dataset)
- **Unique routes**: 16,570
- **Same-station returns**: 458,546 trips (55.6%) - concentrated at median 21.8 seconds, likely user errors or bike tests
- **Different-station trips**: 366,801 trips (44.4%) - wider duration range, mixed validity

*Long Trips (>12 hours):*
- **Count**: 17,678 trips (0.06% of dataset) 
- **Unique routes**: 15,778
- **Same-station returns**: 1,068 trips (6.0%) - likely abandoned bikes
- **Different-station trips**: 16,610 trips (94.0%) - concentrated 12-40 hours (median 22.5h, mean 68.8h), indicating lost/stolen bikes en route to recovery center
- **Max duration**: 6,187 hours (~258 days) - data quality flag

### ⚠️ Data Quality Issues Discovered:

**1. Station Name Inconsistencies (Year-Based)**
- 20 GPS locations have 2-3 different station names
- Caused by progressive standardization: 2024 → 2025 → 2026
- Examples: spacing inconsistencies ("Duluth  / St-Denis" vs "Duluth / St-Denis"), hyphenation changes, naming convention updates
- **Impact**: Top routes may show self-loops (fragmented by naming), Station popularity metrics need consolidation

**2. Anomalous "cyclo" Station (Loss/Theft Recovery Center)**
- Special endpoint capturing 2,584 trips (0.009% of dataset)
- **Direction split**: 99.3% destination (2,567 trips TO cyclo), 0.7% origin (17 trips FROM cyclo)
- **Duration pattern**: Strong concentration 12-40 hours, median 23 hours
- **Interpretation**: Almost certainly a loss/theft recovery center, maintenance facility, or system error catch-all for lost/abandoned bikes
- **Action needed**: Filter from network/route analysis or mark as special status; do not include in station popularity metrics
- **Example long route**: Métro Henri-Bourassa → cyclo (653 hours avg), multiple similar patterns from different stations

**3. Trip Validity Concerns (3.7% of trips)**
- **Short anomalies** (<2 min, 3.0%): Mostly same-station rapid returns (55.6%), likely user errors or bike tests; different-station variants show mixed validity
- **Long anomalies** (>12 hours, 0.06%): Bimodal pattern - same-station returns (6%) indicate abandoned bikes; different-station majority (94%) suggest legitimate extended journeys or lost bikes in transit to recovery

**4. Recommendations for Phase 4**
- Use `(start_station_latitude, start_station_longitude)` as canonical station ID instead of station names
- Create exclusion filter for "cyclo" station in network/route analyses
- Create dimension table: `station_id → all_name_variants` with year-based tracking
- Distance calculations: Use coordinate pairs with haversine formula
- Network analysis: Aggregate by coordinates; apply trip validity thresholds (e.g., exclude <2 min same-station trips, flag >12h trips)
- Data lineage: Document transformation from original_name → canonical_name for auditability

---

## 📌 Key Takeaways & Findings

### Data Quality Summary
- **Overall quality**: High - less than 0.5% of rows contains nulls , consistent data types, parquet compression validated
- **Anomaly scope**: 3.7% of trips flagged for potential validity issues (short <2min or long >12h)
- **Usable data**: ~96.3% of trips appear normal and suitable for analysis

### Trip Behavior Insights
1. **Normal Commute Patterns**: Median trip 10.4 min, concentrated in commute hours (5-6 AM, 7-8 PM)
2. **District Structure**: Strong intra-district preference (76%), suggesting localized micromobility demand
3. **Network Asymmetry**: Standard routes show clear origin-destination patterns; stations with 100+ unique destinations indicate major transit hubs
4. **High-Duration Trips**: 0.06% with >12 hour durations correlate with recovery/maintenance operations, not end-user behavior
5. **Rapid-Return Pattern**: 55.6% of sub-2-minute trips are same-station, indicating testing or immediate undocking

### Data Structure Readiness
- GPS coordinates are reliable for station consolidation (20 naming variants at identical coordinates)
- Year-based data quality trend improves from 2024→2025→2026 (standardization in progress)
- Time data precision (milliseconds) supports second-level analysis
- District assignments are consistent and useful for geographic segmentation

### Station Cleaning Workbook Results (01_dataeng_stations.ipynb)

**Task 1 — Identical-coordinate name variants:**
- Multi-name coordinate locations: **179**
- Trips affected: **3,281,706**

**Task 2 — Name normalization impact:**
- Raw multi-name locations: **179**
- Post-normalization multi-name locations: **141**
- Fully resolved by normalization: **38**
- Total variants reduced: **371 → 329**

**Task 2b — Fuzzy matching and nearby-name analysis:**
- Combined fuzzy methods (`ratio`, `partial_ratio`, token overlap) validated and debugged
- Similar-name pairs at different coordinates: **2,667**
- Distance buckets: **336 (<=0.2km), 794 (0.2–1.0km), 1,537 (>1.0km)**

**0.05km grouping rule applied:**
- Pairs within 0.05km: **115**
- Grouped normalized names: **192**
- Station groups: **87**

**Task 3 — Canonical mapping table built:**
- Raw observed station variants (name+coord): **1,922**
- Unique coordinates before canonicalization: **1,730**
- Canonical stations after mapping: **1,577**
- Mapping rows assigned: **1,922**

**Task 4 — Validation completed:**
- Coordinate reduction: **153** (**8.84%**)
- Coordinate-to-multiple-canonical-ID conflicts: **0**
- Unmapped trips: **148 / 27,462,260** (**0.0005%**)
- Raw-name self-loops: **1,068,037**
- Canonical self-loops: **1,073,149**
- Canonical self-loops are slightly higher (+5,112), indicating consolidation captures additional same-station returns that raw naming previously fragmented.
- Remaining unresolved name conflicts: **226 normalized names mapping to multiple canonical IDs**

### Next Phase Priorities
1. **Operationalize canonical mapping in pipeline** (PySpark, CSV input) with versioned mapping outputs
2. **Resolve remaining 226 multi-canonical normalized-name conflicts** using rule tables and manual curation loop
3. **Apply validity filters** based on trip duration and direction patterns (exclude/flag anomalous records)
4. **Implement "cyclo" special handling** (exclude from standard metrics, track separately for loss/recovery analysis)
5. **Re-run route/network KPIs with canonical IDs** and compare to pre-cleaning baselines

---

## 🔧 Station Name Standardization Status

### Problem Statement
Multiple station names exist for identical GPS coordinates (20+ locations identified), causing:
- Fragmented route analysis (self-loops in network)
- Inflated station count metrics
- Year-based naming inconsistencies (2024 vs 2025 vs 2026 data quality)

### Completion Status (Workbook)
- ✅ Step 1 completed: Identical-coordinate variants identified and quantified.
- ✅ Step 2 completed: Similar-name nearby-coordinate matching implemented with fuzzy + proximity logic.
- ✅ Step 3 completed: Same-name multi-coordinate conflicts surfaced through Task 4 conflict analysis.
- ✅ Canonical mapping table produced: `station_canonical_mapping_df` and `station_canonical_summary_df`.
- ✅ Validation completed: consolidation, coverage, self-loop comparison, and conflict reporting.

### Standardization Algorithm

#### Step 1: Identify Identical Coordinates
```python
# Find locations with multiple names at exact same lat/long
SELECT 
    start_station_latitude,
    start_station_longitude,
    COUNT(DISTINCT start_station_name) as num_variants,
    ARRAY_AGG(DISTINCT start_station_name) as all_names
FROM read_parquet('data/**/*.parquet')
GROUP BY start_station_latitude, start_station_longitude
HAVING COUNT(DISTINCT start_station_name) > 1
```

**Action**: For identical coordinates, calculate string similarity (Levenshtein/Jaro-Winkler) and flag as duplicates if similarity > 85%

#### Step 2: Identify Close Coordinates
```python
# Find nearby stations with similar names (within tolerance)
# tolerance = user_defined (default: 50 meters / 0.0004 degrees)
SELECT 
    s1.start_station_name,
    s2.start_station_name,
    SQRT(POW(s1.start_station_latitude - s2.start_station_latitude, 2) + 
         POW(s1.start_station_longitude - s2.start_station_longitude, 2)) as distance
FROM stations s1
JOIN stations s2 ON s1.station_id < s2.station_id
WHERE distance < tolerance
```

**Action**: If names are similar (>80% match) AND within spatial tolerance, flag as potential duplicates

#### Step 3: Identical Names, Check Distance
```python
# Find stations with same name but different coordinates
SELECT 
    start_station_name,
    COUNT(DISTINCT (start_station_latitude, start_station_longitude)) as num_locations,
    ARRAY_AGG(DISTINCT (start_station_latitude, start_station_longitude)) as all_coords
FROM read_parquet('data/**/*.parquet')
GROUP BY start_station_name
HAVING COUNT(DISTINCT (start_station_latitude, start_station_longitude)) > 1
```

**Action**: Calculate distance between locations. If distance exceeds tolerance, they are separate stations. If within tolerance, flag as data quality issue.

### Implementation Details

**Similarity Algorithms**:
- Exact match: 100%
- Levenshtein distance: For character-level differences (spacing, accents)
- Jaro-Winkler: For transpositions and partial matches
- Use threshold: 85% for identical coordinates, 80% for nearby

**Distance Tolerance** (User Configurable):
- Default: 50 meters (~0.0004 degrees at Montreal latitude)
- Options: 30m (precise locations), 100m (neighborhood-level grouping)
- Apply Haversine formula for accurate distances

**Output**: Station Canonical Mapping
```python
{
    "canonical_station_id": "STATION_001",
    "canonical_name": "Duluth / St-Denis",
    "canonical_lat": 45.520626,
    "canonical_lon": -73.575951,
    "name_variants": [
        "Duluth  / St-Denis",  # extra space (2024)
        "Duluth / St-Denis"    # standardized (2025-2026)
    ],
    "variant_trip_counts": {
        "Duluth  / St-Denis": 93760,
        "Duluth / St-Denis": 4034
    },
    "data_quality_year_notes": "Name standardization occurred between 2024-2025"
}
```

### Remaining Implementation Work
1. Export canonical outputs (`station_canonical_mapping`, summary, conflicts) as governed artifacts.
2. Integrate mapping logic into production PySpark pipeline over CSV input.
3. Create conflict-resolution rule table (approved merges/splits with effective dates).
4. Re-aggregate downstream route/popularity/network metrics on canonical IDs.
5. Add lineage + QA checkpoints per batch run.

### Success Criteria
- △ Reduced naming-fragmented loops and improved station identity consistency (validated)
- ✓ Coordinate-to-canonical mapping is deterministic and conflict-free at coordinate level
- ✓ Coverage is near-complete (unmapped trip rate ~0.0005%)
- ✓ Standardization outputs are ready for pipeline operationalization

---
## �📝 Next Steps
1. Convert notebook logic into a scheduled PySpark CSV pipeline (bronze/silver/gold).
2. Productionize canonical station mapping as a versioned dimension table.
3. Implement conflict triage workflow for the 226 unresolved normalized names.
4. Recompute network and route analytics using canonical station IDs.
5. Publish QA dashboard (coverage, conflict count, station reduction, loop deltas) per run.
