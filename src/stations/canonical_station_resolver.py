"""Canonical station resolver utilities.

Use this module to map raw station inputs (name, lat, lon) to a
`canonical_station_id` using the same staged matching logic as the notebook:
1) exact coordinate key
2) exact normalized name + nearest coordinate
3) nearest canonical station within threshold
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import requests


class CanonicalStationResolver:
    """Resolve raw station inputs to canonical station IDs.

    Example:
        resolver = CanonicalStationResolver(canonical_mapping_df)
        result = resolver.resolve("Clark / Ontario", 45.510494, -73.566921)
        station_id = result["canonical_station_id"]
    """

    @staticmethod
    def load_live_stations(
        info_url: str = "https://gbfs.velobixi.com/gbfs/2-2/en/station_information.json",
        timeout_sec: int = 30,
    ) -> pd.DataFrame:
        """Load and return live BIXI station information as a DataFrame.

        Args:
            info_url: GBFS station information endpoint.
            timeout_sec: HTTP timeout in seconds.

        Returns:
            DataFrame containing live station records (`live_stations_df`).
        """
        response = requests.get(info_url, timeout=timeout_sec)
        response.raise_for_status()
        info_data = response.json()

        live_stations_df = pd.DataFrame(info_data["data"]["stations"]).copy()
        required_live_cols = {"station_id", "name", "capacity", "lat", "lon"}
        missing = required_live_cols - set(live_stations_df.columns)
        if missing:
            raise ValueError(f"Live feed is missing required columns: {sorted(missing)}")

        return live_stations_df

    def __init__(
        self,
        canonical_mapping: pd.DataFrame,
        max_name_coord_km: float = 0.20,
        max_nearest_km: float = 0.05,
    ) -> None:
        """Build lookup indexes from the canonical mapping DataFrame.

        Args:
            canonical_mapping: DataFrame with columns
                canonical_station_id, canonical_lat, canonical_lon,
                normalized_name, coord_key.
            max_name_coord_km: Threshold for name-based nearest coordinate match.
            max_nearest_km: Threshold for nearest canonical fallback match.
        """
        required_cols = {
            "canonical_station_id",
            "canonical_lat",
            "canonical_lon",
            "normalized_name",
            "coord_key",
        }
        missing = required_cols - set(canonical_mapping.columns)
        if missing:
            raise ValueError(f"canonical_mapping is missing required columns: {sorted(missing)}")

        self.max_name_coord_km = float(max_name_coord_km)
        self.max_nearest_km = float(max_nearest_km)

        self.canon_unique_df = (
            canonical_mapping[
                [
                    "canonical_station_id",
                    "canonical_lat",
                    "canonical_lon",
                    "normalized_name",
                    "coord_key",
                ]
            ]
            .dropna(subset=["canonical_station_id", "canonical_lat", "canonical_lon"])
            .drop_duplicates()
            .copy()
        )

        coord_lookup_df = (
            self.canon_unique_df[["coord_key", "canonical_station_id"]]
            .drop_duplicates("coord_key")
            .reset_index(drop=True)
        )
        self.coord_to_station = dict(
            zip(coord_lookup_df["coord_key"], coord_lookup_df["canonical_station_id"])
        )
        self.by_name = {
            key: value.reset_index(drop=True)
            for key, value in self.canon_unique_df.groupby("normalized_name")
        }

        self.canon_station_centers = (
            self.canon_unique_df[["canonical_station_id", "canonical_lat", "canonical_lon"]]
            .drop_duplicates("canonical_station_id")
            .reset_index(drop=True)
        )
        self.center_lats = self.canon_station_centers["canonical_lat"].to_numpy()
        self.center_lons = self.canon_station_centers["canonical_lon"].to_numpy()

    @staticmethod
    def _normalize_station_name(text: str | None) -> str:
        """Normalize station names to improve matching consistency."""
        if text is None:
            return ""
        text = str(text).strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s*/\s*", " / ", text)
        text = re.sub(r"\s*-\s*", "-", text)
        return text.strip()

    @staticmethod
    def _haversine_km_vec(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """Compute vectorized Haversine distance (km)."""
        radius_km = 6371.0
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        return 2 * radius_km * np.arcsin(np.sqrt(a))

    def resolve(self, raw_name: str | None, lat: float, lon: float) -> dict[str, Any]:
        """Resolve one raw station input.

        Args:
            raw_name: Uncleaned station name.
            lat: Station latitude.
            lon: Station longitude.

        Returns:
            A dict containing canonical_station_id, match_method,
            match_distance_km, normalized_name, and coord_key.
        """
        normalized_name = self._normalize_station_name(raw_name)
        coord_key = f"{float(lat):.6f},{float(lon):.6f}"

        station_id = self.coord_to_station.get(coord_key)
        if station_id is not None:
            return {
                "canonical_station_id": station_id,
                "match_method": "exact_coord_key",
                "match_distance_km": 0.0,
                "normalized_name": normalized_name,
                "coord_key": coord_key,
            }

        if normalized_name in self.by_name:
            cand = self.by_name[normalized_name]
            dists = self._haversine_km_vec(
                float(lat),
                float(lon),
                cand["canonical_lat"].to_numpy(),
                cand["canonical_lon"].to_numpy(),
            )
            min_pos = int(np.argmin(dists))
            min_dist = float(dists[min_pos])
            if min_dist <= self.max_name_coord_km:
                return {
                    "canonical_station_id": cand.iloc[min_pos]["canonical_station_id"],
                    "match_method": "exact_normalized_name_nearest_coord",
                    "match_distance_km": min_dist,
                    "normalized_name": normalized_name,
                    "coord_key": coord_key,
                }

        dists = self._haversine_km_vec(float(lat), float(lon), self.center_lats, self.center_lons)
        min_pos = int(np.argmin(dists))
        min_dist = float(dists[min_pos])
        if min_dist <= self.max_nearest_km:
            return {
                "canonical_station_id": self.canon_station_centers.iloc[min_pos]["canonical_station_id"],
                "match_method": "nearest_canonical_within_0.05km",
                "match_distance_km": min_dist,
                "normalized_name": normalized_name,
                "coord_key": coord_key,
            }

        return {
            "canonical_station_id": None,
            "match_method": "unmatched",
            "match_distance_km": np.nan,
            "normalized_name": normalized_name,
            "coord_key": coord_key,
        }

    def resolve_id(self, raw_name: str | None, lat: float, lon: float) -> str | None:
        """Return only the canonical station ID for simple request handlers."""
        return self.resolve(raw_name, lat, lon)["canonical_station_id"]


if __name__ == "__main__":
    print("This module provides the CanonicalStationResolver class for mapping raw station inputs to canonical_station_id.")

    from pathlib import Path
    # Path to mappings
    mapping_parquet_dir = Path("data/silver/station_cleaning/station_direct_match_mapping")
    mapping_csv_dir = Path("data/silver/station_cleaning/station_direct_match_mapping_csv")

    ## VARS
    MAX_NEAREST_KM = 0.05
    MAX_NAME_COORD_KM = 0.20


    if mapping_parquet_dir.exists():
        canonical_mapping_df = pd.read_parquet(mapping_parquet_dir)
        mapping_source = str(mapping_parquet_dir)
    elif mapping_csv_dir.exists():
        csv_parts = sorted(mapping_csv_dir.glob("part-*.csv"))
        if not csv_parts:
            raise FileNotFoundError(f"No part CSV files found in: {mapping_csv_dir}")
        canonical_mapping_df = pd.read_csv(csv_parts[0])
        mapping_source = str(csv_parts[0])
    else:
        raise FileNotFoundError(
            "Missing mapping artifact. Expected one of: "
            f"{mapping_parquet_dir} or {mapping_csv_dir}"
        )

    resolver = CanonicalStationResolver(canonical_mapping=canonical_mapping_df, max_nearest_km=MAX_NEAREST_KM, max_name_coord_km=MAX_NAME_COORD_KM)
    live_stations_df = resolver.load_live_stations()

    sample = live_stations_df.iloc[0]
    sample_result = resolver.resolve(sample['name'], sample['lat'], sample['lon'])
    print('Sample request input:')
    print(sample[['station_id', 'name', 'lat', 'lon']])
    print('\nSample resolver output:')
    print(sample_result)

