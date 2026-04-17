# feature_utils.py

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

def add_misery_index(
    df: DataFrame, 
    temp_col: str = "temperature", 
    precip_col: str = "precipitation",
    output_col: str = "temp_precip_int"
) -> DataFrame:
    """
    Calculates the 'Misery Index' by multiplying temperature and precipitation.
    Captures the exponential drop in demand when it is both cold and wet.
    """
    return df.withColumn(
        output_col, 
        F.col(temp_col) * F.col(precip_col)
    )

def add_temporal_context(
    df: DataFrame, 
    hour_col: str = "hour_of_day", 
    weekend_col: str = "is_weekend",
    output_col: str = "hod_is_weekend_int"
) -> DataFrame:
    """
    Creates an interaction between the hour of the day and the weekend flag.
    Helps the model differentiate a Tuesday morning rush from a Saturday morning.
    """
    # Assuming is_weekend is a 0 or 1 integer. 
    # If it is a boolean, you might need to cast it: F.col(weekend_col).cast("int")
    return df.withColumn(
        output_col, 
        F.col(hour_col) * F.col(weekend_col)
    )

def add_regional_momentum(
    df: DataFrame, 
    regional_demand_col: str = "regional_demand", 
    hour_col: str = "hour_of_day",
    output_col: str = "regional_momentum_hod"
) -> DataFrame:
    """
    Pairs nearby station demand with the cyclical hour to capture systemic network pressure.
    """
    return df.withColumn(
        output_col, 
        F.col(regional_demand_col) * F.col(hour_col)
    )

def build_all_features(df: DataFrame) -> DataFrame:
    """
    A master function to apply all feature engineering steps at once.
    """
    # Using PySpark's transform for clean method chaining
    return (
        df.transform(add_misery_index)
          .transform(add_temporal_context)
          .transform(add_regional_momentum)
    )