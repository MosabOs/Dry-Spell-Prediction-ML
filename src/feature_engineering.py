import numpy as np



def add_lag_and_rolling_features(
    df,
    features,
    lags=(1, 3, 7),
    windows=(7, 14)
):
    """
    Generate lag and rolling mean features from historical observations.

    Temporal calculations are performed separately for each year.
    Rolling means use shifted observations to prevent data leakage.
    """

    data = df.copy()

    # Group observations by year
    year_groups = data["date"].dt.year

    for feature in features:

        # Lag features
        for lag in lags:
            data[f"{feature}_lag_{lag}"] = (
                data.groupby(year_groups)[feature].shift(lag)
            )

        # Rolling mean features
        for window in windows:
            data[f"{feature}_rolling_mean_{window}"] = (
                data.groupby(year_groups)[feature]
                .transform(
                    lambda series: (
                        series.shift(1)
                        .rolling(window=window, min_periods=window)
                        .mean()
                    )
                )
            )

    return data





def add_calendar_features(df):
    """
    Generate calendar-based features from the date column.
    """

    data = df.copy()

    day_of_year = data["date"].dt.day_of_year

    data["month"] = data["date"].dt.month
    data["day_of_year"] = day_of_year

    data["day_of_year_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )

    data["day_of_year_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    return data




def add_domain_informed_features(df):
    """
    Generate domain-informed climate features.

    Features are derived from available climate variables.
    Temporal differences use observations from the same year.
    """

    data = df.copy()

    year_groups = data["date"].dt.year

    # 1. Soil moisture change over the previous 7 days
    data["soil_moisture_change_7d"] = (
        data.groupby(year_groups)["5cm_soli_moist"]
        .diff(7)
    )

    # 2. Potential water deficit change over the previous 7 days
    data["water_deficit_change_7d"] = (
        data.groupby(year_groups)["potential_water_deficit"]
        .diff(7)
    )

    # 3. Temperature and dew point spread
    data["temperature_dewpoint_spread"] = (
        data["max_temp"] - data["mean_dew_point_temp"]
    )

    # 4. SST contrast between Indian Ocean and Cameroon
    data["sst_indian_cameroon_contrast"] = (
        data["sst_indian_ocean_mean_temperature_celsius"]
        - data["sst_cameroon_mean_temperature_celsius"]
    )

    return data
