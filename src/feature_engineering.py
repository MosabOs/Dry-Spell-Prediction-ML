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

    day_of_year = data["date"].dt.dayofyear

    data["month"] = data["date"].dt.month
    data["day_of_year"] = day_of_year

    data["day_of_year_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )

    data["day_of_year_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    return data
