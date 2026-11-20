
from pathlib import Path

import pandas as pd


def inspect_file(file_path, dataset_type, chunksize=100_000):
    """
    Inspect a CSV file in chunks without loading the entire file
    into memory at once.
    """

    file_path = Path(file_path)

    total_rows = 0
    missing_counts = None
    column_names = None
    data_types = None
    preview = None

    for chunk in pd.read_csv(file_path, chunksize=chunksize):

        if column_names is None:
            column_names = chunk.columns.tolist()
            data_types = chunk.dtypes.astype(str).to_dict()
            missing_counts = pd.Series(0, index=column_names)
            preview = chunk.head(3)

        total_rows += len(chunk)
        missing_counts += chunk.isnull().sum()

    return {
        "Dataset": dataset_type,
        "File": file_path.name,
        "Rows": total_rows,
        "Columns": len(column_names),
        "Column Names": column_names,
        "Data Types": data_types,
        "Missing Values": missing_counts.to_dict(),
        "Preview": preview
    }


def build_dataset_inventory(train_dir, test_dir, chunksize=100_000):
    """
    Inspect all CSV files in the training and testing directories.
    """

    dataset_inventory = []

    for file_path in sorted(Path(train_dir).glob("*.csv")):
        dataset_inventory.append(
            inspect_file(file_path, "Train", chunksize)
        )

    for file_path in sorted(Path(test_dir).glob("*.csv")):
        dataset_inventory.append(
            inspect_file(file_path, "Test", chunksize)
        )

    return dataset_inventory





def analyze_date_alignment(directory, dataset_name):
    """
    Analyze date alignment across all CSV files in a directory.

    Returns:
        A dictionary containing:
        - all_dates: Union of dates across all files.
        - common_dates: Intersection of dates across all files.
        - summary: Alignment summary DataFrame.
        - excluded_dates: Dates not shared by every file.
    """

    directory = Path(directory)
    csv_files = sorted(directory.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {directory}")

    date_sets = {}

    for file_path in csv_files:

        df = pd.read_csv(file_path, usecols=["date"])

        dates = pd.to_datetime(
            df["date"],
            errors="coerce"
        ).dropna()

        date_sets[file_path.name] = set(dates)

    # Union: all dates appearing in at least one file
    all_dates = set.union(*date_sets.values())

    # Intersection: dates shared by every file
    common_dates = set.intersection(*date_sets.values())

    # Compare each file against the union
    alignment_summary = []

    for file_name, dates in date_sets.items():

        missing_dates = all_dates - dates

        alignment_summary.append({
            "Dataset": dataset_name,
            "File": file_name,
            "Available Dates": len(dates),
            "Missing Compared to Union": len(missing_dates),
            "Common Dates": len(dates & common_dates)
        })

    summary_df = pd.DataFrame(alignment_summary)

    excluded_dates = sorted(all_dates - common_dates)

    return {
        "all_dates": all_dates,
        "common_dates": common_dates,
        "summary": summary_df,
        "excluded_dates": excluded_dates
    }




def build_date_availability_matrix(directory):
    """
    Build a date availability matrix for all CSV files
    in a directory.

    Rows represent unique dates.
    Columns represent CSV files.
    1 indicates date availability.
    0 indicates date absence.
    """

    directory = Path(directory)
    csv_files = sorted(directory.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {directory}")

    date_data = {}

    for file_path in csv_files:

        df = pd.read_csv(file_path, usecols=["date"])

        dates = pd.to_datetime(
            df["date"],
            errors="coerce"
        ).dropna()

        date_data[file_path.stem] = set(dates)

    # Build a complete date index from the union
    all_dates = sorted(set.union(*date_data.values()))

    availability = pd.DataFrame(index=all_dates)

    for file_name, dates in date_data.items():

        availability[file_name] = [
            int(date in dates)
            for date in all_dates
        ]

    availability.index.name = "date"

    return availability




def inspect_dataset_columns(directory, dataset_name):
    """
    Inspect column names and data types across CSV files
    in a directory.

    Reads only the first five rows of each file.
    """

    directory = Path(directory)
    csv_files = sorted(directory.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {directory}")

    column_records = []

    for file_path in csv_files:

        df = pd.read_csv(file_path, nrows=5)

        for column in df.columns:

            column_records.append({
                "Dataset": dataset_name,
                "File": file_path.name,
                "Column": column,
                "Data Type": str(df[column].dtype)
            })

    return pd.DataFrame(column_records)


#_______________________________________________________________

### 1.6.3 Define a Column Renaming Strategy

# _______________________________________________________________

# Original SST column names
SST_COLUMNS = {
    "mean temperature kelvin": "mean_temperature_kelvin",
    "mean temperature deg C": "mean_temperature_celsius",
    "mean temperature uncertainty": "mean_temperature_uncertainty",
    "fraction of sea-ice-covered ocean": "fraction_of_sea_ice_covered_ocean"
}


def get_rename_mapping(file_name, columns):
    mapping = {}

    # Identify SST region
    if "sst_cameroon" in file_name:
        prefix = "sst_cameroon"

    elif "sst_indian_ocean" in file_name:
        prefix = "sst_indian_ocean"

    else:
        prefix = None

    # Rename only existing SST columns
    if prefix:
        for old_name, new_name in SST_COLUMNS.items():
            if old_name in columns:
                mapping[old_name] = f"{prefix}_{new_name}"

    # Correct VPD spelling only if the column exists
    if "vapor_pressure_dificit" in columns:
        mapping["vapor_pressure_dificit"] = "vapor_pressure_deficit"

    return mapping


def build_rename_preview(train_dir, test_dir):
    rename_records = []

    for directory, dataset_name in [
        (train_dir, "Train"),
        (test_dir, "Test")
    ]:
        for file_path in sorted(Path(directory).glob("*.csv")):
            df = pd.read_csv(file_path, nrows=5)

            mapping = get_rename_mapping(
                file_path.name,
                df.columns
            )

            for old_name, new_name in mapping.items():
                rename_records.append({
                    "Dataset": dataset_name,
                    "File": file_path.name,
                    "Original Column": old_name,
                    "New Column": new_name
                })

    return pd.DataFrame(rename_records)

# _______________________________________________________________
# _______________________________________________________________


def load_and_standardize_datasets(directory):
    datasets = {}

    csv_files = sorted(Path(directory).glob("*.csv"))

    for file_path in csv_files:

        # Load CSV
        df = pd.read_csv(file_path)

        # Apply validated renaming strategy
        rename_mapping = get_rename_mapping(
            file_path.name,
            df.columns
        )

        df = df.rename(columns=rename_mapping)

        # Convert date column
        df["date"] = pd.to_datetime(
            df["date"],
            errors="raise"
        )

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

        # Store using filename as dictionary key
        datasets[file_path.stem] = df

    return datasets



def merge_datasets(datasets):
    if not datasets:
        raise ValueError("No datasets provided for merging.")

    merged_df = None

    for dataset_name, df in datasets.items():

        if "date" not in df.columns:
            raise ValueError(
                f"Dataset '{dataset_name}' does not contain a date column."
            )

        if df["date"].duplicated().any():
            raise ValueError(
                f"Dataset '{dataset_name}' contains duplicate dates."
            )

        if merged_df is None:
            merged_df = df.copy()
            continue

        # Check for overlapping feature names
        overlapping_columns = (
            set(merged_df.columns)
            & set(df.columns)
        ) - {"date"}

        if overlapping_columns:
            raise ValueError(
                f"Duplicate columns found in '{dataset_name}': "
                f"{overlapping_columns}"
            )

        # Merge while preserving all dates
        merged_df = pd.merge(
            merged_df,
            df,
            on="date",
            how="outer",
            validate="one_to_one"
        )

    merged_df = (
        merged_df
        .sort_values("date")
        .reset_index(drop=True)
    )

    return merged_df