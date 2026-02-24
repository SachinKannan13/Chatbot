import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Strings commonly representing null/missing
_NULL_STRINGS = {
    "", "null", "none", "na", "n/a", "nan", "#n/a", "-", "--", "n.a.", "not available",
}

_DATE_FORMAT_CANDIDATES = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%m-%d-%Y",
    "%m/%d/%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
)

_FORCE_DATE_COLUMNS = {"dob", "doj"}


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize a raw DataFrame fetched from Supabase.

    Operations (in order):
    1. Drop fully empty rows
    2. Strip whitespace from all string columns
    3. Standardize null representations
    4. Convert numeric-looking strings to float
    5. Infer and coerce date columns

    Column names are preserved exactly as-is.
    """
    if df.empty:
        return df

    df = df.copy()

    # 1. Drop rows where every cell is NaN / None
    df.dropna(how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 2. Strip whitespace from string columns
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # 3. Replace null-like strings with np.nan
    df[str_cols] = df[str_cols].apply(
        lambda col: col.map(
            lambda x: np.nan if isinstance(x, str) and x.lower() in _NULL_STRINGS else x
        )
    )

    # 4. Convert numeric-looking strings (e.g. "78%", "4.5 ") to float
    for col in df.select_dtypes(include=["object"]).columns:
        converted = _try_numeric_conversion(df[col])
        if converted is not None:
            df[col] = converted
            logger.debug("column_converted_to_numeric", column=col)

    # 5. Infer dates for remaining string columns
    for col in df.select_dtypes(include=["object"]).columns:
        converted = _try_date_conversion(df[col], col_name=col)
        if converted is not None:
            df[col] = converted
            logger.debug("column_converted_to_date", column=col)

    # 6. Force datetime conversion for known date fields (DOB/DOJ)
    for col in df.columns:
        if col.lower() in _FORCE_DATE_COLUMNS and not pd.api.types.is_datetime64_any_dtype(df[col]):
            forced = _parse_with_known_formats(df[col], min_valid_ratio=0.0)
            if forced is not None:
                df[col] = forced
                logger.debug("column_forced_to_date", column=col)

    if "DOB" in df.columns:
        df["DOB"] = pd.to_datetime(df["DOB"], format="%Y-%m-%d", errors="coerce")
    if "DOJ" in df.columns:
        df["DOJ"] = pd.to_datetime(df["DOJ"], format="%Y-%m-%d", errors="coerce")

    logger.info(
        "dataframe_cleaned",
        rows=len(df),
        columns=len(df.columns),
        dtypes={c: str(t) for c, t in df.dtypes.items()},
    )
    return df


def _try_numeric_conversion(series: pd.Series) -> pd.Series | None:
    """
    Attempt to convert a string series to float.
    Returns converted series if >=80% of non-null values parse successfully,
    otherwise None.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return None

    def _parse(val):
        if not isinstance(val, str):
            return val
        clean = val.strip().rstrip("%").strip()
        try:
            return float(clean)
        except ValueError:
            return np.nan

    converted = non_null.apply(_parse)
    valid_ratio = converted.notna().sum() / len(non_null)

    if valid_ratio >= 0.8:
        return series.apply(lambda x: _parse(x) if pd.notna(x) else np.nan)
    return None


def _try_date_conversion(series: pd.Series, col_name: str = "") -> pd.Series | None:
    """
    Attempt to parse a string series as datetime.
    Returns converted series only if enough values parse successfully.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return None

    min_valid_ratio = 0.6 if col_name.lower() in _FORCE_DATE_COLUMNS else 0.8

    return _parse_with_known_formats(series, min_valid_ratio=min_valid_ratio)


def _parse_with_known_formats(series: pd.Series, min_valid_ratio: float) -> pd.Series | None:
    """Parse dates using only explicit formats to avoid dateutil fallback warnings."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return None

    best_fmt = None
    best_ratio = 0.0
    for fmt in _DATE_FORMAT_CANDIDATES:
        converted = pd.to_datetime(non_null, format=fmt, errors="coerce")
        valid_ratio = converted.notna().sum() / len(non_null)
        if valid_ratio > best_ratio:
            best_ratio = valid_ratio
            best_fmt = fmt

    if best_fmt is None or best_ratio < min_valid_ratio:
        return None

    return pd.to_datetime(series, format=best_fmt, errors="coerce")


def get_column_type_summary(df: pd.DataFrame) -> dict:
    """
    Return a dict mapping column name to inferred type label.
    Labels: 'numeric', 'date', 'categorical', 'identifier'
    """
    summary = {}
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            summary[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            summary[col] = "date"
        else:
            n_unique = df[col].nunique(dropna=True)
            n_rows = len(df)
            if n_unique > 0.5 * n_rows and n_rows > 20:
                summary[col] = "identifier"
            else:
                summary[col] = "categorical"
    return summary
