"""Preprocessing transformers for the Early Spring project.

The classes live here instead of in the Preparation notebook so `preprocessor.pkl`
(and any model saved with the preprocessor inside it) can be loaded anywhere this
file is importable, without rerunning the Preparation notebook.
"""

import calendar
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

TARGET = "early_spring"
TOTAL_VARIABLES = ["precipitation", "snowfall"]
NINO_COLUMNS = ["nino34_nov", "nino34_dec", "nino34_jan"]
SKEWED_VARIABLES = ["precipitation", "snowfall", "snowdepth"]
FREEZING_F = 32
LAST_DAYS = 10
MONTHS = {1: "jan", 2: "feb"}


def daily_columns(columns):
    """Group daily columns like `max_temp_01_15` as {variable: {month: [columns]}}."""
    grouped = {}
    for col in columns:
        match = re.fullmatch(r"(.+)_(\d{2})_\d{2}", col)
        if match:
            month = MONTHS[int(match[2])]
            grouped.setdefault(match[1], {}).setdefault(month, []).append(col)
    return grouped


def days_before_march(X, variable, n=LAST_DAYS):
    """Values for each of the `n` days before March 1, counted back so leap years line up."""
    leap = X["year"].map(calendar.isleap)
    return {
        f"{variable}_mar1_minus{k:02d}": X[f"{variable}_02_{30 - k:02d}"].where(leap, X[f"{variable}_02_{29 - k:02d}"])
        for k in range(n, 0, -1)
    }


def share_of_days(condition, values):
    """Fraction of non-missing days meeting `condition` (NaN if every day is missing)."""
    observed = values.notna().sum(axis=1)
    return condition.sum(axis=1) / observed.replace(0, np.nan)


class FrameTransformer(TransformerMixin, BaseEstimator, auto_wrap_output_keys=None):
    """Base for stateless steps that take and return a DataFrame."""

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):
        """Nothing is learned in `fit`, so the step is always ready to transform."""
        return True

    def set_output(self, *, transform=None):
        """Output is already a DataFrame, so there is nothing to configure."""
        return self


class DropColumns(FrameTransformer):
    """Drop the target (if present) and March 1."""

    def transform(self, X):
        dropped = [col for col in X.columns if col.endswith("_03_01")]
        return X.drop(columns=dropped + [TARGET], errors="ignore")


class WeatherAggregator(FrameTransformer):
    """Replace the daily weather and soil columns with monthly summaries.

    Precipitation and snowfall are totaled; everything else is averaged. Average
    temperature also gets its spread, extremes, and share of days above freezing,
    and snowfall gets its share of snowy days. Shares are out of the days that
    were observed, so missing days are not counted as cold or snow-free.

    Each of the 10 days before March 1 also keeps its own values, counted back
    from March 1 (`<variable>_mar1_minus01` is the day before). February 29 is a
    normal day: it counts toward February and is `mar1_minus01` in leap years.
    """

    def transform(self, X):
        daily = daily_columns(X.columns)
        windows = {
            variable: {month: X[cols] for month, cols in months.items()}
            for variable, months in daily.items()
        }
        features = {}

        for variable, months in windows.items():
            for month, values in months.items():
                if variable in TOTAL_VARIABLES:
                    features[f"{variable}_{month}_total"] = values.sum(axis=1, min_count=1)
                else:
                    features[f"{variable}_{month}_mean"] = values.mean(axis=1)

        for month, temps in windows["avg_temp"].items():
            features[f"avg_temp_{month}_std"] = temps.std(axis=1)
            features[f"avg_temp_{month}_min"] = temps.min(axis=1)
            features[f"avg_temp_{month}_max"] = temps.max(axis=1)
            features[f"above_freezing_{month}_pct"] = share_of_days(temps > FREEZING_F, temps)

        for month, snow in windows["snowfall"].items():
            features[f"snowy_days_{month}_pct"] = share_of_days(snow > 0, snow)

        for variable in daily:
            features |= days_before_march(X, variable)

        raw = [col for months in daily.values() for cols in months.values() for col in cols]
        return pd.concat([X.drop(columns=raw), pd.DataFrame(features, index=X.index)], axis=1)


class ClimateFeatures(FrameTransformer):
    """Optionally average the three Nino 3.4 months, which are almost perfectly correlated."""

    def __init__(self, collapse_nino=True):
        self.collapse_nino = collapse_nino

    def transform(self, X):
        if not self.collapse_nino:
            return X
        return X.drop(columns=NINO_COLUMNS).assign(nino34=X[NINO_COLUMNS].mean(axis=1))


class YearPhil(FrameTransformer):
    """Encode Phil's prediction as 0/1 and optionally drop `year` or `phil_pred`."""

    def __init__(self, include_year=True, include_phil=True):
        self.include_year = include_year
        self.include_phil = include_phil

    def transform(self, X):
        X = X.assign(phil_pred=X["phil_pred"].astype("float64"))
        keep = {"year": self.include_year, "phil_pred": self.include_phil}
        return X.drop(columns=[col for col, kept in keep.items() if not kept])


def build_preprocessor():
    """Unfitted preprocessing pipeline. Every step's settings can be tuned with `set_params`."""
    skewed = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ])
    skewed_names = "|".join(SKEWED_VARIABLES)
    columns = ColumnTransformer(
        [("skewed", skewed, make_column_selector(pattern=f"^(?:{skewed_names})_"))],
        remainder=SimpleImputer(strategy="median"),
        verbose_feature_names_out=False,
    )
    preprocessor = Pipeline([
        ("drop", DropColumns()),
        ("aggregate", WeatherAggregator()),
        ("climate", ClimateFeatures()),
        ("year_phil", YearPhil()),
        ("columns", columns),
        ("scale", StandardScaler()),
        ("reduce", "passthrough"),
    ])
    return preprocessor.set_output(transform="pandas")
