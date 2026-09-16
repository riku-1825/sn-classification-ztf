from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from .preprocessing import BANDS


def _band_stats(band_df: pd.DataFrame) -> dict:
    if len(band_df) == 0:
        return {
            "n_det": 0, "amp": np.nan, "rise_time": np.nan,
            "decline_rate": np.nan, "skew": np.nan, "kurt": np.nan,
            "cadence_mean": np.nan, "cadence_std": np.nan,
        }
    flux = band_df["flux"].values
    mjd = band_df["mjd"].values
    peak_idx = np.argmax(flux)
    peak_flux = flux[peak_idx]
    peak_mjd = mjd[peak_idx]

    rise_time = peak_mjd - mjd.min()
    tail = band_df[band_df["mjd"] >= peak_mjd]
    if len(tail) > 1:
        decline_rate = (peak_flux - tail["flux"].values[-1]) / max(
            tail["mjd"].values[-1] - peak_mjd, 1e-3
        )
    else:
        decline_rate = np.nan

    cadence = np.diff(np.sort(mjd))
    return {
        "n_det": len(band_df),
        "amp": peak_flux - flux.min(),
        "rise_time": rise_time,
        "decline_rate": decline_rate,
        "skew": skew(flux) if len(flux) > 2 else np.nan,
        "kurt": kurtosis(flux) if len(flux) > 2 else np.nan,
        "cadence_mean": cadence.mean() if len(cadence) else np.nan,
        "cadence_std": cadence.std() if len(cadence) else np.nan,
    }


def _interpolate_band(band_df: pd.DataFrame, grid_step_days: float = 1.0) -> pd.DataFrame:
    """Resample a single band's irregular detections onto a regular grid
    via linear interpolation, spanning only the band's own observed
    time range (no extrapolation beyond the first/last real detection).
    Falls back to returning the raw points unchanged if fewer than 2
    detections are available (interpolation is undefined otherwise).
    """
    if len(band_df) < 2:
        return band_df
    band_df = band_df.sort_values("mjd")
    mjd = band_df["mjd"].values
    flux = band_df["flux"].values
    grid = np.arange(mjd.min(), mjd.max() + grid_step_days, grid_step_days)
    flux_interp = np.interp(grid, mjd, flux)
    return pd.DataFrame({"mjd": grid, "flux": flux_interp})


def build_feature_table(df: pd.DataFrame, interpolate: bool = False,
                         grid_step_days: float = 1.0) -> pd.DataFrame:
    """One row per object_id, engineered features across all bands +
    simple cross-band color/ratio features.

    interpolate=False (default): features computed directly from the
    raw, irregularly-sampled detections -- this is what train.py uses.

    interpolate=True: each band is first resampled onto a regular grid
    via linear interpolation (see _interpolate_band), THEN the same
    feature set is computed from the resampled series. Used by
    ablation_interpolation.py to test whether smoothing over irregular
    sampling helps or hurts the engineered-feature LightGBM baseline --
    interpolation can suppress noise, but it can also manufacture
    fake structure in sparsely-sampled regions, so this is a genuine
    empirical question rather than an obviously-good preprocessing step.
    """
    rows = []
    for oid, sub in df.groupby("object_id"):
        row = {"object_id": oid, "label": sub["label"].iloc[0]}
        band_peak_flux = {}
        for band in BANDS:
            band_df = sub[sub["band"] == band]
            if interpolate:
                band_df = _interpolate_band(band_df, grid_step_days=grid_step_days)
            stats = _band_stats(band_df)
            for k, v in stats.items():
                row[f"{band}_{k}"] = v
            if len(band_df) > 0:
                band_peak_flux[band] = band_df["flux"].max()

        # cross-band color / amplitude ratio features
        if "g" in band_peak_flux and "r" in band_peak_flux and band_peak_flux["r"] != 0:
            row["gr_ratio"] = band_peak_flux["g"] / band_peak_flux["r"]
        else:
            row["gr_ratio"] = np.nan

        row["total_n_det"] = sum(
            row.get(f"{b}_n_det", 0) for b in BANDS if not np.isnan(row.get(f"{b}_n_det", np.nan))
        )
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Import build_feature_table(long_df) after preprocessing.records_to_long_df().")
