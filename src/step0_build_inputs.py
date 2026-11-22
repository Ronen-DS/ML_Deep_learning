"""
STEP 0 - Build the input files for the deep-learning pipeline.

Reads the CTU-UHB intrapartum CTG database (PhysioNet), cleans the FHR and UC
signals, extracts handcrafted + clinical tabular features, and writes:

  1. ctg_features.csv                       raw per-record features (cache)
  2. cnn_tabular_features_final[_no_missingness].csv
                                            modeling-ready tabular features
                                            + record_id + y_class  -> STEP 3
  3. ctu_full_length_signals_cache.npz      full-length 4-channel signals
                                            (FHR, UC, FHR-mask, UC-mask) -> STEP 2

Run from the repository root:
    python src/step0_build_inputs.py

This is a trimmed version of the original exploratory script: the EDA plots,
the tabular-only LightGBM baseline, anomaly/outlier PDF export and other
diagnostics were removed. The functions that produce the outputs are unchanged.
"""

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from tqdm import tqdm
from scipy.ndimage import label, median_filter
from scipy.interpolate import PchipInterpolator
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR: Optional[str] = "data/CTU_DB"   # folder with the CTU-UHB .dat/.hea files
PN_DIR        = None                       # used only if DATA_DIR is None
RECORD_IDS    = list(range(1001, 1507)) + list(range(2001, 2047))
FS            = 4                          # Hz (1 sample = 0.25 s)

# Outputs (written to the current working directory)
OUTPUT_CSV      = "ctg_features.csv"                  # raw feature cache
FULL_CACHE_PATH = "ctu_full_length_signals_cache.npz" # -> step 2

# Label: acidemia = umbilical artery pH <= 7.15
TARGET         = "ph"
TARGET_thresh  = 7.15
TARGET2        = "base_excess"   # read by get_combined_emergency_category (not used for the label)
TARGET_thresh2 = -12
TARGET3        = "apgar5"
TARGET_thresh3 = 6
ACTIVE_TARGETS = [TARGET]

# Cohort selection
STAGE_SELECT       = "stage1"   # "stage1", "both" or "either"
CTG_thresh         = 100        # max % missing FHR/UC allowed (100 = keep all)
EXCLUDE_OUTLIERS   = True
OUTLIER_RECORD_IDS = [1188]
RECORD_60_MIN_TRIM = 0          # 1 = keep only the last 60 min of each record

# Drop the signal-missingness features from the tabular table.
# True reproduces "cnn_tabular_features_final_no_missingness.csv" used in step 3.
DROP_MISSINGNESS_FEATURES = True
MISSINGNESS_FEATURES = [
    "fhr_missing_pct",
    "fhr_missing_pct_escalation",
    "fhr_missing_pct_first_15min",
    "fhr_missing_pct_last_15min",
    "uc_missing_pct",
]
FINAL_FEATURES_CSV = (
    "cnn_tabular_features_final_no_missingness.csv"
    if DROP_MISSINGNESS_FEATURES
    else "cnn_tabular_features_final.csv"
)

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 & 2 — Load WFDB Records & Parse Clinical Metadata
# ══════════════════════════════════════════════════════════════════════════════

def load_record(record_id: int) -> Optional[wfdb.Record]:
    name = str(record_id)
    try:
        if DATA_DIR:
            return wfdb.rdrecord(os.path.join(DATA_DIR, name))
        return wfdb.rdrecord(name, pn_dir=PN_DIR)
    except Exception:
        return None
 
_FIELD_MAP = {
    "pH":            "ph",
    "BDecf":         "bdecf",
    "pCO2":          "pco2",
    "BE":            "base_excess",
    "Apgar1":        "apgar1",
    "Apgar5":        "apgar5",
    "NICU days":     "nicu_days",
    "Seizures":      "seizures",
    "HIE":           "hie",
    "Intubation":    "intubation",
    "Main diag.":    "main_diag",
    "Other diag.":   "other_diag",
    "Gest. weeks":   "gest_weeks",
    "Weight(g)":     "birthweight_g",
    "Sex":           "fetal_sex",
    "Age":           "maternal_age",
    "Gravidity":     "gravidity",
    "Parity":        "parity",
    "Diabetes":      "diabetes",
    "Hypertension":  "hypertension",
    "Preeclampsia":  "preeclampsia",
    "Liq. praecox":  "liq_praecox",
    "Pyrexia":       "pyrexia",
    "Meconium":      "meconium",
    "Presentation":  "presentation",
    "Induced":       "induced",
    "I.stage":       "stage1_min",
    "NoProgress":    "no_progress",
    "CK/KP":         "ck_kp",
    "II.stage":      "stage2_min",
    "Deliv. type":   "delivery_type",
    "Rec. type":     "rec_type",
    "Pos. II.st.":   "pos_stage2",
    "Sig2Birth":     "sig2birth",
}

def parse_comments(comments: list[str]) -> dict:
    raw: dict = {}
    for line in comments:
        line = line.strip()
        if not line or line.startswith("-"):
            continue
        parts = line.rsplit(None, 1)      
        if len(parts) == 2:
            field, value = parts[0].strip(), parts[1].strip()
            try:
                raw[field] = float(value)
            except ValueError:
                raw[field] = value

    return {clean: raw[orig] for orig, clean in _FIELD_MAP.items() if orig in raw}

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PREPROCESSING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def select_uncorrelated_features(
    X: pd.DataFrame,
    method: str = "spearman",
    corr_threshold: float = 0.85,
    tiebreak: str = "missingness",   # "missingness", "variance", or "manual"
    manual_priority: list = None,     # ordered list of preferred feature names, used if tiebreak="manual"
    plot: bool = True,
):
    """
    Cluster correlated features and keep one representative per cluster.
    Operates only on X (no y involved) -> safe to run once on full dataset,
    no CV leakage risk.

    Parameters
    ----------
    X : predictor matrix (numeric columns only get clustered; non-numeric
        columns are passed through untouched)
    corr_threshold : features with |correlation| >= this are considered
        redundant and clustered together
    tiebreak : how to choose which feature in a cluster survives
        - "missingness": keep the feature with fewest NaNs
        - "variance": keep the feature with highest variance
        - "manual": keep whichever feature in the cluster appears first
          in manual_priority (use this to encode domain knowledge, e.g.
          prefer a clinically validated FHR metric over a derived transform)
    """

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in X.columns if c not in numeric_cols]

    if len(numeric_cols) < 2:
        return X.copy(), {}, None

    corr = X[numeric_cols].corr(method=method).abs()

    # Hierarchical clustering on 1 - |corr| distance
    distance = 1 - corr
    np.fill_diagonal(distance.values, 0)
    condensed = squareform(distance.values, checks=False)
    Z = linkage(condensed, method="average")

    # distance threshold corresponding to corr_threshold
    dist_thresh = 1 - corr_threshold
    cluster_labels = fcluster(Z, t=dist_thresh, criterion="distance")

    clusters = {}
    for feat, label in zip(numeric_cols, cluster_labels):
        clusters.setdefault(label, []).append(feat)

    selected_features = []
    cluster_report = {}

    for label, feats in clusters.items():
        if len(feats) == 1:
            selected_features.append(feats[0])
            cluster_report[feats[0]] = {"cluster": label, "dropped": []}
            continue

        if tiebreak == "missingness":
            scores = X[feats].notna().mean()  # higher = fewer NaNs = better
            winner = scores.idxmax()
        elif tiebreak == "variance":
            scores = X[feats].var()
            winner = scores.idxmax()
        elif tiebreak == "manual":
            assert manual_priority is not None, "manual_priority required for tiebreak='manual'"
            ranked = [f for f in manual_priority if f in feats]
            winner = ranked[0] if ranked else feats[0]
        else:
            raise ValueError(f"Unknown tiebreak: {tiebreak}")

        dropped = [f for f in feats if f != winner]
        selected_features.append(winner)
        cluster_report[winner] = {"cluster": label, "dropped": dropped}

    final_cols = selected_features + non_numeric_cols
    X_reduced = X[final_cols].copy()

    print(f"[FEATURE SELECTION] {len(numeric_cols)} numeric features -> "
          f"{len(selected_features)} kept ({len(numeric_cols) - len(selected_features)} dropped as redundant)")
    for winner, info in cluster_report.items():
        if info["dropped"]:
            print(f"    kept '{winner}'  (dropped: {info['dropped']})")

    if plot:
        plt.figure(figsize=(14, 8))
        dendrogram(Z, labels=numeric_cols, leaf_rotation=90)
        plt.axhline(y=dist_thresh, color="red", linestyle="--",
                    label=f"cut @ corr={corr_threshold}")
        plt.title(f"Feature Correlation Clustering ({method.capitalize()})")
        plt.ylabel("Distance (1 - |correlation|)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return X_reduced, cluster_report, Z

def trim_signal_borders(
    fhr: np.ndarray, 
    uc: np.ndarray = None, 
    trim_start: bool = True, 
    trim_end: bool = True, 
    threshold_min: int = 5, 
    fs: int = FS
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Trims leading and/or trailing missingness/zeros from FHR based on 
    valid data bounds, and applies the exact same slice boundaries to UC.
    
    Parameters:
    - fhr: Fetal Heart Rate signal array
    - uc: Uterine Contraction signal array (optional)
    - trim_start: Whether to trim leading zeros/gaps at the beginning
    - trim_end: Whether to trim trailing zeros/gaps at the end
    - threshold_min: Minimum duration in minutes required to trigger a trim (optional safeguard)
    - fs: Sampling frequency in Hz
    
    Returns:
    - Tuple of trimmed (fhr, uc) arrays
    """
    fhr_s = fhr.astype(float)
    uc_s = uc.astype(float) if uc is not None else None
    
    valid_mask = ~np.isnan(fhr_s) & (fhr_s != 0)
    if not np.any(valid_mask):
        return fhr_s, uc_s
        
    valid_indices = np.where(valid_mask)[0]
    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]
    
    start_idx = 0
    end_idx = len(fhr_s)
    
    # Handle Start Trimming
    if trim_start:
        lead_len_min = first_valid / (fs * 60)
        if lead_len_min >= threshold_min:
            start_idx = first_valid
            
    # Handle End Trimming
    if trim_end:
        tail_len_min = (len(fhr_s) - 1 - last_valid) / (fs * 60)
        if tail_len_min >= threshold_min:
            end_idx = last_valid + 1
            
    # Slice FHR
    trimmed_fhr = fhr_s[start_idx:end_idx]
    
    # Slice UC identically if present
    trimmed_uc = uc_s[start_idx:end_idx] if uc_s is not None else None
    
    return trimmed_fhr, trimmed_uc


def _clean_fhr(raw: np.ndarray, fs: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = raw.copy().astype(float)
    
    # 1. Zeros and existing NaNs are unified as missing data
    raw_zeros_mask = np.isnan(s) | (s == 0)
    s[raw_zeros_mask] = np.nan
    
    #s = smooth_signal(s, window_sec=60.0, fs=fs, method="median")
    
    # Physiological ceiling check
    #s[s > 220] = np.nan
    s[(s < 50) | (s > 200)] = np.nan
    
    # 2. Robust outlier rejection (Hampel-style local median/MAD check)
    s_median = median_filter(s, size=5)
    spike_diff = np.abs(s - s_median)
    s[spike_diff > 35] = np.nan
    
    cleaned = s.copy()
    nans = np.isnan(cleaned)
    
    if not np.all(nans) and nans.any():
        labeled_gaps, num_features = label(nans)
        x_indices = np.arange(len(cleaned))
        valid_idx = x_indices[~nans]
        valid_val = cleaned[~nans]
        
        if len(valid_idx) > 1:
            # Base array starts with NaNs for gaps to prevent artificial linear bridging
            interpolated = cleaned.copy()
            
            MAX_MICRO_GAP = 20  # ~5 seconds at 4Hz
            MAX_GAP_SAMPLES = 15 * fs  # 60 seconds max threshold
            
            pchip = PchipInterpolator(valid_idx, valid_val, extrapolate=False)
            
            # Compute gap lengths efficiently using bincount
            gap_sizes = np.bincount(labeled_gaps.ravel())
            
            for feat in range(1, num_features + 1):
                gap_len = gap_sizes[feat]
                gap_mask = (labeled_gaps == feat)
                
                if gap_len <= MAX_MICRO_GAP:
                    # Shape-preserving PCHIP only for safe micro-gaps
                    gap_indices = x_indices[gap_mask]
                    interpolated[gap_mask] = pchip(gap_indices)
                elif gap_len <= MAX_GAP_SAMPLES:
                    # Optional: fallback to linear only for bounded medium gaps if desired, 
                    # otherwise leave as NaN to prevent hallucinated trends.
                    pass 
                # Gaps > MAX_GAP_SAMPLES naturally remain NaN
                
            cleaned = interpolated
        
    final_nan_mask = np.isnan(cleaned)
    return cleaned, raw_zeros_mask, final_nan_mask


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (FHR & UC)
# ══════════════════════════════════════════════════════════════════════════════

def _segment_fft_features(signal: np.ndarray, fs: int) -> tuple[float, float, float]:
    min_seg_len = int(2 * 60 * fs)   

    valid = ~np.isnan(signal)

    lf_list = []
    hf_list = []
    ratio_list = []

    i = 0
    n = len(signal)

    while i < n:
        if valid[i]:
            j = i
            while j < n and valid[j]:
                j += 1

            seg = signal[i:j]

            if len(seg) >= min_seg_len:
                seg = (
                    pd.Series(seg)
                    .interpolate(limit=3)
                    .to_numpy()
                )

                seg = seg - np.mean(seg)

                fft = np.abs(np.fft.rfft(seg)) ** 2
                freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs)

                total = fft.sum()

                if total > 0:
                    lf = fft[(freqs >= 0.04) & (freqs < 0.15)].sum()
                    hf = fft[(freqs >= 0.15) & (freqs < 0.40)].sum()

                    lf_list.append(lf / total)
                    hf_list.append(hf / total)
                    ratio_list.append(lf / (hf + 1e-12))

            i = j
        else:
            i += 1

    if len(lf_list) == 0:
        return np.nan, np.nan, np.nan

    return (
        float(np.nanmedian(lf_list)),
        float(np.nanmedian(hf_list)),
        float(np.nanmedian(ratio_list)),
    )

def _decel_recovery_time_mean_s(
    signal: np.ndarray,
    baseline: np.ndarray,
    fs: int,
    delta: float = 15,
    min_duration_s: float = 15,
    max_recovery_s: float = 120,
) -> float:
    mask = signal < (baseline - delta)
    min_samples = int(min_duration_s * fs)
    recovery_times = []
    i = 0

    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1

            if (j - i) >= min_samples:
                segment = signal[i:j]
                if not np.all(np.isnan(segment)):
                    nadir_idx = i + np.nanargmin(segment)
                    recovery_limit = min(len(signal), nadir_idx + int(max_recovery_s * fs))
                    recovery_threshold = baseline[nadir_idx] - (delta / 2)

                    for k in range(nadir_idx, recovery_limit):
                        if np.isnan(signal[k]):
                            continue
                        if signal[k] >= recovery_threshold:
                            recovery_times.append((k - nadir_idx) / fs)
                            break
            i = j
        else:
            i += 1

    if len(recovery_times) > 0:
        return float(np.mean(recovery_times))

    return np.nan

def _decel_burden_per_hour(
    signal: np.ndarray,
    baseline: np.ndarray,
    fs: int,
    delta: float = 15,
    min_duration_s: float = 15,
) -> float:
    mask = signal < (baseline - delta)
    min_samples = int(min_duration_s * fs)

    total_burden = 0.0
    i = 0

    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1

            if (j - i) >= min_samples:
                total_burden += np.nansum(baseline[i:j] - signal[i:j])
            i = j
        else:
            i += 1

    total_burden /= fs

    # FIX: normalize by valid (non-NaN) monitored duration, not raw record
    # length. Using len(signal) silently mixes dropout-related missingness
    # into this feature, and dropout is correlated with outcome (see the
    # 22-record anomaly review) — this was injecting outcome-correlated
    # noise into a feature that's supposed to measure deceleration burden.
    valid_duration_hr = np.sum(~np.isnan(signal)) / fs / 3600.0

    if valid_duration_hr == 0:
        return np.nan

    return total_burden / valid_duration_hr

def _baseline_trend_30m(signal: np.ndarray, fs: int) -> float:
    min_len = int(60 * fs)
    n_min = len(signal) // min_len

    # FIX: insufficient data should be NaN, not 0.0 — 0.0 is a real,
    # meaningful value (a genuinely flat baseline trend), and collapsing
    # "not enough data to compute" into the same value as "trend is truly
    # flat" makes the two indistinguishable to the model and imputer.
    if n_min < 5:
        return np.nan

    min_means = np.nanmean(
        signal[: n_min * min_len].reshape(n_min, min_len),
        axis=1
    )

    last_n = min(30, len(min_means))
    last_means = min_means[-last_n:]
    valid = np.isfinite(last_means)

    if np.sum(valid) < 5:
        return np.nan

    x = np.arange(last_n)[valid]
    y = last_means[valid]
    slope = np.polyfit(x, y, 1)[0]

    return float(slope)

def _decel_cumulative_time_per_hour(
    signal: np.ndarray,
    baseline: np.ndarray,
    fs: int,
    delta: float = 15,
    min_duration_s: float = 15,
) -> float:
    mask = signal < (baseline - delta)
    min_samples = int(min_duration_s * fs)

    total_samples = 0
    i = 0

    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1

            if (j - i) >= min_samples:
                total_samples += (j - i)
            i = j
        else:
            i += 1

    total_seconds = total_samples / fs

    # FIX: same valid-duration normalization as _decel_burden_per_hour
    valid_duration_hours = np.sum(~np.isnan(signal)) / fs / 3600.0

    if valid_duration_hours == 0:
        return np.nan

    return total_seconds / valid_duration_hours

def _last_n_min_fhr_features(signal: np.ndarray, fs: int, n_min: int = 15) -> dict:
    n_samples = int(n_min * 60 * fs)
    tail = signal[-n_samples:] if len(signal) >= n_samples else signal

    valid_frac = np.mean(~np.isnan(tail))

    return {
        f"fhr_std_last_{n_min}min": float(np.nanstd(tail)) if valid_frac > 0 else np.nan,
        f"fhr_iqr_last_{n_min}min": float(
            np.nanpercentile(tail, 75) - np.nanpercentile(tail, 25)
        ) if valid_frac > 0 else np.nan,
        f"fhr_missing_pct_last_{n_min}min": float(np.mean(np.isnan(tail))) * 100,
        f"fhr_mean_last_{n_min}min": float(np.nanmean(tail)) if valid_frac > 0 else np.nan,
    }


def extract_fhr_features(raw_fhr: np.ndarray, fs: int = FS) -> dict:
    signal, raw_zeros_mask, final_nan_mask = _clean_fhr(raw_fhr)
    if np.all(np.isnan(signal)):
        return {}

    f: dict = {}
    f["fhr_n_acc"] = np.nan
    f["fhr_n_dec"] = np.nan
    f["fhr_acc_amp"] = np.nan
    f["fhr_dec_amp"] = np.nan
    f["fhr_acc_per_hour"] = np.nan
    f["fhr_dec_per_hour"] = np.nan

    f["fhr_missing_pct"] = float(np.mean(raw_zeros_mask)) * 100
    f["fhr_unrecoverable_pct"] = float(np.mean(final_nan_mask)) * 100

    f["fhr_mean"]        = float(np.nanmean(signal))
    f["fhr_std"]         = float(np.nanstd(signal))
    f["fhr_min"]         = float(np.nanmin(signal))
    f["fhr_max"]         = float(np.nanmax(signal))
    f["fhr_range"]       = f["fhr_max"] - f["fhr_min"]
    f["fhr_median"]      = float(np.nanmedian(signal))
    f["fhr_iqr"]         = float(np.nanpercentile(signal, 75) - np.nanpercentile(signal, 25))
    f["fhr_skew"]        = float(pd.Series(signal).skew(skipna=True))
    f["fhr_kurtosis"]    = float(pd.Series(signal).kurt(skipna=True))

    signal = signal.astype(float)
    win = int(2 * 60 * fs)

    if len(signal) >= win:
        signal_interp = pd.Series(signal).interpolate(limit=3).to_numpy()

        baseline = pd.Series(signal_interp).rolling(
            win, center=True, min_periods=1
        ).mean().values

        f["fhr_baseline_mean"] = float(np.nanmean(baseline))
        f["fhr_baseline_std"]  = float(np.nanstd(baseline))

        epoch = fs
        n_ep = len(signal_interp) // epoch

        if n_ep > 1:
            ep_means = np.nanmean(
                signal_interp[: n_ep * epoch].reshape(n_ep, epoch),
                axis=1
            )
            valid_fraction = np.mean(~np.isnan(ep_means))

            if valid_fraction > 0.8:
                f["fhr_stv"] = float(np.nanmean(np.abs(np.diff(ep_means))))
            else:
                f["fhr_stv"] = np.nan

        min_len = int(60 * fs)
        n_min = len(signal_interp) // min_len

        if n_min > 1:
            min_means = np.nanmean(
                signal_interp[: n_min * min_len].reshape(n_min, min_len),
                axis=1
            )
            valid_fraction = np.mean(~np.isnan(min_means))

            if valid_fraction > 0.8:
                f["fhr_ltv"] = float(np.nanstd(min_means))
            else:
                f["fhr_ltv"] = np.nan

            f["fhr_ltv_valid_fraction"] = float(valid_fraction)

        DELTA = 15
        MIN_SAMP = int(15 * fs)

        def _count_events(mask: np.ndarray, baseline: np.ndarray) -> tuple[int, float]:
            count, amps = 0, []
            i = 0

            while i < len(mask):
                if mask[i]:
                    j = i
                    while j < len(mask) and mask[j]:
                        j += 1

                    if (j - i) >= MIN_SAMP:
                        count += 1
                        amps.append(float(np.nanmean(np.abs(signal[i:j] - baseline[i:j]))))
                    i = j
                else:
                    i += 1

            return count, float(np.nanmean(amps)) if amps else 0.0

        f["fhr_n_acc"], f["fhr_acc_amp"] = _count_events(signal > baseline + DELTA, baseline)
        f["fhr_n_dec"], f["fhr_dec_amp"] = _count_events(signal < baseline - DELTA, baseline)

        valid_duration_hr = np.sum(~np.isnan(signal)) / fs / 3600.0

        if valid_duration_hr > 0:
            f["fhr_acc_per_hour"] = f["fhr_n_acc"] / valid_duration_hr
            f["fhr_dec_per_hour"] = f["fhr_n_dec"] / valid_duration_hr
        else:
            f["fhr_acc_per_hour"] = np.nan
            f["fhr_dec_per_hour"] = np.nan

        f["fhr_decel_cum_time_per_hour"] = _decel_cumulative_time_per_hour(signal, baseline, fs, delta=DELTA, min_duration_s=15)
        f["fhr_decel_burden_per_hour"] = _decel_burden_per_hour(signal, baseline, fs, delta=DELTA, min_duration_s=15)
        f["fhr_decel_recovery_time_mean_s"] = _decel_recovery_time_mean_s(signal, baseline, fs, delta=DELTA, min_duration_s=15)
        f["fhr_baseline_trend_30m_bpm_per_min"] = _baseline_trend_30m(signal, fs)

        # ===== NEW: last-N-minute + escalation features =====
        last_15 = _last_n_min_fhr_features(signal, fs, n_min=15)
        f.update(last_15)

        n_first_samples = int(15 * 60 * fs)
        first_15_signal = signal[:n_first_samples] if len(signal) >= n_first_samples else signal
        first_15 = {
            "fhr_std_first_15min": float(np.nanstd(first_15_signal)),
            "fhr_missing_pct_first_15min": float(np.mean(np.isnan(first_15_signal))) * 100,
        }
        f.update(first_15)

        f["fhr_std_escalation"] = f["fhr_std_last_15min"] - f["fhr_std_first_15min"]
        f["fhr_missing_pct_escalation"] = (
            f["fhr_missing_pct_last_15min"] - f["fhr_missing_pct_first_15min"]
        )
        
    lf, hf, ratio = _segment_fft_features(signal, fs)
    f["fhr_lf_power"] = lf
    f["fhr_hf_power"] = hf
    f["fhr_lf_hf_ratio"] = ratio

    def _sampen(x: np.ndarray, m: int = 2, r_fac: float = 0.2) -> float:
        x = x.astype(float)
        x = x[:800].copy()
        r = r_fac * np.std(x)
        N = len(x)

        def _phi(m_):
            c = 0
            for i in range(N - m_):
                c += int(np.sum(np.abs(x[i + 1: N - m_ + 1] - x[i]) <= r)) - 1
            return c / max(N - m_, 1)

        try:
            return float(-np.log((_phi(m + 1) + 1e-10) / (_phi(m) + 1e-10)))
        except Exception:
            return np.nan

    def _segment_sampen(signal: np.ndarray, fs: int) -> float:
        min_seg_len = int(2 * 60 * fs)
        valid = ~np.isnan(signal)
        segments = []
        i = 0
        n = len(signal)

        while i < n:
            if valid[i]:
                j = i
                while j < n and valid[j]:
                    j += 1
                seg = signal[i:j]
                if len(seg) >= min_seg_len:
                    seg = pd.Series(seg).interpolate(limit=3).to_numpy()
                    val = _sampen(seg)
                    if not np.isnan(val):
                        segments.append(val)
                i = j
            else:
                i += 1

        return float(np.nanmedian(segments)) if len(segments) > 0 else np.nan

    if len(signal) >= 100:
        try:
            f["fhr_sampen"] = _segment_sampen(signal, fs)
        except Exception:
            f["fhr_sampen"] = np.nan

    return f

def detect_uc_disconnects(
    uc_signal: np.ndarray,
    fs: int = FS,
    min_duration_mins: int = 10,
    var_threshold: float = 0.001,
    val_tolerance: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
  """Detects machine disconnects/flatlines in the UC signal while preserving

  normal physiological contraction pauses.

  Returns:
  - cleaned_uc: UC array with disconnect zones converted to NaN
  - disconnect_mask: Boolean array indicating where disconnects occurred
  """
  uc_s = uc_signal.astype(float).copy()
  min_samples = int(min_duration_mins * 60 * fs)
  window_size = int(30 * fs)

  s = pd.Series(uc_s)
  rolling_var = s.rolling(window=window_size, center=True, min_periods=1).var()

  # Identify low-variance (flat) regions
  is_flat = rolling_var < var_threshold

  block_id = (is_flat != is_flat.shift()).cumsum()
  disconnect_mask = np.zeros_like(uc_s, dtype=bool)

  for _, group in s.groupby(block_id):
    if is_flat.loc[group.index[0]] and len(group) >= min_samples:
      # Check if it's pegged near zero or stuck at a constant low baseline offset
      if group.mean() <= val_tolerance or group.nunique() <= 3:
        disconnect_mask[group.index] = True

  cleaned_uc = uc_s.copy()
  cleaned_uc[disconnect_mask] = np.nan
  return cleaned_uc, disconnect_mask

def clean_uc_signal(
    uc_raw: np.ndarray,
    fs: int = FS,
    min_duration_mins: int = 10,
    var_threshold: float = 0.001,
    val_tolerance: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shared UC preprocessing for both feature engineering and NPZ saving.

    Returns
    -------
    uc_clean : np.ndarray
        Cleaned UC signal (spikes removed, tiny gaps interpolated,
        disconnects and zeros set to NaN before interpolation).
    uc_disconnect_mask : np.ndarray (bool)
        Boolean mask of detected disconnect/flatline regions.
    uc_final_nan_mask : np.ndarray (bool)
        Final missing-data mask after cleaning (NaNs in uc_clean).
    """
    # 1. Detect disconnects / long flatlines
    uc_cleaned_disconnects, uc_disconnect_mask = detect_uc_disconnects(
        uc_raw,
        fs=fs,
        min_duration_mins=min_duration_mins,
        var_threshold=var_threshold,
        val_tolerance=val_tolerance,
    )

    # 2. Work on a float copy and unify zeros + existing NaNs as missing
    s = uc_cleaned_disconnects.astype(float)
    s[s == 0] = np.nan
    #s = smooth_signal(s, window_sec=1.0, fs=fs, method="median")
    
    # 3. Remove isolated spike artifacts (same logic as in extract_uc_features)
    median = median_filter(s, size=5)
    spike_mask = (np.abs(s - median) > 40) & (
        (np.abs(np.roll(s, 1) - s) > 30) |
        (np.abs(np.roll(s, -1) - s) > 30)
    )
    spike_mask[[0, -1]] = False
    s[spike_mask] = np.nan

    # 4. Fill only tiny gaps created by spike removal
    s = pd.Series(s).interpolate(limit=2).to_numpy()

    uc_clean = s
    uc_final_nan_mask = np.isnan(uc_clean)

    return uc_clean, uc_disconnect_mask, uc_final_nan_mask


def extract_uc_features(uc: np.ndarray, fs: int = FS) -> dict:
    # Use the shared UC cleaning pipeline (identical for FC and NPZ)
    uc_clean, uc_disconnect_mask, uc_final_nan_mask = clean_uc_signal(uc, fs=fs)

    s = uc_clean  # already cleaned; no need to repeat steps

    f: dict = {
        "uc_mean": float(np.nanmean(s)) if not np.all(np.isnan(s)) else 0.0,
        "uc_std":  float(np.nanstd(s)) if not np.all(np.isnan(s)) else 0.0,
        "uc_max":  float(np.nanmax(s)) if not np.all(np.isnan(s)) else 0.0,
        "uc_energy": float(np.nansum(s ** 2)),
        "uc_missing_pct": float(np.mean(uc_final_nan_mask)) * 100,
        "uc_disconnect_pct": float(np.mean(uc_disconnect_mask)) * 100,
    }

    f["uc_contractions_last_10min"] = np.nan
    f["uc_contractions_per_min_last_10min"] = np.nan
    f["uc_valid_fraction_last_10min"] = np.nan

    s_filled = np.nan_to_num(s, nan=0.0)
    window_size = int(fs * 2)
    s_smoothed = np.convolve(
        s_filled, np.ones(window_size) / window_size, mode="same"
    )

    valid_data = s_smoothed[s_smoothed > 0]
    threshold = np.percentile(valid_data, 75) if len(valid_data) > 0 else 0.0

    above = (s_smoothed > threshold).astype(int)
    above[np.isnan(s)] = 0

    trans = np.diff(above)
    starts = np.where(trans == 1)[0]
    ends = np.where(trans == -1)[0]

    resting_mask = (above == 0) & (~np.isnan(s))

    if np.any(resting_mask):
        f["uc_resting_tone"] = float(np.nanmedian(s[resting_mask]))
    else:
        f["uc_resting_tone"] = 0.0

    if len(starts) > 0 and len(ends) > 0:
        if ends[0] < starts[0]:
            ends = ends[1:]

        n_all = min(len(starts), len(ends))
        starts = starts[:n_all]
        ends = ends[:n_all]

        durs = (ends - starts) / fs
        mask = durs > 45.0

        valid_durs = durs[mask]
        n_valid = len(valid_durs)

        rec_min = len(s) / fs / 60

        f["uc_n_contractions"] = int(n_valid)
        f["uc_mean_duration_s"] = float(np.nanmean(valid_durs)) if n_valid > 0 else 0.0
        f["uc_contractions_per_min"] = n_valid / (rec_min + 1e-6)

        valid_starts = starts[mask]
        valid_ends = ends[mask]

        f["uc_starts_idx"] = valid_starts
        f["uc_ends_idx"] = valid_ends

        last_10_min_samples = int(10 * 60 * fs)
        recent_start = max(0, len(s) - last_10_min_samples)
        recent_segment = s[recent_start:]

        valid_fraction_10min = (
            np.mean(~np.isnan(recent_segment)) if len(recent_segment) > 0 else 0.0
        )

        f["uc_valid_fraction_last_10min"] = float(valid_fraction_10min)

        if valid_fraction_10min >= 0.8 and len(valid_starts) > 0:
            recent_mask = valid_starts >= recent_start
            f["uc_contractions_last_10min"] = int(np.sum(recent_mask))
            f["uc_contractions_per_min_last_10min"] = (
                f["uc_contractions_last_10min"] / 10.0
            )
        else:
            f["uc_contractions_last_10min"] = np.nan
            f["uc_contractions_per_min_last_10min"] = np.nan

        auc_list = []

        for start, end in zip(valid_starts, valid_ends):
            contraction_segment = s[start:end] - f["uc_resting_tone"]
            auc = np.nansum(contraction_segment) / fs
            auc_list.append(auc)

        f["uc_total_auc"] = float(np.sum(auc_list)) if auc_list else 0.0
        f["uc_mean_auc"] = float(np.mean(auc_list)) if auc_list else 0.0

    else:
        f["uc_n_contractions"] = 0
        f["uc_mean_duration_s"] = 0.0
        f["uc_contractions_per_min"] = 0.0
        f["uc_total_auc"] = 0.0
        f["uc_mean_auc"] = 0.0
        f["uc_starts_idx"] = np.array([], dtype=int)
        f["uc_ends_idx"] = np.array([], dtype=int)

    return f


# ══════════════════════════════════════════════════════════════════════════════
# DATASET BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def isolate_final_60_minutes_raw(fhr: np.ndarray, uc: np.ndarray, fs: int = FS) -> tuple[np.ndarray, np.ndarray]:
    """
    Isolates the final 60 minutes of the raw FHR and UC recording before feature extraction.
    If the recording is shorter than 60 minutes, it returns the original arrays intact.
    """
    window_duration_samples = 60 * 60 * fs
    total_samples = len(fhr)
    
    if total_samples > window_duration_samples:
        start_idx = total_samples - window_duration_samples
        trimmed_fhr = fhr[start_idx:]
        trimmed_uc = uc[start_idx:] if uc is not None else None
        return trimmed_fhr, trimmed_uc
    
    return fhr, uc


def build_dataset(record_ids: list[int] = RECORD_IDS) -> pd.DataFrame:
    rows = []
    for rid in tqdm(record_ids, desc="Processing records", unit="rec"):
        rec = load_record(rid)
        if rec is None:
            continue

        row: dict = {"record_id": rid}
        row.update(parse_comments(rec.comments))

        sig_upper = [s.upper() for s in rec.sig_name]
        fhr_idx = next((i for i, s in enumerate(sig_upper) if "FHR" in s), None)
        uc_idx = next((i for i, s in enumerate(sig_upper) if "UC" in s), None)
        
        fhr_raw = rec.p_signal[:, fhr_idx] if fhr_idx is not None else None
        uc_raw = rec.p_signal[:, uc_idx] if uc_idx is not None else None

        if fhr_raw is not None:
            fhr_raw, uc_raw = trim_signal_borders(
                fhr=fhr_raw, 
                uc=uc_raw, 
                trim_start=False, #True, 
                trim_end=True, 
                threshold_min=5, 
                fs=FS
            )
            
        # ── ACTIVATE 60-MINUTE ISOLATION HERE ──────────────────────────────
        if fhr_raw is not None and RECORD_60_MIN_TRIM:
            fhr_raw, uc_raw = isolate_final_60_minutes_raw(fhr_raw, uc_raw, fs=FS)
        # ───────────────────────────────────────────────────────────────────
            
        if fhr_raw is not None:
            row.update(extract_fhr_features(fhr_raw))
        if uc_raw is not None:
            row.update(extract_uc_features(uc_raw))

        row["recording_duration_min"] = rec.sig_len / (FS * 60)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: FULL-LENGTH SIGNAL PREPROCESSING & CACHING
# ══════════════════════════════════════════════════════════════════════════════

def extract_full_length_signals(record_id: int):
    record = load_record(record_id)
    if record is None:
        return None

    signals = record.p_signal
    fhr_raw = signals[:, 0]
    uc_raw  = signals[:, 1] if signals.shape[1] > 1 else np.zeros_like(fhr_raw)

    # FHR cleaning (unchanged)
    fhr_clean, _, fhr_final_nan_mask = _clean_fhr(fhr_raw, fs=FS)

    # UC cleaning via shared function
    uc_clean, uc_disconnect_mask, uc_final_nan_mask = clean_uc_signal(uc_raw, fs=FS)

    # Prepare numerical arrays and masks
    fhr_signal_numeric = np.nan_to_num(fhr_clean, nan=0.0)
    uc_signal_numeric  = np.nan_to_num(uc_clean,  nan=0.0)

    fhr_mask_numeric = fhr_final_nan_mask.astype(np.float32)
    uc_mask_numeric  = uc_final_nan_mask.astype(np.float32)

    return np.stack([
        fhr_signal_numeric, 
        uc_signal_numeric, 
        fhr_mask_numeric, 
        uc_mask_numeric
    ], axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING & MODELING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_combined_emergency_category(row):
    t1 = row[TARGET]
    t2 = row[TARGET2]
    t3 = row[TARGET3]

    if (t1 <= TARGET_thresh):
    #if (t2 <= TARGET_thresh2):
    #if (t1 <= TARGET_thresh and t2 <= TARGET_thresh2):
    #if (t1 <= TARGET_thresh and t3 <= TARGET_thresh3):
    #if (t1 <= TARGET_thresh and t2 <= TARGET_thresh2 and t3 <= TARGET_thresh3):
    #if ((t1 <= TARGET_thresh and t2 <= TARGET_thresh2) 
    #    or (t1 <= TARGET_thresh and t3 <= TARGET_thresh3) 
    #    or (t2 <= TARGET_thresh2 and t3 <= TARGET_thresh3)):
        return 1  # Emergency
    else:
        return 0  # Normal

def preprocess_regression(df: pd.DataFrame, max_missing: float = 0.50) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = df.copy()
    
    print("\n" + "═"*60)
    print("        ACTIVE FORENSIC PREPROCESSING REPORT")
    print("═"*60)

    initial_count = len(df)
    #df = df.dropna(subset=[TARGET, TARGET2])
    df = df.dropna(subset=ACTIVE_TARGETS)
    
    dropped_missing_targets = initial_count - len(df)
    if dropped_missing_targets > 0:
        print(f"    -> Dropped {dropped_missing_targets} records due to missing '{TARGET}' or '{TARGET2}' metadata.")

    
    y_class = df.apply(get_combined_emergency_category, axis=1)
    # ── HANDLE NEW NAN ROWS (BORDERLINE / UNCLASSED) ────────────────────────
    valid_mask = y_class.notna()
    df = df[valid_mask].copy()
    y_class = y_class[valid_mask].astype(int)
    # ────────────────────────────────────────────────────────────────────────
    
    post_birth_outcomes = [
        "ph", "bdecf", "pco2", "base_excess", "apgar1",
        "apgar5", "nicu_days", "seizures", "hie", "intubation", "acidosis", "birthweight_g", 
        "delivery_type","stage1_min", "stage2_min","recording_duration_min"
    ] 
    
    post_birth_outcomes = [c for c in post_birth_outcomes if c != TARGET]
    zero_variance_dead_cols = ["main_diag", "other_diag", "sig2birth"]

    df["fetal_sex"]=df["fetal_sex"]-1 #0 = male # 1 = female
    
    drop_list = post_birth_outcomes + zero_variance_dead_cols + ["record_id"]
    drop = [c for c in drop_list if c in df.columns]

    X = df.drop(columns=drop, errors="ignore")
    if TARGET in X.columns:
        X = X.drop(columns=[TARGET])
    
    X = X.select_dtypes(include=[np.number])

    #if "stage1_min" in X.columns:
    #    mask = (X["stage1_min"] < 0) | (X["stage1_min"] > 1440)
    #    X.loc[mask, "stage1_min"] = np.nan
    #if "stage2_min" in X.columns:
    #    mask = X["stage2_min"] < 0
    #    X.loc[mask, "stage2_min"] = np.nan
    if "fhr_sampen" in X.columns:
        mask = X["fhr_sampen"] < 0
        X.loc[mask, "fhr_sampen"] = np.nan

    X = X.loc[:, X.isnull().mean() < max_missing]

    common_index = X.index
    X = X.loc[common_index]
    y_class = y_class.loc[common_index]
    df_filtered = df.loc[common_index]

    print(f"    -> Final dataset shape: {X.shape}")
    print("═"*60 + "\n")

    return X, y_class, df_filtered


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── 1. Per-record features (cached in OUTPUT_CSV) ────────────────────────
    if Path(OUTPUT_CSV).exists():
        print(f"[INFO] Loading cached features from '{OUTPUT_CSV}'")
        df_raw = pd.read_csv(OUTPUT_CSV)
    else:
        print("[INFO] Extracting features from all records …")
        df_raw = build_dataset()
        df_raw.to_csv(OUTPUT_CSV, index=False)
        print(f"[INFO] Saved → '{OUTPUT_CSV}'  ({len(df_raw)} records, {df_raw.shape[1]} cols)")

    # ── 2. Signal-quality filtering: drop flat-line recordings ───────────────
    df_raw = df_raw[(df_raw["fhr_missing_pct"] <= CTG_thresh) & (df_raw["uc_missing_pct"] <= CTG_thresh)].copy()

    uc_flat_mask = df_raw["uc_std"] <= 0.01    # entire UC channel is flat
    fhr_flat_mask = df_raw["fhr_std"] <= 0.1   # entire FHR channel is flat
    df_clean = df_raw[~uc_flat_mask & ~fhr_flat_mask].copy()
    print(f"[AUDIT] Records: {len(df_raw)} -> {len(df_clean)} after removing flat signals")

    # ── 3. Labor-stage selection ─────────────────────────────────────────────
    if STAGE_SELECT == "both":
        df = df_clean[(df_clean["stage1_min"] > 0) & (df_clean["stage2_min"] > 0)].copy()
    elif STAGE_SELECT == "stage1":
        df = df_clean[(df_clean["stage1_min"] > 0)].copy()
    elif STAGE_SELECT == "either":
        df = df_clean[(df_clean["stage1_min"] > 0) | (df_clean["stage2_min"] > 0)].copy()

    df.drop(columns=["pos_stage2"], inplace=True, errors="ignore")

    # Negative values are placeholders for "unknown"
    for col in ["stage1_min", "stage2_min", "pos_stage2"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    df["stage"] = np.where(df["stage2_min"] > 0, 1, 0)

    # ── 4. Outlier removal ───────────────────────────────────────────────────
    if EXCLUDE_OUTLIERS:
        n_before = len(df)
        df = df[~df["record_id"].isin(OUTLIER_RECORD_IDS)].copy()
        print(f"[OUTLIERS] Removed {n_before - len(df)} records → {len(df)} remaining")

    for col in ["ph", "base_excess", "apgar5"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if DROP_MISSINGNESS_FEATURES:
        df = df.drop(columns=MISSINGNESS_FEATURES, errors="ignore")

    # ── 5. Labels + leakage-safe predictor matrix ───────────────────────────
    X, y_class, df_filtered = preprocess_regression(df)

    leakage_watchlist = [
        "ph", "base_excess", "bdecf", "pco2",
        "apgar1", "apgar5", "nicu_days", "seizures", "hie", "intubation",
        "acidosis", "birthweight_g", "delivery_type",
    ]
    leaked = [c for c in leakage_watchlist if c in X.columns]
    if leaked:
        raise ValueError(f"!!! LEAKAGE DETECTED !!! These columns must not be in X: {leaked}")
    print("[OK] No label-defining / post-birth columns in X.")

    # ── 6. Remove redundant (highly correlated) features ────────────────────
    X, cluster_report, Z = select_uncorrelated_features(
        X,
        method="spearman",
        corr_threshold=0.85,
        tiebreak="missingness",
        plot=False,
    )

    print(f"  • Records          : {len(X)}")
    print(f"  • Class 0 (Normal) : {(y_class == 0).sum()}")
    print(f"  • Class 1 (Acidemia): {(y_class == 1).sum()}")

    # ── 7. Save tabular features for STEP 3 ─────────────────────────────────
    final_df = X.copy()
    final_df["record_id"] = df_filtered["record_id"].values
    final_df["y_class"] = y_class.values
    final_df.to_csv(FINAL_FEATURES_CSV, index=False)
    print(f"[CACHE] Saved tabular features → '{FINAL_FEATURES_CSV}' "
          f"({final_df.shape[0]} rows, {final_df.shape[1]} cols)")

    # ── 8. Save full-length cleaned signals for STEP 2 ──────────────────────
    if not os.path.exists(FULL_CACHE_PATH):
        print("[INFO] Cache not found. Starting signal extraction...")
        save_dict = {}
        valid_ids, skipped = [], []

        for rid in tqdm(df_filtered["record_id"], desc="Caching signals"):
            signal_matrix = extract_full_length_signals(int(rid))
            if signal_matrix is None:
                skipped.append(rid)
                continue
            save_dict[f"rec_{int(rid)}"] = signal_matrix
            valid_ids.append(int(rid))

        save_dict["record_ids"] = np.array(valid_ids, dtype=np.int32)
        np.savez_compressed(FULL_CACHE_PATH, **save_dict)

        print(f"[CACHE] Saved {len(valid_ids)}/{len(df_filtered)} records → '{FULL_CACHE_PATH}'")
        if skipped:
            print(f"[CACHE] Skipped {len(skipped)} missing record(s): {skipped}")
    else:
        print(f"[CACHE] '{FULL_CACHE_PATH}' already exists. Skipping extraction.")

    print("\n[DONE] Upload the two output files to Google Drive for steps 2 and 3.")
