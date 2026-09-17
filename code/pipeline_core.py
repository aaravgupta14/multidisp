"""
Reconstructed cyber-resilient SoC estimation pipeline, built directly from the equations
and design decisions already specified in the manuscript (Section 7.3, Equations 1-9).
The original implementation was lost; this is a clean re-derivation, not a copy.
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
TRAIN_FILES = ['feb_bms_dli.csv', 'March_Merged_All_CSV_Files.csv', 'May_combined_excel_data.csv']
TEST_FILE = 'JuneMerged_CSV_Files.csv'

FEATURES = ['Pack Voltage (V)', 'Pack Current (A)', 'Cell Temp 1 (C)']
TARGET = 'Pack SoC (%)'
ALL_COLS = ['Pack Voltage (V)', 'Pack Current (A)', 'Cell Temp 1 (C)', 'Motor Speed (rpm)',
            'Pack SoC (%)', 't_seconds', 'Key Status', 'Source_File']

DT = 0.5           # nominal logger period, seconds (confirmed via t_seconds diffs in this dataset)


def remove_isolated_glitches(df, col, jump_thresh):
    """Drop single-row sensor glitches: a sample that jumps far from BOTH neighbors while
    those neighbors agree closely with each other (a real physical excursion would not
    snap back to the pre-jump trend one sample later)."""
    x = df[col].values
    if len(x) < 3:
        return df
    d_prev = np.abs(np.diff(x, prepend=x[0]))
    d_next = np.abs(np.diff(x, append=x[-1]))
    neighbor_gap = np.abs(np.r_[x[2:], x[-1], x[-1]] - np.r_[x[0], x[0], x[:-2]])
    is_glitch = (d_prev > jump_thresh) & (d_next > jump_thresh) & (neighbor_gap < jump_thresh)
    is_glitch[0] = is_glitch[-1] = False
    return df[~is_glitch].reset_index(drop=True)


def load_month(fname):
    path = os.path.join(DATA_DIR, fname)
    cols = pd.read_csv(path, nrows=0).columns
    use = [c for c in ALL_COLS if c in cols]
    df = pd.read_csv(path, low_memory=False, usecols=use)
    for c in ALL_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df.dropna(subset=['Pack Voltage (V)', 'Pack Current (A)', 'Cell Temp 1 (C)', 'Pack SoC (%)'])
    df = df[(df['Pack SoC (%)'] >= 0) & (df['Pack SoC (%)'] <= 100) & (df['Pack Voltage (V)'] > 25)]
    if 'Key Status' in df.columns:
        df = df[df['Key Status'] != 0]
    df = df.reset_index(drop=True)
    df = remove_isolated_glitches(df, TARGET, jump_thresh=5.0)
    return df.reset_index(drop=True)


def _gap_split(df, max_gap_s):
    """Split wherever the logger clock jumps forward by more than max_gap_s, or does not
    advance at all (t_seconds resets to ~0 at the start of a new recording session, which
    shows up as a large *negative* diff -- a forward-only gap check misses this entirely
    and silently splices a charge event between two unrelated sessions into one "trip")."""
    if 't_seconds' not in df.columns or df['t_seconds'].isna().all():
        return [df]
    t = df['t_seconds'].values
    d = np.diff(t)
    breaks = np.where((d > max_gap_s) | (d <= 0))[0] + 1
    return np.split(df, breaks)


def contiguous_trips(df, max_gap_s=5.0, min_len=200):
    """Split a month's log into contiguous driving trips.

    When the file carries a 'Source_File' column (June only, one entry per calendar-day
    logger file), that is the authoritative trip boundary: t_seconds is not a single
    monotonic clock across different daily files, so a global time-gap split across the
    whole month can silently splice two unrelated days together (or fragment one day) and
    corrupt a downstream Coulomb-counting integral. Where Source_File is unavailable
    (Feb/Mar/May), a within-file time-gap split is used, which produced physically
    consistent per-trip capacity estimates on this dataset (68.6-74.2 Ah).
    """
    if 'Source_File' in df.columns and df['Source_File'].notna().any():
        groups = [g for _, g in df.groupby('Source_File', sort=False)]
        segs = []
        for g in groups:
            segs.extend(_gap_split(g, max_gap_s))
    else:
        segs = _gap_split(df, max_gap_s)
    out = []
    for i, s in enumerate(segs):
        s = s.reset_index(drop=True)
        if len(s) >= min_len:
            s.attrs['trip'] = '%d' % i
            out.append(s)
    return out


def content_hash(df):
    return hash(tuple(np.round(df['Pack Voltage (V)'].values[:50], 3)))


def deduplicate(trips, blocklist=None):
    seen = set() if blocklist is None else set(blocklist)
    kept = []
    for t in trips:
        h = content_hash(t)
        if h in seen:
            continue
        seen.add(h)
        kept.append(t)
    return kept, seen


def trip_capacity_ah(trip):
    """Solve for the cell capacity implied by a single trip's own Coulomb-counting
    self-consistency: integrated current draw over the observed SoC drop."""
    I = trip['Pack Current (A)'].values
    soc = trip[TARGET].values
    dsoc = soc[-1] - soc[0]
    if abs(dsoc) < 3.0:
        return np.nan
    ah = -np.trapezoid(I, dx=DT) / 3600.0     # +Ah delivered by the pack (I<0 is discharge)
    cn = ah / (-dsoc / 100.0) if dsoc < 0 else np.nan
    return cn if cn is not None and 20 < cn < 200 else np.nan


def identify_capacity(trips):
    vals = np.array([trip_capacity_ah(t) for t in trips])
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)), vals


def trip_dt(trip):
    return DT


# --------------------------------------------------------------------------- Eq. (1)-(5), (7)-(9)
ALPHA_NOMINAL = 0.96
PCTL = 99.5


def physics_path(I, soc0, cn, dt=DT):
    """Eq. (1): open-loop Coulomb-counting reference. Eq. (1) is written for a
    discharge-positive current convention; this dataset logs discharge current as
    negative, so the sign is flipped here to apply the same physical law correctly."""
    soc = np.empty(len(I))
    prev = soc0
    for k in range(len(I)):
        prev = prev + I[k] * dt / (cn * 3600.0) * 100.0
        soc[k] = prev
    return soc


def dual_observer(soc_ml, I, soc0, tau, lam, cn, dt=DT, alpha_nominal=ALPHA_NOMINAL,
                   trust_prev_seed=1.0):
    """Eq. (1)-(5): physics path, trust-weighted convex fusion, monotonicity clamp under
    discharge, non-parametric residual thresholding."""
    n = len(soc_ml)
    soc_phys = np.empty(n)
    soc_fused = np.empty(n)
    residual = np.empty(n)
    trust = np.empty(n)
    flag = np.zeros(n, dtype=bool)

    phys_prev = soc0
    fused_prev = soc0
    for k in range(n):
        phys_prev = phys_prev + I[k] * dt / (cn * 3600.0) * 100.0
        phys_k = phys_prev
        ml_k = soc_ml[k]

        r = abs(phys_k - ml_k)                       # Eq. (3), magnitude form (Eq. 8/9 note)
        t_v = float(np.exp(-lam * r))                 # Eq. (4)
        is_flag = r > tau                              # Eq. (5)

        alpha = alpha_nominal * t_v                    # alpha(k) = alpha_nominal * T_V(k)
        fused_k = alpha * ml_k + (1.0 - alpha) * phys_k

        if I[k] < -0.5:                                 # discharging: enforce monotonicity
            phys_k = min(phys_prev, phys_k)
            fused_k = min(fused_prev, fused_k)

        soc_phys[k] = phys_k
        soc_fused[k] = fused_k
        residual[k] = r
        trust[k] = t_v
        flag[k] = is_flag

        phys_prev = phys_k
        fused_prev = fused_k

    return {'SoC': soc_fused, 'Physics': soc_phys, 'Residual': residual,
            'Trust': trust, 'Flag': flag}


def sliding_windows(trips, win=1000, stride=1000):
    for t in trips:
        n = len(t)
        for a in range(0, max(1, n - win + 1), stride):
            yield t.iloc[a:a + win].reset_index(drop=True)


# ---------------------------------------------------------- Section 7.3.8: identification/localization
CHANNEL_COL = {'V': 'Pack Voltage (V)', 'I': 'Pack Current (A)', 'T': 'Cell Temp 1 (C)'}


def rolling_reference(x, window=20):
    """A causal, online-computable 'locally expected' value for a channel: the median of
    its own last `window` samples. A real embedded system can compute this as a cheap
    rolling filter; it stands in for 'what this channel would read under undisturbed
    conditions' without needing the true clean signal at runtime."""
    ref = pd.Series(x).rolling(window, min_periods=1).median().shift(1)
    return ref.bfill().values


def channel_features(w):
    feats = w[FEATURES].copy()
    refs = {c: rolling_reference(w[CHANNEL_COL[c]].values) for c in 'VIT'}
    return feats, refs


def channel_diagnostics(RF, I, refs, feats, cn):
    """D_V, D_I, D_T (Eq. 8-style per-channel deviation features): the marginal shift in
    the ML prediction if only that one channel were replaced by its recent, locally
    expected value. Current additionally gets its instantaneous marginal contribution to
    the physics/Coulomb-counting path, since it is the one channel that feeds both
    observers (Section 7.3.8's stated reason its trust reduction dominates)."""
    base_pred = RF.predict(feats)
    D = {}
    for c in 'VIT':
        alt = feats.copy()
        alt[CHANNEL_COL[c]] = refs[c]
        D[c] = np.abs(base_pred - RF.predict(alt))
    phys_marg = np.abs(I - refs['I']) * DT / (cn * 3600.0) * 100.0
    D['I'] = D['I'] + phys_marg
    return D, base_pred


def calibrate_channel_thresholds(RF, trips, cn):
    Dall = {'V': [], 'I': [], 'T': []}
    for w in sliding_windows(trips):
        feats, refs = channel_features(w)
        I = w['Pack Current (A)'].values
        D, _ = channel_diagnostics(RF, I, refs, feats, cn)
        for c in 'VIT':
            Dall[c].append(D[c])
    tau_c = {c: float(np.percentile(np.concatenate(Dall[c]), PCTL)) for c in 'VIT'}
    lam_c = {c: float(np.log(2) / max(tau_c[c], 1e-6)) for c in 'VIT'}
    return tau_c, lam_c


def classify_subtype(residual_window):
    """Bias vs drift vs stochastic, from the shape of the residual after attack onset:
    a bias jumps once and plateaus (near-zero trend, low roughness); a bounded-exponential
    drift (Eq. 7) rises smoothly with positive trend; anything rougher/non-monotonic is
    scored as stochastic perturbation."""
    r = np.asarray(residual_window, dtype=float)
    n = len(r)
    if n < 5 or r.max() < 1e-9:
        return 'bias'
    x = np.arange(n)
    slope = np.polyfit(x, r, 1)[0]
    roughness = np.mean(np.abs(np.diff(r, 2))) / (np.mean(r) + 1e-9) if n > 2 else 0.0
    norm_slope = slope * n / (np.mean(r) + 1e-9)
    if roughness > 0.9:
        return 'stochastic'
    if norm_slope > 0.6:
        return 'drift'
    return 'bias'
