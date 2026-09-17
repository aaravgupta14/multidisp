import sys, os, warnings, json
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_core import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
import numpy as np, pandas as pd

RNG = np.random.default_rng(42)

# --- build pipeline (same as validate.py) ---
tr = []
for f in TRAIN_FILES:
    tr += contiguous_trips(load_month(f))
tr, seen = deduplicate(tr)
CN, _ = identify_capacity(tr)
te, _ = deduplicate(contiguous_trips(load_month(TEST_FILE)), blocklist=seen)
TRAIN = pd.concat(tr, ignore_index=True)
RF = RandomForestRegressor(n_estimators=60, max_depth=9, random_state=42, n_jobs=-1)
RF.fit(TRAIN[FEATURES], TRAIN[TARGET])

nom = np.concatenate([dual_observer(RF.predict(w[FEATURES]), w['Pack Current (A)'].values,
                                     w[TARGET].values[0], np.inf, 0.0, cn=CN)['Residual']
                       for w in sliding_windows(tr)])
TAU = float(np.percentile(nom, PCTL))
LAM = float(np.log(2) / TAU)
tau_c, lam_c = calibrate_channel_thresholds(RF, tr, CN)
print('Detection TAU=%.3f  channel tau_c=%s' % (TAU, tau_c))

ATTACK_TYPES = [
    ('Voltage bias',                 'V', 'bias'),
    ('Voltage drift',                'V', 'drift'),
    ('Voltage stochastic noise',     'V', 'stochastic'),
    ('Current bias',                 'I', 'bias'),
    ('Current drift',                'I', 'drift'),
    ('Temperature manipulation',     'T', 'bias'),
]
AMP_RANGE = {'V': (0.5, 3.0), 'I': (8.0, 45.0), 'T': (1.5, 8.0)}
KAPPA_RANGE = (0.008, 0.05)
WIN = 1800
N_EPISODES = 25


def make_attack(n, onset, kind, amp, kappa, seed):
    rng = np.random.default_rng(seed)
    d = np.zeros(n)
    if kind == 'bias':
        d[onset:] = amp + rng.normal(0, amp * 0.03, n - onset)
    elif kind == 'drift':
        kk = np.arange(n - onset)
        d[onset:] = amp * (1 - np.exp(-kappa * kk)) + rng.normal(0, amp * 0.02, n - onset)
    elif kind == 'stochastic':
        d[onset:] = rng.normal(0, amp, n - onset)
    return d


def run_episode(base_window, channel, kind, ep_seed):
    n = len(base_window)
    lo_on = min(300, n // 4)
    hi_on = max(lo_on + 1, n - max(200, n // 4))
    onset = int(RNG.integers(lo_on, hi_on))
    lo, hi = AMP_RANGE[channel]
    amp = float(np.exp(RNG.uniform(np.log(lo), np.log(hi))))
    kappa = float(RNG.uniform(*KAPPA_RANGE))
    d = make_attack(n, onset, kind, amp, kappa, ep_seed)

    attacked = base_window.copy()
    attacked[CHANNEL_COL[channel]] = base_window[CHANNEL_COL[channel]].values + d
    I_for_physics = attacked['Pack Current (A)'].values   # attacked I if I is the target, true I otherwise

    soc_ml = RF.predict(attacked[FEATURES])
    obs = dual_observer(soc_ml, I_for_physics, base_window[TARGET].values[0], TAU, LAM, cn=CN)

    feats = attacked[FEATURES].copy()
    refs = {c: rolling_reference(attacked[CHANNEL_COL[c]].values) for c in 'VIT'}
    D, _ = channel_diagnostics(RF, I_for_physics, refs, feats, CN)

    post_flags = obs['Flag'][onset:]
    detected = bool(post_flags.any())
    delay = int(np.argmax(post_flags)) if detected else None

    loc_ok = ident_ok = False
    if detected:
        k_det = onset + delay
        k_end = min(n, k_det + 20)
        score = {c: float(np.median(D[c][k_det:k_end]) / tau_c[c]) for c in 'VIT'}
        winner = max(score, key=score.get)
        loc_ok = (winner == channel)
        if loc_ok:
            win_len = min(300, n - onset)
            sub = classify_subtype(D[channel][onset:onset + win_len])
            ident_ok = (sub == kind)
        else:
            ident_ok = False
    else:
        ident_ok = False

    y_true = (np.arange(n) >= onset).astype(int)
    y_pred = obs['Flag'].astype(int)
    return dict(detected=detected, delay=delay, loc_ok=loc_ok, ident_ok=ident_ok,
                y_true=y_true, y_pred=y_pred)


results = {}
for name, channel, kind in ATTACK_TYPES:
    eps = []
    seed_ctr = 1000 * (hash((channel, kind)) % 97)
    for i in range(N_EPISODES):
        base = te[i % len(te)]
        n = len(base)
        if n <= WIN + 800:
            w = base.reset_index(drop=True)
        else:
            a0 = int(RNG.integers(0, n - WIN))
            w = base.iloc[a0:a0 + WIN].reset_index(drop=True)
        eps.append(run_episode(w, channel, kind, seed_ctr + i))

    det = np.mean([e['detected'] for e in eps]) * 100
    loc = np.mean([e['loc_ok'] for e in eps]) * 100
    idn = np.mean([e['ident_ok'] for e in eps]) * 100
    delays = [e['delay'] for e in eps if e['delay'] is not None]
    delay_mean = float(np.mean(delays)) if delays else float('nan')
    y_true = np.concatenate([e['y_true'] for e in eps])
    y_pred = np.concatenate([e['y_pred'] for e in eps])
    f1 = f1_score(y_true, y_pred) * 100

    results[name] = dict(detection=det, identification=idn, localization=loc,
                          f1=f1, delay=delay_mean, n=len(eps))
    print('%-28s det=%5.1f%%  ident=%5.1f%%  loc=%5.1f%%  F1=%5.1f%%  delay=%.1f (n=%d)'
          % (name, det, idn, loc, f1, delay_mean, len(eps)))

RESULTS_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'table2_results.json')
with open(RESULTS_OUT, 'w') as f:
    json.dump(results, f, indent=2)
print('\nsaved table2_results.json')
