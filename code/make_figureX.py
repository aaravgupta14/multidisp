import sys, warnings
warnings.filterwarnings('ignore')
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_core import *
from sklearn.ensemble import RandomForestRegressor
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RNG = np.random.default_rng(7)

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

PANELS = [
    ('Voltage bias',             'V', 'bias',       1.6),
    ('Voltage drift',            'V', 'drift',      1.8),
    ('Voltage stochastic noise', 'V', 'stochastic', 1.4),
    ('Current bias',             'I', 'bias',       25.0),
    ('Current drift',            'I', 'drift',      30.0),
    ('Temperature manipulation', 'T', 'bias',       5.0),
]


def make_attack(n, onset, kind, amp, kappa, seed):
    rng = np.random.default_rng(seed)
    d = np.zeros(n)
    if kind == 'bias':
        d[onset:] = amp + rng.normal(0, amp * 0.03, n - onset)
    elif kind == 'drift':
        kk = np.arange(n - onset)
        d[onset:] = amp * (1 - np.exp(-0.02 * kk)) + rng.normal(0, amp * 0.02, n - onset)
    elif kind == 'stochastic':
        d[onset:] = rng.normal(0, amp, n - onset)
    return d


fig, axes = plt.subplots(3, 2, figsize=(11, 11.5))
axes = axes.ravel()
COLORS = {'V': '#1f77b4', 'I': '#ff7f0e', 'T': '#2ca02c'}

for idx, (title, channel, kind, amp) in enumerate(PANELS):
    base = te[idx % len(te)].reset_index(drop=True)
    n = min(len(base), 1800)
    w = base.iloc[:n].reset_index(drop=True)
    onset = max(150, n // 3)
    d = make_attack(n, onset, kind, amp, 0.02, seed=100 + idx)

    attacked = w.copy()
    attacked[CHANNEL_COL[channel]] = w[CHANNEL_COL[channel]].values + d
    I_for_physics = attacked['Pack Current (A)'].values
    soc_ml = RF.predict(attacked[FEATURES])
    obs = dual_observer(soc_ml, I_for_physics, w[TARGET].values[0], TAU, LAM, cn=CN)

    feats = attacked[FEATURES].copy()
    refs = {c: rolling_reference(attacked[CHANNEL_COL[c]].values) for c in 'VIT'}
    D, _ = channel_diagnostics(RF, I_for_physics, refs, feats, CN)

    ax = axes[idx]
    t = np.arange(n)
    ax.plot(t, obs['Residual'], color='black', linewidth=1.1, label=r'Innovation residual $r_{SOC}(k)$')
    ax.axhline(TAU, color='red', linestyle='--', linewidth=1.0, label=r'Detection threshold $\tau$')
    ax.axvspan(onset, n, color='red', alpha=0.05)
    ax.axvline(onset, color='gray', linestyle=':', linewidth=1.0)

    ax2 = ax.twinx()
    for c in 'VIT':
        score = D[c] / tau_c[c]
        ax2.plot(t, score, color=COLORS[c], linewidth=1.0, alpha=0.75,
                 label=r'$D_%s(k)/\tau_%s$' % (c, c))
    ax2.axhline(1.0, color='dimgray', linestyle=':', linewidth=0.8)
    ax2.set_ylim(0, max(3.0, np.percentile(np.concatenate([D[c]/tau_c[c] for c in 'VIT']), 99) * 1.1))

    ax.set_title('%s (%s channel)' % (title, channel), fontsize=10)
    ax.set_xlabel('Time Horizon (Samples)', fontsize=8)
    ax.set_ylabel('Residual Magnitude', fontsize=8)
    ax2.set_ylabel('Normalized channel score', fontsize=8)
    ax.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7)

from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color='black', linewidth=1.1, label=r'Innovation residual $r_{SOC}(k)$'),
    Line2D([0], [0], color='red', linestyle='--', linewidth=1.0, label=r'Detection threshold $\tau$'),
    Line2D([0], [0], color=COLORS['V'], linewidth=1.0, label=r'$D_V(k)/\tau_V$ (voltage score)'),
    Line2D([0], [0], color=COLORS['I'], linewidth=1.0, label=r'$D_I(k)/\tau_I$ (current score)'),
    Line2D([0], [0], color=COLORS['T'], linewidth=1.0, label=r'$D_T(k)/\tau_T$ (temperature score)'),
]
fig.legend(handles=legend_handles, loc='upper center', ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, 1.0), frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.93])
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'figure15_residual_trust_scores.png')
plt.savefig(OUT, dpi=200, facecolor='white')
print('saved', OUT)
