"""
Trains the SoC estimator on Feb/Mar/May and calibrates the physics-fusion pipeline
exactly as described in the paper (Eq. 1-9), then saves everything needed to
reproduce the framework's decisions without retraining:

  models/soc_estimator_rf.joblib   - the trained RandomForestRegressor
  models/pipeline_config.json      - calibrated capacity, thresholds, lambdas

Run from anywhere; paths are resolved relative to this file.
"""
import os
import sys
import json
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

from pipeline_core import (
    TRAIN_FILES, FEATURES, TARGET, PCTL,
    load_month, contiguous_trips, deduplicate, identify_capacity,
    dual_observer, sliding_windows, calibrate_channel_thresholds,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    trips = []
    for f in TRAIN_FILES:
        trips += contiguous_trips(load_month(f))
    trips, _ = deduplicate(trips)
    cn, cn_vals = identify_capacity(trips)
    print('Identified pack capacity: %.2f Ah (n=%d trips, range [%.2f, %.2f])'
          % (cn, len(cn_vals), cn_vals.min(), cn_vals.max()))

    train_df = pd.concat(trips, ignore_index=True)
    rf = RandomForestRegressor(n_estimators=60, max_depth=9, random_state=42, n_jobs=-1)
    rf.fit(train_df[FEATURES], train_df[TARGET])
    print('Trained RandomForestRegressor on %d rows' % len(train_df))
    print('Feature importances:', dict(zip(FEATURES, rf.feature_importances_.round(4))))

    nominal = np.concatenate([
        dual_observer(rf.predict(w[FEATURES]), w['Pack Current (A)'].values,
                       w[TARGET].values[0], np.inf, 0.0, cn=cn)['Residual']
        for w in sliding_windows(trips)
    ])
    tau = float(np.percentile(nominal, PCTL))
    lam = float(np.log(2) / tau)
    print('Detection threshold tau=%.4f, lambda=%.4f' % (tau, lam))

    tau_c, lam_c = calibrate_channel_thresholds(rf, trips, cn)
    print('Per-channel thresholds:', tau_c)

    model_path = os.path.join(MODELS_DIR, 'soc_estimator_rf.joblib')
    joblib.dump(rf, model_path)
    print('Saved model to', model_path)

    config = {
        'pack_capacity_ah': cn,
        'detection_threshold_tau': tau,
        'trust_lambda': lam,
        'channel_thresholds_tau_c': tau_c,
        'channel_lambdas_lam_c': lam_c,
        'features': FEATURES,
        'target': TARGET,
        'rf_params': {'n_estimators': 60, 'max_depth': 9, 'random_state': 42},
    }
    config_path = os.path.join(MODELS_DIR, 'pipeline_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print('Saved config to', config_path)


if __name__ == '__main__':
    main()
