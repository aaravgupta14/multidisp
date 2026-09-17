# multidisp

Physics-grounded, cyber-resilient battery state-of-charge (SoC) estimation framework for
electric vehicles, validated on real multi-month fleet telemetry and stress-tested against
sensor-spoofing attacks on voltage, current, and temperature channels.

## Contents

- **`data/`** - raw monthly BMS telemetry CSVs (Feb, Mar, May for training; June held out as an
  out-of-distribution test month with highway driving and thermal stress).
- **`models/`** - the trained SoC estimator (`soc_estimator_rf.joblib`) and its calibrated
  runtime configuration (`pipeline_config.json`: pack capacity, detection threshold,
  per-channel identification thresholds), plus the attack-identification benchmark results
  and figure.
- **`code/`** - the reconstructed estimation pipeline (`pipeline_core.py`), the training
  script that produces the artifacts in `models/` (`train_and_save_model.py`), and the
  scripts that generated the cybersecurity-layer benchmark and figure.
- **`paper/`** - the manuscript (`Manuscript_FINAL.docx`) and the earlier pre-review draft
  (`Manuscript_original.docx`).

## Reproducing the model

```bash
cd code
pip install pandas numpy scikit-learn joblib
python train_and_save_model.py
```

This retrains the RandomForest SoC estimator on Feb/Mar/May, re-identifies the pack
capacity via Coulomb-counting self-consistency, and recalibrates the detection and
per-channel identification thresholds, overwriting the files in `models/`.

## Method summary

The framework fuses a data-driven (RandomForest) SoC estimator with an open-loop
Coulomb-counting physics reference through an adaptive, trust-weighted convex combination.
An online innovation-residual detector (non-parametric 99.5th-percentile threshold) flags
sensor manipulation; a lightweight residual-and-trust signature then attributes a confirmed
anomaly to the most likely compromised channel (voltage, current, or temperature) without
disconnecting unaffected sensors. Full derivation, equations, and results are in
`paper/Manuscript_FINAL.docx`.
