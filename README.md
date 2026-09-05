# HSI Compression Benchmark - Methods

Current benchmark lineup in this repository:

- Deep learning: `LSSIR`, `FHNeRF`, `Finer`, `HINER++`
- Traditional: `JPEG`, `JPEG2000`, `AVC`, `HEVC`

Notes:

- `Siren` is retained on disk only for `LSSIR` ablation and paper-side baseline analysis.
- `MGIR` is kept on disk as an exploratory baseline but is not part of the final benchmark lineup.
- `methods/deep_learning/HINER/hiner/` is the author-code snapshot; `methods/deep_learning/HINER/` contains the benchmark-native wrapper used by this repository.

## Directory Structure

```text
methods/
├── README.md
├── deep_learning/
│   ├── LSSIR/
│   ├── FHNeRF/
│   ├── Finer/
│   ├── HINER/
│   ├── Siren/              # Ablation only
│   └── MGIR/               # Not part of the final benchmark lineup
├── traditional/
│   ├── jpeg.py
│   ├── jpeg2k.py
│   ├── avc.py
│   └── hevc.py
└── scripts/
    ├── quick_test.sh
    ├── test_all_inr.sh
    ├── tune_brain_baselines.sh
    └── select_brain_baselines.py
```

## Recommended Entry Points

Run each model from its own directory so presets and output paths stay self-contained:

- `deep_learning/LSSIR/main.py`
- `deep_learning/FHNeRF/main.py`
- `deep_learning/Finer/main.py`
- `deep_learning/HINER/main.py` or `deep_learning/HINER/run_presets.sh`
- `traditional/*.py` for codec baselines

## Current Status

- Top-level benchmark documentation now assumes the final deep lineup is `Finer / FHNeRF / HINER++ / LSSIR`.
- The older method-wide orchestration docs that referenced `Siren` as part of the main benchmark are no longer valid.
- Some helper scripts in `methods/scripts/` are legacy utilities from the earlier Siren-centered comparison phase; use them only if their assumptions match your current experiment.
