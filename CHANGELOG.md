# Changelog

## v1.1.0 (2026-11-11)

- Figures 1 and 2 redesigned. Figure 1 (intraday) now shows both configurations in a single panel: for each configuration, the range between the lowest and highest error of the six algorithms and the RF line, together with smart persistence and climatology. Figure 2 (day-ahead) shows, per configuration, the same range and RF line by target hour.
- Two prediction files (`dayahead_nwp_SVR.csv.gz`, `intraday_nwp_RandomForest.csv.gz`) moved from `results/` to `results/predictions/`, where `03_results.ipynb` expects them. In v1.0.0 the notebook stopped at the integrity check when run from a fresh clone.
- `03_results.ipynb` re-executed and stored with its outputs.

No change to the code in `src/`, the data or the numerical results: all 128 result tables produced by `03_results.ipynb` are identical to v1.0.0.

## v1.0.0 (2026-11-10)

First release.
