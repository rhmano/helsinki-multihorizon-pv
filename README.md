# Intraday and day-ahead PV power forecasting with an operational regional weather forecast (Helsinki-Kumpula)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24119837.svg)](https://doi.org/10.5281/zenodo.24119837)

Code, data and results for the article

> Rahmanu, M., Karakaya, A. *İşletimsel Bölgesel Hava Tahmininin Gün İçi ve Ertesi Gün Fotovoltaik Güç Tahminine Katkısı: Helsinki Örneğinde Altı Makine Öğrenmesi Algoritmasının Karşılaştırılması* (under review).

Six algorithms (MLR, KNN, SVR, random forest, XGBoost, LSTM) are compared on the Helsinki-Kumpula PV system (2016–2021) in two tasks — intraday (1–6 h ahead, issued hourly) and day-ahead (issued at 10:00 UTC for the 24 hours of the next day) — with and without the MEPS numerical weather prediction. Models are trained with an expanding window and evaluated on 2019, 2020 and 2021 against smart persistence and climatology; differences are tested with Diebold–Mariano tests and Holm correction.

## Reproduce the article without re-running the experiment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rhmano/helsinki-multihorizon-pv/blob/main/notebooks/03_results.ipynb)

`notebooks/03_results.ipynb` rebuilds every table, statistical test and figure from the saved predictions in `results/`. It trains nothing and runs on a CPU in a few minutes. The notebook is stored with its outputs, so the results can also be read directly on GitHub.

Locally:

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook 03_results.ipynb
```

The outputs were verified with Python 3.12, numpy 2.5, pandas 3.0, scipy 1.18, scikit-learn 1.9 and matplotlib 3.11.

## Repository structure

```
src/multistep_core.py           core module (features, models, evaluation, tests)
notebooks/
  01_data_preparation.ipynb     raw FMI minute data → hourly data set (2016–2021)
  01b_meps_extraction.ipynb     MEPS forecasts at the station (2017–2021)
  02_experiment.ipynb           training of all models (GPU recommended for LSTM)
  03_results.ipynb              tables, tests and figures of the article
data/
  Helsinki_MultiYear.csv        hourly data set
  Helsinki_dataQC.csv           daily FMI quality flag
  nwp/meps_helsinki_00Z.csv     MEPS 00 UTC forecasts at the nearest grid point
results/
  results_*.csv                 metrics per task, configuration, algorithm, fold and horizon
  predictions/*.csv.gz          every prediction (issue time, horizon, fold, seed)
  tuning/                       nested tuning results per fold
```

## Where the article numbers come from

| Article | File in `results/` |
|---|---|
| Table 3 (nRMSE and skill, all horizons) | `pooled_skill_{task}_{config}.csv` |
| Solar-elevation sensitivity | `pooled_skill_{task}_{config}_elev10.csv` |
| Best algorithm and indistinguishable ones | `best_set_{task}_{config}.csv`, `dm_pairs_{task}_{config}.csv` |
| Contribution of MEPS (nRMSE reduction, DM test) | `dm_nwp_gain_{task}.csv` |
| Reference choice | `reference_choice_{task}.csv` |
| Selected hyperparameters | `tuning_summary.csv` |
| Figures 1 and 2 | `Fig1_intraday_nRMSE.png`, `Fig2_dayahead_nRMSE.png` |

For the LSTM, each of the five seeds is tested separately and the largest p-value is reported; the nRMSE reduction in `dm_nwp_gain_*.csv` belongs to that seed.

## Rebuilding everything from scratch

1. Download the FMI data set and place the Helsinki file at `data/raw/FMI_Helsinki_PV.csv`; run `01_data_preparation.ipynb`.
2. Run `01b_meps_extraction.ipynb` (requires access to `thredds.met.no`, about 2–3 hours).
3. Run `02_experiment.ipynb`.
4. Run `03_results.ipynb`.

## Data sources

- PV and meteorological measurements: Karhu, J. A. & Lindfors, A. V. (2025). PV production data with ancillary PV and meteorological data from FMI's outdoor solar laboratories in Helsinki, Kuopio and Sodankylä. Finnish Meteorological Institute. https://doi.org/10.57707/fmi-b2share.c7b3ec9f19324a6497639d6d8c0cde84
- Numerical weather prediction: MetCoOp Ensemble Prediction System (MEPS), archive of the Norwegian Meteorological Institute, https://thredds.met.no

The FMI data are licensed under Creative Commons Attribution (CC BY); the derived files in `data/` are redistributed under the same licence with attribution to the sources above.

## Licence

Code: MIT (see `LICENSE`). Data in `data/` and results in `results/`: Creative Commons Attribution (CC BY). Please cite the article and this repository (see `CITATION.cff`).
