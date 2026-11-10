"""Core module of the multi-horizon PV power forecasting experiment (Helsinki-Kumpula).

Tasks
    intraday : issued every hour, targets the next 1-6 hours.
    dayahead : issued daily at 10:00 UTC, targets the 24 hours of the next day.
Configurations
    no_nwp   : measurements only.
    nwp      : measurements + MEPS numerical weather prediction (00 UTC run, used only
               from 04:00 UTC, i.e. when it is actually available).

Methodological rules
    1. No future leakage: only data available at issue time enter the model.
    2. Expanding window: test years 2019, 2020, 2021; hyperparameters tuned in a nested
       way on the last training year; the test year is never seen.
    3. All algorithms receive the same inputs.
    4. Evaluation on daytime targets only (computed clear-sky GHI); imputed targets and
       suspected outage hours are excluded from training and evaluation.
    5. Every model is compared with smart persistence and climatology on the same rows.
    6. Hourly values are means over [t, t+1); a forecast issued at hour tau knows the hour
       labelled tau-1, and horizon h targets the hour labelled tau+h-1.
    7. Strategy: direct (one model per horizon) for intraday, one pooled model for the 24
       day-ahead hours; LSTM predicts all horizons jointly (MIMO) with several seeds.

Usage
    import sys; sys.path.append("src")
    from multistep_core import *
"""

from __future__ import annotations
import os
import sys
import time
import shutil
import logging
import warnings
from dataclasses import dataclass, field, replace
import json
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
log = logging.getLogger("multistep")
MSG = {
    "copied": "📋 Source file copied into the experiment folder: %s",
    "exists": "ℹ️ File already present — not copied: %s",
    "nwp_title": "🌦️ Extracting MEPS forecasts at the station point",
    "nwp_month_skip": "   ⏭️ %s already done",
    "nwp_month_done": "   ✅ %s | %d runs ok | %d failed | %.0fs",
    "nwp_grid": "   📍 Grid %s → y=%d, x=%d (%.4f, %.4f) | distance %.2f km",
    "nwp_table": "📦 NWP table: %s rows | %d runs | %s → %s",
    "nwp_fail_report": "⚠️ Days without a complete run: %d — details in %s",
    "align_title": "🕒 Time alignment of forecast irradiance",
    "align_best": "📌 Best shift: %+d h (r = %.3f)",
    "base_title": "📂 Preparing the hourly series",
    "base_flags": "   🏷️ Imputed targets: %s | suspected outage hours: %s",
    "samples": "   🧱 %s | %s valid daytime rows | dropped (features) %s | no NWP %s",
    "fold_line": "   📅 Fold %d: train %s (%s rows) ← test %d (%s rows)",
    "run_header": "🚀 %s | %s | %s",
    "fold_done": "  ✓ fold %d | mean skill(SP) over horizons = %+.3f | fit %.1fs",
    "tune_title": "🔧 Tuning (%s → %s)",
    "tune_combo": "      %-42s RMSE=%.4f",
    "tune_best": "   ✅ Best: %s",
    "tune_skip": "   ⏭️ %s: no tuning",
    "saved": "💾 %s",
    "no_nwp_table": "ℹ️ No NWP table — only the no_nwp configuration runs.",
    "unknown_algo": "Unknown algorithm: %s",
    "sum_title": "📊 Summary: %s | %s",
}


def t(key: str) -> str:
    return MSG.get(key, key)


def section(title: str, char: str = "=", width: int = 72) -> None:
    log.info(char * width)
    log.info(title)
    log.info(char * width)


ALGORITHMS = ("MLR", "KNN", "RandomForest", "XGBoost", "SVR", "LSTM")
TASKS = ("intraday", "dayahead")


@dataclass(frozen=True)
class MultistepConfig:
    base_dir: str = os.environ.get(
        "MULTISTEP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    data_filename: str = "Helsinki_MultiYear.csv"
    dataqc_filename: str = "Helsinki_dataQC.csv"
    results_folder_name: str = "results"
    nwp_folder_name: str = "nwp"
    time_col: str = "utctime"
    target_col: str = "Power_PU"
    imputed_target_col: str = "Power_Actual_W_Imputed"
    latitude: float = 60.204
    longitude: float = 24.961
    years: tuple = (2017, 2018, 2019, 2020, 2021)
    tune_train_years: tuple = (2017,)
    tune_valid_year: int = 2018
    test_years: tuple = (2019, 2020, 2021)
    intraday_horizons: tuple = (1, 2, 3, 4, 5, 6)
    dayahead_issue_hour_utc: int = 10
    power_lags_h: tuple = (0, 1, 2, 3, 24)
    ghi_lags_h: tuple = (0, 1, 2)
    meteo_now: tuple = ("Temp_Ambient", "Humidity", "Wind_Speed")
    k_valid_min_elev_deg: float = 3.0
    k_valid_min_cs: float = 30.0
    k_ffill_limit_h: int = 36
    k_clip: tuple = (0.0, 1.5)
    daytime_cs_threshold: float = 10.0
    outage_day_poa_min: float = 1000.0
    outage_day_ppu_max: float = 0.01
    outage_hour_poa_min: float = 100.0
    outage_hour_ppu_max: float = 0.001
    outage_min_daily_temp_c: float = 5.0
    mape_min_power_pu: float = 0.05
    nwp_root: str = "https://thredds.met.no/thredds/dodsC/meps25epsarchive"
    nwp_scheme_switch: str = "2020-02-04"
    nwp_cycle_utc: int = 0
    nwp_availability_lag_h: int = 4
    nwp_shift_hours: int = -1
    nwp_shift_hours_post: int | None = None
    nwp_workers: int = 3
    nwp_retries: int = 2
    nwp_max_grid_km: float = 3.0
    require_nwp_rows: bool = True
    scaled_algorithms: tuple = ("KNN", "SVR", "LSTM")
    svr_max_train: int = 15000
    tune_algorithms: tuple = ("KNN", "SVR", "RandomForest", "XGBoost", "LSTM")
    tune_horizons_intraday: tuple = (1, 3, 6)
    tuning_subsample: int = 10**9
    param_grids: dict = field(
        default_factory=lambda: {
            "KNN": [
                {"n_neighbors": k, "weights": w}
                for k in (10, 25, 50, 100, 200, 400)
                for w in ("uniform", "distance")
            ],
            "SVR": [
                {"C": c, "gamma": g}
                for c in (1.0, 3.0, 10.0, 30.0)
                for g in (0.003, 0.01, 0.03, "scale")
            ],
            "RandomForest": [
                {"max_features": mf, "min_samples_leaf": ml}
                for mf in (1.0, 0.5, "sqrt")
                for ml in (1, 5, 20, 50, 100)
            ],
            "XGBoost": [
                {"max_depth": d, "learning_rate": lr, "n_estimators": n}
                for d in (1, 2, 3, 4, 6, 9)
                for lr in (0.03, 0.1)
                for n in (100, 300, 800)
            ],
            "LSTM": [
                {"lstm_units": u, "lstm_lookback": lb} for u in (32, 64, 128) for lb in (24, 48)
            ],
        }
    )
    lstm_lookback: int = 24
    input_guard_h: int = 48
    lstm_past_cols: tuple = (
        "Power_PU",
        "GHI",
        "Temp_Ambient",
        "Humidity",
        "Wind_Speed",
        "HOUR_SIN",
        "HOUR_COS",
    )
    lstm_units: int = 64
    lstm_epochs_max: int = 150
    lstm_patience: int = 10
    lstm_val_fraction: float = 0.1
    lstm_learning_rate: float = 0.001
    lstm_batch_size: int = 256
    lstm_seeds: tuple = (42, 43, 44, 45, 46)
    lstm_max_restarts: int = 2
    random_seed: int = 42

    @property
    def data_path(self) -> str:
        return os.path.join(self.base_dir, "data", self.data_filename)

    @property
    def dataqc_path(self) -> str:
        return os.path.join(self.base_dir, "data", self.dataqc_filename)

    @property
    def results_dir(self) -> str:
        return os.path.join(self.base_dir, self.results_folder_name)

    @property
    def predictions_dir(self) -> str:
        return os.path.join(self.results_dir, "predictions")

    @property
    def nwp_dir(self) -> str:
        return os.path.join(self.base_dir, "data", self.nwp_folder_name)

    @property
    def nwp_raw_dir(self) -> str:
        return os.path.join(self.nwp_dir, "monthly")

    @property
    def nwp_table_path(self) -> str:
        return os.path.join(self.nwp_dir, "meps_helsinki_00Z.csv")

    @property
    def nwp_failures_path(self) -> str:
        return os.path.join(self.nwp_dir, "failed_runs.csv")


CFG = MultistepConfig()
np.random.seed(CFG.random_seed)


def ensure_dirs(cfg: MultistepConfig = CFG) -> None:
    for d in (cfg.base_dir, cfg.results_dir, cfg.predictions_dir, cfg.nwp_dir, cfg.nwp_raw_dir):
        os.makedirs(d, exist_ok=True)


def solar_elevation_deg(times: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    ts = pd.DatetimeIndex(times)
    doy = ts.dayofyear.to_numpy()
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    decl = (
        0.006918
        - 0.399912 * np.cos(g)
        + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g)
        + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g)
        + 0.00148 * np.sin(3 * g)
    )
    eqt = 229.18 * (
        7.5e-05
        + 0.001868 * np.cos(g)
        - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g)
        - 0.040849 * np.sin(2 * g)
    )
    tst = hour * 60 + eqt + 4 * lon
    ha = np.radians(tst / 4.0 - 180.0)
    phi = np.radians(lat)
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    return np.degrees(np.arcsin(np.clip(cosz, -1, 1)))


def clearsky_ghi(elev_deg: np.ndarray) -> np.ndarray:
    cosz = np.sin(np.radians(elev_deg))
    out = np.zeros_like(cosz, dtype=float)
    m = cosz > 0.01
    out[m] = 1098.0 * cosz[m] * np.exp(-0.057 / cosz[m])
    return out


NWP_VARS = {
    "nwp_ghi": "surface_downwelling_shortwave_flux_in_air",
    "nwp_t2m": "air_temperature_2m",
    "nwp_rh": "relative_humidity_2m",
    "nwp_cloud": "cloud_area_fraction",
}
_GRID_CACHE: dict = {}


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = (np.radians(lat1), np.radians(lat2))
    dphi, dl = (p2 - p1, np.radians(lon2 - lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(6371.0 * 2 * np.arcsin(np.sqrt(a)))


def nearest_gridpoint(nc, cfg: MultistepConfig = CFG) -> tuple[int, int]:
    lat_v, lon_v = (nc.variables["latitude"], nc.variables["longitude"])
    key = tuple(lat_v.shape)
    if key in _GRID_CACHE:
        iy, ix = _GRID_CACHE[key]
        d = _haversine_km(cfg.latitude, cfg.longitude, float(lat_v[iy, ix]), float(lon_v[iy, ix]))
        if d <= cfg.nwp_max_grid_km:
            return (iy, ix)
    lat, lon = (np.asarray(lat_v[:]), np.asarray(lon_v[:]))
    dist2 = (lat - cfg.latitude) ** 2 + (
        (lon - cfg.longitude) * np.cos(np.radians(cfg.latitude))
    ) ** 2
    iy, ix = np.unravel_index(np.argmin(dist2), lat.shape)
    d = _haversine_km(cfg.latitude, cfg.longitude, float(lat[iy, ix]), float(lon[iy, ix]))
    if d > cfg.nwp_max_grid_km:
        raise ValueError(f"nearest grid point {d:.1f} km away (> {cfg.nwp_max_grid_km})")
    _GRID_CACHE[key] = (int(iy), int(ix))
    log.info(t("nwp_grid"), "×".join(map(str, key)), iy, ix, lat[iy, ix], lon[iy, ix], d)
    return (int(iy), int(ix))


def _point_series(nc, name: str, iy: int, ix: int) -> np.ndarray:
    var = nc.variables[name]
    idx = []
    for dim in var.dimensions:
        if dim == "time":
            idx.append(slice(None))
        elif dim == "y":
            idx.append(iy)
        elif dim == "x":
            idx.append(ix)
        else:
            idx.append(0)
    vals = np.ma.filled(np.ma.asarray(var[tuple(idx)]).astype(float), np.nan)
    units = str(getattr(var, "units", "")).strip()
    if name == NWP_VARS["nwp_t2m"] and units.upper().startswith("K"):
        vals = vals - 273.15
    elif name == NWP_VARS["nwp_cloud"] and units == "%":
        vals = vals / 100.0
    elif name == NWP_VARS["nwp_rh"] and units in ("1", ""):
        vals = vals * 100.0
    return vals


def _valid_times(nc) -> pd.DatetimeIndex:
    import netCDF4

    tv = nc.variables["time"]
    dts = netCDF4.num2date(
        tv[:], tv.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True
    )
    return pd.DatetimeIndex(pd.to_datetime([str(d) for d in dts])).tz_localize(None)


def meps_run_files(run_time: pd.Timestamp, cfg: MultistepConfig = CFG) -> list[dict]:
    d = run_time.strftime("%Y%m%d")
    c = f"{run_time.hour:02d}"
    path = run_time.strftime("%Y/%m/%d")
    old = {
        "nwp_ghi": f"meps_mbr0_full_backup_2_5km_{d}T{c}Z.nc",
        "nwp_t2m": f"meps_mbr0_pp_2_5km_{d}T{c}Z.nc",
        "nwp_rh": f"meps_mbr0_pp_2_5km_{d}T{c}Z.nc",
        "nwp_cloud": f"meps_mbr0_pp_2_5km_{d}T{c}Z.nc",
    }
    new = {k: f"meps_det_2_5km_{d}T{c}Z.nc" for k in NWP_VARS}
    order = [old, new] if run_time < pd.Timestamp(cfg.nwp_scheme_switch) else [new, old]
    return [{k: f"{cfg.nwp_root}/{path}/{v}" for k, v in o.items()} for o in order]


def _open_with_retry(url: str, cfg: MultistepConfig):
    import netCDF4

    for attempt in range(cfg.nwp_retries + 1):
        try:
            return netCDF4.Dataset(url)
        except OSError:
            if attempt == cfg.nwp_retries:
                raise
            time.sleep(2 * (attempt + 1))


def _read_layout(layout: dict, run_time: pd.Timestamp, cfg: MultistepConfig):
    opened, errs, out = ({}, [], {})
    try:
        for url in dict.fromkeys(layout.values()):
            try:
                opened[url] = _open_with_retry(url, cfg)
            except OSError as exc:
                errs.append(f"{url.rsplit('/', 1)[-1]}: {type(exc).__name__}")
        if not opened:
            return (None, errs)
        for key, url in layout.items():
            nc = opened.get(url)
            if nc is None:
                continue
            for attempt in range(cfg.nwp_retries + 3):
                try:
                    times = _valid_times(nc)
                    iy, ix = nearest_gridpoint(nc, cfg)
                    out[key] = pd.Series(_point_series(nc, NWP_VARS[key], iy, ix), index=times)
                    break
                except (KeyError, IndexError, ValueError) as exc:
                    errs.append(f"{key}: {type(exc).__name__}")
                    break
                except RuntimeError:
                    if attempt == cfg.nwp_retries + 2:
                        raise
                    time.sleep(3 * (attempt + 1))
        if not out:
            return (None, errs)
        df = pd.DataFrame(out).reindex(columns=list(NWP_VARS))
        df.index.name = "valid_time"
        df = df.reset_index()
        df.insert(0, "run_time", run_time)
        df["lead_h"] = ((df["valid_time"] - run_time) / pd.Timedelta(hours=1)).round().astype(int)
        return (df, errs)
    finally:
        for nc in opened.values():
            try:
                nc.close()
            except Exception:
                pass


def fetch_meps_run(
    run_time: pd.Timestamp, cfg: MultistepConfig = CFG
) -> tuple[pd.DataFrame | None, str]:
    last_err = ""
    for layout in meps_run_files(run_time, cfg):
        for attempt in range(cfg.nwp_retries + 2):
            try:
                df, errs = _read_layout(layout, run_time, cfg)
                break
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {str(exc)[:120]}"
                df, errs = (None, [last_err])
                time.sleep(5 * (attempt + 1))
        if df is not None:
            return (df, "; ".join(errs))
        last_err = "; ".join(errs) or last_err
    return (None, last_err)


def extract_meps_archive(
    start: str = "2017-01-01", end: str = "2021-12-31", cfg: MultistepConfig = CFG
) -> None:
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    ensure_dirs(cfg)
    section(t("nwp_title"))
    days = pd.date_range(start, end, freq="D")
    months = [(m, g.index) for m, g in pd.Series(days, index=days).groupby(days.to_period("M"))]
    todo = [
        (m, d) for m, d in months if not os.path.exists(os.path.join(cfg.nwp_raw_dir, f"{m}.csv"))
    ]
    for m, _ in months:
        if (m, _) not in todo:
            log.info(t("nwp_month_skip"), m)
    pool = ProcessPoolExecutor(max_workers=cfg.nwp_workers) if cfg.nwp_workers > 1 else None
    try:
        for month, mdays in todo:
            t0 = time.time()
            runs = [d + pd.Timedelta(hours=cfg.nwp_cycle_utc) for d in mdays]
            fn = partial(fetch_meps_run, cfg=cfg)
            results = list(pool.map(fn, runs)) if pool else [fn(r) for r in runs]
            frames, fails = ([], [])
            for r, (df, err) in zip(runs, results):
                if df is None:
                    fails.append({"run_time": r, "error": err, "type": "full"})
                else:
                    frames.append(df)
                    if err:
                        fails.append({"run_time": r, "error": err, "type": "partial"})
            if frames:
                pd.concat(frames, ignore_index=True).to_csv(
                    os.path.join(cfg.nwp_raw_dir, f"{month}.csv"), index=False
                )
            if fails:
                pd.DataFrame(fails).to_csv(
                    cfg.nwp_failures_path,
                    mode="a",
                    index=False,
                    header=not os.path.exists(cfg.nwp_failures_path),
                )
            log.info(t("nwp_month_done"), month, len(frames), len(fails), time.time() - t0)
    finally:
        if pool:
            pool.shutdown()


def retry_failed_runs(cfg: MultistepConfig = CFG) -> pd.DataFrame:
    if not os.path.exists(cfg.nwp_failures_path):
        return pd.DataFrame()
    f = pd.read_csv(cfg.nwp_failures_path, parse_dates=["run_time"])
    keep, fixed = ([], 0)
    for _, row in f.iterrows():
        if row["type"] != "full":
            keep.append(row)
            continue
        df, err = fetch_meps_run(row["run_time"], cfg)
        if df is None:
            keep.append(row)
            continue
        month_path = os.path.join(cfg.nwp_raw_dir, f"{row['run_time'].to_period('M')}.csv")
        df.to_csv(month_path, mode="a", index=False, header=not os.path.exists(month_path))
        if err:
            keep.append(pd.Series({"run_time": row["run_time"], "error": err, "type": "partial"}))
        fixed += 1
    rest = pd.DataFrame(keep)
    rest.to_csv(cfg.nwp_failures_path, index=False)
    log.info(
        "🔁 %d runs recovered | remaining records: %d (full: %d)",
        fixed,
        len(rest),
        int((rest["type"] == "full").sum()) if len(rest) else 0,
    )
    return rest


def build_nwp_table(cfg: MultistepConfig = CFG) -> pd.DataFrame:
    files = sorted((f for f in os.listdir(cfg.nwp_raw_dir) if f.endswith(".csv")))
    df = pd.concat(
        [
            pd.read_csv(os.path.join(cfg.nwp_raw_dir, f), parse_dates=["run_time", "valid_time"])
            for f in files
        ],
        ignore_index=True,
    )
    df = df.drop_duplicates(["run_time", "valid_time"]).sort_values(["run_time", "valid_time"])
    df.to_csv(cfg.nwp_table_path, index=False)
    log.info(
        t("nwp_table"),
        f"{len(df):,}",
        df["run_time"].nunique(),
        df["run_time"].min().date(),
        df["run_time"].max().date(),
    )
    if os.path.exists(cfg.nwp_failures_path):
        f = pd.read_csv(cfg.nwp_failures_path)
        log.info(t("nwp_fail_report"), int((f["type"] == "full").sum()), cfg.nwp_failures_path)
    return df


def load_nwp_table(cfg: MultistepConfig = CFG) -> pd.DataFrame | None:
    if not os.path.exists(cfg.nwp_table_path):
        return None
    return pd.read_csv(cfg.nwp_table_path, parse_dates=["run_time", "valid_time"])


def prepare_base(cfg: MultistepConfig = CFG) -> pd.DataFrame:
    section(t("base_title"))
    df = pd.read_csv(cfg.data_path, parse_dates=[cfg.time_col])
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col]).dt.tz_localize(None)
    df = df.drop_duplicates(cfg.time_col).set_index(cfg.time_col).sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full)
    df.index.name = cfg.time_col
    elev = solar_elevation_deg(df.index + pd.Timedelta(minutes=30), cfg.latitude, cfg.longitude)
    df["ELEV"] = elev
    df["GHI_CS"] = clearsky_ghi(elev)
    h, mth = (df.index.hour.to_numpy(), df.index.month.to_numpy())
    df["HOUR_SIN"], df["HOUR_COS"] = (np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24))
    df["MONTH_SIN"], df["MONTH_COS"] = (np.sin(2 * np.pi * mth / 12), np.cos(2 * np.pi * mth / 12))
    imp = (
        df[cfg.imputed_target_col].fillna(True).astype(bool)
        if cfg.imputed_target_col in df.columns
        else df[cfg.target_col].isna()
    )
    df["FLAG_IMPUTED"] = imp | df[cfg.target_col].isna()
    df["FLAG_OUTAGE"] = detect_outage(df, df["FLAG_IMPUTED"], cfg)
    if not os.path.exists(cfg.dataqc_path):
        raise FileNotFoundError(f"{cfg.dataqc_path} not found — run notebook 01 first")
    qc = pd.read_csv(cfg.dataqc_path, parse_dates=["day"])
    bad_days = pd.DatetimeIndex(qc.loc[qc["dataQC"] == 0, "day"]).normalize()
    df["FLAG_QC"] = df.index.normalize().isin(bad_days) & ~df["FLAG_IMPUTED"]
    log.info(
        "   🧾 FMI dataQC = 0: %d days | additionally excluded measured hours: %s (daytime: %s) | outage hours also flagged by the rule: %s",
        len(bad_days),
        f"{int(df['FLAG_QC'].sum()):,}",
        f"{int((df['FLAG_QC'] & (df['GHI_CS'] > cfg.daytime_cs_threshold)).sum()):,}",
        f"{int((df['FLAG_QC'] & df['FLAG_OUTAGE']).sum()):,}",
    )
    valid = (df["ELEV"] > cfg.k_valid_min_elev_deg) & (df["GHI_CS"] > cfg.k_valid_min_cs)
    lo, hi = cfg.k_clip
    p_meas = df[cfg.target_col].where(~df["FLAG_IMPUTED"] & ~df["FLAG_OUTAGE"] & ~df["FLAG_QC"])
    g_imp = df["GHI_Imputed"].fillna(True).astype(bool) if "GHI_Imputed" in df else df["GHI"].isna()
    g_meas = df["GHI"].where(~g_imp)
    kp = (p_meas / (df["GHI_CS"] / 1000.0)).where(valid).clip(lo, hi)
    kg = (g_meas / df["GHI_CS"]).where(valid).clip(lo, hi)
    df["KP"], df["KG"] = (kp, kg)
    df["KP_LAST"] = kp.ffill(limit=cfg.k_ffill_limit_h)
    df["KG_LAST"] = kg.ffill(limit=cfg.k_ffill_limit_h)
    daily_kp = kp.groupby(df.index.normalize()).mean()
    df["KP_DAY"] = daily_kp.reindex(df.index.normalize()).to_numpy()
    night = df["GHI_CS"] <= cfg.daytime_cs_threshold
    p_fill = (df["KP_LAST"] * df["GHI_CS"] / 1000.0).where(~night, 0.0)
    g_fill = (df["KG_LAST"] * df["GHI_CS"]).where(~night, 0.0)
    df[f"{cfg.target_col}_IN"] = p_meas.fillna(p_fill)
    df["GHI_IN"] = g_meas.fillna(g_fill)
    for c in ("Temp_Ambient", "Humidity", "Wind_Speed"):
        ic = f"{c}_Imputed"
        m = df[ic].fillna(True).astype(bool) if ic in df else df[c].isna()
        df[f"{c}_IN"] = df[c].where(~m).ffill(limit=24)
    n_in = {
        c: int((df[c].notna() & df[c.replace("_IN", "")].ne(df[c])).sum())
        for c in (
            f"{cfg.target_col}_IN",
            "GHI_IN",
            "Temp_Ambient_IN",
            "Humidity_IN",
            "Wind_Speed_IN",
        )
    }
    log.info("   🔁 causal input filling (hours): %s", n_in)
    log.info(
        t("base_flags"), f"{int(df['FLAG_IMPUTED'].sum()):,}", f"{int(df['FLAG_OUTAGE'].sum()):,}"
    )
    return df


def detect_outage(df: pd.DataFrame, imputed: pd.Series, cfg: MultistepConfig = CFG) -> pd.Series:
    poa = df["Irradiance_POA"] if "Irradiance_POA" in df.columns else df["GHI"]
    p = df[cfg.target_col].where(~imputed)
    day = df.index.normalize()
    warm = df["Temp_Ambient"].groupby(day).transform("mean") > cfg.outage_min_daily_temp_c
    day_poa = poa.groupby(day).transform("sum")
    day_p = p.groupby(day).transform(lambda x: x.sum(min_count=1))
    day_imp = (imputed & (poa > 0)).groupby(day).transform("sum") > 0
    rule_a = (day_poa > cfg.outage_day_poa_min) & (day_p < cfg.outage_day_ppu_max) & ~day_imp
    rule_b = (poa > cfg.outage_hour_poa_min) & (p < cfg.outage_hour_ppu_max)
    return ((rule_a | rule_b) & warm).fillna(False).astype(bool)


def feature_columns(task: str, with_nwp: bool, cfg: MultistepConfig = CFG) -> list[str]:
    cols = [f"P_lag{h}" for h in cfg.power_lags_h] + [f"GHI_lag{h}" for h in cfg.ghi_lags_h]
    cols += [f"{c}_now" for c in cfg.meteo_now] + ["KP_LAST_now", "KG_LAST_now"]
    if task == "dayahead":
        cols += ["P_target_m48", "KP_DAY_prev"]
    cols += ["T_HOUR_SIN", "T_HOUR_COS", "T_MONTH_SIN", "T_MONTH_COS", "T_GHI_CS", "T_ELEV"]
    if with_nwp:
        cols += ["NWP_GHI", "NWP_K", "NWP_CLOUD", "NWP_T2M", "NWP_RH"]
    return cols


CYCLIC = ("T_HOUR_SIN", "T_HOUR_COS", "T_MONTH_SIN", "T_MONTH_COS", "HOUR_SIN", "HOUR_COS")


def _at(base: pd.DataFrame, times, col: str) -> np.ndarray:
    return base[col].reindex(pd.DatetimeIndex(times)).to_numpy(dtype=float)


def _nwp_shift_for_runs(run_time: pd.Series, cfg: MultistepConfig) -> np.ndarray:
    post = cfg.nwp_shift_hours if cfg.nwp_shift_hours_post is None else cfg.nwp_shift_hours_post
    is_post = pd.to_datetime(run_time).to_numpy() >= np.datetime64(
        pd.Timestamp(cfg.nwp_scheme_switch)
    )
    return np.where(is_post, post, cfg.nwp_shift_hours).astype(float)


def _attach_nwp(s: pd.DataFrame, nwp: pd.DataFrame, cfg: MultistepConfig) -> pd.DataFrame:
    n = nwp.copy()
    n["valid_time"] = n["valid_time"] + pd.to_timedelta(
        _nwp_shift_for_runs(n["run_time"], cfg), unit="h"
    )
    n = n[["run_time", "valid_time", "nwp_ghi", "nwp_t2m", "nwp_rh", "nwp_cloud"]].drop_duplicates(
        ["run_time", "valid_time"]
    )
    lag = pd.Timedelta(hours=cfg.nwp_availability_lag_h)
    run = (s["issue_time"] - lag - pd.Timedelta(hours=cfg.nwp_cycle_utc)).dt.floor(
        "D"
    ) + pd.Timedelta(hours=cfg.nwp_cycle_utc)
    out = s.assign(run_time=run).merge(
        n, left_on=["run_time", "target_time"], right_on=["run_time", "valid_time"], how="left"
    )
    miss = out["nwp_ghi"].isna()
    if miss.any():
        fb = s.loc[miss.to_numpy()].assign(run_time=run[miss.to_numpy()] - pd.Timedelta(days=1))
        fb = fb.merge(
            n, left_on=["run_time", "target_time"], right_on=["run_time", "valid_time"], how="left"
        )
        for c in ("nwp_ghi", "nwp_t2m", "nwp_rh", "nwp_cloud", "run_time"):
            out.loc[miss.to_numpy(), c] = fb[c].to_numpy()
    out["NWP_GHI"] = out["nwp_ghi"]
    out["NWP_K"] = (out["nwp_ghi"] / np.maximum(out["T_GHI_CS"], cfg.k_valid_min_cs)).clip(
        *cfg.k_clip
    )
    out["NWP_CLOUD"], out["NWP_T2M"], out["NWP_RH"] = (
        out["nwp_cloud"],
        out["nwp_t2m"],
        out["nwp_rh"],
    )
    out["NWP_LEAD_H"] = ((out["target_time"] - out["run_time"]) / pd.Timedelta(hours=1)).where(
        out["nwp_ghi"].notna()
    )
    return out.drop(columns=["valid_time", "nwp_ghi", "nwp_t2m", "nwp_rh", "nwp_cloud"])


def build_samples(
    base: pd.DataFrame, task: str, cfg: MultistepConfig = CFG, nwp: pd.DataFrame | None = None
) -> pd.DataFrame:
    idx = base.index
    first_ok = idx.min() + pd.Timedelta(hours=48)
    if task == "intraday":
        issues = idx[idx >= first_ok]
        parts = []
        for h in cfg.intraday_horizons:
            parts.append(
                pd.DataFrame(
                    {
                        "issue_time": issues,
                        "target_time": issues + pd.Timedelta(hours=h - 1),
                        "horizon": h,
                    }
                )
            )
        s = pd.concat(parts, ignore_index=True)
    elif task == "dayahead":
        days = pd.date_range(
            first_ok.normalize() + pd.Timedelta(days=1),
            idx.max().normalize() - pd.Timedelta(days=1),
            freq="D",
        )
        issues = days + pd.Timedelta(hours=cfg.dayahead_issue_hour_utc)
        parts = []
        for hr in range(24):
            tgt = days + pd.Timedelta(days=1, hours=hr)
            parts.append(pd.DataFrame({"issue_time": issues, "target_time": tgt}))
        s = pd.concat(parts, ignore_index=True)
        s["horizon"] = ((s["target_time"] - s["issue_time"]) / pd.Timedelta(hours=1)).astype(
            int
        ) + 1
    else:
        raise ValueError(task)
    s = s[s["target_time"] <= idx.max()].reset_index(drop=True)
    it, tt = (s["issue_time"], s["target_time"])
    known = it - pd.Timedelta(hours=1)
    p_in = f"{cfg.target_col}_IN"
    for h in cfg.power_lags_h:
        s[f"P_lag{h}"] = _at(base, known - pd.Timedelta(hours=h), p_in)
    for h in cfg.ghi_lags_h:
        s[f"GHI_lag{h}"] = _at(base, known - pd.Timedelta(hours=h), "GHI_IN")
    for c in cfg.meteo_now:
        s[f"{c}_now"] = _at(base, known, f"{c}_IN")
    s["KP_LAST_now"] = _at(base, known, "KP_LAST")
    s["KG_LAST_now"] = _at(base, known, "KG_LAST")
    if task == "dayahead":
        s["P_target_m48"] = _at(base, tt - pd.Timedelta(hours=48), p_in)
        s["KP_DAY_prev"] = _at(base, it.dt.normalize() - pd.Timedelta(days=1), "KP_DAY")
    for src, dst in (
        ("HOUR_SIN", "T_HOUR_SIN"),
        ("HOUR_COS", "T_HOUR_COS"),
        ("MONTH_SIN", "T_MONTH_SIN"),
        ("MONTH_COS", "T_MONTH_COS"),
        ("GHI_CS", "T_GHI_CS"),
        ("ELEV", "T_ELEV"),
    ):
        s[dst] = _at(base, tt, src)
    s["y_true"] = _at(base, tt, cfg.target_col)
    s["flag_imputed"] = (
        base["FLAG_IMPUTED"].reindex(pd.DatetimeIndex(tt)).fillna(True).to_numpy(bool)
    )
    s["flag_qc"] = base["FLAG_QC"].reindex(pd.DatetimeIndex(tt)).fillna(False).to_numpy(bool)
    s["flag_outage"] = (
        base["FLAG_OUTAGE"].reindex(pd.DatetimeIndex(tt)).fillna(False).to_numpy(bool)
    )
    s["year"] = tt.dt.year
    out_cum = np.concatenate([[0], np.cumsum(base["FLAG_OUTAGE"].to_numpy(int))])
    pos = base.index.get_indexer(pd.DatetimeIndex(known))
    lo_ = np.maximum(pos - cfg.input_guard_h + 1, 0)
    bad_in = out_cum[pos + 1] - out_cum[lo_] > 0
    if task == "dayahead":
        bad_in |= (
            base["FLAG_OUTAGE"]
            .reindex(pd.DatetimeIndex(tt - pd.Timedelta(hours=48)))
            .fillna(False)
            .to_numpy(bool)
        )
        prevday = base["FLAG_OUTAGE"].groupby(base.index.normalize()).any()
        bad_in |= (
            prevday.reindex(pd.DatetimeIndex(it.dt.normalize() - pd.Timedelta(days=1)))
            .fillna(False)
            .to_numpy(bool)
        )
    s["flag_bad_input"] = bad_in
    s = s[s["T_GHI_CS"] > cfg.daytime_cs_threshold].reset_index(drop=True)
    if task == "intraday":
        s["k_ref"] = s["KP_LAST_now"]
    else:
        s["k_ref"] = s["KP_DAY_prev"]
    has_nwp = np.zeros(len(s), bool)
    if nwp is not None:
        s = _attach_nwp(s, nwp, cfg)
        has_nwp = s[["NWP_GHI", "NWP_CLOUD", "NWP_T2M", "NWP_RH"]].notna().all(axis=1).to_numpy()
    s["has_nwp"] = has_nwp
    feats = feature_columns(task, with_nwp=False, cfg=cfg)
    ok = s[feats].notna().all(axis=1) & s["y_true"].notna()
    n_drop = int((~ok).sum())
    s = s[ok].reset_index(drop=True)
    s["eval_ok"] = ~(s["flag_imputed"] | s["flag_outage"] | s["flag_qc"] | s["flag_bad_input"])
    if nwp is not None and cfg.require_nwp_rows:
        s["eval_ok"] &= s["has_nwp"]
    log.info(
        t("samples"),
        task,
        f"{int(s['eval_ok'].sum()):,}",
        f"{n_drop:,}",
        f"{int((~s['has_nwp']).sum()):,}" if nwp is not None else "—",
    )
    return s


def check_nwp_alignment(
    base: pd.DataFrame,
    nwp: pd.DataFrame,
    cfg: MultistepConfig = CFG,
    years: tuple | None = None,
    shifts=(-2, -1, 0, 1, 2),
) -> int:
    section(t("align_title"))
    years = years or cfg.tune_train_years + (cfg.tune_valid_year,)
    n = nwp[(nwp["lead_h"] >= 24) & (nwp["lead_h"] <= 47)]
    n = n[n["valid_time"].dt.year.isin(years)]
    obs = base["GHI"]
    rows = []
    for sh in shifts:
        v = n["valid_time"] + pd.Timedelta(hours=sh)
        o = obs.reindex(pd.DatetimeIndex(v)).to_numpy(float)
        m = np.isfinite(o) & np.isfinite(n["nwp_ghi"].to_numpy(float)) & (o > 5)
        r = np.corrcoef(o[m], n["nwp_ghi"].to_numpy(float)[m])[0, 1] if m.sum() > 100 else np.nan
        rows.append((sh, r))
        log.info("   shift %+d h → r = %.4f (n=%d)", sh, r, int(m.sum()))
    best = max(rows, key=lambda x: -1 if np.isnan(x[1]) else x[1])
    log.info(t("align_best"), best[0], best[1])
    return int(best[0])


def compute_metrics(y_true, y_pred, cfg: MultistepConfig = CFG) -> dict:
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    y_true, y_pred = (np.asarray(y_true, float), np.asarray(y_pred, float))
    m = {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }
    m["nMAE_pct"], m["nRMSE_pct"] = (m["MAE"] * 100.0, m["RMSE"] * 100.0)
    mask = y_true > cfg.mape_min_power_pu
    m["MAPE"] = (
        float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        if mask.any()
        else np.nan
    )
    return m


def reference_predictions(
    train: pd.DataFrame, test: pd.DataFrame, cfg: MultistepConfig = CFG
) -> dict[str, np.ndarray]:
    k_fill = float(np.nanmedian(train["k_ref"]))
    k = test["k_ref"].fillna(k_fill).to_numpy(float)
    sp = np.clip(k * test["T_GHI_CS"].to_numpy(float) / 1000.0, 0, None)
    tr_key = list(zip(train["target_time"].dt.month, train["target_time"].dt.hour))
    clim = (
        pd.Series(train["y_true"].to_numpy(float), index=pd.MultiIndex.from_tuples(tr_key))
        .groupby(level=[0, 1])
        .mean()
    )
    te_key = pd.MultiIndex.from_arrays([test["target_time"].dt.month, test["target_time"].dt.hour])
    cl = clim.reindex(te_key).fillna(float(train["y_true"].mean())).to_numpy(float)
    return {"SmartPersistence": sp, "Climatology": cl}


def build_model(algorithm: str, cfg: MultistepConfig = CFG, params: dict | None = None):
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR

    p = dict(params or {})
    if algorithm == "MLR":
        return LinearRegression()
    if algorithm == "RandomForest":
        return RandomForestRegressor(
            **{"n_estimators": 300, "min_samples_leaf": 2, **p},
            n_jobs=-1,
            random_state=cfg.random_seed,
        )
    if algorithm == "KNN":
        return KNeighborsRegressor(**{"n_neighbors": 15, "weights": "distance", **p}, n_jobs=-1)
    if algorithm == "SVR":
        return SVR(kernel="rbf", **{"C": 10.0, "epsilon": 0.01, "gamma": "scale", **p})
    if algorithm == "XGBoost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            **{
                "n_estimators": 400,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                **p,
            },
            random_state=cfg.random_seed,
            n_jobs=-1,
            verbosity=0,
        )
    raise ValueError(t("unknown_algo") % algorithm)


def _scale(X_tr, X_te, features, algorithm, cfg):
    from sklearn.preprocessing import StandardScaler

    if algorithm not in cfg.scaled_algorithms:
        return (X_tr, X_te)
    mask = np.array([f not in CYCLIC for f in features])
    sc = StandardScaler().fit(X_tr[:, mask])
    X_tr, X_te = (X_tr.copy(), X_te.copy())
    X_tr[:, mask], X_te[:, mask] = (sc.transform(X_tr[:, mask]), sc.transform(X_te[:, mask]))
    return (X_tr, X_te)


def _fit_predict(algorithm, train, test, features, cfg, params, seed_offset=0):
    if algorithm == "SVR" and len(train) > cfg.svr_max_train:
        train = train.sample(cfg.svr_max_train, random_state=cfg.random_seed + seed_offset)
    X_tr = train[features].to_numpy(float)
    X_te = test[features].to_numpy(float)
    X_tr, X_te = _scale(X_tr, X_te, features, algorithm, cfg)
    model = build_model(algorithm, cfg, params)
    t0 = time.time()
    model.fit(X_tr, train["y_true"].to_numpy(float))
    fit_s = time.time() - t0
    return (np.clip(model.predict(X_te), 0, None), fit_s)


def _train_rows(s: pd.DataFrame, years: tuple, before: pd.Timestamp) -> pd.DataFrame:
    m = s["year"].isin(years) & (s["target_time"] < before) & s["eval_ok"]
    return s[m]


def tabular_task_predict(algorithm, task, s, train_years, test_year, features, cfg, params):
    start = pd.Timestamp(f"{test_year}-01-01")
    train_all = _train_rows(s, train_years, start)
    train_all = train_all[train_all[features].notna().all(axis=1)]
    test_all = s[(s["year"] == test_year) & s["eval_ok"]]
    preds, fit_total = (pd.Series(np.nan, index=test_all.index), 0.0)
    fit_ok = test_all[features].notna().all(axis=1)
    if task == "intraday":
        for h in cfg.intraday_horizons:
            tr = train_all[train_all["horizon"] == h]
            te = test_all[(test_all["horizon"] == h) & fit_ok]
            if len(tr) == 0 or len(te) == 0:
                continue
            p, f = _fit_predict(algorithm, tr, te, features, cfg, params)
            preds.loc[te.index] = p
            fit_total += f
    else:
        te = test_all[fit_ok]
        p, f = _fit_predict(algorithm, train_all, te, features, cfg, params)
        preds.loc[te.index] = p
        fit_total += f
    return (train_all, test_all, preds.to_numpy(float), fit_total)


def _lstm_future_cols(with_nwp: bool) -> list[str]:
    cols = ["T_HOUR_SIN", "T_HOUR_COS", "T_MONTH_SIN", "T_MONTH_COS", "T_GHI_CS", "T_ELEV"]
    return cols + (["NWP_GHI", "NWP_K", "NWP_CLOUD", "NWP_T2M", "NWP_RH"] if with_nwp else [])


def _lstm_arrays(base, s_task, task, with_nwp, cfg):
    H = (
        list(cfg.intraday_horizons)
        if task == "intraday"
        else list(range(25 - cfg.dayahead_issue_hour_utc, 49 - cfg.dayahead_issue_hour_utc))
    )
    issues = pd.DatetimeIndex(sorted(s_task["issue_time"].unique()))
    past = base[
        [f"{c}_IN" if f"{c}_IN" in base.columns else c for c in cfg.lstm_past_cols]
    ].to_numpy(float)
    pos = base.index.get_indexer(issues - pd.Timedelta(hours=1))
    lb = cfg.lstm_lookback
    keep = pos >= lb - 1
    issues, pos = (issues[keep], pos[keep])
    Xp = np.stack([past[p - lb + 1 : p + 1] for p in pos])
    _oc = np.concatenate([[0], np.cumsum(base["FLAG_OUTAGE"].to_numpy(int))])
    badwin = _oc[pos + 1] - _oc[pos - lb + 1] > 0
    fcols = _lstm_future_cols(with_nwp)
    lut = s_task.set_index(["issue_time", "horizon"])
    Xf = np.zeros((len(issues), len(H), len(fcols)))
    Y = np.zeros((len(issues), len(H)))
    bad = np.zeros((len(issues), len(H)), bool)
    for j, h in enumerate(H):
        tgt = issues + pd.Timedelta(hours=h - 1)
        for k, c in enumerate(fcols[:6]):
            src = {
                "T_HOUR_SIN": "HOUR_SIN",
                "T_HOUR_COS": "HOUR_COS",
                "T_MONTH_SIN": "MONTH_SIN",
                "T_MONTH_COS": "MONTH_COS",
                "T_GHI_CS": "GHI_CS",
                "T_ELEV": "ELEV",
            }[c]
            Xf[:, j, k] = _at(base, tgt, src)
        Y[:, j] = np.nan_to_num(_at(base, tgt, cfg.target_col), nan=0.0)
        flags = (
            (base["FLAG_IMPUTED"] | base["FLAG_OUTAGE"] | base["FLAG_QC"]).reindex(tgt).fillna(True)
        )
        bad[:, j] = flags.to_numpy(bool) & (Xf[:, j, 4] > cfg.daytime_cs_threshold)
        if with_nwp:
            mi = pd.MultiIndex.from_arrays([issues, np.full(len(issues), h)])
            sub = lut.reindex(mi)
            for k, c in enumerate(fcols[6:], start=6):
                Xf[:, j, k] = sub[c].to_numpy(float) if c in sub else np.nan
    ok = np.isfinite(Xp).all(axis=(1, 2))
    if not with_nwp:
        ok &= np.isfinite(Xf).all(axis=(1, 2))
    else:
        Xf = np.nan_to_num(Xf, nan=0.0)
    return (issues[ok], Xp[ok], Xf[ok], Y[ok], bad[ok].any(axis=1) | badwin[ok], H)


def _build_lstm(n_past, n_fut_flat, n_out, cfg, seed):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf.keras.utils.set_random_seed(seed)
    a = layers.Input(shape=(cfg.lstm_lookback, n_past))
    b = layers.Input(shape=(n_fut_flat,))
    x = layers.LSTM(cfg.lstm_units)(a)
    x = layers.Concatenate()([x, b])
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_out, activation="linear")(x)
    m = models.Model([a, b], out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lstm_learning_rate, clipnorm=1.0),
        loss="mse",
    )
    return m


def _fit_lstm_once(Xp_tr, Xf_tr, Y_tr, n_out, cfg, seed):
    import tensorflow as tf

    n_val = max(1, int(round(len(Y_tr) * cfg.lstm_val_fraction)))
    fit_idx, val_idx = (slice(0, len(Y_tr) - n_val), slice(len(Y_tr) - n_val, None))
    model = _build_lstm(Xp_tr.shape[2], Xf_tr.shape[1], n_out, cfg, seed)
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=cfg.lstm_patience, restore_best_weights=True
    )
    hist = model.fit(
        [Xp_tr[fit_idx], Xf_tr[fit_idx]],
        Y_tr[fit_idx],
        validation_data=([Xp_tr[val_idx], Xf_tr[val_idx]], Y_tr[val_idx]),
        epochs=cfg.lstm_epochs_max,
        batch_size=cfg.lstm_batch_size,
        callbacks=[es],
        verbose=0,
    )
    best_val = float(np.min(hist.history["val_loss"]))
    const = Y_tr[fit_idx].mean(axis=0, keepdims=True)
    const_val = float(np.mean((Y_tr[val_idx] - const) ** 2))
    epochs = int(np.argmin(hist.history["val_loss"])) + 1
    return (model, best_val, const_val, epochs)


def lstm_task_predict(base, s, task, train_years, test_year, with_nwp, cfg, seed):
    from sklearn.preprocessing import StandardScaler

    start = pd.Timestamp(f"{test_year}-01-01")
    issues, Xp, Xf, Y, bad, H = _lstm_arrays(base, s, task, with_nwp, cfg)
    last_target = issues + pd.Timedelta(hours=max(H) - 1)
    tr = np.isin(issues.year, train_years) & (last_target < start) & ~bad
    te = np.isin((issues + pd.Timedelta(hours=min(H) - 1)).year, [test_year]) | np.isin(
        last_target.year, [test_year]
    )
    sp = StandardScaler().fit(Xp[tr].reshape(-1, Xp.shape[2]))
    Xp_s = sp.transform(Xp.reshape(-1, Xp.shape[2])).reshape(Xp.shape)
    Xf_flat = Xf.reshape(len(Xf), -1)
    sf = StandardScaler().fit(Xf_flat[tr])
    Xf_s = sf.transform(Xf_flat)
    t0 = time.time()
    restarts = 0
    for attempt in range(cfg.lstm_max_restarts + 1):
        model, best_val, const_val, epochs = _fit_lstm_once(
            Xp_s[tr], Xf_s[tr], Y[tr], len(H), cfg, seed + 1000 * attempt
        )
        if best_val < const_val:
            break
        restarts += 1
        log.info(
            "   ⚠️ LSTM seed=%d: val MSE %.5f ≥ constant model %.5f → restart %d",
            seed,
            best_val,
            const_val,
            restarts,
        )
    fit_s = time.time() - t0
    P = np.clip(model.predict([Xp_s[te], Xf_s[te]], verbose=0), 0.0, None)
    long = pd.DataFrame(
        {
            "issue_time": np.repeat(issues[te], len(H)),
            "horizon": np.tile(H, int(te.sum())),
            "y_pred_lstm": P.ravel(),
        }
    )
    test_all = s[(s["year"] == test_year) & s["eval_ok"]]
    merged = test_all.merge(long, on=["issue_time", "horizon"], how="left")
    train_all = _train_rows(s, train_years, start)
    info = {
        "Restarts": restarts,
        "Epochs": epochs,
        "Val_MSE": round(best_val, 6),
        "Val_MSE_const": round(const_val, 6),
        "n_train_issues": int(tr.sum()),
    }
    return (train_all, test_all, merged["y_pred_lstm"].to_numpy(float), fit_s, info)


def _tune_cache_path(algorithm, task, with_nwp, cfg, valid_year):
    return os.path.join(
        cfg.results_dir,
        "tuning",
        f"tuning_{task}_{('nwp' if with_nwp else 'no_nwp')}_{algorithm}_val{valid_year}.json",
    )


def tune(
    algorithm,
    task,
    s,
    features,
    cfg: MultistepConfig = CFG,
    base: pd.DataFrame | None = None,
    train_years: tuple | None = None,
    valid_year: int | None = None,
) -> dict:
    ensure_dirs(cfg)
    grid = cfg.param_grids.get(algorithm)
    if algorithm not in cfg.tune_algorithms or not grid:
        log.info(t("tune_skip"), algorithm)
        return {}
    train_years = tuple(train_years or cfg.tune_train_years)
    valid_year = int(valid_year or cfg.tune_valid_year)
    with_nwp = "NWP_GHI" in features
    cache = _tune_cache_path(algorithm, task, with_nwp, cfg, valid_year)
    sig = repr(
        (
            grid,
            list(features),
            len(s),
            cfg.nwp_shift_hours,
            cfg.nwp_shift_hours_post,
            cfg.nwp_availability_lag_h,
            train_years,
            valid_year,
            cfg.tune_horizons_intraday,
            cfg.tuning_subsample,
            cfg.random_seed,
        )
    )
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if os.path.exists(cache):
        c = json.load(open(cache))
        if c.get("sig") == sig:
            log.info(
                "   ♻️ %s | %s | %s: cached tuning → %s",
                algorithm,
                task,
                "nwp" if with_nwp else "no_nwp",
                c["best"],
            )
            return c["best"]
    log.info(t("tune_title"), train_years, valid_year)
    start = pd.Timestamp(f"{valid_year}-01-01")
    rows = []
    for params in grid:
        if algorithm == "LSTM":
            if base is None:
                raise ValueError("LSTM tuning requires base")
            cfg_p = replace(cfg, **params)
            _, te, yp, _, _ = lstm_task_predict(
                base, s, task, train_years, valid_year, with_nwp, cfg_p, cfg.random_seed
            )
            yt = te["y_true"].to_numpy(float)
            m = np.isfinite(yp)
            rmse = float(np.sqrt(np.mean((yp[m] - yt[m]) ** 2)))
        else:
            tr_all = _train_rows(s, train_years, start)
            va_all = s[(s["year"] == valid_year) & s["eval_ok"]]
            hs = cfg.tune_horizons_intraday if task == "intraday" else (None,)
            errs = []
            for h in hs:
                tr = tr_all if h is None else tr_all[tr_all["horizon"] == h]
                va = va_all if h is None else va_all[va_all["horizon"] == h]
                if len(tr) > cfg.tuning_subsample:
                    tr = tr.sample(cfg.tuning_subsample, random_state=cfg.random_seed)
                p, _ = _fit_predict(algorithm, tr, va, features, cfg, params)
                errs.append(float(np.sqrt(np.mean((p - va["y_true"].to_numpy(float)) ** 2))))
            rmse = float(np.mean(errs))
        rows.append({**{k: str(v) for k, v in params.items()}, "RMSE_2018": rmse})
        log.info(t("tune_combo"), str(params), rmse)
    k = int(np.argmin([r["RMSE_2018"] for r in rows]))
    best = grid[k]
    log.info(t("tune_best"), best)
    for key, val in best.items():
        vals = sorted({g[key] for g in grid if not isinstance(g[key], str)})
        if len(vals) > 2 and (not isinstance(val, str)) and (val in (vals[0], vals[-1])):
            log.info("   ⚠️ %s = %s is at the edge of the grid — consider widening it", key, val)
    pd.DataFrame(rows).to_csv(cache.replace(".json", ".csv"), index=False)
    json.dump({"sig": sig, "best": best}, open(cache, "w"))
    return best


def tuning_summary(cfg: MultistepConfig = CFG) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        for config in ("no_nwp", "nwp"):
            for algo in ALGORITHMS:
                for vy in sorted({ty - 1 for ty in cfg.test_years}):
                    f = _tune_cache_path(algo, task, config == "nwp", cfg, vy)
                    if os.path.exists(f):
                        rows.append(
                            {
                                "Task": task,
                                "Config": config,
                                "Model": algo,
                                "Test_year": vy + 1,
                                "Validation_year": vy,
                                "Selected": json.load(open(f))["best"],
                            }
                        )
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(cfg.results_dir, "tuning_summary.csv"), index=False)
    return out


def _metrics_rows(test_all, y_pred, refs, meta, cfg):
    d = test_all[["horizon", "y_true"]].reset_index(drop=True).assign(_p=np.asarray(y_pred, float))
    for name, arr in refs.items():
        d[f"_ref_{name}"] = np.asarray(arr, float)
    rows = []
    for h, g in d.groupby("horizon"):
        g = g[np.isfinite(g["_p"])]
        if len(g) < 10:
            continue
        yt, yp = (g["y_true"].to_numpy(float), g["_p"].to_numpy(float))
        r = {**meta, "Horizon": int(h), "n_test": int(len(g)), **compute_metrics(yt, yp, cfg)}
        for name in refs:
            ref = g[f"_ref_{name}"].to_numpy(float)
            rmse_ref = float(np.sqrt(np.mean((ref - yt) ** 2)))
            r[f"Skill_{name}"] = 1.0 - r["RMSE"] / rmse_ref if rmse_ref > 0 else np.nan
        rows.append(r)
    return rows


def run_algorithm(
    algorithm: str,
    task: str,
    config: str,
    base: pd.DataFrame,
    s: pd.DataFrame,
    cfg: MultistepConfig = CFG,
    tune_params: bool = True,
):
    ensure_dirs(cfg)
    with_nwp = config == "nwp"
    features = feature_columns(task, with_nwp, cfg)
    section(t("run_header") % (algorithm, task, config))
    seeds = cfg.lstm_seeds if algorithm == "LSTM" else (None,)
    res_rows, pred_frames = ([], [])
    for i, test_year in enumerate(cfg.test_years, start=1):
        train_years = tuple((y for y in cfg.years if y < test_year))
        params = (
            tune(
                algorithm,
                task,
                s,
                features,
                cfg,
                base=base,
                train_years=train_years[:-1],
                valid_year=train_years[-1],
            )
            if tune_params
            else {}
        )
        cfg_run = replace(cfg, **params) if algorithm == "LSTM" and params else cfg
        for seed in seeds:
            info = {}
            if algorithm == "LSTM":
                tr, te, yp, fit_s, info = lstm_task_predict(
                    base, s, task, train_years, test_year, with_nwp, cfg_run, seed
                )
            else:
                tr, te, yp, fit_s = tabular_task_predict(
                    algorithm, task, s, train_years, test_year, features, cfg, params
                )
            if seed in (None, seeds[0]):
                log.info(
                    t("fold_line"),
                    i,
                    "–".join(map(str, train_years)),
                    f"{len(tr):,}",
                    test_year,
                    f"{len(te):,}",
                )
            refs = reference_predictions(tr, te, cfg)
            meta = {
                "Task": task,
                "Config": config,
                "Model": algorithm,
                "Fold": test_year,
                "Seed": seed if seed is not None else cfg.random_seed,
                "Params": str(params) if params else "default",
                "Fit_s": round(fit_s, 2),
                **info,
            }
            rows = _metrics_rows(te, yp, refs, meta, cfg)
            res_rows += rows
            log.info(
                t("fold_done"),
                test_year,
                float(np.nanmean([r["Skill_SmartPersistence"] for r in rows])),
                fit_s,
            )
            if True:
                pred_frames.append(
                    pd.DataFrame(
                        {
                            "issue_time": te["issue_time"].to_numpy(),
                            "target_time": te["target_time"].to_numpy(),
                            "horizon": te["horizon"].to_numpy(),
                            "fold": test_year,
                            "y_true": te["y_true"].to_numpy(),
                            "y_pred": yp,
                            "ref_sp": refs["SmartPersistence"],
                            "ref_clim": refs["Climatology"],
                            "seed": seed if seed is not None else cfg.random_seed,
                        }
                    )
                )
    tag = f"{task}_{config}_{algorithm}"
    res = pd.DataFrame(res_rows)
    res.to_csv(os.path.join(cfg.results_dir, f"results_{tag}.csv"), index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(
        os.path.join(cfg.predictions_dir, f"{tag}.csv.gz"), index=False
    )
    log.info(t("saved"), f"results_{tag}.csv")
    return res


def run_references(task: str, config: str, s: pd.DataFrame, cfg: MultistepConfig = CFG):
    ensure_dirs(cfg)
    rows = []
    for test_year in cfg.test_years:
        train_years = tuple((y for y in cfg.years if y < test_year))
        tr = _train_rows(s, train_years, pd.Timestamp(f"{test_year}-01-01"))
        te = s[(s["year"] == test_year) & s["eval_ok"]]
        refs = reference_predictions(tr, te, cfg)
        for name, arr in refs.items():
            meta = {
                "Task": task,
                "Config": config,
                "Model": name,
                "Fold": test_year,
                "Seed": cfg.random_seed,
                "Params": "-",
                "Fit_s": 0.0,
            }
            rows += _metrics_rows(te, arr, refs, meta, cfg)
    res = pd.DataFrame(rows)
    res.to_csv(
        os.path.join(cfg.results_dir, f"results_{task}_{config}_References.csv"), index=False
    )
    return res


def load_results(task: str, config: str, cfg: MultistepConfig = CFG) -> pd.DataFrame:
    frames = []
    for name in ALGORITHMS + ("References",):
        f = os.path.join(cfg.results_dir, f"results_{task}_{config}_{name}.csv")
        if os.path.exists(f):
            frames.append(pd.read_csv(f))
    if not frames:
        raise RuntimeError(f"no results for {task}/{config}")
    return pd.concat(frames, ignore_index=True)


def summary_table(
    task: str, config: str, cfg: MultistepConfig = CFG, metric: str = "Skill_SmartPersistence"
) -> pd.DataFrame:
    r = load_results(task, config, cfg)
    g = r.groupby(["Model", "Horizon"])[metric].agg(["mean", "std"])
    tbl = g["mean"].unstack("Horizon").round(3)
    sd = g["std"].unstack("Horizon").round(3)
    section(t("sum_title") % (task, config))
    print(f"{metric} — mean")
    print(tbl.to_string())
    print(f"\n{metric} — std (folds{(' × seeds' if 'LSTM' in tbl.index else '')})")
    print(sd.to_string())
    out = os.path.join(cfg.results_dir, f"table_{task}_{config}_{metric}.csv")
    tbl.to_csv(out)
    return tbl


def plot_skill_by_horizon(
    task: str,
    config: str,
    cfg: MultistepConfig = CFG,
    metric: str = "Skill_SmartPersistence",
    dpi: int = 300,
):
    import matplotlib.pyplot as plt

    r = load_results(task, config, cfg)
    r = r[~r["Model"].isin(["SmartPersistence"])] if metric == "Skill_SmartPersistence" else r
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for m, g in r.groupby("Model"):
        s = g.groupby("Horizon")[metric].agg(["mean", "std"])
        ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", capsize=3, label=m)
    ax.axhline(0, color="#555", linewidth=1, linestyle="--")
    ax.set_xlabel("Horizon (h)")
    ax.set_ylabel(metric)
    ax.set_title(
        f"{task} | {config} — mean ± std ({len(cfg.test_years)} folds)",
        loc="left",
        fontweight="bold",
    )
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(fontsize=9, ncol=3)
    fig.tight_layout()
    out = os.path.join(cfg.results_dir, f"fig_skill_{task}_{config}.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.show()
    log.info(t("saved"), out)
    return fig


def check_nwp_alignment_by_source(
    base: pd.DataFrame, nwp: pd.DataFrame, cfg: MultistepConfig = CFG, shifts=(-2, -1, 0, 1, 2)
) -> pd.DataFrame:
    ensure_dirs(cfg)
    section("🕒 NWP time alignment — per file scheme")
    n = nwp[(nwp["lead_h"] >= 24) & (nwp["lead_h"] <= 47)].copy()
    switch = pd.Timestamp(cfg.nwp_scheme_switch)
    periods = {
        "full_backup+pp (2017–2018)": n[
            n["run_time"].dt.year.isin(cfg.tune_train_years + (cfg.tune_valid_year,))
        ],
        "meps_det (2020-02-04 → 2020-12-31)": n[
            (n["run_time"] >= switch) & (n["run_time"].dt.year == 2020)
        ],
    }
    obs = base["GHI"]
    rows = []
    for name, sub in periods.items():
        for sh in shifts:
            o = obs.reindex(pd.DatetimeIndex(sub["valid_time"] + pd.Timedelta(hours=sh))).to_numpy(
                float
            )
            f = sub["nwp_ghi"].to_numpy(float)
            m = np.isfinite(o) & np.isfinite(f) & (o > 5)
            r = np.corrcoef(o[m], f[m])[0, 1] if m.sum() > 100 else np.nan
            rows.append({"Scheme": name, "Shift_h": sh, "r": round(float(r), 4), "n": int(m.sum())})
    out = pd.DataFrame(rows)
    best = out.loc[out.groupby("Scheme")["r"].idxmax()][["Scheme", "Shift_h", "r"]]
    print(out.pivot(index="Shift_h", columns="Scheme", values="r").to_string())
    print("\nBest shift:")
    print(best.to_string(index=False))
    out.to_csv(os.path.join(cfg.nwp_dir, "nwp_alignment_by_scheme.csv"), index=False)
    return best


def nwp_quality_by_year(
    base: pd.DataFrame,
    nwp: pd.DataFrame,
    cfg: MultistepConfig = CFG,
    lead_ranges: dict | None = None,
) -> pd.DataFrame:
    ensure_dirs(cfg)
    section("🌦️ Annual accuracy of MEPS irradiance forecasts")
    lead_ranges = lead_ranges or {"intraday (4–33 h)": (4, 33), "dayahead (24–47 h)": (24, 47)}
    n = nwp.copy()
    n["vt"] = n["valid_time"] + pd.to_timedelta(_nwp_shift_for_runs(n["run_time"], cfg), unit="h")
    n["obs"] = base["GHI"].reindex(pd.DatetimeIndex(n["vt"])).to_numpy(float)
    n["cs"] = base["GHI_CS"].reindex(pd.DatetimeIndex(n["vt"])).to_numpy(float)
    n["Scheme"] = np.where(
        n["run_time"] >= pd.Timestamp(cfg.nwp_scheme_switch), "meps_det", "full_backup+pp"
    )
    n["Year"] = n["vt"].dt.year
    rows = []
    for lname, (a, b) in lead_ranges.items():
        sub = n[
            (n["lead_h"] >= a)
            & (n["lead_h"] <= b)
            & (n["cs"] > cfg.daytime_cs_threshold)
            & n["obs"].notna()
            & n["nwp_ghi"].notna()
        ]
        for (yr, sema), g in sub.groupby(["Year", "Scheme"]):
            e = g["nwp_ghi"].to_numpy(float) - g["obs"].to_numpy(float)
            mean_obs = float(g["obs"].mean())
            rows.append(
                {
                    "Lead": lname,
                    "Year": yr,
                    "Scheme": sema,
                    "n": len(g),
                    "RMSE_Wm2": round(float(np.sqrt(np.mean(e**2))), 1),
                    "Bias_Wm2": round(float(np.mean(e)), 1),
                    "nRMSE_pct_meanGHI": round(float(np.sqrt(np.mean(e**2)) / mean_obs * 100), 1),
                    "r": round(float(np.corrcoef(g["nwp_ghi"], g["obs"])[0, 1]), 3),
                }
            )
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(cfg.results_dir, "nwp_quality_by_year.csv"), index=False)
    return out


def choose_reference(task: str, s: pd.DataFrame, cfg: MultistepConfig = CFG) -> pd.DataFrame:
    ensure_dirs(cfg)
    tr = _train_rows(s, cfg.tune_train_years, pd.Timestamp(f"{cfg.tune_valid_year}-01-01"))
    va = s[(s["year"] == cfg.tune_valid_year) & s["eval_ok"]]
    refs = reference_predictions(tr, va, cfg)
    yt = va["y_true"].to_numpy(float)
    rows = []
    for grp, m in [("all", np.ones(len(va), bool))] + [
        (int(h), (va["horizon"] == h).to_numpy()) for h in sorted(va["horizon"].unique())
    ]:
        for k, v in refs.items():
            rows.append(
                {
                    "Task": task,
                    "Group": grp,
                    "Reference": k,
                    "nRMSE_2018": round(float(np.sqrt(np.mean((v[m] - yt[m]) ** 2))) * 100, 3),
                }
            )
    out = pd.DataFrame(rows)
    piv = out.pivot(index="Group", columns="Reference", values="nRMSE_2018")
    piv["Stronger"] = piv.idxmin(axis=1)
    section(f"📏 Reference comparison (2017 → 2018, test years unseen): {task}")
    print(piv.to_string())
    out.to_csv(os.path.join(cfg.results_dir, f"reference_choice_{task}.csv"), index=False)
    return piv


def load_predictions(
    task: str, config: str, cfg: MultistepConfig = CFG, include_refs: bool = True
) -> pd.DataFrame:
    frames = []
    for algo in ALGORITHMS:
        f = os.path.join(cfg.predictions_dir, f"{task}_{config}_{algo}.csv.gz")
        if not os.path.exists(f):
            continue
        p = pd.read_csv(f, parse_dates=["issue_time", "target_time"])
        if "seed" not in p.columns:
            p["seed"] = cfg.random_seed
        frames.append(
            p.assign(Model=algo)[
                [
                    "issue_time",
                    "target_time",
                    "horizon",
                    "fold",
                    "seed",
                    "Model",
                    "y_true",
                    "y_pred",
                ]
            ]
        )
        if include_refs and algo == "MLR":
            for col, name in (("ref_sp", "SmartPersistence"), ("ref_clim", "Climatology")):
                frames.append(
                    p.assign(Model=name, y_pred=p[col])[
                        [
                            "issue_time",
                            "target_time",
                            "horizon",
                            "fold",
                            "seed",
                            "Model",
                            "y_true",
                            "y_pred",
                        ]
                    ]
                )
    if not frames:
        raise RuntimeError(f"no predictions for {task}/{config} in {cfg.predictions_dir}")
    out = pd.concat(frames, ignore_index=True)
    return out[np.isfinite(out["y_pred"])]


def _common_rows(P: pd.DataFrame) -> pd.DataFrame:
    key = ["issue_time", "horizon"]
    cnt = P.drop_duplicates(key + ["Model"]).groupby(key)["Model"].nunique()
    full = cnt[cnt == P["Model"].nunique()].index
    return P.set_index(key).loc[lambda d: d.index.isin(full)].reset_index()


def _rows_both_configs(task: str, cfg: MultistepConfig) -> pd.MultiIndex:
    keys = None
    for config in ("no_nwp", "nwp"):
        P = _common_rows(load_predictions(task, config, cfg))
        k = pd.MultiIndex.from_frame(P[["issue_time", "horizon"]].drop_duplicates())
        keys = k if keys is None else keys.intersection(k)
    return keys


def pooled_skill_table(
    task: str,
    config: str,
    cfg: MultistepConfig = CFG,
    min_elev_deg: float | None = None,
    same_rows_across_configs: bool = True,
) -> pd.DataFrame:
    ensure_dirs(cfg)
    P = _common_rows(load_predictions(task, config, cfg))
    if same_rows_across_configs:
        keep = _rows_both_configs(task, cfg)
        P = P[pd.MultiIndex.from_frame(P[["issue_time", "horizon"]]).isin(keep)]
    if min_elev_deg is not None:
        tt = pd.DatetimeIndex(P["target_time"]) + pd.Timedelta(minutes=30)
        P = P[solar_elevation_deg(tt, cfg.latitude, cfg.longitude) >= min_elev_deg]
    refs = {}
    for name in ("SmartPersistence", "Climatology"):
        r = P[P["Model"] == name]
        refs[name] = r.groupby("fold").apply(
            lambda g: float(np.sqrt(np.mean((g["y_pred"] - g["y_true"]) ** 2)))
        )
    rows = []
    for (model, fold, seed), g in P.groupby(["Model", "fold", "seed"]):
        rmse = float(np.sqrt(np.mean((g["y_pred"] - g["y_true"]) ** 2)))
        rows.append(
            {
                "Model": model,
                "Fold": fold,
                "Seed": seed,
                "nRMSE_pct": rmse * 100,
                "Skill_SP": 1 - rmse / refs["SmartPersistence"][fold],
                "Skill_Clim": 1 - rmse / refs["Climatology"][fold],
                "n": len(g),
            }
        )
    R = pd.DataFrame(rows)
    agg = R.groupby("Model")[["nRMSE_pct", "Skill_SP", "Skill_Clim"]].agg(["mean", "std"]).round(3)
    agg = agg.sort_values(("nRMSE_pct", "mean"))
    tag = f"{task}_{config}" + (f"_elev{min_elev_deg:g}" if min_elev_deg is not None else "")
    section(f"📊 Pooled skill: {tag}  (rows: {int(R['n'].max()):,} per fold)")
    print(agg.to_string())
    R.to_csv(os.path.join(cfg.results_dir, f"pooled_skill_{tag}.csv"), index=False)
    return agg


def _nw_lrv(d: np.ndarray, lag: int) -> float:
    d = d - d.mean()
    T = len(d)
    v = float(np.dot(d, d) / T)
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * float(np.dot(d[k:], d[:-k]) / T)
    return v


def dm_test(e1: np.ndarray, e2: np.ndarray, days: np.ndarray) -> dict:
    from scipy import stats

    d = pd.Series(e1**2 - e2**2).groupby(np.asarray(days)).sum().to_numpy(float)
    T = len(d)
    if T < 30:
        return {"N_days": T, "DM": np.nan, "p": np.nan}
    lag = int(np.floor(4 * (T / 100) ** (2 / 9)))
    lrv = _nw_lrv(d, lag)
    if lrv <= 0:
        return {"N_days": T, "DM": np.nan, "p": np.nan}
    dm = d.mean() / np.sqrt(lrv / T)
    dm *= np.sqrt((T - 1) / T)
    p = 2 * stats.t.sf(abs(dm), df=T - 1)
    return {"N_days": T, "DM": float(dm), "p": float(p)}


def _holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    m = len(q)
    if m == 0:
        return out
    order = np.argsort(q)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * q[j])
        adj[j] = min(1.0, run)
    out[ok] = adj
    return out


def _seed_list(P, model):
    return sorted(P.loc[P["Model"] == model, "seed"].unique())


def pairwise_dm(
    task: str, config: str, cfg: MultistepConfig = CFG, alpha: float = 0.05, by_horizon: bool = True
) -> pd.DataFrame:
    ensure_dirs(cfg)
    P = _common_rows(load_predictions(task, config, cfg))
    P["day"] = P["target_time"].dt.normalize()
    models = sorted(P["Model"].unique())
    groups = [("all", P)]
    if by_horizon:
        groups += [(int(h), g) for h, g in P.groupby("horizon")]
    rows = []
    for gname, G in groups:
        piv = {}
        for m in models:
            for sd in _seed_list(G, m):
                piv[m, sd] = (
                    G[(G["Model"] == m) & (G["seed"] == sd)]
                    .set_index(["issue_time", "horizon"])
                    .sort_index()
                )
        grows = []
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                best = None
                for sa in _seed_list(G, a):
                    for sb in _seed_list(G, b):
                        A, B = (piv[a, sa], piv[b, sb])
                        idx = A.index.intersection(B.index)
                        A_, B_ = (A.loc[idx], B.loc[idx])
                        r = dm_test(
                            (A_["y_pred"] - A_["y_true"]).to_numpy(float),
                            (B_["y_pred"] - B_["y_true"]).to_numpy(float),
                            A_["day"].to_numpy(),
                        )
                        r["dRMSE_pct"] = 100 * (
                            np.sqrt(np.mean((A_["y_pred"] - A_["y_true"]) ** 2))
                            - np.sqrt(np.mean((B_["y_pred"] - B_["y_true"]) ** 2))
                        )
                        if best is None or (np.isfinite(r["p"]) and r["p"] > best["p"]):
                            best = r
                grows.append(
                    {"Task": task, "Config": config, "Group": gname, "A": a, "B": b, **best}
                )
        gdf = pd.DataFrame(grows)
        gdf["p_holm"] = _holm(gdf["p"].to_numpy())
        gdf["Result"] = np.where(
            gdf["p_holm"] >= alpha,
            "indistinguishable",
            np.where(gdf["DM"] < 0, "A better", "B better"),
        )
        rows.append(gdf)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(cfg.results_dir, f"dm_pairs_{task}_{config}.csv"), index=False)
    return out


def best_set(
    dm: pd.DataFrame, task: str, config: str, cfg: MultistepConfig = CFG, alpha: float = 0.05
) -> pd.DataFrame:
    ensure_dirs(cfg)
    P = _common_rows(load_predictions(task, config, cfg))
    rows = []
    for gname in dm["Group"].unique():
        G = P if gname == "all" else P[P["horizon"] == gname]
        rm = (
            G[G["Model"].isin(ALGORITHMS)]
            .groupby("Model")
            .apply(lambda g: float(np.sqrt(np.mean((g["y_pred"] - g["y_true"]) ** 2))) * 100)
            .sort_values()
        )
        top = rm.index[0]
        d = dm[dm["Group"] == gname]
        tied = []
        for m in rm.index[1:]:
            r = d[(d["A"] == top) & (d["B"] == m) | (d["A"] == m) & (d["B"] == top)]
            if len(r) and r["p_holm"].iloc[0] >= alpha:
                tied.append(m)
        rows.append(
            {
                "Group": gname,
                "Best": top,
                "nRMSE_best": round(rm.iloc[0], 3),
                "Indistinguishable": ", ".join(tied) or "—",
            }
        )
    out = pd.DataFrame(rows)
    section(f"🏁 Best set (Holm, α = {alpha}): {task} | {config}")
    print(out.to_string(index=False))
    out.to_csv(os.path.join(cfg.results_dir, f"best_set_{task}_{config}.csv"), index=False)
    return out


def nwp_gain_dm(task: str, cfg: MultistepConfig = CFG, alpha: float = 0.05) -> pd.DataFrame:
    ensure_dirs(cfg)
    A = load_predictions(task, "nwp", cfg, include_refs=False)
    B = load_predictions(task, "no_nwp", cfg, include_refs=False)
    key = ["Model", "seed", "issue_time", "horizon"]
    M = A.merge(B[key + ["y_pred"]], on=key, suffixes=("_nwp", "_no"))
    M["day"] = M["target_time"].dt.normalize()
    rows = []
    for gname, G in [("all", M)] + [(int(h), g) for h, g in M.groupby("horizon")]:
        grows = []
        for algo, g in G.groupby("Model"):
            best = None
            for sd, gs in g.groupby("seed"):
                e1 = (gs["y_pred_nwp"] - gs["y_true"]).to_numpy(float)
                e0 = (gs["y_pred_no"] - gs["y_true"]).to_numpy(float)
                r = dm_test(e1, e0, gs["day"].to_numpy())
                r["nRMSE_no"] = 100 * np.sqrt(np.mean(e0**2))
                r["nRMSE_nwp"] = 100 * np.sqrt(np.mean(e1**2))
                r["Gain_pct"] = 100 * (1 - np.sqrt(np.mean(e1**2)) / np.sqrt(np.mean(e0**2)))
                if best is None or (np.isfinite(r["p"]) and r["p"] > best["p"]):
                    best = r
            grows.append({"Task": task, "Group": gname, "Model": algo, **best})
        gdf = pd.DataFrame(grows)
        gdf["p_holm"] = _holm(gdf["p"].to_numpy())
        rows.append(gdf)
    out = pd.concat(rows, ignore_index=True).round(4)
    section(f"🌦️ NWP gain (DM, Holm): {task}")
    print(out[out["Group"] == "all"].to_string(index=False))
    out.to_csv(os.path.join(cfg.results_dir, f"dm_nwp_gain_{task}.csv"), index=False)
    return out


def r2_table(cfg: MultistepConfig = CFG) -> pd.DataFrame:
    ensure_dirs(cfg)
    from sklearn.metrics import r2_score

    rows = []
    for task in TASKS:
        for config in ("no_nwp", "nwp"):
            try:
                P = _common_rows(load_predictions(task, config, cfg))
            except RuntimeError:
                continue
            for (m, fold, sd), g in P.groupby(["Model", "fold", "seed"]):
                rows.append(
                    {
                        "Task": task,
                        "Config": config,
                        "Model": m,
                        "Fold": fold,
                        "Seed": sd,
                        "R2": r2_score(g["y_true"], g["y_pred"]),
                    }
                )
    R = pd.DataFrame(rows)
    agg = R.groupby(["Model", "Task", "Config"])["R2"].agg(["mean", "std"])
    tbl = (
        agg["mean"].round(3).astype(str) + " ± " + agg["std"].fillna(0).round(3).astype(str)
    ).unstack(["Task", "Config"])
    print(tbl.to_string())
    R.to_csv(os.path.join(cfg.results_dir, "r2_all_algorithms.csv"), index=False)
    return tbl


def learning_curve(
    base: pd.DataFrame,
    samples: dict,
    cfg: MultistepConfig = CFG,
    test_year: int = 2021,
    config: str = "nwp",
    algos=("MLR", "KNN", "RandomForest", "XGBoost"),
) -> pd.DataFrame:
    ensure_dirs(cfg)
    windows = {
        k: tuple(range(test_year - k, test_year)) for k in range(1, test_year - min(cfg.years) + 1)
    }
    rows = []
    for task in TASKS:
        s = samples[task]
        feats = feature_columns(task, config == "nwp", cfg)
        params = {
            a: tune(
                a,
                task,
                s,
                feats,
                cfg,
                train_years=tuple(range(min(cfg.years), test_year - 1)),
                valid_year=test_year - 1,
            )
            for a in algos
        }
        for n_years, train_years in windows.items():
            for algo in algos:
                tr, te, yp, _ = tabular_task_predict(
                    algo, task, s, train_years, test_year, feats, cfg, params[algo]
                )
                refs = reference_predictions(tr, te, cfg)
                yt = te["y_true"].to_numpy(float)
                m = np.isfinite(yp)
                rmse = float(np.sqrt(np.mean((yp[m] - yt[m]) ** 2)))
                rows.append(
                    {
                        "Task": task,
                        "Model": algo,
                        "N_train_years": n_years,
                        "R2": compute_metrics(yt[m], yp[m], cfg)["R2"],
                        "nRMSE_pct": rmse * 100,
                        "Skill_SP": 1
                        - rmse / np.sqrt(np.mean((refs["SmartPersistence"][m] - yt[m]) ** 2)),
                        "Skill_Clim": 1
                        - rmse / np.sqrt(np.mean((refs["Climatology"][m] - yt[m]) ** 2)),
                    }
                )
    LC = pd.DataFrame(rows)
    for metric in ("R2", "nRMSE_pct"):
        print(
            f"\n=== {metric} — test {test_year}, {config} | columns: number of training years ==="
        )
        print(
            LC.pivot_table(index=["Task", "Model"], columns="N_train_years", values=metric)
            .round(3)
            .to_string()
        )
    LC.to_csv(
        os.path.join(cfg.results_dir, f"learning_curve_{config}_{test_year}.csv"), index=False
    )
    return LC
