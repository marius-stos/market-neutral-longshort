"""
signal_enhancers.py
-------------------
Signal enhancement utilities for the quant long/short strategy.
Includes earnings filtering, HMM regime detection, ML composite scoring,
and VWAP slippage adjustment.

Python 3.9 compatible.
"""

import json
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VWAP_SLIPPAGE_BPS = 8
OUTPUT_DIR = Path("output")
EARNINGS_CACHE_FILE = OUTPUT_DIR / "earnings_cache.json"

# Global in-memory ML training buffer
ML_TRAINING_DATA: List[dict] = []

# ---------------------------------------------------------------------------
# 1. Earnings calendar
# ---------------------------------------------------------------------------


def load_earnings_calendar(tickers: List[str], cache_days: int = 6) -> Dict[str, Optional[datetime]]:
    """
    Fetch the next earnings date for each ticker via yfinance.

    Results are cached to output/earnings_cache.json and refreshed if
    the cache is older than *cache_days* days.

    Returns
    -------
    dict
        {ticker: datetime or None}
    """
    import yfinance as yf  # imported here so the rest of the module loads even if yf is absent

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- load / validate cache ---
    cached: Dict[str, Optional[str]] = {}
    cache_stale = True
    if EARNINGS_CACHE_FILE.exists():
        try:
            raw = json.loads(EARNINGS_CACHE_FILE.read_text())
            ts = raw.get("_ts", 0)
            if (datetime.utcnow().timestamp() - ts) < cache_days * 86_400:
                cached = raw.get("data", {})
                cache_stale = False
                logger.debug("Earnings cache hit (%d tickers).", len(cached))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read earnings cache: %s", exc)

    if not cache_stale:
        result: Dict[str, Optional[datetime]] = {}
        for tk in tickers:
            raw_val = cached.get(tk)
            result[tk] = datetime.fromisoformat(raw_val) if raw_val else None
        return result

    # --- fetch in parallel ---
    def _fetch_one(ticker: str) -> Tuple[str, Optional[datetime]]:
        try:
            info = yf.Ticker(ticker).calendar
            if info is not None and not info.empty:
                # calendar is a DataFrame; earnings date lives in first column
                ed_raw = info.iloc[0, 0]
                ed = pd.Timestamp(ed_raw).to_pydatetime() if ed_raw else None
                return ticker, ed
        except Exception as exc:  # noqa: BLE001
            logger.debug("Earnings fetch failed for %s: %s", ticker, exc)
        return ticker, None

    result = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_one, tk): tk for tk in tickers}
        for fut in as_completed(futures):
            tk, ed = fut.result()
            result[tk] = ed

    # --- persist cache ---
    try:
        serialisable = {tk: (v.isoformat() if v else None) for tk, v in result.items()}
        EARNINGS_CACHE_FILE.write_text(
            json.dumps({"_ts": datetime.utcnow().timestamp(), "data": serialisable}, indent=2)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write earnings cache: %s", exc)

    logger.info("Earnings calendar fetched for %d tickers.", len(result))
    return result


# ---------------------------------------------------------------------------
# 2. Earnings filter
# ---------------------------------------------------------------------------


def filter_earnings_tickers(
    tickers: List[str],
    rebal_date: pd.Timestamp,
    earnings_cal: Dict[str, Optional[datetime]],
    window_days: int = 3,
) -> Tuple[List[str], List[str]]:
    """
    Split *tickers* into (safe, excluded) based on proximity to earnings.

    A ticker is excluded when its next earnings date falls within
    *window_days* calendar days of *rebal_date*.
    """
    safe: List[str] = []
    excluded: List[str] = []

    rebal_dt = rebal_date.to_pydatetime().replace(tzinfo=None)
    cutoff_lo = rebal_dt - timedelta(days=window_days)
    cutoff_hi = rebal_dt + timedelta(days=window_days)

    for tk in tickers:
        ed = earnings_cal.get(tk)
        if ed is None:
            safe.append(tk)
            continue
        ed_naive = ed.replace(tzinfo=None) if ed.tzinfo else ed
        if cutoff_lo <= ed_naive <= cutoff_hi:
            excluded.append(tk)
            logger.debug("Excluding %s: earnings on %s", tk, ed_naive.date())
        else:
            safe.append(tk)

    logger.info(
        "Earnings filter: %d safe, %d excluded (window=±%dd, rebal=%s).",
        len(safe), len(excluded), window_days, rebal_date.date(),
    )
    return safe, excluded


# ---------------------------------------------------------------------------
# 3. HMM regime detection
# ---------------------------------------------------------------------------

_REGIME_LABELS = {0: "Bull", 1: "Choppy", 2: "Bear"}

_REGIME_META = {
    "Bull":   {"allow_short": False, "gross_long": 0.95},
    "Choppy": {"allow_short": False, "gross_long": 0.65},
    "Bear":   {"allow_short": True,  "gross_long": 0.50},
}


def detect_regime_hmm(spy_returns: pd.Series, n_states: int = 3) -> dict:
    """
    Detect the current market regime using a GaussianHMM.

    Falls back to an SMA(200) rule when fewer than 252 data points are
    available.  Returns a dict with keys: label, allow_short, gross_long.
    """
    MIN_OBS = 252

    def _sma_fallback(returns: pd.Series) -> dict:
        cumulative = (1 + returns).cumprod()
        sma200 = cumulative.rolling(200, min_periods=1).mean()
        if cumulative.iloc[-1] > sma200.iloc[-1]:
            label = "Bull"
        elif cumulative.iloc[-1] < sma200.iloc[-1] * 0.97:
            label = "Bear"
        else:
            label = "Choppy"
        logger.info("Regime (SMA fallback): %s", label)
        return {"label": label, **_REGIME_META[label]}

    if len(spy_returns) < MIN_OBS:
        logger.warning(
            "Only %d return observations; need %d for HMM. Using SMA fallback.",
            len(spy_returns), MIN_OBS,
        )
        return _sma_fallback(spy_returns)

    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore
    except ImportError:
        logger.warning("hmmlearn not installed; falling back to SMA regime.")
        return _sma_fallback(spy_returns)

    rets = spy_returns.dropna().values.reshape(-1, 1)
    vol21 = pd.Series(spy_returns.dropna()).rolling(21).std().bfill().values.reshape(-1, 1)
    vol63 = pd.Series(spy_returns.dropna()).rolling(63).std().bfill().values.reshape(-1, 1)
    X = np.hstack([rets, vol21, vol63])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        model.fit(X)

    hidden_states = model.predict(X)
    current_state = int(hidden_states[-1])

    # Rank states by mean return: highest → Bull, lowest → Bear
    state_means = {s: float(X[hidden_states == s, 0].mean()) for s in range(n_states)}
    ranked = sorted(state_means, key=state_means.get, reverse=True)  # type: ignore[arg-type]
    state_to_label = {ranked[0]: "Bull", ranked[1]: "Choppy", ranked[2]: "Bear"}

    label = state_to_label[current_state]
    logger.info(
        "Regime (HMM): %s (state=%d, mean_ret=%.5f).",
        label, current_state, state_means[current_state],
    )
    return {"label": label, **_REGIME_META[label]}


# ---------------------------------------------------------------------------
# 4 & 5. ML training data recording and composite scoring
# ---------------------------------------------------------------------------


def record_signals_for_ml(signals: pd.DataFrame, forward_returns: pd.Series) -> None:
    """
    Append (signal_row, forward_return) pairs to the global ML_TRAINING_DATA buffer.

    Each record is a flat dict of signal values plus a 'fwd_ret' key.
    """
    common = signals.index.intersection(forward_returns.index)
    for ticker in common:
        row = signals.loc[ticker].to_dict()
        row["fwd_ret"] = float(forward_returns.loc[ticker])
        ML_TRAINING_DATA.append(row)

    logger.debug("ML buffer now has %d records.", len(ML_TRAINING_DATA))


_EXCLUDE_COLS = {"composite", "rvol_raw", "mom_3m_raw", "mom_6m_raw", "sector"}


def compute_ml_composite(signals: pd.DataFrame):  # -> Optional[pd.Series]
    """
    Train a GradientBoostingClassifier on ML_TRAINING_DATA and return a
    ranked composite score [0, 1] for each ticker in *signals*.

    Returns None when fewer than 200 training records are available
    (approximately 18 months of data).
    """
    MIN_RECORDS = 200

    if len(ML_TRAINING_DATA) < MIN_RECORDS:
        logger.debug(
            "ML composite skipped: only %d records (need %d).",
            len(ML_TRAINING_DATA), MIN_RECORDS,
        )
        return None

    try:
        from sklearn.ensemble import GradientBoostingClassifier  # type: ignore
    except ImportError:
        logger.warning("scikit-learn not installed; ML composite unavailable.")
        return None

    # Build training matrix
    sample_keys = [k for k in ML_TRAINING_DATA[0] if k not in _EXCLUDE_COLS and k != "fwd_ret"]
    rows, labels = [], []
    for rec in ML_TRAINING_DATA:
        try:
            rows.append([float(rec.get(k, 0.0) or 0.0) for k in sample_keys])
            labels.append(1 if float(rec["fwd_ret"]) > 0 else 0)
        except (ValueError, KeyError):
            continue

    if len(rows) < MIN_RECORDS:
        return None

    X_train = np.array(rows, dtype=np.float32)
    y_train = np.array(labels, dtype=np.int32)

    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
    )
    clf.fit(X_train, y_train)

    # Predict on current signals
    feat_cols = [c for c in signals.columns if c not in _EXCLUDE_COLS]
    X_pred = signals[feat_cols].select_dtypes(include=[np.number]).fillna(0.0).values
    pred_keys = [c for c in signals.select_dtypes(include=[np.number]).columns if c not in _EXCLUDE_COLS]

    # Align feature order with training
    col_idx = {k: i for i, k in enumerate(pred_keys)}
    ordered_idx = [col_idx[k] for k in sample_keys if k in col_idx]
    X_pred_aligned = X_pred[:, ordered_idx] if ordered_idx else X_pred

    proba = clf.predict_proba(X_pred_aligned)[:, 1]
    scores = pd.Series(proba, index=signals.index)
    ranked = scores.rank(pct=True)

    logger.info("ML composite computed for %d tickers.", len(ranked))
    return ranked


# ---------------------------------------------------------------------------
# 6. VWAP slippage
# ---------------------------------------------------------------------------


def apply_vwap_slippage(port_returns: pd.Series, gross_exposure: float) -> pd.Series:
    """
    Deduct estimated VWAP round-trip slippage from the first return period.

    Slippage = VWAP_SLIPPAGE_BPS * 2 * gross_exposure / 10_000
    """
    slippage = VWAP_SLIPPAGE_BPS * 2 * gross_exposure / 10_000
    adjusted = port_returns.copy()
    if len(adjusted) > 0:
        adjusted.iloc[0] -= slippage
        logger.debug(
            "VWAP slippage applied: %.2f bps (gross_exp=%.2f).",
            VWAP_SLIPPAGE_BPS * 2, gross_exposure,
        )
    return adjusted
