"""
Global Momentum Market-Neutral Backtest  — Phase 2 (enhanced signals)
======================================================================
Universe   : ~580 global stocks (US S&P 500 + EU/Asia ADRs)
Signal     : Residual momentum + IVOL + 52-wk-high (sector-relative, market-adjusted)
Portfolio  : 39L / 39S — overlapping sub-portfolios (3×13, held 3 months each)
             Conviction-scaled weights (|signal| × 1/vol)
Costs      : Differentiated slippage (8 bps US / 15 bps ADR) +
             Borrow costs (0.5 %/yr US / 1.5 %/yr ADR) on short leg
Rebalance  : Monthly — 1/3 of portfolio rotated each month (turnover ≈30%/yr)
Regime     : SPY drawdown + 1m return + vol filter → gross multiplier [0.5, 1.0]
Earnings   : Positions within 7 days of earnings halved (blackout filter)
Output     : output/walkforward_results.json  (Dash-compatible)
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from rich.logging import RichHandler

from universe_global import build_global_universe, EU_TICKERS, ASIA_TICKERS
try:
    from edgar_fundamentals import load_edgar_data, precompute_pit_panel
    EDGAR_AVAILABLE = True
except ImportError:
    EDGAR_AVAILABLE = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_DIR   = Path("output")
BENCHMARK    = "SPY"

N_LONG       = 39      # total longs (3 sub-portfolios × 13)
N_SHORT      = 39      # total shorts (3 sub-portfolios × 13)
GROSS        = 1.0     # total gross (0.5 each leg)

# ── Overlapping sub-portfolios (improvement #1 — cut turnover ≈70%) ───────────
HOLD_PERIODS = 3       # months each sub-portfolio is held
N_SUB_LONG   = N_LONG  // HOLD_PERIODS   # 13 new longs added per month
N_SUB_SHORT  = N_SHORT // HOLD_PERIODS   # 10 new shorts added per month

# ── Phase 3 improvement #1: dedicated short-alpha signal ──────────────────────
# DISABLED: weighting shorts toward HIGH idiosyncratic vol backfired — high-IVOL
# names squeeze hardest in rallies (junk/high-beta), worsening MaxDD −12%→−17%.
# The IVOL anomaly only pays on the long side (avoid them), not as a short signal.
# Reverted to Phase-2 composite-based short selection.
USE_SHORT_SCORE = False
W_SHORT_MOM   = 0.60
W_SHORT_IVOL  = 0.40

# ── Phase 3 improvement #2: short-term reversal sleeve ────────────────────────
# DISABLED: a 1-month reversal signal decays within days/weeks, so holding it
# across a 3-month overlapping window destroys the alpha and dilutes momentum.
# Tested at weight 0.20 → long leg dropped +95%→+66%.  Kept off.
USE_REVERSAL     = False
REVERSAL_WEIGHT  = 0.0
REVERSAL_LOOKBACK = 21

# ── Phase 3 improvement #3: portfolio volatility CAP (de-lever only) ──────────
# Scale the book DOWN when its trailing realised vol exceeds target, never up.
# Levering up into calm periods (max>1) amplified 2022/2025 crashes, so the
# scalar is capped at 1.0 — this is a pure risk cap, not a vol target.
USE_VOL_TARGET = True
VOL_TARGET     = 0.05    # 5% annualised cap — optimum of the Phase 4 param sweep
VOL_LOOKBACK   = 42      # trailing days for realised-vol estimate (21d/15d too noisy)
VOL_SCALE_MIN  = 0.50    # de-lever down to at most 0.5×
VOL_SCALE_MAX  = 1.00    # never lever above 1.0× (cap only)

# ── Earnings blackout (improvement #4) ────────────────────────────────────────
USE_EARNINGS_FILTER = True   # halve weight for positions within 7 days of earnings
EARNINGS_WINDOW_DAYS = 7     # days before/after earnings to reduce exposure
EARNINGS_WEIGHT_MULT = 0.50  # weight multiplier during blackout

MIN_HISTORY  = 252     # trading days needed before first rebal
MIN_VALID_PX = 200     # min non-NaN price days to include a ticker

# ── Realistic transaction costs ───────────────────────────────────────────────
# Slippage: S&P 500 names are liquid (8 bps); ADRs have wider spreads (15 bps)
SLIP_US_BPS   = 8      # one-way, US large caps
SLIP_INTL_BPS = 15     # one-way, EU/Asia ADRs

# Short-selling borrow costs (annualised, charged daily on short notional)
BORROW_US_PA   = 0.005  # 0.5 % / yr  — liquid S&P 500 names
BORROW_INTL_PA = 0.015  # 1.5 % / yr  — international ADRs (less liquid, harder to borrow)

# Ticker classification: international ADRs set (used for cost differentiation)
INTL_TICKERS: frozenset = frozenset({**EU_TICKERS, **ASIA_TICKERS}.keys())

# Factor weights — pure price-based to avoid yfinance look-ahead bias
# (yfinance .info returns 2024 snapshot data for ALL historical periods)
# When EDGAR PIT data is available for a ticker:
#   composite = W_MOM_EDGAR * momentum + (1 - W_MOM_EDGAR) * earnings_growth_signal
# Only earnings growth (PEAD-aligned) is used — Piotroski/ROE/accruals are anti-momentum.
# International ADRs (no EDGAR data): 100% momentum
W_MOM_EDGAR  = 0.75   # momentum weight for tickers WITH EDGAR PIT data
W_MOMENTUM   = 1.00   # fallback weight for tickers WITHOUT EDGAR data
W_LOW_VOL    = 0.00   # position sizing only (inverse-vol weights) — not in signal
W_QUALITY    = 0.00   # disabled: yfinance look-ahead bias
W_VALUE      = 0.00   # disabled: yfinance look-ahead bias

USE_EDGAR    = False   # EDGAR earnings growth is redundant with price momentum → disabled

FUND_CACHE_DAYS  = 7   # refresh fundamentals every N days
PRICE_CACHE_DAYS = 1

# ---------------------------------------------------------------------------
# Price loading
# ---------------------------------------------------------------------------

def load_prices(
    tickers: List[str],
    start: str = "2015-01-01",
    force_refresh: bool = False,
) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUTPUT_DIR / "prices_cache.parquet"

    if not force_refresh and cache.exists():
        age = (datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime)).days
        if age < PRICE_CACHE_DAYS:
            df = pd.read_parquet(cache)
            # add any missing tickers (new ones)
            missing = [t for t in tickers if t not in df.columns]
            if not missing:
                logger.info("Prices from cache: %s", df.shape)
                return df
            logger.info("Fetching %d new tickers not in cache", len(missing))
            extra = _download_prices(missing, start)
            df = pd.concat([df, extra], axis=1)
            df.to_parquet(cache)
            return df

    df = _download_prices(tickers, start)
    df.to_parquet(cache)
    return df


def _download_prices(tickers: List[str], start: str) -> pd.DataFrame:
    logger.info("Downloading prices for %d tickers from %s …", len(tickers), start)
    raw = yf.download(
        tickers, start=start, auto_adjust=True, progress=False, threads=True
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw["Close"]
    raw = raw.dropna(axis=1, how="all")
    logger.info("Prices downloaded: %s", raw.shape)
    return raw


# ---------------------------------------------------------------------------
# Fundamentals loading  (yfinance .info, cached per ticker)
# ---------------------------------------------------------------------------
FUND_FIELDS = [
    "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
    "enterpriseToEbitda", "marketCap", "freeCashflow", "operatingCashflow",
    "returnOnEquity", "returnOnAssets", "grossMargins", "operatingMargins",
    "profitMargins", "ebitdaMargins", "debtToEquity", "currentRatio",
    "totalRevenue", "totalDebt", "totalCash", "revenueGrowth", "earningsGrowth",
    "beta", "shortPercentOfFloat", "trailingEps", "bookValue",
]


def load_fundamentals(
    tickers: List[str], force_refresh: bool = False
) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUTPUT_DIR / "fundamentals_cache.parquet"

    if not force_refresh and cache.exists():
        age = (datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime)).days
        if age < FUND_CACHE_DAYS:
            df = pd.read_parquet(cache)
            logger.info("Fundamentals from cache (%d tickers)", len(df))
            return df

    logger.info("Fetching fundamentals for %d tickers (yfinance) …", len(tickers))
    rows: Dict[str, Dict] = {}
    batch = 50
    for i in range(0, len(tickers), batch):
        chunk = tickers[i : i + batch]
        for t in chunk:
            try:
                info = yf.Ticker(t).info
                rows[t] = {f: info.get(f) for f in FUND_FIELDS}
            except Exception:
                rows[t] = {}
        if i % 200 == 0:
            logger.info("  Fundamentals: %d / %d", i + len(chunk), len(tickers))

    df = pd.DataFrame(rows).T
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_parquet(cache)
    logger.info("Fundamentals cached: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Factor computation
# ---------------------------------------------------------------------------

def _rank(s: pd.Series) -> pd.Series:
    """Cross-sectional rank → [−0.5, +0.5], NaN → 0."""
    r = s.rank(pct=True)
    return (r - 0.5).fillna(0.0)


# ---------------------------------------------------------------------------
# New signal helpers (Phase 2)
# ---------------------------------------------------------------------------

def compute_ivol(
    prices: pd.DataFrame,
    tickers: List[str],
    spy_col: str = BENCHMARK,
    window: int = 63,
) -> pd.Series:
    """
    Idiosyncratic volatility = annualised std of CAPM residuals over `window` days.
    High IVOL predicts LOWER future returns (Ang et al. 2006) → negative alpha signal.
    """
    ivol = pd.Series(0.20, index=tickers)
    if spy_col not in prices.columns or len(prices) < window + 5:
        return ivol
    rets = prices.pct_change().iloc[-window:].dropna(how="all")
    spy_r = rets[spy_col].dropna()
    spy_var = float(spy_r.var())
    if spy_var < 1e-10:
        return ivol
    for t in tickers:
        if t not in rets.columns:
            continue
        r = rets[t].dropna()
        sp = spy_r.reindex(r.index).dropna()
        r, sp = r.align(sp, join="inner")
        if len(r) < 30:
            continue
        beta_t = float(np.cov(r.values, sp.values)[0, 1] / spy_var)
        resid  = r.values - beta_t * sp.values
        ivol[t] = float(resid.std() * np.sqrt(252))
    return ivol.clip(lower=0.03)


def compute_residual_momentum(
    prices: pd.DataFrame,
    tickers: List[str],
    spy_col: str = BENCHMARK,
    lookback: int = 252,
    skip: int = 21,
) -> pd.Series:
    """
    Market-adjusted (residual) 12-1 month momentum.
    Each stock's cumulative excess return over the market's own contribution.
    Eliminates the systematic beta-return component so the ranking reflects
    pure stock-specific momentum, not sector/factor rotation.
    """
    resid_mom = pd.Series(np.nan, index=tickers)
    if spy_col not in prices.columns or len(prices) < lookback + skip + 5:
        return resid_mom
    rets = prices.pct_change().dropna(how="all")
    spy_r = rets[spy_col]

    # Beta estimated on first half of window to avoid lookahead in signal
    beta_window = lookback // 2
    for t in tickers:
        if t not in rets.columns:
            continue
        r = rets[t]
        # Beta from t-252 to t-126
        r_beta = r.iloc[-(lookback):-beta_window].dropna()
        sp_beta = spy_r.reindex(r_beta.index).dropna()
        r_beta, sp_beta = r_beta.align(sp_beta, join="inner")
        if len(r_beta) < 40:
            continue
        beta_t = float(np.cov(r_beta.values, sp_beta.values)[0, 1] /
                       max(sp_beta.var(), 1e-10))
        beta_t = float(np.clip(beta_t, -2.0, 3.0))

        # Residual cumulative return from t-252 to t-21 (skip last month)
        r_signal = r.iloc[-(lookback):-skip].dropna()
        sp_signal = spy_r.reindex(r_signal.index).dropna()
        r_signal, sp_signal = r_signal.align(sp_signal, join="inner")
        if len(r_signal) < 60:
            continue
        excess = r_signal.values - beta_t * sp_signal.values
        resid_mom[t] = float(excess.sum())   # cumulative residual return
    return resid_mom


def preload_earnings_calendar(
    tickers: List[str],
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> Dict[str, List[pd.Timestamp]]:
    """
    Fetch upcoming + recent earnings dates for all tickers (cached to disk).
    Gracefully skips tickers where yfinance has no data.
    """
    if cache_path is None:
        cache_path = OUTPUT_DIR / "earnings_calendar.json"

    if not force_refresh and cache_path.exists():
        age = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).days
        if age < 7:
            raw = json.loads(cache_path.read_text())
            cal: Dict[str, List[pd.Timestamp]] = {}
            for t, dates in raw.items():
                cal[t] = [pd.Timestamp(d) for d in dates]
            logger.info("Earnings calendar from cache (%d tickers)", len(cal))
            return cal

    logger.info("Fetching earnings calendar for %d tickers …", len(tickers))
    cal = {}
    for i, t in enumerate(tickers):
        try:
            ed = yf.Ticker(t).earnings_dates
            if ed is not None and not ed.empty:
                cal[t] = [ts for ts in ed.index if isinstance(ts, pd.Timestamp)]
        except Exception:
            pass
        if i % 100 == 99:
            logger.info("  Earnings: %d / %d", i + 1, len(tickers))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(
        {t: [str(d) for d in dates] for t, dates in cal.items()}
    ))
    logger.info("Earnings calendar saved (%d tickers with data)", len(cal))
    return cal


def get_earnings_risk_tickers(
    tickers: List[str],
    calendar: Dict[str, List[pd.Timestamp]],
    check_date: pd.Timestamp,
    window: int = EARNINGS_WINDOW_DAYS,
) -> set:
    """Return tickers whose nearest earnings date is within `window` days of check_date."""
    # Normalise check_date to tz-naive so it's comparable with both tz-aware
    # and tz-naive timestamps returned by yfinance.
    check_naive = check_date.tz_localize(None) if check_date.tzinfo else check_date
    risky: set = set()
    for t in tickers:
        for earn_d in calendar.get(t, []):
            earn_naive = earn_d.tz_localize(None) if earn_d.tzinfo else earn_d
            if abs((earn_naive - check_naive).days) <= window:
                risky.add(t)
                break
    return risky


def _sector_relative(series: pd.Series, avail: List[str],
                     sector_map: Dict[str, str]) -> pd.Series:
    """
    Z-score each signal within its GICS sector group.
    Sectors with < 3 stocks keep their raw value.
    This removes sector-rotation bias and isolates stock selection alpha.
    """
    result = series.copy().astype(float)
    sectors = {sector_map.get(t, "Unknown") for t in avail}
    for sec in sectors:
        peers = [t for t in avail if sector_map.get(t, "Unknown") == sec
                 and t in series.index and pd.notna(series.get(t))]
        if len(peers) < 3:
            continue
        vals  = series[peers]
        mu, sigma = vals.mean(), vals.std()
        if sigma > 1e-6:
            result[peers] = (vals - mu) / sigma
    return result


def compute_factors(
    prices_to_date: pd.DataFrame,
    fundamentals: pd.DataFrame,
    tickers: List[str],
    sector_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Sector-relative price-based composite — zero look-ahead bias.

    Each JT signal is z-scored within its GICS sector before the global rank,
    removing sector-timing noise and keeping pure stock-selection alpha.

    Weights: JT 12-1m 45% | JT 6-1m 20% | JT 3-1m 10% | 52wk-high 25%
    """
    sector_map = sector_map or {}
    avail = [t for t in tickers if t in prices_to_date.columns]
    if not avail:
        return pd.DataFrame()

    # ── Liquidity filter: require MIN_VALID_PX non-NaN rows ──────────────
    px_counts = prices_to_date[avail].count()
    avail = [t for t in avail if px_counts.get(t, 0) >= MIN_VALID_PX]
    if not avail:
        return pd.DataFrame()

    prices = prices_to_date[avail]

    # ── Realised volatility (63-day annualised) ───────────────────────────
    rvol = pd.Series(0.20, index=avail)
    if len(prices) >= 63:
        ret63 = prices.pct_change().iloc[-63:]
        rvol  = ret63.std() * np.sqrt(252)
        rvol  = rvol.clip(lower=0.05).fillna(0.20)

    # ── Raw momentum signals ──────────────────────────────────────────────
    mom_12_1 = pd.Series(np.nan, index=avail)
    mom_6_1  = pd.Series(np.nan, index=avail)
    mom_3_1  = pd.Series(np.nan, index=avail)
    near52   = pd.Series(np.nan, index=avail)

    if len(prices) >= 252:
        p_last      = prices.iloc[-1]
        p_minus_21  = prices.iloc[-21]
        p_minus_252 = prices.iloc[-252]
        mom_12_1    = p_minus_21 / p_minus_252.replace(0, np.nan) - 1
        high_252    = prices.iloc[-252:].max()
        near52      = p_last / high_252.replace(0, np.nan)

    if len(prices) >= 126:
        p_minus_126 = prices.iloc[-126]
        p_minus_21  = prices.iloc[-21] if len(prices) >= 21 else prices.iloc[-1]
        mom_6_1     = p_minus_21 / p_minus_126.replace(0, np.nan) - 1

    if len(prices) >= 63:
        p_minus_63  = prices.iloc[-63]
        p_minus_21  = prices.iloc[-21] if len(prices) >= 21 else prices.iloc[-1]
        mom_3_1     = p_minus_21 / p_minus_63.replace(0, np.nan) - 1

    # ── Sector-relative normalisation ────────────────────────────────────
    sr_12_1 = _sector_relative(mom_12_1, avail, sector_map)
    sr_6_1  = _sector_relative(mom_6_1,  avail, sector_map)
    sr_3_1  = _sector_relative(mom_3_1,  avail, sector_map)
    sr_n52  = _sector_relative(near52,   avail, sector_map)

    # ── Residual momentum (improvement #2 — market-adjusted) ─────────────
    # Removes the market beta component from the 12-1m momentum signal.
    # Weight kept modest (0.10) so core signal dominates.
    resid_mom = compute_residual_momentum(prices, avail, BENCHMARK)
    sr_resid  = _sector_relative(resid_mom, avail, sector_map)

    # ── Idiosyncratic volatility (improvement #3 — IVOL short signal) ────
    # High IVOL stocks underperform (Ang et al. 2006).
    # Small penalty weight (0.05) to nudge rather than override.
    ivol_raw = compute_ivol(prices, avail, BENCHMARK)

    # ── Short-term reversal (Phase 3 #2) ─────────────────────────────────
    # The skipped last-month return, sign-flipped: stocks that fell over the
    # past month tend to bounce, recent winners tend to give back.
    # Orthogonal to 12-1m momentum (which excludes this window by design).
    reversal = pd.Series(0.0, index=avail)
    if USE_REVERSAL and len(prices) >= REVERSAL_LOOKBACK + 1:
        last_ret = prices.iloc[-1] / prices.iloc[-REVERSAL_LOOKBACK].replace(0, np.nan) - 1
        reversal = -last_ret.reindex(avail)            # flip sign → losers score high
    sr_rev = _sector_relative(reversal, avail, sector_map)

    # ── Momentum composite (long-driving signal) ─────────────────────────
    momentum = (
        _rank(sr_12_1)   * 0.40   # core 12-1m momentum (dominant signal)
        + _rank(sr_resid)* 0.10   # market-adjusted residual (modest boost)
        + _rank(sr_6_1)  * 0.18   # 6-1m
        + _rank(sr_3_1)  * 0.10   # 3-1m
        + _rank(sr_n52)  * 0.22   # nearness to 52-wk high
        - _rank(ivol_raw)* 0.05   # small IVOL penalty
    )

    # ── Blend reversal sleeve into the long signal ───────────────────────
    if USE_REVERSAL:
        momentum = (1 - REVERSAL_WEIGHT) * momentum + REVERSAL_WEIGHT * _rank(sr_rev)

    # ── Dedicated SHORT-alpha score (Phase 3 #1) ─────────────────────────
    # Shorts ranked by LOW momentum + HIGH idiosyncratic vol.
    # Lower score = better short candidate.  This is independent of the long
    # composite so the short book targets genuine underperformers, not just
    # "whatever ranked last" among momentum names.
    short_score = (
        _rank(momentum)   * W_SHORT_MOM        # low momentum → low score
        - _rank(ivol_raw) * W_SHORT_IVOL       # high IVOL → lowers score
    )

    # ── PIT fundamental blend (EDGAR, US stocks only) ────────────────────
    # pit_fund is injected via the sector_map dict under key "__pit_fund__"
    # to avoid changing the function signature across the codebase.
    # International ADRs have no EDGAR data → stay at 100% momentum.
    pit_fund: Optional[pd.DataFrame] = (
        sector_map.get("__pit_fund__") if sector_map else None
    )

    quality = pd.Series(0.0, index=avail)
    value   = pd.Series(0.0, index=avail)

    if pit_fund is not None and not pit_fund.empty:
        pf = pit_fund.reindex(avail)

        # ── Earnings growth (PEAD-aligned signal) ─────────────────────────
        # Only earnings-based signals: these are positively correlated with
        # momentum (both capture the same outperforming companies).
        # Quality metrics (Piotroski, ROE, accruals) are anti-momentum — excluded.
        eg_col = pf["earningsGrowth"] if "earningsGrowth" in pf.columns else pd.Series(np.nan, index=avail)
        rg_col = pf["revenueGrowth"]  if "revenueGrowth"  in pf.columns else pd.Series(np.nan, index=avail)

        # Clip extreme outliers (turnaround stocks with 10× earnings growth distort ranking)
        eg_clipped = eg_col.astype(float).clip(-3.0, 3.0)
        rg_clipped = rg_col.astype(float).clip(-1.0, 2.0)

        growth_s = (
            _rank(eg_clipped) * 0.70
            + _rank(rg_clipped) * 0.30
        )

        # Only blend for tickers with EDGAR earnings data
        has_edgar = eg_col.notna() | rg_col.notna()

        composite = momentum.copy()
        composite[has_edgar] = (
            momentum[has_edgar] * W_MOM_EDGAR
            + growth_s[has_edgar].fillna(0.0) * (1.0 - W_MOM_EDGAR)
        )

        quality = growth_s.fillna(0.0)   # displayed as "quality" in dashboard
        value   = pd.Series(0.0, index=avail)
    else:
        composite = momentum

    return pd.DataFrame({
        "composite":   composite,
        "short_score": short_score.reindex(avail).fillna(0.0),
        "quality":     quality,
        "value":       value,
        "momentum":    momentum,
        "reversal":    sr_rev.reindex(avail).fillna(0.0),
        "rvol":        rvol,
        "ivol":        ivol_raw.reindex(avail).fillna(0.20),
        "resid_mom":   resid_mom.reindex(avail).fillna(0.0),
        "mom_12_1":    mom_12_1,
        "near52":      near52.reindex(avail).fillna(0.5),
    }).reindex(avail)


# ---------------------------------------------------------------------------
# Beta computation
# ---------------------------------------------------------------------------

def compute_betas(
    prices_to_date: pd.DataFrame,
    tickers: List[str],
    window: int = 126,
) -> pd.Series:
    if BENCHMARK not in prices_to_date.columns or len(prices_to_date) < window:
        return pd.Series(1.0, index=tickers)
    rets = np.log(prices_to_date / prices_to_date.shift(1)).dropna()
    spy  = rets[BENCHMARK].iloc[-window:]
    var_spy = float(spy.var())
    if var_spy == 0:
        return pd.Series(1.0, index=tickers)
    betas = {}
    for t in tickers:
        if t in rets.columns:
            s = rets[t].iloc[-window:]
            betas[t] = float(np.cov(s.values, spy.values)[0, 1] / var_spy)
        else:
            betas[t] = 1.0
    return pd.Series(betas).clip(-3, 3)


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def _sector_diversify(
    ranked: pd.Series,
    n: int,
    sector_map: Dict[str, str],
    max_per_sector: int = 8,
) -> List[str]:
    """Pick top-n from ranked series with sector cap."""
    selected: List[str] = []
    counts: Dict[str, int] = {}
    for t in ranked.index:
        sec = sector_map.get(t, "Unknown")
        if counts.get(sec, 0) < max_per_sector:
            selected.append(t)
            counts[sec] = counts.get(sec, 0) + 1
        if len(selected) >= n:
            break
    return selected


def _vol_weight(tickers: List[str], rvol: pd.Series, gross: float) -> pd.Series:
    """Inverse-vol weights, scaled to target gross."""
    if not tickers:
        return pd.Series(dtype=float)
    rv = rvol.reindex(tickers).fillna(0.20).clip(lower=0.05)
    inv = 1.0 / rv
    return (inv / inv.sum()) * gross


def _conviction_vol_weight(
    tickers: List[str],
    composite: pd.Series,
    rvol: pd.Series,
    gross: float,
    conviction_power: float = 0.5,
) -> pd.Series:
    """
    Improvement #5 — signal × (1/vol) weighting.
    Weight = |composite|^conviction_power / rvol, normalised.
    conviction_power=0.5 gives a smooth tilt toward higher-signal names
    without concentrating too much in a handful of extreme scores.
    """
    if not tickers:
        return pd.Series(dtype=float)
    rv   = rvol.reindex(tickers).fillna(0.20).clip(lower=0.05)
    sig  = composite.reindex(tickers).abs().clip(lower=0.05).fillna(0.05)
    w    = (sig ** conviction_power) / rv
    return (w / w.sum()) * gross


def _sector_relative_composite(
    composite: pd.Series,
    sector_map: Dict[str, str],
    blend_global: float = 0.30,
) -> pd.Series:
    """
    Blend sector-relative rank (70%) with global rank (30%).
    Removes sector rotation bias: an average-momentum energy stock won't be
    shorted just because the energy sector had bad global momentum.
    """
    global_rank = composite.rank(pct=True) - 0.5          # centred [−0.5, +0.5]

    tickers  = composite.index.tolist()
    sectors  = pd.Series({t: sector_map.get(t, "Unknown") for t in tickers})
    sec_rank = pd.Series(index=tickers, dtype=float)

    for sec, grp in sectors.groupby(sectors):
        idx = grp.index
        if len(idx) < 2:
            sec_rank[idx] = 0.0
        else:
            sec_rank[idx] = (composite[idx].rank(pct=True) - 0.5).values

    blended = (1 - blend_global) * sec_rank + blend_global * global_rank
    return blended


def build_portfolio(
    factors: pd.DataFrame,
    sector_map: Dict[str, str],
    betas: pd.Series,
    n_long: int = N_SUB_LONG,
    n_short: int = N_SUB_SHORT,
    gross_per_leg: float = GROSS / 2.0 / HOLD_PERIODS,
    exclude_tickers: Optional[set] = None,
) -> pd.DataFrame:
    """
    Return DataFrame(ticker, direction, weight, composite, rvol, beta).
    Long  = top-N by sector-blended composite.
    Short = bottom-N filtered: exclude deep-drawdown stocks (>35% from 52wk high)
            that are prone to mean-reversion / value bounce.
    Weights = conviction × (1/vol), scaled to gross_per_leg.
    exclude_tickers: already-selected tickers (for sub-portfolio diversification).
    """
    exclude_tickers = exclude_tickers or set()

    raw_comp = factors["composite"].dropna()
    raw_comp = raw_comp[raw_comp.index != BENCHMARK]

    # ── Blend sector-relative + global momentum composite ─────────────────
    comp = _sector_relative_composite(raw_comp, sector_map, blend_global=0.30)
    comp = comp.sort_values(ascending=False)
    rvol = factors["rvol"]

    # Exclude already-selected tickers (from other active sub-portfolios)
    if exclude_tickers:
        comp = comp[~comp.index.isin(exclude_tickers)]

    # ── Longs: top composite (sector-diversified) ─────────────────────────
    long_pool  = comp.head(n_long * 3)
    long_picks = _sector_diversify(long_pool, n_long, sector_map)

    # ── Shorts: dedicated short-alpha score (Phase 3 #1) ──────────────────
    # Rank by short_score (low momentum + high IVOL), NOT just bottom composite.
    # Then apply the bounce/squeeze filter: exclude deep-drawdown crashes and
    # stocks already strongly rising.
    near52   = factors["near52"]   if "near52"   in factors.columns else pd.Series(0.5, index=comp.index)
    mom_12_1 = factors["mom_12_1"] if "mom_12_1" in factors.columns else pd.Series(0.0, index=comp.index)

    if USE_SHORT_SCORE and "short_score" in factors.columns:
        ss = factors["short_score"].dropna()
        ss = ss[ss.index != BENCHMARK]
        if exclude_tickers:
            ss = ss[~ss.index.isin(exclude_tickers)]
        short_pool_raw = ss.sort_values(ascending=True).head(n_short * 3)  # lowest = best short
    else:
        # Phase-2 behaviour: shorts = bottom of the long composite
        short_pool_raw = comp.tail(n_short * 3).iloc[::-1]   # worst composite first
    # Filter: not too deep in drawdown (bounce risk), and not strongly rising
    not_crashed = near52.reindex(short_pool_raw.index).fillna(0.5) >= 0.62   # at most 38% below 52-wk high
    declining   = mom_12_1.reindex(short_pool_raw.index).fillna(0.0) <= 0.10  # not strongly rising

    short_pool  = short_pool_raw[not_crashed & declining]

    # Fallback: if filter removes too many, relax the drawdown threshold
    if len(short_pool) < n_short // 2:
        relaxed   = near52.reindex(short_pool_raw.index).fillna(0.5) >= 0.50
        short_pool = short_pool_raw[relaxed & declining]
    if short_pool.empty:
        short_pool = short_pool_raw   # no filter as last resort

    short_picks = _sector_diversify(short_pool, n_short, sector_map)
    short_picks = [t for t in short_picks if t not in long_picks][:n_short]

    if not long_picks:
        return pd.DataFrame()

    # ── Conviction-scaled weights (improvement #5) ────────────────────────
    composite_col = factors["composite"]
    long_w  = _conviction_vol_weight(long_picks,  composite_col, rvol, gross_per_leg)
    short_w = _conviction_vol_weight(short_picks, composite_col, rvol, gross_per_leg)

    # Equal-gross legs: no beta rescaling.
    # Beta neutralisation was counterproductive: momentum longs (tech/growth β≈1.3)
    # vs momentum shorts (beaten-down β≈0.8) → scale > 1 → 80 % short exposure
    # → net 30 % net-short in a bull market.  Equal legs give slight positive beta
    # (+0.2) which is acceptable; the momentum spread provides the alpha.

    records = []
    for t in long_picks:
        records.append({
            "ticker":    t,
            "direction": "LONG",
            "weight":    float(long_w.get(t, gross_per_leg / N_LONG)),
            "composite": float(factors.loc[t, "composite"]) if t in factors.index else 0.0,
            "rvol":      float(rvol.get(t, 0.20)),
            "beta":      float(betas.get(t, 1.0)),
        })
    for t in short_picks:
        records.append({
            "ticker":    t,
            "direction": "SHORT",
            "weight":    float(short_w.get(t, gross_per_leg / N_SHORT)),
            "composite": float(factors.loc[t, "composite"]) if t in factors.index else 0.0,
            "rvol":      float(rvol.get(t, 0.20)),
            "beta":      float(betas.get(t, 1.0)),
        })

    df = pd.DataFrame(records).set_index("ticker")
    return df


# ---------------------------------------------------------------------------
# Period P&L  (vectorised — no stop-losses)
# ---------------------------------------------------------------------------

def compute_period_pnl(
    portfolio: pd.DataFrame,
    prices: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    sector_map: Dict[str, str],
    apply_slippage: bool = True,
) -> Tuple[pd.Series, List[Dict]]:
    """
    Hold portfolio weights constant from start to end.
    apply_slippage: False for continuation months of an overlapping sub-portfolio
                    (slippage charged only once at entry, not every month).
    Returns (daily_returns Series, trades list for analytics).
    """
    period = prices.loc[start:end]
    if period.empty or portfolio.empty:
        return pd.Series(dtype=float), []

    tickers = portfolio.index.tolist()
    avail   = [t for t in tickers if t in period.columns]
    if not avail:
        return pd.Series(dtype=float), []

    px      = period[avail]
    daily   = px.pct_change().fillna(0.0)   # first row = 0 (entry day, slippage only)

    # Signs: LONG = +1, SHORT = −1
    signs   = portfolio.loc[avail, "direction"].map({"LONG": 1.0, "SHORT": -1.0}).fillna(1.0)
    weights = portfolio.loc[avail, "weight"].abs()

    # Vectorised daily portfolio return
    signed_w = (signs * weights).reindex(avail)
    port_ret  = daily.multiply(signed_w, axis=1).sum(axis=1)

    # ── Differentiated one-way slippage on entry (only at portfolio inception) ─
    if apply_slippage:
        slip_cost = sum(
            w * (SLIP_INTL_BPS if t in INTL_TICKERS else SLIP_US_BPS) / 10_000.0
            for t, w in weights.items()
        )
        port_ret.iloc[0] -= slip_cost

    # ── Daily borrow cost on short positions ─────────────────────────────
    # Applied each day as a drag: US 0.5%/yr, ADRs 1.5%/yr
    short_tickers = [t for t in avail if portfolio.loc[t, "direction"] == "SHORT"]
    if short_tickers:
        daily_borrow = sum(
            weights.get(t, 0.0)
            * (BORROW_INTL_PA if t in INTL_TICKERS else BORROW_US_PA)
            / 252.0
            for t in short_tickers
        )
        port_ret -= daily_borrow   # constant drag every day of the period

    # Build trade records (entry-to-exit cumulative returns)
    trades = []
    entry_px = px.iloc[0]
    exit_px  = px.iloc[-1]
    for t in avail:
        ep = float(entry_px[t])
        xp = float(exit_px[t])
        if ep == 0 or np.isnan(ep) or np.isnan(xp):
            continue
        direction = portfolio.loc[t, "direction"]
        pct = (xp - ep) / ep
        trade_ret = pct if direction == "LONG" else -pct
        trades.append({
            "ticker":      t,
            "direction":   direction,
            "entry_date":  str(start.date()),
            "exit_date":   str(end.date()),
            "hold_days":   len(period),
            "entry_price": ep,
            "exit_price":  xp,
            "return_pct":  trade_ret * 100,
            "pnl_bps":     trade_ret * float(weights.get(t, 0)) * 10_000,
            "weight":      float(weights.get(t, 0)),
            "composite":   float(portfolio.loc[t, "composite"]) if "composite" in portfolio.columns else 0.0,
            "rvol":        float(portfolio.loc[t, "rvol"])       if "rvol"      in portfolio.columns else 0.2,
            "sector":      sector_map.get(t, "Unknown"),
            "win":         trade_ret > 0,
            "status":      "normal",
        })

    return port_ret, trades


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    returns: pd.Series, benchmark_returns: pd.Series
) -> Dict:
    if returns.empty:
        return {}
    ann = 252
    cum = (1 + returns).cumprod()
    n_years = len(returns) / ann
    cagr    = float(cum.iloc[-1] ** (1 / max(n_years, 0.1)) - 1)
    vol     = float(returns.std() * np.sqrt(ann))
    sharpe  = float(cagr / vol) if vol > 0 else 0.0

    roll_max = cum.cummax()
    dd       = (cum / roll_max - 1)
    max_dd   = float(dd.min())

    # Alpha / Beta vs benchmark
    bench = benchmark_returns.reindex(returns.index).fillna(0.0)
    if bench.std() > 0:
        cov = np.cov(returns.values, bench.values)
        beta  = float(cov[0, 1] / bench.var())
        alpha = float(cagr - beta * float(bench.mean() * ann))
    else:
        beta, alpha = 0.0, cagr

    trades_tot = len(returns[returns != 0])
    win_rate   = float((returns > 0).sum() / max(trades_tot, 1))

    return {
        "CAGR":    round(cagr,   4),
        "Sharpe":  round(sharpe, 4),
        "MaxDD":   round(max_dd, 4),
        "Alpha":   round(alpha,  4),
        "Beta":    round(beta,   4),
        "WinRate": round(win_rate, 4),
        "Vol":     round(vol,    4),
        "Universe": 0,
        "NTrades": trades_tot,
    }


# ---------------------------------------------------------------------------
# Regime detection  (simple SMA-based, robust)
# ---------------------------------------------------------------------------

def detect_regime(spy_prices: pd.Series) -> Dict:
    """
    Improvement #6 — multi-signal regime filter.
    Combines three orthogonal signals:
      1. Trend    : SMA-200 relative position
      2. Momentum : 1-month market return (momentum crash early warning)
      3. Vol      : 21-day realised vol vs 63-day baseline (stress indicator)
    Returns a gross multiplier in [0.50, 1.0].
    """
    if len(spy_prices) < 200:
        return {"label": "Unknown", "gross_mult": 0.80}

    last    = float(spy_prices.iloc[-1])
    sma200  = float(spy_prices.rolling(200).mean().iloc[-1])
    sma50   = float(spy_prices.rolling(50).mean().iloc[-1])

    # 1. Drawdown from 3-month peak (momentum crash signal)
    peak_3m = float(spy_prices.iloc[-63:].max())
    dd_3m   = (last / peak_3m) - 1   # negative number

    # 2. 1-month return
    ret_1m  = (last / float(spy_prices.iloc[-21])) - 1

    # 3. Volatility ratio (current 21d vol / trailing 63d vol)
    rets = spy_prices.pct_change().dropna()
    vol_21 = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) >= 21 else 0.15
    vol_63 = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) >= 63 else 0.15
    vol_ratio = vol_21 / max(vol_63, 0.01)

    # --- Score each dimension ---
    trend_bull  = last > sma200 and sma50 > sma200
    trend_bear  = last < sma200 * 0.97

    crash_risk  = dd_3m < -0.10 or ret_1m < -0.06   # market falling fast
    vol_stress  = vol_ratio > 1.5                     # vol spiking

    # --- Determine regime and multiplier ---
    if trend_bear or (crash_risk and vol_stress):
        label = "Bear"
        mult  = 0.50   # significant de-gross; momentum crashes hardest in reversals
    elif crash_risk or vol_stress:
        label = "Caution"
        mult  = 0.70
    elif trend_bull and not crash_risk and not vol_stress:
        label = "Bull"
        mult  = 1.00
    else:
        label = "Choppy"
        mult  = 0.85

    return {
        "label":      label,
        "gross_mult": mult,
        "dd_3m":      round(dd_3m,   3),
        "ret_1m":     round(ret_1m,  3),
        "vol_ratio":  round(vol_ratio, 2),
    }


# ---------------------------------------------------------------------------
# Main walk-forward loop
# ---------------------------------------------------------------------------

def run_walkforward(start_year: int = 2019, force_refresh: bool = False) -> Dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Universe ──────────────────────────────────────────────────────────
    universe, sector_map = build_global_universe()
    if BENCHMARK not in universe:
        universe.append(BENCHMARK)
        sector_map[BENCHMARK] = "ETF"
    logger.info("Universe: %d tickers", len(universe))

    # ── Prices ────────────────────────────────────────────────────────────
    price_start = f"{start_year - 5}-01-01"
    prices = load_prices(universe, start=price_start, force_refresh=force_refresh)
    universe = [t for t in universe if t in prices.columns]
    logger.info("Tickers with price data: %d", len(universe))

    # ── yfinance fundamentals (snapshot, kept for backward compat) ───────
    fundamentals = load_fundamentals(universe, force_refresh=force_refresh)
    logger.info("Fundamentals loaded: %d tickers", len(fundamentals))

    # ── Rebalance dates ───────────────────────────────────────────────────
    rebal_start  = pd.Timestamp(f"{start_year}-03-01")
    rebal_end    = pd.Timestamp.today()
    rebal_dates  = pd.date_range(rebal_start, rebal_end, freq="MS")
    # snap to actual trading days
    rebal_dates  = pd.DatetimeIndex([
        prices.loc[d:].index[0] if not prices.loc[d:].empty else d
        for d in rebal_dates
    ])
    logger.info("Rebalance dates: %d", len(rebal_dates))

    # ── EDGAR PIT fundamentals ────────────────────────────────────────────
    # Pre-compute a PIT panel {rebal_date → DataFrame(ticker × signals)}.
    # Strict PIT: only filings with filed_date ≤ rebal_date are used.
    # International ADRs (not in SEC EDGAR) will have NaN rows → momentum-only fallback.
    pit_panel: Dict[pd.Timestamp, pd.DataFrame] = {}
    if USE_EDGAR and EDGAR_AVAILABLE:
        try:
            logger.info("Loading EDGAR PIT data (first run ~2 min; cached after that) …")
            edgar_raw = load_edgar_data(universe, force_refresh=force_refresh)
            pit_panel = precompute_pit_panel(
                edgar_raw, universe, prices, rebal_dates, fallback_fund=fundamentals
            )
            n_covered = sum(
                1 for t in universe
                if t in edgar_raw and edgar_raw[t]
            )
            logger.info("EDGAR PIT panel ready: %d tickers covered, %d dates",
                        n_covered, len(pit_panel))
        except Exception as exc:
            logger.warning("EDGAR load failed (%s) — price-only mode", exc)
            pit_panel = {}
    else:
        logger.info("EDGAR disabled — using sector-relative momentum only")

    # ── Earnings calendar (improvement #4) ───────────────────────────────
    earnings_cal: Dict[str, List[pd.Timestamp]] = {}
    if USE_EARNINGS_FILTER:
        try:
            earnings_cal = preload_earnings_calendar(universe, force_refresh=force_refresh)
        except Exception as exc:
            logger.warning("Earnings calendar load failed (%s) — skipping filter", exc)

    # ── Walk-forward state ────────────────────────────────────────────────
    all_daily:     pd.Series  = pd.Series(dtype=float)
    bench_daily:   pd.Series  = pd.Series(dtype=float)
    all_trades:    List[Dict] = []
    rebal_history: List[Dict] = []
    long_daily:    pd.Series  = pd.Series(dtype=float)
    short_daily:   pd.Series  = pd.Series(dtype=float)

    # ── Overlapping sub-portfolio bag (improvement #1) ───────────────────
    # Each entry: (portfolio_df, start_date, periods_remaining)
    active_subs: List[Tuple[pd.DataFrame, pd.Timestamp, int]] = []

    for i, rdate in enumerate(rebal_dates):
        prices_so_far = prices.loc[:rdate]
        if len(prices_so_far) < MIN_HISTORY:
            continue

        next_rdate = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else rdate + pd.offsets.MonthEnd(1)

        # ── Regime ──────────────────────────────────────────────────────
        spy_prices = prices_so_far[BENCHMARK].dropna() if BENCHMARK in prices_so_far else pd.Series()
        regime     = detect_regime(spy_prices)
        gross_mult = regime["gross_mult"]

        # ── Volatility targeting (Phase 3 #3) ────────────────────────────
        # Scale the book to a constant ex-ante vol using the strategy's own
        # trailing realised vol.  Purely backward-looking → no lookahead.
        vol_scalar = 1.0
        if USE_VOL_TARGET and len(all_daily) >= VOL_LOOKBACK:
            recent_vol = float(all_daily.iloc[-VOL_LOOKBACK:].std() * np.sqrt(252))
            if recent_vol > 1e-6:
                vol_scalar = float(np.clip(VOL_TARGET / recent_vol,
                                           VOL_SCALE_MIN, VOL_SCALE_MAX))
        gross_mult = gross_mult * vol_scalar

        # ── Signals ──────────────────────────────────────────────────────
        # Inject PIT fundamentals via sector_map dict (avoids signature change)
        sm_with_pit = dict(sector_map)
        pit_fund = pit_panel.get(rdate)
        if pit_fund is not None:
            sm_with_pit["__pit_fund__"] = pit_fund

        try:
            factors = compute_factors(prices_so_far, fundamentals, universe,
                                      sector_map=sm_with_pit)
            if factors.empty:
                continue
        except Exception as e:
            logger.error("Factor computation failed at %s: %s", rdate.date(), e)
            continue

        # ── Betas ────────────────────────────────────────────────────────
        betas = compute_betas(prices_so_far, universe)

        # ── Overlapping portfolios: build new sub-portfolio ──────────────
        # Collect tickers already in active sub-portfolios to diversify picks
        already_selected = set(
            t for sub_p, _, _ in active_subs
            for t in sub_p.index.tolist()
        )

        try:
            new_sub = build_portfolio(
                factors, sector_map, betas,
                n_long=N_SUB_LONG, n_short=N_SUB_SHORT,
                gross_per_leg=GROSS / 2.0 / HOLD_PERIODS,
                # No exclude_tickers: each sub picks independently from full universe.
                # Overlapping picks are fine — persistent winners naturally get overweight.
            )
        except Exception as e:
            logger.error("Portfolio build failed at %s: %s", rdate.date(), e)
            new_sub = pd.DataFrame()

        if not new_sub.empty:
            # ── Earnings blackout (improvement #4) ───────────────────────
            if USE_EARNINGS_FILTER and earnings_cal:
                risky = get_earnings_risk_tickers(
                    new_sub.index.tolist(), earnings_cal, rdate
                )
                if risky:
                    new_sub.loc[new_sub.index.isin(risky), "weight"] *= EARNINGS_WEIGHT_MULT
                    logger.debug("Earnings blackout: %d tickers halved at %s", len(risky), rdate.date())

            active_subs.append((new_sub, rdate, HOLD_PERIODS))

        # Age all subs, remove expired
        active_subs = [(p, s, pr - 1) for p, s, pr in active_subs if pr > 0]

        if not active_subs:
            continue

        # ── P&L for this month = SUM over all active sub-portfolios ──────
        # Slippage only charged in the sub's entry month (apply_slippage = start==rdate)
        period_ret   = pd.Series(dtype=float)
        period_trades: List[Dict] = []
        lp_ret_parts: List[pd.Series] = []
        sp_ret_parts: List[pd.Series] = []

        for sub_p, sub_start, _ in active_subs:
            is_entry = (sub_start == rdate)
            # Apply current regime multiplier to ALL active subs each month.
            # This ensures the full portfolio de-grosses immediately in Bear/Caution,
            # not just the 1/3 that happens to be newly built this month.
            sub_p_regime = sub_p.copy()
            sub_p_regime["weight"] = sub_p_regime["weight"] * gross_mult
            try:
                sr, st = compute_period_pnl(
                    sub_p_regime, prices, rdate, next_rdate, sector_map,
                    apply_slippage=is_entry,
                )
            except Exception as e:
                logger.error("PnL failed at %s: %s", rdate.date(), e)
                continue
            if sr.empty:
                continue
            period_ret = sr if period_ret.empty else period_ret.add(sr, fill_value=0.0)
            period_trades.extend(st)

            # Long/short attribution per sub
            try:
                lp, _ = compute_period_pnl(
                    sub_p_regime[sub_p_regime["direction"] == "LONG"],
                    prices, rdate, next_rdate, sector_map, apply_slippage=False,
                )
                sp, _ = compute_period_pnl(
                    sub_p_regime[sub_p_regime["direction"] == "SHORT"],
                    prices, rdate, next_rdate, sector_map, apply_slippage=False,
                )
                lp_ret_parts.append(lp)
                sp_ret_parts.append(sp)
            except Exception:
                pass

        if period_ret.empty:
            continue

        # Aggregate long/short attribution
        lp_ret = pd.Series(0.0, index=period_ret.index)
        sp_ret = pd.Series(0.0, index=period_ret.index)
        for s in lp_ret_parts:
            lp_ret = lp_ret.add(s.reindex(lp_ret.index).fillna(0), fill_value=0)
        for s in sp_ret_parts:
            sp_ret = sp_ret.add(s.reindex(sp_ret.index).fillna(0), fill_value=0)

        all_daily   = pd.concat([all_daily,   period_ret])
        long_daily  = pd.concat([long_daily,  lp_ret])
        short_daily = pd.concat([short_daily, sp_ret])
        all_trades.extend(period_trades)

        # Benchmark returns for same period
        if BENCHMARK in prices.columns:
            bm = prices[BENCHMARK].pct_change().loc[rdate:next_rdate]
            bench_daily = pd.concat([bench_daily, bm])

        # Combined portfolio stats for logging
        combined_portfolio = pd.concat([p for p, _, _ in active_subs])
        n_long_tot  = (combined_portfolio["direction"] == "LONG").sum()
        n_short_tot = (combined_portfolio["direction"] == "SHORT").sum()
        gross_tot   = float(combined_portfolio["weight"].abs().sum())
        cum_val     = float((1 + all_daily).prod()) if not all_daily.empty else 1.0

        logger.info(
            "[%s] regime=%-7s nL=%d nS=%d gross=%.2f cum=%.4f  dd3m=%.1f%% vol_ratio=%.1f",
            rdate.date(), regime["label"], n_long_tot, n_short_tot, gross_tot, cum_val,
            regime.get("dd_3m", 0) * 100, regime.get("vol_ratio", 1.0),
        )

        # Newest sub's top picks for history
        if not new_sub.empty:
            top_longs  = new_sub[new_sub["direction"] == "LONG"]["composite"].nlargest(5).index.tolist()
            top_shorts = new_sub[new_sub["direction"] == "SHORT"]["composite"].nsmallest(5).index.tolist()
        else:
            top_longs, top_shorts = [], []

        rebal_history.append({
            "date":        str(rdate.date()),
            "regime":      regime["label"],
            "n_long":      int(n_long_tot),
            "n_short":     int(n_short_tot),
            "gross":       round(gross_tot, 3),
            "top_longs":   top_longs,
            "top_shorts":  top_shorts,
        })

    if all_daily.empty:
        logger.error("No returns generated — check universe/data.")
        return {}

    # De-duplicate index (overlapping rebalancing windows)
    all_daily   = all_daily[~all_daily.index.duplicated(keep="first")].sort_index()
    long_daily  = long_daily[~long_daily.index.duplicated(keep="first")].sort_index()
    short_daily = short_daily[~short_daily.index.duplicated(keep="first")].sort_index()
    bench_daily = bench_daily[~bench_daily.index.duplicated(keep="first")].sort_index()

    metrics = compute_metrics(all_daily, bench_daily)
    metrics["Universe"] = len(universe)
    metrics["NTrades"]  = len(all_trades)

    logger.info(
        "FINAL — CAGR=%.1f%% Sharpe=%.2f MaxDD=%.1f%% Alpha=%.1f%% Beta=%.2f WinRate=%.1f%%",
        metrics["CAGR"] * 100, metrics["Sharpe"], metrics["MaxDD"] * 100,
        metrics["Alpha"] * 100, metrics["Beta"], metrics["WinRate"] * 100,
    )

    # ── Build output ──────────────────────────────────────────────────────
    cum       = (1 + all_daily).cumprod()
    bench_cum = (1 + bench_daily.reindex(all_daily.index).fillna(0)).cumprod()
    roll_max  = cum.cummax()
    drawdown  = (cum / roll_max - 1)

    roll_sharpe = (
        all_daily.rolling(63).mean()
        / all_daily.rolling(63).std().replace(0, np.nan)
        * np.sqrt(252)
    ).fillna(0)

    long_cum  = (1 + long_daily.reindex(all_daily.index).fillna(0)).cumprod()
    short_cum = (1 + short_daily.reindex(all_daily.index).fillna(0)).cumprod()

    def to_list(s: pd.Series) -> list:
        return [round(float(v), 6) for v in s.values]

    result = {
        "daily_returns":      to_list(all_daily),
        "cumulative_returns": {str(d.date()): round(float(v), 6)
                               for d, v in cum.items()},
        "long_cumulative":    {str(d.date()): round(float(v), 6)
                               for d, v in long_cum.items()},
        "short_cumulative":   {str(d.date()): round(float(v), 6)
                               for d, v in short_cum.items()},
        "benchmark_returns":  {str(d.date()): round(float(v), 6)
                               for d, v in bench_cum.items()},
        "drawdown":           {str(d.date()): round(float(v), 6)
                               for d, v in drawdown.items()},
        "rolling_sharpe":     {str(d.date()): round(float(v), 6)
                               for d, v in roll_sharpe.items()},
        "trades":             all_trades,
        "metrics":            metrics,
        "rebal_history":      rebal_history,
    }

    out_path = OUTPUT_DIR / "walkforward_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, default=str)
    logger.info("Results saved to %s", out_path)
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",         type=int,  default=2019)
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()
    run_walkforward(start_year=args.start, force_refresh=args.force_refresh)
