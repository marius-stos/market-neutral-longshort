"""
Global Momentum Market-Neutral Backtest  — Phase 1 (realistic costs)
=====================================================================
Universe  : ~580 global stocks (US S&P 500 + EU/Asia ADRs)
Signal    : Sector-relative momentum (JT 12-1m + 6-1m + 3-1m + 52-wk-high)
Portfolio : Top-40 long / Bottom-40 short, inverse-vol weighted
Costs     : Differentiated slippage (8 bps US / 15 bps ADR) +
            Borrow costs (0.5 %/yr US / 1.5 %/yr ADR) on short leg
Rebalance : Monthly (1st trading day of each month)
PnL       : Vectorised — no stop-losses, holds constant within each period
Output    : output/walkforward_results.json  (Dash-compatible)
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

N_LONG       = 40      # long positions per rebal
N_SHORT      = 40      # short positions per rebal
GROSS        = 1.0     # total gross (0.5 each leg)

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

    # ── Composite: global rank of sector-relative signals ────────────────
    momentum = (
        _rank(sr_12_1) * 0.45
        + _rank(sr_6_1)  * 0.20
        + _rank(sr_3_1)  * 0.10
        + _rank(sr_n52)  * 0.25
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
        "composite": composite,
        "quality":   quality,
        "value":     value,
        "momentum":  momentum,
        "rvol":      rvol,
        "mom_12_1":  mom_12_1,
        "near52":    near52.reindex(avail).fillna(0.5),
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
) -> pd.DataFrame:
    """
    Return DataFrame(ticker, direction, weight, composite, rvol, beta).
    Long  = top-N by sector-blended composite.
    Short = bottom-N filtered: exclude deep-drawdown stocks (>35% from 52wk high)
            that are prone to mean-reversion / value bounce.
    Equal gross on both legs (no beta rescaling — see comment below).
    """
    gross_per_leg = GROSS / 2.0

    raw_comp = factors["composite"].dropna()
    raw_comp = raw_comp[raw_comp.index != BENCHMARK]

    # ── Blend sector-relative + global momentum composite ─────────────────
    comp = _sector_relative_composite(raw_comp, sector_map, blend_global=0.30)
    comp = comp.sort_values(ascending=False)
    rvol = factors["rvol"]

    # ── Longs: top composite (sector-diversified) ─────────────────────────
    long_pool  = comp.head(N_LONG * 3)
    long_picks = _sector_diversify(long_pool, N_LONG, sector_map)

    # ── Shorts: bottom composite, with quality filter ─────────────────────
    # Exclude deep-drawdown stocks (>35% below 52-wk high) — these are
    # bounce candidates (value rotation, short-squeeze risk, M&A).
    # Only short stocks in a GRADUAL downtrend, not ones that already crashed.
    near52   = factors["near52"]   if "near52"   in factors.columns else pd.Series(0.5, index=comp.index)
    mom_12_1 = factors["mom_12_1"] if "mom_12_1" in factors.columns else pd.Series(0.0, index=comp.index)

    short_pool_raw = comp.tail(N_SHORT * 3).iloc[::-1]   # worst composite first
    # Filter: not too deep in drawdown, and actually declining
    not_crashed = near52.reindex(short_pool_raw.index).fillna(0.5) >= 0.62   # at most 38% below 52-wk high
    declining   = mom_12_1.reindex(short_pool_raw.index).fillna(0.0) <= 0.10  # not strongly rising

    short_pool  = short_pool_raw[not_crashed & declining]

    # Fallback: if filter removes too many, relax the drawdown threshold
    if len(short_pool) < N_SHORT // 2:
        relaxed   = near52.reindex(short_pool_raw.index).fillna(0.5) >= 0.50
        short_pool = short_pool_raw[relaxed & declining]
    if short_pool.empty:
        short_pool = short_pool_raw   # no filter as last resort

    short_picks = _sector_diversify(short_pool, N_SHORT, sector_map)
    short_picks = [t for t in short_picks if t not in long_picks][:N_SHORT]

    if not long_picks:
        return pd.DataFrame()

    long_w  = _vol_weight(long_picks,  rvol, gross_per_leg)
    short_w = _vol_weight(short_picks, rvol, gross_per_leg)

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
) -> Tuple[pd.Series, List[Dict]]:
    """
    Hold portfolio weights constant from start to end.
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

    # ── Differentiated one-way slippage on entry ──────────────────────────
    # US large caps: 8 bps; international ADRs: 15 bps (wider spreads)
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
    """SMA-200 based regime. Returns gross multiplier."""
    if len(spy_prices) < 200:
        return {"label": "Choppy", "gross_mult": 0.8}
    sma200 = spy_prices.rolling(200).mean()
    last   = float(spy_prices.iloc[-1])
    sma    = float(sma200.iloc[-1])
    sma50  = float(spy_prices.rolling(50).mean().iloc[-1])

    if last > sma200.iloc[-1] and sma50 > sma:
        return {"label": "Bull",   "gross_mult": 1.0}
    elif last < sma200.iloc[-1] * 0.95:
        return {"label": "Bear",   "gross_mult": 0.7}
    else:
        return {"label": "Choppy", "gross_mult": 0.85}


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

    # ── Walk-forward state ────────────────────────────────────────────────
    all_daily:     pd.Series  = pd.Series(dtype=float)
    bench_daily:   pd.Series  = pd.Series(dtype=float)
    all_trades:    List[Dict] = []
    rebal_history: List[Dict] = []
    long_daily:    pd.Series  = pd.Series(dtype=float)
    short_daily:   pd.Series  = pd.Series(dtype=float)

    for i, rdate in enumerate(rebal_dates):
        prices_so_far = prices.loc[:rdate]
        if len(prices_so_far) < MIN_HISTORY:
            continue

        next_rdate = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else rdate + pd.offsets.MonthEnd(1)

        # ── Regime ──────────────────────────────────────────────────────
        spy_prices = prices_so_far[BENCHMARK].dropna() if BENCHMARK in prices_so_far else pd.Series()
        regime     = detect_regime(spy_prices)
        gross_mult = regime["gross_mult"]

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

        # ── Portfolio ────────────────────────────────────────────────────
        try:
            portfolio = build_portfolio(factors, sector_map, betas)
            if portfolio.empty:
                continue
        except Exception as e:
            logger.error("Portfolio build failed at %s: %s", rdate.date(), e)
            continue

        # Apply regime gross multiplier
        portfolio["weight"] = portfolio["weight"] * gross_mult

        # ── P&L ──────────────────────────────────────────────────────────
        try:
            period_ret, period_trades = compute_period_pnl(
                portfolio, prices, rdate, next_rdate, sector_map
            )
        except Exception as e:
            logger.error("PnL failed at %s: %s", rdate.date(), e)
            continue

        if period_ret.empty:
            continue

        # Long / Short attribution
        long_port  = portfolio[portfolio["direction"] == "LONG"]
        short_port = portfolio[portfolio["direction"] == "SHORT"]
        try:
            lp_ret, _ = compute_period_pnl(long_port,  prices, rdate, next_rdate, sector_map)
            sp_ret, _ = compute_period_pnl(short_port, prices, rdate, next_rdate, sector_map)
        except Exception:
            lp_ret = sp_ret = pd.Series(0.0, index=period_ret.index)

        all_daily   = pd.concat([all_daily,   period_ret])
        long_daily  = pd.concat([long_daily,  lp_ret])
        short_daily = pd.concat([short_daily, sp_ret])
        all_trades.extend(period_trades)

        # Benchmark returns for same period
        if BENCHMARK in prices.columns:
            bm = prices[BENCHMARK].pct_change().loc[rdate:next_rdate]
            bench_daily = pd.concat([bench_daily, bm])

        n_long  = (portfolio["direction"] == "LONG").sum()
        n_short = (portfolio["direction"] == "SHORT").sum()
        cum_val = float((1 + all_daily).prod())
        logger.info(
            "[%s] regime=%-6s nL=%d nS=%d gross=%.2f cum=%.4f",
            rdate.date(), regime["label"], n_long, n_short,
            portfolio["weight"].abs().sum(), cum_val
        )

        # Top longs / shorts for history
        top_longs  = portfolio[portfolio["direction"] == "LONG"]["composite"].nlargest(5).index.tolist()
        top_shorts = portfolio[portfolio["direction"] == "SHORT"]["composite"].nsmallest(5).index.tolist()
        rebal_history.append({
            "date":        str(rdate.date()),
            "regime":      regime["label"],
            "n_long":      int(n_long),
            "n_short":     int(n_short),
            "gross":       round(float(portfolio["weight"].abs().sum()), 3),
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
