"""
Live Signal Generator
=====================
Runs the same sector-relative momentum model used in the backtest,
but on TODAY's prices to produce actionable trade recommendations.

Usage:
    python3 live_signals.py [--capital 100000] [--output output/live_signals.json]

Output JSON keys:
    generated_at   : ISO timestamp
    regime         : Bull / Choppy / Bear
    gross_mult     : exposure multiplier from regime
    capital        : reference capital in USD
    longs          : list of {ticker, sector, weight, notional, score, mom_12_1, near52}
    shorts         : list of {ticker, sector, weight, notional, score, mom_12_1, near52}
    rebalance_needed : True if holdings differ significantly from last run
    summary        : dict of aggregate stats
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Re-use all the infrastructure from walkforward_backtest
sys.path.insert(0, str(Path(__file__).parent))
from walkforward_backtest import (
    BENCHMARK, N_LONG, N_SHORT, GROSS, MIN_HISTORY, MIN_VALID_PX,
    INTL_TICKERS, SLIP_US_BPS, SLIP_INTL_BPS, BORROW_US_PA, BORROW_INTL_PA,
    load_prices, compute_factors, compute_betas, build_portfolio, detect_regime,
)
from universe_global import build_global_universe

OUTPUT_DIR  = Path("output")
SIGNAL_FILE = OUTPUT_DIR / "live_signals.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core signal computation
# ---------------------------------------------------------------------------

def compute_live_signals(capital: float = 100_000.0) -> dict:
    """
    Compute today's long/short portfolio with position sizes in USD.
    Returns a dict ready to be serialised to JSON.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Universe ──────────────────────────────────────────────────────────
    universe, sector_map = build_global_universe()
    if BENCHMARK not in universe:
        universe.append(BENCHMARK)
        sector_map[BENCHMARK] = "ETF"

    # ── Prices (uses cache, refreshed daily) ─────────────────────────────
    prices = load_prices(universe, start="2015-01-01", force_refresh=False)
    universe = [t for t in universe if t in prices.columns]

    if len(prices) < MIN_HISTORY:
        raise RuntimeError(f"Not enough price history ({len(prices)} rows)")

    # ── Regime ────────────────────────────────────────────────────────────
    spy_px = prices[BENCHMARK].dropna()
    regime = detect_regime(spy_px)
    gross_mult = regime["gross_mult"]

    # ── Factors ───────────────────────────────────────────────────────────
    factors = compute_factors(prices, pd.DataFrame(), universe,
                              sector_map=sector_map)
    if factors.empty:
        raise RuntimeError("Factor computation returned empty DataFrame")

    # ── Betas ─────────────────────────────────────────────────────────────
    betas = compute_betas(prices, universe)

    # ── Portfolio construction ────────────────────────────────────────────
    portfolio = build_portfolio(factors, sector_map, betas)
    if portfolio.empty:
        raise RuntimeError("Portfolio construction returned empty DataFrame")

    portfolio["weight"] *= gross_mult   # regime adjustment

    # ── Enrich with metadata ──────────────────────────────────────────────
    today = prices.index[-1]

    def to_positions(df: pd.DataFrame, direction: str) -> list:
        sub = df[df["direction"] == direction].copy()
        records = []
        for ticker, row in sub.iterrows():
            w        = float(row["weight"])
            notional = round(w * capital, 0)
            f_row    = factors.loc[ticker] if ticker in factors.index else pd.Series()
            slip_bps = SLIP_INTL_BPS if ticker in INTL_TICKERS else SLIP_US_BPS
            borrow_pa = BORROW_INTL_PA if ticker in INTL_TICKERS else BORROW_US_PA
            records.append({
                "ticker":     ticker,
                "sector":     sector_map.get(ticker, "Unknown"),
                "direction":  direction,
                "weight_pct": round(w * 100, 2),
                "notional":   int(notional),
                "score":      round(float(f_row.get("composite", 0)), 4),
                "momentum":   round(float(f_row.get("momentum", 0)), 4),
                "mom_12_1":   round(float(f_row.get("mom_12_1", np.nan)), 4),
                "near52":     round(float(f_row.get("near52", np.nan)), 4),
                "rvol_ann":   round(float(f_row.get("rvol", 0.20)), 4),
                "beta":       round(float(betas.get(ticker, 1.0)), 3),
                "slip_bps":   slip_bps,
                "borrow_pa_pct": round(borrow_pa * 100, 2) if direction == "SHORT" else 0.0,
                "intl":       ticker in INTL_TICKERS,
            })
        return sorted(records, key=lambda x: -abs(x["notional"]))

    longs  = to_positions(portfolio, "LONG")
    shorts = to_positions(portfolio, "SHORT")

    # ── Aggregate stats ───────────────────────────────────────────────────
    long_notional  = sum(p["notional"] for p in longs)
    short_notional = sum(p["notional"] for p in shorts)
    gross_exp      = long_notional + short_notional
    net_exp        = long_notional - short_notional
    avg_beta_long  = np.mean([p["beta"] for p in longs])  if longs  else 0
    avg_beta_short = np.mean([p["beta"] for p in shorts]) if shorts else 0
    portfolio_beta = (
        sum(p["weight_pct"] / 100 * p["beta"] for p in longs)
        - sum(p["weight_pct"] / 100 * p["beta"] for p in shorts)
    )
    est_annual_borrow = sum(
        p["notional"] * p["borrow_pa_pct"] / 100 for p in shorts
    )
    est_slip_cost = sum(
        p["notional"] * p["slip_bps"] / 10_000 for p in longs + shorts
    )

    summary = {
        "as_of_date":          str(today.date()),
        "regime":              regime["label"],
        "gross_mult":          gross_mult,
        "n_longs":             len(longs),
        "n_shorts":            len(shorts),
        "capital":             capital,
        "long_notional":       long_notional,
        "short_notional":      short_notional,
        "gross_exposure_pct":  round(gross_exp / capital * 100, 1),
        "net_exposure_pct":    round(net_exp  / capital * 100, 1),
        "portfolio_beta":      round(portfolio_beta, 3),
        "avg_beta_longs":      round(float(avg_beta_long), 3),
        "avg_beta_shorts":     round(float(avg_beta_short), 3),
        "est_annual_borrow_$": round(est_annual_borrow, 0),
        "est_slip_per_rebal_$": round(est_slip_cost, 0),
        "n_sectors_long":  len({p["sector"] for p in longs}),
        "n_sectors_short": len({p["sector"] for p in shorts}),
        "n_intl_long":  sum(1 for p in longs  if p["intl"]),
        "n_intl_short": sum(1 for p in shorts if p["intl"]),
    }

    out = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "regime":        regime["label"],
        "gross_mult":    gross_mult,
        "capital":       capital,
        "longs":         longs,
        "shorts":        shorts,
        "summary":       summary,
    }

    # ── Persist ───────────────────────────────────────────────────────────
    SIGNAL_FILE.write_text(json.dumps(out, indent=2, default=str))
    logger.info("Live signals saved → %s", SIGNAL_FILE)

    return out


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_signals(sig: dict) -> None:
    s = sig["summary"]
    print(f"\n{'='*60}")
    print(f"  LIVE PORTFOLIO  —  {s['as_of_date']}")
    print(f"  Régime: {s['regime']} (gross ×{s['gross_mult']})  |  Capital: ${s['capital']:,.0f}")
    print(f"  Gross: {s['gross_exposure_pct']:.0f}%  Net: {s['net_exposure_pct']:+.0f}%  Beta: {s['portfolio_beta']:+.3f}")
    print(f"  Frais emprunt estimés/an: ${s['est_annual_borrow_$']:,.0f}")
    print(f"{'='*60}")

    print(f"\n🟢 LONGS ({s['n_longs']} positions)")
    print(f"  {'Ticker':<8} {'Sector':<28} {'Weight':>7} {'Notional':>10} {'Score':>7} {'Mom12-1':>8} {'52WkH':>7}")
    print(f"  {'-'*80}")
    for p in sig["longs"][:15]:
        print(f"  {p['ticker']:<8} {p['sector'][:27]:<28} {p['weight_pct']:>6.1f}%"
              f" ${p['notional']:>9,.0f} {p['score']:>7.3f} {p['mom_12_1']:>8.1%} {p['near52']:>7.1%}")

    print(f"\n🔴 SHORTS ({s['n_shorts']} positions)")
    print(f"  {'Ticker':<8} {'Sector':<28} {'Weight':>7} {'Notional':>10} {'Score':>7} {'Mom12-1':>8} {'52WkH':>7} {'Borrow':>8}")
    print(f"  {'-'*88}")
    for p in sig["shorts"][:15]:
        print(f"  {p['ticker']:<8} {p['sector'][:27]:<28} {p['weight_pct']:>6.1f}%"
              f" ${p['notional']:>9,.0f} {p['score']:>7.3f} {p['mom_12_1']:>8.1%}"
              f" {p['near52']:>7.1%} {p['borrow_pa_pct']:>7.1f}%")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute live long/short signals")
    parser.add_argument("--capital", type=float, default=100_000,
                        help="Reference portfolio capital in USD (default: 100000)")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output JSON path")
    args = parser.parse_args()

    if args.output:
        SIGNAL_FILE = Path(args.output)

    signals = compute_live_signals(capital=args.capital)
    print_signals(signals)
