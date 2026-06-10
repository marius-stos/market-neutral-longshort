# Market-Neutral Long/Short Strategy

Systematic equity long/short built in Python with full walk-forward backtesting and a live Dash dashboard.

---

## Strategy — Sector-Relative Momentum Long/Short

**Approach:** Market-neutral equity long/short using Jegadeesh-Titman momentum, sector-normalized within GICS sectors to isolate stock-selection alpha. Dollar-neutral, ~zero beta.

**Signals (composite, sector-relative z-scored before global rank):**
- 12-1m momentum (40%), residual/market-adjusted momentum (10%), 6-1m (18%), 3-1m (10%), 52-week-high proximity (22% — George & Hwang 2004), small IVOL penalty (−5%)

**Portfolio construction:**
- 39 long / 39 short, **overlapping sub-portfolios** (3×13, each held 3 months) → turnover cut ~70%
- Conviction-scaled weights: |signal| × inverse-volatility
- Sector caps + 52-week-high filter on shorts (avoids crashed-stock bounce risk)

**Risk overlay:**
- **3-signal regime filter** (SMA-200 trend + 1m return + vol ratio) → gross multiplier [0.50, 1.0]
- **5% volatility cap** (de-lever only) using 42-day trailing realised vol — tuned via parameter sweep
- Earnings blackout: positions within 7 days of earnings halved

**Universe:** ~600 tickers — US large/mid caps + European and Asian ADRs across all GICS sectors

**Costs modelled:**
- Slippage: 8 bps one-way (US) / 15 bps (ADRs)
- Short borrow: 0.5%/yr (US) / 1.5%/yr (ADRs)

**Backtest results (2019–2026, walk-forward):**
| Metric | Value |
|--------|-------|
| CAGR | +3.97% |
| Sharpe | **0.64** |
| Max Drawdown | -8.3% |
| Alpha | +4.1% |
| Beta | -0.01 ✅ |
| Ann. Vol | 6.2% |
| Win Rate | 54.6% |
| Return skew / kurtosis | -0.27 / 2.5 |

*Developed in phases with controlled, single-variable backtests. The volatility cap was tuned via a parameter sweep (`param_sweep.py`) showing a smooth Sharpe plateau peaking at 5%/42d — not a single lucky point. Candidate signals that failed (short-term reversal, high-IVOL shorts) were rejected by the backtest and documented in the code.*

---

## Files

| File | Description |
|------|-------------|
| `walkforward_backtest.py` | Full walk-forward backtest engine |
| `live_signals.py` | Generates today's long/short signals |
| `dashboard_v2.py` | Dash dashboard (backtest + live signals) |
| `universe_global.py` | ~600-ticker universe with sector mapping |
| `param_sweep.py`, `param_sweep2.py` | Volatility-cap parameter sweeps (analysis artifacts) |
| `edgar_fundamentals.py` | SEC EDGAR XBRL PIT fundamentals loader |
| `alternative_data.py` | Alternative data pipeline |
| `signal_enhancers.py` | Signal enhancement utilities |

---

## Setup

```bash
pip install yfinance pandas numpy dash dash-bootstrap-components plotly requests
```

**Run backtest:**
```bash
python3 walkforward_backtest.py
```

**Generate live signals:**
```bash
python3 live_signals.py --capital 100000
```

**Launch dashboard:**
```bash
python3 dashboard_v2.py
# → http://127.0.0.1:8050
```

**Reproduce the volatility-cap parameter sweep:**
```bash
python3 param_sweep.py    # coarse: cap 6–10%, lookback 21/42/63d
python3 param_sweep2.py   # fine: cap 4.5–6% × lookback 15/21/42d
```
