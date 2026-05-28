# Market-Neutral Long/Short Strategy

Systematic equity long/short built in Python with full walk-forward backtesting and a live Dash dashboard.

---

## Strategy — Sector-Relative Momentum Long/Short

**Approach:** Market-neutral equity long/short using Jegadeesh-Titman momentum, sector-normalized within GICS sectors to isolate stock selection alpha.

**Signals:**
- 12-1m momentum (45%), 6-1m (20%), 3-1m (10%), 52-week high proximity (25%) — George & Hwang 2004
- Signals Z-scored within sector before global ranking → removes sector-timing noise
- Regime filter (Bull/Choppy/Bear via SMA-200) adjusting gross exposure

**Universe:** ~600 tickers — US large/mid caps + European and Asian ADRs across all GICS sectors

**Costs modelled:**
- Slippage: 8 bps one-way (US) / 15 bps (ADRs)
- Short borrow: 0.5%/yr (US) / 1.5%/yr (ADRs)

**Backtest results (2015–2026):**
| Metric | Value |
|--------|-------|
| CAGR | 1.8% |
| Sharpe | 0.26 |
| Max Drawdown | -15.1% |
| Alpha | 1.8% |
| Beta | 0.00 ✅ |
| Win Rate | 54.6% |

---

## Files

| File | Description |
|------|-------------|
| `walkforward_backtest.py` | Full walk-forward backtest engine |
| `live_signals.py` | Generates today's long/short signals |
| `dashboard_v2.py` | Dash dashboard (backtest + live signals) |
| `universe_global.py` | ~600-ticker universe with sector mapping |
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

**Scan M&A deals:**
```bash
python3 deal_scraper.py --days 60 --capital 100000
```
