"""
Phase 4 — Volatility-cap parameter robustness sweep.
Loads cached prices once, runs the walk-forward for several (VOL_TARGET,
VOL_LOOKBACK) configs, and prints a comparison table.  A robust parameter
shows a smooth plateau, not a single lucky spike.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.INFO)   # silence the per-rebal chatter

import walkforward_backtest as wf

CONFIGS = [
    # (VOL_TARGET, VOL_LOOKBACK)
    (0.06, 42),
    (0.07, 42),
    (0.08, 42),   # current
    (0.09, 42),
    (0.10, 42),
    (0.08, 21),
    (0.08, 63),
    (None, None),  # vol cap OFF (Phase 2 baseline reference)
]

rows = []
for vt, vl in CONFIGS:
    if vt is None:
        wf.USE_VOL_TARGET = False
        label = "OFF (Phase 2)"
    else:
        wf.USE_VOL_TARGET = True
        wf.VOL_TARGET   = vt
        wf.VOL_LOOKBACK = vl
        label = f"target={vt:.0%} lookback={vl}d"

    res = wf.run_walkforward(start_year=2019, force_refresh=False)
    m = res["metrics"]
    rows.append((label, m["CAGR"], m["Sharpe"], m["MaxDD"], m["Vol"]))
    print(f"{label:28s}  CAGR={m['CAGR']*100:+5.2f}%  "
          f"Sharpe={m['Sharpe']:.3f}  MaxDD={m['MaxDD']*100:6.1f}%  Vol={m['Vol']*100:.1f}%")

print("\n" + "=" * 70)
print("BEST BY SHARPE:")
for label, cagr, sh, dd, vol in sorted(rows, key=lambda r: -r[2])[:3]:
    print(f"  {label:28s}  Sharpe={sh:.3f}  CAGR={cagr*100:+.2f}%  MaxDD={dd*100:.1f}%")
