"""
Phase 4 — Extension sweep: cross tight vol caps with short lookbacks.
First sweep showed monotonic improvement toward tighter caps and shorter
lookbacks (best so far: 6%/42d=0.600, 8%/21d=0.587). This crosses both axes
to find the turning point — where the cap starts amputating return, not just vol.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.INFO)

import walkforward_backtest as wf

CONFIGS = [
    (0.045, 21), (0.05, 21), (0.055, 21), (0.06, 21),
    (0.045, 15), (0.05, 15), (0.055, 15), (0.06, 15),
    (0.05, 42),  (0.055, 42),
]

rows = []
for vt, vl in CONFIGS:
    wf.USE_VOL_TARGET = True
    wf.VOL_TARGET   = vt
    wf.VOL_LOOKBACK = vl
    label = f"target={vt:.1%} lookback={vl}d"
    res = wf.run_walkforward(start_year=2019, force_refresh=False)
    m = res["metrics"]
    rows.append((label, m["CAGR"], m["Sharpe"], m["MaxDD"], m["Vol"]))
    print(f"{label:28s}  CAGR={m['CAGR']*100:+5.2f}%  "
          f"Sharpe={m['Sharpe']:.3f}  MaxDD={m['MaxDD']*100:6.1f}%  Vol={m['Vol']*100:.1f}%")

print("\n" + "=" * 70)
print("BEST BY SHARPE:")
for label, cagr, sh, dd, vol in sorted(rows, key=lambda r: -r[2])[:4]:
    print(f"  {label:28s}  Sharpe={sh:.3f}  CAGR={cagr*100:+.2f}%  MaxDD={dd*100:.1f}%")
