"""
Market-Neutral Strategy Dashboard  — dark theme, Dash + Bootstrap
Reads from output/walkforward_results.json (auto-refreshes every 30 s).
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

logger = logging.getLogger(__name__)

RESULTS_PATH = Path("output") / "walkforward_results.json"
SIGNAL_PATH  = Path("output") / "live_signals.json"

# ── Colour palette ────────────────────────────────────────────────────────────
BG     = "#0d1117"
CARD   = "#161b22"
BORDER = "#30363d"
GREEN  = "#3fb950"
RED    = "#f85149"
BLUE   = "#58a6ff"
AMBER  = "#d29922"
GREY   = "#8b949e"
WHITE  = "#e6edf3"
FONT   = "Inter, sans-serif"

BASE_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=CARD,
    font=dict(family=FONT, color=WHITE, size=12),
    margin=dict(l=55, r=20, t=40, b=40),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data() -> dict:
    if not RESULTS_PATH.exists():
        return {}
    try:
        with open(RESULTS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def load_live_signals() -> dict:
    if not SIGNAL_PATH.exists():
        return {}
    try:
        with open(SIGNAL_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def position_table(positions: list, side: str) -> dbc.Table:
    """Render a dark-styled positions table."""
    is_short = (side == "SHORT")
    hdr_cells = ["Ticker", "Sector", "Wt %", "Notional", "Score", "12-1m", "52WkH", "Beta"]
    if is_short:
        hdr_cells.append("Borrow/yr")
    header = html.Thead(
        html.Tr([html.Th(h, style={"color": GREY, "fontSize": "0.72rem",
                                    "padding": "4px 8px"}) for h in hdr_cells]),
    )
    rows = []
    for p in positions[:15]:
        color = GREEN if side == "LONG" else RED
        m121  = p.get("mom_12_1", 0) or 0
        n52   = p.get("near52",   0) or 0
        cells = [
            html.Td(p["ticker"],  style={"fontWeight": 700, "color": color, "padding": "3px 8px"}),
            html.Td(p.get("sector","")[:22], style={"color": GREY, "fontSize": "0.75rem", "padding": "3px 8px"}),
            html.Td(f"{p['weight_pct']:+.1f}%",   style={"padding": "3px 8px"}),
            html.Td(f"${p['notional']:,.0f}",      style={"padding": "3px 8px"}),
            html.Td(f"{p['score']:+.3f}",          style={"padding": "3px 8px"}),
            html.Td(f"{m121:.1%}", style={"color": GREEN if m121 > 0 else RED, "padding": "3px 8px"}),
            html.Td(f"{n52:.1%}",  style={"padding": "3px 8px"}),
            html.Td(f"{p.get('beta',1):.2f}",      style={"padding": "3px 8px"}),
        ]
        if is_short:
            bp = p.get("borrow_pa_pct", 0) or 0
            cells.append(html.Td(f"{bp:.1f}%", style={"color": AMBER, "padding": "3px 8px"}))
        rows.append(html.Tr(cells))
    body = html.Tbody(rows)
    return dbc.Table([header, body],
                     bordered=False, hover=True, size="sm",
                     style={"color": WHITE, "fontSize": "0.8rem", "marginBottom": 0})


def to_series(obj) -> pd.Series:
    if isinstance(obj, dict):
        return pd.Series({pd.Timestamp(k): float(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return pd.Series(obj, dtype=float)
    return pd.Series(dtype=float)


def empty_fig(msg: str = "No data yet…") -> go.Figure:
    fig = go.Figure(layout={**BASE_LAYOUT})
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color=GREY, size=13))
    return fig


def kpi_card(label: str, value: str, color: str = WHITE) -> dbc.Col:
    return dbc.Col(
        dbc.Card(dbc.CardBody([
            html.P(label, className="mb-1",
                   style={"fontSize": "0.7rem", "color": GREY, "letterSpacing": "0.06em",
                          "textTransform": "uppercase"}),
            html.H4(value, style={"color": color, "fontWeight": 700, "margin": 0}),
        ]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": 8}),
        xs=6, sm=4, md=3, lg=2, className="mb-3",
    )


# ── App ───────────────────────────────────────────────────────────────────────

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
           title="Momentum Market-Neutral", suppress_callback_exceptions=True)

app.layout = dbc.Container(fluid=True,
    style={"backgroundColor": BG, "minHeight": "100vh", "padding": "24px"},
    children=[
        # ── Header ─────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.H3("⚡ Global Momentum Market-Neutral Strategy",
                        style={"color": WHITE, "fontWeight": 700, "marginBottom": 2}),
                html.Small("Sector-relative momentum · 39L/39S overlapping · "
                           "regime filter + 5% volatility cap · dollar-neutral",
                           style={"color": GREY}),
            ]),
            dbc.Col(html.Small(id="ts-label", style={"color": GREY}),
                    width="auto", className="d-flex align-items-center"),
        ], className="mb-4"),

        # ── KPIs ────────────────────────────────────────────────────────────
        dbc.Row(id="kpi-row", className="mb-3"),

        # ── Row 1: Cumulative | Drawdown ────────────────────────────────────
        dbc.Row([
            dbc.Col(dcc.Graph(id="fig-cum",    config={"displayModeBar": False}), md=8),
            dbc.Col(dcc.Graph(id="fig-dd",     config={"displayModeBar": False}), md=4),
        ], className="mb-3"),

        # ── Row 2: Long/Short decomp | Rolling Sharpe ───────────────────────
        dbc.Row([
            dbc.Col(dcc.Graph(id="fig-ls",     config={"displayModeBar": False}), md=6),
            dbc.Col(dcc.Graph(id="fig-sharpe", config={"displayModeBar": False}), md=6),
        ], className="mb-3"),

        # ── Row 3: Monthly heatmap | Return distribution ────────────────────
        dbc.Row([
            dbc.Col(dcc.Graph(id="fig-heat",   config={"displayModeBar": False}), md=7),
            dbc.Col(dcc.Graph(id="fig-dist",   config={"displayModeBar": False}), md=5),
        ], className="mb-3"),

        # ── Row 4: Sector P&L | Regime timeline ─────────────────────────────
        dbc.Row([
            dbc.Col(dcc.Graph(id="fig-sector", config={"displayModeBar": False}), md=6),
            dbc.Col(dcc.Graph(id="fig-regime", config={"displayModeBar": False}), md=6),
        ], className="mb-3"),

        # ── Row 5: Trade scatter | Top holdings ──────────────────────────────
        dbc.Row([
            dbc.Col(dcc.Graph(id="fig-scatter", config={"displayModeBar": False}), md=6),
            dbc.Col(html.Div(id="holdings"),                                        md=6),
        ], className="mb-3"),

        # ── Divider ───────────────────────────────────────────────────────────
        html.Hr(style={"borderColor": BORDER, "margin": "24px 0"}),

        # ── Live Signals section ─────────────────────────────────────────────
        dbc.Row([
            dbc.Col(html.H4("📡 Portfolio du Jour — Live Signals",
                            style={"color": WHITE, "fontWeight": 700})),
            dbc.Col(html.Small(id="live-ts-label", style={"color": GREY}),
                    width="auto", className="d-flex align-items-center"),
        ], className="mb-3"),

        # Live KPI row
        dbc.Row(id="live-kpi-row", className="mb-3"),

        # Long / Short tables
        dbc.Row([
            dbc.Col([
                html.H6("🟢 LONGS", style={"color": GREEN, "fontWeight": 700, "marginBottom": 8}),
                html.Div(id="live-longs-table"),
            ], md=6),
            dbc.Col([
                html.H6("🔴 SHORTS", style={"color": RED, "fontWeight": 700, "marginBottom": 8}),
                html.Div(id="live-shorts-table"),
            ], md=6),
        ], className="mb-4"),

        dcc.Interval(id="tick", interval=30_000, n_intervals=0),
    ],
)


def build_live_section(sig: dict):
    """Build KPI cards + position tables from live_signals.json."""
    if not sig:
        placeholder = html.P("Run live_signals.py to generate signals.",
                              style={"color": GREY, "fontStyle": "italic"})
        return [], placeholder, placeholder

    s = sig["summary"]
    regime_label = s.get("regime", "—")
    regime_color = {"Bull": GREEN, "Bear": RED, "Choppy": AMBER,
                    "Caution": AMBER, "Unknown": GREY}.get(regime_label, WHITE)

    live_kpis = dbc.Row([
        kpi_card("Regime",       regime_label,                         regime_color),
        kpi_card("Gross ×",      f"{s.get('gross_mult', 1):.2f}",      WHITE),
        kpi_card("Gross Exp",    f"{s.get('gross_exposure_pct', 0):.0f}%", WHITE),
        kpi_card("Net Exp",      f"{s.get('net_exposure_pct', 0):+.0f}%",  WHITE),
        kpi_card("Port Beta",    f"{s.get('portfolio_beta', 0):+.3f}",  WHITE),
        kpi_card("Longs",        str(s.get("n_longs",  0)),             GREEN),
        kpi_card("Shorts",       str(s.get("n_shorts", 0)),             RED),
        kpi_card("Borrow/yr",    f"${s.get('est_annual_borrow_$', 0):,.0f}", AMBER),
        kpi_card("Slip/Rebal",   f"${s.get('est_slip_per_rebal_$', 0):,.0f}", GREY),
    ])

    longs_tbl  = dbc.Card(
        dbc.CardBody(position_table(sig.get("longs", []),  "LONG"),  style={"padding": 8}),
        style={"background": CARD, "border": f"1px solid {BORDER}"},
    ) if sig.get("longs") else html.P("No longs", style={"color": GREY})

    shorts_tbl = dbc.Card(
        dbc.CardBody(position_table(sig.get("shorts", []), "SHORT"), style={"padding": 8}),
        style={"background": CARD, "border": f"1px solid {BORDER}"},
    ) if sig.get("shorts") else html.P("No shorts", style={"color": GREY})

    return live_kpis, longs_tbl, shorts_tbl


@app.callback(
    Output("kpi-row",          "children"),
    Output("fig-cum",          "figure"),
    Output("fig-dd",           "figure"),
    Output("fig-ls",           "figure"),
    Output("fig-sharpe",       "figure"),
    Output("fig-heat",         "figure"),
    Output("fig-dist",         "figure"),
    Output("fig-sector",       "figure"),
    Output("fig-regime",       "figure"),
    Output("fig-scatter",      "figure"),
    Output("holdings",         "children"),
    Output("ts-label",         "children"),
    Output("live-kpi-row",     "children"),
    Output("live-longs-table", "children"),
    Output("live-shorts-table","children"),
    Output("live-ts-label",    "children"),
    Input("tick",              "n_intervals"),
)
def refresh(_n):
    data    = load_data()
    metrics = data.get("metrics", {})
    trades  = data.get("trades", [])
    rebal   = data.get("rebal_history", [])

    cum_s   = to_series(data.get("cumulative_returns", {}))
    bench_s = to_series(data.get("benchmark_returns",  {}))
    long_s  = to_series(data.get("long_cumulative",    {}))
    short_s = to_series(data.get("short_cumulative",   {}))
    dd_s    = to_series(data.get("drawdown",            {}))
    rs_s    = to_series(data.get("rolling_sharpe",      {}))

    ts = datetime.now().strftime("Updated %H:%M:%S")

    # ── Live signals (always attempted, independent of backtest data) ─────────
    sig       = load_live_signals()
    live_ts   = f"Signals as of {sig['summary']['as_of_date']}" if sig else "No signals yet"
    live_kpis_row, live_longs_div, live_shorts_div = build_live_section(sig)

    if not data:
        kpis = [kpi_card("Status", "Waiting for backtest…", AMBER)]
        ef   = empty_fig()
        return (kpis, ef, ef, ef, ef, ef, ef, ef, ef, ef,
                html.P("No data", className="text-muted"), ts,
                live_kpis_row, live_longs_div, live_shorts_div, live_ts)

    # ── KPIs ─────────────────────────────────────────────────────────────────
    m = metrics
    cagr, sharpe, maxdd = m.get("CAGR",0), m.get("Sharpe",0), m.get("MaxDD",0)
    alpha, beta, wr     = m.get("Alpha",0), m.get("Beta",0),   m.get("WinRate",0)
    vol                 = m.get("Vol",0)

    kpis = dbc.Row([
        kpi_card("CAGR",      f"{cagr*100:+.1f}%",   GREEN if cagr > 0 else RED),
        kpi_card("Sharpe",    f"{sharpe:.2f}",        GREEN if sharpe > 0.5 else (AMBER if sharpe > 0 else RED)),
        kpi_card("Max DD",    f"{maxdd*100:.1f}%",    RED),
        kpi_card("Alpha",     f"{alpha*100:+.1f}%",   GREEN if alpha > 0 else RED),
        kpi_card("Beta",      f"{beta:.2f}",          WHITE),
        kpi_card("Win Rate",  f"{wr*100:.1f}%",       GREEN if wr > 0.5 else AMBER),
        kpi_card("Ann Vol",   f"{vol*100:.1f}%",      WHITE),
        kpi_card("Universe",  str(m.get("Universe","—")), WHITE),
        kpi_card("# Trades",  str(m.get("NTrades","—")), WHITE),
    ])

    # ── Cumulative returns ────────────────────────────────────────────────────
    fig_cum = go.Figure(layout={**BASE_LAYOUT, "title": {"text": "Cumulative Returns vs SPY"}})
    if not cum_s.empty:
        fig_cum.add_trace(go.Scatter(x=cum_s.index,   y=cum_s.values-1,
                                     name="Strategy", line=dict(color=BLUE, width=2.5)))
    if not bench_s.empty:
        fig_cum.add_trace(go.Scatter(x=bench_s.index, y=bench_s.values-1,
                                     name="SPY", line=dict(color=GREY, width=1.5, dash="dot")))
    fig_cum.update_yaxes(tickformat=".0%")
    fig_cum.add_hline(y=0, line_color=BORDER, line_width=1)

    # ── Drawdown ──────────────────────────────────────────────────────────────
    fig_dd = go.Figure(layout={**BASE_LAYOUT, "title": {"text": "Drawdown"}})
    if not dd_s.empty:
        fig_dd.add_trace(go.Scatter(x=dd_s.index, y=dd_s.values,
                                    fill="tozeroy", line=dict(color=RED, width=1.5),
                                    fillcolor="rgba(248,81,73,0.18)", name="DD"))
    fig_dd.update_yaxes(tickformat=".0%")

    # ── Long / Short decomposition ────────────────────────────────────────────
    fig_ls = go.Figure(layout={**BASE_LAYOUT, "title": {"text": "Long vs Short Leg P&L"}})
    if not long_s.empty:
        fig_ls.add_trace(go.Scatter(x=long_s.index,  y=long_s.values-1,  name="Long",
                                    line=dict(color=GREEN, width=2)))
    if not short_s.empty:
        fig_ls.add_trace(go.Scatter(x=short_s.index, y=short_s.values-1, name="Short",
                                    line=dict(color=RED,   width=2)))
    if not cum_s.empty:
        fig_ls.add_trace(go.Scatter(x=cum_s.index,   y=cum_s.values-1,   name="Combined",
                                    line=dict(color=BLUE,  width=2, dash="dot")))
    fig_ls.update_yaxes(tickformat=".0%")

    # ── Rolling Sharpe ────────────────────────────────────────────────────────
    fig_sharpe = go.Figure(layout={**BASE_LAYOUT, "title": {"text": "Rolling Sharpe (63 d)"}})
    if not rs_s.empty:
        fig_sharpe.add_trace(go.Bar(x=rs_s.index, y=rs_s.values,
                                    marker_color=[GREEN if v >= 0 else RED for v in rs_s.values],
                                    name="Sharpe"))
    fig_sharpe.add_hline(y=0,  line_color=GREY,  line_width=1)
    fig_sharpe.add_hline(y=1,  line_color=GREEN, line_dash="dot", line_width=1)
    fig_sharpe.add_hline(y=-1, line_color=RED,   line_dash="dot", line_width=1)

    # ── Monthly heatmap ───────────────────────────────────────────────────────
    fig_heat = empty_fig("No monthly data")
    if not cum_s.empty:
        monthly = (cum_s.resample("ME").last().pct_change().dropna() * 100)
        if not monthly.empty:
            df_m = pd.DataFrame({"Y": monthly.index.year,
                                  "M": monthly.index.strftime("%b"),
                                  "R": monthly.values})
            months = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"]
            piv = df_m.pivot_table(index="Y", columns="M", values="R")
            piv = piv.reindex(columns=[m for m in months if m in piv.columns])
            z   = piv.values
            txt = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z]
            fig_heat = go.Figure(
                go.Heatmap(z=z, x=piv.columns.tolist(), y=piv.index.tolist(),
                           colorscale=[[0,RED],[0.5,CARD],[1,GREEN]],
                           zmid=0, text=txt, texttemplate="%{text}",
                           colorbar=dict(title="%")),
                layout={**BASE_LAYOUT, "title": {"text": "Monthly Returns (%)"}},
            )

    # ── Return distribution ───────────────────────────────────────────────────
    fig_dist = empty_fig("No daily data")
    if not cum_s.empty:
        dr = cum_s.pct_change().dropna() * 100
        fig_dist = go.Figure(layout={**BASE_LAYOUT, "title": {"text": "Daily Return Distribution"}})
        fig_dist.add_trace(go.Histogram(x=dr.values, nbinsx=70,
                                         marker_color=BLUE, opacity=0.7, name="Strategy"))
        if not bench_s.empty:
            br = bench_s.pct_change().dropna() * 100
            fig_dist.add_trace(go.Histogram(x=br.values, nbinsx=70,
                                             marker_color=GREY, opacity=0.4, name="SPY"))
        fig_dist.update_layout(barmode="overlay")
        fig_dist.update_xaxes(title="Daily Return (%)", ticksuffix="%")

    # ── Sector P&L ────────────────────────────────────────────────────────────
    fig_sector = empty_fig("No trade data")
    if trades:
        df_t = pd.DataFrame(trades)
        if {"sector","return_pct","direction"}.issubset(df_t.columns):
            grp = df_t.groupby(["sector","direction"]).agg(
                avg_ret=("return_pct","mean"), count=("return_pct","size")).reset_index()
            fig_sector = go.Figure(layout={**BASE_LAYOUT, "title": {"text": "Avg Return by Sector & Side"}})
            for side, color in [("LONG", GREEN), ("SHORT", RED)]:
                sub = grp[grp["direction"] == side].sort_values("avg_ret")
                fig_sector.add_trace(go.Bar(y=sub["sector"], x=sub["avg_ret"],
                                            orientation="h", name=side,
                                            marker_color=color, opacity=0.8))
            fig_sector.update_layout(barmode="group")
            fig_sector.update_xaxes(ticksuffix="%")

    # ── Regime timeline ───────────────────────────────────────────────────────
    fig_regime = empty_fig("No rebalancing history")
    if rebal:
        df_r = pd.DataFrame(rebal)
        df_r["date"] = pd.to_datetime(df_r["date"])
        # Effective gross = raw gross × (regime × vol-cap multiplier). Falls back
        # to raw gross for older result files without the field.
        gcol = "effective_gross" if "effective_gross" in df_r.columns else "gross"
        clrs = {"Bull": GREEN, "Choppy": AMBER, "Caution": AMBER,
                "Bear": RED, "Unknown": GREY}
        fig_regime = go.Figure(layout={**BASE_LAYOUT,
                                       "title": {"text": "Regime & Effective Gross (incl. vol cap)"}})
        # Continuous grey line so the de-grossing trajectory is visible…
        fig_regime.add_trace(go.Scatter(
            x=df_r["date"], y=df_r[gcol], mode="lines",
            line=dict(color=GREY, width=1), name="Effective gross",
            showlegend=False, hoverinfo="skip",
        ))
        # …then colour the markers by regime
        for lbl, grp in df_r.groupby("regime"):
            fig_regime.add_trace(go.Scatter(
                x=grp["date"], y=grp[gcol],
                mode="markers", name=lbl,
                marker=dict(size=7, color=clrs.get(lbl, GREY)),
            ))
        fig_regime.update_yaxes(title="Deployed gross exposure")

    # ── Trade scatter ─────────────────────────────────────────────────────────
    fig_scatter = empty_fig("No trades")
    if trades:
        df_t = pd.DataFrame(trades)
        if {"composite","return_pct","direction"}.issubset(df_t.columns):
            fig_scatter = go.Figure(layout={**BASE_LAYOUT,
                                            "title": {"text": "Factor Score vs Realised Return"}})
            for side, color in [("LONG", GREEN), ("SHORT", RED)]:
                sub = df_t[df_t["direction"] == side]
                if not sub.empty:
                    fig_scatter.add_trace(go.Scatter(
                        x=sub["composite"], y=sub["return_pct"],
                        mode="markers", name=side,
                        marker=dict(color=color, size=4, opacity=0.45),
                    ))
            fig_scatter.update_xaxes(title="Composite Score")
            fig_scatter.update_yaxes(title="Return %", ticksuffix="%")
            # add trend line annotation
            fig_scatter.add_hline(y=0, line_color=BORDER, line_width=1)
            fig_scatter.add_vline(x=0, line_color=BORDER, line_width=1)

    # ── Top holdings ──────────────────────────────────────────────────────────
    holdings = html.P("No holdings", className="text-muted")
    if rebal:
        last = rebal[-1]
        def rows(tickers, side, color):
            return [html.Tr([
                html.Td(t, style={"fontWeight": 600}),
                html.Td(side, style={"color": color}),
            ]) for t in tickers]

        table = dbc.Table([
            html.Thead(html.Tr([html.Th("Ticker"), html.Th("Side")]),
                       style={"color": GREY}),
            html.Tbody(
                rows(last.get("top_longs",  []), "LONG",  GREEN) +
                rows(last.get("top_shorts", []), "SHORT", RED)
            ),
        ], bordered=False, hover=True, size="sm", style={"color": WHITE})

        holdings = dbc.Card([
            dbc.CardHeader(
                html.Div([
                    html.Strong("Latest Portfolio — top picks", style={"color": WHITE}),
                    html.Small(f"  {last['date']}  ·  "
                               f"{last['n_long']}L / {last['n_short']}S  ·  "
                               f"regime {last['regime']}",
                               style={"color": GREY}),
                ]),
                style={"background": CARD, "borderColor": BORDER},
            ),
            dbc.CardBody(table, style={"padding": "0 12px"}),
        ], style={"background": CARD, "border": f"1px solid {BORDER}"})

    return (kpis, fig_cum, fig_dd, fig_ls, fig_sharpe,
            fig_heat, fig_dist, fig_sector, fig_regime,
            fig_scatter, holdings, ts,
            live_kpis_row, live_longs_div, live_shorts_div, live_ts)


if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("DASH_PORT", 8050)), debug=False)
