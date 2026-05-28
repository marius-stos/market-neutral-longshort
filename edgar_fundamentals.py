"""
edgar_fundamentals.py
---------------------
Point-in-time fundamental data loader using SEC EDGAR XBRL API.
No look-ahead bias: only filings with filed_date <= as_of_date are used.
Python 3.9 compatible (no X|Y union types).
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = Path("output/edgar_cache")
HEADERS = {"User-Agent": "QuantResearch/1.0 research@example.com"}
CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
XBRL_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

XBRL_CONCEPTS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "total_assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "shares_outstanding": ["CommonStockSharesOutstanding"],
    "operating_cf": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "rd_expense": ["ResearchAndDevelopmentExpense"],
}

STOCK_FIELDS = {
    "total_assets",
    "current_assets",
    "current_liabilities",
    "equity",
    "long_term_debt",
    "cash",
    "shares_outstanding",
}

FLOW_FIELDS = set(XBRL_CONCEPTS.keys()) - STOCK_FIELDS

# Forms to keep
VALID_FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}

# Cache TTLs
CACHE_TTL_DAYS = 30
FRED_TTL_DAYS = 7

# Rate limiting
_REQUEST_INTERVAL = 0.13  # ~8 req/s
_last_request_time = 0.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rate_limited_get(url: str, timeout: int = 30) -> requests.Response:
    """GET with SEC rate-limiting (max ~8 req/s)."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _REQUEST_INTERVAL:
        time.sleep(_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    return response


def _cache_is_fresh(path: Path, ttl_days: int = CACHE_TTL_DAYS) -> bool:
    """Return True if the cache file exists and is newer than ttl_days."""
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(days=ttl_days)


def _safe_div(numerator: float, denominator: float) -> float:
    """Division guarded against zero, NaN, and negative denominators for ratios."""
    if (
        denominator is None
        or np.isnan(denominator)
        or denominator == 0
        or numerator is None
        or np.isnan(numerator)
    ):
        return np.nan
    return numerator / denominator


# ---------------------------------------------------------------------------
# 1. CIK map
# ---------------------------------------------------------------------------


def _get_cik_map(force_refresh: bool = False) -> Dict[str, str]:
    """Fetch ticker -> zero-padded CIK from SEC, cache to edgar_cache/cik_map.json.

    Returns dict mapping UPPERCASE ticker -> 10-digit CIK string.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "cik_map.json"

    if not force_refresh and _cache_is_fresh(cache_path, ttl_days=CACHE_TTL_DAYS):
        with open(cache_path, "r") as fh:
            return json.load(fh)

    logger.info("Fetching CIK map from SEC …")
    try:
        resp = _rate_limited_get(CIK_MAP_URL)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:
        logger.error("Failed to fetch CIK map: %s", exc)
        if cache_path.exists():
            logger.warning("Using stale CIK map cache.")
            with open(cache_path, "r") as fh:
                return json.load(fh)
        raise

    # raw is {idx: {cik_str: "...", ticker: "...", title: "..."}
    cik_map: Dict[str, str] = {}
    for entry in raw.values():
        ticker = entry.get("ticker", "").upper().strip()
        cik_raw = entry.get("cik_str", "")
        if ticker and cik_raw:
            cik_map[ticker] = str(cik_raw).zfill(10)

    with open(cache_path, "w") as fh:
        json.dump(cik_map, fh)
    logger.info("CIK map cached: %d entries", len(cik_map))
    return cik_map


# ---------------------------------------------------------------------------
# 2. Fetch company facts
# ---------------------------------------------------------------------------


def _fetch_company_facts(cik: str) -> Optional[Dict]:
    """Fetch XBRL companyfacts JSON for a given 10-digit CIK.

    Returns the parsed dict or None on 404/error.
    """
    url = XBRL_URL.format(cik=cik)
    try:
        resp = _rate_limited_get(url)
        if resp.status_code == 404:
            logger.debug("No EDGAR facts for CIK %s (404)", cik)
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as exc:
        logger.warning("HTTP error fetching CIK %s: %s", cik, exc)
        return None
    except Exception as exc:
        logger.warning("Error fetching CIK %s: %s", cik, exc)
        return None


# ---------------------------------------------------------------------------
# 3. Extract concept
# ---------------------------------------------------------------------------


def _extract_concept(facts: Dict, concepts: List[str]) -> pd.DataFrame:
    """Try each concept name in list, return first match as a DataFrame.

    Columns: filed(datetime64), period_end(datetime64), val(float),
             form(str), fp(str), is_annual(bool)

    Only keeps 10-K, 10-Q, 10-K/A, 10-Q/A forms.
    Deduplicates by period_end keeping the last filed.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    for concept in concepts:
        concept_data = us_gaap.get(concept)
        if concept_data is None:
            continue

        units = concept_data.get("units", {})
        # Prefer USD, fall back to shares or first available unit
        unit_data = (
            units.get("USD")
            or units.get("shares")
            or (next(iter(units.values())) if units else None)
        )
        if not unit_data:
            continue

        rows = []
        for entry in unit_data:
            form = entry.get("form", "")
            if form not in VALID_FORMS:
                continue
            try:
                filed = pd.to_datetime(entry["filed"])
                period_end = pd.to_datetime(entry["end"])
                val = float(entry["val"])
                fp = entry.get("fp", "")
                is_annual = form in ("10-K", "10-K/A") or fp == "FY"
            except (KeyError, ValueError, TypeError):
                continue
            rows.append(
                {
                    "filed": filed,
                    "period_end": period_end,
                    "val": val,
                    "form": form,
                    "fp": fp,
                    "is_annual": is_annual,
                }
            )

        if not rows:
            continue

        df = pd.DataFrame(rows)
        df.sort_values("filed", inplace=True)
        # Deduplicate by period_end keeping last filed
        df = df.drop_duplicates(subset=["period_end"], keep="last")
        df.reset_index(drop=True, inplace=True)
        return df

    return pd.DataFrame(
        columns=["filed", "period_end", "val", "form", "fp", "is_annual"]
    )


# ---------------------------------------------------------------------------
# 4. De-annualize quarterly (YTD -> single quarter)
# ---------------------------------------------------------------------------


def _deannualize_quarterly(quarterly_df: pd.DataFrame) -> pd.Series:
    """Convert potentially cumulative YTD quarterly values to single-quarter values.

    fp field: Q1 is always a single quarter.
    Q2, Q3, Q4 may be YTD cumulative — subtract previous period.
    Annual rows (FY / 10-K) are excluded from the output.

    Returns Series indexed by period_end with single-quarter values.
    """
    df = quarterly_df.copy()
    # Exclude annual rows
    df = df[~df["is_annual"]].copy()
    if df.empty:
        return pd.Series(dtype=float)

    df.sort_values("period_end", inplace=True)
    df.reset_index(drop=True, inplace=True)

    results = {}
    for i, row in df.iterrows():
        fp = row["fp"]
        period_end = row["period_end"]
        val = row["val"]

        if fp == "Q1":
            # Always a single quarter
            results[period_end] = val
        elif fp in ("Q2", "Q3", "Q4"):
            # Find the previous quarter's period_end
            # Q2: look for Q1 ~3 months before
            # Q3: look for Q2 ~3 months before
            # Q4: look for Q3 ~3 months before
            expected_prev_end = period_end - pd.DateOffset(months=3)
            # Look for a row within ±15 days of expected
            prev_rows = df[
                (df["period_end"] >= expected_prev_end - pd.Timedelta(days=15))
                & (df["period_end"] <= expected_prev_end + pd.Timedelta(days=15))
            ]
            if not prev_rows.empty:
                prev_val = prev_rows.iloc[-1]["val"]
                results[period_end] = val - prev_val
            else:
                # Cannot deannualize — store as-is (may be non-cumulative already)
                results[period_end] = val
        else:
            # FY or unknown — skip
            pass

    if not results:
        return pd.Series(dtype=float)

    series = pd.Series(results, name="val")
    series.index = pd.DatetimeIndex(series.index)
    series.sort_index(inplace=True)
    return series


# ---------------------------------------------------------------------------
# 5. Load EDGAR data
# ---------------------------------------------------------------------------


def _fetch_and_cache_ticker(
    ticker: str, cik: str, force_refresh: bool
) -> Optional[Dict[str, pd.DataFrame]]:
    """Fetch EDGAR facts for one ticker and return the field->DataFrame dict."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{cik}.json"

    if not force_refresh and _cache_is_fresh(cache_path, ttl_days=CACHE_TTL_DAYS):
        try:
            with open(cache_path, "r") as fh:
                raw = json.load(fh)
            # Deserialise DataFrames from JSON records
            result = {}
            for field, records in raw.items():
                if records:
                    df = pd.DataFrame(records)
                    df["filed"] = pd.to_datetime(df["filed"])
                    df["period_end"] = pd.to_datetime(df["period_end"])
                    result[field] = df
                else:
                    result[field] = pd.DataFrame(
                        columns=["filed", "period_end", "val", "form", "fp", "is_annual"]
                    )
            return result
        except Exception as exc:
            logger.warning("Cache read failed for %s: %s — re-fetching", ticker, exc)

    facts = _fetch_company_facts(cik)
    if facts is None:
        return None

    result = {}
    for field, concepts in XBRL_CONCEPTS.items():
        df = _extract_concept(facts, concepts)
        result[field] = df

    # Serialise to JSON cache
    try:
        serialisable = {}
        for field, df in result.items():
            if df.empty:
                serialisable[field] = []
            else:
                tmp = df.copy()
                tmp["filed"] = tmp["filed"].dt.strftime("%Y-%m-%d")
                tmp["period_end"] = tmp["period_end"].dt.strftime("%Y-%m-%d")
                serialisable[field] = tmp.to_dict(orient="records")
        with open(cache_path, "w") as fh:
            json.dump(serialisable, fh)
    except Exception as exc:
        logger.warning("Cache write failed for %s: %s", ticker, exc)

    return result


def load_edgar_data(
    tickers: List[str], force_refresh: bool = False
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load EDGAR fundamental data for a list of tickers.

    Returns {ticker: {field: pd.DataFrame(filed, period_end, val, form, fp, is_annual)}}.
    Missing tickers are fetched in parallel (max_workers=4).
    Progress is logged every 50 tickers.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cik_map = _get_cik_map(force_refresh=force_refresh)

    tickers_upper = [t.upper() for t in tickers]
    edgar_data: Dict[str, Dict[str, pd.DataFrame]] = {}

    # Determine which tickers need fetching
    to_fetch = []
    for ticker in tickers_upper:
        cik = cik_map.get(ticker)
        if cik is None:
            logger.debug("No CIK for ticker %s — skipping", ticker)
            continue
        cache_path = CACHE_DIR / f"{cik}.json"
        if force_refresh or not _cache_is_fresh(cache_path, ttl_days=CACHE_TTL_DAYS):
            to_fetch.append((ticker, cik))
        else:
            # Will be loaded lazily in the parallel block below
            to_fetch.append((ticker, cik))

    logger.info(
        "Loading EDGAR data for %d tickers (%d to fetch/check cache)",
        len(tickers_upper),
        len(to_fetch),
    )

    completed = 0

    def _worker(args: tuple) -> tuple:
        ticker, cik = args
        data = _fetch_and_cache_ticker(ticker, cik, force_refresh)
        return ticker, data

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_worker, args): args for args in to_fetch}
        for future in as_completed(futures):
            ticker, data = future.result()
            if data is not None:
                edgar_data[ticker] = data
            completed += 1
            if completed % 50 == 0:
                logger.info("EDGAR progress: %d / %d tickers", completed, len(to_fetch))

    logger.info(
        "EDGAR load complete: %d / %d tickers have data",
        len(edgar_data),
        len(tickers_upper),
    )
    return edgar_data


# ---------------------------------------------------------------------------
# 6. get_pit_value
# ---------------------------------------------------------------------------


def get_pit_value(
    ticker_data: Dict[str, pd.DataFrame],
    field: str,
    as_of_date: pd.Timestamp,
    use_ttm: bool = True,
) -> float:
    """Return a point-in-time fundamental value for one ticker+field.

    PIT filter: only filings with filed <= as_of_date are considered.
    Flow fields: TTM = sum of last 4 non-overlapping quarters.
    Stock fields: most recent single value.
    Returns np.nan if data is unavailable.
    """
    df = ticker_data.get(field)
    if df is None or df.empty:
        return np.nan

    # PIT filter
    pit_df = df[df["filed"] <= as_of_date].copy()
    if pit_df.empty:
        return np.nan

    if field in STOCK_FIELDS:
        # Most recent filing
        pit_df.sort_values("filed", inplace=True)
        return float(pit_df.iloc[-1]["val"])

    # Flow field
    if not use_ttm:
        pit_df.sort_values("filed", inplace=True)
        return float(pit_df.iloc[-1]["val"])

    # TTM: sum of last 4 non-overlapping quarterly values
    # Use de-annualised quarterly series
    quarterly = pit_df[~pit_df["is_annual"]].copy()
    if quarterly.empty:
        # Fall back to most recent annual
        annual = pit_df[pit_df["is_annual"]].copy()
        if annual.empty:
            return np.nan
        annual.sort_values("filed", inplace=True)
        return float(annual.iloc[-1]["val"])

    quarterly_vals = _deannualize_quarterly(quarterly)
    if quarterly_vals.empty:
        return np.nan

    # Only keep periods within last ~15 months (4 quarters + buffer)
    cutoff = as_of_date - pd.DateOffset(months=15)
    quarterly_vals = quarterly_vals[quarterly_vals.index >= cutoff]
    if quarterly_vals.empty:
        return np.nan

    quarterly_vals = quarterly_vals.sort_index()

    # Select last 4 non-overlapping quarters
    selected = []
    last_start = None
    for period_end in reversed(quarterly_vals.index.tolist()):
        approx_start = period_end - pd.DateOffset(months=3)
        if last_start is None or period_end < last_start:
            selected.append(quarterly_vals[period_end])
            last_start = approx_start
        if len(selected) == 4:
            break

    if len(selected) < 4:
        # Try annual as fallback
        annual = pit_df[pit_df["is_annual"]].copy()
        if not annual.empty:
            annual.sort_values("filed", inplace=True)
            return float(annual.iloc[-1]["val"])
        if selected:
            return float(sum(selected))
        return np.nan

    return float(sum(selected))


# ---------------------------------------------------------------------------
# 7. _build_pit_series
# ---------------------------------------------------------------------------


def _build_pit_series(
    ticker_data: Dict[str, pd.DataFrame], field: str
) -> pd.Series:
    """Materialise a PIT time series indexed by filing date for one ticker+field.

    For flow fields: rolling TTM at each quarterly filing date.
    For stock fields: value at each filing date.
    Returns pd.Series(index=DatetimeIndex, values=float).
    """
    df = ticker_data.get(field)
    if df is None or df.empty:
        return pd.Series(dtype=float)

    filing_dates = df["filed"].dropna().unique()
    filing_dates = pd.DatetimeIndex(sorted(filing_dates))

    if filing_dates.empty:
        return pd.Series(dtype=float)

    values = {}
    for fdate in filing_dates:
        v = get_pit_value(ticker_data, field, fdate, use_ttm=(field in FLOW_FIELDS))
        values[fdate] = v

    series = pd.Series(values, dtype=float)
    series.index = pd.DatetimeIndex(series.index)
    series.sort_index(inplace=True)
    return series


# ---------------------------------------------------------------------------
# 8. precompute_pit_panel
# ---------------------------------------------------------------------------


def precompute_pit_panel(
    edgar_data: Dict[str, Dict[str, pd.DataFrame]],
    tickers: List[str],
    prices: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    fallback_fund: Optional[pd.DataFrame] = None,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Pre-compute PIT fundamental panel for all tickers and rebalance dates.

    Returns {rebal_date: pd.DataFrame(index=tickers, columns=ratio_fields)}.
    Uses .asof() for O(1) lookup after materialising PIT series once per ticker+field.
    """
    logger.info(
        "precompute_pit_panel: %d tickers x %d rebal dates",
        len(tickers),
        len(rebal_dates),
    )

    all_fields = list(XBRL_CONCEPTS.keys())

    # Materialise all PIT series upfront: {ticker: {field: pd.Series}}
    pit_series: Dict[str, Dict[str, pd.Series]] = {}
    for ticker in tickers:
        tdata = edgar_data.get(ticker)
        if tdata is None:
            continue
        pit_series[ticker] = {}
        for field in all_fields:
            pit_series[ticker][field] = _build_pit_series(tdata, field)

    ratio_columns = [
        "trailingPE",
        "priceToBook",
        "priceToSalesTrailing12Months",
        "returnOnEquity",
        "returnOnAssets",
        "grossMargins",
        "operatingMargins",
        "profitMargins",
        "ebitdaMargins",
        "debtToEquity",
        "currentRatio",
        "quickRatio",
        "freeCashflow",
        "operatingCashflow",
        "totalRevenue",
        "totalDebt",
        "totalCash",
        "marketCap",
        "revenueGrowth",
        "earningsGrowth",
        "rdExpense",
        "piotroski_pit",
        "accruals_pit",
        "asset_bloat_pit",
        "rev_decel_pit",
    ]

    panel: Dict[pd.Timestamp, pd.DataFrame] = {}

    for rdate in rebal_dates:
        rdate_1y = rdate - pd.DateOffset(months=12)
        rdate_2y = rdate - pd.DateOffset(months=24)

        rows = {}
        for ticker in tickers:
            ts = pit_series.get(ticker)
            if ts is None:
                # Use fallback if available
                if fallback_fund is not None and ticker in fallback_fund.index:
                    rows[ticker] = fallback_fund.loc[ticker].to_dict()
                else:
                    rows[ticker] = {col: np.nan for col in ratio_columns}
                continue

            def asof(field: str, ref_date: pd.Timestamp = rdate) -> float:
                s = ts.get(field)
                if s is None or s.empty:
                    return np.nan
                val = s.asof(ref_date)
                return float(val) if not pd.isna(val) else np.nan

            # Current values
            revenue = asof("revenue")
            gross_profit = asof("gross_profit")
            operating_income = asof("operating_income")
            net_income = asof("net_income")
            total_assets = asof("total_assets")
            current_assets = asof("current_assets")
            current_liabilities = asof("current_liabilities")
            equity = asof("equity")
            long_term_debt = asof("long_term_debt")
            cash = asof("cash")
            shares = asof("shares_outstanding")
            operating_cf = asof("operating_cf")
            capex = asof("capex")
            rd_expense = asof("rd_expense")

            # 1-year-ago values
            revenue_1y = asof("revenue", rdate_1y)
            net_income_1y = asof("net_income", rdate_1y)
            total_assets_1y = asof("total_assets", rdate_1y)
            long_term_debt_1y = asof("long_term_debt", rdate_1y)
            current_assets_1y = asof("current_assets", rdate_1y)
            current_liabilities_1y = asof("current_liabilities", rdate_1y)
            gross_profit_1y = asof("gross_profit", rdate_1y)
            operating_cf_1y = asof("operating_cf", rdate_1y)
            shares_1y = asof("shares_outstanding", rdate_1y)

            # 2-year-ago values
            revenue_2y = asof("revenue", rdate_2y)

            # Price
            price = np.nan
            if ticker in prices.columns:
                price_series = prices[ticker].asof(rdate)
                price = float(price_series) if not pd.isna(price_series) else np.nan

            # Market cap
            market_cap = np.nan
            if not np.isnan(price) and not np.isnan(shares) and shares > 0:
                market_cap = price * shares

            # --- Standard ratios ---

            # trailingPE = price / (net_income_ttm / shares)
            eps = _safe_div(net_income, shares)
            trailing_pe = np.nan
            if not np.isnan(eps) and eps > 0 and not np.isnan(price):
                raw_pe = _safe_div(price, eps)
                if not np.isnan(raw_pe):
                    trailing_pe = float(np.clip(raw_pe, 3, 200))

            # priceToBook = market_cap / equity
            price_to_book = np.nan
            if not np.isnan(market_cap) and not np.isnan(equity) and equity > 0:
                price_to_book = _safe_div(market_cap, equity)

            # priceToSalesTrailing12Months = market_cap / revenue
            price_to_sales = np.nan
            if not np.isnan(market_cap) and not np.isnan(revenue) and revenue > 0:
                price_to_sales = _safe_div(market_cap, revenue)

            # returnOnEquity = net_income / equity
            roe = _safe_div(net_income, equity) if not np.isnan(equity) and equity > 0 else np.nan

            # returnOnAssets = net_income / total_assets
            roa = _safe_div(net_income, total_assets) if not np.isnan(total_assets) and total_assets > 0 else np.nan

            # grossMargins = gross_profit / revenue
            gross_margins = _safe_div(gross_profit, revenue) if not np.isnan(revenue) and revenue > 0 else np.nan

            # operatingMargins = operating_income / revenue
            op_margins = _safe_div(operating_income, revenue) if not np.isnan(revenue) and revenue > 0 else np.nan

            # profitMargins = net_income / revenue
            profit_margins = _safe_div(net_income, revenue) if not np.isnan(revenue) and revenue > 0 else np.nan

            # ebitdaMargins = operatingMargins * 1.15 (proxy)
            ebitda_margins = op_margins * 1.15 if not np.isnan(op_margins) else np.nan

            # debtToEquity = long_term_debt / equity
            debt_to_equity = _safe_div(long_term_debt, equity) if not np.isnan(equity) and equity > 0 else np.nan

            # currentRatio = current_assets / current_liabilities
            current_ratio = _safe_div(current_assets, current_liabilities) if not np.isnan(current_liabilities) and current_liabilities > 0 else np.nan

            # quickRatio = current_assets * 0.7 / current_liabilities
            quick_ratio = _safe_div(current_assets * 0.7, current_liabilities) if not np.isnan(current_assets) and not np.isnan(current_liabilities) and current_liabilities > 0 else np.nan

            # freeCashflow = operating_cf - capex
            free_cashflow = np.nan
            if not np.isnan(operating_cf) and not np.isnan(capex):
                free_cashflow = operating_cf - capex
            elif not np.isnan(operating_cf):
                free_cashflow = operating_cf

            # revenueGrowth = (revenue - revenue_1y) / abs(revenue_1y)
            revenue_growth = np.nan
            if not np.isnan(revenue) and not np.isnan(revenue_1y) and revenue_1y != 0:
                revenue_growth = (revenue - revenue_1y) / abs(revenue_1y)

            # earningsGrowth = (net_income - net_income_1y) / abs(net_income_1y)
            earnings_growth = np.nan
            if not np.isnan(net_income) and not np.isnan(net_income_1y) and net_income_1y != 0:
                earnings_growth = (net_income - net_income_1y) / abs(net_income_1y)

            # --- Piotroski F-score (9 binary signals) ---
            roa_1y = _safe_div(net_income_1y, total_assets_1y) if not np.isnan(total_assets_1y) and total_assets_1y > 0 else np.nan

            # F1: ROA > 0
            f1 = 1.0 if (not np.isnan(roa) and roa > 0) else 0.0
            # F2: OCF > 0
            f2 = 1.0 if (not np.isnan(operating_cf) and operating_cf > 0) else 0.0
            # F3: ROA improved YoY
            f3 = 1.0 if (not np.isnan(roa) and not np.isnan(roa_1y) and roa > roa_1y) else 0.0
            # F4: OCF / Assets > ROA (accruals quality)
            ocf_over_assets = _safe_div(operating_cf, total_assets) if not np.isnan(total_assets) and total_assets > 0 else np.nan
            f4 = 1.0 if (not np.isnan(ocf_over_assets) and not np.isnan(roa) and ocf_over_assets > roa) else 0.0
            # F5: leverage (LTD/Assets) decreased YoY
            lev_now = _safe_div(long_term_debt, total_assets) if not np.isnan(total_assets) and total_assets > 0 else np.nan
            lev_1y = _safe_div(long_term_debt_1y, total_assets_1y) if not np.isnan(total_assets_1y) and total_assets_1y > 0 else np.nan
            f5 = 1.0 if (not np.isnan(lev_now) and not np.isnan(lev_1y) and lev_now < lev_1y) else 0.0
            # F6: currentRatio improved YoY
            cr_1y = _safe_div(current_assets_1y, current_liabilities_1y) if not np.isnan(current_liabilities_1y) and current_liabilities_1y > 0 else np.nan
            f6 = 1.0 if (not np.isnan(current_ratio) and not np.isnan(cr_1y) and current_ratio > cr_1y) else 0.0
            # F7: no dilution (shares <= shares_1y * 1.02)
            f7 = 1.0 if (not np.isnan(shares) and not np.isnan(shares_1y) and shares_1y > 0 and shares <= shares_1y * 1.02) else 0.0
            # F8: grossMargin improved YoY
            gm_1y = _safe_div(gross_profit_1y, revenue_1y) if not np.isnan(revenue_1y) and revenue_1y > 0 else np.nan
            f8 = 1.0 if (not np.isnan(gross_margins) and not np.isnan(gm_1y) and gross_margins > gm_1y) else 0.0
            # F9: assetTurnover (Rev/Assets) improved YoY
            at_now = _safe_div(revenue, total_assets) if not np.isnan(total_assets) and total_assets > 0 else np.nan
            at_1y = _safe_div(revenue_1y, total_assets_1y) if not np.isnan(total_assets_1y) and total_assets_1y > 0 else np.nan
            f9 = 1.0 if (not np.isnan(at_now) and not np.isnan(at_1y) and at_now > at_1y) else 0.0

            piotroski = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9

            # --- Accruals: (netIncome - operatingCF) / totalAssets ---
            accruals_pit = np.nan
            if not np.isnan(net_income) and not np.isnan(operating_cf) and not np.isnan(total_assets) and total_assets > 0:
                accruals_pit = (net_income - operating_cf) / total_assets

            # --- Asset bloat: assetGrowthYoY - revenueGrowthYoY ---
            asset_bloat_pit = np.nan
            if not np.isnan(total_assets) and not np.isnan(total_assets_1y) and total_assets_1y > 0:
                asset_growth = (total_assets - total_assets_1y) / abs(total_assets_1y)
                if not np.isnan(revenue_growth):
                    asset_bloat_pit = asset_growth - revenue_growth

            # --- Rev decel: revenueGrowth_1yAgo - revenueGrowth_now ---
            rev_decel_pit = np.nan
            revenue_growth_1y_ago = np.nan
            if not np.isnan(revenue_1y) and not np.isnan(revenue_2y) and revenue_2y != 0:
                revenue_growth_1y_ago = (revenue_1y - revenue_2y) / abs(revenue_2y)
            if not np.isnan(revenue_growth_1y_ago) and not np.isnan(revenue_growth):
                rev_decel_pit = revenue_growth_1y_ago - revenue_growth

            rows[ticker] = {
                "trailingPE": trailing_pe,
                "priceToBook": price_to_book,
                "priceToSalesTrailing12Months": price_to_sales,
                "returnOnEquity": roe,
                "returnOnAssets": roa,
                "grossMargins": gross_margins,
                "operatingMargins": op_margins,
                "profitMargins": profit_margins,
                "ebitdaMargins": ebitda_margins,
                "debtToEquity": debt_to_equity,
                "currentRatio": current_ratio,
                "quickRatio": quick_ratio,
                "freeCashflow": free_cashflow,
                "operatingCashflow": operating_cf,
                "totalRevenue": revenue,
                "totalDebt": long_term_debt,
                "totalCash": cash,
                "marketCap": market_cap,
                "revenueGrowth": revenue_growth,
                "earningsGrowth": earnings_growth,
                "rdExpense": rd_expense,
                "piotroski_pit": piotroski,
                "accruals_pit": accruals_pit,
                "asset_bloat_pit": asset_bloat_pit,
                "rev_decel_pit": rev_decel_pit,
            }

        panel[rdate] = pd.DataFrame.from_dict(rows, orient="index", columns=ratio_columns)

    logger.info("precompute_pit_panel: complete for %d rebal dates", len(rebal_dates))
    return panel


# ---------------------------------------------------------------------------
# 9. load_fred_macro
# ---------------------------------------------------------------------------


def load_fred_macro(series_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """Download macro series from FRED and return a combined DataFrame.

    Default series: T10Y2Y, BAMLH0A0HYM2, DCOILWTICO, VIXCLS, UNRATE.
    Cached to output/edgar_cache/fred_macro.parquet (refresh after 7 days).
    NaNs forward-filled.
    """
    if series_ids is None:
        series_ids = ["T10Y2Y", "BAMLH0A0HYM2", "DCOILWTICO", "VIXCLS", "UNRATE"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "fred_macro.parquet"

    if _cache_is_fresh(cache_path, ttl_days=FRED_TTL_DAYS):
        try:
            df = pd.read_parquet(cache_path)
            logger.debug("FRED macro loaded from cache")
            return df
        except Exception as exc:
            logger.warning("FRED cache read failed: %s — re-fetching", exc)

    frames = {}
    for sid in series_ids:
        url = FRED_URL.format(sid=sid)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            from io import StringIO

            text = resp.text
            # Find the date column (may be 'DATE', 'date', etc.)
            raw_df = pd.read_csv(StringIO(text))
            date_col = None
            for col in raw_df.columns:
                if "date" in col.lower():
                    date_col = col
                    break
            if date_col is None:
                date_col = raw_df.columns[0]

            val_col = None
            for col in raw_df.columns:
                if col != date_col:
                    val_col = col
                    break
            if val_col is None:
                logger.warning("No value column found for FRED series %s", sid)
                continue

            raw_df[date_col] = pd.to_datetime(raw_df[date_col], errors="coerce")
            raw_df = raw_df.dropna(subset=[date_col])
            raw_df.set_index(date_col, inplace=True)
            raw_df.index.name = "date"

            series = pd.to_numeric(raw_df[val_col], errors="coerce")
            series.name = sid
            frames[sid] = series

            logger.debug("FRED %s: %d observations", sid, series.notna().sum())
        except Exception as exc:
            logger.warning("Failed to fetch FRED series %s: %s", sid, exc)

    if not frames:
        logger.error("No FRED series fetched successfully")
        return pd.DataFrame()

    combined = pd.concat(frames.values(), axis=1)
    combined.sort_index(inplace=True)
    combined.ffill(inplace=True)

    try:
        combined.to_parquet(cache_path)
    except Exception as exc:
        logger.warning("FRED cache write failed: %s", exc)

    return combined


# ---------------------------------------------------------------------------
# 10. load_insider_signals
# ---------------------------------------------------------------------------


def load_insider_signals(tickers: List[str]) -> pd.Series:
    """Fetch insider transaction signals from OpenInsider.

    Computes (buy_value - sell_value) / (buy_value + sell_value + 1e-8) per ticker
    for transactions in the last 90 days.
    Returns pd.Series(0.0, index=tickers) on any failure.
    """
    default = pd.Series(0.0, index=tickers)

    try:
        url = "http://openinsider.com/screener"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        html = resp.text
    except Exception as exc:
        logger.info("Insider data unavailable: %s", exc)
        return default

    try:
        # Attempt to parse HTML tables
        tables = pd.read_html(html)
        if not tables:
            return default

        # Use the largest table
        df = max(tables, key=lambda t: len(t))

        # Normalise column names
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Find ticker column
        ticker_col = None
        for col in df.columns:
            if "ticker" in col or "symbol" in col:
                ticker_col = col
                break
        if ticker_col is None:
            return default

        # Find transaction type column
        type_col = None
        for col in df.columns:
            if "type" in col or "transaction" in col:
                type_col = col
                break

        # Find value column
        val_col = None
        for col in df.columns:
            if "value" in col:
                val_col = col
                break

        # Find date column
        date_col = None
        for col in df.columns:
            if "date" in col and "filing" not in col:
                date_col = col
                break
        if date_col is None:
            for col in df.columns:
                if "date" in col:
                    date_col = col
                    break

        if type_col is None or val_col is None:
            return default

        df[ticker_col] = df[ticker_col].astype(str).str.upper().str.strip()

        # Parse dates and filter to last 90 days
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
            df = df[df[date_col] >= cutoff]

        if df.empty:
            return default

        # Parse values (strip $, commas, etc.)
        df["_val_clean"] = (
            df[val_col]
            .astype(str)
            .str.replace(r"[$,\s]", "", regex=True)
            .str.replace(r"\(.*\)", "", regex=True)
        )
        df["_val_num"] = pd.to_numeric(df["_val_clean"], errors="coerce").fillna(0.0)
        df["_type"] = df[type_col].astype(str).str.strip().str.upper()

        buy_val = (
            df[df["_type"] == "P"]
            .groupby(ticker_col)["_val_num"]
            .sum()
        )
        sell_val = (
            df[df["_type"] == "S"]
            .groupby(ticker_col)["_val_num"]
            .sum()
        )

        all_tickers_in_data = buy_val.index.union(sell_val.index)
        signals = {}
        for t in all_tickers_in_data:
            bv = float(buy_val.get(t, 0.0))
            sv = float(sell_val.get(t, 0.0))
            signals[t] = (bv - sv) / (bv + sv + 1e-8)

        result = default.copy()
        for t, v in signals.items():
            if t in result.index:
                result[t] = v
        return result

    except Exception as exc:
        logger.info("Insider signal parsing failed: %s", exc)
        return default
