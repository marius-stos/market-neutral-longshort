"""
alternative_data.py
-------------------
Alternative-data sentiment module for the quant long/short strategy.

Fetches and blends sentiment signals from:
  - Yahoo Finance RSS feeds
  - AlphaVantage News Sentiment API
  - Reddit (placeholder — requires further integration)
  - Google Trends (via pytrends)
  - CNN Fear & Greed index

Auto-loads .env from the same directory at import time.

Python 3.9 compatible.
"""

import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# .env loader (runs at import time)
# ---------------------------------------------------------------------------

_ENV_FILE = Path(__file__).parent / ".env"

if _ENV_FILE.exists():
    try:
        for _line in _ENV_FILE.read_text().splitlines():
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _val = _line.split("=", 1)
            os.environ.setdefault(_key.strip(), _val.strip())
        logger.debug("Loaded .env from %s", _ENV_FILE)
    except Exception as _exc:  # noqa: BLE001
        logger.warning("Could not parse .env: %s", _exc)

# ---------------------------------------------------------------------------
# Global config (populated from env vars at runtime)
# ---------------------------------------------------------------------------

ALT_CONFIG: Dict[str, str] = {
    "newsapi_key":          os.environ.get("NEWSAPI_KEY", ""),
    "reddit_client_id":     os.environ.get("REDDIT_CLIENT_ID", ""),
    "reddit_client_secret": os.environ.get("REDDIT_SECRET", ""),
    "alphavantage_key":     os.environ.get("ALPHAVANTAGE_KEY", ""),
}

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    _vader = SentimentIntensityAnalyzer()
    _VADER_OK = True
except ImportError:
    logger.warning("vaderSentiment not installed; Yahoo RSS sentiment will return 0.0.")
    _vader = None
    _VADER_OK = False

try:
    from pytrends.request import TrendReq  # type: ignore
    _PYTRENDS_OK = True
except ImportError:
    logger.warning("pytrends not installed; Google Trends sentiment will be skipped.")
    _PYTRENDS_OK = False

OUTPUT_DIR = Path("output")
ALT_CACHE_FILE = OUTPUT_DIR / "alt_sentiment_cache.json"
ALT_CACHE_TTL_HOURS = 6

# ---------------------------------------------------------------------------
# 1. Yahoo Finance RSS sentiment
# ---------------------------------------------------------------------------


def fetch_yahoo_rss_batch(tickers: List[str], max_workers: int = 8) -> Dict[str, float]:
    """
    Fetch Yahoo Finance RSS headlines and score them with VADER.

    Returns {ticker: mean_compound_score} in [-1, 1].
    """
    if not _VADER_OK:
        return {}

    def _fetch_one(ticker: str) -> tuple:
        url = (
            f"https://feeds.finance.yahoo.com/rss/2.0/headline"
            f"?s={ticker}&region=US&lang=en-US"
        )
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            titles = [item.findtext("title") or "" for item in root.iter("item")]
            if not titles:
                return ticker, 0.0
            scores = [_vader.polarity_scores(t)["compound"] for t in titles]  # type: ignore[union-attr]
            return ticker, float(np.mean(scores))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Yahoo RSS failed for %s: %s", ticker, exc)
            return ticker, None

    results: Dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, tk): tk for tk in tickers}
        for fut in as_completed(futures):
            tk, score = fut.result()
            if score is not None:
                results[tk] = score

    logger.info("Yahoo RSS: scored %d / %d tickers.", len(results), len(tickers))
    return results


# ---------------------------------------------------------------------------
# 2. AlphaVantage News Sentiment
# ---------------------------------------------------------------------------

_AV_TOPICS = ["earnings", "financial_markets", "economy_macro"]
_AV_BASE = "https://www.alphavantage.co/query"


def fetch_alphavantage_sentiment(tickers: List[str], limit: int = 40) -> Dict[str, float]:
    """
    Pull news sentiment from AlphaVantage (free tier: 5 req/min).

    Uses topics= parameter (not tickers=) to avoid per-ticker rate limits.
    Returns {ticker: mean_ticker_sentiment_score} in [-1, 1].
    """
    api_key = ALT_CONFIG.get("alphavantage_key") or os.environ.get("ALPHAVANTAGE_KEY", "")
    if not api_key:
        logger.warning("ALPHAVANTAGE_KEY not set; skipping AlphaVantage sentiment.")
        return {}

    ticker_scores: Dict[str, List[float]] = {tk: [] for tk in tickers}
    ticker_set = set(tickers)

    for i, topic in enumerate(_AV_TOPICS):
        if i > 0:
            time.sleep(12)  # free tier: 5 req/min

        params = {
            "function": "NEWS_SENTIMENT",
            "topics": topic,
            "limit": limit,
            "apikey": api_key,
        }
        try:
            resp = requests.get(_AV_BASE, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("AlphaVantage request failed (topic=%s): %s", topic, exc)
            continue

        for article in data.get("feed", []):
            for ts_entry in article.get("ticker_sentiment", []):
                symbol = ts_entry.get("ticker", "")
                if symbol not in ticker_set:
                    continue
                try:
                    relevance = float(ts_entry.get("relevance_score", 0))
                    sentiment = float(ts_entry.get("ticker_sentiment_score", 0))
                except ValueError:
                    continue
                if relevance >= 0.10:
                    ticker_scores[symbol].append(sentiment)

        logger.debug("AlphaVantage topic '%s' processed.", topic)

    result = {
        tk: float(np.mean(vals))
        for tk, vals in ticker_scores.items()
        if vals
    }
    logger.info("AlphaVantage: scored %d / %d tickers.", len(result), len(tickers))
    return result


# ---------------------------------------------------------------------------
# 3. Google Trends
# ---------------------------------------------------------------------------


def fetch_google_trends(tickers: List[str], timeframe: str = "today 3-m") -> Dict[str, float]:
    """
    Fetch relative search interest via pytrends.

    Score = (last_2w_mean / historical_mean) - 1, clipped to [-1, 1].
    Returns {} on any failure or if pytrends is unavailable.
    """
    if not _PYTRENDS_OK:
        return {}

    tickers_limited = tickers[:60]
    results: Dict[str, float] = {}

    try:
        pt = TrendReq(hl="en-US", tz=360, timeout=(5, 30))
        batch_size = 5

        for start in range(0, len(tickers_limited), batch_size):
            batch = tickers_limited[start: start + batch_size]
            try:
                pt.build_payload(batch, timeframe=timeframe)
                df = pt.interest_over_time()
                if df.empty:
                    continue
                for tk in batch:
                    if tk not in df.columns:
                        continue
                    series = df[tk].astype(float)
                    hist_mean = series.mean()
                    if hist_mean == 0:
                        continue
                    last_2w = series.iloc[-14:].mean() if len(series) >= 14 else series.mean()
                    score = float(np.clip((last_2w / hist_mean) - 1, -1, 1))
                    results[tk] = score
            except Exception as exc:  # noqa: BLE001
                logger.debug("Google Trends batch failed %s: %s", batch, exc)

            if start + batch_size < len(tickers_limited):
                time.sleep(1)

    except Exception as exc:  # noqa: BLE001
        logger.warning("Google Trends fetch failed entirely: %s", exc)
        return {}

    logger.info("Google Trends: scored %d / %d tickers.", len(results), len(tickers_limited))
    return results


# ---------------------------------------------------------------------------
# 4. CNN Fear & Greed
# ---------------------------------------------------------------------------

_FNG_URL = "https://api.alternative.me/fng/"


def fetch_fear_greed() -> dict:
    """
    Fetch the CNN Fear & Greed index.

    Returns {"fear_greed_score": float 0-100, "fear_greed_mult": float 0.3-1.5}.
    """
    try:
        resp = requests.get(_FNG_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        score = float(data["data"][0]["value"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fear & Greed fetch failed: %s; defaulting to 50.", exc)
        score = 50.0

    mult = 0.3 + 1.2 * (score / 100.0)
    mult = float(np.clip(mult, 0.3, 1.5))
    logger.debug("Fear & Greed score=%.1f, mult=%.3f", score, mult)
    return {"fear_greed_score": score, "fear_greed_mult": mult}


# ---------------------------------------------------------------------------
# 5. Composite alt sentiment loader
# ---------------------------------------------------------------------------

_WEIGHTS = {
    "yahoo":         0.35,
    "alphavantage":  0.25,
    "reddit":        0.20,
    "google_trends": 0.20,
}


def load_alt_sentiment(tickers: List[str], force_refresh: bool = False) -> pd.Series:
    """
    Load blended alternative sentiment for *tickers*.

    Results are cached for ALT_CACHE_TTL_HOURS hours.
    Returns a pd.Series indexed by ticker with values in [-1, 1].
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- cache check ---
    if not force_refresh and ALT_CACHE_FILE.exists():
        try:
            cached = json.loads(ALT_CACHE_FILE.read_text())
            ts = cached.get("_ts", 0)
            age_h = (datetime.now(tz=timezone.utc).timestamp() - ts) / 3600
            if age_h < ALT_CACHE_TTL_HOURS:
                scores = cached.get("data", {})
                result = pd.Series({tk: float(scores.get(tk, 0.0)) for tk in tickers})
                logger.info("Alt sentiment cache hit (age=%.1fh).", age_h)
                return result.clip(-1, 1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Alt sentiment cache read failed: %s", exc)

    # --- fetch all sources ---
    yahoo_scores = fetch_yahoo_rss_batch(tickers)
    av_scores = fetch_alphavantage_sentiment(tickers)
    reddit_scores: Dict[str, float] = {tk: 0.0 for tk in tickers}  # placeholder
    gt_scores = fetch_google_trends(tickers)
    fng = fetch_fear_greed()

    # --- blend ---
    source_data = {
        "yahoo":         yahoo_scores,
        "alphavantage":  av_scores,
        "reddit":        reddit_scores,
        "google_trends": gt_scores,
    }

    blended: Dict[str, float] = {}
    for tk in tickers:
        total_weight = 0.0
        total_score = 0.0
        for src, weight in _WEIGHTS.items():
            val = source_data[src].get(tk)
            if val is not None:
                total_score += weight * val
                total_weight += weight
        raw = (total_score / total_weight) if total_weight > 0 else 0.0
        blended[tk] = float(np.clip(raw * fng["fear_greed_mult"], -1, 1))

    # --- cache result ---
    try:
        ALT_CACHE_FILE.write_text(
            json.dumps(
                {"_ts": datetime.now(tz=timezone.utc).timestamp(), "data": blended},
                indent=2,
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write alt sentiment cache: %s", exc)

    logger.info("Alt sentiment computed for %d tickers (fng=%.1f).", len(blended), fng["fear_greed_score"])
    return pd.Series(blended).clip(-1, 1)


# ---------------------------------------------------------------------------
# 6. Blend alt sentiment into signals
# ---------------------------------------------------------------------------


def blend_alt_sentiment(
    signals: pd.DataFrame,
    alt_scores: pd.Series,
    weight: float = 0.10,
) -> pd.DataFrame:
    """
    Blend alternative sentiment into the signals composite score.

    sig["composite"] = (1 - weight) * sig["composite"] + weight * alt_rank

    *alt_rank* is the percentile rank of *alt_scores* across all tickers.
    Returns the modified signals DataFrame.
    """
    if "composite" not in signals.columns:
        logger.warning("'composite' column missing from signals; skipping alt blend.")
        return signals

    sig = signals.copy()
    alt_rank = alt_scores.rank(pct=True)

    common = sig.index.intersection(alt_rank.index)
    if common.empty:
        logger.warning("No overlapping tickers between signals and alt_scores.")
        return signals

    sig.loc[common, "composite"] = (
        (1 - weight) * sig.loc[common, "composite"]
        + weight * alt_rank.loc[common]
    )

    logger.info(
        "Alt sentiment blended (weight=%.0f%%) into composite for %d tickers.",
        weight * 100, len(common),
    )
    return sig
