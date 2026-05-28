"""
Global universe builder: S&P 500 + major European + Asian + EM ADRs/tickers.
All tickers are USD-denominated (US listings or ADRs) for consistent currency handling.
"""
import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# US universe: S&P 500
# ---------------------------------------------------------------------------
SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
SP500_FALLBACK = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"

# ---------------------------------------------------------------------------
# European companies: large-cap ADRs / US-listed
# ---------------------------------------------------------------------------
EU_TICKERS = {
    # Healthcare & Pharma
    "NVS": "Healthcare",   # Novartis (CH)
    "NVO": "Healthcare",   # Novo Nordisk (DK)
    "AZN": "Healthcare",   # AstraZeneca (UK)
    "GSK": "Healthcare",   # GSK (UK)
    "SNY": "Healthcare",   # Sanofi (FR)
    "RHHBY": "Healthcare", # Roche (CH)
    "BNTX": "Healthcare",  # BioNTech (DE)
    # Technology
    "ASML": "Technology",  # ASML (NL)
    "SAP": "Technology",   # SAP (DE)
    "ERIC": "Technology",  # Ericsson (SE)
    "NOK": "Technology",   # Nokia (FI)
    "STM": "Technology",   # STMicroelectronics (FR/IT)
    "INFY": "Technology",  # Infosys (IN) — listed US
    "WIT": "Technology",   # Wipro (IN)
    # Energy
    "BP": "Energy",        # BP (UK)
    "SHEL": "Energy",      # Shell (UK/NL)
    "TTE": "Energy",       # TotalEnergies (FR)
    "E": "Energy",         # Eni (IT)
    "EQNR": "Energy",      # Equinor (NO)
    # Financials
    "HSBC": "Financials",  # HSBC (UK)
    "ING": "Financials",   # ING Groep (NL)
    "BBVA": "Financials",  # BBVA (ES)
    "SAN": "Financials",   # Santander (ES)
    "BCS": "Financials",   # Barclays (UK)
    "LYG": "Financials",   # Lloyds (UK)
    "DB": "Financials",    # Deutsche Bank (DE)
    "UBS": "Financials",   # UBS (CH)
    "CS": "Financials",    # Credit Suisse (now UBS) — may be delisted
    "AXA": "Financials",   # AXA (FR) — may trade as ADR
    # Consumer
    "UL": "Consumer Staples",    # Unilever (UK/NL)
    "NSRGY": "Consumer Staples", # Nestle (CH) OTC
    "LRLCY": "Consumer Staples", # L'Oreal (FR) OTC
    "DEO": "Consumer Staples",   # Diageo (UK)
    "BTI": "Consumer Staples",   # BAT (UK)
    "PM": "Consumer Staples",    # Philip Morris (US, ex-Altria intl)
    # Auto & Industrials
    "STLA": "Consumer Discretionary", # Stellantis (NL/FR/IT)
    "VWAGY": "Consumer Discretionary",# VW (DE) OTC
    "BMWYY": "Consumer Discretionary",# BMW (DE) OTC
    "MBGYY": "Consumer Discretionary",# Mercedes (DE) OTC
    "EADSY": "Industrials",      # Airbus (FR/DE/ES) OTC
    "ABB": "Industrials",        # ABB (CH)
    "SIEGY": "Industrials",      # Siemens (DE) OTC
    "ALIZF": "Financials",       # Allianz (DE) OTC
    # Mining & Materials
    "RIO": "Materials",   # Rio Tinto (UK/AU)
    "BHP": "Materials",   # BHP (AU/UK)
    "VALE": "Materials",  # Vale (BR)
    "NEM": "Materials",   # Newmont (US but global miner)
    # Luxury
    "LVMUY": "Consumer Discretionary", # LVMH (FR) OTC
    "CFRUY": "Consumer Discretionary", # Richemont (CH) OTC
}

# ---------------------------------------------------------------------------
# Asian companies: large-cap ADRs / US-listed
# ---------------------------------------------------------------------------
ASIA_TICKERS = {
    # Japan
    "TM":    "Consumer Discretionary", # Toyota
    "HMC":   "Consumer Discretionary", # Honda
    "SONY":  "Technology",             # Sony
    "NTDOY": "Technology",             # Nintendo OTC
    "SFTBY": "Technology",             # SoftBank OTC
    "MUFG":  "Financials",             # Mitsubishi UFJ
    "SMFG":  "Financials",             # Sumitomo Mitsui
    "NMR":   "Financials",             # Nomura
    "MFG":   "Financials",             # Mizuho
    "FANUY": "Industrials",            # Fanuc OTC
    "KYCCF": "Industrials",            # Kyocera OTC (some data issues)
    "TOELY": "Industrials",            # Toyota Industries OTC
    # Korea
    "KB":    "Financials",             # KB Financial
    "SHG":   "Financials",             # Shinhan Financial
    "PKX":   "Materials",              # POSCO (steel)
    "LPL":   "Technology",             # LG Display
    # China / HK
    "BABA":  "Technology",             # Alibaba
    "BIDU":  "Technology",             # Baidu
    "JD":    "Consumer Discretionary", # JD.com
    "PDD":   "Consumer Discretionary", # PDD Holdings
    "TCEHY": "Technology",             # Tencent OTC
    "NTES":  "Technology",             # NetEase
    "TCOM":  "Consumer Discretionary", # Trip.com
    "YUMC":  "Consumer Discretionary", # Yum China
    "ZTO":   "Industrials",            # ZTO Express
    "NIO":   "Consumer Discretionary", # NIO (EV)
    "LI":    "Consumer Discretionary", # Li Auto (EV)
    # Taiwan
    "TSM":   "Technology",  # TSMC
    "UMC":   "Technology",  # United Microelectronics
    # India
    "HDB":   "Financials",  # HDFC Bank
    "IBN":   "Financials",  # ICICI Bank
    "TTM":   "Consumer Discretionary", # Tata Motors
    "WIT":   "Technology",  # Wipro (duplicate, will dedup)
    # Australia
    "BHP":   "Materials",   # BHP (duplicate, will dedup)
    # Singapore / SEA
    "SE":    "Technology",  # Sea Limited (gaming/ecommerce)
    "GRAB":  "Technology",  # Grab Holdings
    # Brazil / LatAm
    "PBR":   "Energy",      # Petrobras (BR)
    "ITUB":  "Financials",  # Itaú Unibanco (BR)
    "BBD":   "Financials",  # Bradesco (BR)
    "ABEV":  "Consumer Staples", # Ambev (BR)
    "CX":    "Materials",   # Cemex (MX)
    "AMX":   "Communication Services", # América Móvil (MX)
}


def build_global_universe() -> tuple[list[str], dict[str, str]]:
    """
    Returns (tickers, sector_map) for the full global universe.
    US S&P 500 + EU ADRs + Asia ADRs, deduplicated.
    """
    sector_map: dict[str, str] = {}

    # --- S&P 500 ---
    sp500_tickers = []
    try:
        df = pd.read_csv(SP500_URL)
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        sec_col = "GICS Sector" if "GICS Sector" in df.columns else (
            "Sector" if "Sector" in df.columns else None
        )
        sp500_tickers = df[col].str.strip().str.replace(".", "-", regex=False).tolist()
        if sec_col:
            for _, row in df.iterrows():
                t = str(row[col]).strip().replace(".", "-")
                sector_map[t] = str(row[sec_col])
        logger.info("S&P 500: %d tickers", len(sp500_tickers))
    except Exception as e:
        logger.warning("S&P 500 load failed (%s), trying fallback", e)
        try:
            df = pd.read_csv(SP500_FALLBACK)
            col = df.columns[0]
            sp500_tickers = df[col].str.strip().tolist()
        except Exception as e2:
            logger.error("S&P 500 fallback failed: %s", e2)

    # --- International ---
    for t, sec in EU_TICKERS.items():
        sector_map[t] = sec
    for t, sec in ASIA_TICKERS.items():
        sector_map[t] = sec

    intl_tickers = list({**EU_TICKERS, **ASIA_TICKERS}.keys())

    # --- Combine & deduplicate ---
    seen = set()
    all_tickers = []
    for t in sp500_tickers + intl_tickers:
        if t not in seen:
            seen.add(t)
            all_tickers.append(t)
            if t not in sector_map:
                sector_map[t] = "Unknown"

    logger.info("Global universe: %d tickers (%d US + %d intl)",
                len(all_tickers), len(sp500_tickers), len(intl_tickers))
    return all_tickers, sector_map
