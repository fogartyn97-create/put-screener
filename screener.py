"""
Put options screener — filters stocks by RSI, Bollinger Bands, IV, and IV Rank.
"""
from __future__ import annotations
import math
import time
import warnings
import yfinance as yf
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

_SP500 = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB","ARE","ALGN",
    "ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AAL","AEP","AXP","AIG","AMT","AWK",
    "AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","AAPL","AMAT","APTV","ACGL","ADM","ANET","AJG",
    "AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR","BALL","BAC","BK","BBWI","BAX","BDX",
    "BRK-B","BBY","BIO","TECH","BIIB","BLK","BX","BA","BCR","BSX","BMY","AVGO","BR","BRO","BF-B","BLDR",
    "BG","CDNS","CZR","CPT","CPB","COF","CAH","KMX","CCL","CARR","CTLT","CAT","CBOE","CBRE","CDW","CE",
    "COR","CNC","CNP","CF","CHRW","CRL","SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO",
    "C","CFG","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW",
    "CPAY","CTVA","CSGP","COST","CTRA","CRWD","CCI","CSX","CMI","CVS","DHR","DRI","DVA","DAY","DECK",
    "DE","DAL","DVN","DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW","DHI","DTE","DUK",
    "DD","EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV","LLY","EMR","ENPH","ETR","EOG","EPAM","EQT",
    "EFX","EQIX","EQR","ESS","EL","ETSY","EG","EVRG","ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS",
    "FICO","FAST","FRT","FDX","FIS","FITB","FSLR","FE","FI","FMC","F","FTNT","FTV","FOXA","FOX","BEN",
    "FCX","GRMN","IT","GE","GEHC","GEV","GEN","GNRC","GD","GIS","GM","GPC","GILD","GPN","GL","GDDY",
    "GS","HAL","HIG","HAS","HCA","DOC","HSIC","HSY","HES","HPE","HLT","HOLX","HD","HON","HRL","HST",
    "HWM","HPQ","HUBB","HUM","HBAN","HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD","INTC","ICE",
    "IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV","IRM","JBHT","JBL","JKHY","J","JNJ","JCI","JPM",
    "JNPR","K","KVUE","KDP","KEY","KEYS","KMB","KIM","KMI","KKR","KLAC","KHC","KR","LHX","LH","LRCX",
    "LW","LVS","LDOS","LEN","LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX",
    "MAR","MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT","MRK","META","MET","MTD","MGM","MCHP",
    "MU","MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI",
    "NDAQ","NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG",
    "NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PLTR",
    "PANW","PARA","PH","PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL",
    "PPG","PPL","PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL",
    "RJF","RTX","O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM",
    "SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM","SW","SNA","SOLV","SO","LUV","SWK","SBUX",
    "STT","STLD","STE","SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL",
    "TDY","TFX","TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL","TSN",
    "USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VLTO","VRSN","VRSK",
    "VZ","VRTX","VTRS","VICI","V","VST","VMC","WRB","GWW","WAB","WBA","WMT","DIS","WBD","WM","WAT",
    "WEC","WFC","WELL","WST","WDC","WY","WHR","WMB","WTW","WDAY","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS",
]


# Smaller optionable stocks $5+ with liquid options
_SMALL_CAP = [
    # Crypto miners (high IV, great for put selling)
    "MARA","RIOT","CLSK","HUT","IREN","CORZ",

    # EV / clean transport
    "LCID","RIVN","JOBY","XPEV","NIO","LI",

    # Fintech / neobank
    "SOFI","HOOD","OPEN","RKT","UWMC","PFSI","QFIN",

    # Airlines (cyclical, good IV)
    "AAL","JBLU","HA","SKYW",

    # Clean energy
    "PLUG","BE","NOVA","ARRY",

    # Cannabis (high IV)
    "TLRY","CGC","ACB","CRON",

    # Telecom / legacy tech
    "NOK","BB","ERIC","LUMN","DISH","SIRI",

    # Metals & mining
    "VALE","HL","EGO","KGC","BTG","RIG","BORR","PAAS","NG","CDE",

    # Mortgage REITs (high yield, liquid options)
    "AGNC","NLY","TWO","RITM","ORC","ARR","RC",

    # Media / entertainment
    "WBD","PARA","AMC",

    # Biotech / pharma small cap (high IV)
    "NVAX","MNKD","HIMS","ACMR","BEAM","EDIT","FATE","NKTR","ARCT","IMVT","PRTA",

    # AI / tech small cap
    "IONQ","AI","SOUN","SNAP","BMBL",

    # Consumer
    "GDRX","GRAB",

    # Energy small cap
    "CRK","SWN","RRC","SM","VTLE","GPOR",
]


def get_sp500_tickers() -> list[str]:
    return list(dict.fromkeys(_SP500 + _SMALL_CAP))  # combined, no duplicates


def compute_rsi(closes: pd.Series, period: int = 14) -> float:
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def compute_bollinger(closes: pd.Series, period: int = 20, num_std: float = 2.0):
    sma = closes.rolling(period).mean()
    std = closes.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    last_close = closes.iloc[-1]
    last_lower = lower.iloc[-1]
    last_upper = upper.iloc[-1]
    last_sma = sma.iloc[-1]
    # % distance from lower band: negative means below lower band
    pct_b = (last_close - last_lower) / (last_upper - last_lower) if (last_upper - last_lower) != 0 else 0.5
    return round(last_lower, 2), round(last_upper, 2), round(last_sma, 2), round(pct_b, 4)


def compute_hv(closes: pd.Series, period: int = 30) -> float:
    """Annualized historical volatility over `period` trading days."""
    log_ret = np.log(closes / closes.shift(1)).dropna()
    if len(log_ret) < period:
        return float("nan")
    hv = log_ret.iloc[-period:].std() * math.sqrt(252)
    return round(hv * 100, 2)


def compute_iv_rank(closes: pd.Series, current_hv: float) -> float:
    """
    Approximate IV Rank using rolling 30-day HV over the past year.
    Real IV Rank requires historical IV data, which yfinance doesn't provide.
    This uses HV as a proxy — high HV rank ≈ elevated vol environment.
    """
    log_ret = np.log(closes / closes.shift(1)).dropna()
    if len(log_ret) < 252 + 30:
        return float("nan")
    hvs = [
        log_ret.iloc[i : i + 30].std() * math.sqrt(252) * 100
        for i in range(len(log_ret) - 30)
    ]
    hvs = hvs[-252:]  # last year
    if not hvs:
        return float("nan")
    hv_min, hv_max = min(hvs), max(hvs)
    if hv_max == hv_min:
        return 50.0
    rank = (current_hv - hv_min) / (hv_max - hv_min) * 100
    return round(rank, 1)


def get_atm_iv(ticker_obj, current_price: float) -> float | None:
    """Get ATM implied volatility from nearest expiry with 20-60 DTE."""
    try:
        exps = ticker_obj.options
        if not exps:
            return None
        today = pd.Timestamp.today()
        # Find expiry with 20-60 DTE
        target_exp = None
        for exp in exps:
            dte = (pd.Timestamp(exp) - today).days
            if 20 <= dte <= 60:
                target_exp = exp
                break
        if not target_exp:
            target_exp = exps[0]

        chain = ticker_obj.option_chain(target_exp)
        puts = chain.puts[["strike", "impliedVolatility", "openInterest"]].dropna()
        if puts.empty:
            return None
        # Nearest strike to current price
        puts = puts[puts["openInterest"] > 0]
        if puts.empty:
            return None
        idx = (puts["strike"] - current_price).abs().idxmin()
        iv = puts.loc[idx, "impliedVolatility"]
        return round(iv * 100, 2) if iv > 0 else None
    except Exception:
        return None


def _fetch_history(ticker: str, retries: int = 3) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="2y", auto_adjust=True)
            if not hist.empty:
                return hist, tk
        except Exception:
            pass
        time.sleep(2 + attempt * 2)
    return pd.DataFrame(), None


def screen_ticker(ticker: str, rsi_threshold: float, pct_b_threshold: float, iv_threshold: float, ivr_threshold: float, price_min: float = 0, price_max: float = 0) -> dict | None:
    try:
        hist, tk = _fetch_history(ticker)
        if hist.empty or len(hist) < 60:
            return None

        closes = hist["Close"]
        current_price = round(closes.iloc[-1], 2)

        rsi = compute_rsi(closes)
        lower_bb, upper_bb, mid_bb, pct_b = compute_bollinger(closes)
        hv30 = compute_hv(closes)
        ivr = compute_iv_rank(closes, hv30)
        time.sleep(0.3)  # small pause between price and options fetch
        iv = get_atm_iv(tk, current_price)

        # Apply filters
        if price_min > 0 and current_price < price_min:
            return None
        if price_max > 0 and current_price > price_max:
            return None
        if rsi > rsi_threshold:
            return None
        if pct_b > pct_b_threshold:
            return None
        if iv is not None and iv < iv_threshold:
            return None
        if not math.isnan(ivr) and ivr < ivr_threshold:
            return None

        return {
            "ticker": ticker,
            "price": current_price,
            "rsi": rsi,
            "lower_bb": lower_bb,
            "upper_bb": upper_bb,
            "pct_b": round(pct_b * 100, 1),
            "hv30": hv30,
            "iv_atm": iv,
            "iv_rank": ivr if not math.isnan(ivr) else None,
        }
    except Exception:
        return None


def run_screen(
    tickers: list[str] | None = None,
    rsi_threshold: float = 40,
    pct_b_threshold: float = 0.35,
    iv_threshold: float = 20,
    ivr_threshold: float = 40,
    price_min: float = 0,
    price_max: float = 0,
    max_workers: int = 8,
    progress_callback=None,
) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if tickers is None:
        tickers = get_sp500_tickers()

    results = []
    completed = 0
    total = len(tickers)

    def worker(t):
        return screen_ticker(t, rsi_threshold, pct_b_threshold, iv_threshold, ivr_threshold, price_min, price_max)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, t): t for t in tickers}
        for fut in as_completed(futures):
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
            res = fut.result()
            if res:
                results.append(res)

    results.sort(key=lambda x: (x["rsi"], x["pct_b"]))
    return results
