
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import os
import math
import time as pytime
import asyncio
import numpy as np
import pandas as pd
import requests

# ---------------- Small utils ----------------
def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _round_to_tick(x: float, tick: Optional[float], mode: str = "nearest") -> float:
    if not tick or tick <= 0:
        return float(x)
    n = x / tick
    if mode == "floor":
        n = math.floor(n)
    elif mode == "ceil":
        n = math.ceil(n)
    else:
        n = round(n)
    return float(n * tick)

def _anchor_day_from_ts(ts: pd.Timestamp) -> datetime:
    base = ts.to_pydatetime().astimezone(timezone.utc)
    return base.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

# ---------------- Heikin-Ashi ----------------
def _heikin_ashi_df(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index)
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    hac = (o + h + l + c) / 4.0
    hao = np.zeros_like(hac)
    if len(hac) > 0:
        hao[0] = (o[0] + c[0]) / 2.0
        for i in range(1, len(hac)):
            hao[i] = (hao[i - 1] + hac[i - 1]) / 2.0
    hah = np.maximum.reduce([h, hao, hac])
    hal = np.minimum.reduce([l, hao, hac])
    ha["open"] = hao
    ha["high"] = hah
    ha["low"] = hal
    ha["close"] = hac
    return ha

# ---------------- RVOL / TR / CHOP / FDI ----------------
def _rvol(series_volume: pd.Series, lookback: int = 96) -> float:
    if len(series_volume) < lookback + 2:
        return 1.0
    med = float(series_volume.tail(lookback).median() + 1e-9)
    return float(series_volume.iloc[-1] / med if med > 0 else 1.0)

def _combined_rvol(vol5: pd.Series, vol15: pd.Series, lk5: int = 96, lk15: int = 96) -> float:
    r5 = _rvol(vol5, lk5)
    r15 = _rvol(vol15, lk15)
    return float(0.4 * r5 + 0.6 * r15)

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr

def _choppiness_index(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if len(df) < period + 2:
        return None
    high, low, close = df["high"], df["low"], df["close"]
    tr = _true_range(high, low, close)
    sum_tr = float(tr.tail(period).sum())
    hh = float(high.tail(period).max())
    ll = float(low.tail(period).min())
    denom = (hh - ll) + 1e-12
    if sum_tr <= 0 or denom <= 0:
        return None
    chop = 100.0 * math.log10(sum_tr / denom) / math.log10(period)
    return float(max(0.0, min(100.0, chop)))

def _fractal_dimension_index(close: pd.Series, window: int = 100) -> Optional[float]:
    if len(close) < window + 5:
        return None
    y = close.tail(window).values.astype(float)
    L = np.sum(np.abs(np.diff(y))) + 1e-12
    d = np.abs(y - y[0])
    d_max = float(np.max(d) + 1e-12)
    n = float(window)
    fdi = (math.log10(n) / (math.log10(n) + math.log10(d_max / L)))
    return float(min(2.0, max(1.0, fdi)))

# ---------------- Fractals / ZigZag / Channel ----------------
def _bw_fractals(df: pd.DataFrame) -> Tuple[List[int], List[int]]:
    highs, lows = [], []
    h = df["high"].values
    l = df["low"].values
    n = len(df)
    for i in range(2, n - 2):
        if h[i] > h[i - 1] and h[i] > h[i - 2] and h[i] > h[i + 1] and h[i] > h[i + 2]:
            highs.append(i)
        if l[i] < l[i - 1] and l[i] < l[i - 2] and l[i] < l[i + 1] and l[i] < l[i + 2]:
            lows.append(i)
    return highs, lows

def _zigzag_idx(close: pd.Series, pct: float = 1.0) -> List[int]:
    if len(close) < 10:
        return []
    th = abs(pct) / 100.0
    idxs = []
    direction = 0
    last_pivot_val = float(close.iloc[0])
    for i in range(1, len(close)):
        c = float(close.iloc[i])
        change = (c - last_pivot_val) / (last_pivot_val + 1e-12)
        if direction >= 0 and change >= th:
            idxs.append(i)
            last_pivot_val = c
            direction = -1
        elif direction <= 0 and change <= -th:
            idxs.append(i)
            last_pivot_val = c
            direction = 1
        else:
            if direction >= 0 and c > last_pivot_val:
                last_pivot_val = c
                if idxs:
                    idxs[-1] = i
            elif direction <= 0 and c < last_pivot_val:
                last_pivot_val = c
                if idxs:
                    idxs[-1] = i
    return sorted(set(idxs))

def _regression_channel(y: pd.Series, window: int = 120) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    if len(y) < window + 2:
        return None, None, None, None, None
    ys = y.tail(window).values.astype(float)
    xs = np.arange(len(ys)).astype(float)
    a, b = np.polyfit(xs, ys, 1)
    pred = a * xs + b
    resid = ys - pred
    sd = np.std(resid) + 1e-12
    last_x = xs[-1]
    mid = a * last_x + b
    return float(a), float(b), float(mid - 2 * sd), float(mid), float(mid + 2 * sd)

def _channel_confluence(price: float, lo: float, mid: float, hi: float, side: str) -> float:
    if any(v is None or not np.isfinite(v) for v in [price, lo, mid, hi]):
        return 0.0
    if side == "LONG":
        if price <= lo:
            return 0.25
        if price >= hi:
            return -0.25
    else:
        if price >= hi:
            return 0.25
        if price <= lo:
            return -0.25
    return 0.0

# ---------------- SMC: FVG / OB / EQH-EQL / SFP / QM / BOS+displacement ----------------
def _find_fvg(df: pd.DataFrame, lookback: int = 100) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    if df is None or len(df) < 3:
        return out
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    for i in range(1, min(lookback, len(df) - 1)):
        j = len(df) - 1 - i
        if j - 1 >= 0 and j + 1 < len(df):
            if l[j + 1] > h[j - 1]:
                out.append({"type": 1, "upper": l[j + 1], "lower": h[j - 1], "bar": j})
            if h[j + 1] < l[j - 1]:
                out.append({"type": -1, "upper": l[j - 1], "lower": h[j + 1], "bar": j})
    return out

def _fvg_status(fvgs: List[Dict[str, float]], price: float) -> Tuple[Optional[Dict[str, float]], float]:
    if not fvgs:
        return None, float("inf")
    best = None
    best_dist = float("inf")
    for z in fvgs:
        if z["type"] == 1:
            if price <= z["lower"]:
                d = abs(z["lower"] - price)
            elif price >= z["upper"]:
                d = abs(price - z["upper"])
            else:
                d = 0.0
        else:
            if price <= z["lower"]:
                d = abs(z["lower"] - price)
            elif price >= z["upper"]:
                d = abs(price - z["upper"])
            else:
                d = 0.0
        if d < best_dist:
            best = z
            best_dist = d
    return best, float(best_dist)

def _displacement_bar(df: pd.DataFrame, atr_fn, atr_period: int = 14, k: float = 1.4) -> bool:
    if df is None or len(df) < atr_period + 2:
        return False
    atrv = float(atr_fn(df, atr_period).iloc[-1])
    rng = float(df["high"].iloc[-1] - df["low"].iloc[-1])
    body = abs(float(df["close"].iloc[-1] - df["open"].iloc[-1]))
    return bool(rng > k * atrv and body > 0.6 * rng)

def _find_order_block(df: pd.DataFrame, side: str, atr_fn, atr_period: int = 14) -> Optional[Dict[str, float]]:
    if df is None or len(df) < atr_period + 10:
        return None
    atrv = float(atr_fn(df, atr_period).iloc[-1])
    med_body = float((df["close"] - df["open"]).abs().tail(50).median())
    for i in range(len(df) - 3, atr_period, -1):
        o = float(df["open"].iloc[i]); c = float(df["close"].iloc[i])
        h = float(df["high"].iloc[i]); l = float(df["low"].iloc[i])
        body = abs(c - o); rng = h - l
        strong = (rng > 1.2 * atrv) and (body > 0.8 * rng) and (body > 1.2 * med_body)
        if not strong:
            continue
        if side == "LONG" and c > o:
            j = i - 1
            while j >= max(0, i - 8):
                o2 = float(df["open"].iloc[j]); c2 = float(df["close"].iloc[j])
                if c2 < o2:
                    lo = min(o2, c2); hi = max(o2, c2)
                    return {"type": 1, "low": lo, "high": hi, "mid": (lo + hi) / 2.0, "bar": j}
                j -= 1
        if side == "SHORT" and c < o:
            j = i - 1
            while j >= max(0, i - 8):
                o2 = float(df["open"].iloc[j]); c2 = float(df["close"].iloc[j])
                if c2 > o2:
                    lo = min(o2, c2); hi = max(o2, c2)
                    return {"type": -1, "low": lo, "high": hi, "mid": (lo + hi) / 2.0, "bar": j}
                j -= 1
    return None

def _equal_highs_lows(df: pd.DataFrame, lookback: int = 50, tol_frac: float = 0.0008) -> Tuple[bool, bool]:
    if df is None or len(df) < lookback + 2:
        return False, False
    h = df["high"].tail(lookback).values.astype(float)
    l = df["low"].tail(lookback).values.astype(float)
    hi = np.max(h); lo = np.min(l)
    eqh = np.sum(np.abs(h - hi) <= tol_frac * hi) >= 2
    eql = np.sum(np.abs(l - lo) <= tol_frac * lo) >= 2
    return bool(eqh), bool(eql)

def _sfp_2b(df: pd.DataFrame, side: str, lookback: int = 30) -> bool:
    if df is None or len(df) < lookback + 3:
        return False
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    H = np.max(h[-(lookback + 1):-1])
    L = np.min(l[-(lookback + 1):-1])
    if side == "SHORT":
        return bool(h[-1] > H and c[-1] < H)
    else:
        return bool(l[-1] < L and c[-1] > L)

def _quasimodo_zigzag(close: pd.Series, pct: float = 1.0) -> bool:
    idxs = _zigzag_idx(close, pct=pct)
    if len(idxs) < 5:
        return False
    piv = close.iloc[idxs[-5:]].values.astype(float)
    cond_short = (piv[1] < piv[0] and piv[2] > piv[0] and piv[3] < piv[2])
    cond_long = (piv[1] > piv[0] and piv[2] < piv[0] and piv[3] > piv[2])
    return bool(cond_short or cond_long)

def _bos_displacement(df: pd.DataFrame, side: str, atr_fn, atr_p: int = 14, lookback: int = 30, k: float = 1.2) -> bool:
    if len(df) < lookback + 5:
        return False
    atrv = float(atr_fn(df, atr_p).iloc[-1])
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    H = np.max(h[-(lookback + 1):-1])
    L = np.min(l[-(lookback + 1):-1])
    rng = h[-1] - l[-1]
    body = abs(c[-1] - float(df["open"].iloc[-1]))
    strong = (rng > k * atrv and body > 0.6 * rng)
    if not strong:
        return False
    return bool(c[-1] > H) if side == "LONG" else bool(c[-1] < L)

# ---------------- VWAP bands / IB / NR / Gaps ----------------
def _vwap_sigma_bands(df15: pd.DataFrame, lookback: int = 96, anchored_vwap=None) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if df15 is None or df15.empty or anchored_vwap is None:
        return None, None, None
    vwap_series = anchored_vwap(df15)
    if vwap_series is None or len(vwap_series) < lookback + 2:
        return None, None, None
    close = df15["close"]
    dist = close - vwap_series
    sigma = float(dist.tail(lookback).std(ddof=0) + 1e-12)
    v = float(vwap_series.iloc[-1])
    return v, v + sigma, v - sigma

def _initial_balance_levels(df15: pd.DataFrame, ib_hours: int = 1) -> Tuple[Optional[float], Optional[float]]:
    if df15 is None or len(df15) < 5:
        return None, None
    ts_last = df15["ts"].iloc[-1]
    try:
        ts_last = pd.to_datetime(ts_last, utc=True)
    except Exception:
        return None, None
    day_anchor = _anchor_day_from_ts(ts_last)
    mask = df15["ts"] >= pd.Timestamp(day_anchor, tz=timezone.utc)
    df_day = df15.loc[mask]
    if df_day.empty:
        return None, None
    bars_needed = max(1, int(ib_hours * 60 / 15))
    df_ib = df_day.head(bars_needed)
    if df_ib.empty:
        return None, None
    return float(df_ib["high"].max()), float(df_ib["low"].min())

def _is_nr_n(df: pd.DataFrame, n: int = 7) -> bool:
    if df is None or len(df) < n + 2:
        return False
    rng = (df["high"] - df["low"]).tail(n)
    if rng.isna().any():
        return False
    return bool(rng.iloc[-1] == rng.min())

def _is_breakout(df: pd.DataFrame, side: str, lookback: int = 3) -> bool:
    if df is None or len(df) < lookback + 2:
        return False
    h = df["high"].iloc[-(lookback + 1):-1].max()
    l = df["low"].iloc[-(lookback + 1):-1].min()
    c = float(df["close"].iloc[-1])
    return bool(c > float(h) if side == "LONG" else c < float(l))

def _micro_gap(df: pd.DataFrame, atr_fn, tf: str = "15m", k: float = 0.8) -> Tuple[bool, str, float]:
    if df is None or len(df) < 20:
        return False, "none", 0.0
    atrv = float(atr_fn(df, 14).iloc[-1])
    o = float(df["open"].iloc[-1])
    pc = float(df["close"].iloc[-2])
    d = o - pc
    if abs(d) > k * atrv:
        return True, ("up" if d > 0 else "down"), abs(d)
    return False, "none", 0.0

# ---------------- RSI regime / BTC shocks / Dominance & Breadth ----------------
def _rsi_regime(val: float) -> str:
    if val is None or not np.isfinite(val):
        return "neutral"
    if val >= 55:
        return "bull"
    if val <= 45:
        return "bear"
    return "neutral"

def _btc_shock(market, lookback: int = 96, sigma_thr: float = 2.0) -> Tuple[bool, int, float]:
    try:
        btc5 = market.fetch_ohlcv("BTC/USDT", "5m", max(200, lookback + 5))
        if btc5 is None or len(btc5) < lookback + 5:
            return False, 0, 0.0
        close = btc5["close"].astype(float)
        ret = close.pct_change().dropna().tail(lookback)
        if len(ret) < lookback - 2:
            return False, 0, 0.0
        mu = float(ret.mean())
        sd = float(ret.std(ddof=0) + 1e-12)
        last = float((close.iloc[-1] / close.iloc[-2]) - 1.0)
        z = (last - mu) / sd if sd > 0 else 0.0
        if abs(z) >= sigma_thr:
            return True, (1 if z > 0 else -1), float(z)
        return False, 0, float(z)
    except Exception:
        return False, 0, 0.0

_BTCD_CACHE: Dict[str, Any] = {"ts": 0.0, "val": None, "prev": None}
_BREADTH_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}

def _fetch_btc_dominance() -> Tuple[Optional[float], Optional[float]]:
    # Возвращает (btc.d %, d_btc.d за 10 мин). Кеш 10 минут.
    ts = pytime.time()
    if _BTCD_CACHE["val"] is not None and ts - _BTCD_CACHE["ts"] < 600:
        val = _BTCD_CACHE["val"]
        prev = _BTCD_CACHE["prev"]
        delta = None if prev is None else (val - prev)
        return val, delta
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        if r.status_code == 200:
            data = r.json()
            m = data.get("data", {}).get("market_cap_percentage", {})
            btc = float(m.get("btc", 0.0))
            prev = _BTCD_CACHE["val"]
            _BTCD_CACHE.update({"ts": ts, "val": btc, "prev": prev})
            delta = None if prev is None else (btc - prev)
            return btc, delta
    except Exception:
        pass
    return None, None

def _compute_breadth(market, symbols: List[str], ema_fn, ttl_sec: int = 300) -> Dict[str, Any]:
    ts = pytime.time()
    if _BREADTH_CACHE["data"] is not None and ts - _BREADTH_CACHE["ts"] < ttl_sec:
        return _BREADTH_CACHE["data"]
    try:
        above50_1h = above200_1h = above50_4h = above200_4h = adv = decl = total = 0
        for sym in symbols:
            df1h = market.fetch_ohlcv(sym, "1h", 260)
            df4h = market.fetch_ohlcv(sym, "4h", 260)
            if df1h is None or df4h is None or len(df1h) < 210 or len(df4h) < 60:
                continue
            total += 1
            df1h["ema50"] = ema_fn(df1h["close"], 50)
            df1h["ema200"] = ema_fn(df1h["close"], 200)
            df4h["ema50"] = ema_fn(df4h["close"], 50)
            df4h["ema200"] = ema_fn(df4h["close"], 200)
            c1 = float(df1h["close"].iloc[-1])
            c4 = float(df4h["close"].iloc[-1])
            if c1 > float(df1h["ema50"].iloc[-1]): above50_1h += 1
            if c1 > float(df1h["ema200"].iloc[-1]): above200_1h += 1
            if c4 > float(df4h["ema50"].iloc[-1]): above50_4h += 1
            if c4 > float(df4h["ema200"].iloc[-1]): above200_4h += 1
            ret1h = c1 - float(df1h["close"].iloc[-2])
            if ret1h >= 0: adv += 1
            else: decl += 1
        data = {
            "total": total,
            "pct50_1h": (above50_1h / total * 100.0) if total else 0.0,
            "pct200_1h": (above200_1h / total * 100.0) if total else 0.0,
            "pct50_4h": (above50_4h / total * 100.0) if total else 0.0,
            "pct200_4h": (above200_4h / total * 100.0) if total else 0.0,
            "ad_line": adv - decl,
        }
        _BREADTH_CACHE.update({"ts": ts, "data": data})
        return data
    except Exception:
        return {"total": 0}

# ---------------- Funding / OI / Basis (best-effort) ----------------
def _fetch_funding_rate(market, symbol: str) -> Optional[float]:
    try:
        for name, ex in market._available_exchanges():
            if not hasattr(ex, "fetchFundingRate"):
                continue
            resolved = market.resolve_symbol(ex, symbol) or symbol
            if resolved not in ex.markets:
                continue
            try:
                fr = ex.fetchFundingRate(resolved)
                if isinstance(fr, dict):
                    val = fr.get("fundingRate") or fr.get("info", {}).get("lastFundingRate")
                    if val is None:
                        continue
                    return float(val)
            except Exception:
                continue
    except Exception:
        pass
    return None

def _fetch_open_interest(market, symbol: str) -> Optional[float]:
    try:
        for name, ex in market._available_exchanges():
            # разные биржи: fetchOpenInterest или fetchOpenInterestHistory
            resolved = market.resolve_symbol(ex, symbol) or symbol
            if resolved not in ex.markets:
                continue
            try:
                if hasattr(ex, "fetchOpenInterest"):
                    oi = ex.fetchOpenInterest(resolved)
                    if isinstance(oi, dict):
                        val = oi.get("openInterest") or oi.get("info", {}).get("openInterest")
                        if val is not None:
                            return float(val)
                # history — можно брать последнее значение
                if hasattr(ex, "fetchOpenInterestHistory"):
                    arr = ex.fetchOpenInterestHistory(resolved, limit=2)
                    if isinstance(arr, list) and arr:
                        last = arr[-1]
                        if isinstance(last, dict) and "openInterest" in last:
                            return float(last["openInterest"])
            except Exception:
                continue
    except Exception:
        pass
    return None

def _basis_spot_perp(market, symbol: str) -> Optional[float]:
    # пытаемся взять spot и линейный swap по одному эксчейнджу
    base, quote = symbol.split("/")
    try:
        for name, ex in market._available_exchanges():
            spot_key = None; swap_key = None
            for mkey, m in ex.markets.items():
                try:
                    if str(m.get("base")).upper() != base or str(m.get("quote")).upper() != quote:
                        continue
                    if m.get("spot"):
                        spot_key = mkey
                    if m.get("swap") and m.get("linear"):
                        swap_key = mkey
                except Exception:
                    continue
            if not spot_key or not swap_key:
                continue
            ts = ex.fetch_ticker(spot_key)
            tp = ex.fetch_ticker(swap_key)
            spot = float(ts.get("last") or ts.get("close") or 0.0)
            perp = None
            if "mark" in tp and tp["mark"]:
                perp = float(tp["mark"])
            if perp is None:
                inf = tp.get("info", {})
                for k in ("markPrice", "indexPrice", "mark_price", "index_price"):
                    if k in inf and inf[k]:
                        perp = float(inf[k]); break
            if perp is None:
                perp = float(tp.get("last") or tp.get("close") or 0.0)
            if spot and perp:
                return float((perp - spot) / (spot + 1e-12))
    except Exception:
        pass
    return None

# ---------------- CVD / Footprint-lite ----------------
def _cvd_slope(df: pd.DataFrame, window: int = 60) -> Optional[float]:
    if df is None or len(df) < window + 3:
        return None
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)
    sign = np.sign(close.diff().fillna(0.0).values)
    delta = sign * vol.values
    cvd = np.cumsum(delta)
    return float(cvd[-1] - cvd[-window])

# ---------------- Volume profile (POC/VAH/VAL) ----------------
def _volume_profile(df: pd.DataFrame, bins: int = 40, lookback: int = 240) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    # Аппрокс: распределяем объёмы по типичной цене (tp) последних lookback баров
    if df is None or len(df) < 20:
        return None, None, None
    x = df.tail(lookback).copy()
    tp = (x["high"] + x["low"] + x["close"]) / 3.0
    vol = x["volume"].astype(float)
    prices = tp.values.astype(float)
    weights = vol.values
    lo = float(np.min(prices)); hi = float(np.max(prices))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return None, None, None
    hist, edges = np.histogram(prices, bins=bins, range=(lo, hi), weights=weights)
    if hist.sum() <= 0:
        return None, None, None
    poc_idx = int(np.argmax(hist))
    poc = float(0.5 * (edges[poc_idx] + edges[poc_idx + 1]))
    # Value Area ~ 68% веса вокруг POC
    total = hist.sum()
    order = np.argsort(hist)[::-1]
    acc = 0.0
    mask = np.zeros_like(hist, dtype=bool)
    for idx in order:
        mask[idx] = True
        acc += hist[idx]
        if acc / total >= 0.68:
            break
    sel_edges = edges[np.r_[np.where(mask)[0].min(), np.where(mask)[0].max() + 1]]
    val = float(sel_edges[0]); vah = float(sel_edges[1])
    return poc, vah, val

# ---------------- Order book stats ----------------
def _orderbook_stats(market, symbol: str, depth: int = 25) -> Dict[str, Any]:
    out = {"imb": None, "ask_wall": False, "bid_wall": False}
    try:
        for name, ex in market._available_exchanges():
            resolved = market.resolve_symbol(ex, symbol) or symbol
            if resolved not in ex.markets:
                continue
            try:
                ob = ex.fetch_order_book(resolved, limit=depth)
                bids = ob.get("bids") or []
                asks = ob.get("asks") or []
                sb = float(sum([b[1] for b in bids[:depth]]) or 0.0)
                sa = float(sum([a[1] for a in asks[:depth]]) or 0.0)
                if sb + sa > 0:
                    out["imb"] = float((sb - sa) / (sb + sa))
                # стены: если топ-уровень сильно выше медианы
                def wall(levels):
                    if not levels:
                        return False
                    sizes = np.array([x[1] for x in levels[:min(depth, len(levels))]], dtype=float)
                    med = float(np.median(sizes) + 1e-9)
                    return bool(np.max(sizes) > 3.5 * med and np.max(sizes) > 0.0)
                out["ask_wall"] = wall(asks)
                out["bid_wall"] = wall(bids)
                return out
            except Exception:
                continue
    except Exception:
        pass
    return out

# ---------------- kNN shape matching (5m) ----------------
def _znorm(arr: np.ndarray) -> np.ndarray:
    m = np.mean(arr)
    s = np.std(arr) + 1e-12
    return (arr - m) / s

def _knn_forward_edge(df5: pd.DataFrame, window: int = 48, horizon: int = 12, step: int = 4, k: int = 12) -> Optional[Tuple[float, int]]:
    try:
        if df5 is None or len(df5) < window + horizon + 20:
            return None
        close = df5["close"].astype(float).values
        ret = np.diff(close) / (close[:-1] + 1e-12)
        if len(ret) < window + horizon + 10:
            return None
        target = _znorm(ret[-window:])
        dists = []; fwd = []
        for i in range(window + horizon, len(ret) - horizon, step):
            seg = _znorm(ret[i - window:i])
            d = np.linalg.norm(seg - target)
            dists.append(d)
            fwd_ret = float(np.sum(ret[i:i + horizon]))
            fwd.append(fwd_ret)
        if not dists:
            return None
        idx = np.argsort(dists)[:k]
        fwd_sel = np.array(fwd)[idx]
        return float(np.mean(fwd_sel) * 100.0), int(len(idx))
    except Exception:
        return None

# ---------------- Simple change-point detector (vol regime) ----------------
def _cpd_volatility(close: pd.Series, w_short: int = 24, w_long: int = 96, thr: float = 1.6) -> str:
    if len(close) < w_long + 5:
        return "unknown"
    ret = close.pct_change().dropna()
    vs = float(ret.tail(w_short).std(ddof=0) + 1e-12)
    vl = float(ret.tail(w_long).std(ddof=0) + 1e-12)
    ratio = vs / vl
    if ratio >= thr:
        return "shock"
    elif ratio <= 1.0 / thr:
        return "calm"
    else:
        return "normal"

# ---------------- ML-lite (logistic) ----------------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _ml_probability(features: Dict[str, float], weight_map: Dict[str, float], bias: float = 0.0) -> float:
    z = float(bias)
    for k, w in weight_map.items():
        z += float(features.get(k, 0.0)) * float(w)
    return float(_sigmoid(z))

def _parse_weights_env(env_name: str) -> Dict[str, float]:
    # Формат: "feat1:0.3,feat2:-0.1"
    txt = os.getenv(env_name, "").strip()
    out = {}
    if not txt:
        return out
    for part in txt.split(","):
        if ":" in part:
            f, v = part.split(":", 1)
            f = f.strip(); v = v.strip()
            try:
                out[f] = float(v)
            except Exception:
                continue
    return out

# ---------------- Patch entry ----------------
def patch(app: Dict[str, Any]) -> None:
    logger = app.get("logger")

    # Алиасы индикаторов
    if "rsi" not in app and "rsi_wilder" in app:
        app["rsi"] = app["rsi_wilder"]; logger and logger.info("TA patch: rsi -> rsi_wilder")
    if "atr" not in app and "atr_wilder" in app:
        app["atr"] = app["atr_wilder"]; logger and logger.info("TA patch: atr -> atr_wilder")
    if "adx" not in app and "adx_wilder" in app:
        app["adx"] = app["adx_wilder"]; logger and logger.info("TA patch: adx -> adx_wilder")

    # Prometheus метрики
    if app.get("PROMETHEUS_OK"):
        try:
            from prometheus_client import Histogram, Counter  # noqa
            if "MET_OHLCV_CACHE_HIT" not in app:
                app["MET_OHLCV_CACHE_HIT"] = Counter("ohlcv_cache_hit_total", "OHLCV cache hits")
            if "MET_OHLCV_LATENCY" not in app:
                app["MET_OHLCV_LATENCY"] = Histogram("ohlcv_fetch_latency_seconds", "OHLCV fetch latency")
            if "MET_TICKER_LATENCY" not in app:
                app["MET_TICKER_LATENCY"] = Histogram("ticker_fetch_latency_seconds", "Ticker fetch latency")
        except Exception as e:
            logger and logger.warning("TA patch: Prometheus init fail: %s", e)

    # Оригиналы
    orig_score_symbol_core = app.get("score_symbol_core")
    orig_build_reason = app.get("build_reason")
    orig_update_trailing = app.get("update_trailing")
    orig_tech_risk_trigger = app.get("_tech_risk_trigger")
    orig_watch_signal_price = app.get("watch_signal_price")

    # Ссылки
    now_msk = app["now_msk"]
    market = app["market"]
    ema = app["ema"]; rsi = app["rsi"]; macd = app["macd"]
    atr = app["atr"]; adx = app["adx"]
    anchored_vwap = app["anchored_vwap"]; week_anchor_from_df = app["week_anchor_from_df"]
    prev_session_levels = app["prev_session_levels"]
    supertrend = app["supertrend"]; chandelier_exit_level = app["chandelier_exit_level"]
    format_price = app["format_price"]

    # ENV-конфиги
    TA_SMC = int(os.getenv("TA_SMC", "1"))
    TA_ZIGZAG_PCT = float(os.getenv("TA_ZIGZAG_PCT", "1.0"))
    TA_VWAP_BANDS = int(os.getenv("TA_VWAP_BANDS", "1"))
    TA_BREADTH = int(os.getenv("TA_BREADTH", "1"))
    TA_BTC_DOM = int(os.getenv("TA_BTC_DOM", "1"))
    TA_PATTERN_LIB = int(os.getenv("TA_PATTERN_LIB", "1"))
    TA_HISTORY_MATCH = int(os.getenv("TA_HISTORY_MATCH", "1"))
    TA_FUNDING = int(os.getenv("TA_FUNDING", "1"))
    TA_SCORE_MIN = float(os.getenv("TA_SCORE_MIN", "1.9"))
    TA_TIME_STOP_BARS = int(os.getenv("TA_TIME_STOP_BARS", "24"))
    TA_BTC_SHOCK_SIGMA = float(os.getenv("TA_BTC_SHOCK_SIGMA", "2.0"))
    TA_VWAP_BANDS_LOOKBACK = int(os.getenv("TA_VWAP_BANDS_LOOKBACK", "96"))
    TA_RVOL_LK5 = int(os.getenv("TA_RVOL_LK5", "96"))
    TA_RVOL_LK15 = int(os.getenv("TA_RVOL_LK15", "96"))
    TA_CHOP_PERIOD = int(os.getenv("TA_CHOP_PERIOD", "14"))
    TA_ASSET_PROFILE_LOOKBACK = int(os.getenv("TA_ASSET_PROFILE_LOOKBACK", "96"))
    TA_ML_LIGHT = int(os.getenv("TA_ML_LIGHT", "1"))
    TA_ML_P_THRESHOLD = float(os.getenv("TA_ML_P_THRESHOLD", "0.52"))
    TA_ML_BIAS = float(os.getenv("TA_ML_BIAS", "0.0"))
    TA_ML_WEIGHTS = _parse_weights_env("TA_ML_WEIGHTS")  # пример: "rvol:0.3,adx:0.2,chop:-0.1,nn:0.15,btcdom:-0.2"

    TRAIL_ATR_CAP = app.get("TRAIL_ATR_CAP", 0.30)

    # Кэш профилей актива (адаптивные пороги)
    _ASSET_PROFILE_CACHE: Dict[str, Dict[str, float]] = {}

    def _asset_profile(symbol: str, df15: pd.DataFrame) -> Dict[str, float]:
        base = symbol.split("/")[0]
        now_ts = pytime.time()
        ent = _ASSET_PROFILE_CACHE.get(base)
        if ent and now_ts - ent.get("ts", 0) < 600:
            return ent
        try:
            # ATR% и ADX квантиль на окне
            atr_series = atr(df15, 14) / (df15["close"].astype(float) + 1e-12) * 100.0
            adx_series = adx(df15, 14)
            atr_q20 = float(np.nanpercentile(atr_series.tail(TA_ASSET_PROFILE_LOOKBACK), 20))
            atr_q80 = float(np.nanpercentile(atr_series.tail(TA_ASSET_PROFILE_LOOKBACK), 80))
            adx_med = float(np.nanmedian(adx_series.tail(TA_ASSET_PROFILE_LOOKBACK)))
            prof = {"atr_q20": atr_q20, "atr_q80": atr_q80, "adx_med": adx_med, "ts": now_ts}
            _ASSET_PROFILE_CACHE[base] = prof
            return prof
        except Exception:
            return {"atr_q20": 0.1, "atr_q80": 1.5, "adx_med": 20.0, "ts": now_ts}

    # ---------- Enhanced reason ----------
    def _enhanced_reason(details: Dict[str, Any]) -> str:
        base = ""
        try:
            base = orig_build_reason(details) if orig_build_reason else ""
        except Exception:
            base = ""
        extra_bits = []
        # SMC
        if details.get("fvg_near"): extra_bits.append(f"FVG {details.get('fvg_type','?')}")
        if details.get("ob_near"): extra_bits.append(f"OB {details.get('ob_type','?')}")
        if details.get("eqh") or details.get("eql"): extra_bits.append(("EQH" if details.get("eqh") else "") + (" EQL" if details.get("eql") else ""))
        if details.get("sfp"): extra_bits.append("SFP/2B")
        if details.get("bos_disp"): extra_bits.append("BOS↑" if details.get("bos_disp_dir") == "up" else "BOS↓")
        # Каналы/фракталы/zigzag
        if details.get("zz_trend"): extra_bits.append(f"ZigZag {details['zz_trend']}")
        if details.get("channel_conf"): extra_bits.append("Channel confluence")
        # Профиль
        if details.get("poc") is not None: extra_bits.append("POC/VA")
        # Гэпы
        if details.get("gap_15m"): extra_bits.append(f"Gap15 {details.get('gap15_dir','')}")
        if details.get("gap_1h"): extra_bits.append(f"Gap1h {details.get('gap1h_dir','')}")
        # Паттерны
        if details.get("pattern_list"): extra_bits.append("Pattern: " + ", ".join(details["pattern_list"][:2]))
        # История
        if "nn_edge" in details and details["nn_edge"] is not None: extra_bits.append(f"History {details['nn_edge']:+.2f}% (k={details.get('nn_k',0)})")
        # BTC.D + breadth
        if details.get("btc_dom_trend"): extra_bits.append(f"BTC.D {details['btc_dom_trend']}")
        if "breadth_pct50_1h" in details: extra_bits.append(f"Breadth {details['breadth_pct50_1h']:.0f}%")
        # Ордерфлоу
        if "ob_imb" in details and details["ob_imb"] is not None: extra_bits.append(f"OB-imb {details['ob_imb']:+.2f}")
        if details.get("ask_wall") or details.get("bid_wall"): extra_bits.append("walls")
        if "cvd_slope" in details and details["cvd_slope"] is not None: extra_bits.append(f"CVD {details['cvd_slope']:+.0f}")
        if "basis" in details and details["basis"] is not None: extra_bits.append(f"basis {details['basis']*100:+.2f}%")
        if "funding_rate" in details and details["funding_rate"] is not None: extra_bits.append(f"funding {details['funding_rate']*100:.3f}%/8h")
        # RVOL / FDI / CHOP / режим
        if "rvol_combo" in details: extra_bits.append(f"RVOL x{details['rvol_combo']:.2f}")
        if "fdi" in details and details["fdi"] is not None: extra_bits.append(f"FDI {details['fdi']:.2f}")
        if details.get("nr4"): extra_bits.append("NR4")
        if details.get("nr7"): extra_bits.append("NR7")
        if details.get("vol_regime"): extra_bits.append(f"reg {details['vol_regime']}")

        # ML-фильтр
        if "ml_p" in details and details["ml_p"] is not None:
            extra_bits.append(f"ML p={details['ml_p']:.2f}")

        # Score breakdown (топ 3)
        br = details.get("score_breakdown", {})
        if isinstance(br, dict) and br:
            parts = sorted(br.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            if parts:
                extra_bits.append("[" + ", ".join([f"{k}:{v:+.2f}" for k, v in parts]) + "]")
        extra = " • ".join([x for x in extra_bits if x])
        if base and extra:
            return f"{base} • {extra}"
        return base or extra

    # ---------- Enhanced trailing ----------
    async def _enhanced_update_trailing(sig) -> None:
        try:
            df15 = market.fetch_ohlcv(sig.symbol, "15m", 240)
            if df15 is None or len(df15) < 60:
                return
            st_line, st_dir = supertrend(df15, 10, 3.0)
            atr_val = float(atr(df15, 14).iloc[-1])
            close = float(df15["close"].iloc[-1])
            tick = market.get_tick_size(sig.symbol)
            ema21 = ema(df15["close"], 21)
            vwap_day = anchored_vwap(df15)

            if not hasattr(sig, "_init_sl"):
                sig._init_sl = float(sig.sl)
                sig._init_r = abs(float(sig.entry) - float(sig.sl))
                sig._vwap_tests = 0

            if len(df15) >= 3:
                vd = float(vwap_day.iloc[-1])
                touched = (df15["low"].tail(3).min() <= vd <= df15["high"].tail(3).max())
                if touched:
                    sig._vwap_tests = getattr(sig, "_vwap_tests", 0) + 1

            ch_val = chandelier_exit_level(df15, atr_mult=2.5, period=22, side=sig.side)
            k = 1.7 if sig.tp_hit < 2 else 1.25

            if sig.side == "LONG":
                atr_stop = close - k * atr_val
                stp = max(float(st_line.iloc[-1]), float(ch_val), atr_stop)
                if sig.tp_hit >= 1 and sig._vwap_tests >= 3:
                    stp = max(stp, float(ema21.iloc[-1]), float(vwap_day.iloc[-1]))
                if sig.tp_hit >= 2 and hasattr(sig, "_init_r"):
                    stp = max(stp, float(sig.entry) + 0.3 * float(sig._init_r))
                stp = max(stp, float(sig.entry) - TRAIL_ATR_CAP * atr_val)
                stp = _round_to_tick(stp, tick, "floor")
                if stp > sig.sl:
                    sig.sl = stp
            else:
                atr_stop = close + k * atr_val
                stp = min(float(st_line.iloc[-1]), float(ch_val), atr_stop)
                if sig.tp_hit >= 1 and sig._vwap_tests >= 3:
                    stp = min(stp, float(ema21.iloc[-1]), float(vwap_day.iloc[-1]))
                if sig.tp_hit >= 2 and hasattr(sig, "_init_r"):
                    stp = min(stp, float(sig.entry) - 0.3 * float(sig._init_r))
                stp = min(stp, float(sig.entry) + TRAIL_ATR_CAP * atr_val)
                stp = _round_to_tick(stp, tick, "ceil")
                if stp < sig.sl:
                    sig.sl = stp
        except Exception:
            pass

    # ---------- Enhanced tech risk trigger ----------
    def _enhanced_tech_risk_trigger(symbol: str, side: str) -> Optional[str]:
        try:
            df15 = market.fetch_ohlcv(symbol, "15m", 240)
            df5 = market.fetch_ohlcv(symbol, "5m", 360)
            if df15 is None or df5 is None:
                return None
            vwap_day = anchored_vwap(df15)
            c15 = float(df15["close"].iloc[-1]); vd = float(vwap_day.iloc[-1])

            ha5 = _heikin_ashi_df(df5.set_index("ts").reset_index())
            ha_bear = bool(ha5["close"].iloc[-1] < ha5["open"].iloc[-1])
            ha_bull = not ha_bear
            df5["rsi"] = rsi(df5["close"], 14); rsi5 = float(df5["rsi"].iloc[-1])
            st_line, st_dir = supertrend(df15, 10, 3.0)
            st_last = int(st_dir.iloc[-1]); st_prev = int(st_dir.iloc[-2]) if len(st_dir) >= 2 else st_last

            shocked, dir_, z = _btc_shock(market, lookback=96, sigma_thr=TA_BTC_SHOCK_SIGMA)

            # BTC.D spike / basis shock
            bdom, ddom = _fetch_btc_dominance()
            basis = _basis_spot_perp(market, symbol)

            atr15 = float(atr(df15, 14).iloc[-1])
            amp = float(df5["high"].tail(TA_TIME_STOP_BARS).max() - df5["low"].tail(TA_TIME_STOP_BARS).min()) if len(df5) >= TA_TIME_STOP_BARS else float("inf")
            stagnation = bool(amp < 0.5 * atr15)

            mom_fail_long = (np.all(np.diff(df5["high"].tail(6)) < 0) if len(df5) >= 6 else False) and c15 < vd
            mom_fail_short = (np.all(np.diff(df5["low"].tail(6)) > 0) if len(df5) >= 6 else False) and c15 > vd

            msgs = []
            if side == "LONG":
                if st_last < 0 and st_prev < 0: msgs.append("SuperTrend(15m) уверенно вниз.")
                if c15 < vd and ha_bear and rsi5 < 45: msgs.append("Потеря VWAP, HA(5m) медв., RSI<45.")
                if shocked and dir_ < 0: msgs.append(f"BTC shock {z:.1f}σ вниз.")
                if stagnation: msgs.append("Стагнация диапазона — подумать о частичной фиксации.")
                if mom_fail_long: msgs.append("Momentum failure: lower highs и ниже VWAP.")
                if ddom is not None and ddom > 0.15: msgs.append("BTC.D spike вверх — риск для альтов.")
                if basis is not None and basis < -0.004: msgs.append("Негат. basis — риск для лонга.")
            else:
                if st_last > 0 and st_prev > 0: msgs.append("SuperTrend(15m) уверенно вверх.")
                if c15 > vd and ha_bull and rsi5 > 55: msgs.append("Выше VWAP, HA(5m) быч., RSI>55.")
                if shocked and dir_ > 0: msgs.append(f"BTC shock {z:.1f}σ вверх.")
                if stagnation: msgs.append("Стагнация диапазона — подумать о частичной фиксации.")
                if mom_fail_short: msgs.append("Momentum failure: higher lows и выше VWAP.")
                if ddom is not None and ddom < -0.15: msgs.append("BTC.D spike вниз — риск для доминации альтов (шортам осторожно).")
                if basis is not None and basis > 0.004: msgs.append("Позит. basis — риск для шорта.")
            if msgs:
                return " ".join(msgs[:2])
            return None
        except Exception:
            return None

    # ---------- Enhanced score ----------
    def _enhanced_score_symbol_core(symbol: str, relax: bool = False):
        base = orig_score_symbol_core(symbol, relax) if orig_score_symbol_core else None
        if base is None:
            return None
        side_score, side, details = base

        df5 = market.fetch_ohlcv(symbol, "5m", 700)
        df15 = market.fetch_ohlcv(symbol, "15m", 500)
        df1h = market.fetch_ohlcv(symbol, "1h", 420)
        df1d = market.fetch_ohlcv(symbol, "1d", 420)
        df1w = market.fetch_ohlcv(symbol, "1w", 260)
        if df15 is None or df5 is None:
            return base

        breakdown: Dict[str, float] = details.get("score_breakdown", {}).copy() if isinstance(details.get("score_breakdown"), dict) else {}

        c15 = float(df15["close"].iloc[-1])
        atr15 = float(details.get("atr", None) or atr(df15, 14).iloc[-1])
        tick = market.get_tick_size(symbol)

        # RVOL/FDI/CHOP/Vol-regime
        rvol_combo = _combined_rvol(df5["volume"], df15["volume"], lk5=TA_RVOL_LK5, lk15=TA_RVOL_LK15)
        chop15 = _choppiness_index(df15, TA_CHOP_PERIOD)
        fdi = _fractal_dimension_index(df15["close"], window=120)
        vol_regime = _cpd_volatility(df5["close"], w_short=24, w_long=96, thr=1.6)
        details["vol_regime"] = vol_regime

        # Адаптивный профиль актива
        prof = _asset_profile(symbol, df15)
        atr_pct_now = float(atr15 / (c15 + 1e-12) * 100.0)
        # штраф/бонус, если текущая волатильность сильно вне типичного диапазона актива
        if atr_pct_now < prof.get("atr_q20", 0.1):
            breakdown["AssetFit"] = breakdown.get("AssetFit", 0.0) - 0.10
        elif atr_pct_now > prof.get("atr_q80", 1.5):
            breakdown["AssetFit"] = breakdown.get("AssetFit", 0.0) + 0.05

        # SMC
        if TA_SMC:
            fvg15 = _find_fvg(df15, lookback=120)
            fvg1h = _find_fvg(df1h, lookback=80) if df1h is not None else []
            fvg_all = (fvg15 or []) + (fvg1h or [])
            z, dist = _fvg_status(fvg_all, c15)
            if z:
                d_atr = dist / max(atr15, 1e-9)
                details["fvg_near"] = True
                details["fvg_type"] = "bull" if z["type"] == 1 else "bear"
                if (side == "LONG" and z["type"] == -1 and d_atr < 1.0) or (side == "SHORT" and z["type"] == 1 and d_atr < 1.0):
                    breakdown["FVG"] = breakdown.get("FVG", 0.0) - 0.15
                else:
                    breakdown["FVG"] = breakdown.get("FVG", 0.0) + 0.05
            ob = _find_order_block(df15, side, atr_fn=atr, atr_period=14)
            if ob:
                details["ob_near"] = True
                details["ob_type"] = "demand" if ob["type"] == 1 else "supply"
                if side == "LONG":
                    if ob["low"] <= c15 <= ob["high"]:
                        breakdown["OB"] = breakdown.get("OB", 0.0) - 0.15
                    elif c15 >= ob["low"] and (c15 - ob["low"]) / max(atr15, 1e-9) < 0.6:
                        breakdown["OB"] = breakdown.get("OB", 0.0) + 0.20
                else:
                    if ob["low"] <= c15 <= ob["high"]:
                        breakdown["OB"] = breakdown.get("OB", 0.0) - 0.15
                    elif c15 <= ob["high"] and (ob["high"] - c15) / max(atr15, 1e-9) < 0.6:
                        breakdown["OB"] = breakdown.get("OB", 0.0) + 0.20
            eqh, eql = _equal_highs_lows(df15, lookback=50, tol_frac=0.0008)
            details["eqh"] = bool(eqh); details["eql"] = bool(eql)
            if _sfp_2b(df15, side, lookback=30):
                breakdown["SFP"] = breakdown.get("SFP", 0.0) + 0.15
                details["sfp"] = True
            if _quasimodo_zigzag(df15["close"], pct=max(0.6, TA_ZIGZAG_PCT)):
                breakdown["QM"] = breakdown.get("QM", 0.0) + 0.10
            if _bos_displacement(df15, side, atr_fn=atr, atr_p=14, lookback=40, k=1.2):
                breakdown["BOS"] = breakdown.get("BOS", 0.0) + 0.20
                details["bos_disp"] = True
                details["bos_disp_dir"] = "up" if side == "LONG" else "down"

        # ZigZag/Channel
        zz_idx = _zigzag_idx(df15["close"], pct=max(0.6, TA_ZIGZAG_PCT))
        zz_trend = ""
        if len(zz_idx) >= 3:
            a, b, c = df15["close"].iloc[zz_idx[-3]], df15["close"].iloc[zz_idx[-2]], df15["close"].iloc[zz_idx[-1]]
            if c > b > a: zz_trend = "HH/HL"
            elif c < b < a: zz_trend = "LH/LL"
        if zz_trend:
            details["zz_trend"] = zz_trend
            if side == "LONG":
                breakdown["ZZ"] = breakdown.get("ZZ", 0.0) + (0.15 if zz_trend == "HH/HL" else -0.10)
            else:
                breakdown["ZZ"] = breakdown.get("ZZ", 0.0) + (0.15 if zz_trend == "LH/LL" else -0.10)
        slope, intercept, lo, mid, hi = _regression_channel(df15["close"], window=120)
        conf = _channel_confluence(c15, lo, mid, hi, side)
        if conf != 0.0:
            breakdown["Channel"] = breakdown.get("Channel", 0.0) + conf
            details["channel_conf"] = True

        # VWAP бэнды
        if TA_VWAP_BANDS:
            v, up1, dn1 = _vwap_sigma_bands(df15, lookback=TA_VWAP_BANDS_LOOKBACK, anchored_vwap=anchored_vwap)
            if v is not None and up1 is not None and dn1 is not None:
                if side == "LONG" and (up1 - c15) < 0.25 * atr15:
                    breakdown["VWAP±σ2"] = breakdown.get("VWAP±σ2", 0.0) - 0.10
                if side == "SHORT" and (c15 - dn1) < 0.25 * atr15:
                    breakdown["VWAP±σ2"] = breakdown.get("VWAP±σ2", 0.0) - 0.10

        # NR/IB/Гэпы
        nr4 = _is_nr_n(df15, 4); nr7 = _is_nr_n(df15, 7)
        if nr4: breakdown["NR4"] = breakdown.get("NR4", 0.0) + 0.05
        if nr7: breakdown["NR7"] = breakdown.get("NR7", 0.0) + 0.05
        details["nr4"] = bool(nr4); details["nr7"] = bool(nr7)
        ib_hi, ib_lo = _initial_balance_levels(df15, ib_hours=1)
        if ib_hi is not None and ib_lo is not None:
            if side == "LONG" and ib_hi - c15 < 0.3 * atr15:
                breakdown["IB"] = breakdown.get("IB", 0.0) - 0.10
            if side == "SHORT" and c15 - ib_lo < 0.3 * atr15:
                breakdown["IB"] = breakdown.get("IB", 0.0) - 0.10
        gap15, gap15_dir, _ = _micro_gap(df15, atr_fn=atr, tf="15m", k=0.8)
        gap1h, gap1h_dir, _ = _micro_gap(df1h, atr_fn=atr, tf="1h", k=0.8) if df1h is not None else (False, "none", 0.0)
        details["gap_15m"] = bool(gap15); details["gap15_dir"] = gap15_dir
        details["gap_1h"] = bool(gap1h); details["gap1h_dir"] = gap1h_dir
        if gap15 or gap1h:
            if (side == "LONG" and (gap15_dir == "down" or gap1h_dir == "down")) or (side == "SHORT" and (gap15_dir == "up" or gap1h_dir == "up")):
                breakdown["Gap"] = breakdown.get("Gap", 0.0) + 0.05
            else:
                breakdown["Gap"] = breakdown.get("Gap", 0.0) - 0.05

        # Паттерны (лайт)
        patterns = []
        if TA_PATTERN_LIB:
            if len(df15) >= 3:
                h1 = float(df15["high"].iloc[-1]); l1 = float(df15["low"].iloc[-1])
                h0 = float(df15["high"].iloc[-2]); l0 = float(df15["low"].iloc[-2])
                if h1 < h0 and l1 > l0:
                    patterns.append("Inside")
                    breakdown["PatternInside"] = breakdown.get("PatternInside", 0.0) + (0.10 if rvol_combo >= 1.2 else 0.0)
                if h1 > h0 and l1 < l0:
                    patterns.append("Outside")
            if len(df15) >= 4:
                h2 = float(df15["high"].iloc[-2]); l2 = float(df15["low"].iloc[-2])
                h3 = float(df15["high"].iloc[-3]); l3 = float(df15["low"].iloc[-3])
                if (h2 < h3 and l2 > l3):
                    brk_up = float(df15["close"].iloc[-1]) < h2 and float(df15["high"].iloc[-1]) > h2
                    brk_dn = float(df15["close"].iloc[-1]) > l2 and float(df15["low"].iloc[-1]) < l2
                    if brk_up or brk_dn:
                        patterns.append("Hikkake")
                        breakdown["Hikkake"] = breakdown.get("Hikkake", 0.0) + 0.10
            # Дополнительно — треугольник/флаг
            if slope is not None and lo is not None and hi is not None:
                width = hi - lo
                if width < 0.7 * atr15 and nr7:
                    patterns.append("Triangle/Flag")
                    breakdown["FlagTri"] = breakdown.get("FlagTri", 0.0) + 0.10
            if patterns:
                details["pattern_list"] = patterns

        # Исторический kNN
        if TA_HISTORY_MATCH:
            nn = _knn_forward_edge(df5, window=48, horizon=12, step=4, k=12)
            if nn:
                nn_edge, nn_k = nn
                details["nn_edge"] = float(nn_edge); details["nn_k"] = int(nn_k)
                breakdown["HistoryNN"] = breakdown.get("HistoryNN", 0.0) + (0.20 if ((side == "LONG" and nn_edge > 0) or (side == "SHORT" and nn_edge < 0)) else -0.10)

        # BTC.D + breadth
        if TA_BTC_DOM or TA_BREADTH:
            try:
                symbols = app.get("SYMBOLS", [])
                breadth = _compute_breadth(market, symbols, ema_fn=ema) if TA_BREADTH else {}
                if breadth:
                    details["breadth_pct50_1h"] = float(breadth.get("pct50_1h", 0.0))
                if TA_BTC_DOM:
                    btc_d, ddom = _fetch_btc_dominance()
                    if btc_d is not None:
                        trend = "up" if (ddom is not None and ddom > 0) else ("down" if (ddom is not None and ddom < 0) else "flat")
                        details["btc_dom"] = float(btc_d); details["btc_dom_trend"] = trend
                        base = symbol.split("/")[0]
                        is_alt = base not in ("BTC", "ETH")
                        if is_alt:
                            if side == "LONG" and trend == "up":
                                breakdown["BTC.D"] = breakdown.get("BTC.D", 0.0) - 0.25
                            if side == "SHORT" and trend == "up":
                                breakdown["BTC.D"] = breakdown.get("BTC.D", 0.0) + 0.10
                        if breadth and is_alt and side == "LONG":
                            if float(breadth.get("pct50_1h", 0.0)) < 45.0:
                                breakdown["Breadth"] = breakdown.get("Breadth", 0.0) - 0.20
                            elif float(breadth.get("pct50_1h", 0.0)) > 60.0:
                                breakdown["Breadth"] = breakdown.get("Breadth", 0.0) + 0.10
            except Exception:
                pass

        # Funding / OI / basis / CVD / L2
        if TA_FUNDING:
            try:
                fr = _fetch_funding_rate(market, symbol)
                if fr is not None:
                    details["funding_rate"] = float(fr)
                    if side == "SHORT" and fr >= 0.0006:
                        breakdown["Funding"] = breakdown.get("Funding", 0.0) - 0.20
                    elif side == "LONG" and fr <= -0.0006:
                        breakdown["Funding"] = breakdown.get("Funding", 0.0) - 0.20
            except Exception:
                pass
        oi = _fetch_open_interest(market, symbol)
        if oi is not None:
            details["oi"] = float(oi)
        basis = _basis_spot_perp(market, symbol)
        if basis is not None:
            details["basis"] = float(basis)
            if side == "LONG" and basis < -0.004:
                breakdown["Basis"] = breakdown.get("Basis", 0.0) - 0.10
            if side == "SHORT" and basis > +0.004:
                breakdown["Basis"] = breakdown.get("Basis", 0.0) - 0.10
        cvd = _cvd_slope(df5, window=60)
        if cvd is not None:
            details["cvd_slope"] = float(cvd)
            if side == "LONG" and cvd > 0:
                breakdown["CVD"] = breakdown.get("CVD", 0.0) + 0.10
            if side == "SHORT" and cvd < 0:
                breakdown["CVD"] = breakdown.get("CVD", 0.0) + 0.10
        ob_stats = _orderbook_stats(market, symbol, depth=25)
        if ob_stats:
            details["ob_imb"] = float(ob_stats.get("imb")) if ob_stats.get("imb") is not None else None
            details["ask_wall"] = bool(ob_stats.get("ask_wall", False))
            details["bid_wall"] = bool(ob_stats.get("bid_wall", False))
            if details["ob_imb"] is not None:
                if side == "LONG":
                    breakdown["L2imb"] = breakdown.get("L2imb", 0.0) + (0.10 if details["ob_imb"] > 0.15 else (-0.10 if details["ob_imb"] < -0.15 else 0.0))
                else:
                    breakdown["L2imb"] = breakdown.get("L2imb", 0.0) + (0.10 if details["ob_imb"] < -0.15 else (-0.10 if details["ob_imb"] > 0.15 else 0.0))
            if side == "LONG" and ob_stats.get("ask_wall", False):
                breakdown["L2walls"] = breakdown.get("L2walls", 0.0) - 0.10
            if side == "SHORT" and ob_stats.get("bid_wall", False):
                breakdown["L2walls"] = breakdown.get("L2walls", 0.0) - 0.10

        # Профиль объёма POC/VAH/VAL
        poc, vah, val = _volume_profile(df15, bins=40, lookback=240)
        details["poc"] = poc; details["vah"] = vah; details["val"] = val
        if poc is not None and vah is not None and val is not None:
            if side == "LONG":
                if c15 >= vah: breakdown["POC/VA"] = breakdown.get("POC/VA", 0.0) - 0.10
                elif c15 <= val: breakdown["POC/VA"] = breakdown.get("POC/VA", 0.0) + 0.05
            else:
                if c15 <= val: breakdown["POC/VA"] = breakdown.get("POC/VA", 0.0) - 0.10
                elif c15 >= vah: breakdown["POC/VA"] = breakdown.get("POC/VA", 0.0) + 0.05

        # ML-lite (если включён)
        if TA_ML_LIGHT:
            feats = {
                "rvol": float(rvol_combo),
                "adx": float(adx(df15, 14).iloc[-1] / 50.0),  # нормировка
                "chop": float((chop15 or 50.0) / 100.0),
                "nn": float(details.get("nn_edge", 0.0) / 5.0),
                "btcdom": float(1.0 if details.get("btc_dom_trend") == "up" else (-1.0 if details.get("btc_dom_trend") == "down" else 0.0)),
                "breadth": float(details.get("breadth_pct50_1h", 50.0) / 100.0),
                "basis": float(details.get("basis", 0.0) * 100.0),
                "funding": float(details.get("funding_rate", 0.0) * 1000.0),
                "cvd": float(details.get("cvd_slope", 0.0) / 1e6),
                "l2": float(details.get("ob_imb", 0.0) or 0.0),
                "zz": 1.0 if details.get("zz_trend") in ("HH/HL", "LH/LL") else 0.0,
                "ob": 1.0 if details.get("ob_near") else 0.0,
                "fvg": 1.0 if details.get("fvg_near") else 0.0,
            }
            p = _ml_probability(feats, TA_ML_WEIGHTS, bias=TA_ML_BIAS)
            details["ml_p"] = float(p)
            # мягкий гейтинг
            if p < TA_ML_P_THRESHOLD and not relax:
                breakdown["MLgate"] = breakdown.get("MLgate", 0.0) - 0.25

        # Итоговый скор
        adj = float(sum(breakdown.values()))
        side_score = float(side_score + adj)

        # Адаптация TP/SL под SMC/POC
        entry = float(details.get("c5", c15))
        tps = [float(x) for x in details.get("tps", [])]
        sl_old = float(details.get("sl"))
        if TA_SMC and tps:
            cap_up = None; cap_dn = None
            if poc is not None and vah is not None and val is not None:
                if side == "LONG": cap_up = poc if c15 < poc else vah
                else: cap_dn = poc if c15 > poc else val
            ob = _find_order_block(df15, side, atr_fn=atr, atr_period=14)
            if ob:
                if side == "LONG":
                    cap_up = cap_up if cap_up is not None else ob["high"] + 0.25 * atr15
                else:
                    cap_dn = cap_dn if cap_dn is not None else ob["low"] - 0.25 * atr15
            if cap_up is not None and side == "LONG":
                tps = [min(tp, cap_up) for tp in tps]
            if cap_dn is not None and side == "SHORT":
                tps = [max(tp, cap_dn) for tp in tps]
            if ob:
                if side == "LONG":
                    sl_new = min(sl_old, ob["low"] - 0.15 * atr15)
                else:
                    sl_new = max(sl_old, ob["high"] + 0.15 * atr15)
                if tick:
                    sl_new = _round_to_tick(sl_new, tick, "floor" if side == "LONG" else "ceil")
                details["sl"] = float(sl_new)
            if tick:
                tps = [_round_to_tick(x, tick, "nearest") for x in tps]
            details["tps"] = tps
            details["tp_adjusted"] = True

        # Сохраняем
        details["rvol_combo"] = float(rvol_combo)
        details["chop"] = float(chop15) if chop15 is not None else None
        details["fdi"] = float(fdi) if fdi is not None else None
        details["score_breakdown"] = breakdown

        # Мягкий порог
        if side_score < TA_SCORE_MIN and not relax:
            side_score -= 0.05

        return side_score, side, details

    # ---------- Enhanced watch ----------
    async def _enhanced_watch_signal_price(bot, chat_id: int, sig):
        MET_TP1 = app.get("MET_TP1"); MET_TP2 = app.get("MET_TP2"); MET_TP3 = app.get("MET_TP3")
        MET_STOP = app.get("MET_STOP"); MET_BE = app.get("MET_BE"); MET_WATCH_ERR = app.get("MET_WATCH_ERR")
        db = app.get("db")
        _should_alert = app.get("_should_alert")
        _news_risk_trigger = app.get("_news_risk_trigger")
        active_watch_tasks = app.get("active_watch_tasks", {})

        last_trail_update = 0.0
        last_risk_check = 0.0
        last_latch_check = 0.0
        try:
            logger and logger.info("Мониторинг+: старт %s %s (до %s)", sig.symbol, sig.side, sig.watch_until.isoformat())
            while now_msk() < sig.watch_until and sig.active:
                price = market.fetch_mark_price(sig.symbol)
                if price is None:
                    await asyncio.sleep(10)
                    continue
                now_ts = pytime.time()

                if now_ts - last_risk_check > 60:
                    last_risk_check = now_ts
                    news_msg = _news_risk_trigger(sig.side, sig.news_note)
                    if news_msg and _should_alert(sig.id or -1, "news"):
                        await bot.send_message(chat_id, f"⚠️ Риск-алерт {sig.symbol.split('/')[0]}: {news_msg}\nРекомендация: сократить/закрыть позицию.")
                    tech_msg = _enhanced_tech_risk_trigger(sig.symbol, sig.side)
                    if tech_msg and _should_alert(sig.id or -1, "tech"):
                        await bot.send_message(chat_id, f"⚠️ Риск-алерт {sig.symbol.split('/')[0]}: {tech_msg}\nРекомендация: сократить/закрыть позицию.")

                if now_ts - last_latch_check > 45 and hasattr(sig, "tps") and sig.tps:
                    last_latch_check = now_ts
                    try:
                        df5 = market.fetch_ohlcv(sig.symbol, "5m", 220)
                        df15 = market.fetch_ohlcv(sig.symbol, "15m", 200)
                        if df5 is not None and df15 is not None:
                            rvol_now = _combined_rvol(df5["volume"], df15["volume"], lk5=64, lk15=64)
                            near_tp = any(abs(price - float(tp)) <= 0.2 * float(atr(df15, 14).iloc[-1]) for tp in sig.tps)
                            if near_tp and rvol_now < 0.9 and _should_alert(sig.id or -1, "latch"):
                                await bot.send_message(chat_id, f"ℹ️ {sig.symbol.split('/')[0]}: Цена пилит TP при падающем RVOL — подумайте о частичной фиксации.")
                    except Exception:
                        pass

                if sig.tp_hit >= 1 and sig.trailing and now_ts - last_trail_update > 60:
                    await _enhanced_update_trailing(sig)
                    last_trail_update = now_ts
                    if db: await db.update_signal(sig)

                if sig.side == "LONG":
                    if sig.tp_hit < 1 and price >= sig.tps[0]:
                        sig.tp_hit = 1; sig.sl = sig.entry; sig.trailing = True; sig.trail_mode = "supertrend"
                        if MET_TP1: MET_TP1.inc()
                        await bot.send_message(chat_id, f"✅ TP1 по {sig.symbol.split('/')[0]} ({format_price(price)}). Стоп в БУ ({format_price(sig.sl)}), трейлинг включён.")
                        if db: await db.update_signal(sig)
                    if sig.tp_hit < 2 and price >= sig.tps[1]:
                        sig.tp_hit = 2
                        if MET_TP2: MET_TP2.inc()
                        await bot.send_message(chat_id, f"✅ TP2 по {sig.symbol.split('/')[0]} ({format_price(price)}).")
                        if db: await db.update_signal(sig)
                    if sig.tp_hit < 3 and price >= sig.tps[2]:
                        sig.tp_hit = 3; sig.active = False
                        if MET_TP3: MET_TP3.inc()
                        await bot.send_message(chat_id, f"🎯 TP3 по {sig.symbol.split('/')[0]} ({format_price(price)}). Сигнал завершён.")
                        if db: await db.update_signal(sig)
                        break
                    if price <= sig.sl:
                        sig.active = False
                        if sig.tp_hit >= 1 and sig.sl >= sig.entry:
                            if MET_BE: MET_BE.inc()
                            await bot.send_message(chat_id, f"🟨 Закрыто по БУ {sig.symbol.split('/')[0]}.")
                        else:
                            if MET_STOP: MET_STOP.inc()
                            await bot.send_message(chat_id, f"🛑 Стоп по {sig.symbol.split('/')[0]} ({format_price(price)}).")
                        if db: await db.update_signal(sig)
                        break
                else:
                    if sig.tp_hit < 1 and price <= sig.tps[0]:
                        sig.tp_hit = 1; sig.sl = sig.entry; sig.trailing = True; sig.trail_mode = "supertrend"
                        if MET_TP1: MET_TP1.inc()
                        await bot.send_message(chat_id, f"✅ TP1 по {sig.symbol.split('/')[0]} ({format_price(price)}). Стоп в БУ ({format_price(sig.sl)}), трейлинг включён.")
                        if db: await db.update_signal(sig)
                    if sig.tp_hit < 2 and price <= sig.tps[1]:
                        sig.tp_hit = 2
                        if MET_TP2: MET_TP2.inc()
                        await bot.send_message(chat_id, f"✅ TP2 по {sig.symbol.split('/')[0]} ({format_price(price)}).")
                        if db: await db.update_signal(sig)
                    if sig.tp_hit < 3 and price <= sig.tps[2]:
                        sig.tp_hit = 3; sig.active = False
                        if MET_TP3: MET_TP3.inc()
                        await bot.send_message(chat_id, f"🎯 TP3 по {sig.symbol.split('/')[0]} ({format_price(price)}). Сигнал завершён.")
                        if db: await db.update_signal(sig)
                        break
                    if price >= sig.sl:
                        sig.active = False
                        if sig.tp_hit >= 1 and sig.sl <= sig.entry:
                            if MET_BE: MET_BE.inc()
                            await bot.send_message(chat_id, f"🟨 Закрыто по БУ {sig.symbol.split('/')[0]}.")
                        else:
                            if MET_STOP: MET_STOP.inc()
                            await bot.send_message(chat_id, f"🛑 Стоп по {sig.symbol.split('/')[0]} ({format_price(price)}).")
                        if db: await db.update_signal(sig)
                        break

                await asyncio.sleep(10)

            if now_msk() >= sig.watch_until and sig.active:
                sig.active = False
                await bot.send_message(chat_id, f"⏱ Мониторинг по {sig.symbol.split('/')[0]} завершён по времени.")
                if db: await db.update_signal(sig)
        except Exception:
            if app.get("MET_WATCH_ERR"): app["MET_WATCH_ERR"].inc()
            logger and logger.exception("Мониторинг+: ошибка")
            try:
                await bot.send_message(chat_id, "⚠️ Ошибка мониторинга сигнала.")
            except Exception:
                pass
        finally:
            try:
                tasks = active_watch_tasks.get(sig.user_id, [])
                active_watch_tasks[sig.user_id] = [t for t in tasks if not t.done()]
            except Exception:
                pass

    # Применяем патчи
    app["score_symbol_core"] = _enhanced_score_symbol_core
    app["build_reason"] = _enhanced_reason
    app["update_trailing"] = _enhanced_update_trailing
    app["_tech_risk_trigger"] = _enhanced_tech_risk_trigger
    try:
        app["watch_signal_price"] = _enhanced_watch_signal_price
    except Exception:
        pass

    logger and logger.info("TA patch Iteration 3 подключён: Orderflow/Derivatives/VolumeProfile/Adaptive/Regimes/ML/BTC.D/Breadth/Alerts.")
