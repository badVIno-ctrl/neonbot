# new.py
# Ultra-TA finalizer patch: microstructure/regimes/SMC/RS/breadth/news + final SL/TP sanity
# Works as an add-on via patch(app), no edits in other files.
# Load last:
#   TA_PATCH_MODULES=main,lock,chat,neon,ta,tacoin,new

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
import sys
import os
import math
import asyncio
import contextlib
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# ============================ ENV FLAGS / WEIGHTS ============================
NEW_LOOP_GUARD     = os.getenv("NEW_LOOP_GUARD", "1") == "1"     # idempotent loop guards for channel/admin loops
NEW_CITY_FIX       = os.getenv("NEW_CITY_FIX", "1") == "1"       # ensure users.city before loops
NEW_SCORE_ENGINE   = os.getenv("NEW_SCORE_ENGINE", "1") == "1"   # TA engine wrap (microstructure/regimes/SMC/etc)
NEW_REASON_WRAP    = os.getenv("NEW_REASON_WRAP", "1") == "1"    # add 'Качество:' only if not present
NEW_TRAILING_WRAP  = os.getenv("NEW_TRAILING_WRAP", "1") == "1"  # dynamic TP tune + regime-aware BE
NEW_WATCH_SIDECAR  = os.getenv("NEW_WATCH_SIDECAR", "1") == "1"  # microstructure alerts
NEW_TA_REPORT_WRAP = os.getenv("NEW_TA_REPORT_WRAP", "1") == "1" # extra context in TA report (/tacoin)
NEW_FINAL_SIG_FIX  = os.getenv("NEW_FINAL_SIG_FIX", "1") == "1"  # FINAL SL/TP sanity via format_signal_message wrap

# Microstructure
NEW_OB_SPOOF     = os.getenv("NEW_OB_SPOOF", "1") == "1"
NEW_OFI_VPIN     = os.getenv("NEW_OFI_VPIN", "1") == "1"
NEW_MICRO_GRAD   = os.getenv("NEW_MICRO_GRAD", "1") == "1"  # slope/microprice drift

# Regimes / filters
NEW_HMM          = os.getenv("NEW_HMM", "1") == "1"
NEW_ALIGN_MTF    = os.getenv("NEW_ALIGN_MTF", "1") == "1"
NEW_BREADTH      = os.getenv("NEW_BREADTH", "1") == "1"
NEW_RS           = os.getenv("NEW_RS", "1") == "1"
NEW_BTC_DOM      = os.getenv("NEW_BTC_DOM", "1") == "1"  # оффлайн-грубая оценка, без HTTP

# Structure / SMC
NEW_SMC_STRICT   = os.getenv("NEW_SMC_STRICT", "1") == "1"
NEW_EQ_TOL       = float(os.getenv("NEW_EQ_TOL", "0.0008"))

# News probabilistic tweak (без HTTP — только лог по уже имеющемуся note)
NEW_NEWS_LOG     = os.getenv("NEW_NEWS_LOG", "1") == "1"

# Dynamic TP/BE tuning
NEW_TP_TUNE      = os.getenv("NEW_TP_TUNE", "1") == "1"
NEW_BE_REGIME    = os.getenv("NEW_BE_REGIME", "1") == "1"

# Weights (soft nudges)
W_ALIGN    = float(os.getenv("NEW_W_ALIGN", "0.14"))
W_OFI      = float(os.getenv("NEW_W_OFI", "0.11"))
W_VPIN     = float(os.getenv("NEW_W_VPIN", "0.10"))
W_SPOOF    = float(os.getenv("NEW_W_SPOOF", "-0.15"))
W_SPREAD   = float(os.getenv("NEW_W_SPREAD", "-0.10"))
W_CHURN    = float(os.getenv("NEW_W_CHURN", "-0.08"))
W_GRAD     = float(os.getenv("NEW_W_GRAD", "0.08"))
W_HMM      = float(os.getenv("NEW_W_HMM", "0.12"))
W_RS       = float(os.getenv("NEW_W_RS", "0.10"))
W_BREADTH  = float(os.getenv("NEW_W_BREADTH", "0.10"))
W_BTC_DOM  = float(os.getenv("NEW_W_BTC_DOM", "-0.08"))
W_SMC      = float(os.getenv("NEW_W_SMC", "0.14"))
W_NEWS     = float(os.getenv("NEW_W_NEWS", "0.08"))

# Thresholds / params
OB_CACHE_SEC     = float(os.getenv("NEW_OB_CACHE_SEC", "8.0"))
OFI_WIN          = int(os.getenv("NEW_OFI_WIN", "60"))           # bars for OFI
VPIN_BUCKETS     = int(os.getenv("NEW_VPIN_BUCKETS", "20"))
SPREAD_Z_MAX     = float(os.getenv("NEW_SPREAD_Z_MAX", "3.0"))
CHURN_MAX        = float(os.getenv("NEW_CHURN_MAX", "1.8"))
SPOOF_K          = float(os.getenv("NEW_SPOOF_K", "3.0"))
SPOOF_DROP       = float(os.getenv("NEW_SPOOF_DROP", "0.60"))
NEAR_PCT         = float(os.getenv("NEW_NEAR_PCT", "0.0010"))    # ±0.10% of mid
ALIGN_PENALTY    = float(os.getenv("NEW_ALIGN_PENALTY", "0.15"))
BARRIER_ATR      = float(os.getenv("NEW_BARRIER_ATR", "0.20"))   # TP tune if barrier within N*ATR
RVOL_SPIKE       = float(os.getenv("NEW_RVOL_SPIKE", "1.25"))

# Final signal sanity defaults (if app's env missing)
SAN_SL_MIN_ATR   = float(os.getenv("TA_SAN_SL_MIN_ATR", "0.25"))
SAN_SL_MAX_ATR   = float(os.getenv("TA_SAN_SL_MAX_ATR", "5.0"))
SAN_TP_MIN_ATR   = float(os.getenv("TA_SAN_TP_MIN_STEP_ATR", "0.15"))
TP_CAP_ATR_MAX   = float(os.getenv("TA_TP_CAP_ATR_MAX", "4.0"))
TP_CAP_PDR_MULT  = float(os.getenv("TA_TP_CAP_PDR_MULT", "1.2"))
TP_MAX_PCT       = float(os.getenv("TAC_TP_MAX_PCT", "0.25"))    # cap by percentage from entry

BE_MIN           = float(os.getenv("NEW_BE_MIN_ATR", "1.0"))
BE_MAX           = float(os.getenv("NEW_BE_MAX_ATR", "1.5"))

SAFE_DEC_MIN     = int(os.getenv("TAC_SAFE_DECIMALS_MIN", "6"))
SAFE_DEC_MAX     = int(os.getenv("TAC_SAFE_DECIMALS_MAX", "8"))

# ============================ UTILS ============================
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if (v == v) else float(default)
    except Exception:
        return float(default)

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _tanh_clip(x: float, s: float = 1.0) -> float:
    try:
        return float(np.tanh(x / max(1e-9, s)))
    except Exception:
        return 0.0

# ============================ LOOP GUARDS / CITY FIX ============================
def _wrap_loop_once(logger, key_name: str, orig_fn):
    async def wrapper(app: Dict[str, Any], bot, *args, **kwargs):
        once = app.setdefault("_new_once", {"loops": set()})
        if key_name in once["loops"]:
            logger and logger.info("new: loop '%s' start skipped (already running).", key_name)
            return
        once["loops"].add(key_name)
        return await orig_fn(app, bot, *args, **kwargs)
    setattr(wrapper, "_new_wrapped", True)
    return wrapper

async def _ensure_city_column(app: Dict[str, Any]):
    db = app.get("db")
    if not db or not getattr(db, "conn", None):
        return
    with contextlib.suppress(Exception):
        await db.conn.execute("ALTER TABLE users ADD COLUMN city TEXT")
        await db.conn.commit()

# ============================ MICROSTRUCTURE ============================
_ORDERBOOK_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

def _ob_cache_key(ex_id: str, resolved: str) -> str:
    return f"{ex_id}:{resolved}"

def _fetch_order_book(app: Dict[str, Any], symbol: str, depth: int = 25) -> Optional[Dict[str, Any]]:
    market = app.get("market")
    last_err = None
    for name, ex in market._available_exchanges():
        resolved = market.resolve_symbol(ex, symbol) or symbol
        if resolved not in ex.markets:
            continue
        key = _ob_cache_key(ex.id, resolved)
        ts_now = time.time()
        cached = _ORDERBOOK_CACHE.get(key)
        if cached and ts_now - cached[0] < OB_CACHE_SEC:
            return cached[1]
        try:
            ob = ex.fetch_order_book(resolved, limit=depth)
            if isinstance(ob, dict):
                _ORDERBOOK_CACHE[key] = (ts_now, ob)
                return ob
        except Exception as e:
            last_err = e
            continue
    app.get("logger") and app["logger"].debug("new: orderbook fetch failed for %s: %s", symbol, last_err)
    return None

def _book_features(ob: Dict[str, Any]) -> Dict[str, float]:
    out = {"spread_norm": None, "book_imb": None, "near_depth_ratio": None,
           "churn": None, "spoof_score": None, "slope": None, "microprice_drift": None}
    try:
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        if not bids or not asks:
            return out
        pb, qb = float(bids[0][0]), float(bids[0][1])
        pa, qa = float(asks[0][0]), float(asks[0][1])
        mid = 0.5 * (pa + pb)
        spread = max(0.0, pa - pb)
        denom = max(1e-9, (mid * 0.001))
        out["spread_norm"] = float(spread / denom)

        sb = float(sum([float(q) for _, q in bids[:25]]))
        sa = float(sum([float(q) for _, q in asks[:25]]))
        if sb + sa > 0:
            out["book_imb"] = float((sb - sa) / (sb + sa))

        def near_ratio(levels):
            keep = 0.0
            tot = 0.0
            for p, q in levels[:25]:
                p = float(p); q = float(q)
                tot += q
                if abs(p - mid) / (mid + 1e-9) <= NEAR_PCT:
                    keep += q
            return float(keep / (tot + 1e-9))
        out["near_depth_ratio"] = float(0.5 * (near_ratio(bids) + near_ratio(asks)))

        # churn/spoof
        key = f"default:{mid:.8f}"
        prev = _ORDERBOOK_CACHE.get(key)
        snap = {"bids": [(float(p), float(q)) for p, q in bids[:10]],
                "asks": [(float(p), float(q)) for p, q in asks[:10]]}
        _ORDERBOOK_CACHE[key] = (time.time(), snap)
        if prev and isinstance(prev, tuple) and isinstance(prev[1], dict):
            prv = prev[1]
            def churn_side(cur, prv):
                m = min(len(cur), len(prv))
                if m == 0: return 0.0
                diffs = []
                for i in range(m):
                    qc = float(cur[i][1]); qp = float(prv[i][1]) + 1e-9
                    diffs.append(abs(qc - qp) / qp)
                return float(np.mean(diffs))
            churn_b = churn_side(snap["bids"], prv.get("bids", []))
            churn_a = churn_side(snap["asks"], prv.get("asks", []))
            out["churn"] = float(0.5 * (churn_b + churn_a))

            def max_wall(levels):
                if not levels: return 0.0
                sizes = np.array([float(q) for _, q in levels], dtype=float)
                med = float(np.median(sizes) + 1e-9)
                mx = float(np.max(sizes))
                return float(mx / max(1e-9, med))
            cur_bk = max_wall(snap["bids"]); prv_bk = max_wall(prv.get("bids", []))
            cur_ak = max_wall(snap["asks"]); prv_ak = max_wall(prv.get("asks", []))
            spoof_b = 1.0 if (prv_bk >= SPOOF_K and (cur_bk / max(1e-9, prv_bk)) <= (1.0 - SPOOF_DROP)) else 0.0
            spoof_a = 1.0 if (prv_ak >= SPOOF_K and (cur_ak / max(1e-9, prv_ak)) <= (1.0 - SPOOF_DROP)) else 0.0
            out["spoof_score"] = float(spoof_b + spoof_a)

        # liquidity slope + microprice drift
        def liquidity_slope(levels, sign: int) -> float:
            if not levels:
                return 0.0
            xs = np.arange(min(10, len(levels)), dtype=float)
            ys = np.array([float(q) for _, q in levels[:len(xs)]], dtype=float)
            if np.std(xs) < 1e-12:
                return 0.0
            a, _b = np.polyfit(xs, ys, 1)
            return float(sign * a)
        slope = liquidity_slope(asks, +1) + liquidity_slope(bids, -1)
        out["slope"] = float(slope)

        mp = (pa * qb + pb * qa) / (qa + qb + 1e-9)
        out["microprice_drift"] = float((mp - mid) / (mid + 1e-9))
    except Exception:
        pass
    return out

def _ofi_vpin(df: Optional[pd.DataFrame]) -> Tuple[Optional[float], Optional[float]]:
    try:
        if df is None or len(df) < max(OFI_WIN + 5, VPIN_BUCKETS + 5):
            return None, None
        c = df["close"].astype(float).values
        v = df["volume"].astype(float).values
        ret = np.diff(c, prepend=c[0])
        sign = np.sign(ret)
        ofi = float(np.sum(sign[-OFI_WIN:] * v[-OFI_WIN:]))
        tot = float(np.sum(v[-VPIN_BUCKETS:]) + 1e-9)
        pos = float(np.sum(v[-VPIN_BUCKETS:][ret[-VPIN_BUCKETS:] >= 0.0]))
        vpin = abs(pos - (tot - pos)) / tot
        return ofi, float(vpin)
    except Exception:
        return None, None

# ============================ REGIMES / INDICATORS ============================
def _adx(df: Optional[pd.DataFrame], period: int = 14) -> Optional[float]:
    try:
        if df is None: return None
        high = df["high"]; low = df["low"]; close = df["close"]
        up_move = high.diff(); down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_ = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / (atr_ + 1e-9)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / (atr_ + 1e-9)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        return float(adx.iloc[-1])
    except Exception:
        return None

def _bbwp(series: Optional[pd.Series], length: int = 20, lookback: int = 96) -> Optional[float]:
    try:
        if series is None: return None
        if len(series) < max(length, lookback) + 5:
            return None
        basis = series.rolling(length).mean()
        dev = series.rolling(length).std(ddof=0)
        bbw = (series - basis + 2 * dev) / (4 * dev + 1e-9)
        bbwp = bbw.rolling(lookback).rank(pct=True) * 100.0
        return float(np.clip(bbwp.iloc[-1], 0.0, 100.0))
    except Exception:
        return None

def _hmm_regime(df: Optional[pd.DataFrame]) -> str:
    try:
        if df is None or len(df) < 120:
            return "unknown"
        close = df["close"].astype(float)
        ret = close.pct_change().dropna()
        vs = float(ret.tail(24).std(ddof=0) + 1e-12)
        vl = float(ret.tail(96).std(ddof=0) + 1e-12)
        ratio = vs / vl
        if ratio >= 1.8:
            return "shock"
        if ratio <= 0.8:
            return "calm"
        return "trend" if ratio > 1.1 else "range"
    except Exception:
        return "unknown"

def _alignment_penalty(details: Dict[str, Any], side: str) -> float:
    c4 = bool(details.get("cond4h_up"))
    c1 = bool(details.get("cond1h_up"))
    c15= bool(details.get("cond15_up"))
    votes = (c4 + c1 + c15) if side == "LONG" else ((not c4) + (not c1) + (not c15))
    if votes <= 1:
        return -ALIGN_PENALTY
    elif votes == 2:
        return +0.05
    return +0.08

# ============================ SMC STRICT ============================
def _improved_fvg_ob(df: Optional[pd.DataFrame], atr_val: float) -> Tuple[bool, bool]:
    try:
        if df is None or len(df) < 60 or atr_val <= 0:
            return False, False
        o = df["open"].astype(float).values
        h = df["high"].astype(float).values
        l = df["low"].astype(float).values
        c = df["close"].astype(float).values
        strong_fvg = False
        for i in range(len(df) - 10, len(df) - 1):
            rng = h[i] - l[i]
            body = abs(c[i] - o[i])
            if l[i+1] > h[i-1] and (body >= 0.8*atr_val) and (rng >= 1.2*atr_val):
                strong_fvg = True; break
            if h[i+1] < l[i-1] and (body >= 0.8*atr_val) and (rng >= 1.2*atr_val):
                strong_fvg = True; break
        strong_ob = False
        for i in range(len(df) - 12, len(df) - 4):
            rng = h[i] - l[i]
            body = abs(c[i] - o[i])
            if body >= 0.75 * rng and rng >= 1.1*atr_val:
                lo = min(o[i-1], c[i-1]); hi = max(o[i-1], c[i-1])
                z = df.iloc[i+1:i+8]
                if not z.empty and (((z["low"] <= hi) & (z["high"] >= lo)).any()):
                    strong_ob = True; break
        return strong_fvg, strong_ob
    except Exception:
        return False, False

def _bos_choch_strict(df: Optional[pd.DataFrame], lookback: int = 80, k: float = 1.2) -> Tuple[int, bool]:
    try:
        if df is None or len(df) < lookback + 10:
            return 0, False
        x = df.tail(lookback)
        high = x["high"].astype(float).values
        low  = x["low"].astype(float).values
        close= x["close"].astype(float).values
        atrv = float(((x["high"] - x["low"]).rolling(14).mean()).iloc[-1] or 0.0)
        if atrv <= 0: return 0, False
        H = float(np.max(high[:-1])); L = float(np.min(low[:-1]))
        rng = x["high"].iloc[-1] - x["low"].iloc[-1]
        body = abs(x["close"].iloc[-1] - x["open"].iloc[-1])
        retest = False
        if close[-1] > H and (rng >= k*atrv) and (body >= 0.6*rng):
            retest = bool((x["low"].tail(4).min() <= H) and (x["close"].iloc[-1] > H))
            return 1, retest
        if close[-1] < L and (rng >= k*atrv) and (body >= 0.6*rng):
            retest = bool((x["high"].tail(4).max() >= L) and (x["close"].iloc[-1] < L))
            return -1, retest
        return 0, False
    except Exception:
        return 0, False

# ============================ RS / BREADTH / DOMINANCE ============================
_BREADTH_CACHE_NEW: Dict[str, Any] = {"ts": 0.0, "data": None}

def _compute_breadth(app: Dict[str, Any], ttl_sec: int = 300) -> Dict[str, Any]:
    ts = time.time()
    if _BREADTH_CACHE_NEW["data"] is not None and ts - _BREADTH_CACHE_NEW["ts"] < ttl_sec:
        return _BREADTH_CACHE_NEW["data"]
    try:
        market = app.get("market")
        ema_fn = app.get("ema")
        symbols = app.get("SYMBOLS", [])
        above50_1h = above200_1h = above50_4h = above200_4h = total = 0
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
            c1 = float(df1h["close"].iloc[-1]); c4 = float(df4h["close"].iloc[-1])
            if c1 > float(df1h["ema50"].iloc[-1]):  above50_1h += 1
            if c1 > float(df1h["ema200"].iloc[-1]): above200_1h += 1
            if c4 > float(df4h["ema50"].iloc[-1]):  above50_4h += 1
            if c4 > float(df4h["ema200"].iloc[-1]): above200_4h += 1
        data = {
            "total": total,
            "pct50_1h": (above50_1h / total * 100.0) if total else 0.0,
            "pct200_1h": (above200_1h / total * 100.0) if total else 0.0,
            "pct50_4h": (above50_4h / total * 100.0) if total else 0.0,
            "pct200_4h": (above200_4h / total * 100.0) if total else 0.0,
        }
        _BREADTH_CACHE_NEW.update({"ts": ts, "data": data})
        return data
    except Exception:
        return {"total": 0}

def _estimate_btc_dom_trend(app: Dict[str, Any], win: int = 72) -> Optional[str]:
    """
    Оценка «тренда доминации» BTC без HTTP:
    сравниваем склон (slope) лог-отношения BTC к «альт индексу» за окно win*5m (~6h).
    """
    try:
        market = app.get("market")
        symbols = [s for s in app.get("SYMBOLS", []) if isinstance(s, str)]
        alts = [s for s in symbols if not s.startswith("BTC/") and not s.startswith("ETH/")]
        btc = market.fetch_ohlcv("BTC/USDT", "5m", max(win + 10, 200))
        if btc is None or len(btc) < win + 10 or not alts:
            return None
        # построим «альт-индекс» как средний лог-цен
        alt_closes: List[np.ndarray] = []
        count = 0
        for sym in alts[:20]:  # ограничим до 20, чтобы не грузить
            df = market.fetch_ohlcv(sym, "5m", max(win + 10, 200))
            if df is not None and len(df) >= win + 10:
                alt_closes.append(df["close"].astype(float).tail(win).values)
                count += 1
        if count == 0:
            return None
        btc_y = btc["close"].astype(float).tail(win).values
        alt_mat = np.vstack([x for x in alt_closes if len(x) == win])
        alt_idx = np.nanmean(alt_mat, axis=0)
        y = np.log((btc_y + 1e-9) / (alt_idx + 1e-9))
        x = np.arange(len(y), dtype=float)
        if np.std(x) < 1e-9:
            return None
        sl, _ = np.polyfit(x, y, 1)
        if sl > 0.0:
            return "up"
        if sl < 0.0:
            return "down"
        return "flat"
    except Exception:
        return None

# ============================ NEWS PROBABILITY ============================
def _news_logistic_p(note: str) -> Optional[float]:
    try:
        t = (note or "").lower()
        if not t: return None
        pos_kw = ["listing","approval","approve","etf","funding","partnership","mainnet","burn","integration","adoption","рост","inflow","upgrade","launch","support"]
        neg_kw = ["hack","exploit","rug","lawsuit","ban","sec","delist","outage","breach","dump","sell-off","падение","crackdown","security"]
        x = 0.0
        for k in pos_kw:
            if k in t: x += 1.0
        for k in neg_kw:
            if k in t: x -= 1.0
        p = _sigmoid(0.9 * x)
        return float(p)
    except Exception:
        return None

# ============================ TP/SL UTILITIES ============================
def _infer_safe_tick(entry: float, df15: Optional[pd.DataFrame]) -> float:
    try:
        if entry <= 0:
            return 10 ** (-SAFE_DEC_MIN)
        if entry >= 100: decimals = 2
        elif entry >= 10: decimals = 3
        elif entry >= 1: decimals = 4
        elif entry >= 0.1: decimals = 5
        else: decimals = SAFE_DEC_MIN
        decimals = int(max(2, min(SAFE_DEC_MAX, decimals)))
        tick = 10 ** (-decimals)
        if df15 is not None and len(df15) >= 20:
            med_rng = float((df15["high"] - df15["low"]).tail(20).median())
            est_spread = med_rng * 0.1
            if est_spread > 0:
                tick = min(tick, est_spread / 10.0)
        return float(max(tick, 10 ** (-SAFE_DEC_MAX)))
    except Exception:
        return 10 ** (-SAFE_DEC_MIN)

def _round_tick(x: float, tick: float, mode: str) -> float:
    try:
        n = float(x) / float(tick if tick>0 else 1e-9)
        if mode == "floor": n = math.floor(n)
        elif mode == "ceil": n = math.ceil(n)
        else: n = round(n)
        return float(n * float(tick if tick>0 else 1e-9))
    except Exception:
        return float(x)

def _compute_pdh_pdl(df15: Optional[pd.DataFrame]) -> Optional[Tuple[float, float]]:
    try:
        if df15 is None or len(df15) < 60: return None
        ts = df15["ts"]
        last_ts = pd.to_datetime(ts.iloc[-1], utc=True, errors="coerce").to_pydatetime()
        day_anchor = last_ts.replace(hour=0, minute=0, second=0, microsecond=0)
        prev_day_start = day_anchor - timedelta(days=1)
        prev = df15[(ts >= prev_day_start) & (ts < day_anchor)]
        if prev is None or prev.empty: return None
        pdh = float(prev["high"].max()); pdl = float(prev["low"].min())
        if not (math.isfinite(pdh) and math.isfinite(pdl) and pdh > pdl): return None
        return pdh, pdl
    except Exception:
        return None

def _ensure_tp_monotonic_with_step(side: str, entry: float, tps: List[float], atr: float, tick: float, step_atr: float) -> List[float]:
    tps = [float(x) for x in (tps[:3] if tps else [])]
    if len(tps) < 3:
        while len(tps) < 3: tps.append(tps[-1] if tps else entry)
    step = max(1e-9, float(step_atr) * float(atr))
    if side == "LONG":
        tps = sorted(tps)
        tps[0] = _round_tick(max(tps[0], entry + step), tick, "ceil")
        tps[1] = _round_tick(max(tps[1], tps[0] + step), tick, "ceil")
        tps[2] = _round_tick(max(tps[2], tps[1] + step), tick, "ceil")
    else:
        tps = sorted(tps, reverse=True)
        tps[0] = _round_tick(min(tps[0], entry - step), tick, "floor")
        tps[1] = _round_tick(min(tps[1], tps[0] - step), tick, "floor")
        tps[2] = _round_tick(min(tps[2], tps[1] - step), tick, "floor")
    return tps[:3]

def _cap_tp(entry: float, tp: float, atr: float, side: str, tp_max_pct: float, tp_cap_atr_max: float) -> float:
    if side == "LONG":
        cap1 = entry * (1.0 + tp_max_pct)
        cap2 = entry + tp_cap_atr_max * atr
        return float(min(tp, cap1, cap2))
    else:
        cap1 = entry * (1.0 - tp_max_pct)
        cap2 = entry - tp_cap_atr_max * atr
        return float(max(tp, cap1, cap2))

def _cap_tp_by_pdr(entry: float, tp: float, side: str, pdh_pdl: Optional[Tuple[float,float]]) -> float:
    try:
        if not pdh_pdl: return tp
        pdh, pdl = pdh_pdl
        dr = abs(pdh - pdl)
        if dr <= 0: return tp
        cap = entry + (TP_CAP_PDR_MULT * dr) * (1 if side=="LONG" else -1)
        return float(min(tp, cap) if side=="LONG" else max(tp, cap))
    except Exception:
        return tp

def _final_signal_sanity(app: Dict[str, Any], sig) -> Tuple[bool, List[str]]:
    notes: List[str] = []
    changed = False
    try:
        market = app.get("market")
        atr_fn = app.get("atr")
        df15 = market.fetch_ohlcv(sig.symbol, "15m", 240)
        entry = float(sig.entry); sl = float(sig.sl)
        tps = [float(x) for x in list(sig.tps or [])]
        if df15 is None or len(df15) < 30 or entry <= 0:
            return False, notes
        atrv = float(atr_fn(df15, 14).iloc[-1]) if callable(atr_fn) else float(((df15["high"]-df15["low"]).rolling(14).mean()).iloc[-1])
        if not (atrv > 0):
            return False, notes
        tick_real = 0.0
        try:
            tick_real = float(market.get_tick_size(sig.symbol) or 0.0)
        except Exception:
            tick_real = 0.0
        safe_tick = _infer_safe_tick(entry, df15)
        tick = float(min(safe_tick if safe_tick>0 else 1e-9, tick_real if tick_real>0 else safe_tick))
        a_min = float(app.get("TA_SAN_SL_MIN_ATR", SAN_SL_MIN_ATR))
        a_max = float(app.get("TA_SAN_SL_MAX_ATR", SAN_SL_MAX_ATR))
        step_atr = float(app.get("TA_SAN_TP_MIN_STEP_ATR", SAN_TP_MIN_ATR))
        cap_atr = float(app.get("TA_TP_CAP_ATR_MAX", TP_CAP_ATR_MAX))
        pct_cap = float(os.getenv("TAC_TP_MAX_PCT", str(TP_MAX_PCT)))
        if sig.side == "LONG":
            if not (sl < entry): sl = entry - a_min*atrv; changed = True; notes.append("SL side")
        else:
            if not (sl > entry): sl = entry + a_min*atrv; changed = True; notes.append("SL side")
        risk = abs(entry - sl)
        if risk < a_min*atrv:
            sl = entry - a_min*atrv if sig.side=="LONG" else entry + a_min*atrv
            changed = True; notes.append("SL<minATR")
        elif risk > a_max*atrv:
            sl = entry - a_max*atrv if sig.side=="LONG" else entry + a_max*atrv
            changed = True; notes.append("SL>maxATR")
        sl = _round_tick(sl, tick, "floor" if sig.side=="LONG" else "ceil")
        if sl <= 0:
            sl = _round_tick(entry - a_min*atrv if sig.side=="LONG" else entry + a_min*atrv, tick, "floor" if sig.side=="LONG" else "ceil")
            changed = True; notes.append("SL zero-guard")

        if not tps:
            muls = [0.8, 1.5, 2.4]
            tps = [entry + m*atrv if sig.side=="LONG" else entry - m*atrv for m in muls]

        pd = _compute_pdh_pdl(df15)
        tps0 = list(tps)
        capped = []
        for tp in tps:
            tp1 = _cap_tp(entry, tp, atrv, sig.side, pct_cap, cap_atr)
            tp2 = _cap_tp_by_pdr(entry, tp1, sig.side, pd)
            capped.append(tp2)
        if capped != tps0:
            tps = capped; changed = True; notes.append("TP cap ATR/PDR")

        tps = _ensure_tp_monotonic_with_step(sig.side, entry, tps, atrv, tick, step_atr)

        if len({round(x, 12) for x in tps}) < 3:
            if sig.side == "LONG":
                tps[1] = max(tps[1], tps[0] + step_atr*atrv)
                tps[2] = max(tps[2], tps[1] + step_atr*atrv)
            else:
                tps[1] = min(tps[1], tps[0] - step_atr*atrv)
                tps[2] = min(tps[2], tps[1] - step_atr*atrv)
            tps = _ensure_tp_monotonic_with_step(sig.side, entry, tps, atrv, tick, step_atr)
            changed = True; notes.append("TP de-dup")

        tps = [ _round_tick(x, tick, "ceil" if sig.side=="LONG" else "floor") for x in tps ]

        if float(sig.sl) != float(sl) or [float(x) for x in sig.tps] != [float(x) for x in tps]:
            sig.sl = float(sl)
            sig.tps = [float(x) for x in tps]
            changed = True

    except Exception:
        pass
    return changed, notes

# ============================ GRADE LINE ============================
def _grade(exec_spread, near_ratio, spoof, churn, adx, hmm, vpin) -> Tuple[str, str, str, str, str]:
    def g_exec():
        if exec_spread is None or near_ratio is None: return "C"
        if exec_spread <= SPREAD_Z_MAX and near_ratio >= 0.55: return "A"
        if exec_spread <= SPREAD_Z_MAX*1.3 and near_ratio >= 0.40: return "B"
        return "C"
    def g_break():
        if spoof and spoof >= 1.0: return "C"
        if churn and churn > CHURN_MAX: return "C"
        return "B"
    def g_reg():
        if adx and adx >= 30: return "A"
        if adx and adx >= 20: return "B"
        return "C"
    def g_liq():
        if vpin is None: return "C"
        if vpin <= 0.35: return "A"
        if vpin <= 0.5: return "B"
        return "C"
    def g_hmm():
        return {"trend":"A","range":"B","calm":"A","shock":"C"}.get(hmm or "range","B")
    return g_exec(), g_break(), g_reg(), g_liq(), g_hmm()

def _append_quality_if_absent(text: str, d: Dict[str, Any]) -> str:
    if "Качество:" in (text or ""):
        return text
    exec_spread = d.get("spread_norm")
    near_ratio  = d.get("near_depth_ratio")
    spoof       = d.get("spoof_score")
    churn       = d.get("churn")
    adx         = d.get("adx15") or d.get("ta_adx15_alt")
    hmm         = d.get("regime_hmm")
    vpin        = d.get("vpin")
    q1,q2,q3,q4,q5 = _grade(exec_spread, near_ratio, spoof, churn, adx, hmm, vpin)
    lm = d.get("ta_liq_summary") or d.get("liq_summary") or ""
    line = f"Качество: Exec {q1} • Break {q2} • Reg {q3} • Liq {q4} • HMM {q5}" + (f" • LM: {lm}" if lm else "")
    return (text + (" • " if text else "") + line)[:1100]

# ============================ HMM-BE & TP TUNE ============================
def _tp_dynamic_adjust(app: Dict[str, Any], symbol: str, side: str, tps: List[float], entry: float, atr_val: float) -> List[float]:
    try:
        market = app.get("market")
        df15 = market.fetch_ohlcv(symbol, "15m", 240)
        if df15 is None or len(df15) < 60:
            return tps
        x = df15.tail(240).copy()
        tp = (x["high"] + x["low"] + x["close"]) / 3.0
        vol = x["volume"].astype(float)
        prices = tp.values.astype(float)
        weights = vol.values
        lo = float(np.min(prices)); hi = float(np.max(prices))
        hist, edges = np.histogram(prices, bins=40, range=(lo, hi), weights=weights)
        if hist.sum() <= 0:
            return tps
        poc_idx = int(np.argmax(hist))
        poc = float(0.5*(edges[poc_idx] + edges[poc_idx+1]))
        order = np.argsort(hist)[::-1]
        acc=0.0; total=float(hist.sum())
        mask=np.zeros_like(hist, dtype=bool)
        for idx in order:
            mask[idx]=True; acc+=float(hist[idx])
            if acc/total >= 0.68: break
        sel = np.where(mask)[0]
        val = float(edges[sel.min()]); vah=float(edges[sel.max()+1])
        rv = float((x["volume"].tail(64).iloc[-1]) / (x["volume"].tail(64).median()+1e-9))
        if not tps: return tps
        step = SAN_TP_MIN_ATR * atr_val
        near = vah if side=="LONG" else val
        tps2 = list(tps)
        if near and abs(near - tps2[0]) <= BARRIER_ATR * atr_val:
            tps2[0] = min(tps2[0], near) if side=="LONG" else max(tps2[0], near)
            if len(tps2) >= 2:
                if side=="LONG":
                    tps2[1] = max(tps2[1], tps2[0] + (step if rv >= RVOL_SPIKE else 0.5*step))
                else:
                    tps2[1] = min(tps2[1], tps2[0] - (step if rv >= RVOL_SPIKE else 0.5*step))
        return tps2
    except Exception:
        return tps

def _be_regime_adjust(side: str, entry: float, sl: float, atr_val: float, hmm: str) -> float:
    try:
        if hmm == "trend": be_k = 1.5
        elif hmm == "range": be_k = 1.0
        elif hmm == "shock": be_k = 0.8
        else: be_k = 1.2
        k = float(np.clip(be_k, BE_MIN, BE_MAX))
        if side == "LONG": return max(sl, entry - k*atr_val)
        else: return min(sl, entry + k*atr_val)
    except Exception:
        return sl

# ============================ WATCH: SIDECAR ALERTS ============================
async def _watch_sidecar(app: Dict[str, Any], bot, chat_id: int, sig):
    logger = app.get("logger")
    _should_alert = app.get("_should_alert")
    market = app.get("market")
    atr_fn = app.get("atr")
    try:
        while getattr(sig, "active", False) and app["now_msk"]() < sig.watch_until:
            df15 = market.fetch_ohlcv(sig.symbol, "15m", 140)
            ob = _fetch_order_book(app, sig.symbol, depth=25)
            feats = _book_features(ob) if ob else {}
            if df15 is not None and len(df15) >= 30:
                atrv = float(atr_fn(df15, 14).iloc[-1]) if callable(atr_fn) else float(((df15["high"] - df15["low"]).rolling(14).mean()).iloc[-1])
            else:
                atrv = 0.0
            try:
                if feats.get("spoof_score", 0.0) >= 1.0 and _should_alert and _should_alert(sig.id or -1, "spoof"):
                    await bot.send_message(chat_id, f"⚠️ Spoof‑сигнал по {sig.symbol.split('/')[0]} — частичная фиксация уместна.")
                if feats.get("spread_norm") and feats["spread_norm"] > SPREAD_Z_MAX and _should_alert and _should_alert(sig.id or -1, "spread"):
                    await bot.send_message(chat_id, f"⚠️ Широкий спред по {sig.symbol.split('/')[0]} — риск проскальзывания.")
                if feats.get("churn") and feats["churn"] > CHURN_MAX and _should_alert and _should_alert(sig.id or -1, "book_churn"):
                    await bot.send_message(chat_id, f"⚠️ Нестабильная книга {sig.symbol.split('/')[0]} — повышен риск фейков.")
            except Exception:
                pass

            try:
                if df15 is not None and len(df15) >= 60 and atrv>0:
                    x = df15.tail(240).copy()
                    tp = (x["high"] + x["low"] + x["close"]) / 3.0
                    vol = x["volume"].astype(float)
                    prices = tp.values.astype(float)
                    weights = vol.values
                    lo = float(np.min(prices)); hi = float(np.max(prices))
                    hist, edges = np.histogram(prices, bins=40, range=(lo, hi), weights=weights)
                    if hist.sum()>0:
                        poc_idx = int(np.argmax(hist))
                        poc = float(0.5*(edges[poc_idx] + edges[poc_idx+1]))
                        order = np.argsort(hist)[::-1]
                        acc=0.0; total=float(hist.sum()); mask=np.zeros_like(hist, dtype=bool)
                        for idx in order:
                            mask[idx]=True; acc+=float(hist[idx])
                            if acc/total >= 0.68: break
                        sel = np.where(mask)[0]
                        val = float(edges[sel.min()]); vah=float(edges[sel.max()+1])
                        px = float(df15["close"].iloc[-1])
                        fmt = app.get("format_price", lambda v: f"{v:.4f}")
                        if sig.side == "LONG" and vah and (0 <= (vah - px) <= 0.4 * atrv) and _should_alert and _should_alert(sig.id or -1, "vp_barrier"):
                            await bot.send_message(chat_id, f"ℹ️ Рядом VAH ({fmt(vah)}) по {sig.symbol.split('/')[0]} — возможен откат.")
                        if sig.side == "SHORT" and val and (0 <= (px - val) <= 0.4 * atrv) and _should_alert and _should_alert(sig.id or -1, "vp_barrier"):
                            await bot.send_message(chat_id, f"ℹ️ Рядом VAL ({fmt(val)}) по {sig.symbol.split('/')[0]} — возможен отскок.")
            except Exception:
                pass
            await asyncio.sleep(22)
    except asyncio.CancelledError:
        return
    except Exception as e:
        logger and logger.debug("new sidecar error: %s", e)

# ============================ KeepAlive + throttle (free-host friendly) ============================
try:
    import aiohttp
    from aiohttp import web as _web
except Exception:
    aiohttp = None
    _web = None

_KEEPALIVE_STARTED = False
_MARK_CACHE: Dict[str, Tuple[float, Optional[float]]] = {}
_PCT_CACHE: Dict[str, Tuple[float, Optional[float]]] = {}

async def _start_keepalive_http(app: Dict[str, Any]):
    global _KEEPALIVE_STARTED
    if _KEEPALIVE_STARTED:
        return
    if _web is None:
        app.get("logger") and app["logger"].warning("KeepAlive HTTP: aiohttp не доступен, пропускаю.")
        return
    port = int(os.getenv("KEEPALIVE_PORT") or os.getenv("PORT") or "8080")

    wa = _web.Application()

    async def _h_health(request):
        return _web.Response(text="OK", content_type="text/plain")
    async def _h_time(request):
        return _web.json_response({"ts": int(time.time())})

    wa.add_routes([_web.get("/health", _h_health), _web.get("/time", _h_time)])

    runner = _web.AppRunner(wa)
    await runner.setup()
    site = _web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    app["_keepalive_runner"] = runner
    app["_keepalive_site"] = site
    _KEEPALIVE_STARTED = True
    app.get("logger") and app["logger"].info(f"KeepAlive HTTP: запущен на :{port} (/health,/time)")

async def _keepalive_ping_loop(app: Dict[str, Any]):
    urls_env = os.getenv("KEEPALIVE_URLS") or os.getenv("KEEPALIVE_URL") or ""
    urls = [u.strip() for u in urls_env.split(",") if u.strip()]
    guess = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("RAILWAY_STATIC_URL") or os.getenv("WEB_URL") or ""
    if not urls and guess:
        if not guess.startswith("http"):
            guess = "https://" + guess
        urls = [guess.rstrip("/") + "/health"]

    interval = int(os.getenv("KEEPALIVE_INTERVAL_SEC", "240"))
    if not urls:
        app.get("logger") and app["logger"].info("KeepAlive ping: URL не задан(ы). Рекомендую KEEPALIVE_URL=https://<ваш-хост>/health")
    session = None
    try:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=12)) if aiohttp else None
        while True:
            for url in urls:
                try:
                    if session:
                        async with session.get(url, headers={"User-Agent": "NeonKeepAlive/1.0"}) as resp:
                            _ = await resp.text()
                    else:
                        import urllib.request
                        urllib.request.urlopen(url, timeout=10).read()
                    app.get("logger") and app["logger"].debug(f"KeepAlive ping -> {url}")
                except Exception:
                    app.get("logger") and app["logger"].debug(f"KeepAlive ping fail: {url}")
            try:
                bot = app.get("bot_instance")
                if bot:
                    with contextlib.suppress(Exception):
                        await bot.get_me()
            except Exception:
                pass
            await asyncio.sleep(interval + (int(time.time()) % 7))
    finally:
        if session:
            with contextlib.suppress(Exception):
                await session.close()

def _patch_mark_price_cache(app: Dict[str, Any]):
    market = app.get("market")
    if not market or not hasattr(market, "fetch_mark_price"):
        return
    ttl = float(os.getenv("NEW_MARK_PRICE_CACHE_SEC", "8.0"))
    orig = market.fetch_mark_price

    def _cached(symbol: str) -> Optional[float]:
        now = time.time()
        ts, val = _MARK_CACHE.get(symbol, (0.0, None))
        if val is not None and (now - ts) < ttl:
            return val
        try:
            v = orig(symbol)
            if v is not None:
                _MARK_CACHE[symbol] = (now, float(v))
            return v
        except Exception:
            return val

    market.fetch_mark_price = _cached
    app.get("logger") and app["logger"].info(f"MarkPrice cache: включён (TTL {ttl:.1f}s)")

def _bump_ohlcv_ttl(app: Dict[str, Any]):
    market = app.get("market")
    try:
        bump = int(os.getenv("NEW_OHLCV_TTL_BUMP", "120"))
        if market and hasattr(market, "OHLCV_TTL"):
            old = int(getattr(market, "OHLCV_TTL", 60))
            if bump > old:
                market.OHLCV_TTL = bump
                app.get("logger") and app["logger"].info(f"OHLCV TTL: {old}s -> {bump}s")
    except Exception:
        pass

# ============================ DM TICKER (private chat "plaque") ============================
DM_TICKER_ENABLED       = os.getenv("DM_TICKER_ENABLED", "1") == "1"
DM_TICKER_COINS         = [s.strip().upper() for s in (os.getenv("DM_TICKER_COINS", "BTC,ETH,SOL,BNB,TON")).split(",") if s.strip()]
DM_TICKER_ROTATE_SEC    = int(os.getenv("DM_TICKER_ROTATE_SEC", "3"))
DM_TICKER_LIFETIME_SEC  = int(os.getenv("DM_TICKER_LIFETIME_SEC", "900"))   # 0 = бесконечно
DM_TICKER_PIN           = os.getenv("DM_TICKER_PIN", "1") == "1"
DM_TICKER_PCT_TTL_SEC   = int(os.getenv("DM_TICKER_PCT_TTL_SEC", "60"))     # кэш процента на минуту

def _fmt_price_usd(v: Optional[float]) -> str:
    if v is None:
        return "—"
    try:
        x = float(v)
        if x >= 1000: s = f"{x:,.0f}$"
        elif x >= 10: s = f"{x:,.2f}$"
        else: s = f"{x:.4f}$"
        return s.replace(",", " ")
    except Exception:
        return "—"

def _fmt_ticker_line(base: str, price: Optional[float], pct: Optional[float]) -> str:
    ptxt = _fmt_price_usd(price)
    if pct is None:
        return f"• {base}: {ptxt}"
    arrow = "🟢▲" if pct >= 0 else "🔴▼"
    return f"{arrow} {base}: {ptxt} ({pct:+.2f}% 24ч)"

def _get_price_and_pct(app: Dict[str, Any], symbol: str) -> Tuple[Optional[float], Optional[float]]:
    market = app.get("market")
    price = None; pct = None
    with contextlib.suppress(Exception):
        price = market.fetch_mark_price(symbol)
    # percentage с кэшем
    now = time.time()
    ts, prev = _PCT_CACHE.get(symbol, (0.0, None))
    if prev is not None and (now - ts) < DM_TICKER_PCT_TTL_SEC:
        return price, prev
    try:
        t = market.fetch_ticker(symbol) or {}
        pct = float(t.get("percentage") or t.get("info", {}).get("priceChangePercent") or 0.0)
        _PCT_CACHE[symbol] = (now, pct)
    except Exception:
        pct = prev
    return price, pct

async def _dm_ticker_loop(app: Dict[str, Any], bot, user_id: int):
    coins = DM_TICKER_COINS[:]
    if not coins:
        coins = ["BTC","ETH","SOL","BNB","TON"]
    # найти/создать сообщение
    dm_state = app.setdefault("_dm_tickers", {})
    ent = dm_state.setdefault(user_id, {"mid": None, "expires": 0.0, "task": None})
    # если сообщения нет — создаём
    if not ent["mid"]:
        try:
            m = await bot.send_message(user_id, "💹 Живые цены — инициализация...")
            ent["mid"] = m.message_id
            if DM_TICKER_PIN:
                with contextlib.suppress(Exception):
                    await bot.pin_chat_message(user_id, m.message_id, disable_notification=True)
        except Exception:
            return
    # основной цикл
    while True:
        # остановка по lifetime (если задан)
        if DM_TICKER_LIFETIME_SEC > 0 and time.time() > float(ent.get("expires", 0.0)):
            break
        for base in coins:
            symbol = f"{base}/USDT"
            price, pct = _get_price_and_pct(app, symbol)
            txt = "💹 Живые цены (автообновление)\n" + _fmt_ticker_line(base, price, pct)
            try:
                await bot.edit_message_text(chat_id=user_id, message_id=ent["mid"], text=txt)
            except Exception:
                # возможно, юзер удалил/очистил — пробуем заново создать
                with contextlib.suppress(Exception):
                    m2 = await bot.send_message(user_id, txt)
                    ent["mid"] = m2.message_id
                    if DM_TICKER_PIN:
                        with contextlib.suppress(Exception):
                            await bot.pin_chat_message(user_id, m2.message_id, disable_notification=True)
            await asyncio.sleep(max(2, DM_TICKER_ROTATE_SEC))
    # по окончании — опционально можно раззакрепить

async def _ensure_dm_ticker(app: Dict[str, Any], bot, user_id: int):
    if not DM_TICKER_ENABLED:
        return
    dm_state = app.setdefault("_dm_tickers", {})
    ent = dm_state.setdefault(user_id, {"mid": None, "expires": 0.0, "task": None})
    # продлеваем lifetime
    ent["expires"] = time.time() + (DM_TICKER_LIFETIME_SEC if DM_TICKER_LIFETIME_SEC > 0 else 24*3600*365)
    # если таск не запущен — стартуем
    t = ent.get("task")
    if not t or t.done():
        task = asyncio.create_task(_dm_ticker_loop(app, bot, user_id))
        ent["task"] = task

# ============================ DAILY SEND GUARDS (one per day) ============================
def _install_daily_send_guards_new(app: Dict[str, Any]) -> None:
    import asyncio as _aio
    logger = app.get("logger")
    once = app.setdefault("_daily_once", {"lock": _aio.Lock(), "sent": set()})

    def _day(app_):
        return app_["now_msk"]().date().isoformat()

    def _wrap_send_once(tag_fn, orig_fn):
        async def wrapper(*args, **kwargs):
            key = tag_fn(*args, **kwargs)
            async with once["lock"]:
                if key in once["sent"]:
                    logger and logger.info("Daily send guard: skip %s", key)
                    return
                once["sent"].add(key)
            return await orig_fn(*args, **kwargs)
        setattr(wrapper, "_new_wrapped", True)
        return wrapper

    m = sys.modules.get("main")
    if m and hasattr(m, "_post_morning_report") and callable(m._post_morning_report) and not getattr(m._post_morning_report, "_new_wrapped", False):
        orig = m._post_morning_report
        def _tag_channel(app_, bot, channel_id):
            return ("channel", str(channel_id), _day(app_))
        m._post_morning_report = _wrap_send_once(_tag_channel, orig)
        logger and logger.info("new: daily guard -> main._post_morning_report")

    if m and hasattr(m, "_post_admin_greetings") and callable(m._post_admin_greetings) and not getattr(m._post_admin_greetings, "_new_wrapped", False):
        orig = m._post_admin_greetings
        def _tag_admin_main(app_, bot):
            return ("admin_daily", "all", _day(app_))
        m._post_admin_greetings = _wrap_send_once(_tag_admin_main, orig)
        logger and logger.info("new: daily guard -> main._post_admin_greetings")

    c = sys.modules.get("chat")
    if c and hasattr(c, "_post_admin_greetings_once") and callable(c._post_admin_greetings_once) and not getattr(c._post_admin_greetings_once, "_new_wrapped", False):
        orig = c._post_admin_greetings_once
        def _tag_admin_chat(app_, bot):
            return ("admin_daily", "all", _day(app_))
        c._post_admin_greetings_once = _wrap_send_once(_tag_admin_chat, orig)
        logger and logger.info("new: daily guard -> chat._post_admin_greetings_once")

# ============================ PATCH ENTRY ============================
def patch(app: Dict[str, Any]) -> None:
    logger = app.get("logger")

    # ---- Loop guards + city fix ----
    if NEW_LOOP_GUARD:
        m = sys.modules.get("main")
        c = sys.modules.get("chat")
        try:
            if "start_daily_channel_post_loop" in app and callable(app["start_daily_channel_post_loop"]) and not getattr(app["start_daily_channel_post_loop"], "_new_wrapped", False):
                app["start_daily_channel_post_loop"] = _wrap_loop_once(logger, "channel_loop", app["start_daily_channel_post_loop"])
                logger and logger.info("new: loop guard applied for app.start_daily_channel_post_loop")
            if m and hasattr(m, "start_daily_channel_post_loop") and callable(m.start_daily_channel_post_loop) and not getattr(m.start_daily_channel_post_loop, "_new_wrapped", False):
                m.start_daily_channel_post_loop = _wrap_loop_once(logger, "channel_loop", m.start_daily_channel_post_loop)
                logger and logger.info("new: loop guard applied for main.start_daily_channel_post_loop")
        except Exception:
            pass
        try:
            if m and hasattr(m, "start_daily_admin_greetings_loop") and callable(m.start_daily_admin_greetings_loop) and not getattr(m.start_daily_admin_greetings_loop, "_new_wrapped", False):
                m.start_daily_admin_greetings_loop = _wrap_loop_once(logger, "admin_loop", m.start_daily_admin_greetings_loop)
                logger and logger.info("new: loop guard applied for main.start_daily_admin_greetings_loop")
        except Exception:
            pass
        try:
            if c and hasattr(c, "_start_daily_admin_greetings_loop") and callable(c._start_daily_admin_greetings_loop) and not getattr(c._start_daily_admin_greetings_loop, "_new_wrapped", False):
                c._start_daily_admin_greetings_loop = _wrap_loop_once(logger, "admin_loop", c._start_daily_admin_greetings_loop)
                logger and logger.info("new: loop guard applied for chat._start_daily_admin_greetings_loop")
        except Exception:
            pass

    # сразу установим guards отправки (один раз в день)
    _install_daily_send_guards_new(app)

    orig_on_startup = app.get("on_startup")
    async def _on_startup_new(bot):
        # ensure city column
        if NEW_CITY_FIX:
            with contextlib.suppress(Exception):
                await _ensure_city_column(app)
        # start keepalive http + self-ping + throttles
        with contextlib.suppress(Exception):
            await _start_keepalive_http(app)
        with contextlib.suppress(Exception):
            asyncio.create_task(_keepalive_ping_loop(app))
        with contextlib.suppress(Exception):
            _patch_mark_price_cache(app)
        with contextlib.suppress(Exception):
            _bump_ohlcv_ttl(app)
        # ещё раз — на случай позднего импорта модулей
        with contextlib.suppress(Exception):
            _install_daily_send_guards_new(app)
        if orig_on_startup:
            await orig_on_startup(bot)
    app["on_startup"] = _on_startup_new
    logger and logger.info("new: on_startup patched (city fix + keepalive + throttle + loop guards + daily guards).")

    # ---- Score engine wrap ----
    orig_score = app.get("score_symbol_core")
    if NEW_SCORE_ENGINE and callable(orig_score):
        def _score_new(symbol: str, relax: bool = False):
            base = orig_score(symbol, relax)
            if base is None:
                return None
            score, side, d = base
            d = dict(d or {})
            breakdown = d.get("score_breakdown", {}) or {}
            market = app.get("market"); atr_fn = app.get("atr")
            try: df5  = market.fetch_ohlcv(symbol, "5m", 300)
            except Exception: df5  = None
            try: df15 = market.fetch_ohlcv(symbol, "15m", 300)
            except Exception: df15 = None
            try: df1h = market.fetch_ohlcv(symbol, "1h", 260)
            except Exception: df1h = None
            try: atr15 = float(atr_fn(df15, 14).iloc[-1]) if (df15 is not None and callable(atr_fn)) else float(((df15["high"]-df15["low"]).rolling(14).mean()).iloc[-1]) if df15 is not None else 0.0
            except Exception: atr15 = 0.0

            ob = _fetch_order_book(app, symbol, 25)
            feats = _book_features(ob) if ob else {}
            d.update(feats)

            ofi, vpin = _ofi_vpin(df5)
            if ofi is not None:
                d["ofi"] = float(ofi)
                sdev = float(np.std(df5["volume"].astype(float).tail(OFI_WIN))+1e-9) if df5 is not None else 1.0
                adj = W_OFI * _tanh_clip(ofi, s=sdev)
                score += adj; breakdown["OFI"] = breakdown.get("OFI", 0.0) + adj
            if vpin is not None:
                d["vpin"] = float(vpin)
                adj = W_VPIN * (0.5 - vpin)
                score += adj; breakdown["VPIN"] = breakdown.get("VPIN", 0.0) + adj

            if feats.get("spread_norm") and feats["spread_norm"] > SPREAD_Z_MAX:
                score += W_SPREAD; breakdown["Spread"] = breakdown.get("Spread", 0.0) + W_SPREAD
            if feats.get("churn") and feats["churn"] > CHURN_MAX:
                score += W_CHURN; breakdown["Churn"] = breakdown.get("Churn", 0.0) + W_CHURN
            if feats.get("spoof_score") and feats["spoof_score"] >= 1.0:
                score += W_SPOOF; breakdown["Spoof"] = breakdown.get("Spoof", 0.0) + W_SPOOF
            if NEW_MICRO_GRAD and feats.get("slope") is not None and feats.get("microprice_drift") is not None:
                sign = 1.0 if side=="LONG" else -1.0
                grad = sign * (0.3*_tanh_clip(feats["slope"], 100.0) + 0.7*_tanh_clip(feats["microprice_drift"], 0.002))
                adj = W_GRAD * grad
                score += adj; breakdown["BookGrad"] = breakdown.get("BookGrad", 0.0) + adj

            if NEW_HMM:
                reg = _hmm_regime(df15)
                d["regime_hmm"] = reg
                if reg == "trend": score += W_HMM * 0.5; breakdown["HMM"] = breakdown.get("HMM", 0.0) + W_HMM*0.5
                elif reg == "range": score -= W_HMM * 0.25; breakdown["HMM"] = breakdown.get("HMM", 0.0) - W_HMM*0.25
                elif reg == "shock": score -= W_HMM * 0.35; breakdown["HMM"] = breakdown.get("HMM", 0.0) - W_HMM*0.35
            try:
                adx15 = _adx(df15, 14) if df15 is not None else None
                if adx15 is not None: d["adx15"] = adx15
            except Exception: pass
            try:
                bb = _bbwp(df15["close"]) if df15 is not None else None
                if bb is not None: d["bbwp"] = bb
            except Exception: pass

            if NEW_ALIGN_MTF:
                adj = _alignment_penalty(d, side)
                score += adj; breakdown["AlignMTF"] = breakdown.get("AlignMTF", 0.0) + adj

            if NEW_SMC_STRICT:
                has_fvg, has_ob = _improved_fvg_ob(df15, atr15)
                if has_fvg: score += W_SMC * 0.5; breakdown["FVG+"] = breakdown.get("FVG+", 0.0) + W_SMC*0.5
                if has_ob:  score += W_SMC * 0.5; breakdown["OB+"]  = breakdown.get("OB+", 0.0) + W_SMC*0.5
                bos_dir, bos_retest = _bos_choch_strict(df15, 80, 1.2)
                d["bos_strict"] = int(bos_dir); d["bos_retest_strict"] = bool(bos_retest)

            if NEW_RS:
                try:
                    btc = app.get("market").fetch_ohlcv("BTC/USDT", "5m", 300)
                    eth = app.get("market").fetch_ohlcv("ETH/USDT", "5m", 300)
                except Exception:
                    btc = eth = None
                rs_btc = None
                if df5 is not None and btc is not None and len(df5) >= 240 and len(btc) >= 240:
                    ya = df5["close"].astype(float).tail(240).values + 1e-9
                    yb = btc["close"].astype(float).tail(240).values + 1e-9
                    y = np.log(ya/yb); x = np.arange(len(y), dtype=float)
                    sl, _ = np.polyfit(x, y, 1)
                    rs_btc = float(sl)
                if rs_btc is not None:
                    d["RS_btc_new"] = rs_btc
                    adj = W_RS if ((side=="LONG" and rs_btc>0) or (side=="SHORT" and rs_btc<0)) else -W_RS/2
                    score += adj; breakdown["RSbtc+"] = breakdown.get("RSbtc+", 0.0) + adj

                rs_eth = None
                if df5 is not None and eth is not None and len(df5) >= 240 and len(eth) >= 240:
                    ya = df5["close"].astype(float).tail(240).values + 1e-9
                    yb = eth["close"].astype(float).tail(240).values + 1e-9
                    y = np.log(ya/yb); x = np.arange(len(y), dtype=float)
                    sl, _ = np.polyfit(x, y, 1)
                    rs_eth = float(sl)
                if rs_eth is not None:
                    d["RS_eth_new"] = rs_eth

            if NEW_BREADTH:
                br = _compute_breadth(app, 300)
                d["breadth"] = br
                base_sym = symbol.split("/")[0]
                is_alt = base_sym not in ("BTC","ETH")
                if is_alt and side=="LONG" and br and br.get("pct50_1h", 50.0) < 45.0:
                    score -= W_BREADTH; breakdown["Breadth"] = breakdown.get("Breadth", 0.0) - W_BREADTH
                if is_alt and side=="LONG" and br and br.get("pct50_1h", 50.0) > 60.0:
                    score += W_BREADTH*0.5; breakdown["Breadth"] = breakdown.get("Breadth", 0.0) + W_BREADTH*0.5

            if NEW_BTC_DOM:
                trend = _estimate_btc_dom_trend(app)
                if trend:
                    d["btc_dom_trend_new"] = trend
                    base_sym = symbol.split("/")[0]
                    is_alt = base_sym not in ("BTC","ETH")
                    if is_alt:
                        if side=="LONG" and trend=="up":
                            score += W_BTC_DOM; breakdown["BTC.D"] = breakdown.get("BTC.D", 0.0) + W_BTC_DOM
                        if side=="SHORT" and trend=="up":
                            score -= W_BTC_DOM*0.5; breakdown["BTC.D"] = breakdown.get("BTC.D", 0.0) - W_BTC_DOM*0.5

            if NEW_NEWS_LOG:
                p = _news_logistic_p(d.get("news_note",""))
                if p is not None:
                    d["news_prob_new"] = p
                    adj = W_NEWS * ((p-0.5) * 2.0)
                    score += adj; breakdown["NewsLog"] = breakdown.get("NewsLog", 0.0) + adj

            d["quality_line"] = _append_quality_if_absent("", d)
            d["score_breakdown"] = breakdown
            d["score"] = float(score)
            return float(score), side, d
        app["score_symbol_core"] = _score_new
        logger and logger.info("new: score_symbol_core wrapped (microstructure/regimes/SMC/RS/breadth/news + quality, no HTTP).")

    # ---- Reason wrap (no double 'Качество') ----
    orig_reason = app.get("build_reason")
    if NEW_REASON_WRAP and callable(orig_reason):
        def _build_reason_new(details: Dict[str, Any]) -> str:
            base = ""
            try:
                base = orig_reason(details) or ""
            except Exception:
                base = ""
            try:
                return _append_quality_if_absent(base, details)
            except Exception:
                return base
        app["build_reason"] = _build_reason_new
        logger and logger.info("new: build_reason wrapped (no-duplicate 'Качество').")

    # ---- Trailing wrap (TP tune + regime BE) ----
    if NEW_TRAILING_WRAP and callable(app.get("update_trailing")):
        orig_trailing = app["update_trailing"]
        async def _trailing_new(sig):
            try:
                await orig_trailing(sig)
            except Exception:
                pass
            try:
                market = app.get("market"); atr_fn = app.get("atr")
                df15 = market.fetch_ohlcv(sig.symbol, "15m", 220)
                if df15 is None or len(df15) < 60:
                    return
                atrv = float(atr_fn(df15, 14).iloc[-1]) if callable(atr_fn) else float(((df15["high"]-df15["low"]).rolling(14).mean()).iloc[-1])
                if NEW_TP_TUNE:
                    new_tps = _tp_dynamic_adjust(app, sig.symbol, sig.side, list(sig.tps), float(sig.entry), atrv)
                    if new_tps and new_tps != list(sig.tps):
                        sig.tps = new_tps
                if NEW_BE_REGIME:
                    hmm = _hmm_regime(df15)
                    sig.sl = _be_regime_adjust(sig.side, float(sig.entry), float(sig.sl), atrv, hmm)
            except Exception:
                pass
        app["update_trailing"] = _trailing_new
        logger and logger.info("new: update_trailing wrapped (TP tune + regime-aware BE).")

    # ---- Watch sidecar alerts ----
    orig_watch = app.get("watch_signal_price")
    if NEW_WATCH_SIDECAR and callable(orig_watch):
        async def _watch_wrap(bot, chat_id: int, sig):
            sidecar = None
            try:
                sidecar = asyncio.create_task(_watch_sidecar(app, bot, chat_id, sig))
            except Exception:
                sidecar = None
            try:
                await orig_watch(bot, chat_id, sig)
            finally:
                if sidecar and not sidecar.done():
                    with contextlib.suppress(Exception):
                        sidecar.cancel()
        app["watch_signal_price"] = _watch_wrap
        logger and logger.info("new: watch_signal_price wrapped (microstructure alerts).")

    # ---- TA report extras (/tacoin), оффлайн ----
    if NEW_TA_REPORT_WRAP and "ta" in sys.modules:
        t = sys.modules.get("ta")
        if hasattr(t, "_build_tech_report") and callable(t._build_tech_report):
            orig_ta_report = t._build_tech_report
            def _build_tech_report_new(app_: Dict[str, Any], symbol: str) -> Optional[str]:
                txt = orig_ta_report(app_, symbol)
                if not txt:
                    return txt
                try:
                    market = app_.get("market")
                    df15 = market.fetch_ohlcv(symbol, "15m", 240)
                    hmm = _hmm_regime(df15)
                    br = _compute_breadth(app_, 300) if NEW_BREADTH else {}
                    trend = _estimate_btc_dom_trend(app_) if NEW_BTC_DOM else None
                    df5  = market.fetch_ohlcv(symbol, "5m", 240)
                    ofi, vpin = _ofi_vpin(df5) if NEW_OFI_VPIN else (None, None)
                    ob = _fetch_order_book(app_, symbol, 25) if NEW_OB_SPOOF else None
                    feats = _book_features(ob) if ob else {}
                    lines = []
                    lines.append("")
                    lines.append("Доп. контекст")
                    line = f"• HMM: {hmm}"
                    if br:
                        line += f" • Breadth 1h>50: {br.get('pct50_1h',0):.0f}% • 4h>50: {br.get('pct50_4h',0):.0f}%"
                    if trend:
                        arrow = "↑" if trend=="up" else ("↓" if trend=="down" else "•")
                        line += f" • BTC.D(trend): {arrow}"
                    lines.append(line)
                    micro = []
                    if ofi is not None: micro.append(f"OFI {ofi:+.2e}")
                    if vpin is not None: micro.append(f"VPIN {vpin:.2f}")
                    if feats.get("spread_norm") is not None: micro.append(f"SpreadZ {feats['spread_norm']:.2f}")
                    if feats.get("spoof_score") is not None and feats["spoof_score"]>=1.0: micro.append("Spoof!")
                    if micro:
                        lines.append("• Micro: " + ", ".join(micro))
                    return (txt + "\n" + "\n".join(lines))[:4000]
                except Exception:
                    return txt
            t._build_tech_report = _build_tech_report_new
            logger and logger.info("new: TA report wrapped (extra HMM/Breadth/Micro, no HTTP).")

    # ---- FINAL SIGNAL SANITY (wrap format_signal_message) ----
    orig_format = app.get("format_signal_message")
    if NEW_FINAL_SIG_FIX and callable(orig_format):
        def _format_msg_new(sig):
            changed, notes = _final_signal_sanity(app, sig)
            try:
                if changed:
                    rsn = getattr(sig, "reason", "") or ""
                    if "TP/SL нормализованы" not in rsn:
                        add = (" • " if rsn else "") + "TP/SL нормализованы"
                        sig.reason = (rsn + add)[:950]
            except Exception:
                pass
            return orig_format(sig)
        app["format_signal_message"] = _format_msg_new
        logger and logger.info("new: format_signal_message wrapped (final SL/TP sanity).")

    # ---- Hook DM TICKER on 'Подробнее' (neon:detail) ----
    try:
        router = app.get("router")
        if router:
            obs = router.callback_query
            handlers = getattr(obs, "handlers", [])
            target_cb = None
            for h in handlers:
                cbf = getattr(h, "callback", None)
                if getattr(cbf, "__name__", "") == "_h_cb":  # общий обработчик из chat.py
                    target_cb = h
                    break
            if target_cb:
                orig_h_cb = target_cb.callback
                async def _h_cb_wrapped(cb):
                    data = cb.data or ""
                    try:
                        if DM_TICKER_ENABLED and data.startswith("neon:detail:"):
                            bot = app.get("bot_instance")
                            if bot and cb.from_user:
                                await _ensure_dm_ticker(app, bot, cb.from_user.id)
                    except Exception:
                        pass
                    return await orig_h_cb(cb)
                setattr(target_cb, "callback", _h_cb_wrapped)
                logger and logger.info("new: chat._h_cb wrapped (DM ticker on neon:detail).")
    except Exception:
        logger and logger.warning("new: unable to hook DM ticker on neon:detail")

    logger and logger.info("new: patch applied.")
