# tacoin.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os
import math
import asyncio
import contextlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# ============================== ENV/CONFIG ==============================
def _env_bool(name: str, default: str = "1") -> bool:
    try:
        return bool(int(os.getenv(name, default)))
    except Exception:
        txt = os.getenv(name, str(default)).strip().lower()
        return txt in {"1", "true", "yes", "on"}

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_list_float(name: str, default: str) -> List[float]:
    txt = os.getenv(name, default).strip()
    out: List[float] = []
    for p in txt.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except Exception:
            pass
    return out

# Основные блоки
TAC_EXEC        = _env_bool("TAC_EXEC", "1")          # микроструктура/исполнение
TAC_BREAKOUT    = _env_bool("TAC_BREAKOUT", "1")      # предпробой/качество пробоя/quickfail
TAC_LIQMAP      = _env_bool("TAC_LIQMAP", "1")        # карта ликвидности + глубина риска
TAC_VOLREG      = _env_bool("TAC_VOLREG", "1")        # режимы, персистентность, MR‑риск
TAC_DERIV_DEEP  = _env_bool("TAC_DERIV_DEEP", "1")    # деривативы: jerk/divergences
TAC_META        = _env_bool("TAC_META", "1")          # мета‑p* калибровка (легкая)
TAC_TRAIL2      = _env_bool("TAC_TRAIL2", "1")        # trailing 2.0
TAC_ALERTS      = _env_bool("TAC_ALERTS", "1")        # алерты в мониторинге
TAC_DRYRUN      = _env_bool("TAC_DRYRUN", "0")        # считаем, но score не меняем
TAC_AB          = _env_bool("TAC_AB", "0")            # лог "до/после"

# TP/SL builder (универсальный фиксер)
TAC_TPB_ENABLE  = _env_bool("TAC_TPB_ENABLE", "1")    # включить TP/SL builder
TAC_TPB_AUTO    = _env_bool("TAC_TPB_AUTO", "1")      # авто-исправление при аномалиях
TAC_TPB_FORCE   = _env_bool("TAC_TPB_FORCE", "0")     # всегда перестраивать TP/SL
TAC_TP_LADDER   = _env_list_float("TAC_TP_LADDER", "0.7,1.6,2.6")  # умолчание для обычных
TAC_TP_LADDER_MEME = _env_list_float("TAC_TP_LADDER_MEME", "0.6,1.2,2.0")
TAC_TP_MAX_PCT  = _env_float("TAC_TP_MAX_PCT", 0.25)  # максимум растяжки от входа в %
TAC_SAFE_DECIMALS_MIN = _env_int("TAC_SAFE_DECIMALS_MIN", 6)
TAC_SAFE_DECIMALS_MAX = _env_int("TAC_SAFE_DECIMALS_MAX", 8)

# Веса скоринга
TAC_W_EXEC              = _env_float("TAC_W_EXEC", 0.20)
TAC_W_BREAKOUT          = _env_float("TAC_W_BREAKOUT", 0.18)
TAC_W_LIQMAP            = _env_float("TAC_W_LIQMAP", 0.20)
TAC_W_PERSIST           = _env_float("TAC_W_PERSIST", 0.12)
TAC_W_MRISK             = _env_float("TAC_W_MRISK", -0.12)
TAC_W_META_CAL          = _env_float("TAC_W_META_CAL", 0.10)
TAC_W_SPOOF_PENALTY     = _env_float("TAC_W_SPOOF_PENALTY", -0.15)
TAC_W_SPREADZ_PENALTY   = _env_float("TAC_W_SPREADZ_PENALTY", -0.12)
TAC_W_QUICKFAIL         = _env_float("TAC_W_QUICKFAIL", -0.15)
TAC_W_LIQRISK           = _env_float("TAC_W_LIQRISK", -0.10)

# Пороги/параметры
TAC_EQ_TOL_FRAC         = _env_float("TAC_EQ_TOL_FRAC", 0.0006)
TAC_OB_LOOKBACK         = _env_int("TAC_OB_LOOKBACK", 80)
TAC_FVG_LOOKBACK        = _env_int("TAC_FVG_LOOKBACK", 120)
TAC_NEAR_PCT            = _env_float("TAC_NEAR_PCT", 0.0010)   # 0.10% от mid
TAC_NEAR_DEPTH_MIN      = _env_float("TAC_NEAR_DEPTH_MIN", 0.55)
TAC_SPOOF_K             = _env_float("TAC_SPOOF_K", 3.0)
TAC_SPOOF_DROP          = _env_float("TAC_SPOOF_DROP", 0.60)
TAC_SPREAD_Z_MAX        = _env_float("TAC_SPREAD_Z_MAX", 3.0)
TAC_CHURN_MAX           = _env_float("TAC_CHURN_MAX", 1.80)

TAC_COMPRESSION_Q_MIN   = _env_float("TAC_COMPRESSION_Q_MIN", 55.0)   # bbwp‑ранг %
TAC_BREAKOUT_BODY_ATR   = _env_float("TAC_BREAKOUT_BODY_ATR", 0.80)
TAC_BREAKOUT_CLOSE_PCT  = _env_float("TAC_BREAKOUT_CLOSE_PCT", 0.25)  # доля диапазона за экстремумом
TAC_QUICKFAIL_RISK_MAX  = _env_float("TAC_QUICKFAIL_RISK_MAX", 0.60)

TAC_NO_PROGRESS_MIN     = _env_int("TAC_NO_PROGRESS_MIN", 60)         # минут
TAC_TIME_STOP_ATR       = _env_float("TAC_TIME_STOP_ATR", 0.80)       # < 0.8×ATR → no‑progress
TAC_LADDER_STEP_ATR     = _env_float("TAC_LADDER_STEP_ATR", 0.35)
TAC_CAT_STOP_ATR        = _env_float("TAC_CAT_STOP_ATR", 2.70)

TAC_KAMA_FAST           = _env_int("TAC_KAMA_FAST", 2)
TAC_KAMA_SLOW           = _env_int("TAC_KAMA_SLOW", 30)
TAC_EHLERS_PERIOD       = _env_int("TAC_EHLERS_PERIOD", 20)

TAC_SIDE_TASK_INTERVAL  = _env_int("TAC_SIDE_TASK_INTERVAL", 22)      # интервал побочного мониторинга

# ============================== METRICS (optional) ==============================
_MET_OB_CACHE_SEC   = 8.0
_ORDERBOOK_CACHE    : Dict[str, Tuple[float, Dict[str, Any]]] = {}
_LAST_TOPLEVELS     : Dict[str, Dict[str, Any]] = {}  # symbol_key -> snapshot

def _prom_init(app: Dict[str, Any]):
    if not app.get("PROMETHEUS_OK"):
        return {}
    try:
        from prometheus_client import Histogram, Counter
        return {
            "TA_BUNDLE_LAT": Histogram("tacoin_bundle_latency_seconds", "tacoin bundle compute latency"),
            "TA_OB_LAT":     Histogram("tacoin_orderbook_fetch_seconds", "tacoin orderbook fetch latency"),
            "TA_ALERTS":     Counter("tacoin_alerts_total", "tacoin alerts", ["kind"]),
            "TA_TP_ZERO":    Counter("tacoin_tp_zero_total", "tp rounded to zero"),
            "TA_TP_EQUAL":   Counter("tacoin_tp_equal_total", "tp equal after rounding"),
            "TA_TP_OUTLIER": Counter("tacoin_tp_outlier_total", "tp outlier pct from entry"),
        }
    except Exception:
        return {}

# ============================== UTILS ==============================
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if (v == v) else float(default)
    except Exception:
        return float(default)

def _znorm(arr: np.ndarray) -> np.ndarray:
    m = float(np.mean(arr))
    s = float(np.std(arr) + 1e-12)
    return (arr - m) / s

def _bbwp(series: pd.Series, length: int = 20, lookback: int = 96) -> Optional[float]:
    try:
        if len(series) < max(length, lookback) + 5:
            return None
        basis = series.rolling(length).mean()
        dev = series.rolling(length).std(ddof=0)
        bbw = (series - basis + 2 * dev) / (4 * dev + 1e-12)
        bbwp = bbw.rolling(lookback).rank(pct=True) * 100.0
        return float(np.clip(bbwp.iloc[-1], 0.0, 100.0))
    except Exception:
        return None

def _round_tick(x: float, tick: Optional[float], mode: str = "round") -> float:
    try:
        if not tick or tick <= 0: return float(x)
        n = float(x) / float(tick)
        if mode == "floor": n = math.floor(n)
        elif mode == "ceil": n = math.ceil(n)
        else: n = round(n)
        return float(n * float(tick))
    except Exception:
        return float(x)

def _percent_change(a: float, b: float) -> float:
    if b == 0: return 0.0
    return abs(a - b) / abs(b)

def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))

# ============================== ORDERBOOK ==============================
def _ob_cache_key(ex_id: str, resolved: str) -> str:
    return f"{ex_id}:{resolved}"

def _fetch_order_book(app: Dict[str, Any], symbol: str, depth: int = 25) -> Optional[Dict[str, Any]]:
    market = app.get("market")
    prom = app.setdefault("_TAC_PROM", _prom_init(app))
    last_err = None
    for name, ex in market._available_exchanges():
        resolved = market.resolve_symbol(ex, symbol) or symbol
        if resolved not in ex.markets:
            continue
        key = _ob_cache_key(ex.id, resolved)
        ts_now = _now_utc().timestamp()
        cached = _ORDERBOOK_CACHE.get(key)
        if cached and ts_now - cached[0] < _MET_OB_CACHE_SEC:
            return cached[1]
        try:
            t0 = _now_utc().timestamp()
            ob = ex.fetch_order_book(resolved, limit=depth)
            t1 = _now_utc().timestamp()
            if prom.get("TA_OB_LAT"):
                with contextlib.suppress(Exception):
                    prom["TA_OB_LAT"].observe(max(0.0, t1 - t0))
            if not isinstance(ob, dict):
                continue
            _ORDERBOOK_CACHE[key] = (ts_now, ob)
            return ob
        except Exception as e:
            last_err = e
            continue
    app.get("logger") and app["logger"].debug("tacoin: orderbook fetch failed for %s: %s", symbol, last_err)
    return None

def _book_features(ob: Dict[str, Any], atr_val: float) -> Dict[str, float]:
    """
    Возвращает:
      near_depth_ratio, spread_norm, book_imb, churn (если есть prev), spoof_score (эвристика)
    """
    out = {"near_depth_ratio": None, "spread_norm": None, "book_imb": None, "churn": None, "spoof_score": None}
    try:
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        if not bids or not asks:
            return out
        pb, qb = float(bids[0][0]), float(bids[0][1])
        pa, qa = float(asks[0][0]), float(asks[0][1])
        mid = 0.5 * (pa + pb)
        spread = max(0.0, pa - pb)
        # нормируем спред к 0.1% цены
        denom = max(1e-9, (mid * 0.001))
        out["spread_norm"] = float(spread / denom)

        # доля объёма в ближней зоне (± TAC_NEAR_PCT)
        def near_ratio(levels):
            keep = []
            for p, q in levels[:25]:
                p = float(p); q = float(q)
                if abs(p - mid) / (mid + 1e-9) <= TAC_NEAR_PCT:
                    keep.append(float(q))
            total = float(sum([float(x[1]) for x in levels[:25]])) + 1e-9
            return float(sum(keep) / total)

        near_bid = near_ratio(bids)
        near_ask = near_ratio(asks)
        near_all = 0.5 * (near_bid + near_ask)
        out["near_depth_ratio"] = float(np.clip(near_all, 0.0, 1.0))

        sb = float(sum([float(x[1]) for x in bids[:25]]))
        sa = float(sum([float(x[1]) for x in asks[:25]]))
        if (sb + sa) > 0:
            out["book_imb"] = float((sb - sa) / (sb + sa))

        # churn & spoof: сравнение с прошлым снапшотом топ‑N
        key = f"default:{mid:.8f}"
        prev = _LAST_TOPLEVELS.get(key)
        snap = {
            "bids": [(float(p), float(q)) for p, q in bids[:10]],
            "asks": [(float(p), float(q)) for p, q in asks[:10]],
        }
        _LAST_TOPLEVELS[key] = snap
        if prev:
            # churn = средняя относительная смена объёма первых уровней
            def churn_side(cur, prv):
                m = min(len(cur), len(prv))
                if m == 0: return 0.0
                diffs = []
                for i in range(m):
                    qc = float(cur[i][1]); qp = float(prv[i][1]) + 1e-9
                    diffs.append(abs(qc - qp) / qp)
                return float(np.mean(diffs))
            churn_b = churn_side(snap["bids"], prev["bids"])
            churn_a = churn_side(snap["asks"], prev["asks"])
            out["churn"] = float(0.5 * (churn_b + churn_a))

            # spoof: была доминирующая стена, резко исчезла
            def max_wall(levels):
                if not levels: return 0.0, 0
                sizes = np.array([float(q) for _, q in levels], dtype=float)
                med = float(np.median(sizes) + 1e-9)
                mx = float(np.max(sizes))
                idx = int(np.argmax(sizes))
                return (mx / max(med, 1e-9)), idx
            cur_bk, _ = max_wall(snap["bids"]); prv_bk, _ = max_wall(prev["bids"])
            cur_ak, _ = max_wall(snap["asks"]); prv_ak, _ = max_wall(prev["asks"])
            spoof_b = 1.0 if (prv_bk >= TAC_SPOOF_K and (cur_bk / max(prv_bk, 1e-9)) <= (1.0 - TAC_SPOOF_DROP)) else 0.0
            spoof_a = 1.0 if (prv_ak >= TAC_SPOOF_K and (cur_ak / max(prv_ak, 1e-9)) <= (1.0 - TAC_SPOOF_DROP)) else 0.0
            out["spoof_score"] = float(spoof_b + spoof_a)  # 0/1/2
    except Exception:
        pass
    return out

# ============================== SMC/Liquidity ==============================
def _fractals(df: pd.DataFrame, left: int = 3, right: int = 3) -> Tuple[List[int], List[int]]:
    highs, lows = [], []
    try:
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        n = len(df)
        for i in range(left, n - right):
            seg_h = h[i - left:i + right + 1]
            seg_l = l[i - left:i + right + 1]
            if h[i] == np.max(seg_h) and np.argmax(seg_h) == left:
                highs.append(i)
            if l[i] == np.min(seg_l) and np.argmin(seg_l) == left:
                lows.append(i)
    except Exception:
        return [], []
    return highs, lows

def _eqh_eql(df: pd.DataFrame, lookback: int = 50, tol_frac: float = TAC_EQ_TOL_FRAC) -> Tuple[bool, bool]:
    try:
        h = df["high"].tail(lookback).values.astype(float)
        l = df["low"].tail(lookback).values.astype(float)
        hi = float(np.max(h)); lo = float(np.min(l))
        eqh = np.sum(np.abs(h - hi) <= tol_frac * max(1e-9, hi)) >= 2
        eql = np.sum(np.abs(l - lo) <= tol_frac * max(1e-9, lo)) >= 2
        return bool(eqh), bool(eql)
    except Exception:
        return False, False

def _fvg(df: pd.DataFrame, lookback: int = TAC_FVG_LOOKBACK) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    if df is None or len(df) < 3:
        return out
    try:
        h = df["high"].astype(float).values
        l = df["low"].astype(float).values
        for i in range(1, min(lookback, len(df)-1)):
            j = len(df) - 1 - i
            if j - 1 >= 0 and j + 1 < len(df):
                if l[j + 1] > h[j - 1]:
                    out.append({"type": 1, "upper": l[j + 1], "lower": h[j - 1], "bar": j})
                if h[j + 1] < l[j - 1]:
                    out.append({"type": -1, "upper": l[j - 1], "lower": h[j + 1], "bar": j})
    except Exception:
        return out
    return out

def _ob_find(df: pd.DataFrame, side: str, atr_val: float, lookback: int = TAC_OB_LOOKBACK) -> Optional[Dict[str, float]]:
    try:
        if df is None or len(df) < lookback + 10:
            return None
        med_body = float((df["close"] - df["open"]).abs().tail(50).median())
        for i in range(len(df) - 3, max(10, len(df) - lookback), -1):
            o = float(df["open"].iloc[i]); c = float(df["close"].iloc[i])
            h = float(df["high"].iloc[i]); l = float(df["low"].iloc[i])
            body = abs(c - o); rng = h - l
            strong = (rng > 1.1 * atr_val) and (body > 0.75 * rng) and (body > 1.2 * med_body)
            if not strong:
                continue
            if side == "LONG" and c > o:
                for j in range(i - 1, max(0, i - 8), -1):
                    o2 = float(df["open"].iloc[j]); c2 = float(df["close"].iloc[j])
                    if c2 < o2:
                        lo = min(o2, c2); hi = max(o2, c2)
                        return {"type": 1, "low": lo, "high": hi, "mid": (lo + hi)/2.0, "bar": j}
            if side == "SHORT" and c < o:
                for j in range(i - 1, max(0, i - 8), -1):
                    o2 = float(df["open"].iloc[j]); c2 = float(df["close"].iloc[j])
                    if c2 > o2:
                        lo = min(o2, c2); hi = max(o2, c2)
                        return {"type": -1, "low": lo, "high": hi, "mid": (lo + hi)/2.0, "bar": j}
    except Exception:
        return None
    return None

def _volume_profile(df: pd.DataFrame, bins: int = 40, lookback: int = 240) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if df is None or len(df) < 20:
        return None, None, None
    try:
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
        total = float(hist.sum())
        order = np.argsort(hist)[::-1]
        acc = 0.0
        mask = np.zeros_like(hist, dtype=bool)
        for idx in order:
            mask[idx] = True
            acc += float(hist[idx])
            if acc / total >= 0.68:
                break
        sel = np.where(mask)[0]
        val = float(edges[sel.min()])
        vah = float(edges[sel.max() + 1])
        return poc, vah, val
    except Exception:
        return None, None, None

def _initial_balance(df15: pd.DataFrame, ib_hours: int = 1) -> Tuple[Optional[float], Optional[float]]:
    try:
        if df15 is None or len(df15) < 5:
            return None, None
        ts = pd.to_datetime(df15["ts"], errors="coerce", utc=True)
        last_ts = ts.iloc[-1]
        day_anchor = last_ts.replace(hour=0, minute=0, second=0, microsecond=0)
        cur = df15[ts >= day_anchor]
        if cur.empty:
            return None, None
        bars = max(1, int(ib_hours * 60 / 15))
        ib = cur.head(bars)
        return float(ib["high"].max()), float(ib["low"].min())
    except Exception:
        return None, None

# ============================== BREAKOUT/COMPRESSION ==============================
def _breakout_quality(df15: pd.DataFrame, atr_val: float) -> Dict[str, float]:
    """
    pre_compression_rank (0..100), breakout_disp (в ATR), quickfail_risk (0..1)
    """
    out = {"pre_compression_rank": None, "breakout_disp": None, "quickfail_risk": None}
    try:
        if df15 is None or len(df15) < 60 or atr_val <= 0:
            return out
        bb = _bbwp(df15["close"], 20, 96)
        out["pre_compression_rank"] = bb

        prev_hi = float(df15["high"].iloc[-41:-1].max())
        prev_lo = float(df15["low"].iloc[-41:-1].min())
        o = float(df15["open"].iloc[-1]); c = float(df15["close"].iloc[-1]); h = float(df15["high"].iloc[-1]); l = float(df15["low"].iloc[-1])

        body = abs(c - o)
        disp = float((h - l) / max(1e-9, atr_val))
        cond_up = (c > prev_hi) and (body >= TAC_BREAKOUT_BODY_ATR * atr_val) and ((c - prev_hi) >= TAC_BREAKOUT_CLOSE_PCT * (h - l + 1e-9))
        cond_dn = (c < prev_lo) and (body >= TAC_BREAKOUT_BODY_ATR * atr_val) and ((prev_lo - c) >= TAC_BREAKOUT_CLOSE_PCT * (h - l + 1e-9))
        out["breakout_disp"] = float(disp if (cond_up or cond_dn) else 0.0)

        qf = 0.0
        if cond_up and float(df15["close"].iloc[-1]) <= prev_hi:
            qf = 1.0
        if cond_dn and float(df15["close"].iloc[-1]) >= prev_lo:
            qf = 1.0
        out["quickfail_risk"] = float(qf)
    except Exception:
        pass
    return out

# ============================== REGIMES / MRISK ==============================
def _adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    try:
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

def _mean_reversion_risk(df15: pd.DataFrame, atr_val: float) -> Optional[float]:
    try:
        if df15 is None or len(df15) < 80 or atr_val <= 0:
            return None
        ema21 = df15["close"].ewm(span=21, adjust=False).mean()
        dev = (df15["close"] - ema21)
        mr = float((dev.iloc[-1]) / (atr_val + 1e-9))
        risk = float(1.0 / (1.0 + math.exp(-abs(mr))))
        return float(np.clip(risk, 0.0, 1.0))
    except Exception:
        return None

# ============================== LIQUIDITY MAP ==============================
def _liquidity_map(app: Dict[str, Any], symbol: str, df15: pd.DataFrame, df1h: pd.DataFrame, atr_val: float, side: Optional[str]) -> Dict[str, Any]:
    out = {
        "ta_eqh": False, "ta_eql": False,
        "ta_poc": None, "ta_vah": None, "ta_val": None,
        "ta_ib_hi": None, "ta_ib_lo": None,
        "ta_ob_type": None, "ta_ob_low": None, "ta_ob_high": None,
        "ta_fvg_side": None, "ta_fvg_upper": None, "ta_fvg_lower": None,
        "ta_risk_depth_atr": None, "ta_risk_zone": None,
        "ta_liq_summary": "",
    }
    try:
        eqh, eql = _eqh_eql(df15, lookback=50, tol_frac=TAC_EQ_TOL_FRAC)
        out["ta_eqh"] = bool(eqh); out["ta_eql"] = bool(eql)

        poc, vah, val = _volume_profile(df15, bins=40, lookback=240)
        out["ta_poc"], out["ta_vah"], out["ta_val"] = poc, vah, val

        ib_hi, ib_lo = _initial_balance(df15, ib_hours=1)
        out["ta_ib_hi"], out["ta_ib_lo"] = ib_hi, ib_lo

        c15 = float(df15["close"].iloc[-1])

        ob15 = _ob_find(df15, side or "LONG", atr_val, lookback=TAC_OB_LOOKBACK) if side else None
        if not ob15 and df1h is not None:
            ob15 = _ob_find(df1h, side or "LONG", atr_val, lookback=TAC_OB_LOOKBACK)
        if ob15:
            out["ta_ob_type"] = ("demand" if ob15["type"] == 1 else "supply")
            out["ta_ob_low"], out["ta_ob_high"] = float(ob15["low"]), float(ob15["high"])

        fvgs = _fvg(df15, lookback=TAC_FVG_LOOKBACK)
        if not fvgs and df1h is not None:
            fvgs = _fvg(df1h, lookback=min(TAC_FVG_LOOKBACK, 80))
        if fvgs:
            z = fvgs[0]
            out["ta_fvg_side"] = ("bull" if z["type"] == 1 else "bear")
            out["ta_fvg_upper"], out["ta_fvg_lower"] = float(z["upper"]), float(z["lower"])

        # Depth of risk: минимальная дистанция в ATR до опасной зоны по направлению сделки
        def atr_dist(price: float) -> float:
            return abs(price - c15) / (atr_val + 1e-9)

        candidates = []
        if side == "LONG":
            if vah and vah > c15: candidates.append(("VAH", atr_dist(vah)))
            if poc and poc > c15: candidates.append(("POC", atr_dist(poc)))
            if eqh:
                hi1h = float(df1h["high"].tail(120).max()) if df1h is not None and len(df1h) >= 30 else float(df15["high"].tail(80).max())
                if hi1h > c15: candidates.append(("EQH/High", atr_dist(hi1h)))
            if out["ta_ob_type"] == "supply" and out["ta_ob_high"] and out["ta_ob_high"] > c15:
                candidates.append(("OB(supply)", atr_dist(out["ta_ob_high"])))
            if ib_hi and ib_hi > c15: candidates.append(("IB-HI", atr_dist(ib_hi)))
        elif side == "SHORT":
            if val and val < c15: candidates.append(("VAL", atr_dist(val)))
            if poc and poc < c15: candidates.append(("POC", atr_dist(poc)))
            if eql:
                lo1h = float(df1h["low"].tail(120).min()) if df1h is not None and len(df1h) >= 30 else float(df15["low"].tail(80).min())
                if lo1h < c15: candidates.append(("EQL/Low", atr_dist(lo1h)))
            if out["ta_ob_type"] == "demand" and out["ta_ob_low"] and out["ta_ob_low"] < c15:
                candidates.append(("OB(demand)", atr_dist(out["ta_ob_low"])))
            if ib_lo and ib_lo < c15: candidates.append(("IB-LO", atr_dist(ib_lo)))

        if candidates:
            zone, depth = sorted(candidates, key=lambda x: x[1])[0]
            out["ta_risk_zone"] = zone
            out["ta_risk_depth_atr"] = float(depth)

        bits = []
        if eqh: bits.append("EQH↑")
        if eql: bits.append("EQL↓")
        if vah and val: bits.append("VAH/VAL")
        if poc: bits.append("POC")
        if ib_hi and ib_lo: bits.append("IB")
        if out["ta_ob_type"]: bits.append("OB " + ("S" if out["ta_ob_type"]=="supply" else "D"))
        if out["ta_fvg_side"]: bits.append("FVG " + ("↑" if out["ta_fvg_side"]=="bull" else "↓"))
        if out["ta_risk_zone"] and out["ta_risk_depth_atr"] is not None:
            bits.append(f"Risk {out['ta_risk_depth_atr']:.2f} ATR → {out['ta_risk_zone']}")
        out["ta_liq_summary"] = " | ".join(bits[:6])
    except Exception:
        pass
    return out

# ============================== BUNDLE ==============================
def _compute_ta_bundle(app: Dict[str, Any], symbol: str, side: Optional[str]) -> Dict[str, Any]:
    logger = app.get("logger")
    prom = app.setdefault("_TAC_PROM", _prom_init(app))
    market = app.get("market")
    atr_fn = app.get("atr")

    bundle: Dict[str, Any] = {}
    t0 = _now_utc().timestamp()
    try:
        df5  = market.fetch_ohlcv(symbol, "5m", 700)
        df15 = market.fetch_ohlcv(symbol, "15m", 500)
        df1h = market.fetch_ohlcv(symbol, "1h", 420)
        df4h = market.fetch_ohlcv(symbol, "4h", 420)
        if df15 is None or df5 is None or len(df15) < 80 or len(df5) < 80:
            return bundle

        atr15 = float(atr_fn(df15, 14).iloc[-1]) if callable(atr_fn) else float((df15["high"]-df15["low"]).rolling(14).mean().iloc[-1])

        # EXEC (orderbook)
        if TAC_EXEC:
            ob = _fetch_order_book(app, symbol, depth=25)
            if ob:
                feat = _book_features(ob, atr15)
                for k, v in feat.items():
                    bundle["ta_" + k] = v

        # BREAKOUT/COMPRESSION
        if TAC_BREAKOUT:
            br = _breakout_quality(df15, atr15)
            bundle.update({ "ta_pre_comp_rank": br.get("pre_compression_rank"),
                            "ta_breakout_disp": br.get("breakout_disp"),
                            "ta_quickfail_risk": br.get("quickfail_risk") })

        # REGIME / MRISK
        if TAC_VOLREG:
            adx15 = _adx(df15, 14)
            bundle["ta_adx15_alt"] = adx15
            mrisk = _mean_reversion_risk(df15, atr15)
            bundle["ta_mr_risk"] = mrisk

        # LIQUIDITY MAP
        if TAC_LIQMAP:
            liq = _liquidity_map(app, symbol, df15, df1h, atr15, side)
            bundle.update(liq)

        # DERIV DEEP (best-effort, каркас)
        if TAC_DERIV_DEEP:
            bundle.setdefault("ta_fund_jerk", 0.0)
            bundle.setdefault("ta_oi_jerk", 0.0)
            bundle.setdefault("ta_basis_slope", 0.0)
            bundle.setdefault("ta_oi_div", 0.0)

        # META p*
        if TAC_META:
            probs = []
            for key, scale, center in [
                ("ta_pre_comp_rank", 0.01, 0.5),      # 0..100 → 0..1
                ("ta_breakout_disp",  0.25, 0.0),     # ATR‑норм
                ("ta_adx15_alt",      1.0/50.0, 0.0), # ~50 → 1.0
            ]:
                v = bundle.get(key)
                if v is None:
                    continue
                p = float(np.clip(scale * float(v) + center, 0.0, 1.0))
                probs.append(p)
            if probs:
                p_meta = float(np.mean(probs))
                bundle["ta_p_meta"] = p_meta
                bundle["ta_grade"] = "A" if p_meta >= 0.75 else "B" if p_meta >= 0.55 else "C"
    except Exception as e:
        logger and logger.debug("tacoin: bundle error for %s: %s", symbol, e)
    finally:
        t1 = _now_utc().timestamp()
        if prom.get("TA_BUNDLE_LAT"):
            with contextlib.suppress(Exception):
                prom["TA_BUNDLE_LAT"].observe(max(0.0, t1 - t0))
    return bundle

# ============================== SCORE INTEGRATION ==============================
def _apply_tacoin_scoring(details: Dict[str, Any], side: str) -> Tuple[float, Dict[str, float]]:
    brk: Dict[str, float] = {}
    adj = 0.0

    # EXEC
    if TAC_EXEC:
        near = details.get("ta_near_depth_ratio")
        imb  = details.get("ta_book_imb")
        spr  = details.get("ta_spread_norm")
        churn= details.get("ta_churn")
        spoof= details.get("ta_spoof_score")

        if isinstance(near, (int, float)) and near >= TAC_NEAR_DEPTH_MIN:
            brk["ExecNear"] = brk.get("ExecNear", 0.0) + TAC_W_EXEC * 0.5
            adj += TAC_W_EXEC * 0.5
        if isinstance(imb, (int, float)):
            if (side == "LONG" and imb > 0.15) or (side == "SHORT" and imb < -0.15):
                brk["ExecImb"] = brk.get("ExecImb", 0.0) + TAC_W_EXEC * 0.5
                adj += TAC_W_EXEC * 0.5
        if isinstance(spr, (int, float)) and spr > TAC_SPREAD_Z_MAX:
            brk["ExecSpread"] = brk.get("ExecSpread", 0.0) + TAC_W_SPREADZ_PENALTY
            adj += TAC_W_SPREADZ_PENALTY
        if isinstance(churn, (int, float)) and churn > TAC_CHURN_MAX:
            brk["ExecChurn"] = brk.get("ExecChurn", 0.0) + (TAC_W_EXEC * -0.5)
            adj += (TAC_W_EXEC * -0.5)
        if isinstance(spoof, (int, float)) and spoof >= 1.0:
            brk["ExecSpoof"] = brk.get("ExecSpoof", 0.0) + TAC_W_SPOOF_PENALTY
            adj += TAC_W_SPOOF_PENALTY

    # BREAKOUT/COMPRESSION
    if TAC_BREAKOUT:
        pre = details.get("ta_pre_comp_rank")
        disp = details.get("ta_breakout_disp")
        qf = details.get("ta_quickfail_risk")
        if isinstance(pre, (int, float)) and pre >= TAC_COMPRESSION_Q_MIN:
            brk["BreakComp"] = brk.get("BreakComp", 0.0) + (TAC_W_BREAKOUT * 0.5)
            adj += (TAC_W_BREAKOUT * 0.5)
        if isinstance(disp, (int, float)) and disp >= TAC_BREAKOUT_BODY_ATR:
            brk["BreakDisp"] = brk.get("BreakDisp", 0.0) + (TAC_W_BREAKOUT * 0.5)
            adj += (TAC_W_BREAKOUT * 0.5)
        if isinstance(qf, (int, float)) and qf >= TAC_QUICKFAIL_RISK_MAX:
            brk["QuickFail"] = brk.get("QuickFail", 0.0) + TAC_W_QUICKFAIL
            adj += TAC_W_QUICKFAIL

    # LIQUIDITY RISK
    if TAC_LIQMAP:
        depth = details.get("ta_risk_depth_atr")
        if isinstance(depth, (int, float)) and depth < 0.6:
            brk["LiqRisk"] = brk.get("LiqRisk", 0.0) + TAC_W_LIQRISK
            adj += TAC_W_LIQRISK
        else:
            brk["LiqSafe"] = brk.get("LiqSafe", 0.0) + (abs(TAC_W_LIQRISK) * 0.25)
            adj += (abs(TAC_W_LIQRISK) * 0.25)

    # VOLREG / MR
    if TAC_VOLREG:
        mr = details.get("ta_mr_risk")
        if isinstance(mr, (int, float)) and mr > 0.8:
            brk["MRisk"] = brk.get("MRisk", 0.0) + TAC_W_MRISK
            adj += TAC_W_MRISK
        adx_alt = details.get("ta_adx15_alt")
        if isinstance(adx_alt, (int, float)) and adx_alt >= 25.0:
            brk["Persist"] = brk.get("Persist", 0.0) + TAC_W_PERSIST
            adj += TAC_W_PERSIST

    # META
    if TAC_META:
        p = details.get("ta_p_meta")
        if isinstance(p, (int, float)):
            shift = (p - 0.5) * 2.0  # -1..+1
            adj += (TAC_W_META_CAL * shift)
            if shift >= 0:
                brk["Meta+"] = brk.get("Meta+", 0.0) + (TAC_W_META_CAL * shift)
            else:
                brk["Meta-"] = brk.get("Meta-", 0.0) + (TAC_W_META_CAL * shift)

    return float(adj), brk

# ============================== TP/SL BUILDER ==============================
def _infer_safe_tick(entry: float, df15: Optional[pd.DataFrame]) -> float:
    """
    Если биржевой tick подозрителен/отсутствует — оцениваем безопасный шаг.
    Базово от цены + корректировка по типичной волатильности (0.01..1% цены).
    """
    try:
        if entry <= 0:
            return 1e-6
        # базовые десятичные
        if entry >= 100: decimals = 2
        elif entry >= 10: decimals = 3
        elif entry >= 1: decimals = 4
        elif entry >= 0.1: decimals = 5
        else:
            # «мем»-диапазон
            decimals = TAC_SAFE_DECIMALS_MIN
        decimals = int(_clamp(decimals, TAC_SAFE_DECIMALS_MIN if entry < 0.01 else 2, TAC_SAFE_DECIMALS_MAX))
        tick = 10 ** (-decimals)
        # поправка по спреду (если доступен df15): чем меньше спред, тем меньше шаг
        if df15 is not None and len(df15) >= 20:
            med_rng = float((df15["high"] - df15["low"]).tail(20).median())
            est_spread = med_rng * 0.1
            if est_spread > 0:
                tick = min(tick, est_spread / 10.0)
        return float(max(tick, 10 ** (-TAC_SAFE_DECIMALS_MAX)))
    except Exception:
        return 10 ** (-TAC_SAFE_DECIMALS_MIN)

def _round_levels_near(price: float) -> List[float]:
    """
    Возвращает «круглые» уровни рядом с ценой (для привязки TP).
    Варьируем шаг в зависимости от масштаба цены.
    """
    try:
        if price <= 0:
            return []
        step = 0.0
        if price < 0.01:
            step = 10 ** (-6)
        elif price < 0.1:
            step = 10 ** (-4)
        elif price < 1:
            step = 0.01
        elif price < 10:
            step = 0.1
        elif price < 100:
            step = 1.0
        else:
            step = 5.0
        base = math.floor(price / step) * step
        return [base - 2*step, base - step, base, base + step, base + 2*step]
    except Exception:
        return []

def _ensure_tp_monotonic_with_step(side: str, entry: float, tps: List[float], atr: float, tick: float, min_step_atr: float) -> List[float]:
    tps = [float(x) for x in tps[:3]] + ([tps[-1]] * max(0, 3 - len(tps)))
    step = max(1e-9, float(min_step_atr) * float(atr))
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

def _tp_cap(entry: float, tp: float, atr: float, side: str, tp_max_pct: float, tp_cap_atr_max: float) -> float:
    if side == "LONG":
        cap1 = entry * (1.0 + tp_max_pct)
        cap2 = entry + tp_cap_atr_max * atr
        return min(tp, cap1, cap2)
    else:
        cap1 = entry * (1.0 - tp_max_pct)
        cap2 = entry - tp_cap_atr_max * atr
        return max(tp, cap1, cap2)

def _meme_mode(entry: float) -> bool:
    return entry < 0.01

def _tp_sl_builder(app: Dict[str, Any],
                   symbol: str,
                   side: str,
                   entry: float,
                   sl: float,
                   tps: List[float],
                   atr: float,
                   df15: Optional[pd.DataFrame],
                   df1h: Optional[pd.DataFrame],
                   tick_real: Optional[float]) -> Tuple[float, List[float]]:
    """
    Единый перестроитель TP/SL.
    - безопасный шаг (если tick_real подозрителен);
    - структурный SL в пределах [min..max] ATR;
    - TP‑ladder с привязкой к IB/POC/VA/«круглым».
    """
    prom = app.setdefault("_TAC_PROM", _prom_init(app))
    logger = app.get("logger")
    # шаг цены
    safe_tick = _infer_safe_tick(entry, df15)
    tick = float(min(tick_real or safe_tick, safe_tick))
    # min/max риск к SL (повторяем логику chat.py, но мягко)
    min_risk_atr = app.get("TA_SAN_SL_MIN_ATR", 0.25)
    max_risk_atr = app.get("TA_SAN_SL_MAX_ATR", 5.0)
    min_risk = float(min_risk_atr) * float(atr)
    max_risk = float(max_risk_atr) * float(atr)

    # SL: исправить «сторону» и риск-дистанцию
    s = float(sl)
    if side == "LONG":
        if not (s < entry):
            s = entry - min_risk
        risk = abs(entry - s)
        if risk < min_risk: s = entry - min_risk
        if risk > max_risk: s = entry - max_risk
        s = _round_tick(s, tick, "floor")
        if s <= 0: s = _round_tick(max(entry - min_risk, tick * 5), tick, "floor")
    else:
        if not (s > entry):
            s = entry + min_risk
        risk = abs(entry - s)
        if risk < min_risk: s = entry + min_risk
        if risk > max_risk: s = entry + max_risk
        s = _round_tick(s, tick, "ceil")
        if s <= 0: s = _round_tick(entry + min_risk, tick, "ceil")

    # базовые множители
    muls = TAC_TP_LADDER_MEME if _meme_mode(entry) else TAC_TP_LADDER
    if not muls:
        muls = [0.7, 1.6, 2.6]
    # черновые TP от ATR
    raw = []
    for m in muls[:3]:
        if side == "LONG":
            raw.append(entry + m * atr)
        else:
            raw.append(entry - m * atr)

    # привязка к уровням (IB/POC/VA/круглые)
    poc, vah, val = None, None, None
    ib_hi, ib_lo = None, None
    try:
        poc, vah, val = _volume_profile(df15, 40, 240) if df15 is not None else (None, None, None)
    except Exception:
        pass
    try:
        ib_hi, ib_lo = _initial_balance(df15, ib_hours=1) if df15 is not None else (None, None)
    except Exception:
        pass

    def snap_one(t: float, side: str) -> float:
        # кандидатные уровни
        cands: List[float] = _round_levels_near(t)
        if poc: cands.append(float(poc))
        if vah: cands.append(float(vah))
        if val: cands.append(float(val))
        if ib_hi: cands.append(float(ib_hi))
        if ib_lo: cands.append(float(ib_lo))
        # выбрать ближайший по направлению (не ухудшая монотонность далее)
        if side == "LONG":
            cands = [x for x in cands if x > entry]
            if not cands: return t
            best = min(cands, key=lambda x: abs(x - t))
            return best
        else:
            cands = [x for x in cands if x < entry]
            if not cands: return t
            best = min(cands, key=lambda x: abs(x - t))
            return best

    snapped = [snap_one(t, side) for t in raw]
    # кап по ATR и по % (для мемов особенно)
    tp_cap_atr_max = float(app.get("TA_TP_CAP_ATR_MAX", 4.0))
    tp_max_pct = float(TAC_TP_MAX_PCT)
    capped = [_tp_cap(entry, tp, atr, side, tp_max_pct, tp_cap_atr_max) for tp in snapped]

    # финальная монотонность и шаг
    final = _ensure_tp_monotonic_with_step(side, entry, capped, atr, tick, float(app.get("TA_SAN_TP_MIN_STEP_ATR", 0.15)))

    # завершающее округление
    if side == "LONG":
        final = [_round_tick(x, tick, "ceil") for x in final]
    else:
        final = [_round_tick(x, tick, "floor") for x in final]

    # валидация: TP != 0/1, различаются друг от друга и от entry
    prom_c = app.setdefault("_TAC_PROM", _prom_init(app))
    def _is_bad(v: float) -> bool:
        return v <= 0 or (entry < 0.5 and abs(v - 1.0) < 1e-12)
    if any(_is_bad(v) for v in final):
        if prom_c.get("TA_TP_ZERO"): prom_c["TA_TP_ZERO"].inc()
        # отступим от крайностей безопасным шагом
        final = [max(v, tick*5) for v in final]

    # одинаковые TP после округления — раздвинуть на min_step
    if len({round(v, 12) for v in final}) < 3:
        if prom_c.get("TA_TP_EQUAL"): prom_c["TA_TP_EQUAL"].inc()
        final = _ensure_tp_monotonic_with_step(side, entry, final, atr, tick, float(app.get("TA_SAN_TP_MIN_STEP_ATR", 0.15)))

    # кап по % от entry (защита от «1.000000»)
    def pct_cap(v: float) -> float:
        if side == "LONG":
            return min(v, entry * (1.0 + tp_max_pct))
        else:
            return max(v, entry * (1.0 - tp_max_pct))
    final = [pct_cap(v) for v in final]

    # повторное округление
    if side == "LONG":
        final = [_round_tick(x, tick, "ceil") for x in final]
    else:
        final = [_round_tick(x, tick, "floor") for x in final]

    # sanity на outliers
    for v in final:
        pct = _percent_change(v, entry)
        if pct > tp_max_pct * 1.5 and prom_c.get("TA_TP_OUTLIER"):
            prom_c["TA_TP_OUTLIER"].inc()

    return float(s), [float(x) for x in final[:3]]

def _tp_anomaly(entry: float, sl: float, tps: List[float]) -> bool:
    if not tps or len(tps) < 1:
        return True
    # нули/единицы/NaN
    if any((not isinstance(x, (int, float)) or x <= 0.0) for x in tps):
        return True
    # одинаковые TP (после округления)
    if len({round(x, 8) for x in tps}) < 3:
        return True
    # SL нулевой/не с той стороны
    if entry > 0 and ((sl <= 0) or (sl >= entry and tps[0] > entry) or (sl <= entry and tps[0] < entry)):
        return True
    return False

# ============================== REASON INTEGRATION ==============================
def _quality_line(details: Dict[str, Any]) -> str:
    def g_exec():
        spr = details.get("ta_spread_norm"); near = details.get("ta_near_depth_ratio"); spoof = details.get("ta_spoof_score")
        if spr is None or near is None: return "—"
        if (spr <= TAC_SPREAD_Z_MAX and near >= TAC_NEAR_DEPTH_MIN and (not spoof or spoof < 1)): return "A"
        if (spr <= TAC_SPREAD_Z_MAX * 1.3 and near >= TAC_NEAR_DEPTH_MIN*0.85): return "B"
        return "C"
    def g_break():
        pre = details.get("ta_pre_comp_rank") or 0.0
        disp= details.get("ta_breakout_disp") or 0.0
        qf  = details.get("ta_quickfail_risk") or 0.0
        if pre >= TAC_COMPRESSION_Q_MIN and disp >= TAC_BREAKOUT_BODY_ATR and qf < TAC_QUICKFAIL_RISK_MAX: return "A"
        if (pre >= (TAC_COMPRESSION_Q_MIN*0.7) and disp >= (TAC_BREAKOUT_BODY_ATR*0.7)): return "B"
        return "C"
    def g_reg():
        adx = details.get("ta_adx15_alt") or 0.0
        if adx >= 30: return "A"
        if adx >= 20: return "B"
        return "C"
    def g_liq():
        have = 0
        for k in ("ta_vah","ta_val","ta_poc","ta_ib_hi","ta_ib_lo"):
            if details.get(k) is not None: have += 1
        depth = details.get("ta_risk_depth_atr") or 2.0
        if have >= 3 and depth >= 1.0: return "A"
        if have >= 2 and depth >= 0.6: return "B"
        return "C"
    def g_risk():
        depth = details.get("ta_risk_depth_atr")
        if depth is None: return "—"
        if depth >= 1.2: return "A"
        if depth >= 0.6: return "B"
        return "C"
    lm = details.get("ta_liq_summary") or ""
    return f"Качество: Exec {g_exec()} • Break {g_break()} • Reg {g_reg()} • Liq {g_liq()} • Risk {g_risk()} • LM: {lm}"

# ============================== TRAILING 2.0 ==============================
def _kama(series: pd.Series, er_period: int = 10, fast: int = TAC_KAMA_FAST, slow: int = TAC_KAMA_SLOW) -> pd.Series:
    x = series.astype(float).values
    n = len(x)
    if n == 0: return series.copy()
    change = np.abs(x - np.roll(x, er_period)); change[:er_period] = np.nan
    vol = np.zeros(n)
    for i in range(1, er_period + 1):
        vol += np.abs(x - np.roll(x, i))
    vol[:er_period] = np.nan
    er = np.divide(change, vol, out=np.zeros_like(change), where=(vol != 0))
    fast_sc = 2.0 / (fast + 1.0); slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    out = np.full(n, np.nan, dtype=float)
    out[0] = x[0]
    for i in range(1, n):
        sc_i = sc[i] if not np.isnan(sc[i]) else slow_sc**2
        out[i] = out[i - 1] + sc_i * (x[i] - out[i - 1])
    return pd.Series(out, index=series.index)

def _ehlers_ss(series: pd.Series, period: int = TAC_EHLERS_PERIOD) -> pd.Series:
    x = series.astype(float).values
    n = len(x)
    if n == 0 or period <= 2:
        return series.copy()
    a1 = math.exp(-1.414 * math.pi / period)
    b1 = 2.0 * a1 * math.cos(1.414 * math.pi / period)
    c2 = b1; c3 = -a1 * a1; c1 = 1.0 - c2 - c3
    y = np.zeros(n, dtype=float)
    y[0] = x[0]
    if n >= 2:
        y[1] = (x[0] + x[1]) / 2.0
    for i in range(2, n):
        y[i] = c1 * (x[i] + x[i - 1]) / 2.0 + c2 * y[i - 1] + c3 * y[i - 2]
    return pd.Series(y, index=series.index)

async def _trailing_plus(app: Dict[str, Any], sig) -> None:
    if not TAC_TRAIL2:
        return
    try:
        market = app.get("market")
        atr_fn = app.get("atr")
        df15 = market.fetch_ohlcv(sig.symbol, "15m", 240)
        if df15 is None or len(df15) < 60:
            return
        atrv = float(atr_fn(df15, 14).iloc[-1]) if callable(atr_fn) else float((df15["high"]-df15["low"]).rolling(14).mean().iloc[-1])
        close = df15["close"].astype(float)
        kama = _kama(close, 10, TAC_KAMA_FAST, TAC_KAMA_SLOW)
        ss   = _ehlers_ss(close, TAC_EHLERS_PERIOD)
        cap = float(app.get("TRAIL_ATR_CAP", 0.30))
        if sig.side == "LONG":
            tgt = float(min(kama.iloc[-1], ss.iloc[-1]))
            tgt = max(tgt, sig.entry - cap * atrv)
            if tgt > sig.sl:
                sig.sl = float(tgt)
        else:
            tgt = float(max(kama.iloc[-1], ss.iloc[-1]))
            tgt = min(tgt, sig.entry + cap * atrv)
            if tgt < sig.sl:
                sig.sl = float(tgt)
    except Exception:
        pass

# ============================== WATCH SIDE‑CAR ALERTS ==============================
async def _watch_sidecar(app: Dict[str, Any], bot, chat_id: int, sig):
    if not TAC_ALERTS:
        return
    logger = app.get("logger")
    _should_alert = app.get("_should_alert")
    prom = app.setdefault("_TAC_PROM", _prom_init(app))
    market = app.get("market")
    atr_fn = app.get("atr")
    try:
        while getattr(sig, "active", False) and app["now_msk"]() < sig.watch_until:
            df15 = market.fetch_ohlcv(sig.symbol, "15m", 120)
            if df15 is None or len(df15) < 30:
                await asyncio.sleep(TAC_SIDE_TASK_INTERVAL)
                continue
            atrv = float(atr_fn(df15, 14).iloc[-1]) if callable(atr_fn) else float((df15["high"]-df15["low"]).rolling(14).mean().iloc[-1])
            ob = _fetch_order_book(app, sig.symbol, depth=25)
            feats = _book_features(ob, atrv) if ob else {}
            try:
                if feats.get("spoof_score", 0.0) >= 1.0 and _should_alert and _should_alert(sig.id or -1, "spoof"):
                    await bot.send_message(chat_id, f"⚠️ Spoof‑сигнал в книге по {sig.symbol.split('/')[0]} — подумайте о частичной фиксации.")
                    if prom.get("TA_ALERTS"): prom["TA_ALERTS"].labels(kind="spoof").inc()
                if feats.get("spread_norm") and feats["spread_norm"] > TAC_SPREAD_Z_MAX and _should_alert and _should_alert(sig.id or -1, "spread"):
                    await bot.send_message(chat_id, f"⚠️ Широкий спред по {sig.symbol.split('/')[0]} — риск проскальзывания/ухудшения исполнения.")
                    if prom.get("TA_ALERTS"): prom["TA_ALERTS"].labels(kind="spread").inc()
                if feats.get("churn") and feats["churn"] > TAC_CHURN_MAX and _should_alert and _should_alert(sig.id or -1, "book_churn"):
                    await bot.send_message(chat_id, f"⚠️ Нестабильная книга (churn) по {sig.symbol.split('/')[0]} — повышен риск фейков.")
                    if prom.get("TA_ALERTS"): prom["TA_ALERTS"].labels(kind="book_churn").inc()
            except Exception:
                pass

            try:
                dur_min = max(1.0, (app["now_msk"]() - sig.created_at).total_seconds() / 60.0)
                dist_atr = abs(float(df15["close"].iloc[-1]) - float(sig.entry)) / (atrv + 1e-9)
                if dur_min >= TAC_NO_PROGRESS_MIN and dist_atr < TAC_TIME_STOP_ATR and _should_alert and _should_alert(sig.id or -1, "no_progress"):
                    await bot.send_message(chat_id, f"ℹ️ Нет прогресса по {sig.symbol.split('/')[0]} (<{TAC_TIME_STOP_ATR:.1f}×ATR за {int(dur_min)}м). Рассмотрите частичную фиксацию/перезаход.")
                    if prom.get("TA_ALERTS"): prom["TA_ALERTS"].labels(kind="no_progress").inc()
            except Exception:
                pass

            try:
                poc, vah, val = _volume_profile(df15, 40, 240)
                px = float(df15["close"].iloc[-1])
                fmt = app.get("format_price", lambda v: f"{v:.4f}")
                if sig.side == "LONG" and vah and (0 <= (vah - px) <= 0.4 * atrv) and _should_alert and _should_alert(sig.id or -1, "vp_barrier"):
                    await bot.send_message(chat_id, f"ℹ️ Рядом VAH ({fmt(vah)}) по {sig.symbol.split('/')[0]} — возможен откат, частичная фиксация уместна.")
                    if prom.get("TA_ALERTS"): prom["TA_ALERTS"].labels(kind="vp_barrier").inc()
                if sig.side == "SHORT" and val and (0 <= (px - val) <= 0.4 * atrv) and _should_alert and _should_alert(sig.id or -1, "vp_barrier"):
                    await bot.send_message(chat_id, f"ℹ️ Рядом VAL ({fmt(val)}) по {sig.symbol.split('/')[0]} — возможен отскок, частичная фиксация уместна.")
                    if prom.get("TA_ALERTS"): prom["TA_ALERTS"].labels(kind="vp_barrier").inc()
            except Exception:
                pass

            await asyncio.sleep(TAC_SIDE_TASK_INTERVAL)
    except asyncio.CancelledError:
        return
    except Exception as e:
        logger and logger.debug("tacoin sidecar error: %s", e)

# ============================== PATCH ENTRY ==============================
def patch(app: Dict[str, Any]) -> None:
    logger = app.get("logger")
    logger and logger.info("tacoin: patch start (exec/breakout/liquidity/regimes/meta/trailing/alerts + TP/SL builder)")

    # Оригиналы
    orig_score = app.get("score_symbol_core")
    orig_build_reason = app.get("build_reason")
    orig_watch = app.get("watch_signal_price")
    orig_trailing = app.get("update_trailing")

    # ---------- SCORE WRAP ----------
    if callable(orig_score):
        def _score_tacoin(symbol: str, relax: bool = False):
            base = orig_score(symbol, relax)
            if base is None:
                return None
            score, side, d = base
            d = dict(d or {})
            s_side = str(side or d.get("side") or "LONG")

            # bundle
            t_bundle = _compute_ta_bundle(app, symbol, s_side)
            for k, v in t_bundle.items():
                if k.startswith("ta_") or (k not in d):
                    d[k] = v

            if TAC_AB:
                logger and logger.info("tacoin AB before score=%.3f", float(score))

            # скоринг
            if not TAC_DRYRUN:
                adj, delta = _apply_tacoin_scoring(d, s_side)
                d["score_breakdown"] = {**(d.get("score_breakdown", {}) or {}), **delta}
                score = float(score) + float(adj)

            # TP/SL builder (фиксация дубликатов/нулей/«1.000000» и т.п.)
            if TAC_TPB_ENABLE:
                try:
                    entry = float(d.get("c5"))
                    sl0 = float(d.get("sl"))
                    atr = float(d.get("atr", 0.0))
                    tps0 = [float(x) for x in (d.get("tps") or [])]
                    # попытка получить биржевой tick
                    tick_real = None
                    try:
                        tick_real = float(app.get("market").get_tick_size(symbol) or 0.0)
                    except Exception:
                        tick_real = None
                    # df для оценки уровней
                    df15 = app.get("market").fetch_ohlcv(symbol, "15m", 240)
                    df1h = app.get("market").fetch_ohlcv(symbol, "1h", 420)
                    need_fix = TAC_TPB_FORCE or (TAC_TPB_AUTO and _tp_anomaly(entry, sl0, tps0))
                    if need_fix and atr > 0 and entry > 0:
                        s1, t1 = _tp_sl_builder(app, symbol, s_side, entry, sl0, tps0, atr, df15, df1h, tick_real)
                        d["sl"] = float(s1)
                        d["tps"] = [float(x) for x in t1]
                except Exception as e:
                    logger and logger.debug("tacoin: tp/sl builder error for %s: %s", symbol, e)

            # LM summary
            if t_bundle.get("ta_liq_summary"):
                d["ta_liq_summary"] = t_bundle["ta_liq_summary"]

            if TAC_AB:
                logger and logger.info("tacoin AB after  score=%.3f", float(score))

            return float(score), s_side, d

        app["score_symbol_core"] = _score_tacoin
        logger and logger.info("tacoin: score_symbol_core wrapped.")

    # ---------- REASON WRAP ----------
    if callable(orig_build_reason):
        def _build_reason_tacoin(details: Dict[str, Any]) -> str:
            base = ""
            try:
                base = orig_build_reason(details) or ""
            except Exception:
                base = ""
            try:
                qline = _quality_line(details)
                if qline:
                    if base:
                        return (base + " • " + qline)[:1100]
                    return qline[:1100]
            except Exception:
                return base
        app["build_reason"] = _build_reason_tacoin
        logger and logger.info("tacoin: build_reason wrapped.")

    # ---------- TRAILING WRAP ----------
    if callable(orig_trailing):
        async def _trailing_wrap(sig):
            try:
                await orig_trailing(sig)
            except Exception:
                pass
            try:
                await _trailing_plus(app, sig)
            except Exception:
                pass
        app["update_trailing"] = _trailing_wrap
        logger and logger.info("tacoin: update_trailing wrapped.")

    # ---------- WATCH WRAP (side‑car alerts) ----------
    if callable(orig_watch):
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
        logger and logger.info("tacoin: watch_signal_price wrapped.")

    logger and logger.info("tacoin: patch applied successfully.")
