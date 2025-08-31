# dio.py
# Вспомогательная аналитика: macro, корреляции, BOCPD-lite, новости, L2-фичи, калибратор outcomes, announcements.
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os, math, time, contextlib, asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# optional feedparser for announcements
try:
    import feedparser
except Exception:
    feedparser = None

TF_SEC = {"1m":60,"3m":180,"5m":300,"15m":900,"30m":1800,"1h":3600,"4h":14400,"1d":86400,"1w":604800}

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

# ---------- Macro events ----------
def parse_macro_events_env(env_val: str) -> List[datetime]:
    out = []
    for part in (env_val or "").split(","):
        s = part.strip()
        if not s: continue
        try:
            if s.endswith("Z"): s = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
            out.append(dt.astimezone(timezone.utc))
        except Exception:
            continue
    return sorted(out)

def minutes_to_next_event(events: List[datetime]) -> Optional[float]:
    now = now_utc()
    future = [e for e in events if e > now]
    if not future: return None
    mins = (future[0] - now).total_seconds() / 60.0
    return float(mins)

# ---------- Stale guard ----------
def df_is_stale(df: Optional[pd.DataFrame], tf: str, max_mult: float = 3.0) -> bool:
    if df is None or len(df) == 0: return True
    try:
        last_ts = pd.to_datetime(df["ts"].iloc[-1], utc=True, errors="coerce")
        if pd.isna(last_ts): return True
        sec = TF_SEC.get(tf, 0) or 60
        delta = (now_utc() - last_ts.to_pydatetime()).total_seconds()
        return delta > max_mult * sec
    except Exception:
        return False

# ---------- Correlation & trend ----------
def pearson_corr(a: pd.Series, b: pd.Series, window: int = 180) -> float:
    try:
        ar = a.pct_change().dropna().tail(window); br = b.pct_change().dropna().tail(window)
        n = min(len(ar), len(br))
        if n < 30: return 0.0
        return float(np.corrcoef(ar.tail(n), br.tail(n))[0,1])
    except Exception:
        return 0.0

def simple_trend(series: pd.Series, win: int = 200) -> int:
    try:
        y = series.tail(win).astype(float).values
        x = np.arange(len(y), dtype=float)
        if len(y) < 10 or np.std(x) < 1e-9: return 0
        sl,_ = np.polyfit(x,y,1)
        return 1 if sl>0 else (-1 if sl<0 else 0)
    except Exception:
        return 0

# ---------- HMM-lite / regimes ----------
def hmm_regime(df: Optional[pd.DataFrame]) -> str:
    try:
        if df is None or len(df) < 120: return "unknown"
        close = df["close"].astype(float)
        ret = close.pct_change().dropna()
        vs = float(ret.tail(24).std(ddof=0) + 1e-12)
        vl = float(ret.tail(96).std(ddof=0) + 1e-12)
        ratio = vs / vl
        if ratio >= 1.8: return "shock"
        if ratio <= 0.8: return "calm"
        return "trend" if ratio > 1.1 else "range"
    except Exception:
        return "unknown"

def cp_volatility(df: Optional[pd.DataFrame]) -> float:
    # change-point proxy: |σ_short/σ_long - 1|
    try:
        if df is None or len(df) < 120: return 0.0
        ret = df["close"].astype(float).pct_change().dropna()
        vs = float(ret.tail(24).std(ddof=0) + 1e-12)
        vl = float(ret.tail(96).std(ddof=0) + 1e-12)
        return float(abs(vs/vl - 1.0))
    except Exception:
        return 0.0

# ---------- News scoring ----------
POS_KW = ["listing","approval","approve","etf","funding","partnership","mainnet","burn","integration","adoption","inflow","upgrade","launch","support","roadmap","ath","all-time high"]
NEG_KW = ["hack","exploit","rug","lawsuit","ban","sec","delist","outage","breach","dump","sell-off","crackdown","security","suspension","vulnerability","phishing","sanction","maintenance"]

def news_logistic_p(note: str) -> Optional[float]:
    try:
        t = (note or "").lower()
        if not t: return None
        x = 0.0
        for k in POS_KW:
            if k in t: x += 1.0
        for k in NEG_KW:
            if k in t: x -= 1.0
        return float(1.0/(1.0 + math.exp(-0.9*x)))
    except Exception:
        return None

def parse_news_note(app: Dict[str, Any], note: str) -> Tuple[float,float]:
    try:
        p = app.get("_parse_news_note")
        if callable(p):
            neg,pos,_ = p(note or "")
            return float(neg or 0.0), float(pos or 0.0)
    except Exception:
        pass
    return 0.0, 0.0

# ---------- Orderbook features ----------
def fetch_order_book(app: Dict[str, Any], symbol: str, depth: int = 25) -> Optional[Dict[str, Any]]:
    market = app.get("market")
    last_err = None
    try:
        for name, ex in market._available_exchanges():
            resolved = market.resolve_symbol(ex, symbol) or symbol
            if resolved not in ex.markets:
                continue
            try:
                return ex.fetch_order_book(resolved, limit=depth)
            except Exception as e:
                last_err = e
                continue
    except Exception:
        pass
    logger = app.get("logger")
    logger and logger.debug("dio: order_book fetch failed: %s", last_err)
    return None

def ob_features(ob: Dict[str,Any]) -> Dict[str,float]:
    out = {"q_imb": None, "best_spread": None}
    try:
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        if not bids or not asks: return out
        pb,qb = float(bids[0][0]), float(bids[0][1])
        pa,qa = float(asks[0][0]), float(asks[0][1])
        out["best_spread"] = float(max(0.0, pa - pb))
        tot = qb + qa + 1e-9
        out["q_imb"] = float((qb - qa) / tot)
        return out
    except Exception:
        return out

# ---------- Calibrator (outcomes-driven) ----------
async def vino_calibrator(app: Dict[str, Any], interval_sec: int = 6*3600):
    logger = app.get("logger")
    db = app.get("db")
    await asyncio.sleep(10)
    while True:
        try:
            if not db or not getattr(db, "conn", None):
                await asyncio.sleep(interval_sec); continue
            cur = await db.conn.execute("SELECT outcome, COUNT(1) AS n FROM outcomes GROUP BY outcome")
            rows = await cur.fetchall()
            stats = {str(r["outcome"]): int(r["n"]) for r in rows or []}
            total = sum(stats.values()) or 1
            stop_rate = (stats.get("STOP",0)+stats.get("TIME",0)) / total
            tp1_rate = stats.get("TP1",0)/total
            cfg = app.setdefault("_vino_cfg", {"min_score_adj":0.0, "ladder":{}})
            target_adj = cfg.get("min_score_adj", 0.0)
            target_adj += (0.05 if stop_rate > 0.45 else (-0.05 if tp1_rate > 0.55 else 0.0))
            cfg["min_score_adj"] = max(-0.20, min(0.40, target_adj))
            # per-coin pseudo ladder adapt
            try:
                cur2 = await db.conn.execute("SELECT symbol, AVG(rr1) AS r1, AVG(rr2) AS r2 FROM outcomes WHERE rr1 IS NOT NULL GROUP BY symbol")
                rows2 = await cur2.fetchall()
                lad = cfg.get("ladder", {})
                for r in rows2 or []:
                    sym = str(r["symbol"])
                    r1 = float(r["r1"] or 1.2)
                    r2 = float(r["r2"] or 1.6)
                    m1 = float(max(0.6, min(1.0, r1 * 0.8)))
                    m2 = float(max(1.2, min(1.8, r2 * 0.9)))
                    lad[sym] = (m1, m2, 2.6)
                cfg["ladder"] = lad
            except Exception:
                pass
            logger and logger.info("VINO calibrator: outcomes=%s → min_score_adj=%.2f, ladders=%d", stats, cfg["min_score_adj"], len(cfg.get("ladder",{})))
        except Exception as e:
            logger and logger.warning("VINO calibrator error: %s", e)
        await asyncio.sleep(interval_sec)

# ---------- Recent outcomes (stop-day guard) ----------
async def recent_outcomes_stats(app: Dict[str, Any], hours: int = 12) -> Dict[str,int]:
    db = app.get("db")
    res = {"TP1":0,"TP2":0,"TP3":0,"BE":0,"STOP":0,"TIME":0,"N":0}
    try:
        if not db or not getattr(db, "conn", None):
            return res
        since = (now_utc() - timedelta(hours=hours)).isoformat()
        cur = await db.conn.execute(
            "SELECT outcome, COUNT(1) AS n FROM outcomes WHERE finished_at IS NOT NULL AND finished_at >= ? GROUP BY outcome",
            (since,)
        )
        rows = await cur.fetchall()
        for r in rows or []:
            k = str(r["outcome"]).upper()
            n = int(r["n"] or 0)
            if k in res: res[k] += n
            res["N"] += n
    except Exception:
        pass
    return res

# ---------- Announcements (RSS) ----------
_ANN_CACHE: Dict[str, Tuple[float, Dict[str, Tuple[float,str,bool]]]] = {}

def _coin_aliases(app: Dict[str,Any], base: str) -> List[str]:
    try:
        syn = app.get("COIN_SYNONYMS") or {}
        arr = syn.get(base.upper(), [])
        out = [base.upper(), "$"+base.upper(), base.lower()]
        out += [str(x) for x in arr]
        return list(set(out))
    except Exception:
        return [base.upper(), base.lower()]

def announcements_boost(app: Dict[str,Any], base: str, ttl: int = 600) -> Tuple[float,str,bool]:
    if not (os.getenv("VINO_ANN_ENABLE","0") == "1" and feedparser is not None):
        return 0.0, "", False
    try:
        sources = [s.strip() for s in (os.getenv("VINO_ANN_SOURCES","")).split(",") if s.strip()]
        if not sources: 
            return 0.0, "", False
        key = "|".join(sources)
        ts_now = time.time()
        cached = _ANN_CACHE.get(key)
        data: Dict[str, Tuple[float,str,bool]] = {}
        if cached and ts_now - cached[0] < ttl:
            data = cached[1]
        else:
            data = {}
            cutoff = now_utc() - timedelta(hours=12)
            for url in sources:
                try:
                    feed = feedparser.parse(url)
                    entries = feed.entries[:100]
                    for e in entries:
                        title = (getattr(e,"title","") or "").strip()
                        link  = (getattr(e,"link","") or "").strip()
                        summary = (getattr(e,"summary","") or "").strip()
                        txt = f"{title} {summary}".lower()
                        dt = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
                        if dt is not None:
                            ts = datetime.fromtimestamp(time.mktime(dt), tz=timezone.utc)
                            if ts < cutoff: 
                                continue
                        for cur_base in app.get("BASES", []):
                            aliases = _coin_aliases(app, cur_base)
                            if any(a.lower() in txt for a in aliases):
                                neg = any(k in txt for k in ["delist","suspension","maintenance","halt","outage"])
                                pos = any(k in txt for k in ["listing","launch","support","perp","margin"])
                                if pos or neg:
                                    b = 0.3 if pos else (-0.4 if neg else 0.0)
                                    prev = data.get(cur_base, (0.0,"",False))
                                    nb = prev[0] + b
                                    note = (prev[1] + " | " if prev[1] else "") + f"{title}"
                                    data[cur_base] = (nb, note, prev[2] or neg)
                except Exception:
                    continue
            _ANN_CACHE[key] = (ts_now, data)
        return data.get(base.upper(), (0.0,"",False))
    except Exception:
        return 0.0, "", False
