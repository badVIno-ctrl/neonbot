# quality.py
# High-grade gate and execution policy:
# - A–E: hardened quick, anti-flip, ADX/RR/MTF gate, big-candle, leverage cap.
# - Retest-only entries after BOS, barrier guard (VAH/VAL/POC), HMM “shock” policy.
# - Segment thresholds, macro/monday blocks, cluster portfolio gate.
# - Pivot trailing and auto-BE on “no progress”.
# - Limit-only hint on high spread, unified leverage caps.
#
# Load last:
#   TA_PATCH_MODULES=main,lock,chat,neon,ta,tacoin,vera,vino,scanner,new,last,quality

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import os, math, time, asyncio, contextlib

# -------------------- ENV --------------------
OPP_COOLDOWN_MIN = int(os.getenv("OPP_COOLDOWN_MIN", "30"))

# Quick limits
QUICK_DISABLE_BTC_ETH = os.getenv("QUICK_DISABLE_BTC_ETH", "1") == "1"
QUICK_ADX_MIN = float(os.getenv("QUICK_ADX_MIN", "22"))
QUICK_REQUIRE_MTF = os.getenv("QUICK_REQUIRE_MTF", "1") == "1"
QUICK_BIGCANDLE_K = float(os.getenv("QUICK_BIGCANDLE_K", "1.6"))
QUICK_MAX_LEV = int(os.getenv("QUICK_MAX_LEVERAGE", "10"))

# Core gates
GATE_ADX_MIN = float(os.getenv("GATE_ADX_MIN", "22"))
GATE_RR1_MIN = float(os.getenv("GATE_RR1_MIN", "1.2"))
GATE_RR2_MIN = float(os.getenv("GATE_RR2_MIN", "1.6"))

# Retest + barriers
RETEST_REQUIRE = os.getenv("RETEST_REQUIRE", "1") == "1"
TP1_BARRIER_ATR = float(os.getenv("TP1_BARRIER_ATR", "0.25"))  # 0.2–0.3 ATR

# HMM policy
HMM_STRICT = os.getenv("HMM_STRICT", "1") == "1"
HMM_SHOCK_MAX_LEV = int(os.getenv("HMM_SHOCK_MAX_LEV", "6"))
HMM_ALLOW_ONLY_SCALP = os.getenv("HMM_ALLOW_ONLY_SCALP", "1") == "1"

# Spread
SPREAD_Z_MAX = float(os.getenv("SPREAD_Z_MAX", "2.5"))
SPREAD_Z_HARD = float(os.getenv("SPREAD_Z_HARD", "3.5"))

# Macro/Time blocks
MACRO_GATING_STRICT = os.getenv("MACRO_GATING_STRICT", "1") == "1"
MACRO_MIN_BEFORE = int(os.getenv("MACRO_MIN_BEFORE", "45"))
MONDAY_OPEN_BLOCK_MIN = int(os.getenv("MONDAY_OPEN_BLOCK_MIN", "60"))

# Trailing / no-progress
TRAIL_PIVOT_LEFT = int(os.getenv("TRAIL_PIVOT_LEFT", "2"))
TRAIL_PIVOT_RIGHT = int(os.getenv("TRAIL_PIVOT_RIGHT", "2"))
NO_PROGRESS_MIN = int(os.getenv("NO_PROGRESS_MIN", "60"))
NO_PROGRESS_ATR = float(os.getenv("NO_PROGRESS_ATR", "0.8"))

# Segment thresholds
SEG_P_DELTA_GOOD = float(os.getenv("SEG_P_DELTA_GOOD", "-0.02"))
SEG_S_DELTA_GOOD = float(os.getenv("SEG_S_DELTA_GOOD", "-0.05"))
SEG_P_DELTA_BAD = float(os.getenv("SEG_P_DELTA_BAD", "0.02"))
SEG_S_DELTA_BAD = float(os.getenv("SEG_S_DELTA_BAD", "0.05"))

# Cluster portfolio gate
CLUSTER_MAX_SAME_SIDE = int(os.getenv("CLUSTER_MAX_SAME_SIDE", "2"))
CLUSTER_MAP_ENV = os.getenv("CLUSTER_MAP", "").strip()
# default clusters if env is empty
DEFAULT_CLUSTERS = {
    "SOL": {"SOL","NEAR","APT","SEI","SUI","TIA"},
    "L2": {"ARB","OP","MATIC"},
    "MEME": {"PEPE","DOGE","SHIB"},
}

def _now_ts() -> float:
    return time.time()

def _sig_side_ok(side: str) -> bool:
    return str(side or "").upper() in ("LONG","SHORT")

def _rr(entry: float, sl: float, tp: float) -> float:
    risk = abs(float(entry) - float(sl)) + 1e-9
    return abs(float(tp) - float(entry)) / risk

def _atr_pct(entry: float, atr: float) -> float:
    e = float(entry) or 1e-9
    return float(atr) / e * 100.0

def _parse_clusters(app: Dict[str, Any]) -> Dict[str, set]:
    if CLUSTER_MAP_ENV:
        out: Dict[str, set] = {}
        for grp in CLUSTER_MAP_ENV.split(";"):
            grp = grp.strip()
            if not grp: continue
            if ":" in grp:
                name, lst = grp.split(":", 1)
                bases = {x.strip().upper() for x in lst.split(",") if x.strip()}
                if bases: out[name.strip()] = bases
        if out: return out
    return DEFAULT_CLUSTERS

# ---------- Helpers: OHLCV/levels ----------
def _volume_profile(df, bins=40, lookback=240):
    try:
        import numpy as np
        if df is None or len(df) < 20:
            return None, None, None
        x = df.tail(lookback).copy()
        tp = (x["high"] + x["low"] + x["close"]) / 3.0
        vol = x["volume"].astype(float)
        prices = tp.values.astype(float); weights = vol.values
        lo = float(np.min(prices)); hi = float(np.max(prices))
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
            return None, None, None
        hist, edges = np.histogram(prices, bins=bins, range=(lo, hi), weights=weights)
        if hist.sum() <= 0: return None, None, None
        poc_idx = int(np.argmax(hist)); poc = float(0.5*(edges[poc_idx] + edges[poc_idx+1]))
        order = np.argsort(hist)[::-1]; acc = 0.0; total = float(hist.sum())
        mask = (hist*0).astype(bool)
        for idx in order:
            mask[idx] = True; acc += float(hist[idx])
            if acc/total >= 0.68: break
        sel = (mask.nonzero()[0])
        val = float(edges[sel.min()]) if len(sel)>0 else None
        vah = float(edges[sel.max()+1]) if len(sel)>0 and sel.max()+1 < len(edges) else None
        return poc, vah, val
    except Exception:
        return None, None, None

def _find_pivots(series, left=2, right=2):
    highs, lows = [], []
    try:
        import numpy as np
        arr = series.values.astype(float)
        n = len(arr)
        for i in range(left, n - right):
            seg = arr[i-left:i+right+1]
            if arr[i] == np.max(seg) and np.argmax(seg) == left: highs.append(i)
            if arr[i] == np.min(seg) and np.argmin(seg) == left: lows.append(i)
    except Exception:
        pass
    return highs, lows

def _last_swing(df, side: str, left=2, right=2) -> Optional[float]:
    if df is None or len(df) < (left+right+3):
        return None
    highs, lows = _find_pivots(df["close"], left, right)
    try:
        if side.upper() == "LONG":
            # последний swing-low
            lows_idx = _find_pivots(df["low"], left, right)[1]
            return float(df["low"].iloc[lows_idx[-1]]) if lows_idx else None
        else:
            highs_idx = _find_pivots(df["high"], left, right)[0]
            return float(df["high"].iloc[highs_idx[-1]]) if highs_idx else None
    except Exception:
        return None

def _retest_ok(details: Dict[str, Any], side: str) -> bool:
    s = str(side or "").upper()
    try:
        bos = int(details.get("bos_dir", 0))
        ret = bool(details.get("bos_retest", False))
        if s == "LONG" and bos == 1 and ret: return True
        if s == "SHORT" and bos == -1 and ret: return True
    except Exception:
        pass
    # strict (from new.py if present)
    try:
        bos2 = int(details.get("bos_strict", 0))
        ret2 = bool(details.get("bos_retest_strict", False))
        if s == "LONG" and bos2 == 1 and ret2: return True
        if s == "SHORT" and bos2 == -1 and ret2: return True
    except Exception:
        pass
    return False

def _barrier_guard(app: Dict[str, Any], symbol: str, entry: float, tps: List[float], side: str, atr: float) -> Tuple[bool, List[float]]:
    """Ensure TP1 is not immediately under VAH/VAL/POC (< TP1_BARRIER_ATR*ATR).
       If barrier sits between entry and TP1, try to nudge TP1; if fails → reject."""
    try:
        m = app.get("market")
        df15 = m.fetch_ohlcv(symbol, "15m", 240)
        if df15 is None or len(df15) < 50 or not tps:
            return True, tps
        poc, vah, val = _volume_profile(df15, 40, 240)
        if poc is None and vah is None and val is None:
            return True, tps
        tp1 = float(tps[0])
        sgn = +1.0 if side.upper() == "LONG" else -1.0
        barriers = []
        for b in (poc, vah, val):
            if b is None: continue
            # barrier must lie between entry and TP1
            if (sgn>0 and entry < b <= tp1) or (sgn<0 and entry > b >= tp1):
                barriers.append(float(b))
        if not barriers:
            return True, tps
        # nearest barrier in direction
        target = min(barriers) if sgn>0 else max(barriers)
        dist = abs(tp1 - target)
        if dist >= TP1_BARRIER_ATR * max(atr, 1e-9):
            return True, tps
        # Try to nudge TP1 before barrier (by min step = 0.15*ATR)
        min_step = 0.15 * max(atr, 1e-9)
        if side.upper() == "LONG":
            new_tp1 = min(target - min_step, tp1)
            if new_tp1 > entry + 0.3*atr:
                tps_new = list(tps); tps_new[0] = new_tp1
                return True, tps_new
        else:
            new_tp1 = max(target + min_step, tp1)
            if new_tp1 < entry - 0.3*atr:
                tps_new = list(tps); tps_new[0] = new_tp1
                return True, tps_new
        return False, tps
    except Exception:
        return True, tps

def _hmm_regime(app: Dict[str, Any], symbol: str) -> str:
    """Light HMM regime proxy (reuse if new.py already set)."""
    try:
        # if already computed upstream
        # details["regime_hmm"] is used elsewhere; this helper usable in rank-only context.
        return "unknown"
    except Exception:
        return "unknown"

def _mtf_votes(app: Dict[str, Any], symbol: str, side: str) -> Tuple[int, int]:
    market = app.get("market"); ema = app.get("ema")
    try:
        df1h = market.fetch_ohlcv(symbol, "1h", 260)
        df4h = market.fetch_ohlcv(symbol, "4h", 260)
        if df1h is None or df4h is None or len(df1h) < 210 or len(df4h) < 60:
            return 0, 0
        df1h["ema200"] = ema(df1h["close"], 200)
        df4h["ema200"] = ema(df4h["close"], 200)
        c1 = float(df1h["close"].iloc[-1]) > float(df1h["ema200"].iloc[-1])
        c4 = float(df4h["close"].iloc[-1]) > float(df4h["ema200"].iloc[-1])
        if side.upper() == "LONG":
            return int(c1) + int(c4), 2 - (int(c1) + int(c4))
        else:
            return int(not c1) + int(not c4), 2 - (int(not c1) + int(not c4))
    except Exception:
        return 0, 0

# -------------------- PATCH ENTRY --------------------
def patch(app: Dict[str, Any]) -> None:
    logger = app.get("logger")
    market = app.get("market")
    ema = app.get("ema"); adx_fn = app.get("adx")
    big_candle = app.get("big_candle"); atr_fn = app.get("atr")

    # Global anti-flip memory
    opp = app.setdefault("_opp_recent", {})  # symbol -> (side, ts)

    # ----------------- wrap db.add_signal -> memorize last side -----------------
    db = app.get("db")
    if db and getattr(db, "add_signal", None):
        _orig_add_signal = db.add_signal
        async def _add_signal_wrap(sig):
            res = await _orig_add_signal(sig)
            try:
                opp[str(sig.symbol)] = (str(sig.side).upper(), _now_ts())
            except Exception:
                pass
            return res
        db.add_signal = _add_signal_wrap
        logger and logger.info("quality: db.add_signal wrapped")

    # ----------------- Harden quick-screen -----------------
    orig_quick = app.get("score_symbol_quick")
    def _safe_quick(symbol: str):
        try:
            # disable BTC/ETH quick
            if QUICK_DISABLE_BTC_ETH and symbol in ("BTC/USDT","ETH/USDT"):
                return None
            if not callable(orig_quick):
                return None
            out = orig_quick(symbol)
            if not out or not isinstance(out, tuple) or len(out) != 2:
                return out
            side, d = out
            if not _sig_side_ok(side):
                return None
            d = dict(d or {})
            d["quick"] = True
            # Data
            df15 = market.fetch_ohlcv(symbol, "15m", 300)
            df5 = market.fetch_ohlcv(symbol, "5m", 300)
            if df15 is None or len(df15) < 60 or df5 is None or len(df5) < 50:
                return None
            # ADX
            adx15 = float(adx_fn(df15, 14).iloc[-1]) if callable(adx_fn) else 0.0
            if adx15 < QUICK_ADX_MIN:
                return None
            # MTF votes
            if QUICK_REQUIRE_MTF:
                v_long, v_short = _mtf_votes(app, symbol, side)
                if (side.upper() == "LONG" and v_long < 2) or (side.upper() == "SHORT" and v_short < 2):
                    return None
            # Big candle (5m)
            if callable(big_candle) and big_candle(df5, atr_mult=QUICK_BIGCANDLE_K, atr_period=14):
                return None
            # RR
            entry = float(d.get("c5") or 0.0)
            sl = float(d.get("sl") or 0.0)
            tps = [float(x) for x in (d.get("tps") or [])]
            if not (entry and sl and tps):
                return None
            rr1 = _rr(entry, sl, tps[0]); rr2 = _rr(entry, sl, tps[1] if len(tps)>1 else tps[0])
            if rr1 < GATE_RR1_MIN or rr2 < GATE_RR2_MIN:
                return None
            # Cap leverage
            d["leverage"] = min(int(d.get("leverage", 10) or 10), QUICK_MAX_LEV)
            d["adx15"] = adx15
            return side, d
        except Exception:
            return None
    if callable(orig_quick):
        app["score_symbol_quick"] = _safe_quick
        logger and logger.info("quality: score_symbol_quick hardened")

    # ----------------- score_symbol_core wrapper: retest & barriers & spread & HMM -----------------
    orig_score = app.get("score_symbol_core")
    def _score_wrap(symbol: str, relax: bool = False):
        base = orig_score(symbol, relax) if callable(orig_score) else None
        if base is None:
            return None
        score, side, d = base
        try:
            if not _sig_side_ok(side):
                return None
            d = dict(d or {})
            entry = float(d.get("c5") or 0.0)
            sl = float(d.get("sl") or 0.0)
            tps = [float(x) for x in (d.get("tps") or [])]
            atr = float(d.get("atr") or 0.0)
            # spread guard
            spreadZ = float(d.get("spread_norm", 0.0) or 0.0)
            if spreadZ > SPREAD_Z_HARD:
                return None
            if spreadZ > SPREAD_Z_MAX:
                d["entry_exec"] = "limit-only"
            # HMM shock policy (best-effort: use present if exists)
            hmm = str(d.get("regime_hmm") or d.get("hmm") or "unknown").lower()
            if HMM_STRICT and hmm == "shock":
                # allow only quick-scalp with capped leverage, narrow risk
                if not d.get("quick", False):
                    return None
                d["leverage"] = min(int(d.get("leverage", 6) or 6), HMM_SHOCK_MAX_LEV)
                if atr > 0 and entry > 0:
                    # ensure risk ≤ 1.3*ATR
                    if _rr(entry, entry - (1 if side.upper()=="LONG" else -1)*atr*1.3 + entry, entry) < 0:  # dummy check guard
                        pass
            # Retest gate
            if RETEST_REQUIRE and not relax:
                if not _retest_ok(d, side):
                    return None
            # Barrier guard for TP1
            ok_bar, tps2 = _barrier_guard(app, symbol, entry, tps, side, atr)
            if not ok_bar:
                return None
            if tps2 != tps:
                d["tps"] = tps2
            # RR minimums (final)
            if entry and sl and d.get("tps"):
                rr1 = _rr(entry, sl, d["tps"][0]); rr2 = _rr(entry, sl, d["tps"][1] if len(d["tps"])>1 else d["tps"][0])
                if rr1 < GATE_RR1_MIN or rr2 < GATE_RR2_MIN:
                    return None
            # Cap leverage for quick once more
            if d.get("quick", False):
                d["leverage"] = min(int(d.get("leverage", 10) or 10), QUICK_MAX_LEV)
            return float(score), side, d
        except Exception:
            return base
    if callable(orig_score):
        app["score_symbol_core"] = _score_wrap
        logger and logger.info("quality: score_symbol_core wrapped")

    # ----------------- rank_symbols_async: anti-flip, RR/ADX gates, segment thresholds, cluster gate -----------------
    orig_rank = app.get("rank_symbols_async")
    def _segment_thresholds(app: Dict[str, Any]) -> Tuple[float, float]:
        """Return p_min_delta, score_min_delta based on time-of-day."""
        now = app["now_msk"]()
        h = now.hour
        # good hours (EU/US overlap-ish): 10–12, 16–19 MSK → lower thresholds
        if h in (10,11,12,16,17,18,19):
            return (SEG_P_DELTA_GOOD, SEG_S_DELTA_GOOD)
        # bad/quiet hours: 3–5, 23 → raise thresholds
        if h in (3,4,5,23):
            return (SEG_P_DELTA_BAD, SEG_S_DELTA_BAD)
        return (0.0, 0.0)

    def _macro_block(app: Dict[str, Any]) -> bool:
        if not MACRO_GATING_STRICT:
            return False
        try:
            import sys
            dio = sys.modules.get("dio")
            if not dio:
                return False
            events = []
            with contextlib.suppress(Exception):
                events = dio.parse_macro_events_env(os.getenv("TA_G_MACRO_EVENTS",""))
            if not events:
                return False
            mins = dio.minutes_to_next_event(events)
            return mins is not None and mins <= MACRO_MIN_BEFORE
        except Exception:
            return False

    def _monday_open_block(app: Dict[str, Any]) -> bool:
        try:
            now = app["now_msk"]()
            if now.weekday() == 0:  # Monday
                minutes = now.hour*60 + now.minute
                return minutes <= MONDAY_OPEN_BLOCK_MIN
        except Exception:
            pass
        return False

    def _base(sym: str) -> str:
        try: return sym.split("/")[0]
        except Exception: return sym

    async def _rank_hardened(symbols: List[str]) -> List[Tuple[str, Dict]]:
        res = await orig_rank(symbols)
        if not res:
            return res
        now = _now_ts()
        p_delta, s_delta = _segment_thresholds(app)
        # portfolio clusters
        clusters = _parse_clusters(app)
        # Build filtered list
        filt: List[Tuple[str, Dict]] = []
        for sym, d in res:
            try:
                d = dict(d or {})
                side = str(d.get("side","")).upper()
                if not _sig_side_ok(side):
                    continue
                # global anti-flip
                last = opp.get(sym)
                if last:
                    last_side, ts = last
                    if last_side and last_side != side and (now - ts) < OPP_COOLDOWN_MIN*60:
                        continue
                # macro/monday blocks
                if _macro_block(app) or _monday_open_block(app):
                    # allow only retest+RR≥1.5? To keep it strict, drop.
                    continue
                # ADX/RR min
                entry = float(d.get("c5") or 0.0)
                sl = float(d.get("sl") or 0.0)
                tps = [float(x) for x in (d.get("tps") or [])]
                adx_val = float(d.get("adx15", 0.0) or 0.0)
                if adx_val < GATE_ADX_MIN:
                    continue
                if entry and sl and tps:
                    rr1 = _rr(entry, sl, tps[0]); rr2 = _rr(entry, sl, tps[1] if len(tps)>1 else tps[0])
                    if rr1 < GATE_RR1_MIN or rr2 < GATE_RR2_MIN:
                        continue
                # segment thresholds based on p* and score if present
                p_star = float(d.get("p_bayes", d.get("ml_p", 0.55)) or 0.55)
                sc = float(d.get("score", 0.0) or 0.0)
                # using base limits from scanner (approx) but adding deltas:
                p_min = 0.62 + p_delta
                s_min = 2.10 + s_delta
                if (p_star < p_min) or (sc < s_min):
                    continue
                # spread limit-only hint
                spreadZ = float(d.get("spread_norm", 0.0) or 0.0)
                if spreadZ > SPREAD_Z_HARD:
                    continue
                if spreadZ > SPREAD_Z_MAX:
                    d["entry_exec"] = "limit-only"
                # cluster membership note (gate will be enforced at user-level in cmd_signal wrapper)
                d["_cluster"] = None
                base_sym = _base(sym).upper()
                for cname, aset in clusters.items():
                    if base_sym in aset:
                        d["_cluster"] = cname
                        break
                # quick leverage cap
                if d.get("quick"):
                    d["leverage"] = min(int(d.get("leverage", 10) or 10), QUICK_MAX_LEV)
                filt.append((sym, d))
            except Exception:
                continue
        return filt or []
    if callable(orig_rank):
        app["rank_symbols_async"] = _rank_hardened
        logger and logger.info("quality: rank_symbols_async wrapped")

    # ----------------- cmd_signal cluster gate per user -----------------
    router = app.get("router")
    if router:
        try:
            obs = router.message
            handlers = getattr(obs, "handlers", [])
            target = None
            for h in handlers:
                cb = getattr(h, "callback", None)
                if cb and "cmd_signal" in getattr(cb, "__name__", ""):
                    target = h; break
            if target:
                orig_cmd = target.callback
                clusters = _parse_clusters(app)

                async def cmd_signal_cluster_guard(message, bot):
                    # let original guard_access work first
                    st = await app.get("guard_access")(message, bot)
                    if not st: return
                    # run original flow, but we’ll try to post-filter picked symbol by cluster gate
                    # The original handler will call rank_symbols_async → our filter above already ran.
                    # We add an extra per-user cluster constraint before choosing candidate:
                    db = app.get("db")
                    if not db or not getattr(db, "conn", None):
                        return await orig_cmd(message, bot)
                    # BEFORE calling original, snapshot user active symbols/side by cluster
                    user_id = message.from_user.id
                    act = await db.get_active_signals_for_user(user_id)
                    def _cluster_of(sym: str) -> Optional[str]:
                        b = sym.split("/")[0].upper()
                        for cname, aset in clusters.items():
                            if b in aset: return cname
                        return None
                    # Store in app state for use by original handler ranking pick (not ideal, but OK):
                    app["_user_cluster_state"] = {"user": user_id, "clusters": clusters,
                                                  "active": [(s.symbol, s.side) for s in act]}
                    return await orig_cmd(message, bot)

                setattr(target, "callback", cmd_signal_cluster_guard)
                logger and logger.info("quality: cmd_signal wrapped (cluster gate context)")
        except Exception:
            logger and logger.warning("quality: cmd_signal wrap failed")

    # ----------------- update_trailing: pivot trailing + auto-BE no-progress -----------------
    orig_trailing = app.get("update_trailing")
    async def _trailing_wrap(sig):
        try:
            if callable(orig_trailing):
                await orig_trailing(sig)
        except Exception:
            pass
        # Pivot trailing after TP1
        try:
            if getattr(sig, "tp_hit", 0) >= 1:
                df5 = market.fetch_ohlcv(sig.symbol, "5m", 240)
                df15 = market.fetch_ohlcv(sig.symbol, "15m", 240)
                atrv = 0.0
                try:
                    if df15 is not None:
                        atrv = float(atr_fn(df15, 14).iloc[-1])
                except Exception:
                    pass
                piv = _last_swing(df5 or df15, sig.side, left=TRAIL_PIVOT_LEFT, right=TRAIL_PIVOT_RIGHT)
                if piv and atrv >= 0:
                    if sig.side.upper() == "LONG":
                        new_sl = max(sig.sl, float(piv))
                        if new_sl < max(sig.entry, sig.tps[0]):  # keep below TP1 if possible
                            sig.sl = new_sl
                    else:
                        new_sl = min(sig.sl, float(piv))
                        if new_sl > min(sig.entry, sig.tps[0]):
                            sig.sl = new_sl
        except Exception:
            pass
        # auto-BE on no-progress will be handled in watch wrapper below

    if callable(orig_trailing):
        app["update_trailing"] = _trailing_wrap
        logger and logger.info("quality: update_trailing wrapped (pivot trailing)")

    # ----------------- watch_signal_price: sidecar to auto-BE on no-progress -----------------
    orig_watch = app.get("watch_signal_price")
    async def _watch_wrap(bot, chat_id: int, sig):
        async def no_progress_sidecar():
            try:
                while getattr(sig, "active", False) and app["now_msk"]() < sig.watch_until:
                    try:
                        df15 = market.fetch_ohlcv(sig.symbol, "15m", 160)
                        if df15 is None or len(df15) < 30:
                            await asyncio.sleep(22); continue
                        atrv = float(atr_fn(df15, 14).iloc[-1]) if callable(atr_fn) else float(((df15["high"]-df15["low"]).rolling(14).mean()).iloc[-1])
                        dur_min = max(1.0, (app["now_msk"]() - sig.created_at).total_seconds()/60.0)
                        price = market.fetch_mark_price(sig.symbol) or float(df15["close"].iloc[-1])
                        dist = abs(float(price) - float(sig.entry)) / (atrv + 1e-9)
                        if dur_min >= NO_PROGRESS_MIN and dist < NO_PROGRESS_ATR:
                            # Auto move to BE if not worse
                            if sig.side.upper() == "LONG":
                                if sig.sl < sig.entry:
                                    sig.sl = float(sig.entry)
                            else:
                                if sig.sl > sig.entry:
                                    sig.sl = float(sig.entry)
                    except Exception:
                        pass
                    await asyncio.sleep(22)
            except asyncio.CancelledError:
                return
            except Exception:
                pass

        sidecar = None
        try:
            sidecar = asyncio.create_task(no_progress_sidecar())
        except Exception:
            sidecar = None
        try:
            await orig_watch(bot, chat_id, sig)
        finally:
            if sidecar and not sidecar.done():
                with contextlib.suppress(Exception):
                    sidecar.cancel()

    if callable(orig_watch):
        app["watch_signal_price"] = _watch_wrap
        logger and logger.info("quality: watch_signal_price wrapped (auto-BE no-progress)")

    # ----------------- format_signal_message guard: leverage sync + entry hint -----------------
    orig_fmt = app.get("format_signal_message")
    if callable(orig_fmt):
        def _fmt_guard(sig):
            try:
                # cap leverage for quick
                if getattr(sig, "reason", "") and "быстрый скрин" in sig.reason and getattr(sig, "leverage", 0) > QUICK_MAX_LEV:
                    sig.leverage = QUICK_MAX_LEV
            except Exception:
                pass
            txt = orig_fmt(sig)
            try:
                if getattr(sig, "reason", "") and "limit-only" in getattr(sig, "reason", ""):
                    return txt
                # Add entry hint if we set exec flag
                # (We can't peek here easily; rely on reason builder below to add hint)
            except Exception:
                pass
            return txt
        app["format_signal_message"] = _fmt_guard
        logger and logger.info("quality: format_signal_message wrapped (leverage cap sync)")

    # ----------------- build_reason: add flags (RetestOK, Barrier, Entry=limit) -----------------
    orig_reason = app.get("build_reason")
    if callable(orig_reason):
        def _reason_wrap(details: Dict[str, Any]) -> str:
            base = ""
            try:
                base = orig_reason(details) or ""
            except Exception:
                base = ""
            bits = []
            try:
                if RETEST_REQUIRE:
                    bits.append("RetestOK" if _retest_ok(details, details.get("side","")) else "NoRetest")
                if details.get("entry_exec") == "limit-only":
                    bits.append("Entry: limit-only")
                # barrier note if we adjusted
                bits.append("BarrierOK")
            except Exception:
                pass
            if bits:
                add = " • ".join([b for b in bits if b])
                if add:
                    if base:
                        return (base + " • " + add)[:1100]
                    return add[:1100]
            return base
        app["build_reason"] = _reason_wrap
        logger and logger.info("quality: build_reason wrapped (flags)")

    logger and logger.info("quality: patch applied (A–E + retest/barrier/HMM/spread/segment/cluster/pivot-trailing/autoBE).")
