# scanner.py
# Always-on scanner + hourly digest + channel immediate posts (cap 2/day) + snapshots + simple calibrator
# + inline p* trainer + /backup (admin-only) + /restore (admin-only) + auto-backup to admins
# + round-robin scanning + portfolio gate + session threshold adj + adaptive TP-ladder per-asset
# + model cache (no event loop in thread) + TP/SL sanitize + anti-opposite flip + /bd export+restore
# + support bridge (“Поддержка” / “🛟 Поддержка”)

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import os, asyncio, contextlib, json, time, math, heapq, tempfile
from datetime import datetime, timedelta, timezone

from aiogram import F
from aiogram.filters import Command
from aiogram.types import FSInputFile, Message

# optional helpers (корреляции/макро, если есть)
try:
    import dio
except Exception:
    dio = None

# ------------ ENV ------------
SCAN_INTERVAL_SEC     = int(os.getenv("SCAN_INTERVAL_SEC", "45"))
DIGEST_INTERVAL_MIN   = int(os.getenv("DIGEST_INTERVAL_MIN", "60"))
TOPK_STORE            = int(os.getenv("SCAN_TOPK_STORE", "50"))
TOPK_DIGEST           = int(os.getenv("SCAN_TOPK_DIGEST", "5"))
IMMEDIATE_P_MIN       = float(os.getenv("SCAN_IMMEDIATE_P_MIN", "0.62"))
IMMEDIATE_SCORE_MIN   = float(os.getenv("SCAN_IMMEDIATE_SCORE_MIN", "2.10"))
CHANNEL_DAILY_CAP     = int(os.getenv("CHANNEL_DAILY_CAP", "2"))
CHANNEL_USERNAME      = os.getenv("CHANNEL_USERNAME", "@NeonFakTrading").strip()

# round-robin
SCAN_ROUND_ROBIN      = os.getenv("SCAN_ROUND_ROBIN","1") == "1"
SCAN_BATCH            = int(os.getenv("SCAN_BATCH","6"))

# session adj
SCAN_SESSION_ADJ      = os.getenv("SCAN_SESSION_ADJ","1") == "1"

# portfolio gate
PORTF_BLOCK_MINUTES   = int(os.getenv("PORTF_BLOCK_MINUTES","120"))
PORTF_MAX_SAME_SIDE   = int(os.getenv("PORTF_MAX_SAME_SIDE","2"))

# калибратор порогов
CALIB_P_BOUNDS        = (0.55, 0.80)
CALIB_SCORE_BOUNDS    = (1.80, 2.60)
CALIB_LOOKBACK_HOURS  = int(os.getenv("CALIB_LOOKBACK_HOURS", "48"))

# встроенный p* тренинг
ML_RETRAIN_SEC        = int(os.getenv("SCAN_ML_RETRAIN_SEC", "21600"))
ML_MIN_ROWS           = int(os.getenv("SCAN_ML_MIN_ROWS", "300"))
ML_BLEND_W            = float(os.getenv("SCAN_ML_BLEND_W", "0.5"))
ML_RIDGE_L2           = float(os.getenv("SCAN_ML_RIDGE_L2", "0.1"))

# авто-бэкап DB всем админам
SCAN_AUTO_BACKUP_SEC  = int(os.getenv("SCAN_AUTO_BACKUP_SEC", "21600"))

# adaptive ladder per-asset
LADDER_ENABLE         = os.getenv("LADDER_ENABLE","1") == "1"
LADDER_REFRESH_SEC    = int(os.getenv("LADDER_REFRESH_SEC","21600"))

# --- Model cache for ML (avoid event loop in thread) ---
SCAN_MODEL_TTL_SEC    = int(os.getenv("SCAN_MODEL_TTL_SEC", "600"))

# --- Opposite-post cooldown for channel immediate posts (minutes) ---
SCAN_OPP_POST_MIN     = int(os.getenv("SCAN_OPP_POST_MIN", "7"))

# фичи для ML
NUM_KEYS = [k.strip() for k in (os.getenv("SCAN_ML_NUM_KEYS","score,p_bayes,adx15,rr1,rr2,news_boost,vol_z,r2_1h,bbwp,ta_adx15_alt,ta_pre_comp_rank,ta_breakout_disp,ta_mr_risk,ta_near_depth_ratio,spread_norm,book_imb,spoof_score,churn,vpin,ofi,breadth_pct50_1h,basis,ob_imb,cvd_slope,RS_btc,RS_eth").split(","))]
CAT_KEYS = [k.strip() for k in (os.getenv("SCAN_ML_CAT_KEYS","btc_dom_trend,regime_hmm,side").split(","))]

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

# ------------ DB ------------
async def _ensure_scan_tables(app: Dict[str, Any]) -> None:
    db = app.get("db")
    if not db or not getattr(db, "conn", None):
        return
    stmts = [
        """
        CREATE TABLE IF NOT EXISTS scan_feature_snapshot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER,
            created_at TEXT NOT NULL,
            symbol TEXT,
            side TEXT,
            score REAL,
            p_bayes REAL,
            adx15 REAL,
            rr1 REAL,
            rr2 REAL,
            news_boost REAL,
            payload TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS scan_channel_cap (
            date TEXT PRIMARY KEY,
            count INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS scan_posts (
            date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            PRIMARY KEY(date, symbol, side)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS scan_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            symbol TEXT,
            side TEXT,
            score REAL,
            p_bayes REAL,
            rank REAL,
            payload TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS scan_model (
            id INTEGER PRIMARY KEY CHECK (id=1),
            updated_at TEXT,
            kind TEXT,
            weights TEXT,
            mean TEXT,
            std TEXT,
            features TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS scan_ladder (
            symbol TEXT PRIMARY KEY,
            m1 REAL,
            m2 REAL,
            m3 REAL,
            updated_at TEXT
        )
        """,
    ]
    for s in stmts:
        with contextlib.suppress(Exception):
            await db.conn.execute(s)
    with contextlib.suppress(Exception):
        await db.conn.commit()

async def _get_channel_cap(app: Dict[str, Any]) -> int:
    db = app.get("db")
    if not db or not getattr(db, "conn", None):
        return 0
    dkey = app["now_msk"]().date().isoformat()
    try:
        cur = await db.conn.execute("SELECT count FROM scan_channel_cap WHERE date=?", (dkey,))
        row = await cur.fetchone()
        return int(row["count"]) if row else 0
    except Exception:
        return 0

async def _inc_channel_cap(app: Dict[str, Any], symbol: str, side: str) -> None:
    db = app.get("db"); dkey = app["now_msk"]().date().isoformat()
    try:
        with contextlib.suppress(Exception):
            await db.conn.execute(
                "INSERT OR IGNORE INTO scan_posts(date, symbol, side) VALUES (?, ?, ?)",
                (dkey, symbol, side)
            )
        cur = await db.conn.execute("SELECT COUNT(1) AS cnt FROM scan_posts WHERE date=?", (dkey,))
        row = await cur.fetchone()
        cnt = int(row["cnt"]) if row else 0
        await db.conn.execute(
            "INSERT INTO scan_channel_cap(date, count) VALUES (?, ?) ON CONFLICT(date) DO UPDATE SET count=excluded.count",
            (dkey, cnt)
        )
        await db.conn.commit()
    except Exception:
        pass

async def _save_candidate(app: Dict[str, Any], cand: Dict[str, Any]) -> None:
    db = app.get("db")
    if not db or not getattr(db, "conn", None):
        return
    try:
        payload = json.dumps(cand.get("details", {}), ensure_ascii=False)
        await db.conn.execute(
            "INSERT INTO scan_candidates(ts, symbol, side, score, p_bayes, rank, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (app["now_msk"]().isoformat(), cand["symbol"], cand["side"], cand["score"], cand["p_bayes"], cand["rank"], payload)
        )
        await db.conn.commit()
    except Exception:
        pass

async def _save_snapshot_for_signal(app: Dict[str, Any], signal_id: int, symbol: str, side: str, details: Dict[str, Any]) -> None:
    db = app.get("db")
    if not db or not getattr(db, "conn", None):
        return
    try:
        rr1 = rr2 = None
        try:
            entry = float(details.get("c5") or 0.0)
            sl = float(details.get("sl") or 0.0)
            tps = [float(x) for x in (details.get("tps") or [])]
            if entry > 0 and sl > 0 and tps:
                risk = abs(entry - sl) + 1e-9
                rr1 = abs(tps[0] - entry) / risk if len(tps) > 0 else None
                rr2 = abs(tps[1] - entry) / risk if len(tps) > 1 else None
        except Exception:
            pass
        payload = json.dumps(details, ensure_ascii=False)
        await db.conn.execute(
            """INSERT INTO scan_feature_snapshot
               (signal_id, created_at, symbol, side, score, p_bayes, adx15, rr1, rr2, news_boost, payload)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                int(signal_id),
                app["now_msk"]().isoformat(),
                symbol, side,
                float(details.get("score", 0.0) or 0.0),
                float(details.get("p_bayes", details.get("ml_p", 0.55) or 0.0) or 0.0),
                float(details.get("adx15", 0.0) or 0.0),
                rr1 if rr1 is not None else None,
                rr2 if rr2 is not None else None,
                float(details.get("news_boost", details.get("vino_news_boost", 0.0)) or 0.0),
                payload
            )
        )
        await db.conn.commit()
    except Exception:
        pass

# ------------ SCAN STATE ------------
def _state(app: Dict[str, Any]) -> Dict[str, Any]:
    st = app.setdefault("_scan", {
        "cfg": {"p_min": IMMEDIATE_P_MIN, "score_min": IMMEDIATE_SCORE_MIN},
        "top": [],
        "last_digest_at": 0.0,
        "rr_idx": 0,            # round-robin индекс
        "published": [],        # [(ts, symbol, side)]
        "last_post": {},        # symbol -> (side, ts)
    })
    return st

def _rank_of(score: float, p_bayes: float, details: Dict[str, Any]) -> float:
    r = float(0.6 * score + 0.4 * p_bayes)
    try:
        entry = float(details.get("c5") or 0.0); sl = float(details.get("sl") or 0.0)
        tps = [float(x) for x in (details.get("tps") or [])]
        rr1 = (abs(tps[0]-entry)/max(abs(entry-sl),1e-9)) if (entry and sl and tps) else 0.0
        if rr1 >= 1.5: r += 0.05
        if rr1 < 1.0:  r -= 0.05
    except Exception:
        pass
    try:
        adx = float(details.get("adx15", 0.0) or 0.0)
        if adx >= 25: r += 0.03
        if adx < 18:  r -= 0.03
    except Exception:
        pass
    try:
        spr = float(details.get("spread_norm", 0.0) or 0.0)
        spoof = float(details.get("spoof_score", 0.0) or 0.0)
        near = float(details.get("ta_near_depth_ratio", details.get("near_depth_ratio", 0.0)) or 0.0)
        if spr > 3.0: r -= 0.05
        if spoof >= 1.0: r -= 0.05
        if near >= 0.6: r += 0.03
    except Exception:
        pass
    try:
        trend = details.get("btc_dom_trend_new") or details.get("btc_dom_trend") or ""
        side = details.get("side","")
        if trend == "up" and side == "LONG": r -= 0.03
        if trend == "up" and side == "SHORT": r += 0.02
    except Exception:
        pass
    return r

# ------------ SCORE ONE ------------
async def _score_one(app: Dict[str, Any], symbol: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    fn = app.get("score_symbol_core")
    if not fn:
        return None
    loop = asyncio.get_running_loop()
    try:
        res = await loop.run_in_executor(app.get("EXECUTOR"), fn, symbol, False)
        if res is None:
            res = await loop.run_in_executor(app.get("EXECUTOR"), fn, symbol, True)
        return res
    except Exception:
        return None

# ------------ SESSION ADJ ------------
def _session_adj(msk_now: datetime) -> Tuple[float, float]:
    if not SCAN_SESSION_ADJ:
        return 0.0, 0.0
    h = msk_now.hour
    if h in (10,11,16,17):
        return (-0.02, -0.05)  # p*, score
    if h in (3,4,5,23):
        return (+0.02, +0.05)
    return 0.0, 0.0

# ------------ PORTFOLIO GATE ------------
def _can_publish_portfolio(app: Dict[str, Any], side: str) -> bool:
    st = _state(app)
    ts_now = time.time()
    st["published"] = [(t,s,sd) for (t,s,sd) in st["published"] if ts_now - t <= PORTF_BLOCK_MINUTES*60]
    same = sum(1 for (_,_,sd) in st["published"] if sd == side)
    return same < PORTF_MAX_SAME_SIDE

def _append_published(app: Dict[str, Any], symbol: str, side: str):
    st = _state(app)
    st["published"].append((time.time(), symbol, side))

# ------------ ADAPTIVE LADDER ------------
async def _refresh_ladder_loop(app: Dict[str, Any]):
    if not LADDER_ENABLE:
        return
    await asyncio.sleep(15)
    db = app.get("db"); logger = app.get("logger")
    if not db or not getattr(db,"conn",None):
        return
    while True:
        try:
            with contextlib.suppress(Exception):
                cur = await db.conn.execute("SELECT symbol, AVG(rr1) AS r1, AVG(rr2) AS r2 FROM outcomes WHERE rr1 IS NOT NULL GROUP BY symbol")
                rows = await cur.fetchall()
                for r in rows or []:
                    sym = str(r["symbol"])
                    r1 = float(r["r1"] or 1.2)
                    r2 = float(r["r2"] or 1.6)
                    m1 = max(0.6, min(1.2, r1*0.8))
                    m2 = max(1.2, min(2.2, r2*0.9))
                    m3 = min(3.0, max(m2+0.6, 2.4))
                    await db.conn.execute(
                        "INSERT INTO scan_ladder(symbol,m1,m2,m3,updated_at) VALUES(?,?,?,?,?) "
                        "ON CONFLICT(symbol) DO UPDATE SET m1=excluded.m1, m2=excluded.m2, m3=excluded.m3, updated_at=excluded.updated_at",
                        (sym, m1, m2, m3, app["now_msk"]().isoformat())
                    )
                await db.conn.commit()
                logger and logger.info("Scanner: ladder refreshed (rows=%d)", len(rows or []))
            await asyncio.sleep(max(3600, LADDER_REFRESH_SEC))
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger and logger.warning("Scanner ladder refresh error: %s", e)
            await asyncio.sleep(60)

async def _get_ladder(app: Dict[str, Any], symbol: str) -> Optional[Tuple[float,float,float]]:
    if not LADDER_ENABLE:
        return None
    db = app.get("db")
    if not db or not getattr(db,"conn",None):
        return None
    try:
        cur = await db.conn.execute("SELECT m1,m2,m3 FROM scan_ladder WHERE symbol=?", (symbol,))
        row = await cur.fetchone()
        if not row:
            return None
        return float(row["m1"]), float(row["m2"]), float(row["m3"])
    except Exception:
        return None

async def _apply_adaptive_tps(app: Dict[str, Any], sig, details: Dict[str, Any]) -> None:
    if not LADDER_ENABLE:
        return
    try:
        ladder = await _get_ladder(app, sig.symbol)
        if not ladder:
            return
        m1,m2,m3 = ladder
        atr = float(details.get("atr", 0.0) or 0.0)
        entry = float(details.get("c5", 0.0) or 0.0)
        if atr <= 0 or entry <= 0:
            return
        market = app.get("market")
        tick = 0.0
        with contextlib.suppress(Exception):
            tick = float(market.get_tick_size(sig.symbol) or 0.0)
        if sig.side == "LONG":
            tps = [entry + m1*atr, entry + m2*atr, entry + m3*atr]
            if tick > 0: tps = [math.ceil(tp/tick)*tick for tp in tps]
        else:
            tps = [entry - m1*atr, entry - m2*atr, entry - m3*atr]
            if tick > 0: tps = [math.floor(tp/tick)*tick for tp in tps]
        if tps and tps[0] > 0:
            sig.tps = [float(x) for x in tps[:3]]
    except Exception:
        pass

# ------------ SANITIZE LEVELS ------------
def _sanitize_levels(app: Dict[str, Any], details: Dict[str, Any], symbol: str, side: str) -> None:
    try:
        entry = float(details.get("c5") or 0.0)
        atr   = float(details.get("atr") or 0.0)
        sl    = float(details.get("sl") or 0.0)
        tps   = [float(x) for x in (details.get("tps") or [])]
        if entry <= 0:
            return
        market = app.get("market")
        df15 = market.fetch_ohlcv(symbol, "15m", 240)
        if atr <= 0 or df15 is None or len(df15) < 20:
            df1h = market.fetch_ohlcv(symbol, "1h", 360)
            df5  = market.fetch_ohlcv(symbol, "5m", 360)
            try:
                if df15 is not None and len(df15) >= 20:
                    atr = float((df15["high"] - df15["low"]).rolling(14).mean().iloc[-1])
                elif df1h is not None and len(df1h) >= 50:
                    atr = float((df1h["high"] - df1h["low"]).rolling(14).mean().iloc[-1]) / 4.0
                elif df5 is not None and len(df5) >= 100:
                    atr = float((df5["high"] - df5["low"]).rolling(14).mean().iloc[-1]) * 3.0
            except Exception:
                pass
            if not (isinstance(atr, (int,float)) and atr > 0):
                atr = max(1e-9, 0.001 * entry)
        try:
            tick = float(market.get_tick_size(symbol) or 0.0)
        except Exception:
            tick = 0.0
        def _rt(x, mode="round"):
            if tick and tick > 0:
                n = x / tick
                if mode == "floor": n = math.floor(n)
                elif mode == "ceil": n = math.ceil(n)
                else: n = round(n)
                return n * tick
            return x
        min_r = float(os.getenv("TA_SAN_SL_MIN_ATR", "0.25")) * atr
        max_r = float(os.getenv("TA_SAN_SL_MAX_ATR", "5.0")) * atr
        if side == "LONG":
            if not (sl < entry): sl = entry - min_r
            risk = abs(entry - sl)
            if risk < min_r: sl = entry - min_r
            if risk > max_r: sl = entry - max_r
            sl = _rt(sl, "floor")
        else:
            if not (sl > entry): sl = entry + min_r
            risk = abs(entry - sl)
            if risk < min_r: sl = entry + min_r
            if risk > max_r: sl = entry + max_r
            sl = _rt(sl, "ceil")
        bad = (
            not tps or
            len({round(x, 10) for x in tps}) < 3 or
            any(x <= 0 for x in tps) or
            (max(abs(x - entry) for x in tps) / (atr + 1e-9) > 8.0)
        )
        m1, m2, m3 = 0.8, 1.5, 2.4
        if bad:
            tps = [entry + m1*atr, entry + m2*atr, entry + m3*atr] if side == "LONG" else \
                  [entry - m1*atr, entry - m2*atr, entry - m3*atr]
        step = float(os.getenv("TA_SAN_TP_MIN_STEP_ATR", "0.15")) * atr
        if side == "LONG":
            tps = sorted(tps)
            tps[0] = max(tps[0], entry + step)
            tps[1] = max(tps[1], tps[0] + step)
            tps[2] = max(tps[2], tps[1] + step)
            tps = [_rt(x, "ceil") for x in tps]
        else:
            tps = sorted(tps, reverse=True)
            tps[0] = min(tps[0], entry - step)
            tps[1] = min(tps[1], tps[0] - step)
            tps[2] = min(tps[2], tps[1] - step)
            tps = [_rt(x, "floor") for x in tps]
        if len({round(x, 10) for x in tps}) < 3:
            step_tick = max(tick, 1e-12) * 2.0
            if side == "LONG":
                tps = sorted(tps)
                for i in range(1, len(tps)):
                    if tps[i] <= tps[i-1]:
                        tps[i] = _rt(tps[i-1] + step_tick, "ceil")
            else:
                tps = sorted(tps, reverse=True)
                for i in range(1, len(tps)):
                    if tps[i] >= tps[i-1]:
                        tps[i] = _rt(tps[i-1] - step_tick, "floor")
        details["sl"]  = float(sl)
        details["tps"] = [float(x) for x in tps[:3]]
    except Exception:
        pass

# ------------ PUBLISH ------------
async def _publish_immediate(app: Dict[str, Any], cand: Dict[str, Any]) -> bool:
    bot = app.get("bot_instance")
    if not bot or not CHANNEL_USERNAME:
        return False
    try:
        if not _can_publish_portfolio(app, cand["side"]):
            return False

        # Anti‑opposite cooldown per symbol
        st = _state(app)
        last_map = st.setdefault("last_post", {})
        now_ts = time.time()
        prev = last_map.get(cand["symbol"])
        if prev:
            prev_side, prev_ts = prev
            if cand["side"] != prev_side and (now_ts - prev_ts) < SCAN_OPP_POST_MIN * 60:
                return False

        Signal = app.get("Signal"); build_reason = app.get("build_reason"); fmt = app.get("format_signal_message")
        if not (Signal and fmt): return False

        # Санитация уровней в деталях перед сборкой сигнала
        d = cand["details"]
        try:
            _sanitize_levels(app, d, cand["symbol"], cand["side"])
        except Exception:
            pass

        reason = ""
        with contextlib.suppress(Exception): reason = build_reason(d) or ""
        sig = Signal(
            user_id=0, symbol=cand["symbol"], side=cand["side"],
            entry=float(d["c5"]), tps=[float(x) for x in d["tps"]], sl=float(d["sl"]),
            leverage=int(d.get("leverage", 5)), risk_level=int(d.get("risk_level", 5)),
            created_at=app["now_msk"](), news_note=d.get("news_note", ""), atr_value=float(d.get("atr", 0.0)),
            reason=reason
        )
        await _apply_adaptive_tps(app, sig, d)

        text = fmt(sig) + "\n\n🤖 Опубликовано автоматически • NEON Bot"
        await bot.send_message(CHANNEL_USERNAME, text, disable_web_page_preview=True)
        await _inc_channel_cap(app, cand["symbol"], cand["side"])
        _append_published(app, cand["symbol"], cand["side"])
        last_map[cand["symbol"]] = (cand["side"], time.time())
        return True
    except Exception:
        return False

# ------------ SCAN LOOP ------------
async def _scan_loop(app: Dict[str, Any]):
    await _ensure_scan_tables(app)
    logger = app.get("logger"); syms = app.get("SYMBOLS", [])
    if not syms:
        logger and logger.warning("Scanner: SYMBOLS empty"); return
    logger and logger.info("Scanner: start, pairs=%d, interval=%ss", len(syms), SCAN_INTERVAL_SEC)
    while True:
        try:
            t0 = time.time()
            batch_syms = syms
            if SCAN_ROUND_ROBIN:
                st = _state(app)
                i = int(st.get("rr_idx", 0))
                if i >= len(syms): i = 0
                j = min(len(syms), i + max(1, SCAN_BATCH))
                batch_syms = syms[i:j]
                st["rr_idx"] = j % len(syms)
            tasks = [asyncio.create_task(_score_one(app, s)) for s in batch_syms]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            st = _state(app); top_heap = st["top"]
            for s, res in zip(batch_syms, results):
                if not (isinstance(res, tuple) and len(res) == 3): continue
                score, side, d = res
                p = float(d.get("p_bayes", d.get("ml_p", 0.55)) or 0.55)
                rank = _rank_of(float(score), p, d)
                cand = {"symbol": s, "side": side, "score": float(score), "p_bayes": p, "rank": rank, "details": d}
                with contextlib.suppress(Exception): await _save_candidate(app, cand)
                heapq.heappush(top_heap, (rank, cand))
                if len(top_heap) > TOPK_STORE: heapq.heappop(top_heap)
            p_adj, s_adj = _session_adj(app["now_msk"]())
            thresh_p = _state(app)["cfg"]["p_min"] + p_adj
            thresh_s = _state(app)["cfg"]["score_min"] + s_adj
            cap = await _get_channel_cap(app)
            if cap < CHANNEL_DAILY_CAP:
                best = sorted([c for _, c in top_heap], key=lambda x: x["rank"], reverse=True)
                for c in best:
                    if c["p_bayes"] >= thresh_p and c["score"] >= thresh_s:
                        ok = await _publish_immediate(app, c)
                        if ok:
                            cap += 1
                            if cap >= CHANNEL_DAILY_CAP: break
            dt = time.time() - t0
            await asyncio.sleep(max(1.0, SCAN_INTERVAL_SEC - dt))
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger and logger.warning("Scanner loop error: %s", e)
            await asyncio.sleep(5)

# ------------ DIGEST LOOP ------------
async def _digest_loop(app: Dict[str, Any]):
    logger = app.get("logger"); bot = app.get("bot_instance")
    if not bot: return
    while True:
        try:
            await asyncio.sleep(5)
            st = _state(app); last = float(st.get("last_digest_at", 0.0))
            if time.time() - last < DIGEST_INTERVAL_MIN * 60:
                await asyncio.sleep(10); continue
            st["last_digest_at"] = time.time()
            best = sorted([c for _, c in st["top"]], key=lambda x: x["rank"], reverse=True)[:TOPK_DIGEST]
            if not best: continue
            lines = ["📊 Топ‑сетки часа (NEON Bot)", "━━━━━━━━━━━━━━━━━━"]
            fmt_price = app.get("format_price") or (lambda v: f"{v:.4f}")
            for c in best:
                d = c["details"]; base = c["symbol"].split("/")[0]
                entry = float(d.get("c5", 0.0) or 0.0); sl = float(d.get("sl", 0.0) or 0.0)
                tps = [float(x) for x in (d.get("tps") or [])]
                rr1 = (abs(tps[0] - entry) / (abs(entry - sl) + 1e-9)) if tps and entry and sl else 0.0
                qline = d.get("quality_line", "")
                lines.append(f"• {base} {c['side']} • score {c['score']:.2f} • p* {c['p_bayes']:.2f} • RR1 {rr1:.2f} • TBX {fmt_price(entry)} • SL {fmt_price(sl)}" + (f"\n  {qline}" if qline else ""))
            lines.append("\n🤖 Авто‑дайджест • NEON Bot")
            with contextlib.suppress(Exception):
                await bot.send_message(CHANNEL_USERNAME, "\n".join(lines), disable_web_page_preview=True)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger and logger.warning("Digest loop error: %s", e)
            await asyncio.sleep(10)

# ------------ CALIBRATOR (thresholds) ------------
async def _calibration_loop(app: Dict[str, Any]):
    await _ensure_scan_tables(app)
    logger = app.get("logger"); db = app.get("db")
    if not db or not getattr(db, "conn", None): return
    while True:
        try:
            since = (app["now_msk"]() - timedelta(hours=CALIB_LOOKBACK_HOURS)).isoformat()
            cur = await db.conn.execute("""
                SELECT o.outcome, s.p_bayes, s.score
                FROM outcomes o
                JOIN scan_feature_snapshot s ON s.signal_id = o.signal_id
                WHERE o.finished_at IS NOT NULL AND o.finished_at >= ?
            """, (since,))
            rows = await cur.fetchall()
            wins = losses = 0
            for r in rows or []:
                out = str(r["outcome"]).upper()
                if out in ("TP1","TP2","TP3","BE"): wins += 1
                elif out in ("STOP","TIME"): losses += 1
            cfg = _state(app)["cfg"]; total = max(1, wins + losses)
            stop_rate = losses / total
            d_p = (0.02 if stop_rate > 0.50 else (-0.01 if wins/total > 0.60 else 0.0))
            d_s = (0.10 if stop_rate > 0.50 else (-0.05 if wins/total > 0.60 else 0.0))
            cfg["p_min"] = float(min(CALIB_P_BOUNDS[1], max(CALIB_P_BOUNDS[0], cfg["p_min"] + d_p)))
            cfg["score_min"] = float(min(CALIB_SCORE_BOUNDS[1], max(CALIB_SCORE_BOUNDS[0], cfg["score_min"] + d_s)))
            logger and logger.info("Scanner calibrator: wins=%d losses=%d stop_rate=%.2f -> p_min=%.2f, score_min=%.2f",
                                   wins, losses, stop_rate, cfg["p_min"], cfg["score_min"])
            await asyncio.sleep(3 * 3600)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger and logger.warning("Calibrator loop error: %s", e)
            await asyncio.sleep(60)

# ------------ ML PART ------------
def _cat_to_num(val: Any, key: str) -> float:
    v = str(val or "").lower()
    if key == "btc_dom_trend": return {"up":1.0, "down":-1.0, "flat":0.0}.get(v, 0.0)
    if key == "regime_hmm": return {"trend":0.5, "range":0.0, "calm":0.2, "shock":-0.5}.get(v, 0.0)
    if key == "side": return 1.0 if v == "long" else (-1.0 if v == "short" else 0.0)
    return 0.0

def _extract_features(payload_json: str) -> Dict[str, float]:
    try: d = json.loads(payload_json or "{}")
    except Exception: d = {}
    out: Dict[str, float] = {}
    for k in NUM_KEYS:
        try: out[k] = float(d.get(k)) if d.get(k) is not None else 0.0
        except Exception: out[k] = 0.0
    for k in CAT_KEYS: out[k] = _cat_to_num(d.get(k), k)
    return out

async def _load_dataset(app: Dict[str, Any]) -> Tuple[List[List[float]], List[int]]:
    db = app.get("db"); X: List[List[float]] = []; y: List[int] = []
    if not db or not getattr(db, "conn", None): return X, y
    try:
        cur = await db.conn.execute("""
            SELECT o.outcome, s.payload
            FROM outcomes o
            JOIN scan_feature_snapshot s ON s.signal_id = o.signal_id
            WHERE o.finished_at IS NOT NULL
        """)
        rows = await cur.fetchall(); feats_order = NUM_KEYS + CAT_KEYS
        for r in rows or []:
            out = str(r["outcome"]).upper(); label = 1 if out in ("TP1","TP2","TP3","BE") else 0
            fd = _extract_features(r["payload"]); X.append([float(fd.get(k, 0.0)) for k in feats_order]); y.append(label)
    except Exception:
        pass
    return X, y

async def _save_model(app: Dict[str, Any], model: Dict[str, Any]) -> None:
    db = app.get("db"); 
    if not db or not getattr(db, "conn", None): return
    try:
        await db.conn.execute(
            "INSERT INTO scan_model(id, updated_at, kind, weights, mean, std, features) VALUES (1, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at, kind=excluded.kind, weights=excluded.weights, mean=excluded.mean, std=excluded.std, features=excluded.features",
            (app["now_msk"]().isoformat(), model.get("kind","linlogit"), json.dumps(model.get("w", [])), json.dumps(model.get("mean", [])),
             json.dumps(model.get("std", [])), json.dumps(model.get("features", [])))
        )
        await db.conn.commit()
    except Exception:
        pass

async def _load_model_db(app: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    db = app.get("db"); 
    if not db or not getattr(db, "conn", None): return None
    try:
        cur = await db.conn.execute("SELECT kind, weights, mean, std, features FROM scan_model WHERE id=1")
        row = await cur.fetchone()
        if not row: return None
        return {"kind": str(row["kind"]), "w": json.loads(row["weights"] or "[]"), "mean": json.loads(row["mean"] or "[]"),
                "std": json.loads(row["std"] or "[]"), "features": json.loads(row["features"] or "[]")}
    except Exception:
        return None

def _standardize(X: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    if not X: return [], [], []
    m = len(X); n = len(X[0]); mean = [0.0]*n; std = [0.0]*n
    for j in range(n):
        s = 0.0
        for i in range(m): s += X[i][j]
        mean[j] = s / max(1, m)
    for j in range(n):
        s = 0.0
        for i in range(m):
            d = X[i][j] - mean[j]; s += d*d
        std[j] = math.sqrt(s / max(1, m)) + 1e-9
    Xs = []
    for i in range(m):
        row = []
        for j in range(n):
            row.append((X[i][j] - mean[j]) / std[j])
        Xs.append(row)
    return Xs, mean, std

def _fit_linlogit(Xs: List[List[float]], y: List[int], l2: float = 0.1) -> List[float]:
    if not Xs: return []
    m = len(Xs); n = len(Xs[0])
    XT_X = [[0.0]*(n+1) for _ in range(n+1)]; XT_y = [0.0]*(n+1)
    for i in range(m):
        xi = Xs[i] + [1.0]; yi = float(y[i])
        for j in range(n+1): XT_y[j] += xi[j]*yi
        for j in range(n+1):
            for k in range(n+1):
                XT_X[j][k] += xi[j]*xi[k]
    for j in range(n): XT_X[j][j] += l2
    A = [row[:] + [XT_y[i]] for i, row in enumerate(XT_X)]; N = n+1
    for col in range(N):
        piv = col
        for r in range(col+1, N):
            if abs(A[r][col]) > abs(A[piv][col]):
                A[piv], A[col] = A[col], A[piv]
        if abs(A[col][col]) < 1e-12: continue
        div = A[col][col]
        for c in range(col, N+1): A[col][c] /= div
        for r in range(N):
            if r == col: continue
            fac = A[r][col]
            for c in range(col, N+1): A[r][c] -= fac * A[col][c]
    w = [A[i][N] if i < len(A) else 0.0 for i in range(N)]
    return w

def _sigmoid(z: float) -> float:
    try: return 1.0/(1.0+math.exp(-z))
    except Exception: return 0.5

def _predict_p(model: Dict[str, Any], row_raw: List[float]) -> float:
    try:
        mean = model.get("mean") or []; std = model.get("std") or []; w = model.get("w") or []
        if model.get("kind") in ("lgbm","sk") and not w:
            pass
        if not (mean and std and w): return 0.55
        n = len(mean); z = 0.0
        for j in range(n):
            xj = (row_raw[j] - mean[j]) / (std[j] if std[j] != 0.0 else 1.0)
            z += xj * w[j]
        z += w[-1]
        return float(min(0.99, max(0.01, _sigmoid(z))))
    except Exception:
        return 0.55

async def _train_loop(app: Dict[str, Any]):
    await _ensure_scan_tables(app)
    logger = app.get("logger")
    await asyncio.sleep(15)
    while True:
        try:
            X, y = await _load_dataset(app)
            if len(X) >= ML_MIN_ROWS:
                Xs, mean, std = _standardize(X)
                model_kind = "linlogit"; w: List[float] = []
                ok_saved = False
                try:
                    import lightgbm as lgb, numpy as np
                    ds = lgb.Dataset(np.array(X), label=np.array(y))
                    params = dict(objective="binary", metric="auc", verbosity=-1, learning_rate=0.05, num_leaves=31, max_depth=-1)
                    mdl = lgb.train(params, ds, num_boost_round=300)
                    await _save_model(app, {"kind":"lgbm","w":[],"mean":mean,"std":std,"features": (NUM_KEYS+CAT_KEYS)})
                    ok_saved = True; model_kind="lgbm"
                except Exception:
                    try:
                        from sklearn.linear_model import LogisticRegression
                        import numpy as np
                        lr = LogisticRegression(max_iter=200); lr.fit(np.array(X), np.array(y))
                        await _save_model(app, {"kind":"sk","w":[],"mean":mean,"std":std,"features": (NUM_KEYS+CAT_KEYS)})
                        ok_saved = True; model_kind="sk"
                    except Exception:
                        pass
                if not ok_saved:
                    w = _fit_linlogit(Xs, y, l2=ML_RIDGE_L2)
                    await _save_model(app, {"kind":"linlogit","w":w,"mean":mean,"std":std,"features": (NUM_KEYS+CAT_KEYS)})
                    model_kind="linlogit"
                logger and logger.info("Scanner-ML: trained on %d rows (%s)", len(X), model_kind)
            else:
                logger and logger.info("Scanner-ML: not enough data for training (%d < %d)", len(X), ML_MIN_ROWS)
            await asyncio.sleep(ML_RETRAIN_SEC)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger and logger.warning("Scanner-ML: train error: %s", e)
            await asyncio.sleep(60)

def _details_to_row(details: Dict[str, Any]) -> List[float]:
    row: List[float] = []
    for k in NUM_KEYS: 
        try: row.append(float(details.get(k, 0.0) or 0.0))
        except Exception: row.append(0.0)
    for k in CAT_KEYS: row.append(_cat_to_num(details.get(k), k))
    return row

def _wrap_score_with_ml(app: Dict[str, Any]):
    logger = app.get("logger"); orig = app.get("score_symbol_core")
    if not callable(orig): return
    def _score_ml(symbol: str, relax: bool = False):
        base = orig(symbol, relax)
        if base is None: return None
        score, side, details = base; details = dict(details or {})
        p_base = float(details.get("p_bayes", details.get("ml_p", 0.55)) or 0.55)
        cache = app.setdefault("_scan_model_cache", {"ts": 0.0, "model": None})
        mdl = cache.get("model")
        p_ml = None
        try:
            if mdl and mdl.get("kind") in ("linlogit","sk","lgbm"):
                p_ml = _predict_p(mdl, _details_to_row(details))
        except Exception:
            p_ml = None
        if p_ml is not None:
            p_star = float(ML_BLEND_W * p_ml + (1.0 - ML_BLEND_W) * p_base)
            details["p_ml"] = float(p_ml); details["p_bayes"] = float(p_star)
            score = float(score + (p_star - p_base) * 0.8)
        try:
            _sanitize_levels(app, details, symbol, side)
        except Exception:
            pass
        return float(score), side, details
    app["score_symbol_core"] = _score_ml
    logger and logger.info("Scanner-ML: score patched (blend p_ml into p*).")

def _patch_cmd_signal_snapshot(app: Dict[str, Any]):
    logger = app.get("logger"); router = app.get("router")
    if not router: return
    try:
        obs = router.message; handlers = getattr(obs, "handlers", [])
        target = None
        for h in handlers:
            cb = getattr(h, "callback", None)
            if cb and "cmd_signal" in getattr(cb, "__name__", ""): target = h; break
        if not target: return
        orig = target.callback
        async def cmd_signal_wrapped(message, bot):
            await orig(message, bot)
            try:
                db = app.get("db")
                if not db or not db.conn: return
                cur = await db.conn.execute("SELECT id, symbol, side FROM signals WHERE user_id=? ORDER BY id DESC LIMIT 1", (message.from_user.id,))
                row = await cur.fetchone()
                if not row: return
                signal_id = int(row["id"]); symbol = str(row["symbol"]); side = str(row["side"])
                res = await _score_one(app, symbol)
                if res and isinstance(res, tuple) and len(res)==3:
                    score, _side, details = res; details = dict(details or {}); details["score"] = float(score)
                    await _save_snapshot_for_signal(app, signal_id, symbol, side, details)
            except Exception:
                pass
        setattr(target, "callback", cmd_signal_wrapped)
        logger and logger.info("Scanner: cmd_signal wrapped with feature snapshot.")
    except Exception:
        pass

def _patch_backup_command(app: Dict[str, Any]):
    router = app.get("router"); logger = app.get("logger")
    if not router: return
    async def _h_backup(message: Message):
        try:
            db = app.get("db"); bot = app.get("bot_instance")
            if not db or not getattr(db,"conn",None) or not bot:
                with contextlib.suppress(Exception): await message.answer("Сервис БД недоступен.")
                return
            admins = []
            with contextlib.suppress(Exception): admins = await db.get_admin_user_ids()
            if message.from_user.id not in (admins or []):
                with contextlib.suppress(Exception): await message.answer("Доступно только администраторам.")
                return
            db_path = getattr(db, "path", os.getenv("DB_PATH","neon_bot.db"))
            if not os.path.exists(db_path):
                with contextlib.suppress(Exception): await message.answer(f"Файл БД не найден: {db_path}")
                return
            doc = FSInputFile(db_path, filename=os.path.basename(db_path))
            with contextlib.suppress(Exception):
                size_kb = os.path.getsize(db_path)//1024
                await bot.send_document(message.chat.id, doc, caption=f"Бэкап БД ({size_kb} KB)")
        except Exception:
            with contextlib.suppress(Exception): await message.answer("Ошибка бэкапа.")
    router.message.register(_h_backup, Command("backup"))
    try:
        obs = router.message; handlers = getattr(obs, "handlers", [])
        target_fb = None
        for h in handlers:
            cb = getattr(h, "callback", None)
            if cb and "fallback" in getattr(cb, "__name__", ""):
                target_fb = h; break
        if target_fb:
            orig_fb = target_fb.callback
            async def fb_wrap(message: Message, bot):
                text = (message.text or "").strip().lower()
                if text.startswith("/backup"):
                    await _h_backup(message); return
                return await orig_fb(message, bot)
            setattr(target_fb, "callback", fb_wrap)
            logger and logger.info("Scanner: fallback wrapped for /backup.")
    except Exception:
        pass
    logger and logger.info("Scanner: /backup registered.")

def _patch_restore_command(app: Dict[str, Any]):
    router = app.get("router"); logger = app.get("logger")
    if not router: return
    async def _do_restore_from_doc(message: Message, doc):
        db = app.get("db"); bot = app.get("bot_instance")
        if not db or not getattr(db,"conn",None) or not bot:
            with contextlib.suppress(Exception): await message.answer("Сервис БД недоступен.")
            return
        admins = []
        with contextlib.suppress(Exception): admins = await db.get_admin_user_ids()
        if message.from_user.id not in (admins or []):
            with contextlib.suppress(Exception): await message.answer("Доступно только администраторам.")
            return
        db_path = getattr(db, "path", os.getenv("DB_PATH","neon_bot.db"))
        target = f"{db_path}.new"
        try:
            await bot.download(doc, destination=target)
            size_kb = os.path.getsize(target)//1024
            await message.answer(f"✅ Файл принят и сохранён как {os.path.basename(target)} ({size_kb} KB).\n"
                                 f"Для применения замените {os.path.basename(db_path)} -> {os.path.basename(target)} при остановленном боте и перезапустите.")
        except Exception:
            with contextlib.suppress(Exception): await message.answer("Ошибка сохранения. Отправьте файл ещё раз.")
    async def _h_restore_cmd(message: Message):
        if getattr(message, "reply_to_message", None) and getattr(message.reply_to_message, "document", None):
            await _do_restore_from_doc(message, message.reply_to_message.document)
        else:
            with contextlib.suppress(Exception):
                await message.answer("Пришлите файл БД и ответьте на него командой /restore, либо отправьте документ с подписью /restore.")
    async def _h_restore_doc(message: Message):
        if not getattr(message, "document", None):
            return
        caption = (message.caption or "").strip().lower()
        if not caption.startswith("/restore"):
            return
        await _do_restore_from_doc(message, message.document)
    router.message.register(_h_restore_cmd, Command("restore"))
    router.message.register(_h_restore_doc, F.document)
    try:
        obs = router.message; handlers = getattr(obs, "handlers", [])
        target_fb = None
        for h in handlers:
            cb = getattr(h, "callback", None)
            if cb and "fallback" in getattr(cb, "__name__", ""):
                target_fb = h; break
        if target_fb:
            orig_fb = target_fb.callback
            async def fb_wrap(message: Message, bot):
                text = (message.text or "").strip().lower()
                if text.startswith("/restore"):
                    await _h_restore_cmd(message); return
                return await orig_fb(message, bot)
            setattr(target_fb, "callback", fb_wrap)
            logger and logger.info("Scanner: fallback wrapped for /restore.")
    except Exception:
        pass
    logger and logger.info("Scanner: /restore registered.")

async def _ensure_users_columns_for_bd(app: Dict[str, Any]) -> None:
    db = app.get("db")
    if not db or not getattr(db, "conn", None):
        return
    cols = [
        "ALTER TABLE users ADD COLUMN pro_until TEXT",
        "ALTER TABLE users ADD COLUMN pro_notified INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE users ADD COLUMN pro_pre_notified INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE users ADD COLUMN city TEXT",
        "ALTER TABLE users ADD COLUMN last_seen TEXT",
        "ALTER TABLE users ADD COLUMN ta_count INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE users ADD COLUMN ta_date TEXT",
        "ALTER TABLE users ADD COLUMN analysis_count INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE users ADD COLUMN analysis_date TEXT",
    ]
    for s in cols:
        with contextlib.suppress(Exception):
            await db.conn.execute(s)
    with contextlib.suppress(Exception):
        await db.conn.commit()

async def _fetch_users_snapshot(app: Dict[str, Any]) -> List[Dict[str, Any]]:
    await _ensure_users_columns_for_bd(app)
    db = app.get("db")
    rows_out: List[Dict[str, Any]] = []
    try:
        cur = await db.conn.execute("""
            SELECT user_id, date, count, unlimited, support_mode, admin,
                   pro_until, pro_notified, pro_pre_notified, city,
                   ta_count, ta_date, analysis_count, analysis_date, last_seen
            FROM users
        """)
        rows = await cur.fetchall()
        for r in rows or []:
            rows_out.append({k: r[k] for k in r.keys()})
    except Exception:
        pass
    return rows_out

async def _apply_users_snapshot(app: Dict[str, Any], users: List[Dict[str, Any]]) -> int:
    await _ensure_users_columns_for_bd(app)
    db = app.get("db"); applied = 0
    if not users:
        return 0
    try:
        for u in users:
            try:
                vals = {
                    "user_id": int(u.get("user_id")),
                    "date": str(u.get("date") or ""),
                    "count": int(u.get("count") or 0),
                    "unlimited": int(u.get("unlimited") or 0),
                    "support_mode": int(u.get("support_mode") or 0),
                    "admin": int(u.get("admin") or 0),
                    "pro_until": u.get("pro_until"),
                    "pro_notified": int(u.get("pro_notified") or 0),
                    "pro_pre_notified": int(u.get("pro_pre_notified") or 0),
                    "city": u.get("city"),
                    "ta_count": int(u.get("ta_count") or 0),
                    "ta_date": u.get("ta_date"),
                    "analysis_count": int(u.get("analysis_count") or 0),
                    "analysis_date": u.get("analysis_date"),
                    "last_seen": u.get("last_seen"),
                }
            except Exception:
                continue
            await db.conn.execute("""
                INSERT INTO users(user_id, date, count, unlimited, support_mode, admin, pro_until, pro_notified, pro_pre_notified, city, ta_count, ta_date, analysis_count, analysis_date, last_seen)
                VALUES (:user_id, :date, :count, :unlimited, :support_mode, :admin, :pro_until, :pro_notified, :pro_pre_notified, :city, :ta_count, :ta_date, :analysis_count, :analysis_date, :last_seen)
                ON CONFLICT(user_id) DO UPDATE SET
                    date=excluded.date,
                    count=excluded.count,
                    unlimited=excluded.unlimited,
                    support_mode=excluded.support_mode,
                    admin=excluded.admin,
                    pro_until=excluded.pro_until,
                    pro_notified=excluded.pro_notified,
                    pro_pre_notified=excluded.pro_pre_notified,
                    city=excluded.city,
                    ta_count=excluded.ta_count,
                    ta_date=excluded.ta_date,
                    analysis_count=excluded.analysis_count,
                    analysis_date=excluded.analysis_date,
                    last_seen=excluded.last_seen
            """, vals)
            applied += 1
        await db.conn.commit()
    except Exception:
        pass
    return applied

def _patch_bd_command(app: Dict[str, Any]):
    router = app.get("router"); logger = app.get("logger")
    if not router: return

    async def _h_bd(message: Message):
        try:
            db = app.get("db"); bot = app.get("bot_instance")
            if not db or not getattr(db,"conn",None) or not bot:
                with contextlib.suppress(Exception): await message.answer("Сервис БД недоступен.")
                return
            admins = []
            with contextlib.suppress(Exception): admins = await db.get_admin_user_ids()
            if message.from_user.id not in (admins or []):
                with contextlib.suppress(Exception): await message.answer("Доступно только администраторам.")
                return
            users = await _fetch_users_snapshot(app)
            data = json.dumps({"exported_at": app["now_msk"]().isoformat(), "users": users}, ensure_ascii=False, indent=2)
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
                tmp.write(data)
                path = tmp.name
            doc = FSInputFile(path, filename="users_export.json")
            await bot.send_document(message.chat.id, doc, caption=f"Экспорт пользователей ({len(users)})")
        except Exception:
            with contextlib.suppress(Exception): await message.answer("Ошибка экспорта /bd.")

    async def _bd_restore_from_doc(message: Message, doc):
        try:
            db = app.get("db"); bot = app.get("bot_instance")
            if not db or not getattr(db,"conn",None) or not bot:
                with contextlib.suppress(Exception): await message.answer("Сервис БД недоступен.")
                return
            admins = []
            with contextlib.suppress(Exception): admins = await db.get_admin_user_ids()
            if message.from_user.id not in (admins or []):
                with contextlib.suppress(Exception): await message.answer("Доступно только администраторам.")
                return
            with tempfile.TemporaryDirectory() as td:
                target = os.path.join(td, "users_import.json")
                await bot.download(doc, destination=target)
                obj = None
                with open(target, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                users = obj.get("users") if isinstance(obj, dict) else None
                if not isinstance(users, list):
                    await message.answer("Формат не распознан. Ожидается JSON с ключом 'users'."); return
                applied = await _apply_users_snapshot(app, users)
                await message.answer(f"✅ Импортировано/обновлено пользователей: {applied}")
        except Exception:
            with contextlib.suppress(Exception): await message.answer("Ошибка импорта /bd_restore.")

    async def _h_bd_restore_cmd(message: Message):
        if getattr(message, "reply_to_message", None) and getattr(message.reply_to_message, "document", None):
            await _bd_restore_from_doc(message, message.reply_to_message.document)
        else:
            with contextlib.suppress(Exception):
                await message.answer("Пришлите JSON (users_export.json) и ответьте на него командой /bd_restore, либо отправьте документ с подписью /bd_restore.")

    async def _h_bd_restore_doc(message: Message):
        if not getattr(message, "document", None):
            return
        caption = (message.caption or "").strip().lower()
        if not caption.startswith("/bd_restore"):
            return
        await _bd_restore_from_doc(message, message.document)

    router.message.register(_h_bd, Command("bd"))
    router.message.register(_h_bd_restore_cmd, Command("bd_restore"))
    router.message.register(_h_bd_restore_doc, F.document)

    try:
        obs = router.message; handlers = getattr(obs, "handlers", [])
        target_fb = None
        for h in handlers:
            cb = getattr(h, "callback", None)
            if cb and "fallback" in getattr(cb, "__name__", ""):
                target_fb = h; break
        if target_fb:
            orig_fb = target_fb.callback
            async def fb_wrap(message: Message, bot):
                text = (message.text or "").strip().lower()
                if text.startswith("/bd"):
                    await _h_bd(message); return
                if text.startswith("/bd_restore"):
                    await _h_bd_restore_cmd(message); return
                return await orig_fb(message, bot)
            setattr(target_fb, "callback", fb_wrap)
            logger and logger.info("Scanner: fallback wrapped for /bd and /bd_restore.")
    except Exception:
        pass
    logger and logger.info("Scanner: /bd + /bd_restore registered.")

async def _auto_backup_loop(app: Dict[str, Any]):
    await asyncio.sleep(10)
    db = app.get("db"); bot = app.get("bot_instance"); logger = app.get("logger")
    if not db or not getattr(db,"conn",None) or not bot:
        return
    while True:
        try:
            admins = []
            with contextlib.suppress(Exception): admins = await db.get_admin_user_ids()
            db_path = getattr(db, "path", os.getenv("DB_PATH","neon_bot.db"))
            if os.path.exists(db_path) and admins:
                doc = FSInputFile(db_path, filename=os.path.basename(db_path))
                size_kb = os.path.getsize(db_path)//1024
                for aid in admins:
                    with contextlib.suppress(Exception):
                        await bot.send_document(aid, doc, caption=f"Автобэкап БД ({size_kb} KB)")
            await asyncio.sleep(max(3600, SCAN_AUTO_BACKUP_SEC))
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger and logger.warning("Auto-backup loop error: %s", e)
            await asyncio.sleep(60)

async def _refresh_model_cache_loop(app: Dict[str, Any]):
    app["_scan_model_cache"] = {"ts": 0.0, "model": None}
    while True:
        try:
            mdl = await _load_model_db(app)
            app["_scan_model_cache"] = {"ts": time.time(), "model": mdl}
        except Exception:
            pass
        await asyncio.sleep(max(60, SCAN_MODEL_TTL_SEC))

def _patch_support_bridge(app: Dict[str, Any]):
    router = app.get("router"); logger = app.get("logger")
    if not router:
        return

    _bridge_state = app.setdefault("_support_bridge", {})  # user_id -> last_ts

    async def _h_support_bridge(message: Message):
        try:
            uid = int(getattr(message.from_user, "id", 0) or 0)
            now = time.time()
            last = float(_bridge_state.get(uid, 0.0))
            if now - last < 2.0:
                return
            _bridge_state[uid] = now
            db = app.get("db"); bot = app.get("bot_instance")
            support_kb = app.get("support_kb")
            if db and getattr(db, "conn", None):
                await db.set_support_mode(uid, True)
            if bot:
                if callable(support_kb):
                    await bot.send_message(message.chat.id, "Напишите ваш вопрос", reply_markup=support_kb())
                else:
                    await bot.send_message(message.chat.id, "Напишите ваш вопрос")
        except Exception:
            pass

    router.message.register(
        _h_support_bridge,
        F.chat.type == "private",
        F.text.in_({"Поддержка", "Поддержка"})
    )
    logger and logger.info("Scanner: support bridge registered (emoji and plain).")

def patch(app: Dict[str, Any]) -> None:
    logger = app.get("logger")
    _wrap_score_with_ml(app)
    _patch_cmd_signal_snapshot(app)
    _patch_backup_command(app)
    _patch_restore_command(app)
    _patch_bd_command(app)
    _patch_support_bridge(app)
    orig_on_startup = app.get("on_startup")
    async def _on_startup_scanner(bot):
        await _ensure_scan_tables(app)
        asyncio.create_task(_scan_loop(app))
        asyncio.create_task(_digest_loop(app))
        asyncio.create_task(_calibration_loop(app))
        asyncio.create_task(_train_loop(app))
        asyncio.create_task(_auto_backup_loop(app))
        asyncio.create_task(_refresh_ladder_loop(app))
        asyncio.create_task(_refresh_model_cache_loop(app))
        if orig_on_startup:
            await orig_on_startup(bot)
    app["on_startup"] = _on_startup_scanner
    logger and logger.info("Scanner: patch applied (scan+digest+calibrator+inline ML+/backup+/restore+/bd+support-bridge+unified-fallback+auto-backup+RR+portfolio+session+ladder+model-cache+sanitize+anti-flip).")
