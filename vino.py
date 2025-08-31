# vino.py
# Ultra quality-gate + news/macro/microstructure/correlation + trailing/alerts + admin deactivation + cooldown + announcements + size hint + scalping.
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, List
import os, math, asyncio, contextlib
from datetime import datetime, timedelta

from aiogram.filters import Command
from aiogram.types import Message

import dio  # helper module

# -------- ENV --------
VINO_CLEAR_UNLIMITED = os.getenv("VINO_CLEAR_UNLIMITED","1") == "1"
VINO_CLEAR_SUPPORT   = os.getenv("VINO_CLEAR_SUPPORT","0") == "1"
VINO_MENU_REFRESH    = os.getenv("VINO_MENU_REFRESH","1") == "1"

# Strictness and minimum score (calibrator may adjust in-memory)
VINO_STRICT     = os.getenv("VINO_STRICT","1") == "1"
VINO_MIN_SCORE  = float(os.getenv("VINO_MIN_SCORE","1.90"))
VINO_ADX_MIN    = float(os.getenv("VINO_ADX_MIN","22"))
VINO_RR1_MIN    = float(os.getenv("VINO_RR1_MIN","1.2"))
VINO_RR2_MIN    = float(os.getenv("VINO_RR2_MIN","1.6"))
VINO_LEV_CAP    = int(os.getenv("VINO_LEV_CAP","8"))

# ML-gate
VINO_ML_P_MIN   = float(os.getenv("VINO_ML_P_MIN","0.55"))

# Macro gating
MACRO_GATING    = os.getenv("TA_G_MACRO_GATING","0") == "1"
MACRO_EVENTS    = dio.parse_macro_events_env(os.getenv("TA_G_MACRO_EVENTS",""))
MACRO_MIN_BEFORE= int(os.getenv("TA_G_MACRO_MIN_BEFORE","45"))

# News parameters
VINO_NEWS_WEIGHT        = float(os.getenv("VINO_NEWS_WEIGHT","0.35"))
VINO_NEWS_MIN_SCORE     = float(os.getenv("VINO_NEWS_MIN_SCORE","0.60"))
VINO_NEWS_STRICT        = os.getenv("VINO_NEWS_STRICT","1") == "1"
VINO_BTC_ETH_CROSSBOOST = float(os.getenv("VINO_BTC_ETH_CROSSBOOST","0.25"))
VINO_NEWS_ALERTS        = os.getenv("VINO_NEWS_ALERTS","1") == "1"
VINO_NEWS_CHECK_SEC     = int(os.getenv("VINO_NEWS_CHECK_SEC","90"))

# Announcements
VINO_ANN_ENABLE         = os.getenv("VINO_ANN_ENABLE","0") == "1"
VINO_ANN_SOURCES        = os.getenv("VINO_ANN_SOURCES","")
VINO_ANN_WEIGHT         = float(os.getenv("VINO_ANN_WEIGHT","0.50"))
VINO_ANN_STRICT         = os.getenv("VINO_ANN_STRICT","1") == "1"
VINO_ANN_TTL_SEC        = int(os.getenv("VINO_ANN_TTL_SEC","600"))

# Correlations
CORR_WINDOW     = int(os.getenv("VINO_CORR_WINDOW","180"))
CORR_THRESH     = float(os.getenv("VINO_CORR_THRESH","0.50"))

# Anti-stale
STALE_MULT_5M   = float(os.getenv("VINO_STALE_MULT_5M","3.0"))
STALE_MULT_15M  = float(os.getenv("VINO_STALE_MULT_15M","3.0"))
STALE_MULT_1H   = float(os.getenv("VINO_STALE_MULT_1H","3.0"))
VINO_STALE_SKIP_1W = os.getenv("VINO_STALE_SKIP_1W","1") == "1"

# Stop-day guard + cooldown
VINO_DAY_GUARD          = os.getenv("VINO_DAY_GUARD","1") == "1"
VINO_DAY_GUARD_HOURS    = int(os.getenv("VINO_DAY_GUARD_HOURS","12"))
VINO_DAY_STOP_RATE_MAX  = float(os.getenv("VINO_DAY_STOP_RATE_MAX","0.55"))
VINO_DAY_MIN_TRADES     = int(os.getenv("VINO_DAY_MIN_TRADES","5"))
VINO_COOLDOWN_AFTER_LOSS= int(os.getenv("VINO_COOLDOWN_AFTER_LOSS","45"))

# Session/time blocks
VINO_SESSION_BLOCK      = os.getenv("VINO_SESSION_BLOCK","").upper()
VINO_TIME_BLOCK         = os.getenv("VINO_TIME_BLOCK","")

# Whitelist/blacklist & meme rules
VINO_WHITELIST = [s.strip().upper() for s in os.getenv("VINO_WHITELIST","").split(",") if s.strip()]
VINO_BLACKLIST = [s.strip().upper() for s in os.getenv("VINO_BLACKLIST","").split(",") if s.strip()]
VINO_MEME_PRICE_MAX     = float(os.getenv("VINO_MEME_PRICE_MAX","0.005"))
VINO_MEME_SPREAD_Z_MAX  = float(os.getenv("VINO_MEME_SPREAD_Z_MAX","2.5"))

# Perfect-only mode
VINO_PERFECT_ONLY       = os.getenv("VINO_PERFECT_ONLY","0") == "1"

# Size hint (Kelly-lite)
VINO_SHOW_SIZE_HINT     = os.getenv("VINO_SHOW_SIZE_HINT","1") == "1"
VINO_SIZE_MAX_RISK_PCT  = float(os.getenv("VINO_SIZE_MAX_RISK_PCT","1.0"))
VINO_SIZE_MAX_LEV       = int(os.getenv("VINO_SIZE_MAX_LEV","10"))

# SCALPING MODE
VINO_SCALP_MODE         = os.getenv("VINO_SCALP_MODE","0") == "1"
VINO_SCALP_FORCE        = os.getenv("VINO_SCALP_FORCE","1") == "1"   # 1 — применять ко всем сетапам
VINO_SCALP_TP_PCTS      = [float(x) for x in os.getenv("VINO_SCALP_TP_PCTS","0.016,0.052,0.092").split(",")]
VINO_SCALP_SL_PCT       = float(os.getenv("VINO_SCALP_SL_PCT","0.041"))
VINO_SCALP_FORMAT       = os.getenv("VINO_SCALP_FORMAT","1") == "1"

# Sidecar intervals
SIDE_NEWS_INT   = max(30, VINO_NEWS_CHECK_SEC)
SIDE_MACRO_INT  = int(os.getenv("VINO_SIDE_MACRO_INT","60"))

# Prometheus counters
def _init_metrics(app: Dict[str,Any]) -> Dict[str,Any]:
    if not app.get("PROMETHEUS_OK"):
        return {}
    try:
        from prometheus_client import Counter
        return {
            "G_DROP": Counter("vino_gate_drop_total","vino gate drop total",["reason"]),
            "G_PEN":  Counter("vino_gate_penalty_total","vino gate penalty total",["reason"]),
            "ALERTS": Counter("vino_alerts_total","vino alerts total",["kind"]),
        }
    except Exception:
        return {}

# -------- News util --------
def _news_for_base(app: Dict[str, Any], base: str) -> Tuple[float,str]:
    try:
        f = app.get("fetch_news_sentiment_cached")
        if callable(f):
            b, note = f(base)
            return float(b), str(note or "")
    except Exception:
        pass
    return 0.2, ""

# -------- /admin helpers --------
async def _ensure_columns(app: Dict[str, Any]) -> None:
    db = app.get("db")
    if not db or not getattr(db, "conn", None):
        return
    for s in (
        "ALTER TABLE users ADD COLUMN admin INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE users ADD COLUMN unlimited INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE users ADD COLUMN support_mode INTEGER NOT NULL DEFAULT 0",
    ):
        with contextlib.suppress(Exception):
            await db.conn.execute(s)
    with contextlib.suppress(Exception):
        await db.conn.commit()

async def _do_admin_off(app: Dict[str, Any], message: Message) -> None:
    db = app.get("db"); bot = app.get("bot_instance")
    if not db or not getattr(db,"conn",None):
        with contextlib.suppress(Exception): await message.answer("Сервис БД недоступен. Попробуйте позже.")
        return
    await _ensure_columns(app)
    uid = int(message.from_user.id)
    with contextlib.suppress(Exception): await db.get_user_state(uid)
    fields = ["admin=0"]; 
    if VINO_CLEAR_UNLIMITED: fields.append("unlimited=0")
    if VINO_CLEAR_SUPPORT:   fields.append("support_mode=0")
    try:
        await db.conn.execute(f"UPDATE users SET {', '.join(fields)} WHERE user_id=?", (uid,))
        await db.conn.commit()
    except Exception:
        with contextlib.suppress(Exception): await message.answer("⚠️ Не удалось снять админские права. Попробуйте позже.")
        return
    parts = ["🔓 Админ-режим отключён."]
    if VINO_CLEAR_UNLIMITED: parts.append("Ограничения восстановлены.")
    if VINO_CLEAR_SUPPORT:   parts.append("Режим поддержки выключен.")
    with contextlib.suppress(Exception): await message.answer(" ".join(parts))
    if VINO_MENU_REFRESH and getattr(message.chat,"type",None)=="private":
        menu = app.get("main_menu_kb"); kb = menu(False) if callable(menu) else None
        with contextlib.suppress(Exception): await bot.send_message(message.chat.id,"Вы вернулись к основному функционалу.", reply_markup=kb)

# -------- Helpers --------
def _base_of_symbol(symbol: str) -> str:
    try: return symbol.split("/")[0]
    except Exception: return symbol

def _session_of_hour(h: int) -> str:
    if 0 <= h < 8: return "ASIA"
    if 7 <= h < 13: return "EU"
    if 12 <= h < 20: return "US"
    return "ASIA"

def _in_time_blocks(now_msk: datetime, blocks: str) -> bool:
    if not blocks: return False
    for part in blocks.split(","):
        s = part.strip()
        if not s or "-" not in s: continue
        a,b = s.split("-",1)
        try:
            h1,m1 = [int(x) for x in a.split(":")]
            h2,m2 = [int(x) for x in b.split(":")]
            t1 = now_msk.replace(hour=h1,minute=m1,second=0,microsecond=0)
            t2 = now_msk.replace(hour=h2,minute=m2,second=0,microsecond=0)
            if t1 <= t2:
                if t1 <= now_msk <= t2: return True
            else:
                if now_msk >= t1 or now_msk <= t2: return True
        except Exception:
            continue
    return False

# -------- Market anti-stale wrappers --------
def _wrap_market_guard(app: Dict[str, Any]) -> None:
    market = app.get("market"); logger = app.get("logger")
    if not market: 
        logger and logger.warning("VINO: market not found for stale guard.")
        return
    if not hasattr(market, "_vino_wrapped_ohlcv"):
        orig_ohlcv = market.fetch_ohlcv
        def fetch_ohlcv_guard(symbol: str, timeframe: str, limit: int = 250):
            df = orig_ohlcv(symbol, timeframe, limit)
            # skip 1w stale drop if enabled
            if timeframe == "1w" and VINO_STALE_SKIP_1W:
                return df
            mult = 3.0
            if timeframe == "5m": mult = STALE_MULT_5M
            elif timeframe == "15m": mult = STALE_MULT_15M
            elif timeframe in ("1h","4h"): mult = STALE_MULT_1H
            if dio.df_is_stale(df, timeframe, max_mult=mult):
                logger and logger.warning("VINO: stale OHLCV %s %s → drop", symbol, timeframe)
                return None
            return df
        market.fetch_ohlcv = fetch_ohlcv_guard
        setattr(market, "_vino_wrapped_ohlcv", True)
        logger and logger.info("VINO: market.fetch_ohlcv wrapped (anti-stale).")

# -------- Score helper --------
def _apply_min_gates(score: float, side: str, d: Dict[str,Any], relax: bool, logger, MET) -> Tuple[Optional[float], Dict[str,float]]:
    br = d.get("score_breakdown", {}) or {}
    adx = float(d.get("adx15", 0.0) or 0.0)
    rr1 = d.get("rr1"); rr2=d.get("rr2")
    if adx < VINO_ADX_MIN and not relax:
        score -= 0.15; br["VINO_adx"] = br.get("VINO_adx",0.0) - 0.15
        MET.get("G_PEN") and MET["G_PEN"].labels("low_adx").inc()
    if isinstance(rr1,(int,float)) and rr1 < VINO_RR1_MIN and not relax:
        score -= 0.25; br["VINO_rr1"] = br.get("VINO_rr1",0.0) - 0.25
        MET.get("G_PEN") and MET["G_PEN"].labels("rr1").inc()
    if isinstance(rr2,(int,float)) and rr2 < VINO_RR2_MIN and not relax:
        score -= 0.15; br["VINO_rr2"] = br.get("VINO_rr2",0.0) - 0.15
        MET.get("G_PEN") and MET["G_PEN"].labels("rr2").inc()
    p_ml = d.get("p_bayes")
    if isinstance(p_ml,(int,float)) and p_ml < VINO_ML_P_MIN and not relax:
        score -= 0.20; br["VINO_ml"] = br.get("VINO_ml",0.0) - 0.20
        MET.get("G_PEN") and MET["G_PEN"].labels("ml").inc()
    adj = 0.0
    try:
        cfg = d.get("_vino_cfg") or {}
        adj = cfg.get("min_score_adj", 0.0)
    except Exception:
        pass
    min_score = VINO_MIN_SCORE + adj
    if VINO_STRICT and not relax and score < min_score:
        logger and logger.info("VINO: DROP by min_score %.2f<%.2f for %s", score, min_score, d.get("symbol","?"))
        MET.get("G_DROP") and MET["G_DROP"].labels("min_score").inc()
        return None, br
    return score, br

# -------- Scalping helpers --------
def _apply_scalp_levels(app: Dict[str,Any], d: Dict[str,Any], symbol: str, side: str) -> None:
    if not VINO_SCALP_MODE: 
        return
    try:
        entry = float(d.get("c5") or 0.0)
        if entry <= 0: 
            return
        market = app.get("market")
        tick = 0.0
        with contextlib.suppress(Exception):
            tick = float(market.get_tick_size(symbol) or 0.0)
        tps = []
        if side == "LONG":
            for p in VINO_SCALP_TP_PCTS:
                tps.append(entry * (1.0 + p))
            sl = entry * (1.0 - VINO_SCALP_SL_PCT)
            if tick>0:
                tps = [round(tp/tick)*tick for tp in tps]
                sl  = math.floor(sl/tick)*tick
        else:
            for p in VINO_SCALP_TP_PCTS:
                tps.append(entry * (1.0 - p))
            sl = entry * (1.0 + VINO_SCALP_SL_PCT)
            if tick>0:
                tps = [round(tp/tick)*tick for tp in tps]
                sl  = math.ceil(sl/tick)*tick
        d["tps"] = [float(x) for x in tps[:3]]
        d["sl"]  = float(sl)
    except Exception:
        pass

# -------- Sidecars --------
async def _news_sidecar(app: Dict[str,Any], bot, chat_id: int, sig, MET) -> None:
    try:
        last_boost = None
        base_sym = _base_of_symbol(str(getattr(sig,"symbol","")))
        while getattr(sig,"active",False) and app["now_msk"]() < sig.watch_until:
            boost, note = _news_for_base(app, base_sym)
            btc_b,_ = _news_for_base(app, "BTC"); eth_b,_ = _news_for_base(app, "ETH")
            boost = max(0.0, min(2.0, boost + ((btc_b+eth_b)/2.0)*VINO_BTC_ETH_CROSSBOOST))
            if last_boost is None: last_boost = boost
            try:
                _should_alert = app.get("_should_alert")
                can = True
                if callable(_should_alert): can = _should_alert(sig.id or -1, "news_vino2")
                if can:
                    neg,pos = dio.parse_news_note(app, note)
                    worsening = (last_boost - boost) >= 0.25
                    against_long = (sig.side=="LONG" and (neg - pos) >= 1.2)
                    against_short= (sig.side=="SHORT" and (pos - neg) >= 1.2)
                    if worsening or against_long or against_short:
                        txt = f"🗞 Новостной фон по {_base_of_symbol(sig.symbol)} ухудшился: {boost:.2f} (было {last_boost:.2f})."
                        if against_long or against_short: txt += " Против направления позиции."
                        txt += "\nРекомендация: частичная фиксация / поджать стоп."
                        with contextlib.suppress(Exception):
                            await bot.send_message(chat_id, txt)
                        MET.get("ALERTS") and MET["ALERTS"].labels("news").inc()
            except Exception:
                pass
            last_boost = boost
            await asyncio.sleep(SIDE_NEWS_INT)
    except asyncio.CancelledError:
        return
    except Exception:
        pass

async def _macro_sidecar(app: Dict[str,Any], bot, chat_id: int, sig, MET)->None:
    if not MACRO_GATING or not MACRO_EVENTS: return
    try:
        while getattr(sig,"active",False) and app["now_msk"]() < sig.watch_until:
            mins = dio.minutes_to_next_event(MACRO_EVENTS)
            if mins is not None and mins <= MACRO_MIN_BEFORE:
                _should_alert = app.get("_should_alert")
                can = True
                if callable(_should_alert): can = _should_alert(sig.id or -1, "macro")
                if can:
                    with contextlib.suppress(Exception):
                        await bot.send_message(chat_id, f"📢 Через {int(mins)} мин макро‑событие. Рассмотрите снижение риска/фиксацию.")
                    MET.get("ALERTS") and MET["ALERTS"].labels("macro").inc()
            await asyncio.sleep(SIDE_MACRO_INT)
    except asyncio.CancelledError:
        return
    except Exception:
        pass

# -------- Patch entry --------
def patch(app: Dict[str, Any]) -> None:
    logger = app.get("logger")
    router = app.get("router")
    MET = _init_metrics(app)

    # /admin command + fallback intercept
    async def _h_admin_cmd(message: Message):
        await _do_admin_off(app, message)
    if router:
        router.message.register(_h_admin_cmd, Command("admin"))
        try:
            obs = router.message; handlers = list(getattr(obs,"handlers",[]))
            target_fb = None
            for h in handlers:
                name = getattr(getattr(h,"callback",None),"__name__", "")
                if "fallback" in name: target_fb = h; break
            if target_fb:
                orig_fb = target_fb.callback
                async def fb_wrap(message: Message, bot):
                    txt = (message.text or "").strip().lower()
                    if txt.startswith("/admin"):
                        await _do_admin_off(app, message); return
                    return await orig_fb(message, bot)
                setattr(target_fb,"callback", fb_wrap)
                logger and logger.info("VINO: fallback wrapped to intercept /admin first.")
        except Exception:
            pass
    else:
        logger and logger.warning("VINO: router not found; /admin handler not registered.")

    # Market anti-stale
    _wrap_market_guard(app)

    # Score wrapper
    orig_score = app.get("score_symbol_core")
    if not callable(orig_score):
        logger and logger.warning("VINO: score_symbol_core not found — gating disabled.")
    else:
        def _vino_score(symbol: str, relax: bool = False):
            base = orig_score(symbol, relax)
            if base is None:
                return None
            score, side, d = base
            d = dict(d or {}); br = d.get("score_breakdown", {}) or {}
            d["_vino_cfg"] = app.get("_vino_cfg", {})  # for calibrator adj
            base_sym = _base_of_symbol(d.get("symbol", symbol))

            # Whitelist/blacklist
            if VINO_BLACKLIST and base_sym.upper() in VINO_BLACKLIST:
                MET.get("G_DROP") and MET["G_DROP"].labels("blacklist").inc()
                return None

            now_msk = app["now_msk"]()
            sess = _session_of_hour(now_msk.hour)
            if VINO_SESSION_BLOCK and sess in {s.strip() for s in VINO_SESSION_BLOCK.split(",") if s.strip()} and not relax:
                MET.get("G_DROP") and MET["G_DROP"].labels("session").inc()
                return None
            if _in_time_blocks(now_msk, VINO_TIME_BLOCK) and not relax:
                MET.get("G_DROP") and MET["G_DROP"].labels("time").inc()
                return None

            # Macro gating
            if MACRO_GATING and MACRO_EVENTS and not relax:
                mins = dio.minutes_to_next_event(MACRO_EVENTS)
                if mins is not None and mins <= MACRO_MIN_BEFORE:
                    logger and logger.info("VINO: DROP by macro window %s side=%s", d.get("symbol","?"), side)
                    MET.get("G_DROP") and MET["G_DROP"].labels("macro").inc()
                    return None

            # News + announcements
            boost, note = _news_for_base(app, base_sym)
            if not note and d.get("news_note"): note = d["news_note"]
            if VINO_ANN_ENABLE:
                extra, annote, strict_neg = dio.announcements_boost(app, base_sym, ttl=VINO_ANN_TTL_SEC)
                if annote:
                    note = (note + " | " if note else "") + f"ANN: {annote}"
                if strict_neg and VINO_ANN_STRICT and not relax:
                    MET.get("G_DROP") and MET["G_DROP"].labels("announce_strict").inc()
                    return None
                boost = max(0.0, min(2.0, boost + VINO_ANN_WEIGHT * extra))
            btc_b,_ = _news_for_base(app, "BTC"); eth_b,_ = _news_for_base(app, "ETH")
            boost = max(0.0, min(2.0, boost + ((btc_b+eth_b)/2.0) * VINO_BTC_ETH_CROSSBOOST))
            neg,pos = dio.parse_news_note(app, note)
            if (VINO_NEWS_STRICT and ((side=="LONG" and (neg-pos)>=1.5) or (side=="SHORT" and (pos-neg)>=1.5))) and not relax:
                MET.get("G_DROP") and MET["G_DROP"].labels("news_strict").inc()
                return None
            p_news = dio.news_logistic_p(note)
            adj = VINO_NEWS_WEIGHT * (boost - VINO_NEWS_MIN_SCORE)
            score = float(score) + float(adj)
            br["VINO_news"] = br.get("VINO_news",0.0) + adj
            d["vino_news_boost"] = float(boost)
            d["vino_p_news"] = float(p_news) if p_news is not None else None

            # Correlation & BTC trend alignment
            try:
                mkt = app.get("market")
                df5_self = mkt.fetch_ohlcv(d.get("symbol", symbol), "5m", 240)
                df5_btc  = mkt.fetch_ohlcv("BTC/USDT", "5m", 240)
                if df5_self is not None and df5_btc is not None:
                    corr = dio.pearson_corr(df5_self["close"], df5_btc["close"], CORR_WINDOW)
                    btc_tr = dio.simple_trend(df5_btc["close"], 220)
                    if abs(corr) >= CORR_THRESH and ((btc_tr>0 and side=="SHORT") or (btc_tr<0 and side=="LONG")) and not relax:
                        score -= 0.20; br["VINO_corr"] = br.get("VINO_corr",0.0) - 0.20
                        MET.get("G_PEN") and MET["G_PEN"].labels("corr_btc").inc()
            except Exception:
                pass

            # CP shock penalty
            try:
                df15 = app.get("market").fetch_ohlcv(d.get("symbol", symbol), "15m", 300)
                cp = dio.cp_volatility(df15)
                if cp >= 0.6 and not relax:
                    score -= 0.20; br["VINO_cp"] = br.get("VINO_cp",0.0) - 0.20
                    MET.get("G_PEN") and MET["G_PEN"].labels("shock_cp").inc()
            except Exception:
                pass

            # Minimal gates (ADX/RR/ML)
            score, br2 = _apply_min_gates(score, side, d, relax, logger, MET)
            if score is None:
                return None
            br.update(br2)

            # SCALP levels override (по желанию — ко всем сетапам)
            if VINO_SCALP_MODE and (VINO_SCALP_FORCE or d.get("quick")):
                _apply_scalp_levels(app, d, d.get("symbol", symbol), side)

            # Perfect-only mode (если включено)
            if VINO_PERFECT_ONLY and not relax:
                ok = True
                adx = float(d.get("adx15", 0.0) or 0.0)
                corr_ok = True
                try:
                    mkt = app.get("market")
                    df5_self = mkt.fetch_ohlcv(d.get("symbol", symbol), "5m", 240)
                    df5_btc = mkt.fetch_ohlcv("BTC/USDT", "5m", 240)
                    if df5_self is not None and df5_btc is not None:
                        corr = dio.pearson_corr(df5_self["close"], df5_btc["close"], CORR_WINDOW)
                        btc_tr = dio.simple_trend(df5_btc["close"], 220)
                        corr_ok = not (abs(corr) >= CORR_THRESH and ((btc_tr>0 and side=="SHORT") or (btc_tr<0 and side=="LONG")))
                except Exception:
                    pass
                ok = ok and (adx >= 25.0) and (d.get("vino_news_boost", VINO_NEWS_MIN_SCORE) >= 0.80) and corr_ok and (d.get("rr1",1.0) >= 1.4) and (d.get("vino_p_news",0.5) is None or d.get("vino_p_news",0.5) >= 0.55)
                if not ok:
                    MET.get("G_DROP") and MET["G_DROP"].labels("perfect_only").inc()
                    return None

            # cap leverage if negative news
            try:
                if d.get("vino_news_boost", VINO_NEWS_MIN_SCORE) < VINO_NEWS_MIN_SCORE:
                    lev = int(d.get("leverage", 10) or 10)
                    d["leverage"] = min(lev, VINO_LEV_CAP)
            except Exception:
                pass

            d["score_breakdown"] = br; d["score"] = float(score)
            return float(score), side, d

        app["score_symbol_core"] = _vino_score
        logger and logger.info("VINO: score gate enabled (strict=%s, min_score=%.2f).", str(VINO_STRICT), VINO_MIN_SCORE)

    # Trailing & watch wrappers (news + macro sidecars)
    orig_watch = app.get("watch_signal_price")
    if callable(orig_watch):
        async def _watch_wrap(bot, chat_id: int, sig):
            t1 = t2 = None
            try:
                if VINO_NEWS_ALERTS:
                    t1 = asyncio.create_task(_news_sidecar(app, bot, chat_id, sig, MET))
                if MACRO_GATING and MACRO_EVENTS:
                    t2 = asyncio.create_task(_macro_sidecar(app, bot, chat_id, sig, MET))
            except Exception:
                pass
            try:
                await orig_watch(bot, chat_id, sig)
            finally:
                for t in (t1,t2):
                    if t and not t.done():
                        with contextlib.suppress(Exception): t.cancel()
        app["watch_signal_price"] = _watch_wrap
        logger and logger.info("VINO: watch sidecars (news/macro) enabled.")

    # Stop-day guard — обёртка cmd_signal + cooldown после потерь
    if router and (VINO_DAY_GUARD or VINO_COOLDOWN_AFTER_LOSS>0):
        try:
            obs = router.message; handlers = list(getattr(obs,"handlers",[]))
            target = None
            for h in handlers:
                cb = getattr(h,"callback",None)
                name = getattr(cb,"__name__", "")
                if "cmd_signal" in name: target = h; break
            if target:
                orig_cmd = target.callback
                async def cmd_signal_guard(message: Message, bot):
                    if VINO_DAY_GUARD:
                        stats = await dio.recent_outcomes_stats(app, hours=VINO_DAY_GUARD_HOURS)
                        n = stats.get("N",0); stops = stats.get("STOP",0)+stats.get("TIME",0)
                        rate = (stops/max(1,n)) if n else 0.0
                        if n >= VINO_DAY_MIN_TRADES and rate >= VINO_DAY_STOP_RATE_MAX:
                            with contextlib.suppress(Exception):
                                await message.answer("🛑 Режим стоп‑день: слишком много убыточных исходов за последнее время. Попробуйте позже.")
                            return
                    if VINO_COOLDOWN_AFTER_LOSS>0:
                        try:
                            db = app.get("db")
                            if db and db.conn:
                                cur = await db.conn.execute(
                                    "SELECT outcome, finished_at FROM outcomes WHERE user_id=? AND finished_at IS NOT NULL ORDER BY id DESC LIMIT 2",
                                    (message.from_user.id,)
                                )
                                rows = await cur.fetchall()
                                if rows and len(rows)>=2:
                                    bad = {str(rows[0]['outcome']).upper(), str(rows[1]['outcome']).upper()}
                                    if bad.issubset({"STOP","TIME"}):
                                        last_ts = max(datetime.fromisoformat(rows[0]['finished_at']).timestamp(),
                                                      datetime.fromisoformat(rows[1]['finished_at']).timestamp())
                                        rem = VINO_COOLDOWN_AFTER_LOSS*60 - (datetime.now().timestamp() - last_ts)
                                        if rem > 0:
                                            mins = int(max(1, rem//60))
                                            with contextlib.suppress(Exception):
                                                await message.answer(f"⏳ Кулдаун после серии убытков: подождите ещё ~{mins} мин.")
                                            return
                        except Exception:
                            pass
                    return await orig_cmd(message, bot)
                setattr(target,"callback", cmd_signal_guard)
                logger and logger.info("VINO: cmd_signal wrapped with stop‑day/cooldown guards.")
        except Exception:
            logger and logger.warning("VINO: cmd_signal guard wrap failed.")

    # build_reason wrapper: exec hint + Kelly-lite size hint
    orig_reason = app.get("build_reason")
    if callable(orig_reason):
        def _build_reason_vino(details: Dict[str,Any]) -> str:
            txt = ""
            try:
                txt = orig_reason(details) or ""
            except Exception:
                txt = ""
            parts = []
            spreadZ = details.get("spread_norm")
            if isinstance(spreadZ,(int,float)) and spreadZ>0:
                parts.append(f"spreadZ {spreadZ:.2f}")
            qimb = details.get("ob_imb") or details.get("q_imb")
            if isinstance(qimb,(int,float)):
                parts.append(f"L2imb {qimb:+.2f}")
            if parts:
                hint = f"Вход: лимитом у VWAP/IB mid; избегать маркет при {', '.join(parts)}"
                if "Вход:" not in txt and "Entry:" not in txt:
                    txt = (txt + (" • " if txt else "")) + hint
            if VINO_SHOW_SIZE_HINT:
                try:
                    entry = float(details.get("c5") or 0.0)
                    sl    = float(details.get("sl") or 0.0)
                    tps   = details.get("tps") or []
                    if entry>0 and sl>0 and tps:
                        risk = abs(entry - sl)
                        r1 = abs(tps[0] - entry) / (risk + 1e-9)
                        p  = None
                        if isinstance(details.get("p_bayes"), (int,float)):
                            p = float(details["p_bayes"])
                        elif isinstance(details.get("vino_p_news"), (int,float)):
                            p = float(details["vino_p_news"])
                        if p is None:
                            p = 0.55
                        b = max(0.5, min(3.0, r1))
                        q = 1.0 - p
                        f = (b*p - q) / (b + 1e-9)
                        f = max(0.0, min(VINO_SIZE_MAX_RISK_PCT/100.0, f))
                        risk_pct = f*100.0
                        size_hint = f"Риск ≤ {risk_pct:.2f}% депо • плечо ≤ {min(details.get('leverage', VINO_SIZE_MAX_LEV), VINO_SIZE_MAX_LEV)}x"
                        txt = (txt + (" • " if txt else "")) + size_hint
                except Exception:
                    pass
            return txt[:1100]
        app["build_reason"] = _build_reason_vino

    # format_signal_message (скальп-вид)
    if VINO_SCALP_FORMAT:
        orig_fmt = app.get("format_signal_message")
        fmt_price = app.get("format_price") or (lambda x: f"{x:.4f}")
        def _fmt_scalp(sig) -> str:
            base = _base_of_symbol(sig.symbol)
            tps_fmt = " / ".join(fmt_price(x) for x in sig.tps)
            disclaimer = "\n\n⚠️ Не является финансовым советом. Торгуйте ответственно."
            lines = [
                f"{sig.side.title()}  {base}/USDT",
                f"Плечо: {sig.leverage}x",
                f"Вход: {fmt_price(sig.entry)} / по рынку",
                f"Тейки: {tps_fmt}",
                f"Стоп: {fmt_price(sig.sl)}"
            ]
            return "\n".join(lines) + disclaimer
        app["format_signal_message"] = _fmt_scalp

    # on_startup: запускаем калибратор
    orig_on_startup = app.get("on_startup")
    async def _on_startup_vino(bot):
        try:
            asyncio.create_task(dio.vino_calibrator(app, interval_sec=6*3600))
            logger and logger.info("VINO: calibrator started (6h interval).")
        except Exception as e:
            logger and logger.warning("VINO: calibrator start failed: %s", e)
        if orig_on_startup:
            await orig_on_startup(bot)
    app["on_startup"] = _on_startup_vino
