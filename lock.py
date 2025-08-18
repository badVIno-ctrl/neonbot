from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Set
import os
import math
import asyncio
import random
import contextlib
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

# ===================== ВСПОМОГАТЕЛЬНОЕ: безопасная работа с датами =====================

def _safe_ts_to_utc(ts_obj) -> pd.Timestamp:
    try:
        ts = pd.to_datetime(ts_obj)
    except Exception:
        return pd.Timestamp.now(tz=timezone.utc)
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is None:
            return ts.tz_localize(timezone.utc)
        try:
            return ts.tz_convert(timezone.utc)
        except Exception:
            return ts
    return pd.Timestamp(ts, tz=timezone.utc)

def _safe_series_to_utc(s: pd.Series) -> pd.Series:
    s2 = pd.to_datetime(s, errors="coerce")
    try:
        tz = getattr(s2.dt, "tz", None)
        if tz is None:
            return s2.dt.tz_localize(timezone.utc)
        return s2.dt.tz_convert(timezone.utc)
    except Exception:
        return s2

def _day_anchor_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = _safe_ts_to_utc(ts)
    return ts.replace(hour=0, minute=0, second=0, microsecond=0)

# ===================== TA-утилиты (экспортируем в app) =====================

def kama(series: pd.Series, er_period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    x = series.astype(float).values
    n = len(x)
    if n == 0:
        return series.copy()
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
    return pd.Series(out, index=series.index, name="kama")

def ehlers_super_smoother(series: pd.Series, period: int = 20) -> pd.Series:
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
    return pd.Series(y, index=series.index, name="ssmooth")

def donchian_channel(high: pd.Series, low: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    upper = high.rolling(window).max()
    lower = low.rolling(window).min()
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    pc = close.shift(1)
    return pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)

def keltner_channel(high: pd.Series, low: pd.Series, close: pd.Series, ema_period: int = 20, atr_period: int = 20, mult: float = 1.5) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema = close.ewm(span=ema_period, adjust=False).mean()
    atr = _true_range(high, low, close).rolling(atr_period).mean()
    upper = ema + mult * atr
    lower = ema - mult * atr
    return upper, ema, lower

def vix_fix(close: pd.Series, low: pd.Series, n: int = 22) -> pd.Series:
    hh = close.rolling(n).max()
    out = (hh - low) / (hh + 1e-12) * 100.0
    return out.rename("vixfix")

def tsf_slope(series: pd.Series, window: int = 20) -> pd.Series:
    y = series.astype(float).values
    x = np.arange(len(y))
    out = np.full(len(y), np.nan, dtype=float)
    for i in range(window, len(y) + 1):
        yi = y[i - window:i]
        xi = x[i - window:i]
        a, _b = np.polyfit(xi, yi, 1)
        out[i - 1] = a / (abs(yi.mean()) + 1e-12)
    return pd.Series(out, index=series.index, name="tsf_slope")

# ===================== Утренний пост в канал 08:00 =====================

COIN_MAP = [
    ("BTC", "BTC/USDT"),
    ("ETH", "ETH/USDT"),
    ("SOL", "SOL/USDT"),
    ("TON", "TON/USDT"),
    ("BNB", "BNB/USDT"),
]

def _format_usd(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    if x >= 1000:
        s = f"{x:,.0f}$"
    elif x >= 10:
        s = f"{x:,.2f}$"
    else:
        s = f"{x:,.4f}$"
    return s.replace(",", "'")

def _ensure_channel_id(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("https://t.me/"):
        return "@" + raw.split("/")[-1]
    return raw

async def _post_morning_report(app: Dict[str, Any], bot, channel_id: str) -> None:
    market = app["market"]
    now_msk = app["now_msk"]
    logger = app.get("logger")
    dt = now_msk()
    date_str = dt.strftime("%d.%m.%Y")
    time_str = dt.strftime("%H:%M")

    lines = []
    lines.append("☀️ Доброе утро всем!")
    lines.append("━━━━━━━━━━━━━━━━━━")
    lines.append("📊 Ключевые цены:")
    for sym, mkt in COIN_MAP:
        price = None
        with contextlib.suppress(Exception):
            price = market.fetch_mark_price(mkt)
        lines.append(f"• {sym}: {_format_usd(price)}")
    lines.append("")
    lines.append(f"🕗 Время: {time_str} МСК • {date_str}")
    lines.append("🔗 Бот: @neons_crypto_bot")
    lines.append("📣 Канал: https://t.me/NeonFakTrading")
    lines.append("")
    lines.append("#BTC #ETH #SOL #TON #BNB #crypto #Neon #trading")
    lines.append("")
    lines.append("🄽🄴🄾🄽")

    msg = "\n".join(lines)
    try:
        send_retry_html = app.get("send_retry_html")
        if send_retry_html:
            await send_retry_html(bot, channel_id, msg)
        else:
            await bot.send_message(channel_id, msg)
        logger and logger.info("Morning post sent to %s at %s MSK", channel_id, time_str)
    except Exception as e:
        logger and logger.exception("Failed to send morning post: %s", e)

def _seconds_until_next_msk(now_msk_fn, hour: int) -> float:
    now = now_msk_fn()
    target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if now >= target:
        target = target + timedelta(days=1)
    return max(1.0, (target - now).total_seconds())

async def start_daily_channel_post_loop(app: Dict[str, Any], bot, channel_id: Optional[str] = None) -> None:
    logger = app.get("logger")
    ch = _ensure_channel_id(channel_id or os.getenv("TG_CHANNEL") or "@NeonFakTrading")
    if not ch:
        logger and logger.warning("Daily post loop not started: channel_id is not set")
        return
    logger and logger.info("Starting daily channel post loop for %s ...", ch)
    await asyncio.sleep(2)
    while True:
        try:
            wait_s = _seconds_until_next_msk(app["now_msk"], 8)
            await asyncio.sleep(wait_s)
            await _post_morning_report(app, bot, ch)
        except asyncio.CancelledError:
            logger and logger.info("Daily channel post loop cancelled.")
            break
        except Exception as e:
            logger and logger.exception("Daily channel loop error: %s", e)
            await asyncio.sleep(30)

# ===================== Админам 06:00 МСК =====================

ADMIN_JOKES = [
    "— Пап, а что такое трейдинг?\n— Это когда ты уверен, что всё будет расти… и падает.\n— А если уверен, что будет падать?\n— Тогда растёт.",
    "У каждого трейдера есть план. До первой сделки.",
    "Купил на хаях, продал на лоях — зато опыт бесценный.",
    "Самая точная стратегия: купить, когда страшно. Продать, когда страшнее.",
    "— Как дела у твоей стратегии?\n— Как у биткоина: то на Луне, то в подземелье.",
    "— Почему ты не спишь?\n— У меня позиция в открытом космосе.",
    "Трейдинг — это как спорт: тяжело выигрывать, но легко проигрывать.",
    "Если стоп близко — заберут. Если далеко — дойдут.",
    "Лучшая кнопка на бирже — «Выйти и подумать».",
    "Рынок — отличный психолог. Быстро исправляет завышенную самооценку.",
]

ADMIN_WISHES = [
    "Пусть сегодня будет прибыльным! 🚀",
    "Плавных трендов и чётких пробоев! 📈",
    "Холодной головы и горячих сетапов! 💡",
    "Пусть риск-менеджмент будет на вашей стороне! 🛡️",
    "Больше TP, меньше SL! 🎯",
]

CITY_PENDING: set[int] = set()

async def _ensure_city_column(db) -> None:
    try:
        await db.conn.execute("ALTER TABLE users ADD COLUMN city TEXT")
        await db.conn.commit()
    except Exception:
        pass

async def _get_city(db, user_id: int) -> str:
    try:
        cur = await db.conn.execute("SELECT city FROM users WHERE user_id=?", (user_id,))
        row = await cur.fetchone()
        if row and row["city"]:
            return str(row["city"]).strip()
    except Exception:
        pass
    return "Москва"

async def _set_city(db, user_id: int, city: str) -> None:
    try:
        await db.conn.execute("UPDATE users SET city=? WHERE user_id=?", (city.strip(), user_id))
        await db.conn.commit()
    except Exception:
        pass

def _wttr_line(city: str) -> str:
    try:
        url = f"https://wttr.in/{city}?format=3&m&lang=ru"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            return r.text.strip()
    except Exception:
        pass
    return f"Погода: {city} — недоступна сейчас"

async def _post_admin_greetings(app: Dict[str, Any], bot) -> None:
    logger = app.get("logger")
    db = app.get("db")
    market = app.get("market")
    now_msk = app["now_msk"]
    if not (db and db.conn):
        return
    try:
        admin_ids = await db.get_admin_user_ids()
    except Exception:
        admin_ids = []
    if not admin_ids:
        return

    prices = []
    for sym, mkt in COIN_MAP:
        p = None
        with contextlib.suppress(Exception):
            p = market.fetch_mark_price(mkt)
        prices.append(f"• {sym}: {_format_usd(p)}")

    for uid in admin_ids:
        try:
            city = await _get_city(db, uid)
            weather = _wttr_line(city)
            joke = random.choice(ADMIN_JOKES)
            wish = random.choice(ADMIN_WISHES)
            dt = now_msk()
            date_str = dt.strftime("%d.%m.%Y")
            time_str = dt.strftime("%H:%M")

            lines = []
            lines.append("☕ Доброе утро, админ!")
            lines.append("━━━━━━━━━━━━━━━━━━")
            lines.append(weather)
            lines.append("")
            lines.append("📊 Ключевые цены:")
            lines.extend(prices)
            lines.append("")
            lines.append(f"🕕 Время: {time_str} МСК • {date_str}")
            lines.append(f"✨ {wish}")
            lines.append("")
            lines.append("Анекдот дня:")
            lines.append(joke)

            msg = "\n".join(lines)
            send_retry_html = app.get("send_retry_html")
            with contextlib.suppress(Exception):
                if send_retry_html:
                    await send_retry_html(bot, uid, msg)
                else:
                    await bot.send_message(uid, msg)
        except Exception as e:
            logger and logger.warning("Admin greet send fail to %s: %s", uid, e)

async def start_daily_admin_greetings_loop(app: Dict[str, Any], bot) -> None:
    logger = app.get("logger")
    await asyncio.sleep(2)
    logger and logger.info("Starting daily admin greetings loop...")
    while True:
        try:
            wait_s = _seconds_until_next_msk(app["now_msk"], 6)
            await asyncio.sleep(wait_s)
            await _post_admin_greetings(app, bot)
        except asyncio.CancelledError:
            logger and logger.info("Admin greetings loop cancelled.")
            break
        except Exception as e:
            logger and logger.exception("Admin greetings loop error: %s", e)
            await asyncio.sleep(30)

# ===================== Анализ монеты (/coin) и выдача сигнала =====================

COIN_PENDING: set[int] = set()

def _pretty_analysis_header(base: str, side: str, score: Optional[float], p_bayes: Optional[float]) -> str:
    conf_txt = ""
    if p_bayes is not None:
        conf_txt = f" • уверенность {p_bayes:.2f}"
    elif isinstance(score, (int, float)):
        conf_txt = f" • оценка {score:.2f}"
    return f"🔎 Анализ {base} • направление {side}{conf_txt}"

def _format_analysis_text(build_reason_fn, fmt_price, details: Dict[str, Any], base: str, side: str, side_score: Optional[float]) -> str:
    try:
        reason = build_reason_fn(details) if build_reason_fn else ""
    except Exception:
        reason = ""
    p_bayes = details.get("p_bayes")
    header = _pretty_analysis_header(base, side, side_score, p_bayes)
    tail = "\n\nℹ️ Это аналитический обзор, не торговый сигнал."
    return f"{header}\n{reason}{tail}"

def _norm_token(txt: str) -> str:
    t = "".join(ch for ch in (txt or "") if ch.isalnum() or ch in "/").upper().strip()
    if t.endswith("USDT"):
        t = t[:-4]
    if "/" in t:
        t = t.split("/")[0]
    return t

def _resolve_symbol_any(app: Dict[str, Any], raw: str) -> Optional[str]:
    market = app.get("market")
    token = _norm_token(raw)
    if not token or len(token) < 2:
        return None
    candidate = f"{token}/USDT"
    try:
        for name, ex in market._available_exchanges():
            resolved = market.resolve_symbol(ex, candidate) or None
            if resolved and resolved in ex.markets:
                return candidate
    except Exception:
        pass
    with contextlib.suppress(Exception):
        rsfq = app.get("resolve_symbol_from_query")
        sym = rsfq(raw) if rsfq else None
        if sym:
            return sym
    return None

async def _do_coin_analysis_flow(app: Dict[str, Any], message, bot, user_id: int, symbol: str) -> None:
    logger = app.get("logger")
    db = app.get("db")
    EXECUTOR = app.get("EXECUTOR")
    build_reason_fn = app.get("build_reason")
    fmt_price = app.get("format_price") or (lambda x: f"{x:.4f}")
    Signal = app.get("Signal")
    format_signal_message = app.get("format_signal_message")
    now_msk = app["now_msk"]
    DAILY_LIMIT = app.get("DAILY_LIMIT", 3)
    watch_signal_price = app.get("watch_signal_price")
    send_retry_html = app.get("send_retry_html")
    MET_SIGNALS_GEN = app.get("MET_SIGNALS_GEN", None)

    st = await db.get_user_state(user_id)
    base = symbol.split("/")[0]
    with contextlib.suppress(Exception):
        if send_retry_html:
            await send_retry_html(bot, message.chat.id, f"⏳ Делаю анализ {base}…")
        else:
            await message.answer(f"⏳ Делаю анализ {base}…")

    loop = asyncio.get_running_loop()
    side = None
    side_score = None
    details = None

    try:
        score_symbol_core = app.get("score_symbol_core")
        res = await loop.run_in_executor(EXECUTOR, score_symbol_core, symbol, False)
        if res is None:
            res = await loop.run_in_executor(EXECUTOR, score_symbol_core, symbol, True)
        if res is not None:
            side_score, side, details = res
            details = dict(details or {})
        else:
            score_symbol_quick = app.get("score_symbol_quick")
            quick = score_symbol_quick(symbol)
            if quick:
                side, details = quick
                side_score = float(details.get("score", 0.9))
                details = dict(details or {})
            else:
                raise RuntimeError("Нет данных для анализа.")
    except Exception as e:
        logger and logger.warning("Coin analyze failed for %s: %s", symbol, e)
        with contextlib.suppress(Exception):
            if send_retry_html:
                await send_retry_html(bot, message.chat.id, "⚠️ Не удалось выполнить анализ. Попробуйте позже.")
            else:
                await message.answer("⚠️ Не удалось выполнить анализ. Попробуйте позже.")
        return

    details["symbol"] = symbol

    # Лимит: если исчерпан и не админ — только анализ
    is_admin = bool(st.get("admin"))
    unlimited = bool(st.get("unlimited"))
    if (not unlimited) and (not is_admin) and (st.get("count", 0) >= DAILY_LIMIT):
        txt = _format_analysis_text(build_reason_fn, fmt_price, details, base, side, side_score)
        await message.answer(txt)
        return

    # Анти-дубль (у этого пользователя)
    existing = await db.get_active_signals_for_user(user_id)
    if any(s.symbol == symbol and s.side == side and s.active for s in existing):
        txt = _format_analysis_text(build_reason_fn, fmt_price, details, base, side, side_score)
        txt += "\n\n⚠️ У вас уже есть активный сигнал по этой паре и стороне — дубликат не выдам."
        await message.answer(txt)
        return

    # Подготовка сигнала (полноценный сигнал с мониторингом)
    entry = float(details["c5"])
    sl = float(details["sl"])
    tps = [float(x) for x in (details.get("tps") or [])]
    leverage = int(details.get("leverage", 5))
    risk_level = int(details.get("risk_level", 5))
    news_note = details.get("news_note", "")
    atr_value = float(details.get("atr", 0.0))
    watch_seconds = int(details.get("watch_seconds", 4 * 3600))
    reason = build_reason_fn(details) if build_reason_fn else ""

    if side == "LONG":
        if not (all(tp > entry for tp in tps) and sl < entry):
            txt = _format_analysis_text(build_reason_fn, fmt_price, details, base, side, side_score)
            txt += "\n\n⚠️ Сигнал не прошёл валидацию."
            await message.answer(txt)
            return
    else:
        if not (all(tp < entry for tp in tps) and sl > entry):
            txt = _format_analysis_text(build_reason_fn, fmt_price, details, base, side, side_score)
            txt += "\n\n⚠️ Сигнал не прошёл валидацию."
            await message.answer(txt)
            return

    sig = Signal(
        user_id=user_id,
        symbol=symbol,
        side=side,
        entry=entry,
        tps=tps,
        sl=sl,
        leverage=leverage,
        risk_level=risk_level,
        created_at=now_msk(),
        news_note=news_note,
        atr_value=atr_value,
        watch_until=now_msk() + timedelta(seconds=watch_seconds),
        reason=reason,
    )
    text = format_signal_message(sig)
    await message.answer(text)

    st["count"] = st.get("count", 0) + 1
    await db.save_user_state(user_id, st)

    sig.id = await app["db"].add_signal(sig)
    task = asyncio.create_task(watch_signal_price(bot, message.chat.id, sig))
    active_watch_tasks = app.get("active_watch_tasks", {})
    active_watch_tasks.setdefault(user_id, []).append(task)
    if MET_SIGNALS_GEN:
        MET_SIGNALS_GEN.inc()

# ===================== Патч UI/Handlers/Reason + TZ FIXES =====================

# Новые TA-функции для улучшений

def _session_bounds_utc(now_utc: datetime) -> Dict[str, Tuple[datetime, datetime]]:
    day_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    return {
        "ASIA": (day_start + timedelta(hours=0), day_start + timedelta(hours=8)),
        "EU": (day_start + timedelta(hours=7), day_start + timedelta(hours=13)),
        "US": (day_start + timedelta(hours=12), day_start + timedelta(hours=20)),
    }

def _active_session(now_utc: datetime) -> str:
    h = now_utc.hour
    if 0 <= h < 8: return "ASIA"
    if 7 <= h < 13: return "EU"
    if 12 <= h < 20: return "US"
    return "ASIA"  # fallback

def _session_mask(df: pd.DataFrame, session: str, bounds: Dict[str, Tuple[datetime, datetime]]) -> pd.Series:
    ts = _safe_series_to_utc(df["ts"])
    start, end = bounds[session]
    return (ts >= start) & (ts < end)

def _session_vwap(df: pd.DataFrame, session: str, bounds: Dict[str, Tuple[datetime, datetime]]) -> Optional[float]:
    mask = _session_mask(df, session, bounds)
    if not mask.any(): return None
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (tp[mask] * df["volume"][mask]).sum()
    vv = df["volume"][mask].sum() + 1e-9
    return pv / vv

def _initial_balance_levels_session(df: pd.DataFrame, session: str, bounds: Dict[str, Tuple[datetime, datetime]], ib_minutes: int = 60) -> Tuple[Optional[float], Optional[float]]:
    mask = _session_mask(df, session, bounds)
    if not mask.any(): return None, None
    df_sess = df[mask]
    bars_needed = max(1, int(ib_minutes / 15))  # assuming 15m tf
    df_ib = df_sess.head(bars_needed)
    if df_ib.empty: return None, None
    return float(df_ib["high"].max()), float(df_ib["low"].min())

def _session_breakout(df: pd.DataFrame, session: str, bounds: Dict[str, Tuple[datetime, datetime]], close: float, ib_hi: float, ib_lo: float) -> int:
    mask = _session_mask(df, session, bounds)
    if not mask.any(): return 0
    df_sess = df[mask]
    last_close = df_sess["close"].iloc[-1]
    if last_close > ib_hi: return 1  # breakout up
    if last_close < ib_lo: return -1  # breakout down
    return 0  # inside

def _bbwp(series: pd.Series, length: int = 20, lookback: int = 96) -> float:
    basis = series.rolling(length).mean()
    dev = series.rolling(length).std()
    bbw = (series - basis + 2 * dev) / (4 * dev + 1e-12)
    bbwp = bbw.rolling(lookback).rank(pct=True) * 100
    return bbwp.iloc[-1]

def _daily_pivots(df1d: pd.DataFrame) -> Dict[str, float]:
    if df1d.empty or len(df1d) < 2: return {}
    prev = df1d.iloc[-2]
    P = (prev["high"] + prev["low"] + prev["close"]) / 3
    R1 = 2 * P - prev["low"]
    S1 = 2 * P - prev["high"]
    R2 = P + (prev["high"] - prev["low"])
    S2 = P - (prev["high"] - prev["low"])
    return {"P": P, "R1": R1, "R2": R2, "S1": S1, "S2": S2}

def _near_round_level(price: float, step: float = 0.25) -> Tuple[float, float]:
    base = math.floor(price / step) * step
    dist = price - base
    return (base if dist < step / 2 else base + step), dist / step

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - 100 / (1 + rs)

def _rsi_map(df5: pd.DataFrame, df15: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame) -> Dict[str, float]:
    return {
        "5m": _rsi(df5["close"], 14).iloc[-1] if df5 is not None and not df5.empty else 50.0,
        "15m": _rsi(df15["close"], 14).iloc[-1] if df15 is not None and not df15.empty else 50.0,
        "1h": _rsi(df1h["close"], 14).iloc[-1] if df1h is not None and not df1h.empty else 50.0,
        "4h": _rsi(df4h["close"], 14).iloc[-1] if df4h is not None and not df4h.empty else 50.0,
    }

def _rsi_consensus(rsi_map: Dict[str, float], side: str) -> int:
    bull_count = sum(1 for v in rsi_map.values() if v > 55)
    bear_count = sum(1 for v in rsi_map.values() if v < 45)
    if side == "LONG": return bull_count - bear_count
    return bear_count - bull_count

def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = (df["open"].shift(1) + df["close"].shift(1)) / 2
    ha_high = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame({"open": ha_open, "high": ha_high, "low": ha_low, "close": ha_close})

def _ha_run_exhaustion(ha: pd.DataFrame) -> Tuple[int, bool]:
    direction = np.sign(ha["close"] - ha["open"])
    run = 0
    prev_d = None
    for d in direction.iloc[::-1]:
        if d == 0:
            break
        if prev_d is None or d == prev_d:
            run += 1
        else:
            break
        prev_d = d
    signed_run = 0 if prev_d is None else int(run * prev_d)
    last_range = float(ha["high"].iloc[-1] - ha["low"].iloc[-1])
    body = float(abs(ha["close"].iloc[-1] - ha["open"].iloc[-1]))
    exhaustion = (last_range > 0) and (body < 0.3 * last_range)
    return signed_run, exhaustion

def _ema_ribbon(df: pd.DataFrame, periods: List[int] = [8,13,21,34]) -> float:
    emas = [df["close"].ewm(span=p, adjust=False).mean() for p in periods]
    r = pd.concat(emas, axis=1)
    width = (r.max(axis=1) - r.min(axis=1)).iloc[-1]
    return float(width)  # absolute width in price units

def _ribbon_state(ribbon_w: float, atr: float) -> str:
    if ribbon_w < 0.5 * atr: return "compressed"
    if ribbon_w > 1.5 * atr: return "expanded"
    return "neutral"

def _swing_points(df: pd.DataFrame, lookback: int = 5) -> Tuple[float, float]:
    hi = df["high"].rolling(lookback).max().shift(1)
    lo = df["low"].rolling(lookback).min().shift(1)
    return float(hi.iloc[-1]), float(lo.iloc[-1])

def _fib_levels(high: float, low: float) -> Dict[float, float]:
    diff = high - low
    return {
        0.382: high - 0.382 * diff,
        0.5: high - 0.5 * diff,
        0.618: high - 0.618 * diff,
        1.272: high + 0.272 * diff,
        1.618: high + 0.618 * diff,
    }

def _day_type(df15: pd.DataFrame, ib_hi: float, ib_lo: float) -> str:
    close = df15["close"].iloc[-1]
    open_day = df15["open"].iloc[0]
    if close > ib_hi and open_day < ib_hi: return "Open-Drive Up"
    if close < ib_lo and open_day > ib_lo: return "Open-Drive Down"
    if abs(close - open_day) < (ib_hi - ib_lo): return "Balanced"
    return "ORR"  # Open Range Reversal

# Env-флаги и веса (примерные дефолты)
TA_SESSION = os.getenv("TA_SESSION", "1") == "1"
TA_BBWP = os.getenv("TA_BBWP", "1") == "1"
TA_PIVOTS = os.getenv("TA_PIVOTS", "1") == "1"
TA_ROUND = os.getenv("TA_ROUND", "1") == "1"
TA_RSI_STACK = os.getenv("TA_RSI_STACK", "1") == "1"
TA_HA = os.getenv("TA_HA", "1") == "1"
TA_RIBBON = os.getenv("TA_RIBBON", "1") == "1"
TA_FIB = os.getenv("TA_FIB", "1") == "1"
TA_DAYTYPE = os.getenv("TA_DAYTYPE", "1") == "1"
TA_SMART_WATCH = os.getenv("TA_SMART_WATCH", "1") == "1"

W_SESSION = float(os.getenv("W_SESSION", "0.15"))
W_BBWP = float(os.getenv("W_BBWP", "0.10"))
W_PIVOTS = float(os.getenv("W_PIVOTS", "0.12"))
W_ROUND = float(os.getenv("W_ROUND", "0.08"))
W_RSI = float(os.getenv("W_RSI", "0.10"))
W_HA = float(os.getenv("W_HA", "0.09"))
W_RIBBON = float(os.getenv("W_RIBBON", "0.11"))
W_FIB = float(os.getenv("W_FIB", "0.13"))
W_DAYTYPE = float(os.getenv("W_DAYTYPE", "0.14"))

def patch(app: Dict[str, Any]) -> None:
    """
    - RU‑reason 6–10 строк
    - Кнопка «🔎 Анализ монеты», /coin (с лимитами и анти‑дублем)
    - /code → спросить город; /mycity
    - Патч /signal: не выдаём один и тот же сигнал одному пользователю повторно
    - Утренние рассылки: канал 08:00; админам 06:00 (wttr.in + анекдот + цены)
    - TZ‑фиксы (anchored_vwap, week_anchor_from_df, resample_ohlcv, IB/KeyLevels/CME/Season)
    - TZ‑safe патч VWAP‑бэндов для main (устраняет падение /coin)
    """
    logger = app.get("logger")
    router = app.get("router")

    # Экспорт утилит / утренний пост
    app["kama"] = kama
    app["ehlers_super_smoother"] = ehlers_super_smoother
    app["donchian_channel"] = donchian_channel
    app["keltner_channel"] = keltner_channel
    app["vix_fix"] = vix_fix
    app["tsf_slope"] = tsf_slope
    app["start_daily_channel_post_loop"] = start_daily_channel_post_loop
    app["start_daily_admin_greetings_loop"] = start_daily_admin_greetings_loop

    # ---------- TZ-safe overrides ----------
    def _anchored_vwap_safe(df: pd.DataFrame, anchor: Optional[datetime] = None) -> pd.Series:
        ts_series = pd.to_datetime(df["ts"], errors="coerce")
        try:
            if ts_series.dt.tz is None:
                ts_series = ts_series.dt.tz_localize(timezone.utc)
            else:
                ts_series = ts_series.dt.tz_convert(timezone.utc)
        except Exception:
            pass
        if anchor is None:
            last_ts = ts_series.iloc[-1]
            anchor_ts = last_ts.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            a = pd.to_datetime(anchor)
            if isinstance(a, pd.Timestamp):
                anchor_ts = a.tz_localize(timezone.utc) if a.tzinfo is None else a.tz_convert(timezone.utc)
            else:
                anchor_ts = pd.Timestamp(a, tz=timezone.utc)
        mask = ts_series >= anchor_ts
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        pv = (tp[mask] * df.loc[mask, "volume"]).cumsum()
        vv = df.loc[mask, "volume"].cumsum() + 1e-9
        vw = pv / vv
        out = pd.Series(index=df.index, dtype=float)
        out.loc[mask] = vw
        out.ffill(inplace=True)
        return out

    def _week_anchor_from_df_safe(df: pd.DataFrame) -> datetime:
        ts_series = pd.to_datetime(df["ts"], errors="coerce")
        try:
            if ts_series.dt.tz is None:
                ts_series = ts_series.dt.tz_localize(timezone.utc)
            else:
                ts_series = ts_series.dt.tz_convert(timezone.utc)
        except Exception:
            pass
        last_ts = ts_series.iloc[-1]
        start = (last_ts - timedelta(days=int(getattr(last_ts, "dayofweek", last_ts.weekday())))).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return start.to_pydatetime()

    drop_incomplete = app.get("drop_incomplete")
    def _resample_ohlcv_safe(df: Optional[pd.DataFrame], freq: str) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        x = df.copy()
        ts = _safe_series_to_utc(x["ts"])
        x = x.set_index(ts).sort_index()
        o = x["open"].resample(freq).first()
        h = x["high"].resample(freq).max()
        l = x["low"].resample(freq).min()
        c = x["close"].resample(freq).last()
        v = x["volume"].resample(freq).sum()
        out = pd.concat([o, h, l, c, v], axis=1)
        out.columns = ["open", "high", "low", "close", "volume"]
        out = out.dropna()
        out.index.name = "ts"
        out = out.reset_index()
        out["ts"] = _safe_series_to_utc(out["ts"])
        tf = "1d" if freq.upper() in ("1D", "D") else ("1w" if freq.upper() in ("1W", "W") else freq.lower())
        return drop_incomplete(out[["ts", "open", "high", "low", "close", "volume"]], tf) if drop_incomplete else out

    app["anchored_vwap"] = _anchored_vwap_safe
    app["week_anchor_from_df"] = _week_anchor_from_df_safe
    app["resample_ohlcv"] = _resample_ohlcv_safe
    logger and logger.info("TZ-safe overrides applied: anchored_vwap + week_anchor_from_df + resample_ohlcv.")

    # Ещё TZ‑safe на функции из main.py + VWAP bands fix
    try:
        m = sys.modules.get("main")
        if m:
            def _initial_balance_levels_safe(df15: pd.DataFrame, ib_hours: int = 1) -> Tuple[Optional[float], Optional[float]]:
                if df15 is None or len(df15) < 5:
                    return None, None
                ts_last = _safe_ts_to_utc(df15["ts"].iloc[-1])
                day_anchor = _day_anchor_utc(ts_last)
                mask = df15["ts"] >= day_anchor
                df_day = df15.loc[mask]
                if df_day.empty:
                    return None, None
                bars_needed = max(1, int(ib_hours * 60 / 15))
                df_ib = df_day.head(bars_needed)
                if df_ib.empty:
                    return None, None
                return float(df_ib["high"].max()), float(df_ib["low"].min())

            def _key_levels_safe(df15: pd.DataFrame, df1h: Optional[pd.DataFrame], df1d: Optional[pd.DataFrame]) -> Dict[str, float]:
                out: Dict[str, float] = {}
                try:
                    if df1d is not None and len(df1d) >= 2:
                        out["PDH"] = float(df1d["high"].iloc[-2])
                        out["PDL"] = float(df1d["low"].iloc[-2])
                        out["PDC"] = float(df1d["close"].iloc[-2])
                        out["PDO"] = float(df1d["open"].iloc[-1])
                    if df15 is not None and len(df15) >= 10:
                        anchor = _day_anchor_utc(df15["ts"].iloc[-1])
                        cur = df15[df15["ts"] >= anchor]
                        if not cur.empty:
                            out["SDH"] = float(cur["high"].max())
                            out["SDL"] = float(cur["low"].min())
                except Exception:
                    pass
                return out

            def _cme_gap_approx_safe(df1h: pd.DataFrame) -> Optional[Tuple[float, float]]:
                try:
                    if df1h is None or len(df1h) < 400:
                        return None
                    x = df1h.copy()
                    x["ts"] = _safe_series_to_utc(x["ts"])
                    x["dow"] = x["ts"].dt.dayofweek; x["hour"] = x["ts"].dt.hour
                    fri = x[(x["dow"] == 4) & (x["hour"] == 21)]
                    mon = x[(x["dow"] == 0) & (x["hour"] == 0)]
                    if fri.empty or mon.empty:
                        return None
                    last_fri = float(fri["close"].iloc[-1]); first_mon = float(mon["open"].iloc[-1])
                    gap = first_mon - last_fri
                    return float(gap), float(first_mon)
                except Exception:
                    return None

            def _seasonality_score_safe(ts_obj) -> float:
                try:
                    ts = _safe_ts_to_utc(ts_obj)
                except Exception:
                    return 0.0
                h = ts.hour; dow = ts.dayofweek
                s = 0.0
                if h in (13, 14, 15): s += 0.05
                if dow in (0, 2): s += 0.05
                return float(s)

            def _vwap_sigma_bands_safe(df15: pd.DataFrame, lookback: int = 96, anchored_vwap=None) -> Tuple[Optional[float], Optional[float], Optional[float]]:
                f = app.get("anchored_vwap")
                if df15 is None or df15.empty or f is None:
                    return None, None, None
                vwap_series = f(df15)
                if vwap_series is None or len(vwap_series) < lookback + 2:
                    return None, None, None
                close = df15["close"].astype(float)
                dist = close - vwap_series
                sigma = float(dist.tail(lookback).std(ddof=0) + 1e-12)
                v = float(vwap_series.iloc[-1])
                return v, v + sigma, v - sigma

            setattr(m, "_initial_balance_levels", _initial_balance_levels_safe)
            setattr(m, "_key_levels", _key_levels_safe)
            setattr(m, "_cme_gap_approx", _cme_gap_approx_safe)
            setattr(m, "_seasonality_score", _seasonality_score_safe)
            setattr(m, "_vwap_sigma_bands", _vwap_sigma_bands_safe)

            logger and logger.info("TZ-safe overrides applied to main.py functions (IB/Key/CME/Season) + VWAP bands fixed.")
    except Exception as e:
        logger and logger.warning("TZ override failed: %s", e)

    # ---------- Оригиналы ----------
    orig_on_startup = app.get("on_startup")
    orig_build_reason = app.get("build_reason")
    orig_main_menu_kb = app.get("main_menu_kb")
    fmt_price = app.get("format_price") or (lambda x: f"{x:.4f}")

    # ---------- Улучшенная «Причина» (6–10 строк, RU) ----------
    def _ru_build_reason(details: Dict[str, Any]) -> str:
        try:
            side = details.get("side", "")
            entry = float(details.get("c5", 0.0) or 0.0)
            sl = float(details.get("sl", 0.0) or 0.0)
            atr = float(details.get("atr", 0.0) or 0.0)
            atr_pct = float(details.get("atr_pct", 0.0) or 0.0)
            tps = [float(x) for x in (details.get("tps") or [])]
            risk_level = details.get("risk_level")
            lev = details.get("leverage")

            t4h = "4h выше EMA200" if details.get("cond4h_up") else "4h ниже EMA200"
            t1h = "1h выше EMA200" if details.get("cond1h_up") else "1h ниже EMA200"
            t15 = "15m EMA50 выше EMA200" if details.get("cond15_up") else "15m EMA50 ниже EMA200"

            rsi5 = details.get("rsi5"); macdh5 = details.get("macdh5")
            mom = f"RSI(5m) {rsi5:.0f}, MACD {'+' if (macdh5 or 0) > 0 else '-'}" if rsi5 is not None else f"MACD {'+' if (macdh5 or 0) > 0 else '-'}"

            adx15 = details.get("adx15"); r2 = details.get("r2_1h")
            st_dir = details.get("st_dir"); st_txt = "ST ↑" if st_dir and st_dir > 0 else "ST ↓"
            bos_dir = details.get("bos_dir"); bos_txt = "BOS↑" if bos_dir == 1 else ("BOS↓" if bos_dir == -1 else "")
            bos_txt = bos_txt + ("+retest" if details.get("bos_retest") else "")
            don = "Donchian↑" if (details.get("don_break_long") and side == "LONG") else ("Donchian↓" if (details.get("don_break_short") and side == "SHORT") else "")
            vwap_conf = "у VWAP" if details.get("vwap_conf") else ""
            vol_z = details.get("vol_z")

            cvd = details.get("cvd_slope"); ob_imb = details.get("ob_imb")
            rvol = details.get("rvol_combo"); basis = details.get("basis")
            dom_tr = details.get("btc_dom_trend")
            nb = details.get("news_boost")

            risk_pct = abs(entry - sl) / (entry + 1e-9) * 100.0 if entry and sl else 0.0
            def _tp_pct(tp: float) -> float:
                return ((tp - entry) / entry * 100.0) if side == "LONG" else ((entry - tp) / entry * 100.0)
            tp_txt = ""
            if len(tps) >= 3:
                tp_txt = f"TP1 {_tp_pct(tps[0]):.2f}% • TP2 {_tp_pct(tps[1]):.2f}% • TP3 {_tp_pct(tps[2]):.2f}%"

            br = details.get("score_breakdown", {})
            topf = ""
            if isinstance(br, dict) and br:
                top3 = sorted(br.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
                topf = ", ".join([f"{k}: {v:+.2f}" for k, v in top3])

            # Новые поля для reason
            session = details.get("session", "N/A")
            ib_hi = details.get("ib_hi"); ib_lo = details.get("ib_lo")
            ib_txt = f"Сессия: {fmt_price(ib_hi)}/{fmt_price(ib_lo)}" if ib_hi and ib_lo else ""
            bbwp = details.get("bbwp", 50); bbwp_txt = f"BBWP {bbwp:.0f}%"
            pivots = details.get("pivots", {}); piv_txt = f"P {fmt_price(pivots.get('P'))}, R1 {fmt_price(pivots.get('R1'))}, S1 {fmt_price(pivots.get('S1'))}" if pivots else ""
            round_level = details.get("round_level"); round_txt = f"Round: {fmt_price(round_level)}" if round_level else ""
            rsi_map = details.get("rsi_map", {}); rsi_cons = _rsi_consensus(rsi_map, side); rsi_txt = f"RSI stack: {rsi_cons}/4"
            ha_run = details.get("ha_run", 0); ha_txt = f"HA run: {ha_run}"
            day_type = details.get("day_type", "N/A"); day_txt = f"Day: {day_type}"
            ribbon_state = details.get("ribbon_state", "neutral"); rib_txt = f"Ribbon: {ribbon_state}"

            lines = []
            lines.append(f"Сторона: {side} • Риск-L {risk_level}/10 • Плечо {lev}x")
            lines.append(f"Тренд MTF: {t4h}, {t1h}, {t15}")
            lines.append(f"Моментум: {mom}")
            lines.append(f"Режим: ADX {adx15:.0f} • R2(1h) {r2:.2f} • ATR(15m)≈{atr:.4f} ({atr_pct:.2f}%)")
            setup_bits = " • ".join([x for x in [st_txt, bos_txt, don, vwap_conf] if x])
            if setup_bits:
                lines.append(f"Сетапы: {setup_bits}")
            lines.append(f"Управление риском: SL {fmt_price(sl)} ({risk_pct:.2f}% от входа) • {tp_txt}")
            micro_bits = []
            if vol_z is not None: micro_bits.append(f"vol_z {vol_z:.2f}")
            if rvol is not None: micro_bits.append(f"RVOL x{rvol:.2f}")
            if cvd is not None: micro_bits.append(f"CVD {cvd:+.0f}")
            if ob_imb is not None: micro_bits.append(f"L2-imb {ob_imb:+.2f}")
            if basis is not None: micro_bits.append(f"basis {basis*100:+.2f}%")
            if dom_tr: micro_bits.append(f"BTC.D {dom_tr}")
            if micro_bits:
                lines.append("Потоки/объём: " + ", ".join(micro_bits))
            if nb is not None:
                lines.append(f"Новости: +{nb:.2f}")
            if topf:
                lines.append("Топ-факторы: " + topf)
            lines.append("Окно сделки: ~40м–6ч")
            # Добавляем новые
            if ib_txt: lines.append(ib_txt)
            lines.append(bbwp_txt)
            if piv_txt: lines.append(piv_txt)
            if round_txt: lines.append(round_txt)
            lines.append(rsi_txt)
            lines.append(ha_txt)
            lines.append(rib_txt)
            lines.append(day_txt)
            return "\n".join(lines[:10])
        except Exception:
            try:
                return orig_build_reason(details) if orig_build_reason else ""
            except Exception:
                return ""

    app["build_reason"] = _ru_build_reason

    # ---------- Меню: кнопка «🔎 Анализ монеты» ----------
    def _patched_main_menu_kb(is_admin: bool = False):
        try:
            from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
            kb = [
                [KeyboardButton(text="📈 Получить сигнал")],
                [KeyboardButton(text="🔎 Анализ монеты")],
                [KeyboardButton(text="ℹ️ Помощь")],
                [KeyboardButton(text="Поддержка")],
            ]
            return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)
        except Exception:
            return app.get("main_menu_kb", lambda *_: None)(is_admin)

    app["main_menu_kb"] = _patched_main_menu_kb

    # ---------- Патч /code: спросить город ----------
    def _patch_cmd_code():
        try:
            obs = router.message
            handlers = getattr(obs, "handlers", [])
            target = None
            for h in handlers:
                cb = getattr(h, "callback", None)
                if getattr(cb, "__name__", "") == "cmd_code":
                    target = h; break
            if not target:
                logger and logger.warning("cmd_code handler not found to patch city ask.")
                return
            orig_cmd_code = target.callback
            ADMIN_ACCESS_CODE = app.get("ADMIN_ACCESS_CODE", "2604")

            async def cmd_code_patched(message, command, bot):
                await orig_cmd_code(message, command, bot)
                args = (getattr(command, "args", None) or "").strip()
                if args == ADMIN_ACCESS_CODE:
                    CITY_PENDING.add(message.from_user.id)
                    with contextlib.suppress(Exception):
                        await message.answer("В каком городе вы живёте? Напишите одним сообщением (пример: Москва).")

            setattr(target, "callback", cmd_code_patched)
            logger and logger.info("cmd_code patched to ask city after activation.")
        except Exception as e:
            logger and logger.warning("cmd_code patch error: %s", e)

    # ---------- Патч /signal: анти‑дубль ----------
    def _patch_cmd_signal():
        try:
            db = app.get("db")
            rank_symbols_async = app.get("rank_symbols_async")
            score_symbol_quick = app.get("score_symbol_quick")
            guard_access = app.get("guard_access")
            format_signal_message = app.get("format_signal_message")
            edit_retry_html = app.get("edit_retry_html")
            now_msk = app["now_msk"]
            Signal = app.get("Signal")
            MET_SIGNALS_GEN = app.get("MET_SIGNALS_GEN", None)
            active_watch_tasks = app.get("active_watch_tasks", {})
            SYMBOLS = app.get("SYMBOLS", [])

            obs = router.message
            handlers = getattr(obs, "handlers", [])
            target = None
            for h in handlers:
                cb = getattr(h, "callback", None)
                if getattr(cb, "__name__", "") == "cmd_signal":
                    target = h; break
            if not target:
                logger and logger.warning("cmd_signal handler not found to patch.")
                return
            orig = target.callback

            async def cmd_signal_patched(message, bot):
                user_id = message.from_user.id
                st = await guard_access(message, bot)
                if not st: return
                working_msg = await message.answer("🔍 Ищу торговую пару...")
                try:
                    ranked = await rank_symbols_async(SYMBOLS)
                    if not ranked:
                        await edit_retry_html(working_msg, "Не удалось найти подходящий сигнал. Попробуйте позже.")
                        return
                    existing = await db.get_active_signals_for_user(user_id)
                    def is_dup(sym: str, side: str) -> bool:
                        return any(s.symbol == sym and s.side == side and s.active for s in existing)
                    picked = None
                    for sym, det in ranked:
                        if not is_dup(sym, det["side"]):
                            picked = (sym, det); break
                    if picked is None:
                        best_q = None
                        for sym in SYMBOLS:
                            q = score_symbol_quick(sym)
                            if not q: continue
                            side, det = q
                            if is_dup(sym, side): continue
                            det["side"] = side
                            if (best_q is None) or (float(det.get("score", 0.0)) > float(best_q[1].get("score", 0.0))):
                                best_q = (sym, det)
                        if best_q: picked = best_q
                    if picked is None:
                        await edit_retry_html(working_msg, "Сейчас нет нового уникального сигнала для вас.\nПопробуйте позже — я подберу другой сетап.")
                        return

                    symbol, details = picked
                    side = details["side"]; entry = details["c5"]; sl = details["sl"]; tps = details["tps"]
                    leverage = details["leverage"]; risk_level = details["risk_level"]
                    news_note = details["news_note"]; atr_value = details["atr"]; watch_seconds = details["watch_seconds"]
                    reason = app["build_reason"](details)

                    if side == "LONG":
                        if not (all(tp > entry for tp in tps) and sl < entry):
                            txt = _format_analysis_text(app["build_reason"], app.get("format_price"), details, symbol.split("/")[0], side, details.get("score"))
                            txt += "\n\n⚠️ Сигнал не прошёл валидацию."
                            await edit_retry_html(working_msg, txt)
                            return
                    else:
                        if not (all(tp < entry for tp in tps) and sl > entry):
                            txt = _format_analysis_text(app["build_reason"], app.get("format_price"), details, symbol.split("/")[0], side, details.get("score"))
                            txt += "\n\n⚠️ Сигнал не прошёл валидацию."
                            await edit_retry_html(working_msg, txt)
                            return

                    sig = Signal(
                        user_id=user_id, symbol=symbol, side=side, entry=entry, tps=tps, sl=sl,
                        leverage=leverage, risk_level=risk_level, created_at=now_msk(),
                        news_note=news_note, atr_value=atr_value,
                        watch_until=now_msk() + timedelta(seconds=watch_seconds), reason=reason,
                    )
                    text = format_signal_message(sig)
                    await edit_retry_html(working_msg, text)
                    st["count"] = st.get("count", 0) + 1
                    await db.save_user_state(user_id, st)
                    sig.id = await db.add_signal(sig)
                    task = asyncio.create_task(app["watch_signal_price"](bot, message.chat.id, sig))
                    active_watch_tasks.setdefault(user_id, []).append(task)
                    if MET_SIGNALS_GEN: MET_SIGNALS_GEN.inc()
                except Exception as e:
                    logger and logger.exception("Signal generation error: %s", e)
                    with contextlib.suppress(Exception):
                        await edit_retry_html(working_msg, "⚠️ Произошла ошибка при поиске сигнала.")

            setattr(target, "callback", cmd_signal_patched)
            logger and logger.info("Signal handler patched: anti-duplicate enabled.")
        except Exception as e:
            logger and logger.warning("Signal handler patch error: %s", e)

    # ---------- Перехват fallback: /coin, кнопка, /mycity, ввод города/тикера ----------
    def _patch_fallback():
        try:
            guard_access = app.get("guard_access")
            resolve_symbol_from_query = app.get("resolve_symbol_from_query")
            db = app.get("db")

            obs = router.message
            handlers = getattr(obs, "handlers", [])
            target = None
            for h in handlers:
                cb = getattr(h, "callback", None)
                if getattr(cb, "__name__", "") == "fallback":
                    target = h; break
            if not target:
                logger and logger.warning("fallback handler not found to patch.")
                return
            orig_fallback = target.callback

            async def fallback_patched(message, bot):
                uid = message.from_user.id
                raw = (message.text or "")
                text = raw.strip()
                low = text.lower()

                # 1) ждём город после /code
                if uid in CITY_PENDING and text and not text.startswith("/"):
                    with contextlib.suppress(Exception):
                        await _set_city(db, uid, text)
                        await message.answer(f"Город сохранён: {text}")
                    CITY_PENDING.discard(uid)
                    return

                # 2) /mycity (только админам)
                if low.startswith("/mycity"):
                    st = await db.get_user_state(uid)
                    if not st.get("admin"):
                        await message.answer("Доступно только для админов.")
                        return
                    city = await _get_city(db, uid)
                    await message.answer(f"Ваш город: {city}")
                    return

                # 3) Анализ монеты
                if ("анализ монеты" in low) or ("анализ конкретной монеты" in low) or (text == "🔎 Анализ монеты") or low.startswith("/coin"):
                    st = await guard_access(message, bot)
                    if not st: return

                    parts = text.split(maxsplit=1)
                    if len(parts) > 1 and parts[0].lower().startswith("/coin"):
                        q = parts[1].strip()
                        sym = None
                        with contextlib.suppress(Exception):
                            sym = resolve_symbol_from_query(q)
                        if not sym:
                            sym = _resolve_symbol_any(app, q)
                        if not sym:
                            await message.answer("Монета не распознана. Пример: BTC, ETH, SOL.")
                            return
                        await _do_coin_analysis_flow(app, message, bot, uid, sym)
                        return

                    COIN_PENDING.add(uid)
                    await message.answer("Введите название монеты (например, BTC или ETH)")
                    return

                # 4) Если ждём тикер — ловим
                if uid in COIN_PENDING and text and not text.startswith("/"):
                    st = await guard_access(message, bot)
                    if not st: return
                    sym = None
                    with contextlib.suppress(Exception):
                        sym = resolve_symbol_from_query(text)
                    if not sym:
                        sym = _resolve_symbol_any(app, text)
                    if not sym:
                        await message.answer("Монета не распознана. Пример: BTC, ETH, SOL.")
                        return
                    COIN_PENDING.discard(uid)
                    await _do_coin_analysis_flow(app, message, bot, uid, sym)
                    return

                # 5) прочие сообщения — оригинальный фолбек
                await orig_fallback(message, bot)

            setattr(target, "callback", fallback_patched)
            logger and logger.info("fallback patched: /coin, кнопка, /mycity и ввод города/тикера обрабатываются приоритетно.")
        except Exception as e:
            logger and logger.warning("fallback patch error: %s", e)

    # ---------- Улучшения: обёртка score_symbol_core ----------
    orig_score = app.get("score_symbol_core")
    if orig_score:
        def _score_plus(symbol: str, relax: bool = False):
            base = orig_score(symbol, relax)
            if base is None:
                logger and logger.warning(f"Score plus skipped for {symbol}: base score is None")
                return None
            score, side, d = base
            d = dict(d or {})
            breakdown = d.get("score_breakdown", {}) or {}
            logger and logger.info(f"Score plus started for {symbol}, side {side}")
            try:
                # Предполагаем, что в d есть df5, df15, df1h, df4h, df1d, close= d["c5"], atr=d["atr"]
                df5 = d.get("df5"); df15 = d.get("df15"); df1h = d.get("df1h"); df4h = d.get("df4h"); df1d = d.get("df1d")
                close = float(d.get("c5", 0))
                atr = float(d.get("atr", 0))
                now_utc = datetime.now(timezone.utc)

                # Проверка валидности данных
                if df15 is None or df15.empty:
                    logger and logger.warning(f"Score plus for {symbol}: df15 is None or empty, skipping TA")
                    d["score_breakdown"] = breakdown
                    return base

                # 1) BBWP
                if TA_BBWP and df15 is not None and not df15.empty:
                    try:
                        bbwp = _bbwp(df15["close"])
                        d["bbwp"] = bbwp
                        bbwp_pct = bbwp
                        d["bbwp_pct"] = bbwp_pct
                        adx15 = d.get("adx15", 0)
                        if bbwp_pct < 15 and adx15 < 18:
                            breakdown["BBWP"] = breakdown.get("BBWP", 0) - W_BBWP
                        elif bbwp_pct > 85:
                            breakdown["BBWP"] = breakdown.get("BBWP", 0) + W_BBWP * 0.5
                        logger and logger.info(f"BBWP calculated: {bbwp}, adjustment {breakdown.get('BBWP')}")
                    except Exception as e:
                        logger and logger.warning(f"BBWP failed for {symbol}: {e}")

                # 2) Сессии
                if TA_SESSION:
                    try:
                        bounds = _session_bounds_utc(now_utc)
                        session = _active_session(now_utc)
                        d["session"] = session
                        ib_hi, ib_lo = _initial_balance_levels_session(df15, session, bounds)
                        d["ib_hi"] = ib_hi; d["ib_lo"] = ib_lo
                        vwaps = {}
                        for s in ["ASIA", "EU", "US"]:
                            vwaps[s] = _session_vwap(df15, s, bounds)
                        d["vwaps"] = vwaps
                        breakout = _session_breakout(df15, session, bounds, close, ib_hi or 0, ib_lo or 0)
                        d["session_breakout"] = breakout
                        dist_to_ib = min(abs(close - (ib_hi or close)), abs(close - (ib_lo or close))) / atr if atr > 0 else 0
                        if breakout * (1 if side == "LONG" else -1) > 0:
                            breakdown["Session"] = breakdown.get("Session", 0) + W_SESSION
                        elif dist_to_ib < 0.3:
                            breakdown["Session"] = breakdown.get("Session", 0) - W_SESSION
                        logger and logger.info(f"Session {session}, breakout {breakout}, adjustment {breakdown.get('Session')}")
                    except Exception as e:
                        logger and logger.warning(f"Session TA failed for {symbol}: {e}")

                # 3) Пивоты
                if TA_PIVOTS and df1d is not None and not df1d.empty:
                    try:
                        pivots = _daily_pivots(df1d)
                        d["pivots"] = pivots
                        levels = [pivots.get(k) for k in ["R1", "R2", "S1", "S2"] if pivots.get(k)]
                        if levels:
                            closest = min(levels, key=lambda lv: abs(close - lv))
                            dist = abs(close - closest) / atr if atr > 0 else 0
                            is_resist = (side == "LONG" and closest > close) or (side == "SHORT" and closest < close)
                            if dist < 0.25 and is_resist:
                                breakdown["Pivots"] = breakdown.get("Pivots", 0) - W_PIVOTS
                            logger and logger.info(f"Pivots closest {closest}, dist {dist}, adjustment {breakdown.get('Pivots')}")
                    except Exception as e:
                        logger and logger.warning(f"Pivots failed for {symbol}: {e}")

                # 4) Round levels
                if TA_ROUND:
                    try:
                        step = atr * 0.5 if atr > 0 else 0.25  # dynamic step
                        round_level, round_dist = _near_round_level(close, step)
                        d["round_level"] = round_level
                        d["round_dist_atr"] = round_dist * step / atr if atr > 0 else 0
                        is_against = (side == "LONG" and round_level > close) or (side == "SHORT" and round_level < close)
                        if d["round_dist_atr"] < 0.2 and is_against:
                            breakdown["Round"] = breakdown.get("Round", 0) - W_ROUND
                        logger and logger.info(f"Round level {round_level}, dist_atr {d['round_dist_atr']}, adjustment {breakdown.get('Round')}")
                    except Exception as e:
                        logger and logger.warning(f"Round levels failed for {symbol}: {e}")

                # 5) RSI heatmap
                if TA_RSI_STACK:
                    try:
                        rsi_map = _rsi_map(df5, df15, df1h, df4h)
                        d["rsi_map"] = rsi_map
                        consensus = _rsi_consensus(rsi_map, side)
                        if consensus >= 3:
                            breakdown["RSIstack"] = breakdown.get("RSIstack", 0) + W_RSI
                        elif consensus <= -2:
                            breakdown["RSIstack"] = breakdown.get("RSIstack", 0) - W_RSI
                        logger and logger.info(f"RSI consensus {consensus}, adjustment {breakdown.get('RSIstack')}")
                    except Exception as e:
                        logger and logger.warning(f"RSI heatmap failed for {symbol}: {e}")

                # 6) HA run
                if TA_HA and df15 is not None and not df15.empty:
                    try:
                        ha = _heikin_ashi(df15)
                        ha_run, ha_exhaustion = _ha_run_exhaustion(ha)
                        d["ha_run"] = ha_run
                        d["ha_exhaustion"] = ha_exhaustion
                        run_dir = 1 if ha_run > 0 else -1 if ha_run < 0 else 0
                        if abs(ha_run) > 6 and run_dir == (1 if side == "LONG" else -1):
                            breakdown["HA"] = breakdown.get("HA", 0) - W_HA  # не догонять
                        if ha_exhaustion:
                            breakdown["HA"] = breakdown.get("HA", 0) - W_HA * 0.5
                        logger and logger.info(f"HA run {ha_run}, exhaustion {ha_exhaustion}, adjustment {breakdown.get('HA')}")
                    except Exception as e:
                        logger and logger.warning(f"HA run failed for {symbol}: {e}")

                # 7) Ribbon
                if TA_RIBBON and df15 is not None and not df15.empty:
                    try:
                        ribbon_w = _ema_ribbon(df15)
                        d["ribbon_w"] = ribbon_w
                        ribbon_state = _ribbon_state(ribbon_w, atr)
                        d["ribbon_state"] = ribbon_state
                        adx15 = d.get("adx15", 0)
                        if ribbon_state == "compressed" and adx15 > 25:
                            breakdown["Ribbon"] = breakdown.get("Ribbon", 0) + W_RIBBON
                        elif ribbon_state == "expanded" and d.get("bbwp_pct", 50) > 90:
                            breakdown["Ribbon"] = breakdown.get("Ribbon", 0) - W_RIBBON
                        logger and logger.info(f"Ribbon state {ribbon_state}, adjustment {breakdown.get('Ribbon')}")
                    except Exception as e:
                        logger and logger.warning(f"Ribbon failed for {symbol}: {e}")

                # 8) Fib
                if TA_FIB and df1h is not None and not df1h.empty and df4h is not None and not df4h.empty:
                    try:
                        swing_hi1h, swing_lo1h = _swing_points(df1h)
                        fib1h = _fib_levels(swing_hi1h, swing_lo1h)
                        swing_hi4h, swing_lo4h = _swing_points(df4h)
                        fib4h = _fib_levels(swing_hi4h, swing_lo4h)
                        d["fib"] = {**{f"1h_{k}": v for k, v in fib1h.items()}, **{f"4h_{k}": v for k, v in fib4h.items()}}
                        logger and logger.info(f"Fib levels calculated for 1h/4h")
                    except Exception as e:
                        logger and logger.warning(f"Fib levels failed for {symbol}: {e}")

                # 9) Day type
                if TA_DAYTYPE and df15 is not None and not df15.empty and "ib_hi" in d:
                    try:
                        day_type = _day_type(df15, d["ib_hi"], d["ib_lo"])
                        d["day_type"] = day_type
                        if "Open-Drive" in day_type and ("Up" if side == "LONG" else "Down") in day_type:
                            breakdown["DayType"] = breakdown.get("DayType", 0) + W_DAYTYPE
                        elif "Balanced" in day_type:
                            breakdown["DayType"] = breakdown.get("DayType", 0) - W_DAYTYPE * 0.5
                        logger and logger.info(f"Day type {day_type}, adjustment {breakdown.get('DayType')}")
                    except Exception as e:
                        logger and logger.warning(f"Day type calculation failed for {symbol}: {e}")

                # 10) finalize
                d["score_breakdown"] = breakdown
                try:
                    adj = float(sum(breakdown.values())) if breakdown else 0.0
                except Exception:
                    adj = 0.0
                score2 = float(score) + adj
                d["score"] = score2
                return score2, side, d
            except Exception as e:
                logger and logger.warning(f"Score plus overall failed for {symbol}: {e}")
                return base

        app["score_symbol_core"] = _score_plus

    # Включаем патчи хендлеров
    _patch_cmd_code()
    _patch_cmd_signal()
    _patch_fallback()

    # Обновляем on_startup для поддержки новых фич (опционально)
    async def _on_startup_patched(bot):
        # ensure city column exists
        with contextlib.suppress(Exception):
            if app.get("db"):
                await _ensure_city_column(app["db"])
        # запустить фоновые утренние рассылки, если нужно
        with contextlib.suppress(Exception):
            asyncio.create_task(start_daily_channel_post_loop(app, bot))
        with contextlib.suppress(Exception):
            asyncio.create_task(start_daily_admin_greetings_loop(app, bot))
        # вызвать оригинальный on_startup
        if orig_on_startup:
            await orig_on_startup(bot)

    app["on_startup"] = _on_startup_patched
