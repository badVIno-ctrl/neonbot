# lock.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os
import math
import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# ============ Новые TA-алгоритмы ============

def kama(series: pd.Series, er_period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """
    Kaufman Adaptive Moving Average.
    er_period: окно для Efficiency Ratio (обычно 10)
    fast, slow: периоды быстрых/медленных EMA (2 и 30 по дефолту)
    """
    x = series.astype(float).values
    n = len(x)
    if n == 0:
        return series.copy()

    change = np.abs(x - np.roll(x, er_period))
    change[:er_period] = np.nan
    vol = np.zeros(n)
    for i in range(1, er_period + 1):
        vol += np.abs(x - np.roll(x, i))
    vol[:er_period] = np.nan

    er = np.divide(change, vol, out=np.zeros_like(change), where=(vol != 0))
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    out = np.full(n, np.nan, dtype=float)
    if n > 0:
        out[0] = x[0]
        for i in range(1, n):
            sc_i = sc[i] if not np.isnan(sc[i]) else slow_sc**2
            out[i] = out[i - 1] + sc_i * (x[i] - out[i - 1])
    return pd.Series(out, index=series.index, name="kama")


def ehlers_super_smoother(series: pd.Series, period: int = 20) -> pd.Series:
    """
    Ehlers 2-pole Super Smoother Filter.
    """
    x = series.astype(float).values
    n = len(x)
    if n == 0 or period <= 2:
        return series.copy()

    a1 = math.exp(-1.414 * math.pi / period)
    b1 = 2.0 * a1 * math.cos(1.414 * math.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1.0 - c2 - c3

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
    """
    VIX Fix (Larry Williams): оценивает "страх" как всплески локальных минимумов.
    """
    hh = close.rolling(n).max()
    out = (hh - low) / (hh + 1e-12) * 100.0
    return out.rename("vixfix")


def tsf_slope(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Linear Regression slope (per window). Возвращает наклон, нормированный на цену.
    """
    y = series.astype(float).values
    x = np.arange(len(y))
    out = np.full(len(y), np.nan, dtype=float)
    for i in range(window, len(y) + 1):
        yi = y[i - window:i]
        xi = x[i - window:i]
        a, b = np.polyfit(xi, yi, 1)
        out[i - 1] = a / (abs(yi.mean()) + 1e-12)
    return pd.Series(out, index=series.index, name="tsf_slope")


# ============ Утренний пост в канал ============

COIN_MAP = [
    ("BTC", "BTC/USDT"),
    ("ETH", "ETH/USDT"),
    ("SOL", "SOL/USDT"),
    ("TON", "TON/USDT"),   # при необходимости можно попробовать 'TONCOIN/USDT'
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
    lines.append("")
    lines.append("🄽🄴🄾🄽")
    lines.append("")
    lines.append("Ключевые цены:")

    # Получаем цены
    for sym, mkt in COIN_MAP:
        price = None
        try:
            price = market.fetch_mark_price(mkt)
        except Exception:
            price = None
        lines.append(f"• {sym} — {_format_usd(price)}")

    lines.append("")
    lines.append(f"Время: {time_str} МСК • {date_str}")
    lines.append("")
    lines.append("#BTC #ETH #SOL #TON #BNB #crypto #Neon #trading")

    msg = "\n".join(lines)
    try:
        await bot.send_message(channel_id, msg)
        logger and logger.info("Morning post sent to %s at %s MSK", channel_id, time_str)
    except Exception as e:
        logger and logger.exception("Failed to send morning post: %s", e)

def _seconds_until_next_8_msk(now_msk_fn) -> float:
    now = now_msk_fn()
    target = now.replace(hour=8, minute=0, second=0, microsecond=0)
    if now >= target:
        target = target + timedelta(days=1)
    return max(1.0, (target - now).total_seconds())

async def start_daily_channel_post_loop(app: Dict[str, Any], bot, channel_id: Optional[str] = None) -> None:
    """
    Стартует бесконечный фоновой таск, который раз в день в 08:00 МСК отправляет пост в канал.
    channel_id: можно передать напрямую или задать ENV TG_CHANNEL (например, '@NeonFakTrading').
    """
    logger = app.get("logger")
    ch = _ensure_channel_id(channel_id or os.getenv("TG_CHANNEL") or "@NeonFakTrading")
    if not ch:
        logger and logger.warning("Daily post loop not started: channel_id is not set")
        return

    logger and logger.info("Starting daily channel post loop for %s ...", ch)
    await asyncio.sleep(2)  # дать системе подняться
    while True:
        try:
            wait_s = _seconds_until_next_8_msk(app["now_msk"])
            await asyncio.sleep(wait_s)
            await _post_morning_report(app, bot, ch)
        except asyncio.CancelledError:
            logger and logger.info("Daily channel post loop cancelled.")
            break
        except Exception as e:
            logger and logger.exception("Daily channel loop error: %s", e)
            await asyncio.sleep(30)  # небольшой бэкофф и продолжим


# ============ Patch: подключение TA-алгоритмов и мягкое усиление скоринга ============

def patch(app: Dict[str, Any]) -> None:
    """
    Подключает новые TA-функции и обогащает текущий скоринг без изменения main.
    В app добавляются ссылки на функции + обёртка score_symbol_core.
    """
    logger = app.get("logger")
    market = app.get("market")

    # Экспортируем TA-функции в app
    app["kama"] = kama
    app["ehlers_super_smoother"] = ehlers_super_smoother
    app["donchian_channel"] = donchian_channel
    app["keltner_channel"] = keltner_channel
    app["vix_fix"] = vix_fix
    app["tsf_slope"] = tsf_slope

    # Утренний пост: экспорт функции старта фоновой задачи
    app["start_daily_channel_post_loop"] = start_daily_channel_post_loop

    # Сохраняем оригинальный скоринг
    orig_score_symbol_core = app.get("score_symbol_core")

    def _lock_enhanced_score(symbol: str, relax: bool = False):
        if orig_score_symbol_core:
            base = orig_score_symbol_core(symbol, relax)
        else:
            return None

        if base is None:
            return None

        side_score, side, details = base
        # Подтягиваем данные
        try:
            df15 = app["market"].fetch_ohlcv(symbol, "15m", 500)
            if df15 is None or len(df15) < 60:
                return base

            close = df15["close"].astype(float)
            high = df15["high"].astype(float)
            low = df15["low"].astype(float)
            breakdown: Dict[str, float] = details.get("score_breakdown", {}).copy() if isinstance(details.get("score_breakdown"), dict) else {}

            # Параметры
            atr = app.get("atr")
            adx = app.get("adx")
            atr15 = float(atr(df15, 14).iloc[-1]) if atr else float((high - low).rolling(14).mean().iloc[-1])

            # KAMA
            k = kama(close, er_period=10)
            kama_slope = (k.iloc[-1] - k.iloc[-5]) if len(k.dropna()) >= 5 else 0.0
            details["kama"] = float(k.iloc[-1]) if np.isfinite(k.iloc[-1]) else None
            if side == "LONG":
                if close.iloc[-1] > k.iloc[-1] and kama_slope > 0:
                    breakdown["KAMA"] = breakdown.get("KAMA", 0.0) + 0.10
                elif close.iloc[-1] < k.iloc[-1]:
                    breakdown["KAMA"] = breakdown.get("KAMA", 0.0) - 0.05
            else:
                if close.iloc[-1] < k.iloc[-1] and kama_slope < 0:
                    breakdown["KAMA"] = breakdown.get("KAMA", 0.0) + 0.10
                elif close.iloc[-1] > k.iloc[-1]:
                    breakdown["KAMA"] = breakdown.get("KAMA", 0.0) - 0.05

            # Donchian
            up_d, lo_d, mid_d = donchian_channel(high, low, window=20)
            details["donchian_up"] = float(up_d.iloc[-1]) if np.isfinite(up_d.iloc[-1]) else None
            details["donchian_lo"] = float(lo_d.iloc[-1]) if np.isfinite(lo_d.iloc[-1]) else None
            if side == "LONG" and close.iloc[-1] > up_d.iloc[-1] - 1e-9:
                breakdown["Donchian"] = breakdown.get("Donchian", 0.0) + 0.10
            if side == "SHORT" and close.iloc[-1] < lo_d.iloc[-1] + 1e-9:
                breakdown["Donchian"] = breakdown.get("Donchian", 0.0) + 0.10

            # Keltner
            up_k, mid_k, lo_k = keltner_channel(high, low, close, ema_period=20, atr_period=20, mult=1.5)
            details["keltner_mid"] = float(mid_k.iloc[-1]) if np.isfinite(mid_k.iloc[-1]) else None
            if side == "LONG" and (up_k.iloc[-1] - close.iloc[-1]) < 0.2 * atr15:
                breakdown["Keltner"] = breakdown.get("Keltner", 0.0) - 0.08
            if side == "SHORT" and (close.iloc[-1] - lo_k.iloc[-1]) < 0.2 * atr15:
                breakdown["Keltner"] = breakdown.get("Keltner", 0.0) - 0.08

            # VIX Fix (сигнал к развороту после "страха")
            vix = vix_fix(close, low, n=22)
            details["vixfix"] = float(vix.iloc[-1]) if np.isfinite(vix.iloc[-1]) else None
            if side == "LONG" and vix.iloc[-1] > 8.0:
                breakdown["VIXFix"] = breakdown.get("VIXFix", 0.0) + 0.08
            if side == "SHORT" and vix.iloc[-1] < 2.0:
                breakdown["VIXFix"] = breakdown.get("VIXFix", 0.0) + 0.05

            # Super Smoother
            ss = ehlers_super_smoother(close, period=20)
            ss_slope = ss.iloc[-1] - ss.iloc[-3] if len(ss.dropna()) >= 3 else 0.0
            details["ssmooth"] = float(ss.iloc[-1]) if np.isfinite(ss.iloc[-1]) else None
            if side == "LONG" and ss_slope > 0 and close.iloc[-1] > ss.iloc[-1]:
                breakdown["SSmooth"] = breakdown.get("SSmooth", 0.0) + 0.05
            if side == "SHORT" and ss_slope < 0 and close.iloc[-1] < ss.iloc[-1]:
                breakdown["SSmooth"] = breakdown.get("SSmooth", 0.0) + 0.05

            # TSF slope
            tsf = tsf_slope(close, window=20)
            tsf_last = float(tsf.iloc[-1]) if np.isfinite(tsf.iloc[-1]) else 0.0
            details["tsf_slope"] = tsf_last
            if side == "LONG" and tsf_last > 0:
                breakdown["TSF"] = breakdown.get("TSF", 0.0) + 0.05
            if side == "SHORT" and tsf_last < 0:
                breakdown["TSF"] = breakdown.get("TSF", 0.0) + 0.05

            # ADX поддержка для Donchian выноса (если есть)
            if adx:
                adx_val = float(adx(df15, 14).iloc[-1])
                if adx_val > 22:
                    if side == "LONG" and close.iloc[-1] > up_d.iloc[-1]:
                        breakdown["ADXxTrend"] = breakdown.get("ADXxTrend", 0.0) + 0.05
                    if side == "SHORT" and close.iloc[-1] < lo_d.iloc[-1]:
                        breakdown["ADXxTrend"] = breakdown.get("ADXxTrend", 0.0) + 0.05

            # Обновляем детали/скор
            details["score_breakdown"] = breakdown
            side_score = float(side_score + sum(breakdown.values()))
            return side_score, side, details
        except Exception:
            return base

    # Подменяем скоринг новой обёрткой
    if orig_score_symbol_core:
        app["score_symbol_core"] = _lock_enhanced_score

    # Лог об успешном применении
    logger and logger.info("Lock patch applied: TA extras (KAMA, Ehlers SS, Donchian/Keltner, VIXFix, TSF) + daily channel post exported.")
