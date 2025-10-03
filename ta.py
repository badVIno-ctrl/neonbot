# ta.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Set
import os
import math
import asyncio
import contextlib
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from aiogram import F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton

# ===================== RUNTIME / ENV =====================
_TA_PENDING: Set[int] = set()
NEON_TA_DAILY_LIMIT = int(os.getenv("NEON_TA_DAILY_LIMIT", "4"))

# ===================== Helpers =====================
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _fmt_price(app: Dict[str, Any], x: float) -> str:
    f = app.get("format_price") or (lambda v: f"{v:.4f}")
    try:
        return f(float(x))
    except Exception:
        return f"{x:.4f}"

def _today_key_msk(app: Dict[str, Any]) -> str:
    f = app.get("today_key")
    if callable(f):
        try:
            return f()
        except Exception:
            pass
    now = _now_utc()
    return (now + pd.Timedelta(hours=3)).strftime("%Y-%m-%d")

async def _ensure_ta_columns(app: Dict[str, Any]) -> None:
    db = app.get("db")
    if not db or not getattr(db, "conn", None):
        return
    stmts = (
        "ALTER TABLE users ADD COLUMN ta_count INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE users ADD COLUMN ta_date TEXT",
    )
    for s in stmts:
        with contextlib.suppress(Exception):
            await db.conn.execute(s)
            await db.conn.commit()

async def _ta_allowed_and_inc(app: Dict[str, Any], user_id: int, st: Dict[str, Any]) -> Tuple[bool, int, int]:
    await _ensure_ta_columns(app)
    db = app.get("db")
    if not db or not getattr(db, "conn", None):
        return False, 0, NEON_TA_DAILY_LIMIT
    if bool(st.get("unlimited") or st.get("admin")):
        return True, 0, 0
    dkey = _today_key_msk(app)
    try:
        cur = await db.conn.execute("SELECT ta_count, ta_date FROM users WHERE user_id=?", (user_id,))
        row = await cur.fetchone()
        if not row:
            await db.get_user_state(user_id)
            cur = await db.conn.execute("SELECT ta_count, ta_date FROM users WHERE user_id=?", (user_id,))
            row = await cur.fetchone()
        cnt = int(row["ta_count"] if row and row["ta_count"] is not None else 0)
        date = str(row["ta_date"] or "")
        if date != dkey:
            cnt = 0
        if cnt >= NEON_TA_DAILY_LIMIT:
            return False, cnt, NEON_TA_DAILY_LIMIT
        cnt2 = cnt + 1
        await db.conn.execute("UPDATE users SET ta_count=?, ta_date=? WHERE user_id=?", (cnt2, dkey, user_id))
        await db.conn.commit()
        return True, cnt2, NEON_TA_DAILY_LIMIT
    except Exception:
        return False, 0, NEON_TA_DAILY_LIMIT

def _norm_token(txt: str) -> str:
    t = "".join(ch for ch in (txt or "") if ch.isalnum() or ch in "/").upper().strip()
    if t.endswith("USDT"):
        t = t[:-4]
    if "/" in t:
        t = t.split("/")[0]
    return t

def _resolve_symbol_any(app: Dict[str, Any], raw: str) -> Optional[str]:
    market = app.get("market")
    rsfq = app.get("resolve_symbol_from_query")
    if callable(rsfq):
        try:
            sym = rsfq(raw)
            if sym:
                return sym
        except Exception:
            pass
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
    return candidate

# ===================== TA computations =====================
def _ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()

def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    rs = up.rolling(period).mean() / (dn.rolling(period).mean() + 1e-12)
    return 100 - 100 / (1 + rs)

def _macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def _bbwp(s: pd.Series, length: int = 20, lookback: int = 96) -> Optional[float]:
    if len(s) < max(length, lookback) + 5:
        return None
    basis = s.rolling(length).mean()
    dev = s.rolling(length).std(ddof=0)
    width = (2 * dev * 2) / (basis.abs() + 1e-12)  # (UB-LB)/|mid| ~ 4*dev/|mid|
    rank = width.rolling(lookback).rank(pct=True)
    val = float(rank.iloc[-1] * 100.0)
    return max(0.0, min(100.0, val))

def _rvol(vol: pd.Series, lookback: int = 96) -> Optional[float]:
    if len(vol) < lookback + 5:
        return None
    med = float(vol.tail(lookback).median() + 1e-9)
    return float(vol.iloc[-1] / med) if med > 0 else None

def _obv(df: pd.DataFrame) -> Optional[float]:
    if df is None or len(df) < 40:
        return None
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)
    sign = np.sign(close.diff().fillna(0.0).values)
    cvd = np.cumsum(sign * vol.values)
    return float(cvd[-1] - cvd[-30])

def _fractals(df: pd.DataFrame, left: int = 3, right: int = 3) -> Tuple[List[int], List[int]]:
    highs, lows = [], []
    h = df["high"].values
    l = df["low"].values
    n = len(df)
    for i in range(left, n - right):
        seg_h = h[i - left:i + right + 1]
        seg_l = l[i - left:i + right + 1]
        if h[i] == np.max(seg_h) and np.argmax(seg_h) == left:
            highs.append(i)
        if l[i] == np.min(seg_l) and np.argmin(seg_l) == left:
            lows.append(i)
    return highs, lows

def _cluster_levels(levels: List[float], tol: float) -> List[Tuple[float,float]]:
    if not levels:
        return []
    xs = sorted([float(x) for x in levels if math.isfinite(x)])
    if not xs:
        return []
    clusters = [[xs[0]]]
    for v in xs[1:]:
        if abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [(min(c), max(c)) for c in clusters]

def _structure_hh_hl(close: pd.Series) -> str:
    if len(close) < 30:
        return "переменная"
    y = close.values.astype(float)
    highs, lows = [], []
    for i in range(2, len(y)-2):
        if y[i] > y[i-1] and y[i] > y[i+1] and y[i] > y[i-2] and y[i] > y[i+2]:
            highs.append(i)
        if y[i] < y[i-1] and y[i] < y[i+1] and y[i] < y[i-2] and y[i] < y[i+2]:
            lows.append(i)
    highs = highs[-3:]; lows = lows[-3:]
    if len(highs) >= 2 and len(lows) >= 2:
        hh = y[highs[-1]] > y[highs[-2]]
        hl = y[lows[-1]] > y[lows[-2]]
        if hh and hl:
            return "HH/HL"
        lh = y[highs[-1]] < y[highs[-2]]
        ll = y[lows[-1]] < y[lows[-2]]
        if lh and ll:
            return "LH/LL"
    return "переменная"

def _daily_pivots(df1d: pd.DataFrame) -> Dict[str, float]:
    if df1d is None or len(df1d) < 2:
        return {}
    prev = df1d.iloc[-2]
    P = (float(prev["high"]) + float(prev["low"]) + float(prev["close"])) / 3.0
    R1 = 2 * P - float(prev["low"])
    S1 = 2 * P - float(prev["high"])
    R2 = P + (float(prev["high"]) - float(prev["low"]))
    S2 = P - (float(prev["high"]) - float(prev["low"]))
    return {"P": P, "R1": R1, "S1": S1, "R2": R2, "S2": S2}

def _inside_bar(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 3:
        return False
    h1, l1 = float(df["high"].iloc[-1]), float(df["low"].iloc[-1])
    h0, l0 = float(df["high"].iloc[-2]), float(df["low"].iloc[-2])
    return (h1 <= h0 and l1 >= l0)

def _sfp(df: pd.DataFrame, lookback: int = 30) -> Tuple[bool, Optional[str]]:
    if df is None or len(df) < lookback + 3:
        return False, None
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    H = np.max(h[-(lookback + 1):-1])
    L = np.min(l[-(lookback + 1):-1])
    if h[-1] > H and c[-1] < H:
        return True, "bear"
    if l[-1] < L and c[-1] > L:
        return True, "bull"
    return False, None

def _volume_profile(df: pd.DataFrame, bins: int = 40, lookback: int = 240) -> Tuple[Optional[float], Optional[float], Optional[float]]:
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

def _trend_label(df: pd.DataFrame) -> str:
    if df is None or len(df) < 220:
        return "неопределён"
    ema200 = _ema(df["close"], 200)
    c = float(df["close"].iloc[-1])
    e = float(ema200.iloc[-1])
    if c > e * 1.004:
        return "восходящий"
    if c < e * 0.996:
        return "нисходящий"
    return "боковой"

def _range_text(lo: float, hi: float, app: Dict[str, Any]) -> str:
    mid = (lo + hi) / 2.0
    return f"{_fmt_price(app, lo)} – {_fmt_price(app, hi)}, середина ~{_fmt_price(app, mid)}"

def _cme_gap_approx(df1h: pd.DataFrame) -> Optional[Tuple[float, float]]:
    try:
        if df1h is None or len(df1h) < 400:
            return None
        x = df1h.copy()
        x["ts"] = pd.to_datetime(x["ts"], utc=True, errors="coerce")
        x["dow"] = x["ts"].dt.dayofweek
        x["hour"] = x["ts"].dt.hour
        fri = x[(x["dow"] == 4) & (x["hour"] == 21)]
        mon = x[(x["dow"] == 0) & (x["hour"] == 0)]
        if fri.empty or mon.empty:
            return None
        last_fri = float(fri["close"].iloc[-1]); first_mon = float(mon["open"].iloc[-1])
        gap = first_mon - last_fri
        return float(gap), float(first_mon)
    except Exception:
        return None

# ===================== Report builder =====================
def _build_tech_report(app: Dict[str, Any], symbol: str) -> Optional[str]:
    market = app.get("market")
    now_msk = app["now_msk"]
    anchored_vwap = app.get("anchored_vwap")

    df1w = market.fetch_ohlcv(symbol, "1w", 260)
    df1d = market.fetch_ohlcv(symbol, "1d", 400)
    df4h = market.fetch_ohlcv(symbol, "4h", 400)
    df1h = market.fetch_ohlcv(symbol, "1h", 420)
    df15 = market.fetch_ohlcv(symbol, "15m", 400)

    if any(x is None or len(x) < 50 for x in (df1d, df4h, df1h)):
        return None

    base = symbol.split("/")[0]
    c1h = float(df1h["close"].iloc[-1])

    # Индикаторы
    rsi1w = float(_rsi(df1w["close"], 14).iloc[-1]) if df1w is not None and len(df1w) >= 20 else float("nan")
    rsi1d = float(_rsi(df1d["close"], 14).iloc[-1]) if len(df1d) >= 20 else float("nan")
    rsi4h = float(_rsi(df4h["close"], 14).iloc[-1]) if len(df4h) >= 20 else float("nan")
    rsi1h = float(_rsi(df1h["close"], 14).iloc[-1]) if len(df1h) >= 20 else float("nan")

    macd4, macds4, macdh4 = _macd(df4h["close"]) if len(df4h) >= 40 else (None, None, None)
    macd1, macds1, macdh1 = _macd(df1h["close"]) if len(df1h) >= 40 else (None, None, None)

    bbwp1h = _bbwp(df1h["close"])  # 0..100
    rvol1h = _rvol(df1h["volume"])
    obv_slope = _obv(df1h)

    lo4 = float(df4h["low"].tail(72).min()); hi4 = float(df4h["high"].tail(72).max())    # 4H диапазон
    lo1 = float(df1h["low"].tail(120).min()); hi1 = float(df1h["high"].tail(120).max())  # 1H диапазон

    # Тренды/структура
    trend_1d = _trend_label(df1d)
    trend_4h = _trend_label(df4h)
    trend_1h = _trend_label(df1h)
    struct_1d = _structure_hh_hl(df1d["close"])
    struct_4h = _structure_hh_hl(df4h["close"])
    struct_1h = _structure_hh_hl(df1h["close"])

    # EMA контекст
    ema50_1d = float(_ema(df1d["close"], 50).iloc[-1]) if len(df1d) >= 60 else float("nan")
    ema200_1d = float(_ema(df1d["close"], 200).iloc[-1]) if len(df1d) >= 220 else float("nan")
    ema50_4h = float(_ema(df4h["close"], 50).iloc[-1]) if len(df4h) >= 60 else float("nan")
    ema200_4h = float(_ema(df4h["close"], 200).iloc[-1]) if len(df4h) >= 220 else float("nan")
    ema50_1h = float(_ema(df1h["close"], 50).iloc[-1]) if len(df1h) >= 60 else float("nan")
    ema200_1h = float(_ema(df1h["close"], 200).iloc[-1]) if len(df1h) >= 220 else float("nan")

    # Фрактальные кластеры уровней (1H)
    highs, lows = _fractals(df1h, 3, 3)
    atr1h = float((df1h["high"] - df1h["low"]).rolling(14).mean().iloc[-1] or 0.0)
    tol = max(atr1h * 0.6, (c1h * 0.001))
    res_levels = _cluster_levels([float(df1h["high"].iloc[i]) for i in highs if i < len(df1h)-1 and df1h["high"].iloc[i] > c1h], tol)[:4]
    sup_levels = _cluster_levels([float(df1h["low"].iloc[i]) for i in lows if i < len(df1h)-1 and df1h["low"].iloc[i] < c1h], tol)[:6]

    # Профиль объёма (1H блок 240 баров)
    poc, vah, val = _volume_profile(df1h, bins=40, lookback=240)

    # Пивоты (прошлый день)
    piv = _daily_pivots(df1d)

    # VWAP (дневной, по 15m)
    vwap_txt = ""
    if callable(anchored_vwap) and df15 is not None and len(df15) >= 20:
        try:
            day_vwap = anchored_vwap(df15)
            vwap_val = float(day_vwap.iloc[-1])
            if c1h > vwap_val * 1.002:
                vwap_txt = "выше дневного VWAP (преимущество быков внутридня)"
            elif c1h < vwap_val * 0.998:
                vwap_txt = "ниже дневного VWAP (преимущество продавцов внутридня)"
            else:
                vwap_txt = "вблизи дневного VWAP (баланс)"
        except Exception:
            vwap_txt = ""

    # Паттерны (простые)
    pats: List[str] = []
    if _inside_bar(df4h):
        pats.append("4H inside‑bar (сжатие/ожидание импульса)")
    if _inside_bar(df1d):
        pats.append("1D inside‑bar (консолидация на дневке)")
    sfp4, side4 = _sfp(df4h, 40)
    if sfp4:
        pats.append("SFP на 4H (ловушка/сквиз)")
    # Равные хай/лоу (EQH/EQL) на 1H (в пределах tol)
    try:
        last_h = float(df1h["high"].iloc[-1]); prev_h = float(df1h["high"].iloc[-2])
        last_l = float(df1h["low"].iloc[-1]); prev_l = float(df1h["low"].iloc[-2])
        if abs(last_h - prev_h) <= max(atr1h*0.15, c1h*0.0002):
            pats.append("EQH (равные хаи) → над уровнями скопление стопов")
        if abs(last_l - prev_l) <= max(atr1h*0.15, c1h*0.0002):
            pats.append("EQL (равные лои) → под уровнями скопление стопов")
    except Exception:
        pass

    # CME‑gap (приблизительно)
    cme_txt = ""
    cme = _cme_gap_approx(df1h)
    if cme:
        gap, mon_open = cme
        if abs(gap) >= max(atr1h, c1h * 0.002):
            direction = "вверх" if gap > 0 else "вниз"
            cme_txt = f"CME‑gap {direction} ~ { _fmt_price(app, mon_open) } (наблюдать на ближайших сессиях)"

    # Дата/время
    dt = now_msk()
    date_str = dt.strftime("%d.%m.%Y")
    time_str = dt.strftime("%H:%M")

    # Резюме — «живые» формулировки
    def _trend_phrase(tf: str, tr: str, rsi: float, struct: str, extra: str = "") -> str:
        rsi_txt = f"RSI {int(rsi)}" if rsi == rsi else "RSI —"
        parts = [f"{tf}: {('восходящий' if tr=='восходящий' else 'нисходящий' if tr=='нисходящий' else 'боковой')}"]
        parts.append(f"{rsi_txt}")
        if struct and struct != "переменная":
            parts.append(f"структура: {struct}")
        if extra:
            parts.append(extra)
        return " • ".join(parts)

    emaph_1d = ("над EMA200" if (not math.isnan(ema200_1d) and float(df1d['close'].iloc[-1]) > ema200_1d) else
                "под EMA200" if not math.isnan(ema200_1d) else "")
    emaph_4h = ("над EMA200" if (not math.isnan(ema200_4h) and float(df4h['close'].iloc[-1]) > ema200_4h) else
                "под EMA200" if not math.isnan(ema200_4h) else "")
    emaph_1h = ("над EMA200" if (not math.isnan(ema200_1h) and float(df1h['close'].iloc[-1]) > ema200_1h) else
                "под EMA200" if not math.isnan(ema200_1h) else "")

    res_lines = []
    res_lines.append(_trend_phrase("HTF‑тренд (1D)", trend_1d, rsi1d, struct_1d, emaph_1d))
    res_lines.append(_trend_phrase("4H", trend_4h, rsi4h, struct_4h, emaph_4h))
    res_lines.append(_trend_phrase("1H", trend_1h, rsi1h, struct_1h, emaph_1h))

    # BBWP/объёмы/OBV/MACD — комментарии
    flow_bits = []
    if bbwp1h is not None:
        if bbwp1h <= 15:
            flow_bits.append(f"BBWP {bbwp1h:.0f}% (сужение — возможен импульс)")
        elif bbwp1h >= 85:
            flow_bits.append(f"BBWP {bbwp1h:.0f}% (расширение — волатильность повышена)")
        else:
            flow_bits.append(f"BBWP {bbwp1h:.0f}%")
    if rvol1h is not None:
        if rvol1h >= 1.5:
            flow_bits.append(f"RVOL x{rvol1h:.2f} (высокая активность)")
        elif rvol1h < 0.9:
            flow_bits.append(f"RVOL x{rvol1h:.2f} (ниже нормы)")
        else:
            flow_bits.append(f"RVOL x{rvol1h:.2f}")
    if obv_slope is not None:
        if obv_slope > 0:
            flow_bits.append("OBV↑ (приток объёма)")
        elif obv_slope < 0:
            flow_bits.append("OBV↓ (отток объёма)")
    if macdh1 is not None:
        last_h = float(macdh1.iloc[-1])
        flow_bits.append("MACD hist +" if last_h > 0 else "MACD hist −")

    # Ключевые уровни (форматированные диапазоны)
    def _fmt_range(lo: float, hi: float) -> str:
        if abs(hi - lo) < max(atr1h*0.2, c1h*0.0003):
            v = (lo + hi) / 2.0
            return _fmt_price(app, v) + "–" + _fmt_price(app, v)
        return _fmt_price(app, lo) + "–" + _fmt_price(app, hi)

    sr_lines: List[str] = []
    sr_lines.append("Сопротивления:")
    if res_levels:
        for lo, hi in res_levels:
            sr_lines.append(f"• {_fmt_range(lo, hi)} — продавец/верх диапазона")
    else:
        sr_lines.append("• n/a")
    sr_lines.append("")
    sr_lines.append("Поддержки:")
    if sup_levels:
        for lo, hi in sup_levels:
            sr_lines.append(f"• {_fmt_range(lo, hi)} — спрос/локальные лоу")
    else:
        sr_lines.append("• n/a")

    # Контекст / заметки
    notes: List[str] = []
    if piv:
        notes.append(f"Пивоты (вчера): P { _fmt_price(app, piv.get('P')) }, R1 { _fmt_price(app, piv.get('R1')) }, S1 { _fmt_price(app, piv.get('S1')) }")
    if vwap_txt:
        notes.append(f"VWAP: {vwap_txt}")
    if pats:
        notes.append("Паттерны: " + ", ".join(pats[:3]))
    if cme_txt:
        notes.append(cme_txt)
    if poc is not None and vah is not None and val is not None:
        notes.append(f"POC { _fmt_price(app, poc) }, VAH { _fmt_price(app, vah) }, VAL { _fmt_price(app, val) }")
    if flow_bits:
        notes.append("Потоки/вола: " + ", ".join(flow_bits))

    # Сценарии (нейтрально, без сигналов/TP/SL)
    scen_lines: List[str] = []
    scen_lines.append("• Бычий: удержание над серединой диапазона и закрепление выше ближайшего сопротивления → тест верхних зон/кластеров.")
    scen_lines.append("• Нейтральный‑диапазон: отбой от верхней границы → возврат к середине/нижней области баланса.")
    scen_lines.append("• Медвежий: потеря ближайшей поддержки → тест нижних блоков спроса/нижней границы рейнджа.")

    # Сборка текста
    lines: List[str] = []
    lines.append(f"🧭 <b>{base}/USDT</b> — технический анализ")
    lines.append("━━━━━━━━━━━━━━━━━━")
    lines.append(f"📅 {date_str} • {time_str} МСК")
    lines.append("⏱ Таймфреймы: 1W / 1D / 4H / 1H")
    lines.append(f"Диапазон (4H): {_range_text(lo4, hi4, app)}")
    lines.append("")
    lines.append("Резюме (коротко)")
    lines.append("\n".join(res_lines))
    lines.append("")
    lines.append("🗺 Ключевые уровни")
    lines.extend(sr_lines)
    lines.append("")
    if notes:
        lines.append("ℹ️ Контекст")
        lines.extend(["• " + n for n in notes])
        lines.append("")
    lines.append("🎯 Сценарии")
    lines.extend(scen_lines)
    lines.append("")
    lines.append("⚠️ Аналитика не является торговой рекомендацией.")

    return "\n".join(lines)

# ===================== Handlers =====================
async def _do_ta_flow(app: Dict[str, Any], message: Message, bot, user_id: int, symbol: str) -> None:
    guard_access = app.get("guard_access")
    st = await guard_access(message, bot)
    if not st:
        return

    ok, used, limit = await _ta_allowed_and_inc(app, user_id, st)
    if not ok:
        with contextlib.suppress(Exception):
            await message.answer(f"⛔ Лимит теханализа исчерпан: {limit}/день. Попробуйте завтра.")
        return

    base = symbol.split("/")[0]
    with contextlib.suppress(Exception):
        await message.answer(f"⏳ Готовлю теханализ {base}…")

    try:
        text = _build_tech_report(app, symbol)
        if not text:
            with contextlib.suppress(Exception):
                await message.answer("⚠️ Не удалось собрать данные для теханализа. Попробуйте позже.")
            return
        await message.answer(text)
    except Exception:
        with contextlib.suppress(Exception):
            await message.answer("⚠️ Ошибка при подготовке теханализа.")

# ===================== Patch entry =====================
def patch(app: Dict[str, Any]) -> None:
    """
    - «🧭 Технический анализ монеты» в меню
    - /tacoin (private only)
    - отдельный лимит (ta_count/ta_date)
    - перехват pending до общего fallback
    - поддержка: кнопка «🛟 Поддержка» в группах/супергруппах (устойчивый матчинг)
    """
    logger = app.get("logger")
    router = app.get("router")

    # ---- Меню ----
    orig_menu = app.get("main_menu_kb")
    def _menu_with_ta(is_admin: bool = False):
        try:
            kb = [
                [KeyboardButton(text="📈 Получить сигнал")],
                [KeyboardButton(text="🔎 Анализ монеты")],
                [KeyboardButton(text="🧭 Технический анализ монеты")],
                [KeyboardButton(text="ℹ️ Помощь")],
                [KeyboardButton(text="Поддержка")],
            ]
            return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)
        except Exception:
            if callable(orig_menu):
                return orig_menu(is_admin)
            return None
    app["main_menu_kb"] = _menu_with_ta
    logger and logger.info("TA: main menu patched (added 🧭 TA, emoji for Support).")

    # ---- /tacoin ----
    async def _h_tacoin_cmd(message: Message):
        try:
            if getattr(message.chat, "type", None) != "private":
                return
            bot = app.get("bot_instance")
            guard_access = app.get("guard_access")
            st = await guard_access(message, bot)
            if not st:
                return
            parts = (message.text or "").split(maxsplit=1)
            if len(parts) > 1:
                sym = _resolve_symbol_any(app, parts[1])
                if not sym:
                    await message.answer("Монета не распознана. Пример: BTC, ETH, SOL.")
                    return
                await _do_ta_flow(app, message, bot, message.from_user.id, sym)
            else:
                _TA_PENDING.add(message.from_user.id)
                await message.answer("Введите название монеты (например, BTC или ETH) для технического анализа")
        except Exception:
            logger and logger.exception("TA: /tacoin error")
    router.message.register(_h_tacoin_cmd, F.chat.type == "private", F.text.startswith("/tacoin"))

    # ---- Кнопка ----
    async def _h_ta_button(message: Message):
        try:
            if getattr(message.chat, "type", None) != "private":
                return
            bot = app.get("bot_instance")
            guard_access = app.get("guard_access")
            st = await guard_access(message, bot)
            if not st:
                return
            _TA_PENDING.add(message.from_user.id)
            await message.answer("Введите название монеты (например, BTC или ETH) для технического анализа")
        except Exception:
            logger and logger.exception("TA: button handler error")
    router.message.register(_h_ta_button, F.chat.type == "private", F.text == "🧭 Технический анализ монеты")

    # ---- Pending тикер ----
    async def _h_ta_pending(message: Message):
        try:
            if getattr(message.chat, "type", None) != "private":
                return
            uid = int(message.from_user.id) if message.from_user else 0
            if uid not in _TA_PENDING:
                return
            if not message.text or message.text.startswith("/"):
                return
            bot = app.get("bot_instance")
            _TA_PENDING.discard(uid)
            sym = _resolve_symbol_any(app, message.text)
            if not sym:
                await message.answer("Монета не распознана. Пример: BTC, ETH, SOL.")
                return
            await _do_ta_flow(app, message, bot, uid, sym)
        except Exception:
            logger and logger.exception("TA: pending handler error")
    router.message.register(_h_ta_pending, F.chat.type == "private")

    # ---- Поддержка (emoji) ----
    async def _h_support_emoji(message: Message):
        try:
            db = app.get("db")
            bot = app.get("bot_instance")
            support_kb = app.get("support_kb")
            if db:
                await db.set_support_mode(message.from_user.id, True)
            with contextlib.suppress(Exception):
                if callable(support_kb):
                    await bot.send_message(message.chat.id, "Напишите ваш вопрос", reply_markup=support_kb())
                else:
                    await bot.send_message(message.chat.id, "Напишите ваш вопрос")
        except Exception:
            logger and logger.exception("TA: support emoji handler error")

    # 1) Прямой хендлер для групп/супергрупп — устойчивый регэксп (эмодзи/пробелы/регистр)
    router.message.register(
        _h_support_emoji,
        F.chat.type.in_({"group", "supergroup"}),
        F.text.regexp(r"(?i)^\s*(?:🛟\s*)?поддержка\s*$")
    )

    # ---- Fallback wrap: приоритет pending/кнопки + поддержка для групп ----
    try:
        obs = router.message
        handlers = getattr(obs, "handlers", [])
        target_fb = None
        for h in handlers:
            cb = getattr(h, "callback", None)
            if cb and "fallback" in getattr(cb, "__name__", ""):
                target_fb = h
                break
        if target_fb:
            orig_fallback = target_fb.callback
            async def fallback_with_ta(message: Message, bot):
                try:
                    txt = (message.text or "").strip()
                    low = txt.lower()
                    uid = message.from_user.id if message.from_user else 0
                    chat_type = getattr(message.chat, "type", None)

                    # Группы: перехват «поддержка» в любом удобном написании
                    if chat_type in {"group", "supergroup"} and "поддержка" in low:
                        await _h_support_emoji(message)
                        return

                    # Приват: приоритет кнопки TA и pending тикера
                    if chat_type == "private":
                        if txt == "🧭 Технический анализ монеты":
                            await _h_ta_button(message); return
                        if uid in _TA_PENDING and txt and not txt.startswith("/"):
                            await _h_ta_pending(message); return

                    await orig_fallback(message, bot)
                except Exception:
                    with contextlib.suppress(Exception):
                        await orig_fallback(message, bot)
            setattr(target_fb, "callback", fallback_with_ta)
            logger and logger.info("TA: fallback wrapped (pending first + group support intercept).")
    except Exception as e:
        logger and logger.warning("TA: fallback wrap error: %s", e)

    logger and logger.info("TA patch loaded: /tacoin, rich tech report, limits, pending fix.")
