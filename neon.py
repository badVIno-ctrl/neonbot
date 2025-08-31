# neon.py
from __future__ import annotations
import os
import asyncio
import contextlib
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple, List, Set

from aiogram import F, BaseMiddleware
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton

# ===================== ENV / CONST =====================
NEON_SESSION_TIMEOUT_SEC = int(os.getenv("NEON_SESSION_TIMEOUT_SEC", "900"))       # 15 мин — таймаут сессии
NEON_ANALYSIS_DAILY_LIMIT = int(os.getenv("NEON_ANALYSIS_DAILY_LIMIT", "5"))       # 5 анализов/сутки
NEON_MAX_LIST_SHOW = int(os.getenv("NEON_MAX_LIST_SHOW", "50"))                    # максимум имён в списках
NEON_NOTIFY_COOLDOWN_SEC = int(os.getenv("NEON_NOTIFY_COOLDOWN_SEC", "300"))       # антиспам уведомлений админам (сек)

# ===================== RUNTIME STATE =====================
_SESSIONS: Dict[int, Dict[str, float]] = {}     # user_id -> {"last": ts, "notified": ts}
_NEON_COIN_PENDING: Set[int] = set()            # ожидание ввода тикера пользователем

# ===================== HELPERS =====================
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _user_display_name(app: Dict[str, Any], u: Any) -> str:
    f = app.get("user_display_name")
    if callable(f):
        try:
            return f(u)
        except Exception:
            pass
    if not u:
        return "пользователь"
    if getattr(u, "username", None):
        return f"@{u.username}"
    fn = (getattr(u, "first_name", "") or "").strip()
    ln = (getattr(u, "last_name", "") or "").strip()
    return (fn + " " + ln).strip() or "пользователь"

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
    return candidate  # последний шанс

async def _ensure_user_row_and_touch(app: Dict[str, Any], user_id: int) -> None:
    db = app.get("db")
    if not db or not getattr(db, "conn", None):
        return
    with contextlib.suppress(Exception):
        await db.get_user_state(user_id)
    try:
        await db.conn.execute("ALTER TABLE users ADD COLUMN last_seen TEXT")
        await db.conn.commit()
    except Exception:
        pass
    try:
        await db.conn.execute("UPDATE users SET last_seen=? WHERE user_id=?", (_now_utc().isoformat(), user_id))
        await db.conn.commit()
    except Exception:
        pass

async def _ensure_analysis_columns(app: Dict[str, Any]) -> None:
    db = app.get("db")
    if not db or not getattr(db, "conn", None):
        return
    for s in (
        "ALTER TABLE users ADD COLUMN analysis_count INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE users ADD COLUMN analysis_date TEXT",
    ):
        with contextlib.suppress(Exception):
            await db.conn.execute(s)
            await db.conn.commit()

def _today_key_msk(app: Dict[str, Any]) -> str:
    f = app.get("today_key")
    if callable(f):
        try:
            return f()
        except Exception:
            pass
    now = datetime.now(timezone.utc) + timedelta(hours=3)
    return now.strftime("%Y-%m-%d")

async def _analysis_allowed_and_inc(app: Dict[str, Any], user_id: int, st: Dict[str, Any]) -> Tuple[bool, int, int]:
    await _ensure_analysis_columns(app)
    db = app.get("db")
    if not db or not getattr(db, "conn", None):
        return False, 0, NEON_ANALYSIS_DAILY_LIMIT
    if bool(st.get("unlimited") or st.get("admin")):
        return True, 0, 0
    dkey = _today_key_msk(app)
    try:
        cur = await db.conn.execute("SELECT analysis_count, analysis_date FROM users WHERE user_id=?", (user_id,))
        row = await cur.fetchone()
        if not row:
            await db.get_user_state(user_id)
            cur = await db.conn.execute("SELECT analysis_count, analysis_date FROM users WHERE user_id=?", (user_id,))
            row = await cur.fetchone()
        cnt = int(row["analysis_count"] if row and row["analysis_count"] is not None else 0)
        date = str(row["analysis_date"] or "")
        if date != dkey:
            cnt = 0
        if cnt >= NEON_ANALYSIS_DAILY_LIMIT:
            return False, cnt, NEON_ANALYSIS_DAILY_LIMIT
        cnt2 = cnt + 1
        await db.conn.execute("UPDATE users SET analysis_count=?, analysis_date=? WHERE user_id=?", (cnt2, dkey, user_id))
        await db.conn.commit()
        return True, cnt2, NEON_ANALYSIS_DAILY_LIMIT
    except Exception:
        return False, 0, NEON_ANALYSIS_DAILY_LIMIT

def _format_brief_header(details: Dict[str, Any], base: str, side_score: Optional[float]) -> str:
    p_bayes = details.get("p_bayes")
    if isinstance(p_bayes, (int, float)):
        return f"🔎 Анализ {base} • направление {details.get('side')} • уверенность {p_bayes:.2f}"
    if isinstance(side_score, (int, float)):
        return f"🔎 Анализ {base} • направление {details.get('side')} • оценка {side_score:.2f}"
    return f"🔎 Анализ {base} • направление {details.get('side')}"

def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if (v == v) else float(default)
    except Exception:
        return float(default)

def _fmt_price(app: Dict[str, Any], x: float) -> str:
    f = app.get("format_price") or (lambda v: f"{v:.4f}")
    try:
        return f(float(x))
    except Exception:
        return f"{x:.4f}"

def _extract_news_tops(app: Dict[str, Any], note: str) -> List[str]:
    parser = app.get("_parse_news_note")
    if callable(parser):
        try:
            _neg, _pos, tops = parser(note)
            return tops or []
        except Exception:
            return []
    return []

def _pct_of_entry(side: str, entry: float, price: float) -> float:
    if not entry:
        return 0.0
    if side == "LONG":
        return (price - entry) / (entry + 1e-9) * 100.0
    else:
        return (entry - price) / (entry + 1e-9) * 100.0

def _make_pros_cons(details: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    pros, cons = [], []
    if details.get("cond4h_up") and details.get("cond1h_up"):
        pros.append("бычий MTF (4h/1h > EMA200)")
    elif details.get("cond4h_up") or details.get("cond1h_up"):
        pros.append("смешанный MTF (частично > EMA200)")
    else:
        cons.append("слабый MTF (4h/1h < EMA200)")
    if details.get("cond15_up"):
        pros.append("15m EMA50 > EMA200 (поддержка тренда)")
    st_dir = int(details.get("st_dir") or 0)
    pros.append("SuperTrend ↑" if st_dir > 0 else ("SuperTrend ↓" if st_dir < 0 else ""))
    bos_dir = int(details.get("bos_dir") or 0)
    if bos_dir == 1: pros.append("BOS↑ (за лонг)")
    if bos_dir == -1: cons.append("BOS↓ (за шорт)")
    if details.get("don_break_long"): pros.append("Donchian↑")
    if details.get("don_break_short"): cons.append("Donchian↓")
    adx = _safe_float(details.get("adx15"))
    if adx >= 50: pros.append(f"ADX {adx:.0f} — сильный тренд")
    elif adx >= 25: pros.append(f"ADX {adx:.0f} — трендовый режим")
    else: cons.append(f"ADX {adx:.0f} — боковой/рваный режим")
    rsi5 = _safe_float(details.get("rsi5"))
    macd_pos = (details.get("macdh5") or 0) > 0
    if rsi5 >= 70: cons.append(f"RSI(5m) {rsi5:.0f} — перекупленность")
    elif rsi5 >= 55: pros.append(f"RSI(5m) {rsi5:.0f} — умеренно бычий")
    elif rsi5 <= 45: cons.append(f"RSI(5m) {rsi5:.0f} — слабый моментум")
    if macd_pos: pros.append("MACD + (подтверждение импульса)")
    else: cons.append("MACD − (импульс не подтверждён)")
    rvol = details.get("rvol_combo")
    if isinstance(rvol, (int, float)):
        if rvol >= 1.5: pros.append(f"RVOL x{rvol:.2f} — сильная активность")
        elif rvol >= 1.2: pros.append(f"RVOL x{rvol:.2f} — достаточная активность")
        elif rvol < 0.9: cons.append(f"RVOL x{rvol:.2f} — ниже нормы")
    vol_z = details.get("vol_z")
    if isinstance(vol_z, (int, float)):
        if vol_z > 1.0: pros.append(f"vol_z {vol_z:.2f} — выделяется по объёму")
        elif vol_z < 0: cons.append(f"vol_z {vol_z:.2f} — активность ниже медианы")
    cvd = details.get("cvd_slope")
    if isinstance(cvd, (int, float)):
        if cvd > 1e6: pros.append("CVD — сильная агрессия покупателей")
        elif cvd < -1e6: cons.append("CVD — сильная агрессия продавцов")
    l2 = details.get("ob_imb")
    if isinstance(l2, (int, float)):
        if l2 > 0.15: pros.append(f"L2-imb {l2:+.2f} — перевес бидов")
        elif l2 < -0.15: cons.append(f"L2-imb {l2:+.2f} — перевес асков")
        else: cons.append(f"L2-imb {l2:+.2f} — слабая поддержка")
    basis = details.get("basis")
    if isinstance(basis, (int, float)):
        if basis > 0.004: cons.append(f"basis {basis*100:+.2f}% — риск шорт-сквиза")
        elif basis < -0.004: cons.append(f"basis {basis*100:+.2f}% — риск лонг-сквиза")
    bdom = details.get("btc_dom_trend")
    if bdom == "down": pros.append("BTC.D down — фон за альты")
    elif bdom == "up": cons.append("BTC.D up — риск для альтов")
    nb = details.get("news_boost")
    if isinstance(nb, (int, float)):
        if nb >= 1.0: pros.append(f"Новости +{nb:.2f} — сильная поддержка")
        elif nb >= 0.4: pros.append(f"Новости +{nb:.2f} — умеренная поддержка")
    br = details.get("score_breakdown") or {}
    if isinstance(br, dict):
        if br.get("sanityRR", 0.0) < 0: cons.append("sanityRR < 0 — математика TP/SL плохая")
        if br.get("microRR1", 0.0) < 0: cons.append("microRR1 < порога — слабый TP1")
    pros = [x for x in pros if x][:6]
    cons = [x for x in cons if x][:6]
    return pros, cons

def _mk_detail_sections_fancy(app: Dict[str, Any], symbol: str, side: str, details: Dict[str, Any], side_score: Optional[float]) -> Tuple[str, str]:
    fm = lambda x: _fmt_price(app, x)
    base = symbol.split("/")[0]
    p_bayes = details.get("p_bayes")
    risk_level = details.get("risk_level")
    lev = details.get("leverage")
    window_txt = "Окно сделки: ~40м–6ч"

    header = f"🔎 <b>{base}</b> — расширенный теханалитический обзор"
    sub = f"Сторона: <b>{side}</b>"
    if isinstance(p_bayes, (int, float)):
        sub += f" • Уверенность: <b>{p_bayes:.2f}</b>"
    elif isinstance(side_score, (int, float)):
        sub += f" • Оценка: <b>{side_score:.2f}</b>"
    sub += f" • Риск-L: <b>{risk_level}/10</b> • Плечо: <b>{lev}x</b>\nℹ️ Это аналитический обзор, не торговый сигнал."

    pros, cons = _make_pros_cons(details)
    # Вывод
    adx = _safe_float(details.get("adx15"))
    rvol = details.get("rvol_combo")
    macd_pos = (details.get("macdh5") or 0) > 0
    concl = []
    if (details.get("cond4h_up") and details.get("cond1h_up") and details.get("cond15_up")) or (adx >= 25 and macd_pos):
        concl.append("идея лонга жизнеспособна при подтверждении объёмом и дисциплине стопа")
    else:
        concl.append("идея требует подтверждения (моментум/объёмы) и аккуратного риск‑менеджмента")
    if isinstance(rvol, (int, float)) and rvol < 1.0:
        concl.append("желателен рост активности: RVOL ≥ 1.2–1.5")
    conclusion = "Вывод: " + "; ".join(concl) + "."

    # Dashboard
    t4h = "↑" if details.get("cond4h_up") else "↓"
    t1h = "↑" if details.get("cond1h_up") else "↓"
    t15 = "↑" if details.get("cond15_up") else "↓"
    st = "↑" if (details.get("st_dir") or 0) > 0 else ("↓" if (details.get("st_dir") or 0) < 0 else "•")
    adx_txt = f"{_safe_float(details.get('adx15')):.0f}"
    r2_txt = f"{_safe_float(details.get('r2_1h')):.2f}"
    atr = _safe_float(details.get("atr"))
    atr_pct = _safe_float(details.get("atr_pct"))
    rsi5 = _safe_float(details.get("rsi5"))
    macd_txt = "+" if (details.get("macdh5") or 0) > 0 else "-"
    vol_z = details.get("vol_z")
    rvol_txt = details.get("rvol_combo")
    cvd = details.get("cvd_slope")
    ob_imb = details.get("ob_imb")
    basis = details.get("basis")
    bdom = details.get("btc_dom_trend")
    news_boost = details.get("news_boost")

    dash_lines = [
        "📊 <b>Пульс</b>",
        f"• MTF: 4h {t4h} • 1h {t1h} • 15m {t15}",
        f"• ST: {st} • ADX: {adx_txt} • R2(1h): {r2_txt}",
        f"• ATR(15m): {atr:.4f} ({atr_pct:.2f}%)",
        f"• RSI(5m): {rsi5:.0f} • MACD: {macd_txt}",
    ]
    flows = []
    if isinstance(vol_z, (int, float)): flows.append(f"vol_z {vol_z:.2f}")
    if isinstance(rvol_txt, (int, float)): flows.append(f"RVOL x{rvol_txt:.2f}")
    if isinstance(cvd, (int, float)): flows.append(f"CVD {cvd:+.0f}")
    if isinstance(ob_imb, (int, float)): flows.append(f"L2-imb {ob_imb:+.2f}")
    if isinstance(basis, (int, float)): flows.append(f"basis {basis*100:+.2f}%")
    if isinstance(news_boost, (int, float)): flows.append(f"News +{news_boost:.2f}")
    if bdom: flows.append(f"BTC.D {bdom}")
    if flows:
        dash_lines.append("• " + " • ".join(flows))

    # Уровни
    lvl_lines = []
    pdh = details.get("PDH"); pdl = details.get("PDL")
    poc = details.get("poc"); vah = details.get("vah"); val = details.get("val")
    round_level = details.get("round_level")
    ib_hi = details.get("ib_hi"); ib_lo = details.get("ib_lo")
    if pdh: lvl_lines.append(f"PDH {fm(pdh)}")
    if pdl: lvl_lines.append(f"PDL {fm(pdl)}")
    if ib_hi and ib_lo: lvl_lines.append(f"IB {fm(ib_hi)}/{fm(ib_lo)}")
    if poc is not None and vah is not None and val is not None:
        lvl_lines.append(f"POC {fm(poc)}, VAH {fm(vah)}, VAL {fm(val)}")
    if round_level: lvl_lines.append(f"Round {fm(round_level)}")
    lvl_txt = ("• " + " • ".join(lvl_lines)) if lvl_lines else "• n/a"

    # Part 1
    part1 = "\n".join([
        f"{header}",
        "━━━━━━━━━━━━━━━━━━",
        sub,
        "",
        "✨ <b>Краткое резюме</b>",
        ("За:\n" + "\n".join([f"• ✅ {s}" for s in pros])) if pros else "",
        ("Против:\n" + "\n".join([f"• ❌ {s}" for s in cons])) if cons else "",
        conclusion,
        "",
        *dash_lines,
        "",
        "🗺 <b>Уровни / профиль</b>",
        lvl_txt,
        "",
        window_txt,
    ])[:3500]

    # Part 2 — риск/сценарии
    entry = _safe_float(details.get("c5"))
    sl = _safe_float(details.get("sl"))
    tps = [float(x) for x in (details.get("tps") or [])]
    risk_pct = abs(entry - sl) / (entry + 1e-9) * 100.0 if (entry and sl) else 0.0
    rr1 = details.get("rr1"); rr2 = details.get("rr2")

    alt_mult = 1.2
    atr = _safe_float(details.get("atr"))
    alt_stop_usd = alt_mult * atr
    alt_stop_pct = (alt_stop_usd / (entry + 1e-9)) * 100.0 if entry else 0.0
    R1, R2, R3 = 1.0, 1.75, 2.5
    if side == "LONG":
        tp1_p, tp2_p, tp3_p = entry + R1*alt_stop_usd, entry + R2*alt_stop_usd, entry + R3*alt_stop_usd
    else:
        tp1_p, tp2_p, tp3_p = entry - R1*alt_stop_usd, entry - R2*alt_stop_usd, entry - R3*alt_stop_usd

    scen = [
        "A) Продолжение тренда (breakout/continuation)",
        "• Триггеры: 15m закрытие выше локального хая при RVOL ≥ 1.2–1.5, рост CVD; L2-imb не уходит в минус.",
        "• Вход: на ретесте пробитого уровня или после микро‑консолидации над ним.",
        f"• Стоп: 1.0–1.5×ATR ({atr:.4f} ⇒ {atr*1.0:.4f}–{atr*1.5:.4f}). Цели: +1R / +1.75R / +2.5R. Плечо: 3–6x (консервативно).",
        "B) Откат к поддержке (buy the dip по тренду)",
        "• Триггеры: откат к 15m EMA50/зоне баланса при удержании 1h > EMA200; свечной ре‑клейм уровня; рост CVD.",
        "• Вход: над восстановленным уровнем, подтверждение объёмом (RVOL).",
        "• Стоп: за свинг‑лоу отката или 1.2–1.5×ATR. Цели: к локальному максимуму дня, затем к пивотам (R1/R2).",
    ]
    inv = [
        "• 15m закрытия под 1h EMA200 или перелом структуры (LL/LH).",
        "• Срыв CVD вниз при росте RVOL (агрессия продавца).",
        "• Резкий рост BTC.D при падающих объёмах по активу.",
    ]
    mgmt = [
        "• Риск на сделку: 0.5–1.0% от депозита.",
        "• Перенос в БУ: после достижения +1.0R.",
        "• Частичная фиксация: 30–40% на TP1, 30–40% на TP2, остаток — трейлинг 1.0–1.5×ATR.",
        "• Временной стоп: если за 40м–6ч нет ≥0.8×ATR в вашу сторону и RVOL не растёт — переоценить/перезайти.",
    ]
    chk = [
        "• RVOL ≥ 1.2 на сигнальной свече/пробое.",
        "• CVD растёт; нет налёта продавца.",
        "• L2-imb ≥ 0 и не ухудшается на пробое.",
        "• 4h/1h остаются > EMA200; 15m удерживает EMA50.",
        "• Стоп ≤ 1.5×ATR и соответствует вашему Risk%.",
        ("• Фон новостей позитивный — учитывайте ускорения/волатильность." if isinstance(details.get("news_boost"), (int, float)) and details["news_boost"] >= 1.0 else "• Нет негативных новостей (News)."),
    ]
    cmt = []
    br = details.get("score_breakdown") or {}
    if br.get("HistoryNN", 0) > 0 or details.get("nn_edge"):
        cmt.append(f"• HistoryNN {details.get('nn_edge', 0):+,.2f}% — лёгкая историческая поддержка.")
    if details.get("nn_edge_dtw"):
        cmt.append(f"• DTW {details.get('nn_edge_dtw', 0):+,.2f}% — подтверждает паттерн движения.")
    if isinstance(basis, (int, float)) and abs(basis) < 0.001:
        cmt.append("• basis ~0% — нет перегрева деривативов.")
    if br.get("sanityRR", 0) < 0 or br.get("microRR1", 0) < 0:
        cmt.append("• sanityRR/microRR: пересоберите TP/SL под ATR/R (см. альтернативу ниже).")

    lev_warn = ""
    lev_val = int(details.get("leverage") or 0)
    if lev_val >= 10 and risk_pct > 3.0:
        lev_warn = "⚠️ При плече ≥10x глубокий стоп может не дожить до SL (ликвидация раньше) — держите стоп ≤ 1.5×ATR."

    now_math = []
    if entry:
        now_math.append(f"• TBX: {fm(entry)} • SL: {fm(sl)} ({risk_pct:.2f}% от входа)")
    if tps:
        tparts = [f"TP{i} {fm(tp)} ({_pct_of_entry(side, entry, tp):.2f}%)" for i, tp in enumerate(tps[:3], 1)]
        now_math.append("• " + " • ".join(tparts))
    if isinstance(rr1, (int, float)) or isinstance(rr2, (int, float)):
        now_math.append(f"• microRR: RR1 {rr1 if rr1 is not None else 0:.2f} • RR2 {rr2 if rr2 is not None else 0:.2f}")
    if lev_warn:
        now_math.append(lev_warn)

    alt = [
        "Альтернатива (по ATR/R):",
        f"• Рекомендуемый стоп: <b>{alt_mult:.1f}×ATR</b> ≈ {alt_stop_usd:.4f}$ (~{alt_stop_pct:.2f}%)",
        f"• Цели: 1.0R → {fm(tp1_p)} ({_pct_of_entry(side, entry, tp1_p):.2f}%), 1.75R → {fm(tp2_p)} ({_pct_of_entry(side, entry, tp2_p):.2f}%), 2.5R → {fm(tp3_p)} ({_pct_of_entry(side, entry, tp3_p):.2f}%)",
    ]

    part2 = "\n".join([
        f"🧭 <b>Риск‑менеджмент и сценарии • {base}</b>",
        "━━━━━━━━━━━━━━━━━━",
        "Текущие параметры:" if now_math else "",
        *now_math,
        "",
        *alt,
        "",
        "Сценарии отработки:",
        *[("• " + s if not s.startswith(("A)", "B)")) else s) for s in scen],
        "",
        "Инвалидация идеи:",
        *inv,
        "",
        "Менеджмент позиции:",
        *mgmt,
        "",
        "Чек‑лист перед входом:",
        *chk,
        "",
        "Комментарии к метрикам:" if cmt else "",
        *cmt[:6],
        "",
        "⚠️ Помните: это аналитика, а не торговая рекомендация.",
    ])[:3500]

    return part1, part2

async def _send_admin_user_overview(app: Dict[str, Any], admin_id: int, bot=None):
    logger = app.get("logger")
    db = app.get("db")
    bot = bot or app.get("bot_instance")
    if not db or not getattr(db, "conn", None) or not bot:
        return
    with contextlib.suppress(Exception):
        await db.conn.execute("ALTER TABLE users ADD COLUMN last_seen TEXT")
        await db.conn.commit()
    now = _now_utc()
    cutoff_now = (now - timedelta(seconds=NEON_SESSION_TIMEOUT_SEC)).isoformat()
    cutoff_24h = (now - timedelta(hours=24)).isoformat()

    try:
        cur = await db.conn.execute("SELECT user_id FROM users WHERE last_seen IS NOT NULL AND last_seen >= ? ORDER BY last_seen DESC", (cutoff_now,))
        rows_now = await cur.fetchall()
        ids_now = [int(r["user_id"]) for r in rows_now] if rows_now else []
    except Exception:
        ids_now = []
    try:
        cur = await db.conn.execute("SELECT user_id FROM users WHERE last_seen IS NOT NULL AND last_seen >= ? ORDER BY last_seen DESC", (cutoff_24h,))
        rows_24 = await cur.fetchall()
        ids_24 = [int(r["user_id"]) for r in rows_24] if rows_24 else []
    except Exception:
        ids_24 = []

    async def _names(id_list: List[int]) -> List[str]:
        out = []
        for uid in id_list[:NEON_MAX_LIST_SHOW]:
            try:
                ch = await bot.get_chat(uid)
                out.append(_user_display_name(app, ch))
            except Exception:
                out.append(f"id {uid}")
        return out

    names_now = await _names(ids_now)
    names_24 = await _names(ids_24)
    more_now = max(0, len(ids_now) - len(names_now))
    more_24 = max(0, len(ids_24) - len(names_24))

    msg1 = "\n".join([
        "👥 Активны сейчас (≤ 15 мин): " + str(len(ids_now)),
        "━━━━━━━━━━━━━━━━━━",
        *([f"• {nm}" for nm in names_now] if names_now else ["• нет активных сейчас"]),
        *(["… и ещё " + str(more_now)] if more_now > 0 else []),
    ])
    msg2 = "\n".join([
        "📅 За 24 часа: " + str(len(ids_24)),
        "━━━━━━━━━━━━━━━━━━",
        *([f"• {nm}" for nm in names_24] if names_24 else ["• не было активности"]),
        *(["… и ещё " + str(more_24)] if more_24 > 0 else []),
    ])
    with contextlib.suppress(Exception):
        await bot.send_message(admin_id, msg1)
    with contextlib.suppress(Exception):
        await bot.send_message(admin_id, msg2)

async def _touch_session_and_maybe_notify(app: Dict[str, Any], user: Any) -> None:
    import time as _time
    user_id = int(getattr(user, "id", 0) or 0)
    if user_id <= 0:
        return
    await _ensure_user_row_and_touch(app, user_id)

    ts_now = _time.time()
    st = _SESSIONS.get(user_id, {"last": 0.0, "notified": 0.0})
    inactive = (ts_now - st.get("last", 0.0)) >= NEON_SESSION_TIMEOUT_SEC
    st["last"] = ts_now
    _SESSIONS[user_id] = st

    # «одно уведомление за сессию» + антиспам кулдаун
    if inactive and (ts_now - st.get("notified", 0.0) >= NEON_NOTIFY_COOLDOWN_SEC):
        bot = app.get("bot_instance")
        db = app.get("db")
        logger = app.get("logger")
        if not bot or not db:
            return
        try:
            admins = await db.get_admin_user_ids()
        except Exception:
            admins = []
        if not admins:
            return
        name = _user_display_name(app, user)
        text = f"пользователь {name} находится в боте"
        for aid in admins:
            with contextlib.suppress(Exception):
                await bot.send_message(aid, text)
        st["notified"] = ts_now
        _SESSIONS[user_id] = st
        logger and logger.info("NEON: session start notify sent for user_id=%s", user_id)

# ===================== ANALYSIS (ONLY ANALYSIS) =====================
async def _analysis_only_flow(app: Dict[str, Any], message: Message, bot, user_id: int, symbol: str):
    logger = app.get("logger")
    EXECUTOR = app.get("EXECUTOR")
    build_reason_fn = app.get("build_reason")

    st_guard = await app.get("guard_access")(message, bot)
    if not st_guard:
        return

    ok, used, limit = await _analysis_allowed_and_inc(app, user_id, st_guard)
    if not ok:
        with contextlib.suppress(Exception):
            await message.answer(f"⛔ Лимит анализа исчерпан: {limit}/день. Попробуйте завтра или получите сигнал: /signal")
        return

    base = symbol.split("/")[0]
    with contextlib.suppress(Exception):
        await message.answer(f"⏳ Делаю анализ {base}…")

    loop = asyncio.get_running_loop()
    side = None
    side_score = None
    details: Dict[str, Any] = {}
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
        logger and logger.exception("NEON analysis failed for %s: %s", symbol, e)
        with contextlib.suppress(Exception):
            await message.answer("⚠️ Не удалось выполнить анализ. Попробуйте позже.")
        return

    details["symbol"] = symbol
    header = _format_brief_header(details, base, side_score)
    try:
        reason = build_reason_fn(details) if callable(build_reason_fn) else ""
    except Exception:
        reason = ""
    tail = "\n\nℹ️ Это аналитический обзор, не торговый сигнал."
    text = f"{header}\n{reason}{tail}"

    cb_data = f"neon:detail:{symbol.replace('/', '_')}"
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔬 Сделать более подробный анализ", callback_data=cb_data)]
    ])
    with contextlib.suppress(Exception):
        await message.answer(text, reply_markup=kb)

# ===================== INTERNAL DETAIL HANDLER =====================
async def _handle_neon_detail(app: Dict[str, Any], cb: CallbackQuery):
    logger = app.get("logger")
    with contextlib.suppress(Exception):
        await cb.answer("Готовлю подробный анализ…")
    sym_enc = cb.data.split(":", 2)[-1]
    symbol = sym_enc.replace("_", "/")
    try:
        EXECUTOR = app.get("EXECUTOR")
        loop = asyncio.get_running_loop()
        score_symbol_core = app.get("score_symbol_core")
        res = await loop.run_in_executor(EXECUTOR, score_symbol_core, symbol, False)
        if res is None:
            res = await loop.run_in_executor(EXECUTOR, score_symbol_core, symbol, True)
        if res is None:
            with contextlib.suppress(Exception):
                await cb.message.answer("⚠️ Не удалось подготовить подробный анализ. Попробуйте позже.")
            return
        side_score, side, details = res
        details = dict(details or {})
        details["symbol"] = symbol
        part1, part2 = _mk_detail_sections_fancy(app, symbol, side, details, side_score)
        with contextlib.suppress(Exception):
            await cb.message.answer(part1)
        with contextlib.suppress(Exception):
            await cb.message.answer(part2)
        logger and logger.info("NEON: detail sent for %s", symbol)
    except Exception:
        logger and logger.exception("NEON: detail error")
        with contextlib.suppress(Exception):
            await cb.answer("Ошибка", show_alert=False)

# ===================== Middleware для трекинга активности =====================
class NeonSessionMiddleware(BaseMiddleware):
    def __init__(self, app: Dict[str, Any]):
        super().__init__()
        self.app = app

    async def __call__(self, handler, event, data):
        try:
            # Message in private
            m = event if isinstance(event, Message) else getattr(event, "message", None)
            if m and getattr(m.chat, "type", None) == "private" and m.from_user and not m.from_user.is_bot:
                await _touch_session_and_maybe_notify(self.app, m.from_user)
            # Callback in private
            if isinstance(event, CallbackQuery) and event.from_user and not event.from_user.is_bot:
                msg = event.message
                if msg and getattr(msg.chat, "type", None) == "private":
                    await _touch_session_and_maybe_notify(self.app, event.from_user)
        except Exception:
            pass
        return await handler(event, data)

# ===================== Антидубли для утренних постов (once per day) =====================
def _install_daily_once_guards(app: Dict[str, Any]) -> None:
    import sys
    logger = app.get("logger")
    once = app.setdefault("_daily_once", {"lock": asyncio.Lock(), "sent": set()})

    def _day(app_):
        return app_["now_msk"]().date().isoformat()

    # main._post_morning_report(app, bot, channel_id) — канал 08:00
    m = sys.modules.get("main")
    if m and hasattr(m, "_post_morning_report"):
        orig_morning = m._post_morning_report

        async def _post_morning_report_once(app_, bot, channel_id):
            key = ("channel", str(channel_id), _day(app_))
            async with once["lock"]:
                if key in once["sent"]:
                    logger and logger.info("Morning post: skipped (already sent today) channel=%s", channel_id)
                    return
            await orig_morning(app_, bot, channel_id)
            async with once["lock"]:
                once["sent"].add(key)

        m._post_morning_report = _post_morning_report_once
        logger and logger.info("NEON: guard applied to main._post_morning_report (once/day).")

    # main._post_admin_greetings(app, bot) — утренние сообщения админам (если есть)
    if m and hasattr(m, "_post_admin_greetings"):
        orig_admin = m._post_admin_greetings

        async def _post_admin_greetings_once(app_, bot):
            key = ("admin_daily", "all", _day(app_))
            async with once["lock"]:
                if key in once["sent"]:
                    logger and logger.info("Admin morning digest (main): skipped (once/day).")
                    return
            await orig_admin(app_, bot)
            async with once["lock"]:
                once["sent"].add(key)

        m._post_admin_greetings = _post_admin_greetings_once
        logger and logger.info("NEON: guard applied to main._post_admin_greetings (once/day).")

    # chat._post_admin_greetings_once(app, bot) — второй источник дубля
    c = sys.modules.get("chat")
    if c and hasattr(c, "_post_admin_greetings_once"):
        orig_chat_admin_once = c._post_admin_greetings_once

        async def _chat_post_admin_greetings_once_guard(app_, bot):
            key = ("admin_daily", "all", _day(app_))
            async with once["lock"]:
                if key in once["sent"]:
                    logger and logger.info("Admin morning digest (chat): skipped (once/day).")
                    return
            await orig_chat_admin_once(app_, bot)
            async with once["lock"]:
                once["sent"].add(key)

        c._post_admin_greetings_once = _chat_post_admin_greetings_once_guard
        logger and logger.info("NEON: guard applied to chat._post_admin_greetings_once (once/day).")

# ===================== PATCH ENTRY =====================
def patch(app: Dict[str, Any]) -> None:
    logger = app.get("logger")
    router = app.get("router")
    logger and logger.info("NEON patch подключён: сессии/уведомления админам + анализ-ТОЛЬКО с лимитом и «Подробнее» (WOW-формат) + утренние антидубли.")

    # Middleware для трекинга активности (до всех хэндлеров)
    router.message.middleware(NeonSessionMiddleware(app))
    router.callback_query.middleware(NeonSessionMiddleware(app))

    # Установить антидубли на утренние посты (если функции уже доступны)
    _install_daily_once_guards(app)

    orig_on_startup = app.get("on_startup")
    async def _on_startup_neon(bot):
        app["bot_instance"] = bot
        db = app.get("db")
        if db and db.conn:
            with contextlib.suppress(Exception):
                await db.conn.execute("ALTER TABLE users ADD COLUMN last_seen TEXT")
                await db.conn.commit()
            await _ensure_analysis_columns(app)
        logger and logger.info("NEON: on_startup — модуль активен, колонки подготовлены.")
        # На случай, если main/chat импортированы позже — повторно применим guard
        with contextlib.suppress(Exception):
            _install_daily_once_guards(app)
        if orig_on_startup:
            await orig_on_startup(bot)
    app["on_startup"] = _on_startup_neon

    # Патч /code — отправка обзора пользователей сразу после апгрейда
    try:
        obs = router.message
        handlers = getattr(obs, "handlers", [])
        target = None
        for h in handlers:
            cb = getattr(h, "callback", None)
            if getattr(cb, "__name__", "") == "cmd_code":
                target = h; break
        if target:
            orig_cmd_code = target.callback
            ADMIN_ACCESS_CODE = app.get("ADMIN_ACCESS_CODE", "2604")
            async def cmd_code_patched(message, command, bot):
                await orig_cmd_code(message, command, bot)
                args = (getattr(command, "args", None) or "").strip()
                if args == ADMIN_ACCESS_CODE:
                    with contextlib.suppress(Exception):
                        await _send_admin_user_overview(app, message.from_user.id, bot)
            setattr(target, "callback", cmd_code_patched)
            logger and logger.info("NEON: cmd_code patched — отправка обзора пользователей после апгрейда.")
    except Exception as e:
        logger and logger.warning("NEON: cmd_code patch error: %s", e)

    # Анализ монеты — fallback перехват
    try:
        obs = router.message
        handlers = getattr(obs, "handlers", [])
        target_fb = None
        for h in handlers:
            cb = getattr(h, "callback", None)
            if "fallback" in getattr(cb, "__name__", ""):
                target_fb = h; break
        if not target_fb:
            logger and logger.warning("NEON: fallback handler not found — регистрирую прямые хендлеры /coin и ввода тикера.")
            async def _h_coin_cmd(message: Message):
                if getattr(message.chat, "type", None) != "private":
                    return
                txt = (message.text or "").strip()
                if not (txt.lower().startswith("/coin") or txt == "🔎 Анализ монеты" or "анализ монеты" in txt.lower()):
                    return
                st = await app.get("guard_access")(message, app.get("bot_instance"))
                if not st:
                    return
                parts = txt.split(maxsplit=1)
                if len(parts) > 1 and parts[0].lower().startswith("/coin"):
                    sym = _resolve_symbol_any(app, parts[1])
                    if not sym:
                        await message.answer("Монета не распознана. Пример: BTC, ETH, SOL.")
                        return
                    await _analysis_only_flow(app, message, app.get("bot_instance"), message.from_user.id, sym)
                    return
                _NEON_COIN_PENDING.add(message.from_user.id)
                await message.answer("Введите название монеты (например, BTC или ETH)")
            async def _h_coin_pending(message: Message):
                if getattr(message.chat, "type", None) != "private":
                    return
                uid = message.from_user.id if message.from_user else 0
                if uid in _NEON_COIN_PENDING and message.text and not message.text.startswith("/"):
                    st = await app.get("guard_access")(message, app.get("bot_instance"))
                    if not st:
                        return
                    sym = _resolve_symbol_any(app, message.text)
                    if not sym:
                        await message.answer("Монета не распознана. Пример: BTC, ETH, SOL.")
                        return
                    _NEON_COIN_PENDING.discard(uid)
                    await _analysis_only_flow(app, message, app.get("bot_instance"), uid, sym)
            router.message.register(_h_coin_cmd, F.chat.type == "private")
            router.message.register(_h_coin_pending, F.chat.type == "private")
        else:
            orig_fallback = target_fb.callback
            async def fallback_neon(message: Message, bot):
                try:
                    uid = message.from_user.id if message.from_user else 0
                    text = (message.text or "").strip()
                    low = text.lower()
                    if getattr(message.chat, "type", None) == "private" and (text == "🔎 Анализ монеты" or "анализ монеты" in low or low.startswith("/coin")):
                        st = await app.get("guard_access")(message, bot)
                        if not st:
                            return
                        parts = text.split(maxsplit=1)
                        if len(parts) > 1 and parts[0].lower().startswith("/coin"):
                            sym = _resolve_symbol_any(app, parts[1])
                            if not sym:
                                await message.answer("Монета не распознана. Пример: BTC, ETH, SOL.")
                                return
                            await _analysis_only_flow(app, message, bot, uid, sym)
                            return
                        _NEON_COIN_PENDING.add(uid)
                        await message.answer("Введите название монеты (например, BTC или ETH)")
                        return
                    if getattr(message.chat, "type", None) == "private" and uid in _NEON_COIN_PENDING and text and not text.startswith("/"):
                        st = await app.get("guard_access")(message, bot)
                        if not st:
                            return
                        sym = _resolve_symbol_any(app, text)
                        if not sym:
                            await message.answer("Монета не распознана. Пример: BTC, ETH, SOL.")
                            return
                        _NEON_COIN_PENDING.discard(uid)
                        await _analysis_only_flow(app, message, bot, uid, sym)
                        return
                    await orig_fallback(message, bot)
                except Exception:
                    with contextlib.suppress(Exception):
                        await orig_fallback(message, bot)
            setattr(target_fb, "callback", fallback_neon)
            logger and logger.info("NEON: fallback перехвачен — только анализ с лимитом и кнопкой «Подробнее».")
    except Exception as e:
        logger and logger.warning("NEON: fallback patch error: %s", e)

    # ❶ Handler под neon:detail
    async def _h_detail(cb: CallbackQuery):
        if cb.data and cb.data.startswith("neon:detail:"):
            await _handle_neon_detail(app, cb)
    router.callback_query.register(_h_detail, F.data.startswith("neon:detail:"))
    logger and logger.info("NEON: direct callback handler for neon:detail registered.")

    # ❷ Обёртка общего callback‑хендлера из chat.py, чтобы перехватить neon:detail первее
    try:
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
            async def _h_cb_wrapped(cb: CallbackQuery):
                data = cb.data or ""
                if data.startswith("neon:detail:"):
                    await _handle_neon_detail(app, cb)
                    return
                return await orig_h_cb(cb)
            setattr(target_cb, "callback", _h_cb_wrapped)
            logger and logger.info("NEON: chat._h_cb wrapped to handle neon:detail first.")
        else:
            logger and logger.warning("NEON: chat._h_cb not found; rely on direct handler.")
    except Exception as e:
        logger and logger.warning("NEON: wrap chat._h_cb failed: %s", e)
