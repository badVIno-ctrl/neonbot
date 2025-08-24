# chat.py
from __future__ import annotations
import os
import re
import math
import asyncio
import contextlib
import random
import requests
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple, List

from aiogram import F
from aiogram.types import (
    Message, ChatPermissions, ChatMemberUpdated,
    InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
)

# ====================== ENV / CONFIG ======================
MOD_WARN_LIMIT = int(os.getenv("MOD_WARN_LIMIT", "3"))
MOD_AUTOMUTE_MIN = int(os.getenv("MOD_AUTOMUTE_MIN", "30"))
MOD_FLOOD_N = int(os.getenv("MOD_FLOOD_N", "6"))
MOD_FLOOD_T = int(os.getenv("MOD_FLOOD_T", "30"))
MOD_LOG = bool(int(os.getenv("MOD_LOG", "1")))

# Совместимость: автокоммент включается, если хотя бы одна переменная =1
MOD_AUTOCOMMENT = bool(int(os.getenv("MOD_AUTOCOMMENT", "0")))
MOD_REPLY_AUTOCOMMENT = bool(int(os.getenv("MOD_REPLY_AUTOCOMMENT", "0")))
AUTOCOMMENT_ENABLED = MOD_AUTOCOMMENT or MOD_REPLY_AUTOCOMMENT

MOD_MAX_PRICE_COINS = int(os.getenv("MOD_MAX_PRICE_COINS", "10"))
CMD_COOLDOWN_SEC = int(os.getenv("CMD_COOLDOWN_SEC", "20"))

# Warn-decay: через N дней без нарушений снимаем 1 предупреждение
WARN_DECAY_DAYS = int(os.getenv("WARN_DECAY_DAYS", "14"))

# CAPTCHA/hold для новичков
NEWUSER_CAPTCHA = bool(int(os.getenv("NEWUSER_CAPTCHA", "1")))
NEWUSER_HOLD_MIN = int(os.getenv("NEWUSER_HOLD_MIN", "8"))

# Разрешаем команды в группах без проверки подписки
ALLOW_CHAT_COMMANDS_WITHOUT_SUB = True

# Авто-удаление нарушающих сообщений
DELETE_ON_VIOLATION = bool(int(os.getenv("DELETE_ON_VIOLATION", "1")))

# ====== TA booster ENV (мягкие флаги/пороги — всё в chat.py, ядро не трогаем) ======
TA_G_RR_NOGO = float(os.getenv("TA_G_RR_NOGO", "1.2"))   # microRR1 минимум
TA_G_RR2_MIN = float(os.getenv("TA_G_RR2_MIN", "1.6"))   # microRR2 минимум
TA_G_SPEED_FAST = float(os.getenv("TA_G_SPEED_FAST", "1.2"))  # k*ATR за 6 баров — штраф
TA_G_SPEED_SLOW = float(os.getenv("TA_G_SPEED_SLOW", "0.4"))  # “подтягивается” — бонус
TA_G_IB_ATR = float(os.getenv("TA_G_IB_ATR", "0.25"))    # дистанция до IB границы в ATR
TA_G_DERIV_FUND = float(os.getenv("TA_G_DERIV_FUND", "0.0008"))  # 0.08%/8h
TA_G_DERIV_BASIS = float(os.getenv("TA_G_DERIV_BASIS", "0.004")) # 0.4%
TA_G_LEV_CAP = int(os.getenv("TA_G_LEV_CAP", "8"))              # cap плеча при перекосе
TA_G_KILLZONES = os.getenv("TA_G_KILLZONES", "1") == "1"        # killzones включены
TA_G_MACRO_GATING = os.getenv("TA_G_MACRO_GATING", "0") == "1"  # макро-гейт (вкл. при желании)
TA_G_MACRO_EVENTS = os.getenv("TA_G_MACRO_EVENTS", "")          # CSV ISO8601, напр: 2025-08-20T17:30:00Z,...
TA_G_MACRO_MIN_BEFORE = int(os.getenv("TA_G_MACRO_MIN_BEFORE", "45"))  # минут до события — штраф

# --- Signals anti-dup helpers (per user) ---
_USER_SIGNAL_LOCKS: Dict[int, asyncio.Lock] = {}
_RECENT_SIGS: Dict[int, List[Tuple[str, str, float]]] = {}  # user_id -> [(symbol, side, ts)]
RECENT_SIG_TTL = int(os.getenv("RECENT_SIG_TTL", "7200"))  # сек, по умолчанию 2 часа

def _recent_purge(user_id: int):
    import time
    arr = _RECENT_SIGS.get(user_id, [])
    now = time.time()
    arr = [x for x in arr if now - x[2] < RECENT_SIG_TTL]
    _RECENT_SIGS[user_id] = arr

def _recent_has(user_id: int, symbol: str, side: str) -> bool:
    _recent_purge(user_id)
    for s, sd, _ in _RECENT_SIGS.get(user_id, []):
        if s == symbol and sd == side:
            return True
    return False

def _recent_add(user_id: int, symbol: str, side: str):
    import time
    _RECENT_SIGS.setdefault(user_id, []).append((symbol, side, time.time()))

# ====================== Модерация: словари ======================
# Мат разрешён; пермабан за семью/религию/нацизм/18+
HEAVY_FAMILY = [
    "твою мать", "мать твою", "ебал мам", "мамашу", "мамку трах", "мать шлюха", "мать ебал", "еб вашу мать",
]
HEAVY_RELIGION_NATIONAL = [
    "жид", "чурка", "хач", "хохол", "русня", "нацист", "фашист",
    r"смерть [а-яa-z]+", r"ненавижу [а-яa-z]+",
]
ADULT_SHOCK = ["18+", "nsfw", "xxx", "porn", "порн", "шок-контент", "краш-медиа", "gore"]

REF_BEG_SPAM = ["ref=", "рефка", "реф", "invite link", "приглашай по ссылке", "зови в лс", "пиши в лс", "write in dm", "dm me", "direct me"]
AD_LINK_HINTS = ["http://", "https://", "t.me/", "telegram.me/", "discord.gg", "bit.ly", "t.co", "goo.gl"]
WHITELIST_LINKS = [
    "t.me/NeonFakTrading", "https://t.me/NeonFakTrading",
    "t.me/neons_crypto_bot", "https://t.me/neons_crypto_bot",
]

# Какие нарушения удаляем сразу
RULES_DELETE = {"ad_link", "ref_or_beg", "adult_shock", "insult_family", "hate_speech", "dm_solicit", "forward_channel"}

# ====================== HELPERS ======================
@dataclass
class ModRecord:
    chat_id: int
    user_id: int
    warns: int = 0
    muted_until: Optional[datetime] = None
    banned_until: Optional[datetime] = None
    last_decay_at: Optional[datetime] = None  # для warn-decay

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _fmt_usd(x: Optional[float]) -> str:
    try:
        v = float(x) if x is not None else float("nan")
    except Exception:
        v = float("nan")
    if not math.isfinite(v):
        return "—"
    if v >= 1000:
        s = f"{v:,.0f}$"
    elif v >= 10:
        s = f"{v:,.2f}$"
    else:
        s = f"{v:,.4f}$"
    return s.replace(",", " ")

def _is_admin_status(member) -> bool:
    return getattr(member, "status", None) in ("administrator", "creator")

def _is_group(msg: Message) -> bool:
    return msg.chat and msg.chat.type in ("group", "supergroup")

def _clean_text(text: str) -> str:
    return (text or "").lower().strip()

def _contains_any(text: str, patterns: List[str]) -> bool:
    t = _clean_text(text)
    for p in patterns:
        try:
            if re.search(p, t, flags=re.IGNORECASE):
                return True
        except re.error:
            if p in t:
                return True
    return False

def _contains_link(text: str) -> bool:
    t = _clean_text(text)
    for w in AD_LINK_HINTS:
        if w in t:
            for white in WHITELIST_LINKS:
                if white in t:
                    return False
            return True
    return False

# ====================== DB: ensure tables ======================
async def _ensure_tables(db):
    try:
        await db.conn.execute("""
            CREATE TABLE IF NOT EXISTS moderation (
                chat_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                warns INTEGER NOT NULL DEFAULT 0,
                muted_until TEXT,
                banned_until TEXT,
                last_action_at TEXT,
                last_decay_at TEXT,
                notes TEXT,
                PRIMARY KEY(chat_id, user_id)
            )
        """)
        await db.conn.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                chat_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                message_id INTEGER,
                rule TEXT NOT NULL,
                action TEXT NOT NULL,
                reason TEXT
            )
        """)
        await db.conn.execute("""
            CREATE TABLE IF NOT EXISTS mod_chats (
                chat_id INTEGER PRIMARY KEY,
                title TEXT,
                first_seen TEXT
            )
        """)
        await db.conn.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                user_id INTEGER,
                symbol TEXT,
                side TEXT,
                created_at TEXT,
                finished_at TEXT,
                outcome TEXT,    -- TP1|TP2|TP3|STOP|BE|TIME
                rr1 REAL,
                rr2 REAL,
                rr3 REAL
            )
        """)
        await db.conn.commit()
    except Exception:
        pass

async def _register_chat(db, chat_id: int, title: Optional[str]):
    try:
        await db.conn.execute(
            "INSERT OR IGNORE INTO mod_chats (chat_id, title, first_seen) VALUES (?, ?, ?)",
            (chat_id, (title or "")[:128], _now_utc().isoformat())
        )
        await db.conn.commit()
    except Exception:
        pass

async def _list_mod_chats(db) -> List[int]:
    try:
        cur = await db.conn.execute("SELECT chat_id FROM mod_chats")
        rows = await cur.fetchall()
        return [int(r["chat_id"]) for r in rows]
    except Exception:
        return []

async def _get_mod_record(db, chat_id: int, user_id: int) -> ModRecord:
    try:
        cur = await db.conn.execute("SELECT * FROM moderation WHERE chat_id=? AND user_id=?", (chat_id, user_id))
        row = await cur.fetchone()
        if row:
            mu = datetime.fromisoformat(row["muted_until"]) if row["muted_until"] else None
            bu = datetime.fromisoformat(row["banned_until"]) if row["banned_until"] else None
            ld = datetime.fromisoformat(row["last_decay_at"]) if row["last_decay_at"] else None
            return ModRecord(chat_id=chat_id, user_id=user_id, warns=row["warns"], muted_until=mu, banned_until=bu, last_decay_at=ld)
        await db.conn.execute("INSERT INTO moderation (chat_id, user_id, warns) VALUES (?, ?, 0)", (chat_id, user_id))
        await db.conn.commit()
    except Exception:
        pass
    return ModRecord(chat_id=chat_id, user_id=user_id)

async def _save_mod_record(db, rec: ModRecord, note: str = ""):
    try:
        mu = rec.muted_until.isoformat() if rec.muted_until else None
        bu = rec.banned_until.isoformat() if rec.banned_until else None
        ld = rec.last_decay_at.isoformat() if rec.last_decay_at else None
        await db.conn.execute(
            "UPDATE moderation SET warns=?, muted_until=?, banned_until=?, last_action_at=?, last_decay_at=?, notes=? WHERE chat_id=? AND user_id=?",
            (rec.warns, mu, bu, _now_utc().isoformat(), ld, note, rec.chat_id, rec.user_id)
        )
        await db.conn.commit()
    except Exception:
        pass

async def _log_violation(db, chat_id: int, user_id: int, message_id: Optional[int], rule: str, action: str, reason: str = ""):
    if not MOD_LOG:
        return
    try:
        await db.conn.execute(
            "INSERT INTO violations (ts, chat_id, user_id, message_id, rule, action, reason) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (_now_utc().isoformat(), chat_id, user_id, message_id or 0, rule, action, reason)
        )
        await db.conn.commit()
    except Exception:
        pass

async def _last_violation_ts(db, chat_id: int, user_id: int) -> Optional[datetime]:
    try:
        cur = await db.conn.execute(
            "SELECT ts FROM violations WHERE chat_id=? AND user_id=? ORDER BY id DESC LIMIT 1",
            (chat_id, user_id)
        )
        row = await cur.fetchone()
        if row and row["ts"]:
            return datetime.fromisoformat(row["ts"])
    except Exception:
        pass
    return None

# ====================== Sanctions ======================
async def _ensure_is_admin(bot, chat_id: int, user_id: int) -> bool:
    try:
        member = await bot.get_chat_member(chat_id, user_id)
        return _is_admin_status(member)
    except Exception:
        return False

async def _warn_user(app, message: Message, reason: str):
    db = app.get("db")
    bot = app.get("bot_instance")
    rec = await _get_mod_record(db, message.chat.id, message.from_user.id)
    rec.warns += 1
    await _save_mod_record(db, rec, note=f"warn: {reason}")
    await _log_violation(db, message.chat.id, message.from_user.id, message.message_id, "WARN", "warn", reason)
    txt = (
        f"⚠️ Предупреждение пользователю ID {message.from_user.id}\n"
        f"Причина: {reason}\n"
        f"Предупреждений: {rec.warns}/{MOD_WARN_LIMIT}\n"
        f"После {MOD_WARN_LIMIT} предупреждений — бан на 30 дней."
    )
    with contextlib.suppress(Exception):
        await bot.send_message(message.chat.id, txt, message_thread_id=message.message_thread_id)
    if rec.warns >= MOD_WARN_LIMIT:
        await _ban_user(app, message, days=30, reason="3 предупреждения → бан 30 дней")

async def _mute_user(app, chat_id: int, user_id: int, minutes: int, reason: str, thread_id: Optional[int] = None):
    bot = app.get("bot_instance")
    db = app.get("db")
    until = _now_utc() + timedelta(minutes=minutes)
    perms = ChatPermissions(
        can_send_messages=False, can_send_media_messages=False, can_send_polls=False,
        can_send_other_messages=False, can_add_web_page_previews=False,
        can_change_info=False, can_invite_users=False, can_pin_messages=False
    )
    with contextlib.suppress(Exception):
        await bot.restrict_chat_member(chat_id, user_id, permissions=perms, until_date=until)
    rec = await _get_mod_record(db, chat_id, user_id)
    rec.muted_until = until
    await _save_mod_record(db, rec, note=f"mute {minutes}m: {reason}")
    await _log_violation(db, chat_id, user_id, None, "MUTE", f"{minutes}m", reason)
    with contextlib.suppress(Exception):
        await bot.send_message(chat_id, f"🔇 Мут на {minutes}м. Причина: {reason}", message_thread_id=thread_id)

async def _ban_user(app, message: Message, days: int, reason: str):
    bot = app.get("bot_instance")
    db = app.get("db")
    until = _now_utc() + timedelta(days=days)
    with contextlib.suppress(Exception):
        await bot.ban_chat_member(message.chat.id, message.from_user.id, until_date=until)
    rec = await _get_mod_record(db, message.chat.id, message.from_user.id)
    rec.banned_until = until
    await _save_mod_record(db, rec, note=f"ban {days}d: {reason}")
    await _log_violation(db, message.chat.id, message.from_user.id, message.message_id, "BAN", f"{days}d", reason)
    with contextlib.suppress(Exception):
        await bot.send_message(message.chat.id, f"⛔ Бан на {days} дн. Причина: {reason}", message_thread_id=message.message_thread_id)

async def _perma_ban_user(app, message: Message, reason: str):
    bot = app.get("bot_instance")
    db = app.get("db")
    with contextlib.suppress(Exception):
        await bot.ban_chat_member(message.chat.id, message.from_user.id)
    rec = await _get_mod_record(db, message.chat.id, message.from_user.id)
    rec.banned_until = None
    await _save_mod_record(db, rec, note=f"perma ban: {reason}")
    await _log_violation(db, message.chat.id, message.from_user.id, message.message_id, "PERMABAN", "permanent", reason)
    with contextlib.suppress(Exception):
        await bot.send_message(message.chat.id, f"🛑 Пермабан. Причина: {reason}", message_thread_id=message.message_thread_id)

async def _unmute_user(app, chat_id: int, user_id: int):
    bot = app.get("bot_instance")
    db = app.get("db")
    perms = ChatPermissions(
        can_send_messages=True, can_send_media_messages=True, can_send_polls=True,
        can_send_other_messages=True, can_add_web_page_previews=True,
        can_change_info=False, can_invite_users=True, can_pin_messages=False
    )
    with contextlib.suppress(Exception):
        await bot.restrict_chat_member(chat_id, user_id, permissions=perms, until_date=_now_utc())
    rec = await _get_mod_record(db, chat_id, user_id)
    rec.muted_until = None
    await _save_mod_record(db, rec, note="unmute")
    await _log_violation(db, chat_id, user_id, None, "UNMUTE", "-", "unmute")

async def _unban_user(app, chat_id: int, user_id: int):
    bot = app.get("bot_instance")
    db = app.get("db")
    with contextlib.suppress(Exception):
        await bot.unban_chat_member(chat_id, user_id, only_if_banned=True)
    rec = await _get_mod_record(db, chat_id, user_id)
    rec.banned_until = None
    await _save_mod_record(db, rec, note="unban")
    await _log_violation(db, chat_id, user_id, None, "UNBAN", "-", "unban")

# ====================== Violation detection ======================
def _is_from_channel_forward(msg: Message) -> bool:
    try:
        if getattr(msg, "is_automatic_forward", False):
            return True
        origin = getattr(msg, "forward_origin", None)
        if origin and getattr(origin, "type", None) == "channel":
            return True
        if getattr(msg, "forward_from_chat", None) and getattr(msg.forward_from_chat, "type", None) == "channel":
            return True
    except Exception:
        pass
    return False

def _violation_for_text(text: str) -> Optional[Tuple[str, str]]:
    t = _clean_text(text)
    if any(p in t for p in HEAVY_FAMILY):
        return "insult_family", "perma"
    if _contains_any(t, HEAVY_RELIGION_NATIONAL):
        return "hate_speech", "perma"
    if _contains_any(t, ADULT_SHOCK):
        return "adult_shock", "perma"
    if _contains_link(t):
        if any(k in t for k in REF_BEG_SPAM):
            return "ref_or_beg", "ban"
        return "ad_link", "mute"
    if _contains_any(t, ["fud", "фад", "скам", "scam", "rug", "пирамида", "лохотрон", "обман"]):
        return "fud_scam", "mute"
    if any(k in t for k in ["в лс", "пиши в лс", "в личку", "direct", "dm me", "write in dm", "прайс в лс"]):
        return "dm_solicit", "ban"
    return None

# ====================== Anti-flood (msgs & commands) ======================
_FLOOD: Dict[Tuple[int, int], List[float]] = {}
def _flood_push(chat_id: int, user_id: int, ts: float) -> int:
    key = (chat_id, user_id)
    arr = _FLOOD.get(key, [])
    arr.append(ts)
    arr = [x for x in arr if ts - x <= MOD_FLOOD_T]
    _FLOOD[key] = arr
    return len(arr)

_CMD_LAST: Dict[int, float] = {}
def _cmd_allowed(user_id: int, now_ts: float) -> bool:
    last = _CMD_LAST.get(user_id, 0.0)
    if now_ts - last >= CMD_COOLDOWN_SEC:
        _CMD_LAST[user_id] = now_ts
        return True
    return False

# ====================== Price cache (для автокоммента) ======================
_PRICE_CACHE: Dict[str, Tuple[float, float]] = {}  # symbol -> (ts, price)
def _get_price_cached(market, symbol: str, ttl: int = 10) -> Optional[float]:
    import time
    ts_now = time.time()
    ts_p, val = _PRICE_CACHE.get(symbol, (0.0, None))
    if val is not None and ts_now - ts_p < ttl:
        return val
    try:
        p = market.fetch_mark_price(symbol)
        if p is not None:
            _PRICE_CACHE[symbol] = (ts_now, float(p))
            return float(p)
    except Exception:
        pass
    return None

# ====================== Auto-comment in threads ======================
async def _auto_comment_for_channel_post(app, message: Message):
    if not _is_group(message):
        return
    if not _is_from_channel_forward(message):
        return
    if not AUTOCOMMENT_ENABLED:
        return

    bot = app.get("bot_instance")
    market = app.get("market")
    SYMBOLS = app.get("SYMBOLS", [])

    tops = ["BTC/USDT", "ETH/USDT", "TON/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT"]
    coins = []
    seen = set()
    for s in tops:
        if len(coins) >= MOD_MAX_PRICE_COINS:
            break
        coins.append(s); seen.add(s)
    for s in SYMBOLS:
        if len(coins) >= MOD_MAX_PRICE_COINS:
            break
        if s not in seen:
            coins.append(s); seen.add(s)

    lines = []
    lines.append("━━━━━━━━━━━━━━━━━━")
    lines.append("🛡️ бот NEON на страже комментариев")
    lines.append("📊 ключевые цены:")
    for s in coins:
        base = s.split("/")[0]
        price = _get_price_cached(market, s)
        lines.append(f"• {base}: {_fmt_usd(price)}")
    lines.append("")
    lines.append("🤖 Бот: @neons_crypto_bot")
    lines.append("📣 Канал: https://t.me/NeonFakTrading")
    lines.append("🄽🄴🄾🄽")
    lines.append("━━━━━━━━━━━━━━━━━━")

    text = "\n".join(lines)
    with contextlib.suppress(Exception):
        await bot.send_message(message.chat.id, text, message_thread_id=message.message_thread_id)

# ====================== Group moderation handler ======================
async def on_group_message(app, message: Message, bot):
    if not _is_group(message) or not message.from_user:
        return
    # Регистрируем чат для отчётности
    with contextlib.suppress(Exception):
        db = app.get("db")
        if db and db.conn:
            await _register_chat(db, message.chat.id, getattr(message.chat, "title", ""))

    # Пропускаем ботов
    if message.from_user.is_bot:
        return

    # Антифлуд
    cnt = _flood_push(message.chat.id, message.from_user.id, datetime.now().timestamp())
    if cnt >= MOD_FLOOD_N:
        await _warn_user(app, message, reason="flood")
        if cnt >= MOD_FLOOD_N + 2:
            with contextlib.suppress(Exception):
                await bot.delete_message(message.chat.id, message.message_id)
            await _mute_user(app, message.chat.id, message.from_user.id, minutes=max(6*60, MOD_AUTOMUTE_MIN), reason="flood", thread_id=message.message_thread_id)
        return

    # Пересылка из каналов → мут + удаление
    if _is_from_channel_forward(message):
        if DELETE_ON_VIOLATION:
            with contextlib.suppress(Exception):
                await bot.delete_message(message.chat.id, message.message_id)
        await _mute_user(app, message.chat.id, message.from_user.id, minutes=max(12*60, MOD_AUTOMUTE_MIN), reason="forward_channel", thread_id=message.message_thread_id)
        return

    # Текстовые нарушения
    text = message.text or message.caption or ""
    if not text:
        return
    viol = _violation_for_text(text)
    if viol:
        rule, severity = viol
        if DELETE_ON_VIOLATION and rule in RULES_DELETE:
            with contextlib.suppress(Exception):
                await bot.delete_message(message.chat.id, message.message_id)
        if severity == "perma":
            await _perma_ban_user(app, message, reason=rule)
        elif severity == "ban":
            await _ban_user(app, message, days=7, reason=rule)
        elif severity == "mute":
            await _mute_user(app, message.chat.id, message.from_user.id, minutes=max(12*60, MOD_AUTOMUTE_MIN), reason=rule, thread_id=message.message_thread_id)
        else:
            await _warn_user(app, message, reason=rule)

# ====================== Welcome / CAPTCHA / Rules / Report ======================
_PENDING_VERIFY: Dict[Tuple[int,int], int] = {}  # (chat_id, user_id) -> verify_message_id

async def on_member_update(app, upd: ChatMemberUpdated):
    try:
        if upd.chat and upd.chat.type in ("group", "supergroup"):
            old = getattr(upd, "old_chat_member", None)
            new = getattr(upd, "new_chat_member", None)
            if old and new and getattr(old, "status", None) in ("left", "kicked") and getattr(new, "status", None) == "member":
                bot = app.get("bot_instance")
                db = app.get("db")
                if db and db.conn:
                    await _register_chat(db, upd.chat.id, getattr(upd.chat, "title", ""))
                # Hold/капча
                if NEWUSER_CAPTCHA:
                    perms = ChatPermissions(
                        can_send_messages=False, can_send_media_messages=False, can_send_polls=False,
                        can_send_other_messages=False, can_add_web_page_previews=False,
                        can_change_info=False, can_invite_users=False, can_pin_messages=False
                    )
                    until = _now_utc() + timedelta(minutes=NEWUSER_HOLD_MIN)
                    with contextlib.suppress(Exception):
                        await bot.restrict_chat_member(upd.chat.id, new.user.id, permissions=perms, until_date=until)
                    kb = InlineKeyboardMarkup(inline_keyboard=[
                        [InlineKeyboardButton(text="Я не бот ✅", callback_data=f"verify:{upd.chat.id}:{new.user.id}")]
                    ])
                    txt = (
                        f"👋 Добро пожаловать, {getattr(new.user,'full_name',new.user.id)}!\n"
                        f"Нажмите кнопку ниже, чтобы писать в чат.\n"
                        f"Ограничение снимется автоматически через {NEWUSER_HOLD_MIN} минут."
                    )
                    msg = None
                    with contextlib.suppress(Exception):
                        msg = await bot.send_message(upd.chat.id, txt, reply_markup=kb)
                    if msg:
                        _PENDING_VERIFY[(upd.chat.id, new.user.id)] = msg.message_id
                else:
                    txt = (
                        "👋 Добро пожаловать!\n"
                        "Пожалуйста, соблюдайте правила чата: /rules\n"
                        "Команды бота доступны здесь: /signal, /price, /status и др. (лимит 3 сигнала/24ч)."
                    )
                    with contextlib.suppress(Exception):
                        await bot.send_message(upd.chat.id, txt)
    except Exception:
        pass

async def cb_verify(app, cb: CallbackQuery):
    try:
        data = cb.data or ""
        if not data.startswith("verify:"):
            return
        _, chat_id_s, user_id_s = data.split(":", 2)
        chat_id = int(chat_id_s); user_id = int(user_id_s)
        if cb.from_user.id != user_id:
            with contextlib.suppress(Exception):
                await cb.answer("Эта кнопка не для вас.", show_alert=True)
            return
        bot = app.get("bot_instance")
        perms = ChatPermissions(
            can_send_messages=True, can_send_media_messages=True, can_send_polls=True,
            can_send_other_messages=True, can_add_web_page_previews=True,
            can_change_info=False, can_invite_users=True, can_pin_messages=False
        )
        with contextlib.suppress(Exception):
            await bot.restrict_chat_member(chat_id, user_id, permissions=perms, until_date=_now_utc())
        mid = _PENDING_VERIFY.pop((chat_id, user_id), None)
        if mid:
            with contextlib.suppress(Exception):
                await bot.delete_message(chat_id, mid)
        with contextlib.suppress(Exception):
            await cb.answer("Готово! Пишите в чат 🙌")
    except Exception:
        with contextlib.suppress(Exception):
            await cb.answer()

async def cmd_rules(app, message: Message, bot):
    if not _is_group(message): return
    short = (
        "📜 Правила чата (кратко)\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "• Оскорбления: семья/религия/нац. → пермабан; иное — пред/мут/бан\n"
        "• Реклама/ссылки без разрешения → мут/бан; рефералки/зазыв в ЛС → бан\n"
        "• Пересылка из каналов → мут\n"
        "• 18+/шок‑контент → пермабан\n"
        "• Флуд/громкие медиа → пред → мут → бан\n"
        "• Мошенничество/FUD/клевета → мут/бан\n"
        "• Утечка личных данных → пермабан\n"
        "• Языки: RU/UA/BY/EN\n"
        f"• {MOD_WARN_LIMIT} предупреждения → бан на 30 дней"
    )
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Полные правила", callback_data="rules_full")]
    ])
    with contextlib.suppress(Exception):
        await bot.send_message(message.chat.id, short, reply_markup=kb, message_thread_id=message.message_thread_id)

async def cb_rules_full(app, cb: CallbackQuery):
    if cb.data != "rules_full": return
    full = (
        "📜 Правила чата (полная версия)\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "Запрещено:\n"
        "— Оскорбления и конфликты:\n"
        "   • Оскорбления семьи, религии, национализм → пермабан.\n"
        "   • Прочие оскорбления → предупреждение/мут/бан.\n"
        "— Реклама и спам:\n"
        "   • Реклама/ссылки/пиар без разрешения → мут → бан.\n"
        "   • Реферальные ссылки, попрошайничество, трейд в ЛС → бан.\n"
        "   • Пересылка сообщений из каналов → мут.\n"
        "— Запрещённый контент:\n"
        "   • 18+, шок‑контент, краш‑медиа → пермабан.\n"
        "— Флуд:\n"
        "   • Текст/стикеры/реакции → пред → мут → бан.\n"
        "   • Громкие/эпилептические медиа → пред → мут.\n"
        "— Мошенничество и FUD:\n"
        "   • Ложная информация/клевета → мут/бан.\n"
        "   • Распространение FUD → бан.\n"
        "— Конфиденциальность:\n"
        "   • Утечка личных данных → пермабан.\n"
        "— Зазыв в ЛС → бан.\n\n"
        "Дополнительно: 3 предупреждения → бан на 30 дней.\n"
        "Доступные языки: русский, украинский, белорусский, английский.\n"
        "Мы не даём финансовых советов. Мы делимся сделками → результат → учимся и развиваемся."
    )
    with contextlib.suppress(Exception):
        await cb.message.answer(full, message_thread_id=getattr(cb.message, "message_thread_id", None))
        await cb.answer("Ок")

async def cmd_report(app, message: Message, bot):
    if not _is_group(message): return
    if not message.reply_to_message:
        with contextlib.suppress(Exception):
            await message.reply("Ответьте на сообщение-нарушение командой /report с кратким описанием.")
        return
    reason = (message.text or "").split(maxsplit=1)
    reason = reason[1].strip() if len(reason) > 1 else "report"
    target = message.reply_to_message.from_user
    if not target:
        with contextlib.suppress(Exception):
            await message.reply("Не удалось определить пользователя (нет автора).")
        return
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🔇 Mute 6h", callback_data=f"modact:mute6h:{message.chat.id}:{target.id}:{message.reply_to_message.message_id}"),
            InlineKeyboardButton(text="⛔ Ban 7d", callback_data=f"modact:ban7d:{message.chat.id}:{target.id}:{message.reply_to_message.message_id}"),
            InlineKeyboardButton(text="🛑 Perma", callback_data=f"modact:perma:{message.chat.id}:{target.id}:{message.reply_to_message.message_id}"),
        ],
        [InlineKeyboardButton(text="🗑 Удалить сообщение", callback_data=f"modact:del:{message.chat.id}:{target.id}:{message.reply_to_message.message_id}")]
    ])
    admins = []
    with contextlib.suppress(Exception):
        admins = await app.get("db").get_admin_user_ids()
    txt = (
        f"🚨 Репорт в чате {getattr(message.chat, 'title', message.chat.id)}\n"
        f"От: {message.from_user.id}\n"
        f"На: {target.id}\n"
        f"Причина: {reason}\n"
        f"Сообщение: id={message.reply_to_message.message_id}"
    )
    for aid in admins or []:
        with contextlib.suppress(Exception):
            await bot.send_message(aid, txt, reply_markup=kb)

async def cb_mod_action(app, cb: CallbackQuery):
    try:
        data = cb.data or ""
        if not data.startswith("modact:"):
            return
        _, action, chat_id_s, user_id_s, msg_id_s = data.split(":", 4)
        chat_id = int(chat_id_s); user_id = int(user_id_s); msg_id = int(msg_id_s)
        bot = app.get("bot_instance")
        if not await _ensure_is_admin(bot, chat_id, cb.from_user.id):
            with contextlib.suppress(Exception):
                await cb.answer("Только для админов чата.", show_alert=True)
            return
        if action == "mute6h":
            await _mute_user(app, chat_id, user_id, minutes=360, reason="report_action", thread_id=None)
            await cb.answer("Muted 6h")
        elif action == "ban7d":
            pseudo = Message(chat=type("C", (), {"id": chat_id}), message_id=msg_id, date=datetime.now(), from_user=type("U", (), {"id": user_id}))
            await _ban_user(app, pseudo, days=7, reason="report_action")
            await cb.answer("Banned 7d")
        elif action == "perma":
            pseudo = Message(chat=type("C", (), {"id": chat_id}), message_id=msg_id, date=datetime.now(), from_user=type("U", (), {"id": user_id}))
            await _perma_ban_user(app, pseudo, reason="report_action")
            await cb.answer("Perma banned")
        elif action == "del":
            with contextlib.suppress(Exception):
                await bot.delete_message(chat_id, msg_id)
            await cb.answer("Сообщение удалено")
        else:
            await cb.answer()
    except Exception:
        with contextlib.suppress(Exception):
            await cb.answer()

# ====================== Mod commands ======================
def _extract_target_user_id(message: Message) -> Optional[int]:
    if getattr(message, "reply_to_message", None) and message.reply_to_message.from_user:
        return int(message.reply_to_message.from_user.id)
    m = re.search(r"\b(\d{6,15})\b", message.text or "")
    if m:
        try: return int(m.group(1))
        except Exception: return None
    return None

async def cmd_warn(app, message: Message, bot):
    if not _is_group(message): return
    if not await _ensure_is_admin(bot, message.chat.id, message.from_user.id): return
    target_id = _extract_target_user_id(message)
    if not target_id:
        with contextlib.suppress(Exception): await message.reply("Укажите пользователя ответом на его сообщение или id.")
        return
    pseudo = Message(chat=message.chat, message_id=message.message_id, date=message.date, from_user=type("U", (), {"id": target_id}))
    await _warn_user(app, pseudo, reason="manual")

async def cmd_mute(app, message: Message, bot):
    if not _is_group(message): return
    if not await _ensure_is_admin(bot, message.chat.id, message.from_user.id): return
    target_id = _extract_target_user_id(message)
    if not target_id:
        with contextlib.suppress(Exception): await message.reply("Укажите пользователя ответом на его сообщение или id.")
        return
    parts = (message.text or "").split()
    minutes = MOD_AUTOMUTE_MIN
    if len(parts) >= 2 and parts[1].isdigit():
        minutes = max(1, int(parts[1]))
    reason = " ".join(parts[2:]) if len(parts) >= 3 else "manual"
    await _mute_user(app, message.chat.id, target_id, minutes, reason, message.message_thread_id)

async def cmd_ban(app, message: Message, bot):
    if not _is_group(message): return
    if not await _ensure_is_admin(bot, message.chat.id, message.from_user.id): return
    target_id = _extract_target_user_id(message)
    if not target_id:
        with contextlib.suppress(Exception): await message.reply("Укажите пользователя ответом на его сообщение или id.")
        return
    parts = (message.text or "").split()
    days = 7
    if len(parts) >= 2 and parts[1].isdigit():
        days = max(1, int(parts[1]))
    reason = " ".join(parts[2:]) if len(parts) >= 3 else "manual"
    pseudo = Message(chat=message.chat, message_id=message.message_id, date=message.date, from_user=type("U", (), {"id": target_id}))
    await _ban_user(app, pseudo, days, reason)

async def cmd_unmute(app, message: Message, bot):
    if not _is_group(message): return
    if not await _ensure_is_admin(bot, message.chat.id, message.from_user.id): return
    target_id = _extract_target_user_id(message)
    if not target_id:
        with contextlib.suppress(Exception): await message.reply("Укажите пользователя ответом на его сообщение или id.")
        return
    await _unmute_user(app, message.chat.id, target_id)
    with contextlib.suppress(Exception): await message.reply("✅ С пользователя снят мут.")

async def cmd_unban(app, message: Message, bot):
    if not _is_group(message): return
    if not await _ensure_is_admin(bot, message.chat.id, message.from_user.id): return
    target_id = _extract_target_user_id(message)
    if not target_id:
        with contextlib.suppress(Exception): await message.reply("Укажите пользователя ответом на его сообщение или id.")
        return
    await _unban_user(app, message.chat.id, target_id)
    with contextlib.suppress(Exception): await message.reply("✅ Пользователь разбанен.")

async def cmd_warns(app, message: Message, bot):
    if not _is_group(message): return
    db = app.get("db")
    target_id = _extract_target_user_id(message) or message.from_user.id
    rec = await _get_mod_record(db, message.chat.id, target_id)
    mu = rec.muted_until.strftime("%Y-%m-%d %H:%M") if rec.muted_until else "нет"
    bu = rec.banned_until.strftime("%Y-%m-%d %H:%M") if rec.banned_until else "нет"
    txt = f"👁 Профиль модерации (user {target_id}): warns={rec.warns}, mute_until={mu}, ban_until={bu}"
    with contextlib.suppress(Exception):
        await bot.send_message(message.chat.id, txt, message_thread_id=message.message_thread_id)

# ====================== Guard access patch (commands in groups) + лимит-подсказка ======================
def _patch_guard_access_for_groups(app):
    orig_guard = app.get("guard_access")
    logger = app.get("logger")
    DAILY_LIMIT = app.get("DAILY_LIMIT", 3)

    async def guard_access_patched(message: Message, bot):
        # Группы: разрешаем команды без подписки, но работаем только если БД готова
        if _is_group(message) and ALLOW_CHAT_COMMANDS_WITHOUT_SUB:
            db = app.get("db")
            if not db or not getattr(db, "conn", None):
                # БД ещё не готова — мягко выходим
                return None
            st = await db.get_user_state(message.from_user.id)
            # подсказка остатка для /signal
            txt = (message.text or "").strip().lower()
            if txt.startswith("/signal") or "получить сигнал" in txt or "📈" in txt:
                left = max(0, int(DAILY_LIMIT) - int(st.get("count", 0)))
                with contextlib.suppress(Exception):
                    await bot.send_message(
                        message.chat.id,
                        f"ℹ️ Осталось сигналов сегодня: {left}/{DAILY_LIMIT}",
                        message_thread_id=getattr(message, "message_thread_id", None)
                    )
            return st
        # Прочие случаи — оригинальная защита
        return await orig_guard(message, bot)

    app["guard_access"] = guard_access_patched
    logger and logger.info("Guard access patched: group commands allowed without subscription, limit tips enabled (DB dynamic).")

# ====================== TA utilities (inside chat.py) ======================
def _volume_profile(values_price: List[float], values_vol: List[float], bins: int = 40) -> Optional[Tuple[float,float,float]]:
    try:
        import numpy as np
        if not values_price or not values_vol or len(values_price) != len(values_vol) or len(values_price) < 10:
            return None
        lo, hi = float(min(values_price)), float(max(values_price))
        if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
            return None
        hist, edges = np.histogram(np.array(values_price, dtype=float), bins=bins, range=(lo, hi), weights=np.array(values_vol, dtype=float))
        if hist.sum() <= 0:
            return None
        poc_idx = int(hist.argmax())
        poc = float(0.5 * (edges[poc_idx] + edges[poc_idx + 1]))
        total = float(hist.sum()); order = hist.argsort()[::-1]; acc = 0.0; mask = [False]*len(hist)
        for idx in order:
            mask[idx] = True
            acc += float(hist[idx])
            if acc/total >= 0.68: break
        idxs = [i for i,m in enumerate(mask) if m]
        vah = float(edges[max(idxs)+1]); val = float(edges[min(idxs)])
        return poc, vah, val
    except Exception:
        return None

def _parse_macro_events(env_val: str) -> List[datetime]:
    out = []
    for part in (env_val or "").split(","):
        s = part.strip()
        if not s: continue
        try:
            # поддержка Z/UTC
            if s.endswith("Z"):
                s = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            out.append(dt.astimezone(timezone.utc))
        except Exception:
            continue
    return out

def _minutes_to_next_event(events: List[datetime]) -> Optional[float]:
    now = datetime.now(timezone.utc)
    mins = []
    for e in events:
        if e > now:
            mins.append((e - now).total_seconds() / 60.0)
    return min(mins) if mins else None

# --- New: TA Sanity & Caps ENV ---
TA_SAN_SL_MIN_ATR = float(os.getenv("TA_SAN_SL_MIN_ATR", "0.25"))   # min риск до SL: ≥ 0.25*ATR
TA_SAN_SL_MAX_ATR = float(os.getenv("TA_SAN_SL_MAX_ATR", "5.0"))    # max риск до SL: ≤ 5*ATR
TA_SAN_TP_MIN_STEP_ATR = float(os.getenv("TA_SAN_TP_MIN_STEP_ATR", "0.15"))  # шаг между TP: ≥ 0.15*ATR
TA_TP_CAP_ATR_MAX = float(os.getenv("TA_TP_CAP_ATR_MAX", "4.0"))     # TP не дальше 4*ATR
TA_TP_CAP_PDR_MULT = float(os.getenv("TA_TP_CAP_PDR_MULT", "1.2"))   # TP не дальше 1.2×(PDH-PDL)
TA_VP_BINS_SESSION = int(os.getenv("TA_VP_BINS_SESSION", "32"))
TA_VP_BINS_DAY = int(os.getenv("TA_VP_BINS_DAY", "48"))
TA_VP_APPLY_TP3_ONLY = os.getenv("TA_VP_APPLY_TP3_ONLY", "1") == "1" # капить только TP3
TA_VP_APPLY_TP2_TOO  = os.getenv("TA_VP_APPLY_TP2_TOO",  "0") == "1" # можно капить и TP2
TA_VP_STABILITY_MIN_NBARS = int(os.getenv("TA_VP_STABILITY_MIN_NBARS", "40"))
TA_VP_STABILITY_PEAK_RATIO = float(os.getenv("TA_VP_STABILITY_PEAK_RATIO", "1.6"))  # max/median > 1.6
# RR пороги (усилено)
TA_MRR1_MIN = float(os.getenv("TA_MRR1_MIN", "1.2"))
TA_MRR2_MIN = float(os.getenv("TA_MRR2_MIN", "1.6"))
# “запрет” слишком далёких входов от якорей (VWAP/IB)
TA_ANCHOR_MAX_DIST_ATR = float(os.getenv("TA_ANCHOR_MAX_DIST_ATR", "2.0"))

# --- New: TA sanity helpers ---
def _round_tick(x: float, tick: float, mode: str = "round") -> float:
    try:
        x = float(x)
        if not tick or tick <= 0:
            return x
        n = x / float(tick)
        if mode == "floor": n = math.floor(n)
        elif mode == "ceil": n = math.ceil(n)
        else: n = round(n)
        return float(n * float(tick))
    except Exception:
        return float(x)

def _micro_rr(side: str, entry: float, sl: float, tp1: float, tp2: float) -> Tuple[float, float]:
    risk = abs(float(entry) - float(sl)) + 1e-12
    rr1 = abs(float(tp1) - float(entry)) / risk
    rr2 = abs(float(tp2) - float(entry)) / risk
    return rr1, rr2

def _ensure_tp_monotonic_with_step(side: str, entry: float, tps: List[float], atr: float, tick: float, step_atr: float) -> List[float]:
    # Принудительно: TP1 < TP2 < TP3 с шагом ≥ step_atr*ATR
    if not tps: return []
    tps = [float(x) for x in tps[:3]] + ([tps[-1]] * max(0, 3 - len(tps)))
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

def _compute_pdh_pdl(df15) -> Optional[Tuple[float, float]]:
    # Берём предыдущий полный день относительно UTC
    try:
        if df15 is None or len(df15) < 60:
            return None
        ts = df15["ts"]
        last_ts = ts.iloc[-1].to_pydatetime().astimezone(timezone.utc)
        day_anchor = last_ts.replace(hour=0, minute=0, second=0, microsecond=0)
        prev_day_start = day_anchor - timedelta(days=1)
        prev = df15[(ts >= prev_day_start) & (ts < day_anchor)]
        if prev is None or prev.empty:
            return None
        pdh = float(prev["high"].max())
        pdl = float(prev["low"].min())
        if not (math.isfinite(pdh) and math.isfinite(pdl) and pdh > pdl):
            return None
        return pdh, pdl
    except Exception:
        return None

def _cap_by_atr_and_pdr(side: str, entry: float, tps: List[float], atr: float, pdh_pdl: Optional[Tuple[float,float]]) -> List[float]:
    if not tps: return []
    rng_cap = None
    if pdh_pdl:
        pdh, pdl = pdh_pdl
        day_range = abs(pdh - pdl)
        rng_cap = float(entry) + (TA_TP_CAP_PDR_MULT * day_range) * (1 if side=="LONG" else -1)
    atr_cap = float(entry) + (TA_TP_CAP_ATR_MAX * float(atr)) * (1 if side=="LONG" else -1)

    capped = []
    for tp in tps:
        lims = []
        if atr_cap is not None:
            lims.append(atr_cap)
        if rng_cap is not None:
            lims.append(rng_cap)
        if side == "LONG":
            v = min([tp] + [x for x in lims if x is not None and x > entry])
        else:
            v = max([tp] + [x for x in lims if x is not None and x < entry])
        capped.append(v)
    return capped

def _volume_profile_with_hist(values_price: List[float], values_vol: List[float], bins: int = 40):
    # Возвращает (poc, vah, val, hist) или None
    try:
        import numpy as np
        if not values_price or not values_vol or len(values_price) != len(values_vol) or len(values_price) < 10:
            return None
        lo, hi = float(min(values_price)), float(max(values_price))
        if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
            return None
        hist, edges = np.histogram(np.array(values_price, dtype=float), bins=bins, range=(lo, hi), weights=np.array(values_vol, dtype=float))
        if hist.sum() <= 0:
            return None
        poc_idx = int(hist.argmax())
        poc = float(0.5 * (edges[poc_idx] + edges[poc_idx + 1]))
        total = float(hist.sum())
        order = hist.argsort()[::-1]
        acc = 0.0
        mask = [False]*len(hist)
        for idx in order:
            mask[idx] = True
            acc += float(hist[idx])
            if acc / total >= 0.68: break
        idxs = [i for i,m in enumerate(mask) if m]
        vah = float(edges[max(idxs)+1])
        val = float(edges[min(idxs)])
        return poc, vah, val, hist
    except Exception:
        return None

def _vp_is_stable(hist) -> bool:
    try:
        import numpy as np
        if hist is None or len(hist) < 10:
            return False
        nz = (hist > 0).sum()
        if nz < max(8, len(hist)*0.2):
            return False
        med = float(np.median(hist[hist > 0])) if (hist > 0).any() else 0.0
        mx = float(hist.max())
        if med <= 0:
            return False
        return (mx / med) >= TA_VP_STABILITY_PEAK_RATIO
    except Exception:
        return False

def _choose_vp_cap(side: str, entry: float, vp_levels: List[Tuple[str, float]]) -> Optional[float]:
    # Берём ближайший барьер по профилю в сторону цели
    try:
        if side == "LONG":
            ups = [lv for name, lv in vp_levels if math.isfinite(lv) and lv > entry]
            return min(ups) if ups else None
        else:
            downs = [lv for name, lv in vp_levels if math.isfinite(lv) and lv < entry]
            return max(downs) if downs else None
    except Exception:
        return None

def _sanity_tp_sl_rr_fix(symbol: str, side: str, entry: float, sl: float, tps: List[float], df15, atr: float, tick: float):
    """
    Возвращает: ok, sl2, tps2, info_dict
    Гарантии:
    - SL не 0, не «неправильной стороны», риск в [TA_SAN_SL_MIN_ATR, TA_SAN_SL_MAX_ATR] * ATR
    - TP строго монотонны и с шагом ≥ TA_SAN_TP_MIN_STEP_ATR*ATR
    - TP кап по ATR и дневному диапазону (PDH/PDL)
    - RR1/RR2 >= порогам, иначе ok=False
    """
    info = {"changed": False, "notes": []}
    e = float(entry)
    s = float(sl)
    atrv = float(max(1e-9, atr))
    tick = float(tick or 0.0)

    # 1) SL сторона + базовый пол
    if side == "LONG":
        if not (s < e):
            s = e - TA_SAN_SL_MIN_ATR * atrv
            info["changed"] = True; info["notes"].append("fix: SL side")
    else:
        if not (s > e):
            s = e + TA_SAN_SL_MIN_ATR * atrv
            info["changed"] = True; info["notes"].append("fix: SL side")

    # 2) Риск до SL в пределах [minATR, maxATR]
    risk = abs(e - s)
    min_risk = TA_SAN_SL_MIN_ATR * atrv
    max_risk = TA_SAN_SL_MAX_ATR * atrv
    if risk < min_risk:
        if side == "LONG":
            s = e - min_risk
        else:
            s = e + min_risk
        info["changed"] = True; info["notes"].append("fix: SL < minATR")
    elif risk > max_risk:
        if side == "LONG":
            s = e - max_risk
        else:
            s = e + max_risk
        info["changed"] = True; info["notes"].append("fix: SL > maxATR")

    # round SL by tick
    s = _round_tick(s, tick, "floor" if side=="LONG" else "ceil")

    # 3) TP шаги и монотонность
    old_tps = list(tps)
    tps = _ensure_tp_monotonic_with_step(side, e, tps, atrv, tick, TA_SAN_TP_MIN_STEP_ATR)
    if tps != old_tps:
        info["changed"] = True; info["notes"].append("fix: TP monotonic/step")

    # 4) Капы по ATR и дневному диапазону
    pdh_pdl = _compute_pdh_pdl(df15)
    tps_capped = _cap_by_atr_and_pdr(side, e, tps, atrv, pdh_pdl)
    if tps_capped != tps:
        tps = tps_capped
        info["changed"] = True; info["notes"].append("cap: ATR/PDR")
    # доп. округление
    if side == "LONG":
        tps = [_round_tick(tp, tick, "ceil") for tp in tps]
    else:
        tps = [_round_tick(tp, tick, "floor") for tp in tps]

    # 5) RR проверка
    rr1, rr2 = _micro_rr(side, e, s, tps[0], tps[1])
    info["rr1"] = rr1; info["rr2"] = rr2
    ok = not (rr1 < TA_MRR1_MIN or rr2 < TA_MRR2_MIN)
    if not ok:
        info["notes"].append("reject: microRR below thresholds")

    return ok, float(s), list(tps), info

# ====================== TA booster: усиление (RR/IB/killzones/speed/derivatives/RS + trailing POC lock-in) ======================
def _patch_ta_booster(app):
    logger = app.get("logger")
    market = app.get("market")
    ema = app.get("ema")
    atr_fn = app.get("atr")
    adx_fn = app.get("adx")
    anchored_vwap = app.get("anchored_vwap")

    # Trailing patch: POC lock-in + adaptive BE ±0.2*ATR по режиму
    orig_update_trailing = app.get("update_trailing")
    async def update_trailing_patched(sig):
        # вызов оригинала
        try:
            await orig_update_trailing(sig)
        except Exception:
            pass
        # дополнения
        try:
            df15 = market.fetch_ohlcv(sig.symbol, "15m", 280)
            if df15 is None or len(df15) < 120:
                return
            # POC lock-in (после TP1)
            prices = ((df15["high"] + df15["low"] + df15["close"]) / 3.0).tail(240).astype(float).tolist()
            vols = df15["volume"].tail(240).astype(float).tolist()
            vp = _volume_profile(prices, vols, bins=40)
            if vp:
                poc, vah, val = vp
                if sig.tp_hit >= 1:
                    if sig.side == "LONG":
                        sig.sl = max(sig.sl, float(poc))
                    else:
                        sig.sl = min(sig.sl, float(poc))
            # Adaptive BE: после TP1 — BE = entry ± 0.2*ATR
            if sig.tp_hit >= 1:
                atrv = float(atr_fn(df15, 14).iloc[-1])
                if atrv > 0:
                    if sig.side == "LONG":
                        sig.sl = max(sig.sl, float(sig.entry) - 0.2 * atrv)
                    else:
                        sig.sl = min(sig.sl, float(sig.entry) + 0.2 * atrv)
        except Exception:
            pass

    if orig_update_trailing:
        app["update_trailing"] = update_trailing_patched
        logger and logger.info("Trailing patched: POC lock-in + adaptive BE.")

    # Score booster
    orig = app.get("score_symbol_core")
    if not orig:
        return

    MACRO_EVENTS = _parse_macro_events(TA_G_MACRO_EVENTS) if TA_G_MACRO_GATING else []

    def boosted(symbol: str, relax: bool = False):
        base = orig(symbol, relax)
        if base is None:
            return None
        side_score, side, d = base
        d = dict(d or {})
        breakdown = d.get("score_breakdown", {}) or {}
        entry = float(d.get("c5")); sl = float(d.get("sl")); tps = list(d.get("tps") or [])
        # заранее достанем ATR/tick/df15
        try:
            df15 = market.fetch_ohlcv(symbol, "15m", 320)
        except Exception:
            df15 = None
        try:
            atr_val = float(d.get("atr", 0.0)) or (float(atr_fn(df15, 14).iloc[-1]) if df15 is not None and len(df15)>=20 else 0.0)
        except Exception:
            atr_val = 0.0
        try:
            tick = float(market.get_tick_size(symbol) or 0.0)
        except Exception:
            tick = 0.0

        # 0) Жёсткий sanity‑чек TP/SL/RR (+ кап ATR/PDR, шаги и монотон)
        try:
            ok, sl2, tps2, sinfo = _sanity_tp_sl_rr_fix(symbol, side, entry, sl, tps, df15, atr_val, tick)
            d["rr1"] = float(sinfo.get("rr1", float("nan")))
            d["rr2"] = float(sinfo.get("rr2", float("nan")))
            if not ok:
                # жёстко топим кандидата, но не ломаем пайплайн
                side_score -= 9.99
                breakdown["sanityRR"] = breakdown.get("sanityRR", 0.0) - 1.0
                d["reject_reason"] = "microRR"
            else:
                if sinfo.get("changed"):
                    breakdown["sanityFix"] = breakdown.get("sanityFix", 0.0) + 0.05
                d["sl"] = sl = sl2
                d["tps"] = tps = tps2
        except Exception:
            pass

        # 1) microRR штраф/бонус (поверх уже рассчитанного)
        try:
            if len(tps) >= 2:
                rr1, rr2 = _micro_rr(side, entry, sl, tps[0], tps[1])
                if rr1 < TA_MRR1_MIN:
                    side_score -= 0.5; breakdown["microRR1"] = breakdown.get("microRR1", 0.0) - 0.5
                if rr2 < TA_MRR2_MIN:
                    side_score -= 0.2; breakdown["microRR2"] = breakdown.get("microRR2", 0.0) - 0.2
                if rr1 >= 1.5 and rr2 >= 2.0:
                    side_score += 0.1; breakdown["microRR+"] = breakdown.get("microRR+", 0.0) + 0.1
        except Exception:
            pass

        # 2) IB + скорость + killzones
        try:
            if df15 is not None and len(df15) >= 50:
                if not atr_val:
                    atr_val = float(atr_fn(df15, 14).iloc[-1])
                if atr_val and atr_val > 0:
                    ts = df15["ts"]; last_ts = ts.iloc[-1]
                    day_anchor = (last_ts.to_pydatetime().astimezone(timezone.utc)).replace(hour=0, minute=0, second=0, microsecond=0)
                    cur = df15[df15["ts"] >= day_anchor]
                    if len(cur) >= 4:
                        ib = cur.head(4)
                        ib_hi = float(ib["high"].max()); ib_lo = float(ib["low"].min())
                        if side == "LONG" and (ib_hi - float(entry)) < TA_G_IB_ATR * float(atr_val):
                            side_score -= 0.15; breakdown["IBgate"] = breakdown.get("IBgate", 0.0) - 0.15
                        if side == "SHORT" and (float(entry) - ib_lo) < TA_G_IB_ATR * float(atr_val):
                            side_score -= 0.15; breakdown["IBgate"] = breakdown.get("IBgate", 0.0) - 0.15
                    import numpy as np
                    close = df15["close"].astype(float).values
                    if len(close) >= 6:
                        speed = abs(close[-1] - close[-6]) / (float(atr_val) + 1e-9)
                        if speed > TA_G_SPEED_FAST:
                            side_score -= 0.12; breakdown["approach"] = breakdown.get("approach", 0.0) - 0.12
                        elif speed < TA_G_SPEED_SLOW:
                            side_score += 0.06; breakdown["approach"] = breakdown.get("approach", 0.0) + 0.06
                    if TA_G_KILLZONES:
                        now_msk = app["now_msk"](); hr = now_msk.hour
                        if hr in (10,11,16,17):
                            side_score += 0.06; breakdown["killzone"] = breakdown.get("killzone", 0.0) + 0.06
                        elif hr in (3,4,5,23):
                            side_score -= 0.06; breakdown["killzone"] = breakdown.get("killzone", 0.0) - 0.06
        except Exception:
            pass

        # 3) Derivatives (момент)
        try:
            funding = d.get("funding_rate"); basis = d.get("basis")
            if isinstance(funding, (int,float)) and isinstance(basis,(int,float)):
                pos_press = (funding > TA_G_DERIV_FUND) and (basis > TA_G_DERIV_BASIS)
                neg_press = (funding < -TA_G_DERIV_FUND) and (basis < -TA_G_DERIV_BASIS)
                if side == "LONG" and neg_press:
                    side_score -= 0.15; breakdown["Deriv-"] = breakdown.get("Deriv-", 0.0) - 0.15
                if side == "SHORT" and pos_press:
                    side_score -= 0.15; breakdown["Deriv-"] = breakdown.get("Deriv-", 0.0) - 0.15
                lev = int(d.get("leverage", 10) or 10)
                if pos_press or neg_press:
                    d["leverage"] = min(lev, TA_G_LEV_CAP)
        except Exception:
            pass

        # 4) RS поддержка
        try:
            rs_btc = d.get("RS_btc"); rs_eth = d.get("RS_eth")
            adj = 0.0
            if isinstance(rs_btc, (int, float)):
                adj += (0.12 if ((side == "LONG" and rs_btc > 0) or (side == "SHORT" and rs_btc < 0)) else -0.06)
            if isinstance(rs_eth, (int, float)):
                adj += (0.06 if ((side == "LONG" and rs_eth > 0) or (side == "SHORT" and rs_eth < 0)) else -0.03)
            side_score += adj
            if adj != 0.0:
                breakdown["RS"] = breakdown.get("RS", 0.0) + adj
        except Exception:
            pass

        # 5) Кап TP по Volume Profile: только TP3 (и опционально TP2), session/day + стабильность
        try:
            if df15 is not None and len(df15) >= max(120, TA_VP_STABILITY_MIN_NBARS):
                prices = ((df15["high"] + df15["low"] + df15["close"]) / 3.0).astype(float)
                vols = df15["volume"].astype(float)
                # День
                ts = df15["ts"]; last_ts = ts.iloc[-1].to_pydatetime().astimezone(timezone.utc)
                day_anchor = last_ts.replace(hour=0, minute=0, second=0, microsecond=0)
                day = df15[ts >= day_anchor]
                # Сессия ~ 6 часов
                sess = df15.tail(24)  # 24*15m = 6h

                vp_day = _volume_profile_with_hist(prices.tail(len(day)).tolist(), vols.tail(len(day)).tolist(), bins=TA_VP_BINS_DAY) if len(day) >= TA_VP_STABILITY_MIN_NBARS else None
                vp_ses = _volume_profile_with_hist(sess["close"].astype(float).tolist(), sess["volume"].astype(float).tolist(), bins=TA_VP_BINS_SESSION) if len(sess) >= 16 else None

                vp_levels = []
                if vp_day and _vp_is_stable(vp_day[3]):
                    poc_d, vah_d, val_d, _ = vp_day
                    vp_levels += [("day_poc",poc_d), ("day_vah",vah_d), ("day_val",val_d)]
                if vp_ses and _vp_is_stable(vp_ses[3]):
                    poc_s, vah_s, val_s, _ = vp_ses
                    vp_levels += [("sess_poc",poc_s), ("sess_vah",vah_s), ("sess_val",val_s)]

                cap_level = _choose_vp_cap(side, entry, vp_levels) if vp_levels else None

                if cap_level is not None and tps:
                    new_tps = list(tps)
                    idxs = [2] + ([1] if TA_VP_APPLY_TP2_TOO else [])
                    for i in idxs:
                        if i < len(new_tps):
                            if side == "LONG":
                                new_tps[i] = min(new_tps[i], cap_level)
                            else:
                                new_tps[i] = max(new_tps[i], cap_level)
                    # Окончательно упорядочим и округлим
                    new_tps = _ensure_tp_monotonic_with_step(side, entry, new_tps, atr_val, tick, TA_SAN_TP_MIN_STEP_ATR)
                    d["tps"] = tps = new_tps
                    breakdown["VPcap"] = breakdown.get("VPcap", 0.0) + 0.05
        except Exception:
            pass

        # 6) Macro gating
        try:
            if TA_G_MACRO_GATING and MACRO_EVENTS:
                mins = _minutes_to_next_event(MACRO_EVENTS)
                if mins is not None and mins <= TA_G_MACRO_MIN_BEFORE:
                    side_score -= 0.2; breakdown["macro"] = breakdown.get("macro", 0.0) - 0.2
        except Exception:
            pass

        # 7) “Запрет на вход слишком далеко от якорей”
        try:
            if atr_val and df15 is not None and len(df15) >= 50 and anchored_vwap:
                ts = df15["ts"]
                day_anchor = ts.iloc[-1].to_pydatetime().astimezone(timezone.utc).replace(hour=0,minute=0,second=0,microsecond=0)
                if hasattr(anchored_vwap, "__call__"):
                    vwap_day = anchored_vwap(df15, anchor=day_anchor)  # ожидаем серию/значение
                    vwap_val = float(vwap_day.iloc[-1] if hasattr(vwap_day, "iloc") else vwap_day)
                    dist = abs(float(entry) - vwap_val) / (atr_val + 1e-9)
                    if dist > TA_ANCHOR_MAX_DIST_ATR:
                        side_score -= 0.12; breakdown["anchorDist"] = breakdown.get("anchorDist", 0.0) - 0.12
        except Exception:
            pass

        d["score_breakdown"] = breakdown
        return float(side_score), side, d

    app["score_symbol_core"] = boosted
    logger and logger.info("TA booster enabled (sanity TP/SL/RR + microRR/IB/killzones/speed/Derivatives/RS + VP caps TP3 + trailing lock-in).")

# ====================== Warn-decay loop ======================
async def _warn_decay_loop(app):
    db = app.get("db")
    if not db or not db.conn or WARN_DECAY_DAYS <= 0:
        return
    await asyncio.sleep(5)
    while True:
        try:
            cur = await db.conn.execute("SELECT chat_id, user_id, warns, last_decay_at FROM moderation WHERE warns>0")
            rows = await cur.fetchall()
            now = _now_utc()
            for r in rows or []:
                chat_id = int(r["chat_id"]); user_id = int(r["user_id"]); warns = int(r["warns"])
                last_decay_at = datetime.fromisoformat(r["last_decay_at"]) if r["last_decay_at"] else None
                # опорная точка — максимум из последнего нарушения и последнего decay
                last_vi = await _last_violation_ts(db, chat_id, user_id)
                ref = max([x for x in [last_vi, last_decay_at] if x is not None], default=None)
                if ref is None:
                    ref = now - timedelta(days=WARN_DECAY_DAYS+1)
                if (now - ref) >= timedelta(days=WARN_DECAY_DAYS):
                    rec = await _get_mod_record(db, chat_id, user_id)
                    if rec.warns > 0:
                        rec.warns -= 1
                        rec.last_decay_at = now
                        await _save_mod_record(db, rec, note="auto warn decay")
            await asyncio.sleep(60 * 60 * 12)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(30)

# ====================== Command wrappers (cooldown) ======================
def _wrap_with_cooldown(handler):
    async def wrapped(app, message: Message, bot):
        if _is_group(message):
            now_ts = datetime.now().timestamp()
            if not _cmd_allowed(message.from_user.id, now_ts):
                with contextlib.suppress(Exception):
                    await message.reply(f"⏳ Подождите {CMD_COOLDOWN_SEC}с перед следующей командой.")
                return
        await handler(app, message, bot)
    return wrapped

# ====================== Admin morning greetings (06:00 MSK) ======================
ADMIN_JOKES = [
    "Купил на хаях, продал на лоях — зато опыт бесценный.",
    "У каждого трейдера есть план. До первой сделки.",
    "Самая точная стратегия: купить, когда страшно. Продать, когда страшнее.",
    "Рынок — отличный психолог. Быстро исправляет завышенную самооценку.",
    "Лучшая кнопка на бирже — «Выйти и подумать».",
]
ADMIN_WISHES = [
    "Больше TP, меньше SL! 🎯",
    "Пусть сегодня будет прибыльным! 🚀",
    "Плавных трендов и чётких пробоев! 📈",
    "Холодной головы и чётких сетапов! 💡",
]

COIN_MAP = [
    ("BTC", "BTC/USDT"),
    ("ETH", "ETH/USDT"),
    ("SOL", "SOL/USDT"),
    ("TON", "TON/USDT"),
    ("BNB", "BNB/USDT"),
]

async def _ensure_city_column(app):
    db = app.get("db")
    if not db or not db.conn:
        return
    try:
        await db.conn.execute("ALTER TABLE users ADD COLUMN city TEXT")
        await db.conn.commit()
    except Exception:
        pass

async def _get_city_admin(app, user_id: int) -> str:
    db = app.get("db")
    if not db or not db.conn:
        return "Москва"
    try:
        cur = await db.conn.execute("SELECT city FROM users WHERE user_id=?", (user_id,))
        row = await cur.fetchone()
        if row and row["city"]:
            return str(row["city"]).strip()
    except Exception:
        pass
    return "Москва"

def _wttr_celsius_line(city: str) -> str:
    # Без ключей; просим метрические данные, но на всякий случай конвертируем °F в °C
    import re
    try:
        url = f"https://wttr.in/{city}?format=%C+%t&m&lang=ru"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            txt = r.text.strip()  # пример: "Ясно +18°C" или "Sunny +64°F"
            m = re.search(r"([+\-]?\d+(?:\.\d+)?)°F", txt)
            if m:
                f = float(m.group(1))
                c = (f - 32.0) / 1.8
                txt = re.sub(r"([+\-]?\d+(?:\.\d+)?)°F", f"{c:.0f}°C", txt)
            return f"{city}: {txt}"
    except Exception:
        pass
    return f"{city}: погода недоступна"

def _seconds_until_next_msk(now_msk_fn, hour: int) -> float:
    now = now_msk_fn()
    target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    return max(1.0, (target - now).total_seconds())

async def _post_admin_greetings_once(app, bot):
    logger = app.get("logger")
    market = app.get("market")
    now_msk = app["now_msk"]

    admins = []
    try:
        admins = await app.get("db").get_admin_user_ids()
    except Exception:
        admins = []
    if not admins:
        return

    # Список цен
    prices_lines = []
    for sym, mkt in COIN_MAP:
        p = None
        with contextlib.suppress(Exception):
            p = market.fetch_mark_price(mkt)
        prices_lines.append(f"• {sym}: {_fmt_usd(p)}")

    for uid in admins:
        try:
            city = await _get_city_admin(app, uid)
            weather = _wttr_celsius_line(city)
            joke = random.choice(ADMIN_JOKES)
            wish = random.choice(ADMIN_WISHES)
            dt = now_msk()
            date_str = dt.strftime("%d.%m.%Y")
            time_str = dt.strftime("%H:%M")

            lines = [
                "☕ Доброе утро, админ!",
                "━━━━━━━━━━━━━━━━━━",
                weather,
                "",
                "📊 Ключевые цены:",
                *prices_lines,
                "",
                f"🕕 Время: {time_str} МСК • {date_str}",
                f"✨ {wish}",
                "",
                "Анекдот дня:",
                joke,
            ]
            msg = "\n".join(lines)
            with contextlib.suppress(Exception):
                await bot.send_message(uid, msg)
        except Exception as e:
            logger and logger.warning("Admin greet send fail to %s: %s", uid, e)

async def _start_daily_admin_greetings_loop(app, bot):
    logger = app.get("logger")
    await _ensure_city_column(app)
    await asyncio.sleep(2)
    logger and logger.info("Starting daily admin greetings loop at 06:00 MSK...")
    while True:
        try:
            wait_s = _seconds_until_next_msk(app["now_msk"], 6)
            await asyncio.sleep(wait_s)
            await _post_admin_greetings_once(app, bot)
        except asyncio.CancelledError:
            logger and logger.info("Admin greetings loop cancelled.")
            break
        except Exception as e:
            logger and logger.exception("Admin greetings loop error: %s", e)
            await asyncio.sleep(30)

# ====================== Patch entry ======================
def patch(app: Dict[str, Any]) -> None:
    logger = app.get("logger")
    router = app.get("router")

    # Команды в группах без подписки + подсказка лимита
    _patch_guard_access_for_groups(app)
    # TA booster
    _patch_ta_booster(app)

    # Startup
    orig_on_startup = app.get("on_startup")
    async def on_startup_patched(bot):
        app["bot_instance"] = bot
        db = app.get("db")
        if db and db.conn:
            await _ensure_tables(db)
        if orig_on_startup:
            await orig_on_startup(bot)
        # warn-decay цикл
        asyncio.create_task(_warn_decay_loop(app))
        # admin morning greetings at 06:00 MSK
        asyncio.create_task(_start_daily_admin_greetings_loop(app, bot))
        # safety: ensure users.city exists
        await _ensure_city_column(app)
        logger and logger.info("chat.py startup done: tables ready, warn-decay loop + admin greetings loop started.")
    app["on_startup"] = on_startup_patched

    # Перехват /signal: per-user lock + daily limit + recent anti-dup
    def _patch_cmd_signal():
        try:
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
            DAILY_LIMIT = app.get("DAILY_LIMIT", 3)
            market = app.get("market")  # для quick sanity

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
            orig = target.callback  # ссылка на оригинал

            async def cmd_signal_patched(message, bot):
                user_id = message.from_user.id
                # Per-user lock to prevent races
                lock = _USER_SIGNAL_LOCKS.setdefault(user_id, asyncio.Lock())
                async with lock:
                    st = await guard_access(message, bot)
                    if not st:
                        return

                    # Enforce daily limit
                    db = app.get("db")
                    if not db or not getattr(db, "conn", None):
                        with contextlib.suppress(Exception):
                            await message.answer("Сервис БД недоступен. Попробуйте чуть позже.")
                        return

                    unlimited = bool(st.get("unlimited")) or bool(st.get("admin"))
                    if not unlimited and int(st.get("count", 0)) >= int(DAILY_LIMIT):
                        with contextlib.suppress(Exception):
                            await message.answer("Лимит 3 сигнала в день исчерпан. Введите /code для безлимита.")
                        return

                    working_msg = await message.answer("🔍 Ищу торговую пару...")

                    try:
                        ranked = await rank_symbols_async(SYMBOLS)
                        if not ranked:
                            await edit_retry_html(working_msg, "Не удалось найти подходящий сигнал. Попробуйте позже.")
                            return

                        existing = await db.get_active_signals_for_user(user_id)

                        def is_active_dup(sym: str, side: str) -> bool:
                            return any(s.symbol == sym and s.side == side and s.active for s in existing)

                        # Подбор уникального кандидата
                        picked = None
                        for sym, det in ranked:
                            side = det["side"]
                            if is_active_dup(sym, side):
                                continue
                            if _recent_has(user_id, sym, side):
                                continue
                            if det.get("reject_reason"):
                                continue  # жёстко отбрасываем по sanity
                            picked = (sym, det)
                            break

                        # fallback: quick-скрининг
                        if picked is None:
                            best_q = None
                            for sym in SYMBOLS:
                                q = score_symbol_quick(sym)
                                if not q:
                                    continue
                                sside, det = q
                                if is_active_dup(sym, sside) or _recent_has(user_id, sym, sside):
                                    continue
                                det["side"] = sside
                                # Санити-фиксация quick‑деталей: TP монотон, шаг, ATR/PDR капы, RR-порог
                                try:
                                    df15_q = None
                                    with contextlib.suppress(Exception):
                                        df15_q = market.fetch_ohlcv(sym, "15m", 240)
                                    tick_q = 0.0
                                    with contextlib.suppress(Exception):
                                        tick_q = float(market.get_tick_size(sym) or 0.0)
                                    entry_q = float(det.get("c5")); sl_q = float(det.get("sl")); tps_q = list(det.get("tps") or [])
                                    atr_q = float(det.get("atr", 0.0))
                                    if (not atr_q) and df15_q is not None and len(df15_q)>=20:
                                        with contextlib.suppress(Exception):
                                            atr_q = float(app.get("atr")(df15_q, 14).iloc[-1])
                                    ok_q, sl_q2, tps_q2, _info_q = _sanity_tp_sl_rr_fix(sym, sside, entry_q, sl_q, tps_q, df15_q, atr_q, tick_q)
                                    if not ok_q:
                                        continue
                                    det["sl"] = sl_q2; det["tps"] = tps_q2
                                except Exception:
                                    continue

                                if (best_q is None) or (float(det.get("score", 0.0)) > float(best_q[1].get("score", 0.0))):
                                    best_q = (sym, det)
                            if best_q:
                                picked = best_q

                        if picked is None:
                            await edit_retry_html(working_msg, "Сейчас нет нового уникального сигнала для вас.\nПопробуйте позже — я подберу другой сетап.")
                            return

                        symbol, details = picked
                        side = details["side"]; entry = details["c5"]; sl = details["sl"]; tps = details["tps"]
                        leverage = details["leverage"]; risk_level = details["risk_level"]
                        news_note = details["news_note"]; atr_value = details["atr"]; watch_seconds = details["watch_seconds"]
                        reason = app["build_reason"](details)

                        # Валидация
                        if side == "LONG":
                            if not (all(tp > entry for tp in tps) and sl < entry):
                                await edit_retry_html(working_msg, "Сигнал не прошёл валидацию. Попробуйте ещё раз.")
                                return
                        else:
                            if not (all(tp < entry for tp in tps) and sl > entry):
                                await edit_retry_html(working_msg, "Сигнал не прошёл валидацию. Попробуйте ещё раз.")
                                return

                        sig = Signal(
                            user_id=user_id, symbol=symbol, side=side, entry=entry, tps=tps, sl=sl,
                            leverage=leverage, risk_level=risk_level, created_at=now_msk(),
                            news_note=news_note, atr_value=atr_value,
                            watch_until=now_msk() + timedelta(seconds=watch_seconds), reason=reason,
                        )
                        text = format_signal_message(sig)
                        await edit_retry_html(working_msg, text)

                        # Count +1
                        st["count"] = int(st.get("count", 0)) + 1
                        await db.save_user_state(user_id, st)

                        sig.id = await db.add_signal(sig)
                        task = asyncio.create_task(app["watch_signal_price"](bot, message.chat.id, sig))
                        active_watch_tasks.setdefault(user_id, []).append(task)
                        if MET_SIGNALS_GEN: MET_SIGNALS_GEN.inc()

                        # Запомнить недавний сигнал для пользователя
                        _recent_add(user_id, symbol, side)

                    except Exception as e:
                        logger and logger.exception("Signal generation error: %s", e)
                        with contextlib.suppress(Exception):
                            await edit_retry_html(working_msg, "⚠️ Произошла ошибка при поиске сигнала.")

            setattr(target, "callback", cmd_signal_patched)
            logger and logger.info("Signal handler patched: per-user lock + daily limit + recent anti-dup + sanity filter enabled (DB dynamic).")
        except Exception as e:
            logger and logger.warning("Signal handler patch error: %s", e)

    _patch_cmd_signal()

    # Автокомментарий при автофорвардах из канала
    async def _h_autocomment(message: Message):
        try:
            await _auto_comment_for_channel_post(app, message)
        except Exception:
            logger and logger.exception("autocomment handler error")
    router.message.register(_h_autocomment, F.chat.type.in_({"group","supergroup"}), F.is_automatic_forward == True)

    # Модерация сообщений
    async def _h_group(message: Message):
        try:
            await on_group_message(app, message, app.get("bot_instance"))
        except Exception:
            logger and logger.exception("group moderation handler error")
    router.message.register(_h_group, F.chat.type.in_({"group","supergroup"}))

    # Приветствие / CAPTCHA
    async def _h_member(upd: ChatMemberUpdated):
        try:
            await on_member_update(app, upd)
        except Exception:
            logger and logger.exception("member update handler error")
    router.chat_member.register(_h_member)

    # Callbacks: verify + rules_full + mod actions
    async def _h_cb(cb: CallbackQuery):
        try:
            if cb.data and cb.data.startswith("verify:"):
                await cb_verify(app, cb)
            elif cb.data == "rules_full":
                await cb_rules_full(app, cb)
            elif cb.data and cb.data.startswith("modact:"):
                await cb_mod_action(app, cb)
        except Exception:
            with contextlib.suppress(Exception):
                await cb.answer()
            logger and logger.exception("callback handler error")
    router.callback_query.register(_h_cb)

    # Публичные / мод-команды
    async def _h_rules(message: Message): await cmd_rules(app, message, app.get("bot_instance"))
    async def _h_report(message: Message): await cmd_report(app, message, app.get("bot_instance"))

    async def _h_warn(message: Message): await cmd_warn(app, message, app.get("bot_instance"))
    async def _h_mute(message: Message): await cmd_mute(app, message, app.get("bot_instance"))
    async def _h_ban(message: Message): await cmd_ban(app, message, app.get("bot_instance"))
    async def _h_unmute(message: Message): await cmd_unmute(app, message, app.get("bot_instance"))
    async def _h_unban(message: Message): await cmd_unban(app, message, app.get("bot_instance"))
    async def _h_warns(message: Message): await cmd_warns(app, message, app.get("bot_instance"))

    router.message.register(_h_rules, F.chat.type.in_({"group","supergroup"}), F.text.startswith("/rules"))
    router.message.register(_h_report, F.chat.type.in_({"group","supergroup"}), F.text.startswith("/report"))

    router.message.register(_h_warn, F.chat.type.in_({"group","supergroup"}), F.text.startswith("/warn"))
    router.message.register(_h_mute, F.chat.type.in_({"group","supergroup"}), F.text.startswith("/mute"))
    router.message.register(_h_ban, F.chat.type.in_({"group","supergroup"}), F.text.startswith("/ban"))
    router.message.register(_h_unmute, F.chat.type.in_({"group","supergroup"}), F.text.startswith("/unmute"))
    router.message.register(_h_unban, F.chat.type.in_({"group","supergroup"}), F.text.startswith("/unban"))
    router.message.register(_h_warns, F.chat.type.in_({"group","supergroup"}), F.text.startswith("/warns"))

    logger and logger.info("chat.py patch applied: moderation + autocomment + CAPTCHA + report buttons + TA booster + warn-decay + limit tips + signal anti-dup + admin greetings 06:00 MSK (DB dynamic).")
