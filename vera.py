# vera.py
# PRO-Subscription patch (monthly) with inline purchase flow, admin confirmation and status UI.
# Connect via TA_PATCH_MODULES=...,vera (load after all other patches).

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
import os
import asyncio
import contextlib
from datetime import datetime, timedelta, timezone

from aiogram import F
from aiogram.filters import Command
from aiogram.types import (
    Message, CallbackQuery,
    InlineKeyboardMarkup, InlineKeyboardButton,
    ReplyKeyboardMarkup, KeyboardButton,
)

# ===================== CONFIG =====================
PRO_PRICE_USD = float(os.getenv("PRO_PRICE_USD", "10"))
PRO_DURATION_DAYS = int(os.getenv("PRO_DURATION_DAYS", "30"))
PRO_TON_ADDRESS = os.getenv("PRO_TON_ADDRESS", "UQDdVr_BV6q5RnemNhYYZOA7l2o8EaCwdar45t0QXGqmzQD3").strip()

# Expiry check interval and pre-notify window
PRO_EXPIRY_SCAN_SEC = int(os.getenv("PRO_EXPIRY_SCAN_SEC", "600"))   # 10 min
PRO_PRENOTIFY_HOURS = int(os.getenv("PRO_PRENOTIFY_HOURS", "24"))    # за 24ч предупредить

# Limits (fallback defaults; real limits originate in app/neon/ta)
LIMIT_SIGNALS = int(os.getenv("DAILY_LIMIT", "3"))
LIMIT_TA = int(os.getenv("NEON_TA_DAILY_LIMIT", "4"))
LIMIT_ANALYSIS = int(os.getenv("NEON_ANALYSIS_DAILY_LIMIT", "5"))

# ===================== HELPERS =====================
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _fmt_dt_msk(app: Dict[str, Any], dt: datetime) -> str:
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        msk_ref = app["now_msk"]()
        msk_dt = dt.astimezone(msk_ref.tzinfo)
        return msk_dt.strftime("%d.%m.%Y %H:%M") + " МСК"
    except Exception:
        return dt.strftime("%Y-%m-%d %H:%M")

def _fmt_time_left(msk_now: datetime, msk_until: datetime) -> str:
    delta = msk_until - msk_now
    if delta.total_seconds() <= 0:
        return "0ч"
    days = delta.days
    hours = delta.seconds // 3600
    mins = (delta.seconds % 3600) // 60
    if days > 0:
        return f"{days}д {hours}ч"
    if hours > 0:
        return f"{hours}ч {mins}м"
    return f"{mins}м"

async def _ensure_pro_columns(app: Dict[str, Any]) -> None:
    db = app.get("db")
    if not db or not getattr(db, "conn", None):
        return
    stmts = [
        "ALTER TABLE users ADD COLUMN pro_until TEXT",
        "ALTER TABLE users ADD COLUMN pro_notified INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE users ADD COLUMN pro_pre_notified INTEGER NOT NULL DEFAULT 0",
    ]
    for s in stmts:
        try:
            await db.conn.execute(s)
        except Exception:
            pass
    with contextlib.suppress(Exception):
        await db.conn.commit()

async def _get_user_row(app: Dict[str, Any], user_id: int) -> Dict[str, Any]:
    db = app.get("db")
    row = {}
    try:
        cur = await db.conn.execute(
            "SELECT user_id, date, count, unlimited, support_mode, admin, "
            "pro_until, pro_notified, pro_pre_notified, "
            "ta_count, ta_date, analysis_count, analysis_date "
            "FROM users WHERE user_id=?",
            (user_id,)
        )
        r = await cur.fetchone()
        if r:
            row = dict(r)
    except Exception:
        pass
    return row

def _is_pro_active_row(row: Dict[str, Any]) -> bool:
    try:
        pu = row.get("pro_until")
        if not pu:
            return False
        dt = datetime.fromisoformat(pu)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt > _now_utc()
    except Exception:
        return False

async def _set_pro(app: Dict[str, Any], user_id: int, days: int) -> Optional[datetime]:
    db = app.get("db")
    if not db or not getattr(db, "conn", None):
        return None
    try:
        # Extend from current pro_until if still active, else from now
        base = _now_utc()
        cur = await db.conn.execute("SELECT pro_until FROM users WHERE user_id=?", (user_id,))
        r = await cur.fetchone()
        if r and r["pro_until"]:
            try:
                cur_dt = datetime.fromisoformat(r["pro_until"])
                if cur_dt.tzinfo is None:
                    cur_dt = cur_dt.replace(tzinfo=timezone.utc)
                if cur_dt > base:
                    base = cur_dt
            except Exception:
                pass
        new_until = base + timedelta(days=max(1, int(days)))
        await db.conn.execute(
            "UPDATE users SET pro_until=?, pro_notified=0, pro_pre_notified=0, unlimited=1 WHERE user_id=?",
            (new_until.isoformat(), user_id)
        )
        await db.conn.commit()
        return new_until
    except Exception:
        return None

async def _clear_unlimited_if_expired(app: Dict[str, Any], user_id: int) -> bool:
    db = app.get("db")
    try:
        cur = await db.conn.execute("SELECT pro_until, unlimited FROM users WHERE user_id=?", (user_id,))
        r = await cur.fetchone()
        if not r:
            return False
        pu = r["pro_until"]
        uni = int(r["unlimited"] or 0)
        if not pu:
            return False
        dt = datetime.fromisoformat(pu)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt <= _now_utc() and uni == 1:
            await db.conn.execute("UPDATE users SET unlimited=0 WHERE user_id=?", (user_id,))
            await db.conn.commit()
            return True
    except Exception:
        pass
    return False

def _today_key_msk(app: Dict[str, Any]) -> str:
    f = app.get("today_key")
    try:
        return f() if callable(f) else (_now_utc() + timedelta(hours=3)).strftime("%Y-%m-%d")
    except Exception:
        return (_now_utc() + timedelta(hours=3)).strftime("%Y-%m-%d")

async def _limits_snapshot(app: Dict[str, Any], user_id: int) -> Tuple[int, int, int]:
    row = await _get_user_row(app, user_id)
    dkey = _today_key_msk(app)

    cnt_sig = int(row.get("count", 0) or 0)
    date_sig = row.get("date") or dkey
    sig_left = max(0, LIMIT_SIGNALS - (0 if date_sig != dkey else cnt_sig))

    cnt_ta = int(row.get("ta_count", 0) or 0)
    date_ta = row.get("ta_date") or dkey
    ta_left = max(0, LIMIT_TA - (0 if date_ta != dkey else cnt_ta))

    cnt_an = int(row.get("analysis_count", 0) or 0)
    date_an = row.get("analysis_date") or dkey
    an_left = max(0, LIMIT_ANALYSIS - (0 if date_an != dkey else cnt_an))

    return sig_left, ta_left, an_left

# ===================== UI BUILDERS =====================
def _kb_get_pro() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="💎 Получить PRO статус", callback_data="vera:get_pro")]
    ])

def _kb_pay() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔙 Назад", callback_data="vera:back"),
         InlineKeyboardButton(text="✅ Оплатил", callback_data="vera:paid")]
    ])

def _kb_admin_confirm(user_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="❌ Не оплатил", callback_data=f"vera:adm:no:{user_id}"),
         InlineKeyboardButton(text="✅ Оплатил", callback_data=f"vera:adm:yes:{user_id}")]
    ])

def _menu_with_status(orig_menu, is_admin: bool = False):
    try:
        kb = orig_menu(is_admin) if callable(orig_menu) else None
        if isinstance(kb, ReplyKeyboardMarkup):
            texts = [btn.text for row in (kb.keyboard or []) for btn in row if hasattr(btn, "text")]
            if "🧾 Мой статус" not in texts:
                kb.keyboard.append([KeyboardButton(text="🧾 Мой статус")])
            return kb
    except Exception:
        pass
    try:
        return ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="📈 Получить сигнал")],
                [KeyboardButton(text="🔎 Анализ монеты")],
                [KeyboardButton(text="🧭 Технический анализ монеты")],
                [KeyboardButton(text="🧾 Мой статус")],
                [KeyboardButton(text="ℹ️ Помощь")],
                [KeyboardButton(text="🛟 Поддержка")],
            ],
            resize_keyboard=True
        )
    except Exception:
        return None

def _msg_offer_easy(reason: str = "") -> str:
    reason_map = {
        "signals": "Сигналы на сегодня исчерпаны.",
        "ta": "Технический анализ на сегодня исчерпан.",
        "analysis": "Анализ монеты на сегодня исчерпан.",
        "": "Лимит на сегодня исчерпан.",
    }
    rtxt = reason_map.get(reason, reason_map[""])
    return (
        "✨ <b>Ваш статус: EASY</b>\n"
        f"⚠️ {rtxt}\n\n"
        "Хотите перейти на <b>PRO</b>?\n\n"
        "Что даёт <b>PRO</b>:\n"
        "• ♾️ <b>Безлимитные сигналы</b>\n"
        "• ♾️ <b>Безлимитный доступ к анализу</b> (монета + теханализ)\n\n"
        f"Стоимость: <b>${PRO_PRICE_USD:.0f}</b> / <b>{PRO_DURATION_DAYS} дней</b>\n"
        "Сеть: <b>TON</b>"
    )

def _msg_payment_details() -> str:
    return (
        "💳 <b>Получение статуса PRO</b>\n"
        f"Стоимость: <b>${PRO_PRICE_USD:.0f}</b> / <b>{PRO_DURATION_DAYS} дней</b>\n"
        f"Адрес получателя (TON):\n<code>{PRO_TON_ADDRESS}</code>\n"
        "Сеть: <b>TON</b>\n\n"
        "После произведения оплаты нажмите кнопку <b>«Оплатил»</b> ниже."
    )

def _msg_profile_easy(name: str, sig_l: int, ta_l: int, an_l: int) -> str:
    return (
        f"👤 Вы — <b>{name}</b>\n"
        "💼 Ваш статус: <b>EASY</b>\n\n"
        "🔢 Лимиты на сегодня:\n"
        f"• 📈 Сигналы: <b>{sig_l}</b> из 3\n"
        f"• 🧭 Тех.анализ: <b>{ta_l}</b> из 4\n"
        f"• 🔎 Анализ монеты: <b>{an_l}</b> из 5\n\n"
        "Хотите получить <b>PRO</b>?"
    )

def _msg_profile_pro(app: Dict[str, Any], name: str, until: datetime) -> str:
    msk_now = app["now_msk"]()
    msk_until = until.astimezone(msk_now.tzinfo) if until.tzinfo else msk_now
    left_txt = _fmt_time_left(msk_now, msk_until)
    return (
        f"👤 Вы — <b>{name}</b>\n"
        "💼 Ваш статус: <b>PRO</b>\n"
        f"⏳ Действует до: <b>{_fmt_dt_msk(app, until)}</b>\n"
        f"⌛ Осталось: <b>{left_txt}</b>\n\n"
        "♾️ У вас нет лимитов."
    )

# ===================== NOTIFY / OFFER =====================
async def _offer_pro(app: Dict[str, Any], message: Message, reason: str = ""):
    row = await _get_user_row(app, message.from_user.id)
    if _is_pro_active_row(row) or int(row.get("unlimited", 0)) == 1:
        return
    logger = app.get("logger")
    send_retry_html = app.get("send_retry_html")
    text = _msg_offer_easy(reason)
    try:
        if send_retry_html:
            await send_retry_html(app.get("bot_instance"), message.chat.id, text, reply_markup=_kb_get_pro())
        else:
            await message.answer(text, reply_markup=_kb_get_pro())
        logger and logger.info("VERA: PRO offer sent to user_id=%s (reason=%s)", getattr(message.from_user, "id", None), reason)
    except Exception:
        pass

async def _send_payment(app: Dict[str, Any], chat_id: int):
    send_retry_html = app.get("send_retry_html")
    text = _msg_payment_details()
    try:
        if send_retry_html:
            await send_retry_html(app.get("bot_instance"), chat_id, text, reply_markup=_kb_pay())
        else:
            bot = app.get("bot_instance")
            await bot.send_message(chat_id, text, reply_markup=_kb_pay())
    except Exception:
        pass

# ===================== EXPIRY LOOP =====================
async def _pro_expiry_loop(app: Dict[str, Any]):
    logger = app.get("logger")
    await asyncio.sleep(3)
    try:
        await _ensure_pro_columns(app)
    except Exception:
        pass
    db = app.get("db")
    bot = app.get("bot_instance")
    if not db or not getattr(db, "conn", None) or not bot:
        return
    logger and logger.info("VERA: PRO expiry loop started (interval %ss).", PRO_EXPIRY_SCAN_SEC)
    while True:
        try:
            cur = await db.conn.execute(
                "SELECT user_id, pro_until, pro_notified, pro_pre_notified FROM users WHERE pro_until IS NOT NULL"
            )
            rows = await cur.fetchall()
            for r in rows or []:
                # convert to dict to avoid sqlite3.Row.get issues
                row = dict(r)
                uid = int(row.get("user_id"))
                pu = row.get("pro_until")
                notified = int(row.get("pro_notified", 0) or 0)
                pre_notified = int(row.get("pro_pre_notified", 0) or 0)
                if not pu:
                    continue
                try:
                    dt = datetime.fromisoformat(pu)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                except Exception:
                    continue

                now = _now_utc()
                # PRE-NOTIFY
                if dt > now and not pre_notified:
                    hours_left = (dt - now).total_seconds() / 3600.0
                    if hours_left <= PRO_PRENOTIFY_HOURS:
                        with contextlib.suppress(Exception):
                            msk_now = app["now_msk"]()
                            msk_until = dt.astimezone(msk_now.tzinfo)
                            left_txt = _fmt_time_left(msk_now, msk_until)
                            await bot.send_message(
                                uid,
                                f"⏳ До окончания вашего статуса PRO осталось <b>{left_txt}</b>.\n"
                                f"Дата окончания: <b>{_fmt_dt_msk(app, dt)}</b>"
                            )
                        with contextlib.suppress(Exception):
                            await db.conn.execute("UPDATE users SET pro_pre_notified=1 WHERE user_id=?", (uid,))
                            await db.conn.commit()
                        logger and logger.info("VERA: PRO pre-notified user_id=%s", uid)

                # EXPIRY
                if dt <= now:
                    changed = await _clear_unlimited_if_expired(app, uid)
                    if not notified:
                        with contextlib.suppress(Exception):
                            await bot.send_message(
                                uid,
                                "⏰ Срок действия вашего статуса PRO истёк.\n"
                                "Хотите продлить PRO на 30 дней?",
                                reply_markup=_kb_get_pro()
                            )
                        with contextlib.suppress(Exception):
                            await db.conn.execute("UPDATE users SET pro_notified=1 WHERE user_id=?", (uid,))
                            await db.conn.commit()
                        logger and logger.info("VERA: PRO expired and notified user_id=%s", uid)
            await asyncio.sleep(PRO_EXPIRY_SCAN_SEC)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger and logger.warning("VERA: expiry loop error: %s", e)
            await asyncio.sleep(10)

# ===================== HANDLERS =====================
async def _handle_myprofile(app: Dict[str, Any], message: Message):
    logger = app.get("logger")
    await _ensure_pro_columns(app)
    bot = app.get("bot_instance")
    send_retry_html = app.get("send_retry_html")
    try:
        uid = message.from_user.id
        row = await _get_user_row(app, uid)
        name_fn = app.get("user_display_name")
        name = None
        try:
            if callable(name_fn):
                name = name_fn(message.from_user)
        except Exception:
            name = None
        name = name or (getattr(message.from_user, "username", None) and f"@{message.from_user.username}") or (getattr(message.from_user, "first_name", "") or "пользователь")

        if _is_pro_active_row(row):
            until = datetime.fromisoformat(row["pro_until"])
            if until.tzinfo is None:
                until = until.replace(tzinfo=timezone.utc)
            txt = _msg_profile_pro(app, name, until)
            # Если PRO — НЕ показываем кнопки покупки
            if send_retry_html:
                await send_retry_html(bot, message.chat.id, txt)
            else:
                await message.answer(txt)
        else:
            sig_l, ta_l, an_l = await _limits_snapshot(app, uid)
            txt = _msg_profile_easy(name, sig_l, ta_l, an_l)
            if send_retry_html:
                await send_retry_html(bot, message.chat.id, txt, reply_markup=_kb_get_pro())
            else:
                await message.answer(txt, reply_markup=_kb_get_pro())
        logger and logger.info("VERA: myprofile shown to user_id=%s", uid)
    except Exception as e:
        logger and logger.warning("VERA: myprofile error: %s", e)

async def _handle_pro_cmd(app: Dict[str, Any], message: Message):
    row = await _get_user_row(app, message.from_user.id)
    if _is_pro_active_row(row) or int(row.get("unlimited", 0)) == 1:
        try:
            until = datetime.fromisoformat(row["pro_until"]) if row.get("pro_until") else None
            if until and until.tzinfo is None:
                until = until.replace(tzinfo=timezone.utc)
        except Exception:
            until = None
        msk_now = app["now_msk"]()
        txt = "💼 У вас уже активен статус <b>PRO</b>."
        if until:
            left_txt = _fmt_time_left(msk_now, until.astimezone(msk_now.tzinfo))
            txt += f"\n⏳ Действует до: <b>{_fmt_dt_msk(app, until)}</b>\n⌛ Осталось: <b>{left_txt}</b>"
        with contextlib.suppress(Exception):
            await message.answer(txt)
        return
    await _send_payment(app, message.chat.id)

async def _cb_get_pro(app: Dict[str, Any], cb: CallbackQuery):
    row = await _get_user_row(app, cb.from_user.id)
    if _is_pro_active_row(row) or int(row.get("unlimited", 0)) == 1:
        with contextlib.suppress(Exception):
            msk_now = app["now_msk"]()
            until = datetime.fromisoformat(row["pro_until"]) if row.get("pro_until") else None
            if until and until.tzinfo is None:
                until = until.replace(tzinfo=timezone.utc)
            left_txt = _fmt_time_left(msk_now, until.astimezone(msk_now.tzinfo)) if until else ""
            await cb.answer("У вас уже активен PRO.", show_alert=True)
            await cb.message.answer(
                f"💼 Статус <b>PRO</b> уже активен.\n"
                f"⏳ До: <b>{_fmt_dt_msk(app, until)}</b>\n"
                f"⌛ Осталось: <b>{left_txt}</b>"
            )
        return
    with contextlib.suppress(Exception):
        await cb.answer()
    await _send_payment(app, cb.message.chat.id)

async def _cb_back(app: Dict[str, Any], cb: CallbackQuery):
    with contextlib.suppress(Exception):
        await cb.answer("Возврат в меню")
    menu = app.get("main_menu_kb")
    bot = app.get("bot_instance")
    send_retry_html = app.get("send_retry_html")
    try:
        kb = menu(False) if callable(menu) else None
        if send_retry_html:
            await send_retry_html(bot, cb.message.chat.id, "Вы вернулись к основному функционалу.", reply_markup=kb)
        else:
            await bot.send_message(cb.message.chat.id, "Вы вернулись к основному функционалу.", reply_markup=kb)
    except Exception:
        pass

async def _cb_paid(app: Dict[str, Any], cb: CallbackQuery):
    logger = app.get("logger")
    bot = app.get("bot_instance")
    db = app.get("db")
    send_retry_html = app.get("send_retry_html")
    with contextlib.suppress(Exception):
        await cb.answer("Проверяю оплату...")
    try:
        uid = cb.from_user.id
        row = await _get_user_row(app, uid)
        if _is_pro_active_row(row) or int(row.get("unlimited", 0)) == 1:
            with contextlib.suppress(Exception):
                await cb.message.answer("💼 У вас уже активен статус PRO — оплата не требуется.")
            return

        if send_retry_html:
            await send_retry_html(bot, cb.message.chat.id, "🔎 Проверяю оплату... Ожидаю подтверждения администратора.")
        else:
            await bot.send_message(cb.message.chat.id, "🔎 Проверяю оплату... Ожидаю подтверждения администратора.")
        admins = []
        with contextlib.suppress(Exception):
            admins = await db.get_admin_user_ids()
        name_fn = app.get("user_display_name")
        disp = None
        try:
            if callable(name_fn):
                disp = name_fn(cb.from_user)
        except Exception:
            disp = None
        disp = disp or (getattr(cb.from_user, "username", None) and f"@{cb.from_user.username}") or (getattr(cb.from_user, "first_name", "") or f"id {uid}")
        for aid in admins or []:
            with contextlib.suppress(Exception):
                await bot.send_message(
                    aid,
                    f"🧾 Запрос на PRO: пользователь {disp} (id {uid}) сообщил, что <b>оплатил</b> подписку PRO.\n"
                    "Нажмите для подтверждения:",
                    reply_markup=_kb_admin_confirm(uid)
                )
        logger and logger.info("VERA: payment claimed by user_id=%s (sent to %d admins)", uid, len(admins or []))
    except Exception as e:
        logger and logger.warning("VERA: paid callback error: %s", e)

async def _cb_admin_confirm(app: Dict[str, Any], cb: CallbackQuery):
    logger = app.get("logger")
    bot = app.get("bot_instance")
    db = app.get("db")
    data = cb.data or ""
    parts = data.split(":")
    if len(parts) != 4 or parts[0] != "vera" or parts[1] != "adm":
        with contextlib.suppress(Exception):
            await cb.answer()
        return
    action = parts[2]
    try:
        target_id = int(parts[3])
    except Exception:
        with contextlib.suppress(Exception):
            await cb.answer("Ошибка")
        return

    try:
        admins = await db.get_admin_user_ids()
    except Exception:
        admins = []
    if cb.from_user.id not in (admins or []):
        with contextlib.suppress(Exception):
            await cb.answer("Только для администраторов", show_alert=True)
        return

    if action == "no":
        with contextlib.suppress(Exception):
            await cb.answer("Отклонено")
        with contextlib.suppress(Exception):
            await bot.send_message(target_id, "❌ Оплата не подтверждена. Пожалуйста, обратитесь в поддержку.")
        logger and logger.info("VERA: admin %s marked NOT PAID for user_id=%s", cb.from_user.id, target_id)
        return

    if action == "yes":
        with contextlib.suppress(Exception):
            await cb.answer("Подтверждено")
        until = await _set_pro(app, target_id, PRO_DURATION_DAYS)
        if until:
            with contextlib.suppress(Exception):
                await bot.send_message(
                    target_id,
                    f"✅ Оплата прошла успешно. Вы получили статус <b>PRO</b>!\n"
                    f"Действует до: <b>{_fmt_dt_msk(app, until)}</b>\n\n"
                    "Спасибо, что поддерживаете проект!"
                )
            logger and logger.info("VERA: admin %s CONFIRMED PRO for user_id=%s until %s", cb.from_user.id, target_id, until.isoformat())
        else:
            with contextlib.suppress(Exception):
                await bot.send_message(target_id, "⚠️ Не удалось активировать PRO. Напишите в поддержку.")
            logger and logger.warning("VERA: failed to set PRO for user_id=%s", target_id)
        return

# ===================== LIMIT INTERCEPTS =====================
def _patch_cmd_signal_offer(app: Dict[str, Any]):
    logger = app.get("logger")
    router = app.get("router")
    if not router:
        return
    obs = router.message
    handlers = getattr(obs, "handlers", [])
    target = None
    for h in handlers:
        cb = getattr(h, "callback", None)
        name = getattr(cb, "__name__", "")
        if "cmd_signal" in name:  # match cmd_signal, cmd_signal_patched, cmd_signal_wrapped, etc.
            target = h
            break
    if not target:
        logger and logger.warning("VERA: cmd_signal handler not found to patch (will rely on fallback wrapper).")
        return
    orig = target.callback

    async def cmd_signal_wrapped(message: Message, bot):
        st = await app.get("guard_access")(message, bot)
        if not st:
            return
        await _ensure_pro_columns(app)
        uid = message.from_user.id
        with contextlib.suppress(Exception):
            await _clear_unlimited_if_expired(app, uid)
        row = await _get_user_row(app, uid)
        if _is_pro_active_row(row) or int(row.get("unlimited", 0)) == 1:
            return await orig(message, bot)
        cnt = int(st.get("count", 0) or 0)
        if cnt >= LIMIT_SIGNALS:
            return await _offer_pro(app, message, reason="signals")
        return await orig(message, bot)

    setattr(target, "callback", cmd_signal_wrapped)
    logger and logger.info("VERA: cmd_signal patched (limit → PRO offer).")

def _patch_neon_analysis_offer(app: Dict[str, Any]):
    logger = app.get("logger")
    try:
        import sys
        neon = sys.modules.get("neon")
        if not neon or not hasattr(neon, "_analysis_only_flow"):
            logger and logger.warning("VERA: neon._analysis_only_flow not found.")
            return
        orig = neon._analysis_only_flow

        async def analysis_flow_wrapped(app_, message: Message, bot, user_id: int, symbol: str):
            await _ensure_pro_columns(app_)
            row = await _get_user_row(app_, user_id)
            with contextlib.suppress(Exception):
                await _clear_unlimited_if_expired(app_, user_id)
            if _is_pro_active_row(row) or int(row.get("unlimited", 0)) == 1:
                return await orig(app_, message, bot, user_id, symbol)
            dkey = _today_key_msk(app_)
            cnt = int(row.get("analysis_count", 0) or 0)
            date = row.get("analysis_date") or ""
            if (date == dkey and cnt >= LIMIT_ANALYSIS) or (date != dkey and LIMIT_ANALYSIS <= 0):
                return await _offer_pro(app_, message, reason="analysis")
            return await orig(app_, message, bot, user_id, symbol)

        neon._analysis_only_flow = analysis_flow_wrapped
        logger and logger.info("VERA: neon._analysis_only_flow patched (limit → PRO offer).")
    except Exception as e:
        logger and logger.warning("VERA: neon patch error: %s", e)

def _patch_ta_offer(app: Dict[str, Any]):
    logger = app.get("logger")
    try:
        import sys
        ta = sys.modules.get("ta")
        if not ta or not hasattr(ta, "_do_ta_flow"):
            logger and logger.warning("VERA: ta._do_ta_flow not found.")
            return
        orig = ta._do_ta_flow

        async def ta_flow_wrapped(app_, message: Message, bot, user_id: int, symbol: str):
            await _ensure_pro_columns(app_)
            row = await _get_user_row(app_, user_id)
            with contextlib.suppress(Exception):
                await _clear_unlimited_if_expired(app_, user_id)
            if _is_pro_active_row(row) or int(row.get("unlimited", 0)) == 1:
                return await orig(app_, message, bot, user_id, symbol)
            dkey = _today_key_msk(app_)
            cnt = int(row.get("ta_count", 0) or 0)
            date = row.get("ta_date") or ""
            if (date == dkey and cnt >= LIMIT_TA) or (date != dkey and LIMIT_TA <= 0):
                return await _offer_pro(app_, message, reason="ta")
            return await orig(app_, message, bot, user_id, symbol)

        ta._do_ta_flow = ta_flow_wrapped
        logger and logger.info("VERA: ta._do_ta_flow patched (limit → PRO offer).")
    except Exception as e:
        logger and logger.warning("VERA: ta patch error: %s", e)

# ===================== PATCH ENTRY =====================
def patch(app: Dict[str, Any]) -> None:
    logger = app.get("logger")
    router = app.get("router")

    logger and logger.info("VERA: initializing PRO columns...")
    orig_on_startup = app.get("on_startup")

    async def _on_startup_vera(bot):
        await _ensure_pro_columns(app)
        asyncio.create_task(_pro_expiry_loop(app))
        if orig_on_startup:
            await orig_on_startup(bot)
        logger and logger.info("VERA: startup done, expiry loop scheduled.")

    app["on_startup"] = _on_startup_vera

    # Main menu patch (add "🧾 Мой статус")
    orig_menu = app.get("main_menu_kb")
    app["main_menu_kb"] = lambda is_admin=False: _menu_with_status(orig_menu, is_admin)
    logger and logger.info("VERA: main menu patched (added 🧾 Мой статус).")

    # Direct handlers (Command + button)
    if router:
        async def _h_myprofile(message: Message):
            await _handle_myprofile(app, message)

        async def _h_pro(message: Message):
            await _handle_pro_cmd(app, message)

        # Commands with Command filter (handles /cmd@botname)
        router.message.register(_h_myprofile, F.chat.type == "private", Command("myprofile"))
        router.message.register(_h_pro, F.chat.type == "private", Command("pro"))
        # Button text
        router.message.register(_h_myprofile, F.chat.type == "private", F.text == "🧾 Мой статус")

        # Callbacks: our router for vera:*
        async def _cb_router(cb: CallbackQuery):
            data = cb.data or ""
            try:
                if data == "vera:get_pro":
                    logger and logger.info("VERA: cb get_pro from user_id=%s", cb.from_user.id)
                    await _cb_get_pro(app, cb)
                elif data == "vera:back":
                    logger and logger.info("VERA: cb back from user_id=%s", cb.from_user.id)
                    await _cb_back(app, cb)
                elif data == "vera:paid":
                    logger and logger.info("VERA: cb paid from user_id=%s", cb.from_user.id)
                    await _cb_paid(app, cb)
                elif data.startswith("vera:adm:"):
                    logger and logger.info("VERA: cb admin confirm: %s by admin_id=%s", data, cb.from_user.id)
                    await _cb_admin_confirm(app, cb)
            except Exception:
                with contextlib.suppress(Exception):
                    await cb.answer()

        # 1) Прямая регистрация vera:* (на случай если обёртка не сработает)
        router.callback_query.register(_cb_router, F.data.startswith("vera:"))

        # 2) Оборачиваем самый ранний «общий» callback-хендлер,
        #    чтобы vera:* перехватывался ПЕРВЫМ (даже если общий обработчик уже обёрнут другими модулями)
        try:
            obs_cb = router.callback_query
            handlers_cb = list(getattr(obs_cb, "handlers", []))
            target_cb = None

            # Пробуем найти _h_cb/_h_cb_wrapped
            for h in handlers_cb:
                cbf = getattr(h, "callback", None)
                dname = getattr(cbf, "__name__", "")
                if "_h_cb" in dname:
                    target_cb = h
                    break

            # Если не нашли — заворачиваем самый первый обработчик как «catch-all»
            if not target_cb and handlers_cb:
                target_cb = handlers_cb[0]
                logger and logger.info("VERA: global callback wrapper fallback → wrapping first handler: %s",
                                       getattr(getattr(target_cb, 'callback', None), '__name__', 'unknown'))

            if target_cb:
                orig_h_cb = target_cb.callback

                async def _h_cb_wrapped(cb: CallbackQuery):
                    data = cb.data or ""
                    if data and data.startswith("vera:"):
                        await _cb_router(cb)
                        return
                    return await orig_h_cb(cb)

                setattr(target_cb, "callback", _h_cb_wrapped)
                logger and logger.info("VERA: global callback wrapper installed over %s",
                                       getattr(orig_h_cb, "__name__", str(orig_h_cb)))
            else:
                logger and logger.warning("VERA: no callback handler to wrap; relying on direct vera filter only.")
        except Exception as e:
            logger and logger.warning("VERA: global callback wrap error: %s", e)

        # 3) Fallback wrap — приоритет для /myprofile, /pro и кнопки «🧾 Мой статус»
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

                async def fallback_with_vera(message: Message, bot):
                    try:
                        txt = (message.text or "").strip()
                        if getattr(message.chat, "type", None) == "private":
                            low = txt.lower()
                            if low.startswith("/myprofile") or low.startswith("/pro") or txt == "🧾 Мой статус":
                                if low.startswith("/myprofile") or txt == "🧾 Мой статус":
                                    await _handle_myprofile(app, message); return
                                if low.startswith("/pro"):
                                    await _handle_pro_cmd(app, message); return
                        await orig_fallback(message, bot)
                    except Exception:
                        with contextlib.suppress(Exception):
                            await orig_fallback(message, bot)

                setattr(target_fb, "callback", fallback_with_vera)
                logger and logger.info("VERA: fallback wrapped (priority for /myprofile, /pro, '🧾 Мой статус').")
        except Exception as e:
            logger and logger.warning("VERA: fallback wrap error: %s", e)

    # Intercepts for limits
    _patch_cmd_signal_offer(app)
    _patch_neon_analysis_offer(app)
    _patch_ta_offer(app)

    logger and logger.info("VERA: patch applied — PRO subscription UI, admin confirm, status profile, limit intercepts, expiry loop.")
