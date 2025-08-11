import asyncio
import logging
import os
import re
import json
import math
import calendar
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from typing import Dict, List, Optional, Tuple, Any, Set
from concurrent.futures import ThreadPoolExecutor
from time import time
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import pytz
import requests
import feedparser
import aiosqlite

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandObject, Filter
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, Message, ReplyKeyboardMarkup
from dotenv import load_dotenv

try:
    from prometheus_client import start_http_server, Counter
    PROMETHEUS_OK = True
except Exception:
    PROMETHEUS_OK = False

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHANNEL_USERNAME = os.getenv("CHANNEL_USERNAME", "@NeonFakTrading").strip()
DB_PATH = os.getenv("DB_PATH", "neon_bot.db").strip()
LOCK_PATH = os.getenv("LOCK_PATH", "neon_bot.lock").strip()
METRICS_PORT = int(os.getenv("METRICS_PORT", "0").strip() or "0")

DEFAULT_CP_TOKENS = [
    "bdcf4a18c17ba8c14431aca8aa728b9c69c4f553",
    "4b9363a2ada7611bdb0cc31a863e7cb1807f4652",
    "96af5cfa494bba46f969ff27d303dc0de18778a7",
    "d4e36de4cb703f29896b62bb7ea343b193b494cb",
]
CRYPTOPANIC_TOKENS = [
    t.strip()
    for t in os.getenv("CRYPTOPANIC_TOKENS", ",".join(DEFAULT_CP_TOKENS)).split(",")
    if t.strip()
]

EXCHANGE_PRIORITY = [x.strip() for x in os.getenv("EXCHANGE_PRIORITY", "binance,binanceusdm,bybit,okx,mexc,bingx").split(",")]
TZ_NAME = os.getenv("TZ", "Europe/Moscow")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN в .env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("neon-bot")

SYMBOLS = [
    "BTC/USDT","ETH/USDT","SOL/USDT","TON/USDT","SUI/USDT","XRP/USDT","ENA/USDT","ADA/USDT",
    "BNB/USDT","DOGE/USDT","AVAX/USDT","LINK/USDT","DOT/USDT","NEAR/USDT","ARB/USDT","OP/USDT",
    "SEI/USDT","APT/USDT","TRX/USDT","PEPE/USDT","LTC/USDT","BCH/USDT","ETC/USDT","ATOM/USDT","INJ/USDT","TIA/USDT"
]
BASES = [s.split("/")[0] for s in SYMBOLS]
CP_CURRENCY_MAP = {s: s.split("/")[0] for s in SYMBOLS}

router = Router()
tz = pytz.timezone(TZ_NAME)

DAILY_LIMIT = 3
ADMIN_ACCESS_CODE = "2604"

NEWS_TTL = 300
news_cache: Dict[str, Tuple[float, Tuple[float, str]]] = {}
cp_token_cooldowns: Dict[str, float] = {}
cp_token_index: int = 0

EXECUTOR = ThreadPoolExecutor(max_workers=3)
APP_STARTED_AT = datetime.now(UTC)

MIN_ATR_PCT = 0.10
MAX_ATR_PCT = 1.80

VOL_THRESHOLDS_USDT = {
    "BTC": 20_000_000, "ETH": 10_000_000, "SOL": 3_000_000, "TON": 1_500_000, "SUI": 600_000, "XRP": 4_000_000, "ENA": 300_000, "ADA": 2_000_000,
    "BNB": 4_000_000, "DOGE": 3_000_000, "AVAX": 1_200_000, "LINK": 1_000_000, "DOT": 1_000_000, "NEAR": 1_000_000, "ARB": 700_000, "OP": 600_000,
    "SEI": 500_000, "APT": 800_000, "TRX": 2_000_000, "PEPE": 600_000, "LTC": 1_000_000, "BCH": 800_000, "ETC": 800_000, "ATOM": 800_000, "INJ": 600_000, "TIA": 500_000
}
MIN_RISK_PCT_PER = {
    "BTC": 0.0020, "ETH": 0.0025, "SOL": 0.0035, "TON": 0.0035, "SUI": 0.0035, "XRP": 0.0030, "ENA": 0.0035, "ADA": 0.0030,
    "BNB": 0.0028, "DOGE": 0.0040, "AVAX": 0.0035, "LINK": 0.0030, "DOT": 0.0030, "NEAR": 0.0035, "ARB": 0.0035, "OP": 0.0035,
    "SEI": 0.0040, "APT": 0.0035, "TRX": 0.0030, "PEPE": 0.0060, "LTC": 0.0030, "BCH": 0.0030, "ETC": 0.0030, "ATOM": 0.0030, "INJ": 0.0035, "TIA": 0.0040
}

RSS_SOURCES = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://bitcoinmagazine.com/.rss",
    "https://cryptoslate.com/feed/",
    "https://u.today/rss",
]

NEWS_NEG = [
    "hack","exploit","ban","lawsuit","down","drop","investigation","crash","liquidation","vulnerability","outage","halt","crackdown","sell-off","dump","bearish","fine","charged","sues","sec sues","delist","suspension","security breach","rug","rugpull","reorg","slash","sanction","exit scam","phishing"
]
NEWS_POS = [
    "partnership","funding","listing","upgrade","approve","approval","bullish","adoption","integration","launch","mainnet","testnet","etf inflow","etf","burn","buyback","staking","yield","support","roadmap","milestone","all-time high","ath"
]
NEG_STRONG = {"hack":2.0,"exploit":2.0,"sec sues":2.0,"lawsuit":1.5,"delist":1.5,"rug":2.0,"rugpull":2.0,"security breach":1.8}
POS_STRONG = {"listing":1.5,"etf":1.5,"approve":1.5,"approval":1.5,"partnership":1.2,"funding":1.2,"mainnet":1.3,"burn":1.2}

COIN_SYNONYMS: Dict[str, List[str]] = {
    "BTC": ["btc","bitcoin","биткоин","биток","$btc","btcusdt","btc/usdt","btc price"],
    "ETH": ["eth","ethereum","эфир","эфириум","$eth","ethusdt","eth/usdt","eth price"],
    "SOL": ["sol","solana","сол","солана","$sol","solusdt","sol/usdt"],
    "TON": ["ton","toncoin","тон","тонкоин","$ton","ton/usdt","tonusdt","the open network","ton foundation","ton blockchain"],
    "SUI": ["sui","sui network","$sui","sui/usdt","suiusdt","mysten"],
    "XRP": ["xrp","$xrp","ripple","рипл","xrpl","xrp/usdt","xrpusdt"],
    "ENA": ["ethena","эфена","$ena","ena/usdt","enausdt","usde"],
    "ADA": ["cardano","кардано","$ada","ada/usdt","adausdt","ada price"],
    "BNB": ["bnb","бинанс коин","бинб","$bnb","bnb/usdt","bnbusdt"],
    "DOGE": ["doge","dogecoin","доге","додж","догикоин","$doge","doge/usdt","dogeusdt"],
    "AVAX": ["avax","avalanche","аваланч","$avax","avax/usdt","avaxusdt"],
    "LINK": ["link","chainlink","чейнлинк","$link","link/usdt","linkusdt"],
    "DOT": ["dot","polkadot","полкадот","$dot","dot/usdt","dotusdt"],
    "NEAR": ["near","near protocol","$near","нир","нир протокол","near/usdt","nearusdt"],
    "ARB": ["arb","arbitrum","арб","арбитрум","$arb","arb/usdt","arbusdt"],
    "OP": ["op","optimism","оптимижм","оп","$op","op/usdt","opusdt"],
    "SEI": ["sei","сеи","$sei","sei/usdt","seiusdt"],
    "APT": ["apt","aptos","аптос","$apt","apt/usdt","aptusdt"],
    "TRX": ["trx","tron","трон","$trx","trx/usdt","trxusdt"],
    "PEPE": ["pepe","пепе","$pepe","pepe/usdt","pepeusdt"],
    "LTC": ["ltc","litecoin","лайткоин","$ltc","ltc/usdt","ltcusdt"],
    "BCH": ["bch","bitcoin cash","биткоин кэш","$bch","bch/usdt","bchusdt"],
    "ETC": ["etc","ethereum classic","эфир классик","$etc","etc/usdt","etcusdt"],
    "ATOM": ["atom","cosmos","космос","$atom","atom/usdt","atomusdt"],
    "INJ": ["inj","injective","инжектив","$inj","inj/usdt","injusdt"],
    "TIA": ["tia","celestia","целестия","$tia","tia/usdt","tiausdt"],
}

def now_msk() -> datetime:
    return datetime.now(tz)

def now_utc() -> datetime:
    return datetime.now(UTC)

def today_key() -> str:
    return now_msk().strftime("%Y-%m-%d")

def format_price(x: float) -> str:
    if x >= 1000:
        return f"{x:,.0f}".replace(",", " ")
    elif x >= 100:
        return f"{x:,.2f}"
    elif x >= 1:
        return f"{x:,.4f}"
    else:
        return f"{x:.6f}"

def hashtag_symbol(symbol: str) -> str:
    base = symbol.split("/")[0]
    return f"#{base}"

def human_td(delta: timedelta) -> str:
    s = int(max(0, delta.total_seconds()))
    h, rem = divmod(s, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}ч {m}м"

def google_news_url(query_terms: List[str]) -> str:
    q = " OR ".join(query_terms)
    return f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"

def normalize_text(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[\r\n\t]+", " ", t)
    return t

def classify_currencies(text: str) -> Set[str]:
    t = normalize_text(text)
    hits: Set[str] = set()
    for code, words in COIN_SYNONYMS.items():
        for w in words:
            wlow = w.lower()
            if len(wlow) <= 3 and wlow not in {"btc","eth","xrp","sol","ada","dot","bnb","ltc","trx"}:
                continue
            if re.search(rf"(^|[^a-z0-9а-я]){re.escape(wlow)}([^a-z0-9а-я]|$)", t):
                hits.add(code)
                break
    return hits

def entry_time_utc(entry) -> datetime:
    tm = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if tm is None:
        return now_utc()
    ts = calendar.timegm(tm)
    return datetime.fromtimestamp(ts, UTC)

class Database:
    def __init__(self, path: str):
        self.path = path
        self.conn: Optional[aiosqlite.Connection] = None

    async def init(self):
        self.conn = await aiosqlite.connect(self.path)
        self.conn.row_factory = aiosqlite.Row
        await self.conn.execute("CREATE TABLE IF NOT EXISTS users (user_id INTEGER PRIMARY KEY, date TEXT NOT NULL, count INTEGER NOT NULL DEFAULT 0, unlimited INTEGER NOT NULL DEFAULT 0)")
        await self.conn.execute("CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL, entry REAL NOT NULL, tps TEXT NOT NULL, sl REAL NOT NULL, leverage INTEGER NOT NULL, risk_level INTEGER NOT NULL, created_at TEXT NOT NULL, news_note TEXT, atr_value REAL, active INTEGER NOT NULL, tp_hit INTEGER NOT NULL, watch_until TEXT NOT NULL)")
        await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_user_active ON signals(user_id, active)")
        await self.conn.execute("CREATE TABLE IF NOT EXISTS backtest (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, pairs TEXT NOT NULL, days INTEGER NOT NULL, trades INTEGER NOT NULL, tp1 INTEGER NOT NULL, tp2 INTEGER NOT NULL, tp3 INTEGER NOT NULL, stops INTEGER NOT NULL)")
        try:
            await self.conn.execute("ALTER TABLE users ADD COLUMN support_mode INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass
        await self.conn.commit()

    async def get_user_state(self, user_id: int) -> Dict[str, Any]:
        dkey = today_key()
        cur = await self.conn.execute("SELECT user_id, date, count, unlimited, support_mode FROM users WHERE user_id=?", (user_id,))
        row = await cur.fetchone()
        if row:
            date, count, unlimited, support_mode = row["date"], row["count"], bool(row["unlimited"]), bool(row["support_mode"])
            if date != dkey:
                count = 0
                date = dkey
                await self.conn.execute("UPDATE users SET date=?, count=? WHERE user_id=?", (date, count, user_id))
                await self.conn.commit()
            return {"date": date, "count": count, "unlimited": unlimited, "support_mode": support_mode}
        else:
            await self.conn.execute("INSERT INTO users (user_id, date, count, unlimited, support_mode) VALUES (?, ?, 0, 0, 0)", (user_id, dkey))
            await self.conn.commit()
            return {"date": dkey, "count": 0, "unlimited": False, "support_mode": False}

    async def save_user_state(self, user_id: int, state: Dict[str, Any]):
        await self.conn.execute(
            "INSERT INTO users (user_id, date, count, unlimited, support_mode) VALUES (?, ?, ?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET date=excluded.date, count=excluded.count, unlimited=excluded.unlimited, support_mode=excluded.support_mode",
            (user_id, state.get("date", today_key()), int(state.get("count", 0)), 1 if state.get("unlimited", False) else 0, 1 if state.get("support_mode", False) else 0),
        )
        await self.conn.commit()

    async def set_support_mode(self, user_id: int, enabled: bool):
        await self.conn.execute("UPDATE users SET support_mode=? WHERE user_id=?", (1 if enabled else 0, user_id))
        await self.conn.commit()

    async def add_signal(self, sig: "Signal") -> int:
        tps_json = json.dumps(sig.tps)
        await self.conn.execute("INSERT INTO signals (user_id, symbol, side, entry, tps, sl, leverage, risk_level, created_at, news_note, atr_value, active, tp_hit, watch_until) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (sig.user_id, sig.symbol, sig.side, sig.entry, tps_json, sig.sl, sig.leverage, sig.risk_level, sig.created_at.isoformat(), sig.news_note, sig.atr_value, 1 if sig.active else 0, sig.tp_hit, sig.watch_until.isoformat()))
        await self.conn.commit()
        cur = await self.conn.execute("SELECT last_insert_rowid() AS id")
        row = await cur.fetchone()
        return int(row["id"])

    async def update_signal(self, sig: "Signal"):
        tps_json = json.dumps(sig.tps)
        await self.conn.execute("UPDATE signals SET symbol=?, side=?, entry=?, tps=?, sl=?, leverage=?, risk_level=?, created_at=?, news_note=?, atr_value=?, active=?, tp_hit=?, watch_until=? WHERE id=?", (sig.symbol, sig.side, sig.entry, tps_json, sig.sl, sig.leverage, sig.risk_level, sig.created_at.isoformat(), sig.news_note, sig.atr_value, 1 if sig.active else 0, sig.tp_hit, sig.watch_until.isoformat(), sig.id))
        await self.conn.commit()

    async def get_active_signals_for_user(self, user_id: int) -> List["Signal"]:
        cur = await self.conn.execute("SELECT * FROM signals WHERE user_id=? AND active=1 ORDER BY created_at DESC", (user_id,))
        rows = await cur.fetchall()
        return [Signal.from_row(row) for row in rows]

    async def get_all_active_signals(self) -> List["Signal"]:
        cur = await self.conn.execute("SELECT * FROM signals WHERE active=1 ORDER BY created_at DESC")
        rows = await cur.fetchall()
        return [Signal.from_row(row) for row in rows]

    async def save_backtest(self, pairs: List[str], days: int, trades: int, tp1: int, tp2: int, tp3: int, stops: int):
        await self.conn.execute("INSERT INTO backtest (created_at, pairs, days, trades, tp1, tp2, tp3, stops) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (now_msk().isoformat(), ",".join(pairs), days, trades, tp1, tp2, tp3, stops))
        await self.conn.commit()

    async def get_admin_user_ids(self) -> List[int]:
        cur = await self.conn.execute("SELECT user_id FROM users WHERE unlimited=1")
        rows = await cur.fetchall()
        return [int(r["user_id"]) for r in rows]

db: Optional[Database] = None

class MarketClient:
    def __init__(self, priority: List[str]):
        import ccxt
        self.ccxt = ccxt
        self.priority = priority
        self.exchanges = {}
        self.ohlcv_cache: Dict[Tuple[str, str, str], Tuple[float, pd.DataFrame]] = {}
        self.symbol_map_cache: Dict[Tuple[str, str], Optional[str]] = {}
        self.OHLCV_TTL = 60
        for name in priority:
            try:
                ex = getattr(ccxt, name)({"enableRateLimit": True, "timeout": 20000})
                ex.load_markets()
                self.exchanges[name] = ex
                logger.info("Инициализирована биржа: %s (маркетов: %d)", name, len(ex.markets))
            except Exception as e:
                logger.warning("Не удалось инициализировать биржу %s: %s", name, e)

    def _available_exchanges(self):
        for name in self.priority:
            if name in self.exchanges:
                yield name, self.exchanges[name]

    def resolve_symbol(self, ex, symbol: str) -> Optional[str]:
        key = (ex.id, symbol)
        if key in self.symbol_map_cache:
            return self.symbol_map_cache[key]
        base, quote = symbol.split("/")
        best = None
        best_score = -1
        for mkey, m in ex.markets.items():
            try:
                mb = str(m.get("base") or "").upper()
                mq = str(m.get("quote") or "").upper()
                if mb != base or mq != quote:
                    continue
                t = m.get("type") or ("swap" if m.get("swap") else "spot")
                is_linear = bool(m.get("linear"))
                is_contract = bool(m.get("contract"))
                score = 0
                if t == "swap":
                    score += 3
                if is_contract:
                    score += 2
                if is_linear:
                    score += 1
                if ":USDT" in mkey:
                    score += 1
                if t == "spot":
                    score += 1
                if score > best_score:
                    best = mkey
                    best_score = score
            except Exception:
                continue
        if not best and symbol in ex.markets:
            best = symbol
        self.symbol_map_cache[key] = best
        return best

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 250) -> Optional[pd.DataFrame]:
        ts = time()
        last_error = None
        for name, ex in self._available_exchanges():
            resolved = self.resolve_symbol(ex, symbol) or symbol
            cache_key = (name, resolved, timeframe)
            cached = self.ohlcv_cache.get(cache_key)
            if cached and ts - cached[0] < self.OHLCV_TTL:
                return cached[1].copy()
            if resolved not in ex.markets:
                continue
            try:
                data = ex.fetch_ohlcv(resolved, timeframe=timeframe, limit=limit)
                if not data:
                    continue
                df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
                df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                self.ohlcv_cache[cache_key] = (ts, df)
                return df.copy()
            except Exception as e:
                last_error = e
                logger.warning("fetch_ohlcv fail %s %s %s: %s", name, resolved, timeframe, e)
                continue
        logger.error("Нет данных OHLCV для %s (%s). Последняя ошибка: %s", symbol, timeframe, last_error)
        return None

    def fetch_mark_price(self, symbol: str) -> Optional[float]:
        last_error = None
        for name, ex in self._available_exchanges():
            resolved = self.resolve_symbol(ex, symbol) or symbol
            if resolved not in ex.markets:
                continue
            try:
                ticker = ex.fetch_ticker(resolved)
                price = None
                if isinstance(ticker, dict):
                    if "mark" in ticker and ticker["mark"]:
                        price = ticker["mark"]
                    if not price and "info" in ticker and isinstance(ticker["info"], dict):
                        for k in ("markPrice", "mark_price", "indexPrice", "index_price"):
                            if k in ticker["info"] and ticker["info"][k]:
                                price = ticker["info"][k]
                                break
                    if not price:
                        price = ticker.get("last") or ticker.get("close") or ticker.get("bid") or ticker.get("ask")
                if price is not None:
                    return float(price)
            except Exception as e:
                last_error = e
                logger.warning("fetch_mark_price fail %s %s: %s", name, resolved, e)
                continue
        logger.error("Нет mark price для %s. Последняя ошибка: %s", symbol, last_error)
        return None

    def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        last_error = None
        for name, ex in self._available_exchanges():
            resolved = self.resolve_symbol(ex, symbol) or symbol
            if resolved not in ex.markets:
                continue
            try:
                t = ex.fetch_ticker(resolved)
                if not isinstance(t, dict):
                    continue
                out = {
                    "symbol": symbol,
                    "exchange": name,
                    "last": float(t.get("last") or t.get("close") or 0.0),
                    "bid": float(t.get("bid") or 0.0),
                    "ask": float(t.get("ask") or 0.0),
                    "high": float(t.get("high") or 0.0),
                    "low": float(t.get("low") or 0.0),
                    "percentage": float(t.get("percentage") or t.get("info", {}).get("priceChangePercent") or 0.0)
                }
                return out
            except Exception as e:
                last_error = e
                logger.warning("fetch_ticker fail %s %s: %s", name, resolved, e)
                continue
        logger.error("Нет ticker для %s. Последняя ошибка: %s", symbol, last_error)
        return None

market = MarketClient(EXCHANGE_PRIORITY)

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(period).mean()
    roll_down = pd.Series(down, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def obv(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    vol = df["volume"]
    prev_close = close.shift(1)
    direction = np.where(close > prev_close, 1, np.where(close < prev_close, -1, 0))
    return (direction * vol).cumsum()

def bollinger(series: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(n).mean()
    std = series.rolling(n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    return lower, mid, upper

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    tr1 = high - low
    tr2 = (high - df["close"].shift(1)).abs()
    tr3 = (low - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_ = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).sum() / (atr_ * period + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).sum() / (atr_ * period + 1e-9)
    dx = (100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9))
    adx_ = dx.rolling(period).mean()
    return adx_

def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    hl2 = (df["high"] + df["low"]) / 2.0
    atr_ = atr(df, period)
    upperband = hl2 + multiplier * atr_
    lowerband = hl2 - multiplier * atr_
    st = pd.Series(index=df.index, dtype=float)
    dir_ = pd.Series(index=df.index, dtype=int)
    for i in range(len(df)):
        if i == 0:
            st.iloc[i] = upperband.iloc[i]
            dir_.iloc[i] = 1
            continue
        prev_st = st.iloc[i - 1]
        prev_dir = dir_.iloc[i - 1]
        if df["close"].iloc[i] > upperband.iloc[i - 1]:
            dir_.iloc[i] = 1
        elif df["close"].iloc[i] < lowerband.iloc[i - 1]:
            dir_.iloc[i] = -1
        else:
            dir_.iloc[i] = prev_dir
        if dir_.iloc[i] == 1:
            st.iloc[i] = max(lowerband.iloc[i], prev_st if prev_dir == 1 else lowerband.iloc[i])
        else:
            st.iloc[i] = min(upperband.iloc[i], prev_st if prev_dir == -1 else upperband.iloc[i])
    return st, dir_

def keltner_channels(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_mid = ema(df["close"], n)
    atr_ = atr(df, n)
    upper = ema_mid + k * atr_
    lower = ema_mid - k * atr_
    return lower, ema_mid, upper

def donchian(df: pd.DataFrame, n: int = 20) -> Tuple[pd.Series, pd.Series]:
    upper = df["high"].rolling(n).max()
    lower = df["low"].rolling(n).min()
    return lower, upper

def anchored_vwap(df: pd.DataFrame, anchor: Optional[datetime] = None) -> pd.Series:
    if anchor is None:
        last_ts = df["ts"].iloc[-1].to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
        anchor = last_ts.replace(tzinfo=UTC)
    mask = df["ts"] >= pd.Timestamp(anchor, tz=UTC)
    p = df.loc[mask, "close"]
    v = df.loc[mask, "volume"]
    pv = (p * v).cumsum()
    vv = v.cumsum() + 1e-9
    vw = pv / vv
    out = pd.Series(index=df.index, dtype=float)
    out.loc[mask] = vw
    out.ffill(inplace=True)
    return out

def week_anchor_from_df(df: pd.DataFrame) -> datetime:
    last_ts = df["ts"].iloc[-1].to_pydatetime()
    start = last_ts - timedelta(days=last_ts.weekday())
    start = start.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)
    return start

def linear_reg_r2(y: pd.Series, window: int) -> Tuple[float, float]:
    if len(y) < window + 2:
        return 0.0, 0.0
    ys = y.tail(window).values.astype(float)
    xs = np.arange(len(ys)).astype(float)
    if np.std(ys) < 1e-12:
        return 0.0, 0.0
    a, b = np.polyfit(xs, ys, 1)
    y_pred = a * xs + b
    ss_res = np.sum((ys - y_pred) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return float(max(0.0, min(1.0, r2))), float(a)

def zscore(series: pd.Series, window: int = 96) -> float:
    if len(series) < window + 2:
        return 0.0
    s = series.tail(window)
    m = float(s.mean())
    sd = float(s.std(ddof=0)) + 1e-9
    val = float(series.iloc[-1])
    return (val - m) / sd

def find_pivots(series: pd.Series, left: int = 3, right: int = 3) -> Tuple[List[int], List[int]]:
    highs, lows = [], []
    arr = series.values
    n = len(arr)
    for i in range(left, n - right):
        seg = arr[i - left:i + right + 1]
        if arr[i] == np.max(seg) and np.argmax(seg) == left:
            highs.append(i)
        if arr[i] == np.min(seg) and np.argmin(seg) == left:
            lows.append(i)
    return highs, lows

def rsi_divergence(price: pd.Series, rsi_s: pd.Series, lookback: int = 40) -> int:
    p = price.tail(lookback)
    r = rsi_s.tail(lookback)
    hp, lp = find_pivots(p, 2, 2)
    if len(hp) >= 2:
        i1, i2 = hp[-2], hp[-1]
        p1, p2 = p.iloc[i1], p.iloc[i2]
        r1, r2 = r.iloc[i1], r.iloc[i2]
        if p2 > p1 and r2 < r1:
            return -1
    if len(lp) >= 2:
        i1, i2 = lp[-2], lp[-1]
        p1, p2 = p.iloc[i1], p.iloc[i2]
        r1, r2 = r.iloc[i1], r.iloc[i2]
        if p2 < p1 and r2 > r1:
            return 1
    return 0

def bos_choc(df: pd.DataFrame, lookback: int = 50, retest_lookback: int = 20) -> Tuple[int, bool]:
    highs, lows = find_pivots(df["high"], 2, 2)[0], find_pivots(df["low"], 2, 2)[1]
    bos_dir = 0
    retest = False
    if len(highs) >= 2 and len(lows) >= 2:
        last_high = highs[-1]
        last_low = lows[-1]
        if last_high > last_low:
            level = df["high"].iloc[last_high]
            if df["close"].iloc[-1] > level:
                bos_dir = 1
                win = df.iloc[max(0, last_high):].head(retest_lookback)
                if ((win["low"] <= level * 1.002) & (win["close"] > level)).any():
                    retest = True
        level2 = df["low"].iloc[last_low]
        if df["close"].iloc[-1] < level2:
            bos_dir = -1
            win = df.iloc[max(0, last_low):].head(retest_lookback)
            if ((win["high"] >= level2 * 0.998) & (win["close"] < level2)).any():
                retest = True
    return bos_dir, retest

def ichimoku_cloud(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    high = df["high"]; low = df["low"]; close = df["close"]
    conv = (high.rolling(9).max() + low.rolling(9).min()) / 2
    base = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((conv + base) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    lagging = close.shift(-26)
    return conv, base, span_a, span_b, lagging

def resample_ohlcv(df: Optional[pd.DataFrame], freq: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    x = df.set_index("ts").sort_index()
    o = x["open"].resample(freq).first()
    h = x["high"].resample(freq).max()
    l = x["low"].resample(freq).min()
    c = x["close"].resample(freq).last()
    v = x["volume"].resample(freq).sum()
    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ["open","high","low","close","volume"]
    out = out.dropna().reset_index()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out[["ts","open","high","low","close","volume"]]

@dataclass
class Signal:
    user_id: int
    symbol: str
    side: str
    entry: float
    tps: List[float]
    sl: float
    leverage: int
    risk_level: int
    created_at: datetime
    news_note: str = ""
    atr_value: float = 0.0
    active: bool = True
    tp_hit: int = 0
    watch_until: datetime = field(default_factory=lambda: now_msk() + timedelta(hours=12))
    id: Optional[int] = None
    trailing: bool = False
    trail_mode: str = "none"
    reason: str = ""

    @staticmethod
    def from_row(row: aiosqlite.Row) -> "Signal":
        return Signal(
            id=row["id"],
            user_id=row["user_id"],
            symbol=row["symbol"],
            side=row["side"],
            entry=row["entry"],
            tps=json.loads(row["tps"]),
            sl=row["sl"],
            leverage=row["leverage"],
            risk_level=row["risk_level"],
            created_at=datetime.fromisoformat(row["created_at"]),
            news_note=row["news_note"] or "",
            atr_value=row["atr_value"] or 0.0,
            active=bool(row["active"]),
            tp_hit=row["tp_hit"],
            watch_until=datetime.fromisoformat(row["watch_until"]),
        )

def compute_term_weight(text: str) -> Tuple[float, float]:
    t = normalize_text(text)
    neg_w = 0.0
    pos_w = 0.0
    for k in NEWS_NEG:
        if k in t:
            neg_w += 1.0 * NEG_STRONG.get(k, 1.0)
    for k in NEWS_POS:
        if k in t:
            pos_w += 1.0 * POS_STRONG.get(k, 1.0)
    return neg_w, pos_w

def time_decay_weight(age_hours: float) -> float:
    return float(math.exp(-age_hours / 6.0))

def sentiment_to_boost(neg: float, pos: float) -> float:
    if neg == 0 and pos == 0:
        return 0.2
    ratio = (neg + 1.0) / (pos + 1.0)
    boost = min(2.0, max(0.0, 0.6 * ratio))
    return boost

def try_fetch_news_bulk(token: str, currencies: List[str]) -> Tuple[Dict[str, Tuple[float, str]], int]:
    try:
        cur_param = ",".join(currencies)
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={token}&currencies={cur_param}&public=true"
        logger.info("Новости: запрос к CryptoPanic для %d валют", len(currencies))
        r = requests.get(url, timeout=15)
        code = r.status_code
        if code != 200:
            return {}, code
        data = r.json()
        posts = data.get("results", [])[:200]
        neg: Dict[str, float] = {c: 0.0 for c in currencies}
        pos: Dict[str, float] = {c: 0.0 for c in currencies}
        items: Dict[str, List[Tuple[float, str, float, str]]] = {c: [] for c in currencies}
        cutoff = now_utc() - timedelta(hours=12)
        for p in posts:
            pub = p.get("published_at")
            title = (p.get("title") or "").strip()
            desc = (p.get("description") or "").strip()
            text = f"{title} {desc}"
            try:
                pub_dt = datetime.fromisoformat((pub or "").replace("Z", "+00:00"))
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=UTC)
            except Exception:
                pub_dt = now_utc()
            if pub_dt < cutoff:
                continue
            raw_currs = p.get("currencies") or []
            currs = []
            for c in raw_currs:
                if isinstance(c, dict):
                    code_c = (c.get("code") or c.get("symbol") or "").upper()
                    if code_c:
                        currs.append(code_c)
                elif isinstance(c, str):
                    currs.append(c.upper())
            if not currs:
                matched = classify_currencies(text)
                currs = [c for c in matched if c in currencies]
                if not currs:
                    continue
            n_w, p_w = compute_term_weight(text)
            age_h = max(0.0, (now_utc() - pub_dt).total_seconds() / 3600.0)
            w = time_decay_weight(age_h)
            sign = "NEG" if n_w > p_w else ("POS" if p_w > n_w else "NEU")
            sc = (n_w + p_w) * w
            for code_c in currs:
                if code_c not in currencies:
                    continue
                neg[code_c] += n_w * w
                pos[code_c] += p_w * w
                if title:
                    items[code_c].append((sc, title, age_h, sign))
        res: Dict[str, Tuple[float, str]] = {}
        for c in currencies:
            n, pz = neg.get(c, 0.0), pos.get(c, 0.0)
            boost = sentiment_to_boost(n, pz)
            top = sorted(items.get(c, []), key=lambda x: x[0], reverse=True)[:2]
            if top:
                tops = " | ".join([f"[{t[3]}] {t[1]} ({int(t[2])}ч)" for t in top])
                note = f"Новости(CP): нег={n:.1f}, поз={pz:.1f}, риск+{boost:.2f} • Топ: {tops}"
            else:
                note = f"Новости(CP): нег={n:.1f}, поз={pz:.1f}, риск+{boost:.2f}"
            res[c] = (boost, note)
        return res, 200
    except Exception:
        return {}, -1

def fetch_news_rss(currencies: List[str]) -> Dict[str, Tuple[float, str]]:
    logger.info("Новости: обновление RSS/Google News для %d валют", len(currencies))
    cutoff = now_utc() - timedelta(hours=12)
    neg: Dict[str, float] = {c: 0.0 for c in currencies}
    pos: Dict[str, float] = {c: 0.0 for c in currencies}
    items: Dict[str, List[Tuple[float, str, float, str]]] = {c: [] for c in currencies}
    sources: List[str] = RSS_SOURCES[:]
    for code in currencies:
        if code == "BTC":
            q = ["BTC","Bitcoin","Биткоин","Биток"]
        elif code == "ETH":
            q = ["ETH","Ethereum","Эфир","Эфириум"]
        elif code == "SOL":
            q = ["SOL","Solana","Солана"]
        elif code == "TON":
            q = ["TON","Toncoin","Тон","Тонкоин","\"The Open Network\""]
        elif code == "SUI":
            q = ["SUI","\"Sui Network\"","Mysten"]
        elif code == "XRP":
            q = ["XRP","Ripple","Рипл","XRPL"]
        elif code == "ENA":
            q = ["ENA","Ethena","USDe"]
        elif code == "ADA":
            q = ["ADA","Cardano","Кардано"]
        else:
            q = [code]
        sources.append(google_news_url(q))
    for url in sources:
        try:
            feed = feedparser.parse(url)
            entries = feed.entries[:80]
            for e in entries:
                dt = entry_time_utc(e)
                if dt < cutoff:
                    continue
                title = (getattr(e, "title", "") or "").strip()
                summary = (getattr(e, "summary", "") or "").strip()
                text = f"{title} {summary}".strip()
                matched = classify_currencies(text)
                if not matched:
                    continue
                n_w, p_w = compute_term_weight(text)
                age_h = max(0.0, (now_utc() - dt).total_seconds() / 3600.0)
                w = time_decay_weight(age_h)
                sign = "NEG" if n_w > p_w else ("POS" if p_w > n_w else "NEU")
                sc = (n_w + p_w) * w
                for code in matched:
                    if code not in currencies:
                        continue
                    neg[code] += n_w * w
                    pos[code] += p_w * w
                    if title:
                        items[code].append((sc, title, age_h, sign))
        except Exception as ex:
            logger.debug("RSS source fail %s: %s", url, ex)
            continue
    res: Dict[str, Tuple[float, str]] = {}
    for c in currencies:
        n, pz = neg.get(c, 0.0), pos.get(c, 0.0)
        boost = sentiment_to_boost(n, pz)
        top = sorted(items.get(c, []), key=lambda x: x[0], reverse=True)[:2]
        if top:
            tops = " | ".join([f"[{t[3]}] {t[1]} ({int(t[2])}ч)" for t in top])
            res[c] = (boost, f"Новости(RSS): нег={n:.1f}, поз={pz:.1f}, риск+{boost:.2f} • Топ: {tops}")
        else:
            res[c] = (boost, f"Новости(RSS): нег={n:.1f}, поз={pz:.1f}, риск+{boost:.2f}")
    btc_boost = res.get("BTC", (0.2, ""))[0]
    eth_boost = res.get("ETH", (0.2, ""))[0]
    cross = (btc_boost + eth_boost) / 2.0
    for c in currencies:
        if c in ("BTC","ETH"):
            continue
        add = 0.0
        if cross > 1.0:
            add = 0.4
        elif cross > 0.6:
            add = 0.2
        if add > 0:
            b0, note = res[c]
            b = min(2.0, b0 + add)
            res[c] = (b, note + f" • BTC/ETH фон +{add:.2f}")
    logger.info("Новости: обновлены (%d валют)", len(res))
    return res

def fetch_news_with_fallback(currencies: List[str]) -> Dict[str, Tuple[float, str]]:
    global cp_token_index
    now_ts = time()
    for _ in range(len(CRYPTOPANIC_TOKENS) or 1):
        if not CRYPTOPANIC_TOKENS:
            break
        token = CRYPTOPANIC_TOKENS[cp_token_index % len(CRYPTOPANIC_TOKENS)]
        cp_token_index += 1
        last = cp_token_cooldowns.get(token, 0.0)
        if now_ts - last < 60:
            continue
        cp_token_cooldowns[token] = now_ts
        res, code = try_fetch_news_bulk(token, currencies)
        if code == 200 and res:
            return res
    return {}

async def news_updater():
    while True:
        try:
            logger.info("Новости: старт обновления...")
            loop = asyncio.get_running_loop()
            mapping = await loop.run_in_executor(EXECUTOR, fetch_news_rss, BASES)
            if not mapping:
                logger.warning("Новости: RSS пуст, пробую резерв (CryptoPanic)")
                fallback = await loop.run_in_executor(EXECUTOR, fetch_news_with_fallback, BASES)
                mapping = fallback
                if fallback:
                    logger.info("Новости: обновлены через резерв (CryptoPanic).")
            if mapping:
                ts = time()
                for cur, pair in mapping.items():
                    news_cache[cur] = (ts, pair)
                if PROMETHEUS_OK:
                    MET_NEWS_UPDATES.inc()
                logger.info("Новости: кэш обновлён.")
            else:
                logger.warning("Новости: не удалось обновить (использую кэш).")
        except Exception as e:
            logger.exception("news_updater error: %s", e)
        await asyncio.sleep(NEWS_TTL)

def fetch_news_sentiment_cached(currency_code: str) -> Tuple[float, str]:
    ts = time()
    cached = news_cache.get(currency_code)
    if cached and ts - cached[0] < 3600:
        return cached[1]
    return 0.2, "Новости: кэш не прогрет (базовый риск)"

def check_news_health() -> bool:
    try:
        mapping = fetch_news_rss(["BTC", "ETH"])
        return bool(mapping)
    except Exception:
        return False

def pearson_corr(a: pd.Series, b: pd.Series, window: int = 180) -> float:
    a = a.pct_change().dropna().tail(window)
    b = b.pct_change().dropna().tail(window)
    n = min(len(a), len(b))
    if n < 30:
        return 0.0
    return float(np.corrcoef(a.tail(n), b.tail(n))[0, 1])

def ichimoku_cloud_df(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    return ichimoku_cloud(df)

def _choose_watch_seconds(atr_pct: float, regime_score: float) -> int:
    if atr_pct >= 1.2:
        return 60 * 60
    elif atr_pct >= 0.8:
        return 90 * 60
    elif atr_pct >= 0.4:
        return 3 * 3600
    else:
        return 5 * 3600

def _compute_tps_by_atr(entry: float, side: str, atr15: float, regime_score: float) -> List[float]:
    base = [0.8, 1.5, 2.4]
    scale = 0.9 + 0.4 * max(0.0, min(1.0, regime_score))
    muls = [m * scale for m in base]
    muls = [min(m, 3.0) for m in muls]
    if side == "LONG":
        tps = [entry + m * atr15 for m in muls]
    else:
        tps = [entry - m * atr15 for m in muls]
    return [float(x) for x in tps]

def score_symbol_core(symbol: str, relax: bool = False) -> Optional[Tuple[float, str, Dict]]:
    logger.info("Анализ пары: %s (relax=%s)", symbol, relax)
    base = symbol.split("/")[0]
    df4h = market.fetch_ohlcv(symbol, "4h", 400)
    df1h = market.fetch_ohlcv(symbol, "1h", 400)
    df15 = market.fetch_ohlcv(symbol, "15m", 800)
    df5 = market.fetch_ohlcv(symbol, "5m", 800)
    if any(x is None for x in (df4h, df1h, df15, df5)):
        logger.warning("Нет данных для %s, пропуск", symbol)
        return None
    df1d = market.fetch_ohlcv(symbol, "1d", 400)
    if df1d is None:
        base4h = market.fetch_ohlcv(symbol, "4h", 900)
        df1d = resample_ohlcv(base4h, "1D") if base4h is not None else None
    df1w = market.fetch_ohlcv(symbol, "1w", 260)
    if df1w is None:
        df1w = resample_ohlcv(df1d, "1W") if df1d is not None else None
    for df in (df4h, df1h, df15, df5):
        df["ema50"] = ema(df["close"], 50)
        df["ema200"] = ema(df["close"], 200)
    if df1d is not None:
        df1d["ema50"] = ema(df1d["close"], 50)
        df1d["ema200"] = ema(df1d["close"], 200)
    if df1w is not None:
        df1w["ema50"] = ema(df1w["close"], 50)
        df1w["ema200"] = ema(df1w["close"], 200)
    df5["rsi"] = rsi(df5["close"], 14)
    _, _, df5["macdh"] = macd(df5["close"], 12, 26, 9)
    df15["atr"] = atr(df15, 14)
    df5["obv"] = obv(df5)
    df15["bb_low"], df15["bb_mid"], df15["bb_up"] = bollinger(df15["close"], 20, 2.0)
    df15["adx"] = adx(df15, 14)
    st_line, st_dir = supertrend(df15, 10, 3.0)
    df15["st"] = st_line
    df15["st_dir"] = st_dir
    d_lower, d_upper = donchian(df15, 20)
    df15["don_low"] = d_lower
    df15["don_up"] = d_upper
    kl, kmid, ku = keltner_channels(df15, 20, 1.5)
    df15["kc_low"] = kl; df15["kc_mid"] = kmid; df15["kc_up"] = ku
    day_vwap = anchored_vwap(df15)
    week_vwap = anchored_vwap(df15, week_anchor_from_df(df15))
    df15["vwap_day"] = day_vwap
    df15["vwap_week"] = week_vwap
    conv4, base4, span_a4, span_b4, _ = ichimoku_cloud_df(df4h)
    c4h = float(df4h["close"].iloc[-1])
    c1h = float(df1h["close"].iloc[-1])
    c15 = float(df15["close"].iloc[-1])
    c5 = float(df5["close"].iloc[-1])
    c1d = float(df1d["close"].iloc[-1]) if df1d is not None and len(df1d) else None
    c1w = float(df1w["close"].iloc[-1]) if df1w is not None and len(df1w) else None
    atr15 = float(df15["atr"].iloc[-1])
    rsi5 = float(df5["rsi"].iloc[-1])
    macdh5 = float(df5["macdh"].iloc[-1])
    adx15 = float(df15["adx"].iloc[-1])
    vwap15 = float(df15["vwap_day"].iloc[-1])
    vwapw = float(df15["vwap_week"].iloc[-1])
    atr_pct = float(atr15 / max(1e-9, c5)) * 100.0
    atr_min = 0.03 if relax else MIN_ATR_PCT
    atr_max = 3.00 if relax else MAX_ATR_PCT
    if not (atr_min <= atr_pct <= atr_max):
        logger.info("Фильтр ATR: %.2f%% вне [%s..%s] для %s", atr_pct, atr_min, atr_max, symbol)
        return None
    qv = (df15["close"] * df15["volume"]).iloc[-96:]
    med_qv = float(qv.median())
    min_qv = VOL_THRESHOLDS_USDT.get(base, 300_000)
    if med_qv < min_qv and not relax:
        logger.info("Фильтр объёма: мед. квотация %s < %s для %s", f"{int(med_qv):,}".replace(",", " "), f"{int(min_qv):,}".replace(",", " "), symbol)
        return None
    vol_z = zscore(df15["close"] * df15["volume"], 96)
    currency_code = CP_CURRENCY_MAP.get(symbol, "BTC")
    news_boost, news_note = fetch_news_sentiment_cached(currency_code)
    in_play = (vol_z > 0.6) or (news_boost > 0.6) or relax
    if not in_play and not relax:
        logger.info("Фильтр in-play: vol_z=%.2f, news=%.2f для %s", vol_z, news_boost, symbol)
        return None
    ema50_slope_5m = float(df5["ema50"].iloc[-1] - df5["ema50"].iloc[-6])
    ema50_slope_15m = float(df15["ema50"].iloc[-1] - df15["ema50"].iloc[-6])
    cond4h_up = c4h > float(df4h["ema200"].iloc[-1])
    cond1h_up = c1h > float(df1h["ema200"].iloc[-1])
    cond15_up = float(df15["ema50"].iloc[-1]) > float(df15["ema200"].iloc[-1])
    bias_long = 0
    bias_short = 0
    if c1w is not None and "ema200" in df1w:
        bias_long += int(c1w > float(df1w["ema200"].iloc[-1]))
        bias_short += int(c1w < float(df1w["ema200"].iloc[-1]))
    if c1d is not None and "ema200" in df1d:
        bias_long += int(c1d > float(df1d["ema200"].iloc[-1]))
        bias_short += int(c1d < float(df1d["ema200"].iloc[-1]))
    bias_long += cond4h_up + cond1h_up + cond15_up
    bias_short += (not cond4h_up) + (not cond1h_up) + (not cond15_up)
    mom_long = (macdh5 > 0) + (rsi5 > 50) + (ema50_slope_5m > 0) + (ema50_slope_15m > 0)
    mom_short = (macdh5 < 0) + (rsi5 < 50) + (ema50_slope_5m < 0) + (ema50_slope_15m < 0)
    r2_1h, slope_1h = linear_reg_r2(df1h["close"], 120)
    regime_score = 0.5 * min(1.0, adx15 / 25.0) + 0.5 * r2_1h
    st_dir_last = int(df15["st_dir"].iloc[-1])
    st_bias_long = 1 if st_dir_last > 0 else 0
    st_bias_short = 1 if st_dir_last < 0 else 0
    vwap_dist = abs(c15 - vwap15) / (atr15 + 1e-9)
    vwap_confluence = 1 if vwap_dist <= 0.6 else 0
    kc_pos_long = 1 if c15 <= float(df15["kc_mid"].iloc[-1]) and c15 >= float(df15["kc_low"].iloc[-1]) else 0
    kc_pos_short = 1 if c15 >= float(df15["kc_mid"].iloc[-1]) and c15 <= float(df15["kc_up"].iloc[-1]) else 0
    lower, mid, upper = df15["bb_low"].iloc[-1], df15["bb_mid"].iloc[-1], df15["bb_up"].iloc[-1]
    bb_penalty_long = 1 if c15 > mid and (upper - c15) < 0.25 * (upper - mid) else 0
    bb_penalty_short = 1 if c15 < mid and (c15 - lower) < 0.25 * (mid - lower) else 0
    recent_high1 = float(df5["high"].iloc[-10:].max())
    recent_high2 = float(df5["high"].iloc[-20:-10].max())
    recent_low1 = float(df5["low"].iloc[-10:].min())
    recent_low2 = float(df5["low"].iloc[-20:-10].min())
    structure_long = 1 if (recent_high1 > recent_high2 and recent_low1 > recent_low2) else 0
    structure_short = 1 if (recent_high1 < recent_high2 and recent_low1 < recent_low2) else 0
    div_score = rsi_divergence(df5["close"], df5["rsi"], 40)
    bos_dir, bos_retest = bos_choc(df15, 60, 20)
    don_break_long = 1 if c15 > float(df15["don_up"].iloc[-2]) else 0
    don_break_short = 1 if c15 < float(df15["don_low"].iloc[-2]) else 0
    ich_a = float(span_a4.iloc[-1]) if not math.isnan(span_a4.iloc[-1]) else c4h
    ich_b = float(span_b4.iloc[-1]) if not math.isnan(span_b4.iloc[-1]) else c4h
    cloud_top = max(ich_a, ich_b)
    cloud_bot = min(ich_a, ich_b)
    ich_in_cloud = cloud_bot <= c4h <= cloud_top
    long_exhaust = 1 if abs(c1h - vwapw) > 2.0 * atr15 else 0
    short_exhaust = long_exhaust
    long_score = 0.0
    long_score += 1.2 * bias_long + 0.9 * mom_long + 0.9 * regime_score + 0.7 * vwap_confluence + 0.6 * structure_long + 0.6 * st_bias_long + 0.5 * kc_pos_long
    if bos_dir == 1:
        long_score += 0.6
        if bos_retest:
            long_score += 0.4
    if don_break_long:
        long_score += 0.4
    if div_score == -1:
        long_score -= 0.5
    long_score -= 0.7 * bb_penalty_long
    long_score -= 0.6 * long_exhaust
    if ich_in_cloud:
        long_score -= 0.4
    short_score = 0.0
    short_score += 1.2 * bias_short + 0.9 * mom_short + 0.9 * regime_score + 0.7 * vwap_confluence + 0.6 * structure_short + 0.6 * st_bias_short + 0.5 * kc_pos_short
    if bos_dir == -1:
        short_score += 0.6
        if bos_retest:
            short_score += 0.4
    if don_break_short:
        short_score += 0.4
    if div_score == 1:
        short_score -= 0.5
    short_score -= 0.7 * bb_penalty_short
    short_score -= 0.6 * short_exhaust
    if ich_in_cloud:
        short_score -= 0.4
    side = "LONG" if long_score >= short_score else "SHORT"
    side_score = float(max(long_score, short_score))
    window15 = 20
    swing_low_15 = float(df15["low"].iloc[-window15:].min())
    swing_high_15 = float(df15["high"].iloc[-window15:].max())
    atr_mult = 2.0
    if side == "LONG":
        sl_atr = c5 - atr_mult * atr15
        sl = min(sl_atr, swing_low_15)
    else:
        sl_atr = c5 + atr_mult * atr15
        sl = max(sl_atr, swing_high_15)
    min_risk_pct = MIN_RISK_PCT_PER.get(base, 0.003)
    min_risk_abs = min_risk_pct * c5
    risk = abs(c5 - sl)
    if risk < min_risk_abs:
        if side == "LONG":
            sl = c5 - min_risk_abs
        else:
            sl = c5 + min_risk_abs
        risk = min_risk_abs
    tps = _compute_tps_by_atr(c5, side, atr15, regime_score)
    btc1h = market.fetch_ohlcv("BTC/USDT", "1h", 200)
    btc15 = market.fetch_ohlcv("BTC/USDT", "15m", 300)
    btc5 = market.fetch_ohlcv("BTC/USDT", "5m", 400)
    eth1h = market.fetch_ohlcv("ETH/USDT", "1h", 200)
    eth15 = market.fetch_ohlcv("ETH/USDT", "15m", 300)
    eth5 = market.fetch_ohlcv("ETH/USDT", "5m", 400)
    align_score = 0.0
    if btc1h is not None and btc15 is not None and btc5 is not None:
        btc1h["ema200"] = ema(btc1h["close"], 200)
        btc15["ema50"] = ema(btc15["close"], 50)
        btc15["ema200"] = ema(btc15["close"], 200)
        btc_trend = (btc1h["close"].iloc[-1] > btc1h["ema200"].iloc[-1]) + (btc15["ema50"].iloc[-1] > btc15["ema200"].iloc[-1])
        corr_btc = pearson_corr(df5["close"], btc5["close"], 180)
        if corr_btc > 0.4:
            align_score += 1.0 if ((btc_trend > 0 and side == "LONG") or (btc_trend < 0 and side == "SHORT")) else -1.0
    if eth1h is not None and eth15 is not None and eth5 is not None:
        eth1h["ema200"] = ema(eth1h["close"], 200)
        eth15["ema50"] = ema(eth15["close"], 50)
        eth15["ema200"] = ema(eth15["close"], 200)
        eth_trend = (eth1h["close"].iloc[-1] > eth1h["ema200"].iloc[-1]) + (eth15["ema50"].iloc[-1] > eth15["ema200"].iloc[-1])
        corr_eth = pearson_corr(df5["close"], eth5["close"], 180)
        if corr_eth > 0.4:
            align_score += 1.0 if ((eth_trend > 0 and side == "LONG") or (eth_trend < 0 and side == "SHORT")) else -1.0
    obv_slope = float(df5["obv"].iloc[-1] - df5["obv"].iloc[-30])
    if side == "LONG" and obv_slope > 0:
        align_score += 0.4
    elif side == "SHORT" and obv_slope < 0:
        align_score += 0.4
    else:
        align_score -= 0.2
    if news_boost >= 1.2:
        side_score -= 0.4
    elif news_boost <= 0.4:
        side_score += 0.2
    side_score += align_score

    def atr_to_risk(a_pct: float) -> int:
        if a_pct < 0.10:
            return 2
        elif a_pct < 0.25:
            return 3
        elif a_pct < 0.50:
            return 5
        elif a_pct < 1.00:
            return 7
        elif a_pct < 1.50:
            return 8
        else:
            return 9
    base_risk = atr_to_risk(atr_pct)
    risk_level = int(max(1, min(10, round(base_risk + news_boost))))
    leverage = int(round(50 - (risk_level - 1) * (45 / 9)))
    leverage = max(3, min(50, leverage))
    if news_boost > 1.0:
        leverage = min(leverage, 10)
    watch_seconds = _choose_watch_seconds(atr_pct, regime_score)
    details = {
        "side": side,
        "score": float(side_score),
        "c5": float(c5),
        "sl": float(sl),
        "tps": tps,
        "atr": float(atr15),
        "atr_pct": float(atr_pct),
        "risk_level": risk_level,
        "leverage": leverage,
        "news_note": news_note,
        "watch_seconds": int(watch_seconds),
        "vol_z": float(vol_z),
        "regime": float(regime_score),
        "vwap_dist": float(vwap_dist),
        "cond4h_up": bool(cond4h_up),
        "cond1h_up": bool(cond1h_up),
        "cond15_up": bool(cond15_up),
        "rsi5": float(rsi5),
        "macdh5": float(macdh5),
        "adx15": float(adx15),
        "r2_1h": float(r2_1h),
        "st_dir": int(st_dir_last),
        "don_break_long": bool(don_break_long),
        "don_break_short": bool(don_break_short),
        "bos_dir": int(bos_dir),
        "bos_retest": bool(bos_retest),
        "vwap_conf": bool(vwap_confluence),
        "news_boost": float(news_boost),
    }
    logger.info("Скоринг %s: %s score=%.2f ATR%%=%.2f vol_z=%.2f ADX=%.1f", symbol, side, side_score, atr_pct, vol_z, adx15)
    return side_score, side, details

def score_symbol_quick(symbol: str) -> Optional[Tuple[str, Dict]]:
    try:
        logger.info("Quick-скрининг: %s", symbol)
        df15 = market.fetch_ohlcv(symbol, "15m", 400)
        df5 = market.fetch_ohlcv(symbol, "5m", 600)
        if df15 is None or df5 is None or len(df5) < 250 or len(df15) < 100:
            return None
        df5["ema200"] = ema(df5["close"], 200)
        df5["rsi"] = rsi(df5["close"], 14)
        df15["atr"] = atr(df15, 14)
        c5 = float(df5["close"].iloc[-1])
        ema200 = float(df5["ema200"].iloc[-1])
        rsi5 = float(df5["rsi"].iloc[-1])
        atr15 = float(df15["atr"].iloc[-1])
        if atr15 <= 0 or not np.isfinite(atr15):
            return None
        side = "LONG" if (c5 > ema200 and rsi5 >= 48) else "SHORT"
        base = symbol.split("/")[0]
        min_risk_pct = MIN_RISK_PCT_PER.get(base, 0.003)
        min_risk_abs = min_risk_pct * c5
        sl = (c5 - 2.0 * atr15) if side == "LONG" else (c5 + 2.0 * atr15)
        risk = abs(c5 - sl)
        if risk < min_risk_abs:
            sl = c5 - min_risk_abs if side == "LONG" else c5 + min_risk_abs
        regime_score = 0.5
        tps = _compute_tps_by_atr(c5, side, atr15, regime_score)
        atr_pct = float(atr15 / max(1e-9, c5)) * 100.0

        def atr_to_risk(a_pct: float) -> int:
            if a_pct < 0.10:
                return 2
            elif a_pct < 0.25:
                return 3
            elif a_pct < 0.50:
                return 5
            elif a_pct < 1.00:
                return 7
            elif a_pct < 1.50:
                return 8
            else:
                return 9
        base_risk = atr_to_risk(atr_pct)
        news_boost, news_note = fetch_news_sentiment_cached(base)
        risk_level = int(max(1, min(10, round(base_risk + news_boost))))
        leverage = int(round(50 - (risk_level - 1) * (45 / 9)))
        leverage = max(3, min(30, leverage))
        if news_boost > 1.0:
            leverage = min(leverage, 8)
        watch_seconds = _choose_watch_seconds(atr_pct, regime_score)
        details = {
            "side": side,
            "score": 0.9,
            "c5": float(c5),
            "sl": float(sl),
            "tps": [float(x) for x in tps],
            "atr": float(atr15),
            "atr_pct": float(atr_pct),
            "risk_level": risk_level,
            "leverage": leverage,
            "news_note": news_note,
            "watch_seconds": int(watch_seconds),
            "quick": True,
            "cond5_above_ema200": bool(c5 > ema200),
            "rsi5": float(rsi5),
            "news_boost": float(news_boost),
        }
        logger.info("Quick %s: %s ATR%%=%.2f RSI=%.1f", symbol, side, atr_pct, rsi5)
        return side, details
    except Exception as e:
        logger.warning("quick score failed for %s: %s", symbol, e)
        return None

async def score_symbol_async(symbol: str, relax: bool, sem: asyncio.Semaphore):
    loop = asyncio.get_running_loop()
    async with sem:
        return await loop.run_in_executor(EXECUTOR, score_symbol_core, symbol, relax)

async def rank_symbols_async(symbols: List[str]) -> List[Tuple[str, Dict]]:
    logger.info("Ранжирование пар: старт (%d)", len(symbols))
    sem = asyncio.Semaphore(2)
    tasks = [score_symbol_async(s, False, sem) for s in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    candidates: List[Tuple[float, Dict]] = []
    for s, res in zip(symbols, results):
        if isinstance(res, Exception) or res is None:
            continue
        side_score, side, details = res
        details["symbol"] = s
        candidates.append((float(side_score), details))
    if not candidates:
        logger.info("Ранжирование: строгий режим пуст, пробую relax")
        tasks = [score_symbol_async(s, True, sem) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for s, res in zip(symbols, results):
            if isinstance(res, Exception) or res is None:
                continue
            side_score, side, details = res
            details["symbol"] = s
            candidates.append((float(side_score), details))
    if not candidates:
        logger.info("Ранжирование: переходим к quick-фоллбэку")
        fallback_pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        quick_cands: List[Tuple[float, Dict]] = []
        for sym in fallback_pairs:
            quick = score_symbol_quick(sym)
            if quick:
                side, det = quick
                det["symbol"] = sym
                quick_cands.append((float(det.get("score", 0.9)), det))
        if quick_cands:
            quick_cands.sort(key=lambda x: x[0], reverse=True)
            logger.info("Ранжирование: quick-кандидатов %d", len(quick_cands))
            return [(det["symbol"], det) for _, det in quick_cands]
    if not candidates:
        logger.warning("Ранжирование: кандидатов нет")
        return []
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_score = candidates[0][0]
    near_tops = [c for c in candidates if c[0] >= top_score - 0.6]
    ordered = []
    picked_bases: Set[str] = set()
    for sc, det in near_tops:
        base = det["symbol"].split("/")[0]
        if base in picked_bases:
            continue
        ordered.append((det["symbol"], det))
        picked_bases.add(base)
    for sc, det in candidates:
        base = det["symbol"].split("/")[0]
        pair = (det["symbol"], det)
        if base in picked_bases:
            continue
        if pair not in ordered:
            ordered.append(pair)
            picked_bases.add(base)
    logger.info("Ранжирование: итоговых кандидатов %d", len(ordered))
    return ordered

def extract_news_short(note: str) -> str:
    try:
        if "риск+" in note:
            val = note.split("риск+")[-1].strip().split()[0]
            return f"Новости +{val}"
    except Exception:
        pass
    return "Новости n/a"

def _parse_news_note(note: str) -> Tuple[float, float, List[str]]:
    try:
        neg = float(re.search(r"нег=([0-9]+(?:\.[0-9]+)?)", note).group(1))
    except Exception:
        neg = 0.0
    try:
        pos = float(re.search(r"поз=([0-9]+(?:\.[0-9]+)?)", note).group(1))
    except Exception:
        pos = 0.0
    tops = []
    if "Топ:" in note:
        try:
            part = note.split("Топ:", 1)[1].strip()
            tops = [t.strip() for t in re.split(r"\s*\|\s*", part) if t.strip()]
        except Exception:
            pass
    return neg, pos, tops

def build_reason(details: Dict) -> str:
    side = details.get("side", "")
    atr = details.get("atr", 0.0)
    atr_pct = details.get("atr_pct", 0.0)
    entry = details.get("c5", 0.0)
    sl = details.get("sl", 0.0)
    tps = details.get("tps", [])
    if details.get("quick"):
        parts = []
        parts.append("Формация: быстрый скрин (5m EMA200 + RSI)")
        parts.append(f"Сторона: {side}")
        parts.append(f"ATR(15m)≈{atr:.4f} ({atr_pct:.2f}%)")
        parts.append(f"Управление риском: SL {format_price(sl)} ({((entry-sl)/entry*100 if side=='LONG' else (sl-entry)/entry*100):.2f}% от входа)")
        if tps:
            d1 = (tps[0]-entry)/entry*100 if side=="LONG" else (entry-tps[0])/entry*100
            d2 = (tps[1]-entry)/entry*100 if side=="LONG" else (entry-tps[1])/entry*100
            d3 = (tps[2]-entry)/entry*100 if side=="LONG" else (entry-tps[2])/entry*100
            parts.append(f"Таргеты: TP1 {d1:.2f}%, TP2 {d2:.2f}%, TP3 {d3:.2f}% (ближе к ТВХ, на базе ATR)")
        nb = details.get("news_boost", 0.0)
        parts.append(f"Окно сделки: 40м–6ч • Новостной фон: +{nb:.2f}")
        return " • ".join(parts)
    cond4h = details.get("cond4h_up")
    cond1h = details.get("cond1h_up")
    cond15 = details.get("cond15_up")
    rsi5 = details.get("rsi5")
    macdh5 = details.get("macdh5")
    adx15 = details.get("adx15")
    r2 = details.get("r2_1h")
    st_dir = details.get("st_dir")
    bos_dir = details.get("bos_dir")
    bos_retest = details.get("bos_retest")
    don_l = details.get("don_break_long")
    don_s = details.get("don_break_short")
    vwap_conf = details.get("vwap_conf")
    vol_z = details.get("vol_z")
    nb = details.get("news_boost")
    trend = []
    if cond4h is not None:
        trend.append("4h>EMA200" if cond4h else "4h<EMA200")
    if cond1h is not None:
        trend.append("1h>EMA200" if cond1h else "1h<EMA200")
    if cond15 is not None:
        trend.append("15m EMA50>EMA200" if cond15 else "15m EMA50<EMA200")
    mom = []
    if rsi5 is not None:
        mom.append(f"RSI5m {rsi5:.0f}")
    if macdh5 is not None:
        mom.append("MACD+" if macdh5 > 0 else "MACD-")
    regime = []
    if adx15 is not None:
        regime.append(f"ADX {adx15:.0f}")
    if r2 is not None:
        regime.append(f"R2 {r2:.2f}")
    setup = []
    if st_dir is not None:
        setup.append("ST up" if st_dir > 0 else "ST down")
    if bos_dir == 1:
        setup.append("BOS↑" + ("+retest" if bos_retest else ""))
    elif bos_dir == -1:
        setup.append("BOS↓" + ("+retest" if bos_retest else ""))
    if don_l and side == "LONG":
        setup.append("Donchian↑")
    if don_s and side == "SHORT":
        setup.append("Donchian↓")
    if vwap_conf:
        setup.append("у VWAP")
    risk_mgmt = []
    risk_mgmt.append(f"ATR(15m)≈{atr:.4f} ({atr_pct:.2f}%)")
    risk_mgmt.append(f"SL {format_price(sl)} ({((entry-sl)/entry*100 if side=='LONG' else (sl-entry)/entry*100):.2f}% от входа)")
    if tps:
        d1 = (tps[0]-entry)/entry*100 if side=="LONG" else (entry-tps[0])/entry*100
        d2 = (tps[1]-entry)/entry*100 if side=="LONG" else (entry-tps[1])/entry*100
        d3 = (tps[2]-entry)/entry*100 if side=="LONG" else (entry-tps[2])/entry*100
        risk_mgmt.append(f"TP1 {d1:.2f}%, TP2 {d2:.2f}%, TP3 {d3:.2f}% (по ATR, ближние)")
    meta = []
    if vol_z is not None:
        meta.append(f"vol_z {vol_z:.2f}")
    if nb is not None:
        meta.append(f"новости +{nb:.2f}")
    meta.append("окно 40м–6ч")
    pieces = []
    if trend: pieces.append("Тренд: " + ", ".join(trend))
    if mom: pieces.append("Моментум: " + ", ".join(mom))
    if regime: pieces.append("Режим: " + ", ".join(regime))
    if setup: pieces.append("Сетап: " + ", ".join(setup))
    if risk_mgmt: pieces.append("Риск: " + ", ".join(risk_mgmt))
    if meta: pieces.append("Фон: " + ", ".join(meta))
    txt = " • ".join(pieces)
    return txt[:900]

def format_signal_message(sig: Signal) -> str:
    base = sig.symbol.split("/")[0]
    ts_str = sig.created_at.strftime("%Y-%m-%d %H:%M") + " MSK"
    line1 = f"🤝 <b>{base} {sig.side}</b> • {ts_str} {hashtag_symbol(sig.symbol)} #{sig.side}"
    line2 = f"📌 TBX {format_price(sig.entry)}"
    tps_fmt = " / ".join(format_price(x) for x in sig.tps)
    line3 = f"🎯 Тейки: {tps_fmt} (после TP1 стоп в БУ)"
    line4 = f"🛑 Стоп {format_price(sig.sl)}"
    line5 = f"💪 Плечо {sig.leverage} • Риск-L {sig.risk_level}/10 • {extract_news_short(sig.news_note)}"
    line6 = f"🧠 Причина: {sig.reason}" if sig.reason else ""
    neg, pos, tops = _parse_news_note(sig.news_note)
    line7 = ""
    if tops:
        top_txt = " | ".join(tops[:2])
        line7 = f"🗞 Новости: {top_txt}"
    disclaimer = "\n\n⚠️ <i>Бот может работать нестабильно. Не является финансовым советом. Ответственность за сделки несут пользователи.</i>"
    parts = [line1, line2, line3, line4, line5]
    if line6:
        parts.append(line6)
    if line7:
        parts.append(line7)
    return "\n".join(parts) + disclaimer

async def is_user_subscribed(bot: Bot, user_id: int) -> bool:
    try:
        member = await bot.get_chat_member(chat_id=CHANNEL_USERNAME, user_id=user_id)
        status = getattr(member, "status", None)
        return status in ("member", "administrator", "creator")
    except Exception as e:
        logger.warning("Не удалось проверить подписку: %s", e)
        return False

def start_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📣 Подписаться на канал", url=f"https://t.me/{CHANNEL_USERNAME.lstrip('@')}")],
        [InlineKeyboardButton(text="✅ Продолжить", callback_data="continue")],
    ])

def main_menu_kb(is_admin: bool = False) -> ReplyKeyboardMarkup:
    if is_admin:
        kb = [
            [KeyboardButton(text="📈 Получить сигнал")],
            [KeyboardButton(text="ℹ️ Помощь")],
        ]
    else:
        kb = [
            [KeyboardButton(text="📈 Получить сигнал")],
            [KeyboardButton(text="ℹ️ Помощь")],
            [KeyboardButton(text="Поддержка")],
        ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

def support_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text="Назад")]], resize_keyboard=True)

def admin_answer_kb(user_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="Ответить", callback_data=f"answer_user:{user_id}")]])

def user_reply_inline_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Назад", callback_data="support_user_back"), InlineKeyboardButton(text="Ответить", callback_data="support_user_reply")]
    ])

active_watch_tasks: Dict[int, List[asyncio.Task]] = {}
risk_alert_state: Dict[int, Dict[str, float]] = {}
support_reply_targets: Dict[int, int] = {}

if PROMETHEUS_OK:
    MET_SIGNALS_GEN = Counter("signals_generated", "signals_generated")
    MET_TP1 = Counter("signals_tp1", "signals_tp1")
    MET_TP2 = Counter("signals_tp2", "signals_tp2")
    MET_TP3 = Counter("signals_tp3", "signals_tp3")
    MET_STOP = Counter("signals_stop", "signals_stop")
    MET_BE = Counter("signals_be", "signals_be")
    MET_WATCH_ERR = Counter("signals_watch_errors", "signals_watch_errors")
    MET_NEWS_UPDATES = Counter("news_updates", "news_updates")
    MET_SIGNAL_ERR = Counter("signal_errors", "signal_errors")

async def update_trailing(sig: Signal):
    try:
        df15 = market.fetch_ohlcv(sig.symbol, "15m", 200)
        if df15 is None:
            return
        st_line, st_dir = supertrend(df15, 10, 3.0)
        st_val = float(st_line.iloc[-1])
        if sig.side == "LONG":
            new_sl = max(sig.sl, st_val)
            if new_sl < sig.tps[0]:
                sig.sl = new_sl
        else:
            new_sl = min(sig.sl, st_val)
            if new_sl > sig.tps[0]:
                sig.sl = new_sl
    except Exception:
        pass

def _should_alert(sig_id: int, code: str, cooldown_sec: int = 1800) -> bool:
    now_ts = time()
    states = risk_alert_state.setdefault(sig_id, {})
    last = states.get(code, 0.0)
    if now_ts - last >= cooldown_sec:
        states[code] = now_ts
        return True
    return False

def _news_risk_trigger(side: str, news_note: str) -> Optional[str]:
    neg, pos, tops = _parse_news_note(news_note)
    if side == "LONG":
        if neg >= 2.5 and (neg / (pos + 1.0)) > 1.7:
            return f"Негативный новостной фон (нег={neg:.1f} > поз={pos:.1f}). {' | '.join(tops[:1]) if tops else ''}"
    else:
        if pos >= 2.5 and (pos / (neg + 1.0)) > 1.7:
            return f"Позитивный новостной фон против шорта (поз={pos:.1f} > нег={neg:.1f}). {' | '.join(tops[:1]) if tops else ''}"
    if tops:
        joined = " ".join(tops).lower()
        if side == "LONG" and any(k in joined for k in ["hack","exploit","rug","sec","delist","security breach"]):
            return f"Сильный негатив в новостях: {tops[0]}"
        if side == "SHORT" and any(k in joined for k in ["listing","approval","etf","funding","partnership","mainnet"]):
            return f"Сильный позитив в новостях: {tops[0]}"
    return None

def _tech_risk_trigger(symbol: str, side: str) -> Optional[str]:
    try:
        df15 = market.fetch_ohlcv(symbol, "15m", 200)
        df5 = market.fetch_ohlcv(symbol, "5m", 300)
        if df15 is None or df5 is None:
            return None
        df15["ema200"] = ema(df15["close"], 200)
        st_line, st_dir = supertrend(df15, 10, 3.0)
        st_last = int(st_dir.iloc[-1])
        st_prev = int(st_dir.iloc[-2]) if len(st_dir) >= 2 else st_last
        c15 = float(df15["close"].iloc[-1])
        ema200_15 = float(df15["ema200"].iloc[-1])
        df5["rsi"] = rsi(df5["close"], 14)
        _, _, macdh5 = macd(df5["close"], 12, 26, 9)
        rsi5 = float(df5["rsi"].iloc[-1])
        macd5 = float(macdh5.iloc[-1])
        if side == "LONG":
            if st_last < 0 and st_prev < 0:
                return "SuperTrend(15m) сменился вниз против позиции."
            if c15 < ema200_15 and rsi5 < 45 and macd5 < 0:
                return "Цена ниже EMA200(15m), RSI5<45 и MACD<0 — давление вниз."
        else:
            if st_last > 0 and st_prev > 0:
                return "SuperTrend(15m) сменился вверх против позиции."
            if c15 > ema200_15 and rsi5 > 55 and macd5 > 0:
                return "Цена выше EMA200(15m), RSI5>55 и MACD>0 — давление вверх."
        return None
    except Exception:
        return None

async def watch_signal_price(bot: Bot, chat_id: int, sig: Signal):
    last_trail_update = 0.0
    last_risk_check = 0.0
    try:
        logger.info("Мониторинг: старт %s %s (до %s)", sig.symbol, sig.side, sig.watch_until.isoformat())
        while now_msk() < sig.watch_until and sig.active:
            price = market.fetch_mark_price(sig.symbol)
            if price is None:
                await asyncio.sleep(10)
                continue
            now_ts = time()
            if now_ts - last_risk_check > 60:
                last_risk_check = now_ts
                news_msg = _news_risk_trigger(sig.side, sig.news_note)
                if news_msg and _should_alert(sig.id or -1, "news"):
                    await bot.send_message(chat_id, f"⚠️ Риск-алерт по {sig.symbol.split('/')[0]}: {news_msg}\nРекомендация: сократить/закрыть позицию.")
                tech_msg = _tech_risk_trigger(sig.symbol, sig.side)
                if tech_msg and _should_alert(sig.id or -1, "tech"):
                    await bot.send_message(chat_id, f"⚠️ Риск-алерт по {sig.symbol.split('/')[0]}: {tech_msg}\nРекомендация: сократить/закрыть позицию.")
            if sig.tp_hit >= 1 and sig.trailing and now_ts - last_trail_update > 60:
                await update_trailing(sig)
                last_trail_update = now_ts
                if db: await db.update_signal(sig)
            if sig.side == "LONG":
                if sig.tp_hit < 1 and price >= sig.tps[0]:
                    sig.tp_hit = 1
                    sig.sl = sig.entry
                    sig.trailing = True
                    sig.trail_mode = "supertrend"
                    if PROMETHEUS_OK: MET_TP1.inc()
                    logger.info("Мониторинг: TP1 %s", sig.symbol)
                    await bot.send_message(chat_id, f"✅ TP1 достигнут по {sig.symbol.split('/')[0]} (цена {format_price(price)}). Стоп в БУ ({format_price(sig.sl)}), включён трейлинг.")
                    if db: await db.update_signal(sig)
                if sig.tp_hit < 2 and price >= sig.tps[1]:
                    sig.tp_hit = 2
                    if PROMETHEUS_OK: MET_TP2.inc()
                    logger.info("Мониторинг: TP2 %s", sig.symbol)
                    await bot.send_message(chat_id, f"✅ TP2 достигнут по {sig.symbol.split('/')[0]} (цена {format_price(price)}).")
                    if db: await db.update_signal(sig)
                if sig.tp_hit < 3 and price >= sig.tps[2]:
                    sig.tp_hit = 3
                    sig.active = False
                    if PROMETHEUS_OK: MET_TP3.inc()
                    logger.info("Мониторинг: TP3 %s", sig.symbol)
                    await bot.send_message(chat_id, f"🎯 TP3 достигнут по {sig.symbol.split('/')[0]} (цена {format_price(price)}). Сигнал завершён.")
                    if db: await db.update_signal(sig)
                    break
                if price <= sig.sl:
                    sig.active = False
                    if sig.tp_hit >= 1 and sig.sl >= sig.entry:
                        if PROMETHEUS_OK: MET_BE.inc()
                        logger.info("Мониторинг: BE %s", sig.symbol)
                        await bot.send_message(chat_id, f"🟨 Позиция по {sig.symbol.split('/')[0]} закрыта по безубытку (BE).")
                    else:
                        if PROMETHEUS_OK: MET_STOP.inc()
                        logger.info("Мониторинг: STOP %s", sig.symbol)
                        await bot.send_message(chat_id, f"🛑 Стоп сработал по {sig.symbol.split('/')[0]} (цена {format_price(price)}).")
                    if db: await db.update_signal(sig)
                    break
            else:
                if sig.tp_hit < 1 and price <= sig.tps[0]:
                    sig.tp_hit = 1
                    sig.sl = sig.entry
                    sig.trailing = True
                    sig.trail_mode = "supertrend"
                    if PROMETHEUS_OK: MET_TP1.inc()
                    logger.info("Мониторинг: TP1 %s", sig.symbol)
                    await bot.send_message(chat_id, f"✅ TP1 достигнут по {sig.symbol.split('/')[0]} (цена {format_price(price)}). Стоп в БУ ({format_price(sig.sl)}), включён трейлинг.")
                    if db: await db.update_signal(sig)
                if sig.tp_hit < 2 and price <= sig.tps[1]:
                    sig.tp_hit = 2
                    if PROMETHEUS_OK: MET_TP2.inc()
                    logger.info("Мониторинг: TP2 %s", sig.symbol)
                    await bot.send_message(chat_id, f"✅ TP2 достигнут по {sig.symbol.split('/')[0]} (цена {format_price(price)}).")
                    if db: await db.update_signal(sig)
                if sig.tp_hit < 3 and price <= sig.tps[2]:
                    sig.tp_hit = 3
                    sig.active = False
                    if PROMETHEUS_OK: MET_TP3.inc()
                    logger.info("Мониторинг: TP3 %s", sig.symbol)
                    await bot.send_message(chat_id, f"🎯 TP3 достигнут по {sig.symbol.split('/')[0]} (цена {format_price(price)}). Сигнал завершён.")
                    if db: await db.update_signal(sig)
                    break
                if price >= sig.sl:
                    sig.active = False
                    if sig.tp_hit >= 1 and sig.sl <= sig.entry:
                        if PROMETHEUS_OK: MET_BE.inc()
                        logger.info("Мониторинг: BE %s", sig.symbol)
                        await bot.send_message(chat_id, f"🟨 Позиция по {sig.symbol.split('/')[0]} закрыта по безубытку (BE).")
                    else:
                        if PROMETHEUS_OK: MET_STOP.inc()
                        logger.info("Мониторинг: STOP %s", sig.symbol)
                        await bot.send_message(chat_id, f"🛑 Стоп сработал по {sig.symbol.split('/')[0]} (цена {format_price(price)}).")
                    if db: await db.update_signal(sig)
                    break
            await asyncio.sleep(10)
        if now_msk() >= sig.watch_until and sig.active:
            sig.active = False
            logger.info("Мониторинг: завершён по времени %s", sig.symbol)
            await bot.send_message(chat_id, f"⏱ Мониторинг сигнала по {sig.symbol.split('/')[0]} завершён по времени.")
            if db: await db.update_signal(sig)
    except Exception:
        if PROMETHEUS_OK: MET_WATCH_ERR.inc()
        logger.exception("Мониторинг: ошибка")
        with contextlib.suppress(Exception):
            await bot.send_message(chat_id, "⚠️ Ошибка мониторинга сигнала.")
    finally:
        tasks = active_watch_tasks.get(sig.user_id, [])
        active_watch_tasks[sig.user_id] = [t for t in tasks if not t.done()]

@router.message(Command("start"))
async def cmd_start(message: Message, bot: Bot):
    await message.answer(
        "Привет! Чтобы продолжить, подпишитесь на наш канал:\n"
        f"{'https://t.me/' + CHANNEL_USERNAME.lstrip('@')}\n\n"
        "После подписки нажмите «Продолжить».",
        reply_markup=start_keyboard(),
    )

@router.callback_query(F.data == "continue")
async def on_continue(cb: CallbackQuery, bot: Bot):
    if await is_user_subscribed(bot, cb.from_user.id):
        st = await db.get_user_state(cb.from_user.id)
        await cb.message.answer("Отлично! Подписка подтверждена. Можете запросить сигнал.", reply_markup=main_menu_kb(st.get("unlimited", False)))
    else:
        await cb.message.answer("Вы ещё не подписаны на канал. Подпишитесь и нажмите «Продолжить».", reply_markup=start_keyboard())
    await cb.answer()

@router.message(F.text == "ℹ️ Помощь")
@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer("Мы — NEON. Этот бот подбирает и сопровождает крипто‑сигналы на базе теханализа и новостного фона. Бот не даёт финсоветы; все решения — на вашей стороне.", parse_mode=ParseMode.HTML)

@router.message(Command("ping"))
async def cmd_ping(message: Message):
    uptime = human_td(now_utc() - APP_STARTED_AT)
    await message.answer(f"pong • uptime {uptime}")

@router.message(Command("health"))
async def cmd_health(message: Message):
    ok_news = bool(news_cache)
    price_ok = market.fetch_mark_price("BTC/USDT") is not None
    ages = []
    ts_now = time()
    for b in BASES:
        if b in news_cache:
            ages.append(int(ts_now - news_cache[b][0]))
    age_txt = f", avg_news_age={sum(ages)//len(ages)}s" if ages else ""
    await message.answer(f"health • news={'OK' if ok_news else 'NO'} • price={'OK' if price_ok else 'NO'}{age_txt}")

async def ensure_user_counter(user_id: int) -> Dict[str, Any]:
    assert db is not None
    return await db.get_user_state(user_id)

def resolve_symbol_from_query(q: str) -> Optional[str]:
    t = normalize_text(q)
    for code, syns in COIN_SYNONYMS.items():
        for s in syns:
            s2 = normalize_text(s)
            if t == s2 or re.search(rf"(^|[^a-z0-9а-я]){re.escape(s2)}([^a-z0-9а-я]|$)", t):
                return f"{code}/USDT"
    t3 = re.sub(r"[^a-zA-Z]", "", q).upper()
    if t3 and t3 in COIN_SYNONYMS:
        return f"{t3}/USDT"
    return None

async def guard_access(message: Message, bot: Bot) -> Optional[Dict[str, Any]]:
    user_id = message.from_user.id
    if not await is_user_subscribed(bot, user_id):
        await message.answer("Подпишитесь на канал, чтобы пользоваться ботом:", reply_markup=start_keyboard())
        return None
    st = await db.get_user_state(user_id)
    if st.get("support_mode"):
        await message.answer("Вы в режиме поддержки. Напишите ваш вопрос.", reply_markup=support_kb())
        return None
    return st

@router.message(Command("price"))
async def cmd_price(message: Message, command: CommandObject, bot: Bot):
    st = await guard_access(message, bot)
    if not st:
        return
    query = (command.args or "").strip()
    if not query:
        await message.answer("Использование: /price BTC (или: /price биткоин, /price солана и т.д.)")
        return
    symbol = resolve_symbol_from_query(query)
    if not symbol:
        await message.answer("Монета не распознана. Пример: /price BTC")
        return
    t = market.fetch_ticker(symbol) or {}
    last = t.get("last")
    if not last:
        price = market.fetch_mark_price(symbol)
        if price is None:
            await message.answer("Не удалось получить цену.")
            return
        last = price
    pct = t.get("percentage")
    base = symbol.split("/")[0]
    if pct and abs(pct) > 0.001:
        sign = "▲" if pct >= 0 else "▼"
        await message.answer(f"{base}: {format_price(float(last))} ({sign}{pct:.2f}% 24ч)")
    else:
        await message.answer(f"{base}: {format_price(float(last))}")

@router.message(F.text == "Поддержка")
async def on_support(message: Message, bot: Bot):
    if not await is_user_subscribed(bot, message.from_user.id):
        await message.answer("Подпишитесь на канал, чтобы пользоваться ботом:", reply_markup=start_keyboard())
        return
    st = await db.get_user_state(message.from_user.id)
    if st.get("support_mode"):
        await message.answer("Вы уже в режиме поддержки. Напишите ваш вопрос.", reply_markup=support_kb())
        return
    await db.set_support_mode(message.from_user.id, True)
    await message.answer("Напишите ваш вопрос", reply_markup=support_kb())

@router.message(F.text == "Назад")
async def on_support_back_btn(message: Message, bot: Bot):
    st = await db.get_user_state(message.from_user.id)
    await db.set_support_mode(message.from_user.id, False)
    if not await is_user_subscribed(bot, message.from_user.id):
        await message.answer("Вы вышли из режима поддержки. Для доступа к функциям подпишитесь на канал.", reply_markup=start_keyboard())
        return
    await message.answer("Вы вернулись к основному функционалу.", reply_markup=main_menu_kb(st.get("unlimited", False)))

@router.callback_query(F.data == "support_user_back")
async def cb_support_user_back(cb: CallbackQuery):
    await db.set_support_mode(cb.from_user.id, False)
    st = await db.get_user_state(cb.from_user.id)
    await cb.message.answer("Вы вернулись к основному функционалу.", reply_markup=main_menu_kb(st.get("unlimited", False)))
    await cb.answer("Режим поддержки отключён")

@router.callback_query(F.data == "support_user_reply")
async def cb_support_user_reply(cb: CallbackQuery):
    await db.set_support_mode(cb.from_user.id, True)
    await cb.message.answer("Напишите ваш вопрос", reply_markup=support_kb())
    await cb.answer("Режим поддержки включён")

class UserSupportFilter(Filter):
    async def __call__(self, message: Message) -> bool:
        if not message.from_user or not db:
            return False
        if not message.text or message.text.strip() in {"Назад"}:
            return False
        if message.text.startswith("/"):
            return False
        st = await db.get_user_state(message.from_user.id)
        return bool(st.get("support_mode"))

class AdminReplyFilter(Filter):
    async def __call__(self, message: Message) -> bool:
        if not message.from_user:
            return False
        return message.from_user.id in support_reply_targets and bool(message.text)

def user_display_name(u: Any) -> str:
    if not u:
        return "пользователь"
    if getattr(u, "username", None):
        return f"@{u.username}"
    fn = (u.first_name or "").strip() if getattr(u, "first_name", None) else ""
    ln = (u.last_name or "").strip() if getattr(u, "last_name", None) else ""
    nm = (fn + " " + ln).strip() or "пользователь"
    return nm

@router.message(UserSupportFilter())
async def user_support_message(message: Message, bot: Bot):
    txt = message.text.strip()
    await message.answer("Мы уже увидели ваш вопрос, дожидайтесь ответа", reply_markup=support_kb())
    admin_ids = await db.get_admin_user_ids()
    if not admin_ids:
        return
    u = message.from_user
    disp = user_display_name(u)
    for aid in admin_ids:
        with contextlib.suppress(Exception):
            await bot.send_message(aid, f"Вопрос от {disp} (id {u.id}):\n\n{txt}", reply_markup=admin_answer_kb(u.id))

@router.callback_query(F.data.startswith("answer_user:"))
async def cb_admin_answer(cb: CallbackQuery, bot: Bot):
    try:
        target_id = int(cb.data.split(":", 1)[1])
    except Exception:
        await cb.answer()
        return
    target_chat = None
    with contextlib.suppress(Exception):
        target_chat = await bot.get_chat(target_id)
    name = user_display_name(target_chat)
    support_reply_targets[cb.from_user.id] = target_id
    await cb.message.answer(f"Введите ваш ответ пользователю {name}.")
    await cb.answer()

@router.message(AdminReplyFilter())
async def admin_send_answer(message: Message, bot: Bot):
    admin_id = message.from_user.id
    target_id = support_reply_targets.get(admin_id)
    if not target_id:
        return
    txt = message.text.strip()
    with contextlib.suppress(Exception):
        await bot.send_message(target_id, f"Сообщение от админа:\n\n{txt}", reply_markup=user_reply_inline_kb())
    with contextlib.suppress(Exception):
        await message.answer("Сообщение отправлено")
    support_reply_targets.pop(admin_id, None)

@router.message(F.text == "📈 Получить сигнал")
@router.message(Command("signal"))
async def cmd_signal(message: Message, bot: Bot):
    user_id = message.from_user.id
    st = await guard_access(message, bot)
    if not st:
        return
    logger.info("Запрошен сигнал пользователем %s", user_id)
    if not st.get("unlimited") and st.get("count", 0) >= DAILY_LIMIT:
        await message.answer("Лимит 3 сигнала в день исчерпан. Введите /code для безлимита.")
        return
    working_msg = await message.answer("🔍 Ищу торговую пару...")
    logger.info("Поиск пары: старт")
    try:
        ranked = await rank_symbols_async(SYMBOLS)
        if not ranked:
            logger.warning("Поиск пары: кандидатов нет")
            await working_msg.edit_text("Не удалось найти подходящий сигнал. Попробуйте позже.")
            return
        logger.info("Поиск пары: найдено кандидатов %d", len(ranked))
        existing = await db.get_active_signals_for_user(user_id)
        picked = None
        for symbol, details in ranked:
            side = details["side"]
            if any(s.symbol == symbol and s.side == side and s.active for s in existing):
                continue
            picked = (symbol, details)
            break
        if picked is None:
            symbol, details = ranked[0]
        else:
            symbol, details = picked
        side = details["side"]
        entry = details["c5"]
        sl = details["sl"]
        tps = details["tps"]
        leverage = details["leverage"]
        risk_level = details["risk_level"]
        news_note = details["news_note"]
        atr_value = details["atr"]
        watch_seconds = details["watch_seconds"]
        reason = build_reason(details)
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
        if sig.side == "LONG":
            if not (all(tp > sig.entry for tp in sig.tps) and sig.sl < sig.entry):
                logger.warning("Валидация не пройдена для %s LONG", symbol)
                await working_msg.edit_text("Сигнал не прошёл валидацию. Попробуйте ещё раз.")
                return
        else:
            if not (all(tp < sig.entry for tp in sig.tps) and sig.sl > sig.entry):
                logger.warning("Валидация не пройдена для %s SHORT", symbol)
                await working_msg.edit_text("Сигнал не прошёл валидацию. Попробуйте ещё раз.")
                return
        text = format_signal_message(sig)
        await working_msg.edit_text(text, parse_mode=ParseMode.HTML)
        st["count"] = st.get("count", 0) + 1
        await db.save_user_state(user_id, st)
        sig.id = await db.add_signal(sig)
        task = asyncio.create_task(watch_signal_price(bot, message.chat.id, sig))
        active_watch_tasks.setdefault(user_id, []).append(task)
        if PROMETHEUS_OK: MET_SIGNALS_GEN.inc()
        logger.info("Выбран сигнал: %s %s entry=%s sl=%s lev=%s risk=%s", symbol, side, entry, sl, leverage, risk_level)
    except Exception as e:
        if PROMETHEUS_OK: MET_SIGNAL_ERR.inc()
        logger.exception("Ошибка генерации сигнала: %s", e)
        with contextlib.suppress(Exception):
            await working_msg.edit_text("⚠️ Произошла ошибка при поиске сигнала.")

@router.message(Command("status"))
async def cmd_status(message: Message, bot: Bot):
    st = await guard_access(message, bot)
    if not st:
        return
    sigs = await db.get_active_signals_for_user(message.from_user.id)
    if not sigs:
        await message.answer("Активных сигналов нет.")
        return
    lines = []
    for s in sigs:
        left = human_td(s.watch_until - now_msk())
        base = s.symbol.split("/")[0]
        lines.append(f"{base} {s.side} • TBX {format_price(s.entry)} • SL {format_price(s.sl)} • TP {s.tp_hit}/3 • осталось {left}")
    await message.answer("Активные сигналы:\n" + "\n".join(lines))

@router.message(Command("code"))
async def cmd_code(message: Message, command: CommandObject, bot: Bot):
    code = (command.args or "").strip()
    if not code:
        await message.answer("Введите код: /code 2604")
        return
    st = await ensure_user_counter(message.from_user.id)
    if code == ADMIN_ACCESS_CODE:
        st["unlimited"] = True
        await db.save_user_state(message.from_user.id, st)
        await message.answer("✅ Код принят. Ограничения по количеству сигналов сняты.", reply_markup=main_menu_kb(True))
    else:
        await message.answer("❌ Неверный код.")

async def backtest_pair(df15: pd.DataFrame) -> Tuple[int, int, int, int]:
    df = df15.copy()
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["atr"] = atr(df, 14)
    adx15 = adx(df, 14)
    st_line, st_dir = supertrend(df, 10, 3.0)
    df["st_dir"] = st_dir
    trades = 0
    tp1 = 0
    tp2 = 0
    tp3 = 0
    stops = 0
    for i in range(60, len(df) - 1):
        close = float(df["close"].iloc[i])
        atrv = float(df["atr"].iloc[i])
        if atrv <= 0:
            continue
        dir_now = int(df["st_dir"].iloc[i])
        dir_prev = int(df["st_dir"].iloc[i - 1])
        adxv = float(adx15.iloc[i])
        long_sig = dir_prev < 0 and dir_now > 0 and adxv > 18 and df["ema50"].iloc[i] > df["ema200"].iloc[i]
        short_sig = dir_prev > 0 and dir_now < 0 and adxv > 18 and df["ema50"].iloc[i] < df["ema200"].iloc[i]
        if not long_sig and not short_sig:
            continue
        trades += 1
        if long_sig:
            sl = close - 2.0 * atrv
            risk = close - sl
            tps = [close + 1.5 * risk, close + 2.5 * risk, close + 3.5 * risk]
            hit = 0
            for j in range(i + 1, min(i + 96 * 2, len(df))):
                h = float(df["high"].iloc[j])
                l = float(df["low"].iloc[j])
                if l <= sl:
                    stops += 1
                    break
                if hit < 1 and h >= tps[0]:
                    hit = 1
                    tp1 += 1
                if hit < 2 and h >= tps[1]:
                    hit = 2
                    tp2 += 1
                if hit < 3 and h >= tps[2]:
                    hit = 3
                    tp3 += 1
                    break
        else:
            sl = close + 2.0 * atrv
            risk = sl - close
            tps = [close - 1.5 * risk, close - 2.5 * risk, close - 3.5 * risk]
            hit = 0
            for j in range(i + 1, min(i + 96 * 2, len(df))):
                h = float(df["high"].iloc[j])
                l = float(df["low"].iloc[j])
                if h >= sl:
                    stops += 1
                    break
                if hit < 1 and l <= tps[0]:
                    hit = 1
                    tp1 += 1
                if hit < 2 and l <= tps[1]:
                    hit = 2
                    tp2 += 1
                if hit < 3 and l <= tps[2]:
                    hit = 3
                    tp3 += 1
                    break
    return trades, tp1, tp2, tp3, stops

@router.message(Command("backtest"))
async def cmd_backtest(message: Message, command: CommandObject, bot: Bot):
    st_guard = await guard_access(message, bot)
    if not st_guard:
        return
    args = (command.args or "").strip()
    days = 60
    pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
    try:
        if args:
            for part in args.split():
                if part.isdigit():
                    days = int(part)
                elif "," in part:
                    pp = [p.strip().upper().replace("USDT/USDT", "USDT") for p in part.split(",") if p.strip()]
                    if pp:
                        pairs = [p if "/" in p else f"{p}/USDT" for p in pp]
    except Exception:
        pass
    await message.answer(f"Запускаю бэктест: {days}д, пары: {', '.join(pairs)}")
    total = 0
    tp1 = tp2 = tp3 = stops = 0
    for sym in pairs:
        df15 = market.fetch_ohlcv(sym, "15m", min(96 * days + 100, 10000))
        if df15 is None or len(df15) < 200:
            continue
        tr, t1, t2, t3, stp = await backtest_pair(df15)
        total += tr
        tp1 += t1
        tp2 += t2
        tp3 += t3
        stops += stp
    if db:
        await db.save_backtest(pairs, days, total, tp1, tp2, tp3, stops)
    rate1 = f"{(tp1/total*100):.1f}%" if total else "n/a"
    rate2 = f"{(tp2/total*100):.1f}%" if total else "n/a"
    rate3 = f"{(tp3/total*100):.1f}%" if total else "n/a"
    stpr = f"{(stops/total*100):.1f}%" if total else "n/a"
    await message.answer(f"BT итоги • сделки={total} • TP1={tp1} ({rate1}) • TP2={tp2} ({rate2}) • TP3={tp3} ({rate3}) • Stops={stops} ({stpr})")

@router.message()
async def fallback(message: Message):
    text = (message.text or "").strip().lower()
    if text in {"помощь", "help"}:
        await cmd_help(message)
    elif text in {"получить сигнал", "сигнал"}:
        await cmd_signal(message, message.bot)
    else:
        st = await db.get_user_state(message.from_user.id)
        if st.get("support_mode"):
            await message.answer("Вы в режиме поддержки. Напишите ваш вопрос.", reply_markup=support_kb())
            return
        await message.answer("Команда не распознана. Доступно: /start, /help, /signal, /status, /code, /ping, /health, /backtest, /price")

async def on_startup(bot: Bot):
    logger.info("Старт бота: проверка новостей...")
    ok = check_news_health()
    if ok:
        logger.info("Новости RSS: OK")
    else:
        logger.warning("Новости RSS: пока недоступны, будет использован кэш/резерв")
    asyncio.create_task(news_updater())
    try:
        active_sigs = await db.get_all_active_signals()
        for sig in active_sigs:
            if now_msk() >= sig.watch_until:
                sig.active = False
                await db.update_signal(sig)
                continue
            asyncio.create_task(bot.send_message(sig.user_id, f"♻️ Восстановил мониторинг сигнала по {sig.symbol.split('/')[0]} {sig.side}."))
            task = asyncio.create_task(watch_signal_price(bot, sig.user_id, sig))
            active_watch_tasks.setdefault(sig.user_id, []).append(task)
        if active_sigs:
            logger.info("Восстановлено активных сигналов: %d", len(active_sigs))
    except Exception as e:
        logger.exception("Ошибка восстановления сигналов: %s", e)

LOCK_HANDLE = None

def acquire_lock():
    global LOCK_HANDLE
    if os.path.exists(LOCK_PATH):
        raise RuntimeError("Lock file exists. Другой инстанс уже запущен.")
    LOCK_HANDLE = open(LOCK_PATH, "x")
    LOCK_HANDLE.write(f"{os.getpid()} {int(time())}\n")
    LOCK_HANDLE.flush()

def release_lock():
    try:
        if LOCK_HANDLE:
            LOCK_HANDLE.close()
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)
    except Exception:
        pass

async def main():
    global db
    acquire_lock()
    if PROMETHEUS_OK and METRICS_PORT > 0:
        with contextlib.suppress(Exception):
            start_http_server(METRICS_PORT)
            logger.info("Metrics on :%s", METRICS_PORT)
    db = Database(DB_PATH)
    await db.init()
    bot = Bot(token=TELEGRAM_BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    await on_startup(bot)
    logger.info("Бот запущен (long polling).")
    try:
        await dp.start_polling(bot)
    finally:
        release_lock()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен.")
        with contextlib.suppress(Exception):
            release_lock()
