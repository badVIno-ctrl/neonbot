
# last.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os, math, asyncio, contextlib, json, time
from datetime import datetime, timedelta, timezone

LAST_TOPK=int(os.getenv("LAST_TOPK","10"))
LAST_SCORE_MIN=float(os.getenv("LAST_SCORE_MIN","2.00"))
LAST_P_MIN=float(os.getenv("LAST_P_MIN","0.60"))
LAST_ADX_MIN=float(os.getenv("LAST_ADX_MIN","22.0"))
LAST_ATR_PCT_MIN=float(os.getenv("LAST_ATR_PCT_MIN","0.18"))
LAST_ATR_PCT_MAX=float(os.getenv("LAST_ATR_PCT_MAX","1.90"))
LAST_SPREAD_Z_MAX=float(os.getenv("LAST_SPREAD_Z_MAX","3.0"))
LAST_NEAR_DEPTH_MIN=float(os.getenv("LAST_NEAR_DEPTH_MIN","0.50"))
LAST_SPOOF_MAX=float(os.getenv("LAST_SPOOF_MAX","0.99"))
LAST_MIN_RR1=float(os.getenv("LAST_MIN_RR1","1.20"))
LAST_MIN_RR2=float(os.getenv("LAST_MIN_RR2","1.60"))
LAST_K_STOP_ATR=float(os.getenv("LAST_K_STOP_ATR","1.0"))
LAST_R_LADDER=[float(x) for x in os.getenv("LAST_R_LADDER","1.0,1.75,2.5").split(",") if x.strip()]
LAST_TP_CAP_ATR_MAX=float(os.getenv("LAST_TP_CAP_ATR_MAX","4.0"))
LAST_TP_CAP_PCT_MAX=float(os.getenv("LAST_TP_CAP_PCT_MAX","0.26"))
LAST_TP_MIN_STEP_ATR=float(os.getenv("LAST_TP_MIN_STEP_ATR","0.15"))
LAST_SL_MIN_ATR=float(os.getenv("LAST_SL_MIN_ATR","0.25"))
LAST_SL_MAX_ATR=float(os.getenv("LAST_SL_MAX_ATR","5.0"))
LAST_MEME_PRICE_MIN=float(os.getenv("LAST_MEME_PRICE_MIN","0.005"))
LAST_MAX_LEV=int(os.getenv("LAST_MAX_LEV","20"))
LAST_MIN_LEV=int(os.getenv("LAST_MIN_LEV","5"))
LAST_SCALP_FORMAT=os.getenv("LAST_SCALP_FORMAT","1")=="1"
LAST_LOG_REJECTIONS=os.getenv("LAST_LOG_REJECTIONS","1")=="1"
LAST_SCAN_ENABLE=os.getenv("LAST_SCAN_ENABLE","1")=="1"
LAST_SCAN_INTERVAL_SEC=int(os.getenv("LAST_SCAN_INTERVAL_SEC","60"))
LAST_SCAN_P_MIN=float(os.getenv("LAST_SCAN_P_MIN","0.62"))
LAST_SCAN_SCORE_MIN=float(os.getenv("LAST_SCAN_SCORE_MIN","2.10"))
LAST_CHANNEL_DAILY_CAP=int(os.getenv("LAST_CHANNEL_DAILY_CAP","2"))
LAST_ANTI_BTC=os.getenv("LAST_ANTI_BTC","1")=="1"
LAST_CORR_WINDOW=int(os.getenv("LAST_CORR_WINDOW","180"))
LAST_CORR_THRESH=float(os.getenv("LAST_CORR_THRESH","0.45"))
LAST_KILLZONES=os.getenv("LAST_KILLZONES","1")=="1"
LAST_CALIB_ENABLE=os.getenv("LAST_CALIB_ENABLE","1")=="1"
LAST_CALIB_INTERVAL_SEC=int(os.getenv("LAST_CALIB_INTERVAL_SEC","21600"))
LAST_K_MIN_ATR=float(os.getenv("LAST_K_MIN_ATR","0.7"))
LAST_K_MAX_ATR=float(os.getenv("LAST_K_MAX_ATR","1.5"))
LAST_PORTF_BLOCK_MINUTES=int(os.getenv("LAST_PORTF_BLOCK_MINUTES","120"))
LAST_PORTF_MAX_SAME_SIDE=int(os.getenv("LAST_PORTF_MAX_SAME_SIDE","2"))
LAST_OPP_POST_MIN=int(os.getenv("LAST_OPP_POST_MIN","7"))
LAST_IMPULSE_K=float(os.getenv("LAST_IMPULSE_K","1.8"))
LAST_ROUND_N_TICKS=int(os.getenv("LAST_ROUND_N_TICKS","2"))
LAST_DEPTH_ABS_MIN=float(os.getenv("LAST_DEPTH_ABS_MIN","0.0"))
LAST_DEPTH_NEAR_MIN=float(os.getenv("LAST_DEPTH_NEAR_MIN","0.50"))
LAST_EXECUTE_ON_CH=str(os.getenv("TG_CHANNEL") or os.getenv("CHANNEL_USERNAME") or "").strip()
LAST_LOW_PRICE_THRESHOLD = float(os.getenv("LAST_LOW_PRICE_THRESHOLD", "0.10"))
LAST_LOW_TP_PCTS = [float(x) for x in os.getenv("LAST_LOW_TP_PCTS","0.015,0.040,0.070").split(",") if x.strip()]
LAST_LOW_SL_PCT = float(os.getenv("LAST_LOW_SL_PCT","0.030"))

def _now_utc()->datetime:
    return datetime.now(timezone.utc)

def _infer_safe_tick(entry:float,df15)->float:
    try:
        if entry<=0: return 1e-6
        if entry>=100:dec=2
        elif entry>=10:dec=3
        elif entry>=1:dec=4
        elif entry>=0.1:dec=5
        else:dec=6
        tick=10**(-dec)
        if df15 is not None and len(df15)>=20:
            med_rng=float((df15["high"]-df15["low"]).tail(20).median())
            est_spread=med_rng*0.1
            if est_spread>0: tick=min(tick,est_spread/10.0)
        return max(tick,10**(-8))
    except Exception:
        return 10**(-6)

def _round_tick(x:float,tick:float,mode:str)->float:
    try:
        n=float(x)/(tick if tick>0 else 1e-9)
        if mode=="floor": n=math.floor(n)
        elif mode=="ceil": n=math.ceil(n)
        else: n=round(n)
        return float(n*(tick if tick>0 else 1e-9))
    except Exception:
        return float(x)

def _ensure_tp_monotonic_with_step(side:str,entry:float,tps:List[float],atr:float,tick:float,step_atr:float)->List[float]:
    tps=[float(x) for x in tps[:3]]+([tps[-1]]*max(0,3-len(tps))) if tps else [entry,entry,entry]
    step=max(1e-9,float(step_atr)*float(atr if atr>0 else max(1e-9,0.001*entry)))
    if side=="LONG":
        tps=sorted(tps)
        tps[0]=_round_tick(max(tps[0],entry+step),tick,"ceil")
        tps[1]=_round_tick(max(tps[1],tps[0]+step),tick,"ceil")
        tps[2]=_round_tick(max(tps[2],tps[1]+step),tick,"ceil")
    else:
        tps=sorted(tps,reverse=True)
        tps[0]=_round_tick(min(tps[0],entry-step),tick,"floor")
        tps[1]=_round_tick(min(tps[1],tps[0]-step),tick,"floor")
        tps[2]=_round_tick(min(tps[2],tps[1]-step),tick,"floor")
    return tps[:3]

def _cap_tp(entry:float,tp:float,atr:float,side:str)->float:
    if side=="LONG":
        cap1=entry+LAST_TP_CAP_ATR_MAX*atr
        cap2=entry*(1.0+LAST_TP_CAP_PCT_MAX)
        return min(tp,cap1,cap2)
    else:
        cap1=entry-LAST_TP_CAP_ATR_MAX*atr
        cap2=entry*(1.0-LAST_TP_CAP_PCT_MAX)
        return max(tp,cap1,cap2)

def _compute_pdh_pdl(df15)->Optional[Tuple[float,float]]:
    try:
        if df15 is None or len(df15)<60:return None
        ts=df15["ts"]; last=ts.iloc[-1].to_pydatetime().astimezone(timezone.utc)
        day_anchor=last.replace(hour=0,minute=0,second=0,microsecond=0)
        prev=df15[(ts>=day_anchor-timedelta(days=1))&(ts<day_anchor)]
        if prev is None or prev.empty:return None
        pdh=float(prev["high"].max()); pdl=float(prev["low"].min())
        return (pdh,pdl) if pdh>pdl else None
    except Exception:
        return None

def _cap_by_pdr(entry:float,tp:float,side:str,pd:Optional[Tuple[float,float]])->float:
    if not pd:return tp
    pdh,pdl=pd; dr=abs(pdh-pdl)
    if dr<=0:return tp
    cap=entry+(1.2*dr)*(1 if side=="LONG" else -1)
    return min(tp,cap) if side=="LONG" else max(tp,cap)

def _rr(entry:float,sl:float,tp:float)->float:
    risk=abs(entry-sl)+1e-9
    return abs(tp-entry)/risk

def _round_step(price:float)->float:
    try:
        if price<0.01:return 1e-6
        if price<0.1:return 1e-4
        if price<1:return 0.01
        if price<10:return 0.1
        if price<100:return 1.0
        return 5.0
    except Exception:
        return 0.01

def _avoid_round_levels(entry:float,level:float,tick:float,side:str)->float:
    step=_round_step(entry)
    base=math.floor(level/step)*step
    dist=abs(level-base)
    guard=max(LAST_ROUND_N_TICKS*(tick if tick>0 else step/100.0),step*0.002)
    if dist<guard:
        level=base+guard if side=="LONG" else base-guard
    return _round_tick(level,tick,"ceil" if side=="LONG" else "floor")

def _volume_profile_local(df, bins=40, lookback=240):
    try:
        import numpy as np
        if df is None or len(df)<20:
            return None, None, None
        x = df.tail(lookback).copy()
        tp = (x["high"] + x["low"] + x["close"]) / 3.0
        vol = x["volume"].astype(float)
        prices = tp.values.astype(float); weights = vol.values
        lo = float(np.min(prices)); hi = float(np.max(prices))
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo>=hi:
            return None,None,None
        hist, edges = np.histogram(prices, bins=bins, range=(lo, hi), weights=weights)
        if hist.sum()<=0: return None,None,None
        poc_idx=int(np.argmax(hist)); poc=float(0.5*(edges[poc_idx]+edges[poc_idx+1]))
        order = np.argsort(hist)[::-1]
        acc=0.0; total=float(hist.sum())
        mask=np.zeros_like(hist, dtype=bool)
        for idx in order:
            mask[idx]=True; acc+=float(hist[idx])
            if acc/total>=0.68: break
        sel = np.where(mask)[0]
        val=float(edges[sel.min()]); vah=float(edges[sel.max()+1])
        return poc,vah,val
    except Exception:
        return None,None,None

def _initial_balance_local(df15, ib_hours=1):
    try:
        if df15 is None or len(df15)<5:
            return None,None
        ts = df15["ts"]
        last = ts.iloc[-1].to_pydatetime().astimezone(timezone.utc)
        day_anchor = last.replace(hour=0, minute=0, second=0, microsecond=0)
        cur = df15[(ts>=day_anchor)]
        if cur is None or cur.empty:
            return None,None
        bars=max(1, int(ib_hours*60/15))
        ib = cur.head(bars)
        return float(ib["high"].max()), float(ib["low"].min())
    except Exception:
        return None,None

def _swing_levels(df, lookback=30):
    try:
        import numpy as np
        if df is None or len(df)<(lookback+10):
            return [],[]
        close = df["close"].astype(float).values
        highs, lows = [], []
        for i in range(2, len(close)-2):
            if close[i]>close[i-1] and close[i]>close[i+1] and close[i]>close[i-2] and close[i]>close[i+2]:
                highs.append(float(df["high"].iloc[i]))
            if close[i]<close[i-1] and close[i]<close[i+1] and close[i]<close[i-2] and close[i]<close[i+2]:
                lows.append(float(df["low"].iloc[i]))
        return highs[-lookback:], lows[-lookback:]
    except Exception:
        return [],[]

def _ta_candidate_levels(df15, df1h, entry, side):
    poc,vah,val = _volume_profile_local(df15, 40, 240)
    ib_hi, ib_lo = _initial_balance_local(df15, 1)
    pd = _compute_pdh_pdl(df15)
    pdh,pdl = (pd if pd else (None,None))
    hs15, ls15 = _swing_levels(df15, 40)
    hs1h, ls1h = _swing_levels(df1h, 40) if df1h is not None else ([],[])
    ups = []
    dns = []
    for v in [vah, poc, ib_hi, pdh] + hs15 + hs1h:
        if v is not None and v>entry:
            ups.append(float(v))
    for v in [val, poc, ib_lo, pdl] + ls15 + ls1h:
        if v is not None and v<entry:
            dns.append(float(v))
    ups = sorted(list(set(ups)))
    dns = sorted(list(set(dns)), reverse=True)
    return ups, dns, (poc,vah,val,ib_hi,ib_lo,pdh,pdl)

def _snap_targets_to_ta(entry, side, tps, df15, df1h, atr, tick):
    ups, dns, _ = _ta_candidate_levels(df15, df1h, entry, side)
    def _closest_above(x, arr):
        best=None; bd=1e18
        for v in arr:
            if v>entry:
                d=abs(v-x)
                if d<bd: bd=d; best=v
        return best
    def _closest_below(x, arr):
        best=None; bd=1e18
        for v in arr:
            if v<entry:
                d=abs(v-x)
                if d<bd: bd=d; best=v
        return best
    out=[]
    for tp in tps:
        if side=="LONG":
            c = _closest_above(tp, ups)
            if c is None:
                c = tp
            c = max(c, entry + 0.4*max(atr, tick*5))
            out.append(c)
        else:
            c = _closest_below(tp, dns)
            if c is None:
                c = tp
            c = min(c, entry - 0.4*max(atr, tick*5))
            out.append(c)
    out = _ensure_tp_monotonic_with_step(side, entry, out, atr, tick, LAST_TP_MIN_STEP_ATR)
    if side=="LONG":
        out=[_round_tick(_avoid_round_levels(entry,x,tick,side),tick,"ceil") for x in out]
    else:
        out=[_round_tick(_avoid_round_levels(entry,x,tick,side),tick,"floor") for x in out]
    return out[:3]

async def _ensure_last_tables(app:Dict[str,Any])->None:
    db=app.get("db")
    if not db or not getattr(db,"conn",None):return
    try:
        await db.conn.execute("""
        CREATE TABLE IF NOT EXISTS last_channel_cap(
            date TEXT PRIMARY KEY,
            count INTEGER NOT NULL
        )""")
        await db.conn.commit()
    except Exception:
        pass

def _log_reject(app:Dict[str,Any],symbol:str,side:str,reason:str,metrics:Dict[str,Any],relax:bool):
    if not LAST_LOG_REJECTIONS:
        return
    try:
        log = app.setdefault("_last_reject_log", [])
        log.append({
            "ts": app["now_msk"]().isoformat(),
            "symbol": symbol, "side": side,
            "reason": reason, "relax": bool(relax),
            "metrics": metrics
        })
        if len(log) > 2000:
            del log[:len(log)-2000]
    except Exception:
        pass

async def _get_daily_cap(app:Dict[str,Any])->int:
    try:
        db=app.get("db")
        if not db or not getattr(db,"conn",None):return 0
        dkey=app["now_msk"]().date().isoformat()
        cur=await db.conn.execute("SELECT count FROM last_channel_cap WHERE date=?", (dkey,))
        row=await cur.fetchone()
        return int(row["count"]) if row else 0
    except Exception:
        return 0

async def _inc_daily_cap(app:Dict[str,Any]):
    try:
        db=app.get("db")
        if not db or not getattr(db,"conn",None):return
        dkey=app["now_msk"]().date().isoformat()
        cur=await db.conn.execute("SELECT count FROM last_channel_cap WHERE date=?", (dkey,))
        row=await cur.fetchone()
        cnt=int(row["count"])+1 if row else 1
        await db.conn.execute(
            "INSERT INTO last_channel_cap(date,count) VALUES(?,?) ON CONFLICT(date) DO UPDATE SET count=excluded.count",
            (dkey,cnt)
        )
        await db.conn.commit()
    except Exception:
        pass

def _portfolio_state(app:Dict[str,Any])->Dict[str,Any]:
    return app.setdefault("_last_portf",{"posted":[],"last_by_symbol":{}})

def _can_post_portf(app:Dict[str,Any],side:str)->bool:
    st=_portfolio_state(app)
    ts_now=time.time()
    st["posted"]=[(t,s,sd) for (t,s,sd) in st["posted"] if ts_now-t<=LAST_PORTF_BLOCK_MINUTES*60]
    same=sum(1 for (_,_,sd) in st["posted"] if sd==side)
    return same<LAST_PORTF_MAX_SAME_SIDE

def _append_posted(app:Dict[str,Any],symbol:str,side:str):
    st=_portfolio_state(app)
    st["posted"].append((time.time(),symbol,side))
    st["last_by_symbol"][symbol]=(side,time.time())

def _opp_cooldown_ok(app:Dict[str,Any],symbol:str,side:str)->bool:
    st=_portfolio_state(app)
    prev=st["last_by_symbol"].get(symbol)
    if not prev:return True
    pside,pts=prev
    if pside==side:return True
    return (time.time()-pts)>=LAST_OPP_POST_MIN*60

def _btc_eth_corr_trend(app:Dict[str,Any],symbol:str)->Tuple[Optional[float],Optional[int],Optional[float],Optional[int]]:
    try:
        m=app.get("market")
        df_a=m.fetch_ohlcv(symbol,"5m",max(220,LAST_CORR_WINDOW+10))
        df_b=m.fetch_ohlcv("BTC/USDT","5m",max(220,LAST_CORR_WINDOW+10))
        df_e=m.fetch_ohlcv("ETH/USDT","5m",max(220,LAST_CORR_WINDOW+10))
        if df_a is None or df_b is None or df_e is None:return None,None,None,None
        a=df_a["close"].pct_change().dropna().tail(LAST_CORR_WINDOW)
        b=df_b["close"].pct_change().dropna().tail(LAST_CORR_WINDOW)
        e=df_e["close"].pct_change().dropna().tail(LAST_CORR_WINDOW)
        import numpy as np
        def trend_of(series):
            y=series.tail(200).astype(float).values
            x=np.arange(len(y),dtype=float)
            if len(y)<10:return 0
            sl,_=np.polyfit(x,y,1)
            return 1 if sl>0 else (-1 if sl<0 else 0)
        n=min(len(a),len(b),len(e))
        if n<30:return None,None,None,None
        corr_b=float(np.corrcoef(a.tail(n),b.tail(n))[0,1])
        corr_e=float(np.corrcoef(a.tail(n),e.tail(n))[0,1])
        tr_b=trend_of(df_b["close"]); tr_e=trend_of(df_e["close"])
        return corr_b,tr_b,corr_e,tr_e
    except Exception:
        return None,None,None,None

def _ensure_ob_fields(app:Dict[str,Any],d:Dict[str,Any],symbol:str):
    try:
        need=("spread_norm" not in d) or ("near_depth_ratio" not in d) or ("spoof_score" not in d)
        if not need:return
        m=app.get("market")
        ob=None
        for name,ex in m._available_exchanges():
            resolved=m.resolve_symbol(ex,symbol) or symbol
            if resolved not in ex.markets:continue
            try:
                ob=ex.fetch_order_book(resolved,limit=25)
                if isinstance(ob,dict):break
            except Exception:
                continue
        if not isinstance(ob,dict):return
        bids=ob.get("bids") or []
        asks=ob.get("asks") or []
        if not bids or not asks:return
        pb,qb=float(bids[0][0]),float(bids[0][1])
        pa,qa=float(asks[0][0]),float(asks[0][1])
        mid=0.5*(pa+pb)
        spread=max(0.0,pa-pb)
        denom=max(1e-9,mid*0.001)
        d["spread_norm"]=float(spread/denom)
        def near_ratio(levels):
            keep=0.0; tot=0.0
            for p,q in levels[:25]:
                p=float(p); q=float(q)
                tot+=q
                if abs(p-mid)/(mid+1e-9)<=0.001:
                    keep+=q
            return float(keep/(tot+1e-9))
        nb=near_ratio(bids); na=near_ratio(asks)
        d["near_depth_ratio"]=float(0.5*(nb+na))
        import numpy as np
        def max_wall(levels):
            if not levels:return 0.0
            sizes=np.array([float(q) for _,q in levels[:10]],dtype=float)
            med=float(np.median(sizes)+1e-9)
            mx=float(np.max(sizes))
            return float(mx/med)
        d["spoof_score"]=0.0 if max_wall(bids)<3.0 and max_wall(asks)<3.0 else 1.0
    except Exception:
        pass

def _vpin_ofi(df)->Tuple[Optional[float],Optional[float]]:
    try:
        import numpy as np
        if df is None or len(df)<100:return None,None
        c=df["close"].astype(float).values
        v=df["volume"].astype(float).values
        ret=np.diff(c,prepend=c[0])
        sign=np.sign(ret)
        ofi=float(np.sum(sign[-60:]*v[-60:]))
        tot=float(np.sum(v[-20:])+1e-9)
        pos=float(np.sum(v[-20:][ret[-20:]>=0.0]))
        vpin=abs(pos-(tot-pos))/tot
        return ofi,float(vpin)
    except Exception:
        return None,None

def _atr_pct(entry:float,atr:float)->float:
    return float(atr/max(1e-9,entry)*100.0)

def _recent_impulse(app:Dict[str,Any],symbol:str,atr15:float,near_ratio:Optional[float])->bool:
    try:
        m=app.get("market"); df5=m.fetch_ohlcv(symbol,"5m",60)
        if df5 is None or len(df5)<5 or atr15<=0:return False
        rng=float(df5["high"].iloc[-1]-df5["low"].iloc[-1])
        if rng/atr15>=LAST_IMPULSE_K and (near_ratio is not None and near_ratio<LAST_NEAR_DEPTH_MIN):
            return True
        return False
    except Exception:
        return False

def _rank(app:Dict[str,Any],details:Dict[str,Any])->float:
    sc=float(details.get("score",0.0) or 0.0)
    p=float(details.get("p_bayes",details.get("ml_p",0.55)) or 0.55)
    r=0.6*sc+0.4*p
    try:
        adx=float(details.get("adx15",0.0) or 0.0)
        r+=max(0.0,min(0.06,(adx-20.0)/200.0))
    except Exception:
        pass
    try:
        spreadZ=float(details.get("spread_norm",0.0) or 0.0)
        near=float(details.get("near_depth_ratio",details.get("ta_near_depth_ratio",0.0)) or 0.0)
        spoof=float(details.get("spoof_score",0.0) or 0.0)
        r-=max(0.0,(spreadZ-2.5))*0.03
        r+=max(0.0,(near-0.5))*0.04
        r-=max(0.0,spoof)*0.05
    except Exception:
        pass
    if LAST_KILLZONES:
        h=app["now_msk"]().hour
        if h in (3,4,5,23): r-=0.08
        if h in (10,11,16,17): r+=0.03
    return float(r)

def _pass_filters(app:Dict[str,Any],symbol:str,side:str,d:Dict[str,Any],relax:bool=False)->bool:
    reasons=[]
    try:
        entry=float(d.get("c5",0.0) or 0.0)
        atr=float(d.get("atr",0.0) or 0.0)
        if entry<=0 or atr<=0: reasons.append("no_entry_or_atr")
        if entry<LAST_MEME_PRICE_MIN: reasons.append("meme_price")
        ap=_atr_pct(entry,atr) if entry>0 else 0.0
        if ap<LAST_ATR_PCT_MIN or ap>LAST_ATR_PCT_MAX: reasons.append("atr_pct")
        sc=float(d.get("score",0.0) or 0.0)
        p=float(d.get("p_bayes",d.get("ml_p",0.55)) or 0.55)
        if sc<LAST_SCORE_MIN or p<LAST_P_MIN: reasons.append("score_p")
        adx=float(d.get("adx15",0.0) or 0.0)
        if adx<LAST_ADX_MIN: reasons.append("adx")
        _ensure_ob_fields(app,d,symbol)
        spreadZ=float(d.get("spread_norm",0.0) or 0.0)
        near=float(d.get("near_depth_ratio",d.get("ta_near_depth_ratio",0.0)) or 0.0)
        spoof=float(d.get("spoof_score",0.0) or 0.0)
        if spreadZ and spreadZ>LAST_SPREAD_Z_MAX: reasons.append("spread")
        if near and near<LAST_NEAR_DEPTH_MIN: reasons.append("near_depth")
        if spoof and spoof>LAST_SPOOF_MAX: reasons.append("spoof")
        if _recent_impulse(app,symbol,atr,near): reasons.append("impulse")
        sl=float(d.get("sl",0.0) or 0.0)
        tps=[float(x) for x in (d.get("tps") or [])]
        if entry>0 and sl>0 and tps:
            rr1=_rr(entry,sl,tps[0]); rr2=_rr(entry,sl,tps[1] if len(tps)>1 else tps[0])
            if rr1<LAST_MIN_RR1 or rr2<LAST_MIN_RR2: reasons.append("rr")
        if LAST_ANTI_BTC:
            corr_b,tr_b,corr_e,tr_e=_btc_eth_corr_trend(app,symbol)
            if corr_b is not None and tr_b is not None:
                if abs(corr_b)>=LAST_CORR_THRESH and ((tr_b>0 and side=="SHORT") or (tr_b<0 and side=="LONG")):
                    reasons.append("anti_btc")
            if corr_e is not None and tr_e is not None:
                if abs(corr_e)>=LAST_CORR_THRESH and ((tr_e>0 and side=="SHORT") or (tr_e<0 and side=="LONG")):
                    reasons.append("anti_eth")
        if LAST_KILLZONES:
            h=app["now_msk"]().hour
            if h in (3,4,5,23): reasons.append("killzone")
        if reasons:
            _log_reject(app,symbol,side,",".join(reasons),{"score":d.get("score"),"p":d.get("p_bayes",d.get("ml_p")), "adx":d.get("adx15"),"spreadZ":spreadZ,"near":near,"spoof":spoof},relax)
            return False
        return True
    except Exception:
        _log_reject(app,symbol,side,"exception",{},relax)
        return False

def _micro_adapt_ladder(app:Dict[str,Any],symbol:str,entry:float,atr:float,ladder:List[float],side:str)->List[float]:
    try:
        m=app.get("market"); df5=m.fetch_ohlcv(symbol,"5m",240)
        ofi,vpin=_vpin_ofi(df5)
        adj=1.0
        if vpin is not None and vpin>0.5: adj*=0.9
        if ofi is not None:
            if side=="LONG" and ofi<0: adj*=0.93
            if side=="SHORT" and ofi>0: adj*=0.93
        return [max(0.5,min(3.2,x*adj)) for x in ladder[:3]]
    except Exception:
        return ladder[:3]

def _build_lowprice_levels(app: Dict[str,Any], symbol: str, side: str, entry: float, df15, atr: float) -> Tuple[float, List[float]]:
    m = app.get("market")
    tick_real = 0.0
    with contextlib.suppress(Exception):
        tick_real = float(m.get_tick_size(symbol) or 0.0)
    safe_tick = _infer_safe_tick(entry, df15)
    tick = min(tick_real or safe_tick, safe_tick)
    if side=="LONG":
        sl = entry * (1.0 - LAST_LOW_SL_PCT)
        sl = _round_tick(_avoid_round_levels(entry, sl, tick, side), tick, "floor")
    else:
        sl = entry * (1.0 + LAST_LOW_SL_PCT)
        sl = _round_tick(_avoid_round_levels(entry, sl, tick, side), tick, "ceil")
    if side=="LONG":
        tps = [entry*(1.0+p) for p in LAST_LOW_TP_PCTS[:3]]
        tps = [_round_tick(_avoid_round_levels(entry,tp,tick,side),tick,"ceil") for tp in tps]
    else:
        tps = [entry*(1.0-p) for p in LAST_LOW_TP_PCTS[:3]]
        tps = [_round_tick(_avoid_round_levels(entry,tp,tick,side),tick,"floor") for tp in tps]
    pd = _compute_pdh_pdl(df15)
    capped = []
    for tp in tps:
        tp1=_cap_tp(entry,tp,atr,side)
        tp2=_cap_by_pdr(entry,tp1,side,pd)
        capped.append(tp2)
    tps = capped
    tps = _ensure_tp_monotonic_with_step(side, entry, tps, atr if atr>0 else (entry*0.01), tick, LAST_TP_MIN_STEP_ATR)
    floor = max(tick*5, 1e-12)
    for i,tp in enumerate(tps):
        if tp<=0: tps[i]=floor
    return float(sl), [float(x) for x in tps[:3]]

def _build_scalp_levels(app:Dict[str,Any],symbol:str,side:str,entry:float,atr:float,df15,df1h)->Tuple[float,List[float]]:
    if entry <= LAST_LOW_PRICE_THRESHOLD:
        return _build_lowprice_levels(app, symbol, side, entry, df15, atr)
    cfg=app.setdefault("_last_cfg",{"k_atr":LAST_K_STOP_ATR,"ladder":{}})
    k=float(cfg.get("k_atr",LAST_K_STOP_ATR))
    lad_map=cfg.get("ladder",{})
    ladder=list(LAST_R_LADDER)
    if symbol in lad_map and isinstance(lad_map[symbol],(list,tuple)) and len(lad_map[symbol])>=3:
        ladder=[float(lad_map[symbol][0]),float(lad_map[symbol][1]),float(lad_map[symbol][2])]
    ladder=_micro_adapt_ladder(app,symbol,entry,atr,ladder,side)
    m=app.get("market"); tick_real=0.0
    with contextlib.suppress(Exception):
        tick_real=float(m.get_tick_size(symbol) or 0.0)
    safe_tick=_infer_safe_tick(entry,df15)
    tick=min(tick_real or safe_tick,safe_tick)
    risk_abs=max(LAST_SL_MIN_ATR,min(k,LAST_SL_MAX_ATR))*max(atr,1e-9)
    if side=="LONG": sl=_round_tick(entry-risk_abs,tick,"floor")
    else: sl=_round_tick(entry+risk_abs,tick,"ceil")
    tps=[]
    for r in ladder[:3]:
        if side=="LONG": tps.append(entry+r*risk_abs)
        else: tps.append(entry-r*risk_abs)
    pd=_compute_pdh_pdl(df15)
    tps=[_cap_by_pdr(entry,_cap_tp(entry,tp,atr,side),side,pd) for tp in tps]
    if df1h is None:
        with contextlib.suppress(Exception):
            df1h = m.fetch_ohlcv(symbol,"1h",420)
    tps = _snap_targets_to_ta(entry, side, tps, df15, df1h, atr, tick)
    return float(sl),[float(x) for x in tps[:3]]

def _sanitize_levels(app:Dict[str,Any],symbol:str,side:str,entry:float,sl:float,tps:List[float],atr:float,df15)->Tuple[float,List[float]]:
    m=app.get("market"); tick_real=0.0
    with contextlib.suppress(Exception):
        tick_real=float(m.get_tick_size(symbol) or 0.0)
    tick=min(_infer_safe_tick(entry,df15),tick_real or 1e9)
    min_r=LAST_SL_MIN_ATR*atr; max_r=LAST_SL_MAX_ATR*atr
    if side=="LONG":
        if not (sl<entry): sl=entry-min_r
        risk=abs(entry-sl)
        if risk<min_r: sl=entry-min_r
        if risk>max_r: sl=entry-max_r
        sl=_round_tick(_avoid_round_levels(entry,sl,tick,side),tick,"floor")
    else:
        if not (sl>entry): sl=entry+min_r
        risk=abs(entry-sl)
        if risk<min_r: sl=entry+min_r
        if risk>max_r: sl=entry+max_r
        sl=_round_tick(_avoid_round_levels(entry,sl,tick,side),tick,"ceil")
    tps=tps[:3] if tps else []
    bad = (not tps) or (len({round(x,10) for x in tps})<3) or any(x<=0 for x in tps)
    if bad:
        if entry <= LAST_LOW_PRICE_THRESHOLD:
            sl,tps=_build_lowprice_levels(app,symbol,side,entry,df15,atr)
        else:
            df1h=None
            with contextlib.suppress(Exception):
                df1h = m.fetch_ohlcv(symbol,"1h",420)
            sl,tps=_build_scalp_levels(app,symbol,side,entry,atr,df15,df1h)
    tps=_ensure_tp_monotonic_with_step(side,entry,tps,atr,tick,LAST_TP_MIN_STEP_ATR)
    floor = max(tick*5, 1e-12)
    for i,tp in enumerate(tps):
        if tp<=0: tps[i]=floor
    if side=="LONG":
        tps=[_round_tick(_avoid_round_levels(entry,x,tick,side),tick,"ceil") for x in tps]
    else:
        tps=[_round_tick(_avoid_round_levels(entry,x,tick,side),tick,"floor") for x in tps]
    if len({round(x,10) for x in tps})<3:
        step_tick=max(tick,1e-12)*2.0
        if side=="LONG":
            tps=sorted(tps)
            for i in range(1,len(tps)):
                if tps[i]<=tps[i-1]: tps[i]=_round_tick(tps[i-1]+step_tick,tick,"ceil")
        else:
            tps=sorted(tps,reverse=True)
            for i in range(1,len(tps)):
                if tps[i]>=tps[i-1]: tps[i]=_round_tick(tps[i-1]-step_tick,tick,"floor")
    if side=="LONG":
        if not (all(tp>entry for tp in tps) and sl<entry):
            df1h=None
            with contextlib.suppress(Exception): df1h = m.fetch_ohlcv(symbol,"1h",420)
            sl,tps=_build_scalp_levels(app,symbol,side,entry,atr,df15,df1h)
    else:
        if not (all(tp<entry for tp in tps) and sl>entry):
            df1h=None
            with contextlib.suppress(Exception): df1h = m.fetch_ohlcv(symbol,"1h",420)
            sl,tps=_build_scalp_levels(app,symbol,side,entry,atr,df15,df1h)
    return float(sl),[float(x) for x in tps[:3]]

def _suggest_leverage(entry:float,sl:float)->int:
    try:
        risk_pct=abs(entry-sl)/(entry+1e-9)*100.0
        if risk_pct<=0.6:return min(LAST_MAX_LEV,20)
        if risk_pct<=0.9:return min(LAST_MAX_LEV,15)
        if risk_pct<=1.2:return min(LAST_MAX_LEV,12)
        if risk_pct<=1.8:return min(LAST_MAX_LEV,10)
        return max(LAST_MIN_LEV,7)
    except Exception:
        return min(LAST_MAX_LEV,10)

async def _pick_best_scalp(app:Dict[str,Any])->Optional[Tuple[str,str,Dict[str,Any]]]:
    symbols=app.get("SYMBOLS",[]) or []
    score_fn=app.get("score_symbol_core")
    EXECUTOR=app.get("EXECUTOR")
    if not (symbols and score_fn): return None
    loop=asyncio.get_running_loop()
    tasks=[loop.run_in_executor(EXECUTOR,score_fn,s,False) for s in symbols]
    results=await asyncio.gather(*tasks,return_exceptions=True)
    cands=[]
    for s,res in zip(symbols,results):
        if not (isinstance(res,tuple) and len(res)==3): continue
        score,side,d=res
        d=dict(d or {})
        d["score"]=float(score)
        if not _pass_filters(app,s,side,d,relax=False): continue
        rk=_rank(app,d)
        cands.append((rk,(s,side,d)))
    if not cands:
        tasks=[loop.run_in_executor(EXECUTOR,score_fn,s,True) for s in symbols]
        results=await asyncio.gather(*tasks,return_exceptions=True)
        for s,res in zip(symbols,results):
            if not (isinstance(res,tuple) and len(res)==3): continue
            score,side,d=res
            d=dict(d or {})
            d["score"]=float(score)
            if not _pass_filters(app,s,side,d,relax=True): continue
            rk=_rank(app,d)
            cands.append((rk,(s,side,d)))
    if not cands: return None
    cands.sort(key=lambda x:x[0],reverse=True)
    top=cands[:LAST_TOPK]
    return top[0][1] if top else None

def _format_scalp(sig,fmt_price)->str:
    base=sig.symbol.split("/")[0]
    tps_fmt=" / ".join(fmt_price(x) for x in sig.tps)
    lines=[f"{sig.side.title()}  {base}/USDT",
           f"Плечо: {sig.leverage}x",
           f"Вход: {fmt_price(sig.entry)} / по рынку",
           f"Тейки: {tps_fmt}",
           f"Стоп: {fmt_price(sig.sl)}"]
    return "\n".join(lines)+"\n\n⚠️ Не является финансовой рекомендацией. Торгуйте ответственно."

async def _handle_scalp(app:Dict[str,Any],message,bot):
    guard_access=app.get("guard_access")
    db=app.get("db")
    fmt_price=app.get("format_price") or (lambda v:f"{v:.4f}")
    Signal=app.get("Signal")
    format_signal_message=app.get("format_signal_message")
    watch_signal_price=app.get("watch_signal_price")
    st=await guard_access(message,bot)
    if not st:return
    DAILY_LIMIT=int(app.get("DAILY_LIMIT",3))
    if not st.get("unlimited") and int(st.get("count",0))>=DAILY_LIMIT:
        with contextlib.suppress(Exception): await message.answer("Лимит 3 сигнала в день исчерпан. Введите /code для безлимита.")
        return
    working=await message.answer("🔎 Ищу лучший скальп‑сетап...")
    try:
        pick=await _pick_best_scalp(app)
        if not pick:
            with contextlib.suppress(Exception): await working.edit_text("Сейчас нет надёжного скальп‑сетапа. Попробуйте позже.")
            return
        symbol,side,d=pick
        entry=float(d.get("c5",0.0) or 0.0)
        atr=float(d.get("atr",0.0) or 0.0)
        df15=app.get("market").fetch_ohlcv(symbol,"15m",240)
        df1h=app.get("market").fetch_ohlcv(symbol,"1h",420)
        sl0,tps0=_build_scalp_levels(app,symbol,side,entry,atr,df15,df1h)
        sl,tps=_sanitize_levels(app,symbol,side,entry,sl0,tps0,atr,df15)
        lev=_suggest_leverage(entry,sl)
        risk_level=int(min(9,max(4,round((_atr_pct(entry,atr))/0.2))))
        reason_fn=app.get("build_reason"); reason=""
        with contextlib.suppress(Exception): reason=reason_fn(d) if callable(reason_fn) else ""
        sig=Signal(user_id=message.from_user.id,symbol=symbol,side=side,entry=entry,tps=tps,sl=sl,leverage=lev,risk_level=risk_level,created_at=app["now_msk"](),news_note=d.get("news_note",""),atr_value=atr,watch_until=app["now_msk"]()+timedelta(hours=4),reason=reason)
        exist=await db.get_active_signals_for_user(message.from_user.id)
        if any(s.symbol==sig.symbol and s.side==sig.side and s.active for s in exist):
            with contextlib.suppress(Exception): await working.edit_text("У вас уже есть активный сигнал по этой паре и стороне. Попробуйте позже.")
            return
        if LAST_SCALP_FORMAT:
            text=_format_scalp(sig,fmt_price)
            await working.edit_text(text)
        else:
            text=format_signal_message(sig)
            await working.edit_text(text)
        st["count"]=int(st.get("count",0))+1
        await db.save_user_state(message.from_user.id,st)
        sig.id=await db.add_signal(sig)
        task=asyncio.create_task(watch_signal_price(bot,message.chat.id,sig))
        app.get("active_watch_tasks",{}).setdefault(message.from_user.id,[]).append(task)
    except Exception:
        with contextlib.suppress(Exception): await working.edit_text("⚠️ Ошибка при поиске скальп‑сетапа. Попробуйте позже.")

async def _scalp_scan_loop(app:Dict[str,Any]):
    if not LAST_SCAN_ENABLE:return
    await asyncio.sleep(5)
    await _ensure_last_tables(app)
    bot=app.get("bot_instance")
    if not bot:return
    ch=LAST_EXECUTE_ON_CH
    if not ch:return
    while True:
        try:
            cap=await _get_daily_cap(app)
            if cap<LAST_CHANNEL_DAILY_CAP:
                pick=await _pick_best_scalp(app)
                if pick:
                    symbol,side,d=pick
                    if _can_post_portf(app,side) and _opp_cooldown_ok(app,symbol,side):
                        entry=float(d.get("c5",0.0) or 0.0)
                        atr=float(d.get("atr",0.0) or 0.0)
                        df15=app.get("market").fetch_ohlcv(symbol,"15m",240)
                        df1h=app.get("market").fetch_ohlcv(symbol,"1h",420)
                        sl0,tps0=_build_scalp_levels(app,symbol,side,entry,atr,df15,df1h)
                        sl,tps=_sanitize_levels(app,symbol,side,entry,sl0,tps0,atr,df15)
                        lev=_suggest_leverage(entry,sl)
                        fmt_price=app.get("format_price") or (lambda v:f"{v:.4f}")
                        msg=_format_scalp(type("S",(),{"symbol":symbol,"side":side,"tps":tps,"entry":entry,"sl":sl}),fmt_price)
                        with contextlib.suppress(Exception): await bot.send_message(ch,msg,disable_web_page_preview=True)
                        await _inc_daily_cap(app); _append_posted(app,symbol,side)
            await asyncio.sleep(max(15,LAST_SCAN_INTERVAL_SEC))
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(10)

async def _calibrate_last_loop(app:Dict[str,Any]):
    if not LAST_CALIB_ENABLE:return
    await asyncio.sleep(10)
    db=app.get("db")
    if not db or not getattr(db,"conn",None):return
    while True:
        try:
            since=(app["now_msk"]()-timedelta(hours=48)).isoformat()
            cur=await db.conn.execute("""
                SELECT outcome, rr1, rr2, symbol
                FROM outcomes
                WHERE finished_at IS NOT NULL AND finished_at >= ?
            """,(since,))
            rows=await cur.fetchall()
            stats={}; wins=losses=0
            for r in rows or []:
                sym=str(r["symbol"] or "")
                out=str(r["outcome"]).upper()
                rr1=float(r["rr1"] or 0.0)
                rr2=float(r["rr2"] or 0.0)
                if sym not in stats: stats[sym]={"cnt":0,"r1":0.0,"r2":0.0}
                stats[sym]["cnt"]+=1; stats[sym]["r1"]+=rr1; stats[sym]["r2"]+=rr2
                if out in ("TP1","TP2","TP3","BE"): wins+=1
                if out in ("STOP","TIME"): losses+=1
            cfg=app.setdefault("_last_cfg",{"k_atr":LAST_K_STOP_ATR,"ladder":{}})
            total=max(1,wins+losses); stop_rate=losses/total
            if stop_rate>0.5: cfg["k_atr"]=min(LAST_K_MAX_ATR,cfg.get("k_atr",LAST_K_STOP_ATR)*1.05)
            elif wins/total>0.6: cfg["k_atr"]=max(LAST_K_MIN_ATR,cfg.get("k_atr",LAST_K_STOP_ATR)*0.95)
            lad=cfg.get("ladder",{})
            for sym,agg in stats.items():
                if agg["cnt"]<3: continue
                r1=agg["r1"]/max(1,agg["cnt"])
                r2=agg["r2"]/max(1,agg["cnt"])
                m1=max(0.6,min(1.2,r1*0.9))
                m2=max(1.2,min(2.2,r2*0.9))
                m3=min(3.0,max(m2+0.6,2.4))
                lad[sym]=(float(m1),float(m2),float(m3))
            cfg["ladder"]=lad
        except Exception:
            pass
        await asyncio.sleep(max(3600,LAST_CALIB_INTERVAL_SEC))

def patch(app:Dict[str,Any])->None:
    logger=app.get("logger")
    orig_on_startup = app.get("on_startup")
    async def _on_startup_last(bot):
        asyncio.create_task(_scalp_scan_loop(app))
        asyncio.create_task(_calibrate_last_loop(app))
        if orig_on_startup:
            await orig_on_startup(bot)
    app["on_startup"] = _on_startup_last
    logger and logger.info("last.py: patch applied (start loops in on_startup; TA‑snap TP; low‑price mode; no UI).")
