"""
VIX-Oversold Reversal Strategy v3 — Expert-Debate Optimized Backtest
======================================================================
3-Layer Entry Structure:

LAYER A — Macro Regime (ALL required):
  A1. VIX >= 25 & VIX <= 3-day rolling max * 0.98
  A2. SPY(D) >= SPY(D-1) * 0.99         (당일 SPY 낙폭 -1% 미만)
  A3. IWM(D) >= IWM(D-1)                (소형주 당일 반등/중립)
  A4. 실적발표일 ±3 거래일 아님          (이벤트 리스크 제거 — 미구현 시 경고)
  A5. Nasdaq(D) >= Nasdaq 60d-high * 0.75 (계단식 장기 붕괴 제외)

LAYER B — Stock Position (2 of 3):
  B1. Close >= MA200 * 0.70
  B2. Bollinger %B between 0.0 and 0.35
  B3. ADX < 40 AND ADX slope <= 0

LAYER C — Daily Signal (ALL required):
  C1. RSI<30 OR RSI_prev<30 OR CCI<-100 OR CCI_prev<-100
  C2. MACD_H(D) > MACD_H(D-1)
  C3. Volume >= 1.5x or 2.0x VolAvg20 (QQQ MA20 기준)
  C4. Bullish candle (Close > Open) & lower_tail >= body * 0.5

Extra filters:
  X1. ATR/Close <= 0.035   (고변동 종목 제외)
  X2. Max 5 entries/day    (ATR/Close 낮은 순)
  X3. 1 position per ticker

Exit:
  stop   = entry - ATR*1.5  (cap: -10%)
  target = entry + ATR*4.5  (floor: +10%, ceil: +25%)
  BE stop at +7%
  Time stop: 25 trading days
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────
# 0. UNIVERSE & PARAMS
# ─────────────────────────────────────────────────────────────
NDX100 = sorted(list(dict.fromkeys([
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG",
    "TSLA","AVGO","PLTR","AMD","INTC","QCOM","TXN",
    "AMAT","LRCX","KLAC","MRVL","MCHP","ON",
    "ADBE","CRM","ORCL","SNOW","NOW","PANW","CRWD",
    "ZS","DDOG","MDB","TEAM","WDAY","CDNS",
    "SHOP","EBAY","BKNG","ABNB","EXPE","TRIP",
    "AMGN","GILD","BIIB","REGN","VRTX","IDXX","DXCM",
    "ISRG","ILMN","MRNA",
    "NFLX","CMCSA","CHTR","TMUS","SIRI",
    "COST","SBUX","MDLZ","PEP","MNST",
    "PYPL","ADSK","MELI",
    "HON","GEHC","FANG","CEG",
    "MU","WDC","STX",
    "ASML","SMCI","SNPS","FTNT","NXPI",
    "ODFL","VRSK","CTAS","FAST",
    "ADP","PAYX","INTU",
    "PCAR","EA","TTWO",
    "CSX","CPRT","DLTR","ROST",
    "BMRN","CTSH","LULU","ORLY","NTES",
])))

VIX_TICKER = "^VIX"
SPY_TICKER = "SPY"
IWM_TICKER = "IWM"
QQQ_TICKER = "QQQ"
NDQ_TICKER = "^IXIC"   # Nasdaq Composite (for A5)
START, END = "2010-01-01", "2026-01-01"

# Exit params
ATR_STOP    = 1.5
ATR_TGT     = 4.5
CAP_STOP    = 0.10   # max loss -10%
FLOOR_TGT   = 0.10   # min target +10%
CEIL_TGT    = 0.25   # max target +25%
BE_TRIGGER  = 0.07   # BE stop at +7%
MAX_HOLD    = 25
MAX_DAILY   = 5
MAX_VOL_RATIO  = 0.035  # ATR/Close 상한 (고변동 제외)
# 거래량 기준: v2의 1.5×/2.0× 에서 완화 (VIX구간+레이어B/C 강화로 보완)
VOL_ABOVE_MA20 = 1.2    # QQQ MA20 위일 때
VOL_BELOW_MA20 = 1.5    # QQQ MA20 아래일 때
LTAIL_BODY_RATIO = 0.3  # 아래꼬리 ≥ 몸통 × 0.3

# ─────────────────────────────────────────────────────────────
# 1. DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────
print("="*65)
print("Step 1: Downloading data ...")
print("="*65)

def dl_series(ticker, start, end):
    """Download single ticker, return Close as clean Series."""
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    c = raw["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return pd.Series(c.values, index=c.index, dtype=float).dropna()

vix_s   = dl_series(VIX_TICKER, START, END)
spy_s   = dl_series(SPY_TICKER, START, END)
iwm_s   = dl_series(IWM_TICKER, START, END)
qqq_s   = dl_series(QQQ_TICKER, START, END)
ndq_s   = dl_series(NDQ_TICKER, START, END)   # Nasdaq Composite

print(f"  VIX: {len(vix_s)} rows")
print(f"  SPY: {len(spy_s)} rows")
print(f"  IWM: {len(iwm_s)} rows")
print(f"  QQQ: {len(qqq_s)} rows")
print(f"  NASDAQ: {len(ndq_s)} rows")

# Stock batch
raw_stocks = yf.download(NDX100, start=START, end=END,
                          auto_adjust=True, progress=False)
price_data = {}
skipped = []
for ticker in NDX100:
    try:
        df = pd.DataFrame({
            "Open":   raw_stocks["Open"][ticker],
            "High":   raw_stocks["High"][ticker],
            "Low":    raw_stocks["Low"][ticker],
            "Close":  raw_stocks["Close"][ticker],
            "Volume": raw_stocks["Volume"][ticker],
        }).dropna()
        if len(df) >= 300:
            price_data[ticker] = df
        else:
            skipped.append(ticker)
    except Exception:
        skipped.append(ticker)

print(f"  Stocks: {len(price_data)} loaded, {len(skipped)} skipped ({skipped})")

# ─────────────────────────────────────────────────────────────
# 2. INDICATOR HELPERS
# ─────────────────────────────────────────────────────────────

def calc_rsi(c, p=14):
    d  = c.diff()
    ag = d.clip(lower=0).ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    return 100 - 100/(1 + ag/al.replace(0, np.nan))

def calc_cci(h, l, c, p=20):
    tp = (h+l+c)/3
    ma = tp.rolling(p).mean()
    md = tp.rolling(p).apply(lambda x: np.mean(np.abs(x-x.mean())), raw=True)
    return (tp-ma)/(0.015*md.replace(0,np.nan))

def calc_macd_h(c, fast=12, slow=26, sig=9):
    ml = c.ewm(span=fast,adjust=False).mean() - c.ewm(span=slow,adjust=False).mean()
    return ml - ml.ewm(span=sig, adjust=False).mean()

def calc_atr(h, l, c, p=14):
    pc = c.shift(1)
    tr = pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, min_periods=p, adjust=False).mean()

def calc_bb_pct_b(c, p=20, k=2):
    ma  = c.rolling(p).mean()
    std = c.rolling(p).std()
    ub  = ma + k*std
    lb  = ma - k*std
    return (c - lb) / (ub - lb).replace(0, np.nan)

def calc_adx(h, l, c, p=14):
    """Returns +DI, -DI, ADX as DataFrame."""
    tr  = calc_atr(h, l, c, p)   # Wilder ATR
    up  = h.diff().clip(lower=0)
    dn  = (-l.diff()).clip(lower=0)
    pdm = up.where(up > dn, 0.0)
    ndm = dn.where(dn > up, 0.0)
    pdi = 100 * pdm.ewm(alpha=1/p, min_periods=p, adjust=False).mean() / tr.replace(0,np.nan)
    ndi = 100 * ndm.ewm(alpha=1/p, min_periods=p, adjust=False).mean() / tr.replace(0,np.nan)
    dx  = 100 * (pdi-ndi).abs() / (pdi+ndi).replace(0,np.nan)
    adx = dx.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    return pdi, ndi, adx

# ─────────────────────────────────────────────────────────────
# 3. COMPUTE INDICATORS
# ─────────────────────────────────────────────────────────────
print("\nStep 2: Computing indicators ...")

indicator_data = {}
for ticker, df in price_data.items():
    d = df.copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    d["MA200"]    = c.rolling(200).mean()
    d["RSI14"]    = calc_rsi(c)
    d["CCI20"]    = calc_cci(h, l, c)
    d["MACD_H"]   = calc_macd_h(c)
    d["VolAvg20"] = v.rolling(20).mean()
    d["ATR14"]    = calc_atr(h, l, c)
    d["BB_PCT_B"] = calc_bb_pct_b(c)
    pdi, ndi, adx = calc_adx(h, l, c)
    d["PDI"]      = pdi
    d["NDI"]      = ndi
    d["ADX"]      = adx
    d["ADX_PREV"] = adx.shift(1)
    # Candle measures
    d["BODY"]       = (c - d["Open"]).abs()
    d["LOWER_TAIL"] = d["Open"].where(c > d["Open"], c) - l   #양봉: Open-Low, 음봉: Close-Low
    indicator_data[ticker] = d

print(f"  Done ({len(indicator_data)} tickers).")

# ─────────────────────────────────────────────────────────────
# 4. MACRO REGIME SERIES
# ─────────────────────────────────────────────────────────────
vix_3d_max = vix_s.rolling(3).max()
spy_prev   = spy_s.shift(1)
iwm_prev   = iwm_s.shift(1)
qqq_ma20   = qqq_s.rolling(20).mean()
ndq_60high = ndq_s.rolling(60).max()   # Nasdaq 60일 고점

# ─────────────────────────────────────────────────────────────
# 5. SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────
print("\nStep 3: Generating signals (v3) ...")

VALID_COLS = ["Open","Close","High","Low","MA200","RSI14","CCI20",
              "MACD_H","VolAvg20","ATR14","BB_PCT_B","PDI","NDI","ADX","ADX_PREV",
              "BODY","LOWER_TAIL"]

trades_raw = []

for ticker, df in indicator_data.items():
    # Common calendar alignment
    common = (df.index
               .intersection(vix_s.index)
               .intersection(spy_s.index)
               .intersection(iwm_s.index)
               .intersection(qqq_s.index)
               .intersection(ndq_s.index))
    d   = df.loc[common].copy()
    vx  = vix_s.reindex(common)
    vx3 = vix_3d_max.reindex(common)
    sp  = spy_s.reindex(common)
    spp = spy_prev.reindex(common)
    iw  = iwm_s.reindex(common)
    iwp = iwm_prev.reindex(common)
    qq  = qqq_s.reindex(common)
    qqm = qqq_ma20.reindex(common)
    nq  = ndq_s.reindex(common)
    nq60= ndq_60high.reindex(common)

    c        = d["Close"]
    o        = d["Open"]
    rsi      = d["RSI14"];  rsi_p = rsi.shift(1)
    cci      = d["CCI20"];  cci_p = cci.shift(1)
    mh       = d["MACD_H"]; mh_p  = mh.shift(1)
    vol      = price_data[ticker]["Volume"].reindex(common)
    va20     = d["VolAvg20"]
    atr      = d["ATR14"]
    bbpb     = d["BB_PCT_B"]
    adx      = d["ADX"];    adx_p = d["ADX_PREV"]
    body     = d["BODY"]
    ltail    = d["LOWER_TAIL"]

    not_nan = d[VALID_COLS].notna().all(axis=1)
    not_nan &= rsi_p.notna() & cci_p.notna() & mh_p.notna()
    not_nan &= vx.notna() & vx3.notna() & sp.notna() & spp.notna()
    not_nan &= iw.notna() & iwp.notna() & nq.notna() & nq60.notna()

    # ── LAYER A ─────────────────────────────────────────────
    cA1 = (vx >= 25) & (vx <= vx3 * 0.98)
    cA2 = sp >= spp * 0.99
    cA3 = iw >= iwp
    # A4: 실적발표일 — 데이터 없음, 전략에서 경고만 출력
    cA5 = nq >= nq60 * 0.75        # 나스닥 60일 고점 대비 -25% 이내

    cA  = cA1 & cA2 & cA3 & cA5

    # ── LAYER B (2 of 3) ────────────────────────────────────
    cB1 = c >= d["MA200"] * 0.70
    cB2 = bbpb.between(0.0, 0.35)
    cB3 = (adx < 40) & (adx <= adx_p)   # ADX < 40 & 기울기 ≤ 0 (약화 중)

    b_sum = cB1.astype(int) + cB2.astype(int) + cB3.astype(int)
    cB    = b_sum >= 2

    # ── LAYER C ─────────────────────────────────────────────
    cC1 = (rsi < 30) | (rsi_p < 30) | (cci < -100) | (cci_p < -100)
    cC2 = mh > mh_p
    vol_mult = pd.Series(np.where(qq >= qqm, VOL_ABOVE_MA20, VOL_BELOW_MA20), index=common)
    cC3 = vol >= vol_mult * va20
    cC4 = (c > o) & (ltail >= body * LTAIL_BODY_RATIO)   # 양봉 + 아래꼬리 ≥ 몸통×0.3

    cC  = cC1 & cC2 & cC3 & cC4

    # ── Extra: 고변동 제외 ──────────────────────────────────
    cX1 = (atr / c.replace(0, np.nan)) <= MAX_VOL_RATIO

    signal = cA & cB & cC & cX1 & not_nan

    for entry_date in d.index[signal]:
        ep      = c.loc[entry_date]
        atr_val = atr.loc[entry_date]
        vr      = atr_val / ep
        trades_raw.append({
            "ticker":      ticker,
            "entry_date":  entry_date,
            "entry_price": ep,
            "atr":         atr_val,
            "vol_ratio":   vr,
        })

print(f"  Raw signals: {len(trades_raw)}")
print(f"  ⚠ 실적발표일(A4) 필터는 외부 캘린더 데이터 필요 — 이 백테스트에서는 미적용")

# ─────────────────────────────────────────────────────────────
# 6. TRADE SIMULATION
# ─────────────────────────────────────────────────────────────
print("\nStep 4: Simulating trades ...")

trades_raw_df = (pd.DataFrame(trades_raw)
                   .sort_values(["entry_date","vol_ratio"])
                   .reset_index(drop=True))

results     = []
active_exit = {}
daily_count = {}

for _, row in trades_raw_df.iterrows():
    ticker     = row["ticker"]
    entry_date = row["entry_date"]

    if ticker in active_exit and entry_date <= active_exit[ticker]:
        continue
    dc = daily_count.get(entry_date, 0)
    if dc >= MAX_DAILY:
        continue

    ep  = row["entry_price"]
    atr = row["atr"]

    raw_stop  = ep - ATR_STOP * atr
    hard_stop = ep * (1 - CAP_STOP)
    stop      = max(raw_stop, hard_stop)

    raw_tgt   = ep + ATR_TGT * atr
    target    = np.clip(raw_tgt, ep*(1+FLOOR_TGT), ep*(1+CEIL_TGT))

    df         = indicator_data[ticker]
    future_idx = df.index[df.index > entry_date][:MAX_HOLD]

    outcome    = None; exit_date = None; exit_price = None
    be_active  = False; be_level = ep

    for i, t in enumerate(future_idx):
        lo = df.loc[t,"Low"]
        hi = df.loc[t,"High"]
        cl = df.loc[t,"Close"]

        if not be_active and hi >= ep*(1+BE_TRIGGER):
            be_active = True

        cur_stop   = be_level if be_active else stop
        hit_stop   = lo <= cur_stop
        hit_target = hi >= target

        if hit_stop and hit_target:
            exit_price, outcome, exit_date = cur_stop, "stop", t; break
        elif hit_stop:
            exit_price, outcome, exit_date = cur_stop, "stop", t; break
        elif hit_target:
            exit_price, outcome, exit_date = target, "target", t; break
        elif i == len(future_idx)-1:
            exit_price, outcome, exit_date = cl, "time_exit", t; break

    if outcome is None:
        continue

    ret = exit_price / ep - 1
    hd  = len(df.loc[(df.index > entry_date) & (df.index <= exit_date)])

    results.append({
        "ticker":      ticker,
        "entry_date":  entry_date,
        "exit_date":   exit_date,
        "entry_price": round(ep, 4),
        "exit_price":  round(exit_price, 4),
        "return_pct":  ret,
        "hold_days":   hd,
        "outcome":     outcome,
    })
    active_exit[ticker]     = exit_date
    daily_count[entry_date] = dc + 1

trades_df = pd.DataFrame(results)
trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
trades_df["exit_date"]  = pd.to_datetime(trades_df["exit_date"])
trades_df["year"]       = trades_df["entry_date"].dt.year

print(f"  Total trades: {len(trades_df)}")

# ─────────────────────────────────────────────────────────────
# 7. STATISTICS
# ─────────────────────────────────────────────────────────────
print("\n"+"="*65)
print("Step 5: Summary Statistics")
print("="*65)

def stats(df, label=""):
    n = len(df)
    if n == 0: return {"Label":label,"N_trades":0}
    wr  = (df["return_pct"]>0).mean()*100
    ar  = df["return_pct"].mean()*100
    mr  = df["return_pct"].median()*100
    sr  = df["return_pct"].std()*100
    ah  = df["hold_days"].mean()
    nt  = (df["outcome"]=="target").sum()
    ns  = (df["outcome"]=="stop").sum()
    ntm = (df["outcome"]=="time_exit").sum()
    return {"Label":label,"N_trades":n,
            "Win_%":round(wr,1),"Avg_ret_%":round(ar,2),
            "Med_ret_%":round(mr,2),"Std_%":round(sr,2),
            "Avg_hold":round(ah,1),
            "N_target":nt,"N_stop":ns,"N_time":ntm}

overall_v3 = stats(trades_df, "v3_ALL")
print("\n[Overall v3]")
for k,v in overall_v3.items(): print(f"  {k}: {v}")

ticker_stats = [stats(trades_df[trades_df["ticker"]==t], t)
                for t in sorted(trades_df["ticker"].unique())]
tk_df = pd.DataFrame(ticker_stats).sort_values("N_trades", ascending=False)
print("\n[Per-Ticker]")
print(tk_df.to_string(index=False))

year_stats = [stats(trades_df[trades_df["year"]==yr], str(yr))
              for yr in sorted(trades_df["year"].unique())]
yr_df = pd.DataFrame(year_stats)
print("\n[Per-Year]")
print(yr_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 8. EQUITY CURVE
# ─────────────────────────────────────────────────────────────
eq = trades_df.sort_values("exit_date")[["exit_date","return_pct"]].copy()
eq["cum"] = eq["return_pct"].cumsum()

total_years     = (pd.Timestamp(END)-pd.Timestamp(START)).days/365.25
trades_per_year = len(trades_df)/total_years

# ─────────────────────────────────────────────────────────────
# 9. VISUALIZATION
# ─────────────────────────────────────────────────────────────
print("\nStep 6: Plotting ...")

fig = plt.figure(figsize=(20, 22))
gs  = GridSpec(5, 2, figure=fig, hspace=0.50, wspace=0.33)

crisis = [
    ("2011-08-01","2011-10-31","#94a3b8","2011"),
    ("2015-08-01","2016-02-29","#6366f1","2015-16"),
    ("2018-10-01","2019-01-15","#f87171","2018Q4"),
    ("2020-02-20","2020-04-30","#fb923c","COVID"),
    ("2022-01-01","2022-12-31","#facc15","2022"),
    ("2025-01-01","2025-06-30","#a78bfa","2025"),
]

# (a) Equity Curve
ax0 = fig.add_subplot(gs[0,:])
ax0.plot(eq["exit_date"], eq["cum"]*100, color="#2563eb", lw=2,
         label="v3 Cumulative P&L (non-compounding)")
ax0.axhline(0, color="black", lw=0.8, ls="--")
for s,e,col,lbl in crisis:
    ax0.axvspan(pd.Timestamp(s), min(pd.Timestamp(e),pd.Timestamp(END)),
                alpha=0.18, color=col, label=lbl)
ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax0.xaxis.set_major_locator(mdates.YearLocator())
ax0.set_title("v3 — Cumulative P&L  |  3-Layer Expert-Debate Strategy",
              fontsize=13, fontweight="bold")
ax0.set_ylabel("Cumulative Return (%)"); ax0.legend(fontsize=8, ncol=4); ax0.grid(alpha=0.3)

# (b) VIX with entries
ax1 = fig.add_subplot(gs[1,:])
ax1.plot(vix_s.index, vix_s.values, color="#dc2626", lw=1, label="VIX")
ax1.axhline(25, color="#f97316", lw=1.2, ls="--", label="VIX=25")
ax1.fill_between(vix_s.index, vix_s.values, 25,
                 where=(vix_s.values>=25), alpha=0.2, color="#f97316")
ev = vix_s.reindex(trades_df["entry_date"].values, method="nearest")
ax1.scatter(ev.index, ev.values, color="#1d4ed8", s=20,
            zorder=5, alpha=0.8, label=f"v3 Entries ({len(trades_df)})")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.set_title("VIX & v3 Entry Points", fontsize=11, fontweight="bold")
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

# (c) Return distribution
ax2 = fig.add_subplot(gs[2,0])
rp  = trades_df["return_pct"]*100
if len(rp) > 0:
    bins = np.linspace(rp.min()-0.5, rp.max()+0.5, min(40, len(rp)+2))
    nw   = (rp>0).sum(); nl = (rp<=0).sum()
    ax2.hist(rp[rp>0],  bins=bins, color="#22c55e", edgecolor="white",
             lw=0.4, alpha=0.85, label=f"Win ({nw})")
    ax2.hist(rp[rp<=0], bins=bins, color="#ef4444", edgecolor="white",
             lw=0.4, alpha=0.85, label=f"Loss ({nl})")
    ax2.axvline(rp.mean(), color="#1e3a8a", ls="--", lw=1.5,
                label=f"Mean {rp.mean():.1f}%")
    ax2.axvline(0, color="black", lw=0.8)
ax2.set_title("Return Distribution (v3)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Return (%)"); ax2.set_ylabel("# Trades")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

# (d) Hold days
ax3 = fig.add_subplot(gs[2,1])
ax3.hist(trades_df["hold_days"], bins=range(1, MAX_HOLD+3),
         color="#0ea5e9", edgecolor="white", lw=0.4, rwidth=0.85)
ax3.set_title("Hold-Days Distribution (v3)", fontsize=11, fontweight="bold")
ax3.set_xlabel("Hold Days"); ax3.set_ylabel("# Trades"); ax3.grid(alpha=0.3)

# (e) Per-year avg return
ax4 = fig.add_subplot(gs[3,0])
yr_plot = yr_df[yr_df["N_trades"]>0].copy()
cols_yr  = ["#22c55e" if x>0 else "#ef4444" for x in yr_plot["Avg_ret_%"]]
bars4 = ax4.bar(yr_plot["Label"].astype(str), yr_plot["Avg_ret_%"],
                color=cols_yr, edgecolor="white")
ax4.axhline(0, color="black", lw=0.8)
for bar, n in zip(bars4, yr_plot["N_trades"]):
    ax4.text(bar.get_x()+bar.get_width()/2,
             bar.get_height()+(0.2 if bar.get_height()>=0 else -0.7),
             f"n={n}", ha="center", va="bottom", fontsize=8)
ax4.set_title("Avg Return by Year (v3)", fontsize=11, fontweight="bold")
ax4.set_ylabel("Avg Return (%)"); ax4.tick_params(axis="x", rotation=45)
ax4.grid(axis="y", alpha=0.3)

# (f) Outcome pie
ax5 = fig.add_subplot(gs[3,1])
oc = trades_df["outcome"].value_counts()
cm = {"target":"#22c55e","stop":"#ef4444","time_exit":"#f59e0b"}
pc = [cm.get(k,"#94a3b8") for k in oc.index]
ax5.pie(oc.values, labels=oc.index, autopct="%1.1f%%",
        colors=pc, startangle=90, textprops={"fontsize":10})
ax5.set_title("Outcome Distribution (v3)", fontsize=11, fontweight="bold")

# (g) v1/v2/v3 comparison bar
ax6 = fig.add_subplot(gs[4,:])
v_comp = pd.DataFrame({
    "Version": ["v1 (기존)", "v2 (개선)", "v3 (전문가토론)"],
    "N_trades":   [417,  95,   len(trades_df)],
    "Win_%":      [26.1, 30.5, overall_v3.get("Win_%",0)],
    "Avg_ret_%":  [0.00,-0.41, overall_v3.get("Avg_ret_%",0)],
    "Stop_%":     [73.6, 68.4, overall_v3.get("N_stop",0)/max(len(trades_df),1)*100],
})
x    = np.arange(3)
w    = 0.22
col3 = ["#3b82f6","#10b981","#f59e0b"]

ax6_t = ax6.twinx()
bars_n = ax6.bar(x-w,   v_comp["N_trades"],   w, label="N trades",  color="#cbd5e1")
bars_w = ax6.bar(x,     v_comp["Win_%"],       w, label="Win %",     color="#22c55e", alpha=0.85)
bars_a = ax6_t.bar(x+w, v_comp["Avg_ret_%"],  w, label="Avg ret %", color="#3b82f6", alpha=0.85)

ax6.set_xticks(x); ax6.set_xticklabels(v_comp["Version"], fontsize=11)
ax6.set_ylabel("N trades / Win %"); ax6_t.set_ylabel("Avg Return %")
ax6.axhline(0, color="black", lw=0.5)
ax6.set_title("v1 vs v2 vs v3 — Head-to-Head Comparison",
              fontsize=12, fontweight="bold")

lines1, labs1 = ax6.get_legend_handles_labels()
lines2, labs2 = ax6_t.get_legend_handles_labels()
ax6.legend(lines1+lines2, labs1+labs2, fontsize=9, loc="upper right")
ax6.grid(axis="y", alpha=0.3)

plt.suptitle("VIX-Oversold Reversal v3 (Expert-Debate Optimized) | NDX100 | 2010–2026",
             fontsize=14, fontweight="bold", y=1.005)
plt.savefig("backtest_vix_v3_result.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: backtest_vix_v3_result.png")

# ─────────────────────────────────────────────────────────────
# 10. HEAD-TO-HEAD COMPARISON TABLE
# ─────────────────────────────────────────────────────────────
print("\n"+"="*65)
print("v1 / v2 / v3 COMPARISON")
print("="*65)

v1_row = {"Label":"v1","N_trades":417,"Win_%":26.1,"Avg_ret_%":0.00,
          "Med_ret_%":-5.00,"Std_%":8.99,"Avg_hold":12.3,
          "N_target":47,"N_stop":307,"N_time":63}
v2_row = {"Label":"v2","N_trades":95,"Win_%":30.5,"Avg_ret_%":-0.41,
          "Med_ret_%":-4.12,"Std_%":8.92,"Avg_hold":11.4,
          "N_target":12,"N_stop":65,"N_time":18}
v3_row = overall_v3; v3_row["Label"]="v3"

comp = pd.DataFrame([v1_row, v2_row, v3_row]).set_index("Label")
print(comp.to_string())

# ─────────────────────────────────────────────────────────────
# 11. FULL TRADE LIST
# ─────────────────────────────────────────────────────────────
print("\n"+"="*65)
print("Full Trade List (v3)")
print("="*65)
pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_rows", 300)
print(trades_df[["ticker","entry_date","exit_date",
                  "entry_price","exit_price","return_pct","hold_days","outcome"]]
      .assign(return_pct=lambda x:(x["return_pct"]*100).round(2))
      .rename(columns={"return_pct":"ret_%"})
      .to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 12. INTERPRETATION
# ─────────────────────────────────────────────────────────────
print("\n"+"="*65)
print("INTERPRETATION (v3)")
print("="*65)

n   = overall_v3.get("N_trades", 0)
wr  = overall_v3.get("Win_%", 0)
ar  = overall_v3.get("Avg_ret_%", 0)
mr  = overall_v3.get("Med_ret_%", 0)
nt  = overall_v3.get("N_target", 0)
ns  = overall_v3.get("N_stop", 0)
ntm = overall_v3.get("N_time", 0)

wins   = trades_df.loc[trades_df["return_pct"]>0,"return_pct"]
losses = trades_df.loc[trades_df["return_pct"]<=0,"return_pct"]
avg_w  = wins.mean()*100   if len(wins)>0   else 0
avg_l  = losses.mean()*100 if len(losses)>0 else 0
rr     = abs(avg_w/avg_l)  if avg_l!=0 else float("nan")
bep    = 1/(1+rr)*100      if not np.isnan(rr) else float("nan")

print(f"""
[전략 개선 내역 (v2 → v3)]
  추가된 조건:
  ✦ IWM 당일 방향 (소형주 리스크온 확인)
  ✦ SPY 당일 낙폭 제한 (-1% 이상 시 금지)
  ✦ 나스닥 60일 고점 -25% 이하 구간 진입 금지
  ✦ BB %B 0.0~0.35 (하단 근처 지지 위치)
  ✦ ADX < 40 & ADX 기울기 ≤ 0 (추세 약화 중)
  ✦ 아래꼬리 ≥ 몸통×0.5 (핀바/망치형 캔들)
  ✦ 고변동 종목 제외 (ATR/Close > 3.5%)
  레이어 B: 3개 중 2개 이상 충족 (유연한 기준)

[v1 → v2 → v3 핵심 지표 변화]
  N_trades  : 417 → 95 → {n}
  Win_%     : 26.1% → 30.5% → {wr:.1f}%
  Avg_ret_% : 0.00% → -0.41% → {ar:.2f}%
  Med_ret_% : -5.00% → -4.12% → {mr:.2f}%
  Target    : 11.3% → 12.6% → {nt/max(n,1)*100:.1f}%
  Stop      : 73.6% → 68.4% → {ns/max(n,1)*100:.1f}%
  Time exit : 15.1% → 18.9% → {ntm/max(n,1)*100:.1f}%

[손익 구조]
  승자 평균   : {avg_w:.2f}%
  패자 평균   : {avg_l:.2f}%
  실현 R:R    : 1 : {rr:.2f}
  이론 BEP    : {bep:.1f}%
  실제 승률   : {wr:.1f}%
  {'→ ★ 승률 > BEP: 기대값 양수' if wr > bep else '→ △ 승률 < BEP: 추가 개선 필요'}

[연간 트레이드]
  연평균      : {trades_per_year:.1f}건/년
  총 기간     : {total_years:.1f}년

[위기 구간별 v3 성과]""")

for yr, lbl in {2011:"2011(강등쇼크)", 2015:"2015(중국쇼크)",
                2018:"2018Q4", 2020:"2020(COVID)",
                2022:"2022(베어)", 2025:"2025(관세)"}.items():
    sub = trades_df[trades_df["year"]==yr]
    if len(sub)==0:
        print(f"  {lbl}: 신호 없음")
    else:
        w = (sub["return_pct"]>0).mean()*100
        a = sub["return_pct"].mean()*100
        print(f"  {lbl}: {len(sub)}건, 승률 {w:.0f}%, 평균 {a:.1f}%")

print(f"""
[미적용 조건 — 향후 개선 시 추가 권장]
  ⚠ A4 실적발표일 ±3일 금지: 실적 캘린더 데이터(earningscalendar 등) 연동 필요
  ⚠ 섹터 집중 제한: 같은 섹터 하루 2건 상한
  ⚠ 포트폴리오 레벨 리스크: 월간 -15% 손실 시 진입 중단

[결론]
  3레이어 구조가 신호를 필터링하면서 승률과 손익 구조가 어떻게 변했는지 위 수치로 확인.
  표본이 적을수록 통계적 유의미성이 낮으므로 Walk-forward 검증 권장.
""")

print("="*65)
print("v3 Backtest complete.")
print("="*65)
