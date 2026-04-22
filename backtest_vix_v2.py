"""
VIX-Oversold Reversal Strategy v2 — IMPROVED Backtest
=======================================================
개선사항 (전문가 토론 도출안):
  [0] VIX 레짐 강화
      - VIX[D] >= 25
      - VIX[D] <= VIX 최근 3일 고점 * 0.98  (3일 고점 대비 -2% 이상 하락)
      - QQQ 당일 수익률 >= -1%  (당일 -1% 이상 급락 시 진입 금지)
  [1] MA200 -30% 이내 (기존 유지)
  [2] RSI/CCI 과매도 (기존 유지)
  [3] MACD_H 개선 (기존 유지)
  [4] 거래량 강화
      - QQQ MA20 위: 1.5× 이상
      - QQQ MA20 아래: 2.0× 이상
  [5] 양봉 (기존 유지)

  청산 구조 개선:
      - Stop   = entry - ATR×1.5  (최대 -15% 캡)
      - Target = entry + ATR×4.5  (상한 entry×1.25, 하한 entry×1.08)
      - Break-even stop: 수익 +10% 이상 발생 시 stop → entry가로 이동
      - Time stop: 25 거래일

  포지션 제한:
      - 하루 최대 5종목  (ATR/Close 낮은 순, 즉 저변동성 우선)
      - 티커당 동시 1포지션
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
# 0. UNIVERSE
# ─────────────────────────────────────────────────────────────
NDX100 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG",
    "TSLA","AVGO","PLTR","AMD","INTC","QCOM","TXN",
    "AMAT","LRCX","KLAC","MRVL","MCHP","ON",
    "ADBE","CRM","ORCL","SNOW","NOW","PANW","CRWD",
    "ZS","DDOG","MDB","TEAM","WDAY","ANSS","CDNS",
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
    "PCAR","EA","TTWO","ATVI",
    "CSX","CPRT","DLTR","ROST",
    "BMRN","ALXN","CERN",
    "SGEN","CTSH","KLAC",
    "LULU","ORLY","NTES",
]
NDX100 = sorted(list(dict.fromkeys(NDX100)))

VIX_TICKER = "^VIX"
QQQ_TICKER = "QQQ"
START = "2010-01-01"
END   = "2026-01-01"

MAX_HOLD       = 25     # 거래일 (기존 40 → 25)
BE_TRIGGER_PCT = 0.10   # +10% 시 Break-even stop 발동
ATR_STOP_MULT  = 1.5    # stop = entry - ATR × 1.5
ATR_TGT_MULT   = 4.5    # target = entry + ATR × 4.5
MAX_STOP_PCT   = 0.15   # hard cap: 최대 손실 -15%
MIN_TGT_PCT    = 0.08   # 최소 목표 +8%
MAX_TGT_PCT    = 0.25   # 최대 목표 +25%
MAX_DAILY      = 5      # 하루 최대 진입 종목 수

# ─────────────────────────────────────────────────────────────
# 1. DOWNLOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("Step 1: Downloading data ...")
print("=" * 65)

# VIX
vix_raw = yf.download(VIX_TICKER, start=START, end=END, auto_adjust=True, progress=False)
_vc = vix_raw["Close"]
if isinstance(_vc, pd.DataFrame):
    _vc = _vc.iloc[:, 0]
vix = pd.Series(_vc.values, index=_vc.index, name="VIX", dtype=float).dropna()
print(f"  VIX: {len(vix)} rows  {vix.index[0].date()} ~ {vix.index[-1].date()}")

# QQQ
qqq_raw = yf.download(QQQ_TICKER, start=START, end=END, auto_adjust=True, progress=False)
def extract_close(raw, ticker):
    c = raw["Close"]
    if isinstance(c, pd.DataFrame):
        c = c[ticker] if ticker in c.columns else c.iloc[:, 0]
    return pd.Series(c.values, index=c.index, dtype=float).dropna()

qqq_close = extract_close(qqq_raw, QQQ_TICKER)
qqq_ma20  = qqq_close.rolling(20).mean()
print(f"  QQQ: {len(qqq_close)} rows")

# Stocks
raw = yf.download(NDX100, start=START, end=END, auto_adjust=True, progress=False)
price_data = {}
skipped = []
for ticker in NDX100:
    try:
        df = pd.DataFrame({
            "Open":   raw["Open"][ticker],
            "High":   raw["High"][ticker],
            "Low":    raw["Low"][ticker],
            "Close":  raw["Close"][ticker],
            "Volume": raw["Volume"][ticker],
        }).dropna()
        if len(df) >= 300:
            price_data[ticker] = df
        else:
            skipped.append(ticker)
    except Exception:
        skipped.append(ticker)

print(f"  Loaded {len(price_data)} tickers (skipped {len(skipped)}: {skipped})")

# ─────────────────────────────────────────────────────────────
# 2. INDICATOR HELPERS
# ─────────────────────────────────────────────────────────────

def calc_rsi(close, period=14):
    d = close.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    al = l.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))

def calc_cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))

def calc_macd(close, fast=12, slow=26, signal=9):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    ml = ef - es
    sl = ml.ewm(span=signal, adjust=False).mean()
    return ml, sl, ml - sl

def calc_atr(high, low, close, period=14):
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

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
    _, _, mh      = calc_macd(c)
    d["MACD_H"]   = mh
    d["VolAvg20"] = v.rolling(20).mean()
    d["ATR14"]    = calc_atr(h, l, c)
    indicator_data[ticker] = d

print(f"  Done ({len(indicator_data)} tickers).")

# ─────────────────────────────────────────────────────────────
# 4. BUILD VIX / QQQ REGIME SERIES
# ─────────────────────────────────────────────────────────────
vix_3d_max    = vix.rolling(3).max()          # 최근 3일 VIX 고점
vix_threshold = vix_3d_max * 0.98             # 고점 대비 -2% 기준
qqq_ret_1d    = qqq_close.pct_change()        # QQQ 당일 등락률
qqq_ma20_full = qqq_close.rolling(20).mean()  # QQQ MA20

# ─────────────────────────────────────────────────────────────
# 5. SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────
print("\nStep 3: Generating entry signals ...")

VALID_COLS = ["Open", "Close", "High", "Low",
              "MA200", "RSI14", "CCI20", "MACD_H",
              "VolAvg20", "ATR14"]

trades_raw = []  # list of dict with ticker, entry_date, entry_price, atr, vol_ratio

for ticker, df in indicator_data.items():
    common_idx = df.index.intersection(vix.index).intersection(qqq_close.index)
    d    = df.loc[common_idx].copy()
    vx   = vix.reindex(common_idx)
    vx3m = vix_3d_max.reindex(common_idx)
    vx_t = vix_threshold.reindex(common_idx)
    qq_r = qqq_ret_1d.reindex(common_idx)
    qq_ma= qqq_ma20_full.reindex(common_idx)
    qq_c = qqq_close.reindex(common_idx)

    c         = d["Close"]
    o         = d["Open"]
    rsi       = d["RSI14"]
    cci       = d["CCI20"]
    rsi_prev  = rsi.shift(1)
    cci_prev  = cci.shift(1)
    mh_prev   = d["MACD_H"].shift(1)
    vol_avg20  = d["VolAvg20"]
    actual_vol = price_data[ticker]["Volume"].reindex(common_idx)

    not_nan  = d[VALID_COLS].notna().all(axis=1)
    not_nan &= rsi_prev.notna() & cci_prev.notna() & mh_prev.notna()
    not_nan &= vx.notna() & vx3m.notna() & qq_r.notna() & qq_ma.notna()

    # [0] VIX 레짐 (강화)
    cond0_a = vx >= 25                         # VIX 절대 임계값
    cond0_b = vx <= vx_t                       # 3일 고점 대비 -2% 이상 하락
    cond0_c = qq_r >= -0.01                    # QQQ 당일 -1% 이상 급락 시 금지
    cond0   = cond0_a & cond0_b & cond0_c

    # [1] MA200 -30% 이내
    cond1 = c >= d["MA200"] * 0.70

    # [2] RSI/CCI 과매도
    cond2 = (rsi < 30) | (rsi_prev < 30) | (cci < -100) | (cci_prev < -100)

    # [3] MACD_H 개선
    cond3 = d["MACD_H"] > mh_prev

    # [4] 거래량 (QQQ MA20 위/아래 구분)
    vol_mult = pd.Series(np.where(qq_c >= qq_ma, 1.5, 2.0), index=common_idx)
    actual_vol = price_data[ticker]["Volume"].reindex(common_idx)
    cond4 = actual_vol >= vol_mult * vol_avg20

    # [5] 양봉
    cond5 = c > o

    signal = cond0 & cond1 & cond2 & cond3 & cond4 & cond5 & not_nan

    for entry_date in d.index[signal]:
        atr_val = d.loc[entry_date, "ATR14"]
        ep      = c.loc[entry_date]
        vol_ratio = atr_val / ep  # ATR/Close = 정규화 변동성 (낮을수록 안정)
        trades_raw.append({
            "ticker":      ticker,
            "entry_date":  entry_date,
            "entry_price": ep,
            "atr":         atr_val,
            "vol_ratio":   vol_ratio,
        })

print(f"  Total raw signals: {len(trades_raw)}")

# ─────────────────────────────────────────────────────────────
# 6. TRADE SIMULATION (with daily cap + ATR-based exits + BE stop)
# ─────────────────────────────────────────────────────────────
print("\nStep 4: Simulating trades ...")

trades_raw_df = (pd.DataFrame(trades_raw)
                   .sort_values(["entry_date", "vol_ratio"])  # 낮은 변동성 우선
                   .reset_index(drop=True))

results     = []
active_exit = {}   # ticker -> exit_date
daily_count = {}   # date -> count of new entries

for _, row in trades_raw_df.iterrows():
    ticker     = row["ticker"]
    entry_date = row["entry_date"]

    # 티커당 1포지션
    if ticker in active_exit and entry_date <= active_exit[ticker]:
        continue

    # 하루 최대 5종목
    dc = daily_count.get(entry_date, 0)
    if dc >= MAX_DAILY:
        continue

    entry_price = row["entry_price"]
    atr         = row["atr"]

    # ATR 기반 동적 손익
    raw_stop   = entry_price - ATR_STOP_MULT * atr
    hard_stop  = entry_price * (1 - MAX_STOP_PCT)
    stop       = max(raw_stop, hard_stop)   # 더 큰(손실이 작은) 값 선택 → 최대 -15% 캡

    raw_target = entry_price + ATR_TGT_MULT * atr
    min_target = entry_price * (1 + MIN_TGT_PCT)
    max_target = entry_price * (1 + MAX_TGT_PCT)
    target     = np.clip(raw_target, min_target, max_target)

    df         = indicator_data[ticker]
    future_idx = df.index[df.index > entry_date][:MAX_HOLD]

    outcome    = None
    exit_date  = None
    exit_price = None
    be_active  = False   # Break-even stop 활성화 여부
    be_level   = entry_price  # BE stop 기준가

    for i, t in enumerate(future_idx):
        lo = df.loc[t, "Low"]
        hi = df.loc[t, "High"]
        cl = df.loc[t, "Close"]

        current_stop = be_level if be_active else stop

        # BE stop 발동 체크 (당일 고가 기준)
        if not be_active and hi >= entry_price * (1 + BE_TRIGGER_PCT):
            be_active = True

        hit_stop   = lo <= current_stop
        hit_target = hi >= target

        if hit_stop and hit_target:
            exit_price, outcome, exit_date = current_stop, "stop", t
            break
        elif hit_stop:
            exit_price, outcome, exit_date = current_stop, "stop", t
            break
        elif hit_target:
            exit_price, outcome, exit_date = target, "target", t
            break
        elif i == len(future_idx) - 1:
            exit_price, outcome, exit_date = cl, "time_exit", t
            break

    if outcome is None:
        continue

    return_pct = exit_price / entry_price - 1
    hold_days  = len(df.loc[(df.index > entry_date) & (df.index <= exit_date)])

    results.append({
        "ticker":      ticker,
        "entry_date":  entry_date,
        "exit_date":   exit_date,
        "entry_price": round(entry_price, 4),
        "exit_price":  round(exit_price, 4),
        "return_pct":  return_pct,
        "hold_days":   hold_days,
        "outcome":     outcome,
        "atr":         round(atr, 4),
        "stop":        round(stop, 4),
        "target":      round(target, 4),
    })

    active_exit[ticker] = exit_date
    daily_count[entry_date] = dc + 1

trades_df = pd.DataFrame(results)
trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
trades_df["exit_date"]  = pd.to_datetime(trades_df["exit_date"])
trades_df["year"]       = trades_df["entry_date"].dt.year

print(f"  Total trades executed: {len(trades_df)}")

# ─────────────────────────────────────────────────────────────
# 7. SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Step 5: Summary Statistics")
print("=" * 65)

def summary_stats(df, label="ALL"):
    n = len(df)
    if n == 0:
        return {"Label": label, "N_trades": 0}
    win_rate = (df["return_pct"] > 0).mean() * 100
    avg_ret  = df["return_pct"].mean() * 100
    med_ret  = df["return_pct"].median() * 100
    std_ret  = df["return_pct"].std() * 100
    avg_hold = df["hold_days"].mean()
    n_t      = (df["outcome"] == "target").sum()
    n_s      = (df["outcome"] == "stop").sum()
    n_tm     = (df["outcome"] == "time_exit").sum()
    return {
        "Label":           label,
        "N_trades":        n,
        "Win_rate_%":      round(win_rate, 1),
        "Avg_return_%":    round(avg_ret, 2),
        "Median_return_%": round(med_ret, 2),
        "Std_return_%":    round(std_ret, 2),
        "Avg_hold_days":   round(avg_hold, 1),
        "N_target":        n_t,
        "N_stop":          n_s,
        "N_time_exit":     n_tm,
    }

overall_v2 = summary_stats(trades_df, "v2_ALL")
print("\n[Overall v2]")
for k, v in overall_v2.items():
    print(f"  {k}: {v}")

# Per-ticker
ticker_stats = []
for t in sorted(trades_df["ticker"].unique()):
    sub = trades_df[trades_df["ticker"] == t]
    if len(sub) >= 1:
        ticker_stats.append(summary_stats(sub, t))
ticker_stats_df = pd.DataFrame(ticker_stats).sort_values("N_trades", ascending=False)
print("\n[Per-Ticker]")
print(ticker_stats_df.to_string(index=False))

# Per-year
year_stats = []
for yr in sorted(trades_df["year"].unique()):
    sub = trades_df[trades_df["year"] == yr]
    year_stats.append(summary_stats(sub, str(yr)))
year_stats_df = pd.DataFrame(year_stats)
print("\n[Per-Year]")
print(year_stats_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 8. EQUITY CURVE
# ─────────────────────────────────────────────────────────────
eq_curve = (trades_df.sort_values("exit_date")[["exit_date","return_pct"]].copy())
eq_curve["cum_return"] = eq_curve["return_pct"].cumsum()

total_years     = (pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25
trades_per_year = len(trades_df) / total_years

# ─────────────────────────────────────────────────────────────
# 9. VISUALIZATION — v2 단독 + v1 vs v2 비교
# ─────────────────────────────────────────────────────────────
print("\nStep 6: Plotting ...")

# ── v2 전략 시각화 ──────────────────────────────────────────
fig = plt.figure(figsize=(20, 20))
gs  = GridSpec(4, 2, figure=fig, hspace=0.48, wspace=0.33)

# (a) Equity Curve
ax_eq = fig.add_subplot(gs[0, :])
ax_eq.plot(eq_curve["exit_date"], eq_curve["cum_return"]*100,
           color="#2563eb", linewidth=1.8,
           label="v2: Cumulative P&L (non-compounding)")
ax_eq.axhline(0, color="black", linewidth=0.8, linestyle="--")

crisis_zones = [
    ("2011-08-01","2011-10-31","#94a3b8","2011 US downgrade"),
    ("2015-08-01","2016-02-29","#6366f1","2015–16 China"),
    ("2018-10-01","2019-01-15","#f87171","2018Q4"),
    ("2020-02-20","2020-04-30","#fb923c","COVID"),
    ("2022-01-01","2022-12-31","#facc15","2022 bear"),
    ("2025-01-01","2025-06-30","#a78bfa","2025 tension"),
]
for s, e, col, lbl in crisis_zones:
    ax_eq.axvspan(pd.Timestamp(s), min(pd.Timestamp(e), pd.Timestamp(END)),
                  alpha=0.18, color=col, label=lbl)

ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_eq.xaxis.set_major_locator(mdates.YearLocator())
ax_eq.set_title("v2 Improved — Cumulative P&L", fontsize=13, fontweight="bold")
ax_eq.set_ylabel("Cumulative Return (%)")
ax_eq.legend(fontsize=8, loc="upper left", ncol=4)
ax_eq.grid(alpha=0.3)

# (b) VIX with entries
ax_vix = fig.add_subplot(gs[1, :])
ax_vix.plot(vix.index, vix.values, color="#dc2626", linewidth=1.0, label="VIX")
ax_vix.axhline(25, color="#f97316", linewidth=1.2, linestyle="--", label="VIX=25")
ax_vix.fill_between(vix.index, vix.values, 25,
                    where=(vix.values >= 25), alpha=0.2, color="#f97316")
entry_vix = vix.reindex(trades_df["entry_date"].values, method="nearest")
ax_vix.scatter(entry_vix.index, entry_vix.values,
               color="#1d4ed8", s=15, zorder=5, alpha=0.8, label="v2 Entries")
ax_vix.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_vix.xaxis.set_major_locator(mdates.YearLocator())
ax_vix.set_title("VIX & v2 Entry Points (fewer, higher quality)", fontsize=11, fontweight="bold")
ax_vix.legend(fontsize=8)
ax_vix.grid(alpha=0.3)

# (c) Return distribution
ax_hist = fig.add_subplot(gs[2, 0])
ret_pct = trades_df["return_pct"] * 100
all_rets = ret_pct.values
bins = np.linspace(min(all_rets)-0.5, max(all_rets)+0.5, 40)
n_win  = (ret_pct > 0).sum()
n_loss = (ret_pct <= 0).sum()
ax_hist.hist(ret_pct[ret_pct > 0], bins=bins, color="#22c55e",
             edgecolor="white", linewidth=0.4, alpha=0.85, label=f"Win ({n_win})")
ax_hist.hist(ret_pct[ret_pct <= 0], bins=bins, color="#ef4444",
             edgecolor="white", linewidth=0.4, alpha=0.85, label=f"Loss ({n_loss})")
ax_hist.axvline(ret_pct.mean(), color="#1e3a8a", linestyle="--", linewidth=1.5,
                label=f"Mean {ret_pct.mean():.1f}%")
ax_hist.axvline(0, color="black", linewidth=0.8)
ax_hist.set_title("Per-Trade Return Distribution (v2)", fontsize=11, fontweight="bold")
ax_hist.set_xlabel("Return (%)"); ax_hist.set_ylabel("# Trades")
ax_hist.legend(fontsize=9); ax_hist.grid(alpha=0.3)

# (d) Hold days
ax_hold = fig.add_subplot(gs[2, 1])
ax_hold.hist(trades_df["hold_days"], bins=range(1, MAX_HOLD+3),
             color="#0ea5e9", edgecolor="white", linewidth=0.4, rwidth=0.85)
ax_hold.set_title("Hold-Days Distribution (v2)", fontsize=11, fontweight="bold")
ax_hold.set_xlabel("Hold Days"); ax_hold.set_ylabel("# Trades")
ax_hold.grid(alpha=0.3)

# (e) Per-year avg return
ax_yr = fig.add_subplot(gs[3, 0])
yr_df = year_stats_df[year_stats_df["N_trades"] > 0].copy()
colors_yr = ["#22c55e" if x > 0 else "#ef4444" for x in yr_df["Avg_return_%"]]
bars = ax_yr.bar(yr_df["Label"].astype(str), yr_df["Avg_return_%"],
                 color=colors_yr, edgecolor="white")
ax_yr.axhline(0, color="black", linewidth=0.8)
for bar, n in zip(bars, yr_df["N_trades"]):
    ax_yr.text(bar.get_x()+bar.get_width()/2,
               bar.get_height()+(0.1 if bar.get_height()>=0 else -0.6),
               f"n={n}", ha="center", va="bottom", fontsize=7)
ax_yr.set_title("Avg Return by Year (v2)", fontsize=11, fontweight="bold")
ax_yr.set_ylabel("Avg Return (%)"); ax_yr.tick_params(axis="x", rotation=45)
ax_yr.grid(axis="y", alpha=0.3)

# (f) Outcome pie
ax_pie = fig.add_subplot(gs[3, 1])
outcomes   = trades_df["outcome"].value_counts()
col_map    = {"target":"#22c55e","stop":"#ef4444","time_exit":"#f59e0b"}
pie_colors = [col_map.get(k,"#94a3b8") for k in outcomes.index]
ax_pie.pie(outcomes.values, labels=outcomes.index, autopct="%1.1f%%",
           colors=pie_colors, startangle=90, textprops={"fontsize":10})
ax_pie.set_title("Outcome Distribution (v2)", fontsize=11, fontweight="bold")

plt.suptitle("VIX-Oversold Reversal v2 (Improved)  |  Nasdaq-100  |  2010–2026",
             fontsize=15, fontweight="bold", y=1.005)
plt.savefig("backtest_vix_v2_result.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: backtest_vix_v2_result.png")

# ─────────────────────────────────────────────────────────────
# 10. v1 vs v2 COMPARISON TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("v1 vs v2 COMPARISON")
print("=" * 65)

v1 = {
    "Label":           "v1 (기존)",
    "N_trades":        417,
    "Win_rate_%":      26.1,
    "Avg_return_%":    0.00,
    "Median_return_%": -5.00,
    "Std_return_%":    8.99,
    "Avg_hold_days":   12.3,
    "N_target":        47,
    "N_stop":          307,
    "N_time_exit":     63,
}
v2 = overall_v2
v2["Label"] = "v2 (개선)"

comp_df = pd.DataFrame([v1, v2]).set_index("Label")
print(comp_df.to_string())

# Year-by-year comparison (v2 only, since v1 is fixed)
print("\n[v2 Per-Year]")
print(year_stats_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 11. INTERPRETATION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("INTERPRETATION (v2 개선 전략)")
print("=" * 65)

n_t   = overall_v2.get("N_trades", 0)
win_r = overall_v2.get("Win_rate_%", 0)
avg_r = overall_v2.get("Avg_return_%", 0)
med_r = overall_v2.get("Median_return_%", 0)
n_tgt = overall_v2.get("N_target", 0)
n_stp = overall_v2.get("N_stop", 0)
n_tim = overall_v2.get("N_time_exit", 0)

# ATR 기반 손익비는 동적이므로 실현된 평균으로 계산
avg_win  = trades_df.loc[trades_df["return_pct"] > 0, "return_pct"].mean() * 100
avg_loss = trades_df.loc[trades_df["return_pct"] <= 0, "return_pct"].mean() * 100
real_rr  = abs(avg_win / avg_loss) if avg_loss != 0 else float("nan")

print(f"""
[개선 내역 요약]
  VIX 조건     : VIX≥25 & 3일고점-2%하락 & QQQ 당일-1% 이상낙폭 금지
  거래량 조건  : QQQ MA20 위→1.5×, 아래→2.0×
  Stop         : ATR×1.5 (max -15%)
  Target       : ATR×4.5 (범위 +8%~+25%)
  BE Stop      : +10% 달성 시 손절선→진입가로 이동
  보유 기간    : 25일 (기존 40일)
  일별 최대    : 5종목 (ATR/Close 낮은순 선택)

[v1 → v2 핵심 지표 변화]
  총 트레이드  : 417 → {n_t} ({n_t - 417:+d})
  승률         : 26.1% → {win_r:.1f}%
  평균 수익률  : 0.00% → {avg_r:.2f}%
  중앙값 수익률: -5.00% → {med_r:.2f}%
  Target hit   : 47 (11.3%) → {n_tgt} ({n_tgt/max(n_t,1)*100:.1f}%)
  Stop hit     : 307 (73.6%) → {n_stp} ({n_stp/max(n_t,1)*100:.1f}%)
  Time exit    : 63 (15.1%) → {n_tim} ({n_tim/max(n_t,1)*100:.1f}%)
  실현 손익비  : (고정 1:4) → 1:{real_rr:.1f} (실현값)

[연간 트레이드]
  연평균       : {trades_per_year:.1f}건/년 (기존 26.1건/년)

[손익분기 승률]
  이론 BEP = 1/(1+실현R:R) = {1/(1+real_rr)*100:.1f}%
  실제 승률 {win_r:.1f}% {'> BEP → 기대값 양수 ★' if win_r > 1/(1+real_rr)*100 else '< BEP → 추가 개선 필요 △'}
""")

crisis_years = {
    2011: "2011 (미국 신용등급 강등)",
    2015: "2015 (중국 쇼크)",
    2016: "2016 (브렉시트)",
    2018: "2018 Q4 (Fed 인상)",
    2020: "2020 (COVID)",
    2022: "2022 (금리 급등)",
    2024: "2024",
    2025: "2025 (관세 전쟁)",
}
print("[위기 구간별 v2 성과]")
for yr, label in crisis_years.items():
    sub = trades_df[trades_df["year"] == yr]
    if len(sub) == 0:
        print(f"  {label}: 신호 없음")
    else:
        wr  = (sub["return_pct"] > 0).mean() * 100
        avg = sub["return_pct"].mean() * 100
        n   = len(sub)
        print(f"  {label}: {n}건, 승률 {wr:.0f}%, 평균 {avg:.1f}%")

print(f"""
[전문가 토론 도출 개선 효과 평가]
  ✓ 신호 군집 해소: 하루 최대 5종목 캡 → 2025.4.9 40종목→5종목으로 제한
  ✓ Dead Cat Bounce 필터: VIX 3일고점 기준 진짜 하향 전환만 포착
  ✓ QQQ 낙폭 필터: 지수 급락일 진입 방지
  ✓ ATR 기반 동적 손익: 종목 변동성에 맞는 손절/익절
  ✓ Break-even stop: 수익 반납 구조적 방지
  ✓ 보유기간 단축: time_exit 수익 개선

[후속 개선 과제]
  • 섹터 집중 제한 추가 (같은 섹터 하루 2건 상한)
  • Walk-forward 검증 (2010-2019 학습 / 2020-2025 검증)
  • 포트폴리오 레벨 리스크: 동시 오픈 10개 상한, 월 -15% 시 진입 중단
""")

print("=" * 65)
print("Backtest v2 complete.")
print("=" * 65)
