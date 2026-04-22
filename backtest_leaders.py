"""
SWING_LEADERS_20D Backtest
Strategy  : Swing trade on sector leaders during market weakness / crisis regimes
Period    : 2015-01-01 ~ 2026-01-01
Adaptation: C2 uses a rolling 5-day look-back window to capture the pullback state
            rather than requiring the exact prior day to be oversold, which makes the
            condition more realistic (the stock may have been under MA20 for 1-3 days
            before the bounce day D).
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

# ─────────────────────────────────────────────
# 0. UNIVERSE
# ─────────────────────────────────────────────
LEADERS = [
    "NVDA",   # AI / semis
    "TSLA",   # EV / energy ecosystem
    "PLTR",   # AI / data infra
    "AVGO",   # semis
    "SMCI",   # AI infra
    "LMT",    # defense / aerospace
    "NOC",    # defense
    "ENPH",   # energy tech
    "AES",    # power / grid
    "SHOP",   # software / growth
    "MSFT",   # AI platform
    "GOOGL",  # AI platform
    "AMZN",   # infra / cloud
    "META",   # AI / consumer
    "AMD"     # semis
]

MARKET_PROXY = "QQQ"
START = "2015-01-01"
END   = "2026-01-01"

# ─────────────────────────────────────────────
# 1. DATA DOWNLOAD
# ─────────────────────────────────────────────
print("=" * 60)
print("Step 1: Downloading data ...")
print("=" * 60)

all_tickers = LEADERS + [MARKET_PROXY]
raw = yf.download(all_tickers, start=START, end=END, auto_adjust=True, progress=False)

price_data = {}
for ticker in all_tickers:
    try:
        df = pd.DataFrame({
            "Open":   raw["Open"][ticker],
            "High":   raw["High"][ticker],
            "Low":    raw["Low"][ticker],
            "Close":  raw["Close"][ticker],
            "Volume": raw["Volume"][ticker],
        }).dropna()
        price_data[ticker] = df
        print(f"  {ticker}: {len(df)} rows  "
              f"{df.index[0].date()} ~ {df.index[-1].date()}")
    except Exception as e:
        print(f"  {ticker}: ERROR - {e}")

# ─────────────────────────────────────────────
# 2. INDICATOR HELPERS
# ─────────────────────────────────────────────

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_cci(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


# ─────────────────────────────────────────────
# 3. COMPUTE INDICATORS
# ─────────────────────────────────────────────
print("\nStep 2: Computing indicators ...")

indicator_data = {}
for ticker in all_tickers:
    df  = price_data[ticker].copy()
    c   = df["Close"]
    h   = df["High"]
    l   = df["Low"]
    v   = df["Volume"]

    df["MA20"]     = c.rolling(20).mean()
    df["MA60"]     = c.rolling(60).mean()
    df["MA200"]    = c.rolling(200).mean()
    df["RSI14"]    = calc_rsi(c, 14)
    df["CCI20"]    = calc_cci(h, l, c, 20)
    macd_l, macd_s, macd_h = calc_macd(c)
    df["MACD_L"]   = macd_l
    df["MACD_S"]   = macd_s
    df["MACD_H"]   = macd_h
    df["VolAvg20"] = v.rolling(20).mean()
    df["VolSpike"] = (v >= 1.5 * df["VolAvg20"])
    df["ATR14"]    = calc_atr(h, l, c, 14)

    if ticker == MARKET_PROXY:
        df["QQQ_MA200"]   = c.rolling(200).mean()
        df["QQQ_60D_MAX"] = c.rolling(60).max()

    indicator_data[ticker] = df

print("  Done.")

# ─────────────────────────────────────────────
# 4. SIGNAL GENERATION
# ─────────────────────────────────────────────
print("\nStep 3: Generating entry signals ...")

qqq = indicator_data[MARKET_PROXY]

VALID_COLS = ["Close", "MA20", "MA60", "MA200", "RSI14", "CCI20",
              "MACD_H", "ATR14", "VolSpike", "VolAvg20"]

trades_raw = []

for ticker in LEADERS:
    df = indicator_data[ticker].copy()

    # Align calendar with QQQ
    common_idx = df.index.intersection(qqq.index)
    df = df.loc[common_idx]
    q  = qqq.loc[common_idx]

    c   = df["Close"]
    rsi = df["RSI14"]
    cci = df["CCI20"]

    # Previous day (D-1) values
    rsi_prev    = rsi.shift(1)
    macd_h_prev = df["MACD_H"].shift(1)

    # 60-day-ago values for RS calculation
    c_60ago   = c.shift(60)
    q_close_60ago = q["Close"].shift(60)

    # ── [0] QQQ Weakness Regime ──────────────────────────────────
    regime_a = q["Close"] < q["QQQ_MA200"]
    regime_b = q["Close"] <= 0.9 * q["QQQ_60D_MAX"]
    cond0    = regime_a | regime_b

    # ── [1] Strong Leader ────────────────────────────────────────
    cond1_ma = c >= 0.95 * df["MA200"]
    rs60     = (c / c_60ago) - (q["Close"] / q_close_60ago)
    cond1    = cond1_ma & (rs60 > 0)

    # ── [2] Pullback state (rolling 5-day look-back) ─────────────
    # The stock must have been in a pullback zone *at least once*
    # in the 5 days ending yesterday: Close < MA20 AND Close > MA60
    # AND (RSI < 50 OR CCI < -30).
    # Using a rolling window is more realistic than requiring the exact
    # prior day to qualify, because the bounce signal (C3) may appear
    # 1–3 days after the oversold low.
    pullback_day = (
        (c < df["MA20"])
        & (c > df["MA60"])
        & ((rsi < 50) | (cci < -30))
    )
    # rolling(5).max() over the previous 5 days (shift(1) so today is excluded)
    cond2 = (
        pullback_day.shift(1).rolling(5).max().fillna(0).astype(bool)
    )

    # ── [3] Bounce / Momentum Signal (current day D) ─────────────
    body_pct = (c - df["Low"]) / (df["High"] - df["Low"]).replace(0, np.nan)
    cond3 = (
        (rsi > rsi_prev)               # RSI slope positive
        & (df["MACD_H"] > macd_h_prev) # MACD histogram improving
        & (body_pct >= 0.6)            # close in upper 40% of day's range
    )

    # ── [4] Volume Spike ─────────────────────────────────────────
    cond4 = df["VolSpike"]

    # ── NaN guard ────────────────────────────────────────────────
    not_nan = df[VALID_COLS].notna().all(axis=1)
    not_nan &= rsi_prev.notna() & macd_h_prev.notna()
    not_nan &= c_60ago.notna() & q_close_60ago.notna()

    signal = cond0 & cond1 & cond2 & cond3 & cond4 & not_nan

    signal_dates = df.index[signal]
    print(f"  {ticker}: {len(signal_dates)} signals")

    for entry_date in signal_dates:
        trades_raw.append({
            "ticker":      ticker,
            "entry_date":  entry_date,
            "entry_price": df.loc[entry_date, "Close"],
            "atr":         df.loc[entry_date, "ATR14"],
        })

print(f"\n  Total raw signals: {len(trades_raw)}")

# ─────────────────────────────────────────────
# 5. TRADE SIMULATION
# ─────────────────────────────────────────────
print("\nStep 4: Simulating trades ...")

trades_raw_df = (
    pd.DataFrame(trades_raw)
    .sort_values("entry_date")
    .reset_index(drop=True)
)

results    = []
active_exit = {}   # ticker -> exit_date of currently open position

for _, row in trades_raw_df.iterrows():
    ticker     = row["ticker"]
    entry_date = row["entry_date"]

    # One position per ticker at a time
    if ticker in active_exit and entry_date <= active_exit[ticker]:
        continue

    entry_price = row["entry_price"]
    atr         = row["atr"]

    stop   = entry_price - 1.5 * atr
    target = min(entry_price + 2.5 * atr, entry_price * 1.15)

    df         = indicator_data[ticker]
    future_idx = df.index[df.index > entry_date][:20]   # max 20 trading days

    outcome    = None
    exit_date  = None
    exit_price = None

    for i, t in enumerate(future_idx):
        lo = df.loc[t, "Low"]
        hi = df.loc[t, "High"]
        cl = df.loc[t, "Close"]

        hit_stop   = lo <= stop
        hit_target = hi >= target

        if hit_stop and hit_target:
            # Both intraday: assume stop hit first (conservative)
            exit_price, outcome, exit_date = stop, "stop", t
            break
        elif hit_stop:
            exit_price, outcome, exit_date = stop, "stop", t
            break
        elif hit_target:
            exit_price, outcome, exit_date = target, "target", t
            break
        elif i == len(future_idx) - 1:
            exit_price, outcome, exit_date = cl, "time_exit", t
            break

    if outcome is None:
        continue   # No future data (near end of dataset)

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
    })

    active_exit[ticker] = exit_date

trades_df = pd.DataFrame(results)
trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
trades_df["exit_date"]  = pd.to_datetime(trades_df["exit_date"])
trades_df["return_pct_display"] = (trades_df["return_pct"] * 100).round(2)

print(f"\n  Total trades executed: {len(trades_df)}")
print("\n  Full trade list:")
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_rows", 200)
print(trades_df[["ticker","entry_date","exit_date","entry_price","exit_price",
                 "return_pct_display","hold_days","outcome"]].to_string(index=False))

# ─────────────────────────────────────────────
# 6. SUMMARY STATISTICS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 5: Summary Statistics")
print("=" * 60)

def summary_stats(df: pd.DataFrame, label: str = "ALL") -> dict:
    n = len(df)
    if n == 0:
        return {"Label": label, "N_trades": 0}
    win_rate  = (df["return_pct"] > 0).mean() * 100
    avg_ret   = df["return_pct"].mean() * 100
    med_ret   = df["return_pct"].median() * 100
    std_ret   = df["return_pct"].std() * 100
    avg_hold  = df["hold_days"].mean()
    n_target  = (df["outcome"] == "target").sum()
    n_stop    = (df["outcome"] == "stop").sum()
    n_time    = (df["outcome"] == "time_exit").sum()
    return {
        "Label":             label,
        "N_trades":          n,
        "Win_rate_%":        round(win_rate, 1),
        "Avg_return_%":      round(avg_ret, 2),
        "Median_return_%":   round(med_ret, 2),
        "Std_return_%":      round(std_ret, 2),
        "Avg_hold_days":     round(avg_hold, 1),
        "N_target":          n_target,
        "N_stop":            n_stop,
        "N_time_exit":       n_time,
    }

# Overall
overall = summary_stats(trades_df, "ALL")
print("\n[Overall]")
for k, v in overall.items():
    print(f"  {k}: {v}")

# Per-ticker
ticker_stats = []
for t in LEADERS:
    sub = trades_df[trades_df["ticker"] == t]
    if len(sub) > 0:
        ticker_stats.append(summary_stats(sub, t))
ticker_stats_df = pd.DataFrame(ticker_stats)
print("\n[Per-Ticker]")
print(ticker_stats_df.to_string(index=False))

# Per-year
trades_df["year"] = trades_df["entry_date"].dt.year
year_stats = []
for yr in sorted(trades_df["year"].unique()):
    sub = trades_df[trades_df["year"] == yr]
    year_stats.append(summary_stats(sub, str(yr)))
year_stats_df = pd.DataFrame(year_stats)
print("\n[Per-Year]")
print(year_stats_df.to_string(index=False))

# ─────────────────────────────────────────────
# 7. EQUITY CURVE
# ─────────────────────────────────────────────
print("\nStep 6: Building equity curve ...")

eq_curve = (
    trades_df
    .sort_values("exit_date")[["exit_date", "return_pct"]]
    .copy()
)
eq_curve["cum_return"] = eq_curve["return_pct"].cumsum()

total_years  = (pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25
trades_per_year = len(trades_df) / total_years

# ─────────────────────────────────────────────
# 8. VISUALIZATION
# ─────────────────────────────────────────────
print("Step 7: Plotting ...")

fig = plt.figure(figsize=(18, 16))
gs  = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── (a) Equity Curve ──────────────────────────────────────────
ax_eq = fig.add_subplot(gs[0, :])
ax_eq.plot(
    eq_curve["exit_date"],
    eq_curve["cum_return"] * 100,
    color="#2563eb", linewidth=1.8, label="Cumulative P&L (non-compounding)"
)
ax_eq.axhline(0, color="black", linewidth=0.8, linestyle="--")

crisis_zones = [
    ("2018-10-01", "2019-01-15", "#f87171", "2018Q4 sell-off"),
    ("2020-02-20", "2020-04-30", "#fb923c", "COVID crash"),
    ("2022-01-01", "2022-12-31", "#facc15", "2022 bear"),
    ("2025-01-01", "2025-12-31", "#a78bfa", "2025 tension"),
]
for s, e, col, lbl in crisis_zones:
    ax_eq.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                  alpha=0.18, color=col, label=lbl)

ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_eq.xaxis.set_major_locator(mdates.YearLocator())
ax_eq.set_title(
    "SWING_LEADERS_20D — Cumulative P&L (non-compounding, equal weight per trade)",
    fontsize=13, fontweight="bold"
)
ax_eq.set_ylabel("Cumulative Return (%)")
ax_eq.legend(fontsize=8, loc="upper left", ncol=3)
ax_eq.grid(alpha=0.3)

# ── (b) Return % histogram ────────────────────────────────────
ax_hist = fig.add_subplot(gs[1, 0])
ret_pct = trades_df["return_pct"] * 100
bins = np.linspace(ret_pct.min() - 1, ret_pct.max() + 1, 35)
ax_hist.hist(ret_pct, bins=bins, color="#3b82f6",
             edgecolor="white", linewidth=0.5)
ax_hist.axvline(ret_pct.mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"Mean {ret_pct.mean():.1f}%")
ax_hist.axvline(0, color="black", linestyle="-", linewidth=0.8)
ax_hist.set_title("Per-Trade Return Distribution", fontsize=11, fontweight="bold")
ax_hist.set_xlabel("Return (%)")
ax_hist.set_ylabel("# Trades")
ax_hist.legend(fontsize=9)
ax_hist.grid(alpha=0.3)

# ── (c) Hold-days histogram ───────────────────────────────────
ax_hold = fig.add_subplot(gs[1, 1])
ax_hold.hist(trades_df["hold_days"], bins=range(1, 23),
             color="#10b981", edgecolor="white", linewidth=0.5, rwidth=0.85)
ax_hold.set_title("Hold-Days Distribution", fontsize=11, fontweight="bold")
ax_hold.set_xlabel("Hold Days (trading)")
ax_hold.set_ylabel("# Trades")
ax_hold.grid(alpha=0.3)

# ── (d) Per-ticker win rate ───────────────────────────────────
ax_wr = fig.add_subplot(gs[2, 0])
if not ticker_stats_df.empty:
    wr_data = ticker_stats_df[["Label","Win_rate_%","N_trades"]].sort_values(
        "Win_rate_%", ascending=False
    )
    colors_wr = ["#22c55e" if w >= 50 else "#ef4444"
                 for w in wr_data["Win_rate_%"]]
    bars = ax_wr.bar(wr_data["Label"], wr_data["Win_rate_%"],
                     color=colors_wr, edgecolor="white")
    ax_wr.axhline(50, color="black", linestyle="--", linewidth=0.9)
    ax_wr.set_title("Win Rate by Ticker", fontsize=11, fontweight="bold")
    ax_wr.set_ylabel("Win Rate (%)")
    ax_wr.set_ylim(0, 105)
    for bar, n in zip(bars, wr_data["N_trades"]):
        ax_wr.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"n={n}", ha="center", va="bottom", fontsize=7
        )
    ax_wr.tick_params(axis="x", rotation=45)
    ax_wr.grid(axis="y", alpha=0.3)

# ── (e) Outcome pie ───────────────────────────────────────────
ax_pie = fig.add_subplot(gs[2, 1])
outcomes = trades_df["outcome"].value_counts()
colors_pie = {"target": "#22c55e", "stop": "#ef4444", "time_exit": "#f59e0b"}
pie_colors = [colors_pie.get(k, "#94a3b8") for k in outcomes.index]
ax_pie.pie(
    outcomes.values,
    labels=outcomes.index,
    autopct="%1.1f%%",
    colors=pie_colors,
    startangle=90,
    textprops={"fontsize": 10},
)
ax_pie.set_title("Outcome Distribution", fontsize=11, fontweight="bold")

plt.suptitle(
    "SWING_LEADERS_20D Backtest  (2015 – 2026)",
    fontsize=15, fontweight="bold", y=1.01
)
plt.savefig("backtest_leaders_result.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Chart saved: backtest_leaders_result.png")

# ─────────────────────────────────────────────
# 9. TEXTUAL INTERPRETATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print(f"""
[1] 엣지 존재 여부
  - 전체 평균 수익률  : {overall['Avg_return_%']:.2f}%
  - 중앙값 수익률     : {overall['Median_return_%']:.2f}%
  - 승률              : {overall['Win_rate_%']:.1f}%
  - 수익률 표준편차   : {overall['Std_return_%']:.2f}%
  → {'★ 양수 기대값 + 승률 > 50% → 통계적 엣지 존재 가능성 (표본 소규모 주의)'
     if overall.get('Avg_return_%', 0) > 0 and overall.get('Win_rate_%', 0) > 50
     else '△ 평균 수익률 또는 승률 개선 여지 — 임계값 튜닝 권장'}

[2] 연간 트레이드 수
  - 백테스트 기간  : {total_years:.1f}년 ({START} ~ {END})
  - 총 트레이드    : {len(trades_df)}건
  - 연평균         : {trades_per_year:.1f}건/년
  - 월평균         : {trades_per_year/12:.1f}건/월
  → 레짐 필터(QQQ 약세 구간)가 강하게 걸려 있어 거래 빈도가 낮음.
    위기 구간에만 진입하는 구조적 특성.

[3] Outcome 비율
  - Target hit  : {overall.get('N_target', 0)}건  ({overall.get('N_target', 0)/max(len(trades_df),1)*100:.1f}%)
  - Stop hit    : {overall.get('N_stop', 0)}건  ({overall.get('N_stop', 0)/max(len(trades_df),1)*100:.1f}%)
  - Time exit   : {overall.get('N_time_exit', 0)}건  ({overall.get('N_time_exit', 0)/max(len(trades_df),1)*100:.1f}%)
  → R:R = 2.5 ATR / 1.5 ATR ≈ 1.67.
    손절 비율이 높더라도 R:R > 1이면 기대값 양수 유지 가능.

[4] 위기 구간별 성과
""")

crisis_labels = {
    2018: "2018 Q4 (미중 무역전쟁/Fed 금리 인상)",
    2020: "2020 (COVID 충격 및 V자 반등)",
    2022: "2022 (금리 급등 베어마켓)",
    2025: "2025 (지정학적 긴장 / 관세 전쟁)",
}
for yr, label in crisis_labels.items():
    sub = trades_df[trades_df["year"] == yr]
    if len(sub) == 0:
        print(f"  {label}: 신호 없음 (위기 기간이지만 리더 종목 조건 불충족)")
    else:
        wr  = (sub["return_pct"] > 0).mean() * 100
        avg = sub["return_pct"].mean() * 100
        n   = len(sub)
        print(f"  {label}")
        print(f"    → {n}건, 승률 {wr:.0f}%, 평균 {avg:.1f}%")

print(f"""
[5] 전략 설계 특성 요약
  - 시장이 약할 때(QQQ < MA200 OR 고점 -10%)만 진입 → 반등 탄력 최대화
  - 강한 리더 종목(MA200 ±5% + 60일 상대강도 양수)만 대상 → 재건 첫 타깃
  - 눌림 후 거래량 동반 반등 신호 → 세력 재매수 포착
  - 타임스탑 20일 / R:R 1.67 → 포지션 장기 묶임 방지
  - 종목별 동시 1포지션만 허용 → 리스크 분산

[주의] 표본이 적을 경우 통계적 유의성이 낮을 수 있습니다.
       파라미터 과최적화(over-fitting) 방지를 위해 Walk-Forward 검증 권장.
""")

print("=" * 60)
print("Backtest complete.")
print("=" * 60)
