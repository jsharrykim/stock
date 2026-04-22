"""
VIX-Oversold Reversal Strategy — Backtest
Universe  : Nasdaq-100 top 100 constituents (approx)
Period    : Maximum available data (2010-01-01 ~ 2026-01-01)
Strategy  : Long entries when:
  [0] VIX >= 25 AND VIX today < VIX yesterday (fear spike but cooling)
  [1] Stock Close >= MA200 * 0.70  (not more than -30% below 200MA)
  [2] RSI14[D] < 30 OR RSI14[D-1] < 30 OR CCI20[D] < -100 OR CCI20[D-1] < -100
  [3] MACD_H[D] > MACD_H[D-1]
  [4] Volume >= 1.5 * VolAvg20
  [5] Bullish candle: Close[D] > Open[D]
Exit rules:
  stop   = entry * 0.95  (-5%)
  target = entry * 1.20  (+20%)
  time   = 40 trading days (max hold)
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
# 0.  UNIVERSE  — Nasdaq-100 representative tickers
#     (as of 2025; some replaced with long-history equivalents)
# ─────────────────────────────────────────────────────────────
NDX100 = [
    # Mega-cap tech / AI
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG",
    "TSLA", "AVGO", "PLTR", "AMD", "INTC", "QCOM", "TXN",
    "AMAT", "LRCX", "KLAC", "MRVL", "MCHP", "ON",
    # Cloud / SaaS
    "ADBE", "CRM", "ORCL", "SNOW", "NOW", "PANW", "CRWD",
    "ZS", "DDOG", "MDB", "TEAM", "WDAY", "ANSS", "CDNS",
    # Internet / E-commerce
    "SHOP", "EBAY", "BKNG", "ABNB", "EXPE", "TRIP",
    # Biotech / Healthcare
    "AMGN", "GILD", "BIIB", "REGN", "VRTX", "IDXX", "DXCM",
    "ISRG", "ILMN", "MRNA",
    # Consumer / Media
    "NFLX", "CMCSA", "CHTR", "TMUS", "SIRI",
    "COST", "SBUX", "MDLZ", "PEP", "MNST",
    # Financials / Payments
    "PYPL", "ADSK", "MELI",
    # Industrials / Autos
    "HON", "GEHC", "FANG", "CEG",
    # Semis / Hardware
    "MU", "WDC", "STX",
    # Others
    "ASML", "SMCI", "SNPS", "FTNT", "NXPI",
    "ODFL", "VRSK", "CTAS", "FAST",
    "ADP", "PAYX", "INTU",
    "PCAR", "EA", "TTWO", "ATVI",
    "CSX", "CPRT", "DLTR", "ROST",
    "BMRN", "ALXN", "CERN",
    "SGEN", "CTSH", "KLAC",
    "LULU", "ORLY", "NTES",
]
# De-duplicate and keep clean list
NDX100 = sorted(list(dict.fromkeys(NDX100)))
print(f"Universe size: {len(NDX100)} tickers")

VIX_TICKER = "^VIX"
START = "2010-01-01"
END   = "2026-01-01"

STOP_PCT   = 0.05   # -5%
TARGET_PCT = 0.20   # +20%
MAX_HOLD   = 40     # trading days

# ─────────────────────────────────────────────────────────────
# 1.  DOWNLOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("Step 1: Downloading data ...")
print("=" * 65)

# Download VIX
vix_raw = yf.download(VIX_TICKER, start=START, end=END,
                      auto_adjust=True, progress=False)
# yfinance returns MultiIndex columns even for single ticker — flatten
_vix_close = vix_raw["Close"]
if isinstance(_vix_close, pd.DataFrame):
    # MultiIndex: columns = (field, ticker) or just ticker level
    _vix_close = _vix_close.iloc[:, 0]
vix = _vix_close.dropna().squeeze()
vix = pd.Series(vix.values, index=vix.index, name="VIX", dtype=float)
print(f"  VIX: {len(vix)} rows  {vix.index[0].date()} ~ {vix.index[-1].date()}")

# Download all NDX100 tickers in one batch
raw = yf.download(NDX100, start=START, end=END,
                  auto_adjust=True, progress=False)

price_data = {}
skipped    = []
for ticker in NDX100:
    try:
        df = pd.DataFrame({
            "Open":   raw["Open"][ticker],
            "High":   raw["High"][ticker],
            "Low":    raw["Low"][ticker],
            "Close":  raw["Close"][ticker],
            "Volume": raw["Volume"][ticker],
        }).dropna()
        # Keep only tickers with at least 300 rows (enough for MA200)
        if len(df) >= 300:
            price_data[ticker] = df
        else:
            skipped.append(ticker)
    except Exception:
        skipped.append(ticker)

print(f"  Loaded {len(price_data)} tickers "
      f"(skipped {len(skipped)}: {skipped})")

# ─────────────────────────────────────────────────────────────
# 2.  INDICATOR HELPERS
# ─────────────────────────────────────────────────────────────

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
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def calc_macd(close: pd.Series,
              fast: int = 12, slow: int = 26, signal: int = 9):
    ef  = close.ewm(span=fast,   adjust=False).mean()
    es  = close.ewm(span=slow,   adjust=False).mean()
    ml  = ef - es
    sl  = ml.ewm(span=signal, adjust=False).mean()
    return ml, sl, ml - sl


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([high - low,
                    (high - pc).abs(),
                    (low  - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


# ─────────────────────────────────────────────────────────────
# 3.  COMPUTE INDICATORS FOR ALL TICKERS
# ─────────────────────────────────────────────────────────────
print("\nStep 2: Computing indicators ...")

indicator_data = {}
for ticker, df in price_data.items():
    d = df.copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]

    d["MA200"]    = c.rolling(200).mean()
    d["RSI14"]    = calc_rsi(c, 14)
    d["CCI20"]    = calc_cci(h, l, c, 20)
    ml, sl, mh    = calc_macd(c)
    d["MACD_H"]   = mh
    d["VolAvg20"] = v.rolling(20).mean()
    d["VolSpike"] = v >= 1.5 * d["VolAvg20"]
    d["ATR14"]    = calc_atr(h, l, c, 14)
    indicator_data[ticker] = d

print(f"  Done ({len(indicator_data)} tickers).")

# ─────────────────────────────────────────────────────────────
# 4.  SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────
print("\nStep 3: Generating entry signals ...")

VALID_COLS = ["Open", "Close", "High", "Low",
              "MA200", "RSI14", "CCI20", "MACD_H",
              "VolAvg20", "VolSpike", "ATR14"]

trades_raw = []

for ticker, df in indicator_data.items():
    # Align with VIX calendar
    common_idx = df.index.intersection(vix.index)
    d  = df.loc[common_idx].copy()
    vx = vix.loc[common_idx].copy()
    vx_prev = vx.shift(1)

    c        = d["Close"]
    o        = d["Open"]
    rsi      = d["RSI14"]
    cci      = d["CCI20"]
    rsi_prev = rsi.shift(1)
    cci_prev = cci.shift(1)
    mh_prev  = d["MACD_H"].shift(1)

    # NaN guard
    not_nan = d[VALID_COLS].notna().all(axis=1)
    not_nan &= rsi_prev.notna() & cci_prev.notna() & mh_prev.notna()
    not_nan &= vx_prev.notna()

    # ── [0] VIX >= 25 and falling ────────────────────────────
    cond0 = (vx >= 25) & (vx < vx_prev)

    # ── [1] Not too far below MA200 (within -30%) ────────────
    cond1 = c >= d["MA200"] * 0.70

    # ── [2] RSI or CCI oversold (today or yesterday) ─────────
    cond2 = (
        (rsi < 30)
        | (rsi_prev < 30)
        | (cci < -100)
        | (cci_prev < -100)
    )

    # ── [3] MACD histogram improving ────────────────────────
    cond3 = d["MACD_H"] > mh_prev

    # ── [4] Volume spike ─────────────────────────────────────
    cond4 = d["VolSpike"]

    # ── [5] Bullish candle ───────────────────────────────────
    cond5 = c > o

    signal = cond0 & cond1 & cond2 & cond3 & cond4 & cond5 & not_nan

    for entry_date in d.index[signal]:
        trades_raw.append({
            "ticker":      ticker,
            "entry_date":  entry_date,
            "entry_price": c.loc[entry_date],
        })

total_signals = len(trades_raw)
print(f"  Total raw signals: {total_signals}")

# ─────────────────────────────────────────────────────────────
# 5.  TRADE SIMULATION
# ─────────────────────────────────────────────────────────────
print("\nStep 4: Simulating trades ...")

trades_raw_df = (pd.DataFrame(trades_raw)
                   .sort_values("entry_date")
                   .reset_index(drop=True))

results     = []
active_exit = {}   # ticker -> last exit_date

for _, row in trades_raw_df.iterrows():
    ticker     = row["ticker"]
    entry_date = row["entry_date"]

    # One position per ticker at a time
    if ticker in active_exit and entry_date <= active_exit[ticker]:
        continue

    entry_price = row["entry_price"]
    stop        = entry_price * (1 - STOP_PCT)    # -5%
    target      = entry_price * (1 + TARGET_PCT)  # +20%

    df         = indicator_data[ticker]
    future_idx = df.index[df.index > entry_date][:MAX_HOLD]

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
            # Conservative: stop first
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
    })
    active_exit[ticker] = exit_date

trades_df = pd.DataFrame(results)
trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
trades_df["exit_date"]  = pd.to_datetime(trades_df["exit_date"])
trades_df["year"]       = trades_df["entry_date"].dt.year

print(f"  Total trades executed: {len(trades_df)}")

# ─────────────────────────────────────────────────────────────
# 6.  SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Step 5: Summary Statistics")
print("=" * 65)

def summary_stats(df: pd.DataFrame, label: str = "ALL") -> dict:
    n = len(df)
    if n == 0:
        return {"Label": label, "N_trades": 0}
    win_rate = (df["return_pct"] > 0).mean() * 100
    avg_ret  = df["return_pct"].mean() * 100
    med_ret  = df["return_pct"].median() * 100
    std_ret  = df["return_pct"].std() * 100
    avg_hold = df["hold_days"].mean()
    n_target = (df["outcome"] == "target").sum()
    n_stop   = (df["outcome"] == "stop").sum()
    n_time   = (df["outcome"] == "time_exit").sum()
    # Expected Value: E[R] with discrete outcomes
    ev = avg_ret
    return {
        "Label":           label,
        "N_trades":        n,
        "Win_rate_%":      round(win_rate, 1),
        "Avg_return_%":    round(avg_ret, 2),
        "Median_return_%": round(med_ret, 2),
        "Std_return_%":    round(std_ret, 2),
        "Avg_hold_days":   round(avg_hold, 1),
        "N_target":        n_target,
        "N_stop":          n_stop,
        "N_time_exit":     n_time,
    }

overall = summary_stats(trades_df, "ALL")
print("\n[Overall]")
for k, v in overall.items():
    print(f"  {k}: {v}")

# Per-ticker (top 20 by trade count)
ticker_stats = []
for t in sorted(trades_df["ticker"].unique()):
    sub = trades_df[trades_df["ticker"] == t]
    if len(sub) >= 1:
        ticker_stats.append(summary_stats(sub, t))
ticker_stats_df = pd.DataFrame(ticker_stats).sort_values(
    "N_trades", ascending=False
)
print("\n[Per-Ticker (sorted by trade count)]")
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
# 7.  EQUITY CURVE
# ─────────────────────────────────────────────────────────────
print("\nStep 6: Building equity curve ...")

eq_curve = (trades_df
            .sort_values("exit_date")[["exit_date", "return_pct"]]
            .copy())
eq_curve["cum_return"] = eq_curve["return_pct"].cumsum()

total_years     = (pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25
trades_per_year = len(trades_df) / total_years

# VIX regime days count for context
vix_regime_days = (vix >= 25).sum()
print(f"  VIX >= 25 days: {vix_regime_days} out of {len(vix)} total days "
      f"({vix_regime_days/len(vix)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────
# 8.  VISUALIZATION
# ─────────────────────────────────────────────────────────────
print("Step 7: Plotting ...")

fig = plt.figure(figsize=(20, 18))
gs  = GridSpec(4, 2, figure=fig, hspace=0.48, wspace=0.33)

# ── (a) Equity Curve ──────────────────────────────────────────
ax_eq = fig.add_subplot(gs[0, :])
ax_eq.plot(
    eq_curve["exit_date"],
    eq_curve["cum_return"] * 100,
    color="#2563eb", linewidth=1.8,
    label="Cumulative P&L (non-compounding, equal weight)"
)
ax_eq.axhline(0, color="black", linewidth=0.8, linestyle="--")

crisis_zones = [
    ("2011-08-01", "2011-10-31", "#94a3b8", "2011 US downgrade"),
    ("2015-08-01", "2016-02-29", "#6366f1", "2015–16 China slump"),
    ("2018-10-01", "2019-01-15", "#f87171", "2018Q4"),
    ("2020-02-20", "2020-04-30", "#fb923c", "COVID crash"),
    ("2022-01-01", "2022-12-31", "#facc15", "2022 bear"),
    ("2025-01-01", "2025-06-30", "#a78bfa", "2025 tension"),
]
for s, e, col, lbl in crisis_zones:
    ax_eq.axvspan(pd.Timestamp(s), min(pd.Timestamp(e), pd.Timestamp(END)),
                  alpha=0.18, color=col, label=lbl)

ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_eq.xaxis.set_major_locator(mdates.YearLocator())
ax_eq.set_title(
    "VIX-Oversold Reversal  —  Cumulative P&L (non-compounding, equal weight per trade)",
    fontsize=13, fontweight="bold"
)
ax_eq.set_ylabel("Cumulative Return (%)")
ax_eq.legend(fontsize=8, loc="upper left", ncol=4)
ax_eq.grid(alpha=0.3)

# ── (b) VIX with regime shading ───────────────────────────────
ax_vix = fig.add_subplot(gs[1, :])
ax_vix.plot(vix.index, vix.values, color="#dc2626", linewidth=1.0, label="VIX")
ax_vix.axhline(25, color="#f97316", linewidth=1.2, linestyle="--",
               label="VIX = 25 (threshold)")
ax_vix.fill_between(vix.index, vix.values, 25,
                    where=(vix.values >= 25),
                    alpha=0.25, color="#f97316", label="VIX ≥ 25")
# Mark entry dates
entry_vix = vix.reindex(trades_df["entry_date"].values, method="nearest")
ax_vix.scatter(entry_vix.index, entry_vix.values,
               color="#1d4ed8", s=12, zorder=5, alpha=0.7, label="Entries")
ax_vix.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_vix.xaxis.set_major_locator(mdates.YearLocator())
ax_vix.set_title("VIX Level & Strategy Entry Points", fontsize=11, fontweight="bold")
ax_vix.set_ylabel("VIX")
ax_vix.legend(fontsize=8, loc="upper right", ncol=4)
ax_vix.grid(alpha=0.3)

# ── (c) Return distribution ───────────────────────────────────
ax_hist = fig.add_subplot(gs[2, 0])
ret_pct = trades_df["return_pct"] * 100
bins = np.linspace(ret_pct.min() - 0.5, ret_pct.max() + 0.5, 45)
n_win  = (ret_pct > 0).sum()
n_loss = (ret_pct <= 0).sum()
ax_hist.hist(ret_pct[ret_pct > 0], bins=bins,
             color="#22c55e", edgecolor="white", linewidth=0.4,
             alpha=0.8, label=f"Win ({n_win})")
ax_hist.hist(ret_pct[ret_pct <= 0], bins=bins,
             color="#ef4444", edgecolor="white", linewidth=0.4,
             alpha=0.8, label=f"Loss ({n_loss})")
ax_hist.axvline(ret_pct.mean(), color="#1e3a8a", linestyle="--",
                linewidth=1.5, label=f"Mean {ret_pct.mean():.1f}%")
ax_hist.axvline(0, color="black", linewidth=0.8)
ax_hist.set_title("Per-Trade Return Distribution", fontsize=11, fontweight="bold")
ax_hist.set_xlabel("Return (%)")
ax_hist.set_ylabel("# Trades")
ax_hist.legend(fontsize=9)
ax_hist.grid(alpha=0.3)

# ── (d) Hold days ─────────────────────────────────────────────
ax_hold = fig.add_subplot(gs[2, 1])
ax_hold.hist(trades_df["hold_days"], bins=range(1, MAX_HOLD + 3),
             color="#0ea5e9", edgecolor="white", linewidth=0.4, rwidth=0.85)
ax_hold.set_title("Hold-Days Distribution", fontsize=11, fontweight="bold")
ax_hold.set_xlabel("Hold Days (trading)")
ax_hold.set_ylabel("# Trades")
ax_hold.grid(alpha=0.3)

# ── (e) Per-year stats ────────────────────────────────────────
ax_yr = fig.add_subplot(gs[3, 0])
yr_df = year_stats_df[year_stats_df["N_trades"] > 0].copy()
colors_yr = ["#22c55e" if x > 0 else "#ef4444"
             for x in yr_df["Avg_return_%"]]
bars = ax_yr.bar(yr_df["Label"].astype(str), yr_df["Avg_return_%"],
                 color=colors_yr, edgecolor="white", linewidth=0.5)
ax_yr.axhline(0, color="black", linewidth=0.8)
for bar, n in zip(bars, yr_df["N_trades"]):
    ax_yr.text(bar.get_x() + bar.get_width() / 2,
               bar.get_height() + (0.1 if bar.get_height() >= 0 else -0.5),
               f"n={n}", ha="center", va="bottom", fontsize=7)
ax_yr.set_title("Avg Return by Year", fontsize=11, fontweight="bold")
ax_yr.set_ylabel("Avg Return (%)")
ax_yr.tick_params(axis="x", rotation=45)
ax_yr.grid(axis="y", alpha=0.3)

# ── (f) Outcome pie ───────────────────────────────────────────
ax_pie = fig.add_subplot(gs[3, 1])
outcomes    = trades_df["outcome"].value_counts()
colors_pie  = {"target": "#22c55e", "stop": "#ef4444", "time_exit": "#f59e0b"}
pie_colors  = [colors_pie.get(k, "#94a3b8") for k in outcomes.index]
wedges, texts, autotexts = ax_pie.pie(
    outcomes.values, labels=outcomes.index,
    autopct="%1.1f%%", colors=pie_colors,
    startangle=90, textprops={"fontsize": 10},
)
ax_pie.set_title("Outcome Distribution", fontsize=11, fontweight="bold")

plt.suptitle(
    "VIX-Oversold Reversal Strategy  |  Nasdaq-100  |  2010–2026",
    fontsize=15, fontweight="bold", y=1.005
)
plt.savefig("backtest_vix_oversold_result.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Chart saved: backtest_vix_oversold_result.png")

# ─────────────────────────────────────────────────────────────
# 9.  PRINT FULL TRADE TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Full Trade List")
print("=" * 65)
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_rows", 500)
print(trades_df[["ticker","entry_date","exit_date",
                 "entry_price","exit_price","return_pct","hold_days","outcome"]]
      .assign(return_pct=lambda x: (x["return_pct"]*100).round(2))
      .rename(columns={"return_pct":"ret_%"})
      .to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 10.  TEXTUAL INTERPRETATION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("INTERPRETATION")
print("=" * 65)

n_t   = overall.get("N_trades", 0)
win_r = overall.get("Win_rate_%", 0)
avg_r = overall.get("Avg_return_%", 0)
med_r = overall.get("Median_return_%", 0)
std_r = overall.get("Std_return_%", 0)
n_tgt = overall.get("N_target", 0)
n_stp = overall.get("N_stop", 0)
n_tim = overall.get("N_time_exit", 0)

edge_flag = "★ 양수 기대값 + 승률 > 50%" if (avg_r > 0 and win_r > 50) else "△ 개선 여지 있음"

print(f"""
[1] 전략 설정값
  - Stop    : -{STOP_PCT*100:.0f}%  (entry × 0.95)
  - Target  : +{TARGET_PCT*100:.0f}%  (entry × 1.20)
  - R:R     : 1 : {TARGET_PCT/STOP_PCT:.0f}  (손익비 1:4)
  - Max hold: {MAX_HOLD} 거래일

[2] 전체 성과
  - 총 트레이드   : {n_t}건
  - 연평균        : {trades_per_year:.1f}건/년
  - 승률          : {win_r:.1f}%
  - 평균 수익률   : {avg_r:.2f}%
  - 중앙값 수익률 : {med_r:.2f}%
  - 표준편차      : {std_r:.2f}%
  → {edge_flag}

[3] Outcome 분포
  - Target (+20%) : {n_tgt}건 ({n_tgt/max(n_t,1)*100:.1f}%)
  - Stop   (-5%)  : {n_stp}건 ({n_stp/max(n_t,1)*100:.1f}%)
  - Time exit     : {n_tim}건 ({n_tim/max(n_t,1)*100:.1f}%)

  손익비 1:4 이므로 이론적 손익분기 승률 = 1/(1+4) = 20%.
  → 실제 승률이 20%를 초과하면 기대값 양수.

[4] 위기 구간별 성과
""")

crisis_years = {
    2011: "2011 (미국 신용등급 강등)",
    2015: "2015 (중국 쇼크)",
    2016: "2016 (브렉시트/유가 급락)",
    2018: "2018 Q4 (Fed 인상/무역전쟁)",
    2020: "2020 (COVID)",
    2022: "2022 (금리 급등 베어마켓)",
    2024: "2024",
    2025: "2025 (관세 전쟁)",
}
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
[5] 전략 특성 요약
  - VIX ≥ 25 & 전일比 VIX 하락: 공포 피크 이후 진정 국면 포착
  - MA200 -30% 이내: 완전 붕괴 종목 제외, 반등 가능 범위
  - RSI/CCI 과매도: 극단적 매도세 확인
  - MACD_H 개선: 하락 모멘텀 약화 신호
  - 거래량 1.5×: 세력 개입 또는 패닉셀 소화 확인
  - 양봉: 당일 매수세 우위
  - 손익비 1:4: 승률 20% 이상이면 기대값 양수
    (즉, 5번 중 1번만 +20% 달성해도 4번 -5% 만회)

[주의]
  - 표본 수에 따라 통계적 신뢰도가 달라집니다.
  - Stop=5% 고정이므로 변동성이 높은 종목은 ATR 기반 스탑 검토 권장.
  - 진입가 = 당일 종가 가정 (gap 리스크 미반영).
  - Walk-forward / Out-of-sample 검증 필수.
""")

print("=" * 65)
print("Backtest complete.")
print("=" * 65)
