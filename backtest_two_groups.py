"""
backtest_two_groups.py
======================
그룹 A (우량 대형주 11개) vs 그룹 B (레버리지ETF + 스타트업/소형주)
v10 전략 비교 백테스트
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

START = "2010-01-01"
END   = "2026-01-01"

VIX_MIN      = 25
TARGET_PCT   = 0.20
CIRCUIT_PCT  = 0.25
HALF_EXIT    = 60
MAX_HOLD     = 120
MAX_POSITIONS= 5
MAX_DAILY    = 5

GROUPS = {
    "A_우량대형주": [
        "NVDA","AVGO","GOOGL","TSLA","AMZN","MSFT","AAPL","META","MU","STX","PLTR"
    ],
    "B_기타": [
        "IONQ","QBTS","BBAI","RKLB","SLDP","IREN","FLNC","AVAV",
        "ONDS","ASTS","OKLO","BE","PL","SGML","SKYT","LPTH","SNDK",
        "TQQQ","TSLL","SOXL","ETHU",
    ],
}

# ──────────────────────────────────────────────
def dl_ohlcv(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw[["Open","High","Low","Close","Volume"]].copy()

def compute(d):
    d = d.copy()
    c = d["Close"]
    d["MA200"] = c.rolling(200).mean()
    delta = c.diff()
    g = delta.clip(lower=0).rolling(14).mean()
    l = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100 / (1 + g / l.replace(0, np.nan))
    tp     = (d["High"] + d["Low"] + c) / 3
    tp_ma  = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    d["CCI"] = (tp - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))
    return d

def run_backtest(tickers, vix, label):
    print(f"\n{'='*55}")
    print(f"  [{label}] 백테스트 ({len(tickers)}개 종목)")
    print(f"{'='*55}")

    stock_data = {}
    for tk in tickers:
        d = dl_ohlcv(tk, START, END)
        if len(d) < 250:
            print(f"  ⚠️  {tk}: 데이터 부족, 스킵")
            continue
        d = compute(d)
        stock_data[tk] = d
    print(f"  ✅ {len(stock_data)}개 종목 로드")

    signals_by_date = {}
    for tk, d in stock_data.items():
        d_c    = d.dropna(subset=["MA200","RSI","CCI"])
        close  = d_c["Close"]
        rsi    = d_c["RSI"]
        cci    = d_c["CCI"]
        common = d_c.index.intersection(vix.index)
        if len(common) < 50:
            continue
        vx   = vix.reindex(common)
        cond = (
            (close < d_c["MA200"]) &
            (vx >= VIX_MIN) &
            ((rsi < 40) | (cci < -100))
        )
        sig_days = d_c.index[cond.reindex(d_c.index).fillna(False)]
        for sig_day in sig_days:
            idx = d_c.index.get_loc(sig_day)
            if idx + 1 >= len(d_c):
                continue
            entry_day  = d_c.index[idx + 1]
            entry_open = float(d_c["Open"].iloc[idx + 1])
            if pd.isna(entry_open):
                continue
            row = d_c.loc[sig_day]
            if entry_day not in signals_by_date:
                signals_by_date[entry_day] = []
            signals_by_date[entry_day].append({
                "ticker"   : tk,
                "entry_day": entry_day,
                "entry"    : entry_open,
                "rsi"      : float(row["RSI"]),
                "cci"      : float(row["CCI"]),
                "close_sig": float(row["Close"]),
            })

    final_signals = []
    for entry_day, items in sorted(signals_by_date.items()):
        items.sort(key=lambda x: x["rsi"])
        for item in items[:MAX_DAILY]:
            final_signals.append(item)

    trades        = []
    pos_exit_date = {}

    for sig in final_signals:
        tk        = sig["ticker"]
        entry_day = sig["entry_day"]
        entry     = sig["entry"]

        active = {t: ed for t, ed in pos_exit_date.items() if ed > entry_day}
        if len(active) >= MAX_POSITIONS:
            continue
        if tk in active:
            continue

        d      = stock_data[tk]
        future = d.loc[d.index >= entry_day]
        if len(future) == 0:
            continue

        circuit      = entry * (1 - CIRCUIT_PCT)
        target       = entry * (1 + TARGET_PCT)
        half_exited  = False
        exit_records = []

        for i, (fdt, row) in enumerate(future.iterrows()):
            lo  = float(row["Low"])
            hi  = float(row["High"])
            cl  = float(row["Close"])
            ret = (cl - entry) / entry

            if hi >= target:
                exit_records.append((fdt, target, 0.5 if half_exited else 1.0, "target"))
                break
            if lo <= circuit:
                exit_records.append((fdt, circuit, 0.5 if half_exited else 1.0, "circuit"))
                break
            if i + 1 == HALF_EXIT and not half_exited and ret > 0:
                exit_records.append((fdt, cl, 0.5, "half_60d"))
                half_exited = True
                continue
            if i + 1 >= MAX_HOLD:
                exit_records.append((fdt, cl, 0.5 if half_exited else 1.0, "time"))
                break

        if not exit_records:
            continue

        total_pct   = sum(r[2] for r in exit_records)
        weighted    = sum((r[1] - entry) / entry * r[2] for r in exit_records)
        blended_ret = weighted / total_pct if total_pct > 0 else 0
        last_exit   = exit_records[-1]
        reason      = ("+".join(r[3] for r in exit_records)
                       if len(exit_records) > 1 else exit_records[0][3])

        trades.append({
            "entry_date" : entry_day,
            "exit_date"  : last_exit[0],
            "ticker"     : tk,
            "entry"      : entry,
            "return_pct" : blended_ret * 100,
            "hold_days"  : (last_exit[0] - entry_day).days,
            "exit_reason": reason,
            "win"        : blended_ret > 0,
        })
        pos_exit_date[tk] = last_exit[0]

    if not trades:
        print(f"  ⚠️  트레이드 없음")
        return None

    df   = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)
    wins = df[df["win"]]
    loss = df[~df["win"]]
    n    = len(df)

    win_rate = len(wins) / n * 100
    avg_ret  = df["return_pct"].mean()
    avg_win  = wins["return_pct"].mean() if len(wins) else 0
    avg_loss = loss["return_pct"].mean() if len(loss) else 0
    pf       = (wins["return_pct"].sum() / -loss["return_pct"].sum()
                if len(loss) and loss["return_pct"].sum() < 0 else np.nan)
    ev       = win_rate/100 * avg_win + (1 - win_rate/100) * avg_loss

    max_cl = cur = 0
    for r in df["return_pct"]:
        cur = cur + 1 if r < 0 else 0
        max_cl = max(max_cl, cur)

    capital = 1.0
    for ed, group in df.groupby("entry_date"):
        w     = 1.0 / max(len(group), MAX_POSITIONS)
        batch = (group["return_pct"] / 100 * w).sum()
        capital *= (1 + batch)
    years   = (df["exit_date"].max() - df["entry_date"].min()).days / 365.25
    cagr    = capital ** (1 / years) - 1

    val_5y  = 1e8 * ((1 + cagr) ** 5)
    val_10y = 1e8 * ((1 + cagr) ** 10)
    val_15y = 1e8 * ((1 + cagr) ** 15)

    exit_cnt = df["exit_reason"].value_counts()

    print(f"  트레이드: {n}건 | 승률: {win_rate:.1f}% | "
          f"평균수익: {avg_ret:+.2f}% | CAGR: {cagr*100:+.1f}%")

    # 종목별 성과
    by_ticker = df.groupby("ticker").agg(
        건수=("return_pct","count"),
        승률=("win", lambda x: round(x.mean()*100,1)),
        평균수익=("return_pct", lambda x: round(x.mean(),2)),
    ).sort_values("평균수익", ascending=False)

    return {
        "label"    : label,
        "n"        : n,
        "win_rate" : win_rate,
        "avg_ret"  : avg_ret,
        "avg_win"  : avg_win,
        "avg_loss" : avg_loss,
        "pf"       : pf,
        "ev"       : ev,
        "cagr"     : cagr * 100,
        "max_cl"   : max_cl,
        "val_5y"   : val_5y,
        "val_10y"  : val_10y,
        "val_15y"  : val_15y,
        "exit_cnt" : exit_cnt,
        "by_ticker": by_ticker,
        "df"       : df,
    }

# ──────────────────────────────────────────────
print("📥 VIX 다운로드...")
vix_raw = yf.download("^VIX", start=START, end=END,
                      auto_adjust=True, progress=False)
_vc = vix_raw["Close"]
if isinstance(_vc, pd.DataFrame): _vc = _vc.iloc[:, 0]
vix = pd.Series(_vc.values, index=_vc.index, dtype=float).dropna()
print(f"✅ VIX {len(vix)}일")

results = {}
for label, tickers in GROUPS.items():
    r = run_backtest(tickers, vix, label)
    if r:
        results[label] = r

# ──────────────────────────────────────────────
# 결과 출력
# ──────────────────────────────────────────────
print("\n\n" + "="*65)
print("  그룹 A (우량 대형주 11개) vs 그룹 B (기타)")
print("  v10 전략 | 2010-01-01 ~ 2026-01-01")
print("="*65)

fmt = "  {:<20} {:>10} {:>8} {:>8} {:>8} {:>8}"
print(fmt.format("구분", "트레이드", "승률", "평균수익", "PF", "CAGR"))
print("  " + "-"*60)
for label, r in results.items():
    pf_str = f"{r['pf']:.2f}" if not np.isnan(r['pf']) else "N/A"
    name   = "A. 우량 대형주" if "A" in label else "B. 기타"
    print(fmt.format(name, f"{r['n']}건",
                     f"{r['win_rate']:.1f}%",
                     f"{r['avg_ret']:+.2f}%",
                     pf_str,
                     f"{r['cagr']:+.1f}%"))

print("\n" + "="*65)
print("  1억 투자 시 N년 후 자산")
print("="*65)
fmt2 = "  {:<20} {:>14} {:>14} {:>14}"
print(fmt2.format("구분", "5년 후", "10년 후", "15년 후"))
print("  " + "-"*60)
for label, r in results.items():
    name = "A. 우량 대형주" if "A" in label else "B. 기타"
    print(fmt2.format(name,
                      f"{r['val_5y']/1e8:.2f}억",
                      f"{r['val_10y']/1e8:.2f}억",
                      f"{r['val_15y']/1e8:.2f}억"))

for label, r in results.items():
    name = "A. 우량 대형주" if "A" in label else "B. 기타"
    print(f"\n{'='*65}")
    print(f"  [{name}] 상세")
    print(f"{'='*65}")
    print(f"  트레이드   : {r['n']}건")
    print(f"  승률       : {r['win_rate']:.1f}%")
    print(f"  평균 수익  : {r['avg_ret']:+.2f}%")
    print(f"  승자 평균  : {r['avg_win']:+.2f}%")
    print(f"  패자 평균  : {r['avg_loss']:+.2f}%")
    pf_str = f"{r['pf']:.2f}" if not np.isnan(r['pf']) else "N/A"
    print(f"  PF         : {pf_str}")
    print(f"  기대값     : {r['ev']:+.2f}%")
    print(f"  CAGR       : {r['cagr']:+.1f}%")
    print(f"  최대연속손실: {r['max_cl']}건")
    print(f"  1억→5년    : {r['val_5y']/1e8:.2f}억")
    print(f"  1억→10년   : {r['val_10y']/1e8:.2f}억")
    print(f"  1억→15년   : {r['val_15y']/1e8:.2f}억")
    print(f"\n  청산 유형:")
    for reason, cnt in r["exit_cnt"].items():
        print(f"    {reason:<25}: {cnt:4d}건 ({cnt/r['n']*100:.1f}%)")
    print(f"\n  종목별 성과 (평균수익 내림차순):")
    print(r["by_ticker"].to_string())

# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    "v10 전략: 그룹A (우량 대형주 11개) vs 그룹B (기타)\n"
    "2010-2026 | MA200↓ + VIX≥25 + RSI<40 OR CCI<-100",
    fontsize=12, fontweight="bold"
)

colors = {"A_우량대형주": "#2ecc71", "B_기타": "#e74c3c"}
labels = list(results.keys())
names  = ["A. 우량 대형주", "B. 기타"]

ax = axes[0, 0]
cagrs = [results[l]["cagr"] for l in labels]
bars  = ax.bar(names, cagrs, color=[colors[l] for l in labels], edgecolor="white", width=0.5)
ax.axhline(0, color="black", lw=0.8)
for bar, val in zip(bars, cagrs):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
            f"{val:+.1f}%", ha="center", fontsize=13, fontweight="bold")
ax.set_title("CAGR 비교", fontsize=11)
ax.set_ylabel("%")

ax = axes[0, 1]
wrs  = [results[l]["win_rate"] for l in labels]
bars = ax.bar(names, wrs, color=[colors[l] for l in labels], edgecolor="white", width=0.5)
ax.axhline(50, color="red", lw=1, linestyle="--", label="50%")
for bar, val in zip(bars, wrs):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
            f"{val:.1f}%", ha="center", fontsize=13, fontweight="bold")
ax.set_title("승률 비교", fontsize=11)
ax.legend()

ax = axes[0, 2]
val10s = [results[l]["val_10y"] / 1e8 for l in labels]
bars   = ax.bar(names, val10s, color=[colors[l] for l in labels], edgecolor="white", width=0.5)
for bar, val in zip(bars, val10s):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
            f"{val:.1f}억", ha="center", fontsize=13, fontweight="bold")
ax.set_title("1억 → 10년 후 (억원)", fontsize=11)
ax.set_ylabel("억원")

ax = axes[1, 0]
for l, name in zip(labels, names):
    ax.hist(results[l]["df"]["return_pct"], bins=30, alpha=0.6,
            label=name, color=colors[l])
ax.axvline(0, color="black", lw=1.5, linestyle="--")
ax.set_title("수익률 분포 비교", fontsize=11)
ax.legend()

ax = axes[1, 1]
for l, name in zip(labels, names):
    cum = (1 + results[l]["df"]["return_pct"] / 100).cumprod() - 1
    ax.plot(range(len(cum)), cum * 100,
            label=f"{name} (CAGR {results[l]['cagr']:+.1f}%)",
            color=colors[l], lw=2)
ax.axhline(0, color="black", lw=0.8, linestyle="--")
ax.set_title("누적 수익률", fontsize=11)
ax.set_xlabel("Trade #")
ax.legend(fontsize=9)

ax = axes[1, 2]
x     = np.arange(3)
width = 0.35
for j, (l, name) in enumerate(zip(labels, names)):
    vals = [results[l]["val_5y"]/1e8,
            results[l]["val_10y"]/1e8,
            results[l]["val_15y"]/1e8]
    bars = ax.bar(x + j*width, vals, width,
                  label=name, color=colors[l], edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                f"{val:.1f}", ha="center", fontsize=8)
ax.set_xticks(x + width/2)
ax.set_xticklabels(["5년 후", "10년 후", "15년 후"])
ax.set_title("1억 → N년 후 자산 (억원)", fontsize=11)
ax.legend()

plt.tight_layout()
plt.savefig("backtest_two_groups.png", dpi=150, bbox_inches="tight")
print("\n\n📊 backtest_two_groups.png 저장 완료")
print("✅ 완료")
