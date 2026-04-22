"""
backtest_v11_ticker_analysis.py
================================
v11 조건 (RSI<35 OR CCI<-150) 으로 전체 종목 성과 분석

1. 원본 v10 vs v11 전체 성과 비교
2. 종목별 상세 성과 분석 (제거 후보 도출)
   - 승률 < 60%
   - 평균 수익률 < 0%
   - 서킷브레이커 비중 높음
   - 거래 건수 극소 (기회 없음)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

START         = "2010-01-01"
END           = "2026-01-01"
VIX_MIN       = 25
TARGET_PCT    = 0.20
CIRCUIT_PCT   = 0.25
HALF_EXIT     = 60
MAX_HOLD      = 120
MAX_POSITIONS = 5
MAX_DAILY     = 5

# 전체 모니터링 종목 (사용자 리스트)
TICKERS = sorted(set([
    "SNPS","COST","AZN","AMGN","MDLZ","FTNT","CSGP","CDNS","ADP","FAST",
    "ADI","TXN","PAYX","BKNG","KLAC","MNST","ORLY","HOOD","CPRT","ISRG",
    "PANW","CDW","INTC","AAPL","AVGO","AMD","MSFT","GOOGL","NVDA","PLTR",
    "TSLA","META","MELI","MCHP","AMZN","SMCI","AMAT","MU","LRCX","CSX",
    "QCOM","ROP","INTU","ON","NXPI","STX","SNDK","ASTS","AVAV","IONQ",
    "SGML","RKLB",
]))

# 비교 조건
VERSIONS = {
    "v10 원본 (RSI<40 OR CCI<-100)": {"rsi": 40, "cci": -100},
    "v11 개선 (RSI<35 OR CCI<-150)": {"rsi": 35, "cci": -150},
}

# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────
def dl_ohlcv(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    if raw.empty: return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw[["Open","High","Low","Close","Volume"]].copy()

def compute(d):
    d = d.copy(); c = d["Close"]
    d["MA200"] = c.rolling(200).mean()
    delta = c.diff()
    g = delta.clip(lower=0).rolling(14).mean()
    l = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100 / (1 + g / l.replace(0, np.nan))
    tp = (d["High"] + d["Low"] + c) / 3
    tp_ma  = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    d["CCI"] = (tp - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))
    return d

# ──────────────────────────────────────────────
# 데이터
# ──────────────────────────────────────────────
print("📥 다운로드 중...")
vix_raw = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)
_vc = vix_raw["Close"]
if isinstance(_vc, pd.DataFrame): _vc = _vc.iloc[:, 0]
vix = pd.Series(_vc.values, index=_vc.index, dtype=float).dropna()

stock_data = {}
skip_tickers = []
for i, tk in enumerate(TICKERS, 1):
    d = dl_ohlcv(tk, START, END)
    if len(d) < 250:
        skip_tickers.append(tk)
        continue
    stock_data[tk] = compute(d)
    if i % 10 == 0: print(f"  {i}/{len(TICKERS)} 완료")
print(f"✅ VIX {len(vix)}일 | 종목 {len(stock_data)}개 로드")
if skip_tickers:
    print(f"⚠️  데이터 부족 종목 (제외): {skip_tickers}")

# ──────────────────────────────────────────────
# 신호 + 시뮬레이션
# ──────────────────────────────────────────────
def build_and_run(rsi_thresh, cci_thresh, label):
    signals_by_date = {}
    for tk, d in stock_data.items():
        d_c    = d.dropna(subset=["MA200","RSI","CCI"])
        common = d_c.index.intersection(vix.index)
        if len(common) < 50: continue
        vx    = vix.reindex(common)
        close = d_c["Close"].reindex(common)
        ma200 = d_c["MA200"].reindex(common)
        rsi   = d_c["RSI"].reindex(common)
        cci   = d_c["CCI"].reindex(common)
        cond  = (close < ma200) & (vx >= VIX_MIN) & ((rsi < rsi_thresh) | (cci < cci_thresh))
        for sig_day in common[cond.reindex(common).fillna(False)]:
            idx = d_c.index.get_loc(sig_day)
            if idx + 1 >= len(d_c): continue
            entry_day  = d_c.index[idx + 1]
            entry_open = float(d_c["Open"].iloc[idx + 1])
            if pd.isna(entry_open): continue
            row = d_c.loc[sig_day]
            signals_by_date.setdefault(entry_day, []).append({
                "ticker": tk, "entry_day": entry_day, "entry": entry_open,
                "rsi": float(row["RSI"]), "cci": float(row["CCI"]),
            })
    signals = []
    for entry_day, items in sorted(signals_by_date.items()):
        items.sort(key=lambda x: x["rsi"])
        signals.extend(items[:MAX_DAILY])

    trades = []; pos_exit_date = {}
    for sig in signals:
        tk = sig["ticker"]; entry_day = sig["entry_day"]; entry = sig["entry"]
        active = {t: ed for t, ed in pos_exit_date.items() if ed > entry_day}
        if len(active) >= MAX_POSITIONS or tk in active: continue
        d = stock_data[tk]; future = d.loc[d.index >= entry_day]
        if len(future) == 0: continue
        circuit = entry * (1 - CIRCUIT_PCT); target = entry * (1 + TARGET_PCT)
        half_exited = False; exit_records = []
        for i, (fdt, row) in enumerate(future.iterrows()):
            lo = float(row["Low"]); hi = float(row["High"]); cl = float(row["Close"])
            if hi >= target:
                exit_records.append((fdt, target, 0.5 if half_exited else 1.0, "target")); break
            if lo <= circuit:
                exit_records.append((fdt, circuit, 0.5 if half_exited else 1.0, "circuit")); break
            if i + 1 == HALF_EXIT and not half_exited and (cl - entry) / entry > 0:
                exit_records.append((fdt, cl, 0.5, "half_60d")); half_exited = True; continue
            if i + 1 >= MAX_HOLD:
                exit_records.append((fdt, cl, 0.5 if half_exited else 1.0, "time")); break
        if not exit_records: continue
        total_pct   = sum(r[2] for r in exit_records)
        blended_ret = sum((r[1]-entry)/entry * r[2] for r in exit_records) / total_pct
        last_exit   = exit_records[-1]
        reason      = "+".join(r[3] for r in exit_records) if len(exit_records) > 1 else exit_records[0][3]
        trades.append({
            "entry_date": entry_day, "exit_date": last_exit[0],
            "ticker": tk, "entry": entry, "exit_price": last_exit[1],
            "return_pct": blended_ret * 100,
            "hold_days": (last_exit[0] - entry_day).days,
            "exit_reason": reason, "win": blended_ret > 0,
        })
        pos_exit_date[tk] = last_exit[0]

    df = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)
    print(f"  [{label}] 신호 {len(signals):,}건 → 트레이드 {len(df)}건")
    return df

# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────
print("\n⚙️  시뮬레이션...")
dfs = {}
for label, params in VERSIONS.items():
    dfs[label] = build_and_run(params["rsi"], params["cci"], label)

# ──────────────────────────────────────────────
# 전체 통계
# ──────────────────────────────────────────────
def calc_stats(df):
    if df.empty: return {}
    wins = df[df["win"]]; losses = df[~df["win"]]; n = len(df)
    wr  = len(wins) / n * 100
    ar  = df["return_pct"].mean()
    aw  = wins["return_pct"].mean()   if len(wins)   else 0
    al  = losses["return_pct"].mean() if len(losses) else 0
    pf  = (wins["return_pct"].sum() / -losses["return_pct"].sum()
           if len(losses) and losses["return_pct"].sum() < 0 else np.nan)
    ev  = wr/100 * aw + (1 - wr/100) * al
    max_cl = cur = 0
    for r in df["return_pct"]:
        cur = cur+1 if r < 0 else 0; max_cl = max(max_cl, cur)
    capital = 1.0
    for _, g in df.groupby("entry_date"):
        capital *= (1 + (g["return_pct"]/100 * (1/max(len(g), MAX_POSITIONS))).sum())
    years = (df["exit_date"].max() - df["entry_date"].min()).days / 365.25
    cagr  = capital ** (1/max(years, 0.01)) - 1
    return {"n": n, "wr": wr, "ar": ar, "aw": aw, "al": al,
            "pf": pf, "ev": ev, "max_cl": max_cl, "cagr": cagr*100,
            "avg_hold": df["hold_days"].mean()}

# ──────────────────────────────────────────────
# 종목별 성과 분석 (v11 기준)
# ──────────────────────────────────────────────
df_v11 = dfs["v11 개선 (RSI<35 OR CCI<-150)"]
df_v10 = dfs["v10 원본 (RSI<40 OR CCI<-100)"]

def ticker_stats(df):
    if df.empty: return pd.DataFrame()
    g = df.groupby("ticker")
    result = g.agg(
        trades   =("return_pct", "count"),
        avg_ret  =("return_pct", "mean"),
        win_rate =("win",        lambda x: x.mean() * 100),
        avg_hold =("hold_days",  "mean"),
        total_ret=("return_pct", "sum"),
    )
    result["circuit_n"] = df[df["exit_reason"].str.contains("circuit")].groupby("ticker").size()
    result["circuit_n"] = result["circuit_n"].fillna(0).astype(int)
    result["circuit_pct"] = result["circuit_n"] / result["trades"] * 100
    result["min_ret"] = df.groupby("ticker")["return_pct"].min()
    result["max_ret"] = df.groupby("ticker")["return_pct"].max()
    return result.sort_values("avg_ret", ascending=False)

ts_v11 = ticker_stats(df_v11)
ts_v10 = ticker_stats(df_v10)

# ──────────────────────────────────────────────
# 결과 출력 — 전체 비교
# ──────────────────────────────────────────────
v10s = calc_stats(df_v10); v11s = calc_stats(df_v11)
pf_s = lambda v: f"{v:.2f}" if not (isinstance(v,float) and np.isnan(v)) else "N/A"

print("\n" + "="*75)
print("  v10 원본 vs v11 개선 (RSI<35/CCI<-150) 전체 성과 비교")
print("="*75)
print(f"  기간: {START}~{END}  |  종목 {len(stock_data)}개  |  MA200↓ + VIX≥25")
print("="*75)
fmt = "  {:<22} {:>22} {:>22}"
print(fmt.format("지표", "v10 (RSI<40/CCI<-100)", "v11 (RSI<35/CCI<-150)"))
print("  " + "-"*66)
for label, v10v, v11v in [
    ("총 트레이드",     f"{v10s['n']}건",              f"{v11s['n']}건"),
    ("승 률",           f"{v10s['wr']:.1f}%",           f"{v11s['wr']:.1f}%"),
    ("평균 수익률",     f"{v10s['ar']:+.2f}%",          f"{v11s['ar']:+.2f}%"),
    ("기대값(EV)",      f"{v10s['ev']:+.2f}%",          f"{v11s['ev']:+.2f}%"),
    ("승자 평균",       f"{v10s['aw']:+.2f}%",          f"{v11s['aw']:+.2f}%"),
    ("패자 평균",       f"{v10s['al']:+.2f}%",          f"{v11s['al']:+.2f}%"),
    ("Profit Factor",   pf_s(v10s['pf']),               pf_s(v11s['pf'])),
    ("포트CAGR",        f"{v10s['cagr']:+.2f}%",        f"{v11s['cagr']:+.2f}%"),
    ("최대 연속 손실",  f"{v10s['max_cl']}건",           f"{v11s['max_cl']}건"),
    ("평균 보유 일수",  f"{v10s['avg_hold']:.0f}일",     f"{v11s['avg_hold']:.0f}일"),
]:
    print(fmt.format(label, v10v, v11v))
print("="*75)

# ──────────────────────────────────────────────
# 종목별 성과 전체 출력 (v11)
# ──────────────────────────────────────────────
print(f"\n{'='*85}")
print(f"  v11 기준 종목별 성과 (총 {len(ts_v11)}개 종목)")
print(f"{'='*85}")
print(f"  {'종목':<8} {'거래':>5} {'승률':>7} {'평균수익':>9} {'평균보유':>8} {'서킷%':>6} {'최소':>8} {'최대':>8}")
print(f"  {'-'*75}")
for tk, row in ts_v11.iterrows():
    flag = ""
    if row["win_rate"] < 60:             flag += " ⚠️ 승률저조"
    if row["avg_ret"] < 0:               flag += " 🔴 평균손실"
    if row["circuit_pct"] >= 20:         flag += " 💥 서킷다발"
    if row["trades"] <= 2:               flag += " 📉 기회부족"
    print(f"  {tk:<8} {int(row['trades']):>5} {row['win_rate']:>6.1f}% "
          f"{row['avg_ret']:>+8.2f}% {row['avg_hold']:>7.0f}일 "
          f"{row['circuit_pct']:>5.1f}% {row['min_ret']:>+7.2f}% {row['max_ret']:>+7.2f}%"
          f"{flag}")

# ──────────────────────────────────────────────
# 제거 권고 분석
# ──────────────────────────────────────────────
print(f"\n{'='*75}")
print("  제거 권고 종목 분석")
print(f"{'='*75}")

# 기준: v11과 v10 모두에서 성과 불량 종목
remove_candidates = {}

for tk in ts_v11.index:
    reasons = []
    row11 = ts_v11.loc[tk]

    # 기준 1: 승률 < 60%
    if row11["win_rate"] < 60:
        reasons.append(f"승률 {row11['win_rate']:.0f}%")

    # 기준 2: 평균 수익률 음수
    if row11["avg_ret"] < 0:
        reasons.append(f"평균수익 {row11['avg_ret']:+.1f}%")

    # 기준 3: 서킷브레이커 비중 25% 이상
    if row11["circuit_pct"] >= 25:
        reasons.append(f"서킷 {row11['circuit_pct']:.0f}%")

    # 기준 4: 거래 2건 이하 (기회 자체가 없음)
    if row11["trades"] <= 2:
        reasons.append(f"거래 {int(row11['trades'])}건")

    # v10에서도 같은 문제 있으면 가중
    if tk in ts_v10.index:
        row10 = ts_v10.loc[tk]
        if row10["win_rate"] < 65 and row11["win_rate"] < 65:
            if "양쪽 저승률" not in " ".join(reasons):
                reasons.append(f"v10도 승률 {row10['win_rate']:.0f}%")

    if reasons:
        remove_candidates[tk] = reasons

# 심각도 분류
critical   = {tk: r for tk, r in remove_candidates.items()
              if any("평균손실" in x or "서킷" in x for x in r)
              or ts_v11.loc[tk, "win_rate"] < 50}
caution    = {tk: r for tk, r in remove_candidates.items() if tk not in critical}

print("\n🔴 제거 강력 권고 (심각한 성과 불량):")
if critical:
    for tk, reasons in sorted(critical.items()):
        row = ts_v11.loc[tk]
        print(f"  {tk:<8} | {', '.join(reasons)}"
              f" | 거래 {int(row['trades'])}건, 평균 {row['avg_ret']:+.1f}%, "
              f"승률 {row['win_rate']:.0f}%")
else:
    print("  없음")

print("\n🟡 모니터링 주의 (조건부 유지):")
if caution:
    for tk, reasons in sorted(caution.items()):
        row = ts_v11.loc[tk]
        print(f"  {tk:<8} | {', '.join(reasons)}"
              f" | 거래 {int(row['trades'])}건, 평균 {row['avg_ret']:+.1f}%, "
              f"승률 {row['win_rate']:.0f}%")
else:
    print("  없음")

# 데이터 부족 종목
if skip_tickers:
    print(f"\n⬛ 데이터 부족 (백테스트 불가):")
    for tk in skip_tickers:
        print(f"  {tk}")

# 신호 0건 종목 (리스트에 있지만 한 번도 매수 신호 없음)
no_signal = [tk for tk in stock_data if tk not in ts_v11.index]
if no_signal:
    print(f"\n⬜ 신호 미발생 종목 (MA200+VIX+RSI 조건 한 번도 미충족):")
    for tk in sorted(no_signal):
        print(f"  {tk}")

print(f"\n{'='*75}")
print("  최종 권고 요약")
print(f"{'='*75}")
keep   = [tk for tk in ts_v11.index if tk not in remove_candidates and tk not in skip_tickers]
remove = list(critical.keys())
watch  = list(caution.keys())
print(f"  ✅ 유지 권고   : {len(keep)}개  — {', '.join(sorted(keep))}")
print(f"  🔴 제거 권고   : {len(remove)}개  — {', '.join(sorted(remove))}")
print(f"  🟡 주의 모니터 : {len(watch)}개  — {', '.join(sorted(watch))}")
if skip_tickers or no_signal:
    print(f"  ⬛ 데이터부족  : {len(skip_tickers)}개  — {', '.join(skip_tickers)}")
    print(f"  ⬜ 신호미발생  : {len(no_signal)}개  — {', '.join(sorted(no_signal))}")

# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle(
    f"v11 (RSI<35/CCI<-150) 종목별 성과 분석  |  {START}~{END}\n"
    f"전체 {len(ts_v11)}종목  |  승률 {v11s['wr']:.1f}%  |  평균 {v11s['ar']:+.2f}%  |  CAGR {v11s['cagr']:+.1f}%",
    fontsize=10, fontweight="bold"
)

# [0,0] 종목별 평균 수익률 (색상: 승률 기반)
ax = axes[0, 0]
tickers_sorted = ts_v11.sort_values("avg_ret", ascending=True)
colors_bar = []
for _, row in tickers_sorted.iterrows():
    if row["win_rate"] >= 80:    colors_bar.append("#27ae60")
    elif row["win_rate"] >= 60:  colors_bar.append("#f39c12")
    else:                        colors_bar.append("#e74c3c")
y_pos = range(len(tickers_sorted))
ax.barh(list(tickers_sorted.index), tickers_sorted["avg_ret"],
        color=colors_bar, edgecolor="white", alpha=0.85)
ax.axvline(0, color="black", lw=1, linestyle="--")
ax.axvline(v11s["ar"], color="blue", lw=1.5, linestyle="-",
           label=f"전체 평균 {v11s['ar']:+.2f}%")
ax.set_title("종목별 평균 수익률\n(🟢≥80% 🟡60~80% 🔴<60%)")
ax.set_xlabel("평균 수익률 %")
ax.tick_params(axis="y", labelsize=7)
ax.legend(fontsize=8)

# [0,1] 종목별 승률 산점도 (x=거래수, y=승률, 크기=평균수익)
ax = axes[0, 1]
for tk, row in ts_v11.iterrows():
    c = "#e74c3c" if tk in critical else ("#f39c12" if tk in caution else "#27ae60")
    size = max(abs(row["avg_ret"]) * 20, 30)
    ax.scatter(row["trades"], row["win_rate"], s=size, color=c, alpha=0.7, edgecolors="white")
    ax.annotate(tk, (row["trades"], row["win_rate"]),
                fontsize=6.5, ha="center", va="bottom",
                xytext=(0, 4), textcoords="offset points")
ax.axhline(60, color="red", linestyle="--", lw=1, label="60% 기준")
ax.axhline(80, color="green", linestyle=":", lw=1, label="80% 기준")
ax.set_title("거래수 vs 승률\n(🔴제거권고 🟡주의 🟢유지)")
ax.set_xlabel("거래 건수"); ax.set_ylabel("승률 %")
ax.legend(fontsize=7.5)

# [0,2] v10 vs v11 CAGR + 승률 비교
ax = axes[0, 2]
metrics = ["승률(%)", "평균수익(%)", "CAGR(%)", "PF"]
v10_vals = [v10s["wr"], v10s["ar"], v10s["cagr"],
            v10s["pf"] if not np.isnan(v10s["pf"]) else 0]
v11_vals = [v11s["wr"], v11s["ar"], v11s["cagr"],
            v11s["pf"] if not np.isnan(v11s["pf"]) else 0]
x = np.arange(len(metrics)); wid = 0.35
b1 = ax.bar(x - wid/2, v10_vals, wid, color="#2c3e50", alpha=0.8, label="v10 원본", edgecolor="white")
b2 = ax.bar(x + wid/2, v11_vals, wid, color="#e74c3c", alpha=0.8, label="v11 개선", edgecolor="white")
for bar, v in [(b1, v10_vals), (b2, v11_vals)]:
    for b, val in zip(bar, v):
        ax.text(b.get_x()+b.get_width()/2, val+0.2, f"{val:.1f}",
                ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("v10 vs v11 핵심 지표 비교")
ax.legend(fontsize=8)

# [1,0] 누적 수익률 비교
ax = axes[1, 0]
for label, c in [("v10 원본 (RSI<40 OR CCI<-100)", "#2c3e50"),
                  ("v11 개선 (RSI<35 OR CCI<-150)", "#e74c3c")]:
    df = dfs[label]
    s  = calc_stats(df)
    cum = (1 + df["return_pct"]/100).cumprod() - 1
    ax.plot(range(len(cum)), cum*100, color=c, lw=2,
            label=f"{label[:20]} CAGR {s['cagr']:+.1f}%")
ax.axhline(0, color="black", linestyle="--", lw=1)
ax.set_title("v10 vs v11 누적 수익률")
ax.set_xlabel("Trade #"); ax.legend(fontsize=8)

# [1,1] 종목별 서킷브레이커 비중
ax = axes[1, 1]
ts_circ = ts_v11[ts_v11["circuit_pct"] > 0].sort_values("circuit_pct", ascending=False)
bar_c = ["#e74c3c" if v >= 25 else "#f39c12" for v in ts_circ["circuit_pct"]]
ax.bar(ts_circ.index, ts_circ["circuit_pct"], color=bar_c, edgecolor="white", alpha=0.85)
ax.axhline(25, color="red", linestyle="--", lw=1.2, label="25% 경계")
ax.set_title("종목별 서킷브레이커 비중\n(🔴≥25% 제거 고려)")
ax.set_ylabel("%"); ax.tick_params(axis="x", rotation=45, labelsize=7.5)
ax.legend(fontsize=8)

# [1,2] 연도별 히트맵 (v11, 상위 종목)
ax = axes[1, 2]
top_tickers = ts_v11[ts_v11["trades"] >= 3].index.tolist()[:20]
all_years = sorted(df_v11["entry_date"].dt.year.unique())
hm = []
for tk in top_tickers:
    sub = df_v11[df_v11["ticker"] == tk].copy()
    sub["year"] = sub["entry_date"].dt.year
    yr_avg = sub.groupby("year")["return_pct"].mean()
    hm.append([yr_avg.get(y, np.nan) for y in all_years])
hm = np.array(hm, dtype=float)
im = ax.imshow(hm, aspect="auto", cmap="RdYlGn", vmin=-25, vmax=20)
ax.set_xticks(range(len(all_years)))
ax.set_xticklabels([str(y) for y in all_years], rotation=45, fontsize=7)
ax.set_yticks(range(len(top_tickers)))
ax.set_yticklabels(top_tickers, fontsize=7)
ax.set_title(f"v11 종목별 연도별 수익률 히트맵\n(거래 3건 이상 {len(top_tickers)}종목)")
for i in range(len(top_tickers)):
    for j in range(len(all_years)):
        v = hm[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=6, color="black" if abs(v) < 18 else "white")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("backtest_v11_ticker_analysis.png", dpi=150, bbox_inches="tight")
df_v11.to_csv("backtest_v11_trades.csv", index=False)
ts_v11.to_csv("backtest_v11_ticker_stats.csv")
print("\n📊 backtest_v11_ticker_analysis.png 저장 완료")
print("📄 backtest_v11_trades.csv / ticker_stats.csv 저장 완료")
print("✅ 완료")
