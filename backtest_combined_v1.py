"""
backtest_combined_v1.py — 추천 조합 최종 검증
=============================================

비교 그룹:
  원본:    RSI<40 OR CCI<-100  |  60일 절반 + 120일 타임
  조합A:   RSI<35 OR CCI<-150  |  60일 절반 + 90일 타임
  조합B:   RSI<35 OR CCI<-150  |  60일 절반 + 120일 타임  (RSI만 변경)
  조합C:   RSI<40 OR CCI<-100  |  60일 절반 + 90일 타임   (EXIT만 변경)

4개 비교로 조합 효과 vs 개별 효과 분리 확인
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
MAX_POSITIONS = 5
MAX_DAILY     = 5

TICKERS = sorted(set([
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","COST",
    "AMD","QCOM","INTC","TXN","AMGN","INTU","AMAT","MU","LRCX","KLAC",
    "CDNS","SNPS","FTNT","PANW","MNST","ORLY","ISRG","PAYX","MELI",
    "PLTR","CPRT","NXPI","ON","CSX","ROP","ADP","ADI","BKNG",
    "MDLZ","AZN","FAST","MCHP",
]))

# label: (rsi_thresh, cci_thresh, max_hold)
GROUPS = {
    "원본  (RSI<40OR|120일)": (40, -100, 120),
    "조합A (RSI<35OR|90일)":  (35, -150,  90),
    "조합B (RSI<35OR|120일)": (35, -150, 120),
    "조합C (RSI<40OR|90일)":  (40, -100,  90),
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
for i, tk in enumerate(TICKERS, 1):
    d = dl_ohlcv(tk, START, END)
    if len(d) < 250: continue
    stock_data[tk] = compute(d)
    if i % 10 == 0: print(f"  {i}/{len(TICKERS)} 완료")
print(f"✅ VIX {len(vix)}일 | 종목 {len(stock_data)}개")

# ──────────────────────────────────────────────
# 신호 + 시뮬레이션
# ──────────────────────────────────────────────
def build_and_run(rsi_thresh, cci_thresh, max_hold, label):
    # 신호
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

    # 시뮬레이션
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
            if i + 1 >= max_hold:
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
            "rsi_entry": sig["rsi"],
        })
        pos_exit_date[tk] = last_exit[0]

    df = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)
    print(f"  [{label}] 신호 {len(signals):,}건 → 트레이드 {len(df)}건")
    return df, len(signals)

# ──────────────────────────────────────────────
# 통계
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
    exit_detail = df.groupby("exit_reason")["return_pct"].agg(["count","mean"]).sort_values("count", ascending=False)
    return {
        "n": n, "wr": wr, "ar": ar, "aw": aw, "al": al,
        "pf": pf, "ev": ev, "max_cl": max_cl,
        "cagr": cagr * 100,
        "avg_hold": df["hold_days"].mean(),
        "med_hold": df["hold_days"].median(),
        "exit_cnt": df["exit_reason"].value_counts(),
        "exit_detail": exit_detail,
    }

# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────
print("\n⚙️  시뮬레이션...")
results = {}
for label, (rsi_t, cci_t, max_h) in GROUPS.items():
    df, n_sig = build_and_run(rsi_t, cci_t, max_h, label)
    results[label] = {"df": df, "stats": calc_stats(df), "n_sig": n_sig}

# ──────────────────────────────────────────────
# 출력
# ──────────────────────────────────────────────
labels = list(GROUPS.keys())

print("\n" + "="*95)
print("  추천 조합 최종 검증: 원본 vs 조합A(최적) vs 조합B(RSI만) vs 조합C(EXIT만)")
print("="*95)
print(f"  기간: {START}~{END}  |  공통: MA200↓ + VIX≥25, 다음날 시가")
print(f"  조합A = RSI<35/CCI<-150 + 90일  |  조합B = RSI만 변경  |  조합C = EXIT만 변경")
print("="*95)

fmt = "  {:<28}" + " {:>15}" * 4
print(fmt.format("지표", *labels))
print("  " + "-"*88)

pf_s = lambda v: f"{v:.2f}" if not (isinstance(v, float) and np.isnan(v)) else "N/A"
rows = [
    ("신호 건수",        lambda s,l: f"{results[l]['n_sig']:,}건"),
    ("총 트레이드",      lambda s,l: f"{s['n']}건"),
    ("승 률",            lambda s,l: f"{s['wr']:.1f}%"),
    ("평균 수익률",      lambda s,l: f"{s['ar']:+.2f}%"),
    ("기대값(EV)",       lambda s,l: f"{s['ev']:+.2f}%"),
    ("승자 평균",        lambda s,l: f"{s['aw']:+.2f}%"),
    ("패자 평균",        lambda s,l: f"{s['al']:+.2f}%"),
    ("Profit Factor",    lambda s,l: pf_s(s['pf'])),
    ("포트CAGR",         lambda s,l: f"{s['cagr']:+.2f}%"),
    ("최대 연속 손실",   lambda s,l: f"{s['max_cl']}건"),
    ("평균 보유 일수",   lambda s,l: f"{s['avg_hold']:.0f}일"),
    ("중간값 보유",      lambda s,l: f"{s['med_hold']:.0f}일"),
]
for row_label, fn in rows:
    vals = [fn(results[l]["stats"], l) for l in labels]
    print(fmt.format(row_label, *vals))
print("="*95)

# 원본 대비 변화량 강조
base = results[labels[0]]["stats"]
print("\n  원본 대비 변화:")
print(f"  {'':28} {'조합A':>15} {'조합B':>15} {'조합C':>15}")
print("  " + "-"*73)
for metric, key in [("CAGR", "cagr"), ("평균 수익률", "ar"),
                     ("승률", "wr"), ("PF", "pf"), ("평균 보유", "avg_hold")]:
    vals = []
    for l in labels[1:]:
        s = results[l]["stats"]
        v = s[key]; b = base[key]
        if isinstance(v, float) and np.isnan(v): vals.append("N/A")
        else: vals.append(f"{v-b:+.2f}")
    print(f"  {metric:<28} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")

for lbl in labels:
    s = results[lbl]["stats"]
    print(f"\n[{lbl}] 청산 유형:")
    for r in s["exit_cnt"].index:
        cnt = s["exit_cnt"][r]; avg = s["exit_detail"].loc[r, "mean"]
        mark = " ← 타임" if "time" in r else (" ← 손절" if "circuit" in r else "")
        print(f"  {r:<30}: {cnt:4d}건 ({cnt/s['n']*100:.1f}%)  avg {avg:+.2f}%{mark}")

for lbl in labels:
    df2 = results[lbl]["df"].copy(); df2["year"] = df2["entry_date"].dt.year
    y = df2.groupby("year").agg(
        trades  =("return_pct","count"),
        avg_ret =("return_pct","mean"),
        win_rate=("win", lambda x: x.mean()*100),
    )
    print(f"\n연도별 [{lbl}]:")
    print(y.round(1).to_string())

# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
COLORS = ["#2c3e50", "#e74c3c", "#2980b9", "#27ae60"]

fig, axes = plt.subplots(3, 3, figsize=(22, 17))

# 헤더
title_parts = []
for (lbl, _), c in zip(GROUPS.items(), COLORS):
    s = results[lbl]["stats"]
    title_parts.append(
        f"{lbl}: {s['n']}건 | 승률 {s['wr']:.1f}% | 평균 {s['ar']:+.2f}% | "
        f"CAGR {s['cagr']:+.1f}% | 보유 {s['avg_hold']:.0f}일"
    )
fig.suptitle(
    f"추천 조합 최종 검증  |  {START}~{END}\n" +
    "\n".join(title_parts),
    fontsize=8.5, fontweight="bold"
)

clr_exit = {"target":"#27ae60","circuit":"#e74c3c","time":"#f39c12",
            "half_60d":"#3498db","half_60d+target":"#1abc9c",
            "half_60d+time":"#e67e22","half_60d+circuit":"#c0392b"}
def get_clr(k):
    for ck, cv in clr_exit.items():
        if ck in k: return cv
    return "#bdc3c7"

# [0,0] 수익률 분포
ax = axes[0, 0]
for lbl, c in zip(labels, COLORS):
    s = results[lbl]["stats"]
    ax.hist(results[lbl]["df"]["return_pct"], bins=35, alpha=0.5,
            color=c, edgecolor="white",
            label=f"{lbl[:20]} ({s['ar']:+.1f}%)")
ax.axvline(0, color="black", lw=1.5, linestyle="--")
ax.set_title("수익률 분포 비교")
ax.legend(fontsize=7)

# [0,1] CAGR + PF 비교
ax = axes[0, 1]
ax2 = ax.twinx()
cagrs = [results[l]["stats"]["cagr"] for l in labels]
pfs   = [results[l]["stats"]["pf"] if not np.isnan(results[l]["stats"]["pf"]) else 0
         for l in labels]
x = np.arange(len(labels))
bars = ax.bar(x, cagrs, color=COLORS, alpha=0.8, edgecolor="white", label="CAGR")
ax2.plot(x, pfs, "D--", color="black", lw=1.5, markersize=8, label="PF")
for bar, v in zip(bars, cagrs):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.2, f"{v:.1f}%",
            ha="center", fontsize=8, fontweight="bold")
for i, v in enumerate(pfs):
    ax2.text(i, v+0.05, f"{v:.2f}", ha="center", fontsize=8, color="black")
ax.set_xticks(x); ax.set_xticklabels([l[:15] for l in labels], fontsize=7, rotation=10)
ax.set_ylabel("CAGR %"); ax2.set_ylabel("Profit Factor")
ax.set_title("CAGR & Profit Factor 비교")
lines1, lbs1 = ax.get_legend_handles_labels()
lines2, lbs2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, lbs1+lbs2, fontsize=7.5)

# [0,2] 누적 수익률
ax = axes[0, 2]
for lbl, c in zip(labels, COLORS):
    df = results[lbl]["df"]
    cum = (1 + df["return_pct"]/100).cumprod() - 1
    s   = results[lbl]["stats"]
    ax.plot(range(len(cum)), cum*100, color=c, lw=2,
            label=f"{lbl[:18]} CAGR {s['cagr']:+.1f}%")
ax.axhline(0, color="black", linestyle="--", lw=1)
ax.set_title("누적 수익률 비교")
ax.set_xlabel("Trade #"); ax.legend(fontsize=7)

# [1,0] 청산 유형 — 원본
ax = axes[1, 0]
s0 = results[labels[0]]["stats"]
ec = s0["exit_cnt"].head(7)
ax.pie(ec.values, labels=[f"{k[:18]}\n{v}건({v/s0['n']*100:.0f}%)" for k,v in ec.items()],
       colors=[get_clr(k) for k in ec.index], startangle=140,
       textprops={"fontsize": 7})
ax.set_title(f"{labels[0][:20]} 청산 유형")

# [1,1] 청산 유형 — 조합A
ax = axes[1, 1]
sA = results[labels[1]]["stats"]
ec = sA["exit_cnt"].head(7)
ax.pie(ec.values, labels=[f"{k[:18]}\n{v}건({v/sA['n']*100:.0f}%)" for k,v in ec.items()],
       colors=[get_clr(k) for k in ec.index], startangle=140,
       textprops={"fontsize": 7})
ax.set_title(f"{labels[1][:20]} 청산 유형")

# [1,2] 보유 기간 박스플롯
ax = axes[1, 2]
data_bp = [results[l]["df"]["hold_days"].values for l in labels]
bp = ax.boxplot(data_bp, labels=[l[:14] for l in labels],
                patch_artist=True, medianprops={"color":"black","lw":2.5})
for patch, c in zip(bp["boxes"], COLORS):
    patch.set_facecolor(c); patch.set_alpha(0.7)
for i, l in enumerate(labels):
    s = results[l]["stats"]
    ax.text(i+1, s["med_hold"]+3, f"avg {s['avg_hold']:.0f}일",
            ha="center", fontsize=7.5)
ax.set_title("보유 기간 Boxplot"); ax.set_ylabel("Hold Days")
ax.tick_params(axis="x", labelsize=7)

# [2,0] 연도별 히트맵
ax = axes[2, 0]
all_years = sorted(set(
    y for l in labels
    for y in results[l]["df"]["entry_date"].dt.year.unique()
))
hm = []
for l in labels:
    df2 = results[l]["df"].copy(); df2["year"] = df2["entry_date"].dt.year
    yr_avg = df2.groupby("year")["return_pct"].mean()
    hm.append([yr_avg.get(y, np.nan) for y in all_years])
hm = np.array(hm, dtype=float)
im = ax.imshow(hm, aspect="auto", cmap="RdYlGn", vmin=-10, vmax=25)
ax.set_xticks(range(len(all_years)))
ax.set_xticklabels([str(y) for y in all_years], rotation=45, fontsize=7)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels([l[:20] for l in labels], fontsize=7)
ax.set_title("연도별 평균 수익률 히트맵")
for i in range(len(labels)):
    for j in range(len(all_years)):
        v = hm[i,j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=6.5, color="black" if abs(v) < 15 else "white")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# [2,1] 연도별 승률 라인
ax = axes[2, 1]
for lbl, c in zip(labels, COLORS):
    df2 = results[lbl]["df"].copy(); df2["year"] = df2["entry_date"].dt.year
    y = df2.groupby("year")["win"].mean() * 100
    ax.plot(y.index.astype(str), y.values, "o-", color=c, lw=1.8,
            markersize=6, label=lbl[:18])
ax.axhline(80, color="gray", linestyle="--", lw=0.8, label="80% 기준")
ax.set_title("연도별 승률 비교"); ax.set_ylabel("승률 %")
ax.legend(fontsize=6.5); ax.tick_params(axis="x", rotation=45, labelsize=7)

# [2,2] 원본 대비 개선량 비교
ax = axes[2, 2]
metrics  = ["CAGR", "평균수익", "승률", "PF"]
base_s   = results[labels[0]]["stats"]
base_vals= [base_s["cagr"], base_s["ar"], base_s["wr"],
            base_s["pf"] if not np.isnan(base_s["pf"]) else 0]
x3 = np.arange(len(metrics)); wid3 = 0.25
for i, (lbl, c) in enumerate(zip(labels[1:], COLORS[1:]), 0):
    s = results[lbl]["stats"]
    vals = [s["cagr"], s["ar"], s["wr"],
            s["pf"] if not np.isnan(s["pf"]) else 0]
    diffs = [v - b for v, b in zip(vals, base_vals)]
    bars = ax.bar(x3 + (i-1)*wid3, diffs, wid3, color=c, alpha=0.85,
                  label=lbl[:15], edgecolor="white")
    for bar, d in zip(bars, diffs):
        ax.text(bar.get_x()+bar.get_width()/2,
                d + (0.05 if d >= 0 else -0.15),
                f"{d:+.2f}", ha="center", fontsize=7)
ax.axhline(0, color="black", lw=1.2)
ax.set_xticks(x3); ax.set_xticklabels(metrics, fontsize=9)
ax.set_title("원본 대비 개선량")
ax.set_ylabel("Δ (조합 - 원본)"); ax.legend(fontsize=7.5)

plt.tight_layout()
plt.savefig("backtest_combined_v1.png", dpi=150, bbox_inches="tight")
for lbl in labels:
    fname = lbl.replace(" ","_").replace("(","").replace(")","").replace("<","lt").replace("|","_").replace("/","_")
    results[lbl]["df"].to_csv(f"backtest_combined_{fname}.csv", index=False)
print("\n📊 backtest_combined_v1.png 저장 완료")
print("📄 CSV 4개 저장 완료")
print("✅ 완료")
