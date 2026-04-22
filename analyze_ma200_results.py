"""
analyze_ma200_results.py
========================
이미 수집된 ma200_bounce_cases.csv를 읽어서
성공 vs 실패 지표 분포를 분석합니다 (scipy 없이).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("ma200_bounce_cases.csv", parse_dates=["date"])
print(f"✅ 케이스 로드: {len(df):,}건")
print(df["outcome"].value_counts())

success = df[df["outcome"]=="success"]
failure = df[df["outcome"]=="failure"]

print(f"\n성공: {len(success):,}건 ({len(success)/len(df)*100:.1f}%)")
print(f"실패: {len(failure):,}건 ({len(failure)/len(df)*100:.1f}%)")

# scipy 없이 t-통계 직접 계산
def ttest_means(a, b):
    """Welch's t-test (unequal variance)"""
    na, nb = len(a), len(b)
    if na < 5 or nb < 5:
        return np.nan, np.nan
    va = a.var(ddof=1) / na
    vb = b.var(ddof=1) / nb
    se = np.sqrt(va + vb)
    if se == 0:
        return np.nan, np.nan
    t = (a.mean() - b.mean()) / se
    return t, np.nan  # p-val 생략

INDICATORS = [
    ("RSI",            "RSI(14)"),
    ("MACD_H",         "MACD 히스토그램"),
    ("BB_PCT_B",       "Bollinger %B"),
    ("VolRatio",       "거래량/20일평균"),
    ("ATR_PCT",        "ATR/종가 (변동성)"),
    ("Body_Pct",       "캔들 몸통 비율"),
    ("LTail_Pct",      "아래꼬리 비율"),
    ("IsBullish",      "양봉 여부"),
    ("Close_vs_MA200", "종가/MA200 거리"),
    ("Close_vs_MA50",  "종가/MA50 거리"),
    ("Close_vs_MA20",  "종가/MA20 거리"),
    ("Ret1d",          "당일 수익률"),
    ("Ret5d",          "5일 수익률"),
    ("VIX",            "VIX"),
    ("SPY_above_MA200","SPY MA200 위 여부"),
    ("RSI_rising",     "RSI 상승 여부"),
]

print("\n" + "="*80)
print("  지표별 성공 vs 실패 평균 비교  (t값 클수록 구분력 강함)")
print("="*80)
print(f"  {'지표':<25} {'성공평균':>10} {'실패평균':>10} {'차이':>8} {'t값':>8}  {'구분력'}")
print("  " + "-"*75)

analysis_results = []
for col, label in INDICATORS:
    s_vals = success[col].dropna()
    f_vals = failure[col].dropna()
    if len(s_vals) < 10 or len(f_vals) < 10:
        continue
    s_mean = s_vals.mean()
    f_mean = f_vals.mean()
    diff   = s_mean - f_mean
    t_stat, _ = ttest_means(s_vals, f_vals)
    
    power = "★★★" if abs(t_stat) > 30 else ("★★" if abs(t_stat) > 15 else ("★" if abs(t_stat) > 5 else ""))
    print(f"  {label:<25} {s_mean:>10.3f} {f_mean:>10.3f} {diff:>+8.3f} {t_stat:>8.1f}  {power}")
    analysis_results.append({
        "indicator": col, "label": label,
        "success_mean": s_mean, "failure_mean": f_mean,
        "diff": diff, "abs_t": abs(t_stat),
    })

ar = pd.DataFrame(analysis_results).sort_values("abs_t", ascending=False)

# ──────────────────────────────────────────────────────
# 분위별 성공률
# ──────────────────────────────────────────────────────
sf = df[df["outcome"].isin(["success","failure"])].copy()
sf["is_success"] = (sf["outcome"] == "success").astype(int)
overall_rate = sf["is_success"].mean()

print(f"\n  전체 기준선 성공률: {overall_rate*100:.1f}%")

print("\n" + "="*80)
print("  지표별 분위 구간 성공률  (기준선 대비 얼마나 높은지)")
print("="*80)

key_continuous = [
    ("RSI",            "RSI(14)"),
    ("VolRatio",       "거래량비율"),
    ("VIX",            "VIX"),
    ("Close_vs_MA200", "종가/MA200 거리"),
    ("BB_PCT_B",       "Bollinger %B"),
    ("ATR_PCT",        "ATR/종가"),
    ("LTail_Pct",      "아래꼬리"),
    ("Ret1d",          "당일 수익률"),
    ("Ret5d",          "5일 수익률"),
]

quantile_summary = {}
for col, label in key_continuous:
    try:
        sf[f"_q"] = pd.qcut(sf[col], q=4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
        tbl = sf.groupby("_q", observed=True).agg(
            win_rate=("is_success","mean"),
            count=("is_success","count"),
            mean_val=(col,"mean"),
        )
        print(f"\n  [{label}]  (기준선 {overall_rate*100:.1f}%)")
        best_q   = tbl["win_rate"].idxmax()
        for idx, r in tbl.iterrows():
            delta = r["win_rate"]*100 - overall_rate*100
            marker = "◀ 최고" if idx==best_q else ""
            bar_len = max(0, int((r["win_rate"]-0.3)*60))
            bar = "█"*bar_len
            print(f"    {idx} (평균 {r['mean_val']:>7.2f}):  {r['win_rate']*100:>5.1f}%  ({delta:>+5.1f}pp)  {bar}  ({int(r['count'])}건)  {marker}")
        quantile_summary[col] = tbl
    except Exception as e:
        print(f"  [{label}]: 분석 실패 ({e})")

# 이진 변수
key_binary = [
    ("IsBullish",       "양봉 여부"),
    ("RSI_rising",      "RSI 상승 여부"),
    ("SPY_above_MA200", "SPY MA200 위"),
]
print(f"\n  [이진 변수 성공률]")
print(f"  {'변수':<25} {'0 (없음)':>12} {'1 (있음)':>12}  {'차이':>8}")
print("  " + "-"*60)
for col, label in key_binary:
    tbl = sf.groupby(col)["is_success"].mean()
    v0 = tbl.get(0, np.nan)
    v1 = tbl.get(1, np.nan)
    diff = v1 - v0 if not np.isnan(v0) and not np.isnan(v1) else np.nan
    print(f"  {label:<25} {v0*100:>11.1f}%  {v1*100:>11.1f}%  {diff*100:>+7.1f}pp")

# ──────────────────────────────────────────────────────
# 복합 조건 성공률
# ──────────────────────────────────────────────────────
print("\n" + "="*80)
print(f"  복합 조건 성공률  (기준선: {overall_rate*100:.1f}%)")
print("="*80)

combos = [
    ("① 기준선 (전체)",
     pd.Series(True, index=sf.index)),
    ("② SPY MA200 위",
     sf["SPY_above_MA200"]==1),
    ("③ RSI 상승 중",
     sf["RSI_rising"]==1),
    ("④ 양봉",
     sf["IsBullish"]==1),
    ("⑤ 거래량 1.5배+",
     sf["VolRatio"]>1.5),
    ("⑥ RSI 30~50",
     sf["RSI"].between(30,50)),
    ("⑦ VIX 22~40",
     sf["VIX"].between(22,40)),
    ("⑧ MA200 -5%~-25%",
     sf["Close_vs_MA200"].between(-0.25,-0.05)),
    ("─────────────────────────────────────", None),
    ("⑨ ③+④ (RSI상승+양봉)",
     (sf["RSI_rising"]==1) & (sf["IsBullish"]==1)),
    ("⑩ ②+③ (SPY위+RSI상승)",
     (sf["SPY_above_MA200"]==1) & (sf["RSI_rising"]==1)),
    ("⑪ ③+④+⑤ (RSI상승+양봉+거래량)",
     (sf["RSI_rising"]==1) & (sf["IsBullish"]==1) & (sf["VolRatio"]>1.5)),
    ("⑫ ②+③+④ (SPY위+RSI상승+양봉)",
     (sf["SPY_above_MA200"]==1) & (sf["RSI_rising"]==1) & (sf["IsBullish"]==1)),
    ("─────────────────────────────────────", None),
    ("⑬ ②+③+④+⑤ (4개 조합)",
     (sf["SPY_above_MA200"]==1) & (sf["RSI_rising"]==1) & (sf["IsBullish"]==1) & (sf["VolRatio"]>1.5)),
    ("⑭ ⑦+②+③+④+⑤ (VIX추가)",
     sf["VIX"].between(22,40) & (sf["SPY_above_MA200"]==1) & (sf["RSI_rising"]==1) & (sf["IsBullish"]==1) & (sf["VolRatio"]>1.5)),
    ("⑮ ⑦+②+⑥+③+④+⑤ (RSI범위추가)",
     sf["VIX"].between(22,40) & (sf["SPY_above_MA200"]==1) & sf["RSI"].between(30,50) & (sf["RSI_rising"]==1) & (sf["IsBullish"]==1) & (sf["VolRatio"]>1.5)),
    ("─────────────────────────────────────", None),
    ("⑯ ⑬ + MA200 거리 -5%~-25%",
     (sf["SPY_above_MA200"]==1) & (sf["RSI_rising"]==1) & (sf["IsBullish"]==1) & (sf["VolRatio"]>1.5) & sf["Close_vs_MA200"].between(-0.25,-0.05)),
    ("⑰ ⑬ + MACD_H 개선",
     (sf["SPY_above_MA200"]==1) & (sf["RSI_rising"]==1) & (sf["IsBullish"]==1) & (sf["VolRatio"]>1.5) & (sf["MACD_H"] > sf["MACD_H_prev"])),
]

print(f"\n  {'조건':<43} {'성공률':>7} {'기준대비':>8} {'건수':>8}")
print("  " + "-"*72)
for label, mask in combos:
    if mask is None:
        print(f"  {label}")
        continue
    sub = sf[mask]
    if len(sub) < 30:
        print(f"  {label:<43} {'(N부족)':>7}")
        continue
    wr = sub["is_success"].mean()
    n  = len(sub)
    delta = wr - overall_rate
    bar = "█"*int(wr*30)
    marker = " ◀★" if wr > overall_rate*1.20 else (" ◀" if wr > overall_rate*1.10 else "")
    print(f"  {label:<43} {wr*100:>6.1f}%  {delta*100:>+7.1f}pp  {n:>7,}건{marker}")

# ──────────────────────────────────────────────────────
# 핵심 발견 요약
# ──────────────────────────────────────────────────────
print("\n" + "="*80)
print("  📌 데이터가 말하는 핵심 공통점 — 전략 설계 인사이트")
print("="*80)

for col, label in [("RSI","RSI"), ("VolRatio","거래량비율"), ("VIX","VIX"), ("Close_vs_MA200","MA200 거리")]:
    if col not in quantile_summary:
        continue
    tbl = quantile_summary[col]
    best_q   = tbl["win_rate"].idxmax()
    best_rate= tbl.loc[best_q,"win_rate"]*100
    best_val = tbl.loc[best_q,"mean_val"]
    worst_q  = tbl["win_rate"].idxmin()
    worst_rate = tbl.loc[worst_q,"win_rate"]*100
    worst_val  = tbl.loc[worst_q,"mean_val"]
    uplift = best_rate - overall_rate*100
    print(f"\n  [{label}]")
    print(f"    최고 성공 구간: {best_q} (평균 {best_val:.2f}) → 성공률 {best_rate:.1f}%  (+{uplift:.1f}pp)")
    print(f"    최저 성공 구간: {worst_q} (평균 {worst_val:.2f}) → 성공률 {worst_rate:.1f}%")

# ──────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────
print("\n📊 시각화 생성 중...")
fig = plt.figure(figsize=(22, 16))
fig.suptitle("MA200 아래 반등 케이스 분석 — 성공 vs 실패 지표 분포\n"
             f"(전체 {len(df):,}건: 성공 {len(success):,}건 / 실패 {len(failure):,}건)",
             fontsize=13, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.35)

plot_items = [
    ("RSI",            "RSI(14)",          0, 0),
    ("VolRatio",       "거래량/20일평균",   0, 1),
    ("VIX",            "VIX",              0, 2),
    ("Close_vs_MA200", "종가/MA200 거리",   0, 3),
    ("BB_PCT_B",       "Bollinger %B",     1, 0),
    ("ATR_PCT",        "ATR/종가",         1, 1),
    ("LTail_Pct",      "아래꼬리 비율",     1, 2),
    ("Ret1d",          "당일 수익률",       1, 3),
]

for col, label, row, col_idx in plot_items:
    ax = fig.add_subplot(gs[row, col_idx])
    s_data = success[col].dropna()
    f_data = failure[col].dropna()
    lo = min(np.percentile(s_data, 2), np.percentile(f_data, 2))
    hi = max(np.percentile(s_data, 98), np.percentile(f_data, 98))
    bins = np.linspace(lo, hi, 35)
    ax.hist(s_data.clip(lo,hi), bins=bins, alpha=0.55, color="#2ecc71",
            label=f"성공 (avg={s_data.mean():.2f})", density=True)
    ax.hist(f_data.clip(lo,hi), bins=bins, alpha=0.55, color="#e74c3c",
            label=f"실패 (avg={f_data.mean():.2f})", density=True)
    ax.axvline(s_data.mean(), color="#1e8449", linestyle="--", lw=1.5)
    ax.axvline(f_data.mean(), color="#922b21", linestyle="--", lw=1.5)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5)

# 복합 조건 바차트
ax_combo = fig.add_subplot(gs[2, :])
c_labels, c_rates, c_ns = [], [], []
for label, mask in combos:
    if mask is None or "─" in label:
        continue
    sub = sf[mask]
    if len(sub) < 30:
        continue
    c_labels.append(label)
    c_rates.append(sub["is_success"].mean()*100)
    c_ns.append(len(sub))

bar_colors = ["#2ecc71" if r >= overall_rate*100*1.15 else
              ("#f39c12" if r >= overall_rate*100*1.05 else "#e74c3c") for r in c_rates]
y_pos = range(len(c_labels))
ax_combo.barh(y_pos, c_rates, color=bar_colors, edgecolor="white")
ax_combo.set_yticks(y_pos)
ax_combo.set_yticklabels(c_labels, fontsize=8.5)
ax_combo.axvline(overall_rate*100, color="gray", linestyle="--", lw=1.5, label=f"기준선 {overall_rate*100:.1f}%")
ax_combo.set_xlabel("성공률 (%) — +5% 달성 / 20거래일")
ax_combo.set_title("복합 조건별 성공률 (MA200 아래 반등 케이스)", fontsize=11, fontweight="bold")
ax_combo.legend()
for i, (r, n) in enumerate(zip(c_rates, c_ns)):
    ax_combo.text(r+0.2, i, f"{r:.1f}%  ({n:,}건)", va="center", fontsize=8)
ax_combo.set_xlim(0, max(c_rates)*1.15)

plt.savefig("ma200_bounce_analysis.png", dpi=150, bbox_inches="tight")
print("📊 ma200_bounce_analysis.png 저장 완료")
print("✅ 분석 완료")
