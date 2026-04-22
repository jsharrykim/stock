"""
analyze_ma200_bounces.py
========================
MA200 아래에서 반등한 케이스를 모두 수집하고
각 케이스의 지표값 분포를 통계적으로 분석합니다.

반등 정의:
  - 종가가 MA200 아래에 있는 상태
  - 이후 N일 내 +5% 이상 상승 (성공 반등)
  vs
  - 이후 N일 내 +5% 못 미침 (실패)

출력: 성공 vs 실패 케이스별 지표 분포 비교
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

START = "2010-01-01"
END   = "2026-01-01"

# 반등 성공 기준: 진입 후 20거래일 내 +5% 이상
SUCCESS_RET  = 0.05
SUCCESS_DAYS = 20
FAILURE_STOP = -0.07  # -7% 이하면 실패로 확정

TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","TMUS","AMD","ADBE","PEP","CSCO","QCOM","INTC","TXN","AMGN",
    "INTU","AMAT","MU","LRCX","KLAC","CDNS","SNPS","MRVL","FTNT","PANW",
    "CRWD","DDOG","ZS","TEAM","MNST","KDP","MDLZ","ORLY","AZN",
    "ISRG","IDXX","REGN","VRTX","GILD","BIIB","ILMN","MRNA",
    "ODFL","FAST","PCAR","CTSH","PAYX","VRSK","WDAY","MELI","PDD",
    "DLTR","SBUX","ROST","LULU","EBAY","MAR","CTAS","EA",
    "CHTR","CMCSA","EXC","XEL","AEP","MKL",
    "PLTR","CPRT","NXPI","MPWR","ENPH","SEDG","ON","COIN",
    "DOCU","ZM","OKTA","PTON","JD","BIDU","NTES","FANG",
]
TICKERS = sorted(set(TICKERS))

# ──────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────
def dl_ohlcv(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw[["Open","High","Low","Close","Volume"]].copy()

def dl_series(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    if raw.empty:
        return pd.Series(dtype=float)
    s = raw["Close"]
    if isinstance(s, pd.DataFrame): s = s.iloc[:,0]
    return pd.Series(s.values, index=s.index, dtype=float).dropna()

def add_indicators(d):
    d = d.copy()
    c = d["Close"]
    # MAs
    d["MA20"]  = c.rolling(20).mean()
    d["MA50"]  = c.rolling(50).mean()
    d["MA200"] = c.rolling(200).mean()
    # RSI(14)
    delta = c.diff()
    g = delta.clip(lower=0).rolling(14).mean()
    l = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100/(1 + g/l.replace(0,np.nan))
    # MACD(12,26,9)
    e12 = c.ewm(span=12,adjust=False).mean()
    e26 = c.ewm(span=26,adjust=False).mean()
    macd = e12 - e26
    sig  = macd.ewm(span=9,adjust=False).mean()
    d["MACD"]   = macd
    d["MACD_S"] = sig
    d["MACD_H"] = macd - sig
    # Bollinger Band %B
    ma20 = c.rolling(20).mean()
    std20= c.rolling(20).std()
    d["BB_UPPER"] = ma20 + 2*std20
    d["BB_LOWER"] = ma20 - 2*std20
    d["BB_PCT_B"] = (c - (ma20 - 2*std20)) / (4*std20)
    # ATR(14)
    tr = pd.concat([
        d["High"]-d["Low"],
        (d["High"]-c.shift(1)).abs(),
        (d["Low"] -c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean()
    d["ATR_PCT"] = d["ATR"] / c  # ATR/종가 (변동성 크기)
    # Volume
    d["VolAvg20"] = d["Volume"].rolling(20).mean()
    d["VolRatio"] = d["Volume"] / d["VolAvg20"]
    # 캔들
    d["Body"]       = (d["Close"] - d["Open"]).abs()
    d["Range"]      = d["High"] - d["Low"]
    d["Body_Pct"]   = d["Body"] / d["Range"].replace(0,np.nan)
    d["LowerTail"]  = d[["Open","Close"]].min(axis=1) - d["Low"]
    d["UpperTail"]  = d["High"] - d[["Open","Close"]].max(axis=1)
    d["LTail_Pct"]  = d["LowerTail"] / d["Range"].replace(0,np.nan)
    d["IsBullish"]  = (d["Close"] > d["Open"]).astype(int)
    # 가격 위치
    d["Close_vs_MA200"] = c / d["MA200"] - 1   # MA200 대비 거리
    d["Close_vs_MA50"]  = c / d["MA50"]  - 1
    d["Close_vs_MA20"]  = c / d["MA20"]  - 1
    # 모멘텀
    d["Ret1d"]  = c.pct_change(1)
    d["Ret5d"]  = c.pct_change(5)
    d["Ret20d"] = c.pct_change(20)
    return d

# ──────────────────────────────────────────────────────
# 시장 지표 (VIX, QQQ)
# ──────────────────────────────────────────────────────
print("📥 시장 데이터 다운로드 중...")
vix_raw = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)
_vc = vix_raw["Close"]
if isinstance(_vc, pd.DataFrame): _vc = _vc.iloc[:,0]
vix = pd.Series(_vc.values, index=_vc.index, dtype=float).dropna()

qqq_close = dl_series("QQQ", START, END)
spy_close = dl_series("SPY", START, END)
spy_ma200 = spy_close.rolling(200).mean()

# ──────────────────────────────────────────────────────
# 케이스 수집
# ──────────────────────────────────────────────────────
print("📥 종목 데이터 다운로드 & 케이스 수집 중...")

all_cases = []

for i, tk in enumerate(TICKERS, 1):
    raw = dl_ohlcv(tk, START, END)
    if len(raw) < 250:
        continue
    d = add_indicators(raw)
    d = d.dropna(subset=["MA200","RSI","MACD_H","VolRatio","ATR"])

    close   = d["Close"]
    ma200   = d["MA200"]

    for j in range(len(d)-SUCCESS_DAYS-1):
        row = d.iloc[j]
        dt  = d.index[j]

        # 조건: 종가가 MA200 아래
        if row["Close"] >= row["MA200"]:
            continue

        # 미래 수익 계산
        future_close = close.iloc[j+1:j+SUCCESS_DAYS+1]
        max_ret = (future_close.max() - row["Close"]) / row["Close"]
        min_ret = (future_close.min() - row["Close"]) / row["Close"]
        final_ret = (future_close.iloc[-1] - row["Close"]) / row["Close"] if len(future_close)>0 else np.nan

        # 성공 / 실패 분류
        if max_ret >= SUCCESS_RET:
            outcome = "success"
        elif min_ret <= FAILURE_STOP:
            outcome = "failure"
        else:
            outcome = "neutral"  # +5%도 -7%도 안 됨

        # VIX, SPY 정보 추가
        vix_val   = vix.get(dt, np.nan)
        spy_c_val = spy_close.get(dt, np.nan)
        spy_m200  = spy_ma200.get(dt, np.nan)
        qqq_val   = qqq_close.get(dt, np.nan)

        case = {
            "date"           : dt,
            "ticker"         : tk,
            "outcome"        : outcome,
            "max_ret_20d"    : max_ret,
            "min_ret_20d"    : min_ret,
            "final_ret_20d"  : final_ret,
            # 핵심 지표들
            "RSI"            : row["RSI"],
            "MACD_H"         : row["MACD_H"],
            "MACD_H_prev"    : d["MACD_H"].iloc[j-1] if j>0 else np.nan,
            "BB_PCT_B"       : row["BB_PCT_B"],
            "VolRatio"       : row["VolRatio"],
            "ATR_PCT"        : row["ATR_PCT"],
            "Body_Pct"       : row["Body_Pct"],
            "LTail_Pct"      : row["LTail_Pct"],
            "IsBullish"      : row["IsBullish"],
            "Close_vs_MA200" : row["Close_vs_MA200"],
            "Close_vs_MA50"  : row["Close_vs_MA50"],
            "Close_vs_MA20"  : row["Close_vs_MA20"],
            "Ret1d"          : row["Ret1d"],
            "Ret5d"          : row["Ret5d"],
            "VIX"            : vix_val,
            "SPY_above_MA200": 1 if (spy_c_val > spy_m200) else 0,
            "RSI_prev"       : d["RSI"].iloc[j-1] if j>0 else np.nan,
            "RSI_rising"     : int(row["RSI"] > d["RSI"].iloc[j-1]) if j>0 else 0,
        }
        all_cases.append(case)

    if i % 15 == 0:
        print(f"  {i}/{len(TICKERS)} 완료 (누적 케이스: {len(all_cases):,})")

df = pd.DataFrame(all_cases)
print(f"\n✅ 총 케이스: {len(df):,}건")
print(df["outcome"].value_counts())

# ──────────────────────────────────────────────────────
# 통계 분석
# ──────────────────────────────────────────────────────
success = df[df["outcome"]=="success"]
failure = df[df["outcome"]=="failure"]
neutral = df[df["outcome"]=="neutral"]

print(f"\n성공: {len(success):,}건 ({len(success)/len(df)*100:.1f}%)")
print(f"실패: {len(failure):,}건 ({len(failure)/len(df)*100:.1f}%)")
print(f"중립: {len(neutral):,}건 ({len(neutral)/len(df)*100:.1f}%)")

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
print("  지표별 성공 vs 실패 분포 비교")
print("="*80)
print(f"{'지표':<25} {'성공 평균':>10} {'실패 평균':>10} {'차이':>8} {'t-통계':>8}")
print("-"*80)

def ttest_means(a, b):
    na, nb = len(a), len(b)
    if na < 5 or nb < 5:
        return np.nan
    va = a.var(ddof=1) / na
    vb = b.var(ddof=1) / nb
    se = np.sqrt(va + vb)
    if se == 0:
        return np.nan
    return (a.mean() - b.mean()) / se

analysis_results = []
for col, label in INDICATORS:
    s_vals = success[col].dropna()
    f_vals = failure[col].dropna()
    if len(s_vals) < 10 or len(f_vals) < 10:
        continue
    s_mean = s_vals.mean()
    f_mean = f_vals.mean()
    diff   = s_mean - f_mean
    t_stat = ttest_means(s_vals, f_vals)

    power = "★★★" if abs(t_stat) > 30 else ("★★" if abs(t_stat) > 15 else ("★" if abs(t_stat) > 5 else ""))
    print(f"  {label:<23} {s_mean:>10.3f} {f_mean:>10.3f} {diff:>+8.3f} {t_stat:>8.1f} {power}")
    analysis_results.append({
        "indicator": col, "label": label,
        "success_mean": s_mean, "failure_mean": f_mean,
        "diff": diff, "t_stat": t_stat,
    })

ar = pd.DataFrame(analysis_results).sort_values("t_stat", key=abs, ascending=False)

# ──────────────────────────────────────────────────────
# 분위별 성공률 분석 (핵심!)
# ──────────────────────────────────────────────────────
print("\n" + "="*80)
print("  지표별 분위 구간 성공률 (success = +5% 달성)")
print("  (성공률이 높은 구간 = 진입 조건으로 사용 가능)")
print("="*80)

# 성공/실패만 사용 (중립 제외)
sf = df[df["outcome"].isin(["success","failure"])].copy()
sf["is_success"] = (sf["outcome"] == "success").astype(int)

quantile_results = {}
key_cols = ["RSI","VolRatio","BB_PCT_B","VIX","Close_vs_MA200",
            "IsBullish","RSI_rising","SPY_above_MA200","MACD_H","ATR_PCT","LTail_Pct"]

for col in key_cols:
    vals = sf[col].dropna()
    if col in ["IsBullish","RSI_rising","SPY_above_MA200"]:
        # 이진 변수
        tbl = sf.groupby(col)["is_success"].agg(["mean","count"])
        tbl.columns = ["win_rate","count"]
        print(f"\n  [{col}] — 이진 분류:")
        for idx, r in tbl.iterrows():
            print(f"    {idx}: 성공률 {r['win_rate']*100:.1f}%  ({int(r['count'])}건)")
        quantile_results[col] = tbl
    else:
        # 4분위
        try:
            sf[f"{col}_q"] = pd.qcut(sf[col], q=4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
            tbl = sf.groupby(f"{col}_q", observed=True)[["is_success",col]].agg(
                win_rate=("is_success","mean"),
                count=("is_success","count"),
                mean_val=(col,"mean"),
            )
            print(f"\n  [{col}] 분위별 성공률:")
            for idx, r in tbl.iterrows():
                bar = "█" * int(r["win_rate"]*20)
                print(f"    {idx} (평균{r['mean_val']:>7.2f}): {r['win_rate']*100:>5.1f}%  {bar}  ({int(r['count'])}건)")
            quantile_results[col] = tbl
        except Exception as e:
            pass

# ──────────────────────────────────────────────────────
# 복합 조건 테스트 (통계에서 뽑은 상위 조건 조합)
# ──────────────────────────────────────────────────────
print("\n" + "="*80)
print("  복합 조건 조합 성공률 테스트")
print("="*80)

combos = [
    ("RSI < 35",
     sf["RSI"] < 35),
    ("RSI 35~55",
     sf["RSI"].between(35,55)),
    ("VolRatio > 1.5",
     sf["VolRatio"] > 1.5),
    ("RSI < 45 & VolRatio > 1.5",
     (sf["RSI"] < 45) & (sf["VolRatio"] > 1.5)),
    ("RSI < 45 & IsBullish",
     (sf["RSI"] < 45) & (sf["IsBullish"]==1)),
    ("RSI_rising & VolRatio > 1.5",
     (sf["RSI_rising"]==1) & (sf["VolRatio"] > 1.5)),
    ("RSI_rising & IsBullish & VolRatio>1.5",
     (sf["RSI_rising"]==1) & (sf["IsBullish"]==1) & (sf["VolRatio"]>1.5)),
    ("SPY_above_MA200",
     sf["SPY_above_MA200"]==1),
    ("SPY_above_MA200 & RSI_rising & VolRatio>1.5",
     (sf["SPY_above_MA200"]==1) & (sf["RSI_rising"]==1) & (sf["VolRatio"]>1.5)),
    ("SPY_above_MA200 & RSI_rising & IsBullish & VolRatio>1.5",
     (sf["SPY_above_MA200"]==1) & (sf["RSI_rising"]==1) & (sf["IsBullish"]==1) & (sf["VolRatio"]>1.5)),
    ("VIX>22 & RSI_rising & IsBullish & VolRatio>1.5",
     (sf["VIX"]>22) & (sf["RSI_rising"]==1) & (sf["IsBullish"]==1) & (sf["VolRatio"]>1.5)),
    ("VIX>22 & SPY_above & RSI_rising & IsBullish & Vol>1.5",
     (sf["VIX"]>22) & (sf["SPY_above_MA200"]==1) & (sf["RSI_rising"]==1) & (sf["IsBullish"]==1) & (sf["VolRatio"]>1.5)),
    ("Close MA200 -5%~-20% & RSI_rising & Vol>1.5",
     (sf["Close_vs_MA200"].between(-0.20,-0.05)) & (sf["RSI_rising"]==1) & (sf["VolRatio"]>1.5)),
    ("MACD_H 개선 & RSI_rising & Vol>1.5 & IsBullish",
     (sf["MACD_H"] > sf["MACD_H_prev"]) & (sf["RSI_rising"]==1) & (sf["VolRatio"]>1.5) & (sf["IsBullish"]==1)),
    ("전체 평균 (기준선)",
     pd.Series(True, index=sf.index)),
]

print(f"\n  {'조건':<50} {'성공률':>8} {'건수':>8}")
print("  " + "-"*70)
for label, mask in combos:
    sub = sf[mask]
    if len(sub) < 20:
        print(f"  {label:<50} {'(N<20)':>8}")
        continue
    wr = sub["is_success"].mean()
    n  = len(sub)
    bar = "█"*int(wr*30)
    print(f"  {label:<50} {wr*100:>7.1f}%  {n:>6}건  {bar}")

# ──────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────
print("\n📊 시각화 생성 중...")

fig = plt.figure(figsize=(20, 16))
fig.suptitle("MA200 아래 반등 케이스 분석 — 성공 vs 실패 지표 분포",
             fontsize=15, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

plot_items = [
    ("RSI",            "RSI(14)",          0, 0),
    ("VolRatio",       "거래량비율",         0, 1),
    ("BB_PCT_B",       "Bollinger %B",     0, 2),
    ("Close_vs_MA200", "종가/MA200 거리",   0, 3),
    ("VIX",            "VIX",             1, 0),
    ("ATR_PCT",        "ATR/종가",         1, 1),
    ("Body_Pct",       "몸통 비율",         1, 2),
    ("LTail_Pct",      "아래꼬리 비율",     1, 3),
]

for col, label, row, col_idx in plot_items:
    ax = fig.add_subplot(gs[row, col_idx])
    s_data = success[col].dropna()
    f_data = failure[col].dropna()
    # 이상치 제거 (1~99 퍼센타일)
    lo = min(np.percentile(s_data, 1), np.percentile(f_data, 1))
    hi = max(np.percentile(s_data, 99), np.percentile(f_data, 99))
    bins = np.linspace(lo, hi, 30)
    ax.hist(s_data.clip(lo,hi), bins=bins, alpha=0.55, color="#2ecc71",
            label=f"성공({len(s_data):,})", density=True)
    ax.hist(f_data.clip(lo,hi), bins=bins, alpha=0.55, color="#e74c3c",
            label=f"실패({len(f_data):,})", density=True)
    ax.axvline(s_data.mean(), color="#27ae60", linestyle="--", linewidth=1.5)
    ax.axvline(f_data.mean(), color="#c0392b", linestyle="--", linewidth=1.5)
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=7)

# 복합 조건 성공률 바차트
ax_combo = fig.add_subplot(gs[2, :])
combo_labels = []
combo_rates  = []
combo_ns     = []

for label, mask in combos:
    sub = sf[mask]
    if len(sub) < 20:
        continue
    combo_labels.append(label[:45])
    combo_rates.append(sub["is_success"].mean()*100)
    combo_ns.append(len(sub))

bar_colors = ["#2ecc71" if r>50 else ("#f39c12" if r>40 else "#e74c3c") for r in combo_rates]
bars = ax_combo.barh(range(len(combo_labels)), combo_rates, color=bar_colors)
ax_combo.set_yticks(range(len(combo_labels)))
ax_combo.set_yticklabels(combo_labels, fontsize=8)
ax_combo.axvline(combo_rates[-1], color="gray", linestyle="--", linewidth=1)  # 기준선
ax_combo.set_xlabel("성공률 (%)")
ax_combo.set_title("복합 조건별 성공률 (MA200 아래 반등 케이스)", fontsize=11)
for i, (r, n) in enumerate(zip(combo_rates, combo_ns)):
    ax_combo.text(r+0.3, i, f"{r:.1f}%  ({n}건)", va="center", fontsize=7.5)

plt.savefig("ma200_bounce_analysis.png", dpi=150, bbox_inches="tight")
print("📊 ma200_bounce_analysis.png 저장 완료")

# ──────────────────────────────────────────────────────
# 최종 요약 — 전략 설계에 쓸 공통점 추출
# ──────────────────────────────────────────────────────
print("\n" + "="*80)
print("  📌 데이터 기반 공통점 요약")
print("="*80)

# 각 지표의 최적 구간 출력
for col in ["RSI","VolRatio","VIX","Close_vs_MA200"]:
    try:
        sf_temp = sf.copy()
        sf_temp["q"] = pd.qcut(sf_temp[col], q=4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
        tbl = sf_temp.groupby("q", observed=True)["is_success"].mean()
        best_q = tbl.idxmax()
        best_rate = tbl.max()
        worst_q = tbl.idxmin()
        worst_rate = tbl.min()
        print(f"\n  [{col}]")
        print(f"    최고 성공률 구간: {best_q}  ({best_rate*100:.1f}%)")
        print(f"    최저 성공률 구간: {worst_q}  ({worst_rate*100:.1f}%)")
        # 구간별 실제 값
        for q, r in tbl.items():
            mean_val = sf_temp[sf_temp["q"]==q][col].mean()
            print(f"      {q}: 평균 {mean_val:.2f} → 성공률 {r*100:.1f}%")
    except:
        pass

df.to_csv("ma200_bounce_cases.csv", index=False)
print(f"\n📄 전체 케이스 저장: ma200_bounce_cases.csv ({len(df):,}건)")
print("✅ 분석 완료")
