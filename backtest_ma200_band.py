"""
backtest_ma200_band.py — MA200 이격 구간별 비교
================================================

그룹 1 (밴드): 현재가가 MA200 기준 0~10% 아래인 경우만 진입
  조건: -10% < MA200_GAP <= 0%
  즉: MA200 바로 아래 ~ 10% 이내

그룹 2 (깊은 이격): 현재가가 MA200 기준 10% 이상 아래인 경우만 진입
  조건: MA200_GAP <= -10%
  즉: MA200에서 10% 초과 하방

나머지 공통:
  VIX ≥ 25
  RSI < 40 OR CCI < -100
  진입가: 신호 발생일 다음날 시가
  매도: +20% / -25%CB / 60일절반 / 120일
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

TICKERS = sorted(set([
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","COST",
    "AMD","QCOM","INTC","TXN","AMGN","INTU","AMAT","MU","LRCX","KLAC",
    "CDNS","SNPS","FTNT","PANW","MNST","ORLY","ISRG","PAYX","MELI",
    "PLTR","CPRT","NXPI","ON","CSX","ROP","ADP","ADI","BKNG",
    "MDLZ","AZN","FAST","MCHP",
]))

# ──────────────────────────────────────────────
# 헬퍼
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
    d  = d.copy()
    c  = d["Close"]
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
    d["MA200_GAP"] = (c - d["MA200"]) / d["MA200"] * 100
    return d


# ──────────────────────────────────────────────
# 데이터 다운로드
# ──────────────────────────────────────────────
print("📥 VIX 다운로드 중...")
vix_raw = yf.download("^VIX", start=START, end=END,
                      auto_adjust=True, progress=False)
_vc = vix_raw["Close"]
if isinstance(_vc, pd.DataFrame): _vc = _vc.iloc[:, 0]
vix = pd.Series(_vc.values, index=_vc.index, dtype=float).dropna()
print(f"✅ VIX {len(vix)}일")

print("📥 종목 OHLCV 다운로드 중...")
stock_data = {}
for i, tk in enumerate(TICKERS, 1):
    d = dl_ohlcv(tk, START, END)
    if len(d) < 250: continue
    d = compute(d)
    stock_data[tk] = d
    if i % 10 == 0: print(f"  {i}/{len(TICKERS)} 완료")
print(f"✅ {len(stock_data)}개 종목 로드")


# ──────────────────────────────────────────────
# 신호 생성 — 구간 필터 포함
# ──────────────────────────────────────────────
def build_signals(gap_min, gap_max, label):
    """
    gap_min <= MA200_GAP < gap_max 인 경우만 진입
    gap_min, gap_max 단위: % (음수 = MA200 아래)
    """
    signals_by_date = {}
    for tk, d in stock_data.items():
        d_c    = d.dropna(subset=["MA200","RSI","CCI","MA200_GAP"])
        common = d_c.index.intersection(vix.index)
        if len(common) < 50: continue
        vx  = vix.reindex(common)
        gap = d_c["MA200_GAP"].reindex(common)
        rsi = d_c["RSI"].reindex(common)
        cci = d_c["CCI"].reindex(common)

        cond = (
            (gap >= gap_min) & (gap < gap_max) &
            (vx >= VIX_MIN) &
            ((rsi < 40) | (cci < -100))
        )
        for sig_day in common[cond.reindex(common).fillna(False)]:
            idx = d_c.index.get_loc(sig_day)
            if idx + 1 >= len(d_c): continue
            entry_day  = d_c.index[idx + 1]
            entry_open = float(d_c["Open"].iloc[idx + 1])
            if pd.isna(entry_open): continue
            row = d_c.loc[sig_day]
            signals_by_date.setdefault(entry_day, []).append({
                "ticker"   : tk,
                "sig_day"  : sig_day,
                "entry_day": entry_day,
                "entry"    : entry_open,
                "rsi"      : float(row["RSI"]),
                "cci"      : float(row["CCI"]),
                "close_sig": float(row["Close"]),
                "ma200_gap": float(row["MA200_GAP"]),
            })

    final = []
    for entry_day, items in sorted(signals_by_date.items()):
        items.sort(key=lambda x: x["rsi"])
        final.extend(items[:MAX_DAILY])
    print(f"  [{label}] 신호 {len(final):,}건 ({len(signals_by_date)}거래일)")
    return final


# ──────────────────────────────────────────────
# 시뮬레이션
# ──────────────────────────────────────────────
def run_simulation(signals, label):
    trades        = []
    pos_exit_date = {}

    for sig in signals:
        tk        = sig["ticker"]
        entry_day = sig["entry_day"]
        entry     = sig["entry"]

        active = {t: ed for t, ed in pos_exit_date.items() if ed > entry_day}
        if len(active) >= MAX_POSITIONS or tk in active: continue

        d      = stock_data[tk]
        future = d.loc[d.index >= entry_day]
        if len(future) == 0: continue

        circuit     = entry * (1 - CIRCUIT_PCT)
        target      = entry * (1 + TARGET_PCT)
        half_exited = False
        exit_records= []

        for i, (fdt, row) in enumerate(future.iterrows()):
            lo = float(row["Low"])
            hi = float(row["High"])
            cl = float(row["Close"])

            if hi >= target:
                exit_records.append((fdt, target, 0.5 if half_exited else 1.0, "target"))
                break
            if lo <= circuit:
                exit_records.append((fdt, circuit, 0.5 if half_exited else 1.0, "circuit"))
                break
            if i + 1 == HALF_EXIT and not half_exited and (cl - entry) / entry > 0:
                exit_records.append((fdt, cl, 0.5, "half_60d"))
                half_exited = True
                continue
            if i + 1 >= MAX_HOLD:
                exit_records.append((fdt, cl, 0.5 if half_exited else 1.0, "time"))
                break

        if not exit_records: continue

        total_pct   = sum(r[2] for r in exit_records)
        blended_ret = sum((r[1]-entry)/entry * r[2] for r in exit_records) / total_pct
        last_exit   = exit_records[-1]
        reason      = ("+".join(r[3] for r in exit_records)
                       if len(exit_records) > 1 else exit_records[0][3])

        trades.append({
            "entry_date" : entry_day,
            "exit_date"  : last_exit[0],
            "ticker"     : tk,
            "entry"      : entry,
            "exit_price" : last_exit[1],
            "return_pct" : blended_ret * 100,
            "hold_days"  : (last_exit[0] - entry_day).days,
            "exit_reason": reason,
            "win"        : blended_ret > 0,
            "rsi_entry"  : sig["rsi"],
            "ma200_gap"  : sig["ma200_gap"],
        })
        pos_exit_date[tk] = last_exit[0]

    df = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)
    print(f"  [{label}] 트레이드 {len(df)}건")
    return df


# ──────────────────────────────────────────────
# 통계
# ──────────────────────────────────────────────
def calc_stats(df):
    if df.empty: return {}
    wins   = df[df["win"]]
    losses = df[~df["win"]]
    n      = len(df)
    wr     = len(wins) / n * 100
    ar     = df["return_pct"].mean()
    aw     = wins["return_pct"].mean()   if len(wins)   else 0
    al     = losses["return_pct"].mean() if len(losses) else 0
    pf     = (wins["return_pct"].sum() / -losses["return_pct"].sum()
              if len(losses) and losses["return_pct"].sum() < 0 else np.nan)
    ev     = wr/100 * aw + (1 - wr/100) * al

    max_cl = cur = 0
    for r in df["return_pct"]:
        cur = cur+1 if r < 0 else 0
        max_cl = max(max_cl, cur)

    capital = 1.0
    for _, group in df.groupby("entry_date"):
        w     = 1.0 / max(len(group), MAX_POSITIONS)
        batch = (group["return_pct"] / 100 * w).sum()
        capital *= (1 + batch)
    years = (df["exit_date"].max() - df["entry_date"].min()).days / 365.25
    cagr  = capital ** (1 / max(years, 0.01)) - 1

    return {
        "n": n, "wr": wr, "ar": ar, "aw": aw, "al": al,
        "pf": pf, "ev": ev, "max_cl": max_cl,
        "cagr": cagr * 100,
        "avg_hold": df["hold_days"].mean(),
        "avg_gap" : df["ma200_gap"].mean(),
        "exit_cnt": df["exit_reason"].value_counts(),
    }


# ──────────────────────────────────────────────
# 실행 — 3개 그룹
# ──────────────────────────────────────────────
print("\n🔍 신호 생성 중...")
# 원본: 전체 MA200 아래
sigs_all   = build_signals(-999,  0, "원본 전체")
# 밴드: 0~10% 아래
sigs_band  = build_signals(-10,   0, "밴드 0~10%")
# 깊은 이격: 10% 초과 아래
sigs_deep  = build_signals(-999, -10, "깊은 >10%")

print("\n⚙️  시뮬레이션 중...")
df_all  = run_simulation(sigs_all,  "원본 전체")
df_band = run_simulation(sigs_band, "밴드 0~10%")
df_deep = run_simulation(sigs_deep, "깊은 >10%")

sa  = calc_stats(df_all)
sb  = calc_stats(df_band)
sd  = calc_stats(df_deep)


# ──────────────────────────────────────────────
# 결과 출력
# ──────────────────────────────────────────────
print("\n" + "="*80)
print("  MA200 이격 구간 비교: 밴드(0~10%) vs 깊은 이격(>10%)")
print("="*80)
print(f"  기간      : {START} ~ {END}")
print(f"  공통 조건 : VIX≥25 + (RSI<40 OR CCI<-100), 다음날 시가 진입")
print("="*80)

fmt = "  {:<22} {:>16} {:>16} {:>16}"
print(fmt.format("지표", "원본(전체 <MA200)", "밴드(0~10%↓)", "깊은(>10%↓)"))
print("  " + "-"*70)

def pf_s(v): return f"{v:.2f}" if not (isinstance(v,float) and np.isnan(v)) else "N/A"

rows = [
    ("총 트레이드",      f"{sa['n']}건",            f"{sb['n']}건",            f"{sd['n']}건"),
    ("승 률",            f"{sa['wr']:.1f}%",         f"{sb['wr']:.1f}%",         f"{sd['wr']:.1f}%"),
    ("평균 수익률",      f"{sa['ar']:+.2f}%",        f"{sb['ar']:+.2f}%",        f"{sd['ar']:+.2f}%"),
    ("기대값(EV)",       f"{sa['ev']:+.2f}%",        f"{sb['ev']:+.2f}%",        f"{sd['ev']:+.2f}%"),
    ("승자 평균",        f"{sa['aw']:+.2f}%",        f"{sb['aw']:+.2f}%",        f"{sd['aw']:+.2f}%"),
    ("패자 평균",        f"{sa['al']:+.2f}%",        f"{sb['al']:+.2f}%",        f"{sd['al']:+.2f}%"),
    ("Profit Factor",    pf_s(sa['pf']),             pf_s(sb['pf']),             pf_s(sd['pf'])),
    ("포트CAGR",         f"{sa['cagr']:+.2f}%",      f"{sb['cagr']:+.2f}%",      f"{sd['cagr']:+.2f}%"),
    ("최대 연속 손실",   f"{sa['max_cl']}건",         f"{sb['max_cl']}건",         f"{sd['max_cl']}건"),
    ("평균 보유 일수",   f"{sa['avg_hold']:.0f}일",   f"{sb['avg_hold']:.0f}일",   f"{sd['avg_hold']:.0f}일"),
    ("평균 진입 이격",   f"{sa['avg_gap']:.1f}%",     f"{sb['avg_gap']:.1f}%",     f"{sd['avg_gap']:.1f}%"),
]
for row in rows:
    print(fmt.format(*row))
print("="*80)

for label, s in [("원본", sa), ("밴드 0~10%", sb), ("깊은 >10%", sd)]:
    print(f"\n[{label}] 청산 유형:")
    for r, c in s["exit_cnt"].items():
        print(f"  {r:<28}: {c:4d}건 ({c/s['n']*100:.1f}%)")

# 연도별 비교
for label, df in [("원본 전체", df_all), ("밴드 0~10%", df_band), ("깊은 >10%", df_deep)]:
    df2 = df.copy(); df2["year"] = df2["entry_date"].dt.year
    y = df2.groupby("year").agg(
        trades  =("return_pct","count"),
        avg_ret =("return_pct","mean"),
        win_rate=("win", lambda x: x.mean()*100),
        avg_gap =("ma200_gap","mean"),
    )
    print(f"\n연도별 [{label}]:")
    print(y.round(2).to_string())

# 종목별 깊은 이격 성과 (어떤 종목이 많이 나오는지)
print("\n[깊은 이격 >10%] 종목별 성과:")
deep_by_tk = df_deep.groupby("ticker").agg(
    trades  =("return_pct","count"),
    avg_ret =("return_pct","mean"),
    win_rate=("win", lambda x: x.mean()*100),
    avg_gap =("ma200_gap","mean"),
).sort_values("avg_ret", ascending=False)
print(deep_by_tk.round(2).to_string())


# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
C_ALL  = "#2c3e50"
C_BAND = "#27ae60"
C_DEEP = "#e74c3c"

fig, axes = plt.subplots(3, 3, figsize=(21, 17))
fig.suptitle(
    f"MA200 이격 구간별 비교  |  기간: {START}~{END}  |  종목: {len(stock_data)}개\n"
    f"원본(전체): {sa['n']}건 | 승률 {sa['wr']:.1f}% | 평균 {sa['ar']:+.2f}% | CAGR {sa['cagr']:+.1f}%\n"
    f"밴드(0~10%↓): {sb['n']}건 | 승률 {sb['wr']:.1f}% | 평균 {sb['ar']:+.2f}% | CAGR {sb['cagr']:+.1f}%   "
    f"깊은(>10%↓): {sd['n']}건 | 승률 {sd['wr']:.1f}% | 평균 {sd['ar']:+.2f}% | CAGR {sd['cagr']:+.1f}%",
    fontsize=9, fontweight="bold"
)

clr_exit = {"target":"#27ae60","circuit":"#e74c3c","time":"#f39c12",
            "half_60d":"#3498db","half_60d+target":"#1abc9c",
            "half_60d+time":"#e67e22","half_60d+circuit":"#c0392b"}
def get_clr(k):
    for ck, cv in clr_exit.items():
        if ck in k: return cv
    return "#bdc3c7"

# [0,0] 수익률 분포 3개 겹치기
ax = axes[0, 0]
for df, c, lbl, s in [(df_all, C_ALL, "원본", sa),
                       (df_band, C_BAND, "밴드 0~10%", sb),
                       (df_deep, C_DEEP, "깊은 >10%", sd)]:
    ax.hist(df["return_pct"], bins=35, alpha=0.5, color=c, edgecolor="white",
            label=f"{lbl} (avg {s['ar']:+.2f}%)")
ax.axvline(0, color="black", lw=1.5, linestyle="--")
ax.set_title("수익률 분포 비교")
ax.legend(fontsize=8)

# [0,1] 청산 유형 — 밴드
ax = axes[0, 1]
ecb = sb["exit_cnt"].head(7)
ax.pie(ecb.values,
       labels=[f"{k[:18]}\n{v}건({v/sb['n']*100:.0f}%)" for k,v in ecb.items()],
       colors=[get_clr(k) for k in ecb.index],
       startangle=140, textprops={"fontsize": 7})
ax.set_title("밴드(0~10%) 청산 유형")

# [0,2] 청산 유형 — 깊은
ax = axes[0, 2]
ecd = sd["exit_cnt"].head(7)
ax.pie(ecd.values,
       labels=[f"{k[:18]}\n{v}건({v/sd['n']*100:.0f}%)" for k,v in ecd.items()],
       colors=[get_clr(k) for k in ecd.index],
       startangle=140, textprops={"fontsize": 7})
ax.set_title("깊은(>10%) 청산 유형")

# [1,0] 누적 수익률
ax = axes[1, 0]
for df, c, lbl, s in [(df_all, C_ALL, "원본", sa),
                       (df_band, C_BAND, "밴드 0~10%", sb),
                       (df_deep, C_DEEP, "깊은 >10%", sd)]:
    cum = (1 + df["return_pct"] / 100).cumprod() - 1
    ax.plot(range(len(cum)), cum*100, color=c, lw=1.8,
            label=f"{lbl} CAGR {s['cagr']:+.1f}%")
ax.axhline(0, color="black", linestyle="--", lw=1)
ax.set_title("누적 수익률 비교")
ax.set_xlabel("Trade #")
ax.legend(fontsize=8)

# [1,1] 연도별 평균 수익률 — 밴드
ax = axes[1, 1]
df_band2 = df_band.copy(); df_band2["year"] = df_band2["entry_date"].dt.year
yb = df_band2.groupby("year").agg(trades=("return_pct","count"), avg_ret=("return_pct","mean"))
bar_c = ["#27ae60" if v >= 0 else "#e74c3c" for v in yb["avg_ret"]]
ax.bar(yb.index.astype(str), yb["avg_ret"], color=bar_c, edgecolor="white")
ax.axhline(0, color="black", lw=0.8)
for x, (yr, row) in enumerate(yb.iterrows()):
    ax.text(x, row["avg_ret"]+(0.4 if row["avg_ret"]>=0 else -1.5),
            f'{int(row["trades"])}건', ha="center", fontsize=7.5)
ax.set_title("밴드(0~10%) 연도별 평균 수익률")
ax.tick_params(axis="x", rotation=45)

# [1,2] 연도별 평균 수익률 — 깊은
ax = axes[1, 2]
df_deep2 = df_deep.copy(); df_deep2["year"] = df_deep2["entry_date"].dt.year
yd = df_deep2.groupby("year").agg(trades=("return_pct","count"), avg_ret=("return_pct","mean"))
bar_c = ["#27ae60" if v >= 0 else "#e74c3c" for v in yd["avg_ret"]]
ax.bar(yd.index.astype(str), yd["avg_ret"], color=bar_c, edgecolor="white")
ax.axhline(0, color="black", lw=0.8)
for x, (yr, row) in enumerate(yd.iterrows()):
    ax.text(x, row["avg_ret"]+(0.4 if row["avg_ret"]>=0 else -1.5),
            f'{int(row["trades"])}건', ha="center", fontsize=7.5)
ax.set_title("깊은(>10%) 연도별 평균 수익률")
ax.tick_params(axis="x", rotation=45)

# [2,0] 보유 기간 분포
ax = axes[2, 0]
for df, c, lbl, s in [(df_band, C_BAND, "밴드 0~10%", sb),
                       (df_deep, C_DEEP, "깊은 >10%", sd)]:
    ax.hist(df["hold_days"], bins=30, alpha=0.6, color=c, edgecolor="white",
            label=f"{lbl} avg {s['avg_hold']:.0f}일")
ax.set_title("보유 기간 분포 비교")
ax.set_xlabel("Hold Days")
ax.legend(fontsize=8)

# [2,1] 진입 이격 분포
ax = axes[2, 1]
ax.hist(df_band["ma200_gap"], bins=25, alpha=0.7, color=C_BAND, edgecolor="white",
        label=f"밴드 (avg {sb['avg_gap']:.1f}%)")
ax.hist(df_deep["ma200_gap"], bins=25, alpha=0.7, color=C_DEEP, edgecolor="white",
        label=f"깊은 (avg {sd['avg_gap']:.1f}%)")
ax.axvline(-10, color="black", linestyle="--", lw=1.5, label="-10% 경계")
ax.set_title("진입 시 MA200 이격 분포")
ax.set_xlabel("MA200 Gap %")
ax.legend(fontsize=8)

# [2,2] 핵심 지표 비교 막대
ax = axes[2, 2]
metrics = ["승률(%)", "평균수익(%)", "CAGR(%)"]
vals_all  = [sa["wr"], sa["ar"], sa["cagr"]]
vals_band = [sb["wr"], sb["ar"], sb["cagr"]]
vals_deep = [sd["wr"], sd["ar"], sd["cagr"]]
x = np.arange(len(metrics))
w = 0.25
ax.bar(x - w,   vals_all,  w, label="원본",       color=C_ALL,  alpha=0.8)
ax.bar(x,       vals_band, w, label="밴드 0~10%", color=C_BAND, alpha=0.8)
ax.bar(x + w,   vals_deep, w, label="깊은 >10%",  color=C_DEEP, alpha=0.8)
for i, (a, b, d) in enumerate(zip(vals_all, vals_band, vals_deep)):
    ax.text(i - w,   a + 0.3, f"{a:.1f}", ha="center", fontsize=7.5)
    ax.text(i,       b + 0.3, f"{b:.1f}", ha="center", fontsize=7.5)
    ax.text(i + w,   d + 0.3, f"{d:.1f}", ha="center", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("핵심 지표 비교")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("backtest_ma200_band.png", dpi=150, bbox_inches="tight")
df_all.to_csv("backtest_ma200_band_all.csv",   index=False)
df_band.to_csv("backtest_ma200_band_near.csv", index=False)
df_deep.to_csv("backtest_ma200_band_deep.csv", index=False)
print("\n📊 backtest_ma200_band.png 저장 완료")
print("📄 CSV 3개 저장 완료")
print("✅ 완료")
