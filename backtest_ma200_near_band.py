"""
backtest_ma200_near_band.py — 0~10% vs 5~10% 비교
==================================================

그룹 1: MA200 기준 0~10% 아래 (기존 밴드)
  -10% < MA200_GAP <= 0%

그룹 2: MA200 기준 5~10% 아래 (좁은 밴드)
  -10% < MA200_GAP <= -5%

나머지 공통:
  VIX ≥ 25, RSI < 40 OR CCI < -100
  진입: 다음날 시가 / 매도: +20% / -25%CB / 60일절반 / 120일
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
    d["MA200"]    = c.rolling(200).mean()
    delta = c.diff()
    g = delta.clip(lower=0).rolling(14).mean()
    l = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"]      = 100 - 100 / (1 + g / l.replace(0, np.nan))
    tp     = (d["High"] + d["Low"] + c) / 3
    tp_ma  = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    d["CCI"]      = (tp - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))
    d["MA200_GAP"] = (c - d["MA200"]) / d["MA200"] * 100
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
# 신호 생성
# ──────────────────────────────────────────────
def build_signals(gap_min, gap_max, label):
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

    # 청산 유형별 평균 수익률
    exit_detail = df.groupby("exit_reason")["return_pct"].agg(["count","mean"]).sort_values("count", ascending=False)

    return {
        "n": n, "wr": wr, "ar": ar, "aw": aw, "al": al,
        "pf": pf, "ev": ev, "max_cl": max_cl,
        "cagr"       : cagr * 100,
        "avg_hold"   : df["hold_days"].mean(),
        "med_hold"   : df["hold_days"].median(),
        "avg_gap"    : df["ma200_gap"].mean(),
        "exit_cnt"   : df["exit_reason"].value_counts(),
        "exit_detail": exit_detail,
        "hold_p25"   : df["hold_days"].quantile(0.25),
        "hold_p75"   : df["hold_days"].quantile(0.75),
    }

# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────
print("\n🔍 신호 생성 중...")
sigs_wide   = build_signals(-10,  0,  "0~10% 아래")
sigs_narrow = build_signals(-10, -5,  "5~10% 아래")

print("\n⚙️  시뮬레이션 중...")
df_wide   = run_simulation(sigs_wide,   "0~10%")
df_narrow = run_simulation(sigs_narrow, "5~10%")

sw = calc_stats(df_wide)
sn = calc_stats(df_narrow)

# ──────────────────────────────────────────────
# 출력
# ──────────────────────────────────────────────
print("\n" + "="*72)
print("  MA200 하방 구간 비교: 0~10% vs 5~10%")
print("="*72)
print(f"  기간: {START} ~ {END}  |  VIX≥25, RSI<40 OR CCI<-100, 다음날 시가")
print("="*72)

fmt = "  {:<22} {:>20} {:>20}"
print(fmt.format("지표", "0~10% 아래", "5~10% 아래"))
print("  " + "-"*62)
pf_s = lambda v: f"{v:.2f}" if not (isinstance(v,float) and np.isnan(v)) else "N/A"
rows = [
    ("총 트레이드",        f"{sw['n']}건",                f"{sn['n']}건"),
    ("승 률",              f"{sw['wr']:.1f}%",             f"{sn['wr']:.1f}%"),
    ("평균 수익률",        f"{sw['ar']:+.2f}%",            f"{sn['ar']:+.2f}%"),
    ("기대값(EV)",         f"{sw['ev']:+.2f}%",            f"{sn['ev']:+.2f}%"),
    ("승자 평균",          f"{sw['aw']:+.2f}%",            f"{sn['aw']:+.2f}%"),
    ("패자 평균",          f"{sw['al']:+.2f}%",            f"{sn['al']:+.2f}%"),
    ("Profit Factor",      pf_s(sw['pf']),                 pf_s(sn['pf'])),
    ("포트CAGR",           f"{sw['cagr']:+.2f}%",          f"{sn['cagr']:+.2f}%"),
    ("최대 연속 손실",     f"{sw['max_cl']}건",             f"{sn['max_cl']}건"),
    ("평균 보유 일수",     f"{sw['avg_hold']:.0f}일",       f"{sn['avg_hold']:.0f}일"),
    ("중간값 보유 일수",   f"{sw['med_hold']:.0f}일",       f"{sn['med_hold']:.0f}일"),
    ("25th pct 보유",      f"{sw['hold_p25']:.0f}일",       f"{sn['hold_p25']:.0f}일"),
    ("75th pct 보유",      f"{sw['hold_p75']:.0f}일",       f"{sn['hold_p75']:.0f}일"),
    ("평균 진입 이격",     f"{sw['avg_gap']:.1f}%",         f"{sn['avg_gap']:.1f}%"),
]
for row in rows:
    print(fmt.format(*row))
print("="*72)

for label, s in [("0~10%", sw), ("5~10%", sn)]:
    print(f"\n[{label}] 청산 유형 (건수 | 비중 | 평균 수익률):")
    for r in s["exit_cnt"].index:
        cnt  = s["exit_cnt"][r]
        avg  = s["exit_detail"].loc[r, "mean"]
        print(f"  {r:<28}: {cnt:4d}건 ({cnt/s['n']*100:.1f}%)  avg {avg:+.2f}%")

# 연도별
for label, df in [("0~10%", df_wide), ("5~10%", df_narrow)]:
    df2 = df.copy(); df2["year"] = df2["entry_date"].dt.year
    y = df2.groupby("year").agg(
        trades  =("return_pct","count"),
        avg_ret =("return_pct","mean"),
        win_rate=("win", lambda x: x.mean()*100),
        avg_hold=("hold_days","mean"),
        avg_gap =("ma200_gap","mean"),
    )
    print(f"\n연도별 [{label}]:")
    print(y.round(1).to_string())

# 종목별
for label, df in [("0~10%", df_wide), ("5~10%", df_narrow)]:
    t = df.groupby("ticker").agg(
        trades  =("return_pct","count"),
        avg_ret =("return_pct","mean"),
        win_rate=("win", lambda x: x.mean()*100),
        avg_hold=("hold_days","mean"),
        avg_gap =("ma200_gap","mean"),
    ).sort_values("avg_ret", ascending=False)
    print(f"\n종목별 [{label}]:")
    print(t.round(1).to_string())

# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
C_W = "#2980b9"   # 0~10% 파랑
C_N = "#27ae60"   # 5~10% 초록

fig, axes = plt.subplots(3, 3, figsize=(21, 17))
fig.suptitle(
    f"MA200 하방 구간 비교  |  {START} ~ {END}\n"
    f"0~10%↓: {sw['n']}건 | 승률 {sw['wr']:.1f}% | 평균 {sw['ar']:+.2f}% | CAGR {sw['cagr']:+.1f}% | 평균 보유 {sw['avg_hold']:.0f}일\n"
    f"5~10%↓: {sn['n']}건 | 승률 {sn['wr']:.1f}% | 평균 {sn['ar']:+.2f}% | CAGR {sn['cagr']:+.1f}% | 평균 보유 {sn['avg_hold']:.0f}일",
    fontsize=10, fontweight="bold"
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
ax.hist(df_wide["return_pct"],   bins=35, alpha=0.6, color=C_W, edgecolor="white",
        label=f"0~10% (avg {sw['ar']:+.2f}%)")
ax.hist(df_narrow["return_pct"], bins=35, alpha=0.6, color=C_N, edgecolor="white",
        label=f"5~10% (avg {sn['ar']:+.2f}%)")
ax.axvline(0, color="black", lw=1.5, linestyle="--")
ax.set_title("수익률 분포 비교")
ax.legend(fontsize=9)

# [0,1] 청산 유형 — 0~10%
ax = axes[0, 1]
ecw = sw["exit_cnt"].head(7)
ax.pie(ecw.values,
       labels=[f"{k[:18]}\n{v}건({v/sw['n']*100:.0f}%)" for k,v in ecw.items()],
       colors=[get_clr(k) for k in ecw.index],
       startangle=140, textprops={"fontsize": 7.5})
ax.set_title("0~10% 청산 유형")

# [0,2] 청산 유형 — 5~10%
ax = axes[0, 2]
ecn = sn["exit_cnt"].head(7)
ax.pie(ecn.values,
       labels=[f"{k[:18]}\n{v}건({v/sn['n']*100:.0f}%)" for k,v in ecn.items()],
       colors=[get_clr(k) for k in ecn.index],
       startangle=140, textprops={"fontsize": 7.5})
ax.set_title("5~10% 청산 유형")

# [1,0] 누적 수익률
ax = axes[1, 0]
for df, c, lbl, s in [(df_wide, C_W, "0~10%", sw), (df_narrow, C_N, "5~10%", sn)]:
    cum = (1 + df["return_pct"] / 100).cumprod() - 1
    ax.plot(range(len(cum)), cum*100, color=c, lw=2,
            label=f"{lbl} CAGR {s['cagr']:+.1f}%")
ax.axhline(0, color="black", linestyle="--", lw=1)
ax.set_title("누적 수익률")
ax.set_xlabel("Trade #")
ax.legend(fontsize=9)

# [1,1] 보유 기간 분포 (박스플롯 + 히스토그램 겹치기)
ax = axes[1, 1]
ax.hist(df_wide["hold_days"],   bins=30, alpha=0.6, color=C_W, edgecolor="white",
        label=f"0~10% (avg {sw['avg_hold']:.0f}일, med {sw['med_hold']:.0f}일)")
ax.hist(df_narrow["hold_days"], bins=30, alpha=0.6, color=C_N, edgecolor="white",
        label=f"5~10% (avg {sn['avg_hold']:.0f}일, med {sn['med_hold']:.0f}일)")
ax.axvline(sw["avg_hold"], color=C_W, linestyle="--", lw=1.5)
ax.axvline(sn["avg_hold"], color=C_N, linestyle="--", lw=1.5)
ax.set_title("보유 기간 분포")
ax.set_xlabel("Hold Days")
ax.legend(fontsize=8)

# [1,2] 보유 기간 박스플롯
ax = axes[1, 2]
bp = ax.boxplot([df_wide["hold_days"].values, df_narrow["hold_days"].values],
                labels=["0~10%", "5~10%"],
                patch_artist=True, medianprops={"color":"black","lw":2.5})
bp["boxes"][0].set_facecolor(C_W); bp["boxes"][0].set_alpha(0.7)
bp["boxes"][1].set_facecolor(C_N); bp["boxes"][1].set_alpha(0.7)
# 통계 표기
for i, (s, lbl) in enumerate([(sw, "0~10%"), (sn, "5~10%")], 1):
    ax.text(i, s["hold_p75"] + 3,
            f"avg {s['avg_hold']:.0f}일\nmed {s['med_hold']:.0f}일",
            ha="center", fontsize=8.5)
ax.set_title("보유 기간 Boxplot")
ax.set_ylabel("Hold Days")

# [2,0] 연도별 평균 수익률 비교 (나란히)
ax = axes[2, 0]
df_w2 = df_wide.copy();   df_w2["year"] = df_w2["entry_date"].dt.year
df_n2 = df_narrow.copy(); df_n2["year"] = df_n2["entry_date"].dt.year
yw = df_w2.groupby("year")["return_pct"].mean()
yn = df_n2.groupby("year")["return_pct"].mean()
all_years = sorted(set(yw.index) | set(yn.index))
x   = np.arange(len(all_years))
wid = 0.35
vw  = [yw.get(y, 0) for y in all_years]
vn  = [yn.get(y, 0) for y in all_years]
ax.bar(x - wid/2, vw, wid, color=C_W, alpha=0.8, label="0~10%", edgecolor="white")
ax.bar(x + wid/2, vn, wid, color=C_N, alpha=0.8, label="5~10%", edgecolor="white")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([str(y) for y in all_years], rotation=45, fontsize=8)
ax.set_title("연도별 평균 수익률 비교")
ax.legend(fontsize=8)

# [2,1] 진입 이격 분포
ax = axes[2, 1]
ax.hist(df_wide["ma200_gap"],   bins=25, alpha=0.7, color=C_W, edgecolor="white",
        label=f"0~10% (avg {sw['avg_gap']:.1f}%)")
ax.hist(df_narrow["ma200_gap"], bins=25, alpha=0.7, color=C_N, edgecolor="white",
        label=f"5~10% (avg {sn['avg_gap']:.1f}%)")
ax.axvline(-5,  color="gray",  linestyle=":", lw=1.5, label="-5% 경계")
ax.axvline(-10, color="black", linestyle="--", lw=1.5, label="-10% 경계")
ax.set_title("진입 시 MA200 이격 분포")
ax.set_xlabel("MA200 Gap %")
ax.legend(fontsize=7.5)

# [2,2] 핵심 수치 비교
ax = axes[2, 2]
metrics = ["승률(%)", "평균수익(%)", "CAGR(%)", "PF", "평균보유(일/10)"]
vals_w  = [sw["wr"], sw["ar"], sw["cagr"],
           sw["pf"] if not np.isnan(sw["pf"]) else 0, sw["avg_hold"]/10]
vals_n  = [sn["wr"], sn["ar"], sn["cagr"],
           sn["pf"] if not np.isnan(sn["pf"]) else 0, sn["avg_hold"]/10]
x2 = np.arange(len(metrics)); wid2 = 0.35
ax.bar(x2 - wid2/2, vals_w, wid2, color=C_W, alpha=0.85, label="0~10%", edgecolor="white")
ax.bar(x2 + wid2/2, vals_n, wid2, color=C_N, alpha=0.85, label="5~10%", edgecolor="white")
for i, (vw_, vn_) in enumerate(zip(vals_w, vals_n)):
    ax.text(i - wid2/2, vw_ + 0.3, f"{vw_:.1f}", ha="center", fontsize=7.5)
    ax.text(i + wid2/2, vn_ + 0.3, f"{vn_:.1f}", ha="center", fontsize=7.5)
ax.set_xticks(x2); ax.set_xticklabels(metrics, fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("핵심 지표 비교")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("backtest_ma200_near_band.png", dpi=150, bbox_inches="tight")
df_wide.to_csv("backtest_near_band_0to10.csv",  index=False)
df_narrow.to_csv("backtest_near_band_5to10.csv", index=False)
print("\n📊 backtest_ma200_near_band.png 저장 완료")
print("📄 CSV 2개 저장 완료")
print("✅ 완료")
