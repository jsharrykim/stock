"""
backtest_ma200_proximity.py — MA200 근접 구간별 비교
====================================================

현재가 기준 MA200 대비 진입 구간:
  원본:    현재가 < MA200          (MA200 아래)
  +5%:     현재가 < MA200 * 1.05   (MA200 위 5% 이내 포함)
  +10%:    현재가 < MA200 * 1.10   (MA200 위 10% 이내 포함)

즉 +5% 조건은 MA200 아직 안 깼어도 5% 이내면 진입 허용.
나머지 조건 동일: VIX≥25, RSI<40 OR CCI<-100, 다음날 시가
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

# 비교 그룹 정의: (label, multiplier)
# 현재가 < MA200 * multiplier
GROUPS = [
    ("원본 (<MA200)",       1.00),
    ("+5%  (<MA200×1.05)", 1.05),
    ("+10% (<MA200×1.10)", 1.10),
]

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
    tp_mad = tp.rolling(20).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
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
def build_signals(multiplier, label):
    """현재가 < MA200 * multiplier"""
    signals_by_date = {}
    for tk, d in stock_data.items():
        d_c    = d.dropna(subset=["MA200","RSI","CCI","MA200_GAP"])
        common = d_c.index.intersection(vix.index)
        if len(common) < 50: continue
        vx    = vix.reindex(common)
        close = d_c["Close"].reindex(common)
        ma200 = d_c["MA200"].reindex(common)
        rsi   = d_c["RSI"].reindex(common)
        cci   = d_c["CCI"].reindex(common)

        cond = (
            (close < ma200 * multiplier) &
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
                "ma200_gap": float(row["MA200_GAP"]),
                "above_ma" : float(row["MA200_GAP"]) > 0,   # MA200 위인지 여부
            })

    final = []
    for entry_day, items in sorted(signals_by_date.items()):
        items.sort(key=lambda x: x["rsi"])
        final.extend(items[:MAX_DAILY])

    n_above = sum(1 for s in final if s["above_ma"])
    print(f"  [{label}] 신호 {len(final):,}건  (MA200 위 진입 {n_above}건 포함)")
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
            "above_ma"   : sig["above_ma"],
        })
        pos_exit_date[tk] = last_exit[0]

    df = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)
    n_above = df["above_ma"].sum() if "above_ma" in df.columns else 0
    print(f"  [{label}] 트레이드 {len(df)}건  (MA200 위 진입 {n_above}건)")
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

    exit_detail = df.groupby("exit_reason")["return_pct"].agg(["count","mean"]).sort_values("count", ascending=False)

    # MA200 위/아래 분리 성과
    above = df[df["above_ma"]]  if "above_ma" in df.columns else pd.DataFrame()
    below = df[~df["above_ma"]] if "above_ma" in df.columns else df

    return {
        "n": n, "wr": wr, "ar": ar, "aw": aw, "al": al,
        "pf": pf, "ev": ev, "max_cl": max_cl,
        "cagr"       : cagr * 100,
        "avg_hold"   : df["hold_days"].mean(),
        "med_hold"   : df["hold_days"].median(),
        "avg_gap"    : df["ma200_gap"].mean(),
        "exit_cnt"   : df["exit_reason"].value_counts(),
        "exit_detail": exit_detail,
        "n_above"    : len(above),
        "wr_above"   : above["win"].mean()*100 if len(above) > 0 else np.nan,
        "ar_above"   : above["return_pct"].mean() if len(above) > 0 else np.nan,
        "wr_below"   : below["win"].mean()*100 if len(below) > 0 else np.nan,
        "ar_below"   : below["return_pct"].mean() if len(below) > 0 else np.nan,
    }


# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────
print("\n🔍 신호 생성 중...")
all_signals  = {label: build_signals(mult, label) for label, mult in GROUPS}

print("\n⚙️  시뮬레이션 중...")
all_dfs   = {label: run_simulation(sigs, label) for label, sigs in all_signals.items()}
all_stats = {label: calc_stats(df) for label, df in all_dfs.items()}


# ──────────────────────────────────────────────
# 결과 출력
# ──────────────────────────────────────────────
labels = [g[0] for g in GROUPS]

print("\n" + "="*80)
print("  MA200 근접 구간 비교: 원본 vs +5% vs +10%")
print("="*80)
print(f"  기간: {START}~{END}  |  VIX≥25, RSI<40 OR CCI<-100, 다음날 시가")
print("="*80)

fmt = "  {:<24} " + " {:>16}" * len(GROUPS)
print(fmt.format("지표", *[g[0] for g in GROUPS]))
print("  " + "-"*76)

pf_s = lambda v: f"{v:.2f}" if not (isinstance(v,float) and np.isnan(v)) else "N/A"
na_s = lambda v: f"{v:.1f}%" if not (isinstance(v,float) and np.isnan(v)) else "N/A"

stat_rows = [
    ("총 트레이드",      lambda s: f"{s['n']}건"),
    ("승 률",            lambda s: f"{s['wr']:.1f}%"),
    ("평균 수익률",      lambda s: f"{s['ar']:+.2f}%"),
    ("기대값(EV)",       lambda s: f"{s['ev']:+.2f}%"),
    ("승자 평균",        lambda s: f"{s['aw']:+.2f}%"),
    ("패자 평균",        lambda s: f"{s['al']:+.2f}%"),
    ("Profit Factor",    lambda s: pf_s(s['pf'])),
    ("포트CAGR",         lambda s: f"{s['cagr']:+.2f}%"),
    ("최대 연속 손실",   lambda s: f"{s['max_cl']}건"),
    ("평균 보유 일수",   lambda s: f"{s['avg_hold']:.0f}일"),
    ("중간값 보유",      lambda s: f"{s['med_hold']:.0f}일"),
    ("평균 진입 이격",   lambda s: f"{s['avg_gap']:.1f}%"),
    ("── MA200 위 진입", lambda s: f"{s['n_above']}건"),
    ("  위 진입 승률",   lambda s: na_s(s['wr_above'])),
    ("  위 진입 평균",   lambda s: f"{s['ar_above']:+.2f}%" if not np.isnan(s['ar_above']) else "N/A"),
    ("  아래 진입 승률", lambda s: f"{s['wr_below']:.1f}%"),
    ("  아래 진입 평균", lambda s: f"{s['ar_below']:+.2f}%"),
]
for row_label, fn in stat_rows:
    vals = [fn(all_stats[lbl]) for lbl in labels]
    print(fmt.format(row_label, *vals))
print("="*80)

for lbl in labels:
    s = all_stats[lbl]
    print(f"\n[{lbl}] 청산 유형:")
    for r in s["exit_cnt"].index:
        cnt = s["exit_cnt"][r]
        avg = s["exit_detail"].loc[r, "mean"]
        print(f"  {r:<30}: {cnt:4d}건 ({cnt/s['n']*100:.1f}%)  avg {avg:+.2f}%")

# MA200 위 진입 종목 상세 (+5% 기준)
lbl_5 = labels[1]
df_5  = all_dfs[lbl_5]
above_df = df_5[df_5["above_ma"]] if "above_ma" in df_5.columns else pd.DataFrame()
if len(above_df) > 0:
    print(f"\n[{lbl_5}] MA200 위 진입 상세 ({len(above_df)}건):")
    above_detail = above_df.groupby("ticker").agg(
        trades  =("return_pct","count"),
        avg_ret =("return_pct","mean"),
        win_rate=("win", lambda x: x.mean()*100),
        avg_gap =("ma200_gap","mean"),
        avg_hold=("hold_days","mean"),
    ).sort_values("avg_ret", ascending=False)
    print(above_detail.round(1).to_string())

# MA200 위 진입 종목 상세 (+10% 기준)
lbl_10 = labels[2]
df_10  = all_dfs[lbl_10]
above_df10 = df_10[df_10["above_ma"]] if "above_ma" in df_10.columns else pd.DataFrame()
if len(above_df10) > 0:
    print(f"\n[{lbl_10}] MA200 위 진입 상세 ({len(above_df10)}건):")
    above_detail10 = above_df10.groupby("ticker").agg(
        trades  =("return_pct","count"),
        avg_ret =("return_pct","mean"),
        win_rate=("win", lambda x: x.mean()*100),
        avg_gap =("ma200_gap","mean"),
        avg_hold=("hold_days","mean"),
    ).sort_values("avg_ret", ascending=False)
    print(above_detail10.round(1).to_string())

# 연도별
for lbl in labels:
    df2 = all_dfs[lbl].copy(); df2["year"] = df2["entry_date"].dt.year
    y = df2.groupby("year").agg(
        trades  =("return_pct","count"),
        avg_ret =("return_pct","mean"),
        win_rate=("win", lambda x: x.mean()*100),
    )
    print(f"\n연도별 [{lbl}]:")
    print(y.round(2).to_string())


# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
COLORS = ["#2c3e50", "#2980b9", "#e74c3c"]

fig, axes = plt.subplots(3, 3, figsize=(21, 17))
fig.suptitle(
    f"MA200 근접 구간 비교  |  {START}~{END}  |  VIX≥25 + RSI<40/CCI<-100\n" +
    "  |  ".join([
        f"{g[0]}: {all_stats[g[0]]['n']}건 | 승률 {all_stats[g[0]]['wr']:.1f}% | "
        f"평균 {all_stats[g[0]]['ar']:+.2f}% | CAGR {all_stats[g[0]]['cagr']:+.1f}%"
        for g in GROUPS
    ]),
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
for (lbl, _), c in zip(GROUPS, COLORS):
    s = all_stats[lbl]
    ax.hist(all_dfs[lbl]["return_pct"], bins=35, alpha=0.5, color=c,
            edgecolor="white", label=f"{lbl} (avg {s['ar']:+.2f}%)")
ax.axvline(0, color="black", lw=1.5, linestyle="--")
ax.set_title("수익률 분포 비교")
ax.legend(fontsize=7.5)

# [0,1] 청산 유형 — +5%
ax = axes[0, 1]
s5 = all_stats[labels[1]]
ec = s5["exit_cnt"].head(7)
ax.pie(ec.values,
       labels=[f"{k[:18]}\n{v}건({v/s5['n']*100:.0f}%)" for k,v in ec.items()],
       colors=[get_clr(k) for k in ec.index],
       startangle=140, textprops={"fontsize":7})
ax.set_title(f"{labels[1]} 청산 유형")

# [0,2] 청산 유형 — +10%
ax = axes[0, 2]
s10 = all_stats[labels[2]]
ec = s10["exit_cnt"].head(7)
ax.pie(ec.values,
       labels=[f"{k[:18]}\n{v}건({v/s10['n']*100:.0f}%)" for k,v in ec.items()],
       colors=[get_clr(k) for k in ec.index],
       startangle=140, textprops={"fontsize":7})
ax.set_title(f"{labels[2]} 청산 유형")

# [1,0] 누적 수익률
ax = axes[1, 0]
for (lbl, _), c in zip(GROUPS, COLORS):
    df = all_dfs[lbl]
    cum = (1 + df["return_pct"]/100).cumprod() - 1
    ax.plot(range(len(cum)), cum*100, color=c, lw=2,
            label=f"{lbl} CAGR {all_stats[lbl]['cagr']:+.1f}%")
ax.axhline(0, color="black", linestyle="--", lw=1)
ax.set_title("누적 수익률 비교")
ax.set_xlabel("Trade #"); ax.legend(fontsize=7.5)

# [1,1] 진입 이격 분포
ax = axes[1, 1]
for (lbl, _), c in zip(GROUPS, COLORS):
    df = all_dfs[lbl]
    s  = all_stats[lbl]
    ax.hist(df["ma200_gap"], bins=30, alpha=0.5, color=c, edgecolor="white",
            label=f"{lbl} (avg {s['avg_gap']:.1f}%)")
ax.axvline(0,  color="black", linestyle="--", lw=1.5, label="MA200 경계")
ax.axvline(5,  color="blue",  linestyle=":",  lw=1.2, label="+5%")
ax.axvline(10, color="red",   linestyle=":",  lw=1.2, label="+10%")
ax.set_title("진입 시 MA200 이격 분포")
ax.set_xlabel("MA200 Gap %"); ax.legend(fontsize=7)

# [1,2] MA200 위/아래 진입 승률 비교 (stacked bar)
ax = axes[1, 2]
x       = np.arange(2)   # 위 / 아래
wid     = 0.25
offsets = [-wid, 0, wid]
for i, ((lbl, _), c) in enumerate(zip(GROUPS, COLORS)):
    s = all_stats[lbl]
    wrs = [
        s["wr_above"] if not np.isnan(s["wr_above"]) else 0,
        s["wr_below"],
    ]
    bars = ax.bar(x + offsets[i], wrs, wid, color=c, alpha=0.8,
                  label=lbl, edgecolor="white")
    for bar, v in zip(bars, wrs):
        if v > 0:
            ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f"{v:.0f}%",
                    ha="center", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(["MA200 위 진입", "MA200 아래 진입"], fontsize=9)
ax.set_ylim(0, 105); ax.axhline(50, color="gray", linestyle="--", lw=0.8)
ax.set_title("MA200 위/아래 진입 승률 비교")
ax.set_ylabel("승률 %"); ax.legend(fontsize=7.5)

# [2,0] 보유 기간 분포
ax = axes[2, 0]
for (lbl, _), c in zip(GROUPS, COLORS):
    s = all_stats[lbl]
    ax.hist(all_dfs[lbl]["hold_days"], bins=30, alpha=0.5, color=c,
            edgecolor="white", label=f"{lbl} avg {s['avg_hold']:.0f}일")
ax.set_title("보유 기간 분포"); ax.set_xlabel("Hold Days"); ax.legend(fontsize=7.5)

# [2,1] 연도별 비교 (히트맵)
ax = axes[2, 1]
all_years = sorted(set(
    y for lbl, _ in GROUPS
    for y in all_dfs[lbl]["entry_date"].dt.year.unique()
))
hm = []
for lbl, _ in GROUPS:
    df2 = all_dfs[lbl].copy(); df2["year"] = df2["entry_date"].dt.year
    yr_avg = df2.groupby("year")["return_pct"].mean()
    hm.append([yr_avg.get(y, np.nan) for y in all_years])
hm = np.array(hm, dtype=float)
im = ax.imshow(hm, aspect="auto", cmap="RdYlGn", vmin=-15, vmax=25)
ax.set_xticks(range(len(all_years)))
ax.set_xticklabels([str(y) for y in all_years], rotation=45, fontsize=7)
ax.set_yticks(range(len(GROUPS)))
ax.set_yticklabels([g[0] for g in GROUPS], fontsize=7.5)
ax.set_title("연도별 평균 수익률 히트맵")
for i in range(len(GROUPS)):
    for j in range(len(all_years)):
        v = hm[i,j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=6.5, color="black" if abs(v) < 15 else "white")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# [2,2] 핵심 지표 비교 막대
ax = axes[2, 2]
metrics = ["승률(%)", "평균수익(%)", "CAGR(%)"]
x3 = np.arange(len(metrics)); wid3 = 0.25
for i, ((lbl, _), c) in enumerate(zip(GROUPS, COLORS)):
    s = all_stats[lbl]
    vals = [s["wr"], s["ar"], s["cagr"]]
    bars = ax.bar(x3 + (i-1)*wid3, vals, wid3, color=c, alpha=0.85,
                  label=lbl, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.3, f"{v:.1f}",
                ha="center", fontsize=7)
ax.set_xticks(x3); ax.set_xticklabels(metrics, fontsize=9)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("핵심 지표 비교"); ax.legend(fontsize=7.5)

plt.tight_layout()
plt.savefig("backtest_ma200_proximity.png", dpi=150, bbox_inches="tight")
for lbl, _ in GROUPS:
    fname = lbl.replace(" ", "_").replace("<", "").replace("×", "x").replace("%","pct").replace("(","").replace(")","")
    all_dfs[lbl].to_csv(f"backtest_proximity_{fname}.csv", index=False)
print("\n📊 backtest_ma200_proximity.png 저장 완료")
print("📄 CSV 3개 저장 완료")
print("✅ 완료")
