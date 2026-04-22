"""
backtest_bb_target_range.py
============================
BB 기본 조건 (현재가 > MA200 + %B_D ≤ 5) 기준으로
목표수익률 +5% ~ +30% 전 구간 비교

손절(-25%) / 60일절반 / 120일타임 은 고정
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings("ignore")

START         = "2015-01-01"
END           = "2026-01-01"
MAX_POSITIONS = 5
MAX_DAILY     = 5
CIRCUIT_PCT   = 0.25
HALF_DAYS     = 60
MAX_HOLD      = 120
BB_PERIOD     = 20
BB_STD        = 2.0

KR_TICKERS = [
    "000660.KS","005930.KS","277810.KS","000150.KS","034020.KS",
    "005380.KS","329180.KS","267260.KS","298040.KS","010120.KS",
    "012450.KS","042660.KS","039030.KQ","060280.KQ","199430.KQ",
    "042700.KQ","096770.KS","009150.KS","373220.KS","000270.KS",
    "207940.KS","105560.KS","005490.KS","140410.KQ","247540.KQ",
    "357780.KQ","196170.KQ","079550.KS",
]
US_TICKERS = [
    "SNPS","COST","AZN","AMGN","MDLZ","FTNT","CDNS","ADP","FAST",
    "ADI","TXN","PAYX","BKNG","MNST","ORLY","HOOD","CPRT","ISRG",
    "CDW","INTC","AAPL","AVGO","AMD","MSFT","GOOGL","NVDA","TSLA",
    "MCHP","AMZN","AMAT","MU","LRCX","CSX","QCOM","ROP","ON",
    "STX","SNDK","ASTS","AVAV","IONQ","SGML","RKLB",
]

_kr_fonts = [f.name for f in fm.fontManager.ttflist
             if "Apple" in f.name or "Malgun" in f.name
             or "NanumGothic" in f.name or "Noto" in f.name]
if _kr_fonts:
    plt.rcParams["font.family"] = _kr_fonts[0]
plt.rcParams["axes.unicode_minus"] = False

# ─────────────────────────────────────────
# 데이터 로드 & 지표 계산
# ─────────────────────────────────────────
def dl_batch(tickers, start, end, min_rows=200):
    tickers = list(dict.fromkeys(tickers))
    try:
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False, group_by="ticker")
    except Exception:
        return {}
    result = {}
    if raw.empty: return result
    if not isinstance(raw.columns, pd.MultiIndex):
        if len(tickers) == 1:
            cols = [c for c in ["Open","High","Low","Close","Volume"] if c in raw.columns]
            if len(cols) == 5 and len(raw) >= min_rows:
                result[tickers[0]] = raw[cols].copy()
        return result
    for tk in tickers:
        try:
            sub = raw[tk][["Open","High","Low","Close","Volume"]].dropna(how="all")
            if len(sub) >= min_rows:
                result[tk] = sub.copy()
        except Exception:
            continue
    return result

def compute(d):
    d = d.copy()
    c = d["Close"]
    d["MA200"]   = c.rolling(200).mean()
    bb_ma        = c.rolling(BB_PERIOD).mean()
    bb_std_s     = c.rolling(BB_PERIOD).std(ddof=0)
    bb_upper     = bb_ma + BB_STD * bb_std_s
    bb_lower     = bb_ma - BB_STD * bb_std_s
    bb_range     = bb_upper - bb_lower
    d["PCT_B_D"] = ((c - bb_lower) / bb_range.replace(0, np.nan)) * 100
    # RSI (로그용)
    delta = c.diff()
    g  = delta.clip(lower=0).rolling(14).mean()
    lo = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100 / (1 + g / lo.replace(0, np.nan))
    return d

print("📥 데이터 다운로드...")
raw_kr = dl_batch(KR_TICKERS, START, END)
raw_us = dl_batch(US_TICKERS, START, END)
stock_data = {tk: compute(d) for tk, d in {**raw_kr, **raw_us}.items()}
print(f"  ✅ 한국 {len(raw_kr)}개 + 미국 {len(raw_us)}개 = 총 {len(stock_data)}개")

# ─────────────────────────────────────────
# 신호 생성 — BB 기본 (종가 %B ≤ 5, MA200 위)
# ─────────────────────────────────────────
def build_bb_signals():
    rows = []
    for tk, d in stock_data.items():
        d_c = d.dropna(subset=["MA200","PCT_B_D"])
        if len(d_c) < 2: continue
        cond = (
            (d_c["Close"]   > d_c["MA200"]) &
            (d_c["PCT_B_D"] <= 5)
        ).fillna(False)
        for i in np.where(cond.values)[0]:
            if i+1 >= len(d_c): continue
            ed = d_c.index[i+1]
            eo = float(d_c["Open"].iloc[i+1])
            if pd.isna(eo): continue
            rsi_val = float(d_c["RSI"].iloc[i]) if not pd.isna(d_c["RSI"].iloc[i]) else 50
            rows.append({"ticker": tk, "entry_day": ed, "entry": eo, "rsi": rsi_val})
    if not rows: return []
    df_s = pd.DataFrame(rows).sort_values(["entry_day","rsi"])
    df_s = df_s.drop_duplicates(subset=["entry_day","ticker"], keep="first")
    final = []
    for _, grp in df_s.groupby("entry_day"):
        final.extend(grp.to_dict("records")[:MAX_DAILY])
    return final

# ─────────────────────────────────────────
# 시뮬레이션
# ─────────────────────────────────────────
def run_sim(signals, target_pct):
    trades, pos_exit_date = [], {}
    for sig in signals:
        tk, entry_day, entry = sig["ticker"], sig["entry_day"], sig["entry"]
        active = {t: ed for t, ed in pos_exit_date.items() if ed > entry_day}
        if len(active) >= MAX_POSITIONS or tk in active: continue
        d = stock_data.get(tk)
        if d is None: continue
        future = d.loc[d.index >= entry_day]
        if len(future) == 0: continue
        cb_price  = entry * (1 - CIRCUIT_PCT)
        tgt_price = entry * (1 + target_pct)
        half_exited, exit_records = False, []
        for i, (fdt, row) in enumerate(future.iterrows()):
            lo, hi, cl = float(row["Low"]), float(row["High"]), float(row["Close"])
            if hi >= tgt_price:
                exit_records.append((fdt, tgt_price, 0.5 if half_exited else 1.0, "target")); break
            if lo <= cb_price:
                exit_records.append((fdt, cb_price, 0.5 if half_exited else 1.0, "circuit")); break
            if i+1 == HALF_DAYS and not half_exited and (cl-entry)/entry > 0:
                exit_records.append((fdt, cl, 0.5, "half")); half_exited = True; continue
            if i+1 >= MAX_HOLD:
                exit_records.append((fdt, cl, 0.5 if half_exited else 1.0, "time")); break
        if not exit_records: continue
        total_w = sum(r[2] for r in exit_records)
        blended = sum((r[1]-entry)/entry * r[2] for r in exit_records) / total_w
        last_exit = exit_records[-1]
        reason = "+".join(r[3] for r in exit_records) if len(exit_records)>1 else exit_records[0][3]
        trades.append({
            "entry_date": entry_day, "exit_date": last_exit[0], "ticker": tk,
            "return_pct": blended*100, "hold_days": (last_exit[0]-entry_day).days,
            "exit_reason": reason, "win": blended > 0,
        })
        pos_exit_date[tk] = last_exit[0]
    return pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True) if trades else pd.DataFrame()

def calc_stats(df, label):
    if df.empty:
        return {"label": label, "n": 0, "win_rate": 0, "avg_ret": 0,
                "avg_win": 0, "avg_loss": 0, "pf": float("nan"),
                "ev": 0, "cagr": 0, "max_consec_loss": 0, "avg_hold": 0,
                "circuit_rate": 0, "target_rate": 0, "exit_dist": {}, "df": df}
    n = len(df)
    wins, losses = df[df["win"]], df[~df["win"]]
    wr   = len(wins)/n*100
    avg_w= wins["return_pct"].mean()   if len(wins)   else 0
    avg_l= losses["return_pct"].mean() if len(losses) else 0
    pf   = (wins["return_pct"].sum() / -losses["return_pct"].sum()
            if len(losses) and losses["return_pct"].sum()<0 else float("nan"))
    ev   = wr/100*avg_w + (1-wr/100)*avg_l
    max_cl = cur = 0
    for r in df["return_pct"]:
        cur = cur+1 if r<0 else 0; max_cl = max(max_cl, cur)
    cap = 1.0
    for _, grp in df.groupby("entry_date"):
        cap *= (1 + (grp["return_pct"]/100 * (1.0/max(len(grp), MAX_POSITIONS))).sum())
    yrs  = (df["exit_date"].max()-df["entry_date"].min()).days/365.25
    cagr = cap**(1/yrs)-1 if yrs>0 else 0
    ec   = dict(df["exit_reason"].value_counts())
    tt   = sum(ec.values()) or 1
    circuit_rate = (ec.get("circuit",0)) / tt * 100
    target_rate  = (ec.get("target",0) + ec.get("half+target",0) + ec.get("half",0)) / tt * 100
    return {
        "label": label, "n": n, "win_rate": wr,
        "avg_ret": df["return_pct"].mean(), "avg_win": avg_w, "avg_loss": avg_l,
        "pf": pf, "ev": ev, "cagr": cagr*100, "max_consec_loss": max_cl,
        "avg_hold": df["hold_days"].mean(),
        "circuit_rate": circuit_rate, "target_rate": target_rate,
        "exit_dist": ec, "df": df,
    }

def pf_s(r): return f"{r['pf']:.2f}" if not np.isnan(r['pf']) else " N/A"

def score(r):
    pf_val = r["pf"] if not np.isnan(r["pf"]) else 0
    return r["win_rate"]*0.30 + r["cagr"]*0.30 + r["ev"]*0.25 + min(pf_val,5)*0.15*10

# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────
print("\n⚙️  신호 생성...")
signals = build_bb_signals()
print(f"  BB 신호 {len(signals)}건")

TARGET_LIST = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
results = {}
for tgt in TARGET_LIST:
    lbl = f"+{int(tgt*100)}%"
    r   = calc_stats(run_sim(signals, tgt), lbl)
    results[lbl] = r
    print(f"  [{lbl}] {r['n']}건 | 승률 {r['win_rate']:.1f}% | CAGR {r['cagr']:+.1f}% | PF {pf_s(r)} | 손절율 {r['circuit_rate']:.1f}% | 목표달성 {r['target_rate']:.1f}%")

# ─────────────────────────────────────────
# 결과 출력
# ─────────────────────────────────────────
HDR = f"\n  {'목표수익률':<8} {'건수':>5} {'승률':>7} {'평균수익':>9} {'평균익':>8} {'평균손':>8} {'PF':>6} {'기대값':>8} {'CAGR':>8} {'연손':>4} {'보유':>6} {'손절율':>7} {'목표달성':>8}"
SEP = "  " + "-"*110

print("\n" + "="*112)
print("  BB 기본 전략 (현재가>MA200 + %B≤5) — 목표수익률별 비교 (2015-2026, 손절-25% 고정)")
print("="*112)
print(HDR); print(SEP)

best_score = max(score(r) for r in results.values())
for lbl, r in results.items():
    mark = " ★" if abs(score(r) - best_score) < 0.01 else ""
    print(f"  {r['label']:<8} {r['n']:>5} {r['win_rate']:>6.1f}% {r['avg_ret']:>+8.2f}% "
          f"{r['avg_win']:>+7.2f}% {r['avg_loss']:>+7.2f}% {pf_s(r):>6} "
          f"{r['ev']:>+7.2f}% {r['cagr']:>+7.1f}% "
          f"{r['max_consec_loss']:>3}건 {r['avg_hold']:>5.0f}일 "
          f"{r['circuit_rate']:>6.1f}% {r['target_rate']:>7.1f}%{mark}")
print("="*112)

# 손절율 강조 출력
print("\n[손절(-25%) 비율 — 목표수익률이 낮을수록 손절 전에 탈출 가능한지 확인]")
for lbl, r in results.items():
    bar = "█" * int(r["circuit_rate"] / 2)
    print(f"  {lbl:<6} 손절 {r['circuit_rate']:>5.1f}%  {bar}")

# 연도별 손절 건수
print("\n[연도별 손절 건수 비교 — +10% vs +20% vs +30%]")
show_tgts = ["+10%", "+20%", "+30%"]
all_years = sorted(set().union(*[
    set(results[t]["df"]["entry_date"].dt.year)
    for t in show_tgts if not results[t]["df"].empty
]))
hdr2 = f"  {'연도':<6}" + "".join(f" {t:>22}" for t in show_tgts)
print(hdr2); print("  " + "-"*76)
for yr in all_years:
    row_str = f"  {yr:<6}"
    for t in show_tgts:
        df_y = results[t]["df"]
        if df_y.empty:
            row_str += f" {'  -':>22}"; continue
        sub = df_y[df_y["entry_date"].dt.year==yr]
        if not len(sub):
            row_str += f" {'  -':>22}"
        else:
            cb_cnt = sub["exit_reason"].str.contains("circuit").sum()
            row_str += f" {sub['win'].mean()*100:.0f}%/{sub['return_pct'].mean():+.1f}%(손절{cb_cnt}건)"
    print(row_str)

# ─────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("BB 기본 전략 목표수익률별 비교 (MA200↑ + %B≤5, 손절-25% 고정, 2015-2026)",
             fontweight="bold", fontsize=13)

lbls  = [r["label"] for r in results.values()]
colors = ["#e74c3c","#e67e22","#f1c40f","#2ecc71","#3498db","#9b59b6"]

for ax, (metric, title) in zip(axes.flatten(), [
    ("win_rate","승률(%)"), ("avg_ret","평균수익(%)"), ("cagr","CAGR(%)"),
    ("ev","기대값EV(%)"), ("circuit_rate","손절 비율(%)"), ("avg_hold","평균보유(일)")
]):
    vals  = [r[metric] if not np.isnan(r[metric]) else 0 for r in results.values()]
    bars  = ax.bar(range(len(lbls)), vals, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(len(lbls))); ax.set_xticklabels(lbls)
    ax.axhline(0, color="black", lw=0.5)
    ylim = ax.get_ylim(); rng = ylim[1]-ylim[0]
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+rng*0.02,
                f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("backtest_bb_target_range.png", dpi=150, bbox_inches="tight")
print("\n📊 차트 저장 완료")
print("✅ 완료")
