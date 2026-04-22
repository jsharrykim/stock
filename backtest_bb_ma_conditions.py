"""
backtest_bb_ma_conditions.py
=============================
볼린저밴드 전략에서
1) 어느 이평선 위 조건이 가장 좋은지 비교 (MA5/20/50/60/144/200)
2) 7가지 조건 중 어느 게 실제로 기여하는지 분석
3) 추가할 만한 새 조건 탐색 (전일 대비 낙폭, 연속 음봉, 섹터 MA 등)

[기준 전략] BB 기본: 이평선 위 + 저가≤BB하단, 목표+20%, CB-25%
[비교군]
  MA_X 시리즈: 현재가 > MA_X + 저가≤BB하단
  조건 추가 시리즈: BB기본(MA200) + 추가조건 하나씩

종목: 한국 28개 + 미국 43개 = 71개 / 기간: 2015-2026
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
TARGET_PCT    = 0.20

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
# 데이터 로드
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
    for p in [5, 20, 50, 60, 144, 200]:
        d[f"MA{p}"] = c.rolling(p).mean()
    bb_ma       = c.rolling(BB_PERIOD).mean()
    bb_std_s    = c.rolling(BB_PERIOD).std()
    d["BB_lower"] = bb_ma - BB_STD * bb_std_s
    d["BB_upper"] = bb_ma + BB_STD * bb_std_s
    delta = c.diff()
    g  = delta.clip(lower=0).rolling(14).mean()
    lo = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100 / (1 + g / lo.replace(0, np.nan))
    tp     = (d["High"] + d["Low"] + c) / 3
    tp_ma  = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    d["CCI"]    = (tp - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))
    d["HIGH20"] = d["High"].rolling(20).max()
    d["VOL20"]  = d["Volume"].rolling(20).mean()
    d["RET1"]   = c.pct_change(1)   # 전일 대비 등락률
    d["RET3"]   = c.pct_change(3)   # 3일 등락률
    # 연속 음봉 수
    neg = (c.diff() < 0).astype(int)
    consec = neg.copy().astype(float)
    for i in range(1, len(consec)):
        if neg.iloc[i] == 1:
            consec.iloc[i] = consec.iloc[i-1] + 1
        else:
            consec.iloc[i] = 0
    d["CONSEC_NEG"] = consec
    return d

print("📥 VIX 다운로드...")
vix_raw = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)
_vc = vix_raw["Close"]
if isinstance(_vc, pd.DataFrame): _vc = _vc.iloc[:, 0]
vix = pd.Series(_vc.values, index=_vc.index, dtype=float).dropna()
print(f"  ✅ VIX {len(vix)}일")

print("📥 종목 데이터 배치 다운로드...")
raw_kr = dl_batch(KR_TICKERS, START, END)
raw_us = dl_batch(US_TICKERS, START, END)
stock_data = {tk: compute(d) for tk, d in {**raw_kr, **raw_us}.items()}
print(f"  ✅ 한국 {len(raw_kr)}개 + 미국 {len(raw_us)}개 = 총 {len(stock_data)}개 로드")


# ─────────────────────────────────────────
# 신호 생성
# ─────────────────────────────────────────
def _finalize(rows):
    if not rows: return []
    df_s = pd.DataFrame(rows).sort_values(["entry_day","rsi"])
    df_s = df_s.drop_duplicates(subset=["entry_day","ticker"], keep="first")
    final = []
    for _, grp in df_s.groupby("entry_day"):
        final.extend(grp.to_dict("records")[:MAX_DAILY])
    return final

def build_bb_ma(ma_col):
    """현재가 > MA_X + 저가≤BB하단"""
    rows = []
    for tk, d in stock_data.items():
        d_c = d.dropna(subset=[ma_col, "BB_lower", "RSI"])
        if len(d_c) < 2: continue
        cond = (
            (d_c["Close"] > d_c[ma_col]) &
            (d_c["Low"]   <= d_c["BB_lower"])
        ).fillna(False)
        for i in np.where(cond.values)[0]:
            if i + 1 >= len(d_c): continue
            ed = d_c.index[i+1]; eo = float(d_c["Open"].iloc[i+1])
            if pd.isna(eo): continue
            rows.append({"ticker": tk, "entry_day": ed, "entry": eo,
                         "rsi": float(d_c["RSI"].iloc[i]) if not pd.isna(d_c["RSI"].iloc[i]) else 50})
    return _finalize(rows)

def build_bb_with_extra(extra_cond_fn):
    """MA200 위 + BB하단 터치 + 추가조건"""
    rows = []
    for tk, d in stock_data.items():
        need = ["MA200","BB_lower","RSI","HIGH20","VOL20","CONSEC_NEG","RET1","RET3","MA50","MA20"]
        d_c = d.dropna(subset=[c for c in need if c in d.columns])
        if len(d_c) < 2: continue
        base = (
            (d_c["Close"] > d_c["MA200"]) &
            (d_c["Low"]   <= d_c["BB_lower"])
        ).fillna(False)
        extra = extra_cond_fn(d_c).fillna(False)
        cond  = base & extra
        for i in np.where(cond.values)[0]:
            if i + 1 >= len(d_c): continue
            ed = d_c.index[i+1]; eo = float(d_c["Open"].iloc[i+1])
            if pd.isna(eo): continue
            rows.append({"ticker": tk, "entry_day": ed, "entry": eo,
                         "rsi": float(d_c["RSI"].iloc[i]) if not pd.isna(d_c["RSI"].iloc[i]) else 50})
    return _finalize(rows)

def build_v10():
    rows = []
    for tk, d in stock_data.items():
        d_c = d.dropna(subset=["MA200","RSI","CCI"])
        if len(d_c) < 2: continue
        vx = vix.reindex(d_c.index)
        cond = (
            (d_c["Close"] < d_c["MA200"]) &
            (vx >= 25) &
            ((d_c["RSI"] < 40) | (d_c["CCI"] < -100))
        ).fillna(False)
        for i in np.where(cond.values)[0]:
            if i + 1 >= len(d_c): continue
            ed = d_c.index[i+1]; eo = float(d_c["Open"].iloc[i+1])
            if pd.isna(eo): continue
            rows.append({"ticker": tk, "entry_day": ed, "entry": eo,
                         "rsi": float(d_c["RSI"].iloc[i])})
    return _finalize(rows)


# ─────────────────────────────────────────
# 시뮬레이션 & 통계
# ─────────────────────────────────────────
def run_sim(signals, target_pct=TARGET_PCT):
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
                "avg_win": 0, "avg_loss": 0, "pf": float("nan"), "ev": 0,
                "cagr": 0, "max_consec_loss": 0, "avg_hold": 0,
                "exit_dist": {}, "df": df}
    n = len(df)
    wins, losses = df[df["win"]], df[~df["win"]]
    wr   = len(wins)/n*100
    avg_w= wins["return_pct"].mean()  if len(wins)  else 0
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
    return {
        "label": label, "n": n, "win_rate": wr,
        "avg_ret": df["return_pct"].mean(), "avg_win": avg_w, "avg_loss": avg_l,
        "pf": pf, "ev": ev, "cagr": cagr*100, "max_consec_loss": max_cl,
        "avg_hold": df["hold_days"].mean(),
        "exit_dist": dict(df["exit_reason"].value_counts()), "df": df,
    }

def score(r):
    pf_val = r["pf"] if not np.isnan(r["pf"]) else 0
    return r["win_rate"]*0.30 + r["cagr"]*0.30 + r["ev"]*0.25 + min(pf_val,5)*0.15*10

def pf_s(r):
    return f"{r['pf']:.2f}" if not np.isnan(r['pf']) else " N/A"

def print_row(r, mark=""):
    ec = r["exit_dist"]; tt = sum(ec.values()) or 1
    tgt_pct = (ec.get("target",0)+ec.get("half+target",0))/tt*100
    print(f"  {r['label']:<32} {r['n']:>5} {r['win_rate']:>6.1f}% {r['avg_ret']:>+8.2f}% "
          f"{r['ev']:>+7.2f}% {pf_s(r):>6} {r['cagr']:>+7.1f}% "
          f"{r['max_consec_loss']:>3}건 {r['avg_hold']:>5.0f}일 {tgt_pct:>6.1f}%{mark}")


# ─────────────────────────────────────────
# 실행 1: 이평선 조건 비교
# ─────────────────────────────────────────
print("\n⚙️  [1] 이평선 조건별 비교...")

ma_results = {}
for ma_p in [5, 20, 50, 60, 144, 200]:
    col = f"MA{ma_p}"
    lbl = f"BB_MA{ma_p}위"
    sigs = build_bb_ma(col)
    r    = calc_stats(run_sim(sigs), lbl)
    ma_results[lbl] = r
    print(f"  [{lbl}] 신호 {len(sigs):,}건 → 거래 {r['n']}건 | 승률 {r['win_rate']:.1f}% | CAGR {r['cagr']:+.1f}% | PF {pf_s(r)}")

r_v10 = calc_stats(run_sim(build_v10()), "A: v10원본")
print(f"  [A: v10원본]  거래 {r_v10['n']}건 | 승률 {r_v10['win_rate']:.1f}% | CAGR {r_v10['cagr']:+.1f}% | PF {pf_s(r_v10)}")


# ─────────────────────────────────────────
# 실행 2: 추가 조건별 기여도 분석 (MA200 위 + BB하단 기준)
# ─────────────────────────────────────────
print("\n⚙️  [2] 추가 조건별 기여도 분석...")

extra_conditions = {
    "BB기본(MA200+하단)":    lambda d: pd.Series(True, index=d.index),             # 기준
    "+핀바(종가>BB하단)":    lambda d: d["Close"] > d["BB_lower"],
    "+MA50위":               lambda d: d["Close"] > d["MA50"],
    "+RSI30~55":             lambda d: (d["RSI"]>30) & (d["RSI"]<55),
    "+거래량<평균1.5배":     lambda d: d["Volume"] < d["VOL20"] * 1.5,
    "+낙폭<15%":             lambda d: d["Close"] > d["HIGH20"] * 0.85,
    "+연속음봉≥2일":         lambda d: d["CONSEC_NEG"] >= 2,                        # 새 조건
    "+전일-3%이상하락":      lambda d: d["RET1"] <= -0.03,                          # 새 조건
    "+3일-5%이상하락":       lambda d: d["RET3"] <= -0.05,                          # 새 조건
    "+RSI40~60":             lambda d: (d["RSI"]>40) & (d["RSI"]<60),              # 새 조건
    "+BB하단~중간(Close)":   lambda d: (d["Close"] >= d["BB_lower"]) & (d["Close"] <= d["BB_lower"] + (d["BB_upper"]-d["BB_lower"])*0.3),  # 새
    "+MA20위":               lambda d: d["Close"] > d["MA20"],
}

extra_results = {}
for lbl, fn in extra_conditions.items():
    sigs = build_bb_with_extra(fn)
    r    = calc_stats(run_sim(sigs), lbl)
    extra_results[lbl] = r
    print(f"  [{lbl}] 신호 {len(sigs):,}건 → 거래 {r['n']}건 | 승률 {r['win_rate']:.1f}% | CAGR {r['cagr']:+.1f}% | PF {pf_s(r)}")


# ─────────────────────────────────────────
# 결과 출력
# ─────────────────────────────────────────
HDR = f"\n  {'그룹':<32} {'건수':>5} {'승률':>7} {'평균수익':>9} {'기대값':>8} {'PF':>6} {'CAGR':>8} {'연손':>4} {'보유':>6} {'목표달성':>7}"
SEP = "  " + "-"*100

print("\n" + "="*102)
print("  [파트1] 이평선 조건별 비교 (BB 하단 터치 + 현재가 > MA_X)")
print("="*102)
print(HDR); print(SEP)
print_row(r_v10, " ◀v10기준")
print("  ···")
ma_sorted = sorted(ma_results.items(), key=lambda x: -score(x[1]))
for lbl, r in ma_sorted:
    best_mark = " ★" if lbl == ma_sorted[0][0] else ""
    print_row(r, best_mark)
print("="*102)

print("\n" + "="*102)
print("  [파트2] 추가 조건별 기여도 — 기준: BB기본(MA200위+BB하단) + 조건 하나씩 추가")
print("="*102)
print(HDR); print(SEP)
base_r = extra_results["BB기본(MA200+하단)"]
print_row(base_r, " ◀기준")
print("  ···")
extra_sorted = sorted(
    [(k,v) for k,v in extra_results.items() if k != "BB기본(MA200+하단)"],
    key=lambda x: -score(x[1])
)
for lbl, r in extra_sorted:
    d_cagr = r["cagr"] - base_r["cagr"]
    d_wr   = r["win_rate"] - base_r["win_rate"]
    impact = f" (CAGR {d_cagr:+.1f}%p, 승률 {d_wr:+.1f}%p)"
    print_row(r, impact)
print("="*102)

# 복합 점수 종합 순위
print("\n[종합 복합 점수 순위 — 전체]")
all_r = {"A: v10원본": r_v10, **ma_results, **extra_results}
sd = sorted(all_r.items(), key=lambda x: -score(x[1]))
for rank, (lbl, r) in enumerate(sd[:12], 1):
    tag = " ◀v10" if "v10" in lbl else (" ★" if rank==1 else "")
    print(f"  {rank:>2}위 {lbl:<34}: 점수 {score(r):.2f} | 승률 {r['win_rate']:.1f}% | CAGR {r['cagr']:+.1f}% | PF {pf_s(r)}{tag}")


# ─────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────
# 1) 이평선별 CAGR/승률/PF
ma_labels = [f"MA{p}위" for p in [5,20,50,60,144,200]]
ma_rs     = [ma_results[f"BB_MA{p}위"] for p in [5,20,50,60,144,200]]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("볼린저밴드 — 이평선 조건별 비교 (BB하단터치 + 현재가>MA_X, 목표+20%)", fontweight="bold")
colors_ma = plt.cm.Blues(np.linspace(0.4, 0.9, 6))
for ax, (metric, title) in zip(axes, [("cagr","CAGR(%)"),("win_rate","승률(%)"),("pf","PF")]):
    vals = [r[metric] if not np.isnan(r[metric]) else 0 for r in ma_rs]
    bars = ax.bar(ma_labels, vals, color=colors_ma, edgecolor="white")
    # v10 기준선
    ref = r_v10[metric] if not np.isnan(r_v10[metric]) else 0
    ax.axhline(ref, color="#e74c3c", lw=1.5, linestyle="--", label=f"v10원본({ref:.1f})")
    ax.set_title(title, fontweight="bold"); ax.legend(fontsize=8)
    ylim = ax.get_ylim(); rng = ylim[1]-ylim[0]
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+rng*0.02,
                f"{v:.1f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("backtest_bb_ma_compare.png", dpi=150, bbox_inches="tight")
print("\n📊 이평선 비교 차트 저장 완료")

# 2) 추가 조건 기여도 — CAGR 기준 막대
extra_lbls  = [k for k in extra_results]
extra_cagrs = [extra_results[k]["cagr"] for k in extra_lbls]
base_cagr   = base_r["cagr"]
colors_ex   = ["#95a5a6" if k=="BB기본(MA200+하단)" else
               ("#2ecc71" if extra_results[k]["cagr"] >= base_cagr else "#e74c3c")
               for k in extra_lbls]

fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
fig2.suptitle("추가 조건별 기여도 분석 (기준: BB기본 MA200위+BB하단)", fontweight="bold")
for ax, (metric, title) in zip(axes2, [("cagr","CAGR(%)"),("win_rate","승률(%)"),("pf","PF")]):
    vals = [extra_results[k][metric] if not np.isnan(extra_results[k][metric]) else 0 for k in extra_lbls]
    ref  = base_r[metric] if not np.isnan(base_r[metric]) else 0
    bars = ax.bar(range(len(extra_lbls)), vals, color=colors_ex, edgecolor="white")
    ax.axhline(ref, color="#3498db", lw=1.5, linestyle="--", label=f"기준({ref:.1f})")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(len(extra_lbls)))
    ax.set_xticklabels([k.replace("BB기본(MA200+하단)","기준") for k in extra_lbls],
                       rotation=35, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ylim = ax.get_ylim(); rng = ylim[1]-ylim[0]
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+rng*0.02,
                f"{v:.1f}", ha="center", fontsize=7.5)
plt.tight_layout()
plt.savefig("backtest_bb_extra_conditions.png", dpi=150, bbox_inches="tight")
print("📊 추가 조건 기여도 차트 저장 완료")

print("📄 분석 완료\n✅ 완료")
