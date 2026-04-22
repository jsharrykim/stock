"""
backtest_bb_enhanced.py
========================
볼린저밴드 기본 vs 강화 조건 비교 + v10 원본 기준

[A] v10 원본
  현재가<MA200 + VIX≥25 + (RSI<40 OR CCI<-100)
  목표+20% / CB-25% / 60일절반 / 120일

[B] BB 기본 (이전 최고 조건)
  현재가>MA200 + 저가≤BB하단
  VIX없음 / 목표+20% / CB-25% / 60일절반 / 120일

[C] BB 강화 (7개 조건)
  ① 현재가 > MA200
  ② 현재가 > MA50
  ③ 저가 ≤ BB하단
  ④ 종가 > BB하단  (당일 핀바 회복)
  ⑤ 30 < RSI < 55
  ⑥ 거래량 < 20일 평균 × 1.5
  ⑦ 현재가 > 20일 최고가 × 0.85
  목표+20% / CB-25% / 60일절반 / 120일

[D] BB 강화 파라미터 조합
  조건 C + 목표 +10%/+15%/+20%/+25%
  + VIX없음/≥15 조합

종목: 한국 28개 + 미국 43개 = 71개
기간: 2015-2026
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

# D그룹 파라미터 조합
D_CONFIGS = [
    {"label": "BB강화_VIX없음_+10%", "vix_min": 0,  "target": 0.10},
    {"label": "BB강화_VIX없음_+15%", "vix_min": 0,  "target": 0.15},
    {"label": "BB강화_VIX없음_+20%", "vix_min": 0,  "target": 0.20},
    {"label": "BB강화_VIX없음_+25%", "vix_min": 0,  "target": 0.25},
    {"label": "BB강화_VIX≥15_+10%",  "vix_min": 15, "target": 0.10},
    {"label": "BB강화_VIX≥15_+15%",  "vix_min": 15, "target": 0.15},
    {"label": "BB강화_VIX≥15_+20%",  "vix_min": 15, "target": 0.20},
    {"label": "BB강화_VIX≥15_+25%",  "vix_min": 15, "target": 0.25},
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
    d["MA50"]  = c.rolling(50).mean()
    d["MA200"] = c.rolling(200).mean()
    # 볼린저밴드
    bb_ma      = c.rolling(BB_PERIOD).mean()
    bb_std_s   = c.rolling(BB_PERIOD).std()
    d["BB_lower"] = bb_ma - BB_STD * bb_std_s
    d["BB_upper"] = bb_ma + BB_STD * bb_std_s
    # RSI
    delta = c.diff()
    g  = delta.clip(lower=0).rolling(14).mean()
    lo = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100 / (1 + g / lo.replace(0, np.nan))
    # CCI
    tp     = (d["High"] + d["Low"] + c) / 3
    tp_ma  = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    d["CCI"] = (tp - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))
    # 20일 최고가
    d["HIGH20"] = d["High"].rolling(20).max()
    # 20일 평균 거래량
    d["VOL20"] = d["Volume"].rolling(20).mean()
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
raw_all = {**raw_kr, **raw_us}
stock_data = {}
for tk, d in raw_all.items():
    stock_data[tk] = compute(d)
print(f"  ✅ 한국 {len(raw_kr)}개 + 미국 {len(raw_us)}개 = 총 {len(stock_data)}개 로드")


# ─────────────────────────────────────────
# 신호 생성
# ─────────────────────────────────────────
def build_signals_v10():
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

def build_signals_bb_basic():
    """기본: MA200 위 + 저가≤BB하단"""
    rows = []
    for tk, d in stock_data.items():
        d_c = d.dropna(subset=["MA200","BB_lower"])
        if len(d_c) < 2: continue
        cond = (
            (d_c["Close"] > d_c["MA200"]) &
            (d_c["Low"]   <= d_c["BB_lower"])
        ).fillna(False)
        for i in np.where(cond.values)[0]:
            if i + 1 >= len(d_c): continue
            ed = d_c.index[i+1]; eo = float(d_c["Open"].iloc[i+1])
            if pd.isna(eo): continue
            rows.append({"ticker": tk, "entry_day": ed, "entry": eo,
                         "rsi": float(d_c["RSI"].iloc[i]) if not pd.isna(d_c["RSI"].iloc[i]) else 50})
    return _finalize(rows)

def build_signals_bb_enhanced(vix_min=0):
    """강화: 7개 조건 모두"""
    rows = []
    for tk, d in stock_data.items():
        need = ["MA50","MA200","BB_lower","RSI","HIGH20","VOL20"]
        d_c = d.dropna(subset=need)
        if len(d_c) < 2: continue
        vx = vix.reindex(d_c.index)

        cond = (
            (d_c["Close"] > d_c["MA200"]) &                      # ① MA200 위
            (d_c["Close"] > d_c["MA50"]) &                       # ② MA50 위
            (d_c["Low"]   <= d_c["BB_lower"]) &                   # ③ 저가 BB하단 터치
            (d_c["Close"] > d_c["BB_lower"]) &                    # ④ 종가 BB하단 위 (핀바)
            (d_c["RSI"]   > 30) & (d_c["RSI"] < 55) &            # ⑤ RSI 30~55
            (d_c["Volume"] < d_c["VOL20"] * 1.5) &               # ⑥ 거래량 평균 150% 미만
            (d_c["Close"] > d_c["HIGH20"] * 0.85)                 # ⑦ 20일 고점 대비 -15% 이내
        ).fillna(False)

        if vix_min > 0:
            cond = cond & (vx >= vix_min).fillna(False)

        for i in np.where(cond.values)[0]:
            if i + 1 >= len(d_c): continue
            ed = d_c.index[i+1]; eo = float(d_c["Open"].iloc[i+1])
            if pd.isna(eo): continue
            rows.append({"ticker": tk, "entry_day": ed, "entry": eo,
                         "rsi": float(d_c["RSI"].iloc[i])})
    return _finalize(rows)

def _finalize(rows):
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
            if i + 1 == HALF_DAYS and not half_exited and (cl - entry) / entry > 0:
                exit_records.append((fdt, cl, 0.5, "half")); half_exited = True; continue
            if i + 1 >= MAX_HOLD:
                exit_records.append((fdt, cl, 0.5 if half_exited else 1.0, "time")); break
        if not exit_records: continue
        total_w  = sum(r[2] for r in exit_records)
        blended  = sum((r[1]-entry)/entry * r[2] for r in exit_records) / total_w
        last_exit= exit_records[-1]
        reason   = "+".join(r[3] for r in exit_records) if len(exit_records)>1 else exit_records[0][3]
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
    wr      = len(wins)/n*100
    avg_ret = df["return_pct"].mean()
    avg_w   = wins["return_pct"].mean()   if len(wins)   else 0
    avg_l   = losses["return_pct"].mean() if len(losses) else 0
    pf      = (wins["return_pct"].sum() / -losses["return_pct"].sum()
               if len(losses) and losses["return_pct"].sum()<0 else float("nan"))
    ev      = wr/100*avg_w + (1-wr/100)*avg_l
    max_cl = cur = 0
    for r in df["return_pct"]:
        cur = cur+1 if r<0 else 0; max_cl = max(max_cl, cur)
    cap = 1.0
    for _, grp in df.groupby("entry_date"):
        cap *= (1 + (grp["return_pct"]/100 * (1.0/max(len(grp), MAX_POSITIONS))).sum())
    yrs  = (df["exit_date"].max()-df["entry_date"].min()).days/365.25
    cagr = cap**(1/yrs)-1 if yrs>0 else 0
    return {
        "label": label, "n": n, "win_rate": wr, "avg_ret": avg_ret,
        "avg_win": avg_w, "avg_loss": avg_l, "pf": pf, "ev": ev,
        "cagr": cagr*100, "max_consec_loss": max_cl,
        "avg_hold": df["hold_days"].mean(),
        "exit_dist": dict(df["exit_reason"].value_counts()), "df": df,
    }

def pf_s(r):
    return f"{r['pf']:.2f}" if not np.isnan(r['pf']) else " N/A"


# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────
print("\n⚙️  신호 생성 & 시뮬레이션...")

sig_v10   = build_signals_v10()
sig_bb    = build_signals_bb_basic()

# 강화 신호 캐시
sig_enh_cache = {}
for cfg in D_CONFIGS:
    vm = cfg["vix_min"]
    if vm not in sig_enh_cache:
        sig_enh_cache[vm] = build_signals_bb_enhanced(vm)

r_v10   = calc_stats(run_sim(sig_v10, 0.20),  "A: v10원본")
r_bb    = calc_stats(run_sim(sig_bb,  0.20),  "B: BB기본+20%")
print(f"  [A: v10원본]    신호 {len(sig_v10):,}건 → 거래 {r_v10['n']}건 | 승률 {r_v10['win_rate']:.1f}% | CAGR {r_v10['cagr']:+.1f}% | PF {pf_s(r_v10)}")
print(f"  [B: BB기본+20%] 신호 {len(sig_bb):,}건 → 거래 {r_bb['n']}건 | 승률 {r_bb['win_rate']:.1f}% | CAGR {r_bb['cagr']:+.1f}% | PF {pf_s(r_bb)}")

enh_results = {}
for cfg in D_CONFIGS:
    lbl    = cfg["label"]
    vm     = cfg["vix_min"]
    tgt    = cfg["target"]
    sigs   = sig_enh_cache[vm]
    r      = calc_stats(run_sim(sigs, tgt), lbl)
    enh_results[lbl] = r
    print(f"  [{lbl}] 신호 {len(sigs):,}건 → 거래 {r['n']}건 | 승률 {r['win_rate']:.1f}% | CAGR {r['cagr']:+.1f}% | PF {pf_s(r)}")


# ─────────────────────────────────────────
# 결과 출력
# ─────────────────────────────────────────
all_results = {"A: v10원본": r_v10, "B: BB기본+20%": r_bb, **enh_results}

print("\n" + "="*120)
print("  BB 기본 vs BB 강화(7조건) vs v10 원본 비교 (2015-2026)")
print("  강화조건: MA200↑ + MA50↑ + 저가≤BB하단 + 종가>BB하단(핀바) + 30<RSI<55 + 거래량<평균×1.5 + 고점대비-15%이내")
print("="*120)
print(f"\n  {'그룹':<26} {'건수':>5} {'승률':>7} {'평균수익':>9} {'기대값':>8} {'승평균':>8} {'패평균':>8} {'PF':>6} {'CAGR':>8} {'연손':>4} {'보유':>6} {'목표도달':>7}")
print("  " + "-"*115)

for key, r in all_results.items():
    ec  = r["exit_dist"]; tt = sum(ec.values()) or 1
    tgt_share = (ec.get("target",0) + ec.get("half+target",0)) / tt * 100
    sep = "  ···\n" if key == "B: BB기본+20%" else ""
    mark = " ◀기준" if "v10" in key else ""
    print(sep + f"  {key:<26} {r['n']:>5} {r['win_rate']:>6.1f}% {r['avg_ret']:>+8.2f}% "
          f"{r['ev']:>+7.2f}% {r['avg_win']:>+7.2f}% {r['avg_loss']:>+7.2f}% "
          f"{pf_s(r):>6} {r['cagr']:>+7.1f}% {r['max_consec_loss']:>3}건 "
          f"{r['avg_hold']:>5.0f}일 {tgt_share:>6.1f}%{mark}")
print("="*120)

# 복합 점수
print("\n[복합 점수 순위]")
score_data = []
for lbl, r in all_results.items():
    pf_val = r["pf"] if not np.isnan(r["pf"]) else 0
    score  = r["win_rate"]*0.30 + r["cagr"]*0.30 + r["ev"]*0.25 + min(pf_val,5)*0.15*10
    score_data.append((lbl, score, r))
score_data.sort(key=lambda x: -x[1])
for rank, (lbl, score, r) in enumerate(score_data, 1):
    mark = " ◀v10기준" if "v10" in lbl else (" ★최고" if rank==1 and "v10" not in lbl else "")
    print(f"  {rank:>2}위 {lbl:<26}: 점수 {score:.2f} | 승률 {r['win_rate']:.1f}% | CAGR {r['cagr']:+.1f}% | EV {r['ev']:+.2f}% | PF {pf_s(r)} | 보유 {r['avg_hold']:.0f}일{mark}")

# 기본 vs 강화 직접 비교
print("\n[BB 기본 vs 강화 — +20% 기준 직접 비교]")
r_enh20 = enh_results.get("BB강화_VIX없음_+20%", {})
if r_enh20:
    for metric, name in [("n","거래건수"), ("win_rate","승률"), ("avg_ret","평균수익"),
                          ("ev","기대값"), ("cagr","CAGR"), ("max_consec_loss","연속손실"), ("avg_hold","평균보유")]:
        bv = r_bb[metric]; ev2 = r_enh20[metric]
        diff = ev2 - bv
        unit = "%" if metric in ("win_rate","avg_ret","ev","cagr") else ("건" if metric in ("n","max_consec_loss") else "일")
        sign = "+" if diff > 0 else ""
        print(f"  {name:<10}: BB기본 {bv:>7.1f}{unit}  →  BB강화 {ev2:>7.1f}{unit}  ({sign}{diff:.1f}{unit})")

# 연도별
print("\n[연도별 비교]")
keys_show = ["A: v10원본", "B: BB기본+20%", "BB강화_VIX없음_+20%", "BB강화_VIX≥15_+20%"]
show_rs   = [(k, all_results[k]) for k in keys_show if k in all_results]
all_years = sorted(set().union(*[set(r["df"]["entry_date"].dt.year) for _, r in show_rs if not r["df"].empty]))
hdr = f"  {'연도':<6}" + "".join(f" {k.replace('BB강화_','').replace('BB기본','BB기본'):>22}" for k, _ in show_rs)
print(hdr); print("  " + "-"*98)
for yr in all_years:
    row_str = f"  {yr:<6}"
    for _, r in show_rs:
        if r["df"].empty: row_str += f" {'  -':>22}"; continue
        sub = r["df"][r["df"]["entry_date"].dt.year==yr]
        if not len(sub): row_str += f" {'  -':>22}"
        else: row_str += f" {sub['win'].mean()*100:.0f}%/{sub['return_pct'].mean():+.1f}%({len(sub)}건)"
    print(row_str)

# 청산 분포
print("\n[청산 유형 분포]")
for k in keys_show:
    if k not in all_results: continue
    r = all_results[k]; ec = r["exit_dist"]
    if not ec: print(f"  {k:<28}: 거래 없음"); continue
    tt = sum(ec.values())
    tops = sorted(ec.items(), key=lambda x: -x[1])[:4]
    parts = [f"{kk}:{v}({v/tt*100:.0f}%)" for kk, v in tops]
    print(f"  {k:<28}: {' | '.join(parts)}")


# ─────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────
# 1) 핵심 4개 항목 막대
keys_viz  = ["A: v10원본", "B: BB기본+20%", "BB강화_VIX없음_+20%", "BB강화_VIX≥15_+20%",
             "BB강화_VIX없음_+15%", "BB강화_VIX없음_+25%"]
keys_viz  = [k for k in keys_viz if k in all_results]
rviz      = [all_results[k] for k in keys_viz]
clabels   = [k.replace("BB강화_","강화_").replace("BB기본","BB기본") for k in keys_viz]
colors_v  = ["#e74c3c","#95a5a6"] + ["#3498db"]*4

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("BB 기본 vs BB 강화(7조건) vs v10 원본 비교 (2015-2026)\n"
             "강화: MA200↑+MA50↑+BB하단터치+핀바+RSI30~55+거래량필터+낙폭제한",
             fontweight="bold", fontsize=11)
metrics = [
    ("승률 (%)",          [r["win_rate"] for r in rviz]),
    ("평균 수익률 (%)",    [r["avg_ret"]  for r in rviz]),
    ("CAGR (%)",          [r["cagr"]     for r in rviz]),
    ("기대값 EV (%)",     [r["ev"]       for r in rviz]),
    ("Profit Factor",     [r["pf"] if not np.isnan(r["pf"]) else 0 for r in rviz]),
    ("평균 보유 기간 (일)",[r["avg_hold"] for r in rviz]),
]
for ax, (title, vals) in zip(axes.flatten(), metrics):
    bars = ax.bar(range(len(clabels)), vals, color=colors_v, edgecolor="white", linewidth=1.2)
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(len(clabels)))
    ax.set_xticklabels(clabels, rotation=30, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.5)
    ylim = ax.get_ylim(); rng = ylim[1]-ylim[0]
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+rng*0.02,
                f"{v:.1f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig("backtest_bb_enhanced_compare.png", dpi=150, bbox_inches="tight")
print("\n📊 비교 차트 저장 완료")

# 2) 강화 조건 파라미터 히트맵
vix_lbls = ["VIX없음", "VIX≥15"]
tgt_lbls = ["+10%", "+15%", "+20%", "+25%"]
vix_mins = [0, 15]
targets  = [0.10, 0.15, 0.20, 0.25]

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle("BB 강화 조건 파라미터 히트맵", fontweight="bold")
for ax, (metric, mtitle) in zip(axes2, [("win_rate","승률(%)"),("cagr","CAGR(%)"),("ev","기대값EV(%)")]):
    grid = np.zeros((len(vix_mins), len(targets)))
    for i, vm in enumerate(vix_mins):
        for j, tg in enumerate(targets):
            vix_tag = "VIX없음" if vm==0 else f"VIX≥{vm}"
            lbl = f"BB강화_{vix_tag}_{'+' if tg>=0 else ''}{int(tg*100)}%"
            if lbl in enh_results:
                grid[i][j] = enh_results[lbl][metric]
    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(tgt_lbls))); ax.set_xticklabels(tgt_lbls)
    ax.set_yticks(range(len(vix_lbls))); ax.set_yticklabels(vix_lbls)
    ax.set_title(f"BB강화 — {mtitle}", fontweight="bold")
    plt.colorbar(im, ax=ax)
    for i in range(len(vix_mins)):
        for j in range(len(targets)):
            ax.text(j, i, f"{grid[i][j]:.1f}", ha="center", va="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("backtest_bb_enhanced_heatmap.png", dpi=150, bbox_inches="tight")
print("📊 히트맵 저장 완료")

# CSV
best_enh_lbl = score_data[0][0] if "v10" not in score_data[0][0] else score_data[1][0]
all_results[best_enh_lbl]["df"].to_csv("backtest_bb_enhanced_best_trades.csv", index=False)
r_v10["df"].to_csv("backtest_v10_trades.csv", index=False)
print("📄 CSV 저장 완료\n✅ 완료")
