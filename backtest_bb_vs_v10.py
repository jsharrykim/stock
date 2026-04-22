"""
backtest_bb_vs_v10.py
======================
[그룹 A] v10 원본
  진입: 현재가 < MA200 + VIX≥25 + (RSI<40 OR CCI<-100)
  청산: 목표+20% / CB-25% / 60일 절반+수익 / 120일

[그룹 B] 볼린저밴드 하단 터치 (MA200 위)
  진입: 현재가 > MA200 + 당일 저가가 볼린저밴드 하단 이하 터치
  VIX 조건: 없음 / ≥15 / ≥20
  청산 목표: +10% / +15% / +20% / +25%
  CB: -25% 고정 / 60일 절반+수익 / 120일

종목: 한국 28개 + 미국 42개 = 70개 혼합
기간: 2015-2026 (한국 종목 데이터 고려)
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
ALL_TICKERS = KR_TICKERS + US_TICKERS

# 볼린저밴드 그룹 파라미터 조합
BB_CONFIGS = [
    {"label": "BB_VIX없음_+10%",  "vix_min": 0,  "target": 0.10},
    {"label": "BB_VIX없음_+15%",  "vix_min": 0,  "target": 0.15},
    {"label": "BB_VIX없음_+20%",  "vix_min": 0,  "target": 0.20},
    {"label": "BB_VIX없음_+25%",  "vix_min": 0,  "target": 0.25},
    {"label": "BB_VIX≥15_+10%",   "vix_min": 15, "target": 0.10},
    {"label": "BB_VIX≥15_+15%",   "vix_min": 15, "target": 0.15},
    {"label": "BB_VIX≥15_+20%",   "vix_min": 15, "target": 0.20},
    {"label": "BB_VIX≥15_+25%",   "vix_min": 15, "target": 0.25},
    {"label": "BB_VIX≥20_+10%",   "vix_min": 20, "target": 0.10},
    {"label": "BB_VIX≥20_+15%",   "vix_min": 20, "target": 0.15},
    {"label": "BB_VIX≥20_+20%",   "vix_min": 20, "target": 0.20},
    {"label": "BB_VIX≥20_+25%",   "vix_min": 20, "target": 0.25},
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
    d["MA200"] = c.rolling(200).mean()
    # 볼린저밴드
    bb_ma  = c.rolling(BB_PERIOD).mean()
    bb_std = c.rolling(BB_PERIOD).std()
    d["BB_upper"] = bb_ma + BB_STD * bb_std
    d["BB_lower"] = bb_ma - BB_STD * bb_std
    d["BB_mid"]   = bb_ma
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
    return d

print("📥 VIX 다운로드...")
vix_raw = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)
_vc = vix_raw["Close"]
if isinstance(_vc, pd.DataFrame): _vc = _vc.iloc[:, 0]
vix = pd.Series(_vc.values, index=_vc.index, dtype=float).dropna()
print(f"  ✅ VIX {len(vix)}일")

print("📥 종목 데이터 배치 다운로드...")
# 한국/미국 분리 다운로드
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
def build_v10_signals():
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
            ed = d_c.index[i+1]
            eo = float(d_c["Open"].iloc[i+1])
            if pd.isna(eo): continue
            rows.append({"ticker": tk, "entry_day": ed, "entry": eo,
                         "rsi": float(d_c["RSI"].iloc[i])})
    if not rows: return []
    df_s = pd.DataFrame(rows).sort_values(["entry_day","rsi"])
    df_s = df_s.drop_duplicates(subset=["entry_day","ticker"], keep="first")
    final = []
    for _, grp in df_s.groupby("entry_day"):
        final.extend(grp.to_dict("records")[:MAX_DAILY])
    return final

def build_bb_signals(vix_min=0):
    """MA200 위 + 볼린저밴드 하단 터치 (저가 <= BB_lower)"""
    rows = []
    for tk, d in stock_data.items():
        d_c = d.dropna(subset=["MA200","BB_lower","BB_upper"])
        if len(d_c) < 2: continue
        vx = vix.reindex(d_c.index)
        cond = (
            (d_c["Close"] > d_c["MA200"]) &          # MA200 위
            (d_c["Low"]   <= d_c["BB_lower"])         # 저가가 BB 하단 터치
        ).fillna(False)
        if vix_min > 0:
            cond = cond & (vx >= vix_min).fillna(False)
        for i in np.where(cond.values)[0]:
            if i + 1 >= len(d_c): continue
            ed = d_c.index[i+1]
            eo = float(d_c["Open"].iloc[i+1])
            if pd.isna(eo): continue
            rows.append({"ticker": tk, "entry_day": ed, "entry": eo,
                         "rsi": float(d_c["RSI"].iloc[i]) if not pd.isna(d_c["RSI"].iloc[i]) else 50})
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


# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────
print("\n⚙️  신호 생성 & 시뮬레이션...")

# A그룹: v10 원본
sig_v10 = build_v10_signals()
df_v10  = run_sim(sig_v10, 0.20)
r_v10   = calc_stats(df_v10, "A: v10원본")
pf_s = f"{r_v10['pf']:.2f}" if not np.isnan(r_v10['pf']) else "N/A"
print(f"  [A: v10원본] 신호 {len(sig_v10):,}건 → 거래 {r_v10['n']}건 | 승률 {r_v10['win_rate']:.1f}% | CAGR {r_v10['cagr']:+.1f}% | PF {pf_s}")

# B그룹: 볼린저밴드 파라미터 조합
bb_results = {}
bb_signals_cache = {}
for cfg in BB_CONFIGS:
    vix_min = cfg["vix_min"]
    target  = cfg["target"]
    label   = cfg["label"]
    # VIX별 신호 캐시 (같은 VIX는 재사용)
    if vix_min not in bb_signals_cache:
        bb_signals_cache[vix_min] = build_bb_signals(vix_min)
    signals = bb_signals_cache[vix_min]
    df  = run_sim(signals, target)
    r   = calc_stats(df, label)
    bb_results[label] = r
    pf_s = f"{r['pf']:.2f}" if not np.isnan(r['pf']) else "N/A"
    print(f"  [{label}] 신호 {len(signals):,}건 → 거래 {r['n']}건 | 승률 {r['win_rate']:.1f}% | CAGR {r['cagr']:+.1f}% | PF {pf_s}")


# ─────────────────────────────────────────
# 결과 출력
# ─────────────────────────────────────────
def pf_str(r):
    return f"{r['pf']:.2f}" if not np.isnan(r['pf']) else " N/A"

print("\n" + "="*115)
print("  볼린저밴드(MA200 위) vs v10 원본 백테스트 비교 (2015-2026)")
print("  BB: 현재가>MA200 + 저가≤BB하단(20일,2σ) | A: v10원본(현재가<MA200+VIX≥25+RSI<40/CCI<-100)")
print("="*115)
print(f"\n  {'그룹':<24} {'건수':>5} {'승률':>7} {'평균수익':>9} {'기대값':>8} {'승평균':>8} {'패평균':>8} {'PF':>6} {'CAGR':>8} {'연손':>4} {'보유':>6} {'목표도달':>7}")
print("  " + "-"*113)

# A 먼저
r = r_v10
tgt_share = r['exit_dist'].get('target',0) + r['exit_dist'].get('half+target',0)
tt = sum(r['exit_dist'].values()) or 1
print(f"  {'A: v10원본':<24} {r['n']:>5} {r['win_rate']:>6.1f}% {r['avg_ret']:>+8.2f}% "
      f"{r['ev']:>+7.2f}% {r['avg_win']:>+7.2f}% {r['avg_loss']:>+7.2f}% "
      f"{pf_str(r):>6} {r['cagr']:>+7.1f}% {r['max_consec_loss']:>3}건 {r['avg_hold']:>5.0f}일 "
      f"{tgt_share/tt*100:>6.1f}%")
print("  " + "·"*113)

# B 그룹 — VIX별 구분선
prev_vix = None
for label, r in bb_results.items():
    vix_tag = label.split("_")[1]
    if vix_tag != prev_vix:
        print(f"  --- {vix_tag} ---")
        prev_vix = vix_tag
    tgt_share = r['exit_dist'].get('target',0) + r['exit_dist'].get('half+target',0)
    tt = sum(r['exit_dist'].values()) or 1
    mark = " ★" if r['cagr'] > r_v10['cagr'] else ""
    print(f"  {label:<24} {r['n']:>5} {r['win_rate']:>6.1f}% {r['avg_ret']:>+8.2f}% "
          f"{r['ev']:>+7.2f}% {r['avg_win']:>+7.2f}% {r['avg_loss']:>+7.2f}% "
          f"{pf_str(r):>6} {r['cagr']:>+7.1f}% {r['max_consec_loss']:>3}건 {r['avg_hold']:>5.0f}일 "
          f"{tgt_share/tt*100:>6.1f}%{mark}")
print("="*115)

# 복합 점수
print("\n[복합 점수 순위 — 상위 10개]")
all_results = {"A: v10원본": r_v10, **bb_results}
score_data = []
for lbl, r in all_results.items():
    pf_val = r["pf"] if not np.isnan(r["pf"]) else 0
    score  = r["win_rate"]*0.30 + r["cagr"]*0.30 + r["ev"]*0.25 + min(pf_val,5)*0.15*10
    score_data.append((lbl, score, r))
score_data.sort(key=lambda x: -x[1])
for rank, (lbl, score, r) in enumerate(score_data[:10], 1):
    mark = " ◀v10기준" if "v10" in lbl else (" ★BB최고" if rank==1 and "v10" not in lbl else "")
    print(f"  {rank:>2}위 {lbl:<24}: 점수 {score:.2f} | 승률 {r['win_rate']:.1f}% | CAGR {r['cagr']:+.1f}% | EV {r['ev']:+.2f}% | PF {pf_str(r)} | 보유 {r['avg_hold']:.0f}일{mark}")

# 연도별 (상위 BB + v10)
print("\n[연도별 비교 — v10 원본 vs BB 최고 조건]")
top_bb_lbl = score_data[0][0] if "v10" not in score_data[0][0] else score_data[1][0]
top_bb_r   = all_results[top_bb_lbl]
compare_rs = [("A: v10원본", r_v10), (top_bb_lbl, top_bb_r)]
all_years = sorted(set().union(*[set(r["df"]["entry_date"].dt.year) for _, r in compare_rs if not r["df"].empty]))
hdr = f"  {'연도':<6}" + "".join(f" {lbl:>28}" for lbl, _ in compare_rs)
print(hdr); print("  " + "-"*70)
for yr in all_years:
    row_str = f"  {yr:<6}"
    for lbl, r in compare_rs:
        if r["df"].empty: row_str += f" {'  -':>28}"; continue
        sub = r["df"][r["df"]["entry_date"].dt.year==yr]
        if not len(sub): row_str += f" {'  -':>28}"
        else: row_str += f" {sub['win'].mean()*100:.0f}%/{sub['return_pct'].mean():+.1f}%({len(sub)}건)"
    print(row_str)

# 청산 분포
print("\n[청산 유형 분포 — v10 원본 vs BB 상위 3개]")
for lbl, r in [("A: v10원본", r_v10)] + [(l, all_results[l]) for l, _, _ in score_data[1:4] if "v10" not in l]:
    if not r["exit_dist"]: continue
    ec = r["exit_dist"]; tt = sum(ec.values())
    tops = sorted(ec.items(), key=lambda x: -x[1])[:4]
    parts = [f"{k}:{v}({v/tt*100:.0f}%)" for k, v in tops]
    print(f"  {lbl:<26}: {' | '.join(parts)}")


# ─────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────
# 1) BB 파라미터 히트맵 (VIX × 목표수익률 × 지표)
vix_labels = ["VIX없음", "VIX≥15", "VIX≥20"]
tgt_labels = ["+10%", "+15%", "+20%", "+25%"]
vix_mins   = [0, 15, 20]
targets    = [0.10, 0.15, 0.20, 0.25]

for metric, mtitle in [("win_rate","승률(%)"), ("cagr","CAGR(%)"), ("ev","기대값EV(%)")]:
    grid = np.zeros((len(vix_mins), len(targets)))
    for i, vm in enumerate(vix_mins):
        for j, tg in enumerate(targets):
            lbl = f"BB_VIX{'없음' if vm==0 else f'≥{vm}'}_{'+' if tg>=0 else ''}{int(tg*100)}%"
            # 매핑
            matched = [k for k in bb_results if f"VIX{'없음' if vm==0 else f'≥{vm}'}" in k and f"+{int(tg*100)}%" in k]
            if matched:
                grid[i][j] = bb_results[matched[0]][metric]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(tgt_labels))); ax.set_xticklabels(tgt_labels)
    ax.set_yticks(range(len(vix_labels))); ax.set_yticklabels(vix_labels)
    ax.set_title(f"볼린저밴드 전략 파라미터 히트맵 — {mtitle}", fontweight="bold")
    plt.colorbar(im, ax=ax)
    for i in range(len(vix_mins)):
        for j in range(len(targets)):
            ax.text(j, i, f"{grid[i][j]:.1f}", ha="center", va="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"backtest_bb_heatmap_{metric}.png", dpi=150, bbox_inches="tight")
print("📊 히트맵 3개 저장 완료")

# 2) 막대 비교 — v10 vs BB 상위 5개
top5_labels = ["A: v10원본"] + [l for l, _, _ in score_data[:6] if "v10" not in l][:5]
top5_results= [all_results[l] for l in top5_labels]
colors_bar  = ["#e74c3c"] + ["#3498db"]*5

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("v10 원본 vs 볼린저밴드 상위 조건 비교 (2015-2026)", fontweight="bold")
for ax, (metric, title) in zip(axes, [("win_rate","승률(%)"),("cagr","CAGR(%)"),("ev","기대값(%)"),("pf","PF")]):
    vals = [r[metric] if not (isinstance(r[metric],float) and np.isnan(r[metric])) else 0 for r in top5_results]
    bars = ax.bar(range(len(top5_labels)), vals, color=colors_bar, edgecolor="white")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(len(top5_labels)))
    ax.set_xticklabels([l.replace("BB_","").replace("_"," ") for l in top5_labels], rotation=30, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.5)
    ylim = ax.get_ylim(); rng = ylim[1]-ylim[0]
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+rng*0.02, f"{v:.1f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig("backtest_bb_vs_v10_compare.png", dpi=150, bbox_inches="tight")
print("📊 비교 차트 저장 완료")

# CSV
df_v10.to_csv("backtest_v10_trades.csv", index=False)
best_bb_df = all_results[top_bb_lbl]["df"]
if not best_bb_df.empty:
    best_bb_df.to_csv(f"backtest_bb_best_trades.csv", index=False)
print("📄 CSV 저장 완료\n✅ 완료")
