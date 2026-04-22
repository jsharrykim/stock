"""
backtest_bb_percentb.py
========================
기존 볼린저밴드 함수가 반환하는 값(%B)만으로 BB 전략 조건 구성
— 새 컬럼 추가 없이 시트 현재 값만 사용

[A] v10 원본 (기준)
  현재가<MA200 + VIX≥25 + (RSI<40 OR CCI<-100)
  목표+20% / CB-25% / 60일절반 / 120일

[B] BB 기본 (이전 최고 — 저가 기준)
  현재가>MA200 + 저가≤BB하단(Low %B≤0)
  목표+20% / CB-25% / 60일절반 / 120일

[C] %B 종가 단독 ≤ 5%
  현재가>MA200 + 종가 %B ≤ 5%
  목표+20%

[D] %B 종가 + 전일比 반등
  현재가>MA200 + %B_D ≤ 5% + %B_D > %B_D-1 (당일 반등 시작)
  목표+20%

[E] %B 종가 + 전일比 반등 + 3일-5%이상 하락
  현재가>MA200 + %B_D ≤ 5% + %B_D > %B_D-1 + 3일수익률≤-5%
  목표+20%

[F] 밴드폭 스퀴즈 탈출
  현재가>MA200 + %B_D ≤ 20% + bw_D > bw_D-1 + bw_D < avg_bw60
  목표+20%

각 조건별 목표수익률 +10%/+15%/+20%/+25% 세분화 추가

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
    d["MA200"] = c.rolling(200).mean()
    # 볼린저밴드 계산 (Apps Script 동일 로직)
    bb_ma    = c.rolling(BB_PERIOD).mean()
    bb_std_s = c.rolling(BB_PERIOD).std(ddof=0)  # 모표준편차 (Apps Script와 동일)
    bb_upper = bb_ma + BB_STD * bb_std_s
    bb_lower = bb_ma - BB_STD * bb_std_s
    bb_range = bb_upper - bb_lower

    # %B 종가 당일 (bollingerBand_D)
    d["PCT_B_D"]    = ((c - bb_lower) / bb_range.replace(0, np.nan)) * 100
    # %B 고가 당일 (bollingerBand_Peak_D) — 저가 %B 근사에도 활용
    d["PCT_B_HIGH"] = ((d["High"] - bb_lower) / bb_range.replace(0, np.nan)) * 100
    # %B 저가 당일 — 실제 저가 기반 (B 기준 전략용)
    d["PCT_B_LOW"]  = ((d["Low"]  - bb_lower) / bb_range.replace(0, np.nan)) * 100
    # %B 종가 전일 (bollingerBand_D_minus_1)
    d["PCT_B_D1"]   = d["PCT_B_D"].shift(1)
    # 밴드폭 당일/전일/60일평균
    d["BW_D"]       = (bb_range / bb_ma.replace(0, np.nan)) * 100
    d["BW_D1"]      = d["BW_D"].shift(1)
    d["BW_AVG60"]   = d["BW_D"].rolling(60).mean()
    # 3일 수익률
    d["RET3"]       = c.pct_change(3)
    # RSI, CCI
    delta = c.diff()
    g  = delta.clip(lower=0).rolling(14).mean()
    lo = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100 / (1 + g / lo.replace(0, np.nan))
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
            if i+1 >= len(d_c): continue
            ed = d_c.index[i+1]; eo = float(d_c["Open"].iloc[i+1])
            if pd.isna(eo): continue
            rows.append({"ticker": tk, "entry_day": ed, "entry": eo,
                         "rsi": float(d_c["RSI"].iloc[i])})
    return _finalize(rows)

def build_bb_low(pct_b_threshold=0):
    """B 기준: 저가 %B ≤ threshold + MA200 위"""
    rows = []
    for tk, d in stock_data.items():
        d_c = d.dropna(subset=["MA200","PCT_B_LOW","RSI"])
        if len(d_c) < 2: continue
        cond = (
            (d_c["Close"]     > d_c["MA200"]) &
            (d_c["PCT_B_LOW"] <= pct_b_threshold)
        ).fillna(False)
        for i in np.where(cond.values)[0]:
            if i+1 >= len(d_c): continue
            ed = d_c.index[i+1]; eo = float(d_c["Open"].iloc[i+1])
            if pd.isna(eo): continue
            rows.append({"ticker": tk, "entry_day": ed, "entry": eo,
                         "rsi": float(d_c["RSI"].iloc[i]) if not pd.isna(d_c["RSI"].iloc[i]) else 50})
    return _finalize(rows)

def build_pct_b_close(threshold=5):
    """C: 종가 %B ≤ threshold + MA200 위"""
    rows = []
    for tk, d in stock_data.items():
        d_c = d.dropna(subset=["MA200","PCT_B_D","RSI"])
        if len(d_c) < 2: continue
        cond = (
            (d_c["Close"]   > d_c["MA200"]) &
            (d_c["PCT_B_D"] <= threshold)
        ).fillna(False)
        for i in np.where(cond.values)[0]:
            if i+1 >= len(d_c): continue
            ed = d_c.index[i+1]; eo = float(d_c["Open"].iloc[i+1])
            if pd.isna(eo): continue
            rows.append({"ticker": tk, "entry_day": ed, "entry": eo,
                         "rsi": float(d_c["RSI"].iloc[i]) if not pd.isna(d_c["RSI"].iloc[i]) else 50})
    return _finalize(rows)

def build_pct_b_rebound(threshold=5):
    """D: 종가 %B ≤ threshold + 당일 %B > 전일 %B (반등 시작) + MA200 위"""
    rows = []
    for tk, d in stock_data.items():
        d_c = d.dropna(subset=["MA200","PCT_B_D","PCT_B_D1","RSI"])
        if len(d_c) < 2: continue
        cond = (
            (d_c["Close"]    > d_c["MA200"]) &
            (d_c["PCT_B_D"]  <= threshold) &
            (d_c["PCT_B_D"]  > d_c["PCT_B_D1"])
        ).fillna(False)
        for i in np.where(cond.values)[0]:
            if i+1 >= len(d_c): continue
            ed = d_c.index[i+1]; eo = float(d_c["Open"].iloc[i+1])
            if pd.isna(eo): continue
            rows.append({"ticker": tk, "entry_day": ed, "entry": eo,
                         "rsi": float(d_c["RSI"].iloc[i]) if not pd.isna(d_c["RSI"].iloc[i]) else 50})
    return _finalize(rows)

def build_pct_b_rebound_ret3(threshold=5):
    """E: D조건 + 3일수익률≤-5%"""
    rows = []
    for tk, d in stock_data.items():
        d_c = d.dropna(subset=["MA200","PCT_B_D","PCT_B_D1","RET3","RSI"])
        if len(d_c) < 2: continue
        cond = (
            (d_c["Close"]    > d_c["MA200"]) &
            (d_c["PCT_B_D"]  <= threshold) &
            (d_c["PCT_B_D"]  > d_c["PCT_B_D1"]) &
            (d_c["RET3"]     <= -0.05)
        ).fillna(False)
        for i in np.where(cond.values)[0]:
            if i+1 >= len(d_c): continue
            ed = d_c.index[i+1]; eo = float(d_c["Open"].iloc[i+1])
            if pd.isna(eo): continue
            rows.append({"ticker": tk, "entry_day": ed, "entry": eo,
                         "rsi": float(d_c["RSI"].iloc[i]) if not pd.isna(d_c["RSI"].iloc[i]) else 50})
    return _finalize(rows)

def build_squeeze_breakout(pct_b_max=20):
    """F: 밴드폭 스퀴즈 탈출 + 하단 위치 + MA200 위"""
    rows = []
    for tk, d in stock_data.items():
        d_c = d.dropna(subset=["MA200","PCT_B_D","BW_D","BW_D1","BW_AVG60","RSI"])
        if len(d_c) < 2: continue
        cond = (
            (d_c["Close"]    > d_c["MA200"]) &
            (d_c["PCT_B_D"]  <= pct_b_max) &
            (d_c["BW_D"]     > d_c["BW_D1"]) &       # 밴드폭 확장 시작
            (d_c["BW_D"]     < d_c["BW_AVG60"])       # 아직 평균보다 좁음
        ).fillna(False)
        for i in np.where(cond.values)[0]:
            if i+1 >= len(d_c): continue
            ed = d_c.index[i+1]; eo = float(d_c["Open"].iloc[i+1])
            if pd.isna(eo): continue
            rows.append({"ticker": tk, "entry_day": ed, "entry": eo,
                         "rsi": float(d_c["RSI"].iloc[i]) if not pd.isna(d_c["RSI"].iloc[i]) else 50})
    return _finalize(rows)


# ─────────────────────────────────────────
# 시뮬레이션 & 통계
# ─────────────────────────────────────────
def run_sim(signals, target_pct=0.20):
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
    print(f"  {r['label']:<34} {r['n']:>5} {r['win_rate']:>6.1f}% {r['avg_ret']:>+8.2f}% "
          f"{r['ev']:>+7.2f}% {pf_s(r):>6} {r['cagr']:>+7.1f}% "
          f"{r['max_consec_loss']:>3}건 {r['avg_hold']:>5.0f}일 {tgt_pct:>6.1f}%{mark}")


# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────
print("\n⚙️  신호 생성 & 시뮬레이션...")

# 기준 신호 사전 생성
sig_v10  = build_v10()
sig_bb_low = build_bb_low(0)         # 저가 %B ≤ 0 (기존 최고)
sig_c    = build_pct_b_close(5)      # 종가 %B ≤ 5
sig_d    = build_pct_b_rebound(5)    # 종가 %B ≤ 5 + 반등
sig_e    = build_pct_b_rebound_ret3(5)  # D + 3일-5%
sig_f    = build_squeeze_breakout(20)   # 스퀴즈 탈출

print(f"  신호 건수: v10={len(sig_v10)}, BB저가={len(sig_bb_low)}, C={len(sig_c)}, D={len(sig_d)}, E={len(sig_e)}, F={len(sig_f)}")

# 메인 비교 — 목표 +20% 기준
results_main = {}
for lbl, sigs in [
    ("A: v10원본",            sig_v10),
    ("B: BB저가%B≤0",         sig_bb_low),
    ("C: 종가%B≤5%",          sig_c),
    ("D: 종가%B≤5+반등",       sig_d),
    ("E: D+3일-5%이상하락",    sig_e),
    ("F: 스퀴즈탈출%B≤20",    sig_f),
]:
    r = calc_stats(run_sim(sigs, 0.20), lbl)
    results_main[lbl] = r
    print(f"  [{lbl}] 거래 {r['n']}건 | 승률 {r['win_rate']:.1f}% | CAGR {r['cagr']:+.1f}% | PF {pf_s(r)}")

# D 조건 목표수익률 세분화
print("\n  [D조건 목표수익률 세분화]")
results_d_targets = {}
for tgt in [0.10, 0.15, 0.20, 0.25]:
    lbl = f"D_+{int(tgt*100)}%"
    r   = calc_stats(run_sim(sig_d, tgt), lbl)
    results_d_targets[lbl] = r
    print(f"  [{lbl}] 거래 {r['n']}건 | 승률 {r['win_rate']:.1f}% | CAGR {r['cagr']:+.1f}% | PF {pf_s(r)}")

# E 조건 목표수익률 세분화
print("  [E조건 목표수익률 세분화]")
results_e_targets = {}
for tgt in [0.10, 0.15, 0.20, 0.25]:
    lbl = f"E_+{int(tgt*100)}%"
    r   = calc_stats(run_sim(sig_e, tgt), lbl)
    results_e_targets[lbl] = r
    print(f"  [{lbl}] 거래 {r['n']}건 | 승률 {r['win_rate']:.1f}% | CAGR {r['cagr']:+.1f}% | PF {pf_s(r)}")


# ─────────────────────────────────────────
# 결과 출력
# ─────────────────────────────────────────
HDR = f"\n  {'전략':<34} {'건수':>5} {'승률':>7} {'평균수익':>9} {'기대값':>8} {'PF':>6} {'CAGR':>8} {'연손':>4} {'보유':>6} {'목표달성':>7}"
SEP = "  " + "-"*105

print("\n" + "="*107)
print("  %B 기반 BB 전략 비교 (2015-2026) — 목표 +20% 기준")
print("  시트 기존 값만 활용: %B_D(종가당일), %B_D-1(종가전일), BW(밴드폭), BW_AVG60(60일평균밴드폭)")
print("="*107)
print(HDR); print(SEP)
for lbl, r in results_main.items():
    mark = " ◀v10기준" if "v10" in lbl else (" ★" if lbl == "B: BB저가%B≤0" else "")
    print_row(r, mark)
print("="*107)

print("\n[D조건 — 종가%B≤5+반등 — 목표수익률별]")
print(HDR); print(SEP)
for lbl, r in results_d_targets.items():
    best = " ★" if score(r) == max(score(v) for v in results_d_targets.values()) else ""
    print_row(r, best)

print("\n[E조건 — D+3일-5%이상하락 — 목표수익률별]")
print(HDR); print(SEP)
for lbl, r in results_e_targets.items():
    best = " ★" if score(r) == max(score(v) for v in results_e_targets.values()) else ""
    print_row(r, best)

# 종합 복합 점수 순위
all_r = {**results_main, **results_d_targets, **results_e_targets}
sd = sorted(all_r.items(), key=lambda x: -score(x[1]))
print("\n[종합 복합 점수 순위 — 상위 10개]")
for rank, (lbl, r) in enumerate(sd[:10], 1):
    tag = " ◀v10기준" if "v10" in lbl else (" ★" if rank==1 and "v10" not in lbl else "")
    print(f"  {rank:>2}위 {lbl:<34}: 점수 {score(r):.2f} | 승률 {r['win_rate']:.1f}% | CAGR {r['cagr']:+.1f}% | PF {pf_s(r)} | 보유 {r['avg_hold']:.0f}일{tag}")

# 연도별 비교 (v10, BB기본, D최고, E최고)
best_d = max(results_d_targets.items(), key=lambda x: score(x[1]))[0]
best_e = max(results_e_targets.items(), key=lambda x: score(x[1]))[0]
show = [("A: v10원본", results_main["A: v10원본"]),
        ("B: BB저가%B≤0", results_main["B: BB저가%B≤0"]),
        (best_d, results_d_targets[best_d]),
        (best_e, results_e_targets[best_e])]
all_years = sorted(set().union(*[set(r["df"]["entry_date"].dt.year) for _,r in show if not r["df"].empty]))
print(f"\n[연도별 비교 — v10 / BB기본 / {best_d} / {best_e}]")
hdr2 = f"  {'연도':<6}" + "".join(f" {k:>24}" for k,_ in show)
print(hdr2); print("  " + "-"*104)
for yr in all_years:
    row_str = f"  {yr:<6}"
    for _, r in show:
        if r["df"].empty: row_str += f" {'  -':>24}"; continue
        sub = r["df"][r["df"]["entry_date"].dt.year==yr]
        if not len(sub): row_str += f" {'  -':>24}"
        else: row_str += f" {sub['win'].mean()*100:.0f}%/{sub['return_pct'].mean():+.1f}%({len(sub)}건)"
    print(row_str)


# ─────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────
# 1) 메인 비교 막대
labels_m = [r["label"] for r in results_main.values()]
colors_m = ["#e74c3c","#95a5a6","#3498db","#2ecc71","#9b59b6","#f39c12"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("%B 기반 BB 전략 비교 (시트 기존 값만 활용, 목표+20%, 2015-2026)", fontweight="bold")
for ax, (metric, title) in zip(axes.flatten(), [
    ("win_rate","승률(%)"), ("avg_ret","평균수익(%)"), ("cagr","CAGR(%)"),
    ("ev","기대값EV(%)"), ("pf","PF"), ("avg_hold","평균보유(일)")
]):
    vals = [r[metric] if not np.isnan(r[metric]) else 0 for r in results_main.values()]
    bars = ax.bar(range(len(labels_m)), vals, color=colors_m, edgecolor="white", linewidth=1.2)
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(len(labels_m)))
    ax.set_xticklabels([l.replace("A: ","").replace("B: ","") for l in labels_m],
                       rotation=25, ha="right", fontsize=8.5)
    ax.axhline(0, color="black", lw=0.5)
    ylim = ax.get_ylim(); rng = ylim[1]-ylim[0]
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+rng*0.02,
                f"{v:.1f}", ha="center", fontsize=8.5)
plt.tight_layout()
plt.savefig("backtest_bb_percentb_main.png", dpi=150, bbox_inches="tight")
print("\n📊 메인 비교 차트 저장 완료")

# 2) D/E 목표수익률별 히트맵
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4))
fig2.suptitle("D/E 조건 목표수익률별 히트맵", fontweight="bold")
for ax, (metric, title) in zip(axes2, [("win_rate","승률(%)"),("cagr","CAGR(%)"),("ev","기대값(%)")]):
    tgt_lbls = ["+10%","+15%","+20%","+25%"]
    d_vals = [results_d_targets[f"D_+{t}%"][metric] for t in [10,15,20,25]]
    e_vals = [results_e_targets[f"E_+{t}%"][metric] for t in [10,15,20,25]]
    grid   = np.array([d_vals, e_vals])
    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(tgt_lbls)
    ax.set_yticks([0,1]); ax.set_yticklabels(["D조건","E조건"])
    ax.set_title(title, fontweight="bold")
    plt.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(4):
            ax.text(j, i, f"{grid[i][j]:.1f}", ha="center", va="center",
                    fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("backtest_bb_percentb_heatmap.png", dpi=150, bbox_inches="tight")
print("📊 히트맵 저장 완료\n✅ 완료")
