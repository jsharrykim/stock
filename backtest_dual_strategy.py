"""
backtest_dual_strategy.py
==========================
투 트랙 전략: v10 저점반등 + 추세편승을 동시에 운용

[전략 A — v10 저점반등]
  진입: 현재가 < MA200 + VIX≥25 + (RSI<40 OR CCI<-100)
  청산: 목표 +20% / CB -25% / 60일 절반+수익 / 120일 시간청산

[전략 B — 골든크로스 추세]
  진입: MA50>MA200 + 현재가>MA200 + RSI>50
  청산: 목표 +25% / CB -15% / MA50 이탈 즉시 청산 / 120일 시간청산

[비교 기준]
  v10 단독 (원본)
  추세 단독 (+25%, CB -15%, MA50이탈)
  듀얼 전략 (두 전략 동시 운용, MAX_POSITIONS 공유)

공통:
  종목: 96개 (동일)
  기간: 2010-2026
  MAX_POSITIONS: 5 (전체 합산)
  MAX_DAILY: 5
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

START         = "2010-01-01"
END           = "2026-01-01"
MAX_POSITIONS = 5
MAX_DAILY     = 5

TICKERS = sorted(set([
    "SNPS","COST","AZN","AMGN","MDLZ","FTNT","CSGP","CDNS","ADP","FAST",
    "ADI","TXN","PAYX","BKNG","KLAC","MNST","ORLY","HOOD","CPRT","ISRG",
    "PANW","CDW","INTC","AAPL","AVGO","AMD","MSFT","GOOGL","NVDA","META",
    "TSLA","PLTR","MELI","MCHP","AMZN","SMCI","AMAT","MU","LRCX","CSX",
    "QCOM","ROP","INTU","ON","NXPI","STX","ASTS","AVAV","IONQ","SGML",
    "GOOG","NFLX","TMUS","ADBE","PEP","CSCO","MRVL","CRWD","DDOG","ZS",
    "TEAM","KDP","IDXX","REGN","VRTX","GILD","BIIB","ILMN","MRNA",
    "ODFL","PCAR","CTSH","VRSK","WDAY","PDD","DLTR","SBUX","ROST",
    "LULU","EBAY","MAR","CTAS","EA","CHTR","CMCSA","EXC","XEL","AEP",
    "MPWR","ENPH","SEDG","COIN","DOCU","ZM","OKTA","PTON",
]))

_kr_fonts = [f.name for f in fm.fontManager.ttflist
             if "Apple" in f.name or "Malgun" in f.name
             or "NanumGothic" in f.name or "Noto" in f.name]
if _kr_fonts:
    plt.rcParams["font.family"] = _kr_fonts[0]
plt.rcParams["axes.unicode_minus"] = False


# ─────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────
def dl_ohlcv(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty: return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw[["Open","High","Low","Close","Volume"]].copy()

def compute(d):
    d = d.copy()
    c = d["Close"]
    d["MA50"]  = c.rolling(50).mean()
    d["MA200"] = c.rolling(200).mean()
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

print("📥 종목 데이터 다운로드...")
stock_data = {}
for i, tk in enumerate(TICKERS, 1):
    d = dl_ohlcv(tk, START, END)
    if len(d) < 250: continue
    d = compute(d)
    stock_data[tk] = d
    if i % 20 == 0: print(f"  {i}/{len(TICKERS)} 완료")
print(f"  ✅ {len(stock_data)}개 종목 로드")


# ─────────────────────────────────────────
# 신호 생성
# ─────────────────────────────────────────
def build_signals():
    """A(v10)와 B(추세) 신호를 벡터화 방식으로 통합"""
    rows = []
    for tk, d in stock_data.items():
        d_c = d.dropna(subset=["MA50","MA200","RSI","CCI"])
        if len(d_c) < 2: continue
        vx = vix.reindex(d_c.index)

        # 전략 A: 벡터 조건
        cond_a = (
            (d_c["Close"] < d_c["MA200"]) &
            (vx >= 25) &
            ((d_c["RSI"] < 40) | (d_c["CCI"] < -100))
        ).fillna(False)

        # 전략 B: 벡터 조건
        cond_b = (
            (d_c["MA50"]  > d_c["MA200"]) &
            (d_c["Close"] > d_c["MA200"]) &
            (d_c["RSI"]   > 50)
        ).fillna(False)

        # A 우선 — B 중 A와 겹치는 것 제거
        cond_b_only = cond_b & ~cond_a

        for strat, cond in [("A", cond_a), ("B", cond_b_only)]:
            sig_idx = np.where(cond.values)[0]
            for i in sig_idx:
                if i + 1 >= len(d_c): continue
                ed  = d_c.index[i + 1]
                eo  = float(d_c["Open"].iloc[i + 1])
                if pd.isna(eo): continue
                rows.append({
                    "ticker": tk,
                    "sig_date": d_c.index[i],
                    "entry_day": ed,
                    "entry": eo,
                    "strategy": strat,
                    "rsi": float(d_c["RSI"].iloc[i]),
                })

    if not rows: return []
    df_sig = pd.DataFrame(rows).sort_values(["entry_day","strategy","rsi"])

    # 같은 날·같은 종목 중복 제거 (A 우선)
    df_sig = df_sig.drop_duplicates(subset=["entry_day","ticker"], keep="first")

    # 날짜별 MAX_DAILY 제한 — A 우선
    final = []
    for ed, grp in df_sig.groupby("entry_day"):
        a_items = grp[grp["strategy"]=="A"].to_dict("records")
        b_items = grp[grp["strategy"]=="B"].to_dict("records")
        day_picks = (a_items + b_items)[:MAX_DAILY]
        final.extend(day_picks)
    return final

print("\n🔍 신호 생성...")
all_signals = build_signals()
sig_a = [s for s in all_signals if s["strategy"]=="A"]
sig_b = [s for s in all_signals if s["strategy"]=="B"]
print(f"  전략A(v10): {len(sig_a):,}건 | 전략B(추세): {len(sig_b):,}건 | 합계: {len(all_signals):,}건")


# ─────────────────────────────────────────
# 시뮬레이션
# ─────────────────────────────────────────
def run_sim(signals, label=""):
    """각 신호의 strategy 필드에 따라 청산 파라미터 자동 분기"""
    trades, pos_exit_date = [], {}
    for sig in signals:
        tk, entry_day, entry = sig["ticker"], sig["entry_day"], sig["entry"]
        strat = sig.get("strategy", "A")

        # 청산 파라미터 분기
        if strat == "A":
            target_pct, circuit_pct, half_days, max_hold = 0.20, 0.25, 60, 120
            use_ma50_exit = False
        else:  # B
            target_pct, circuit_pct, half_days, max_hold = 0.25, 0.15, 999, 120
            use_ma50_exit = True

        active = {t: ed for t, ed in pos_exit_date.items() if ed > entry_day}
        if len(active) >= MAX_POSITIONS or tk in active: continue

        d = stock_data[tk]
        future = d.loc[d.index >= entry_day]
        if len(future) == 0: continue

        cb_price  = entry * (1 - circuit_pct)
        tgt_price = entry * (1 + target_pct)
        half_exited, exit_records = False, []

        for i, (fdt, row) in enumerate(future.iterrows()):
            lo, hi, cl = float(row["Low"]), float(row["High"]), float(row["Close"])
            ma50_val = float(row["MA50"]) if not pd.isna(row.get("MA50", np.nan)) else None

            if hi >= tgt_price:
                exit_records.append((fdt, tgt_price, 0.5 if half_exited else 1.0, "target"))
                break
            if lo <= cb_price:
                exit_records.append((fdt, cb_price, 0.5 if half_exited else 1.0, "circuit"))
                break
            # 전략B: MA50 이탈 즉시 청산
            if use_ma50_exit and ma50_val and cl < ma50_val:
                exit_records.append((fdt, cl, 0.5 if half_exited else 1.0, "ma50_exit"))
                break
            # 전략A: 60일 절반 청산
            if i + 1 == half_days and not half_exited and (cl - entry) / entry > 0:
                exit_records.append((fdt, cl, 0.5, "half"))
                half_exited = True
                continue
            if i + 1 >= max_hold:
                exit_records.append((fdt, cl, 0.5 if half_exited else 1.0, "time"))
                break

        if not exit_records: continue
        total_w   = sum(r[2] for r in exit_records)
        blended   = sum((r[1]-entry)/entry * r[2] for r in exit_records) / total_w
        last_exit = exit_records[-1]
        reason    = "+".join(r[3] for r in exit_records) if len(exit_records)>1 else exit_records[0][3]
        trades.append({
            "entry_date": entry_day, "exit_date": last_exit[0], "ticker": tk,
            "strategy": strat, "return_pct": blended*100,
            "hold_days": (last_exit[0]-entry_day).days,
            "exit_reason": reason, "win": blended > 0,
        })
        pos_exit_date[tk] = last_exit[0]

    return pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True) if trades else pd.DataFrame()

def calc_stats(df, label):
    if df.empty: return None
    n = len(df)
    wins, losses = df[df["win"]], df[~df["win"]]
    wr      = len(wins)/n*100
    avg_ret = df["return_pct"].mean()
    avg_w   = wins["return_pct"].mean()   if len(wins)   else 0
    avg_l   = losses["return_pct"].mean() if len(losses) else 0
    pf      = (wins["return_pct"].sum() / -losses["return_pct"].sum()
               if len(losses) and losses["return_pct"].sum()<0 else np.nan)
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
# 3가지 시나리오 실행
# ─────────────────────────────────────────
print("\n⚙️  시뮬레이션 실행...")

# 시나리오 1: v10 단독
sig_only_a = [dict(s, strategy="A") for s in all_signals if s["strategy"]=="A"]
df_v10 = run_sim(sig_only_a, "v10 단독")

# 시나리오 2: 추세 단독
sig_only_b = [dict(s, strategy="B") for s in all_signals if s["strategy"]=="B"]
df_trend = run_sim(sig_only_b, "추세 단독")

# 시나리오 3: 듀얼 (A+B 통합, MAX_POSITIONS 공유)
df_dual = run_sim(all_signals, "듀얼")

results = {}
for lbl, df in [("v10 단독\n(기준)", df_v10), ("추세 단독\n(B전략)", df_trend), ("듀얼 전략\n(A+B)", df_dual)]:
    s = calc_stats(df, lbl)
    if s: results[lbl] = s

print("\n" + "="*110)
print("  투 트랙 듀얼 전략 백테스트 (2010-2026)")
print("  A: 현재가<MA200 + VIX≥25 + RSI<40/CCI<-100 → 목표+20%, CB-25%, 120일")
print("  B: MA50>MA200 + 현재가>MA200 + RSI>50       → 목표+25%, CB-15%, MA50이탈")
print("="*110)
print(f"\n  {'전략':<20} {'건수':>5} {'승률':>7} {'평균수익':>9} {'기대값':>8} {'승평균':>8} {'패평균':>8} {'PF':>6} {'CAGR':>8} {'연속손':>5} {'평균보유':>7}")
print("  " + "-"*100)
for key, r in results.items():
    pf_s = f"{r['pf']:.2f}" if not np.isnan(r['pf']) else " N/A"
    lbl  = key.replace("\n"," ")
    mark = " ◀" if "듀얼" in key else ""
    print(f"  {lbl:<20} {r['n']:>5} {r['win_rate']:>6.1f}% {r['avg_ret']:>+8.2f}% "
          f"{r['ev']:>+7.2f}% {r['avg_win']:>+7.2f}% {r['avg_loss']:>+7.2f}% "
          f"{pf_s:>6} {r['cagr']:>+7.1f}% {r['max_consec_loss']:>4}건 {r['avg_hold']:>6.0f}일{mark}")
print("="*110)

# 듀얼 전략 내부 분해
if "듀얼 전략\n(A+B)" in results:
    df_d = results["듀얼 전략\n(A+B)"]["df"]
    print("\n[듀얼 전략 내부 분해]")
    for strat, sname in [("A","v10 저점반등"), ("B","추세편승")]:
        sub = df_d[df_d["strategy"]==strat]
        if sub.empty: continue
        wins = sub[sub["win"]]; losses = sub[~sub["win"]]
        wr   = len(wins)/len(sub)*100
        avg  = sub["return_pct"].mean()
        pf_v = (wins["return_pct"].sum() / -losses["return_pct"].sum()
                if len(losses) and losses["return_pct"].sum()<0 else float("nan"))
        pf_s = f"{pf_v:.2f}" if not np.isnan(pf_v) else "N/A"
        print(f"  전략{strat} {sname:<10}: {len(sub):>4}건 | 승률 {wr:.1f}% | 평균 {avg:+.2f}% | PF {pf_s} | 평균보유 {sub['hold_days'].mean():.0f}일")
    print(f"  A비중: {len(df_d[df_d['strategy']=='A'])}/{len(df_d)} ({len(df_d[df_d['strategy']=='A'])/len(df_d)*100:.1f}%) | "
          f"B비중: {len(df_d[df_d['strategy']=='B'])}/{len(df_d)} ({len(df_d[df_d['strategy']=='B'])/len(df_d)*100:.1f}%)")

# 연도별
print("\n[연도별 비교 — 승률/평균수익(건수)]")
all_years = sorted(set().union(*[set(r["df"]["entry_date"].dt.year) for r in results.values()]))
hdr = f"  {'연도':<6}" + "".join(f" {r['label'].replace(chr(10),' '):>24}" for r in results.values())
print(hdr); print("  " + "-"*84)
for yr in all_years:
    row_str = f"  {yr:<6}"
    for r in results.values():
        sub = r["df"][r["df"]["entry_date"].dt.year==yr]
        if not len(sub): row_str += f" {'  -':>24}"
        else:
            row_str += f" {sub['win'].mean()*100:.0f}%/{sub['return_pct'].mean():+.1f}%({len(sub)})"
    print(row_str)

# 복합 점수
print("\n[복합 점수 순위]")
score_data = []
for key, r in results.items():
    pf_val = r["pf"] if not np.isnan(r["pf"]) else 0
    score  = r["win_rate"]*0.30 + r["cagr"]*0.30 + r["ev"]*0.25 + min(pf_val,5)*0.15*10
    score_data.append((key, score, r))
score_data.sort(key=lambda x: -x[1])
for rank, (key, score, r) in enumerate(score_data, 1):
    pf_s = f"{r['pf']:.2f}" if not np.isnan(r['pf']) else "N/A"
    mark = " ◀듀얼" if "듀얼" in key else (" ◀기준" if "v10" in key else "")
    print(f"  {rank}위 {r['label'].replace(chr(10),' '):<22}: 점수 {score:.2f} | 승률 {r['win_rate']:.1f}% | CAGR {r['cagr']:+.1f}% | EV {r['ev']:+.2f}% | PF {pf_s} | 보유 {r['avg_hold']:.0f}일{mark}")

# ─────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────
labels_p = [r["label"].replace("\n"," ") for r in results.values()]
colors_p = ["#e74c3c", "#3498db", "#9b59b6"]

fig, axes = plt.subplots(2, 4, figsize=(22, 11))
fig.suptitle(
    "투 트랙 듀얼 전략 vs 단독 전략 비교 (2010-2026)\n"
    "빨간=v10단독 | 파란=추세단독 | 보라=듀얼(A+B)",
    fontweight="bold", fontsize=12
)
metric_data = [
    ("승률 (%)",          [r["win_rate"] for r in results.values()]),
    ("평균 수익률 (%)",    [r["avg_ret"]  for r in results.values()]),
    ("CAGR (%)",          [r["cagr"]     for r in results.values()]),
    ("기대값 EV (%)",     [r["ev"]       for r in results.values()]),
    ("Profit Factor",     [r["pf"] if not np.isnan(r["pf"]) else 0 for r in results.values()]),
    ("평균 보유 기간 (일)",[r["avg_hold"] for r in results.values()]),
    ("거래 건수",         [r["n"]        for r in results.values()]),
    ("최대 연속 손실 (건)",[r["max_consec_loss"] for r in results.values()]),
]
for ax, (title, vals) in zip(axes.flatten(), metric_data):
    bars = ax.bar(labels_p, vals, color=colors_p, edgecolor="white", linewidth=1.2)
    ax.set_title(title, fontweight="bold")
    ax.axhline(0, color="black", lw=0.5)
    ylim = ax.get_ylim()
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+(ylim[1]-ylim[0])*0.02,
                f"{v:.1f}", ha="center", fontsize=9)
    ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("backtest_dual_strategy.png", dpi=150, bbox_inches="tight")
print("\n📊 backtest_dual_strategy.png 저장 완료")

# 누적 수익 곡선
fig2, ax2 = plt.subplots(figsize=(14, 6))
ax2.set_title("연도별 평균 수익률 추이 비교", fontweight="bold")
colors_line = ["#e74c3c", "#3498db", "#9b59b6"]
markers     = ["o", "s", "^"]
for (key, r), c, mk in zip(results.items(), colors_line, markers):
    df_r = r["df"].copy()
    df_r["year"] = df_r["entry_date"].dt.year
    yr_avg = df_r.groupby("year")["return_pct"].mean()
    ax2.plot(yr_avg.index, yr_avg.values, marker=mk, color=c,
             label=r["label"].replace("\n"," "), linewidth=2, markersize=6)
ax2.axhline(0, color="black", lw=0.8, linestyle="--")
ax2.set_xlabel("연도"); ax2.set_ylabel("평균 수익률 (%)")
ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("backtest_dual_strategy_trend.png", dpi=150, bbox_inches="tight")
print("📊 backtest_dual_strategy_trend.png 저장 완료")

results["듀얼 전략\n(A+B)"]["df"].to_csv("backtest_dual_trades.csv", index=False)
print("📄 backtest_dual_trades.csv 저장 완료")

print("\n" + "="*80)
print("  ★ 최종 분석 요약")
print("="*80)
r_v10  = results.get("v10 단독\n(기준)")
r_dual = results.get("듀얼 전략\n(A+B)")
r_b    = results.get("추세 단독\n(B전략)")
if r_v10 and r_dual:
    dn = r_dual["n"] - r_v10["n"]
    dw = r_dual["win_rate"] - r_v10["win_rate"]
    dc = r_dual["cagr"]     - r_v10["cagr"]
    de = r_dual["ev"]       - r_v10["ev"]
    dp = r_dual["pf"]       - r_v10["pf"]
    print(f"\n  v10 단독  : 거래 {r_v10['n']:>4}건 | 승률 {r_v10['win_rate']:.1f}% | CAGR {r_v10['cagr']:+.1f}% | PF {r_v10['pf']:.2f} | EV {r_v10['ev']:+.2f}%")
    print(f"  추세 단독 : 거래 {r_b['n']:>4}건 | 승률 {r_b['win_rate']:.1f}% | CAGR {r_b['cagr']:+.1f}% | PF {r_b['pf']:.2f} | EV {r_b['ev']:+.2f}%")
    print(f"  듀얼 전략 : 거래 {r_dual['n']:>4}건 | 승률 {r_dual['win_rate']:.1f}% | CAGR {r_dual['cagr']:+.1f}% | PF {r_dual['pf']:.2f} | EV {r_dual['ev']:+.2f}%")
    print(f"\n  v10 대비 듀얼: 거래수 {dn:+}건 | 승률 {dw:+.1f}%p | CAGR {dc:+.1f}%p | EV {de:+.2f}%p | PF {dp:+.2f}")
    verdict = "개선" if dc > 0 and dw > -5 else ("혼합" if dc > 0 else "비개선")
    print(f"\n  → 판정: [{verdict}] ", end="")
    if verdict == "개선":
        print("듀얼 전략이 v10보다 전반적으로 우수합니다.")
    elif verdict == "혼합":
        print("CAGR은 개선됐으나 승률/PF가 희석됩니다. 거래빈도 ↑, 퀄리티 ↓ 트레이드오프.")
    else:
        print("v10 단독이 더 우수합니다. 추세 신호가 퀄리티를 희석합니다.")
print("\n  ※ 슬리피지·세금 미반영. 과거 데이터 기반.")
print("="*80)
print("\n✅ 완료")
