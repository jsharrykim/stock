"""
backtest_trend_target_compare.py
=================================
추세 편승 조건(+B 골든크로스 — 가장 성과 좋았던 추세 조건)에서
목표 수익률을 +10% ~ +50%까지 다양하게 비교.

[기준 전략: +B 골든크로스]
  매수: MA50>MA200 + 현재가>MA200 + RSI>50
  (VIX 조건 없음 — 추세장이므로)

비교 TARGET: 10%, 15%, 20%, 25%, 30%, 35%, 40%, 50%
CB: -25% 고정 (변경 없음)
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

START        = "2010-01-01"
END          = "2026-01-01"
CIRCUIT_PCT  = 0.25
HALF_EXIT    = 60
MAX_HOLD     = 120
MAX_POSITIONS= 5
MAX_DAILY    = 5

TARGET_GROUPS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

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
    return d

print("📥 종목 데이터 다운로드...")
stock_data = {}
for i, tk in enumerate(TICKERS, 1):
    d = dl_ohlcv(tk, START, END)
    if len(d) < 250: continue
    d = compute(d)
    stock_data[tk] = d
    if i % 20 == 0: print(f"  {i}/{len(TICKERS)} 완료")
print(f"  ✅ {len(stock_data)}개 종목 로드")

# 신호 생성 (TARGET 무관 — 공통)
print("🔍 신호 생성 (골든크로스 추세)...")
signals_by_date = {}
for tk, d in stock_data.items():
    d_c = d.dropna(subset=["MA50","MA200","RSI"])
    cond = (
        (d_c["MA50"]  > d_c["MA200"]) &
        (d_c["Close"] > d_c["MA200"]) &
        (d_c["RSI"]   > 50)
    )
    sig_days = d_c.index[cond]
    for sig_day in sig_days:
        idx = d_c.index.get_loc(sig_day)
        if idx + 1 >= len(d_c): continue
        entry_day  = d_c.index[idx + 1]
        entry_open = float(d_c["Open"].iloc[idx + 1])
        if pd.isna(entry_open): continue
        row = d_c.loc[sig_day]
        if entry_day not in signals_by_date:
            signals_by_date[entry_day] = []
        signals_by_date[entry_day].append({
            "ticker": tk, "entry_day": entry_day,
            "entry": entry_open, "rsi": float(row["RSI"]),
        })

final_signals = []
for entry_day, items in sorted(signals_by_date.items()):
    items.sort(key=lambda x: x["rsi"])
    for item in items[:MAX_DAILY]:
        final_signals.append(item)
print(f"  ✅ 원시 신호: {len(final_signals):,}건")

def run_simulation(signals, target_pct):
    trades, pos_exit_date = [], {}
    for sig in signals:
        tk, entry_day, entry = sig["ticker"], sig["entry_day"], sig["entry"]
        active = {t: ed for t, ed in pos_exit_date.items() if ed > entry_day}
        if len(active) >= MAX_POSITIONS or tk in active: continue
        d = stock_data[tk]
        future = d.loc[d.index >= entry_day]
        if len(future) == 0: continue
        cb_price, tgt_price = entry * (1 - CIRCUIT_PCT), entry * (1 + target_pct)
        half_exited, exit_records = False, []
        for i, (fdt, row) in enumerate(future.iterrows()):
            lo, hi, cl = float(row["Low"]), float(row["High"]), float(row["Close"])
            if hi >= tgt_price:
                exit_records.append((fdt, tgt_price, 0.5 if half_exited else 1.0, "target")); break
            if lo <= cb_price:
                exit_records.append((fdt, cb_price, 0.5 if half_exited else 1.0, "circuit")); break
            if i + 1 == HALF_EXIT and not half_exited and (cl - entry) / entry > 0:
                exit_records.append((fdt, cl, 0.5, "half_60d")); half_exited = True; continue
            if i + 1 >= MAX_HOLD:
                exit_records.append((fdt, cl, 0.5 if half_exited else 1.0, "time")); break
        if not exit_records: continue
        total_pct   = sum(r[2] for r in exit_records)
        weighted    = sum((r[1] - entry) / entry * r[2] for r in exit_records)
        blended_ret = weighted / total_pct if total_pct > 0 else 0
        last_exit   = exit_records[-1]
        reason      = "+".join(r[3] for r in exit_records) if len(exit_records) > 1 else exit_records[0][3]
        trades.append({
            "entry_date": entry_day, "exit_date": last_exit[0], "ticker": tk,
            "return_pct": blended_ret * 100, "hold_days": (last_exit[0] - entry_day).days,
            "exit_reason": reason, "win": blended_ret > 0,
        })
        pos_exit_date[tk] = last_exit[0]
    return pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)

print("\n⚙️  목표 수익률별 시뮬레이션...")
results = {}
for tgt in TARGET_GROUPS:
    lbl = f"+{int(tgt*100)}%"
    df  = run_simulation(final_signals, tgt)
    if df.empty: continue
    n = len(df)
    wins   = df[df["win"]];  losses = df[~df["win"]]
    wr     = len(wins) / n * 100
    avg_ret= df["return_pct"].mean()
    avg_w  = wins["return_pct"].mean()   if len(wins)   else 0
    avg_l  = losses["return_pct"].mean() if len(losses) else 0
    pf     = (wins["return_pct"].sum() / -losses["return_pct"].sum()
              if len(losses) and losses["return_pct"].sum() < 0 else np.nan)
    ev     = wr/100 * avg_w + (1 - wr/100) * avg_l
    max_cl = cur = 0
    for r in df["return_pct"]:
        cur = cur + 1 if r < 0 else 0; max_cl = max(max_cl, cur)
    cap = 1.0
    for _, grp in df.groupby("entry_date"):
        cap *= (1 + (grp["return_pct"] / 100 * (1.0 / max(len(grp), MAX_POSITIONS))).sum())
    yrs  = (df["exit_date"].max() - df["entry_date"].min()).days / 365.25
    cagr = cap ** (1 / yrs) - 1 if yrs > 0 else 0
    exit_dist = df["exit_reason"].value_counts(normalize=True) * 100
    tgt_share = exit_dist.get("target", 0) + exit_dist.get("half_60d+target", 0)
    cb_share  = exit_dist.get("circuit", 0) + exit_dist.get("half_60d+circuit", 0)
    results[lbl] = {
        "target_pct": tgt, "n": n, "win_rate": wr, "avg_ret": avg_ret,
        "avg_win": avg_w, "avg_loss": avg_l, "pf": pf, "ev": ev,
        "cagr": cagr * 100, "max_consec_loss": max_cl,
        "avg_hold_days": df["hold_days"].mean(),
        "target_hit_pct": tgt_share, "circuit_hit_pct": cb_share,
        "exit_dist": dict(df["exit_reason"].value_counts()), "df": df,
    }
    pf_s = f"{pf:.2f}" if not np.isnan(pf) else "N/A"
    print(f"  {lbl}: {n}건, 승률 {wr:.1f}%, 평균 {avg_ret:+.2f}%, CAGR {cagr*100:+.1f}%, PF {pf_s}")

print("\n" + "="*105)
print("  골든크로스 추세 전략 — 목표 수익률별 비교 (2010-2026)")
print("  매수: MA50>MA200 + 현재가>MA200 + RSI>50 | CB: -25%")
print("="*105)
hdr = f"  {'그룹':<7} {'건수':>5} {'승률':>7} {'평균수익':>9} {'기대값':>8} {'승평균':>8} {'패평균':>8} {'PF':>6} {'CAGR':>8} {'연속손':>5} {'평균보유':>7} {'목표%':>6} {'CB%':>5}"
print(hdr); print("  " + "-"*100)
for lbl, r in results.items():
    pf_s = f"{r['pf']:.2f}" if not np.isnan(r['pf']) else " N/A"
    print(f"  {lbl:<7} {r['n']:>5} {r['win_rate']:>6.1f}% {r['avg_ret']:>+8.2f}% {r['ev']:>+7.2f}% "
          f"{r['avg_win']:>+7.2f}% {r['avg_loss']:>+7.2f}% {pf_s:>6} {r['cagr']:>+7.1f}% "
          f"{r['max_consec_loss']:>4}건 {r['avg_hold_days']:>6.0f}일 "
          f"{r['target_hit_pct']:>5.1f}% {r['circuit_hit_pct']:>4.1f}%")
print("="*105)

# 복합 점수
score_data = []
for lbl, r in results.items():
    pf_val = r["pf"] if not np.isnan(r["pf"]) else 0
    score = r["win_rate"]*0.30 + r["cagr"]*0.30 + r["ev"]*0.25 + min(pf_val,5)*0.15*10
    score_data.append((lbl, score, r))
score_data.sort(key=lambda x: -x[1])
print("\n[복합 점수 순위]")
for rank, (lbl, score, r) in enumerate(score_data, 1):
    pf_s = f"{r['pf']:.2f}" if not np.isnan(r['pf']) else "N/A"
    print(f"  {rank}위 {lbl}: 점수 {score:.2f} | 승률 {r['win_rate']:.1f}% | CAGR {r['cagr']:+.1f}% | EV {r['ev']:+.2f}% | PF {pf_s} | 보유 {r['avg_hold_days']:.0f}일")

# 차트
labels = list(results.keys())
colors = ["#e74c3c" if r["target_pct"]==0.20 else "#3498db" for r in results.values()]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("골든크로스 추세 전략 — 목표 수익률별 비교\nMA50>MA200 + 현재가>MA200 + RSI>50 | CB:-25%", fontweight="bold")
metrics = [
    ("승률 (%)", [r["win_rate"] for r in results.values()]),
    ("평균 수익률 (%)", [r["avg_ret"] for r in results.values()]),
    ("CAGR (%)", [r["cagr"] for r in results.values()]),
    ("기대값 EV (%)", [r["ev"] for r in results.values()]),
    ("Profit Factor", [r["pf"] if not np.isnan(r["pf"]) else 0 for r in results.values()]),
    ("평균 보유 기간 (일)", [r["avg_hold_days"] for r in results.values()]),
]
for ax, (title, vals) in zip(axes.flatten(), metrics):
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_title(title, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+ax.get_ylim()[1]*0.01,
                f"{v:.1f}", ha="center", fontsize=8.5)
    ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig("backtest_trend_target_compare.png", dpi=150, bbox_inches="tight")
print("\n📊 backtest_trend_target_compare.png 저장 완료\n✅ 완료")
