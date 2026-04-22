"""
backtest_youtube_strategies.py
================================
영상 3개에서 추출한 전략 백테스트 + v10 원본과 비교

[전략 1] 성승현 — 월봉 12이평선 (일봉 환산 MA250 돌파)
  - 매수: 현재가 > MA250 (월봉 12이평 ≈ 일봉 250일선)
  - 매도: 현재가 < MA250 이탈
  - 특징: 장기 추세 추종, 진입/청산 모두 이평선 기준

[전략 2] 전황 — 60/120일선 추세 + 20일선 눌림목 진입
  - 강세 판단: 현재가 > MA60 AND 현재가 > MA120
  - 진입: 강세 구간에서 현재가가 MA20 근처 (MA20 대비 -3% 이내) 눌림목
  - 청산: 현재가 < MA60 이탈 (추세 이탈) 또는 TARGET_PCT 달성
  - 특징: 추세 중 눌림목 매수

[전략 3] 영상3 오더블록/FBG
  - 알고리즘화 불가 (주관적 패턴 인식) → 제외

[비교 기준] v10 원본
  - 현재가 < MA200 + VIX≥25 + (RSI<40 OR CCI<-100)
  - TARGET +20%, CB -25%

모든 전략 공통:
  - 종목 유니버스: 동일 96개
  - 기간: 2010-2026
  - MAX_POSITIONS: 5
  - MAX_DAILY: 5
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
MAX_POSITIONS= 5
MAX_DAILY    = 5

# 전략별 파라미터
PARAMS = {
    "v10 원본":    {"target": 0.20, "circuit": 0.25, "half_exit": 60,  "max_hold": 120},
    "전략1_월봉MA": {"target": 0.30, "circuit": 0.25, "half_exit": 999, "max_hold": 500},  # 이탈 청산 위주
    "전략2_눌림목": {"target": 0.15, "circuit": 0.10, "half_exit": 60,  "max_hold": 120},  # 단기 손절 빡빡
}

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


# ══════════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════════
def dl_ohlcv(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty: return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw[["Open","High","Low","Close","Volume"]].copy()

def compute(d):
    d = d.copy()
    c = d["Close"]
    d["MA20"]  = c.rolling(20).mean()
    d["MA50"]  = c.rolling(50).mean()
    d["MA60"]  = c.rolling(60).mean()
    d["MA120"] = c.rolling(120).mean()
    d["MA200"] = c.rolling(200).mean()
    d["MA250"] = c.rolling(250).mean()   # 월봉 12이평 근사
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
    if len(d) < 300: continue
    d = compute(d)
    stock_data[tk] = d
    if i % 20 == 0: print(f"  {i}/{len(TICKERS)} 완료")
print(f"  ✅ {len(stock_data)}개 종목 로드")


# ══════════════════════════════════════════════════
# 전략별 신호 생성
# ══════════════════════════════════════════════════
def build_v10_signals():
    """원본 v10: 현재가<MA200 + VIX≥25 + (RSI<40 OR CCI<-100)"""
    sig_map = {}
    for tk, d in stock_data.items():
        d_c  = d.dropna(subset=["MA200","RSI","CCI"])
        comm = d_c.index.intersection(vix.index)
        if len(comm) < 50: continue
        vx = vix.reindex(d_c.index)
        cond = (
            (d_c["Close"] < d_c["MA200"]) &
            (vx >= 25) &
            ((d_c["RSI"] < 40) | (d_c["CCI"] < -100))
        )
        for dt in d_c.index[cond.reindex(d_c.index).fillna(False)]:
            idx = d_c.index.get_loc(dt)
            if idx + 1 >= len(d_c): continue
            ed = d_c.index[idx+1]
            eo = float(d_c["Open"].iloc[idx+1])
            if pd.isna(eo): continue
            if ed not in sig_map: sig_map[ed] = []
            sig_map[ed].append({"ticker": tk, "entry_day": ed, "entry": eo,
                                 "rsi": float(d_c["RSI"].loc[dt])})
    final = []
    for ed, items in sorted(sig_map.items()):
        items.sort(key=lambda x: x["rsi"])
        final.extend(items[:MAX_DAILY])
    return final

def build_ma250_signals():
    """전략1 성승현: 현재가가 MA250 위로 올라올 때 매수 신호"""
    sig_map = {}
    for tk, d in stock_data.items():
        d_c = d.dropna(subset=["MA250"])
        # 전날 MA250 아래, 오늘 MA250 위 → 돌파 신호
        above   = d_c["Close"] > d_c["MA250"]
        crossed = above & ~above.shift(1).fillna(False)
        for dt in d_c.index[crossed]:
            idx = d_c.index.get_loc(dt)
            if idx + 1 >= len(d_c): continue
            ed = d_c.index[idx+1]
            eo = float(d_c["Open"].iloc[idx+1])
            if pd.isna(eo): continue
            if ed not in sig_map: sig_map[ed] = []
            sig_map[ed].append({"ticker": tk, "entry_day": ed, "entry": eo,
                                 "rsi": float(d_c["RSI"].loc[dt]) if not pd.isna(d_c["RSI"].loc[dt]) else 50,
                                 "ma250": float(d_c["MA250"].loc[dt])})
    final = []
    for ed, items in sorted(sig_map.items()):
        items.sort(key=lambda x: x["rsi"])
        final.extend(items[:MAX_DAILY])
    return final

def build_pullback_signals():
    """전략2 전황: MA60/MA120 위 강세장 + MA20 근처 눌림목"""
    sig_map = {}
    for tk, d in stock_data.items():
        d_c = d.dropna(subset=["MA20","MA60","MA120"])
        cond = (
            (d_c["Close"] > d_c["MA60"]) &
            (d_c["Close"] > d_c["MA120"]) &
            (d_c["Close"] / d_c["MA20"] - 1).between(-0.03, 0.03)  # MA20 ±3% 눌림목
        )
        for dt in d_c.index[cond]:
            idx = d_c.index.get_loc(dt)
            if idx + 1 >= len(d_c): continue
            ed = d_c.index[idx+1]
            eo = float(d_c["Open"].iloc[idx+1])
            if pd.isna(eo): continue
            if ed not in sig_map: sig_map[ed] = []
            sig_map[ed].append({"ticker": tk, "entry_day": ed, "entry": eo,
                                 "rsi": float(d_c["RSI"].loc[dt]) if not pd.isna(d_c["RSI"].loc[dt]) else 50,
                                 "ma20": float(d_c["MA20"].loc[dt])})
    final = []
    for ed, items in sorted(sig_map.items()):
        items.sort(key=lambda x: x["rsi"])
        final.extend(items[:MAX_DAILY])
    return final


# ══════════════════════════════════════════════════
# 시뮬레이션 (전략1은 MA250 이탈 시 추가 청산 포함)
# ══════════════════════════════════════════════════
def run_simulation(signals, target_pct, circuit_pct, half_exit, max_hold,
                   use_ma250_exit=False, use_ma60_exit=False):
    trades, pos_exit_date = [], {}
    for sig in signals:
        tk, entry_day, entry = sig["ticker"], sig["entry_day"], sig["entry"]
        active = {t: ed for t, ed in pos_exit_date.items() if ed > entry_day}
        if len(active) >= MAX_POSITIONS or tk in active: continue
        d = stock_data[tk]
        future = d.loc[d.index >= entry_day]
        if len(future) == 0: continue
        cb_price, tgt_price = entry * (1 - circuit_pct), entry * (1 + target_pct)
        half_exited, exit_records = False, []
        for i, (fdt, row) in enumerate(future.iterrows()):
            lo, hi, cl = float(row["Low"]), float(row["High"]), float(row["Close"])
            ma250_val = float(row["MA250"]) if not pd.isna(row.get("MA250", np.nan)) else None
            ma60_val  = float(row["MA60"])  if not pd.isna(row.get("MA60",  np.nan)) else None
            # 목표가
            if hi >= tgt_price:
                exit_records.append((fdt, tgt_price, 0.5 if half_exited else 1.0, "target")); break
            # CB
            if lo <= cb_price:
                exit_records.append((fdt, cb_price, 0.5 if half_exited else 1.0, "circuit")); break
            # 전략1: MA250 이탈 청산
            if use_ma250_exit and ma250_val and cl < ma250_val:
                exit_records.append((fdt, cl, 0.5 if half_exited else 1.0, "ma250_exit")); break
            # 전략2: MA60 이탈 청산
            if use_ma60_exit and ma60_val and cl < ma60_val:
                exit_records.append((fdt, cl, 0.5 if half_exited else 1.0, "ma60_exit")); break
            # 60일 절반
            if i + 1 == half_exit and not half_exited and (cl - entry) / entry > 0:
                exit_records.append((fdt, cl, 0.5, "half")); half_exited = True; continue
            # time exit
            if i + 1 >= max_hold:
                exit_records.append((fdt, cl, 0.5 if half_exited else 1.0, "time")); break
        if not exit_records: continue
        total_pct   = sum(r[2] for r in exit_records)
        weighted    = sum((r[1] - entry) / entry * r[2] for r in exit_records)
        blended_ret = weighted / total_pct if total_pct > 0 else 0
        last_exit   = exit_records[-1]
        reason      = "+".join(r[3] for r in exit_records) if len(exit_records)>1 else exit_records[0][3]
        trades.append({
            "entry_date": entry_day, "exit_date": last_exit[0], "ticker": tk,
            "return_pct": blended_ret*100, "hold_days": (last_exit[0]-entry_day).days,
            "exit_reason": reason, "win": blended_ret > 0,
        })
        pos_exit_date[tk] = last_exit[0]
    return pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True) if trades else pd.DataFrame()

def calc_stats(df, label):
    if df.empty: return None
    n = len(df)
    wins, losses = df[df["win"]], df[~df["win"]]
    wr      = len(wins) / n * 100
    avg_ret = df["return_pct"].mean()
    avg_w   = wins["return_pct"].mean()   if len(wins)   else 0
    avg_l   = losses["return_pct"].mean() if len(losses) else 0
    pf      = (wins["return_pct"].sum() / -losses["return_pct"].sum()
               if len(losses) and losses["return_pct"].sum() < 0 else np.nan)
    ev      = wr/100 * avg_w + (1 - wr/100) * avg_l
    max_cl = cur = 0
    for r in df["return_pct"]:
        cur = cur+1 if r < 0 else 0; max_cl = max(max_cl, cur)
    cap = 1.0
    for _, grp in df.groupby("entry_date"):
        cap *= (1 + (grp["return_pct"]/100 * (1.0/max(len(grp),MAX_POSITIONS))).sum())
    yrs  = (df["exit_date"].max() - df["entry_date"].min()).days / 365.25
    cagr = cap**(1/yrs) - 1 if yrs > 0 else 0
    exit_dist = df["exit_reason"].value_counts(normalize=True)*100
    return {
        "label": label, "n": n, "win_rate": wr, "avg_ret": avg_ret,
        "avg_win": avg_w, "avg_loss": avg_l, "pf": pf, "ev": ev,
        "cagr": cagr*100, "max_consec_loss": max_cl,
        "avg_hold": df["hold_days"].mean(),
        "exit_dist": dict(df["exit_reason"].value_counts()),
        "df": df,
    }


# ══════════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════════
print("\n⚙️  전략별 신호 생성 & 시뮬레이션...")

sig_v10      = build_v10_signals()
sig_ma250    = build_ma250_signals()
sig_pullback = build_pullback_signals()
print(f"  v10 원본: {len(sig_v10):,}건 | 전략1 MA250: {len(sig_ma250):,}건 | 전략2 눌림목: {len(sig_pullback):,}건")

p1 = PARAMS["v10 원본"]
p2 = PARAMS["전략1_월봉MA"]
p3 = PARAMS["전략2_눌림목"]

df_v10   = run_simulation(sig_v10,      p1["target"], p1["circuit"], p1["half_exit"], p1["max_hold"])
df_ma250 = run_simulation(sig_ma250,    p2["target"], p2["circuit"], p2["half_exit"], p2["max_hold"], use_ma250_exit=True)
df_pb    = run_simulation(sig_pullback, p3["target"], p3["circuit"], p3["half_exit"], p3["max_hold"], use_ma60_exit=True)

results = {}
for lbl, df in [("v10 원본", df_v10), ("전략1_성승현\n(MA250돌파)", df_ma250), ("전략2_전황\n(눌림목)", df_pb)]:
    s = calc_stats(df, lbl)
    if s: results[lbl] = s
    wr_str = f"{s['win_rate']:.1f}%" if s else "N/A"
    cagr_str = f"{s['cagr']:+.1f}%" if s else "N/A"
    n_str = str(s['n']) if s else "0"
    print(f"  [{lbl.replace(chr(10),' ')}] {n_str}건 | 승률 {wr_str} | CAGR {cagr_str}")


# ══════════════════════════════════════════════════
# 결과 출력
# ══════════════════════════════════════════════════
print("\n" + "="*100)
print("  영상 전략 vs v10 원본 백테스트 비교 (2010-2026)")
print("="*100)

rows = [
    ("v10 원본",            "현재가<MA200 + VIX≥25 + RSI<40 OR CCI<-100",             "+20%", "-25%", "60일절반/120일"),
    ("전략1 성승현 MA250",   "현재가 > MA250 돌파 진입, MA250 이탈 청산",               "+30%", "-25%", "MA250이탈"),
    ("전략2 전황 눌림목",    "현재가>MA60 AND >MA120 + MA20±3% 눌림목 진입, MA60이탈 청산", "+15%", "-10%", "MA60이탈"),
]
print(f"\n  {'전략':<20} {'조건 요약':<52} {'목표':>5} {'CB':>5} {'청산'}")
print("  " + "-"*95)
for lbl, desc, tgt, cb, exit_type in rows:
    print(f"  {lbl:<20} {desc:<52} {tgt:>5} {cb:>5} {exit_type}")

print(f"\n  {'전략':<22} {'건수':>5} {'승률':>7} {'평균수익':>9} {'기대값':>8} {'승평균':>8} {'패평균':>8} {'PF':>6} {'CAGR':>8} {'연속손':>5} {'평균보유':>7}")
print("  " + "-"*98)
for key, r in results.items():
    pf_s = f"{r['pf']:.2f}" if not np.isnan(r['pf']) else " N/A"
    lbl  = key.replace("\n", " ")
    mark = " ◀기준" if key == "v10 원본" else ""
    print(f"  {lbl:<22} {r['n']:>5} {r['win_rate']:>6.1f}% {r['avg_ret']:>+8.2f}% "
          f"{r['ev']:>+7.2f}% {r['avg_win']:>+7.2f}% {r['avg_loss']:>+7.2f}% "
          f"{pf_s:>6} {r['cagr']:>+7.1f}% {r['max_consec_loss']:>4}건 "
          f"{r['avg_hold']:>6.0f}일{mark}")
print("="*100)

# 연도별
print("\n[연도별 승률 & 평균수익]")
all_years = sorted(set().union(*[set(r["df"]["entry_date"].dt.year) for r in results.values()]))
hdr = f"  {'연도':<6}" + "".join(f" {r['label'].replace(chr(10),' '):>22}" for r in results.values())
print(hdr); print("  " + "-"*80)
for yr in all_years:
    row_str = f"  {yr:<6}"
    for r in results.values():
        sub = r["df"][r["df"]["entry_date"].dt.year == yr]
        if len(sub) == 0: row_str += f" {'  -':>22}"
        else:
            wr  = sub["win"].mean()*100
            avg = sub["return_pct"].mean()
            row_str += f" {wr:.0f}%/{avg:+.1f}%({len(sub)})"
    print(row_str)

# 청산 유형
print("\n[청산 유형 분포]")
for r in results.values():
    ec = r["exit_dist"]; tt = sum(ec.values())
    tops = sorted(ec.items(), key=lambda x: -x[1])[:4]
    parts = [f"{k}: {v}건({v/tt*100:.0f}%)" for k, v in tops]
    print(f"  {r['label'].replace(chr(10),' '):<22}: {' | '.join(parts)}")

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
    mark = " ◀기준" if key == "v10 원본" else ""
    print(f"  {rank}위 {r['label'].replace(chr(10),' '):<22}: 점수 {score:.2f} | "
          f"승률 {r['win_rate']:.1f}% | CAGR {r['cagr']:+.1f}% | EV {r['ev']:+.2f}% | PF {pf_s}{mark}")


# ══════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════
labels_p = [r["label"].replace("\n"," ") for r in results.values()]
colors_p = ["#e74c3c", "#3498db", "#2ecc71"]

fig, axes = plt.subplots(2, 4, figsize=(22, 11))
fig.suptitle(
    "영상 전략 vs v10 원본 백테스트 비교 (2010-2026)\n"
    "빨간=v10원본 | 파란=전략1 성승현(MA250돌파) | 초록=전략2 전황(눌림목)",
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
    ax.tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig("backtest_youtube_strategies.png", dpi=150, bbox_inches="tight")
print("\n📊 backtest_youtube_strategies.png 저장 완료")

for key, r in results.items():
    fname = f"backtest_{key.replace(chr(10),'_').replace(' ','_')}_trades.csv"
    r["df"].to_csv(fname, index=False)
print("📄 CSV 저장 완료")

# ══════════════════════════════════════════════════
# 최종 분석
# ══════════════════════════════════════════════════
print("\n" + "="*80)
print("  ★ 최종 분석 요약")
print("="*80)
base = results.get("v10 원본")
if base:
    print(f"\n  [기준: v10 원본]")
    print(f"    승률 {base['win_rate']:.1f}% | CAGR {base['cagr']:+.1f}% | PF {base['pf']:.2f} | 거래 {base['n']}건")
    for key, r in results.items():
        if key == "v10 원본": continue
        dcagr = r["cagr"] - base["cagr"]
        dwr   = r["win_rate"] - base["win_rate"]
        dpf   = r["pf"] - base["pf"]
        print(f"\n  [{r['label'].replace(chr(10),' ')}]")
        print(f"    승률 {r['win_rate']:.1f}% ({dwr:+.1f}%p) | CAGR {r['cagr']:+.1f}% ({dcagr:+.1f}%p) | PF {r['pf']:.2f} ({dpf:+.2f})")
        print(f"    거래수: {r['n']}건 | 평균보유: {r['avg_hold']:.0f}일 | 연속손실: {r['max_consec_loss']}건")
print("\n  ※ 2010-2026 과거 데이터 기반. 슬리피지·세금 미반영.")
print("="*80)
print("\n✅ 백테스트 완료")
