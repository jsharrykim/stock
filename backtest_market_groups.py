"""
backtest_market_groups.py
==========================
v10 전략을 4개 시장 그룹에 적용해 비교

[전략 조건 — 동일]
  매수: 현재가 < MA200 + VIX≥25 + (RSI<40 OR CCI<-100)
  청산: 목표 +20% / CB -25% / 60일 절반+수익 / 120일 시간청산

[그룹]
  1. 나스닥 100  — QQQ 구성 종목 (현재 공식 100개, 상장 기간 부족 시 제외)
  2. 다우 30     — DJIA 구성 30개
  3. 코스피 100  — 코스피 시총 상위 약 100개 (.KS 접미사)
  4. 코스닥 100  — 코스닥 시총 상위 약 100개 (.KQ 접미사)

기간: 2010-2026
VIX: ^VIX (미국·한국 공통 적용 — 글로벌 공포지수)
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
TARGET_PCT    = 0.20
CIRCUIT_PCT   = 0.25
HALF_DAYS     = 60
MAX_HOLD      = 120

# ── 나스닥 100 (2026년 기준 공식 구성) ──────────────────────────────
NDX100 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","TMUS","CSCO","AMD","PEP","LIN","ADBE","TXN","QCOM","AMGN",
    "INTU","ISRG","BKNG","CMCSA","AMAT","MU","ADI","LRCX","REGN","PANW",
    "VRTX","KLAC","ADP","SBUX","GILD","MDLZ","SNPS","CDNS","MELI","INTC",
    "CTAS","ORLY","CSX","PYPL","ABNB","CRWD","MNST","FTNT","DASH","MRVL",
    "PCAR","ROP","WDAY","ROST","CHTR","KDP","EXC","FAST","IDXX","ODFL",
    "DDOG","VRSK","TEAM","PAYX","CPRT","ZS","CTSH","BKR","ON","BIIB",
    "NXPI","DLTR","MRNA","FANG","CDW","AEP","XEL","TTWO","GFS","APP",
    "TTD","ARM","CEG","GEHC","ILMN","SMCI","EA","MAR","EBAY","LULU",
    "PDD","WBD","MDB","HOOD","PLTR","SIRI","ALGN","DXCM","ENPH","ZM",
]

# ── 다우 30 ───────────────────────────────────────────────────────────
DOW30 = [
    "AAPL","MSFT","UNH","GS","HD","MCD","CAT","AMGN","V","CRM",
    "TRV","AXP","BA","HON","IBM","JPM","JNJ","WMT","PG","CVX",
    "MRK","MMM","NKE","DIS","KO","CSCO","VZ","DOW","INTC","WBA",
]

# ── 코스피 시총 상위 100 (.KS) ────────────────────────────────────────
KOSPI100 = [
    "005930.KS","000660.KS","005380.KS","035420.KS","005490.KS",
    "000270.KS","051910.KS","068270.KS","035720.KS","105560.KS",
    "055550.KS","032830.KS","012330.KS","207940.KS","066570.KS",
    "003550.KS","028260.KS","096770.KS","017670.KS","034730.KS",
    "015760.KS","086790.KS","000810.KS","047050.KS","011170.KS",
    "316140.KS","009150.KS","010130.KS","003490.KS","018260.KS",
    "010950.KS","011790.KS","034020.KS","033780.KS","024110.KS",
    "090430.KS","139480.KS","000100.KS","004020.KS","161390.KS",
    "036570.KS","180640.KS","011200.KS","030200.KS","271560.KS",
    "029780.KS","008770.KS","011780.KS","000720.KS","009540.KS",
    "002380.KS","004490.KS","010140.KS","003670.KS","086280.KS",
    "138040.KS","097950.KS","071050.KS","032640.KS","000080.KS",
    "028050.KS","007070.KS","018880.KS","011170.KS","004370.KS",
    "005830.KS","078930.KS","001800.KS","009830.KS","005940.KS",
    "003230.KS","006400.KS","001450.KS","007310.KS","002790.KS",
    "000990.KS","005935.KS","088350.KS","071970.KS","004800.KS",
    "047810.KS","004170.KS","010060.KS","042660.KS","001040.KS",
    "011070.KS","000120.KS","016360.KS","008060.KS","003410.KS",
    "006800.KS","002760.KS","069960.KS","006120.KS","004000.KS",
    "001630.KS","002350.KS","000240.KS","009200.KS","003000.KS",
]

# ── 코스닥 시총 상위 100 (.KQ) ────────────────────────────────────────
KOSDAQ100 = [
    "247540.KQ","091990.KQ","196170.KQ","086520.KQ","263750.KQ",
    "357780.KQ","145020.KQ","112040.KQ","041510.KQ","035760.KQ",
    "068760.KQ","031980.KQ","054040.KQ","240810.KQ","039030.KQ",
    "950130.KQ","141080.KQ","256840.KQ","031370.KQ","083790.KQ",
    "293490.KQ","042700.KQ","140410.KQ","067160.KQ","058470.KQ",
    "028300.KQ","950170.KQ","214150.KQ","066970.KQ","048260.KQ",
    "078140.KQ","052690.KQ","122870.KQ","036810.KQ","065660.KQ",
    "064760.KQ","226330.KQ","900110.KQ","035420.KQ","950140.KQ",
    "041960.KQ","036530.KQ","049070.KQ","122990.KQ","032280.KQ",
    "090460.KQ","140860.KQ","065130.KQ","222080.KQ","067310.KQ",
    "039200.KQ","082210.KQ","251970.KQ","033500.KQ","101140.KQ",
    "073010.KQ","080530.KQ","041830.KQ","089600.KQ","060310.KQ",
    "950160.KQ","053160.KQ","215600.KQ","058820.KQ","036460.KQ",
    "032500.KQ","064350.KQ","038680.KQ","900290.KQ","076080.KQ",
    "041020.KQ","045890.KQ","036200.KQ","211270.KQ","017510.KQ",
    "290510.KQ","048830.KQ","039440.KQ","108490.KQ","060280.KQ",
    "007390.KQ","083500.KQ","054620.KQ","036810.KQ","057540.KQ",
    "023160.KQ","051500.KQ","078600.KQ","024120.KQ","033290.KQ",
    "060900.KQ","040300.KQ","043150.KQ","028040.KQ","036570.KQ",
    "101000.KQ","215000.KQ","950190.KQ","049630.KQ","032980.KQ",
]

GROUPS = {
    "나스닥100": NDX100,
    "다우30":    DOW30,
    "코스피100": KOSPI100,
    "코스닥100": KOSDAQ100,
}

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
    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty: return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw[["Open","High","Low","Close","Volume"]].copy()
    except Exception:
        return pd.DataFrame()

def compute(d):
    d = d.copy()
    c = d["Close"]
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

def dl_batch(tickers, start, end, min_rows=250):
    """yfinance 배치 다운로드 후 종목별 분리"""
    tickers = list(dict.fromkeys(tickers))
    try:
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False, group_by="ticker")
    except Exception:
        raw = pd.DataFrame()
    result = {}
    if raw.empty: return result
    # 단일 티커면 MultiIndex 없음
    if not isinstance(raw.columns, pd.MultiIndex):
        if len(tickers) == 1:
            tk = tickers[0]
            cols = [c for c in ["Open","High","Low","Close","Volume"] if c in raw.columns]
            if len(cols) == 5 and len(raw) >= min_rows:
                result[tk] = raw[cols].copy()
        return result
    for tk in tickers:
        try:
            sub = raw[tk][["Open","High","Low","Close","Volume"]].dropna(how="all")
            if len(sub) >= min_rows:
                result[tk] = sub.copy()
        except Exception:
            continue
    return result

print("\n📥 그룹별 종목 데이터 배치 다운로드...")
group_data = {}
for gname, tickers in GROUPS.items():
    tickers_dedup = list(dict.fromkeys(tickers))
    raw_data = dl_batch(tickers_dedup, START, END)
    stock_data = {}
    for tk, d in raw_data.items():
        stock_data[tk] = compute(d)
    group_data[gname] = stock_data
    print(f"  [{gname}] 요청 {len(tickers_dedup)}개 → 로드 {len(stock_data)}개")


# ─────────────────────────────────────────
# 신호 생성 (벡터화)
# ─────────────────────────────────────────
def build_signals(stock_data):
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
        sig_idx = np.where(cond.values)[0]
        for i in sig_idx:
            if i + 1 >= len(d_c): continue
            ed = d_c.index[i + 1]
            eo = float(d_c["Open"].iloc[i + 1])
            if pd.isna(eo): continue
            rows.append({
                "ticker": tk, "sig_date": d_c.index[i],
                "entry_day": ed, "entry": eo,
                "rsi": float(d_c["RSI"].iloc[i]),
            })
    if not rows: return []
    df_sig = pd.DataFrame(rows).sort_values(["entry_day","rsi"])
    df_sig = df_sig.drop_duplicates(subset=["entry_day","ticker"], keep="first")
    final = []
    for ed, grp in df_sig.groupby("entry_day"):
        final.extend(grp.to_dict("records")[:MAX_DAILY])
    return final


# ─────────────────────────────────────────
# 시뮬레이션
# ─────────────────────────────────────────
def run_simulation(signals, stock_data):
    trades, pos_exit_date = [], {}
    for sig in signals:
        tk, entry_day, entry = sig["ticker"], sig["entry_day"], sig["entry"]
        active = {t: ed for t, ed in pos_exit_date.items() if ed > entry_day}
        if len(active) >= MAX_POSITIONS or tk in active: continue
        d = stock_data[tk]
        future = d.loc[d.index >= entry_day]
        if len(future) == 0: continue
        cb_price  = entry * (1 - CIRCUIT_PCT)
        tgt_price = entry * (1 + TARGET_PCT)
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
# 그룹별 실행
# ─────────────────────────────────────────
print("\n⚙️  그룹별 신호 생성 & 시뮬레이션...")
results = {}
for gname, stock_data in group_data.items():
    signals = build_signals(stock_data)
    df = run_simulation(signals, stock_data)
    r  = calc_stats(df, gname)
    results[gname] = r
    pf_s = f"{r['pf']:.2f}" if not (isinstance(r['pf'], float) and np.isnan(r['pf'])) else "N/A"
    print(f"  [{gname}] 신호 {len(signals):,}건 → 거래 {r['n']}건 | 승률 {r['win_rate']:.1f}% | CAGR {r['cagr']:+.1f}% | PF {pf_s}")


# ─────────────────────────────────────────
# 결과 출력
# ─────────────────────────────────────────
print("\n" + "="*105)
print("  v10 전략 — 4개 시장 그룹 비교 (2010-2026)")
print("  조건: 현재가<MA200 + VIX≥25 + (RSI<40 OR CCI<-100) | 목표+20% | CB-25% | 120일")
print("="*105)
print(f"\n  {'그룹':<12} {'로드종목':>6} {'거래건수':>7} {'승률':>7} {'평균수익':>9} {'기대값':>8} {'승평균':>8} {'패평균':>8} {'PF':>6} {'CAGR':>8} {'연속손':>5} {'평균보유':>7}")
print("  " + "-"*102)
for gname, r in results.items():
    pf_s = f"{r['pf']:.2f}" if not (isinstance(r['pf'], float) and np.isnan(r['pf'])) else " N/A"
    n_stocks = len(group_data[gname])
    print(f"  {gname:<12} {n_stocks:>6} {r['n']:>7} {r['win_rate']:>6.1f}% {r['avg_ret']:>+8.2f}% "
          f"{r['ev']:>+7.2f}% {r['avg_win']:>+7.2f}% {r['avg_loss']:>+7.2f}% "
          f"{pf_s:>6} {r['cagr']:>+7.1f}% {r['max_consec_loss']:>4}건 {r['avg_hold']:>6.0f}일")
print("="*105)

# 연도별
print("\n[연도별 승률 / 평균수익 / 거래건수]")
all_years = set()
for r in results.values():
    if not r["df"].empty:
        all_years.update(r["df"]["entry_date"].dt.year.unique())
all_years = sorted(all_years)
hdr = f"  {'연도':<6}" + "".join(f" {gn:>22}" for gn in results.keys())
print(hdr); print("  " + "-"*96)
for yr in all_years:
    row_str = f"  {yr:<6}"
    for r in results.values():
        if r["df"].empty:
            row_str += f" {'  -':>22}"; continue
        sub = r["df"][r["df"]["entry_date"].dt.year==yr]
        if not len(sub): row_str += f" {'  -':>22}"
        else: row_str += f" {sub['win'].mean()*100:.0f}%/{sub['return_pct'].mean():+.1f}%({len(sub)})"
    print(row_str)

# 청산 유형
print("\n[청산 유형 분포]")
for gname, r in results.items():
    if not r["exit_dist"]: print(f"  {gname:<12}: 거래 없음"); continue
    ec = r["exit_dist"]; tt = sum(ec.values())
    tops = sorted(ec.items(), key=lambda x: -x[1])[:4]
    parts = [f"{k}: {v}건({v/tt*100:.0f}%)" for k, v in tops]
    print(f"  {gname:<12}: {' | '.join(parts)}")

# 복합 점수
print("\n[복합 점수 순위]")
score_data = []
for gname, r in results.items():
    pf_val = r["pf"] if not (isinstance(r["pf"], float) and np.isnan(r["pf"])) else 0
    score  = r["win_rate"]*0.30 + r["cagr"]*0.30 + r["ev"]*0.25 + min(pf_val,5)*0.15*10
    score_data.append((gname, score, r))
score_data.sort(key=lambda x: -x[1])
for rank, (gname, score, r) in enumerate(score_data, 1):
    pf_s = f"{r['pf']:.2f}" if not (isinstance(r['pf'], float) and np.isnan(r['pf'])) else "N/A"
    print(f"  {rank}위 {gname:<12}: 점수 {score:.2f} | 승률 {r['win_rate']:.1f}% | CAGR {r['cagr']:+.1f}% | EV {r['ev']:+.2f}% | PF {pf_s} | 보유 {r['avg_hold']:.0f}일")


# ─────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────
group_names = list(results.keys())
colors_g = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

fig, axes = plt.subplots(2, 4, figsize=(22, 11))
fig.suptitle("v10 전략 — 4개 시장 그룹 비교 (2010-2026)\n"
             "조건: 현재가<MA200 + VIX≥25 + (RSI<40 OR CCI<-100) | 목표+20% | CB-25%",
             fontweight="bold", fontsize=12)
metric_data = [
    ("승률 (%)",          [r["win_rate"] for r in results.values()]),
    ("평균 수익률 (%)",    [r["avg_ret"]  for r in results.values()]),
    ("CAGR (%)",          [r["cagr"]     for r in results.values()]),
    ("기대값 EV (%)",     [r["ev"]       for r in results.values()]),
    ("Profit Factor",     [r["pf"] if not (isinstance(r["pf"],float) and np.isnan(r["pf"])) else 0 for r in results.values()]),
    ("평균 보유 기간 (일)",[r["avg_hold"] for r in results.values()]),
    ("거래 건수",         [r["n"]        for r in results.values()]),
    ("최대 연속 손실 (건)",[r["max_consec_loss"] for r in results.values()]),
]
for ax, (title, vals) in zip(axes.flatten(), metric_data):
    bars = ax.bar(group_names, vals, color=colors_g, edgecolor="white", linewidth=1.2)
    ax.set_title(title, fontweight="bold")
    ax.axhline(0, color="black", lw=0.5)
    ylim = ax.get_ylim()
    rng  = ylim[1] - ylim[0]
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height() + rng*0.02,
                f"{v:.1f}", ha="center", fontsize=9)
    ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("backtest_market_groups.png", dpi=150, bbox_inches="tight")
print("\n📊 backtest_market_groups.png 저장 완료")

# 연도별 수익 꺾은선
fig2, ax2 = plt.subplots(figsize=(14, 6))
ax2.set_title("연도별 평균 수익률 추이 — 4개 시장 그룹", fontweight="bold")
for (gname, r), c, mk in zip(results.items(), colors_g, ["o","s","^","D"]):
    if r["df"].empty: continue
    df_r = r["df"].copy(); df_r["year"] = df_r["entry_date"].dt.year
    yr_avg = df_r.groupby("year")["return_pct"].mean()
    ax2.plot(yr_avg.index, yr_avg.values, marker=mk, color=c,
             label=gname, linewidth=2, markersize=6)
ax2.axhline(0, color="black", lw=0.8, linestyle="--")
ax2.set_xlabel("연도"); ax2.set_ylabel("평균 수익률 (%)")
ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("backtest_market_groups_trend.png", dpi=150, bbox_inches="tight")
print("📊 backtest_market_groups_trend.png 저장 완료")

for gname, r in results.items():
    if not r["df"].empty:
        r["df"].to_csv(f"backtest_{gname}_trades.csv", index=False)
print("📄 CSV 저장 완료\n✅ 완료")
