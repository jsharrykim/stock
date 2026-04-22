"""
backtest_vix_v8.py  — 데이터 기반 최소 조건 전략
=================================================
88,371건 분석에서 도출된 핵심 발견:
  1. MA200 아래 = 기본 성공률 70.7%
  2. VIX 25+ = 성공률 77.5% (가장 강력한 구분 지표, t=31.0)
  3. RSI < 40 = 성공률 72.8% (낮을수록 좋음)
  4. 나머지 조건 추가 시 오히려 성공률 하락 (65~67%)

전략 설계 원칙:
  - 진입 조건: 딱 3개 (MA200 아래 + VIX 25+ + RSI < 40)
  - 청산: ATR 기반 동적 손익 (고정이 아닌 종목별 변동성 반영)
  - 포지션 관리: 동일 종목 중복 진입 없음, 하루 최대 5종목

손익 구조 (ATR 기반):
  Stop  = entry - ATR × 1.5  (최소 -4%, 최대 -10%)
  Target= entry + ATR × 3.0  (최소 +8%, 최대 +25%)
  → 손익비 항상 1:2 유지, 종목 변동성에 맞춤
  BE stop: +5% 달성 시 원가로
  Max hold: 25거래일
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
START = "2010-01-01"
END   = "2026-01-01"

VIX_MIN      = 25      # VIX 하한 (데이터: 25+ 구간 성공률 77.5%)
RSI_MAX      = 40      # RSI 상한 (데이터: RSI < 40 구간 성공률 72.8%)
ATR_STOP_M   = 1.5     # stop = entry - ATR × 1.5
ATR_TGT_M    = 3.0     # target = entry + ATR × 3.0  (1:2 손익비)
MIN_STOP     = 0.04    # 최소 손절 -4%
MAX_STOP     = 0.10    # 최대 손절 -10%
MIN_TGT      = 0.08    # 최소 목표 +8%
MAX_TGT      = 0.25    # 최대 목표 +25%
BE_TRIGGER   = 0.05    # +5% 달성 시 BE stop
MAX_HOLD     = 25      # 거래일
MAX_DAILY    = 5       # 하루 최대 종목 (ATR/close 높은 순 — 변동성 클수록 수익 가능성 높음)

TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","TMUS","AMD","ADBE","PEP","CSCO","QCOM","INTC","TXN","AMGN",
    "INTU","AMAT","MU","LRCX","KLAC","CDNS","SNPS","MRVL","FTNT","PANW",
    "CRWD","DDOG","ZS","TEAM","MNST","KDP","MDLZ","ORLY","AZN",
    "ISRG","IDXX","REGN","VRTX","GILD","BIIB","ILMN","MRNA",
    "ODFL","FAST","PCAR","CTSH","PAYX","VRSK","WDAY","MELI","PDD",
    "DLTR","SBUX","ROST","LULU","EBAY","MAR","CTAS","EA",
    "CHTR","CMCSA","EXC","XEL","AEP","MKL",
    "PLTR","CPRT","NXPI","MPWR","ENPH","SEDG","ON","COIN",
    "DOCU","ZM","OKTA","PTON","JD","BIDU","NTES","FANG",
]
TICKERS = sorted(set(TICKERS))

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


def compute_indicators(d):
    d = d.copy()
    c = d["Close"]
    d["MA200"] = c.rolling(200).mean()
    # RSI
    delta = c.diff()
    g = delta.clip(lower=0).rolling(14).mean()
    l = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100 / (1 + g / l.replace(0, np.nan))
    # ATR
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - c.shift(1)).abs(),
        (d["Low"]  - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean()
    d["ATR_PCT"] = d["ATR"] / c
    return d


# ──────────────────────────────────────────────
# 시장 데이터
# ──────────────────────────────────────────────
print("📥 시장 데이터 다운로드 중...")
vix_raw = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)
_vc = vix_raw["Close"]
if isinstance(_vc, pd.DataFrame): _vc = _vc.iloc[:, 0]
vix = pd.Series(_vc.values, index=_vc.index, dtype=float).dropna()
print(f"✅ VIX {len(vix)}일")

# ──────────────────────────────────────────────
# 종목 데이터
# ──────────────────────────────────────────────
print("📥 종목 데이터 다운로드 중...")
stock_data = {}
for i, tk in enumerate(TICKERS, 1):
    d = dl_ohlcv(tk, START, END)
    if len(d) < 250:
        continue
    d = compute_indicators(d)
    stock_data[tk] = d
    if i % 20 == 0:
        print(f"  {i}/{len(TICKERS)} 완료")
print(f"✅ {len(stock_data)}개 종목 로드")

# ──────────────────────────────────────────────
# 신호 생성 — 핵심 3조건
# ──────────────────────────────────────────────
print("🔍 신호 생성 중...")
signals_by_date = {}

for tk, d in stock_data.items():
    d_clean = d.dropna(subset=["MA200", "RSI", "ATR"])
    close  = d_clean["Close"]
    ma200  = d_clean["MA200"]
    rsi    = d_clean["RSI"]
    atr_pct= d_clean["ATR_PCT"]

    common = d_clean.index.intersection(vix.index)
    if len(common) < 50:
        continue

    vx = vix.reindex(common)

    # ── 핵심 3조건 ─────────────────────────────
    cond1 = close < ma200          # MA200 아래
    cond2 = vx >= VIX_MIN          # VIX 25 이상
    cond3 = rsi < RSI_MAX          # RSI 40 미만

    sig = cond1 & cond2 & cond3
    sig_dates = d_clean.index[sig.reindex(d_clean.index).fillna(False)]

    for dt in sig_dates:
        row = d_clean.loc[dt]
        if pd.isna(row["ATR"]):
            continue
        if dt not in signals_by_date:
            signals_by_date[dt] = []
        signals_by_date[dt].append({
            "ticker" : tk,
            "close"  : float(row["Close"]),
            "atr_pct": float(row["ATR_PCT"]),
            "atr"    : float(row["ATR"]),
            "rsi"    : float(row["RSI"]),
        })

# 날짜별: ATR/close 높은 순 → 상위 MAX_DAILY (변동성 클수록 수익 폭 큼)
final_signals = []
for dt, items in sorted(signals_by_date.items()):
    items.sort(key=lambda x: -x["atr_pct"])
    for item in items[:MAX_DAILY]:
        final_signals.append({"date": dt, **item})

print(f"✅ 원시 신호: {len(final_signals)}건 ({len(signals_by_date)}거래일)")

# ──────────────────────────────────────────────
# 트레이드 시뮬레이션
# ──────────────────────────────────────────────
print("⚙️  시뮬레이션 중...")
trades = []
open_pos = set()

for sig in final_signals:
    tk, dt, entry, atr = sig["ticker"], sig["date"], sig["close"], sig["atr"]

    if tk in open_pos:
        continue

    d = stock_data[tk]
    future = d.loc[d.index > dt]
    if len(future) == 0:
        continue

    # ATR 기반 손익 계산
    raw_stop_dist = atr * ATR_STOP_M
    raw_tgt_dist  = atr * ATR_TGT_M

    # 최소/최대 클램프
    stop_dist = np.clip(raw_stop_dist, entry * MIN_STOP, entry * MAX_STOP)
    tgt_dist  = np.clip(raw_tgt_dist,  entry * MIN_TGT,  entry * MAX_TGT)

    stop   = entry - stop_dist
    target = entry + tgt_dist

    be_applied = False
    max_ret    = 0.0
    exit_date  = exit_price = exit_reason = None

    for i, (fdt, row) in enumerate(future.iterrows()):
        lo, hi, cl = float(row["Low"]), float(row["High"]), float(row["Close"])
        mr = (hi - entry) / entry
        if mr > max_ret:
            max_ret = mr

        if max_ret >= BE_TRIGGER and not be_applied:
            stop = entry
            be_applied = True

        if lo <= stop:
            exit_price, exit_reason, exit_date = stop, "stop", fdt
            break
        if hi >= target:
            exit_price, exit_reason, exit_date = target, "target", fdt
            break
        if i + 1 >= MAX_HOLD:
            exit_price, exit_reason, exit_date = cl, "time", fdt
            break

    if exit_price is None:
        continue

    ret = (exit_price - entry) / entry
    stop_pct  = stop_dist / entry
    tgt_pct   = tgt_dist  / entry

    trades.append({
        "entry_date" : dt,
        "exit_date"  : exit_date,
        "ticker"     : tk,
        "entry"      : entry,
        "exit"       : exit_price,
        "stop_pct"   : -stop_pct * 100,
        "target_pct" : tgt_pct  * 100,
        "return_pct" : ret * 100,
        "hold_days"  : (exit_date - dt).days,
        "exit_reason": exit_reason,
        "win"        : ret > 0,
        "rsi_entry"  : sig["rsi"],
        "atr_pct"    : sig["atr_pct"] * 100,
    })
    open_pos.add(tk)

print(f"✅ 완료: {len(trades)}건")

# ──────────────────────────────────────────────
# 분석
# ──────────────────────────────────────────────
df = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)

wins   = df[df["win"]]
losses = df[~df["win"]]
n      = len(df)

win_rate = len(wins) / n * 100
avg_ret  = df["return_pct"].mean()
avg_win  = wins["return_pct"].mean()  if len(wins)   else 0
avg_loss = losses["return_pct"].mean() if len(losses) else 0
pf_raw   = wins["return_pct"].sum() / -losses["return_pct"].sum() \
           if len(losses) and losses["return_pct"].sum() < 0 else np.nan
ev       = win_rate/100 * avg_win + (1-win_rate/100) * avg_loss

max_cl = cur = 0
for r in df["return_pct"]:
    cur = cur + 1 if r < 0 else 0
    max_cl = max(max_cl, cur)

exit_cnt = df["exit_reason"].value_counts()

print("\n" + "="*62)
print("    v8 데이터 기반 최소 조건 전략 — 백테스트 결과")
print("="*62)
print(f"  기간           : {START} ~ {END}")
print(f"  유니버스       : {len(stock_data)}개 종목")
print(f"  진입 조건      : MA200↓ + VIX≥25 + RSI<40")
print(f"  총 트레이드    : {n}건")
print(f"  활성 연도      : {df['entry_date'].dt.year.nunique()}개년")
print(f"  승 률          : {win_rate:.1f}%")
print(f"  평균 수익률    : {avg_ret:+.2f}%")
print(f"  기대값         : {ev:+.2f}%")
print(f"  승자 평균      : {avg_win:+.2f}%")
print(f"  패자 평균      : {avg_loss:+.2f}%")
print(f"  Profit Factor  : {pf_raw:.2f}")
print(f"  최대 연속 손실 : {max_cl}건")
print(f"  평균 손절 설정 : {df['stop_pct'].mean():.2f}%")
print(f"  평균 목표 설정 : {df['target_pct'].mean():.2f}%")
print(f"  청산 유형:")
for r, c in exit_cnt.items():
    print(f"    {r:8s}: {c:4d}건 ({c/n*100:.1f}%)")
print("="*62)

df["year"] = df["entry_date"].dt.year
yearly = df.groupby("year").agg(
    trades=("return_pct","count"),
    avg_ret=("return_pct","mean"),
    win_rate=("win", lambda x: x.mean()*100),
    total_ret=("return_pct","sum"),
    avg_stop=("stop_pct","mean"),
    avg_tgt=("target_pct","mean"),
)
print("\n연도별 성과:")
print(yearly.round(2).to_string())

print(f"\n▶ 수익 상위 10:")
print(df.nlargest(10,"return_pct")[
    ["entry_date","ticker","entry","exit","return_pct","stop_pct","target_pct","exit_reason","hold_days"]
].to_string(index=False))
print(f"\n▶ 손실 하위 10:")
print(df.nsmallest(10,"return_pct")[
    ["entry_date","ticker","entry","exit","return_pct","stop_pct","target_pct","exit_reason","hold_days"]
].to_string(index=False))

# ──────────────────────────────────────────────
# 포트폴리오 복리 시뮬레이션
# ──────────────────────────────────────────────
print("\n" + "="*62)
print("    포트폴리오 복리 시뮬레이션")
print("="*62)

capital = 1.0
monthly_cap = {}

for entry_date, group in df.groupby("entry_date"):
    n_pos = len(group)
    weight = 1.0 / n_pos
    batch = (group["return_pct"] / 100 * weight).sum()
    capital *= (1 + batch)
    ym = entry_date.strftime("%Y-%m")
    monthly_cap[ym] = capital

years = (df["exit_date"].max() - df["entry_date"].min()).days / 365.25
cagr  = capital ** (1 / years) - 1

print(f"  누적 수익률 : {(capital-1)*100:+.1f}%")
print(f"  CAGR        : {cagr*100:+.2f}%")
print(f"  1억 투자 시:")
for yr in [1, 3, 5, 10, 15, 20]:
    val = 1.0 * (1 + cagr) ** yr
    print(f"    {yr:2d}년 후 → {val:.2f}억  ({(val-1)*100:+.0f}%)")

# 버전 비교
print("\n" + "="*72)
print("                 전략 버전별 최종 비교")
print("="*72)
history = [
    ("v1", 417,  26.0,  0.00, 0.98, "+3.0%"),
    ("v2",  95,  30.5, -0.41, 0.87, "+0.9%"),
    ("v3",  64,  29.7, -0.14, 0.95, "-"),
    ("v4", 214,  40.7,  1.03, 1.47, "+3.0%"),
    ("v6",  38,  47.4,  2.19, 2.26, "+3.0%"),
    ("v7", 161,  37.3,  1.03, 1.54, "+5.7%"),
    ("v8",   n, round(win_rate,1), round(avg_ret,2),
              round(pf_raw,2) if not np.isnan(pf_raw) else 0,
              f"{cagr*100:+.1f}%"),
]
print(f"  {'버전':>4} {'건수':>6} {'승률':>7} {'건당수익':>9} {'PF':>6}  {'포트CAGR':>10}  {'비고'}")
print("  " + "-"*65)
for row in history:
    nm, n_, wr, ar, pf_, cagr_ = row
    mark = " ◀ 데이터 기반" if nm == "v8" else ""
    print(f"  {nm:>4} {n_:>6} {wr:>6.1f}% {ar:>+8.2f}% {pf_:>6.2f}  {cagr_:>10}{mark}")
print("="*72)

# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    f"v8 데이터 기반 최소 조건 전략 (MA200↓ + VIX≥25 + RSI<40)\n"
    f"총 {n}건 | 승률 {win_rate:.1f}% | 평균 {avg_ret:+.2f}% | PF {pf_raw:.2f} | CAGR {cagr*100:+.1f}%",
    fontsize=12, fontweight="bold"
)

# 1. 수익률 분포
ax = axes[0, 0]
ax.hist(df["return_pct"], bins=40, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(0,       color="red",    linestyle="--", lw=1.5)
ax.axvline(avg_ret, color="orange", linestyle="-",  lw=1.5, label=f"평균 {avg_ret:+.2f}%")
ax.set_title("수익률 분포")
ax.set_xlabel("Return (%)")
ax.legend()

# 2. 청산 유형
ax = axes[0, 1]
clr = {"stop":"#e74c3c","target":"#2ecc71","time":"#f39c12"}
ax.pie(exit_cnt.values,
       labels=[f"{k}\n{v}건\n({v/n*100:.1f}%)" for k,v in exit_cnt.items()],
       colors=[clr.get(k,"#aaa") for k in exit_cnt.index],
       startangle=140)
ax.set_title("청산 유형")

# 3. 누적 수익률 곡선
ax = axes[0, 2]
cum = (1 + df["return_pct"]/100).cumprod() - 1
ax.plot(range(len(cum)), cum*100, color="navy", lw=1.5)
ax.axhline(0, color="red", linestyle="--", lw=1)
ax.fill_between(range(len(cum)), cum*100, 0, where=(cum>=0), alpha=0.2, color="green")
ax.fill_between(range(len(cum)), cum*100, 0, where=(cum<0),  alpha=0.2, color="red")
ax.set_title(f"누적 수익률 (CAGR {cagr*100:+.1f}%)")
ax.set_xlabel("Trade #")
ax.set_ylabel("Cumulative %")

# 4. 연도별 평균 수익률
ax = axes[1, 0]
bar_c = ["#2ecc71" if v>=0 else "#e74c3c" for v in yearly["avg_ret"]]
bars = ax.bar(yearly.index.astype(str), yearly["avg_ret"], color=bar_c, edgecolor="white")
ax.axhline(0, color="black", lw=0.8)
for x, (yr, row) in enumerate(yearly.iterrows()):
    ax.text(x, row["avg_ret"] + (0.2 if row["avg_ret"]>=0 else -0.5),
            f'{int(row["trades"])}건', ha='center', fontsize=8)
ax.set_title("연도별 평균 수익률 (건수)")
ax.tick_params(axis="x", rotation=45)

# 5. 연도별 승률
ax = axes[1, 1]
bar_c2 = ["#2ecc71" if v>=50 else ("#f39c12" if v>=40 else "#e74c3c") for v in yearly["win_rate"]]
ax.bar(yearly.index.astype(str), yearly["win_rate"], color=bar_c2, edgecolor="white")
ax.axhline(50, color="green", linestyle="--", lw=1, label="50%")
ax.axhline(win_rate, color="navy", linestyle=":", lw=1.5, label=f"전체 {win_rate:.1f}%")
ax.set_title("연도별 승률")
ax.tick_params(axis="x", rotation=45)
ax.legend(fontsize=8)

# 6. RSI 진입값 vs 수익률
ax = axes[1, 2]
sc = ax.scatter(df["rsi_entry"], df["return_pct"],
                c=df["return_pct"], cmap="RdYlGn",
                alpha=0.4, s=15, vmin=-15, vmax=15)
ax.axhline(0, color="black", lw=0.8, linestyle="--")
ax.axvline(30, color="blue", lw=1, linestyle=":", label="RSI=30")
ax.set_title("진입 RSI vs 수익률")
ax.set_xlabel("RSI at Entry")
ax.set_ylabel("Return (%)")
ax.legend(fontsize=8)
plt.colorbar(sc, ax=ax, label="Return %")

plt.tight_layout()
plt.savefig("backtest_v8_results.png", dpi=150, bbox_inches="tight")
df.to_csv("backtest_v8_trades.csv", index=False)
print("\n📊 backtest_v8_results.png 저장 완료")
print("📄 backtest_v8_trades.csv 저장 완료")
print("✅ v8 백테스트 완료")
