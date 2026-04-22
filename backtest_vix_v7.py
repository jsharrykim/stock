"""
backtest_vix_v7.py  — 핵심 4가지만 보는 전략
=================================================
지금까지의 모든 복잡한 조건을 버리고 진짜 핵심만:

  1) VIX: 22 이상이고 5일 평균보다 낮음 (진정 중)
  2) RSI: 전일 < 45 였다가 오늘 올라오는 중 (반등 초입)
  3) 거래량: 20일 평균의 1.5배 이상 (자금 유입)
  4) QQQ: 당일 상승 (지수 방향 확인)

손익비: -5% / +10% (1:2) — 현실적인 목표가
BE stop: +5% 달성 시 원가로 이동
Max hold: 15일

추가 안전장치 1개:
- SPY가 200일 이평 위 (시장 구조 확인)
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

STOP_PCT     = 0.05    # -5%
TARGET_PCT   = 0.10    # +10% (손익비 1:2)
BE_TRIGGER   = 0.05    # +5% 달성 시 stop → entry
MAX_HOLD     = 15      # 거래일
VIX_MIN      = 22
VOL_MULT     = 1.5     # 거래량 1.5배
MAX_DAILY    = 10      # 하루 최대 진입 종목

TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","TMUS","AMD","ADBE","PEP","CSCO","QCOM","INTC","TXN","AMGN",
    "INTU","AMAT","MU","LRCX","KLAC","CDNS","SNPS","MRVL","FTNT","PANW",
    "CRWD","DDOG","ZS","TEAM","MNST","KDP","MDLZ","ORLY","AZN",
    "ISRG","IDXX","REGN","VRTX","GILD","BIIB","ILMN","MRNA",
    "ODFL","FAST","PCAR","CTSH","PAYX","VRSK","WDAY",
    "MELI","PDD","JD","BIDU","NTES",
    "DLTR","SBUX","ROST","LULU","EBAY","MAR","CTAS","EA",
    "CHTR","CMCSA","EXC","XEL","AEP","FANG","MKL",
    "PLTR","CPRT","NXPI","MPWR","ENPH","SEDG","ON","COIN",
    "DOCU","ZM","OKTA","PTON",
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


def compute_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ──────────────────────────────────────────────
# 시장 데이터
# ──────────────────────────────────────────────
print("📥 시장 데이터 다운로드 중...")

vix_raw = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)
_vc = vix_raw["Close"]
if isinstance(_vc, pd.DataFrame): _vc = _vc.iloc[:, 0]
vix = pd.Series(_vc.values, index=_vc.index, dtype=float).dropna()
vix_ma5 = vix.rolling(5).mean()

qqq = dl_ohlcv("QQQ", START, END)
qqq_close = pd.Series(qqq["Close"].values, index=qqq.index, dtype=float)
qqq_open  = pd.Series(qqq["Open"].values,  index=qqq.index, dtype=float)

spy = dl_ohlcv("SPY", START, END)
spy_close = pd.Series(spy["Close"].values, index=spy.index, dtype=float)
spy_ma200 = spy_close.rolling(200).mean()

print(f"✅ 시장 데이터 로드 완료")

# ──────────────────────────────────────────────
# 종목 데이터
# ──────────────────────────────────────────────
print("📥 종목 데이터 다운로드 중...")
stock_data = {}
for i, tk in enumerate(TICKERS, 1):
    d = dl_ohlcv(tk, START, END)
    if len(d) < 250:
        continue
    d["RSI"]      = compute_rsi(d["Close"])
    d["VolAvg20"] = d["Volume"].rolling(20).mean()
    stock_data[tk] = d
    if i % 20 == 0:
        print(f"  {i}/{len(TICKERS)} 완료")

print(f"✅ {len(stock_data)}개 종목 로드 완료")

# ──────────────────────────────────────────────
# 신호 생성 — 핵심 4가지
# ──────────────────────────────────────────────
print("🔍 신호 생성 중...")
signals_by_date = {}

for tk, d in stock_data.items():
    d = d.dropna(subset=["RSI","VolAvg20"])
    close    = d["Close"]
    rsi      = d["RSI"]
    rsi_prev = rsi.shift(1)
    vol      = d["Volume"]
    vol_avg  = d["VolAvg20"]

    common = d.index.intersection(vix.index)
    if len(common) < 50:
        continue

    vx      = vix.reindex(common)
    vx_ma5  = vix_ma5.reindex(common)
    qqq_c   = qqq_close.reindex(common)
    qqq_o   = qqq_open.reindex(common)
    spy_c   = spy_close.reindex(common)
    spy_m200= spy_ma200.reindex(common)

    # ── 핵심 4가지 ──────────────────────────────
    cond1 = (vx >= VIX_MIN) & (vx < vx_ma5)          # VIX 진정 중
    cond2 = (rsi_prev < 45) & (rsi > rsi_prev)        # RSI 반등 초입
    cond3 = vol >= VOL_MULT * vol_avg                  # 거래량 터짐
    cond4 = qqq_c > qqq_o                             # QQQ 당일 상승

    # ── 안전장치 1개 ─────────────────────────────
    safe  = spy_c > spy_m200                          # SPY MA200 위

    sig = cond1 & cond2 & cond3 & cond4 & safe
    sig_dates = d.index[sig.fillna(False)]

    for dt in sig_dates:
        if pd.isna(rsi.loc[dt]) or pd.isna(rsi_prev.loc[dt]):
            continue
        rsi_up = float(rsi.loc[dt] - rsi_prev.loc[dt])
        row = d.loc[dt]
        if dt not in signals_by_date:
            signals_by_date[dt] = []
        signals_by_date[dt].append((tk, rsi_up, float(row["Close"])))

# RSI 상승폭 큰 순 → 상위 MAX_DAILY
final_signals = []
for dt, items in sorted(signals_by_date.items()):
    items.sort(key=lambda x: -x[1])
    for tk, rsi_up, entry in items[:MAX_DAILY]:
        final_signals.append({"date": dt, "ticker": tk, "entry": entry})

print(f"✅ 원시 신호: {len(final_signals)}건 ({len(signals_by_date)}거래일)")

# ──────────────────────────────────────────────
# 트레이드 시뮬레이션
# ──────────────────────────────────────────────
print("⚙️  시뮬레이션 중...")
trades = []
open_pos = set()

for sig in final_signals:
    tk, dt, entry = sig["ticker"], sig["date"], sig["entry"]
    if tk in open_pos:
        continue

    d = stock_data[tk]
    future = d.loc[d.index > dt]
    if len(future) == 0:
        continue

    stop   = entry * (1 - STOP_PCT)
    target = entry * (1 + TARGET_PCT)
    be_applied = False
    max_ret = 0.0
    exit_date = exit_price = exit_reason = None

    for i, (fdt, row) in enumerate(future.iterrows()):
        lo, hi, cl = float(row["Low"]), float(row["High"]), float(row["Close"])
        mr = (hi - entry) / entry
        if mr > max_ret:
            max_ret = mr

        # BE stop
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
    trades.append({
        "entry_date" : dt,
        "exit_date"  : exit_date,
        "ticker"     : tk,
        "entry"      : entry,
        "exit"       : exit_price,
        "return_pct" : ret * 100,
        "hold_days"  : (exit_date - dt).days,
        "exit_reason": exit_reason,
        "win"        : ret > 0,
    })

print(f"✅ 완료: {len(trades)}건")

# ──────────────────────────────────────────────
# 분석
# ──────────────────────────────────────────────
df = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)

wins   = df[df["win"]]
losses = df[~df["win"]]

n          = len(df)
win_rate   = wins.__len__() / n * 100
avg_ret    = df["return_pct"].mean()
avg_win    = wins["return_pct"].mean()   if len(wins)   else 0
avg_loss   = losses["return_pct"].mean() if len(losses) else 0
pf         = (wins["return_pct"].sum() / -losses["return_pct"].sum()
              if len(losses) and losses["return_pct"].sum() < 0 else np.nan)
ev         = win_rate/100 * avg_win + (1-win_rate/100) * avg_loss

max_cl = cur = 0
for r in df["return_pct"]:
    cur = cur + 1 if r < 0 else 0
    max_cl = max(max_cl, cur)

exit_cnt = df["exit_reason"].value_counts()

print("\n" + "="*60)
print("    v7 핵심 4조건 전략 — 백테스트 결과")
print("="*60)
print(f"  기간           : {START} ~ {END}")
print(f"  총 트레이드    : {n}건")
print(f"  활성 연도 수   : {df['entry_date'].dt.year.nunique()}개년")
print(f"  승 률          : {win_rate:.1f}%")
print(f"  평균 수익률    : {avg_ret:+.2f}%  (트레이드 1건당)")
print(f"  기대값         : {ev:+.2f}%")
print(f"  승자 평균      : {avg_win:+.2f}%")
print(f"  패자 평균      : {avg_loss:+.2f}%")
print(f"  Profit Factor  : {pf:.2f}")
print(f"  최대 연속 손실 : {max_cl}건")
print(f"  청산 유형:")
for r, c in exit_cnt.items():
    print(f"    {r:8s}: {c:4d}건 ({c/n*100:.1f}%)")
print("="*60)

# 연도별
df["year"] = df["entry_date"].dt.year
yearly = df.groupby("year").agg(
    trades=("return_pct","count"),
    avg_ret=("return_pct","mean"),
    win_rate=("win", lambda x: x.mean()*100),
    total_ret=("return_pct","sum"),
)
print("\n연도별 성과:")
print(yearly.round(2).to_string())

print(f"\n▶ 수익 상위 10:")
print(df.nlargest(10,"return_pct")[
    ["entry_date","ticker","entry","exit","return_pct","exit_reason","hold_days"]
].to_string(index=False))

print(f"\n▶ 손실 하위 10:")
print(df.nsmallest(10,"return_pct")[
    ["entry_date","ticker","entry","exit","return_pct","exit_reason","hold_days"]
].to_string(index=False))

# ──────────────────────────────────────────────
# 포트폴리오 관점 수익 계산
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("    포트폴리오 관점 수익 시뮬레이션")
print("="*60)

# 전체 자본 동시 균등 배분 방식
capital = 1.0
yr_capital = {}
cur_year = None

for entry_date, group in df.groupby("entry_date"):
    n_pos = len(group)
    weight = 1.0 / n_pos
    batch = (group["return_pct"] / 100 * weight).sum()
    capital *= (1 + batch)
    yr = entry_date.year
    if yr not in yr_capital:
        yr_capital[yr] = []
    yr_capital[yr].append(capital)

years = (df["exit_date"].max() - df["entry_date"].min()).days / 365.25
cagr  = capital ** (1 / years) - 1

print(f"  전략 활성 기간 : {df['entry_date'].min().date()} ~ {df['exit_date'].max().date()}")
print(f"  누적 수익률    : {(capital-1)*100:+.1f}%")
print(f"  CAGR           : {cagr*100:+.2f}%")
print(f"  1억 → {capital:.2f}억 (활성 기간 내)")
print()

# 1억 N년 시뮬레이션 (CAGR 복리)
print("  1억 복리 시뮬레이션 (CAGR 기준):")
init = 1.0
for yr in [1, 3, 5, 10, 15]:
    val = init * (1 + cagr) ** yr
    print(f"    {yr:2d}년 후: {val:.2f}억  ({(val-init)*100:+.0f}%)")

print()
print(f"  ※ 이 전략은 '신호가 없을 때는 현금'입니다.")
print(f"     현금 기간에 예금(3.5%)을 병행할 경우:")
active_days = (df["exit_date"].max() - df["entry_date"].min()).days
total_days  = (pd.Timestamp(END) - pd.Timestamp(START)).days
cash_ratio  = 1 - (active_days / total_days)
blended_cagr = cagr * (1-cash_ratio) + 0.035 * cash_ratio
print(f"     혼합 CAGR: {blended_cagr*100:+.2f}%")
print()

# 과거 버전들과 비교
print("="*72)
print("                전략 버전별 최종 비교")
print("="*72)
versions = [
    ("v4", 214, 40.7, 1.03, 1.47, "+3.0%"),
    ("v5",  83, 32.5, 0.34, 1.17, "+0.9%"),
    ("v6",  38, 47.4, 2.19, 2.26, "+3.0%"),
    ("v7",   n, round(win_rate,1), round(avg_ret,2),
              round(pf,2) if not np.isnan(pf) else 0,
              f"{cagr*100:+.1f}%"),
]
print(f"  {'버전':>4} {'건수':>6} {'승률':>7} {'건당수익':>9} {'PF':>6}  {'포트CAGR':>10}")
print(f"  {'-'*55}")
for v in versions:
    nm, n_, wr, ar, pf_, cagr_ = v
    mark = " ◀" if nm == "v7" else ""
    print(f"  {nm:>4} {n_:>6} {wr:>6.1f}% {ar:>+8.2f}% {pf_:>6.2f}  {cagr_:>10}{mark}")
print("="*72)

# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("v7 핵심 4조건 전략 — 백테스트 결과", fontsize=14, fontweight="bold")

ax = axes[0, 0]
ax.hist(df["return_pct"], bins=30, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(0,      color="red",    linestyle="--", linewidth=1.5)
ax.axvline(avg_ret,color="orange", linestyle="-",  linewidth=1.5, label=f"평균 {avg_ret:+.2f}%")
ax.set_title("수익률 분포 (트레이드별)")
ax.set_xlabel("Return (%)")
ax.legend()

ax = axes[0, 1]
clr = {"stop":"#e74c3c","target":"#2ecc71","time":"#f39c12"}
ax.pie(exit_cnt.values,
       labels=[f"{k}\n{v}건" for k,v in exit_cnt.items()],
       colors=[clr.get(k,"#aaa") for k in exit_cnt.index],
       autopct="%1.1f%%", startangle=140)
ax.set_title("청산 유형")

ax = axes[1, 0]
cum = (1 + df["return_pct"]/100).cumprod() - 1
ax.plot(range(len(cum)), cum*100, color="navy", linewidth=1.5)
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.fill_between(range(len(cum)), cum*100, 0, where=(cum>=0), alpha=0.2, color="green")
ax.fill_between(range(len(cum)), cum*100, 0, where=(cum<0),  alpha=0.2, color="red")
ax.set_title(f"누적 수익률 곡선  (CAGR {cagr*100:+.1f}%)")
ax.set_xlabel("Trade #")
ax.set_ylabel("Cumulative %")

ax = axes[1, 1]
bar_c = ["#2ecc71" if v>=0 else "#e74c3c" for v in yearly["avg_ret"]]
ax.bar(yearly.index.astype(str), yearly["avg_ret"], color=bar_c, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8)
# 건수를 bar 위에 표시
for x, (yr, row) in enumerate(yearly.iterrows()):
    ax.text(x, row["avg_ret"] + (0.3 if row["avg_ret"]>=0 else -0.8),
            f'{int(row["trades"])}건', ha='center', fontsize=8)
ax.set_title("연도별 평균 수익률 (건수 표시)")
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("backtest_v7_results.png", dpi=150, bbox_inches="tight")
df.to_csv("backtest_v7_trades.csv", index=False)
print("\n📊 backtest_v7_results.png / backtest_v7_trades.csv 저장 완료")
print("✅ v7 백테스트 완료")
