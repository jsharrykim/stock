"""
backtest_vix_v9.py  — "stop 없이 기다리되 서킷브레이커 + 자본 회전 최적화" 전략
===================================================================================
데이터 기반 설계:
  진입: MA200↓ + VIX≥25 + RSI<40
  
  청산 규칙 (우선순위 순):
    1. +20% 달성 시 즉시 매도
    2. -25% 낙폭 시 서킷브레이커 손절 (최악 케이스 차단)
    3. 60일 경과 & 수익 플러스면 절반 청산 + 나머지 +20% 재도전 (최대 120일)
    4. 120일 time exit (종가)

  포지션: 동시 최대 5종목, 종목당 자본 20%
  우선순위: RSI 낮은 순 (더 과매도된 종목 우선)
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

TARGET_PCT     = 0.20    # +20% 목표
CIRCUIT_PCT    = 0.25    # -25% 서킷브레이커
HALF_EXIT_DAYS = 60      # 60일 경과 & 수익 시 절반 청산
MAX_HOLD       = 120     # 최대 보유 거래일
VIX_MIN        = 25
RSI_MAX        = 40
MAX_POSITIONS  = 5       # 동시 최대 포지션
MAX_DAILY      = 5       # 하루 최대 신호

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


def compute(d):
    d = d.copy()
    c = d["Close"]
    d["MA200"] = c.rolling(200).mean()
    delta = c.diff()
    g = delta.clip(lower=0).rolling(14).mean()
    l = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100 / (1 + g / l.replace(0, np.nan))
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
    d = compute(d)
    stock_data[tk] = d
    if i % 20 == 0:
        print(f"  {i}/{len(TICKERS)} 완료")
print(f"✅ {len(stock_data)}개 종목 로드")

# ──────────────────────────────────────────────
# 신호 생성
# ──────────────────────────────────────────────
print("🔍 신호 생성 중...")
signals_by_date = {}

for tk, d in stock_data.items():
    d_c = d.dropna(subset=["MA200", "RSI"])
    close = d_c["Close"]
    ma200 = d_c["MA200"]
    rsi   = d_c["RSI"]

    common = d_c.index.intersection(vix.index)
    if len(common) < 50:
        continue
    vx = vix.reindex(common)

    cond1 = close < ma200
    cond2 = vx >= VIX_MIN
    cond3 = rsi < RSI_MAX

    sig_dates = d_c.index[(cond1 & cond2 & cond3).reindex(d_c.index).fillna(False)]

    for dt in sig_dates:
        row = d_c.loc[dt]
        if pd.isna(row["RSI"]):
            continue
        if dt not in signals_by_date:
            signals_by_date[dt] = []
        signals_by_date[dt].append({
            "ticker": tk,
            "close" : float(row["Close"]),
            "rsi"   : float(row["RSI"]),
        })

# RSI 낮은 순 정렬 → 상위 MAX_DAILY
final_signals = []
for dt, items in sorted(signals_by_date.items()):
    items.sort(key=lambda x: x["rsi"])   # RSI 낮을수록 우선
    for item in items[:MAX_DAILY]:
        final_signals.append({"date": dt, **item})

print(f"✅ 원시 신호: {len(final_signals):,}건 ({len(signals_by_date)}거래일)")

# ──────────────────────────────────────────────
# 트레이드 시뮬레이션
# ──────────────────────────────────────────────
print("⚙️  시뮬레이션 중...")

trades   = []
# ticker → 포지션 청산 날짜 (이 날짜 이후부터 재진입 가능)
pos_exit_date = {}

for sig in final_signals:
    tk, dt, entry = sig["ticker"], sig["date"], sig["close"]

    # 현재 열린 포지션 수 (아직 청산 안 된 것)
    active = {t: ed for t, ed in pos_exit_date.items() if ed > dt}
    if len(active) >= MAX_POSITIONS:
        continue
    # 동일 종목 포지션이 아직 열려있으면 스킵
    if tk in active:
        continue

    d = stock_data[tk]
    future = d.loc[d.index > dt]
    if len(future) == 0:
        continue

    circuit = entry * (1 - CIRCUIT_PCT)
    target  = entry * (1 + TARGET_PCT)

    half_exited  = False
    exit_records = []

    for i, (fdt, row) in enumerate(future.iterrows()):
        lo  = float(row["Low"])
        hi  = float(row["High"])
        cl  = float(row["Close"])
        ret = (cl - entry) / entry

        if hi >= target:
            pct = 0.5 if half_exited else 1.0
            exit_records.append((fdt, target, pct, "target"))
            break

        if lo <= circuit:
            pct = 0.5 if half_exited else 1.0
            exit_records.append((fdt, circuit, pct, "circuit"))
            break

        if i + 1 == HALF_EXIT_DAYS and not half_exited and ret > 0:
            exit_records.append((fdt, cl, 0.5, "half_60d"))
            half_exited = True
            continue

        if i + 1 >= MAX_HOLD:
            pct = 0.5 if half_exited else 1.0
            exit_records.append((fdt, cl, pct, "time"))
            break

    if not exit_records:
        continue

    total_pct   = sum(r[2] for r in exit_records)
    weighted    = sum((r[1] - entry) / entry * r[2] for r in exit_records)
    blended_ret = weighted / total_pct if total_pct > 0 else 0

    last_exit   = exit_records[-1]
    main_reason = "+".join(r[3] for r in exit_records) if len(exit_records) > 1 else exit_records[0][3]
    hold_days   = (last_exit[0] - dt).days

    trades.append({
        "entry_date"  : dt,
        "exit_date"   : last_exit[0],
        "ticker"      : tk,
        "entry"       : entry,
        "exit_price"  : last_exit[1],
        "return_pct"  : blended_ret * 100,
        "hold_days"   : hold_days,
        "exit_reason" : main_reason,
        "half_exit"   : half_exited,
        "win"         : blended_ret > 0,
        "rsi_entry"   : sig["rsi"],
    })

    # 포지션 청산 날짜 기록 (재진입 차단용)
    pos_exit_date[tk] = last_exit[0]

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
avg_win  = wins["return_pct"].mean()   if len(wins)   else 0
avg_loss = losses["return_pct"].mean() if len(losses) else 0
pf       = (wins["return_pct"].sum() / -losses["return_pct"].sum()
            if len(losses) and losses["return_pct"].sum() < 0 else np.nan)
ev       = win_rate/100 * avg_win + (1-win_rate/100) * avg_loss

max_cl = cur = 0
for r in df["return_pct"]:
    cur = cur + 1 if r < 0 else 0
    max_cl = max(max_cl, cur)

exit_cnt = df["exit_reason"].value_counts()

print("\n" + "="*64)
print("  v9 서킷브레이커 + 자본회전 전략 — 백테스트 결과")
print("="*64)
print(f"  기간           : {START} ~ {END}")
print(f"  유니버스       : {len(stock_data)}개 종목")
print(f"  진입 조건      : MA200↓ + VIX≥25 + RSI<40")
print(f"  청산 규칙      : +20% / -25% CB / 60일절반 / 120일")
print(f"  총 트레이드    : {n}건")
print(f"  활성 연도      : {df['entry_date'].dt.year.nunique()}개년")
print(f"  승 률          : {win_rate:.1f}%")
print(f"  평균 수익률    : {avg_ret:+.2f}%")
print(f"  기대값         : {ev:+.2f}%")
print(f"  승자 평균      : {avg_win:+.2f}%")
print(f"  패자 평균      : {avg_loss:+.2f}%")
print(f"  Profit Factor  : {pf:.2f}")
print(f"  최대 연속 손실 : {max_cl}건")
print(f"  평균 보유 일수 : {df['hold_days'].mean():.0f}일")
print(f"  절반 청산 건수 : {df['half_exit'].sum()}건 ({df['half_exit'].mean()*100:.1f}%)")
print(f"\n  청산 유형:")
for r, c in exit_cnt.items():
    print(f"    {r:<25}: {c:4d}건 ({c/n*100:.1f}%)")
print("="*64)

# 연도별
df["year"] = df["entry_date"].dt.year
yearly = df.groupby("year").agg(
    trades  =("return_pct","count"),
    avg_ret =("return_pct","mean"),
    win_rate=("win", lambda x: x.mean()*100),
    total_ret=("return_pct","sum"),
    avg_hold=("hold_days","mean"),
)
print("\n연도별 성과:")
print(yearly.round(1).to_string())

print(f"\n▶ 수익 상위 10:")
print(df.nlargest(10,"return_pct")[
    ["entry_date","ticker","entry","exit_price","return_pct","exit_reason","hold_days"]
].to_string(index=False))
print(f"\n▶ 손실 하위 10:")
print(df.nsmallest(10,"return_pct")[
    ["entry_date","ticker","entry","exit_price","return_pct","exit_reason","hold_days"]
].to_string(index=False))

# ──────────────────────────────────────────────
# 포트폴리오 복리 시뮬레이션
# ──────────────────────────────────────────────
print("\n" + "="*64)
print("  포트폴리오 복리 시뮬레이션 (동시 5종목 균등배분)")
print("="*64)

capital = 1.0
yearly_capital = {}

for entry_date, group in df.groupby("entry_date"):
    n_pos   = len(group)
    weight  = 1.0 / max(n_pos, MAX_POSITIONS)   # 최대 5등분
    batch   = (group["return_pct"] / 100 * weight).sum()
    capital *= (1 + batch)
    yr = entry_date.year
    yearly_capital[yr] = capital

years = (df["exit_date"].max() - df["entry_date"].min()).days / 365.25
cagr  = capital ** (1 / years) - 1

print(f"  누적 수익률 : {(capital-1)*100:+.1f}%")
print(f"  CAGR        : {cagr*100:+.2f}%")
print(f"\n  1억 투자 시 복리 시뮬레이션:")
print(f"  {'기간':<10} {'금액':>10} {'수익':>10}")
print(f"  {'-'*32}")
for yr in [1, 3, 5, 10, 15, 20]:
    val = 1.0 * (1 + cagr) ** yr
    print(f"  {yr:2d}년 후   {val:>8.2f}억  {(val-1)*100:>+8.0f}%")

# 전략 버전 비교
print("\n" + "="*74)
print("                  전략 버전 최종 비교")
print("="*74)
history = [
    ("v4",   214, 40.7,  1.03, 1.47, "+3.0%",  "복합조건"),
    ("v6",    38, 47.4,  2.19, 2.26, "+3.0%",  "불마켓 전용"),
    ("v7",   161, 37.3,  1.03, 1.54, "+5.7%",  "핵심4조건"),
    ("v8",    84, 26.2, -0.38, 0.89, "-1.0%",  "데이터기반(stop有)"),
    ("v9",     n, round(win_rate,1), round(avg_ret,2),
               round(pf,2) if not np.isnan(pf) else 0,
               f"{cagr*100:+.1f}%", "CB+자본회전(stop無)"),
]
print(f"  {'버전':>4} {'건수':>6} {'승률':>7} {'건당수익':>9} {'PF':>6} {'CAGR':>8}  {'전략특징'}")
print("  " + "-"*70)
for row in history:
    nm, n_, wr, ar, pf_, cagr_, desc = row
    mark = " ◀" if nm == "v9" else ""
    print(f"  {nm:>4} {n_:>6} {wr:>6.1f}% {ar:>+8.2f}% {pf_:>6.2f} {cagr_:>8}  {desc}{mark}")
print("="*74)

# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    f"v9 서킷브레이커 + 자본회전 전략 (MA200↓ + VIX≥25 + RSI<40)\n"
    f"총 {n}건 | 승률 {win_rate:.1f}% | 평균 {avg_ret:+.2f}% | PF {pf:.2f} | CAGR {cagr*100:+.1f}%",
    fontsize=12, fontweight="bold"
)

# 1. 수익률 분포
ax = axes[0, 0]
ax.hist(df["return_pct"], bins=40, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(0,       color="red",    linestyle="--", lw=1.5)
ax.axvline(avg_ret, color="orange", linestyle="-",  lw=1.5, label=f"평균 {avg_ret:+.2f}%")
ax.set_title("수익률 분포", fontweight="bold")
ax.set_xlabel("Return (%)")
ax.legend()

# 2. 청산 유형
ax = axes[0, 1]
top_exits = exit_cnt.head(6)
clr_map = {"target":"#2ecc71","circuit":"#e74c3c","time":"#f39c12",
           "half_60d+target":"#27ae60","half_60d+time":"#e67e22",
           "half_60d+circuit":"#c0392b"}
colors = [clr_map.get(k, "#95a5a6") for k in top_exits.index]
ax.pie(top_exits.values,
       labels=[f"{k[:18]}\n{v}건 ({v/n*100:.0f}%)" for k,v in top_exits.items()],
       colors=colors, startangle=140, textprops={"fontsize":7.5})
ax.set_title("청산 유형", fontweight="bold")

# 3. 누적 수익률
ax = axes[0, 2]
cum = (1 + df["return_pct"]/100).cumprod() - 1
ax.plot(range(len(cum)), cum*100, color="navy", lw=1.5)
ax.axhline(0, color="red", linestyle="--", lw=1)
ax.fill_between(range(len(cum)), cum*100, 0, where=(cum>=0), alpha=0.2, color="green")
ax.fill_between(range(len(cum)), cum*100, 0, where=(cum<0),  alpha=0.2, color="red")
ax.set_title(f"누적 수익률 (CAGR {cagr*100:+.1f}%)", fontweight="bold")
ax.set_xlabel("Trade #")
ax.set_ylabel("Cumulative %")

# 4. 연도별 평균 수익률
ax = axes[1, 0]
bar_c = ["#2ecc71" if v>=0 else "#e74c3c" for v in yearly["avg_ret"]]
ax.bar(yearly.index.astype(str), yearly["avg_ret"], color=bar_c, edgecolor="white")
ax.axhline(0, color="black", lw=0.8)
for x, (yr, row) in enumerate(yearly.iterrows()):
    ax.text(x, row["avg_ret"] + (0.3 if row["avg_ret"]>=0 else -0.8),
            f'{int(row["trades"])}건', ha="center", fontsize=8)
ax.set_title("연도별 평균 수익률", fontweight="bold")
ax.tick_params(axis="x", rotation=45)

# 5. 연도별 승률
ax = axes[1, 1]
bar_c2 = ["#2ecc71" if v>=60 else ("#f39c12" if v>=45 else "#e74c3c") for v in yearly["win_rate"]]
ax.bar(yearly.index.astype(str), yearly["win_rate"], color=bar_c2, edgecolor="white")
ax.axhline(60, color="green", linestyle="--", lw=1, label="60%")
ax.axhline(win_rate, color="navy", linestyle=":", lw=1.5, label=f"전체 {win_rate:.1f}%")
ax.set_title("연도별 승률", fontweight="bold")
ax.tick_params(axis="x", rotation=45)
ax.legend(fontsize=8)

# 6. 보유일수 분포
ax = axes[1, 2]
ax.hist(df["hold_days"], bins=30, color="purple", edgecolor="white", alpha=0.8)
ax.axvline(df["hold_days"].mean(), color="orange", lw=1.5, linestyle="--",
           label=f"평균 {df['hold_days'].mean():.0f}일")
ax.axvline(HALF_EXIT_DAYS, color="blue", lw=1, linestyle=":",
           label=f"절반청산 {HALF_EXIT_DAYS}일")
ax.set_title("보유 기간 분포", fontweight="bold")
ax.set_xlabel("Days Held")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("backtest_v9_results.png", dpi=150, bbox_inches="tight")
df.to_csv("backtest_v9_trades.csv", index=False)
print("\n📊 backtest_v9_results.png 저장 완료")
print("📄 backtest_v9_trades.csv 저장 완료")
print("✅ v9 백테스트 완료")
