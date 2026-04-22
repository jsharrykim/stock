"""
backtest_vix_v4.py  — VIX 모멘텀 전환 전략 v4
=================================================
핵심 변경: "공포 극단 잡기" → "VIX 진정 초입 + RSI 모멘텀 전환" 포착
- VIX_MA5 하향 전환 (D vs D-3) 조건으로 진짜 진정 국면 선별
- RSI 30~52 & 전일보다 상승 & 전일 RSI < 45  (극단 과매도 제외)
- 고정 손익비 1:3 (-7% / +21%) + BE/Lock Stop
- 핵심 조건 7개 (과최적화 방지)
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
# 1. 설정
# ──────────────────────────────────────────────
START = "2010-01-01"
END   = "2026-01-01"

VIX_TICKER = "^VIX"
QQQ_TICKER = "QQQ"

STOP_PCT     = 0.07    # 고정 -7%
TARGET_PCT   = 0.21    # 고정 +21%  (손익비 1:3)
BE_TRIGGER   = 0.05    # +5% 달성 시 stop → entry
LOCK_TRIGGER = 0.10    # +10% 달성 시 stop → +5%
MAX_HOLD     = 20      # 거래일
MAX_DAILY    = 7       # 하루 최대 신호
VIX_MIN      = 22      # VIX 하한
VOL_MULT     = 1.3     # 거래량 배수
GAP_UP_LIMIT = 0.03    # 전일 종가 대비 갭상승 초과 시 진입 금지

# ──────────────────────────────────────────────
# 2. 유니버스 — Nasdaq 100 대표 종목
# ──────────────────────────────────────────────
TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","TMUS","AMD","ADBE","PEP","CSCO","QCOM","INTC","TXN","AMGN",
    "INTU","AMAT","MU","LRCX","KLAC","CDNS","SNPS","MRVL","FTNT","PANW",
    "CRWD","ABNB","DDOG","ZS","TEAM","MNST","KDP","MDLZ","ORLY","AZN",
    "ISRG","IDXX","REGN","VRTX","GILD","BIIB","ILMN","MRNA","SGEN","ALXN",
    "ODFL","FAST","PCAR","CTSH","PAYX","VRSK","ANSS","SPLK","WDAY","OKTA",
    "DOCU","ZM","PTON","COIN","MELI","PDD","JD","BIDU","NTES","WBA",
    "DLTR","SBUX","ROST","LULU","EBAY","MAR","CTAS","EA","TTWO","ATVI",
    "CHTR","CMCSA","SIRI","MTCH","IAC","EXC","XEL","AEP","FANG","MKL",
    "PLTR","CPRT","SLAB","SWKS","XLNX","NXPI","MPWR","ENPH","SEDG","ON",
]
TICKERS = sorted(set(TICKERS))

# ──────────────────────────────────────────────
# 3. 헬퍼
# ──────────────────────────────────────────────
def dl_series(ticker, start, end, col="Close"):
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    if raw.empty:
        return pd.Series(dtype=float, name=ticker)
    s = raw[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.Series(s.values, index=s.index, dtype=float, name=ticker).dropna()


def dl_ohlcv(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw[["Open","High","Low","Close","Volume"]].copy()


def compute_indicators(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    close = d["Close"]
    # MAs
    d["MA20"]  = close.rolling(20).mean()
    d["MA60"]  = close.rolling(60).mean()
    d["MA200"] = close.rolling(200).mean()
    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - 100 / (1 + rs)
    # MACD histogram
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    d["MACD_H"] = macd - sig
    # Volume avg
    d["VolAvg20"] = d["Volume"].rolling(20).mean()
    return d


# ──────────────────────────────────────────────
# 4. 시장 데이터 다운로드
# ──────────────────────────────────────────────
print("📥 시장 데이터 다운로드 중...")
vix_raw = yf.download(VIX_TICKER, start=START, end=END,
                      auto_adjust=True, progress=False)
_vc = vix_raw["Close"]
if isinstance(_vc, pd.DataFrame):
    _vc = _vc.iloc[:, 0]
vix = pd.Series(_vc.values, index=_vc.index, dtype=float, name="VIX").dropna()

qqq_close = dl_series(QQQ_TICKER, START, END)

# VIX 파생 지표
vix_ma5   = vix.rolling(5).mean()
vix_ma5_d3 = vix_ma5.shift(3)   # 3일 전 MA5

print(f"✅ VIX {len(vix)}일, QQQ {len(qqq_close)}일")

# ──────────────────────────────────────────────
# 5. 종목 데이터 다운로드 & 지표 계산
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

print(f"✅ {len(stock_data)}개 종목 로드 완료")

# ──────────────────────────────────────────────
# 6. 신호 생성
# ──────────────────────────────────────────────
print("🔍 신호 생성 중...")

signals_by_date = {}  # date → [(ticker, rsi_delta, row_data)]

for tk, d in stock_data.items():
    d = d.dropna(subset=["MA60","RSI","MACD_H","VolAvg20"])

    close     = d["Close"]
    open_     = d["Open"]
    high      = d["High"]
    rsi       = d["RSI"]
    macd_h    = d["MACD_H"]
    vol       = d["Volume"]
    vol_avg   = d["VolAvg20"]
    ma20      = d["MA20"]
    ma60      = d["MA60"]

    rsi_prev  = rsi.shift(1)
    macd_h_prev = macd_h.shift(1)
    close_prev  = close.shift(1)

    # ── Layer A: VIX 레짐 ─────────────────────────────
    common_idx = d.index.intersection(vix.index)
    if len(common_idx) < 10:
        continue

    vx      = vix.reindex(common_idx)
    vx_ma5  = vix_ma5.reindex(common_idx)
    vx_ma5_d3 = vix_ma5_d3.reindex(common_idx)
    qqq_c   = qqq_close.reindex(common_idx)
    qqq_prev = qqq_c.shift(1)

    A1 = vx >= VIX_MIN                    # VIX >= 22
    A2 = vx < vx_ma5                      # 현재 VIX < 5일 평균 (진정 시작)
    A3 = vx_ma5 < vx_ma5_d3              # MA5 자체가 하향 전환 (D vs D-3)
    A4 = qqq_c >= qqq_prev * 0.99         # QQQ 급락일 금지

    A_mask = A1 & A2 & A3 & A4

    # ── Layer B: 종목 상태 ────────────────────────────
    # RSI 모멘텀 전환
    B1 = rsi.between(30, 52)              # 극단 과매도 제외
    B2 = rsi > rsi_prev                   # RSI 상승 중
    B3 = rsi_prev < 45                    # 전일까지는 눌려있었음
    # 가격 위치
    B4 = close >= ma20 * 0.90             # MA20 -10% 이내
    B5 = close >= ma60 * 0.85             # MA60 -15% 이내

    B_mask = B1 & B2 & B3 & (B4 | B5)   # B4 OR B5 (둘 중 하나)

    # ── Layer C: 당일 신호 ───────────────────────────
    C1 = macd_h > macd_h_prev            # MACD 히스토그램 개선
    C2 = close > open_                    # 양봉
    C3 = vol >= VOL_MULT * vol_avg        # 거래량 1.3배 이상

    # 갭상승 필터 (추격매수 제외)
    gap_up = (open_ / close_prev - 1) > GAP_UP_LIMIT
    C_mask = C1 & C2 & C3 & ~gap_up

    # ── 최종 신호 ─────────────────────────────────────
    sig_mask = A_mask & B_mask & C_mask
    sig_dates = d.index[sig_mask.fillna(False)]

    for dt in sig_dates:
        rsi_delta = float(rsi.loc[dt] - rsi_prev.loc[dt])  # 우선순위 기준
        row = d.loc[dt]
        if dt not in signals_by_date:
            signals_by_date[dt] = []
        signals_by_date[dt].append((tk, rsi_delta, row))

# 날짜별로 RSI 상승폭 큰 순 정렬 → 상위 MAX_DAILY 선택
final_signals = []
for dt, items in sorted(signals_by_date.items()):
    items.sort(key=lambda x: -x[1])  # RSI 상승폭 내림차순
    for tk, rsi_delta, row in items[:MAX_DAILY]:
        final_signals.append({
            "date"  : dt,
            "ticker": tk,
            "entry" : float(row["Close"]),
            "rsi"   : float(row["RSI"]),
            "rsi_delta": rsi_delta,
        })

print(f"✅ 총 원시 신호: {len(final_signals)}건 ({len(signals_by_date)}거래일)")

# ──────────────────────────────────────────────
# 7. 트레이드 시뮬레이션
# ──────────────────────────────────────────────
print("⚙️  트레이드 시뮬레이션 중...")

trades = []
open_positions = {}   # ticker → entry info

for sig in final_signals:
    tk    = sig["ticker"]
    dt    = sig["date"]
    entry = sig["entry"]

    if tk in open_positions:
        continue                 # 이미 포지션 있으면 패스

    d = stock_data[tk]
    future = d.loc[d.index > dt]
    if len(future) == 0:
        continue

    stop   = entry * (1 - STOP_PCT)
    target = entry * (1 + TARGET_PCT)
    max_ret = 0.0
    be_applied   = False
    lock_applied = False
    exit_date    = None
    exit_price   = None
    exit_reason  = None

    for i, (fdt, frow) in enumerate(future.iterrows()):
        low_  = float(frow["Low"])
        high_ = float(frow["High"])
        close_ = float(frow["Close"])

        # 당일 고가 기준 최대 수익 추적 (BE/Lock stop 발동용)
        daily_max_ret = (high_ - entry) / entry
        if daily_max_ret > max_ret:
            max_ret = daily_max_ret

        # Lock stop 업데이트 (+10% 달성 시 stop → +5%)
        if max_ret >= LOCK_TRIGGER and not lock_applied:
            stop = entry * (1 + BE_TRIGGER)   # +5%로 이동
            lock_applied = True
            be_applied   = True

        # Break-even stop (+5% 달성 시 stop → entry)
        elif max_ret >= BE_TRIGGER and not be_applied:
            stop = entry
            be_applied = True

        # 청산 체크 (종가 기준)
        if low_ <= stop:
            exit_price  = stop
            exit_reason = "stop"
            exit_date   = fdt
            break
        if high_ >= target:
            exit_price  = target
            exit_reason = "target"
            exit_date   = fdt
            break
        if i + 1 >= MAX_HOLD:
            exit_price  = close_
            exit_reason = "time"
            exit_date   = fdt
            break

    if exit_price is None:
        continue

    ret = (exit_price - entry) / entry
    hold_days = (exit_date - dt).days

    trades.append({
        "entry_date" : dt,
        "exit_date"  : exit_date,
        "ticker"     : tk,
        "entry"      : entry,
        "exit"       : exit_price,
        "return_pct" : ret * 100,
        "hold_days"  : hold_days,
        "exit_reason": exit_reason,
        "win"        : ret > 0,
    })

print(f"✅ 시뮬레이션 완료: {len(trades)}건 트레이드")

# ──────────────────────────────────────────────
# 8. 결과 분석
# ──────────────────────────────────────────────
df = pd.DataFrame(trades)
df = df.sort_values("entry_date").reset_index(drop=True)

if df.empty:
    print("⚠️  트레이드 없음")
else:
    wins   = df[df["win"]]
    losses = df[~df["win"]]

    win_rate    = len(wins) / len(df) * 100
    avg_ret     = df["return_pct"].mean()
    avg_win     = wins["return_pct"].mean()   if len(wins)   else 0
    avg_loss    = losses["return_pct"].mean() if len(losses) else 0
    profit_factor = (wins["return_pct"].sum() / -losses["return_pct"].sum()
                     if len(losses) and losses["return_pct"].sum() < 0 else np.nan)
    expected_val = win_rate/100 * avg_win + (1-win_rate/100) * avg_loss
    
    max_cons_loss = 0
    cur_cons_loss = 0
    for r in df["return_pct"]:
        if r < 0:
            cur_cons_loss += 1
            max_cons_loss = max(max_cons_loss, cur_cons_loss)
        else:
            cur_cons_loss = 0

    exit_cnt = df["exit_reason"].value_counts()

    print("\n" + "="*56)
    print("          v4 VIX 모멘텀 전환 전략 — 백테스트 결과")
    print("="*56)
    print(f"  기간         : {START} ~ {END}")
    print(f"  유니버스     : Nasdaq 100 ({len(stock_data)}개 종목)")
    print(f"  총 트레이드  : {len(df)}건")
    print(f"  승 률        : {win_rate:.1f}%")
    print(f"  평균 수익률  : {avg_ret:+.2f}%")
    print(f"  기대값       : {expected_val:+.2f}%")
    print(f"  승자 평균    : {avg_win:+.2f}%")
    print(f"  패자 평균    : {avg_loss:+.2f}%")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  최대 연속 손실: {max_cons_loss}건")
    print(f"  청산 유형:")
    for reason, cnt in exit_cnt.items():
        pct = cnt/len(df)*100
        print(f"    {reason:8s}: {cnt:4d}건 ({pct:.1f}%)")
    print("="*56)

    # 연도별 수익률
    df["year"] = pd.to_datetime(df["entry_date"]).dt.year
    yearly = df.groupby("year").agg(
        trades=("return_pct","count"),
        avg_ret=("return_pct","mean"),
        win_rate=("win", lambda x: x.mean()*100),
        total_ret=("return_pct","sum"),
    )
    print("\n연도별 성과:")
    print(yearly.round(2).to_string())

    # 상위 10 / 하위 10 트레이드
    print(f"\n▶ 수익 상위 10:")
    print(df.nlargest(10, "return_pct")[
        ["entry_date","ticker","entry","exit","return_pct","exit_reason","hold_days"]
    ].to_string(index=False))
    print(f"\n▶ 손실 하위 10:")
    print(df.nsmallest(10, "return_pct")[
        ["entry_date","ticker","entry","exit","return_pct","exit_reason","hold_days"]
    ].to_string(index=False))

    # ──────────────────────────────────────────────
    # 9. 시각화
    # ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("v4 VIX 모멘텀 전환 전략 — 백테스트 결과", fontsize=14, fontweight="bold")

    # 9-1. 수익률 히스토그램
    ax = axes[0, 0]
    ax.hist(df["return_pct"], bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(avg_ret, color="orange", linestyle="-", linewidth=1.5, label=f"평균 {avg_ret:+.2f}%")
    ax.set_title("수익률 분포")
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Count")
    ax.legend()

    # 9-2. 청산 유형 파이
    ax = axes[0, 1]
    colors = {"stop": "#e74c3c", "target": "#2ecc71", "time": "#f39c12"}
    ax.pie(
        exit_cnt.values,
        labels=[f"{k}\n{v}건" for k, v in exit_cnt.items()],
        colors=[colors.get(k, "#aaa") for k in exit_cnt.index],
        autopct="%1.1f%%",
        startangle=140,
    )
    ax.set_title("청산 유형 비율")

    # 9-3. 누적 수익률 곡선
    ax = axes[1, 0]
    cum = (1 + df["return_pct"] / 100).cumprod() - 1
    ax.plot(range(len(cum)), cum * 100, color="navy", linewidth=1.5)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.fill_between(range(len(cum)), cum * 100, 0,
                    where=(cum >= 0), alpha=0.15, color="green")
    ax.fill_between(range(len(cum)), cum * 100, 0,
                    where=(cum < 0), alpha=0.15, color="red")
    ax.set_title("누적 수익률 곡선")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative Return (%)")

    # 9-4. 연도별 평균 수익률 Bar
    ax = axes[1, 1]
    colors_bar = ["#2ecc71" if v >= 0 else "#e74c3c" for v in yearly["avg_ret"]]
    ax.bar(yearly.index.astype(str), yearly["avg_ret"], color=colors_bar, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("연도별 평균 수익률")
    ax.set_xlabel("Year")
    ax.set_ylabel("Avg Return (%)")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("backtest_v4_results.png", dpi=150, bbox_inches="tight")
    print("\n📊 차트 저장: backtest_v4_results.png")

    # CSV 저장
    df.to_csv("backtest_v4_trades.csv", index=False)
    print("📄 트레이드 내역 저장: backtest_v4_trades.csv")

print("\n✅ v4 백테스트 완료")
