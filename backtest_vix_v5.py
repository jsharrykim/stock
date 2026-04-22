"""
backtest_vix_v5.py  — VIX 모멘텀 전환 전략 v5
=================================================
v4 대비 핵심 개선:
1. 레짐 분기 (불마켓 / 방어 모드)
   - SPY > MA200 → 불마켓: Stop -7%, Target +21%, Max Hold 20일
   - SPY < MA200 → 방어:   Stop -4%, Target +12%, Max Hold 15일 + 추가 필터
2. 방어 모드 추가 진입 필터
   - RSI 2일 연속 상승
   - 전일도 양봉
   - QQQ 당일 양봉
3. VIX 레짐 강화
   - VIX_MA5(D) < VIX_MA5(D-3) → VIX_MA5(D) < VIX_MA5(D-5) (더 확실한 하향 전환)
4. 갭 상승 필터 + 목표가 동적 적용
목표: Profit Factor > 2.0, 승률 45%+, 평균 수익 +2%+
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
SPY_TICKER = "SPY"

# ── 불마켓 파라미터 (SPY > MA200)
BULL_STOP_PCT   = 0.07   # -7%
BULL_TARGET_PCT = 0.21   # +21%
BULL_MAX_HOLD   = 20

# ── 방어 모드 파라미터 (SPY < MA200)
BEAR_STOP_PCT   = 0.04   # -4%  (타이트하게)
BEAR_TARGET_PCT = 0.12   # +12% (손익비 1:3 유지)
BEAR_MAX_HOLD   = 15

# ── 공통
BE_TRIGGER   = 0.05   # +5% 달성 시 stop → entry
LOCK_TRIGGER = 0.10   # +10% 달성 시 stop → +BE_TRIGGER
VIX_MIN      = 22
VOL_MULT     = 1.3
GAP_UP_LIMIT = 0.03   # 갭 상승 3% 초과 진입 금지
MAX_DAILY    = 7

# ──────────────────────────────────────────────
# 2. 유니버스
# ──────────────────────────────────────────────
TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","TMUS","AMD","ADBE","PEP","CSCO","QCOM","INTC","TXN","AMGN",
    "INTU","AMAT","MU","LRCX","KLAC","CDNS","SNPS","MRVL","FTNT","PANW",
    "CRWD","ABNB","DDOG","ZS","TEAM","MNST","KDP","MDLZ","ORLY","AZN",
    "ISRG","IDXX","REGN","VRTX","GILD","BIIB","ILMN","MRNA","SGEN","ALXN",
    "ODFL","FAST","PCAR","CTSH","PAYX","VRSK","SNPS","WDAY","OKTA",
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
    # ATR
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - d["Close"].shift(1)).abs(),
        (d["Low"]  - d["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    d["ATR14"] = tr.rolling(14).mean()
    return d


# ──────────────────────────────────────────────
# 4. 시장 데이터
# ──────────────────────────────────────────────
print("📥 시장 데이터 다운로드 중...")
vix_raw = yf.download(VIX_TICKER, start=START, end=END, auto_adjust=True, progress=False)
_vc = vix_raw["Close"]
if isinstance(_vc, pd.DataFrame):
    _vc = _vc.iloc[:, 0]
vix = pd.Series(_vc.values, index=_vc.index, dtype=float, name="VIX").dropna()

qqq_raw = dl_ohlcv(QQQ_TICKER, START, END)
qqq_close = dl_series(QQQ_TICKER, START, END)
qqq_open  = pd.Series(qqq_raw["Open"].values, index=qqq_raw.index, dtype=float)

spy_raw   = dl_ohlcv(SPY_TICKER, START, END)
spy_close = pd.Series(spy_raw["Close"].values, index=spy_raw.index, dtype=float)
spy_ma200 = spy_close.rolling(200).mean()

vix_ma5   = vix.rolling(5).mean()
vix_ma5_d5 = vix_ma5.shift(5)   # 5일 전 MA5 (더 확실한 전환 확인)

print(f"✅ VIX {len(vix)}일, QQQ {len(qqq_close)}일, SPY {len(spy_close)}일")

# ──────────────────────────────────────────────
# 5. 종목 데이터
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
signals_by_date = {}

for tk, d in stock_data.items():
    d = d.dropna(subset=["MA60","RSI","MACD_H","VolAvg20","ATR14"])

    close     = d["Close"]
    open_     = d["Open"]
    rsi       = d["RSI"]
    macd_h    = d["MACD_H"]
    vol       = d["Volume"]
    vol_avg   = d["VolAvg20"]
    ma20      = d["MA20"]
    ma60      = d["MA60"]

    rsi_prev    = rsi.shift(1)
    rsi_prev2   = rsi.shift(2)
    macd_h_prev = macd_h.shift(1)
    close_prev  = close.shift(1)
    open_prev   = d["Open"].shift(1)

    common_idx = d.index.intersection(vix.index)
    if len(common_idx) < 50:
        continue

    vx       = vix.reindex(common_idx)
    vx_ma5   = vix_ma5.reindex(common_idx)
    vx_ma5_d5= vix_ma5_d5.reindex(common_idx)
    qqq_c    = qqq_close.reindex(common_idx)
    qqq_o    = qqq_open.reindex(common_idx)
    qqq_prev = qqq_c.shift(1)
    spy_c    = spy_close.reindex(common_idx)
    spy_m200 = spy_ma200.reindex(common_idx)

    # ── 레짐 판별 ─────────────────────────────────
    is_bull = spy_c > spy_m200   # True: 불마켓, False: 방어

    # ── Layer A: VIX 레짐 ─────────────────────────
    A1 = vx >= VIX_MIN
    A2 = vx < vx_ma5                    # VIX < 5일 평균
    A3 = vx_ma5 < vx_ma5_d5            # MA5가 5일 전보다 낮음 (더 확실한 하향)
    A4 = qqq_c >= qqq_prev * 0.99       # QQQ 급락일 금지

    A_mask = A1 & A2 & A3 & A4

    # ── Layer B: 종목 상태 ────────────────────────
    B1 = rsi.between(30, 52)
    B2 = rsi > rsi_prev                  # RSI 상승 중
    B3 = rsi_prev < 45
    B4 = close >= ma20 * 0.90
    B5 = close >= ma60 * 0.85

    B_base = B1 & B2 & B3 & (B4 | B5)

    # 방어 모드 추가 필터: RSI 이틀 연속 상승 + 전일 양봉 + QQQ 당일 양봉
    B_extra = (
        (rsi_prev > rsi_prev2)  &         # RSI 어제도 상승
        (close_prev > open_prev) &         # 전일 양봉
        (qqq_c > qqq_o)                    # QQQ 당일 양봉
    )

    B_bull = B_base
    B_bear = B_base & B_extra

    # ── Layer C: 당일 신호 ───────────────────────
    C1 = macd_h > macd_h_prev
    C2 = close > open_
    C3 = vol >= VOL_MULT * vol_avg

    # 갭 상승 필터
    gap_up = (open_ / close_prev - 1) > GAP_UP_LIMIT

    C_mask = C1 & C2 & C3 & ~gap_up

    # ── 최종 신호 (레짐별 조건 분기) ──────────────
    bull_mask = A_mask & B_bull & C_mask & is_bull
    bear_mask = A_mask & B_bear & C_mask & ~is_bull
    sig_mask  = bull_mask | bear_mask

    sig_dates = d.index[sig_mask.fillna(False)]

    for dt in sig_dates:
        if pd.isna(rsi.loc[dt]) or pd.isna(rsi_prev.loc[dt]):
            continue
        rsi_delta = float(rsi.loc[dt] - rsi_prev.loc[dt])
        row = d.loc[dt]
        regime = "bull" if bool(is_bull.loc[dt]) else "bear"
        if dt not in signals_by_date:
            signals_by_date[dt] = []
        signals_by_date[dt].append((tk, rsi_delta, row, regime))

# 날짜별 정렬 및 상위 MAX_DAILY
final_signals = []
for dt, items in sorted(signals_by_date.items()):
    items.sort(key=lambda x: -x[1])
    for tk, rsi_delta, row, regime in items[:MAX_DAILY]:
        final_signals.append({
            "date"   : dt,
            "ticker" : tk,
            "entry"  : float(row["Close"]),
            "rsi"    : float(row["RSI"]),
            "rsi_delta": rsi_delta,
            "regime" : regime,
        })

print(f"✅ 총 원시 신호: {len(final_signals)}건 ({len(signals_by_date)}거래일)")
bull_cnt = sum(1 for s in final_signals if s["regime"] == "bull")
bear_cnt = sum(1 for s in final_signals if s["regime"] == "bear")
print(f"   불마켓 신호: {bull_cnt}건 / 방어 신호: {bear_cnt}건")

# ──────────────────────────────────────────────
# 7. 트레이드 시뮬레이션
# ──────────────────────────────────────────────
print("⚙️  트레이드 시뮬레이션 중...")
trades = []
open_positions = {}

for sig in final_signals:
    tk     = sig["ticker"]
    dt     = sig["date"]
    entry  = sig["entry"]
    regime = sig["regime"]

    if tk in open_positions:
        continue

    # 레짐별 파라미터
    if regime == "bull":
        stop_pct   = BULL_STOP_PCT
        target_pct = BULL_TARGET_PCT
        max_hold   = BULL_MAX_HOLD
    else:
        stop_pct   = BEAR_STOP_PCT
        target_pct = BEAR_TARGET_PCT
        max_hold   = BEAR_MAX_HOLD

    d = stock_data[tk]
    future = d.loc[d.index > dt]
    if len(future) == 0:
        continue

    stop   = entry * (1 - stop_pct)
    target = entry * (1 + target_pct)
    max_ret = 0.0
    be_applied   = False
    lock_applied = False
    exit_date  = None
    exit_price = None
    exit_reason= None

    for i, (fdt, frow) in enumerate(future.iterrows()):
        low_   = float(frow["Low"])
        high_  = float(frow["High"])
        close_ = float(frow["Close"])

        daily_max_ret = (high_ - entry) / entry
        if daily_max_ret > max_ret:
            max_ret = daily_max_ret

        if max_ret >= LOCK_TRIGGER and not lock_applied:
            stop = entry * (1 + BE_TRIGGER)
            lock_applied = True
            be_applied   = True
        elif max_ret >= BE_TRIGGER and not be_applied:
            stop = entry
            be_applied = True

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
        if i + 1 >= max_hold:
            exit_price  = close_
            exit_reason = "time"
            exit_date   = fdt
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
        "regime"     : regime,
        "win"        : ret > 0,
    })

print(f"✅ 시뮬레이션 완료: {len(trades)}건 트레이드")

# ──────────────────────────────────────────────
# 8. 결과 분석
# ──────────────────────────────────────────────
df = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)

if df.empty:
    print("⚠️  트레이드 없음")
else:
    wins   = df[df["win"]]
    losses = df[~df["win"]]

    win_rate = len(wins) / len(df) * 100
    avg_ret  = df["return_pct"].mean()
    avg_win  = wins["return_pct"].mean()   if len(wins)   else 0
    avg_loss = losses["return_pct"].mean() if len(losses) else 0
    pf       = (wins["return_pct"].sum() / -losses["return_pct"].sum()
                if len(losses) and losses["return_pct"].sum() < 0 else np.nan)
    ev       = win_rate/100 * avg_win + (1-win_rate/100) * avg_loss

    max_cons_loss = cur = 0
    for r in df["return_pct"]:
        cur = cur + 1 if r < 0 else 0
        max_cons_loss = max(max_cons_loss, cur)

    exit_cnt = df["exit_reason"].value_counts()

    print("\n" + "="*60)
    print("       v5 VIX 모멘텀 전환 전략 (레짐 분기) — 결과")
    print("="*60)
    print(f"  기간         : {START} ~ {END}")
    print(f"  유니버스     : Nasdaq 100 ({len(stock_data)}개 종목)")
    print(f"  총 트레이드  : {len(df)}건")
    print(f"  승 률        : {win_rate:.1f}%")
    print(f"  평균 수익률  : {avg_ret:+.2f}%")
    print(f"  기대값       : {ev:+.2f}%")
    print(f"  승자 평균    : {avg_win:+.2f}%")
    print(f"  패자 평균    : {avg_loss:+.2f}%")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  최대 연속 손실: {max_cons_loss}건")
    print(f"  청산 유형:")
    for reason, cnt in exit_cnt.items():
        print(f"    {reason:8s}: {cnt:4d}건 ({cnt/len(df)*100:.1f}%)")
    print("="*60)

    # 레짐별 성과
    print("\n레짐별 성과:")
    regime_stats = df.groupby("regime").agg(
        trades=("return_pct","count"),
        win_rate=("win", lambda x: round(x.mean()*100,1)),
        avg_ret=("return_pct", lambda x: round(x.mean(),2)),
        pf=("return_pct", lambda x: (
            round(x[x>0].sum() / (-x[x<0].sum()), 2)
            if x[x<0].sum() < 0 else np.nan
        )),
    )
    print(regime_stats.to_string())

    # 연도별
    df["year"] = pd.to_datetime(df["entry_date"]).dt.year
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
        ["entry_date","ticker","entry","exit","return_pct","exit_reason","regime","hold_days"]
    ].to_string(index=False))
    print(f"\n▶ 손실 하위 10:")
    print(df.nsmallest(10,"return_pct")[
        ["entry_date","ticker","entry","exit","return_pct","exit_reason","regime","hold_days"]
    ].to_string(index=False))

    # ──────────────────────────────────────────────
    # 9. 시각화
    # ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("v5 VIX 모멘텀 전환 전략 (레짐 분기) — 백테스트 결과", fontsize=14, fontweight="bold")

    # 수익률 히스토그램
    ax = axes[0, 0]
    ax.hist(df["return_pct"], bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(avg_ret, color="orange", linewidth=1.5, label=f"평균 {avg_ret:+.2f}%")
    ax.set_title("수익률 분포"); ax.set_xlabel("Return (%)"); ax.legend()

    # 청산 유형 파이
    ax = axes[0, 1]
    colors = {"stop":"#e74c3c","target":"#2ecc71","time":"#f39c12"}
    ax.pie(exit_cnt.values, labels=[f"{k}\n{v}건" for k,v in exit_cnt.items()],
           colors=[colors.get(k,"#aaa") for k in exit_cnt.index],
           autopct="%1.1f%%", startangle=140)
    ax.set_title("청산 유형")

    # 레짐별 수익률 분포
    ax = axes[0, 2]
    bull_ret = df[df["regime"]=="bull"]["return_pct"]
    bear_ret = df[df["regime"]=="bear"]["return_pct"]
    ax.hist(bull_ret, bins=20, alpha=0.6, color="green", label=f"불마켓({len(bull_ret)}건)")
    ax.hist(bear_ret, bins=20, alpha=0.6, color="red",   label=f"방어({len(bear_ret)}건)")
    ax.axvline(0, color="black", linestyle="--")
    ax.set_title("레짐별 수익률 분포"); ax.legend()

    # 누적 수익률
    ax = axes[1, 0]
    cum = (1 + df["return_pct"]/100).cumprod() - 1
    ax.plot(range(len(cum)), cum*100, color="navy", linewidth=1.5)
    ax.axhline(0, color="red", linestyle="--")
    ax.fill_between(range(len(cum)), cum*100, 0, where=(cum>=0), alpha=0.15, color="green")
    ax.fill_between(range(len(cum)), cum*100, 0, where=(cum<0),  alpha=0.15, color="red")
    ax.set_title("누적 수익률 곡선"); ax.set_xlabel("Trade #"); ax.set_ylabel("Cumulative %")

    # 연도별 avg_ret bar
    ax = axes[1, 1]
    bar_colors = ["#2ecc71" if v>=0 else "#e74c3c" for v in yearly["avg_ret"]]
    ax.bar(yearly.index.astype(str), yearly["avg_ret"], color=bar_colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("연도별 평균 수익률"); ax.tick_params(axis="x", rotation=45)

    # 연도별 win_rate
    ax = axes[1, 2]
    ax.bar(yearly.index.astype(str), yearly["win_rate"],
           color=["#2ecc71" if v>=45 else "#f39c12" if v>=35 else "#e74c3c"
                  for v in yearly["win_rate"]],
           edgecolor="white")
    ax.axhline(45, color="green", linestyle="--", linewidth=1, label="목표 45%")
    ax.set_title("연도별 승률 (%)"); ax.tick_params(axis="x", rotation=45); ax.legend()

    plt.tight_layout()
    plt.savefig("backtest_v5_results.png", dpi=150, bbox_inches="tight")
    print("\n📊 차트 저장: backtest_v5_results.png")
    df.to_csv("backtest_v5_trades.csv", index=False)
    print("📄 트레이드 내역: backtest_v5_trades.csv")

    # ── v4 vs v5 비교 출력 ──
    print("\n" + "="*60)
    print("           v4 vs v5 비교 요약")
    print("="*60)
    print(f"  {'지표':<20} {'v4':>10} {'v5':>10}")
    print(f"  {'-'*40}")

    v4_stats = {
        "총 트레이드": 214,
        "승률 (%)": 40.7,
        "평균 수익률 (%)": 1.03,
        "Profit Factor": 1.47,
        "승자 평균 (%)": 7.85,
        "패자 평균 (%)": -3.65,
    }
    v5_stats = {
        "총 트레이드": len(df),
        "승률 (%)": round(win_rate, 1),
        "평균 수익률 (%)": round(avg_ret, 2),
        "Profit Factor": round(pf, 2) if not np.isnan(pf) else 0,
        "승자 평균 (%)": round(avg_win, 2),
        "패자 평균 (%)": round(avg_loss, 2),
    }
    for k in v4_stats:
        arrow = "▲" if v5_stats[k] > v4_stats[k] else ("▼" if v5_stats[k] < v4_stats[k] else "─")
        print(f"  {k:<20} {v4_stats[k]:>10} {v5_stats[k]:>10} {arrow}")
    print("="*60)

print("\n✅ v5 백테스트 완료")
