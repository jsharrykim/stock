"""
backtest_growth_v1.py — 성장률 조건 추가 비교 백테스트
=======================================================

그룹 A: 기존 5가지 매도 조건
  ① +20% 목표 수익
  ② -25% 서킷브레이커
  ③ 60거래일 경과 & 수익 중
  ④ 120거래일 타임 익시트
  ⑤ SalesQQ + EPS_QQ 동반 하락 (성장률 둔화)

그룹 B: 성장률 조건(⑤)만 적용
  ⑤만 → 나머지 조건 없음 (얼마나 일찍/늦게 탈출하는지 단독 검증)

성장률 데이터:
  - yfinance quarterly financials에서 Revenue, EPS 분기별 수집
  - QoQ 성장률 계산 (YoY는 계절성 제거에 유리하지만 데이터 지연 큼)
  - 분기 실적 발표일 기준: 해당 분기 데이터가 공시된 날 이후부터 적용

성장률 매도 조건 (조건 A):
  SalesQQ 가 직전 분기 대비 하락
  AND EPS_QQ 가 직전 분기 대비 하락
  단, 이전값이 음수인 경우: 더 악화(더 음수)될 때만 트리거

진입 조건 (기존 v10 동일):
  MA200↓ + VIX≥25 + (RSI<40 OR CCI<-100)
  진입가: 신호 발생일 다음날 시가
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
# 파라미터
# ──────────────────────────────────────────────
START        = "2010-01-01"
END          = "2026-01-01"
VIX_MIN      = 25
TARGET_PCT   = 0.20
CIRCUIT_PCT  = 0.25
HALF_EXIT    = 60
MAX_HOLD     = 120
MAX_POSITIONS= 5
MAX_DAILY    = 5

TICKERS = sorted(set([
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","COST",
    "AMD","QCOM","INTC","TXN","AMGN","INTU","AMAT","MU","LRCX","KLAC",
    "CDNS","SNPS","FTNT","PANW","MNST","ORLY","ISRG","PAYX","MELI",
    "PLTR","CPRT","NXPI","ON","CSX","ROP","ADP","ADI","BKNG",
    "MDLZ","AZN","FAST","MCHP",
]))

# ──────────────────────────────────────────────
# 헬퍼: OHLCV + 기술 지표
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
    d  = d.copy()
    c  = d["Close"]
    d["MA200"] = c.rolling(200).mean()
    delta = c.diff()
    g = delta.clip(lower=0).rolling(14).mean()
    l = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100 / (1 + g / l.replace(0, np.nan))
    tp     = (d["High"] + d["Low"] + c) / 3
    tp_ma  = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    d["CCI"] = (tp - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))
    return d


# ──────────────────────────────────────────────
# 헬퍼: 분기 성장률 데이터 수집
# ──────────────────────────────────────────────
def get_quarterly_growth(ticker):
    """
    yfinance quarterly financials에서 매출/EPS QoQ 성장률 시계열 반환.
    반환: DataFrame with columns [date, salesQQ, epsQQ]
      - date: 해당 분기 데이터가 시장에 반영되는 날짜 (실적 발표일 기준)
              yfinance에서 발표일을 직접 제공하지 않으므로
              분기 종료 후 45일을 보수적 발표 시점으로 가정
              (실제 발표는 보통 30~60일 후)
    """
    try:
        tk = yf.Ticker(ticker)

        # 분기별 매출 (Total Revenue)
        qincome = tk.quarterly_income_stmt
        if qincome is None or qincome.empty:
            return pd.DataFrame()

        # 컬럼이 날짜 (분기 종료일), 인덱스가 항목명
        # 날짜 오름차순 정렬
        qincome = qincome.sort_index(axis=1)

        revenue_row = None
        for idx in qincome.index:
            if "Total Revenue" in str(idx) or "Revenue" in str(idx):
                revenue_row = idx
                break
        if revenue_row is None:
            return pd.DataFrame()

        revenue = qincome.loc[revenue_row].dropna()

        # EPS: quarterly_earnings 또는 income_stmt에서 Basic EPS
        eps_row = None
        for idx in qincome.index:
            if "Basic EPS" in str(idx) or "Diluted EPS" in str(idx):
                eps_row = idx
                break

        eps = None
        if eps_row is not None:
            eps = qincome.loc[eps_row].dropna()
        else:
            # EPS가 없으면 Net Income으로 대체
            for idx in qincome.index:
                if "Net Income" in str(idx):
                    eps_row = idx
                    break
            if eps_row:
                eps = qincome.loc[eps_row].dropna()

        if eps is None or eps.empty:
            return pd.DataFrame()

        # 공통 날짜 기준으로 정렬
        common_dates = revenue.index.intersection(eps.index)
        if len(common_dates) < 3:
            return pd.DataFrame()

        revenue = revenue.reindex(common_dates).sort_index()
        eps     = eps.reindex(common_dates).sort_index()

        rows = []
        for i in range(1, len(revenue)):
            q_date    = revenue.index[i]   # 분기 종료일
            rev_curr  = float(revenue.iloc[i])
            rev_prev  = float(revenue.iloc[i - 1])
            eps_curr  = float(eps.iloc[i])
            eps_prev  = float(eps.iloc[i - 1])

            if rev_prev == 0 or np.isnan(rev_prev) or np.isnan(rev_curr):
                sales_qq = np.nan
            else:
                sales_qq = (rev_curr - rev_prev) / abs(rev_prev) * 100

            if eps_prev == 0 or np.isnan(eps_prev) or np.isnan(eps_curr):
                eps_qq = np.nan
            else:
                eps_qq = (eps_curr - eps_prev) / abs(eps_prev) * 100

            # 실적 발표일 = 분기 종료 후 45일 (보수적 가정)
            report_date = pd.Timestamp(q_date) + pd.Timedelta(days=45)

            rows.append({
                "quarter_end" : pd.Timestamp(q_date),
                "report_date" : report_date,
                "salesQQ"     : sales_qq,
                "epsQQ"       : eps_qq,
            })

        df = pd.DataFrame(rows).sort_values("report_date").reset_index(drop=True)
        return df

    except Exception as e:
        return pd.DataFrame()


def build_growth_signal_series(growth_df, price_index):
    """
    growth_df: get_quarterly_growth() 반환값
    price_index: 가격 데이터 날짜 인덱스

    반환: 각 거래일에 대해 "성장률 매도 플래그" (True/False) Series
    로직:
      - 각 report_date 이후 거래일부터 해당 분기 성장률 값 적용
      - salesQQ AND epsQQ 동시 하락 → 플래그 True
      - 이전 분기 대비 하락 판단:
          이전 salesQQ >= 0: 현재 < 이전 → True
          이전 salesQQ < 0: 현재 < 이전 (더 악화) → True
          이전 salesQQ < 0: 현재 > 이전 (개선 중) → False
    """
    flag = pd.Series(False, index=price_index)

    if growth_df.empty or len(growth_df) < 2:
        return flag

    for i in range(1, len(growth_df)):
        curr = growth_df.iloc[i]
        prev = growth_df.iloc[i - 1]

        curr_sq = curr["salesQQ"]
        prev_sq = prev["salesQQ"]
        curr_eq = curr["epsQQ"]
        prev_eq = prev["epsQQ"]

        if any(np.isnan(v) for v in [curr_sq, prev_sq, curr_eq, prev_eq]):
            continue

        # salesQQ 하락 판단
        if prev_sq >= 0:
            sales_down = curr_sq < prev_sq
        else:
            sales_down = curr_sq < prev_sq  # 더 악화

        # 개선 중인 경우 (이전 음수 → 현재 덜 음수) 스킵
        if prev_sq < 0 and curr_sq > prev_sq:
            sales_down = False

        # epsQQ 하락 판단
        if prev_eq >= 0:
            eps_down = curr_eq < prev_eq
        else:
            eps_down = curr_eq < prev_eq

        if prev_eq < 0 and curr_eq > prev_eq:
            eps_down = False

        triggered = sales_down and eps_down

        # report_date 이후 ~ 다음 report_date 전까지 적용
        start_date = curr["report_date"]
        if i + 1 < len(growth_df):
            end_date = growth_df.iloc[i + 1]["report_date"]
        else:
            end_date = price_index[-1] + pd.Timedelta(days=1)

        mask = (price_index >= start_date) & (price_index < end_date)
        flag[mask] = triggered

    return flag


# ──────────────────────────────────────────────
# 데이터 다운로드
# ──────────────────────────────────────────────
print("📥 시장 데이터 다운로드 중...")
vix_raw = yf.download("^VIX", start=START, end=END,
                      auto_adjust=True, progress=False)
_vc = vix_raw["Close"]
if isinstance(_vc, pd.DataFrame):
    _vc = _vc.iloc[:, 0]
vix = pd.Series(_vc.values, index=_vc.index, dtype=float).dropna()
print(f"✅ VIX {len(vix)}일")

print("📥 종목 OHLCV 다운로드 중...")
stock_data = {}
for i, tk in enumerate(TICKERS, 1):
    d = dl_ohlcv(tk, START, END)
    if len(d) < 250:
        continue
    d = compute(d)
    stock_data[tk] = d
    if i % 10 == 0:
        print(f"  {i}/{len(TICKERS)} 완료")
print(f"✅ {len(stock_data)}개 종목 OHLCV 로드")

print("📥 분기 성장률 데이터 수집 중...")
growth_data = {}
for i, tk in enumerate(list(stock_data.keys()), 1):
    gdf = get_quarterly_growth(tk)
    if not gdf.empty:
        price_idx = stock_data[tk].index
        growth_data[tk] = {
            "df"    : gdf,
            "flags" : build_growth_signal_series(gdf, price_idx)
        }
    if i % 10 == 0:
        print(f"  {i}/{len(stock_data)} 완료")
print(f"✅ {len(growth_data)}개 종목 성장률 데이터 로드")


# ──────────────────────────────────────────────
# 신호 생성
# ──────────────────────────────────────────────
print("🔍 매수 신호 생성 중...")
signals_by_date = {}

for tk, d in stock_data.items():
    d_c   = d.dropna(subset=["MA200", "RSI", "CCI"])
    close = d_c["Close"]
    rsi   = d_c["RSI"]
    cci   = d_c["CCI"]
    common = d_c.index.intersection(vix.index)
    if len(common) < 50:
        continue
    vx = vix.reindex(common)

    cond = (
        (close.reindex(common) < d_c["MA200"].reindex(common)) &
        (vx >= VIX_MIN) &
        ((rsi.reindex(common) < 40) | (cci.reindex(common) < -100))
    )
    sig_days = common[cond.reindex(common).fillna(False)]

    for sig_day in sig_days:
        idx = d_c.index.get_loc(sig_day)
        if idx + 1 >= len(d_c):
            continue
        entry_day  = d_c.index[idx + 1]
        entry_open = float(d_c["Open"].iloc[idx + 1])
        if pd.isna(entry_open):
            continue
        row = d_c.loc[sig_day]
        if entry_day not in signals_by_date:
            signals_by_date[entry_day] = []
        signals_by_date[entry_day].append({
            "ticker"   : tk,
            "sig_day"  : sig_day,
            "entry_day": entry_day,
            "entry"    : entry_open,
            "rsi"      : float(row["RSI"]),
            "cci"      : float(row["CCI"]),
            "close_sig": float(row["Close"]),
        })

final_signals = []
for entry_day, items in sorted(signals_by_date.items()):
    items.sort(key=lambda x: x["rsi"])
    for item in items[:MAX_DAILY]:
        final_signals.append(item)

print(f"✅ 신호 {len(final_signals):,}건 ({len(signals_by_date)}거래일)")


# ──────────────────────────────────────────────
# 시뮬레이션 함수
# ──────────────────────────────────────────────
def run_simulation(signals, use_classic_exits=True, use_growth_exit=True, label=""):
    """
    use_classic_exits: ①②③④ (목표/CB/60일절반/120일)
    use_growth_exit:   ⑤ (성장률 둔화)
    """
    trades        = []
    pos_exit_date = {}

    for sig in signals:
        tk        = sig["ticker"]
        entry_day = sig["entry_day"]
        entry     = sig["entry"]

        active = {t: ed for t, ed in pos_exit_date.items() if ed > entry_day}
        if len(active) >= MAX_POSITIONS:
            continue
        if tk in active:
            continue

        d       = stock_data[tk]
        future  = d.loc[d.index >= entry_day]
        if len(future) == 0:
            continue

        # 성장률 플래그 시리즈
        g_flags = growth_data[tk]["flags"] if tk in growth_data else pd.Series(
            False, index=d.index)

        circuit      = entry * (1 - CIRCUIT_PCT)
        target       = entry * (1 + TARGET_PCT)
        half_exited  = False
        exit_records = []

        for i, (fdt, row) in enumerate(future.iterrows()):
            lo = float(row["Low"])
            hi = float(row["High"])
            cl = float(row["Close"])

            # ⑤ 성장률 매도 — 당일 종가로 탈출
            if use_growth_exit and tk in growth_data:
                growth_flag = bool(g_flags.get(fdt, False))
                if growth_flag:
                    exit_records.append((fdt, cl, 0.5 if half_exited else 1.0, "growth"))
                    break

            if use_classic_exits:
                if hi >= target:
                    exit_records.append((fdt, target, 0.5 if half_exited else 1.0, "target"))
                    break
                if lo <= circuit:
                    exit_records.append((fdt, circuit, 0.5 if half_exited else 1.0, "circuit"))
                    break
                if i + 1 == HALF_EXIT and not half_exited and (cl - entry) / entry > 0:
                    exit_records.append((fdt, cl, 0.5, "half_60d"))
                    half_exited = True
                    continue
                if i + 1 >= MAX_HOLD:
                    exit_records.append((fdt, cl, 0.5 if half_exited else 1.0, "time"))
                    break
            else:
                # 성장률만 사용할 때: 120거래일 타임 익시트는 유지 (무한 보유 방지)
                if i + 1 >= MAX_HOLD:
                    exit_records.append((fdt, cl, 1.0, "time_fallback"))
                    break

        if not exit_records:
            continue

        total_pct   = sum(r[2] for r in exit_records)
        weighted    = sum((r[1] - entry) / entry * r[2] for r in exit_records)
        blended_ret = weighted / total_pct if total_pct > 0 else 0
        last_exit   = exit_records[-1]
        reason      = ("+".join(r[3] for r in exit_records)
                       if len(exit_records) > 1 else exit_records[0][3])

        trades.append({
            "entry_date" : entry_day,
            "exit_date"  : last_exit[0],
            "ticker"     : tk,
            "entry"      : entry,
            "exit_price" : last_exit[1],
            "return_pct" : blended_ret * 100,
            "hold_days"  : (last_exit[0] - entry_day).days,
            "exit_reason": reason,
            "half_exit"  : half_exited,
            "win"        : blended_ret > 0,
            "rsi_entry"  : sig["rsi"],
        })
        pos_exit_date[tk] = last_exit[0]

    return pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)


# ──────────────────────────────────────────────
# 두 그룹 실행
# ──────────────────────────────────────────────
print("\n⚙️  그룹 A (①②③④⑤ 전체) 시뮬레이션 중...")
df_a = run_simulation(final_signals, use_classic_exits=True, use_growth_exit=True,
                      label="GroupA_All5")
print(f"✅ 그룹 A: {len(df_a)}건")

print("⚙️  그룹 B (⑤ 성장률만) 시뮬레이션 중...")
df_b = run_simulation(final_signals, use_classic_exits=False, use_growth_exit=True,
                      label="GroupB_GrowthOnly")
print(f"✅ 그룹 B: {len(df_b)}건")


# ──────────────────────────────────────────────
# 통계 계산 함수
# ──────────────────────────────────────────────
def calc_stats(df):
    if df.empty:
        return {}
    wins   = df[df["win"]]
    losses = df[~df["win"]]
    n      = len(df)
    wr     = len(wins) / n * 100
    ar     = df["return_pct"].mean()
    aw     = wins["return_pct"].mean()   if len(wins)   else 0
    al     = losses["return_pct"].mean() if len(losses) else 0
    pf     = (wins["return_pct"].sum() / -losses["return_pct"].sum()
              if len(losses) and losses["return_pct"].sum() < 0 else np.nan)
    ev     = wr / 100 * aw + (1 - wr / 100) * al

    max_cl = cur = 0
    for r in df["return_pct"]:
        cur = cur + 1 if r < 0 else 0
        max_cl = max(max_cl, cur)

    capital = 1.0
    for _, group in df.groupby("entry_date"):
        w     = 1.0 / max(len(group), MAX_POSITIONS)
        batch = (group["return_pct"] / 100 * w).sum()
        capital *= (1 + batch)
    years = (df["exit_date"].max() - df["entry_date"].min()).days / 365.25
    cagr  = capital ** (1 / max(years, 0.01)) - 1

    # 성장률 조건으로 탈출한 비중
    growth_exit_ratio = (df["exit_reason"].str.contains("growth").sum() / n * 100
                         if "exit_reason" in df.columns else 0)

    return {
        "n": n, "wr": wr, "ar": ar, "aw": aw, "al": al,
        "pf": pf, "ev": ev, "max_cl": max_cl,
        "cagr": cagr * 100,
        "avg_hold": df["hold_days"].mean(),
        "growth_exit_pct": growth_exit_ratio,
        "exit_cnt": df["exit_reason"].value_counts(),
    }


sa = calc_stats(df_a)
sb = calc_stats(df_b)


# ──────────────────────────────────────────────
# 결과 출력
# ──────────────────────────────────────────────
print("\n" + "="*72)
print("  성장률 조건 비교 백테스트 결과")
print("="*72)
print(f"  기간          : {START} ~ {END}")
print(f"  진입 조건     : MA200↓ + VIX≥25 + (RSI<40 OR CCI<-100), 다음날 시가")
print(f"  성장률 조건 ⑤ : SalesQQ + EPS_QQ 동시 하락 → 당일 종가 매도")
print("="*72)

fmt = "  {:<22} {:>20} {:>20}"
print(fmt.format("지표", "그룹A (①②③④⑤ 전체)", "그룹B (⑤만)"))
print("  " + "-"*62)
rows_cmp = [
    ("총 트레이드",       f"{sa['n']}건",              f"{sb['n']}건"),
    ("승 률",             f"{sa['wr']:.1f}%",           f"{sb['wr']:.1f}%"),
    ("평균 수익률",       f"{sa['ar']:+.2f}%",          f"{sb['ar']:+.2f}%"),
    ("기대값(EV)",        f"{sa['ev']:+.2f}%",          f"{sb['ev']:+.2f}%"),
    ("승자 평균",         f"{sa['aw']:+.2f}%",          f"{sb['aw']:+.2f}%"),
    ("패자 평균",         f"{sa['al']:+.2f}%",          f"{sb['al']:+.2f}%"),
    ("Profit Factor",     f"{sa['pf']:.2f}",            f"{sb['pf']:.2f}"),
    ("포트CAGR",          f"{sa['cagr']:+.2f}%",        f"{sb['cagr']:+.2f}%"),
    ("최대 연속 손실",    f"{sa['max_cl']}건",           f"{sb['max_cl']}건"),
    ("평균 보유 일수",    f"{sa['avg_hold']:.0f}일",     f"{sb['avg_hold']:.0f}일"),
    ("⑤ 조건 탈출 비중",  f"{sa['growth_exit_pct']:.1f}%", f"{sb['growth_exit_pct']:.1f}%"),
]
for label, va, vb in rows_cmp:
    print(fmt.format(label, va, vb))
print("="*72)

print("\n그룹 A — 청산 유형:")
for r, c in sa["exit_cnt"].items():
    print(f"  {r:<25}: {c:4d}건 ({c/sa['n']*100:.1f}%)")

print("\n그룹 B — 청산 유형:")
for r, c in sb["exit_cnt"].items():
    print(f"  {r:<25}: {c:4d}건 ({c/sb['n']*100:.1f}%)")

# 연도별 비교
df_a["year"] = df_a["entry_date"].dt.year
df_b["year"] = df_b["entry_date"].dt.year
ya = df_a.groupby("year").agg(trades=("return_pct","count"),
                               avg_ret=("return_pct","mean"),
                               win_rate=("win", lambda x: x.mean()*100))
yb = df_b.groupby("year").agg(trades=("return_pct","count"),
                               avg_ret=("return_pct","mean"),
                               win_rate=("win", lambda x: x.mean()*100))
print("\n연도별 — 그룹 A (①②③④⑤):")
print(ya.round(2).to_string())
print("\n연도별 — 그룹 B (⑤만):")
print(yb.round(2).to_string())


# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.suptitle(
    "성장률 조건(⑤) 추가 비교 백테스트\n"
    f"그룹A(①~⑤): {sa['n']}건 | 승률 {sa['wr']:.1f}% | 평균 {sa['ar']:+.2f}% | CAGR {sa['cagr']:+.1f}%   ||   "
    f"그룹B(⑤만): {sb['n']}건 | 승률 {sb['wr']:.1f}% | 평균 {sb['ar']:+.2f}% | CAGR {sb['cagr']:+.1f}%",
    fontsize=10, fontweight="bold"
)

# ── 수익률 분포 비교
ax = axes[0, 0]
ax.hist(df_a["return_pct"], bins=40, alpha=0.6, color="steelblue",
        edgecolor="white", label=f"그룹A (avg {sa['ar']:+.2f}%)")
ax.hist(df_b["return_pct"], bins=40, alpha=0.6, color="tomato",
        edgecolor="white", label=f"그룹B (avg {sb['ar']:+.2f}%)")
ax.axvline(0, color="black", lw=1.5, linestyle="--")
ax.set_title("수익률 분포 비교")
ax.legend(fontsize=8)

# ── 청산 유형 — 그룹 A
ax = axes[0, 1]
clr = {"target":"#2ecc71","circuit":"#e74c3c","time":"#f39c12",
       "half_60d":"#3498db","growth":"#9b59b6","time_fallback":"#95a5a6"}
eca = sa["exit_cnt"].head(6)
ax.pie(eca.values,
       labels=[f"{k[:15]}\n{v}건({v/sa['n']*100:.0f}%)" for k,v in eca.items()],
       colors=[clr.get(k.split("+")[0], "#bdc3c7") for k in eca.index],
       startangle=140, textprops={"fontsize": 7})
ax.set_title("그룹A 청산 유형")

# ── 청산 유형 — 그룹 B
ax = axes[0, 2]
ecb = sb["exit_cnt"].head(6)
ax.pie(ecb.values,
       labels=[f"{k[:15]}\n{v}건({v/sb['n']*100:.0f}%)" for k,v in ecb.items()],
       colors=[clr.get(k.split("+")[0], "#bdc3c7") for k in ecb.index],
       startangle=140, textprops={"fontsize": 7})
ax.set_title("그룹B 청산 유형")

# ── 누적 수익률 비교
ax = axes[1, 0]
cum_a = (1 + df_a["return_pct"] / 100).cumprod() - 1
cum_b = (1 + df_b["return_pct"] / 100).cumprod() - 1
ax.plot(range(len(cum_a)), cum_a * 100, color="steelblue", lw=1.5,
        label=f"그룹A CAGR {sa['cagr']:+.1f}%")
ax.plot(range(len(cum_b)), cum_b * 100, color="tomato",    lw=1.5,
        label=f"그룹B CAGR {sb['cagr']:+.1f}%")
ax.axhline(0, color="black", linestyle="--", lw=1)
ax.set_title("누적 수익률 비교")
ax.set_xlabel("Trade #")
ax.legend(fontsize=8)

# ── 연도별 평균 수익률 — 그룹 A
ax = axes[1, 1]
bar_c = ["#2ecc71" if v >= 0 else "#e74c3c" for v in ya["avg_ret"]]
ax.bar(ya.index.astype(str), ya["avg_ret"], color=bar_c, edgecolor="white")
ax.axhline(0, color="black", lw=0.8)
for x, (yr, row) in enumerate(ya.iterrows()):
    ax.text(x, row["avg_ret"] + (0.3 if row["avg_ret"] >= 0 else -1.0),
            f'{int(row["trades"])}건', ha="center", fontsize=7)
ax.set_title("그룹A 연도별 평균 수익률")
ax.tick_params(axis="x", rotation=45)

# ── 연도별 평균 수익률 — 그룹 B
ax = axes[1, 2]
bar_c = ["#2ecc71" if v >= 0 else "#e74c3c" for v in yb["avg_ret"]]
ax.bar(yb.index.astype(str), yb["avg_ret"], color=bar_c, edgecolor="white")
ax.axhline(0, color="black", lw=0.8)
for x, (yr, row) in enumerate(yb.iterrows()):
    ax.text(x, row["avg_ret"] + (0.3 if row["avg_ret"] >= 0 else -1.0),
            f'{int(row["trades"])}건', ha="center", fontsize=7)
ax.set_title("그룹B 연도별 평균 수익률")
ax.tick_params(axis="x", rotation=45)

# ── 보유 기간 분포 비교
ax = axes[2, 0]
ax.hist(df_a["hold_days"], bins=30, alpha=0.6, color="steelblue",
        edgecolor="white", label=f"그룹A (avg {sa['avg_hold']:.0f}일)")
ax.hist(df_b["hold_days"], bins=30, alpha=0.6, color="tomato",
        edgecolor="white", label=f"그룹B (avg {sb['avg_hold']:.0f}일)")
ax.set_title("보유 기간 분포 비교")
ax.set_xlabel("Hold Days")
ax.legend(fontsize=8)

# ── 그룹A: ⑤ 조건 발동 시 수익률 vs 미발동 시 수익률
ax = axes[2, 1]
if "exit_reason" in df_a.columns:
    g_exits    = df_a[df_a["exit_reason"].str.contains("growth")]["return_pct"]
    non_g_exits= df_a[~df_a["exit_reason"].str.contains("growth")]["return_pct"]
    data_vals  = [g_exits.values, non_g_exits.values]
    bp = ax.boxplot(data_vals, labels=["⑤ 발동", "⑤ 미발동"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#9b59b6")
    bp["boxes"][1].set_facecolor("#3498db")
    ax.axhline(0, color="red", linestyle="--", lw=1)
    ax.set_title(f"그룹A: ⑤ 발동({len(g_exits)}건) vs 미발동({len(non_g_exits)}건)")
    ax.set_ylabel("Return %")

# ── 핵심 지표 비교 막대
ax = axes[2, 2]
metrics   = ["승률(%)", "평균수익(%)", "CAGR(%)", "평균보유(일/10)"]
vals_a    = [sa["wr"], sa["ar"], sa["cagr"], sa["avg_hold"] / 10]
vals_b    = [sb["wr"], sb["ar"], sb["cagr"], sb["avg_hold"] / 10]
x         = np.arange(len(metrics))
width     = 0.35
ax.bar(x - width/2, vals_a, width, label="그룹A", color="steelblue", alpha=0.8)
ax.bar(x + width/2, vals_b, width, label="그룹B", color="tomato",    alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("핵심 지표 비교")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("backtest_growth_comparison.png", dpi=150, bbox_inches="tight")
df_a.to_csv("backtest_growth_groupA.csv", index=False)
df_b.to_csv("backtest_growth_groupB.csv", index=False)
print("\n📊 backtest_growth_comparison.png 저장 완료")
print("📄 backtest_growth_groupA.csv / groupB.csv 저장 완료")
print("✅ 성장률 비교 백테스트 완료")
