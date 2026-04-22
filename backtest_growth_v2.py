"""
backtest_growth_v2.py — 성장률 조건 3그룹 최종 비교
=====================================================

그룹 O (원본):  기존 ①②③④만
그룹 A (엄격):  ①②③④ + ⑤ 엄격 (SalesQQ AND EPS_QQ 동시 하락)
그룹 C (완화):  ①②③④ + ⑤ 완화 (SalesQQ OR  EPS_QQ 하나만 하락해도 트리거)

성장률 매도 조건:
  엄격 ⑤: prev_SalesQQ → curr_SalesQQ 하락  AND  prev_EPS_QQ → curr_EPS_QQ 하락
  완화 ⑤: prev_SalesQQ → curr_SalesQQ 하락  OR   prev_EPS_QQ → curr_EPS_QQ 하락

음수 성장률 처리 (공통):
  이전값 < 0 이고 현재값 > 이전값 (개선 중) → 하락 아님
  이전값 < 0 이고 현재값 < 이전값 (더 악화)  → 하락 으로 처리

데이터 소스:
  - 가격/VIX: yfinance
  - 분기 실적: yfinance quarterly_income_stmt (Revenue, Basic/Diluted EPS)
  - 발표일 가정: 분기 종료 후 45일 (보수적)
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
START         = "2010-01-01"
END           = "2026-01-01"
VIX_MIN       = 25
TARGET_PCT    = 0.20
CIRCUIT_PCT   = 0.25
HALF_EXIT     = 60
MAX_HOLD      = 120
MAX_POSITIONS = 5
MAX_DAILY     = 5

TICKERS = sorted(set([
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","COST",
    "AMD","QCOM","INTC","TXN","AMGN","INTU","AMAT","MU","LRCX","KLAC",
    "CDNS","SNPS","FTNT","PANW","MNST","ORLY","ISRG","PAYX","MELI",
    "PLTR","CPRT","NXPI","ON","CSX","ROP","ADP","ADI","BKNG",
    "MDLZ","AZN","FAST","MCHP",
]))

# ──────────────────────────────────────────────
# OHLCV + 기술 지표
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
# 분기 성장률 수집
# ──────────────────────────────────────────────
def get_quarterly_growth(ticker):
    try:
        tk      = yf.Ticker(ticker)
        qincome = tk.quarterly_income_stmt
        if qincome is None or qincome.empty:
            return pd.DataFrame()
        qincome = qincome.sort_index(axis=1)

        revenue_row = next(
            (idx for idx in qincome.index
             if "Total Revenue" in str(idx) or "Revenue" in str(idx)), None)
        if revenue_row is None:
            return pd.DataFrame()
        revenue = qincome.loc[revenue_row].dropna()

        eps_row = next(
            (idx for idx in qincome.index
             if "Basic EPS" in str(idx) or "Diluted EPS" in str(idx)), None)
        if eps_row is None:
            eps_row = next(
                (idx for idx in qincome.index if "Net Income" in str(idx)), None)
        if eps_row is None:
            return pd.DataFrame()
        eps = qincome.loc[eps_row].dropna()

        common = revenue.index.intersection(eps.index)
        if len(common) < 3:
            return pd.DataFrame()

        revenue = revenue.reindex(common).sort_index()
        eps     = eps.reindex(common).sort_index()

        rows = []
        for i in range(1, len(revenue)):
            q_date   = revenue.index[i]
            rev_curr = float(revenue.iloc[i])
            rev_prev = float(revenue.iloc[i - 1])
            eps_curr = float(eps.iloc[i])
            eps_prev = float(eps.iloc[i - 1])

            sales_qq = ((rev_curr - rev_prev) / abs(rev_prev) * 100
                        if rev_prev != 0 and not np.isnan(rev_prev) and not np.isnan(rev_curr)
                        else np.nan)
            eps_qq   = ((eps_curr - eps_prev) / abs(eps_prev) * 100
                        if eps_prev != 0 and not np.isnan(eps_prev) and not np.isnan(eps_curr)
                        else np.nan)

            report_date = pd.Timestamp(q_date) + pd.Timedelta(days=45)
            rows.append({
                "quarter_end": pd.Timestamp(q_date),
                "report_date": report_date,
                "salesQQ"    : sales_qq,
                "epsQQ"      : eps_qq,
            })
        return pd.DataFrame(rows).sort_values("report_date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def is_declining(curr, prev):
    """
    성장률 하락 여부 판단.
    이전값 < 0 이고 현재값 > 이전값 (개선 중) → False
    그 외 현재 < 이전 → True
    """
    if any(np.isnan(v) for v in [curr, prev]):
        return False
    if prev < 0 and curr > prev:   # 음수 개선 중
        return False
    return curr < prev


def build_growth_flags(growth_df, price_index, mode="strict"):
    """
    mode='strict': SalesQQ AND EPS_QQ 동시 하락
    mode='loose' : SalesQQ OR  EPS_QQ 하나만 하락
    """
    flag = pd.Series(False, index=price_index)
    if growth_df.empty or len(growth_df) < 2:
        return flag

    for i in range(1, len(growth_df)):
        curr = growth_df.iloc[i]
        prev = growth_df.iloc[i - 1]

        sales_down = is_declining(curr["salesQQ"], prev["salesQQ"])
        eps_down   = is_declining(curr["epsQQ"],   prev["epsQQ"])

        if mode == "strict":
            triggered = sales_down and eps_down
        else:  # loose
            triggered = sales_down or eps_down

        start_d = curr["report_date"]
        end_d   = (growth_df.iloc[i + 1]["report_date"]
                   if i + 1 < len(growth_df)
                   else price_index[-1] + pd.Timedelta(days=1))

        mask = (price_index >= start_d) & (price_index < end_d)
        flag[mask] = triggered

    return flag


# ──────────────────────────────────────────────
# 데이터 다운로드
# ──────────────────────────────────────────────
print("📥 VIX 다운로드 중...")
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
print(f"✅ {len(stock_data)}개 종목 로드")

print("📥 분기 성장률 수집 중...")
growth_strict = {}   # mode=strict 플래그
growth_loose  = {}   # mode=loose  플래그
for i, tk in enumerate(list(stock_data.keys()), 1):
    gdf = get_quarterly_growth(tk)
    if not gdf.empty:
        pidx = stock_data[tk].index
        growth_strict[tk] = build_growth_flags(gdf, pidx, mode="strict")
        growth_loose[tk]  = build_growth_flags(gdf, pidx, mode="loose")
    if i % 10 == 0:
        print(f"  {i}/{len(stock_data)} 완료")
print(f"✅ {len(growth_strict)}개 종목 성장률 플래그 생성")


# ──────────────────────────────────────────────
# 매수 신호 생성
# ──────────────────────────────────────────────
print("🔍 매수 신호 생성 중...")
signals_by_date = {}

for tk, d in stock_data.items():
    d_c    = d.dropna(subset=["MA200", "RSI", "CCI"])
    common = d_c.index.intersection(vix.index)
    if len(common) < 50:
        continue
    vx = vix.reindex(common)

    cond = (
        (d_c["Close"].reindex(common)  < d_c["MA200"].reindex(common)) &
        (vx >= VIX_MIN) &
        ((d_c["RSI"].reindex(common) < 40) | (d_c["CCI"].reindex(common) < -100))
    )
    for sig_day in common[cond.reindex(common).fillna(False)]:
        idx = d_c.index.get_loc(sig_day)
        if idx + 1 >= len(d_c):
            continue
        entry_day  = d_c.index[idx + 1]
        entry_open = float(d_c["Open"].iloc[idx + 1])
        if pd.isna(entry_open):
            continue
        row = d_c.loc[sig_day]
        signals_by_date.setdefault(entry_day, []).append({
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
    final_signals.extend(items[:MAX_DAILY])
print(f"✅ 신호 {len(final_signals):,}건 ({len(signals_by_date)}거래일)")


# ──────────────────────────────────────────────
# 시뮬레이션
# ──────────────────────────────────────────────
def run_simulation(signals, growth_flags_map=None, label=""):
    """
    growth_flags_map: None → 성장률 조건 없음 (원본)
                      dict → 해당 플래그 시리즈 사용
    """
    trades        = []
    pos_exit_date = {}

    for sig in signals:
        tk        = sig["ticker"]
        entry_day = sig["entry_day"]
        entry     = sig["entry"]

        active = {t: ed for t, ed in pos_exit_date.items() if ed > entry_day}
        if len(active) >= MAX_POSITIONS or tk in active:
            continue

        d      = stock_data[tk]
        future = d.loc[d.index >= entry_day]
        if len(future) == 0:
            continue

        g_flags = (growth_flags_map.get(tk, pd.Series(False, index=d.index))
                   if growth_flags_map is not None else None)

        circuit     = entry * (1 - CIRCUIT_PCT)
        target      = entry * (1 + TARGET_PCT)
        half_exited = False
        exit_records= []

        for i, (fdt, row) in enumerate(future.iterrows()):
            lo = float(row["Low"])
            hi = float(row["High"])
            cl = float(row["Close"])

            # ⑤ 성장률 조건 (있는 경우)
            if g_flags is not None and bool(g_flags.get(fdt, False)):
                exit_records.append((fdt, cl, 0.5 if half_exited else 1.0, "growth"))
                break

            # ①②③④ 기존 조건
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

        if not exit_records:
            continue

        total_pct   = sum(r[2] for r in exit_records)
        blended_ret = sum((r[1] - entry) / entry * r[2] for r in exit_records) / total_pct
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
            "win"        : blended_ret > 0,
            "rsi_entry"  : sig["rsi"],
        })
        pos_exit_date[tk] = last_exit[0]

    df = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)
    print(f"✅ [{label}] {len(df)}건")
    return df


print("\n⚙️  시뮬레이션 실행 중...")
df_o = run_simulation(final_signals, growth_flags_map=None,          label="원본 ①②③④")
df_a = run_simulation(final_signals, growth_flags_map=growth_strict,  label="엄격 ①②③④⑤-AND")
df_c = run_simulation(final_signals, growth_flags_map=growth_loose,   label="완화 ①②③④⑤-OR")


# ──────────────────────────────────────────────
# 통계
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

    growth_n = (df["exit_reason"].str.contains("growth").sum()
                if "exit_reason" in df.columns else 0)

    return {
        "n": n, "wr": wr, "ar": ar, "aw": aw, "al": al,
        "pf": pf, "ev": ev, "max_cl": max_cl,
        "cagr": cagr * 100,
        "avg_hold": df["hold_days"].mean(),
        "growth_n": growth_n,
        "growth_pct": growth_n / n * 100,
        "exit_cnt": df["exit_reason"].value_counts(),
    }


so = calc_stats(df_o)
sa = calc_stats(df_a)
sc = calc_stats(df_c)


# ──────────────────────────────────────────────
# 출력
# ──────────────────────────────────────────────
print("\n" + "="*80)
print("  성장률 조건 3그룹 최종 비교 백테스트")
print("="*80)
print(f"  기간       : {START} ~ {END}")
print(f"  진입 조건  : MA200↓ + VIX≥25 + (RSI<40 OR CCI<-100), 다음날 시가")
print(f"  ⑤ 엄격    : SalesQQ AND EPS_QQ 동시 하락 → 당일 종가 매도")
print(f"  ⑤ 완화    : SalesQQ OR  EPS_QQ 하나만 하락 → 당일 종가 매도")
print("="*80)

fmt  = "  {:<22} {:>17} {:>17} {:>17}"
hdr  = fmt.format("지표", "원본(①②③④)", "엄격(①~⑤-AND)", "완화(①~⑤-OR)")
print(hdr)
print("  " + "-"*73)

def pf_str(v): return f"{v:.2f}" if not (isinstance(v, float) and np.isnan(v)) else "N/A"

rows = [
    ("총 트레이드",      f"{so['n']}건",              f"{sa['n']}건",              f"{sc['n']}건"),
    ("승 률",            f"{so['wr']:.1f}%",           f"{sa['wr']:.1f}%",           f"{sc['wr']:.1f}%"),
    ("평균 수익률",      f"{so['ar']:+.2f}%",          f"{sa['ar']:+.2f}%",          f"{sc['ar']:+.2f}%"),
    ("기대값(EV)",       f"{so['ev']:+.2f}%",          f"{sa['ev']:+.2f}%",          f"{sc['ev']:+.2f}%"),
    ("승자 평균",        f"{so['aw']:+.2f}%",          f"{sa['aw']:+.2f}%",          f"{sc['aw']:+.2f}%"),
    ("패자 평균",        f"{so['al']:+.2f}%",          f"{sa['al']:+.2f}%",          f"{sc['al']:+.2f}%"),
    ("Profit Factor",    pf_str(so['pf']),             pf_str(sa['pf']),             pf_str(sc['pf'])),
    ("포트CAGR",         f"{so['cagr']:+.2f}%",        f"{sa['cagr']:+.2f}%",        f"{sc['cagr']:+.2f}%"),
    ("최대 연속 손실",   f"{so['max_cl']}건",           f"{sa['max_cl']}건",           f"{sc['max_cl']}건"),
    ("평균 보유 일수",   f"{so['avg_hold']:.0f}일",     f"{sa['avg_hold']:.0f}일",     f"{sc['avg_hold']:.0f}일"),
    ("⑤ 탈출 건수",      "-",                           f"{sa['growth_n']}건",         f"{sc['growth_n']}건"),
    ("⑤ 탈출 비중",      "-",                           f"{sa['growth_pct']:.1f}%",    f"{sc['growth_pct']:.1f}%"),
]
for row in rows:
    print(fmt.format(*row))
print("="*80)

for label, s, df in [("원본", so, df_o), ("엄격", sa, df_a), ("완화", sc, df_c)]:
    print(f"\n[{label}] 청산 유형:")
    for r, c in s["exit_cnt"].items():
        mark = " ← ⑤" if "growth" in r else ""
        print(f"  {r:<30}: {c:4d}건 ({c/s['n']*100:.1f}%){mark}")

# 연도별
for label, df in [("원본 ①②③④", df_o), ("엄격 ①~⑤-AND", df_a), ("완화 ①~⑤-OR", df_c)]:
    df = df.copy()
    df["year"] = df["entry_date"].dt.year
    y = df.groupby("year").agg(
        trades  =("return_pct","count"),
        avg_ret =("return_pct","mean"),
        win_rate=("win", lambda x: x.mean()*100),
    )
    print(f"\n연도별 [{label}]:")
    print(y.round(2).to_string())


# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(21, 17))
fig.suptitle(
    "성장률 조건 3그룹 최종 비교\n"
    f"원본(①②③④): {so['n']}건 | 승률 {so['wr']:.1f}% | 평균 {so['ar']:+.2f}% | CAGR {so['cagr']:+.1f}%\n"
    f"엄격(AND ⑤): {sa['n']}건 | 승률 {sa['wr']:.1f}% | 평균 {sa['ar']:+.2f}% | CAGR {sa['cagr']:+.1f}%   "
    f"완화(OR  ⑤): {sc['n']}건 | 승률 {sc['wr']:.1f}% | 평균 {sc['ar']:+.2f}% | CAGR {sc['cagr']:+.1f}%",
    fontsize=9, fontweight="bold"
)

COLORS = {"원본":"#3498db", "엄격":"#2ecc71", "완화":"#e67e22"}
clr_exit = {"target":"#27ae60","circuit":"#e74c3c","time":"#f39c12",
            "half_60d":"#3498db","growth":"#9b59b6",
            "half_60d+target":"#1abc9c","half_60d+time":"#e67e22",
            "time_fallback":"#95a5a6"}

def get_clr(key):
    for k, v in clr_exit.items():
        if k in key:
            return v
    return "#bdc3c7"

# ── [0,0] 수익률 분포 3개 겹치기
ax = axes[0, 0]
for label, df, c in [("원본", df_o, COLORS["원본"]),
                      ("엄격", df_a, COLORS["엄격"]),
                      ("완화", df_c, COLORS["완화"])]:
    s = calc_stats(df)
    ax.hist(df["return_pct"], bins=35, alpha=0.5, color=c, edgecolor="white",
            label=f"{label} (avg {s['ar']:+.2f}%)")
ax.axvline(0, color="black", lw=1.5, linestyle="--")
ax.set_title("수익률 분포 비교 (3그룹)")
ax.legend(fontsize=8)

# ── [0,1] 청산 유형 — 원본
ax = axes[0, 1]
eca = so["exit_cnt"].head(7)
ax.pie(eca.values,
       labels=[f"{k[:18]}\n{v}건({v/so['n']*100:.0f}%)" for k,v in eca.items()],
       colors=[get_clr(k) for k in eca.index],
       startangle=140, textprops={"fontsize": 7})
ax.set_title("원본(①②③④) 청산 유형")

# ── [0,2] 청산 유형 — 완화
ax = axes[0, 2]
ecc = sc["exit_cnt"].head(7)
ax.pie(ecc.values,
       labels=[f"{k[:18]}\n{v}건({v/sc['n']*100:.0f}%)" for k,v in ecc.items()],
       colors=[get_clr(k) for k in ecc.index],
       startangle=140, textprops={"fontsize": 7})
ax.set_title("완화(①~⑤-OR) 청산 유형")

# ── [1,0] 누적 수익률 비교
ax = axes[1, 0]
for label, df, c in [("원본", df_o, COLORS["원본"]),
                      ("엄격", df_a, COLORS["엄격"]),
                      ("완화", df_c, COLORS["완화"])]:
    cum = (1 + df["return_pct"] / 100).cumprod() - 1
    s   = calc_stats(df)
    ax.plot(range(len(cum)), cum * 100, color=c, lw=1.5,
            label=f"{label} CAGR {s['cagr']:+.1f}%")
ax.axhline(0, color="black", linestyle="--", lw=1)
ax.set_title("누적 수익률 비교")
ax.set_xlabel("Trade #")
ax.legend(fontsize=8)

# ── [1,1] 연도별 평균 수익률 — 원본
for col_idx, (label, df) in enumerate([("원본 ①②③④", df_o),
                                        ("완화 ①~⑤-OR", df_c)]):
    ax = axes[1, col_idx + 1]
    df2 = df.copy(); df2["year"] = df2["entry_date"].dt.year
    y   = df2.groupby("year").agg(
        trades  =("return_pct","count"),
        avg_ret =("return_pct","mean"),
    )
    bar_c = ["#2ecc71" if v >= 0 else "#e74c3c" for v in y["avg_ret"]]
    ax.bar(y.index.astype(str), y["avg_ret"], color=bar_c, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    for x, (yr, row) in enumerate(y.iterrows()):
        ax.text(x, row["avg_ret"] + (0.5 if row["avg_ret"] >= 0 else -1.5),
                f'{int(row["trades"])}', ha="center", fontsize=7)
    ax.set_title(f"{label} 연도별 평균 수익률")
    ax.tick_params(axis="x", rotation=45)

# ── [2,0] 보유 기간 분포
ax = axes[2, 0]
for label, df, c in [("원본", df_o, COLORS["원본"]),
                      ("엄격", df_a, COLORS["엄격"]),
                      ("완화", df_c, COLORS["완화"])]:
    s = calc_stats(df)
    ax.hist(df["hold_days"], bins=30, alpha=0.5, color=c, edgecolor="white",
            label=f"{label} (avg {s['avg_hold']:.0f}일)")
ax.set_title("보유 기간 분포 비교")
ax.set_xlabel("Hold Days")
ax.legend(fontsize=8)

# ── [2,1] ⑤ 발동/미발동 수익률 박스플롯 (완화 기준)
ax = axes[2, 1]
g_exits     = df_c[df_c["exit_reason"].str.contains("growth")]["return_pct"]
non_g_exits = df_c[~df_c["exit_reason"].str.contains("growth")]["return_pct"]
if len(g_exits) > 0:
    bp = ax.boxplot([g_exits.values, non_g_exits.values],
                    labels=[f"⑤ 발동\n({len(g_exits)}건)", f"⑤ 미발동\n({len(non_g_exits)}건)"],
                    patch_artist=True)
    bp["boxes"][0].set_facecolor("#9b59b6")
    bp["boxes"][1].set_facecolor(COLORS["완화"])
    ax.axhline(0, color="red", linestyle="--", lw=1)
ax.set_title("완화: ⑤ 발동 vs 미발동 수익률")
ax.set_ylabel("Return %")

# ── [2,2] 핵심 지표 비교 막대
ax = axes[2, 2]
metrics = ["승률(%)", "평균수익(%)", "CAGR(%)"]
vals_o  = [so["wr"], so["ar"], so["cagr"]]
vals_a  = [sa["wr"], sa["ar"], sa["cagr"]]
vals_c  = [sc["wr"], sc["ar"], sc["cagr"]]
x = np.arange(len(metrics))
w = 0.25
ax.bar(x - w,   vals_o, w, label="원본",  color=COLORS["원본"],  alpha=0.85)
ax.bar(x,       vals_a, w, label="엄격",  color=COLORS["엄격"],  alpha=0.85)
ax.bar(x + w,   vals_c, w, label="완화",  color=COLORS["완화"],  alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("핵심 지표 비교 (3그룹)")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("backtest_growth_v2_final.png", dpi=150, bbox_inches="tight")
for label, df in [("original", df_o), ("strict_and", df_a), ("loose_or", df_c)]:
    df.to_csv(f"backtest_growth_v2_{label}.csv", index=False)
print("\n📊 backtest_growth_v2_final.png 저장 완료")
print("📄 CSV 3개 저장 완료")
print("✅ 3그룹 최종 비교 백테스트 완료")
