"""
backtest_trendline.py
======================
A/B/C 전략 × 하단 추세선 터치 필터 백테스트

[개요]
  기존 A(squeeze)/B(ma200u)/C(ma200d) 진입 조건에
  하단 추세선 근접(터치) 필터를 추가했을 때의 성과를 비교

[하단 추세선 정의]
  최근 N일 저가에 대해 선형 회귀(rolling linear regression)
  → 현재 시점의 추세선 추정값 계산
  → 저가 ≤ 추세선값 × (1 + tolerance) 이면 "터치" 판정
  → 추가 옵션: slope > 0 (상승 추세선) 필터

[비교 케이스 — 전략 3개 × 조건 5개 = 15 조합]
  base       : 추세선 필터 없음 (현재 전략)
  tl_60_3pct : 하단추세선 window=60일, 터치 허용 ±3%
  tl_120_3pct: 하단추세선 window=120일, 터치 허용 ±3%
  tl_60_3pct_up : window=60, ±3%, 상승 추세선만
  tl_120_3pct_up: window=120, ±3%, 상승 추세선만

[손절 / 목표]
  A(squeeze) / B(ma200u): 목표 +8%, 손절 -25%
  C(ma200d)             : 목표 +20%, 손절 -25%
  60거래일 & 수익 중 → 매도 / 120거래일 최대 보유

[종목] backtest_stoploss_compare.py 와 동일 50개
[기간] 2015-01-01 ~ 2026-03-27
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings("ignore")

# ── 기간 ─────────────────────────────────────────────────────────────────────
START = "2015-01-01"
END   = "2026-03-27"

# ── 전략 파라미터 ─────────────────────────────────────────────────────────────
TARGET_SQUEEZE  = 0.08
TARGET_MA200U   = 0.08
TARGET_MA200D   = 0.20
CIRCUIT_PCT     = 0.25
HALF_EXIT_DAYS  = 60
MAX_HOLD_DAYS   = 120

VIX_MIN         = 25
RSI_MAX         = 40
CCI_MIN         = -100
BB_PERIOD       = 20
BB_STD          = 2.0
SQUEEZE_PERIOD  = 60
SQUEEZE_RATIO   = 0.5
PCTB_LOW_MA200U = 5
PCTB_LOW_SQZ    = 50

# ── 추세선 파라미터 ───────────────────────────────────────────────────────────
TL_CONFIGS = {
    "base":           {"window": None,  "tol": None,  "up_only": False},
    "tl_60_3pct":     {"window": 60,    "tol": 0.03,  "up_only": False},
    "tl_120_3pct":    {"window": 120,   "tol": 0.03,  "up_only": False},
    "tl_60_3pct_up":  {"window": 60,    "tol": 0.03,  "up_only": True},
    "tl_120_3pct_up": {"window": 120,   "tol": 0.03,  "up_only": True},
}

# ── 종목 ─────────────────────────────────────────────────────────────────────
KR_TICKERS = [
    "000660.KS",  # SK하이닉스
    "005930.KS",  # 삼성전자
    "277810.KS",  # 레인보우로보틱스
    "034020.KS",  # 두산에너빌리티
    "015760.KS",  # 한국전력
    "005380.KS",  # 현대차
    "012450.KS",  # 한화에어로스페이스
    "042660.KS",  # 한화오션
    "042700.KQ",  # 한미반도체
    "096770.KS",  # SK이노베이션
    "009150.KS",  # 삼성전기
    "000270.KS",  # 기아
    "247540.KQ",  # 에코프로
    "376900.KQ",  # 로킷헬스케어
    "079550.KS",  # LIG넥스원
]
US_TICKERS = [
    "HOOD", "AVGO", "AMD", "MSFT", "GOOGL", "NVDA", "TSLA",
    "MU", "LRCX", "ON", "SNDK", "ASTS", "AVAV", "IONQ",
    "RKLB", "PLTR", "APP", "SOXL", "TSLL", "TE", "ONDS",
    "BE", "PL", "VRT", "LITE", "TER", "ANET",
    "IREN", "HOOG", "SOLT", "ETHU", "NBIS", "LPTH",
    "CONL", "GLW", "FLNC", "VST", "ASX", "SGML",
]
ALL_TICKERS = KR_TICKERS + US_TICKERS

# ── 한글 폰트 ─────────────────────────────────────────────────────────────────
def get_kr_font():
    for c in ["AppleGothic", "NanumGothic", "Malgun Gothic"]:
        if c in {f.name for f in fm.fontManager.ttflist}:
            return c
    return None

KR_FONT = get_kr_font()
if KR_FONT:
    plt.rcParams["font.family"] = KR_FONT
plt.rcParams["axes.unicode_minus"] = False


# ── 롤링 선형 회귀 추세선 ──────────────────────────────────────────────────────
def calc_rolling_trendline(series: pd.Series, window: int):
    """
    각 시점에서 과거 window일 데이터에 대한 선형 회귀 추세선값과 기울기를 반환.
    반환: (trendline_values, slopes) — 모두 pd.Series
    """
    vals = series.values.astype(float)
    n = len(vals)
    tl_vals  = np.full(n, np.nan)
    slopes   = np.full(n, np.nan)

    x     = np.arange(window, dtype=float)
    xmean = x.mean()
    xvar  = ((x - xmean) ** 2).sum()

    for i in range(window - 1, n):
        y = vals[i - window + 1: i + 1]
        if np.isnan(y).any():
            continue
        ymean = y.mean()
        slope = ((x - xmean) * (y - ymean)).sum() / xvar
        intercept = ymean - slope * xmean
        # window-1 인덱스 = 현재 시점
        tl_vals[i] = intercept + slope * (window - 1)
        slopes[i]  = slope

    return (
        pd.Series(tl_vals, index=series.index),
        pd.Series(slopes,  index=series.index),
    )


# ── 지표 계산 ─────────────────────────────────────────────────────────────────
def calc_indicators(df: pd.DataFrame, vix: pd.Series, tl_window: int | None) -> pd.DataFrame:
    df    = df.copy()
    close = df["Close"]
    low   = df["Low"]

    # MA200
    df["ma200"] = close.rolling(200).mean()

    # BB (20일)
    ma20     = close.rolling(BB_PERIOD).mean()
    std20    = close.rolling(BB_PERIOD).std()
    bb_upper = ma20 + BB_STD * std20
    bb_lower = ma20 - BB_STD * std20
    bb_range = bb_upper - bb_lower

    df["bb_width"]     = np.where(ma20 > 0, (bb_upper - bb_lower) / ma20 * 100, np.nan)
    df["bb_width_avg"] = df["bb_width"].rolling(SQUEEZE_PERIOD).mean()
    df["squeeze"]      = df["bb_width"] < df["bb_width_avg"] * SQUEEZE_RATIO
    df["pctb_low"]     = np.where(bb_range > 0, (low - bb_lower) / bb_range * 100, np.nan)

    # RSI(14)
    delta     = close.diff()
    gain      = delta.clip(lower=0).rolling(14).mean()
    loss      = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + np.where(loss == 0, 100.0, gain / loss))

    # CCI(14)
    tp       = (df["High"] + df["Low"] + close) / 3
    tp_ma    = tp.rolling(14).mean()
    tp_md    = tp.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = np.where(tp_md > 0, (tp - tp_ma) / (0.015 * tp_md), 0)

    # VIX 병합
    df = df.join(vix, how="left")
    df["vix"] = df["vix"].ffill()

    # 하단 추세선 (롤링 선형 회귀)
    if tl_window is not None:
        tl_vals, tl_slopes = calc_rolling_trendline(low, tl_window)
        df["tl_lower"]       = tl_vals
        df["tl_lower_slope"] = tl_slopes
    else:
        df["tl_lower"]       = np.nan
        df["tl_lower_slope"] = np.nan

    return df


# ── 데이터 다운로드 ────────────────────────────────────────────────────────────
def download_all(tickers, vix, tl_window):
    print(f"  종목 다운로드 중... ({len(tickers)}개)")
    try:
        raw = yf.download(
            tickers, start=START, end=END,
            auto_adjust=True, progress=False, group_by="ticker"
        )
    except Exception as e:
        print(f"  일괄 다운로드 실패: {e}")
        raw = None

    result = {}
    for t in tickers:
        try:
            df = raw[t].copy() if (raw is not None and len(tickers) > 1) else (
                raw.copy() if raw is not None
                else yf.download(t, start=START, end=END, auto_adjust=True, progress=False)
            )
            df.dropna(how="all", inplace=True)
            if len(df) < 250:
                continue
            df = calc_indicators(df, vix, tl_window)
            result[t] = df
        except Exception as e:
            print(f"    [{t}] 오류: {e}")

    print(f"  → {len(result)}개 종목 로드 완료")
    return result


# ── 추세선 터치 판정 ───────────────────────────────────────────────────────────
def is_trendline_touch(low_val, tl_val, tl_slope, tol, up_only):
    """저가가 하단 추세선 ±tol 이내인지 판정."""
    if np.isnan(tl_val) or tl_val <= 0:
        return False
    if up_only and (np.isnan(tl_slope) or tl_slope <= 0):
        return False
    # 터치 = 저가가 추세선 근처 (아래 혹은 최대 tol 위)
    return low_val <= tl_val * (1 + tol)


# ── 백테스트 엔진 ─────────────────────────────────────────────────────────────
def run_backtest(data: dict, strategy: str, tl_name: str,
                 tl_tol, tl_up_only) -> list:
    """
    strategy: "squeeze" / "ma200u" / "ma200d"
    tl_name:  추세선 설정 이름 (결과 레이블용)
    tl_tol:   추세선 터치 허용 비율 (None = 필터 없음)
    tl_up_only: True면 상승 추세선만 허용
    """
    target_pct = {
        "squeeze": TARGET_SQUEEZE,
        "ma200u":  TARGET_MA200U,
        "ma200d":  TARGET_MA200D,
    }[strategy]

    req_cols = ["ma200", "rsi", "cci", "pctb_low", "bb_width_avg", "squeeze", "vix",
                "tl_lower", "tl_lower_slope"]
    trades = []

    for ticker, df in data.items():
        df_c = df.dropna(subset=["ma200", "rsi", "cci", "vix"]).copy()
        if len(df_c) < 50:
            continue

        in_position = False
        entry_price = 0.0
        entry_idx   = 0
        entry_date  = None

        rows     = df_c.to_dict("index")
        idx_list = list(df_c.index)

        for ii, date in enumerate(idx_list):
            r        = rows[date]
            close    = r["Close"]
            low      = r["Low"]
            ma200    = r["ma200"]
            rsi      = r["rsi"]
            cci      = r["cci"]
            pctb_low = r["pctb_low"] if not pd.isna(r["pctb_low"]) else 999
            squeeze  = bool(r["squeeze"]) if not pd.isna(r["squeeze"]) else False
            vix_val  = r["vix"] if not pd.isna(r["vix"]) else 0
            tl_val   = r["tl_lower"]
            tl_slope = r["tl_lower_slope"]

            if in_position:
                hold = ii - entry_idx
                pnl  = (close - entry_price) / entry_price

                reason = None
                if   pnl >= target_pct:                  reason = "목표"
                elif pnl <= -CIRCUIT_PCT:                reason = "손절"
                elif hold >= HALF_EXIT_DAYS and pnl > 0: reason = "60일수익"
                elif hold >= MAX_HOLD_DAYS:              reason = "기간만료"

                if reason:
                    trades.append({
                        "strategy":  strategy,
                        "tl_config": tl_name,
                        "ticker":    ticker,
                        "entry_date": entry_date,
                        "exit_date":  date,
                        "pnl_pct":   round(pnl * 100, 2),
                        "hold_days": hold,
                        "exit_reason": reason,
                    })
                    in_position = False

            if not in_position:
                # 기본 전략 신호 판단
                signal = False
                if strategy == "squeeze":
                    signal = (close > ma200 and squeeze and pctb_low <= PCTB_LOW_SQZ)
                elif strategy == "ma200u":
                    signal = (close > ma200 and pctb_low <= PCTB_LOW_MA200U)
                elif strategy == "ma200d":
                    signal = (close < ma200 and vix_val >= VIX_MIN
                              and (rsi < RSI_MAX or cci < CCI_MIN))

                # 추세선 터치 필터 적용
                if signal and tl_tol is not None:
                    signal = is_trendline_touch(low, tl_val, tl_slope, tl_tol, tl_up_only)

                if signal:
                    in_position = True
                    entry_price = close
                    entry_date  = date
                    entry_idx   = ii

        # 미청산 처리
        if in_position:
            last = df_c.iloc[-1]
            pnl  = (last["Close"] - entry_price) / entry_price
            hold = len(df_c) - 1 - entry_idx
            trades.append({
                "strategy":  strategy,
                "tl_config": tl_name,
                "ticker":    ticker,
                "entry_date": entry_date,
                "exit_date":  df_c.index[-1],
                "pnl_pct":   round(pnl * 100, 2),
                "hold_days": hold,
                "exit_reason": "미청산",
            })

    return trades


# ── 통계 계산 ─────────────────────────────────────────────────────────────────
def calc_stats(trades: list) -> dict:
    if not trades:
        return {"n": 0, "win_rate": 0, "avg_pnl": 0, "median_pnl": 0,
                "ev": 0, "avg_win": 0, "avg_loss": 0,
                "stop_rate": 0, "avg_hold": 0}

    pnls   = [t["pnl_pct"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    stops  = [t for t in trades if t["exit_reason"] == "손절"]
    holds  = [t["hold_days"] for t in trades]

    return {
        "n":          len(trades),
        "win_rate":   round(len(wins) / len(pnls) * 100, 1),
        "avg_pnl":    round(np.mean(pnls), 2),
        "median_pnl": round(np.median(pnls), 2),
        "ev":         round(np.mean(pnls), 2),
        "avg_win":    round(np.mean(wins)   if wins   else 0, 2),
        "avg_loss":   round(np.mean(losses) if losses else 0, 2),
        "stop_rate":  round(len(stops) / len(trades) * 100, 1),
        "avg_hold":   round(np.mean(holds), 1),
    }


# ── 차트: 전략별 추세선 조건 비교 ─────────────────────────────────────────────
def plot_results(summary_df: pd.DataFrame):
    strategies  = ["squeeze", "ma200u", "ma200d"]
    strat_names = {
        "squeeze": "A그룹 (스퀴즈, 목표+8%)",
        "ma200u":  "B그룹 (BB하단, 목표+8%)",
        "ma200d":  "C그룹 (200일하방, 목표+20%)",
    }
    tl_order = ["base", "tl_60_3pct", "tl_120_3pct", "tl_60_3pct_up", "tl_120_3pct_up"]
    tl_labels = {
        "base":           "기존 (필터없음)",
        "tl_60_3pct":     "추세선 60일 ±3%",
        "tl_120_3pct":    "추세선 120일 ±3%",
        "tl_60_3pct_up":  "추세선 60일+상승",
        "tl_120_3pct_up": "추세선 120일+상승",
    }
    colors = ["#2c7bb6", "#fdae61", "#d7191c", "#1a9641", "#a6d96a"]

    metrics = [
        ("win_rate",  "승률 (%)"),
        ("avg_pnl",   "평균 수익률 (%)"),
        ("ev",        "기댓값 EV (%)"),
        ("n",         "거래 횟수"),
        ("stop_rate", "손절 청산율 (%)"),
        ("avg_hold",  "평균 보유일"),
    ]

    fig, axes = plt.subplots(len(strategies), len(metrics),
                             figsize=(24, 14), constrained_layout=True)
    fig.suptitle("A/B/C 전략 × 하단 추세선 터치 필터 비교 (2015–2026)",
                 fontsize=14, fontweight="bold")

    x = np.arange(len(tl_order))
    for ri, strat in enumerate(strategies):
        df_s = summary_df[summary_df["strategy"] == strat].copy()
        for ci, (metric, ylabel) in enumerate(metrics):
            ax = axes[ri][ci]
            vals = []
            for tl in tl_order:
                row = df_s[df_s["tl_config"] == tl]
                vals.append(row[metric].values[0] if len(row) else 0)

            bars = ax.bar(x, vals, color=colors, alpha=0.80, width=0.65, edgecolor="white")

            # base 강조
            bars[0].set_edgecolor("black")
            bars[0].set_linewidth(1.5)

            for bar, val in zip(bars, vals):
                fmt = f"{val:.0f}" if metric == "n" else f"{val:.1f}"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (max(vals) * 0.02 if max(vals) > 0 else 0.5),
                        fmt, ha="center", va="bottom", fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels(
                [tl_labels[t] for t in tl_order],
                fontsize=7, rotation=25, ha="right"
            )
            ax.set_ylabel(ylabel, fontsize=8)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

            if metric == "win_rate":
                ax.axhline(50, color="red", linewidth=0.8, linestyle=":", alpha=0.6)

            if ci == 0:
                ax.set_title(f"{strat_names[strat]}\n{ylabel}", fontsize=8, fontweight="bold")
            else:
                ax.set_title(ylabel, fontsize=8)

    from matplotlib.patches import Patch
    legend_el = [
        Patch(facecolor=colors[i], label=tl_labels[tl], alpha=0.8)
        for i, tl in enumerate(tl_order)
    ]
    legend_el[0].set_edgecolor("black")
    legend_el[0].set_linewidth(1.5)
    fig.legend(handles=legend_el, loc="lower right", fontsize=9, ncol=5,
               bbox_to_anchor=(0.99, 0.01))

    plt.savefig("backtest_trendline_main.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → backtest_trendline_main.png 저장")


# ── 추세선 개선 효과 상세 출력 ────────────────────────────────────────────────
def print_improvement(summary_df: pd.DataFrame):
    strategies = ["squeeze", "ma200u", "ma200d"]
    tl_configs = ["tl_60_3pct", "tl_120_3pct", "tl_60_3pct_up", "tl_120_3pct_up"]
    tl_short   = {
        "tl_60_3pct":     "60일/±3%",
        "tl_120_3pct":    "120일/±3%",
        "tl_60_3pct_up":  "60일/상승",
        "tl_120_3pct_up": "120일/상승",
    }

    print("\n" + "=" * 90)
    print("  추세선 필터 추가 시 성과 변화 (vs base)")
    print("=" * 90)
    hdr = f"{'전략':<10} {'추세선':>12}  {'거래↓':>6}  {'승률Δ':>7}  {'EV Δ':>7}  {'평균수익Δ':>9}  {'손절율Δ':>8}"
    print(hdr)
    print("-" * 90)

    for strat in strategies:
        base = summary_df[(summary_df["strategy"] == strat) &
                          (summary_df["tl_config"] == "base")].iloc[0]
        for tl in tl_configs:
            row = summary_df[(summary_df["strategy"] == strat) &
                             (summary_df["tl_config"] == tl)]
            if row.empty:
                continue
            row = row.iloc[0]
            d_n    = int(row["n"]) - int(base["n"])
            d_wr   = row["win_rate"]  - base["win_rate"]
            d_ev   = row["ev"]        - base["ev"]
            d_avg  = row["avg_pnl"]   - base["avg_pnl"]
            d_stop = row["stop_rate"] - base["stop_rate"]

            marker = " ★" if (d_ev > 0 and d_wr > 0) else ("  △" if d_ev > 0 else "")
            print(f"  {strat:<10} {tl_short[tl]:>12}  "
                  f"{d_n:>+6d}  {d_wr:>+7.1f}%  {d_ev:>+7.2f}%  "
                  f"{d_avg:>+9.2f}%  {d_stop:>+8.1f}%{marker}")
        print()


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  A/B/C 전략 × 하단 추세선 터치 필터 백테스트")
    print("=" * 65)

    # 1. VIX 다운로드
    print("\n[1] VIX 다운로드")
    vix_raw = yf.download("^VIX", start=START, end=END,
                          auto_adjust=True, progress=False)
    vix = vix_raw["Close"].squeeze()
    vix.name = "vix"
    print(f"  VIX: {len(vix)}일")

    # 2. 종목 데이터 다운로드 (추세선 window별로 1회씩)
    #    window가 다르면 지표 결과가 다르므로 별도 로드
    unique_windows = sorted(set(
        v["window"] for v in TL_CONFIGS.values() if v["window"] is not None
    ))
    unique_windows = [None] + unique_windows  # None = base용

    print("\n[2] 종목 데이터 다운로드 (window별)")
    data_cache = {}
    for w in unique_windows:
        label = f"window={w}" if w else "base"
        print(f"  {label} ...")
        data_cache[w] = download_all(ALL_TICKERS, vix, w)

    # 3. 백테스트 실행
    print("\n[3] 백테스트 실행")
    strategies = ["squeeze", "ma200u", "ma200d"]
    all_trades = []

    for strat in strategies:
        for tl_name, cfg in TL_CONFIGS.items():
            w        = cfg["window"]
            tl_tol   = cfg["tol"]
            up_only  = cfg["up_only"]
            data     = data_cache[w]

            trades = run_backtest(data, strat, tl_name, tl_tol, up_only)
            all_trades.extend(trades)

            stats = calc_stats(trades)
            print(f"  [{strat:8s}] {tl_name:20s} | "
                  f"거래 {stats['n']:3d}건 | "
                  f"승률 {stats['win_rate']:5.1f}% | "
                  f"평균 {stats['avg_pnl']:+6.2f}% | "
                  f"EV {stats['ev']:+6.2f}%")

    # 4. 요약 테이블
    print("\n[4] 요약 테이블 생성")
    rows = []
    for strat in strategies:
        for tl_name in TL_CONFIGS:
            subset = [t for t in all_trades
                      if t["strategy"] == strat and t["tl_config"] == tl_name]
            stats = calc_stats(subset)
            rows.append({
                "strategy":  strat,
                "tl_config": tl_name,
                **stats,
            })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv("backtest_trendline_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_trades).to_csv(
        "backtest_trendline_trades.csv", index=False, encoding="utf-8-sig"
    )
    print("  → backtest_trendline_summary.csv 저장")
    print("  → backtest_trendline_trades.csv 저장")

    # 5. 상세 개선 효과 출력
    print_improvement(summary_df)

    # 6. 전략별 최적 추세선 조합 출력
    print("\n" + "=" * 65)
    print("  전략별 EV 기준 최적 추세선 조합")
    print("=" * 65)
    for strat in strategies:
        df_s = summary_df[summary_df["strategy"] == strat].copy()
        best = df_s.sort_values("ev", ascending=False).iloc[0]
        base = df_s[df_s["tl_config"] == "base"].iloc[0]
        print(f"\n  [{strat}]")
        print(f"    기존 (base): "
              f"거래 {int(base['n'])}건 / 승률 {base['win_rate']}% / EV {base['ev']:+.2f}%")
        print(f"    최적 ({best['tl_config']}): "
              f"거래 {int(best['n'])}건 / 승률 {best['win_rate']}% / EV {best['ev']:+.2f}%")
        if best["tl_config"] != "base":
            print(f"    → EV {best['ev'] - base['ev']:+.2f}%p 개선 / "
                  f"거래수 {int(best['n']) - int(base['n']):+d}건")

    # 7. 차트 생성
    print("\n[7] 차트 생성")
    plot_results(summary_df)

    print("\n완료!")
    print("  backtest_trendline_summary.csv")
    print("  backtest_trendline_trades.csv")
    print("  backtest_trendline_main.png")


if __name__ == "__main__":
    main()
