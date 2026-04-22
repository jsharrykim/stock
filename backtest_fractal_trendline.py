"""
backtest_fractal_trendline.py
==============================
C그룹 (ma200d) × DojiEmoji 스타일 프랙탈 추세선 터치 필터 백테스트

[추세선 방식 — Auto Trendline [DojiEmoji] 재현]
  1. 프랙탈 저점 감지: 좌우 N봉보다 낮은 봉 = 프랙탈 저점
     (룩어헤드 방지: 프랙탈 저점 j는 j+N봉 이후에만 확정)
  2. 확정된 프랙탈 저점 중 가장 최근 HL 쌍 (두 번째 저점 > 첫 번째) 찾기
  3. 두 점을 연결한 직선을 현재까지 연장 → 오늘의 추세선 가격
  4. 조건: slope > 0 (상승 추세선) AND 저가 ≤ 추세선 × (1 + tolerance)

[비교 케이스]
  base          : C그룹 기본 (추세선 없음)
  f10_tol3pct   : 프랙탈 10봉, 터치 ±3%
  f10_tol5pct   : 프랙탈 10봉, 터치 ±5%
  f5_tol3pct    : 프랙탈 5봉,  터치 ±3%
  f5_tol5pct    : 프랙탈 5봉,  터치 ±5%

[C그룹 매수 조건]
  현재가 < MA200 AND VIX ≥ 25 AND (RSI < 40 OR CCI < -100)
[매도 조건]  목표 +20% / 손절 -25% / 60일&수익 / 120일
[종목]       유저 모니터링 리스트
[기간]       2015-01-01 ~ 2026-03-27
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
START = "2013-01-01"   # 프랙탈 워밍업 기간 확보용 (백테스트는 2015~)
END   = "2026-03-27"
BACKTEST_START = "2015-01-01"

# ── 전략 파라미터 ─────────────────────────────────────────────────────────────
TARGET_PCT      = 0.20
CIRCUIT_PCT     = 0.25
HALF_EXIT_DAYS  = 60
MAX_HOLD_DAYS   = 120

BB_PERIOD       = 20
BB_STD          = 2.0
VIX_MIN         = 25
RSI_MAX         = 40
CCI_MIN         = -100

# ── 테스트 설정 ───────────────────────────────────────────────────────────────
CONFIGS = {
    "base":        {"fractal_period": None, "tolerance": None},
    "f10_tol3pct": {"fractal_period": 10,   "tolerance": 0.03},
    "f10_tol5pct": {"fractal_period": 10,   "tolerance": 0.05},
    "f5_tol3pct":  {"fractal_period": 5,    "tolerance": 0.03},
    "f5_tol5pct":  {"fractal_period": 5,    "tolerance": 0.05},
}

# ── 종목 ─────────────────────────────────────────────────────────────────────
KR_TICKERS = [
    "000660.KS", "005930.KS", "277810.KS", "034020.KS", "015760.KS",
    "005380.KS", "012450.KS", "042660.KS", "042700.KQ", "096770.KS",
    "009150.KS", "000270.KS", "247540.KQ", "376900.KQ", "079550.KS",
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


# ── 프랙탈 추세선 계산 ────────────────────────────────────────────────────────
def calc_fractal_trendline(df: pd.DataFrame, fractal_period: int):
    """
    DojiEmoji 스타일 HL(Higher Low) 추세선 계산.
    룩어헤드 방지: 프랙탈 저점 j는 j + fractal_period 이후 봉에서만 사용 가능.

    반환: (trendline_values, slopes) — pd.Series
    """
    lows = df["Low"].values.astype(float)
    n    = len(lows)

    # Step 1: 모든 프랙탈 저점 사전 계산 (확정 시점 포함)
    # fractal_lows: list of (bar_index, low_value, confirmed_at)
    fractal_lows = []
    fp = fractal_period
    for j in range(fp, n - fp):
        segment = lows[j - fp: j + fp + 1]
        if lows[j] == np.min(segment) and np.sum(segment == lows[j]) == 1:
            fractal_lows.append((j, lows[j], j + fp))  # confirmed at j+fp

    # Step 2: 각 봉에서 사용 가능한 프랙탈 저점으로 최근 HL 쌍 찾기
    tl_vals   = np.full(n, np.nan)
    tl_slopes = np.full(n, np.nan)

    # 확정 시점 순으로 정렬 (이미 오름차순이지만 명시)
    fractal_lows.sort(key=lambda x: x[2])

    for i in range(fp * 2, n):
        # i 시점에서 사용 가능한 프랙탈 저점들 (confirmed_at <= i)
        available = [(j, v) for j, v, ca in fractal_lows if ca <= i]

        if len(available) < 2:
            continue

        # 가장 최근 HL 쌍 탐색 (뒤에서부터)
        for m in range(len(available) - 1, 0, -1):
            j2, v2 = available[m]
            j1, v1 = available[m - 1]
            if v2 > v1:  # Higher Low 조건
                slope = (v2 - v1) / (j2 - j1)
                if slope > 0:
                    intercept  = v1 - slope * j1
                    tl_val_now = intercept + slope * i
                    tl_vals[i]   = tl_val_now
                    tl_slopes[i] = slope
                break  # 최근 HL 쌍 1개만 사용

    return (
        pd.Series(tl_vals,   index=df.index),
        pd.Series(tl_slopes, index=df.index),
    )


# ── 지표 계산 ─────────────────────────────────────────────────────────────────
def calc_indicators(df: pd.DataFrame, vix: pd.Series,
                    fractal_period) -> pd.DataFrame:
    df    = df.copy()
    close = df["Close"]

    df["ma200"] = close.rolling(200).mean()

    # RSI(14)
    delta     = close.diff()
    gain      = delta.clip(lower=0).rolling(14).mean()
    loss      = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + np.where(loss == 0, 100.0, gain / loss))

    # CCI(14)
    tp        = (df["High"] + df["Low"] + close) / 3
    tp_ma     = tp.rolling(14).mean()
    tp_md     = tp.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = np.where(tp_md > 0, (tp - tp_ma) / (0.015 * tp_md), 0)

    # VIX 병합
    df = df.join(vix, how="left")
    df["vix"] = df["vix"].ffill()

    # 프랙탈 추세선
    if fractal_period is not None:
        tl_vals, tl_slopes = calc_fractal_trendline(df, fractal_period)
        df["tl_lower"]       = tl_vals
        df["tl_lower_slope"] = tl_slopes
    else:
        df["tl_lower"]       = np.nan
        df["tl_lower_slope"] = np.nan

    return df


# ── 데이터 다운로드 ────────────────────────────────────────────────────────────
def download_all(tickers, vix, fractal_period, label):
    print(f"  [{label}] 종목 다운로드 중... ({len(tickers)}개)")
    try:
        raw = yf.download(
            tickers, start=START, end=END,
            auto_adjust=True, progress=False, group_by="ticker"
        )
    except Exception as e:
        print(f"  다운로드 실패: {e}")
        raw = None

    result = {}
    for t in tickers:
        try:
            df = raw[t].copy() if (raw is not None and len(tickers) > 1) else (
                raw.copy() if raw is not None
                else yf.download(t, start=START, end=END, auto_adjust=True, progress=False)
            )
            df.dropna(how="all", inplace=True)
            if len(df) < 300:
                continue
            df = calc_indicators(df, vix, fractal_period)
            result[t] = df
        except Exception as e:
            print(f"    [{t}] 오류: {e}")

    print(f"  → {len(result)}개 종목 로드 완료")
    return result


# ── 백테스트 엔진 ─────────────────────────────────────────────────────────────
def run_backtest(data: dict, config_name: str, tolerance) -> list:
    trades = []

    for ticker, df in data.items():
        # 백테스트 시작 이후만 사용
        df_c = df[df.index >= BACKTEST_START].copy()
        df_c = df_c.dropna(subset=["ma200", "rsi", "cci", "vix"]).copy()
        if len(df_c) < 50:
            continue

        in_position = False
        entry_price = 0.0
        entry_idx   = 0
        entry_date  = None
        rows        = df_c.to_dict("index")
        idx_list    = list(df_c.index)

        for ii, date in enumerate(idx_list):
            r        = rows[date]
            close    = r["Close"]
            low      = r["Low"]
            ma200    = r["ma200"]
            rsi      = r["rsi"]
            cci      = r["cci"]
            vix_val  = r["vix"] if not pd.isna(r["vix"]) else 0
            tl_val   = r["tl_lower"]
            tl_slope = r["tl_lower_slope"]

            if in_position:
                hold  = ii - entry_idx
                pnl   = (close - entry_price) / entry_price
                reason = None
                if   pnl >= TARGET_PCT:                   reason = "목표"
                elif pnl <= -CIRCUIT_PCT:                 reason = "손절"
                elif hold >= HALF_EXIT_DAYS and pnl > 0:  reason = "60일수익"
                elif hold >= MAX_HOLD_DAYS:               reason = "기간만료"

                if reason:
                    trades.append({
                        "config":      config_name,
                        "ticker":      ticker,
                        "entry_date":  entry_date,
                        "exit_date":   date,
                        "pnl_pct":     round(pnl * 100, 2),
                        "hold_days":   hold,
                        "exit_reason": reason,
                    })
                    in_position = False

            if not in_position:
                # C그룹 기본 조건
                base_signal = (
                    close < ma200
                    and vix_val >= VIX_MIN
                    and (rsi < RSI_MAX or cci < CCI_MIN)
                )

                # 추세선 터치 필터
                if base_signal and tolerance is not None:
                    if (pd.isna(tl_val) or tl_val <= 0 or
                            pd.isna(tl_slope) or tl_slope <= 0):
                        base_signal = False
                    else:
                        base_signal = low <= tl_val * (1 + tolerance)

                if base_signal:
                    in_position = True
                    entry_price = close
                    entry_date  = date
                    entry_idx   = ii

        if in_position:
            last = df_c.iloc[-1]
            pnl  = (last["Close"] - entry_price) / entry_price
            trades.append({
                "config":      config_name,
                "ticker":      ticker,
                "entry_date":  entry_date,
                "exit_date":   df_c.index[-1],
                "pnl_pct":     round(pnl * 100, 2),
                "hold_days":   len(df_c) - 1 - entry_idx,
                "exit_reason": "미청산",
            })

    return trades


# ── 통계 ─────────────────────────────────────────────────────────────────────
def calc_stats(trades: list) -> dict:
    if not trades:
        return {"n": 0, "win_rate": 0, "avg_pnl": 0, "median_pnl": 0,
                "ev": 0, "avg_win": 0, "avg_loss": 0, "stop_rate": 0, "avg_hold": 0}
    pnls   = [t["pnl_pct"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    stops  = [t for t in trades if t["exit_reason"] == "손절"]
    return {
        "n":          len(trades),
        "win_rate":   round(len(wins) / len(pnls) * 100, 1),
        "avg_pnl":    round(np.mean(pnls), 2),
        "median_pnl": round(np.median(pnls), 2),
        "ev":         round(np.mean(pnls), 2),
        "avg_win":    round(np.mean(wins)   if wins   else 0, 2),
        "avg_loss":   round(np.mean(losses) if losses else 0, 2),
        "stop_rate":  round(len(stops) / len(trades) * 100, 1),
        "avg_hold":   round(np.mean([t["hold_days"] for t in trades]), 1),
    }


# ── 차트 ─────────────────────────────────────────────────────────────────────
def plot_results(summary_df: pd.DataFrame):
    configs = list(CONFIGS.keys())
    labels  = {
        "base":        "기존 (필터없음)",
        "f10_tol3pct": "프랙탈10봉 ±3%",
        "f10_tol5pct": "프랙탈10봉 ±5%",
        "f5_tol3pct":  "프랙탈5봉  ±3%",
        "f5_tol5pct":  "프랙탈5봉  ±5%",
    }
    colors  = ["#2c7bb6", "#d7191c", "#fdae61", "#1a9641", "#a6d96a"]
    metrics = [
        ("win_rate",  "승률 (%)"),
        ("avg_pnl",   "평균 수익률 (%)"),
        ("ev",        "기댓값 EV (%)"),
        ("n",         "거래 횟수"),
        ("stop_rate", "손절 청산율 (%)"),
        ("avg_hold",  "평균 보유일"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(
        "C그룹 (ma200d) × DojiEmoji 프랙탈 추세선 터치 필터 비교 (2015–2026)\n"
        "목표 +20% / 손절 -25% / 상승 추세선(slope>0) + 저가 터치 조건",
        fontsize=12, fontweight="bold"
    )

    x = np.arange(len(configs))
    for ax, (metric, ylabel) in zip(axes.flatten(), metrics):
        vals = [summary_df[summary_df["config"] == c][metric].values[0]
                for c in configs]
        bars = ax.bar(x, vals, color=colors, alpha=0.85, width=0.65, edgecolor="white")
        bars[0].set_edgecolor("black")
        bars[0].set_linewidth(2)

        for bar, val in zip(bars, vals):
            fmt = f"{int(val)}" if metric == "n" else f"{val:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(vals) * 0.02 if max(vals) > 0 else 0.3),
                    fmt, ha="center", va="bottom", fontsize=8.5)

        ax.set_xticks(x)
        ax.set_xticklabels([labels[c] for c in configs],
                           fontsize=8, rotation=20, ha="right")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        if metric == "win_rate":
            ax.axhline(50, color="red", linewidth=0.8, linestyle=":", alpha=0.5)

    from matplotlib.patches import Patch
    legend_el = [Patch(facecolor=colors[i], alpha=0.85,
                       label=labels[c] + (" ← 현재" if c == "base" else ""))
                 for i, c in enumerate(configs)]
    fig.legend(handles=legend_el, loc="lower right", fontsize=9, ncol=3)

    plt.savefig("backtest_fractal_trendline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → backtest_fractal_trendline.png 저장")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  C그룹 (ma200d) × DojiEmoji 프랙탈 추세선 백테스트")
    print("=" * 65)

    # 1. VIX
    print("\n[1] VIX 다운로드")
    vix_raw = yf.download("^VIX", start=START, end=END,
                          auto_adjust=True, progress=False)
    vix = vix_raw["Close"].squeeze()
    vix.name = "vix"

    # 2. 데이터 다운로드 (프랙탈 기간별로 캐시)
    print("\n[2] 종목 데이터 다운로드")
    unique_fps = sorted(set(
        v["fractal_period"] for v in CONFIGS.values()
        if v["fractal_period"] is not None
    ))

    data_cache = {}
    # base용 (fractal 없음)
    data_cache[None] = download_all(ALL_TICKERS, vix, None, "base")
    for fp in unique_fps:
        data_cache[fp] = download_all(ALL_TICKERS, vix, fp, f"fractal_{fp}")

    # 3. 백테스트
    print("\n[3] 백테스트 실행")
    config_labels = {
        "base":        "기존(필터없음)     ",
        "f10_tol3pct": "프랙탈10봉/±3%    ",
        "f10_tol5pct": "프랙탈10봉/±5%    ",
        "f5_tol3pct":  "프랙탈5봉/±3%     ",
        "f5_tol5pct":  "프랙탈5봉/±5%     ",
    }
    all_trades = []
    rows       = []

    print(f"\n  {'설정':22} {'거래':>5} {'승률':>7} {'평균수익':>9} "
          f"{'EV':>7} {'손절청산%':>9} {'평균보유일':>10}")
    print("  " + "-" * 72)

    for cfg_name, cfg in CONFIGS.items():
        fp  = cfg["fractal_period"]
        tol = cfg["tolerance"]
        data = data_cache[fp]

        trades = run_backtest(data, cfg_name, tol)
        all_trades.extend(trades)
        s = calc_stats(trades)
        rows.append({"config": cfg_name, **s})

        marker = " ← 현재" if cfg_name == "base" else ""
        print(f"  {config_labels[cfg_name]}  {s['n']:>5d}건  "
              f"{s['win_rate']:>6.1f}%  {s['avg_pnl']:>+8.2f}%  "
              f"{s['ev']:>+6.2f}%  {s['stop_rate']:>8.1f}%  "
              f"{s['avg_hold']:>9.1f}일{marker}")

    # 4. 저장
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv("backtest_fractal_trendline_summary.csv",
                      index=False, encoding="utf-8-sig")
    pd.DataFrame(all_trades).to_csv("backtest_fractal_trendline_trades.csv",
                                    index=False, encoding="utf-8-sig")
    print("\n  → CSV 저장 완료")

    # 5. 개선 효과 요약
    base = summary_df[summary_df["config"] == "base"].iloc[0]
    print("\n" + "=" * 72)
    print("  추세선 필터 추가 시 성과 변화 (vs base)")
    print("=" * 72)
    print(f"  {'설정':22} {'거래Δ':>7} {'승률Δ':>8} {'EV Δ':>8} "
          f"{'손절율Δ':>9} {'판단'}")
    print("  " + "-" * 72)

    for _, row in summary_df.iterrows():
        cfg = row["config"]
        if cfg == "base":
            print(f"  {'기존 (base)':22}  기준  승률 {row['win_rate']}%  "
                  f"EV {row['ev']:+.2f}%")
            continue
        d_n    = int(row["n"])       - int(base["n"])
        d_wr   = row["win_rate"]     - base["win_rate"]
        d_ev   = row["ev"]           - base["ev"]
        d_stop = row["stop_rate"]    - base["stop_rate"]
        verdict = (
            "★★ 강추" if d_ev > 1.0 and d_wr > 2 else
            "★ 추천"  if d_ev > 0  and d_wr > 0  else
            "△ 참고"  if d_ev > 0              else
            "✗ 비추"
        )
        print(f"  {config_labels[cfg]}  {d_n:>+7d}건  {d_wr:>+7.1f}%p  "
              f"{d_ev:>+7.2f}%p  {d_stop:>+8.1f}%p  {verdict}")

    # 6. 차트
    print("\n[6] 차트 생성")
    plot_results(summary_df)

    print("\n완료!")
    print("  backtest_fractal_trendline_summary.csv")
    print("  backtest_fractal_trendline_trades.csv")
    print("  backtest_fractal_trendline.png")


if __name__ == "__main__":
    main()
