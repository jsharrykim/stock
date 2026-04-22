"""
backtest_squeeze_pctb.py
=========================
A그룹 (squeeze) — 저가 %B 임계값 비교 백테스트

[A그룹 스퀴즈 조건]
  1. 현재가 > MA200
  2. BB폭 < BB폭 60일 평균 × 50% (스퀴즈)
  3. 저가 기준 %B ≤ X  ← 여기를 다양하게 변경

[비교 임계값]
  %B ≤ 50  (현재 — BB 중단 이하)
  %B ≤ 40
  %B ≤ 30
  %B ≤ 20
  %B ≤ 10
  %B ≤  5  (BB 하단 근접 — ma200u와 동일 강도)

[매도 조건]  목표 +8% / 손절 -25% / 60일&수익중 / 120일
[종목]       유저 모니터링 리스트 기준
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
START = "2015-01-01"
END   = "2026-03-27"

# ── 전략 파라미터 ─────────────────────────────────────────────────────────────
TARGET_PCT      = 0.08
CIRCUIT_PCT     = 0.25
HALF_EXIT_DAYS  = 60
MAX_HOLD_DAYS   = 120

BB_PERIOD       = 20
BB_STD          = 2.0
SQUEEZE_PERIOD  = 60
SQUEEZE_RATIO   = 0.5

# ── 테스트할 %B 임계값 ────────────────────────────────────────────────────────
PCTB_THRESHOLDS = [50, 40, 30, 20, 10, 5]

# ── 종목 (유저 리스트) ─────────────────────────────────────────────────────────
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


# ── 지표 계산 ─────────────────────────────────────────────────────────────────
def calc_indicators(df: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    df    = df.copy()
    close = df["Close"]
    low   = df["Low"]

    df["ma200"]      = close.rolling(200).mean()
    ma20             = close.rolling(BB_PERIOD).mean()
    std20            = close.rolling(BB_PERIOD).std()
    bb_upper         = ma20 + BB_STD * std20
    bb_lower         = ma20 - BB_STD * std20
    bb_range         = bb_upper - bb_lower
    df["bb_width"]   = np.where(ma20 > 0, (bb_upper - bb_lower) / ma20 * 100, np.nan)
    df["bb_width_avg"] = df["bb_width"].rolling(SQUEEZE_PERIOD).mean()
    df["squeeze"]    = df["bb_width"] < df["bb_width_avg"] * SQUEEZE_RATIO
    df["pctb_low"]   = np.where(bb_range > 0, (low - bb_lower) / bb_range * 100, np.nan)

    df = df.join(vix, how="left")
    df["vix"] = df["vix"].ffill()
    return df


# ── 데이터 다운로드 ────────────────────────────────────────────────────────────
def download_all(tickers, vix):
    print(f"  종목 다운로드 중... ({len(tickers)}개)")
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
            if len(df) < 250:
                continue
            df = calc_indicators(df, vix)
            result[t] = df
        except Exception as e:
            print(f"    [{t}] 오류: {e}")

    print(f"  → {len(result)}개 종목 로드 완료")
    return result


# ── 백테스트 엔진 ─────────────────────────────────────────────────────────────
def run_backtest(data: dict, pctb_threshold: int) -> list:
    trades = []

    for ticker, df in data.items():
        df_c = df.dropna(subset=["ma200", "pctb_low", "bb_width_avg", "squeeze"]).copy()
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
            ma200    = r["ma200"]
            pctb_low = r["pctb_low"] if not pd.isna(r["pctb_low"]) else 999
            squeeze  = bool(r["squeeze"]) if not pd.isna(r["squeeze"]) else False

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
                        "pctb_thr":    pctb_threshold,
                        "ticker":      ticker,
                        "entry_date":  entry_date,
                        "exit_date":   date,
                        "pnl_pct":     round(pnl * 100, 2),
                        "hold_days":   ii - entry_idx,
                        "exit_reason": reason,
                    })
                    in_position = False

            if not in_position:
                signal = (close > ma200 and squeeze and pctb_low <= pctb_threshold)
                if signal:
                    in_position = True
                    entry_price = close
                    entry_date  = date
                    entry_idx   = ii

        if in_position:
            last = df_c.iloc[-1]
            pnl  = (last["Close"] - entry_price) / entry_price
            trades.append({
                "pctb_thr":    pctb_threshold,
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
    metrics = [
        ("win_rate",  "승률 (%)"),
        ("avg_pnl",   "평균 수익률 (%)"),
        ("ev",        "기댓값 EV (%)"),
        ("n",         "거래 횟수"),
        ("stop_rate", "손절 청산율 (%)"),
        ("avg_hold",  "평균 보유일"),
    ]
    colors = ["#d73027", "#fc8d59", "#fee090", "#91cf60", "#1a9850", "#006837"]
    x = np.arange(len(PCTB_THRESHOLDS))
    current_idx = PCTB_THRESHOLDS.index(50)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    fig.suptitle(
        "A그룹 (squeeze) — 저가 %B 임계값별 성과 비교 (2015–2026)\n"
        "목표 +8% / 손절 -25% / 유저 종목 기준",
        fontsize=13, fontweight="bold"
    )
    axes_flat = axes.flatten()

    for ax, (metric, label), color_set in zip(axes_flat, metrics,
                                               [colors] * len(metrics)):
        vals = [summary_df[summary_df["pctb_thr"] == t][metric].values[0]
                for t in PCTB_THRESHOLDS]
        bars = ax.bar(x, vals, color=colors, alpha=0.85, width=0.65, edgecolor="white")

        # 현재 설정(%B≤50) 강조
        bars[current_idx].set_edgecolor("red")
        bars[current_idx].set_linewidth(2.5)

        for bar, val in zip(bars, vals):
            fmt = f"{int(val)}" if metric == "n" else f"{val:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(vals) * 0.02 if max(vals) > 0 else 0.3),
                    fmt, ha="center", va="bottom", fontsize=8.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"%B≤{t}" for t in PCTB_THRESHOLDS], fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        if metric == "win_rate":
            ax.axhline(50, color="red", linewidth=0.8, linestyle=":", alpha=0.5)

    from matplotlib.patches import Patch
    legend_el = [Patch(facecolor=colors[i], alpha=0.85,
                       label=f"%B≤{t}" + (" ← 현재" if t == 50 else ""))
                 for i, t in enumerate(PCTB_THRESHOLDS)]
    fig.legend(handles=legend_el, loc="lower right", fontsize=9, ncol=3)

    plt.savefig("backtest_squeeze_pctb.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → backtest_squeeze_pctb.png 저장")


def plot_exit_dist(all_trades: list):
    """%B 임계값별 청산 사유 분포"""
    df = pd.DataFrame(all_trades)
    reasons = ["목표", "손절", "60일수익", "기간만료", "미청산"]
    colors  = ["#2ecc71", "#e74c3c", "#3498db", "#95a5a6", "#f39c12"]

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    fig.suptitle("A그룹 (squeeze) — %B 임계값별 청산 사유 분포", fontsize=12, fontweight="bold")

    x = np.arange(len(PCTB_THRESHOLDS))
    bottom = np.zeros(len(PCTB_THRESHOLDS))

    for reason, color in zip(reasons, colors):
        counts = []
        for thr in PCTB_THRESHOLDS:
            sub   = df[df["pctb_thr"] == thr]
            total = len(sub)
            cnt   = len(sub[sub["exit_reason"] == reason])
            counts.append(cnt / total * 100 if total > 0 else 0)
        ax.bar(x, counts, bottom=bottom, label=reason, color=color, alpha=0.85)
        bottom += np.array(counts)

    ax.set_xticks(x)
    ax.set_xticklabels([f"%B≤{t}" for t in PCTB_THRESHOLDS], fontsize=10)
    ax.set_ylabel("비율 (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=9)
    ax.axvline(x=0 - 0.3, color="red", linewidth=2, linestyle="--", alpha=0.4, label="현재")

    plt.savefig("backtest_squeeze_pctb_dist.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → backtest_squeeze_pctb_dist.png 저장")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  A그룹 (squeeze) — 저가 %B 임계값 비교 백테스트")
    print("=" * 65)

    # 1. VIX
    print("\n[1] VIX 다운로드")
    vix_raw = yf.download("^VIX", start=START, end=END,
                          auto_adjust=True, progress=False)
    vix = vix_raw["Close"].squeeze()
    vix.name = "vix"

    # 2. 종목 데이터
    print("\n[2] 종목 데이터 다운로드")
    data = download_all(ALL_TICKERS, vix)

    # 3. 백테스트
    print("\n[3] 백테스트 실행")
    all_trades = []
    rows = []

    print(f"\n  {'%B 임계':>8} {'거래':>5} {'승률':>7} {'평균수익':>9} {'EV':>7} {'손절청산%':>9} {'평균보유일':>10}")
    print("  " + "-" * 60)

    for thr in PCTB_THRESHOLDS:
        trades = run_backtest(data, thr)
        all_trades.extend(trades)
        s = calc_stats(trades)
        rows.append({"pctb_thr": thr, **s})
        marker = " ← 현재" if thr == 50 else ""
        print(f"  %B≤{thr:>3d}  {s['n']:>5d}건  {s['win_rate']:>6.1f}%  "
              f"{s['avg_pnl']:>+8.2f}%  {s['ev']:>+6.2f}%  "
              f"{s['stop_rate']:>8.1f}%  {s['avg_hold']:>9.1f}일{marker}")

    # 4. 저장
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv("backtest_squeeze_pctb_summary.csv",
                      index=False, encoding="utf-8-sig")
    pd.DataFrame(all_trades).to_csv("backtest_squeeze_pctb_trades.csv",
                                    index=False, encoding="utf-8-sig")
    print("\n  → backtest_squeeze_pctb_summary.csv 저장")
    print("  → backtest_squeeze_pctb_trades.csv 저장")

    # 5. 개선 효과 요약
    base = summary_df[summary_df["pctb_thr"] == 50].iloc[0]
    print("\n" + "=" * 65)
    print("  %B 임계값 변경 시 성과 변화 (vs 현재 %B≤50)")
    print("=" * 65)
    print(f"  {'임계값':>8}  {'거래Δ':>6}  {'승률Δ':>7}  {'EV Δ':>7}  {'손절율Δ':>8}  {'판단'}")
    print("  " + "-" * 60)
    for _, row in summary_df.iterrows():
        thr = int(row["pctb_thr"])
        if thr == 50:
            print(f"  %B≤{thr:>3d}  (현재 기준)  승률 {row['win_rate']}%  EV {row['ev']:+.2f}%")
            continue
        d_n    = int(row["n"])    - int(base["n"])
        d_wr   = row["win_rate"] - base["win_rate"]
        d_ev   = row["ev"]       - base["ev"]
        d_stop = row["stop_rate"] - base["stop_rate"]
        # 판단: 거래 감소를 감수하고 EV가 오르면 ★, EV도 내리면 ✗
        verdict = "★ 추천" if (d_ev > 0 and abs(d_n) < int(base["n"]) * 0.5) else (
                  "△ 참고" if d_ev > 0 else "✗ 비추")
        print(f"  %B≤{thr:>3d}  {d_n:>+6d}건  {d_wr:>+6.1f}%p  {d_ev:>+6.2f}%p  "
              f"{d_stop:>+7.1f}%p  {verdict}")

    # 6. 차트
    print("\n[6] 차트 생성")
    plot_results(summary_df)
    plot_exit_dist(all_trades)

    print("\n완료!")
    print("  backtest_squeeze_pctb_summary.csv")
    print("  backtest_squeeze_pctb.png")
    print("  backtest_squeeze_pctb_dist.png")


if __name__ == "__main__":
    main()
