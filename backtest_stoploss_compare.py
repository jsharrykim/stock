"""
backtest_stoploss_compare.py
============================
전략별 손절 비율 비교 백테스트

[비교 대상 전략]
  ma200d : 현재가 < MA200  AND  VIX ≥ 25  AND  (RSI < 40 OR CCI < -100)
           목표 수익 +20%

  ma200u : 현재가 > MA200  AND  저가 %B ≤ 5
           목표 수익 +8%

  squeeze: 현재가 > MA200  AND  BB폭 < 60일평균 × 50%  AND  저가 %B ≤ 50
           목표 수익 +8%

[손절 비율]  -5%, -7%, -10%, -15%, -20%, -25%, -30%

[고정 매도 조건]
  - 60거래일 경과 & 수익 중 → 매도
  - 120거래일 타임 익시트

[출력]
  - backtest_stoploss_summary.csv         → 전체 결과 테이블
  - backtest_stoploss_main.png            → 전략별 손절라인 비교 (3×4 그리드)
  - backtest_stoploss_heatmap.png         → 전략 × 손절 히트맵 (EV, 승률)
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
END   = "2026-03-21"

# ── 전략 파라미터 (고정) ──────────────────────────────────────────────────────
TARGET_MA200D     = 0.20   # ma200d 목표 수익
TARGET_MA200U     = 0.08   # ma200u 목표 수익
TARGET_SQUEEZE    = 0.08   # squeeze 목표 수익
HALF_EXIT_DAYS    = 60     # 60거래일 수익 중 매도
MAX_HOLD_DAYS     = 120    # 최대 보유 거래일

VIX_MIN           = 25
RSI_MAX           = 40
CCI_MIN           = -100
BB_PERIOD         = 20
BB_STD            = 2.0
SQUEEZE_PERIOD    = 60
SQUEEZE_RATIO     = 0.5
PCTB_LOW_MA200U   = 5      # ma200u: 저가%B ≤ 5
PCTB_LOW_SQUEEZE  = 50     # squeeze: 저가%B ≤ 50

# ── 테스트할 손절 비율 ────────────────────────────────────────────────────────
STOP_LOSSES = [0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]

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

# ── 지표 계산 ─────────────────────────────────────────────────────────────────
def calc_indicators(df: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
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

    # BB폭 & 스퀴즈
    df["bb_width"]     = np.where(ma20 > 0, (bb_upper - bb_lower) / ma20 * 100, np.nan)
    df["bb_width_avg"] = df["bb_width"].rolling(SQUEEZE_PERIOD).mean()
    df["squeeze"]      = df["bb_width"] < df["bb_width_avg"] * SQUEEZE_RATIO

    # 저가 기준 %B
    df["pctb_low"] = np.where(bb_range > 0, (low - bb_lower) / bb_range * 100, np.nan)

    # RSI(14)
    delta      = close.diff()
    gain       = delta.clip(lower=0).rolling(14).mean()
    loss       = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"]  = 100 - 100 / (1 + np.where(loss == 0, 100.0, gain / loss))

    # CCI(14)
    tp     = (df["High"] + df["Low"] + close) / 3
    tp_ma  = tp.rolling(14).mean()
    tp_md  = tp.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = np.where(tp_md > 0, (tp - tp_ma) / (0.015 * tp_md), 0)

    # VIX 병합
    df = df.join(vix, how="left")
    df["vix"] = df["vix"].ffill()

    return df

# ── 데이터 다운로드 ────────────────────────────────────────────────────────────
def download_all(tickers, vix):
    print(f"종목 다운로드 중... ({len(tickers)}개)")
    try:
        raw = yf.download(
            tickers, start=START, end=END,
            auto_adjust=True, progress=False, group_by="ticker"
        )
    except Exception as e:
        print(f"일괄 다운로드 실패: {e}")
        raw = None

    result = {}
    for t in tickers:
        try:
            df = raw[t].copy() if (raw is not None and len(tickers) > 1) else (
                raw.copy() if raw is not None else yf.download(t, start=START, end=END, auto_adjust=True, progress=False)
            )
            df.dropna(how="all", inplace=True)
            if len(df) < 250:
                continue
            df = calc_indicators(df, vix)
            result[t] = df
        except Exception as e:
            print(f"  [{t}] 오류: {e}")

    print(f"  → {len(result)}개 종목 로드 완료")
    return result

# ── 백테스트 엔진 ─────────────────────────────────────────────────────────────
def run_backtest(data: dict, strategy: str, target_pct: float, circuit_pct: float) -> list:
    trades = []
    req_cols = ["ma200", "rsi", "cci", "pctb_low", "bb_width_avg", "squeeze", "vix"]

    for ticker, df in data.items():
        df_c = df.dropna(subset=req_cols).copy()
        if len(df_c) < 50:
            continue

        in_position = False
        entry_price = 0.0
        entry_idx   = 0

        rows   = df_c.to_dict("index")
        idx_list = list(df_c.index)

        for ii, date in enumerate(idx_list):
            r = rows[date]
            close    = r["Close"]
            ma200    = r["ma200"]
            rsi      = r["rsi"]
            cci      = r["cci"]
            pctb_low = r["pctb_low"] if not pd.isna(r["pctb_low"]) else 999
            squeeze  = bool(r["squeeze"])
            vix_val  = r["vix"] if not pd.isna(r["vix"]) else 0
            bb_ratio = (r["bb_width"] / r["bb_width_avg"]) if (r["bb_width_avg"] > 0 and not pd.isna(r["bb_width_avg"])) else 1

            if in_position:
                hold  = ii - entry_idx
                pnl   = (close - entry_price) / entry_price

                reason = None
                if   pnl >= target_pct:                       reason = "목표"
                elif pnl <= -circuit_pct:                     reason = "손절"
                elif hold >= HALF_EXIT_DAYS and pnl > 0:      reason = "60일수익"
                elif hold >= MAX_HOLD_DAYS:                   reason = "기간만료"

                if reason:
                    trades.append({
                        "strategy":    strategy,
                        "circuit_pct": round(circuit_pct * 100, 1),
                        "ticker":      ticker,
                        "entry_date":  entry_date,
                        "exit_date":   date,
                        "pnl_pct":     round(pnl * 100, 2),
                        "hold_days":   hold,
                        "exit_reason": reason,
                    })
                    in_position = False

            if not in_position:
                signal = False
                if strategy == "ma200d":
                    cond1 = close < ma200
                    cond2 = vix_val >= VIX_MIN
                    cond3 = (rsi < RSI_MAX) or (cci < CCI_MIN)
                    signal = cond1 and cond2 and cond3
                elif strategy == "ma200u":
                    cond1 = close > ma200
                    cond2 = pctb_low <= PCTB_LOW_MA200U
                    signal = cond1 and cond2
                elif strategy == "squeeze":
                    cond1 = close > ma200
                    cond2 = squeeze
                    cond3 = pctb_low <= PCTB_LOW_SQUEEZE
                    signal = cond1 and cond2 and cond3

                if signal:
                    in_position = True
                    entry_price = close
                    entry_date  = date
                    entry_idx   = ii

        # 미청산 처리
        if in_position:
            last  = df_c.iloc[-1]
            pnl   = (last["Close"] - entry_price) / entry_price
            hold  = len(df_c) - 1 - entry_idx
            trades.append({
                "strategy":    strategy,
                "circuit_pct": round(circuit_pct * 100, 1),
                "ticker":      ticker,
                "entry_date":  entry_date,
                "exit_date":   df_c.index[-1],
                "pnl_pct":     round(pnl * 100, 2),
                "hold_days":   hold,
                "exit_reason": "미청산",
            })

    return trades

# ── 통계 계산 ─────────────────────────────────────────────────────────────────
def calc_stats(trades: list) -> dict:
    if not trades:
        return {"n": 0, "win_rate": 0, "avg_pnl": 0, "median_pnl": 0,
                "ev": 0, "avg_win": 0, "avg_loss": 0, "loss_rate_pct": 0}

    pnls     = [t["pnl_pct"] for t in trades]
    wins     = [p for p in pnls if p > 0]
    losses   = [p for p in pnls if p <= 0]
    stops    = [t for t in trades if t["exit_reason"] == "손절"]

    win_rate  = len(wins) / len(pnls) * 100
    avg_pnl   = np.mean(pnls)
    median_pnl= np.median(pnls)
    avg_win   = np.mean(wins)   if wins   else 0
    avg_loss  = np.mean(losses) if losses else 0
    ev        = avg_pnl   # 기댓값 = 평균 수익률
    loss_rate = len(stops) / len(trades) * 100

    return {
        "n":            len(trades),
        "win_rate":     round(win_rate, 1),
        "avg_pnl":      round(avg_pnl, 2),
        "median_pnl":   round(median_pnl, 2),
        "ev":           round(ev, 2),
        "avg_win":      round(avg_win, 2),
        "avg_loss":     round(avg_loss, 2),
        "loss_rate_pct": round(loss_rate, 1),
    }

# ── 차트 그리기 ───────────────────────────────────────────────────────────────
def plot_main(summary_df: pd.DataFrame):
    strategies   = ["ma200d", "ma200u", "squeeze"]
    strat_labels = {
        "ma200d":  "A그룹 — 200일선 하방 (목표 +20%)",
        "ma200u":  "B그룹 — 200일선 상방/BB눌림 (목표 +8%)",
        "squeeze": "C그룹 — 200일선 상방/스퀴즈 (목표 +8%)",
    }
    metrics = [
        ("win_rate",   "승률 (%)",          "royalblue"),
        ("avg_pnl",    "평균 수익률 (%)",   "darkorange"),
        ("ev",         "기댓값 EV (%)",     "green"),
        ("n",          "거래 횟수",         "gray"),
    ]

    fig, axes = plt.subplots(len(strategies), len(metrics),
                             figsize=(20, 13), constrained_layout=True)
    fig.suptitle("전략별 손절 비율 비교 (2015–2026)", fontsize=15, fontweight="bold")

    sl_labels = [f"-{sl}%" for sl in STOP_LOSSES]
    x = np.arange(len(STOP_LOSSES))
    current_idx = STOP_LOSSES.index(0.25)  # 현재 설정(-25%) 위치

    for ri, strategy in enumerate(strategies):
        df_s = summary_df[summary_df["strategy"] == strategy].copy()
        df_s = df_s.sort_values("circuit_pct")

        for ci, (metric, label, color) in enumerate(metrics):
            ax = axes[ri][ci]
            vals = df_s[metric].values

            bars = ax.bar(x, vals, color=color, alpha=0.7, width=0.6)

            # 현재 설정 강조
            bars[current_idx].set_edgecolor("red")
            bars[current_idx].set_linewidth(2)
            bars[current_idx].set_alpha(1.0)

            # 값 레이블
            for bar, val in zip(bars, vals):
                fmt = f"{val:.1f}" if metric != "n" else f"{int(val)}"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.02,
                        fmt, ha="center", va="bottom", fontsize=7.5)

            ax.set_xticks(x)
            ax.set_xticklabels(sl_labels, fontsize=8, rotation=30)
            ax.set_ylabel(label, fontsize=9)
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

            if ci == 0:
                ax.set_title(f"{strat_labels[strategy]}\n{label}", fontsize=9, fontweight="bold")
            else:
                ax.set_title(label, fontsize=9)

            # 참고선 (승률 50%, EV 0)
            if metric == "win_rate":
                ax.axhline(50, color="red", linewidth=0.8, linestyle=":", label="50%")
            if metric in ("ev", "avg_pnl"):
                ax.axhline(0, color="red", linewidth=0.8, linestyle=":")

    # 범례 (현재 설정)
    from matplotlib.patches import Patch
    legend_el = [Patch(facecolor="white", edgecolor="red", linewidth=2, label="현재 설정 (-25%)")]
    fig.legend(handles=legend_el, loc="lower right", fontsize=10)

    plt.savefig("backtest_stoploss_main_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → backtest_stoploss_main.png 저장")

def plot_heatmaps(summary_df: pd.DataFrame):
    strategies  = ["ma200d", "ma200u", "squeeze"]
    sl_vals     = [round(s * 100, 1) for s in STOP_LOSSES]
    strat_short = {"ma200d": "A그룹\n하방", "ma200u": "B그룹\n상방BB", "squeeze": "C그룹\n스퀴즈"}

    metrics_hm = [
        ("ev",       "기댓값 EV (%)"),
        ("win_rate", "승률 (%)"),
        ("avg_pnl",  "평균 수익률 (%)"),
        ("n",        "거래 횟수"),
    ]

    fig, axes = plt.subplots(1, len(metrics_hm), figsize=(18, 5), constrained_layout=True)
    fig.suptitle("전략 × 손절 비율 히트맵 (2015–2026)", fontsize=14, fontweight="bold")

    for ax, (metric, title) in zip(axes, metrics_hm):
        mat = np.zeros((len(strategies), len(sl_vals)))
        for ri, strat in enumerate(strategies):
            for ci, sl in enumerate(sl_vals):
                row = summary_df[(summary_df["strategy"] == strat) & (summary_df["circuit_pct"] == sl)]
                mat[ri, ci] = row[metric].values[0] if len(row) else 0

        cmap = "RdYlGn" if metric in ("ev", "win_rate", "avg_pnl") else "Blues"
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8)

        ax.set_xticks(range(len(sl_vals)))
        ax.set_xticklabels([f"-{s}%" for s in sl_vals], fontsize=9, rotation=30)
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels([strat_short[s] for s in strategies], fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")

        # 값 표시
        for ri in range(len(strategies)):
            for ci in range(len(sl_vals)):
                val = mat[ri, ci]
                fmt = f"{val:.1f}" if metric != "n" else f"{int(val)}"
                ax.text(ci, ri, fmt, ha="center", va="center",
                        fontsize=8.5, fontweight="bold",
                        color="white" if abs(val) > (mat.max() * 0.6) else "black")

        # 현재 설정 테두리 (circuit_pct=25.0 열)
        if 25.0 in sl_vals:
            col_idx = sl_vals.index(25.0)
            for ri in range(len(strategies)):
                ax.add_patch(plt.Rectangle(
                    (col_idx - 0.5, ri - 0.5), 1, 1,
                    fill=False, edgecolor="red", linewidth=2
                ))

    plt.savefig("backtest_stoploss_heatmap_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → backtest_stoploss_heatmap.png 저장")

def plot_exit_distribution(all_trades: list):
    """손절 비율별 청산 사유 분포"""
    df = pd.DataFrame(all_trades)
    strategies = ["ma200d", "ma200u", "squeeze"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    fig.suptitle("손절 비율별 청산 사유 분포", fontsize=13, fontweight="bold")

    reasons  = ["목표", "손절", "60일수익", "기간만료", "미청산"]
    colors   = ["#2ecc71", "#e74c3c", "#3498db", "#95a5a6", "#f39c12"]
    sl_labels= [f"-{int(s*100)}%" for s in STOP_LOSSES]

    for ax, strat in zip(axes, strategies):
        df_s = df[df["strategy"] == strat]
        bottom = np.zeros(len(STOP_LOSSES))
        for reason, color in zip(reasons, colors):
            counts = []
            for sl in STOP_LOSSES:
                sl_str = round(sl * 100, 1)
                total  = len(df_s[df_s["circuit_pct"] == sl_str])
                cnt    = len(df_s[(df_s["circuit_pct"] == sl_str) & (df_s["exit_reason"] == reason)])
                counts.append(cnt / total * 100 if total > 0 else 0)
            ax.bar(range(len(STOP_LOSSES)), counts, bottom=bottom,
                   label=reason, color=color, alpha=0.85)
            bottom += np.array(counts)

        ax.set_xticks(range(len(STOP_LOSSES)))
        ax.set_xticklabels(sl_labels, rotation=30, fontsize=9)
        ax.set_ylabel("비율 (%)")
        ax.set_ylim(0, 105)
        ax.set_title({"ma200d": "A그룹 (하방)", "ma200u": "B그룹 (상방BB)", "squeeze": "C그룹 (스퀴즈)"}[strat],
                     fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)

    plt.savefig("backtest_stoploss_dist_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → backtest_stoploss_dist.png 저장")

# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  전략별 손절 비율 백테스트")
    print("=" * 60)

    # 1. VIX 다운로드
    print("\n[1] VIX 다운로드")
    vix_raw = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)
    vix = vix_raw["Close"].squeeze()
    vix.name = "vix"
    print(f"  VIX: {len(vix)}일")

    # 2. 종목 다운로드
    print("\n[2] 종목 데이터 다운로드")
    data = download_all(ALL_TICKERS, vix)

    # 3. 백테스트 실행
    print("\n[3] 백테스트 실행")
    strategies = {
        "ma200d":  TARGET_MA200D,
        "ma200u":  TARGET_MA200U,
        "squeeze": TARGET_SQUEEZE,
    }

    all_trades = []
    for strat, target in strategies.items():
        for sl in STOP_LOSSES:
            sl_pct = round(sl * 100, 1)
            trades = run_backtest(data, strat, target, sl)
            all_trades.extend(trades)
            stats  = calc_stats(trades)
            print(f"  [{strat:8s}] 손절 -{sl_pct:4.1f}% | "
                  f"거래 {stats['n']:3d}건 | 승률 {stats['win_rate']:5.1f}% | "
                  f"평균 {stats['avg_pnl']:+6.2f}% | EV {stats['ev']:+6.2f}%")

    # 4. 요약 테이블 생성
    print("\n[4] 요약 테이블 생성")
    rows = []
    for strat, target in strategies.items():
        for sl in STOP_LOSSES:
            sl_pct  = round(sl * 100, 1)
            trades  = [t for t in all_trades if t["strategy"] == strat and t["circuit_pct"] == sl_pct]
            stats   = calc_stats(trades)
            rows.append({
                "strategy":       strat,
                "circuit_pct":    sl_pct,
                "target_pct":     round(target * 100, 1),
                **stats
            })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv("backtest_stoploss_summary_v2.csv", index=False, encoding="utf-8-sig")
    print("  → backtest_stoploss_summary.csv 저장")

    # 5. 결과 출력
    print("\n[5] 전략별 최적 손절 요약")
    print("-" * 75)
    print(f"{'전략':<10} {'손절':>7} {'거래':>5} {'승률':>7} {'평균수익':>9} {'EV':>8} {'손절청산%':>9}")
    print("-" * 75)
    for strat in ["ma200d", "ma200u", "squeeze"]:
        df_s = summary_df[summary_df["strategy"] == strat].sort_values("ev", ascending=False)
        best = df_s.iloc[0]
        curr = df_s[df_s["circuit_pct"] == 25.0].iloc[0]
        print(f"{'[최적] '+strat:<10} -{best['circuit_pct']:4.1f}%  "
              f"{int(best['n']):5d}건  {best['win_rate']:6.1f}%  "
              f"{best['avg_pnl']:+8.2f}%  {best['ev']:+7.2f}%  {best['loss_rate_pct']:7.1f}%")
        print(f"{'[현재] '+strat:<10} -{curr['circuit_pct']:4.1f}%  "
              f"{int(curr['n']):5d}건  {curr['win_rate']:6.1f}%  "
              f"{curr['avg_pnl']:+8.2f}%  {curr['ev']:+7.2f}%  {curr['loss_rate_pct']:7.1f}%")
        print()

    # 6. 차트 생성
    print("[6] 차트 생성")
    plot_main(summary_df)
    plot_heatmaps(summary_df)
    plot_exit_distribution(all_trades)

    print("\n완료!")
    print("  backtest_stoploss_summary.csv")
    print("  backtest_stoploss_main.png")
    print("  backtest_stoploss_heatmap.png")
    print("  backtest_stoploss_dist.png")

if __name__ == "__main__":
    main()
