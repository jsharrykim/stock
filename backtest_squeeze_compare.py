"""
backtest_squeeze_compare.py
============================
기존 전략 vs 스퀴즈 단독 vs 스퀴즈 혼합 3-way 비교

[비교 대상]
  ── 200일선 상방 계열 ──────────────────────────────────────────
  S1: 기존 전략 (상방)
      현재가 > MA200 AND 저가%B ≤ 5
      매도: +8% / -25% / 60일&수익중 / 120일

  S2: 스퀴즈 단독 (상방)
      현재가 > MA200 AND 스퀴즈
      매도: +8% / -25% / 60일&수익중 / 120일

  S3: 기존 + 스퀴즈 AND (상방)
      현재가 > MA200 AND 저가%B ≤ 5 AND 스퀴즈
      매도: +8% / -25% / 60일&수익중 / 120일

  S4: 기존 OR 스퀴즈 (상방)
      현재가 > MA200 AND (저가%B ≤ 5 OR 스퀴즈)
      매도: +8% / -25% / 60일&수익중 / 120일

  ── 200일선 하방 계열 ──────────────────────────────────────────
  S5: 기존 전략 (하방)
      현재가 < MA200 AND VIX ≥ 25 AND (RSI < 40 OR CCI < -100)
      매도: +20% / -25% / 60일&수익중 / 120일

  S6: 스퀴즈 단독 (하방)
      현재가 < MA200 AND 스퀴즈
      매도: +20% / -25% / 60일&수익중 / 120일

  S7: 기존 + 스퀴즈 AND (하방)
      현재가 < MA200 AND VIX ≥ 25 AND (RSI < 40 OR CCI < -100) AND 스퀴즈
      매도: +20% / -25% / 60일&수익중 / 120일

  S8: 기존 OR 스퀴즈 (하방)
      현재가 < MA200 AND (VIX ≥ 25 AND (RSI < 40 OR CCI < -100) OR 스퀴즈)
      매도: +20% / -25% / 60일&수익중 / 120일

매도 조건은 실제 운용 전략과 동일하게 적용
기간: 전체 가능 기간 (2015~2026)
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

# ── 파라미터 ─────────────────────────────────────────────────────────────────
START          = "2015-01-01"
END            = "2026-03-17"

# 매도 파라미터 (실제 전략과 동일)
TARGET_U       = 0.08    # 상방 목표 +8%
TARGET_D       = 0.20    # 하방 목표 +20%
CIRCUIT_PCT    = 0.25    # 공통 손절 -25%
HALF_DAYS      = 60      # 60거래일 후 수익 중이면 매도
MAX_HOLD       = 120     # 120일 강제 만료

# 진입 파라미터
BB_PERIOD      = 20
BB_STD         = 2.0
SQUEEZE_PERIOD = 60
SQUEEZE_RATIO  = 0.5
PCTB_LOW_MAX   = 5       # 저가%B ≤ 5
VIX_MIN        = 25      # VIX ≥ 25 (하방 조건)
RSI_THRESH     = 40      # RSI < 40
CCI_THRESH     = -100    # CCI < -100

# ── 종목 ──────────────────────────────────────────────────────────────────────
KR_TICKERS = [
    "000660.KS", "005930.KS", "277810.KS", "034020.KS", "005380.KS",
    "012450.KS", "042660.KS", "042700.KQ", "096770.KS", "009150.KS",
    "000270.KS", "247540.KQ", "376900.KS", "006400.KS", "079550.KS",
]
US_TICKERS = [
    "HOOD", "AAPL", "AVGO", "AMD", "MSFT", "GOOGL", "NVDA", "TSLA",
    "AMZN", "MU", "LRCX", "ON", "SNDK", "ASTS", "AVAV", "IONQ",
    "RKLB", "PLTR", "CRWD", "APP", "SOXL", "TSLL", "TE", "ONDS",
    "BE", "PL", "VRT", "LITE", "TER", "ANET", "IREN", "HOOG",
    "SOLT", "ETHU", "NBIS", "LPTH", "CONL", "INTC",
]
ALL_TICKERS = KR_TICKERS + US_TICKERS

# ── 한글 폰트 ─────────────────────────────────────────────────────────────────
def get_kr_font():
    for c in ["AppleGothic", "NanumGothic", "Malgun Gothic", "DejaVu Sans"]:
        if c in {f.name for f in fm.fontManager.ttflist}:
            return c
    return None

KR_FONT = get_kr_font()
if KR_FONT:
    plt.rcParams["font.family"] = KR_FONT
plt.rcParams["axes.unicode_minus"] = False

# ── VIX 데이터 ────────────────────────────────────────────────────────────────
def download_vix():
    print("VIX 데이터 다운로드 중...")
    vix = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)
    vix = vix["Close"].squeeze()
    vix.name = "vix"
    return vix

# ── 지표 계산 ─────────────────────────────────────────────────────────────────
def calc_indicators(df: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    low   = df["Low"]

    # MA200
    df["ma200"] = close.rolling(200).mean()

    # BB
    ma20   = close.rolling(BB_PERIOD).mean()
    std20  = close.rolling(BB_PERIOD).std()
    bb_upper = ma20 + BB_STD * std20
    bb_lower = ma20 - BB_STD * std20
    df["ma20"]     = ma20
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower

    # BB폭 & 스퀴즈
    df["bb_width"]     = (bb_upper - bb_lower) / ma20 * 100
    df["bb_width_avg"] = df["bb_width"].rolling(SQUEEZE_PERIOD).mean()
    df["squeeze"]      = df["bb_width"] < df["bb_width_avg"] * SQUEEZE_RATIO

    # 저가 %B
    bb_range = bb_upper - bb_lower
    df["pctb_low"] = np.where(bb_range > 0, (low - bb_lower) / bb_range * 100, np.nan)

    # RSI(14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = np.where(loss == 0, 100.0, gain / loss)
    df["rsi"] = 100 - 100 / (1 + rs)

    # CCI(14)
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    tp    = (hi + lo + cl) / 3
    tp_ma = tp.rolling(14).mean()
    tp_md = tp.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = np.where(tp_md > 0, (tp - tp_ma) / (0.015 * tp_md), 0)

    # VIX 병합 (날짜 기준 forward fill)
    df = df.join(vix, how="left")
    df["vix"] = df["vix"].ffill()

    return df

# ── 데이터 다운로드 ────────────────────────────────────────────────────────────
def download_data(tickers, vix):
    print(f"종목 데이터 다운로드 중... ({len(tickers)}개)")
    raw = yf.download(
        tickers, start=START, end=END,
        auto_adjust=True, progress=False,
        group_by="ticker"
    )
    result = {}
    for t in tickers:
        try:
            df = raw[t].copy() if len(tickers) > 1 else raw.copy()
            df.dropna(how="all", inplace=True)
            if len(df) < 250:
                print(f"  [{t}] 데이터 부족 ({len(df)}일) - 제외")
                continue
            df = calc_indicators(df, vix)
            result[t] = df
            print(f"  [{t}] {len(df)}일 로드 완료")
        except Exception as e:
            print(f"  [{t}] 오류: {e}")
    return result

# ── 매도 조건 판단 ────────────────────────────────────────────────────────────
def check_exit(close, entry_price, hold_days, target_pct):
    pnl = (close - entry_price) / entry_price
    if pnl >= target_pct:
        return "목표", pnl
    if pnl <= -CIRCUIT_PCT:
        return "손절", pnl
    if hold_days >= HALF_DAYS and pnl > 0:
        return "60일수익", pnl
    if hold_days >= MAX_HOLD:
        return "기간만료", pnl
    return None, pnl

# ── 백테스트 엔진 ─────────────────────────────────────────────────────────────
def run_backtest(data: dict, scenario_name: str, condition_fn, target_pct: float) -> list:
    trades = []
    for ticker, df in data.items():
        req_cols = ["ma200", "bb_width_avg", "squeeze", "rsi", "cci", "pctb_low", "vix"]
        df_clean = df.dropna(subset=req_cols)
        if len(df_clean) < 10:
            continue

        in_position = False
        entry_price = None
        entry_date  = None
        entry_idx   = None

        idx_list = list(df_clean.index)
        for ii, date in enumerate(idx_list):
            row = df_clean.loc[date]

            if in_position:
                hold_days = ii - entry_idx
                reason, pnl = check_exit(row["Close"], entry_price, hold_days, target_pct)
                if reason:
                    trades.append({
                        "scenario": scenario_name,
                        "ticker": ticker,
                        "entry_date": entry_date,
                        "exit_date": date,
                        "entry_price": entry_price,
                        "exit_price": row["Close"],
                        "pnl_pct": round(pnl * 100, 2),
                        "hold_days": hold_days,
                        "exit_reason": reason,
                    })
                    in_position = False

            if not in_position:
                signal = {
                    "above200":  row["Close"] > row["ma200"],
                    "squeeze":   bool(row["squeeze"]),
                    "pctb_low":  row["pctb_low"],
                    "rsi":       row["rsi"],
                    "cci":       row["cci"],
                    "vix":       row["vix"] if not pd.isna(row["vix"]) else 0,
                }
                try:
                    if condition_fn(signal):
                        in_position = True
                        entry_price = row["Close"]
                        entry_date  = date
                        entry_idx   = ii
                except Exception:
                    pass

        # 미청산
        if in_position and entry_price:
            last = df_clean.iloc[-1]
            pnl  = (last["Close"] - entry_price) / entry_price
            trades.append({
                "scenario": scenario_name,
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": df_clean.index[-1],
                "entry_price": entry_price,
                "exit_price": last["Close"],
                "pnl_pct": round(pnl * 100, 2),
                "hold_days": len(df_clean) - 1 - entry_idx,
                "exit_reason": "미청산",
            })
    return trades

# ── 시나리오 정의 ─────────────────────────────────────────────────────────────
SCENARIOS = [
    # ── 상방 계열 (target +8%) ─────────────────────────────────────────────
    {
        "name": "S1_기존전략_상방",
        "target": TARGET_U,
        "fn": lambda r: r["above200"] and r["pctb_low"] <= PCTB_LOW_MAX,
    },
    {
        "name": "S2_스퀴즈단독_상방",
        "target": TARGET_U,
        "fn": lambda r: r["above200"] and r["squeeze"],
    },
    {
        "name": "S3_기존AND스퀴즈_상방",
        "target": TARGET_U,
        "fn": lambda r: r["above200"] and r["pctb_low"] <= PCTB_LOW_MAX and r["squeeze"],
    },
    {
        "name": "S4_기존OR스퀴즈_상방",
        "target": TARGET_U,
        "fn": lambda r: r["above200"] and (r["pctb_low"] <= PCTB_LOW_MAX or r["squeeze"]),
    },
    # ── 하방 계열 (target +20%) ────────────────────────────────────────────
    {
        "name": "S5_기존전략_하방",
        "target": TARGET_D,
        "fn": lambda r: (not r["above200"])
                        and r["vix"] >= VIX_MIN
                        and (r["rsi"] < RSI_THRESH or r["cci"] < CCI_THRESH),
    },
    {
        "name": "S6_스퀴즈단독_하방",
        "target": TARGET_D,
        "fn": lambda r: (not r["above200"]) and r["squeeze"],
    },
    {
        "name": "S7_기존AND스퀴즈_하방",
        "target": TARGET_D,
        "fn": lambda r: (not r["above200"])
                        and r["vix"] >= VIX_MIN
                        and (r["rsi"] < RSI_THRESH or r["cci"] < CCI_THRESH)
                        and r["squeeze"],
    },
    {
        "name": "S8_기존OR스퀴즈_하방",
        "target": TARGET_D,
        "fn": lambda r: (not r["above200"])
                        and (
                            (r["vix"] >= VIX_MIN and (r["rsi"] < RSI_THRESH or r["cci"] < CCI_THRESH))
                            or r["squeeze"]
                        ),
    },
]

# ── 결과 분석 ─────────────────────────────────────────────────────────────────
def analyze(trades: list, name: str, target_pct: float) -> dict:
    if not trades:
        return {"scenario": name, "trades": 0, "win_rate": 0, "avg_pnl": 0,
                "median_pnl": 0, "best": 0, "worst": 0, "avg_hold": 0,
                "target_pct": 0, "stop_pct": 0, "half_pct": 0, "expire_pct": 0,
                "total_pnl": 0, "target_cfg": target_pct * 100}
    df = pd.DataFrame(trades)
    wins = df[df["pnl_pct"] > 0]
    by_exit = df["exit_reason"].value_counts(normalize=True) * 100
    return {
        "scenario":   name,
        "trades":     len(df),
        "win_rate":   round(len(wins) / len(df) * 100, 1),
        "avg_pnl":    round(df["pnl_pct"].mean(), 2),
        "median_pnl": round(df["pnl_pct"].median(), 2),
        "best":       round(df["pnl_pct"].max(), 2),
        "worst":      round(df["pnl_pct"].min(), 2),
        "avg_hold":   round(df["hold_days"].mean(), 1),
        "total_pnl":  round(df["pnl_pct"].sum(), 1),
        "target_pct": round(by_exit.get("목표", 0), 1),
        "stop_pct":   round(by_exit.get("손절", 0), 1),
        "half_pct":   round(by_exit.get("60일수익", 0), 1),
        "expire_pct": round(by_exit.get("기간만료", 0), 1),
        "target_cfg": target_pct * 100,
    }

# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_comparison(summary_df: pd.DataFrame, all_trades: list):
    upper_df = summary_df[summary_df["scenario"].str.contains("상방")]
    lower_df = summary_df[summary_df["scenario"].str.contains("하방")]

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle(
        "기존 전략 vs 스퀴즈 단독 vs 스퀴즈 혼합 비교\n"
        "매도: 상방 +8%/-25%/60일수익/120일  |  하방 +20%/-25%/60일수익/120일",
        fontsize=13, fontweight="bold"
    )

    colors_u = ["#1565C0", "#42A5F5", "#0D47A1", "#90CAF9"]  # 파란계열 - 상방
    colors_d = ["#B71C1C", "#EF5350", "#7B1FA2", "#EF9A9A"]  # 빨간계열 - 하방

    def draw_bar(ax, df, col, title, colors, fmt="{:.1f}%"):
        vals = df[col].tolist()
        lbls = [s[:14] for s in df["scenario"].tolist()]
        bars = ax.bar(range(len(lbls)), vals, color=colors, alpha=0.85, edgecolor="white")
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xticks(range(len(lbls)))
        ax.set_xticklabels(lbls, rotation=35, ha="right", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (max(vals) - min(vals)) * 0.02 if vals else 0,
                    fmt.format(v), ha="center", va="bottom", fontsize=8)

    # 상방
    draw_bar(axes[0, 0], upper_df, "avg_pnl",  "상방 - 평균 수익률 (%)",  colors_u)
    draw_bar(axes[0, 1], upper_df, "win_rate",  "상방 - 승률 (%)",         colors_u)
    draw_bar(axes[0, 2], upper_df, "trades",    "상방 - 거래 수",           colors_u, fmt="{:.0f}건")

    # 하방
    draw_bar(axes[1, 0], lower_df, "avg_pnl",  "하방 - 평균 수익률 (%)",  colors_d)
    draw_bar(axes[1, 1], lower_df, "win_rate",  "하방 - 승률 (%)",         colors_d)
    draw_bar(axes[1, 2], lower_df, "trades",    "하방 - 거래 수",           colors_d, fmt="{:.0f}건")

    for ax in axes.flat:
        ax.set_ylim(bottom=min(ax.get_ylim()[0], -1))

    plt.tight_layout()
    plt.savefig("/Users/jungsoo.kim/Desktop/backtest/backtest_squeeze_compare.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("차트 저장: backtest_squeeze_compare.png")

    # ── 매도 사유 비교 ────────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
    fig2.suptitle("매도 사유 비율 비교", fontsize=12, fontweight="bold")

    for ax, df, colors, title in zip(
        axes2,
        [upper_df, lower_df],
        [colors_u, colors_d],
        ["200일선 상방 그룹", "200일선 하방 그룹"]
    ):
        x     = range(len(df))
        lbls  = [s[:14] for s in df["scenario"].tolist()]
        t_val = df["target_pct"].tolist()
        s_val = df["stop_pct"].tolist()
        h_val = df["half_pct"].tolist()
        e_val = df["expire_pct"].tolist()
        ax.bar(x, t_val, label="목표달성", color="#4CAF50", alpha=0.85)
        ax.bar(x, s_val, bottom=t_val, label="손절", color="#F44336", alpha=0.85)
        b2 = [a+b for a,b in zip(t_val, s_val)]
        ax.bar(x, h_val, bottom=b2, label="60일수익", color="#2196F3", alpha=0.85)
        b3 = [a+b for a,b in zip(b2, h_val)]
        ax.bar(x, e_val, bottom=b3, label="기간만료", color="#FF9800", alpha=0.85)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(range(len(lbls)))
        ax.set_xticklabels(lbls, rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig("/Users/jungsoo.kim/Desktop/backtest/backtest_squeeze_compare_exit.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("매도사유 차트 저장: backtest_squeeze_compare_exit.png")

    # ── 수익률 분포 비교 (박스플롯) ──────────────────────────────────────────
    all_df = pd.DataFrame(all_trades)
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))
    fig3.suptitle("시나리오별 수익률 분포", fontsize=12, fontweight="bold")

    for ax, df, colors, title, target in zip(
        axes3,
        [upper_df, lower_df],
        [colors_u, colors_d],
        ["200일선 상방", "200일선 하방"],
        [TARGET_U, TARGET_D]
    ):
        scens = df["scenario"].tolist()
        data_list = [all_df[all_df["scenario"]==s]["pnl_pct"].tolist() or [0] for s in scens]
        bp = ax.boxplot(data_list, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axhline(target*100, color="green", linewidth=1, linestyle=":",
                   label=f"+{target*100:.0f}% 목표")
        ax.axhline(-CIRCUIT_PCT*100, color="red", linewidth=1, linestyle=":",
                   label=f"-{CIRCUIT_PCT*100:.0f}% 손절")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(range(1, len(scens)+1))
        ax.set_xticklabels([s[:14] for s in scens], rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylabel("수익률 (%)")

    plt.tight_layout()
    plt.savefig("/Users/jungsoo.kim/Desktop/backtest/backtest_squeeze_compare_dist.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("분포 차트 저장: backtest_squeeze_compare_dist.png")

# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("기존 전략 vs 스퀴즈 단독 vs 스퀴즈 혼합 3-way 비교")
    print(f"기간: {START} ~ {END}")
    print(f"상방 매도: +{TARGET_U*100:.0f}% 목표 / -{CIRCUIT_PCT*100:.0f}% 손절 / 60일수익 / 120일")
    print(f"하방 매도: +{TARGET_D*100:.0f}% 목표 / -{CIRCUIT_PCT*100:.0f}% 손절 / 60일수익 / 120일")
    print("=" * 70)

    vix  = download_vix()
    data = download_data(ALL_TICKERS, vix)
    print(f"\n유효 종목: {len(data)}개\n")

    all_trades = []
    summary_rows = []

    for sc in SCENARIOS:
        name   = sc["name"]
        target = sc["target"]
        fn     = sc["fn"]
        print(f"[{name}] 실행 중...")
        trades = run_backtest(data, name, fn, target)
        all_trades.extend(trades)
        stats = analyze(trades, name, target)
        summary_rows.append(stats)
        print(f"  → 거래: {stats['trades']}건 / 승률: {stats['win_rate']}% / "
              f"평균수익: {stats['avg_pnl']}% / 목표:{stats['target_pct']}% 손절:{stats['stop_pct']}%")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        "/Users/jungsoo.kim/Desktop/backtest/backtest_squeeze_compare_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(all_trades).to_csv(
        "/Users/jungsoo.kim/Desktop/backtest/backtest_squeeze_compare_trades.csv",
        index=False, encoding="utf-8-sig"
    )

    # ── 최종 요약 출력 ────────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print(f"{'시나리오':<30} {'목표':>5} {'거래':>5} {'승률':>7} {'평균수익':>8} {'중간값':>8} "
          f"{'최고':>7} {'최악':>7} {'목표%':>7} {'손절%':>7} {'60일%':>7}")
    print("-" * 95)

    for _, r in summary_df.iterrows():
        side = "▲ 상방" if "상방" in r["scenario"] else "▼ 하방"
        print(f"{r['scenario']:<30} {r['target_cfg']:>4.0f}%  {r['trades']:>5}  "
              f"{r['win_rate']:>6.1f}%  {r['avg_pnl']:>7.2f}%  {r['median_pnl']:>7.2f}%  "
              f"{r['best']:>6.1f}%  {r['worst']:>6.1f}%  "
              f"{r['target_pct']:>6.1f}%  {r['stop_pct']:>6.1f}%  {r['half_pct']:>6.1f}%")

    print("=" * 95)

    print("\n─── 핵심 비교 요약 ───")
    for pair in [("S1_기존전략_상방", "S2_스퀴즈단독_상방", "S3_기존AND스퀴즈_상방", "S4_기존OR스퀴즈_상방"),
                 ("S5_기존전략_하방", "S6_스퀴즈단독_하방", "S7_기존AND스퀴즈_하방", "S8_기존OR스퀴즈_하방")]:
        side = "상방" if "상방" in pair[0] else "하방"
        print(f"\n[{side} 그룹]")
        rows = summary_df[summary_df["scenario"].isin(pair)]
        for _, r in rows.iterrows():
            tag = ""
            if "AND" in r["scenario"]: tag = " ← 기존에 스퀴즈 추가 AND"
            if "OR"  in r["scenario"]: tag = " ← 기존에 스퀴즈 추가 OR"
            if "단독" in r["scenario"]: tag = " ← 스퀴즈만"
            if "기존전략" in r["scenario"]: tag = " ← 현재 운용 중"
            print(f"  {r['scenario']:<30}{tag}")
            print(f"    거래:{r['trades']}건  승률:{r['win_rate']}%  평균수익:{r['avg_pnl']}%  "
                  f"목표달성:{r['target_pct']}%  손절:{r['stop_pct']}%")

    plot_comparison(summary_df, all_trades)
    print("\n완료!")

if __name__ == "__main__":
    main()
