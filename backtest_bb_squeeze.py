"""
backtest_bb_squeeze.py
========================
볼린저밴드 스퀴즈 + 200일선 상/하방 그룹별 백테스트

[스퀴즈 정의]
  BB폭(%) = (BB상단 - BB하단) / MA20 * 100
  스퀴즈 조건: BB폭 < BB폭 60일 평균 × 0.5

[A그룹 - 200일선 상방]
  A1: 상방 + 스퀴즈 + 저가%B ≤ 5
  A2: 상방 + 스퀴즈 + RSI < 50
  A3: 상방 + 스퀴즈 + CCI < 0
  A4: 상방 + 스퀴즈 + RSI < 50 + CCI < 0
  A5: 상방 + 스퀴즈만 (베이스라인)

[B그룹 - 200일선 하방]
  B1: 하방 + 스퀴즈 + RSI < 40
  B2: 하방 + 스퀴즈 + CCI < -100
  B3: 하방 + 스퀴즈 + RSI < 40 + CCI < -100
  B4: 하방 + 스퀴즈만 (베이스라인)

[매도 조건]
  목표: +10% / 손절: -15% / 최대 보유: 60거래일

종목: 사용자 모니터링 49개
기간: 전체 가능 기간
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
START           = "2015-01-01"
END             = "2026-03-17"
TARGET_PCT      = 0.10    # 목표 수익률
STOP_PCT        = 0.15    # 손절률
MAX_HOLD        = 60      # 최대 보유일
BB_PERIOD       = 20
BB_STD          = 2.0
SQUEEZE_PERIOD  = 60      # 스퀴즈 판단 기준 이동평균 기간
SQUEEZE_RATIO   = 0.5     # BB폭 < 60일 평균 × 이 비율
MAX_POSITIONS   = 5       # 동시 최대 포지션
MAX_DAILY       = 3       # 일별 최대 진입

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
    candidates = ["AppleGothic", "NanumGothic", "Malgun Gothic", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return None

KR_FONT = get_kr_font()
if KR_FONT:
    plt.rcParams["font.family"] = KR_FONT
plt.rcParams["axes.unicode_minus"] = False

# ── 지표 계산 ─────────────────────────────────────────────────────────────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # MA200
    df["ma200"] = close.rolling(200).mean()

    # BB
    ma20 = close.rolling(BB_PERIOD).mean()
    std20 = close.rolling(BB_PERIOD).std()
    df["bb_upper"] = ma20 + BB_STD * std20
    df["bb_lower"] = ma20 - BB_STD * std20
    df["ma20"]     = ma20

    # BB폭(%) = (upper - lower) / ma20 * 100
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / ma20 * 100

    # 스퀴즈: BB폭 < 60일 평균 BB폭 × SQUEEZE_RATIO
    df["bb_width_avg60"] = df["bb_width"].rolling(SQUEEZE_PERIOD).mean()
    df["squeeze"] = df["bb_width"] < df["bb_width_avg60"] * SQUEEZE_RATIO

    # 저가 %B = (low - bb_lower) / (bb_upper - bb_lower) * 100
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["pctb_low"] = np.where(bb_range > 0, (low - df["bb_lower"]) / bb_range * 100, np.nan)

    # RSI(14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = np.where(loss == 0, 100.0, gain / loss)
    df["rsi"] = 100 - 100 / (1 + rs)

    # CCI(14)
    tp    = (high + low + close) / 3
    tp_ma = tp.rolling(14).mean()
    tp_md = tp.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = np.where(tp_md > 0, (tp - tp_ma) / (0.015 * tp_md), 0)

    return df

# ── 데이터 다운로드 ────────────────────────────────────────────────────────────
def download_data(tickers):
    print(f"데이터 다운로드 중... ({len(tickers)}개 종목)")
    raw = yf.download(
        tickers, start=START, end=END,
        auto_adjust=True, progress=False,
        group_by="ticker"
    )
    result = {}
    for t in tickers:
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[t].copy()
            df.dropna(how="all", inplace=True)
            if len(df) < 250:
                print(f"  [{t}] 데이터 부족 ({len(df)}일) - 제외")
                continue
            df = calc_indicators(df)
            result[t] = df
            print(f"  [{t}] {len(df)}일 로드 완료")
        except Exception as e:
            print(f"  [{t}] 오류: {e}")
    return result

# ── 진입 조건 정의 ────────────────────────────────────────────────────────────
GROUPS = {
    # 200일선 상방
    "A1_상방_스퀴즈_저가%B≤5":       lambda r: r["above200"] and r["squeeze"] and r["pctb_low"] <= 5,
    "A2_상방_스퀴즈_RSI<50":         lambda r: r["above200"] and r["squeeze"] and r["rsi"] < 50,
    "A3_상방_스퀴즈_CCI<0":          lambda r: r["above200"] and r["squeeze"] and r["cci"] < 0,
    "A4_상방_스퀴즈_RSI<50+CCI<0":   lambda r: r["above200"] and r["squeeze"] and r["rsi"] < 50 and r["cci"] < 0,
    "A5_상방_스퀴즈만":              lambda r: r["above200"] and r["squeeze"],
    # 200일선 하방
    "B1_하방_스퀴즈_RSI<40":         lambda r: not r["above200"] and r["squeeze"] and r["rsi"] < 40,
    "B2_하방_스퀴즈_CCI<-100":       lambda r: not r["above200"] and r["squeeze"] and r["cci"] < -100,
    "B3_하방_스퀴즈_RSI<40+CCI<-100":lambda r: not r["above200"] and r["squeeze"] and r["rsi"] < 40 and r["cci"] < -100,
    "B4_하방_스퀴즈만":              lambda r: not r["above200"] and r["squeeze"],
}

# ── 백테스트 엔진 ─────────────────────────────────────────────────────────────
def run_backtest(data: dict, group_name: str, condition_fn) -> list:
    trades = []

    for ticker, df in data.items():
        df = df.dropna(subset=["ma200", "bb_upper", "bb_lower", "squeeze", "rsi", "cci", "pctb_low"])
        if len(df) < 10:
            continue

        in_position = False
        entry_price = None
        entry_date  = None
        entry_idx   = None

        for i in range(len(df)):
            row = df.iloc[i]
            date = df.index[i]

            if in_position:
                hold_days = i - entry_idx
                pnl = (row["Close"] - entry_price) / entry_price

                exit_reason = None
                if pnl >= TARGET_PCT:
                    exit_reason = "목표"
                elif pnl <= -STOP_PCT:
                    exit_reason = "손절"
                elif hold_days >= MAX_HOLD:
                    exit_reason = "기간만료"

                if exit_reason:
                    trades.append({
                        "group": group_name,
                        "ticker": ticker,
                        "entry_date": entry_date,
                        "exit_date": date,
                        "entry_price": entry_price,
                        "exit_price": row["Close"],
                        "pnl_pct": round(pnl * 100, 2),
                        "hold_days": hold_days,
                        "exit_reason": exit_reason,
                        "above200": row["Close"] > row["ma200"],
                        "squeeze_ratio": round(row["bb_width"] / row["bb_width_avg60"], 3) if row["bb_width_avg60"] > 0 else None,
                    })
                    in_position = False

            if not in_position:
                # NaN 체크
                if pd.isna(row["ma200"]) or pd.isna(row["bb_width_avg60"]):
                    continue
                if pd.isna(row["rsi"]) or pd.isna(row["cci"]) or pd.isna(row["pctb_low"]):
                    continue

                signal_row = {
                    "above200": row["Close"] > row["ma200"],
                    "squeeze":  bool(row["squeeze"]),
                    "pctb_low": row["pctb_low"],
                    "rsi":      row["rsi"],
                    "cci":      row["cci"],
                }

                try:
                    if condition_fn(signal_row):
                        in_position = True
                        entry_price = row["Close"]
                        entry_date  = date
                        entry_idx   = i
                except Exception:
                    pass

        # 미청산 포지션 처리
        if in_position and entry_price and len(df) > 0:
            last_row = df.iloc[-1]
            pnl = (last_row["Close"] - entry_price) / entry_price
            trades.append({
                "group": group_name,
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": df.index[-1],
                "entry_price": entry_price,
                "exit_price": last_row["Close"],
                "pnl_pct": round(pnl * 100, 2),
                "hold_days": len(df) - 1 - entry_idx,
                "exit_reason": "미청산",
                "above200": last_row["Close"] > last_row["ma200"],
                "squeeze_ratio": None,
            })

    return trades

# ── 결과 분석 ─────────────────────────────────────────────────────────────────
def analyze(trades: list, group_name: str) -> dict:
    if not trades:
        return {
            "group": group_name, "trades": 0,
            "win_rate": 0, "avg_pnl": 0, "median_pnl": 0,
            "avg_hold": 0, "total_pnl": 0,
            "best": 0, "worst": 0,
            "target_pct": 0, "stop_pct": 0, "expire_pct": 0
        }
    df = pd.DataFrame(trades)
    wins = df[df["pnl_pct"] > 0]
    by_exit = df["exit_reason"].value_counts(normalize=True) * 100
    return {
        "group": group_name,
        "trades": len(df),
        "win_rate": round(len(wins) / len(df) * 100, 1),
        "avg_pnl": round(df["pnl_pct"].mean(), 2),
        "median_pnl": round(df["pnl_pct"].median(), 2),
        "avg_hold": round(df["hold_days"].mean(), 1),
        "total_pnl": round(df["pnl_pct"].sum(), 1),
        "best": round(df["pnl_pct"].max(), 2),
        "worst": round(df["pnl_pct"].min(), 2),
        "target_pct": round(by_exit.get("목표", 0), 1),
        "stop_pct": round(by_exit.get("손절", 0), 1),
        "expire_pct": round(by_exit.get("기간만료", 0), 1),
    }

# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_results(summary_df: pd.DataFrame, all_trades: list):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("BB 스퀴즈 백테스트 결과\n(매수: 스퀴즈 진입 / 매도: +10% 목표 / -15% 손절 / 60일 만료)",
                 fontsize=14, fontweight="bold")

    groups = summary_df["group"].tolist()
    colors_a = ["#2196F3", "#42A5F5", "#64B5F6", "#90CAF9", "#BBDEFB"]
    colors_b = ["#F44336", "#EF5350", "#E57373", "#EF9A9A"]
    bar_colors = colors_a + colors_b

    # ① 평균 수익률
    ax = axes[0, 0]
    bars = ax.bar(range(len(groups)), summary_df["avg_pnl"], color=bar_colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("평균 수익률 (%)", fontweight="bold")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[:12] for g in groups], rotation=40, ha="right", fontsize=8)
    for bar, val in zip(bars, summary_df["avg_pnl"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    # ② 승률
    ax = axes[0, 1]
    bars = ax.bar(range(len(groups)), summary_df["win_rate"], color=bar_colors, alpha=0.85, edgecolor="white")
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--", label="50% 기준")
    ax.set_title("승률 (%)", fontweight="bold")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[:12] for g in groups], rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    for bar, val in zip(bars, summary_df["win_rate"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

    # ③ 거래 수
    ax = axes[1, 0]
    bars = ax.bar(range(len(groups)), summary_df["trades"], color=bar_colors, alpha=0.85, edgecolor="white")
    ax.set_title("총 거래 수", fontweight="bold")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[:12] for g in groups], rotation=40, ha="right", fontsize=8)
    for bar, val in zip(bars, summary_df["trades"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(int(val)), ha="center", va="bottom", fontsize=8)

    # ④ 매도 사유 스택 바
    ax = axes[1, 1]
    x = range(len(groups))
    ax.bar(x, summary_df["target_pct"], label="목표달성", color="#4CAF50", alpha=0.85)
    ax.bar(x, summary_df["stop_pct"], bottom=summary_df["target_pct"], label="손절", color="#F44336", alpha=0.85)
    ax.bar(x, summary_df["expire_pct"],
           bottom=summary_df["target_pct"] + summary_df["stop_pct"],
           label="기간만료", color="#FF9800", alpha=0.85)
    ax.set_title("매도 사유 비율 (%)", fontweight="bold")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[:12] for g in groups], rotation=40, ha="right", fontsize=8)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("/Users/jungsoo.kim/Desktop/backtest/backtest_bb_squeeze.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("차트 저장: backtest_bb_squeeze.png")

    # 수익률 분포 박스플롯
    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 7))
    fig2.suptitle("그룹별 수익률 분포", fontsize=13, fontweight="bold")

    all_df = pd.DataFrame(all_trades)
    a_groups = [g for g in groups if g.startswith("A")]
    b_groups = [g for g in groups if g.startswith("B")]

    for ax, grp_list, title in zip(axes2, [a_groups, b_groups],
                                   ["200일선 상방 그룹 (A)", "200일선 하방 그룹 (B)"]):
        data_list = [all_df[all_df["group"] == g]["pnl_pct"].tolist() for g in grp_list]
        data_list = [d if d else [0] for d in data_list]
        bp = ax.boxplot(data_list, patch_artist=True, notch=False)
        clrs = colors_a if title.startswith("200일선 상방") else colors_b
        for patch, c in zip(bp["boxes"], clrs):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axhline(TARGET_PCT * 100, color="green", linewidth=0.8, linestyle=":", label=f"+{TARGET_PCT*100:.0f}% 목표")
        ax.axhline(-STOP_PCT * 100, color="red", linewidth=0.8, linestyle=":", label=f"-{STOP_PCT*100:.0f}% 손절")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(range(1, len(grp_list) + 1))
        ax.set_xticklabels([g[:14] for g in grp_list], rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylabel("수익률 (%)")

    plt.tight_layout()
    plt.savefig("/Users/jungsoo.kim/Desktop/backtest/backtest_bb_squeeze_dist.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("분포 차트 저장: backtest_bb_squeeze_dist.png")

# ── 종목별 스퀴즈 현황 체크 ───────────────────────────────────────────────────
def check_current_squeeze(data: dict):
    """현재(최근 데이터) 기준 스퀴즈 종목 현황"""
    rows = []
    for ticker, df in data.items():
        df_clean = df.dropna(subset=["ma200", "bb_width_avg60", "squeeze"])
        if len(df_clean) == 0:
            continue
        last = df_clean.iloc[-1]
        rows.append({
            "종목": ticker,
            "종가": round(last["Close"], 2),
            "MA200": round(last["ma200"], 2),
            "200일선": "상방" if last["Close"] > last["ma200"] else "하방",
            "BB폭(%)": round(last["bb_width"], 2),
            "BB폭60일평균": round(last["bb_width_avg60"], 2),
            "스퀴즈비율": round(last["bb_width"] / last["bb_width_avg60"], 3) if last["bb_width_avg60"] > 0 else None,
            "스퀴즈": "✓ 스퀴즈" if last["squeeze"] else "",
            "RSI": round(last["rsi"], 1) if not pd.isna(last["rsi"]) else None,
            "CCI": round(last["cci"], 1) if not pd.isna(last["cci"]) else None,
            "저가%B": round(last["pctb_low"], 1) if not pd.isna(last["pctb_low"]) else None,
        })
    result = pd.DataFrame(rows).sort_values(["200일선", "스퀴즈비율"])
    return result

# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("BB 스퀴즈 백테스트")
    print(f"기간: {START} ~ {END}")
    print(f"목표: +{TARGET_PCT*100:.0f}% / 손절: -{STOP_PCT*100:.0f}% / 최대{MAX_HOLD}일")
    print(f"스퀴즈: BB폭 < 60일평균 × {SQUEEZE_RATIO}")
    print("=" * 60)

    # 데이터 다운로드
    data = download_data(ALL_TICKERS)
    print(f"\n유효 종목: {len(data)}개\n")

    # 현재 스퀴즈 현황
    print("\n─── 현재 스퀴즈 현황 ───")
    squeeze_now = check_current_squeeze(data)
    print(squeeze_now.to_string(index=False))
    squeeze_now.to_csv("/Users/jungsoo.kim/Desktop/backtest/squeeze_current_status.csv",
                       index=False, encoding="utf-8-sig")
    print("\n현재 스퀴즈 현황 저장: squeeze_current_status.csv")

    # 그룹별 백테스트
    all_trades = []
    summary_rows = []

    for group_name, condition_fn in GROUPS.items():
        print(f"\n[{group_name}] 백테스트 실행 중...")
        trades = run_backtest(data, group_name, condition_fn)
        all_trades.extend(trades)
        stats = analyze(trades, group_name)
        summary_rows.append(stats)
        print(f"  → 거래: {stats['trades']}건 / 승률: {stats['win_rate']}% / "
              f"평균수익: {stats['avg_pnl']}% / 목표:{stats['target_pct']}% 손절:{stats['stop_pct']}%")

    # 결과 저장
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("/Users/jungsoo.kim/Desktop/backtest/backtest_bb_squeeze_summary.csv",
                      index=False, encoding="utf-8-sig")
    print("\n요약 저장: backtest_bb_squeeze_summary.csv")

    all_trades_df = pd.DataFrame(all_trades)
    all_trades_df.to_csv("/Users/jungsoo.kim/Desktop/backtest/backtest_bb_squeeze_trades.csv",
                         index=False, encoding="utf-8-sig")
    print("거래내역 저장: backtest_bb_squeeze_trades.csv")

    # 최종 요약 출력
    print("\n" + "=" * 80)
    print(f"{'그룹':<30} {'거래':>5} {'승률':>7} {'평균수익':>8} {'중간값':>8} {'최고':>7} {'최악':>7} {'목표%':>7} {'손절%':>7}")
    print("-" * 80)
    for _, r in summary_df.iterrows():
        print(f"{r['group']:<30} {r['trades']:>5} {r['win_rate']:>6.1f}% "
              f"{r['avg_pnl']:>7.2f}% {r['median_pnl']:>7.2f}% "
              f"{r['best']:>6.1f}% {r['worst']:>6.1f}% "
              f"{r['target_pct']:>6.1f}% {r['stop_pct']:>6.1f}%")
    print("=" * 80)

    # 시각화
    plot_results(summary_df, all_trades)
    print("\n완료!")

if __name__ == "__main__":
    main()
