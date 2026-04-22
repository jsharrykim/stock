"""
전략 A (200일 상방 + 스퀴즈 + 저가%B≤50) 보유 중
'매수 조건 이탈 → 관망 → 재충족' 복원에 대해
  허용 조건: (직전 매수일 종가 대비 -δ) OR (관망 진입 후 N거래일 경과)
그리드를 돌려 휩소(복원 횟수)와 지연을 비교한다.

※ 시트와 동일하게 보유는 유지하고 의견만 매수/관망이 바뀌는 가정 →
  청산 PnL 경로는 필터와 무관. 지표는 '복원 신호 횟수·차단·관망 체류'에 집중.
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

START = "2015-01-01"
END = "2026-04-05"

TARGET_PCT = 0.08
CIRCUIT_PCT = 0.25
HALF_EXIT_DAYS = 60
MAX_HOLD_DAYS = 120

BB_PERIOD = 20
BB_STD = 2.0
SQUEEZE_PERIOD = 60
SQUEEZE_RATIO = 0.5
PCTB_MAX = 50  # GAS SQUEEZE_PCT_B_MAX

KR_TICKERS = [
    "000660.KS",
    "005930.KS",
    "277810.KS",
    "034020.KS",
    "015760.KS",
    "005380.KS",
    "012450.KS",
    "042660.KS",
    "042700.KQ",
    "096770.KS",
    "009150.KS",
    "000270.KS",
    "247540.KQ",
    "376900.KQ",
    "079550.KS",
]
US_TICKERS = [
    "HOOD",
    "AVGO",
    "AMD",
    "MSFT",
    "GOOGL",
    "NVDA",
    "TSLA",
    "MU",
    "LRCX",
    "ON",
    "SNDK",
    "ASTS",
    "AVAV",
    "IONQ",
    "RKLB",
    "PLTR",
    "APP",
    "SOXL",
    "TSLL",
    "TE",
    "ONDS",
    "BE",
    "PL",
    "VRT",
    "LITE",
    "TER",
    "ANET",
    "IREN",
    "HOOG",
    "SOLT",
    "ETHU",
    "NBIS",
    "LPTH",
    "CONL",
    "GLW",
    "FLNC",
    "VST",
    "ASX",
    "SGML",
]
ALL_TICKERS = KR_TICKERS + US_TICKERS

DELTA_GRID = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
N_GRID = [1, 2, 3, 4, 5, 7]


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    low = df["Low"]
    df["ma200"] = close.rolling(200).mean()
    ma20 = close.rolling(BB_PERIOD).mean()
    std20 = close.rolling(BB_PERIOD).std()
    bb_u = ma20 + BB_STD * std20
    bb_l = ma20 - BB_STD * std20
    rng = bb_u - bb_l
    df["bb_width"] = np.where(ma20 > 0, (bb_u - bb_l) / ma20 * 100, np.nan)
    df["bb_width_avg"] = df["bb_width"].rolling(SQUEEZE_PERIOD).mean()
    df["squeeze"] = df["bb_width"] < df["bb_width_avg"] * SQUEEZE_RATIO
    df["pctb_low"] = np.where(rng > 0, (low - bb_l) / rng * 100, np.nan)
    return df


def raw_buy_a(row) -> bool:
    if pd.isna(row["ma200"]) or pd.isna(row["pctb_low"]) or pd.isna(row["squeeze"]):
        return False
    return (
        row["Close"] > row["ma200"]
        and bool(row["squeeze"])
        and row["pctb_low"] <= PCTB_MAX
    )


def simulate_ticker(df: pd.DataFrame, delta: float | None, n_days: int | None):
    """
    delta, n_days 가 None 이면 필터 없음 (베이스라인).
    앵커: 매수→관망 전환 직전 거래일 종가.
    허용: close <= anchor*(1-δ) OR (i - 관망_start_i) >= N
    """
    df = df.dropna(subset=["ma200", "pctb_low", "squeeze"]).reset_index(drop=True)
    if len(df) < 250:
        return None

    restores = 0
    total_blocked_days = 0  # 보유·관망·raw_buy True인데 필터로 막힌 일수
    quick_restores_within_2d = 0  # 관망 진입 후 2거래일 이내에 허용된 복원

    in_pos = False
    entry_price = 0.0
    entry_i = 0
    opinion_buy = False
    관망_start_i = None
    anchor_price = None

    def reentry_allowed(i, close):
        if delta is None or n_days is None or 관망_start_i is None:
            return True
        dd_ok = anchor_price is not None and close <= anchor_price * (1 - delta)
        days_ok = (i - 관망_start_i) >= n_days
        return dd_ok or days_ok

    for i in range(len(df)):
        row = df.iloc[i]
        close = row["Close"]
        buy = raw_buy_a(row)

        if in_pos:
            hold = i - entry_i
            pnl = (close - entry_price) / entry_price
            reason = None
            if pnl >= TARGET_PCT:
                reason = "목표"
            elif pnl <= -CIRCUIT_PCT:
                reason = "손절"
            elif hold >= HALF_EXIT_DAYS and pnl > 0:
                reason = "60일수익"
            elif hold >= MAX_HOLD_DAYS:
                reason = "기간만료"
            if reason:
                in_pos = False
                opinion_buy = False
                관망_start_i = None
                anchor_price = None

        if not in_pos:
            if buy:
                in_pos = True
                entry_price = close
                entry_i = i
                opinion_buy = True
                관망_start_i = None
                anchor_price = None
            continue

        # 보유 중
        if buy:
            if not opinion_buy:
                if reentry_allowed(i, close):
                    restores += 1
                    if 관망_start_i is not None and (i - 관망_start_i) <= 2:
                        quick_restores_within_2d += 1
                    opinion_buy = True
                    관망_start_i = None
                    anchor_price = None
                else:
                    total_blocked_days += 1
            # opinion_buy True: 유지
        else:
            if opinion_buy:
                opinion_buy = False
                관망_start_i = i
                anchor_price = float(df.iloc[i - 1]["Close"]) if i > 0 else float(close)

    return {
        "restores": restores,
        "quick_2d": quick_restores_within_2d,
        "blocked_wait_days": total_blocked_days,
    }


def merge_stats(rows):
    if not rows:
        return {}
    return {
        "restores": sum(r["restores"] for r in rows),
        "quick_2d": sum(r["quick_2d"] for r in rows),
        "blocked_wait_days": sum(r["blocked_wait_days"] for r in rows),
    }


def main():
    print("VIX + 종목 다운로드…")
    vix = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)[
        "Close"
    ].squeeze()
    vix.name = "vix"

    raw = yf.download(
        ALL_TICKERS, start=START, end=END, auto_adjust=True, progress=False, group_by="ticker"
    )
    data = {}
    for t in ALL_TICKERS:
        try:
            d = raw[t].copy() if len(ALL_TICKERS) > 1 else raw.copy()
            d = d.dropna(how="all")
            if len(d) < 250:
                continue
            data[t] = calc_indicators(d)
        except Exception as e:
            print(f"  skip {t}: {e}")

    print(f"로드 종목 수: {len(data)}")

    baseline_rows = []
    for t, df in data.items():
        r = simulate_ticker(df, None, None)
        if r:
            baseline_rows.append(r)
    base = merge_stats(baseline_rows)
    print("\n[베이스라인: 필터 없음]")
    print(f"  전체 복원 횟수(관망→매수): {base['restores']}")
    print(f"  관망 후 2거래일 이내 복원(휩소 프록시): {base['quick_2d']}")

    results = []
    for dlt in DELTA_GRID:
        for n in N_GRID:
            rows = []
            for t, df in data.items():
                r = simulate_ticker(df, dlt, n)
                if r:
                    rows.append(r)
            st = merge_stats(rows)
            results.append(
                {
                    "delta_pct": round(dlt * 100, 2),
                    "N_days": n,
                    "restores": st["restores"],
                    "quick_2d": st["quick_2d"],
                    "quick_2d_reduction_pct": round(
                        (1 - st["quick_2d"] / max(base["quick_2d"], 1)) * 100, 1
                    ),
                    "restore_reduction_pct": round(
                        (1 - st["restores"] / max(base["restores"], 1)) * 100, 1
                    ),
                    "blocked_bar_days": st["blocked_wait_days"],
                }
            )

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(
        ["quick_2d", "blocked_bar_days"], ascending=[True, True]
    )

    print("\n=== 상위 후보 (2일 이내 급복원 적을수록 우선, 같은면 차단일수 적게) ===")
    print(res_df.head(12).to_string(index=False))

    # 스코어: 휩소 프록시 대비 많이 줄이고, 전체 복원도 너무 죽이지 않기
    res_df["score"] = res_df["quick_2d_reduction_pct"] - 0.35 * res_df["restore_reduction_pct"]
    best = res_df.sort_values("score", ascending=False).iloc[0]

    print("\n=== 제안 (그리드 내 균형 스코어 최대) ===")
    print(
        f"  δ = {best['delta_pct']}% (직전 매수일 종가 대비 하락)\n"
        f"  N = {int(best['N_days'])}거래일 (관망 진입 후 경과 시 OR로 허용)\n"
        f"  → 2일 이내 급복원 {base['quick_2d']} → {int(best['quick_2d'])} "
        f"({best['quick_2d_reduction_pct']}% 감소), "
        f"전체 복원 {base['restores']} → {int(best['restores'])} "
        f"({best['restore_reduction_pct']}% 감소)"
    )

    out = "backtest_a_reentry_filter_grid.csv"
    res_df.sort_values(["delta_pct", "N_days"]).to_csv(out, index=False)
    print(f"\n전체 그리드 저장: {out}")


if __name__ == "__main__":
    main()
