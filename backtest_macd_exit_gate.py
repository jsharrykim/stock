"""
목표 수익률 도달 후 MACD 약화 확인 매도 백테스트

[비교 시나리오]
  1) baseline   : 목표 수익률 도달 즉시 매도
  2) macd_gate  : 목표 수익률 도달 후
                  - MACD 히스토그램 감소
                  - MACD 기울기 음수 전환
                  둘 중 하나가 나오면 매도

[전략]
  squeeze : 현재가 > MA200 AND BB 스퀴즈 AND 저가 %B <= 50   (목표 +8%)
  ma200u  : 현재가 > MA200 AND 저가 %B <= 5                  (목표 +8%)
  ma200d  : 현재가 < MA200 AND VIX >= 25 AND (RSI < 40 OR CCI < -100) (목표 +20%)

[공통 매도]
  - 손절 -25%
  - 60거래일 경과 + 수익 중
  - 120거래일 최대 보유
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))

import warnings

import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── 기간 ─────────────────────────────────────────────────────────────────────
START = "2015-01-01"
END = date.today().isoformat()

# ── 전략 파라미터 ─────────────────────────────────────────────────────────────
TARGET_SQUEEZE = 0.08
TARGET_MA200U = 0.08
TARGET_MA200D = 0.20
CIRCUIT_PCT = 0.25
HALF_EXIT_DAYS = 60
MAX_HOLD_DAYS = 120

VIX_MIN = 25
RSI_MAX = 40
CCI_MIN = -100
BB_PERIOD = 20
BB_STD = 2.0
SQUEEZE_PERIOD = 60
SQUEEZE_RATIO = 0.5
PCTB_LOW_MA200U = 5
PCTB_LOW_SQUEEZE = 50

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


def get_kr_font():
    for candidate in ["AppleGothic", "NanumGothic", "Malgun Gothic"]:
        if candidate in {f.name for f in fm.fontManager.ttflist}:
            return candidate
    return None


KR_FONT = get_kr_font()
if KR_FONT:
    import matplotlib

    matplotlib.rcParams["font.family"] = KR_FONT
    matplotlib.rcParams["axes.unicode_minus"] = False


# ── 지표 계산 ─────────────────────────────────────────────────────────────────
def calc_indicators(df: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    low = df["Low"]

    df["ma200"] = close.rolling(200).mean()

    ma20 = close.rolling(BB_PERIOD).mean()
    std20 = close.rolling(BB_PERIOD).std()
    bb_upper = ma20 + BB_STD * std20
    bb_lower = ma20 - BB_STD * std20
    bb_range = bb_upper - bb_lower

    df["bb_width"] = np.where(ma20 > 0, (bb_upper - bb_lower) / ma20 * 100, np.nan)
    df["bb_width_avg"] = df["bb_width"].rolling(SQUEEZE_PERIOD).mean()
    df["squeeze"] = df["bb_width"] < df["bb_width_avg"] * SQUEEZE_RATIO
    df["pctb_low"] = np.where(bb_range > 0, (low - bb_lower) / bb_range * 100, np.nan)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + np.where(loss == 0, 100.0, gain / loss))

    tp = (df["High"] + df["Low"] + close) / 3
    tp_ma = tp.rolling(14).mean()
    tp_md = tp.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = np.where(tp_md > 0, (tp - tp_ma) / (0.015 * tp_md), 0)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_hist_prev"] = df["macd_hist"].shift(1)
    df["macd_slope"] = df["macd"].diff()

    df = df.join(vix, how="left")
    df["vix"] = df["vix"].ffill()
    return df


def download_all(tickers, vix):
    print(f"종목 다운로드 중... ({len(tickers)}개)")
    try:
        raw = yf.download(
            tickers,
            start=START,
            end=END,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
    except Exception as exc:
        print(f"일괄 다운로드 실패: {exc}")
        raw = None

    result = {}
    for ticker in tickers:
        try:
            if raw is not None and len(tickers) > 1:
                df = raw[ticker].copy()
            elif raw is not None:
                df = raw.copy()
            else:
                df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)

            df.dropna(how="all", inplace=True)
            if len(df) < 250:
                continue
            result[ticker] = calc_indicators(df, vix)
        except Exception as exc:
            print(f"  [{ticker}] 오류: {exc}")

    print(f"  -> {len(result)}개 종목 로드 완료")
    return result


def strategy_signal(strategy: str, row) -> bool:
    close = row["Close"]
    ma200 = row["ma200"]
    pctb_low = row["pctb_low"] if not pd.isna(row["pctb_low"]) else 999
    squeeze = bool(row["squeeze"]) if not pd.isna(row["squeeze"]) else False
    vix_val = row["vix"] if not pd.isna(row["vix"]) else 0
    rsi = row["rsi"]
    cci = row["cci"]

    if strategy == "squeeze":
        return close > ma200 and squeeze and pctb_low <= PCTB_LOW_SQUEEZE
    if strategy == "ma200u":
        return close > ma200 and pctb_low <= PCTB_LOW_MA200U
    if strategy == "ma200d":
        return close < ma200 and vix_val >= VIX_MIN and ((rsi < RSI_MAX) or (cci < CCI_MIN))
    return False


def target_pct_for(strategy: str) -> float:
    return {
        "squeeze": TARGET_SQUEEZE,
        "ma200u": TARGET_MA200U,
        "ma200d": TARGET_MA200D,
    }[strategy]


def run_backtest(data: dict, strategy: str, exit_mode: str) -> list:
    trades = []
    req_cols = [
        "ma200", "rsi", "cci", "pctb_low", "bb_width_avg", "squeeze", "vix",
        "macd", "macd_hist", "macd_hist_prev", "macd_slope",
    ]
    target_pct = target_pct_for(strategy)
    scenario = exit_mode

    for ticker, df in data.items():
        df_c = df.dropna(subset=req_cols).copy()
        if len(df_c) < 50:
            continue

        in_position = False
        entry_price = 0.0
        entry_idx = 0
        entry_date = None
        target_armed = False
        target_armed_date = None

        rows = df_c.to_dict("index")
        idx_list = list(df_c.index)

        for ii, current_date in enumerate(idx_list):
            row = rows[current_date]
            close = row["Close"]

            if in_position:
                hold_days = ii - entry_idx
                pnl = (close - entry_price) / entry_price

                reason = None
                macd_exit_detail = ""

                if pnl <= -CIRCUIT_PCT:
                    reason = "손절"
                elif hold_days >= HALF_EXIT_DAYS and pnl > 0:
                    reason = "60일수익"
                elif hold_days >= MAX_HOLD_DAYS:
                    reason = "기간만료"
                elif exit_mode == "baseline" and pnl >= target_pct:
                    reason = "목표"
                else:
                    if pnl >= target_pct and not target_armed:
                        target_armed = True
                        target_armed_date = current_date

                    macd_ready = False
                    if exit_mode == "macd_gate_armed":
                        macd_ready = target_armed
                    elif exit_mode == "macd_gate_strict":
                        macd_ready = pnl >= target_pct

                    if exit_mode in ("macd_gate_armed", "macd_gate_strict") and macd_ready:
                        hist_shrinking = row["macd_hist"] < row["macd_hist_prev"]
                        slope_negative = row["macd_slope"] < 0
                        if hist_shrinking or slope_negative:
                            if hist_shrinking and slope_negative:
                                macd_exit_detail = "목표후_MACD히스토감소+기울기음수"
                            elif hist_shrinking:
                                macd_exit_detail = "목표후_MACD히스토감소"
                            else:
                                macd_exit_detail = "목표후_MACD기울기음수"
                            reason = macd_exit_detail

                if reason:
                    trades.append({
                        "scenario": scenario,
                        "strategy": strategy,
                        "ticker": ticker,
                        "entry_date": entry_date,
                        "exit_date": current_date,
                        "entry_price": round(entry_price, 4),
                        "exit_price": round(float(close), 4),
                        "pnl_pct": round(pnl * 100, 2),
                        "hold_days": hold_days,
                        "exit_reason": reason,
                        "target_armed": target_armed,
                        "target_armed_date": target_armed_date,
                    })
                    in_position = False
                    target_armed = False
                    target_armed_date = None

            if not in_position and strategy_signal(strategy, row):
                in_position = True
                entry_price = float(close)
                entry_idx = ii
                entry_date = current_date
                target_armed = False
                target_armed_date = None

        if in_position:
            last = df_c.iloc[-1]
            pnl = (last["Close"] - entry_price) / entry_price
            hold_days = len(df_c) - 1 - entry_idx
            trades.append({
                "scenario": scenario,
                "strategy": strategy,
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": df_c.index[-1],
                "entry_price": round(entry_price, 4),
                "exit_price": round(float(last["Close"]), 4),
                "pnl_pct": round(pnl * 100, 2),
                "hold_days": hold_days,
                "exit_reason": "미청산",
                "target_armed": target_armed,
                "target_armed_date": target_armed_date,
            })

    return trades


def calc_stats(trades: list) -> dict:
    if not trades:
        return {
            "n": 0,
            "win_rate": 0,
            "avg_pnl": 0,
            "median_pnl": 0,
            "ev": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "avg_hold": 0,
        }

    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    holds = [t["hold_days"] for t in trades]

    return {
        "n": len(trades),
        "win_rate": round(len(wins) / len(pnls) * 100, 1),
        "avg_pnl": round(float(np.mean(pnls)), 2),
        "median_pnl": round(float(np.median(pnls)), 2),
        "ev": round(float(np.mean(pnls)), 2),
        "avg_win": round(float(np.mean(wins)) if wins else 0, 2),
        "avg_loss": round(float(np.mean(losses)) if losses else 0, 2),
        "avg_hold": round(float(np.mean(holds)) if holds else 0, 1),
    }


def print_compare(summary_df: pd.DataFrame):
    print("\n" + "=" * 96)
    print("전략별 비교: baseline vs macd_gate_armed vs macd_gate_strict")
    print("=" * 96)
    header = (
        f"{'전략':<10} {'시나리오':<12} {'거래':>6} {'승률':>7} "
        f"{'평균수익':>10} {'EV':>8} {'중앙값':>8} {'평균보유':>10}"
    )
    print(header)
    print("-" * 96)

    for strategy in ["squeeze", "ma200u", "ma200d"]:
        df_s = summary_df[summary_df["strategy"] == strategy].copy()
        for scenario in ["baseline", "macd_gate_armed", "macd_gate_strict"]:
            row = df_s[df_s["scenario"] == scenario].iloc[0]
            print(
                f"{strategy:<10} {scenario:<12} {int(row['n']):6d} {row['win_rate']:6.1f}% "
                f"{row['avg_pnl']:>+9.2f}% {row['ev']:>+7.2f}% {row['median_pnl']:>+7.2f}% "
                f"{row['avg_hold']:9.1f}일"
            )

        base = df_s[df_s["scenario"] == "baseline"].iloc[0]
        for scenario in ["macd_gate_armed", "macd_gate_strict"]:
            gate = df_s[df_s["scenario"] == scenario].iloc[0]
            print(
                f"{'delta':<10} {scenario+'-base':<12} "
                f"{int(gate['n'] - base['n']):6d} {(gate['win_rate'] - base['win_rate']):+6.1f}% "
                f"{(gate['avg_pnl'] - base['avg_pnl']):+9.2f}% {(gate['ev'] - base['ev']):+7.2f}% "
                f"{(gate['median_pnl'] - base['median_pnl']):+7.2f}% {(gate['avg_hold'] - base['avg_hold']):+9.1f}일"
            )
        print("-" * 96)


def print_exit_breakdown(trades_df: pd.DataFrame):
    print("\n청산 사유 분포")
    print("-" * 96)
    grouped = (
        trades_df.groupby(["strategy", "scenario", "exit_reason"])
        .size()
        .reset_index(name="count")
        .sort_values(["strategy", "scenario", "count"], ascending=[True, True, False])
    )

    for strategy in ["squeeze", "ma200u", "ma200d"]:
        for scenario in ["baseline", "macd_gate_armed", "macd_gate_strict"]:
            subset = grouped[(grouped["strategy"] == strategy) & (grouped["scenario"] == scenario)]
            if subset.empty:
                continue
            parts = [f"{row.exit_reason}:{int(row.count)}" for row in subset.itertuples()]
            print(f"{strategy:<10} {scenario:<12} " + " | ".join(parts))


def main():
    print("=" * 72)
    print("목표 수익률 도달 후 MACD 약화 확인 매도 백테스트")
    print("=" * 72)
    print(f"기간: {START} ~ {END}")
    print(f"종목 수: {len(ALL_TICKERS)}")

    print("\n[1] VIX 다운로드")
    vix_raw = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)
    vix = vix_raw["Close"].squeeze()
    vix.name = "vix"
    print(f"  VIX: {len(vix)}일")

    print("\n[2] 종목 데이터 다운로드")
    data = download_all(ALL_TICKERS, vix)

    print("\n[3] 백테스트 실행")
    all_trades = []
    for strategy in ["squeeze", "ma200u", "ma200d"]:
        for scenario in ["baseline", "macd_gate_armed", "macd_gate_strict"]:
            trades = run_backtest(data, strategy, scenario)
            all_trades.extend(trades)
            stats = calc_stats(trades)
            print(
                f"  [{strategy:8s} / {scenario:9s}] "
                f"거래 {stats['n']:3d}건 | 승률 {stats['win_rate']:5.1f}% | "
                f"평균 {stats['avg_pnl']:+6.2f}% | EV {stats['ev']:+6.2f}% | "
                f"보유 {stats['avg_hold']:5.1f}일"
            )

    trades_df = pd.DataFrame(all_trades)

    print("\n[4] 요약 저장")
    rows = []
    for strategy in ["squeeze", "ma200u", "ma200d"]:
        for scenario in ["baseline", "macd_gate_armed", "macd_gate_strict"]:
            trades = trades_df[
                (trades_df["strategy"] == strategy) &
                (trades_df["scenario"] == scenario)
            ].to_dict("records")
            rows.append({
                "strategy": strategy,
                "scenario": scenario,
                **calc_stats(trades),
            })

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(os.path.dirname(__file__), "backtest_macd_exit_gate_summary.csv")
    trades_path = os.path.join(os.path.dirname(__file__), "backtest_macd_exit_gate_trades.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
    print(f"  -> {os.path.basename(summary_path)} 저장")
    print(f"  -> {os.path.basename(trades_path)} 저장")

    print_compare(summary_df)
    print_exit_breakdown(trades_df)

    print("\n완료")


if __name__ == "__main__":
    main()
