"""
상방 2개 전략 전용:
strict + MACD histogram only + 최대 대기일 제한 백테스트

규칙:
1) 목표 수익률(8%) 첫 도달 시점을 기록
2) 그 이후 현재 수익률이 여전히 목표 이상이고
   MACD histogram 이 전일보다 감소하면 매도
3) 다만 first target hit 이후 max_wait_days 가 지나도록
   신호가 없으면 강제 매도
"""

import os
import pandas as pd
import yfinance as yf

import backtest_macd_exit_gate as mod


WAIT_DAYS_GRID = [0, 1, 2, 3, 4, 5]
UPPER_STRATEGIES = ["squeeze", "ma200u"]


def run_wait_limited_hist_strict(data: dict, strategy: str, max_wait_days: int) -> list:
    trades = []
    req_cols = [
        "ma200", "rsi", "cci", "pctb_low", "bb_width_avg", "squeeze", "vix",
        "macd_hist", "macd_hist_prev",
    ]
    target_pct = mod.target_pct_for(strategy)

    for ticker, df in data.items():
        df_c = df.dropna(subset=req_cols).copy()
        if len(df_c) < 50:
            continue

        in_position = False
        entry_price = 0.0
        entry_idx = 0
        entry_date = None
        first_target_idx = None
        first_target_date = None

        rows = df_c.to_dict("index")
        idx_list = list(df_c.index)

        for ii, current_date in enumerate(idx_list):
            row = rows[current_date]
            close = row["Close"]

            if in_position:
                hold_days = ii - entry_idx
                pnl = (close - entry_price) / entry_price
                reason = None
                waited_days = None

                if pnl <= -mod.CIRCUIT_PCT:
                    reason = "손절"
                elif hold_days >= mod.HALF_EXIT_DAYS and pnl > 0:
                    reason = "60일수익"
                elif hold_days >= mod.MAX_HOLD_DAYS:
                    reason = "기간만료"
                else:
                    if first_target_idx is None and pnl >= target_pct:
                        first_target_idx = ii
                        first_target_date = current_date

                    if first_target_idx is not None:
                        waited_days = ii - first_target_idx
                        hist_shrinking = row["macd_hist"] < row["macd_hist_prev"]

                        if pnl >= target_pct and hist_shrinking:
                            reason = "목표중_MACD히스토감소"
                        elif waited_days >= max_wait_days:
                            reason = "목표후대기만료"

                if reason:
                    trades.append({
                        "strategy": strategy,
                        "scenario": f"wait_{max_wait_days}d",
                        "ticker": ticker,
                        "entry_date": entry_date,
                        "exit_date": current_date,
                        "pnl_pct": round(pnl * 100, 2),
                        "hold_days": hold_days,
                        "exit_reason": reason,
                        "first_target_date": first_target_date,
                        "waited_days": waited_days,
                    })
                    in_position = False
                    first_target_idx = None
                    first_target_date = None

            if not in_position and mod.strategy_signal(strategy, row):
                in_position = True
                entry_price = float(close)
                entry_idx = ii
                entry_date = current_date
                first_target_idx = None
                first_target_date = None

        if in_position:
            last = df_c.iloc[-1]
            pnl = (last["Close"] - entry_price) / entry_price
            hold_days = len(df_c) - 1 - entry_idx
            trades.append({
                "strategy": strategy,
                "scenario": f"wait_{max_wait_days}d",
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": df_c.index[-1],
                "pnl_pct": round(pnl * 100, 2),
                "hold_days": hold_days,
                "exit_reason": "미청산",
                "first_target_date": first_target_date,
                "waited_days": (len(df_c) - 1 - first_target_idx) if first_target_idx is not None else None,
            })

    return trades


def calc_extra_stats(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {
            "timeout_cnt": 0,
            "timeout_below_target_cnt": 0,
            "timeout_below_target_pct": 0.0,
            "macd_exit_cnt": 0,
        }

    timeout = trades_df[trades_df["exit_reason"] == "목표후대기만료"]
    macd_exit = trades_df[trades_df["exit_reason"] == "목표중_MACD히스토감소"]
    timeout_below = timeout[timeout["pnl_pct"] < 8.0]
    return {
        "timeout_cnt": int(len(timeout)),
        "timeout_below_target_cnt": int(len(timeout_below)),
        "timeout_below_target_pct": round(len(timeout_below) / len(trades_df) * 100, 1),
        "macd_exit_cnt": int(len(macd_exit)),
    }


def main():
    print("=" * 80)
    print("상방 2개 전략: strict + histogram only + 최대 대기일 그리드")
    print("=" * 80)
    print(f"기간: {mod.START} ~ {mod.END}")

    print("\n[1] VIX 다운로드")
    vix_raw = yf.download("^VIX", start=mod.START, end=mod.END, auto_adjust=True, progress=False)
    vix = vix_raw["Close"].squeeze()
    vix.name = "vix"
    print(f"  VIX: {len(vix)}일")

    print("\n[2] 종목 데이터 다운로드")
    data = mod.download_all(mod.ALL_TICKERS, vix)

    print("\n[3] 백테스트 실행")
    all_rows = []
    all_trades = []
    for strategy in UPPER_STRATEGIES:
        baseline = mod.run_backtest(data, strategy, "baseline")
        base_stats = mod.calc_stats(baseline)
        all_rows.append({
            "strategy": strategy,
            "scenario": "baseline",
            **base_stats,
            **calc_extra_stats(pd.DataFrame(baseline)),
        })
        all_trades.extend(baseline)
        print(
            f"  [{strategy:8s} / baseline] "
            f"거래 {base_stats['n']:4d}건 | 승률 {base_stats['win_rate']:5.1f}% | "
            f"평균 {base_stats['avg_pnl']:+6.2f}% | 보유 {base_stats['avg_hold']:5.1f}일"
        )

        for wait_days in WAIT_DAYS_GRID:
            trades = run_wait_limited_hist_strict(data, strategy, wait_days)
            stats = mod.calc_stats(trades)
            extra = calc_extra_stats(pd.DataFrame(trades))
            all_rows.append({
                "strategy": strategy,
                "scenario": f"wait_{wait_days}d",
                "wait_days": wait_days,
                **stats,
                **extra,
            })
            all_trades.extend(trades)
            print(
                f"  [{strategy:8s} / wait_{wait_days}d] "
                f"거래 {stats['n']:4d}건 | 승률 {stats['win_rate']:5.1f}% | "
                f"평균 {stats['avg_pnl']:+6.2f}% | 보유 {stats['avg_hold']:5.1f}일 | "
                f"대기만료<8 {extra['timeout_below_target_cnt']:3d}건"
            )

    summary_df = pd.DataFrame(all_rows)
    trades_df = pd.DataFrame(all_trades)

    summary_path = os.path.join(os.path.dirname(__file__), "backtest_upper_hist_wait_grid_summary.csv")
    trades_path = os.path.join(os.path.dirname(__file__), "backtest_upper_hist_wait_grid_trades.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")

    print("\n[4] 상방 2개 합산")
    compare_rows = []
    for scenario in ["baseline"] + [f"wait_{d}d" for d in WAIT_DAYS_GRID]:
        g = trades_df[
            (trades_df["strategy"].isin(UPPER_STRATEGIES)) &
            (trades_df["scenario"] == scenario)
        ].copy()
        stats = mod.calc_stats(g.to_dict("records"))
        extra = calc_extra_stats(g)
        compare_rows.append({
            "scenario": scenario,
            "wait_days": -1 if scenario == "baseline" else int(scenario.split("_")[1][:-1]),
            **stats,
            **extra,
        })

    compare_df = pd.DataFrame(compare_rows).sort_values("wait_days")
    print(compare_df[[
        "scenario", "n", "win_rate", "avg_pnl", "median_pnl", "avg_hold",
        "macd_exit_cnt", "timeout_cnt", "timeout_below_target_cnt", "timeout_below_target_pct"
    ]].to_string(index=False))

    print(f"\n[5] 저장\n  -> {os.path.basename(summary_path)}\n  -> {os.path.basename(trades_path)}")


if __name__ == "__main__":
    main()
