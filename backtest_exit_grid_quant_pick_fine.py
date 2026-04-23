"""
퀀트 픽 진입 기준 청산 정밀(1% 단위) 그리드 백테스트
=================================================

- 진입 기준은 `backtest_exit_grid_quant_pick.py`의 quant pick 고정
- 청산은 1% 단위 그리드 탐색
- 그룹별 / 통합 포트폴리오 모두 avg_hold 포함 저장
"""

import os
import pandas as pd

import backtest_exit_grid_quant_pick as mod

TARGET_GRID = [x / 100 for x in range(8, 26)]   # 8% ~ 25%
STOP_GRID = [x / 100 for x in range(15, 36)]    # 15% ~ 35%


def main():
    print("=" * 100)
    print("퀀트 픽 진입 기준 청산 정밀(1% 단위) 그리드 백테스트")
    print("=" * 100)

    vix = mod.base.download_vix()
    ixic_filter_df = mod.download_ixic_filter()
    data = mod.download_data(vix, ixic_filter_df)
    mod.build_prev2(data)
    prepared = mod.prepare_data(data)

    print("\n[3] 그룹별 정밀 그리드 실행")
    group_rows = []
    for group in mod.GROUPS:
        print(f"  - {group}그룹")
        for target in TARGET_GRID:
            for stop in STOP_GRID:
                trades = mod.run_group_backtest(prepared, group, target, stop)
                stats = mod.calc_stats(trades)
                group_rows.append(
                    {
                        "group": group,
                        "target_pct": round(target * 100, 1),
                        "stop_pct": round(stop * 100, 1),
                        **stats,
                    }
                )
    group_df = pd.DataFrame(group_rows).sort_values(
        ["group", "avg_pnl", "pf", "win_rate", "n"],
        ascending=[True, False, False, False, False],
    )

    print("\n[4] 통합 포트폴리오 정밀 그리드 실행")
    portfolio_rows = []
    current_trades = mod.run_portfolio_backtest(prepared, mod.CURRENT_CONFIG, "current_quant_pick")
    portfolio_rows.append({"scenario": "current_quant_pick", **mod.calc_stats(current_trades)})

    for target in TARGET_GRID:
        for stop in STOP_GRID:
            cfg = {g: {"target": target, "stop": stop} for g in mod.GROUPS}
            trades = mod.run_portfolio_backtest(
                prepared,
                cfg,
                f"common_t{int(target * 100)}_s{int(stop * 100)}",
            )
            portfolio_rows.append(
                {
                    "scenario": f"common_t{int(target * 100)}_s{int(stop * 100)}",
                    "target_pct": round(target * 100, 1),
                    "stop_pct": round(stop * 100, 1),
                    **mod.calc_stats(trades),
                }
            )
    portfolio_df = pd.DataFrame(portfolio_rows).sort_values(
        ["avg_pnl", "pf", "win_rate", "n"],
        ascending=[False, False, False, False],
    )

    base_dir = os.path.dirname(__file__)
    group_path = os.path.join(base_dir, "backtest_quant_pick_exit_group_grid_fine.csv")
    portfolio_path = os.path.join(base_dir, "backtest_quant_pick_exit_portfolio_grid_fine.csv")
    group_df.to_csv(group_path, index=False, encoding="utf-8-sig")
    portfolio_df.to_csv(portfolio_path, index=False, encoding="utf-8-sig")

    print("\n[5] 그룹별 현재 조합 vs 최고 조합")
    current_by_group = {
        "A": (20.0, 30.0),
        "B": (20.0, 30.0),
        "C": (18.0, 30.0),
        "D": (18.0, 30.0),
        "E": (8.0, 30.0),
        "F": (8.0, 30.0),
    }
    for group in mod.GROUPS:
        gdf = group_df[group_df["group"] == group]
        best = gdf.iloc[0]
        cur_t, cur_s = current_by_group[group]
        cur = gdf[(gdf["target_pct"] == cur_t) & (gdf["stop_pct"] == cur_s)]
        cur = cur.iloc[0] if not cur.empty else None
        print(
            f"  [{group}] 현재 {cur_t:.0f}/{cur_s:.0f}"
            f" -> 거래 {cur['n'] if cur is not None else 0} | 승률 {cur['win_rate'] if cur is not None else 0:.1f}%"
            f" | 평균 {cur['avg_pnl'] if cur is not None else 0:+.2f}% | 보유 {cur['avg_hold'] if cur is not None else 0:.1f}일"
            f" || 최고 {best['target_pct']:.0f}/{best['stop_pct']:.0f}"
            f" -> 거래 {int(best['n'])} | 승률 {best['win_rate']:.1f}% | 평균 {best['avg_pnl']:+.2f}%"
            f" | 보유 {best['avg_hold']:.1f}일 | PF {best['pf']:.2f}"
        )

    print("\n[6] 통합 포트폴리오 상위 15")
    print(
        portfolio_df.head(15)[
            ["scenario", "target_pct", "stop_pct", "n", "win_rate", "avg_pnl", "median_pnl", "avg_hold", "pf", "stop_rate"]
        ].to_string(index=False)
    )
    print(f"\n저장 완료:\n- {group_path}\n- {portfolio_path}")


if __name__ == "__main__":
    main()
