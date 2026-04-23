"""
퀀트 픽 진입 조합 기준 청산 그리드 백테스트
=======================================

진입 조합(quant pick):
- A: 유지
- B: 강화
- C: 개선
- D: 개선
- E: 유지
- F: 개선

청산:
- 그룹별 target/stop 그리드
- 통합 포트폴리오 공통 target/stop 그리드
- E/F는 라이브와 동일하게 목표 도달 후 MACD 둔화 또는 대기 만료 출구 유지
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

import backtest_combined as base

START = "2015-01-01"
END = "2026-04-15"

TARGET_GRID = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
STOP_GRID = [0.15, 0.20, 0.25, 0.30]

HALF_EXIT_DAYS = 60
MAX_HOLD_DAYS = 120
UPPER_EXIT_MAX_WAIT_DAYS = 5

NASDAQ_DIST_UPPER = -3.0
NASDAQ_DIST_LOWER = -12.0
NASDAQ_DIST_RELEASE = -2.5

# quant pick entries
A_PCTB_MIN = 80.0
A_RSI_MIN = 70.0

B_VIX_MIN = 30.0
B_RSI_MAX = 35.0
B_CCI_MIN = -150.0
B_LR_TOUCH = 1.05

C_SQUEEZE_RATIO = 0.45
C_BB_EXPAND_RATIO = 1.00
C_VOL_RATIO = 1.50
C_PCTB_MIN = 55.0

D_ADX_MIN = 30.0
D_PCTB_MIN = 30.0
D_PCTB_MAX = 80.0

E_SQUEEZE_RATIO = 0.50
E_PCTB_LOW_MAX = 50.0

F_PCTB_LOW_MAX = 3.0

GROUPS = ["A", "B", "C", "D", "E", "F"]
PRIORITY = ["A", "B", "C", "D", "E", "F"]

CURRENT_CONFIG = {
    "A": {"target": 0.20, "stop": 0.30},
    "B": {"target": 0.20, "stop": 0.30},
    "C": {"target": 0.18, "stop": 0.30},
    "D": {"target": 0.18, "stop": 0.30},
    "E": {"target": 0.08, "stop": 0.30},
    "F": {"target": 0.08, "stop": 0.30},
}


def compute_ixic_filter(ixic_dist: pd.Series) -> pd.DataFrame:
    latched = False
    active_list = []
    for dist in ixic_dist.ffill():
        in_death = dist > NASDAQ_DIST_LOWER and dist < NASDAQ_DIST_UPPER
        bottom = dist <= NASDAQ_DIST_LOWER
        cleared = dist >= NASDAQ_DIST_RELEASE
        if bottom:
            latched = False
            active = False
        elif in_death:
            latched = True
            active = True
        elif cleared:
            latched = False
            active = False
        else:
            active = latched
        active_list.append(active)
    return pd.DataFrame(
        {"ixic_dist": ixic_dist, "ixic_filter_active": pd.Series(active_list, index=ixic_dist.index)}
    )


def download_ixic_filter():
    print("[1] IXIC 다운로드")
    ixic_raw = yf.download("^IXIC", start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(ixic_raw.columns, pd.MultiIndex):
        ixic_raw.columns = ixic_raw.columns.droplevel(1)
    ixic = ixic_raw.copy()
    ixic["ma200"] = ixic["Close"].rolling(200).mean()
    ixic["ixic_dist"] = (ixic["Close"] / ixic["ma200"] - 1) * 100
    return compute_ixic_filter(ixic["ixic_dist"])


def download_data(vix_series: pd.Series, ixic_filter_df: pd.DataFrame):
    print(f"[2] 종목 다운로드 및 지표 계산 ({len(base.ALL_TICKERS)}개)")
    raw = yf.download(base.ALL_TICKERS, start=START, end=END, auto_adjust=True, progress=False, group_by="ticker")
    result = {}
    for t in base.ALL_TICKERS:
        try:
            df = raw[t].copy() if len(base.ALL_TICKERS) > 1 else raw.copy()
            df.dropna(how="all", inplace=True)
            if len(df) < 250:
                continue
            df = base.calc_indicators(df)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df["vix"] = vix_series.reindex(df.index).ffill()
            df["ixic_dist"] = ixic_filter_df["ixic_dist"].reindex(df.index).ffill()
            df["ixic_filter_active"] = ixic_filter_df["ixic_filter_active"].reindex(df.index).ffill().fillna(False)
            result[t] = df
        except Exception as e:
            print(f"  [{t}] 오류: {e}")
    print(f"  -> {len(result)}개 종목 준비 완료")
    return result


def row_to_signal(row) -> dict:
    fv = lambda k, d=np.nan: float(row.get(k, d)) if pd.notna(row.get(k, d)) else d
    return {
        "close": fv("Close", 0),
        "low": fv("Low", 0),
        "ma200": fv("ma200", np.nan),
        "above200": fv("Close", 0) > fv("ma200", np.inf),
        "pctb_close": fv("pctb_close", np.nan),
        "pctb_low": fv("pctb_low", np.nan),
        "rsi": fv("rsi", np.nan),
        "cci": fv("cci", np.nan),
        "macd_hist": fv("macd_hist", np.nan),
        "macd_prev": fv("macd_prev", np.nan),
        "golden_cross": bool(row.get("golden_cross", False)),
        "plus_di": fv("plus_di", np.nan),
        "minus_di": fv("minus_di", np.nan),
        "adx": fv("adx", np.nan),
        "adx_prev": fv("adx_prev", np.nan),
        "adx_rising": bool(row.get("adx_rising", False)),
        "vol_ratio": fv("vol_ratio", np.nan),
        "bb_width": fv("bb_width", np.nan),
        "bb_width_prev": fv("bb_width_prev", np.nan),
        "bb_width_avg": fv("bb_width_avg", np.nan),
        "squeeze": bool(row.get("squeeze", False)),
        "prev_squeeze": bool(row.get("prev_squeeze", False)),
        "lr_trendline": fv("lr_trendline", np.nan),
        "lr_slope": fv("lr_slope", np.nan),
        "vix": fv("vix", np.nan),
        "ixic_dist": fv("ixic_dist", np.nan),
        "ixic_filter_active": bool(row.get("ixic_filter_active", False)),
    }


def signal_by_group(s: dict) -> dict:
    nasdaq_strict = (not s["ixic_filter_active"]) and s["ixic_dist"] >= NASDAQ_DIST_UPPER
    nasdaq_bottom = not s["ixic_filter_active"]

    a = (
        s["above200"]
        and s["golden_cross"]
        and s["pctb_close"] > A_PCTB_MIN
        and s["rsi"] > A_RSI_MIN
        and nasdaq_strict
    )

    b = (
        (not s["above200"])
        and s["vix"] >= B_VIX_MIN
        and ((s["rsi"] < B_RSI_MAX) or (s["cci"] < B_CCI_MIN))
        and s["lr_slope"] > 0
        and s["lr_trendline"] > 0
        and s["low"] <= s["lr_trendline"] * B_LR_TOUCH
    )

    c = (
        (not a) and (not b)
        and s["above200"]
        and s["bb_width_prev"] < s["bb_width_avg"] * C_SQUEEZE_RATIO
        and s["bb_width"] > s["bb_width_prev"] * C_BB_EXPAND_RATIO
        and s["vol_ratio"] >= C_VOL_RATIO
        and s["pctb_close"] > C_PCTB_MIN
        and s["macd_hist"] > 0
        and nasdaq_strict
    )

    d = (
        (not a) and (not b) and (not c)
        and s["above200"]
        and s["plus_di"] > s["minus_di"]
        and s["adx"] > D_ADX_MIN
        and s["adx_rising"]
        and s["macd_hist"] > 0
        and D_PCTB_MIN <= s["pctb_close"] <= D_PCTB_MAX
        and nasdaq_strict
    )

    e = (
        (not a) and (not b) and (not c) and (not d)
        and s["above200"]
        and s["bb_width"] < s["bb_width_avg"] * E_SQUEEZE_RATIO
        and s["pctb_low"] <= E_PCTB_LOW_MAX
        and nasdaq_bottom
    )

    f = (
        (not a) and (not b) and (not c) and (not d) and (not e)
        and s["above200"]
        and s["pctb_low"] <= F_PCTB_LOW_MAX
        and nasdaq_bottom
    )
    return {"A": a, "B": b, "C": c, "D": d, "E": e, "F": f}


def check_exit_values(group: str, close: float, macd_hist: float, macd_prev: float, macd_prev2: float,
                      entry_price: float, hold_days: int, target: float, stop: float,
                      first_target_idx: int | None, current_idx: int):
    pnl = (close - entry_price) / entry_price
    reason = None
    updated_first_target = first_target_idx

    if pnl <= -stop:
        return "손절", pnl, updated_first_target
    if hold_days >= HALF_EXIT_DAYS and pnl > 0:
        return "60일수익", pnl, updated_first_target
    if hold_days >= MAX_HOLD_DAYS:
        return "기간만료", pnl, updated_first_target

    if group in ("E", "F"):
        if updated_first_target is None and pnl >= target:
            updated_first_target = current_idx

        if updated_first_target is not None:
            hist_turn = False
            if pd.notna(macd_hist) and pd.notna(macd_prev) and pd.notna(macd_prev2):
                hist_turn = (macd_hist - macd_prev) < (macd_prev - macd_prev2)
            wait_days = current_idx - updated_first_target
            if pnl >= target and hist_turn:
                reason = "목표+MACD둔화"
            elif wait_days >= UPPER_EXIT_MAX_WAIT_DAYS:
                reason = "목표후대기만료"
        return reason, pnl, updated_first_target

    if pnl >= target:
        return "목표달성", pnl, updated_first_target
    return None, pnl, updated_first_target


def build_prev2(data: dict):
    for df in data.values():
        df["macd_prev2"] = df["macd_hist"].shift(2)


def prepare_data(data: dict) -> dict:
    prepared = {}
    required = ["ma200", "pctb_close", "pctb_low", "rsi", "cci", "macd_hist", "vix", "ixic_dist", "macd_prev2"]
    for ticker, df in data.items():
        dfc = df.dropna(subset=required).copy()
        if len(dfc) < 250:
            continue
        signal_rows = []
        for row in dfc.to_dict("records"):
            signal_rows.append(signal_by_group(row_to_signal(row)))
        sig_df = pd.DataFrame(signal_rows, index=dfc.index)
        prepared[ticker] = {
            "dates": list(dfc.index),
            "close": dfc["Close"].to_numpy(dtype=float),
            "macd_hist": dfc["macd_hist"].to_numpy(dtype=float),
            "macd_prev": dfc["macd_prev"].to_numpy(dtype=float),
            "macd_prev2": dfc["macd_prev2"].to_numpy(dtype=float),
            "signals": {g: sig_df[g].to_numpy(dtype=bool) for g in GROUPS},
        }
    return prepared


def calc_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "win_rate": 0.0, "avg_pnl": 0.0, "median_pnl": 0.0, "avg_hold": 0.0, "pf": 0.0, "stop_rate": 0.0}
    df = pd.DataFrame(trades)
    wins = df[df["pnl_pct"] > 0]["pnl_pct"]
    losses = df[df["pnl_pct"] <= 0]["pnl_pct"]
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    stop_rate = (df["exit_reason"] == "손절").mean() * 100
    return {
        "n": int(len(df)),
        "win_rate": round((df["pnl_pct"] > 0).mean() * 100, 2),
        "avg_pnl": round(df["pnl_pct"].mean(), 3),
        "median_pnl": round(df["pnl_pct"].median(), 3),
        "avg_hold": round(df["hold_days"].mean(), 2),
        "pf": round(gross_win / gross_loss, 3) if gross_loss > 0 else 999.0,
        "stop_rate": round(stop_rate, 2),
    }


def run_group_backtest(prepared: dict, group: str, target: float, stop: float) -> list[dict]:
    trades = []
    for ticker, bundle in prepared.items():
        dates = bundle["dates"]
        closes = bundle["close"]
        macd_hist = bundle["macd_hist"]
        macd_prev = bundle["macd_prev"]
        macd_prev2 = bundle["macd_prev2"]
        entries = bundle["signals"][group]
        in_pos = False
        entry_price = 0.0
        entry_date = None
        entry_idx = 0
        first_target_idx = None
        for i, dt in enumerate(dates):
            if in_pos:
                reason, pnl, first_target_idx = check_exit_values(
                    group, closes[i], macd_hist[i], macd_prev[i], macd_prev2[i],
                    entry_price, i - entry_idx, target, stop, first_target_idx, i
                )
                if reason:
                    trades.append({
                        "group": group,
                        "ticker": ticker,
                        "entry_date": entry_date,
                        "exit_date": dt,
                        "pnl_pct": round(pnl * 100, 2),
                        "hold_days": i - entry_idx,
                        "exit_reason": reason,
                        "target": round(target * 100, 1),
                        "stop": round(stop * 100, 1),
                    })
                    in_pos = False
                    first_target_idx = None
            if (not in_pos) and entries[i]:
                in_pos = True
                entry_price = float(closes[i])
                entry_date = dt
                entry_idx = i
                first_target_idx = None
    return trades


def run_portfolio_backtest(prepared: dict, config: dict, scenario_name: str) -> list[dict]:
    trades = []
    for ticker, bundle in prepared.items():
        dates = bundle["dates"]
        closes = bundle["close"]
        macd_hist = bundle["macd_hist"]
        macd_prev = bundle["macd_prev"]
        macd_prev2 = bundle["macd_prev2"]
        signals = bundle["signals"]
        in_pos = False
        entry_price = 0.0
        entry_date = None
        entry_idx = 0
        entry_group = None
        first_target_idx = None
        for i, dt in enumerate(dates):
            if in_pos:
                ep = config[entry_group]
                reason, pnl, first_target_idx = check_exit_values(
                    entry_group, closes[i], macd_hist[i], macd_prev[i], macd_prev2[i],
                    entry_price, i - entry_idx, ep["target"], ep["stop"], first_target_idx, i
                )
                if reason:
                    trades.append({
                        "scenario": scenario_name,
                        "ticker": ticker,
                        "group": entry_group,
                        "entry_date": entry_date,
                        "exit_date": dt,
                        "pnl_pct": round(pnl * 100, 2),
                        "hold_days": i - entry_idx,
                        "exit_reason": reason,
                    })
                    in_pos = False
                    entry_group = None
                    first_target_idx = None

            if not in_pos:
                chosen = next((g for g in PRIORITY if signals[g][i]), None)
                if chosen:
                    in_pos = True
                    entry_group = chosen
                    entry_price = float(closes[i])
                    entry_date = dt
                    entry_idx = i
                    first_target_idx = None
    return trades


def main():
    print("=" * 90)
    print("퀀트 픽 진입 기준 익절/손절 그리드 백테스트")
    print("=" * 90)

    vix = base.download_vix()
    ixic_filter_df = download_ixic_filter()
    data = download_data(vix, ixic_filter_df)
    build_prev2(data)
    prepared = prepare_data(data)

    group_rows = []
    print("\n[3] 그룹별 그리드 실행")
    for group in GROUPS:
        print(f"  - {group}그룹")
        for target in TARGET_GRID:
            for stop in STOP_GRID:
                trades = run_group_backtest(prepared, group, target, stop)
                stats = calc_stats(trades)
                group_rows.append({
                    "group": group,
                    "target_pct": round(target * 100, 1),
                    "stop_pct": round(stop * 100, 1),
                    **stats,
                })
    group_df = pd.DataFrame(group_rows).sort_values(["group", "avg_pnl", "pf"], ascending=[True, False, False])

    print("\n[4] 통합 포트폴리오 비교")
    portfolio_rows = []
    current_trades = run_portfolio_backtest(prepared, CURRENT_CONFIG, "current_quant_pick")
    portfolio_rows.append({"scenario": "current_quant_pick", **calc_stats(current_trades)})

    for target in TARGET_GRID:
        for stop in STOP_GRID:
            cfg = {g: {"target": target, "stop": stop} for g in GROUPS}
            trades = run_portfolio_backtest(prepared, cfg, f"common_t{int(target*100)}_s{int(stop*100)}")
            portfolio_rows.append({
                "scenario": f"common_t{int(target*100)}_s{int(stop*100)}",
                "target_pct": round(target * 100, 1),
                "stop_pct": round(stop * 100, 1),
                **calc_stats(trades),
            })
    portfolio_df = pd.DataFrame(portfolio_rows).sort_values(["avg_pnl", "pf"], ascending=[False, False])

    base_dir = os.path.dirname(__file__)
    group_path = os.path.join(base_dir, "backtest_quant_pick_exit_group_grid.csv")
    portfolio_path = os.path.join(base_dir, "backtest_quant_pick_exit_portfolio_grid.csv")
    group_df.to_csv(group_path, index=False, encoding="utf-8-sig")
    portfolio_df.to_csv(portfolio_path, index=False, encoding="utf-8-sig")

    print("\n[5] 그룹별 현재 조합 vs 최고 조합")
    current_by_group = {"A": (20.0, 30.0), "B": (20.0, 30.0), "C": (18.0, 30.0), "D": (18.0, 30.0), "E": (8.0, 30.0), "F": (8.0, 30.0)}
    for group in GROUPS:
        gdf = group_df[group_df["group"] == group]
        best = gdf.iloc[0]
        cur_t, cur_s = current_by_group[group]
        cur = gdf[(gdf["target_pct"] == cur_t) & (gdf["stop_pct"] == cur_s)]
        cur = cur.iloc[0] if not cur.empty else None
        print(
            f"  [{group}] 현재 {cur_t:.0f}/{cur_s:.0f}"
            f" -> 거래 {cur['n'] if cur is not None else 0} | 승률 {cur['win_rate'] if cur is not None else 0:.1f}% | 평균 {cur['avg_pnl'] if cur is not None else 0:+.2f}%"
            f" || 최고 {best['target_pct']:.0f}/{best['stop_pct']:.0f}"
            f" -> 거래 {int(best['n'])} | 승률 {best['win_rate']:.1f}% | 평균 {best['avg_pnl']:+.2f}% | PF {best['pf']:.2f}"
        )

    print("\n[6] 통합 포트폴리오 상위 10")
    print(portfolio_df.head(10)[["scenario", "target_pct", "stop_pct", "n", "win_rate", "avg_pnl", "median_pnl", "pf", "stop_rate"]].to_string(index=False))
    print(f"\n저장 완료:\n- {group_path}\n- {portfolio_path}")


if __name__ == "__main__":
    main()
