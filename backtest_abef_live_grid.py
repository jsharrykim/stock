"""
현재 라이브 A/B/E/F 전략 파라미터 그리드 탐색
=========================================

원칙:
- updateInvestmentOpinion.gs의 현재 라이브 로직을 최대한 그대로 반영
- 출구는 라이브 기준으로 고정하고, 진입 숫자만 그리드 탐색
- 우선순위 제외도 현재 로직 기준으로 반영
"""

import os
import sys
import warnings
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

import backtest_combined as base

START = "2015-01-01"
END = "2026-04-15"

HALF_EXIT_DAYS = 60
MAX_HOLD_DAYS = 120

NASDAQ_DIST_UPPER = -3.0
NASDAQ_DIST_LOWER = -12.0
NASDAQ_DIST_RELEASE = -2.5
UPPER_EXIT_MAX_WAIT_DAYS = 5

LIVE_A = {"pctb_min": 80.0, "rsi_min": 70.0}
LIVE_B = {"vix_min": 25.0, "rsi_max": 40.0, "cci_min": -100.0, "lr_touch": 1.03}
LIVE_C = {"squeeze_ratio": 0.50, "bb_expand_ratio": 1.05, "vol_ratio": 1.50, "pctb_min": 55.0}
LIVE_D = {"adx_min": 20.0, "pctb_min": 30.0, "pctb_max": 75.0}
LIVE_E = {"squeeze_ratio": 0.50, "pctb_low_max": 50.0}
LIVE_F = {"pctb_low_max": 5.0}

# exit: 라이브 기준 고정
EXIT_A = {"target": 0.20, "stop": 0.30}
EXIT_B = {"target": 0.20, "stop": 0.30}
EXIT_E = {"target": 0.08, "stop": 0.30}
EXIT_F = {"target": 0.08, "stop": 0.30}

# grid
A_PCTB_GRID = [70.0, 75.0, 80.0, 85.0, 90.0]
A_RSI_GRID = [60.0, 65.0, 70.0, 75.0, 80.0]
A_NASDAQ_GRID = [-5.0, -4.0, -3.0, -2.0]

B_VIX_GRID = [20.0, 22.0, 25.0, 28.0, 30.0]
B_RSI_GRID = [35.0, 40.0, 45.0, 50.0]
B_CCI_GRID = [-50.0, -75.0, -100.0, -125.0, -150.0]
B_TOUCH_GRID = [1.00, 1.02, 1.03, 1.05]

E_SQUEEZE_GRID = [0.40, 0.45, 0.50, 0.55, 0.60]
E_PCTBLOW_GRID = [30.0, 35.0, 40.0, 45.0, 50.0, 55.0]

F_PCTBLOW_GRID = [3.0, 5.0, 7.0, 10.0, 12.0]


def compute_ixic_filter(ixic_dist: pd.Series, lower=NASDAQ_DIST_LOWER, upper=NASDAQ_DIST_UPPER, release=NASDAQ_DIST_RELEASE):
    latched = False
    active_list = []
    for dist in ixic_dist.ffill():
        in_death = dist > lower and dist < upper
        bottom = dist <= lower
        cleared = dist >= release
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
    print("[1/6] IXIC 다운로드")
    ixic_raw = yf.download("^IXIC", start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(ixic_raw.columns, pd.MultiIndex):
        ixic_raw.columns = ixic_raw.columns.droplevel(1)
    ixic = ixic_raw.copy()
    ixic.index = pd.to_datetime(ixic.index).tz_localize(None)
    ixic["ma200"] = ixic["Close"].rolling(200).mean()
    ixic["ixic_dist"] = (ixic["Close"] / ixic["ma200"] - 1) * 100
    return compute_ixic_filter(ixic["ixic_dist"])


def download_vix():
    print("[2/6] VIX 다운로드")
    vix_raw = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.droplevel(1)
    vix = vix_raw["Close"].copy()
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    return vix


def pack_dataframe(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["macd_prev2"] = df["macd_hist"].shift(2)
    return {
        "close": df["Close"].to_numpy(dtype=float),
        "low": df["Low"].to_numpy(dtype=float),
        "ma200": df["ma200"].to_numpy(dtype=float),
        "pctb_close": df["pctb_close"].to_numpy(dtype=float),
        "pctb_low": df["pctb_low"].to_numpy(dtype=float),
        "rsi": df["rsi"].to_numpy(dtype=float),
        "cci": df["cci"].to_numpy(dtype=float),
        "golden_cross": df["golden_cross"].fillna(False).to_numpy(dtype=bool),
        "macd_hist": df["macd_hist"].to_numpy(dtype=float),
        "macd_prev": df["macd_prev"].to_numpy(dtype=float),
        "macd_prev2": df["macd_prev2"].to_numpy(dtype=float),
        "plus_di": df["plus_di"].to_numpy(dtype=float),
        "minus_di": df["minus_di"].to_numpy(dtype=float),
        "adx": df["adx"].to_numpy(dtype=float),
        "adx_prev": df["adx_prev"].to_numpy(dtype=float),
        "adx_rising": df["adx_rising"].fillna(False).to_numpy(dtype=bool),
        "vol_ratio": df["vol_ratio"].to_numpy(dtype=float),
        "bb_width": df["bb_width"].to_numpy(dtype=float),
        "bb_width_prev": df["bb_width_prev"].to_numpy(dtype=float),
        "bb_width_avg": df["bb_width_avg"].to_numpy(dtype=float),
        "squeeze": df["squeeze"].fillna(False).to_numpy(dtype=bool),
        "prev_squeeze": df["prev_squeeze"].fillna(False).to_numpy(dtype=bool),
        "lr_trendline": df["lr_trendline"].to_numpy(dtype=float),
        "lr_slope": df["lr_slope"].to_numpy(dtype=float),
        "vix": df["vix"].to_numpy(dtype=float),
        "ixic_dist": df["ixic_dist"].to_numpy(dtype=float),
        "ixic_filter_active": df["ixic_filter_active"].fillna(False).to_numpy(dtype=bool),
    }


def download_data(vix_series: pd.Series, ixic_filter_df: pd.DataFrame):
    print(f"[3/6] 종목 다운로드 및 지표 계산 ({len(base.ALL_TICKERS)}개)")
    raw = yf.download(base.ALL_TICKERS, start=START, end=END, auto_adjust=True, progress=False, group_by="ticker")
    packed = {}
    for ticker in base.ALL_TICKERS:
        try:
            df = raw[ticker].copy() if len(base.ALL_TICKERS) > 1 else raw.copy()
            df.dropna(how="all", inplace=True)
            if len(df) < 250:
                continue
            df = base.calc_indicators(df)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df["vix"] = vix_series.reindex(df.index).ffill()
            df["ixic_dist"] = ixic_filter_df["ixic_dist"].reindex(df.index).ffill()
            df["ixic_filter_active"] = ixic_filter_df["ixic_filter_active"].reindex(df.index).ffill().fillna(False)
            packed[ticker] = pack_dataframe(df)
        except Exception as e:
            print(f"  [{ticker}] 오류: {e}")
    print(f"  -> {len(packed)}개 종목 준비 완료")
    return packed


def is_finite(x):
    return np.isfinite(x)


def strict_nasdaq_ok(ixic_dist, ixic_filter_active, upper=NASDAQ_DIST_UPPER):
    return (not bool(ixic_filter_active)) and is_finite(ixic_dist) and ixic_dist >= upper


def bottom_nasdaq_ok(ixic_filter_active):
    return not bool(ixic_filter_active)


def signal_a(data, i, params):
    return (
        is_finite(data["close"][i]) and is_finite(data["ma200"][i]) and data["close"][i] > data["ma200"][i]
        and bool(data["golden_cross"][i])
        and is_finite(data["pctb_close"][i]) and data["pctb_close"][i] > params["pctb_min"]
        and is_finite(data["rsi"][i]) and data["rsi"][i] > params["rsi_min"]
        and strict_nasdaq_ok(data["ixic_dist"][i], data["ixic_filter_active"][i], params["nasdaq_upper"])
    )


def signal_b(data, i, params):
    rsi_ok = is_finite(data["rsi"][i]) and data["rsi"][i] < params["rsi_max"]
    cci_ok = is_finite(data["cci"][i]) and data["cci"][i] < params["cci_min"]
    return (
        is_finite(data["close"][i]) and is_finite(data["ma200"][i]) and data["close"][i] < data["ma200"][i]
        and is_finite(data["vix"][i]) and data["vix"][i] >= params["vix_min"]
        and (rsi_ok or cci_ok)
        and is_finite(data["lr_slope"][i]) and data["lr_slope"][i] > 0
        and is_finite(data["lr_trendline"][i]) and data["lr_trendline"][i] > 0
        and is_finite(data["low"][i]) and data["low"][i] <= data["lr_trendline"][i] * params["lr_touch"]
    )


def signal_c_current(data, i):
    return (
        is_finite(data["close"][i]) and is_finite(data["ma200"][i]) and data["close"][i] > data["ma200"][i]
        and is_finite(data["bb_width_prev"][i]) and is_finite(data["bb_width_avg"][i]) and data["bb_width_avg"][i] > 0
        and data["bb_width_prev"][i] < data["bb_width_avg"][i] * LIVE_C["squeeze_ratio"]
        and is_finite(data["bb_width"][i]) and data["bb_width"][i] > data["bb_width_prev"][i] * LIVE_C["bb_expand_ratio"]
        and is_finite(data["vol_ratio"][i]) and data["vol_ratio"][i] >= LIVE_C["vol_ratio"]
        and is_finite(data["pctb_close"][i]) and data["pctb_close"][i] > LIVE_C["pctb_min"]
        and is_finite(data["macd_hist"][i]) and data["macd_hist"][i] > 0
        and strict_nasdaq_ok(data["ixic_dist"][i], data["ixic_filter_active"][i], NASDAQ_DIST_UPPER)
    )


def signal_d_current(data, i):
    return (
        is_finite(data["close"][i]) and is_finite(data["ma200"][i]) and data["close"][i] > data["ma200"][i]
        and is_finite(data["plus_di"][i]) and is_finite(data["minus_di"][i]) and data["plus_di"][i] > data["minus_di"][i]
        and is_finite(data["adx"][i]) and data["adx"][i] > LIVE_D["adx_min"]
        and bool(data["adx_rising"][i])
        and is_finite(data["macd_hist"][i]) and data["macd_hist"][i] > 0
        and is_finite(data["pctb_close"][i]) and LIVE_D["pctb_min"] <= data["pctb_close"][i] <= LIVE_D["pctb_max"]
        and strict_nasdaq_ok(data["ixic_dist"][i], data["ixic_filter_active"][i], NASDAQ_DIST_UPPER)
    )


def signal_e(data, i, params):
    return (
        is_finite(data["close"][i]) and is_finite(data["ma200"][i]) and data["close"][i] > data["ma200"][i]
        and is_finite(data["bb_width"][i]) and is_finite(data["bb_width_avg"][i]) and data["bb_width_avg"][i] > 0
        and (data["bb_width"][i] / data["bb_width_avg"][i]) < params["squeeze_ratio"]
        and is_finite(data["pctb_low"][i]) and data["pctb_low"][i] <= params["pctb_low_max"]
        and bottom_nasdaq_ok(data["ixic_filter_active"][i])
    )


def signal_f(data, i, params):
    return (
        is_finite(data["close"][i]) and is_finite(data["ma200"][i]) and data["close"][i] > data["ma200"][i]
        and is_finite(data["pctb_low"][i]) and data["pctb_low"][i] <= params["pctb_low_max"]
        and bottom_nasdaq_ok(data["ixic_filter_active"][i])
    )


def check_exit_simple(close, entry_price, hold_days, target, stop):
    pnl = (close - entry_price) / entry_price
    if pnl >= target:
        return "목표달성", pnl
    if pnl <= -stop:
        return "손절", pnl
    if hold_days >= HALF_EXIT_DAYS and pnl > 0:
        return "60일수익", pnl
    if hold_days >= MAX_HOLD_DAYS:
        return "기간만료", pnl
    return None, pnl


def check_exit_ef(data, i, entry_price, hold_days, target, stop, first_target_idx):
    close = data["close"][i]
    pnl = (close - entry_price) / entry_price
    updated = first_target_idx

    if pnl >= target and updated is None:
        updated = i

    if updated is not None:
        hist = data["macd_hist"][i]
        prev = data["macd_prev"][i]
        prev2 = data["macd_prev2"][i]
        hist_turn = (
            is_finite(hist) and is_finite(prev) and is_finite(prev2)
            and (hist - prev) < (prev - prev2)
        )
        if pnl >= target and hist_turn:
            return "목표+MACD둔화", pnl, updated
        if i - updated >= UPPER_EXIT_MAX_WAIT_DAYS:
            return "목표후대기만료", pnl, updated

    if pnl <= -stop:
        return "손절", pnl, updated
    if hold_days >= HALF_EXIT_DAYS and pnl > 0:
        return "60일수익", pnl, updated
    if hold_days >= MAX_HOLD_DAYS:
        return "기간만료", pnl, updated
    return None, pnl, updated


def summarize(trades, meta):
    if not trades:
        return {
            **meta,
            "trades": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "median_pnl": 0.0,
            "avg_hold": 0.0,
            "pf": np.nan,
            "stop_rate": 0.0,
            "ev": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }
    df = pd.DataFrame(trades)
    wins = df[df["pnl_pct"] > 0]
    loss = df[df["pnl_pct"] <= 0]
    wr = len(wins) / len(df)
    avg_win = wins["pnl_pct"].mean() if len(wins) else 0.0
    avg_loss = loss["pnl_pct"].mean() if len(loss) else 0.0
    gross_win = wins["pnl_pct"].sum() if len(wins) else 0.0
    gross_loss = abs(loss["pnl_pct"].sum()) if len(loss) else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else np.nan
    by_exit = df["exit_reason"].value_counts(normalize=True) * 100
    return {
        **meta,
        "trades": int(len(df)),
        "win_rate": round(wr * 100, 2),
        "avg_pnl": round(df["pnl_pct"].mean(), 3),
        "median_pnl": round(df["pnl_pct"].median(), 3),
        "avg_hold": round(df["hold_days"].mean(), 2),
        "pf": round(float(pf), 3) if np.isfinite(pf) else np.nan,
        "stop_rate": round(by_exit.get("손절", 0.0), 2),
        "ev": round(wr * avg_win + (1 - wr) * avg_loss, 3),
        "avg_win": round(avg_win, 3),
        "avg_loss": round(avg_loss, 3),
    }


def simulate_group(data_map, group, params):
    trades = []
    for ticker, data in data_map.items():
        n = len(data["close"])
        in_pos = False
        entry_price = 0.0
        entry_idx = -1
        first_target_idx = None
        for i in range(n):
            close = data["close"][i]
            if not is_finite(close):
                continue

            if in_pos:
                hold_days = i - entry_idx
                if group in ("A", "B"):
                    reason, pnl = check_exit_simple(
                        close, entry_price, hold_days,
                        EXIT_A["target"] if group == "A" else EXIT_B["target"],
                        EXIT_A["stop"] if group == "A" else EXIT_B["stop"],
                    )
                else:
                    reason, pnl, first_target_idx = check_exit_ef(
                        data, i, entry_price, hold_days,
                        EXIT_E["target"] if group == "E" else EXIT_F["target"],
                        EXIT_E["stop"] if group == "E" else EXIT_F["stop"],
                        first_target_idx,
                    )
                if reason:
                    trades.append({"ticker": ticker, "pnl_pct": round(pnl * 100, 3), "hold_days": hold_days, "exit_reason": reason})
                    in_pos = False
                    first_target_idx = None

            if not in_pos:
                a_cur = signal_a(data, i, {"pctb_min": LIVE_A["pctb_min"], "rsi_min": LIVE_A["rsi_min"], "nasdaq_upper": NASDAQ_DIST_UPPER})
                b_cur = signal_b(data, i, LIVE_B)
                c_cur = signal_c_current(data, i)
                d_cur = signal_d_current(data, i)
                e_cur = signal_e(data, i, LIVE_E)

                trigger = False
                if group == "A":
                    trigger = signal_a(data, i, params)
                elif group == "B":
                    trigger = (not a_cur) and signal_b(data, i, params)
                elif group == "E":
                    trigger = (not a_cur) and (not b_cur) and (not c_cur) and (not d_cur) and signal_e(data, i, params)
                elif group == "F":
                    trigger = (not a_cur) and (not b_cur) and (not c_cur) and (not d_cur) and (not e_cur) and signal_f(data, i, params)

                if trigger:
                    in_pos = True
                    entry_price = close
                    entry_idx = i
                    first_target_idx = None

        if in_pos:
            pnl = (data["close"][-1] - entry_price) / entry_price
            trades.append({"ticker": ticker, "pnl_pct": round(pnl * 100, 3), "hold_days": n - 1 - entry_idx, "exit_reason": "미청산"})
    return trades


def run_grid(data_map, group, combos, current_params):
    rows = []
    rows.append(summarize(simulate_group(data_map, group, current_params), {"kind": "current", **current_params}))
    print(f"[{group}] grid start ({len(combos)} combos)")
    for idx, params in enumerate(combos, start=1):
        rows.append(summarize(simulate_group(data_map, group, params), {"kind": "grid", **params}))
        if idx % 100 == 0:
            print(f"  {group} progress: {idx}/{len(combos)}")
    return pd.DataFrame(rows).sort_values(["ev", "pf", "win_rate", "trades"], ascending=[False, False, False, False]).reset_index(drop=True)


def main():
    ixic_filter_df = download_ixic_filter()
    vix_series = download_vix()
    data_map = download_data(vix_series, ixic_filter_df)

    a_combos = [{"pctb_min": p, "rsi_min": r, "nasdaq_upper": n} for p, r, n in product(A_PCTB_GRID, A_RSI_GRID, A_NASDAQ_GRID)]
    b_combos = [{"vix_min": v, "rsi_max": r, "cci_min": c, "lr_touch": t} for v, r, c, t in product(B_VIX_GRID, B_RSI_GRID, B_CCI_GRID, B_TOUCH_GRID)]
    e_combos = [{"squeeze_ratio": s, "pctb_low_max": p} for s, p in product(E_SQUEEZE_GRID, E_PCTBLOW_GRID)]
    f_combos = [{"pctb_low_max": p} for p in F_PCTBLOW_GRID]

    print("[4/6] A 실행")
    a_df = run_grid(data_map, "A", a_combos, {"pctb_min": LIVE_A["pctb_min"], "rsi_min": LIVE_A["rsi_min"], "nasdaq_upper": NASDAQ_DIST_UPPER})
    print("[5/6] B 실행")
    b_df = run_grid(data_map, "B", b_combos, LIVE_B)
    print("[6/6] E/F 실행")
    e_df = run_grid(data_map, "E", e_combos, LIVE_E)
    f_df = run_grid(data_map, "F", f_combos, LIVE_F)

    base_dir = os.path.dirname(__file__)
    a_df.to_csv(os.path.join(base_dir, "backtest_abef_live_grid_a.csv"), index=False, encoding="utf-8-sig")
    b_df.to_csv(os.path.join(base_dir, "backtest_abef_live_grid_b.csv"), index=False, encoding="utf-8-sig")
    e_df.to_csv(os.path.join(base_dir, "backtest_abef_live_grid_e.csv"), index=False, encoding="utf-8-sig")
    f_df.to_csv(os.path.join(base_dir, "backtest_abef_live_grid_f.csv"), index=False, encoding="utf-8-sig")

    def brief(name, df):
        cur = df[df["kind"] == "current"].iloc[0]
        print(f"\n[{name}] current ev={cur['ev']} trades={int(cur['trades'])} win={cur['win_rate']}%")
        print(df.head(8).to_csv(index=False))

    brief("A", a_df)
    brief("B", b_df)
    brief("E", e_df)
    brief("F", f_df)


if __name__ == "__main__":
    main()
