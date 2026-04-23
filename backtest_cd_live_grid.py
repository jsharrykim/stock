"""
현재 라이브 C/D 전략 파라미터 그리드 탐색
=====================================

목적:
1) `updateInvestmentOpinion.gs`의 현재 C/D 진입 로직을 최대한 그대로 반영한다.
2) 기존 결과는 참고만 하고, 실제 라이브 기준의 숫자 축을 넓게 그리드 탐색한다.
3) 출구는 라이브 기준(C/D 공통: +18% / -30% / 60일수익 / 120일만료)으로 고정해
   "진입 숫자"의 품질만 비교한다.

주의:
- C는 라이브 우선순위상 A를 먼저 제외한다.
- D는 라이브 우선순위상 A와 "현재 C 기준"을 먼저 제외한다.
- 나스닥 하락장 필터(A/C/D strict)는 반영한다.
- 이벤트 데이 필터는 반영하지 않는다.
- MACD > 0 자체는 숫자 축이지만 종목 가격 스케일 영향을 크게 받아 이번 그리드에선 고정한다.
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

TARGET = 0.18
STOP = 0.30
HALF_EXIT_DAYS = 60
MAX_HOLD_DAYS = 120

NASDAQ_DIST_UPPER = -3.0
NASDAQ_DIST_LOWER = -12.0
NASDAQ_DIST_RELEASE = -2.5

# 현재 라이브 기준
LIVE_A = {
    "pctb_min": 80.0,
    "rsi_min": 70.0,
}
LIVE_C = {
    "squeeze_ratio": 0.50,
    "bb_expand_ratio": 1.05,
    "vol_ratio": 1.50,
    "pctb_min": 55.0,
}
LIVE_D = {
    "adx_min": 20.0,
    "pctb_min": 30.0,
    "pctb_max": 75.0,
    "adx_delta_min": 0.0,
}

# 그리드 범위
C_SQUEEZE_GRID = [0.40, 0.45, 0.50, 0.55, 0.60]
C_EXPAND_GRID = [1.00, 1.05, 1.10, 1.15]
C_VOL_GRID = [1.00, 1.25, 1.50, 1.75, 2.00]
C_PCTB_GRID = [45.0, 50.0, 55.0, 60.0, 65.0]

D_ADX_GRID = [15.0, 20.0, 25.0, 30.0, 35.0]
D_PCTB_MIN_GRID = [20.0, 25.0, 30.0, 35.0, 40.0]
D_PCTB_MAX_GRID = [65.0, 70.0, 75.0, 80.0, 85.0]
D_ADX_DELTA_GRID = [0.0, 0.5, 1.0]


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


def download_ixic_filter() -> pd.DataFrame:
    print("[1/4] IXIC 다운로드")
    ixic_raw = yf.download("^IXIC", start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(ixic_raw.columns, pd.MultiIndex):
        ixic_raw.columns = ixic_raw.columns.droplevel(1)
    ixic = ixic_raw.copy()
    ixic.index = pd.to_datetime(ixic.index).tz_localize(None)
    ixic["ma200"] = ixic["Close"].rolling(200).mean()
    ixic["ixic_dist"] = (ixic["Close"] / ixic["ma200"] - 1) * 100
    return compute_ixic_filter(ixic["ixic_dist"])


def pack_dataframe(df: pd.DataFrame) -> dict:
    cols = [
        "Close",
        "ma200",
        "pctb_close",
        "rsi",
        "golden_cross",
        "bb_width",
        "bb_width_prev",
        "bb_width_avg",
        "vol_ratio",
        "macd_hist",
        "plus_di",
        "minus_di",
        "adx",
        "adx_prev",
        "adx_rising",
        "ixic_dist",
        "ixic_filter_active",
    ]
    need = [c for c in cols if c in df.columns]
    dfc = df[need].copy()
    return {
        "close": dfc["Close"].to_numpy(dtype=float),
        "ma200": dfc["ma200"].to_numpy(dtype=float),
        "pctb": dfc["pctb_close"].to_numpy(dtype=float),
        "rsi": dfc["rsi"].to_numpy(dtype=float),
        "golden_cross": dfc["golden_cross"].fillna(False).to_numpy(dtype=bool),
        "bb_width": dfc["bb_width"].to_numpy(dtype=float),
        "bb_width_prev": dfc["bb_width_prev"].to_numpy(dtype=float),
        "bb_width_avg": dfc["bb_width_avg"].to_numpy(dtype=float),
        "vol_ratio": dfc["vol_ratio"].to_numpy(dtype=float),
        "macd_hist": dfc["macd_hist"].to_numpy(dtype=float),
        "plus_di": dfc["plus_di"].to_numpy(dtype=float),
        "minus_di": dfc["minus_di"].to_numpy(dtype=float),
        "adx": dfc["adx"].to_numpy(dtype=float),
        "adx_prev": dfc["adx_prev"].to_numpy(dtype=float),
        "adx_rising": dfc["adx_rising"].fillna(False).to_numpy(dtype=bool),
        "ixic_dist": dfc["ixic_dist"].to_numpy(dtype=float),
        "ixic_filter_active": dfc["ixic_filter_active"].fillna(False).to_numpy(dtype=bool),
    }


def download_data(ixic_filter_df: pd.DataFrame) -> dict:
    print(f"[2/4] 종목 다운로드 및 지표 계산 ({len(base.ALL_TICKERS)}개)")
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
            df["ixic_dist"] = ixic_filter_df["ixic_dist"].reindex(df.index).ffill()
            df["ixic_filter_active"] = (
                ixic_filter_df["ixic_filter_active"].reindex(df.index).ffill().fillna(False)
            )
            packed[ticker] = pack_dataframe(df)
        except Exception as e:
            print(f"  [{ticker}] 오류: {e}")
    print(f"  -> {len(packed)}개 종목 준비 완료")
    return packed


def safe_gt(a, b) -> bool:
    return np.isfinite(a) and np.isfinite(b) and a > b


def safe_ge(a, b) -> bool:
    return np.isfinite(a) and np.isfinite(b) and a >= b


def strict_nasdaq_ok(ixic_dist, ixic_filter_active) -> bool:
    return (not bool(ixic_filter_active)) and np.isfinite(ixic_dist) and ixic_dist >= NASDAQ_DIST_UPPER


def signal_a(data: dict, i: int) -> bool:
    return (
        safe_gt(data["close"][i], data["ma200"][i])
        and bool(data["golden_cross"][i])
        and safe_gt(data["pctb"][i], LIVE_A["pctb_min"])
        and safe_gt(data["rsi"][i], LIVE_A["rsi_min"])
        and strict_nasdaq_ok(data["ixic_dist"][i], data["ixic_filter_active"][i])
    )


def signal_c(data: dict, i: int, params: dict) -> bool:
    return (
        safe_gt(data["close"][i], data["ma200"][i])
        and np.isfinite(data["bb_width_prev"][i])
        and np.isfinite(data["bb_width_avg"][i])
        and data["bb_width_avg"][i] > 0
        and data["bb_width_prev"][i] < data["bb_width_avg"][i] * params["squeeze_ratio"]
        and np.isfinite(data["bb_width"][i])
        and data["bb_width"][i] > data["bb_width_prev"][i] * params["bb_expand_ratio"]
        and safe_ge(data["vol_ratio"][i], params["vol_ratio"])
        and safe_gt(data["pctb"][i], params["pctb_min"])
        and safe_gt(data["macd_hist"][i], 0.0)
        and strict_nasdaq_ok(data["ixic_dist"][i], data["ixic_filter_active"][i])
    )


def signal_d(data: dict, i: int, params: dict) -> bool:
    adx_delta = data["adx"][i] - data["adx_prev"][i] if np.isfinite(data["adx"][i]) and np.isfinite(data["adx_prev"][i]) else np.nan
    return (
        safe_gt(data["close"][i], data["ma200"][i])
        and safe_gt(data["plus_di"][i], data["minus_di"][i])
        and safe_gt(data["adx"][i], params["adx_min"])
        and bool(data["adx_rising"][i])
        and safe_ge(adx_delta, params["adx_delta_min"])
        and safe_gt(data["macd_hist"][i], 0.0)
        and np.isfinite(data["pctb"][i])
        and params["pctb_min"] <= data["pctb"][i] <= params["pctb_max"]
        and strict_nasdaq_ok(data["ixic_dist"][i], data["ixic_filter_active"][i])
    )


def check_exit(close: float, entry_price: float, hold_days: int):
    pnl = (close - entry_price) / entry_price
    if pnl >= TARGET:
        return "목표달성", pnl
    if pnl <= -STOP:
        return "손절", pnl
    if hold_days >= HALF_EXIT_DAYS and pnl > 0:
        return "60일수익", pnl
    if hold_days >= MAX_HOLD_DAYS:
        return "기간만료", pnl
    return None, pnl


def summarize(trades: list, meta: dict) -> dict:
    if not trades:
        return {
            **meta,
            "trades": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "median_pnl": 0.0,
            "avg_hold": 0.0,
            "pf": 0.0,
            "target_rate": 0.0,
            "stop_rate": 0.0,
            "half_rate": 0.0,
            "expire_rate": 0.0,
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
    by_exit = df["exit_reason"].value_counts(normalize=True) * 100
    gross_win = wins["pnl_pct"].sum() if len(wins) else 0.0
    gross_loss = abs(loss["pnl_pct"].sum()) if len(loss) else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else np.nan
    return {
        **meta,
        "trades": int(len(df)),
        "win_rate": round(wr * 100, 2),
        "avg_pnl": round(df["pnl_pct"].mean(), 3),
        "median_pnl": round(df["pnl_pct"].median(), 3),
        "avg_hold": round(df["hold_days"].mean(), 2),
        "pf": round(float(pf), 3) if np.isfinite(pf) else np.nan,
        "target_rate": round(by_exit.get("목표달성", 0.0), 2),
        "stop_rate": round(by_exit.get("손절", 0.0), 2),
        "half_rate": round(by_exit.get("60일수익", 0.0), 2),
        "expire_rate": round(by_exit.get("기간만료", 0.0), 2),
        "ev": round(wr * avg_win + (1 - wr) * avg_loss, 3),
        "avg_win": round(avg_win, 3),
        "avg_loss": round(avg_loss, 3),
    }


def simulate_c(data_map: dict, params: dict) -> list:
    trades = []
    for ticker, data in data_map.items():
        n = len(data["close"])
        in_pos = False
        entry_price = 0.0
        entry_idx = -1
        for i in range(n):
            close = data["close"][i]
            if not np.isfinite(close):
                continue
            if in_pos:
                reason, pnl = check_exit(close, entry_price, i - entry_idx)
                if reason:
                    trades.append(
                        {"ticker": ticker, "pnl_pct": round(pnl * 100, 3), "hold_days": i - entry_idx, "exit_reason": reason}
                    )
                    in_pos = False
            if not in_pos:
                if signal_a(data, i):
                    continue
                if signal_c(data, i, params):
                    in_pos = True
                    entry_price = close
                    entry_idx = i
        if in_pos:
            pnl = (data["close"][-1] - entry_price) / entry_price
            trades.append(
                {"ticker": ticker, "pnl_pct": round(pnl * 100, 3), "hold_days": n - 1 - entry_idx, "exit_reason": "미청산"}
            )
    return trades


def simulate_d(data_map: dict, d_params: dict, c_exclusion_params: dict) -> list:
    trades = []
    for ticker, data in data_map.items():
        n = len(data["close"])
        in_pos = False
        entry_price = 0.0
        entry_idx = -1
        for i in range(n):
            close = data["close"][i]
            if not np.isfinite(close):
                continue
            if in_pos:
                reason, pnl = check_exit(close, entry_price, i - entry_idx)
                if reason:
                    trades.append(
                        {"ticker": ticker, "pnl_pct": round(pnl * 100, 3), "hold_days": i - entry_idx, "exit_reason": reason}
                    )
                    in_pos = False
            if not in_pos:
                if signal_a(data, i):
                    continue
                if signal_c(data, i, c_exclusion_params):
                    continue
                if signal_d(data, i, d_params):
                    in_pos = True
                    entry_price = close
                    entry_idx = i
        if in_pos:
            pnl = (data["close"][-1] - entry_price) / entry_price
            trades.append(
                {"ticker": ticker, "pnl_pct": round(pnl * 100, 3), "hold_days": n - 1 - entry_idx, "exit_reason": "미청산"}
            )
    return trades


def run_c_grid(data_map: dict) -> pd.DataFrame:
    rows = []
    combos = list(product(C_SQUEEZE_GRID, C_EXPAND_GRID, C_VOL_GRID, C_PCTB_GRID))
    current_meta = {
        "kind": "current",
        "squeeze_ratio": LIVE_C["squeeze_ratio"],
        "bb_expand_ratio": LIVE_C["bb_expand_ratio"],
        "vol_ratio": LIVE_C["vol_ratio"],
        "pctb_min": LIVE_C["pctb_min"],
    }
    current_trades = simulate_c(data_map, LIVE_C)
    rows.append(summarize(current_trades, current_meta))
    print(f"[3/4] C 그리드 탐색 시작 ({len(combos)} combos)")
    for idx, (sq, ex, vr, pb) in enumerate(combos, start=1):
        params = {
            "squeeze_ratio": sq,
            "bb_expand_ratio": ex,
            "vol_ratio": vr,
            "pctb_min": pb,
        }
        trades = simulate_c(data_map, params)
        rows.append(
            summarize(
                trades,
                {"kind": "grid", **params},
            )
        )
        if idx % 50 == 0:
            print(f"  C progress: {idx}/{len(combos)}")
    df = pd.DataFrame(rows)
    return df.sort_values(["ev", "pf", "win_rate", "trades"], ascending=[False, False, False, False]).reset_index(drop=True)


def run_d_grid(data_map: dict) -> pd.DataFrame:
    rows = []
    combos = []
    for adx_min, pctb_min, pctb_max, adx_delta_min in product(
        D_ADX_GRID, D_PCTB_MIN_GRID, D_PCTB_MAX_GRID, D_ADX_DELTA_GRID
    ):
        if pctb_min >= pctb_max:
            continue
        if pctb_max - pctb_min < 20:
            continue
        combos.append((adx_min, pctb_min, pctb_max, adx_delta_min))

    current_meta = {
        "kind": "current",
        "adx_min": LIVE_D["adx_min"],
        "pctb_min": LIVE_D["pctb_min"],
        "pctb_max": LIVE_D["pctb_max"],
        "adx_delta_min": LIVE_D["adx_delta_min"],
    }
    current_trades = simulate_d(data_map, LIVE_D, LIVE_C)
    rows.append(summarize(current_trades, current_meta))
    print(f"[4/4] D 그리드 탐색 시작 ({len(combos)} combos)")
    for idx, (adx_min, pctb_min, pctb_max, adx_delta_min) in enumerate(combos, start=1):
        params = {
            "adx_min": adx_min,
            "pctb_min": pctb_min,
            "pctb_max": pctb_max,
            "adx_delta_min": adx_delta_min,
        }
        trades = simulate_d(data_map, params, LIVE_C)
        rows.append(summarize(trades, {"kind": "grid", **params}))
        if idx % 50 == 0:
            print(f"  D progress: {idx}/{len(combos)}")
    df = pd.DataFrame(rows)
    return df.sort_values(["ev", "pf", "win_rate", "trades"], ascending=[False, False, False, False]).reset_index(drop=True)


def print_top(df: pd.DataFrame, group_name: str, cols: list, top_n: int = 12):
    print(f"\n[{group_name}] 상위 {top_n}개 조합")
    cur = df[df["kind"] == "current"].iloc[0]
    print(f"  현재 기준 EV={cur['ev']} / 승률={cur['win_rate']}% / 거래={int(cur['trades'])}")
    show = df.head(top_n).copy()
    show["rank"] = range(1, len(show) + 1)
    base_cols = ["rank", "kind"] + cols + ["trades", "win_rate", "avg_pnl", "ev", "pf", "stop_rate", "avg_hold"]
    print(show[base_cols].to_string(index=False))


def main():
    ixic_filter_df = download_ixic_filter()
    data_map = download_data(ixic_filter_df)

    c_df = run_c_grid(data_map)
    d_df = run_d_grid(data_map)

    base_dir = os.path.dirname(__file__)
    c_path = os.path.join(base_dir, "backtest_cd_live_grid_c.csv")
    d_path = os.path.join(base_dir, "backtest_cd_live_grid_d.csv")
    c_df.to_csv(c_path, index=False, encoding="utf-8-sig")
    d_df.to_csv(d_path, index=False, encoding="utf-8-sig")

    print_top(c_df, "C", ["squeeze_ratio", "bb_expand_ratio", "vol_ratio", "pctb_min"])
    print_top(d_df, "D", ["adx_min", "pctb_min", "pctb_max", "adx_delta_min"])

    print("\n저장 완료:")
    print(f"  {c_path}")
    print(f"  {d_path}")


if __name__ == "__main__":
    main()
