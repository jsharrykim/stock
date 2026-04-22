"""
실전 상태머신 패리티 백테스트
================================

목표
1) `updateInvestmentOpinion.gs` + `trackChanges.gs`의 핵심 상태머신을
   일봉 백테스트로 최대한 가깝게 재현한다.
2) 전 유니버스(관심종목, 다우30, 나스닥100, S&P500)에 대해
   실전 기준 성과를 다시 측정한다.
3) 기존 단순 엔진(real current config)과 패리티 엔진의 차이를 비교한다.

포함 로직
- A~F 진입/청산
- IXIC 하락장 필터(히스테리시스)
- 나스닥 고점 경고(일봉 High 근사)
- 매도 후 48시간 대기 / 10거래일 -3% 재진입 제한
- 보유 중 관망 -> 매수 복원(앵커 -3% 또는 3거래일)
- 멀티 슬롯 병행 진입/청산

제한
- `global event`(FOMC/CPI 등)는 `market_event_days.csv`가 있을 때만 반영.
  파일이 없으면 이벤트 차단은 비활성으로 둔다.
- 나스닥 고점 경고의 "당일 중 돌파"는 일봉 `High`로 근사한다.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

import backtest_combined as base
import backtest_exit_grid_current as simple_eng
import backtest_exit_grid_universes as uni


START = "2015-01-01"
END = "2026-04-15"

GROUPS = ["A", "B", "C", "D", "E", "F"]
PRIORITY = ["A", "B", "C", "D", "E", "F"]

REAL_CONFIG: dict[str, dict[str, float]] = {
    "A": {"target": 0.20, "stop": 0.30},
    "B": {"target": 0.20, "stop": 0.30},
    "C": {"target": 0.18, "stop": 0.30},
    "D": {"target": 0.18, "stop": 0.30},
    "E": {"target": 0.08, "stop": 0.30},
    "F": {"target": 0.08, "stop": 0.30},
}

# 실전 GS 로직 상수
VIX_MIN = 25.0
VIX_RELEASE = 23.0
RSI_MAX = 40.0
CCI_MIN = -100.0

GOLDEN_CROSS_PCTB_MIN = 80.0
GOLDEN_CROSS_RSI_MIN = 70.0

BB_EXPAND_RATIO = 1.05
SQUEEZE_BREAKOUT_VOL_RATIO = 1.5
SQUEEZE_BREAKOUT_PCTB_MIN = 55.0

ADX_MIN = 20.0
ADX_PCTB_MIN = 30.0
ADX_PCTB_MAX = 75.0

SQUEEZE_RATIO = 0.5
SQUEEZE_PCT_B_MAX = 50.0
BB_PCT_B_LOW_MAX = 5.0

HALF_EXIT_DAYS = 60
MAX_HOLD_DAYS = 120
SELL_HOLD_HOURS = 48
REENTRY_DAYS = 10
REENTRY_DROP = 0.03

NASDAQ_DIST_UPPER = -3.0
NASDAQ_DIST_LOWER = -12.0
NASDAQ_DIST_RELEASE = -2.5

UPPER_EXIT_MAX_WAIT_DAYS = 5
HOLD_RESTORE_DROP = 0.03
HOLD_RESTORE_MIN_TRADING_DAYS = 3

PEAK_VIX_MIN = 18.0
PEAK_VIX_SPIKE = 0.10
PEAK_IXIC_MA200_MULT = 1.10

EVENT_CALENDAR_FILE = os.path.join(os.path.dirname(__file__), "market_event_days.csv")


@dataclass
class PrimaryState:
    opinion: str = "관망"
    entry_price: float = 0.0
    entry_date: pd.Timestamp | None = None
    entry_idx: int | None = None
    strategy: str | None = None
    sell_date: pd.Timestamp | None = None
    sell_price: float = 0.0
    sell_idx: int | None = None
    hold_anchor: float = 0.0
    hold_watch_idx: int | None = None
    upper_exit_arm_idx: int | None = None

    @property
    def is_holding(self) -> bool:
        return self.entry_price > 0 and self.entry_date is not None and self.strategy is not None

    def clear_entry(self) -> None:
        self.entry_price = 0.0
        self.entry_date = None
        self.entry_idx = None
        self.strategy = None
        self.upper_exit_arm_idx = None
        self.hold_anchor = 0.0
        self.hold_watch_idx = None


@dataclass
class SlotState:
    price: float
    date: pd.Timestamp
    entry_idx: int
    strategy: str
    upper_exit_arm_idx: int | None = None


def normalize_ticker(ticker: str) -> str:
    t = str(ticker).strip().upper()
    if not t or t in {"NAN", "NONE"}:
        return ""
    if t.endswith(".KS") or t.endswith(".KQ"):
        return t
    return t.replace(".", "-")


def load_market_events(path: str = EVENT_CALENDAR_FILE) -> dict[pd.Timestamp, str]:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    date_col = "date" if "date" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else (df.columns[1] if len(df.columns) > 1 else None)
    event_map: dict[pd.Timestamp, str] = {}
    for _, row in df.iterrows():
        try:
            dt = pd.Timestamp(row[date_col]).tz_localize(None).normalize()
        except Exception:
            continue
        label = str(row[label_col]).strip() if label_col else "이벤트"
        event_map[dt] = label or "이벤트"
    return event_map


def flatten_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.droplevel(1)
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out


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


def download_market_state(events: dict[pd.Timestamp, str]) -> pd.DataFrame:
    print("[시장] VIX / IXIC 다운로드")
    vix = base.download_vix()
    vix_prev = vix.shift(1)

    ixic = flatten_ohlcv(yf.download("^IXIC", start=START, end=END, auto_adjust=True, progress=False))
    ixic["ma200"] = ixic["Close"].rolling(200).mean()
    ixic["ixic_dist"] = (ixic["Close"] / ixic["ma200"] - 1) * 100

    filter_df = compute_ixic_filter(ixic["ixic_dist"])

    peak_alert = (
        (ixic["High"] > ixic["ma200"] * PEAK_IXIC_MA200_MULT)
        & ((vix.reindex(ixic.index).ffill() > PEAK_VIX_MIN) | ((vix.reindex(ixic.index).ffill() / vix_prev.reindex(ixic.index).ffill() - 1) >= PEAK_VIX_SPIKE))
    ).fillna(False)

    market = pd.DataFrame(index=ixic.index)
    market["vix"] = vix.reindex(ixic.index).ffill()
    market["ixic_dist"] = filter_df["ixic_dist"]
    market["ixic_filter_active"] = filter_df["ixic_filter_active"].fillna(False)
    market["nasdaq_peak_alert"] = peak_alert
    market["event_label"] = [events.get(pd.Timestamp(dt).normalize(), "당분간 없음") for dt in market.index]
    return market


def download_data_for_tickers(tickers: list[str], market: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tickers = [normalize_ticker(t) for t in tickers]
    tickers = [t for t in dict.fromkeys(tickers) if t]
    print(f"[다운로드] 전체 {len(tickers)}개 티커")
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False, group_by="ticker")
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = raw[ticker].copy() if len(tickers) > 1 else raw.copy()
            df.dropna(how="all", inplace=True)
            if len(df) < 250:
                continue
            df = base.calc_indicators(df)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df["macd_prev2"] = df["macd_hist"].shift(2)
            df["vix"] = market["vix"].reindex(df.index).ffill()
            df["ixic_dist"] = market["ixic_dist"].reindex(df.index).ffill()
            df["ixic_filter_active"] = market["ixic_filter_active"].reindex(df.index).ffill().fillna(False)
            df["nasdaq_peak_alert"] = market["nasdaq_peak_alert"].reindex(df.index).ffill().fillna(False)
            df["event_label"] = market["event_label"].reindex(df.index).fillna("당분간 없음")
            df.dropna(subset=["ma200", "pctb_close", "pctb_low", "rsi", "cci", "macd_hist", "macd_prev2", "vix", "ixic_dist"], inplace=True)
            if len(df) < 250:
                continue
            result[ticker] = df
        except Exception:
            continue
    print(f"[다운로드] 준비 완료: {len(result)}개 종목")
    return result


def calc_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "n": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "median_pnl": 0.0,
            "avg_hold": 0.0,
            "pf": 0.0,
            "stop_rate": 0.0,
        }
    df = pd.DataFrame(trades)
    wins = df[df["pnl_pct"] > 0]["pnl_pct"]
    losses = df[df["pnl_pct"] <= 0]["pnl_pct"]
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    stop_rate = df["exit_reason"].astype(str).str.contains("손절").mean() * 100
    return {
        "n": int(len(df)),
        "win_rate": round((df["pnl_pct"] > 0).mean() * 100, 2),
        "avg_pnl": round(df["pnl_pct"].mean(), 3),
        "median_pnl": round(df["pnl_pct"].median(), 3),
        "avg_hold": round(df["hold_days"].mean(), 2),
        "pf": round(gross_win / gross_loss, 3) if gross_loss > 0 else 999.0,
        "stop_rate": round(stop_rate, 2),
    }


def trade_days_since(entry_idx: int | None, current_idx: int) -> int:
    if entry_idx is None:
        return 0
    return current_idx - entry_idx


def hours_since(past_date: pd.Timestamp | None, current_date: pd.Timestamp) -> float:
    if past_date is None:
        return SELL_HOLD_HOURS + 1
    return (current_date - past_date).total_seconds() / 3600.0


def hold_restore_allowed(state: PrimaryState, current_price: float, current_idx: int) -> bool:
    if state.hold_watch_idx is None:
        return True
    dd_ok = state.hold_anchor > 0 and current_price > 0 and current_price <= state.hold_anchor * (1 - HOLD_RESTORE_DROP)
    days_ok = trade_days_since(state.hold_watch_idx, current_idx) >= HOLD_RESTORE_MIN_TRADING_DAYS
    return dd_ok or days_ok


def reentry_allowed(sell_date: pd.Timestamp | None, sell_price: float, current_price: float, current_date: pd.Timestamp, current_idx: int, sell_idx: int | None) -> bool:
    if sell_date is None:
        return True
    if hours_since(sell_date, current_date) < SELL_HOLD_HOURS:
        return False
    if sell_idx is not None and trade_days_since(sell_idx, current_idx) <= REENTRY_DAYS:
        return sell_price > 0 and current_price <= sell_price * (1 - REENTRY_DROP)
    return True


def is_num(value: Any) -> bool:
    return pd.notna(value)


def row_value(row: pd.Series, key: str, default: float = np.nan) -> float:
    value = row.get(key, default)
    if pd.isna(value):
        return default
    return float(value)


def compute_entry_groups(row: pd.Series) -> tuple[dict[str, bool], str | None]:
    close = row_value(row, "Close", 0.0)
    low = row_value(row, "Low", 0.0)
    ma200 = row_value(row, "ma200")
    rsi = row_value(row, "rsi")
    cci = row_value(row, "cci")
    macd_hist = row_value(row, "macd_hist")
    macd_prev = row_value(row, "macd_prev")
    pctb_close = row_value(row, "pctb_close")
    pctb_low = row_value(row, "pctb_low")
    bb_width = row_value(row, "bb_width")
    bb_width_prev = row_value(row, "bb_width_prev")
    bb_width_avg = row_value(row, "bb_width_avg")
    vol_ratio = row_value(row, "vol_ratio")
    plus_di = row_value(row, "plus_di")
    minus_di = row_value(row, "minus_di")
    adx = row_value(row, "adx")
    adx_prev = row_value(row, "adx_prev")
    lr_slope = row_value(row, "lr_slope")
    lr_trendline = row_value(row, "lr_trendline")
    vix = row_value(row, "vix")
    ixic_dist = row_value(row, "ixic_dist")
    ixic_filter_active = bool(row.get("ixic_filter_active", False))

    bb_pair_ok = is_num(bb_width) and is_num(bb_width_avg) and bb_width_avg > 0
    nasdaq_strict = (not ixic_filter_active) and ixic_dist >= NASDAQ_DIST_UPPER
    nasdaq_bottom = not ixic_filter_active

    has_rsi = is_num(rsi)
    has_cci = is_num(cci)
    rsi_ok = has_rsi and rsi < RSI_MAX
    cci_ok = has_cci and cci < CCI_MIN
    b_cond3 = rsi_ok or cci_ok

    groups = {
        "A": (
            close > ma200
            and macd_prev <= 0
            and macd_hist > 0
            and pctb_close > GOLDEN_CROSS_PCTB_MIN
            and rsi > GOLDEN_CROSS_RSI_MIN
            and nasdaq_strict
        ),
        "B": (
            close < ma200
            and vix >= VIX_MIN
            and b_cond3
            and lr_slope > 0
            and lr_trendline > 0
            and low <= lr_trendline * 1.03
        ),
        "C": False,
        "D": False,
        "E": False,
        "F": False,
    }
    groups["C"] = (
        (not groups["A"])
        and (not groups["B"])
        and close > ma200
        and bb_pair_ok
        and bb_width_prev < bb_width_avg * SQUEEZE_RATIO
        and bb_width > bb_width_prev * BB_EXPAND_RATIO
        and vol_ratio >= SQUEEZE_BREAKOUT_VOL_RATIO
        and pctb_close > SQUEEZE_BREAKOUT_PCTB_MIN
        and macd_hist > 0
        and nasdaq_strict
    )
    groups["D"] = (
        (not groups["A"])
        and (not groups["B"])
        and (not groups["C"])
        and close > ma200
        and plus_di > minus_di
        and adx > ADX_MIN
        and adx > adx_prev
        and macd_hist > 0
        and ADX_PCTB_MIN <= pctb_close <= ADX_PCTB_MAX
        and nasdaq_strict
    )
    groups["E"] = (
        (not groups["A"])
        and (not groups["B"])
        and (not groups["C"])
        and (not groups["D"])
        and close > ma200
        and bb_pair_ok
        and (bb_width / bb_width_avg) < SQUEEZE_RATIO
        and pctb_low <= SQUEEZE_PCT_B_MAX
        and nasdaq_bottom
    )
    groups["F"] = (
        (not groups["A"])
        and (not groups["B"])
        and (not groups["C"])
        and (not groups["D"])
        and (not groups["E"])
        and close > ma200
        and pctb_low <= BB_PCT_B_LOW_MAX
        and nasdaq_bottom
    )
    chosen = next((g for g in PRIORITY if groups[g]), None)
    return groups, chosen


def compute_hold_trigger(row: pd.Series, strategy: str) -> bool:
    close = row_value(row, "Close", 0.0)
    ma200 = row_value(row, "ma200")
    rsi = row_value(row, "rsi")
    cci = row_value(row, "cci")
    macd_hist = row_value(row, "macd_hist")
    pctb_close = row_value(row, "pctb_close")
    pctb_low = row_value(row, "pctb_low")
    bb_width = row_value(row, "bb_width")
    bb_width_avg = row_value(row, "bb_width_avg")
    plus_di = row_value(row, "plus_di")
    minus_di = row_value(row, "minus_di")
    vix = row_value(row, "vix")
    lr_slope = row_value(row, "lr_slope")
    ixic_dist = row_value(row, "ixic_dist")
    ixic_filter_active = bool(row.get("ixic_filter_active", False))

    bb_pair_ok = is_num(bb_width) and is_num(bb_width_avg) and bb_width_avg > 0
    nasdaq_strict = (not ixic_filter_active) and ixic_dist >= NASDAQ_DIST_UPPER
    nasdaq_bottom = not ixic_filter_active
    has_rsi = is_num(rsi)
    has_cci = is_num(cci)
    rsi_ok = has_rsi and rsi < RSI_MAX
    cci_ok = has_cci and cci < CCI_MIN
    b_cond3_hold = (has_rsi or has_cci) and (rsi_ok or cci_ok)

    if strategy == "A":
        return close > ma200 and nasdaq_strict and macd_hist > 0
    if strategy == "B":
        return close < ma200 and vix >= VIX_RELEASE and b_cond3_hold and lr_slope > 0
    if strategy == "C":
        return close > ma200 and nasdaq_strict and macd_hist > 0
    if strategy == "D":
        return close > ma200 and nasdaq_strict and plus_di > minus_di and macd_hist > 0
    if strategy == "E":
        return close > ma200 and nasdaq_bottom and bb_pair_ok and (bb_width / bb_width_avg) < SQUEEZE_RATIO and pctb_low <= SQUEEZE_PCT_B_MAX
    if strategy == "F":
        return close > ma200 and nasdaq_bottom and pctb_low <= BB_PCT_B_LOW_MAX
    return False


def evaluate_primary_exit(state: PrimaryState, row: pd.Series, current_idx: int) -> str | None:
    if not state.is_holding:
        return None

    strategy = state.strategy
    assert strategy is not None
    close = row_value(row, "Close", 0.0)
    macd_hist = row_value(row, "macd_hist")
    macd_prev = row_value(row, "macd_prev")
    macd_prev2 = row_value(row, "macd_prev2")
    peak_alert = bool(row.get("nasdaq_peak_alert", False))

    pnl = (close - state.entry_price) / state.entry_price
    hold_days = trade_days_since(state.entry_idx, current_idx)
    target = REAL_CONFIG[strategy]["target"]
    stop = REAL_CONFIG[strategy]["stop"]

    if peak_alert:
        return "나스닥 고점 경고 — 강제 매도"
    if pnl <= -stop:
        return f"손절 기준 도달 {pnl * 100:+.2f}%"

    is_ef = strategy in {"E", "F"}
    if is_ef and pnl >= target and state.upper_exit_arm_idx is None:
        state.upper_exit_arm_idx = current_idx

    if is_ef and state.upper_exit_arm_idx is not None:
        hist_turn = (
            pd.notna(macd_hist)
            and pd.notna(macd_prev)
            and pd.notna(macd_prev2)
            and (macd_hist - macd_prev) < (macd_prev - macd_prev2)
        )
        wait_days = trade_days_since(state.upper_exit_arm_idx, current_idx)
        if pnl >= target and hist_turn:
            return f"목표 수익 구간 + MACD 둔화전환 {pnl * 100:+.2f}%"
        if wait_days >= UPPER_EXIT_MAX_WAIT_DAYS:
            return f"목표 수익 도달 후 대기 만료 {pnl * 100:+.2f}%"
    elif (not is_ef) and pnl >= target:
        return f"목표 수익 달성 즉시 매도 {pnl * 100:+.2f}%"

    if hold_days >= HALF_EXIT_DAYS and pnl > 0:
        return f"60거래일 경과 + 수익 중 {pnl * 100:+.2f}%"
    if hold_days >= MAX_HOLD_DAYS:
        return f"최대 보유 기간 초과 {pnl * 100:+.2f}%"
    return None


def evaluate_slot_entry(row: pd.Series, strategy: str) -> bool:
    close = row_value(row, "Close", 0.0)
    low = row_value(row, "Low", 0.0)
    ma200 = row_value(row, "ma200")
    rsi = row_value(row, "rsi")
    cci = row_value(row, "cci")
    macd_hist = row_value(row, "macd_hist")
    macd_prev = row_value(row, "macd_prev")
    pctb_close = row_value(row, "pctb_close")
    pctb_low = row_value(row, "pctb_low")
    bb_width = row_value(row, "bb_width")
    bb_width_prev = row_value(row, "bb_width_prev")
    bb_width_avg = row_value(row, "bb_width_avg")
    vol_ratio = row_value(row, "vol_ratio")
    plus_di = row_value(row, "plus_di")
    minus_di = row_value(row, "minus_di")
    adx = row_value(row, "adx")
    adx_prev = row_value(row, "adx_prev")
    lr_slope = row_value(row, "lr_slope")
    lr_trendline = row_value(row, "lr_trendline")
    vix = row_value(row, "vix")
    ixic_dist = row_value(row, "ixic_dist")
    ixic_filter_active = bool(row.get("ixic_filter_active", False))
    peak_alert = bool(row.get("nasdaq_peak_alert", False))
    event_watch = str(row.get("event_label", "당분간 없음")) != "당분간 없음"

    if peak_alert or event_watch or close <= 0 or pd.isna(ma200):
        return False

    bb_pair_ok = is_num(bb_width) and is_num(bb_width_avg) and bb_width_avg > 0
    nasdaq_strict = (not ixic_filter_active) and ixic_dist >= NASDAQ_DIST_UPPER
    nasdaq_bottom = not ixic_filter_active
    has_rsi = is_num(rsi)
    has_cci = is_num(cci)
    cond3 = (has_rsi and rsi < RSI_MAX) or (has_cci and cci < CCI_MIN)

    if strategy == "A":
        return close > ma200 and macd_prev <= 0 and macd_hist > 0 and pctb_close > GOLDEN_CROSS_PCTB_MIN and rsi > GOLDEN_CROSS_RSI_MIN and nasdaq_strict
    if strategy == "B":
        return close < ma200 and vix >= VIX_MIN and cond3 and lr_slope > 0 and lr_trendline > 0 and low <= lr_trendline * 1.03
    if strategy == "C":
        return close > ma200 and bb_pair_ok and bb_width_prev < bb_width_avg * SQUEEZE_RATIO and bb_width > bb_width_prev * BB_EXPAND_RATIO and vol_ratio >= SQUEEZE_BREAKOUT_VOL_RATIO and pctb_close > SQUEEZE_BREAKOUT_PCTB_MIN and macd_hist > 0 and nasdaq_strict
    if strategy == "D":
        return close > ma200 and plus_di > minus_di and adx > ADX_MIN and adx > adx_prev and macd_hist > 0 and ADX_PCTB_MIN <= pctb_close <= ADX_PCTB_MAX and nasdaq_strict
    if strategy == "E":
        return close > ma200 and bb_pair_ok and (bb_width / bb_width_avg) < SQUEEZE_RATIO and pctb_low <= SQUEEZE_PCT_B_MAX and nasdaq_bottom
    if strategy == "F":
        return close > ma200 and pctb_low <= BB_PCT_B_LOW_MAX and nasdaq_bottom
    return False


def evaluate_slot_exit(slot: SlotState, row: pd.Series, current_idx: int) -> str | None:
    close = row_value(row, "Close", 0.0)
    macd_hist = row_value(row, "macd_hist")
    macd_prev = row_value(row, "macd_prev")
    macd_prev2 = row_value(row, "macd_prev2")
    peak_alert = bool(row.get("nasdaq_peak_alert", False))

    pnl = (close - slot.price) / slot.price
    hold_days = trade_days_since(slot.entry_idx, current_idx)
    target = REAL_CONFIG[slot.strategy]["target"]
    stop = REAL_CONFIG[slot.strategy]["stop"]

    if peak_alert:
        return "나스닥 고점 경고 — 강제 매도"
    if pnl <= -stop:
        return f"손절 기준 도달 {pnl * 100:+.2f}%"

    is_ef = slot.strategy in {"E", "F"}
    if is_ef and pnl >= target and slot.upper_exit_arm_idx is None:
        slot.upper_exit_arm_idx = current_idx

    if is_ef and slot.upper_exit_arm_idx is not None:
        hist_turn = (
            pd.notna(macd_hist)
            and pd.notna(macd_prev)
            and pd.notna(macd_prev2)
            and (macd_hist - macd_prev) < (macd_prev - macd_prev2)
        )
        wait_days = trade_days_since(slot.upper_exit_arm_idx, current_idx)
        if pnl >= target and hist_turn:
            return f"목표 수익 구간 + MACD 둔화전환 {pnl * 100:+.2f}%"
        if wait_days >= UPPER_EXIT_MAX_WAIT_DAYS:
            return f"목표 수익 도달 후 대기 만료 {pnl * 100:+.2f}%"
    elif (not is_ef) and pnl >= target:
        return f"목표 수익 달성 {pnl * 100:+.2f}%"

    if hold_days >= HALF_EXIT_DAYS and pnl > 0:
        return f"60거래일 경과 + 수익 중 {pnl * 100:+.2f}%"
    if hold_days >= MAX_HOLD_DAYS:
        return f"최대 보유 기간 초과 {pnl * 100:+.2f}%"
    return None


def simulate_ticker(ticker: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    state = PrimaryState()
    slot_states: dict[str, SlotState | None] = {g: None for g in GROUPS}
    slot_sell_dates: dict[str, pd.Timestamp | None] = {g: None for g in GROUPS}
    slot_sell_prices: dict[str, float] = {g: 0.0 for g in GROUPS}
    slot_sell_indices: dict[str, int | None] = {g: None for g in GROUPS}
    trades: list[dict[str, Any]] = []

    for current_idx, (dt, row) in enumerate(df.iterrows()):
        close = row_value(row, "Close", 0.0)
        ma200 = row_value(row, "ma200")
        if close <= 0 or pd.isna(ma200):
            continue

        event_watch = str(row.get("event_label", "당분간 없음")) != "당분간 없음"
        peak_alert = bool(row.get("nasdaq_peak_alert", False))

        entry_groups, chosen_entry = compute_entry_groups(row)
        if state.is_holding:
            buy_triggered = compute_hold_trigger(row, state.strategy or "A")
        else:
            buy_triggered = chosen_entry is not None

        exit_reason = evaluate_primary_exit(state, row, current_idx) if state.is_holding else None

        # primary state machine
        if event_watch and not peak_alert:
            if state.is_holding:
                if exit_reason:
                    pnl = (close - state.entry_price) / state.entry_price
                    trades.append(
                        {
                            "ticker": ticker,
                            "leg_type": "primary",
                            "group": state.strategy,
                            "entry_date": state.entry_date,
                            "exit_date": dt,
                            "pnl_pct": round(pnl * 100, 2),
                            "hold_days": trade_days_since(state.entry_idx, current_idx),
                            "exit_reason": exit_reason,
                        }
                    )
                    state.clear_entry()
                    state.opinion = "매도"
                    state.sell_date = dt
                    state.sell_price = close
                    state.sell_idx = current_idx
                elif state.opinion == "매수":
                    if buy_triggered:
                        state.opinion = "매수"
                    else:
                        state.opinion = "관망"
                        state.hold_anchor = close
                        state.hold_watch_idx = current_idx
                else:
                    state.opinion = "관망"
            elif state.opinion == "매도":
                if hours_since(state.sell_date, dt) >= SELL_HOLD_HOURS:
                    state.opinion = "관망"
            else:
                state.opinion = "관망"
        elif state.is_holding:
            if exit_reason:
                pnl = (close - state.entry_price) / state.entry_price
                trades.append(
                    {
                        "ticker": ticker,
                        "leg_type": "primary",
                        "group": state.strategy,
                        "entry_date": state.entry_date,
                        "exit_date": dt,
                        "pnl_pct": round(pnl * 100, 2),
                        "hold_days": trade_days_since(state.entry_idx, current_idx),
                        "exit_reason": exit_reason,
                    }
                )
                state.clear_entry()
                state.opinion = "매도"
                state.sell_date = dt
                state.sell_price = close
                state.sell_idx = current_idx
            elif state.opinion == "매수":
                if buy_triggered:
                    state.opinion = "매수"
                else:
                    state.opinion = "관망"
                    state.hold_anchor = close
                    state.hold_watch_idx = current_idx
            else:
                if buy_triggered:
                    if hold_restore_allowed(state, close, current_idx):
                        state.opinion = "매수"
                        state.hold_anchor = 0.0
                        state.hold_watch_idx = None
                    else:
                        state.opinion = "관망"
                else:
                    state.opinion = "관망"
        else:
            if state.opinion == "매도":
                can_reenter = reentry_allowed(state.sell_date, state.sell_price, close, dt, current_idx, state.sell_idx)
                if hours_since(state.sell_date, dt) < SELL_HOLD_HOURS:
                    state.opinion = "매도"
                elif peak_alert:
                    state.opinion = "관망"
                elif buy_triggered and (not event_watch):
                    if can_reenter:
                        state.entry_price = close
                        state.entry_date = dt
                        state.entry_idx = current_idx
                        state.strategy = chosen_entry
                        state.opinion = "매수"
                        state.sell_date = None
                        state.sell_price = 0.0
                        state.sell_idx = None
                        state.upper_exit_arm_idx = None
                        state.hold_anchor = 0.0
                        state.hold_watch_idx = None
                    else:
                        state.opinion = "관망"
                else:
                    state.opinion = "관망"
            else:
                can_reenter = reentry_allowed(state.sell_date, state.sell_price, close, dt, current_idx, state.sell_idx)
                if (not event_watch) and (not peak_alert) and buy_triggered and can_reenter:
                    state.entry_price = close
                    state.entry_date = dt
                    state.entry_idx = current_idx
                    state.strategy = chosen_entry
                    state.opinion = "매수"
                    state.sell_date = None
                    state.sell_price = 0.0
                    state.sell_idx = None
                    state.upper_exit_arm_idx = None
                    state.hold_anchor = 0.0
                    state.hold_watch_idx = None
                else:
                    state.opinion = "관망"

        if state.is_holding and state.opinion == "매수":
            state.hold_anchor = close

        # slot state machine (primary 처리 후)
        primary_strategy = state.strategy if state.is_holding else None
        for strategy in GROUPS:
            if strategy == primary_strategy:
                continue

            slot = slot_states[strategy]
            if slot is not None:
                slot_exit_reason = evaluate_slot_exit(slot, row, current_idx)
                if slot_exit_reason:
                    pnl = (close - slot.price) / slot.price
                    trades.append(
                        {
                            "ticker": ticker,
                            "leg_type": "slot",
                            "group": strategy,
                            "entry_date": slot.date,
                            "exit_date": dt,
                            "pnl_pct": round(pnl * 100, 2),
                            "hold_days": trade_days_since(slot.entry_idx, current_idx),
                            "exit_reason": slot_exit_reason,
                        }
                    )
                    slot_states[strategy] = None
                    slot_sell_dates[strategy] = dt
                    slot_sell_prices[strategy] = close
                    slot_sell_indices[strategy] = current_idx
                continue

            if not state.is_holding:
                continue

            can_enter_slot = evaluate_slot_entry(row, strategy)
            if not can_enter_slot:
                continue

            if not reentry_allowed(slot_sell_dates[strategy], slot_sell_prices[strategy], close, dt, current_idx, slot_sell_indices[strategy]):
                continue

            slot_states[strategy] = SlotState(price=close, date=dt, entry_idx=current_idx, strategy=strategy)

    return trades


def subset_data(data: dict[str, pd.DataFrame], tickers: list[str]) -> dict[str, pd.DataFrame]:
    tickers = [normalize_ticker(t) for t in tickers]
    return {t: data[t] for t in tickers if t in data}


def convert_for_simple_engine(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    cloned: dict[str, pd.DataFrame] = {}
    for ticker, df in data.items():
        cloned[ticker] = df.copy()
    return simple_eng.prepare_data(cloned)


def main() -> None:
    events = load_market_events()
    if events:
        print(f"[이벤트] 로드 완료: {len(events)}일")
    else:
        print("[이벤트] market_event_days.csv 없음 -> 이벤트 차단 비활성")

    current_watchlist = uni.unique_clean(base.ALL_TICKERS)
    dow30 = uni.unique_clean(uni.DOW30)
    nasdaq100 = uni.unique_clean(uni.fetch_nasdaq100())
    sp500 = uni.unique_clean(uni.fetch_sp500())

    universe_map = {
        "current_watchlist": current_watchlist,
        "dow30": dow30,
        "nasdaq100": nasdaq100,
        "sp500": sp500,
    }

    union = uni.unique_clean(current_watchlist + dow30 + nasdaq100 + sp500)
    market = download_market_state(events)
    prepared_all = download_data_for_tickers(union, market)

    simple_prepared_all = convert_for_simple_engine(prepared_all)

    top_rows: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []
    trade_parts: list[pd.DataFrame] = []

    for universe, tickers in universe_map.items():
        subset = subset_data(prepared_all, tickers)
        subset_simple = {t: simple_prepared_all[t] for t in subset if t in simple_prepared_all}
        trades: list[dict[str, Any]] = []
        for ticker, df in subset.items():
            trades.extend(simulate_ticker(ticker, df))

        simple_trades = simple_eng.run_portfolio_backtest(subset_simple, REAL_CONFIG, "simple_real_current")
        parity_stats = calc_stats(trades)
        simple_stats = simple_eng.calc_stats(simple_trades)
        trade_df = pd.DataFrame(trades)
        if not trade_df.empty:
            trade_df.insert(0, "universe", universe)
            trade_parts.append(trade_df)

            grouped = (
                trade_df.groupby(["leg_type", "group"], as_index=False)
                .apply(lambda g: pd.Series(calc_stats(g.to_dict("records"))))
                .reset_index(drop=True)
            )
            grouped.insert(0, "universe", universe)
            strategy_rows.extend(grouped.to_dict("records"))

        slot_count = 0 if trade_df.empty else int((trade_df["leg_type"] == "slot").sum())
        primary_count = 0 if trade_df.empty else int((trade_df["leg_type"] == "primary").sum())
        top_rows.append(
            {
                "universe": universe,
                "coverage": len(subset),
                "event_days_loaded": len(events),
                "simple_real_avg_pnl": simple_stats["avg_pnl"],
                "simple_real_pf": simple_stats["pf"],
                "parity_avg_pnl": parity_stats["avg_pnl"],
                "parity_pf": parity_stats["pf"],
                "delta_parity_vs_simple": round(parity_stats["avg_pnl"] - simple_stats["avg_pnl"], 3),
                "primary_trades": primary_count,
                "slot_trades": slot_count,
                "slot_trade_share_pct": round((slot_count / (primary_count + slot_count) * 100), 2) if (primary_count + slot_count) else 0.0,
            }
        )

    top_df = pd.DataFrame(top_rows)
    strategy_df = pd.DataFrame(strategy_rows)
    trades_df = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()

    base_dir = os.path.dirname(__file__)
    top_path = os.path.join(base_dir, "backtest_live_parity_topline.csv")
    strategy_path = os.path.join(base_dir, "backtest_live_parity_strategy_summary.csv")
    trades_path = os.path.join(base_dir, "backtest_live_parity_trades.csv")

    top_df.to_csv(top_path, index=False, encoding="utf-8-sig")
    strategy_df.to_csv(strategy_path, index=False, encoding="utf-8-sig")
    trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")

    print("\n[Topline]")
    print(top_df.to_string(index=False))
    print("\n저장 완료:")
    print(top_path)
    print(strategy_path)
    print(trades_path)


if __name__ == "__main__":
    main()
