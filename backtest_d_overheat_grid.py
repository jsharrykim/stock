"""
전략 D 과열 필터 비교 백테스트
================================

목적:
1) 현재 실전 `updateInvestmentOpinion.gs`의 전략 D 조건과 최대한 같은 parity base를 만든다.
2) 전략 D에 `IXIC 상단 캡`과 `VIX 상한` AND 필터를 1단위 그리드로 얹어 비교한다.
3) D 단독 성과와 전체 A~F 포트폴리오 성과를 함께 본다.

주의:
- 이벤트 데이 필터는 백테스트 데이터 부재로 미반영
- NasdaqPeakSellState(장중 과열+VIX 급등 조합)도 히스토리 부재로 미반영
- 진입/청산은 기존 스크립트와 동일하게 당일 종가 체결로 단순화
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf


START = "2015-01-01"
END = "2026-04-15"

# updateInvestmentOpinion.gs parity
NASDAQ_DIST_UPPER = -3.0
NASDAQ_DIST_LOWER = -12.0
NASDAQ_DIST_RELEASE = -2.5

VIX_MIN = 30.0
VIX_RELEASE = 23.0
RSI_MAX = 35.0
CCI_MIN = -150.0
LR_TOUCH_RATIO = 1.05

GOLDEN_CROSS_PCTB_MIN = 80.0
GOLDEN_CROSS_RSI_MIN = 70.0

C_SQUEEZE_RATIO = 0.45
BB_EXPAND_RATIO = 1.00
SQUEEZE_BREAKOUT_VOL_RATIO = 1.50
SQUEEZE_BREAKOUT_PCTB_MIN = 55.0

ADX_MIN = 30.0
ADX_PCTB_MIN = 30.0
ADX_PCTB_MAX = 80.0

SQUEEZE_RATIO = 0.50
SQUEEZE_PCT_B_MAX = 50.0
BB_PCT_B_LOW_MAX = 3.0

TARGET_PCT = 0.20
STOP_PCT = 0.30
HALF_EXIT_DAYS = 60
MAX_HOLD_DAYS = 120
UPPER_EXIT_MAX_WAIT_DAYS = 5

PRIORITY = ["A", "B", "C", "D", "E", "F"]

IXIC_CAP_GRID = list(range(3, 16))
VIX_CAP_GRID = list(range(15, 26))
PCTB_CAP_GRID = list(range(65, 81))
BBW_RATIO_CAP_GRID = [round(x, 1) for x in np.arange(1.0, 3.1, 0.1)]
IXIC_BB_COMBO_GRID = [12, 13]

BB_PERIOD = 20
BB_STD = 2.0
BB_AVG = 60
MACD_F, MACD_S, MACD_SIG = 12, 26, 9
ADX_PERIOD = 14
LR_WINDOW = 120

KR_TICKERS = [
    "000660.KS","005930.KS","277810.KS","034020.KS","015760.KS",
    "005380.KS","012450.KS","042660.KS","042700.KQ","096770.KS",
    "009150.KS","000270.KS","247540.KQ","376900.KS","004020.KS",
    "329180.KS","375500.KS","086280.KS","000720.KS","353200.KQ",
    "011070.KS","079550.KS",
]
US_TICKERS = [
    "HOOD","AVGO","AMD","MSFT","GOOGL","NVDA","TSLA",
    "MU","LRCX","ON","SNDK","ASTS","AVAV","IONQ",
    "RKLB","PLTR","APP","SOXL","TSLL","TE","ONDS",
    "BE","PL","VRT","LITE","TER","ANET","IREN","HOOG",
    "SOLT","ETHU","NBIS","LPTH","CONL","GLW","FLNC",
    "VST","ASX","CRCL","SGML","AEHR","MP","PLAB","SKYT",
    "SMTC","COHR","MPWR","CIEN","KLAC","FORM","CRDO",
]
ALL_TICKERS = KR_TICKERS + US_TICKERS


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    df["ma200"] = c.rolling(200).mean()
    df["ma20"] = c.rolling(BB_PERIOD).mean()

    std20 = c.rolling(BB_PERIOD).std()
    bb_upper = df["ma20"] + BB_STD * std20
    bb_lower = df["ma20"] - BB_STD * std20
    bb_range = bb_upper - bb_lower
    df["bb_width"] = (bb_range / df["ma20"] * 100).where(df["ma20"] > 0)
    df["bb_width_avg"] = df["bb_width"].rolling(BB_AVG).mean()
    df["bb_width_prev"] = df["bb_width"].shift(1)
    df["squeeze"] = df["bb_width"] < df["bb_width_avg"] * SQUEEZE_RATIO
    df["prev_squeeze"] = df["squeeze"].shift(1).fillna(False)
    df["pctb_close"] = np.where(bb_range > 0, (c - bb_lower) / bb_range * 100, np.nan)
    df["pctb_low"] = np.where(bb_range > 0, (l - bb_lower) / bb_range * 100, np.nan)

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + np.where(loss == 0, 100.0, gain / loss))
    df["rsi_prev"] = df["rsi"].shift(1)
    df["rsi_rising"] = df["rsi"] > df["rsi_prev"]

    tp = (h + l + c) / 3
    tp_ma = tp.rolling(14).mean()
    tp_md = tp.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = np.where(tp_md > 0, (tp - tp_ma) / (0.015 * tp_md), 0)

    ema_f = c.ewm(span=MACD_F, adjust=False).mean()
    ema_s = c.ewm(span=MACD_S, adjust=False).mean()
    macd_line = ema_f - ema_s
    sig_line = macd_line.ewm(span=MACD_SIG, adjust=False).mean()
    df["macd_hist"] = macd_line - sig_line
    df["macd_prev"] = df["macd_hist"].shift(1)
    df["golden_cross"] = (df["macd_prev"] <= 0) & (df["macd_hist"] > 0)

    p = ADX_PERIOD
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    plus_dm = np.where((h.diff() > 0) & (h.diff() > (-l.diff())), h.diff(), 0.0)
    minus_dm = np.where((-l.diff() > 0) & (-l.diff() > h.diff()), -l.diff(), 0.0)

    atr = pd.Series(tr, index=df.index).rolling(p).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(p).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(p).mean() / atr
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(p).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = adx
    df["adx_prev"] = adx.shift(1)
    df["adx_rising"] = df["adx"] > df["adx_prev"]

    vol20 = v.rolling(20).mean()
    df["vol_ratio"] = v / vol20

    idx = np.arange(len(df))
    lr_trendline = np.full(len(df), np.nan)
    lr_slope = np.full(len(df), np.nan)
    for i in range(LR_WINDOW - 1, len(df)):
        y = c.iloc[i - LR_WINDOW + 1:i + 1].values
        x = idx[i - LR_WINDOW + 1:i + 1]
        if np.isnan(y).any():
            continue
        slope, intercept = np.polyfit(x, y, 1)
        lr_slope[i] = slope
        lr_trendline[i] = slope * idx[i] + intercept
    df["lr_trendline"] = lr_trendline
    df["lr_slope"] = lr_slope
    return df


def download_vix() -> pd.Series:
    print("[0/5] VIX 다운로드")
    vix_raw = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.droplevel(1)
    vix = vix_raw["Close"].copy()
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    return vix


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
    print("[1/5] IXIC 다운로드")
    ixic_raw = yf.download("^IXIC", start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(ixic_raw.columns, pd.MultiIndex):
        ixic_raw.columns = ixic_raw.columns.droplevel(1)
    ixic = ixic_raw.copy()
    ixic.index = pd.to_datetime(ixic.index).tz_localize(None)
    ixic["ma200"] = ixic["Close"].rolling(200).mean()
    ixic["ixic_dist"] = (ixic["Close"] / ixic["ma200"] - 1) * 100
    return compute_ixic_filter(ixic["ixic_dist"])


def pack_dataframe(df: pd.DataFrame) -> dict:
    bb_width_ratio = np.where(df["bb_width_avg"] > 0, df["bb_width"] / df["bb_width_avg"], np.nan)
    ma200_dist = np.where(df["ma200"] > 0, (df["Close"] / df["ma200"] - 1) * 100, np.nan)
    return {
        "dates": list(df.index),
        "close": df["Close"].to_numpy(dtype=float),
        "low": df["Low"].to_numpy(dtype=float),
        "ma200": df["ma200"].to_numpy(dtype=float),
        "ma200_dist": ma200_dist.astype(float),
        "pctb_close": df["pctb_close"].to_numpy(dtype=float),
        "pctb_low": df["pctb_low"].to_numpy(dtype=float),
        "rsi": df["rsi"].to_numpy(dtype=float),
        "cci": df["cci"].to_numpy(dtype=float),
        "macd_hist": df["macd_hist"].to_numpy(dtype=float),
        "macd_prev": df["macd_prev"].to_numpy(dtype=float),
        "macd_prev2": df["macd_prev2"].to_numpy(dtype=float),
        "golden_cross": df["golden_cross"].fillna(False).to_numpy(dtype=bool),
        "plus_di": df["plus_di"].to_numpy(dtype=float),
        "minus_di": df["minus_di"].to_numpy(dtype=float),
        "adx": df["adx"].to_numpy(dtype=float),
        "adx_prev": df["adx_prev"].to_numpy(dtype=float),
        "adx_rising": df["adx_rising"].fillna(False).to_numpy(dtype=bool),
        "vol_ratio": df["vol_ratio"].to_numpy(dtype=float),
        "bb_width": df["bb_width"].to_numpy(dtype=float),
        "bb_width_prev": df["bb_width_prev"].to_numpy(dtype=float),
        "bb_width_avg": df["bb_width_avg"].to_numpy(dtype=float),
        "bb_width_ratio": bb_width_ratio.astype(float),
        "squeeze": df["squeeze"].fillna(False).to_numpy(dtype=bool),
        "lr_trendline": df["lr_trendline"].to_numpy(dtype=float),
        "lr_slope": df["lr_slope"].to_numpy(dtype=float),
        "vix": df["vix"].to_numpy(dtype=float),
        "ixic_dist": df["ixic_dist"].to_numpy(dtype=float),
        "ixic_filter_active": df["ixic_filter_active"].fillna(False).to_numpy(dtype=bool),
    }


def download_data(vix_series: pd.Series, ixic_filter_df: pd.DataFrame) -> dict:
    print(f"[2/5] 종목 다운로드 및 지표 계산 ({len(ALL_TICKERS)}개)")
    raw = yf.download(ALL_TICKERS, start=START, end=END, auto_adjust=True, progress=False, group_by="ticker")
    packed = {}
    for ticker in ALL_TICKERS:
        try:
            df = raw[ticker].copy() if len(ALL_TICKERS) > 1 else raw.copy()
            df.dropna(how="all", inplace=True)
            if len(df) < 250:
                continue
            df = calc_indicators(df)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df["vix"] = vix_series.reindex(df.index).ffill()
            df["ixic_dist"] = ixic_filter_df["ixic_dist"].reindex(df.index).ffill()
            df["ixic_filter_active"] = (
                ixic_filter_df["ixic_filter_active"].reindex(df.index).ffill().fillna(False)
            )
            df["macd_prev2"] = df["macd_hist"].shift(2)
            packed[ticker] = pack_dataframe(df)
        except Exception as e:
            print(f"  [{ticker}] 오류: {e}")
    print(f"  -> {len(packed)}개 종목 준비 완료")
    return packed


def safe_gt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b) & (a > b)


def safe_ge(a: np.ndarray, b: float) -> np.ndarray:
    return np.isfinite(a) & (a >= b)


def safe_le(a: np.ndarray, b: float) -> np.ndarray:
    return np.isfinite(a) & (a <= b)


def build_base_conditions(data: dict) -> dict:
    close = data["close"]
    ma200 = data["ma200"]
    pctb_close = data["pctb_close"]
    pctb_low = data["pctb_low"]
    rsi = data["rsi"]
    cci = data["cci"]
    macd_hist = data["macd_hist"]
    plus_di = data["plus_di"]
    minus_di = data["minus_di"]
    adx = data["adx"]
    adx_rising = data["adx_rising"]
    vol_ratio = data["vol_ratio"]
    bb_width = data["bb_width"]
    bb_width_prev = data["bb_width_prev"]
    bb_width_avg = data["bb_width_avg"]
    low = data["low"]
    lr_trendline = data["lr_trendline"]
    lr_slope = data["lr_slope"]
    ixic_dist = data["ixic_dist"]
    ixic_filter_active = data["ixic_filter_active"]

    above200 = safe_gt(close, ma200)
    below200 = safe_gt(ma200, close)
    nasdaq_strict = (~ixic_filter_active) & np.isfinite(ixic_dist) & (ixic_dist >= NASDAQ_DIST_UPPER)
    nasdaq_bottom = ~ixic_filter_active

    a = (
        above200
        & data["golden_cross"]
        & safe_gt(pctb_close, np.full_like(pctb_close, GOLDEN_CROSS_PCTB_MIN))
        & safe_gt(rsi, np.full_like(rsi, GOLDEN_CROSS_RSI_MIN))
        & nasdaq_strict
    )

    b_oversold = (
        (np.isfinite(rsi) & (rsi < RSI_MAX))
        | (np.isfinite(cci) & (cci < CCI_MIN))
    )
    b = (
        below200
        & np.isfinite(data["vix"]) & (data["vix"] >= VIX_MIN)
        & b_oversold
        & np.isfinite(lr_slope) & (lr_slope > 0)
        & np.isfinite(lr_trendline) & (lr_trendline > 0)
        & np.isfinite(low) & (low <= lr_trendline * LR_TOUCH_RATIO)
    )

    c = (
        (~a) & (~b)
        & above200
        & np.isfinite(bb_width_prev) & np.isfinite(bb_width_avg) & (bb_width_avg > 0)
        & (bb_width_prev / bb_width_avg < C_SQUEEZE_RATIO)
        & np.isfinite(bb_width) & (bb_width > bb_width_prev * BB_EXPAND_RATIO)
        & safe_ge(vol_ratio, SQUEEZE_BREAKOUT_VOL_RATIO)
        & safe_gt(pctb_close, np.full_like(pctb_close, SQUEEZE_BREAKOUT_PCTB_MIN))
        & safe_gt(macd_hist, np.zeros_like(macd_hist))
        & nasdaq_strict
    )

    d_base = (
        (~a) & (~b) & (~c)
        & above200
        & safe_gt(plus_di, minus_di)
        & safe_gt(adx, np.full_like(adx, ADX_MIN))
        & adx_rising
        & safe_gt(macd_hist, np.zeros_like(macd_hist))
        & np.isfinite(pctb_close)
        & (pctb_close >= ADX_PCTB_MIN)
        & (pctb_close <= ADX_PCTB_MAX)
        & nasdaq_strict
    )

    e_core = (
        above200
        & data["squeeze"]
        & np.isfinite(pctb_low) & (pctb_low <= SQUEEZE_PCT_B_MAX)
        & nasdaq_bottom
    )
    f_core = (
        above200
        & np.isfinite(pctb_low) & (pctb_low <= BB_PCT_B_LOW_MAX)
        & nasdaq_bottom
    )

    return {
        "a": a,
        "b": b,
        "c": c,
        "d_base": d_base,
        "e_core": e_core,
        "f_core": f_core,
    }


def build_scenario_signals(data: dict, base_cond: dict, ixic_cap=None, vix_cap=None, pctb_cap=None, bbw_ratio_cap=None) -> dict:
    d_extra = np.ones_like(data["close"], dtype=bool)
    if ixic_cap is not None:
        d_extra &= np.isfinite(data["ixic_dist"]) & (data["ixic_dist"] <= ixic_cap)
    if vix_cap is not None:
        d_extra &= np.isfinite(data["vix"]) & (data["vix"] <= vix_cap)
    if pctb_cap is not None:
        d_extra &= np.isfinite(data["pctb_close"]) & (data["pctb_close"] <= pctb_cap)
    if bbw_ratio_cap is not None:
        d_extra &= np.isfinite(data["bb_width_ratio"]) & (data["bb_width_ratio"] <= bbw_ratio_cap)

    a = base_cond["a"]
    b = base_cond["b"]
    c = base_cond["c"]
    d = base_cond["d_base"] & d_extra
    e = (~a) & (~b) & (~c) & (~d) & base_cond["e_core"]
    f = (~a) & (~b) & (~c) & (~d) & (~e) & base_cond["f_core"]
    return {"A": a, "B": b, "C": c, "D": d, "E": e, "F": f}


def check_exit_values(group: str, data: dict, i: int, entry_price: float, entry_idx: int, first_target_idx):
    close = data["close"][i]
    pnl = (close - entry_price) / entry_price

    if pnl <= -STOP_PCT:
        return "손절", pnl, first_target_idx
    if i - entry_idx >= HALF_EXIT_DAYS and pnl > 0:
        return "60일수익", pnl, first_target_idx
    if i - entry_idx >= MAX_HOLD_DAYS:
        return "기간만료", pnl, first_target_idx

    if group in ("E", "F"):
        if first_target_idx is None and pnl >= TARGET_PCT:
            first_target_idx = i
        if first_target_idx is not None:
            macd_hist = data["macd_hist"][i]
            macd_prev = data["macd_prev"][i]
            macd_prev2 = data["macd_prev2"][i]
            hist_turn = (
                np.isfinite(macd_hist) and np.isfinite(macd_prev) and np.isfinite(macd_prev2)
                and (macd_hist - macd_prev) < (macd_prev - macd_prev2)
            )
            wait_days = i - first_target_idx
            if pnl >= TARGET_PCT and hist_turn:
                return "목표+MACD둔화", pnl, first_target_idx
            if wait_days >= UPPER_EXIT_MAX_WAIT_DAYS:
                return "목표후대기만료", pnl, first_target_idx
        return None, pnl, first_target_idx

    if pnl >= TARGET_PCT:
        return "목표달성", pnl, first_target_idx
    return None, pnl, first_target_idx


def simulate_d_only(data_map: dict, cond_map: dict, ixic_cap=None, vix_cap=None, pctb_cap=None, bbw_ratio_cap=None, scenario_name="baseline") -> list[dict]:
    trades = []
    for ticker, data in data_map.items():
        signals = build_scenario_signals(data, cond_map[ticker], ixic_cap, vix_cap, pctb_cap, bbw_ratio_cap)
        entries = signals["D"]
        in_pos = False
        entry_price = 0.0
        entry_idx = -1
        entry_date = None
        entry_ixic = np.nan
        entry_vix = np.nan
        entry_pctb = np.nan
        entry_adx = np.nan
        entry_bbw_ratio = np.nan
        entry_ma200_dist = np.nan
        entry_rsi = np.nan
        for i, dt in enumerate(data["dates"]):
            close = data["close"][i]
            if not np.isfinite(close):
                continue
            if in_pos:
                reason, pnl, _ = check_exit_values("D", data, i, entry_price, entry_idx, None)
                if reason:
                    trades.append({
                        "scope": "d_only",
                        "scenario": scenario_name,
                        "ticker": ticker,
                        "group": "D",
                        "entry_date": entry_date,
                        "exit_date": dt,
                        "pnl_pct": round(pnl * 100, 3),
                        "hold_days": i - entry_idx,
                        "exit_reason": reason,
                        "entry_ixic_dist": round(float(entry_ixic), 3) if np.isfinite(entry_ixic) else np.nan,
                        "entry_vix": round(float(entry_vix), 3) if np.isfinite(entry_vix) else np.nan,
                        "entry_pctb": round(float(entry_pctb), 3) if np.isfinite(entry_pctb) else np.nan,
                        "entry_adx": round(float(entry_adx), 3) if np.isfinite(entry_adx) else np.nan,
                        "entry_bbw_ratio": round(float(entry_bbw_ratio), 3) if np.isfinite(entry_bbw_ratio) else np.nan,
                        "entry_ma200_dist": round(float(entry_ma200_dist), 3) if np.isfinite(entry_ma200_dist) else np.nan,
                        "entry_rsi": round(float(entry_rsi), 3) if np.isfinite(entry_rsi) else np.nan,
                    })
                    in_pos = False
            if not in_pos and entries[i]:
                in_pos = True
                entry_price = close
                entry_idx = i
                entry_date = dt
                entry_ixic = data["ixic_dist"][i]
                entry_vix = data["vix"][i]
                entry_pctb = data["pctb_close"][i]
                entry_adx = data["adx"][i]
                entry_bbw_ratio = data["bb_width_ratio"][i]
                entry_ma200_dist = data["ma200_dist"][i]
                entry_rsi = data["rsi"][i]
        if in_pos:
            pnl = (data["close"][-1] - entry_price) / entry_price
            trades.append({
                "scope": "d_only",
                "scenario": scenario_name,
                "ticker": ticker,
                "group": "D",
                "entry_date": entry_date,
                "exit_date": data["dates"][-1],
                "pnl_pct": round(pnl * 100, 3),
                "hold_days": len(data["close"]) - 1 - entry_idx,
                "exit_reason": "미청산",
                "entry_ixic_dist": round(float(entry_ixic), 3) if np.isfinite(entry_ixic) else np.nan,
                "entry_vix": round(float(entry_vix), 3) if np.isfinite(entry_vix) else np.nan,
                "entry_pctb": round(float(entry_pctb), 3) if np.isfinite(entry_pctb) else np.nan,
                "entry_adx": round(float(entry_adx), 3) if np.isfinite(entry_adx) else np.nan,
                "entry_bbw_ratio": round(float(entry_bbw_ratio), 3) if np.isfinite(entry_bbw_ratio) else np.nan,
                "entry_ma200_dist": round(float(entry_ma200_dist), 3) if np.isfinite(entry_ma200_dist) else np.nan,
                "entry_rsi": round(float(entry_rsi), 3) if np.isfinite(entry_rsi) else np.nan,
            })
    return trades


def simulate_portfolio(data_map: dict, cond_map: dict, ixic_cap=None, vix_cap=None, pctb_cap=None, bbw_ratio_cap=None, scenario_name="baseline") -> list[dict]:
    trades = []
    for ticker, data in data_map.items():
        signals = build_scenario_signals(data, cond_map[ticker], ixic_cap, vix_cap, pctb_cap, bbw_ratio_cap)
        in_pos = False
        entry_price = 0.0
        entry_idx = 0
        entry_date = None
        entry_group = None
        entry_ixic = np.nan
        entry_vix = np.nan
        first_target_idx = None
        for i, dt in enumerate(data["dates"]):
            if in_pos:
                reason, pnl, first_target_idx = check_exit_values(entry_group, data, i, entry_price, entry_idx, first_target_idx)
                if reason:
                    trades.append({
                        "scope": "portfolio",
                        "scenario": scenario_name,
                        "ticker": ticker,
                        "group": entry_group,
                        "entry_date": entry_date,
                        "exit_date": dt,
                        "pnl_pct": round(pnl * 100, 3),
                        "hold_days": i - entry_idx,
                        "exit_reason": reason,
                        "entry_ixic_dist": round(float(entry_ixic), 3) if np.isfinite(entry_ixic) else np.nan,
                        "entry_vix": round(float(entry_vix), 3) if np.isfinite(entry_vix) else np.nan,
                    })
                    in_pos = False
                    entry_group = None
                    first_target_idx = None
            if not in_pos:
                chosen = next((g for g in PRIORITY if signals[g][i]), None)
                if chosen:
                    in_pos = True
                    entry_group = chosen
                    entry_price = float(data["close"][i])
                    entry_date = dt
                    entry_idx = i
                    entry_ixic = data["ixic_dist"][i]
                    entry_vix = data["vix"][i]
                    first_target_idx = None
    return trades


def calc_stats(trades: list[dict], meta: dict) -> dict:
    if not trades:
        return {
            **meta,
            "trades": 0,
            "d_trades": 0,
            "d_share": 0.0,
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
            "avg_entry_ixic": np.nan,
            "avg_entry_vix": np.nan,
            "avg_entry_pctb": np.nan,
            "avg_entry_bbw_ratio": np.nan,
            "avg_entry_ma200_dist": np.nan,
            "avg_entry_rsi": np.nan,
            "pct_ixic_ge_8": 0.0,
            "pct_ixic_ge_9": 0.0,
            "pct_ixic_ge_10": 0.0,
            "pct_ixic_ge_11": 0.0,
            "pct_ixic_ge_12": 0.0,
        }

    df = pd.DataFrame(trades)
    wins = df[df["pnl_pct"] > 0]
    losses = df[df["pnl_pct"] <= 0]
    wr = len(wins) / len(df)
    avg_win = wins["pnl_pct"].mean() if len(wins) else 0.0
    avg_loss = losses["pnl_pct"].mean() if len(losses) else 0.0
    gross_win = wins["pnl_pct"].sum()
    gross_loss = abs(losses["pnl_pct"].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else np.nan
    d_trades = int((df["group"] == "D").sum()) if "group" in df else len(df)
    ixic_vals = df["entry_ixic_dist"].dropna()
    vix_vals = df["entry_vix"].dropna()
    pctb_vals = df["entry_pctb"].dropna() if "entry_pctb" in df else pd.Series(dtype=float)
    bbw_vals = df["entry_bbw_ratio"].dropna() if "entry_bbw_ratio" in df else pd.Series(dtype=float)
    ma200_dist_vals = df["entry_ma200_dist"].dropna() if "entry_ma200_dist" in df else pd.Series(dtype=float)
    rsi_vals = df["entry_rsi"].dropna() if "entry_rsi" in df else pd.Series(dtype=float)

    return {
        **meta,
        "trades": int(len(df)),
        "d_trades": d_trades,
        "d_share": round(d_trades / len(df) * 100, 2),
        "win_rate": round(wr * 100, 2),
        "avg_pnl": round(df["pnl_pct"].mean(), 3),
        "median_pnl": round(df["pnl_pct"].median(), 3),
        "avg_hold": round(df["hold_days"].mean(), 2),
        "pf": round(float(pf), 3) if np.isfinite(pf) else np.nan,
        "target_rate": round((df["exit_reason"] == "목표달성").mean() * 100, 2),
        "stop_rate": round((df["exit_reason"] == "손절").mean() * 100, 2),
        "half_rate": round((df["exit_reason"] == "60일수익").mean() * 100, 2),
        "expire_rate": round((df["exit_reason"] == "기간만료").mean() * 100, 2),
        "ev": round(wr * avg_win + (1 - wr) * avg_loss, 3),
        "avg_win": round(avg_win, 3),
        "avg_loss": round(avg_loss, 3),
        "avg_entry_ixic": round(ixic_vals.mean(), 3) if len(ixic_vals) else np.nan,
        "avg_entry_vix": round(vix_vals.mean(), 3) if len(vix_vals) else np.nan,
        "avg_entry_pctb": round(pctb_vals.mean(), 3) if len(pctb_vals) else np.nan,
        "avg_entry_bbw_ratio": round(bbw_vals.mean(), 3) if len(bbw_vals) else np.nan,
        "avg_entry_ma200_dist": round(ma200_dist_vals.mean(), 3) if len(ma200_dist_vals) else np.nan,
        "avg_entry_rsi": round(rsi_vals.mean(), 3) if len(rsi_vals) else np.nan,
        "pct_ixic_ge_8": round((ixic_vals >= 8).mean() * 100, 2) if len(ixic_vals) else 0.0,
        "pct_ixic_ge_9": round((ixic_vals >= 9).mean() * 100, 2) if len(ixic_vals) else 0.0,
        "pct_ixic_ge_10": round((ixic_vals >= 10).mean() * 100, 2) if len(ixic_vals) else 0.0,
        "pct_ixic_ge_11": round((ixic_vals >= 11).mean() * 100, 2) if len(ixic_vals) else 0.0,
        "pct_ixic_ge_12": round((ixic_vals >= 12).mean() * 100, 2) if len(ixic_vals) else 0.0,
    }


def scenario_rows(scope: str, data_map: dict, cond_map: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    trades_all = []

    scenarios = [("baseline", None, None, None, None)]
    scenarios += [(f"ixic_cap_{cap}", float(cap), None, None, None) for cap in IXIC_CAP_GRID]
    scenarios += [(f"vix_cap_{cap}", None, float(cap), None, None) for cap in VIX_CAP_GRID]
    scenarios += [
        (f"ixic_cap_{ixic_cap}_vix_cap_{vix_cap}", float(ixic_cap), float(vix_cap), None, None)
        for ixic_cap in IXIC_CAP_GRID
        for vix_cap in VIX_CAP_GRID
    ]
    scenarios += [(f"pctb_cap_{cap}", None, None, float(cap), None) for cap in PCTB_CAP_GRID]
    scenarios += [(f"bbw_ratio_cap_{cap:.1f}", None, None, None, float(cap)) for cap in BBW_RATIO_CAP_GRID]
    scenarios += [
        (f"ixic_cap_{ixic_cap}_pctb_cap_{pctb_cap}", float(ixic_cap), None, float(pctb_cap), None)
        for ixic_cap in IXIC_BB_COMBO_GRID
        for pctb_cap in PCTB_CAP_GRID
    ]
    scenarios += [
        (f"ixic_cap_{ixic_cap}_bbw_ratio_cap_{bbw_cap:.1f}", float(ixic_cap), None, None, float(bbw_cap))
        for ixic_cap in IXIC_BB_COMBO_GRID
        for bbw_cap in BBW_RATIO_CAP_GRID
    ]

    total = len(scenarios)
    for idx, (name, ixic_cap, vix_cap, pctb_cap, bbw_ratio_cap) in enumerate(scenarios, start=1):
        if scope == "d_only":
            trades = simulate_d_only(
                data_map, cond_map,
                ixic_cap=ixic_cap, vix_cap=vix_cap,
                pctb_cap=pctb_cap, bbw_ratio_cap=bbw_ratio_cap,
                scenario_name=name
            )
        else:
            trades = simulate_portfolio(
                data_map, cond_map,
                ixic_cap=ixic_cap, vix_cap=vix_cap,
                pctb_cap=pctb_cap, bbw_ratio_cap=bbw_ratio_cap,
                scenario_name=name
            )
        row = calc_stats(trades, {
            "scope": scope,
            "scenario": name,
            "ixic_cap": ixic_cap,
            "vix_cap": vix_cap,
            "pctb_cap": pctb_cap,
            "bbw_ratio_cap": bbw_ratio_cap,
        })
        rows.append(row)
        trades_all.extend(trades)
        if idx == 1 or idx % 25 == 0 or idx == total:
            print(f"  [{scope}] {idx}/{total} 완료")

    df = pd.DataFrame(rows)
    baseline = df[df["scenario"] == "baseline"].iloc[0]
    df["delta_ev"] = (df["ev"] - baseline["ev"]).round(3)
    df["delta_pf"] = (df["pf"] - baseline["pf"]).round(3)
    df["delta_avg_pnl"] = (df["avg_pnl"] - baseline["avg_pnl"]).round(3)
    df["trade_ratio"] = (df["trades"] / baseline["trades"]).round(3) if baseline["trades"] else 0.0

    sort_cols = ["ev", "pf", "avg_pnl", "win_rate", "trades"]
    df = df.sort_values(sort_cols, ascending=[False, False, False, False, False]).reset_index(drop=True)
    trades_df = pd.DataFrame(trades_all)
    return df, trades_df


def select_recommendation(d_only_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> dict:
    d_base = d_only_df[d_only_df["scenario"] == "baseline"].iloc[0]
    p_base = portfolio_df[portfolio_df["scenario"] == "baseline"].iloc[0]

    candidates = d_only_df[
        (d_only_df["scenario"] != "baseline")
        & (d_only_df["ev"] >= d_base["ev"])
        & (d_only_df["pf"] >= d_base["pf"])
        & (d_only_df["stop_rate"] <= d_base["stop_rate"])
        & (d_only_df["trade_ratio"] >= 0.55)
    ].copy()
    if candidates.empty:
        return {
            "recommendation": "hold",
            "reason": "D 단독 기준으로 baseline을 확실히 이기는 후보가 없음",
        }

    merged = candidates.merge(
        portfolio_df[["scenario", "avg_pnl", "pf", "ev", "trades"]].rename(columns={
            "avg_pnl": "portfolio_avg_pnl",
            "pf": "portfolio_pf",
            "ev": "portfolio_ev",
            "trades": "portfolio_trades",
        }),
        on="scenario",
        how="left",
    )
    merged = merged[
        (merged["portfolio_avg_pnl"] >= p_base["avg_pnl"] - 0.10)
        & (merged["portfolio_pf"] >= p_base["pf"] - 0.05)
    ].copy()
    if merged.empty:
        return {
            "recommendation": "hold",
            "reason": "D 단독 개선 후보는 있지만 전체 포트폴리오 훼손이 큼",
        }

    merged["cap_preference"] = merged["ixic_cap"].notna().astype(int) * 10 + merged["vix_cap"].notna().astype(int)
    merged = merged.sort_values(
        ["delta_ev", "delta_pf", "portfolio_avg_pnl", "trade_ratio", "cap_preference"],
        ascending=[False, False, False, False, False],
    )
    top = merged.iloc[0]
    if pd.notna(top["ixic_cap"]) and pd.isna(top["vix_cap"]):
        rec = "ixic_only"
    elif pd.notna(top["ixic_cap"]) and pd.notna(top["vix_cap"]):
        rec = "ixic_and_vix"
    else:
        rec = "vix_only"
    return {
        "recommendation": rec,
        "scenario": top["scenario"],
        "ixic_cap": top["ixic_cap"],
        "vix_cap": top["vix_cap"],
        "delta_ev": top["delta_ev"],
        "delta_pf": top["delta_pf"],
        "portfolio_avg_pnl": top["portfolio_avg_pnl"],
        "trade_ratio": top["trade_ratio"],
    }


def main():
    print("=" * 88)
    print("전략 D 과열 필터 비교 백테스트")
    print("=" * 88)
    print("실전 parity: ADX>30 / %B 30~80 / target +20% / stop -30%")

    vix = download_vix()
    ixic_filter_df = download_ixic_filter()
    data_map = download_data(vix, ixic_filter_df)

    print("[3/5] 기본 조건 전처리")
    cond_map = {ticker: build_base_conditions(data) for ticker, data in data_map.items()}

    print("[4/5] D 단독 시나리오 비교")
    d_only_df, d_only_trades = scenario_rows("d_only", data_map, cond_map)

    print("[5/5] 전체 포트폴리오 시나리오 비교")
    portfolio_df, portfolio_trades = scenario_rows("portfolio", data_map, cond_map)

    recommendation = select_recommendation(d_only_df, portfolio_df)

    base_dir = os.path.dirname(__file__)
    d_only_path = os.path.join(base_dir, "backtest_d_overheat_d_only.csv")
    d_only_trades_path = os.path.join(base_dir, "backtest_d_overheat_d_only_trades.csv")
    portfolio_path = os.path.join(base_dir, "backtest_d_overheat_portfolio.csv")
    portfolio_trades_path = os.path.join(base_dir, "backtest_d_overheat_portfolio_trades.csv")

    d_only_df.to_csv(d_only_path, index=False, encoding="utf-8-sig")
    d_only_trades.to_csv(d_only_trades_path, index=False, encoding="utf-8-sig")
    portfolio_df.to_csv(portfolio_path, index=False, encoding="utf-8-sig")
    portfolio_trades.to_csv(portfolio_trades_path, index=False, encoding="utf-8-sig")

    d_base = d_only_df[d_only_df["scenario"] == "baseline"].iloc[0]
    p_base = portfolio_df[portfolio_df["scenario"] == "baseline"].iloc[0]

    print("\n[D 단독 baseline]")
    print(
        f"  거래={int(d_base['trades'])} / 승률={d_base['win_rate']}% / 평균손익={d_base['avg_pnl']}% "
        f"/ EV={d_base['ev']} / PF={d_base['pf']} / 손절률={d_base['stop_rate']}%"
    )
    print(
        f"  진입 IXIC 평균={d_base['avg_entry_ixic']} / IXIC>=10% 비중={d_base['pct_ixic_ge_10']}% / "
        f"진입 VIX 평균={d_base['avg_entry_vix']}"
    )

    print("\n[포트폴리오 baseline]")
    print(
        f"  거래={int(p_base['trades'])} / 평균손익={p_base['avg_pnl']}% / EV={p_base['ev']} / PF={p_base['pf']} "
        f"/ D 비중={p_base['d_share']}%"
    )

    print("\n[D 단독 상위 10개]")
    show_cols = [
        "scenario", "ixic_cap", "vix_cap", "trades", "trade_ratio", "win_rate",
        "avg_pnl", "ev", "pf", "stop_rate", "avg_entry_ixic", "avg_entry_vix",
        "pct_ixic_ge_10", "delta_ev", "delta_pf",
    ]
    print(d_only_df.head(10)[show_cols].to_string(index=False))

    print("\n[포트폴리오 상위 10개]")
    show_cols_p = [
        "scenario", "ixic_cap", "vix_cap", "trades", "d_trades", "d_share",
        "win_rate", "avg_pnl", "ev", "pf", "stop_rate", "delta_avg_pnl", "delta_pf",
    ]
    print(portfolio_df.head(10)[show_cols_p].to_string(index=False))

    print("\n[추천]")
    print(recommendation)
    print("\n저장 완료:")
    print(" ", d_only_path)
    print(" ", d_only_trades_path)
    print(" ", portfolio_path)
    print(" ", portfolio_trades_path)


if __name__ == "__main__":
    main()
