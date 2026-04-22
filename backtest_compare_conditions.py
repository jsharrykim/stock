"""
기존(As-Is) vs 수정(To-Be) 진입 조건 비교 백테스트
=====================================================

변경 내용:
  C: vol_ratio 1.5→2.0, pctb_close>55→>60, + plus_di>minus_di 추가
  D: adx>20→>25, plus_di-minus_di>0→>=5, pctb 30~75→40~70
  E: pctb_low<=50→<=25, + macd_hist>=macd_prev 추가
  F: pctb_low<=5 유지, + macd_hist>=macd_prev 추가 (방향 악화 필터)
  A, B: 변경 없음

유니버스: current_watchlist, dow30, nasdaq100, sp500
기간: 2015-01-01 ~ 2026-04-15
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import backtest_combined as base

# ─── 기간/출구 파라미터 ─────────────────────────────────────────────────────
START = "2015-01-01"
END   = "2026-04-15"

REAL_CONFIG = {
    "A": {"target": 0.20, "stop": 0.30},
    "B": {"target": 0.20, "stop": 0.30},
    "C": {"target": 0.18, "stop": 0.30},
    "D": {"target": 0.18, "stop": 0.30},
    "E": {"target": 0.08, "stop": 0.30},
    "F": {"target": 0.08, "stop": 0.30},
}

HALF_EXIT_DAYS = 60
MAX_HOLD_DAYS  = 120
UPPER_EXIT_MAX_WAIT_DAYS = 5

# ─── 시장 필터 ──────────────────────────────────────────────────────────────
NASDAQ_DIST_UPPER   = -3.0
NASDAQ_DIST_LOWER   = -12.0
NASDAQ_DIST_RELEASE = -2.5
VIX_MIN = 25.0

# ─── 진입 조건 파라미터 (As-Is) ──────────────────────────────────────────────
ASIS = {
    "GOLDEN_CROSS_PCTB_MIN": 80.0,
    "GOLDEN_CROSS_RSI_MIN":  70.0,
    "SQUEEZE_RATIO": 0.5,
    "BB_EXPAND_RATIO": 1.05,
    # C
    "C_VOL_RATIO": 1.5,
    "C_PCTB_MIN":  55.0,
    # D
    "D_ADX_MIN":   20.0,
    "D_DI_DIFF":   0.0,   # +DI - -DI >= 0 (즉 +DI > -DI)
    "D_PCTB_MIN":  30.0,
    "D_PCTB_MAX":  75.0,
    # E
    "E_PCTB_MAX":  50.0,
    "E_MACD_GATE": False,  # 진입 시 macd_hist >= macd_prev 체크 없음
    # F
    "F_PCTB_MAX":  5.0,
    "F_MACD_GATE": False,
    # B
    "RSI_MAX": 40.0,
    "CCI_MIN": -100.0,
}

# ─── 진입 조건 파라미터 (To-Be) ──────────────────────────────────────────────
TOBE = {
    "GOLDEN_CROSS_PCTB_MIN": 80.0,
    "GOLDEN_CROSS_RSI_MIN":  70.0,
    "SQUEEZE_RATIO": 0.5,
    "BB_EXPAND_RATIO": 1.05,
    # C: 더 강한 필터
    "C_VOL_RATIO": 2.0,
    "C_PCTB_MIN":  60.0,
    # D: 더 강한 필터
    "D_ADX_MIN":   25.0,
    "D_DI_DIFF":   5.0,   # +DI - -DI >= 5
    "D_PCTB_MIN":  40.0,
    "D_PCTB_MAX":  70.0,
    # E: 더 강한 필터
    "E_PCTB_MAX":  25.0,
    "E_MACD_GATE": True,   # macd_hist >= macd_prev 필요
    # F: MACD 방향 필터 추가
    "F_PCTB_MAX":  5.0,
    "F_MACD_GATE": True,
    # B
    "RSI_MAX": 40.0,
    "CCI_MIN": -100.0,
}

GROUPS   = ["A", "B", "C", "D", "E", "F"]
PRIORITY = ["A", "B", "C", "D", "E", "F"]

# ─── 유니버스 ────────────────────────────────────────────────────────────────
from backtest_exit_grid_universes import fetch_sp500, fetch_nasdaq100, DOW30

UNIVERSES = {
    "current_watchlist": base.ALL_TICKERS,
    "dow30":             DOW30,
    "nasdaq100":         fetch_nasdaq100(),
    "sp500":             fetch_sp500(),
}

# ─── 시장 데이터 캐시 ────────────────────────────────────────────────────────
def download_market_data():
    vix_raw  = yf.download("^VIX",  start=START, end=END, auto_adjust=False, progress=False)
    ixic_raw = yf.download("^IXIC", start=START, end=END, auto_adjust=True,  progress=False)
    vix = vix_raw["Close"].squeeze().rename("vix")
    ixic_close = ixic_raw["Close"].squeeze()
    ixic_ma200 = ixic_close.rolling(200).mean()
    ixic_dist  = ((ixic_close - ixic_ma200) / ixic_ma200 * 100).rename("ixic_dist")

    # 나스닥 hysteresis 필터
    filter_active = pd.Series(False, index=ixic_dist.index, name="ixic_filter_active")
    active = False
    for dt in ixic_dist.index:
        d = ixic_dist.get(dt, np.nan)
        if np.isnan(d):
            filter_active[dt] = active
            continue
        if not active and d < NASDAQ_DIST_LOWER:
            active = True
        elif active and d >= NASDAQ_DIST_RELEASE:
            active = False
        filter_active[dt] = active

    mkt = pd.DataFrame({"vix": vix, "ixic_dist": ixic_dist, "ixic_filter_active": filter_active})
    return mkt.ffill()


def attach_market(df: pd.DataFrame, mkt: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    aligned = mkt.reindex(df.index, method="ffill")
    df["vix"]               = aligned["vix"].values
    df["ixic_dist"]         = aligned["ixic_dist"].values
    df["ixic_filter_active"]= aligned["ixic_filter_active"].values
    return df


def row_to_signal(row: dict, params: dict) -> dict:
    def fv(k, d=np.nan):
        v = row.get(k, d)
        if isinstance(v, bool):
            return v
        try:
            f = float(v)
            return f if not np.isnan(f) else d
        except Exception:
            return d

    close  = fv("Close", 0.0)
    ma200  = fv("ma200", np.inf)
    return {
        "above200":        close > ma200 if not np.isinf(ma200) else False,
        "golden_cross":    bool(row.get("golden_cross", False)),
        "pctb_close":      fv("pctb_close"),
        "pctb_low":        fv("pctb_low"),
        "rsi":             fv("rsi"),
        "cci":             fv("cci"),
        "macd_hist":       fv("macd_hist"),
        "macd_prev":       fv("macd_prev"),
        "adx":             fv("adx"),
        "adx_rising":      bool(row.get("adx_rising", False)),
        "plus_di":         fv("plus_di"),
        "minus_di":        fv("minus_di"),
        "vol_ratio":       fv("vol_ratio"),
        "squeeze":         bool(row.get("squeeze", False)),
        "prev_squeeze":    bool(row.get("prev_squeeze", False)),
        "bb_width":        fv("bb_width"),
        "bb_width_prev":   fv("bb_width_prev"),
        "bb_width_avg":    fv("bb_width_avg"),
        "lr_slope":        fv("lr_slope"),
        "lr_trendline":    fv("lr_trendline"),
        "low":             fv("Low", np.nan),
        "vix":             fv("vix"),
        "ixic_dist":       fv("ixic_dist"),
        "ixic_filter_active": bool(row.get("ixic_filter_active", False)),
    }


def signal_by_group(s: dict, params: dict) -> dict:
    nasdaq_strict = (not s["ixic_filter_active"]) and s["ixic_dist"] >= NASDAQ_DIST_UPPER
    nasdaq_bottom = not s["ixic_filter_active"]

    a = (
        s["above200"]
        and s["golden_cross"]
        and s["pctb_close"] > params["GOLDEN_CROSS_PCTB_MIN"]
        and s["rsi"] > params["GOLDEN_CROSS_RSI_MIN"]
        and nasdaq_strict
    )

    b = (
        (not s["above200"])
        and s["vix"] >= VIX_MIN
        and (s["rsi"] < params["RSI_MAX"] or s["cci"] < params["CCI_MIN"])
        and s["lr_slope"] > 0
        and s["lr_trendline"] > 0
        and s["low"] <= s["lr_trendline"] * 1.03
    )

    # squeeze detection: 전일 스퀴즈 상태이고 오늘 BB 확장
    sq_prev = s["prev_squeeze"]  # 전일 squeeze (calc_indicators에서 shift(1))
    bb_expand = s["bb_width"] > s["bb_width_prev"] * params["BB_EXPAND_RATIO"]

    c_base = (
        (not a) and (not b)
        and s["above200"]
        and sq_prev
        and bb_expand
        and s["vol_ratio"] >= params["C_VOL_RATIO"]
        and s["pctb_close"] > params["C_PCTB_MIN"]
        and s["macd_hist"] > 0
        and nasdaq_strict
    )
    # To-Be 에서만: plus_di > minus_di 추가
    if "C_DI_FILTER" in params and params["C_DI_FILTER"]:
        c = c_base and (s["plus_di"] > s["minus_di"])
    else:
        c = c_base

    d_di_diff = s["plus_di"] - s["minus_di"]
    d = (
        (not a) and (not b) and (not c)
        and s["above200"]
        and d_di_diff >= params["D_DI_DIFF"]
        and s["adx"] > params["D_ADX_MIN"]
        and s["adx_rising"]
        and s["macd_hist"] > 0
        and params["D_PCTB_MIN"] <= s["pctb_close"] <= params["D_PCTB_MAX"]
        and nasdaq_strict
    )

    macd_improving = (
        pd.notna(s["macd_hist"]) and pd.notna(s["macd_prev"])
        and s["macd_hist"] >= s["macd_prev"]
    )

    e_base = (
        (not a) and (not b) and (not c) and (not d)
        and s["above200"]
        and s["squeeze"]
        and s["pctb_low"] <= params["E_PCTB_MAX"]
        and nasdaq_bottom
    )
    e = e_base and (macd_improving if params["E_MACD_GATE"] else True)

    f_base = (
        (not a) and (not b) and (not c) and (not d) and (not e)
        and s["above200"]
        and s["pctb_low"] <= params["F_PCTB_MAX"]
        and nasdaq_bottom
    )
    f = f_base and (macd_improving if params["F_MACD_GATE"] else True)

    return {"A": a, "B": b, "C": c, "D": d, "E": e, "F": f}


def check_exit_values(group: str, close: float, macd_hist: float, macd_prev: float, macd_prev2: float,
                      entry_price: float, hold_days: int, target: float, stop: float,
                      first_target_idx: int | None, current_idx: int):
    pnl = (close - entry_price) / entry_price
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
            if hist_turn or wait_days >= UPPER_EXIT_MAX_WAIT_DAYS:
                return "상단매도", pnl, updated_first_target
    else:
        if pnl >= target:
            return "목표달성", pnl, updated_first_target

    return None, pnl, updated_first_target


def calc_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "win_rate": 0.0, "avg_pnl": 0.0, "pf": 0.0, "stop_rate": 0.0, "avg_hold": 0.0}
    df = pd.DataFrame(trades)
    wins   = df[df["pnl_pct"] > 0]["pnl_pct"]
    losses = df[df["pnl_pct"] <= 0]["pnl_pct"]
    gw = wins.sum()
    gl = abs(losses.sum())
    return {
        "n":         int(len(df)),
        "win_rate":  round((df["pnl_pct"] > 0).mean() * 100, 1),
        "avg_pnl":   round(df["pnl_pct"].mean(), 2),
        "pf":        round(gw / gl, 2) if gl > 0 else 999.0,
        "stop_rate": round((df["exit_reason"] == "손절").mean() * 100, 1),
        "avg_hold":  round(df["hold_days"].mean(), 1),
    }


def run_backtest(prepared: dict, params: dict, config: dict) -> tuple[list, list]:
    """포트폴리오(우선순위) 백테스트 + 그룹별 독립 백테스트"""

    # ── 그룹별 독립 백테스트 ───────────────────────────────────────────────
    group_trades = []
    for group in GROUPS:
        target = config[group]["target"]
        stop   = config[group]["stop"]
        for ticker, bundle in prepared.items():
            entries = bundle["signals"][group]
            in_pos  = False
            entry_price = 0.0
            entry_date  = None
            entry_idx   = 0
            first_target_idx = None
            for i, dt in enumerate(bundle["dates"]):
                if in_pos:
                    reason, pnl, first_target_idx = check_exit_values(
                        group, bundle["close"][i], bundle["macd_hist"][i],
                        bundle["macd_prev"][i], bundle["macd_prev2"][i],
                        entry_price, i - entry_idx, target, stop, first_target_idx, i
                    )
                    if reason:
                        group_trades.append({
                            "group": group, "ticker": ticker,
                            "entry_date": entry_date, "exit_date": dt,
                            "pnl_pct": round(pnl * 100, 2),
                            "hold_days": i - entry_idx,
                            "exit_reason": reason,
                        })
                        in_pos = False
                        first_target_idx = None
                if (not in_pos) and entries[i]:
                    in_pos = True
                    entry_price = float(bundle["close"][i])
                    entry_date  = dt
                    entry_idx   = i
                    first_target_idx = None

    # ── 포트폴리오(우선순위) 백테스트 ─────────────────────────────────────
    port_trades = []
    for ticker, bundle in prepared.items():
        in_pos  = False
        entry_price = 0.0
        entry_date  = None
        entry_idx   = 0
        entry_group = None
        first_target_idx = None
        for i, dt in enumerate(bundle["dates"]):
            if in_pos:
                target = config[entry_group]["target"]
                stop   = config[entry_group]["stop"]
                reason, pnl, first_target_idx = check_exit_values(
                    entry_group, bundle["close"][i], bundle["macd_hist"][i],
                    bundle["macd_prev"][i], bundle["macd_prev2"][i],
                    entry_price, i - entry_idx, target, stop, first_target_idx, i
                )
                if reason:
                    port_trades.append({
                        "group": entry_group, "ticker": ticker,
                        "entry_date": entry_date, "exit_date": dt,
                        "pnl_pct": round(pnl * 100, 2),
                        "hold_days": i - entry_idx,
                        "exit_reason": reason,
                    })
                    in_pos = False
                    first_target_idx = None
            if not in_pos:
                for g in PRIORITY:
                    if bundle["signals"][g][i]:
                        in_pos = True
                        entry_price = float(bundle["close"][i])
                        entry_date  = dt
                        entry_idx   = i
                        entry_group = g
                        first_target_idx = None
                        break

    return group_trades, port_trades


def prepare_bundle(df: pd.DataFrame, params: dict) -> dict | None:
    df = df.copy()
    df["macd_prev2"] = df["macd_hist"].shift(2)
    required_core = ["ma200", "pctb_close", "pctb_low", "rsi", "cci",
                     "macd_hist", "macd_prev", "vix", "ixic_dist"]
    dfc = df.dropna(subset=required_core).copy()
    if len(dfc) < 250:
        return None
    signals_rows = []
    for row in dfc.to_dict("records"):
        s = row_to_signal(row, params)
        signals_rows.append(signal_by_group(s, params))
    sig_df = pd.DataFrame(signals_rows, index=dfc.index)
    return {
        "dates":     list(dfc.index),
        "close":     dfc["Close"].to_numpy(dtype=float),
        "macd_hist": dfc["macd_hist"].to_numpy(dtype=float),
        "macd_prev": dfc["macd_prev"].to_numpy(dtype=float),
        "macd_prev2": dfc["macd_prev2"].to_numpy(dtype=float),
        "signals":   {g: sig_df[g].to_numpy(dtype=bool) for g in GROUPS},
    }


def count_entry_signals(prepared: dict) -> dict:
    """그룹별 진입 신호 횟수 집계"""
    counts = {g: 0 for g in GROUPS}
    for bundle in prepared.values():
        for g in GROUPS:
            counts[g] += int(bundle["signals"][g].sum())
    return counts


def main():
    print("=" * 70)
    print("기존(As-Is) vs 수정(To-Be) 진입 조건 비교 백테스트")
    print("=" * 70)

    # ASIS에 C_DI_FILTER 추가
    asis_params = dict(ASIS, C_DI_FILTER=False)
    tobe_params = dict(TOBE, C_DI_FILTER=True)

    print("\n▶ 시장 데이터 다운로드 중...")
    mkt = download_market_data()

    all_results = []

    for uni_name, tickers in UNIVERSES.items():
        print(f"\n{'─'*60}")
        print(f"유니버스: {uni_name}  ({len(tickers)} tickers)")

        # 1) 데이터 다운로드
        raw_data = {}
        batch = yf.download(tickers, start=START, end=END, auto_adjust=True,
                            progress=False, group_by="ticker")
        for tk in tickers:
            try:
                if len(tickers) == 1:
                    df_tk = batch.copy()
                else:
                    df_tk = batch[tk].copy() if tk in batch.columns.get_level_values(0) else pd.DataFrame()
                if df_tk.empty or len(df_tk) < 250:
                    continue
                raw_data[tk] = df_tk
            except Exception:
                continue

        # 2) 지표 계산
        indicator_data = {}
        for tk, df_tk in raw_data.items():
            try:
                df_ind = base.calc_indicators(df_tk)
                df_ind = attach_market(df_ind, mkt)
                df_ind["low"] = df_tk["Low"]
                indicator_data[tk] = df_ind
            except Exception:
                continue

        print(f"   데이터 준비: {len(indicator_data)} tickers")

        # 3) As-Is 준비
        prepared_asis = {}
        for tk, df_ind in indicator_data.items():
            b = prepare_bundle(df_ind, asis_params)
            if b:
                prepared_asis[tk] = b

        # 4) To-Be 준비
        prepared_tobe = {}
        for tk, df_ind in indicator_data.items():
            b = prepare_bundle(df_ind, tobe_params)
            if b:
                prepared_tobe[tk] = b

        # 5) 진입 신호 수 비교
        asis_cnt = count_entry_signals(prepared_asis)
        tobe_cnt = count_entry_signals(prepared_tobe)
        print(f"\n   [진입 신호 수 비교]")
        print(f"   {'그룹':>4} {'As-Is':>8} {'To-Be':>8} {'변화':>8} {'변화율':>8}")
        for g in GROUPS:
            diff = tobe_cnt[g] - asis_cnt[g]
            rate = diff / asis_cnt[g] * 100 if asis_cnt[g] > 0 else 0
            print(f"   {g:>4} {asis_cnt[g]:>8} {tobe_cnt[g]:>8} {diff:>+8} {rate:>+7.1f}%")

        # 6) 백테스트 실행
        asis_group_trades, asis_port_trades = run_backtest(prepared_asis, asis_params, REAL_CONFIG)
        tobe_group_trades, tobe_port_trades = run_backtest(prepared_tobe, tobe_params, REAL_CONFIG)

        # 7) 그룹별 결과 비교
        print(f"\n   [그룹별 독립 성과 비교]")
        header = f"   {'그룹':>4} | {'As-Is N':>7} {'승률':>6} {'avg%':>7} {'PF':>6} {'손절':>6} | {'To-Be N':>7} {'승률':>6} {'avg%':>7} {'PF':>6} {'손절':>6}"
        print(header)
        print("   " + "-" * (len(header) - 3))

        for g in GROUPS:
            a_tr = [t for t in asis_group_trades if t["group"] == g]
            t_tr = [t for t in tobe_group_trades if t["group"] == g]
            a_s = calc_stats(a_tr)
            t_s = calc_stats(t_tr)
            print(
                f"   {g:>4} | "
                f"{a_s['n']:>7} {a_s['win_rate']:>5.1f}% {a_s['avg_pnl']:>6.2f}% {a_s['pf']:>5.2f}x {a_s['stop_rate']:>5.1f}% | "
                f"{t_s['n']:>7} {t_s['win_rate']:>5.1f}% {t_s['avg_pnl']:>6.2f}% {t_s['pf']:>5.2f}x {t_s['stop_rate']:>5.1f}%"
            )
            all_results.append({
                "universe": uni_name, "group": g,
                "asis_n": a_s["n"], "asis_wr": a_s["win_rate"],
                "asis_avg": a_s["avg_pnl"], "asis_pf": a_s["pf"], "asis_stop": a_s["stop_rate"],
                "tobe_n": t_s["n"], "tobe_wr": t_s["win_rate"],
                "tobe_avg": t_s["avg_pnl"], "tobe_pf": t_s["pf"], "tobe_stop": t_s["stop_rate"],
            })

        # 8) 포트폴리오 비교
        a_port = calc_stats(asis_port_trades)
        t_port = calc_stats(tobe_port_trades)
        print(f"\n   [포트폴리오(우선순위) 전체 성과]")
        print(f"   {'':>6} | {'As-Is N':>7} {'승률':>6} {'avg%':>7} {'PF':>6} {'손절':>6} | {'To-Be N':>7} {'승률':>6} {'avg%':>7} {'PF':>6} {'손절':>6}")
        print(
            f"   {'전체':>6} | "
            f"{a_port['n']:>7} {a_port['win_rate']:>5.1f}% {a_port['avg_pnl']:>6.2f}% {a_port['pf']:>5.2f}x {a_port['stop_rate']:>5.1f}% | "
            f"{t_port['n']:>7} {t_port['win_rate']:>5.1f}% {t_port['avg_pnl']:>6.2f}% {t_port['pf']:>5.2f}x {t_port['stop_rate']:>5.1f}%"
        )

    # 9) 전체 집계
    print(f"\n{'=' * 70}")
    print("전체 유니버스 통합 그룹별 요약")
    print(f"{'=' * 70}")
    df_all = pd.DataFrame(all_results)
    summary = df_all.groupby("group").agg(
        asis_n=("asis_n", "sum"),
        asis_wr=("asis_wr", "mean"),
        asis_avg=("asis_avg", "mean"),
        asis_pf=("asis_pf", "mean"),
        asis_stop=("asis_stop", "mean"),
        tobe_n=("tobe_n", "sum"),
        tobe_wr=("tobe_wr", "mean"),
        tobe_avg=("tobe_avg", "mean"),
        tobe_pf=("tobe_pf", "mean"),
        tobe_stop=("tobe_stop", "mean"),
    ).reset_index()

    print(f"\n{'그룹':>4} | {'As-Is':>35} | {'To-Be':>35} | {'판정':>8}")
    print(f"{'':>4} | {'N':>5} {'승률':>6} {'avg%':>7} {'PF':>6} {'손절%':>6} | {'N':>5} {'승률':>6} {'avg%':>7} {'PF':>6} {'손절%':>6} |")
    print("-" * 100)
    for _, row in summary.iterrows():
        g = row["group"]
        # 판정: avg_pnl & PF 모두 개선되면 "개선", 악화 시 "주의", 혼합 "중립"
        avg_better = row["tobe_avg"] > row["asis_avg"]
        pf_better  = row["tobe_pf"]  > row["asis_pf"]
        stop_better = row["tobe_stop"] < row["asis_stop"]
        score = sum([avg_better, pf_better, stop_better])
        if score >= 2:
            verdict = "✅ 개선"
        elif score == 0:
            verdict = "⚠️ 주의"
        else:
            verdict = "↔ 중립"
        print(
            f"{g:>4} | "
            f"{int(row['asis_n']):>5} {row['asis_wr']:>5.1f}% {row['asis_avg']:>6.2f}% {row['asis_pf']:>5.2f}x {row['asis_stop']:>5.1f}% | "
            f"{int(row['tobe_n']):>5} {row['tobe_wr']:>5.1f}% {row['tobe_avg']:>6.2f}% {row['tobe_pf']:>5.2f}x {row['tobe_stop']:>5.1f}% | "
            f"{verdict:>8}"
        )

    # CSV 저장
    df_all.to_csv("backtest_compare_conditions.csv", index=False, encoding="utf-8-sig")
    print("\n결과 저장: backtest_compare_conditions.csv")


if __name__ == "__main__":
    main()
