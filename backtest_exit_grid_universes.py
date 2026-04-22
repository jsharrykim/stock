"""
동일 엔진 기준 유니버스별 익절/손절 비교
=====================================

유니버스
- current_watchlist: 현재 백테스트 관심 종목군
- dow30
- nasdaq100
- sp500

동일 기준
- 전략 조건 / 지표 / 기간 / 매도 로직 모두 backtest_exit_grid_current.py 기준 고정
- 각 유니버스마다
  1) 현재 설정 성과
  2) 공통 target/stop 그리드 최고안
  3) 그룹별 현재 vs 최고
를 출력/저장
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf

import backtest_combined as base
import backtest_exit_grid_current as eng

SP500_SNAPSHOT = "/Users/jungsoo.kim/.cursor/projects/Users-jungsoo-kim-Desktop-backtest/agent-tools/b1a57c13-c857-414b-81f9-b916c3b8cc14.txt"
NASDAQ100_SNAPSHOT = "/Users/jungsoo.kim/.cursor/projects/Users-jungsoo-kim-Desktop-backtest/agent-tools/3eef208b-cb00-483c-9ec2-5f61bfd12462.txt"

DOW30 = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]


def normalize_ticker(t: str) -> str:
    t = str(t).strip().upper()
    if not t or t in {"NAN", "NONE"}:
        return ""
    if t.endswith(".KS") or t.endswith(".KQ"):
        return t
    return t.replace(".", "-")


def parse_markdown_tickers(path: str, section_title: str, end_title_prefix: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    start = text.index(section_title)
    end = text.index(end_title_prefix, start)
    section = text[start:end]
    tickers = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if not parts:
            continue
        first = parts[0]
        if first.startswith("["):
            first = first.split("]")[0].lstrip("[")
        if first in {"Symbol", "Ticker", "---"}:
            continue
        t = normalize_ticker(first)
        if t:
            tickers.append(t)
    return unique_clean(tickers)


def fetch_sp500() -> list[str]:
    return parse_markdown_tickers(SP500_SNAPSHOT, "## S&P 500 component stocks", "## Selected changes to the list of S&P 500 components")


def fetch_nasdaq100() -> list[str]:
    return parse_markdown_tickers(NASDAQ100_SNAPSHOT, "## Current components", "## Component changes")


def unique_clean(seq):
    out = []
    seen = set()
    for x in seq:
        x = normalize_ticker(x)
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def download_vix_and_ixic():
    vix = base.download_vix()
    ixic_filter_df = eng.download_ixic_filter()
    return vix, ixic_filter_df


def download_data_for_tickers(tickers: list[str], vix_series: pd.Series, ixic_filter_df: pd.DataFrame):
    tickers = unique_clean(tickers)
    print(f"[다운로드] 전체 {len(tickers)}개 티커")
    raw = yf.download(tickers, start=eng.START, end=eng.END, auto_adjust=True, progress=False, group_by="ticker")
    result = {}
    for t in tickers:
        try:
            df = raw[t].copy() if len(tickers) > 1 else raw.copy()
            df.dropna(how="all", inplace=True)
            if len(df) < 250:
                continue
            df = base.calc_indicators(df)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df["vix"] = vix_series.reindex(df.index).ffill()
            df["ixic_dist"] = ixic_filter_df["ixic_dist"].reindex(df.index).ffill()
            df["ixic_filter_active"] = ixic_filter_df["ixic_filter_active"].reindex(df.index).ffill().fillna(False)
            result[t] = df
        except Exception:
            continue
    eng.build_prev2(result)
    return eng.prepare_data(result)


def subset_prepared(prepared: dict, tickers: list[str]) -> dict:
    return {t: prepared[t] for t in unique_clean(tickers) if t in prepared}


def current_target_stop(group: str):
    cfg = eng.CURRENT_CONFIG[group]
    return round(cfg["target"] * 100, 1), round(cfg["stop"] * 100, 1)


def evaluate_universe(name: str, prepared_subset: dict):
    print(f"\n=== {name} ({len(prepared_subset)}개 종목) ===")
    portfolio_rows = []
    current_trades = eng.run_portfolio_backtest(prepared_subset, eng.CURRENT_CONFIG, "current")
    portfolio_rows.append({"scenario": "current", **eng.calc_stats(current_trades)})
    for target in eng.TARGET_GRID:
        for stop in eng.STOP_GRID:
            cfg = {g: {"target": target, "stop": stop} for g in eng.GROUPS}
            trades = eng.run_portfolio_backtest(prepared_subset, cfg, f"common_t{int(target*100)}_s{int(stop*100)}")
            portfolio_rows.append({
                "scenario": f"common_t{int(target*100)}_s{int(stop*100)}",
                "target_pct": round(target * 100, 1),
                "stop_pct": round(stop * 100, 1),
                **eng.calc_stats(trades),
            })
    portfolio_df = pd.DataFrame(portfolio_rows).sort_values(["avg_pnl", "pf"], ascending=[False, False])

    group_rows = []
    for group in eng.GROUPS:
        for target in eng.TARGET_GRID:
            for stop in eng.STOP_GRID:
                trades = eng.run_group_backtest(prepared_subset, group, target, stop)
                group_rows.append({
                    "group": group,
                    "target_pct": round(target * 100, 1),
                    "stop_pct": round(stop * 100, 1),
                    **eng.calc_stats(trades),
                })
    group_df = pd.DataFrame(group_rows).sort_values(["group", "avg_pnl", "pf"], ascending=[True, False, False])

    summary_rows = []
    for group in eng.GROUPS:
        gdf = group_df[group_df["group"] == group]
        best = gdf.iloc[0]
        cur_t, cur_s = current_target_stop(group)
        cur = gdf[(gdf["target_pct"] == cur_t) & (gdf["stop_pct"] == cur_s)]
        cur = cur.iloc[0] if not cur.empty else None
        summary_rows.append({
            "universe": name,
            "group": group,
            "current_target": cur_t,
            "current_stop": cur_s,
            "current_avg_pnl": None if cur is None else cur["avg_pnl"],
            "best_target": best["target_pct"],
            "best_stop": best["stop_pct"],
            "best_avg_pnl": best["avg_pnl"],
            "delta_avg_pnl": None if cur is None else round(best["avg_pnl"] - cur["avg_pnl"], 3),
        })
    summary_df = pd.DataFrame(summary_rows)

    best_common = portfolio_df.iloc[0].to_dict()
    current_row = portfolio_df[portfolio_df["scenario"] == "current"].iloc[0].to_dict()
    return portfolio_df, group_df, summary_df, current_row, best_common


def main():
    current_watchlist = unique_clean(base.ALL_TICKERS)
    dow30 = unique_clean(DOW30)
    nasdaq100 = unique_clean(fetch_nasdaq100())
    sp500 = unique_clean(fetch_sp500())

    universe_map = {
        "current_watchlist": current_watchlist,
        "dow30": dow30,
        "nasdaq100": nasdaq100,
        "sp500": sp500,
    }

    union = unique_clean(current_watchlist + dow30 + nasdaq100 + sp500)

    print("=" * 90)
    print("동일 엔진 기준 유니버스별 익절/손절 비교")
    print("=" * 90)
    vix, ixic_filter_df = download_vix_and_ixic()
    prepared_all = download_data_for_tickers(union, vix, ixic_filter_df)

    portfolio_parts = []
    group_parts = []
    summary_parts = []
    top_rows = []

    for name, tickers in universe_map.items():
        prepared_subset = subset_prepared(prepared_all, tickers)
        portfolio_df, group_df, summary_df, current_row, best_common = evaluate_universe(name, prepared_subset)
        portfolio_df.insert(0, "universe", name)
        group_df.insert(0, "universe", name)
        summary_df.insert(0, "coverage", len(prepared_subset))
        portfolio_parts.append(portfolio_df)
        group_parts.append(group_df)
        summary_parts.append(summary_df)
        top_rows.append({
            "universe": name,
            "coverage": len(prepared_subset),
            "current_avg_pnl": current_row["avg_pnl"],
            "current_pf": current_row["pf"],
            "best_common_scenario": best_common["scenario"],
            "best_common_avg_pnl": best_common["avg_pnl"],
            "best_common_pf": best_common["pf"],
            "delta_common_vs_current": round(best_common["avg_pnl"] - current_row["avg_pnl"], 3),
        })

    portfolio_all = pd.concat(portfolio_parts, ignore_index=True)
    group_all = pd.concat(group_parts, ignore_index=True)
    summary_all = pd.concat(summary_parts, ignore_index=True)
    top_df = pd.DataFrame(top_rows)

    base_dir = os.path.dirname(__file__)
    portfolio_path = os.path.join(base_dir, "backtest_exit_universe_portfolio.csv")
    group_path = os.path.join(base_dir, "backtest_exit_universe_group.csv")
    summary_path = os.path.join(base_dir, "backtest_exit_universe_group_summary.csv")
    top_path = os.path.join(base_dir, "backtest_exit_universe_topline.csv")

    portfolio_all.to_csv(portfolio_path, index=False, encoding="utf-8-sig")
    group_all.to_csv(group_path, index=False, encoding="utf-8-sig")
    summary_all.to_csv(summary_path, index=False, encoding="utf-8-sig")
    top_df.to_csv(top_path, index=False, encoding="utf-8-sig")

    print("\n[Topline]")
    print(top_df.to_string(index=False))
    print("\n저장 완료:")
    print(portfolio_path)
    print(group_path)
    print(summary_path)
    print(top_path)


if __name__ == "__main__":
    main()
