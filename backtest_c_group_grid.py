"""
backtest_c_group_grid.py
========================================
C그룹 (MACD 골든크로스 + MA200 상방) 3차원 그리드 탐색

[고정 조건]
  현재가 > MA200
  MACD 골든크로스: 전일 hist ≤ 0 → 당일 hist > 0

[탐색 차원]
  차원 1: 종가 %B 임계값  — [50, 60, 70, 75, 80, 85, 90]
  차원 2: RSI 하한선       — [None(필터없음), 50, 60, 70]
  차원 3: BB 확장 여부     — [False(필터없음), True(필요)]

  ▶ 총 7 × 4 × 2 = 56가지 조합

[출구: AB방식 고정]
  목표 +8% / 손절 -25% / 60일수익 / 120일만료

기간: 2015-01-01 ~ 2026-04-15 / 54개 종목
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))

import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings("ignore")

START = "2015-01-01"
END   = "2026-04-15"
EXIT  = dict(target=0.08, stop=0.25, half_days=60, max_hold=120)

PCTB_GRID    = [50, 60, 70, 75, 80, 85, 90]
RSI_GRID     = [None, 50, 60, 70]   # None = RSI 필터 없음
BB_EXP_GRID  = [False, True]        # True = BB 확장 조건 추가

KR_TICKERS = [
    "000660.KS","005930.KS","277810.KS","034020.KS","005380.KS",
    "012450.KS","042660.KS","042700.KQ","096770.KS","009150.KS",
    "000270.KS","247540.KQ","376900.KS","006400.KS","079550.KS",
]
US_TICKERS = [
    "HOOD","AAPL","AVGO","AMD","MSFT","GOOGL","NVDA","TSLA",
    "AMZN","MU","LRCX","ON","SNDK","ASTS","AVAV","IONQ",
    "RKLB","PLTR","CRWD","APP","SOXL","TSLL","TE","ONDS",
    "BE","PL","VRT","LITE","TER","ANET","IREN","HOOG",
    "SOLT","ETHU","NBIS","LPTH","CONL","INTC","CRDO","SKYT",
]
ALL_TICKERS = KR_TICKERS + US_TICKERS


def get_kr_font():
    for c in ["AppleGothic","NanumGothic","Malgun Gothic","DejaVu Sans"]:
        if c in {f.name for f in fm.fontManager.ttflist}: return c
    return None
KR_FONT = get_kr_font()
if KR_FONT: plt.rcParams["font.family"] = KR_FONT
plt.rcParams["axes.unicode_minus"] = False


# ── 지표 계산 ──────────────────────────────────────────────────────────────────
def calc_indicators(df):
    df    = df.copy()
    c     = df["Close"]
    h     = df["High"]
    l     = df["Low"]
    ma20  = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bb_u  = ma20 + 2.0 * std20
    bb_l  = ma20 - 2.0 * std20
    bb_r  = bb_u - bb_l

    df["ma200"]      = c.rolling(200).mean()
    df["pctb_close"] = np.where(bb_r > 0, (c - bb_l) / bb_r * 100, np.nan)
    df["pctb_low"]   = np.where(bb_r > 0, (l - bb_l) / bb_r * 100, np.nan)

    # BB 확장 (현재 BB폭 > 60일 평균 BB폭 × 80%)
    bb_w             = (bb_r / ma20 * 100).where(ma20 > 0)
    bb_w_avg         = bb_w.rolling(60).mean()
    df["bb_expanding"] = bb_w > bb_w_avg * 0.80
    df["squeeze"]    = bb_w < bb_w_avg * 0.50

    # RSI (14)
    delta   = c.diff()
    gain    = delta.clip(lower=0).rolling(14).mean()
    loss    = (-delta.clip(upper=0)).rolling(14).mean()
    rs      = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    # MACD 히스토그램
    ema_f   = c.ewm(span=12, adjust=False).mean()
    ema_s   = c.ewm(span=26, adjust=False).mean()
    macd    = ema_f - ema_s
    sig     = macd.ewm(span=9, adjust=False).mean()
    hist    = macd - sig
    df["macd_hist"]    = hist
    df["macd_prev"]    = hist.shift(1)
    df["golden_cross"] = (df["macd_prev"] <= 0) & (df["macd_hist"] > 0)

    return df


def download_data():
    print(f"데이터 다운로드 중... ({len(ALL_TICKERS)}개 종목)")
    raw = yf.download(ALL_TICKERS, start=START, end=END,
                      auto_adjust=True, progress=False, group_by="ticker")
    result = {}
    for t in ALL_TICKERS:
        try:
            df = raw[t].copy() if len(ALL_TICKERS) > 1 else raw.copy()
            df.dropna(how="all", inplace=True)
            if len(df) < 250: continue
            result[t] = calc_indicators(df)
        except:
            pass
    print(f"유효 종목: {len(result)}개\n")
    return result


def check_exit(close, entry, hold):
    pnl = (close - entry) / entry
    if pnl >= EXIT["target"]:                 return "목표", pnl
    if pnl <= -EXIT["stop"]:                  return "손절", pnl
    if hold >= EXIT["half_days"] and pnl > 0: return "60일수익", pnl
    if hold >= EXIT["max_hold"]:              return "기간만료", pnl
    return None, pnl


def run_scenario(data, pctb_th, rsi_th, require_bb_exp):
    """C그룹 조건 조합 하나를 백테스트"""
    req_base = ["ma200", "pctb_close", "golden_cross", "rsi", "bb_expanding"]
    trades = []
    for ticker, df in data.items():
        dfc = df.dropna(subset=req_base)
        if len(dfc) < 10: continue
        in_pos = False
        entry_price = entry_idx = None
        for ii, date in enumerate(dfc.index):
            row = dfc.loc[date]
            if in_pos:
                reason, pnl = check_exit(row["Close"], entry_price, ii - entry_idx)
                if reason:
                    trades.append({"pnl_pct": round(pnl * 100, 2),
                                   "exit_reason": reason,
                                   "hold_days": ii - entry_idx})
                    in_pos = False
            if not in_pos:
                try:
                    above = float(row["Close"]) > float(row["ma200"])
                    gc    = bool(row["golden_cross"])
                    pc    = float(row["pctb_close"]) if not np.isnan(row["pctb_close"]) else -999
                    rsi   = float(row["rsi"])        if not np.isnan(row["rsi"])        else 0
                    bb_ok = bool(row["bb_expanding"])

                    cond = above and gc and (pc > pctb_th)
                    if rsi_th is not None:
                        cond = cond and (rsi > rsi_th)
                    if require_bb_exp:
                        cond = cond and bb_ok

                    if cond:
                        in_pos      = True
                        entry_price = row["Close"]
                        entry_idx   = ii
                except:
                    pass
        if in_pos and entry_price is not None:
            last = dfc.iloc[-1]
            pnl  = (last["Close"] - entry_price) / entry_price
            trades.append({"pnl_pct": round(pnl * 100, 2),
                           "exit_reason": "미청산",
                           "hold_days": len(dfc) - 1 - entry_idx})
    return trades


def run_ref_a(data):
    req = ["ma200", "squeeze", "pctb_low"]
    trades = []
    for ticker, df in data.items():
        dfc = df.dropna(subset=req)
        if len(dfc) < 10: continue
        in_pos = False; entry_price = entry_idx = None
        for ii, date in enumerate(dfc.index):
            row = dfc.loc[date]
            if in_pos:
                reason, pnl = check_exit(row["Close"], entry_price, ii - entry_idx)
                if reason:
                    trades.append({"pnl_pct": round(pnl*100,2), "exit_reason": reason,
                                   "hold_days": ii - entry_idx})
                    in_pos = False
            if not in_pos:
                try:
                    if (float(row["Close"]) > float(row["ma200"])
                            and bool(row["squeeze"])
                            and float(row["pctb_low"]) <= 50):
                        in_pos = True; entry_price = row["Close"]; entry_idx = ii
                except: pass
        if in_pos and entry_price is not None:
            last = dfc.iloc[-1]
            pnl  = (last["Close"] - entry_price) / entry_price
            trades.append({"pnl_pct": round(pnl*100,2), "exit_reason": "미청산",
                           "hold_days": len(dfc)-1-entry_idx})
    return trades


def stats(trades):
    if not trades:
        return {"trades":0,"win_rate":0,"avg_pnl":0,"ev":0,
                "avg_win":0,"avg_loss":0,"target_pct":0,"stop_pct":0,"avg_hold":0}
    df   = pd.DataFrame(trades)
    wins = df[df["pnl_pct"] > 0]
    loss = df[df["pnl_pct"] <= 0]
    wr   = len(wins) / len(df)
    aw   = wins["pnl_pct"].mean() if len(wins) else 0
    al   = loss["pnl_pct"].mean() if len(loss) else 0
    bx   = df["exit_reason"].value_counts(normalize=True) * 100
    return {
        "trades":     len(df),
        "win_rate":   round(wr * 100, 1),
        "avg_pnl":    round(df["pnl_pct"].mean(), 2),
        "ev":         round(wr * aw + (1 - wr) * al, 2),
        "avg_win":    round(aw, 2),
        "avg_loss":   round(al, 2),
        "target_pct": round(bx.get("목표", 0), 1),
        "stop_pct":   round(bx.get("손절", 0), 1),
        "avg_hold":   round(df["hold_days"].mean(), 1),
    }


def label(rsi_th, bb_exp):
    r = f"RSI>{rsi_th}" if rsi_th else "RSI무관"
    b = "+BB확장" if bb_exp else ""
    return f"{r}{b}"


def main():
    print("=" * 80)
    print("C그룹 3차원 그리드 탐색: %B × RSI × BB확장")
    print(f"기간: {START} ~ {END}  |  종목: {len(ALL_TICKERS)}개")
    print(f"출구: 목표+{EXIT['target']*100:.0f}% / 손절-{EXIT['stop']*100:.0f}% / "
          f"{EXIT['half_days']}일수익 / {EXIT['max_hold']}일만료")
    print("=" * 80)

    data = download_data()

    # ── REF_A 기준선 ────────────────────────────────────────────────────────────
    print("REF_A (A그룹 현행) 실행 중...")
    ref_s = stats(run_ref_a(data))
    print(f"  → 거래:{ref_s['trades']}  승률:{ref_s['win_rate']}%  "
          f"EV:{ref_s['ev']}%  손절:{ref_s['stop_pct']}%\n")

    # ── 그리드 탐색 ─────────────────────────────────────────────────────────────
    combos = list(itertools.product(PCTB_GRID, RSI_GRID, BB_EXP_GRID))
    print(f"총 {len(combos)}개 조합 실행 중...\n")

    rows = []
    for i, (pctb_th, rsi_th, bb_exp) in enumerate(combos):
        lbl = label(rsi_th, bb_exp)
        tag = f"%B>{pctb_th} {lbl}"
        t   = run_scenario(data, pctb_th, rsi_th, bb_exp)
        s   = stats(t)
        s.update({"pctb_th": pctb_th, "rsi_th": rsi_th,
                  "bb_exp": bb_exp, "label": lbl, "tag": tag})
        rows.append(s)
        if (i + 1) % 14 == 0 or i == 0:
            print(f"  진행: {i+1}/{len(combos)}")

    df_all = pd.DataFrame(rows)

    # ── 콘솔 결과 출력 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"{'%B임계':>6}  {'조건':>22}  {'거래':>6}  {'승률':>7}  "
          f"{'평균수익':>8}  {'EV':>7}  {'avg승':>7}  {'avg패':>7}  "
          f"{'목표%':>6}  {'손절%':>6}  {'평균보유':>8}")
    print("-" * 100)

    df_sorted = df_all.sort_values("ev", ascending=False)
    for _, r in df_sorted.iterrows():
        marker = " ★" if r["ev"] == df_all["ev"].max() else ""
        print(f"  {r['pctb_th']:>3}    {r['label']:>22}  "
              f"{r['trades']:>6}  {r['win_rate']:>6.1f}%  "
              f"{r['avg_pnl']:>7.2f}%  {r['ev']:>6.2f}%  "
              f"{r['avg_win']:>7.2f}%  {r['avg_loss']:>7.2f}%  "
              f"{r['target_pct']:>5.1f}%  {r['stop_pct']:>5.1f}%  "
              f"{r['avg_hold']:>7.1f}일{marker}")

    print("-" * 100)
    print(f"  REF_A현행  {'A그룹기준선':>22}  "
          f"{ref_s['trades']:>6}  {ref_s['win_rate']:>6.1f}%  "
          f"{ref_s['avg_pnl']:>7.2f}%  {ref_s['ev']:>6.2f}%  "
          f"{ref_s['avg_win']:>7.2f}%  {ref_s['avg_loss']:>7.2f}%  "
          f"{ref_s['target_pct']:>5.1f}%  {ref_s['stop_pct']:>5.1f}%  "
          f"{ref_s['avg_hold']:>7.1f}일 ← 기준선")
    print("=" * 100)

    # ── TOP 10 요약 ─────────────────────────────────────────────────────────────
    print("\n■ EV 상위 10개 조합")
    print(f"  {'순위':>4}  {'%B':>4}  {'조건':>22}  {'거래':>6}  {'승률':>7}  {'EV':>7}  {'손절%':>6}")
    print("  " + "-" * 60)
    for rank, (_, r) in enumerate(df_sorted.head(10).iterrows(), 1):
        flag = " ↑A" if r["ev"] > ref_s["ev"] else ""
        print(f"  {rank:>4}  {r['pctb_th']:>4}  {r['label']:>22}  "
              f"{r['trades']:>6}  {r['win_rate']:>6.1f}%  {r['ev']:>6.2f}%  "
              f"{r['stop_pct']:>5.1f}%{flag}")

    # ── 시각화 ──────────────────────────────────────────────────────────────────
    # 1) BB확장 없음 / 있음 두 개 히트맵 (EV 기준)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"C그룹 3차원 그리드 — EV 히트맵 (%B × RSI 필터)\n"
        f"공통: 현재가>MA200 + MACD 골든크로스  |  "
        f"출구: +{EXIT['target']*100:.0f}%/-{EXIT['stop']*100:.0f}%/{EXIT['half_days']}일  |  "
        f"REF_A EV: {ref_s['ev']}%",
        fontsize=11, fontweight="bold"
    )

    rsi_labels = ["RSI무관", "RSI>50", "RSI>60", "RSI>70"]

    for ax, bb_exp in zip(axes, [False, True]):
        sub = df_all[df_all["bb_exp"] == bb_exp]
        matrix = np.zeros((len(RSI_GRID), len(PCTB_GRID)))
        for i, rsi_th in enumerate(RSI_GRID):
            for j, pctb_th in enumerate(PCTB_GRID):
                row = sub[(sub["rsi_th"].isna() if rsi_th is None else sub["rsi_th"] == rsi_th)
                          & (sub["pctb_th"] == pctb_th)]
                if len(row):
                    matrix[i, j] = row.iloc[0]["ev"]

        vmax = max(df_all["ev"].max(), ref_s["ev"]) + 0.3
        vmin = df_all["ev"].min() - 0.3
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                       vmin=vmin, vmax=vmax, origin="upper")

        ax.set_xticks(range(len(PCTB_GRID)))
        ax.set_xticklabels([f"%B>{t}" for t in PCTB_GRID], rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(RSI_GRID)))
        ax.set_yticklabels(rsi_labels, fontsize=9)
        ax.set_title(f"{'BB 확장 조건 추가' if bb_exp else 'BB 확장 조건 없음'}",
                     fontweight="bold", fontsize=11)

        for i in range(len(RSI_GRID)):
            for j in range(len(PCTB_GRID)):
                v = matrix[i, j]
                row = sub[(sub["rsi_th"].isna() if RSI_GRID[i] is None
                           else sub["rsi_th"] == RSI_GRID[i])
                          & (sub["pctb_th"] == PCTB_GRID[j])]
                n = row.iloc[0]["trades"] if len(row) else 0
                color = "white" if abs(v - (vmax + vmin) / 2) > (vmax - vmin) * 0.25 else "black"
                ax.text(j, i, f"{v:.2f}%\n({n}건)", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

        ax.axhline(-0.5, color="white", lw=2)
        plt.colorbar(im, ax=ax, shrink=0.85, label="EV (%)")

    plt.tight_layout()
    out_heatmap = "/Users/jungsoo.kim/Desktop/backtest/backtest_c_group_heatmap.png"
    plt.savefig(out_heatmap, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n차트 저장: {out_heatmap}")

    # 2) 승률 히트맵
    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 7))
    fig2.suptitle(
        f"C그룹 3차원 그리드 — 승률 히트맵 (%B × RSI 필터)\n"
        f"REF_A 승률: {ref_s['win_rate']}%",
        fontsize=11, fontweight="bold"
    )
    for ax, bb_exp in zip(axes2, [False, True]):
        sub = df_all[df_all["bb_exp"] == bb_exp]
        matrix = np.zeros((len(RSI_GRID), len(PCTB_GRID)))
        for i, rsi_th in enumerate(RSI_GRID):
            for j, pctb_th in enumerate(PCTB_GRID):
                row = sub[(sub["rsi_th"].isna() if rsi_th is None else sub["rsi_th"] == rsi_th)
                          & (sub["pctb_th"] == pctb_th)]
                if len(row):
                    matrix[i, j] = row.iloc[0]["win_rate"]
        vmax_wr = max(df_all["win_rate"].max(), ref_s["win_rate"]) + 1
        vmin_wr = df_all["win_rate"].min() - 1
        im2 = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                        vmin=vmin_wr, vmax=vmax_wr, origin="upper")
        ax.set_xticks(range(len(PCTB_GRID)))
        ax.set_xticklabels([f"%B>{t}" for t in PCTB_GRID], rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(RSI_GRID)))
        ax.set_yticklabels(rsi_labels, fontsize=9)
        ax.set_title(f"{'BB 확장 조건 추가' if bb_exp else 'BB 확장 조건 없음'}",
                     fontweight="bold", fontsize=11)
        for i in range(len(RSI_GRID)):
            for j in range(len(PCTB_GRID)):
                v = matrix[i, j]
                color = "white" if abs(v - (vmax_wr + vmin_wr) / 2) > (vmax_wr - vmin_wr) * 0.25 else "black"
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")
        plt.colorbar(im2, ax=ax, shrink=0.85, label="승률 (%)")

    plt.tight_layout()
    out_wr = "/Users/jungsoo.kim/Desktop/backtest/backtest_c_group_winrate.png"
    plt.savefig(out_wr, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"차트 저장: {out_wr}")

    # ── 최적 조합 ───────────────────────────────────────────────────────────────
    best = df_all.loc[df_all["ev"].idxmax()]
    print("\n" + "=" * 60)
    print("★ 최적 조합 (EV 기준)")
    print(f"  %B 임계값 : {best['pctb_th']}")
    print(f"  RSI 필터  : {best['rsi_th'] if best['rsi_th'] else '없음'}")
    print(f"  BB 확장   : {'필요' if best['bb_exp'] else '없음'}")
    print(f"  EV        : {best['ev']}%")
    print(f"  승률      : {best['win_rate']}%")
    print(f"  거래 수   : {best['trades']}건")
    print(f"  손절 비율 : {best['stop_pct']}%")
    print(f"  REF_A EV  : {ref_s['ev']}%  →  "
          f"{'C그룹이 우위' if best['ev'] > ref_s['ev'] else 'REF_A가 우위'} "
          f"({abs(best['ev'] - ref_s['ev']):.2f}%p)")
    print("=" * 60)
    print("\n완료!")


if __name__ == "__main__":
    main()
