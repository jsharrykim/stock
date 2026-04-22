"""
backtest_golden_cross_pctb_grid.py
====================================
G4 골든크로스 전략의 종가 %B 임계값 그리드 탐색

[고정 조건]
  현재가 > MA200
  MACD 골든크로스 (전일 hist ≤ 0 → 당일 hist > 0)
  종가 %B > THRESHOLD  ← 이 값을 변화시킴

[탐색 범위]
  %B 임계값: 0, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95

[매도: AB 방식 고정]
  목표 +8% / 손절 -25% / 60일수익 / 120일만료

기간: 2015-01-01 ~ 2026-04-15 / 54개 종목
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings("ignore")

START = "2015-01-01"
END   = "2026-04-15"
EXIT  = dict(target=0.08, stop=0.25, half_days=60, max_hold=120)
PCTB_GRID = [0, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95]

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
    df  = df.copy()
    c   = df["Close"]
    ma20    = c.rolling(20).mean()
    std20   = c.rolling(20).std()
    bb_u    = ma20 + 2.0 * std20
    bb_l    = ma20 - 2.0 * std20
    bb_r    = bb_u - bb_l
    df["ma200"]     = c.rolling(200).mean()
    df["pctb_close"]= np.where(bb_r > 0, (c - bb_l) / bb_r * 100, np.nan)
    df["pctb_low"]  = np.where(bb_r > 0, (df["Low"] - bb_l) / bb_r * 100, np.nan)
    # MACD
    ema_f   = c.ewm(span=12, adjust=False).mean()
    ema_s   = c.ewm(span=26, adjust=False).mean()
    macd    = ema_f - ema_s
    sig     = macd.ewm(span=9,  adjust=False).mean()
    hist    = macd - sig
    df["macd_hist"]    = hist
    df["macd_prev"]    = hist.shift(1)
    df["golden_cross"] = (df["macd_prev"] <= 0) & (df["macd_hist"] > 0)
    # BB 스퀴즈 (REF_A용)
    bb_w    = (bb_r / ma20 * 100).where(ma20 > 0)
    df["bb_width_avg"] = bb_w.rolling(60).mean()
    df["squeeze"]      = bb_w < df["bb_width_avg"] * 0.5
    return df

def download_data():
    print(f"다운로드 중... ({len(ALL_TICKERS)}개)")
    raw = yf.download(ALL_TICKERS, start=START, end=END,
                      auto_adjust=True, progress=False, group_by="ticker")
    result = {}
    for t in ALL_TICKERS:
        try:
            df = raw[t].copy() if len(ALL_TICKERS) > 1 else raw.copy()
            df.dropna(how="all", inplace=True)
            if len(df) < 250: continue
            result[t] = calc_indicators(df)
        except: pass
    print(f"유효 종목: {len(result)}개")
    return result

def check_exit(close, entry, hold):
    pnl = (close - entry) / entry
    if pnl >= EXIT["target"]:                          return "목표", pnl
    if pnl <= -EXIT["stop"]:                           return "손절", pnl
    if hold >= EXIT["half_days"] and pnl > 0:          return "60일수익", pnl
    if hold >= EXIT["max_hold"]:                       return "기간만료", pnl
    return None, pnl

def run_scenario(data, threshold, req_cols):
    trades = []
    for ticker, df in data.items():
        dfc = df.dropna(subset=req_cols)
        if len(dfc) < 10: continue
        in_pos = False
        entry_price = entry_idx = None
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
                    above = float(row["Close"]) > float(row["ma200"])
                    gc    = bool(row["golden_cross"])
                    pc    = float(row["pctb_close"]) if not np.isnan(row["pctb_close"]) else 0
                    if above and gc and pc > threshold:
                        in_pos = True
                        entry_price = row["Close"]
                        entry_idx   = ii
                except: pass
        if in_pos and entry_price is not None:
            last = dfc.iloc[-1]
            pnl  = (last["Close"] - entry_price) / entry_price
            trades.append({"pnl_pct": round(pnl*100,2), "exit_reason": "미청산",
                            "hold_days": len(dfc)-1-entry_idx})
    return trades

def run_ref_a(data):
    """REF_A 현행 전략 기준선"""
    trades = []
    req = ["ma200","squeeze","pctb_low"]
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
    if not trades: return {"trades":0,"win_rate":0,"avg_pnl":0,"ev":0,
                           "target_pct":0,"stop_pct":0,"avg_hold":0}
    df   = pd.DataFrame(trades)
    wins = df[df["pnl_pct"]>0]; loss = df[df["pnl_pct"]<=0]
    wr   = len(wins)/len(df)
    aw   = wins["pnl_pct"].mean() if len(wins) else 0
    al   = loss["pnl_pct"].mean() if len(loss) else 0
    bx   = df["exit_reason"].value_counts(normalize=True)*100
    return {
        "trades":     len(df),
        "win_rate":   round(wr*100,1),
        "avg_pnl":    round(df["pnl_pct"].mean(),2),
        "ev":         round(wr*aw+(1-wr)*al,2),
        "avg_win":    round(aw,2),
        "avg_loss":   round(al,2),
        "target_pct": round(bx.get("목표",0),1),
        "stop_pct":   round(bx.get("손절",0),1),
        "avg_hold":   round(df["hold_days"].mean(),1),
    }

def main():
    print("=" * 70)
    print("골든크로스 + %B 임계값 그리드 탐색")
    print(f"기간: {START} ~ {END}")
    print("=" * 70)

    data = download_data()
    req  = ["ma200","pctb_close","golden_cross"]

    rows = []
    for th in PCTB_GRID:
        print(f"  %B > {th:2d} 실행 중...", end=" ")
        t  = run_scenario(data, th, req)
        s  = stats(t)
        s["threshold"] = th
        rows.append(s)
        print(f"거래:{s['trades']}  승률:{s['win_rate']}%  EV:{s['ev']}%  손절:{s['stop_pct']}%")

    # REF_A 기준선
    print("  REF_A 실행 중...", end=" ")
    ref_t = run_ref_a(data)
    ref_s = stats(ref_t)
    ref_s["threshold"] = -1
    print(f"거래:{ref_s['trades']}  승률:{ref_s['win_rate']}%  EV:{ref_s['ev']}%  손절:{ref_s['stop_pct']}%")

    df = pd.DataFrame(rows)

    # ── 콘솔 출력 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"{'%B 임계값':>10}  {'거래':>6}  {'승률':>7}  {'평균수익':>8}  {'EV':>7}  "
          f"{'avg승리':>8}  {'avg손실':>8}  {'목표%':>6}  {'손절%':>6}  {'평균보유':>8}")
    print("-" * 75)
    for _, r in df.iterrows():
        marker = " ★" if r["ev"] == df["ev"].max() else ""
        print(f"  %B > {r['threshold']:2.0f}    {r['trades']:>6}  {r['win_rate']:>6.1f}%  "
              f"{r['avg_pnl']:>7.2f}%  {r['ev']:>6.2f}%  {r['avg_win']:>7.2f}%  "
              f"{r['avg_loss']:>7.2f}%  {r['target_pct']:>5.1f}%  {r['stop_pct']:>5.1f}%  "
              f"{r['avg_hold']:>7.1f}일{marker}")
    print(f"\n  REF_A현행     {ref_s['trades']:>6}  {ref_s['win_rate']:>6.1f}%  "
          f"{ref_s['avg_pnl']:>7.2f}%  {ref_s['ev']:>6.2f}%  {ref_s['avg_win']:>7.2f}%  "
          f"{ref_s['avg_loss']:>7.2f}%  {ref_s['target_pct']:>5.1f}%  {ref_s['stop_pct']:>5.1f}%  "
          f"{ref_s['avg_hold']:>7.1f}일 ← 기준선")
    print("=" * 75)

    # ── 시각화 ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        "골든크로스 전략: 종가 %B 임계값별 성과 (AB출구 고정)\n"
        f"공통: 현재가>MA200 + MACD 골든크로스 + 종가%B > 임계값 | "
        f"매도: +{EXIT['target']*100:.0f}%/-{EXIT['stop']*100:.0f}%/{EXIT['half_days']}일",
        fontsize=11, fontweight="bold"
    )

    th_labels = [f"%B>{t}" for t in PCTB_GRID]
    ref_ev_val = ref_s["ev"]

    def line_bar(ax, y_vals, ylabel, title, ref_val=None, fmt="{:.1f}%"):
        x = range(len(th_labels))
        clr = ["#1565C0" if v < max(y_vals) else "#E53935" for v in y_vals]
        bars = ax.bar(x, y_vals, color=clr, alpha=0.85, edgecolor="white")
        ax.plot(x, y_vals, "o--", color="#333", linewidth=1.2, markersize=4, zorder=5)
        if ref_val is not None:
            ax.axhline(ref_val, color="red", linewidth=1.5, linestyle=":",
                       label=f"REF_A {ref_val:.2f}", alpha=0.8)
            ax.legend(fontsize=8)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xticks(list(x)); ax.set_xticklabels(th_labels, rotation=35, ha="right", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        vr = max(y_vals)-min(y_vals) if y_vals else 1
        for b, v in zip(bars, y_vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+(vr*0.03),
                    fmt.format(v), ha="center", va="bottom", fontsize=8)

    line_bar(axes[0,0], df["ev"].tolist(),       "EV (%)",     "★ 기대값 EV",     ref_ev_val)
    line_bar(axes[0,1], df["win_rate"].tolist(),  "승률 (%)",   "승률",             ref_s["win_rate"])
    line_bar(axes[0,2], df["avg_pnl"].tolist(),   "평균수익 (%)","평균 수익률",      ref_s["avg_pnl"])
    line_bar(axes[1,0], df["trades"].tolist(),    "거래 수",    "거래 수",           ref_s["trades"], fmt="{:.0f}건")
    line_bar(axes[1,1], df["stop_pct"].tolist(),  "손절 비율 (%)","손절 비율",        ref_s["stop_pct"])
    line_bar(axes[1,2], df["avg_hold"].tolist(),  "평균 보유일", "평균 보유일",       ref_s["avg_hold"], fmt="{:.1f}일")

    plt.tight_layout()
    out = "/Users/jungsoo.kim/Desktop/backtest/backtest_golden_pctb_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n차트 저장: {out}")

    # ── EV vs 거래수 산포도 ───────────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(10, 6))
    fig2.suptitle("EV vs 거래 수 — %B 임계값별", fontsize=12, fontweight="bold")
    sc = ax.scatter(df["trades"], df["ev"], c=PCTB_GRID, cmap="Blues",
                    s=120, edgecolors="gray", linewidth=0.8, zorder=5)
    for _, r in df.iterrows():
        ax.annotate(f"%B>{r['threshold']:.0f}",
                    (r["trades"], r["ev"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.scatter([ref_s["trades"]], [ref_s["ev"]], color="red", marker="*", s=220,
               zorder=6, label="REF_A 현행")
    ax.annotate("REF_A", (ref_s["trades"], ref_s["ev"]),
                textcoords="offset points", xytext=(6, 4), fontsize=9, color="red")
    ax.set_xlabel("거래 수 (신호 빈도)")
    ax.set_ylabel("기대값 EV (%)")
    ax.axhline(ref_ev_val, color="red", linewidth=1, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9)
    plt.colorbar(sc, ax=ax, label="%B 임계값")
    plt.tight_layout()
    out2 = "/Users/jungsoo.kim/Desktop/backtest/backtest_golden_pctb_scatter.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight"); plt.close()
    print(f"차트 저장: {out2}")

    # 최적 임계값
    best = df.loc[df["ev"].idxmax()]
    print(f"\n★ 최적 %B 임계값: {best['threshold']:.0f}")
    print(f"  EV {best['ev']}%  승률 {best['win_rate']}%  거래 {best['trades']}건")
    print(f"  REF_A EV: {ref_ev_val}%  →  "
          f"{'골든크로스가 우위' if best['ev']>ref_ev_val else 'REF가 우위'} "
          f"({abs(best['ev']-ref_ev_val):.2f}%p)")
    print("\n완료!")

if __name__ == "__main__":
    main()
