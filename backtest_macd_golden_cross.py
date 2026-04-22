"""
backtest_macd_golden_cross.py
==============================
MACD 골든크로스 진입 전략 — 조합별 비교 + D5/REF_A 기준선

[골든크로스 정의]
  전일 MACD hist ≤ 0  AND  당일 MACD hist > 0
  = MACD 라인이 시그널 라인을 아래에서 위로 교차하는 그 순간

[공통 베이스 조건]
  ① 현재가 > MA200
  ② MACD 골든크로스 발생 (hist: 음→양 전환)

[조합별 추가 필터]
  G1: 베이스만 (필터 없음)
  G2: RSI > 50                          (모멘텀 방향 확인)
  G3: 현재가 기준 %B > 50               (BB 중간선 위 — 가격 위치 확인)
  G4: 현재가 기준 %B > 80               (BB 상단 근처 — 돌파 직전)

[매도 방식 2종 교차 테스트]
  EXIT_AB: 목표 +8%  / 손절 -25% / 60일수익 / 120일만료  (현행 A/B 방식)
  EXIT_D : 목표 +15% / 손절 -15% / 30일수익 /  60일만료  (D그룹 방식)

[비교 기준선]
  REF_A : 현행 A그룹 (스퀴즈 + 저가%B ≤ 50, EXIT_AB)
  D5    : 현행 D그룹 (BB상단 + hist>0 + RSI≤80, EXIT_D)

종목: 기존 풀 54개
기간: 2015-01-01 ~ 2026-04-15
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

# ── 기간 ──────────────────────────────────────────────────────────────────────
START = "2015-01-01"
END   = "2026-04-15"

# ── 매도 파라미터 ──────────────────────────────────────────────────────────────
EXIT_AB = dict(target=0.08, stop=0.25, half_days=60, max_hold=120)
EXIT_D  = dict(target=0.15, stop=0.15, half_days=30, max_hold=60)

# ── 지표 파라미터 ──────────────────────────────────────────────────────────────
BB_PERIOD       = 20
BB_STD          = 2.0
BB_AVG_PERIOD   = 60
SQUEEZE_RATIO   = 0.50
PCTB_LOW_A      = 50
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9

# ── 종목 ──────────────────────────────────────────────────────────────────────
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

# ── 폰트 ──────────────────────────────────────────────────────────────────────
def get_kr_font():
    for c in ["AppleGothic","NanumGothic","Malgun Gothic","DejaVu Sans"]:
        if c in {f.name for f in fm.fontManager.ttflist}:
            return c
    return None
KR_FONT = get_kr_font()
if KR_FONT:
    plt.rcParams["font.family"] = KR_FONT
plt.rcParams["axes.unicode_minus"] = False

# ── 지표 계산 ──────────────────────────────────────────────────────────────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy()
    c, h, l = df["Close"], df["High"], df["Low"]

    df["ma200"]        = c.rolling(200).mean()
    ma20               = c.rolling(BB_PERIOD).mean()
    std20              = c.rolling(BB_PERIOD).std()
    bb_upper           = ma20 + BB_STD * std20
    bb_lower           = ma20 - BB_STD * std20
    bb_range           = bb_upper - bb_lower
    df["ma20"]         = ma20
    df["bb_upper"]     = bb_upper
    df["bb_lower"]     = bb_lower
    df["bb_width"]     = (bb_range / ma20 * 100).where(ma20 > 0)
    df["bb_width_avg"] = df["bb_width"].rolling(BB_AVG_PERIOD).mean()
    df["squeeze"]      = df["bb_width"] < df["bb_width_avg"] * SQUEEZE_RATIO

    # 저가 %B (REF_A용)
    df["pctb_low"]  = np.where(bb_range > 0, (l - bb_lower) / bb_range * 100, np.nan)
    # 현재가(종가) %B (G그룹용)
    df["pctb_close"] = np.where(bb_range > 0, (c - bb_lower) / bb_range * 100, np.nan)
    # 고가 %B (D5용)
    df["pctb_high"] = np.where(bb_range > 0, (h - bb_lower) / bb_range * 100, np.nan)

    # BB 확장 (D5용)
    df["bb_expanding"] = df["bb_width"] > df["bb_width_avg"] * 0.80

    # RSI(14)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + np.where(loss == 0, 100.0, gain / loss))

    # MACD(12,26,9)
    ema_f            = c.ewm(span=MACD_FAST,  adjust=False).mean()
    ema_s            = c.ewm(span=MACD_SLOW,  adjust=False).mean()
    macd_line        = ema_f - ema_s
    signal_line      = macd_line.ewm(span=MACD_SIG, adjust=False).mean()
    df["macd_hist"]  = macd_line - signal_line
    df["macd_prev"]  = df["macd_hist"].shift(1)

    # 골든크로스 여부: 전일 hist ≤ 0  AND  당일 hist > 0
    df["golden_cross"] = (df["macd_prev"] <= 0) & (df["macd_hist"] > 0)

    return df

# ── 데이터 다운로드 ────────────────────────────────────────────────────────────
def download_data():
    print(f"데이터 다운로드 중... ({len(ALL_TICKERS)}개)")
    raw = yf.download(ALL_TICKERS, start=START, end=END,
                      auto_adjust=True, progress=False, group_by="ticker")
    result = {}
    for t in ALL_TICKERS:
        try:
            df = raw[t].copy() if len(ALL_TICKERS) > 1 else raw.copy()
            df.dropna(how="all", inplace=True)
            if len(df) < 250:
                print(f"  [{t}] 데이터 부족 - 제외")
                continue
            result[t] = calc_indicators(df)
            print(f"  [{t}] {len(df)}일")
        except Exception as e:
            print(f"  [{t}] 오류: {e}")
    return result

# ── 매도 체크 ─────────────────────────────────────────────────────────────────
def check_exit(close, entry, hold, ep):
    pnl = (close - entry) / entry
    if pnl >= ep["target"]:             return "목표", pnl
    if pnl <= -ep["stop"]:              return "손절", pnl
    if hold >= ep["half_days"] and pnl > 0: return f"{ep['half_days']}일수익", pnl
    if hold >= ep["max_hold"]:          return "기간만료", pnl
    return None, pnl

# ── 백테스트 엔진 ─────────────────────────────────────────────────────────────
def run_backtest(data: dict, sc: dict) -> list:
    cond_fn  = sc["fn"]
    ep       = sc["exit"]
    req      = sc.get("req", ["ma200","macd_hist","macd_prev","golden_cross",
                               "pctb_close","pctb_low","pctb_high",
                               "rsi","squeeze","bb_expanding"])
    trades = []
    for ticker, df in data.items():
        dfc = df.dropna(subset=req)
        if len(dfc) < 10:
            continue
        in_pos = False
        entry_price = entry_date = entry_idx = None
        idx_list = list(dfc.index)

        for ii, date in enumerate(idx_list):
            row = dfc.loc[date]
            if in_pos:
                hold = ii - entry_idx
                reason, pnl = check_exit(row["Close"], entry_price, hold, ep)
                if reason:
                    trades.append({
                        "scenario": sc["name"], "ticker": ticker,
                        "entry_date": entry_date, "exit_date": date,
                        "entry_price": round(float(entry_price), 4),
                        "exit_price":  round(float(row["Close"]), 4),
                        "pnl_pct": round(pnl * 100, 2),
                        "hold_days": hold, "exit_reason": reason,
                    })
                    in_pos = False

            if not in_pos:
                def v(col):
                    val = row[col]
                    return float(val) if not (isinstance(val, float) and np.isnan(val)) else None

                sig = {
                    "above200":     float(row["Close"]) > float(row["ma200"]),
                    "golden_cross": bool(row["golden_cross"]),
                    "pctb_close":   v("pctb_close") or 0,
                    "pctb_low":     v("pctb_low")   or 999,
                    "pctb_high":    v("pctb_high")  or -999,
                    "rsi":          v("rsi")         or 0,
                    "squeeze":      bool(row["squeeze"]),
                    "bb_expanding": bool(row["bb_expanding"]),
                    "macd_hist":    float(row["macd_hist"]),
                    "hist_rising":  float(row["macd_hist"]) > float(row["macd_prev"]),
                }
                try:
                    if cond_fn(sig):
                        in_pos = True
                        entry_price = row["Close"]
                        entry_date  = date
                        entry_idx   = ii
                except Exception:
                    pass

        if in_pos and entry_price is not None:
            last = dfc.iloc[-1]
            pnl  = (last["Close"] - entry_price) / entry_price
            trades.append({
                "scenario": sc["name"], "ticker": ticker,
                "entry_date": entry_date, "exit_date": dfc.index[-1],
                "entry_price": round(float(entry_price), 4),
                "exit_price":  round(float(last["Close"]), 4),
                "pnl_pct": round(pnl * 100, 2),
                "hold_days": len(dfc) - 1 - entry_idx,
                "exit_reason": "미청산",
            })
    return trades

# ── 시나리오 ──────────────────────────────────────────────────────────────────
SCENARIOS = [
    # ── G그룹: 골든크로스 진입 × EXIT_AB ─────────────────────────────────────
    {
        "name": "G1_골든크로스_AB출구",
        "exit": EXIT_AB,
        "fn":   lambda s: s["above200"] and s["golden_cross"],
    },
    {
        "name": "G2_골든+RSI50_AB출구",
        "exit": EXIT_AB,
        "fn":   lambda s: s["above200"] and s["golden_cross"] and s["rsi"] > 50,
    },
    {
        "name": "G3_골든+%B50_AB출구",
        "exit": EXIT_AB,
        "fn":   lambda s: s["above200"] and s["golden_cross"] and s["pctb_close"] > 50,
    },
    {
        "name": "G4_골든+%B80_AB출구",
        "exit": EXIT_AB,
        "fn":   lambda s: s["above200"] and s["golden_cross"] and s["pctb_close"] > 80,
    },
    # ── G그룹: 골든크로스 진입 × EXIT_D ──────────────────────────────────────
    {
        "name": "G1_골든크로스_D출구",
        "exit": EXIT_D,
        "fn":   lambda s: s["above200"] and s["golden_cross"],
    },
    {
        "name": "G2_골든+RSI50_D출구",
        "exit": EXIT_D,
        "fn":   lambda s: s["above200"] and s["golden_cross"] and s["rsi"] > 50,
    },
    {
        "name": "G3_골든+%B50_D출구",
        "exit": EXIT_D,
        "fn":   lambda s: s["above200"] and s["golden_cross"] and s["pctb_close"] > 50,
    },
    {
        "name": "G4_골든+%B80_D출구",
        "exit": EXIT_D,
        "fn":   lambda s: s["above200"] and s["golden_cross"] and s["pctb_close"] > 80,
    },
    # ── 기준선 ────────────────────────────────────────────────────────────────
    {
        "name": "REF_A현행전략",
        "exit": EXIT_AB,
        "req":  ["ma200","bb_width_avg","squeeze","pctb_low"],
        "fn":   lambda s: s["above200"] and s["pctb_low"] <= 50 and s["squeeze"],
    },
    {
        "name": "D5_현행D그룹",
        "exit": EXIT_D,
        "fn":   lambda s: (
            s["above200"]
            and s["pctb_high"] >= 95
            and s["bb_expanding"]
            and s["macd_hist"] > 0
            and s["rsi"] <= 80
        ),
    },
]

# ── 분석 ──────────────────────────────────────────────────────────────────────
def analyze(trades: list, sc: dict) -> dict:
    name = sc["name"]
    ep   = sc["exit"]
    if not trades:
        return {"scenario": name, "trades": 0, "win_rate": 0, "avg_pnl": 0,
                "median_pnl": 0, "ev": 0, "avg_win": 0, "avg_loss": 0,
                "target_pct": 0, "stop_pct_r": 0, "half_pct": 0,
                "expire_pct": 0, "avg_hold": 0,
                "target_cfg": ep["target"]*100, "stop_cfg": ep["stop"]*100}
    df   = pd.DataFrame(trades)
    wins = df[df["pnl_pct"] > 0]
    loss = df[df["pnl_pct"] <= 0]
    wr   = len(wins) / len(df)
    aw   = wins["pnl_pct"].mean() if len(wins) else 0
    al   = loss["pnl_pct"].mean() if len(loss) else 0
    ev   = round(wr * aw + (1 - wr) * al, 2)
    bx   = df["exit_reason"].value_counts(normalize=True) * 100
    hk   = next((k for k in bx.index if "일수익" in k), None)
    return {
        "scenario":   name,
        "trades":     len(df),
        "win_rate":   round(wr * 100, 1),
        "avg_pnl":    round(df["pnl_pct"].mean(), 2),
        "median_pnl": round(df["pnl_pct"].median(), 2),
        "ev":         ev,
        "avg_win":    round(aw, 2),
        "avg_loss":   round(al, 2),
        "target_pct": round(bx.get("목표", 0), 1),
        "stop_pct_r": round(bx.get("손절", 0), 1),
        "half_pct":   round(bx.get(hk, 0) if hk else 0, 1),
        "expire_pct": round(bx.get("기간만료", 0), 1),
        "avg_hold":   round(df["hold_days"].mean(), 1),
        "target_cfg": ep["target"] * 100,
        "stop_cfg":   ep["stop"] * 100,
    }

# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot(summary_df: pd.DataFrame, all_trades: list):
    g_ab  = summary_df[summary_df["scenario"].str.contains("AB출구")]
    g_d   = summary_df[summary_df["scenario"].str.contains("D출구")]
    refs  = summary_df[summary_df["scenario"].str.startswith(("REF","D5"))]

    c_ab  = ["#1565C0","#1976D2","#42A5F5","#90CAF9"]
    c_d   = ["#6A1B9A","#8E24AA","#BA68C8","#E1BEE7"]
    c_ref = ["#B71C1C","#E65100"]

    fig, axes = plt.subplots(2, 4, figsize=(26, 13))
    fig.suptitle(
        "MACD 골든크로스 진입 전략 — G1~G4 조합 × AB/D 출구 비교\n"
        f"AB출구: 목표+{EXIT_AB['target']*100:.0f}%/-{EXIT_AB['stop']*100:.0f}%/{EXIT_AB['half_days']}일  │  "
        f"D출구: 목표+{EXIT_D['target']*100:.0f}%/-{EXIT_D['stop']*100:.0f}%/{EXIT_D['half_days']}일",
        fontsize=11, fontweight="bold"
    )

    def bar(ax, df, colors, col, title, fmt="{:.1f}%"):
        vals = df[col].tolist()
        lbls = df["scenario"].str.replace("_AB출구","").str.replace("_D출구","").tolist()
        brs  = ax.bar(range(len(lbls)), vals, color=colors[:len(lbls)],
                      alpha=0.85, edgecolor="white")
        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.set_xticks(range(len(lbls)))
        ax.set_xticklabels(lbls, rotation=38, ha="right", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        vr = max(vals)-min(vals) if vals else 1
        for b, v in zip(brs, vals):
            ax.text(b.get_x()+b.get_width()/2,
                    b.get_height()+(vr*0.03 if v>=0 else -vr*0.07),
                    fmt.format(v), ha="center", va="bottom", fontsize=7.5)

    # 상단 행: AB 출구 (G1~G4 + REF_A)
    ab_comb = pd.concat([g_ab, refs[refs["scenario"]=="REF_A현행전략"]], ignore_index=True)
    ab_col  = c_ab + [c_ref[0]]
    bar(axes[0,0], ab_comb, ab_col, "avg_pnl",  "AB출구 — 평균수익 (%)")
    bar(axes[0,1], ab_comb, ab_col, "win_rate", "AB출구 — 승률 (%)")
    bar(axes[0,2], ab_comb, ab_col, "ev",       "AB출구 — EV (%)")
    bar(axes[0,3], ab_comb, ab_col, "trades",   "AB출구 — 거래 수", fmt="{:.0f}건")

    # 하단 행: D 출구 (G1~G4 + D5)
    d_comb = pd.concat([g_d, refs[refs["scenario"]=="D5_현행D그룹"]], ignore_index=True)
    d_col  = c_d + [c_ref[1]]
    bar(axes[1,0], d_comb, d_col, "avg_pnl",  "D출구 — 평균수익 (%)")
    bar(axes[1,1], d_comb, d_col, "win_rate", "D출구 — 승률 (%)")
    bar(axes[1,2], d_comb, d_col, "ev",       "D출구 — EV (%)")
    bar(axes[1,3], d_comb, d_col, "trades",   "D출구 — 거래 수", fmt="{:.0f}건")

    plt.tight_layout()
    p1 = "/Users/jungsoo.kim/Desktop/backtest/backtest_golden_cross_main.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()
    print(f"차트 저장: {p1}")

    # ── EV 전체 비교 한눈에 ────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(20, 7))
    fig2.suptitle("EV 전체 비교 — 골든크로스 G1~G4 × 출구 방식 + 기준선", fontsize=12, fontweight="bold")

    all_rows = pd.concat([g_ab, g_d, refs], ignore_index=True)
    ev_vals  = all_rows["ev"].tolist()
    lbls     = all_rows["scenario"].tolist()
    colors   = (c_ab + c_d + c_ref)[:len(all_rows)]
    brs = ax2.bar(range(len(lbls)), ev_vals, color=colors, alpha=0.85, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.7, linestyle="--")

    # REF/D5 기준선 음영
    ref_ev = refs["ev"].max() if len(refs) else 0
    ax2.axhline(ref_ev, color="red", linewidth=1.5, linestyle=":",
                label=f"기준선 최고 EV {ref_ev:.2f}%")

    for b, v in zip(brs, ev_vals):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                 f"{v:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax2.set_xticks(range(len(lbls)))
    ax2.set_xticklabels(lbls, rotation=38, ha="right", fontsize=9)
    ax2.set_ylabel("기대값 EV (%)")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    p2 = "/Users/jungsoo.kim/Desktop/backtest/backtest_golden_cross_ev.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
    print(f"차트 저장: {p2}")

    # ── 매도 사유 비율 ─────────────────────────────────────────────────────────
    fig3, axes3 = plt.subplots(1, 3, figsize=(24, 7))
    fig3.suptitle("매도 사유 비율", fontsize=12, fontweight="bold")
    for ax, df, colors, title in [
        (axes3[0], ab_comb, ab_col, "골든크로스 × AB출구"),
        (axes3[1], d_comb,  d_col,  "골든크로스 × D출구"),
        (axes3[2], refs,    c_ref,  "기준선 (REF_A / D5)"),
    ]:
        lbls = df["scenario"].str.replace("_AB출구","").str.replace("_D출구","").tolist()
        x    = range(len(lbls))
        tv   = df["target_pct"].tolist()
        sv   = df["stop_pct_r"].tolist()
        hv   = df["half_pct"].tolist()
        ev_  = df["expire_pct"].tolist()
        ax.bar(x, tv, label="목표달성", color="#4CAF50", alpha=0.85)
        ax.bar(x, sv, bottom=tv, label="손절", color="#F44336", alpha=0.85)
        b2 = [a+b for a,b in zip(tv,sv)]
        ax.bar(x, hv, bottom=b2, label="시간수익", color="#2196F3", alpha=0.85)
        b3 = [a+b for a,b in zip(b2,hv)]
        ax.bar(x, ev_, bottom=b3, label="기간만료", color="#FF9800", alpha=0.85)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xticks(list(x)); ax.set_xticklabels(lbls, rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=8); ax.set_ylim(0,105); ax.set_ylabel("비율 (%)")
    plt.tight_layout()
    p3 = "/Users/jungsoo.kim/Desktop/backtest/backtest_golden_cross_exit.png"
    plt.savefig(p3, dpi=150, bbox_inches="tight"); plt.close()
    print(f"차트 저장: {p3}")

    # ── 수익률 분포 박스플롯 ───────────────────────────────────────────────────
    all_df   = pd.DataFrame(all_trades)
    all_rows2 = pd.concat([g_ab, g_d, refs], ignore_index=True)
    sc_list  = all_rows2["scenario"].tolist()
    data_bp  = [all_df[all_df["scenario"]==s]["pnl_pct"].tolist() or [0] for s in sc_list]
    all_col2 = (c_ab + c_d + c_ref)[:len(sc_list)]

    fig4, ax4 = plt.subplots(figsize=(24, 8))
    fig4.suptitle("시나리오별 수익률 분포", fontsize=12, fontweight="bold")
    bp = ax4.boxplot(data_bp, patch_artist=True)
    for patch, col in zip(bp["boxes"], all_col2):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax4.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax4.axhline(EXIT_AB["target"]*100, color="#4CAF50", linewidth=1, linestyle=":",
                label=f"AB목표 +{EXIT_AB['target']*100:.0f}%")
    ax4.axhline(-EXIT_AB["stop"]*100,  color="#EF9A9A", linewidth=1, linestyle=":",
                label=f"AB손절 -{EXIT_AB['stop']*100:.0f}%")
    ax4.axhline(EXIT_D["target"]*100,  color="#81C784", linewidth=1, linestyle="-.",
                label=f"D목표 +{EXIT_D['target']*100:.0f}%")
    ax4.axhline(-EXIT_D["stop"]*100,   color="#F44336", linewidth=1, linestyle="-.",
                label=f"D손절 -{EXIT_D['stop']*100:.0f}%")
    ax4.set_xticks(range(1, len(sc_list)+1))
    ax4.set_xticklabels(sc_list, rotation=38, ha="right", fontsize=8.5)
    ax4.legend(fontsize=9); ax4.set_ylabel("수익률 (%)")
    plt.tight_layout()
    p4 = "/Users/jungsoo.kim/Desktop/backtest/backtest_golden_cross_dist.png"
    plt.savefig(p4, dpi=150, bbox_inches="tight"); plt.close()
    print(f"차트 저장: {p4}")

# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 82)
    print("MACD 골든크로스 진입 전략 백테스트")
    print(f"기간: {START} ~ {END}")
    print("=" * 82)

    data = download_data()
    print(f"\n유효 종목: {len(data)}개\n")

    all_trades, summary_rows = [], []
    for sc in SCENARIOS:
        print(f"[{sc['name']}] 실행 중...")
        trades = run_backtest(data, sc)
        all_trades.extend(trades)
        stats  = analyze(trades, sc)
        summary_rows.append(stats)
        print(f"  → 거래:{stats['trades']}건  승률:{stats['win_rate']}%  "
              f"EV:{stats['ev']}%  평균수익:{stats['avg_pnl']}%  손절:{stats['stop_pct_r']}%")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("/Users/jungsoo.kim/Desktop/backtest/backtest_golden_cross_summary.csv",
                      index=False, encoding="utf-8-sig")
    pd.DataFrame(all_trades).to_csv(
        "/Users/jungsoo.kim/Desktop/backtest/backtest_golden_cross_trades.csv",
        index=False, encoding="utf-8-sig")

    # ── 콘솔 요약 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 108)
    print(f"{'시나리오':<28} {'목표':>5} {'손절':>5} {'거래':>5} {'승률':>7} "
          f"{'평균수익':>8} {'EV':>7} {'avg승리':>8} {'avg손실':>8} "
          f"{'목표%':>6} {'손절%':>6} {'평균보유':>8}")
    print("-" * 108)
    for _, r in summary_df.iterrows():
        tag = " ★" if r["scenario"].startswith(("REF","D5")) else ""
        print(f"{r['scenario']:<28} {r['target_cfg']:>4.0f}% {r['stop_cfg']:>4.0f}%  "
              f"{r['trades']:>5}  {r['win_rate']:>6.1f}%  {r['avg_pnl']:>7.2f}%  "
              f"{r['ev']:>6.2f}%  {r['avg_win']:>7.2f}%  {r['avg_loss']:>7.2f}%  "
              f"{r['target_pct']:>5.1f}%  {r['stop_pct_r']:>5.1f}%  "
              f"{r['avg_hold']:>7.1f}일{tag}")
    print("=" * 108)

    # ── 최적 조합 ─────────────────────────────────────────────────────────────
    g_df = summary_df[summary_df["scenario"].str.startswith("G")]
    if len(g_df) and g_df["trades"].max() > 0:
        best = g_df.loc[g_df["ev"].idxmax()]
        ref_ev = summary_df[summary_df["scenario"]=="REF_A현행전략"]["ev"].values
        ref_ev = float(ref_ev[0]) if len(ref_ev) else 0
        print(f"\n★ EV 기준 최적 G 조합: {best['scenario']}")
        print(f"  거래:{best['trades']}건  승률:{best['win_rate']}%  EV:{best['ev']}%")
        print(f"  REF_A EV: {ref_ev}%  →  "
              f"{'골든크로스가 우위' if best['ev']>ref_ev else 'REF가 우위'} "
              f"({abs(best['ev']-ref_ev):.2f}%p 차이)")

    plot(summary_df, all_trades)
    print("\n완료!")

if __name__ == "__main__":
    main()
