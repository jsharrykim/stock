"""
backtest_d_group.py
====================
D그룹: BB 상단 돌파 모멘텀 전략 — 조합별 비교 + A/B 현행 전략 기준선

[공통 진입 베이스 - D그룹 전체 공통]
  ① 현재가 > MA200
  ② 고가 기준 %B ≥ 95  (고가가 BB 상단 근처 또는 돌파)
  ③ BB폭 > BB폭 60일 평균 × 80%  (밴드 확장 중 — 스퀴즈의 반대)

[조합별 추가 필터]
  D1: MACD hist > 0  AND  hist 증가 중 (오늘 > 어제)
  D2: MACD hist > 0  AND  CCI > 100
  D3: MACD hist > 0  AND  RSI ≥ 55  AND  RSI ≤ 75      (과열 차단)
  D4: MACD hist > 0  AND  hist 증가 중  AND  RSI ≤ 75  AND  CCI > 100   (복합 고확도)
  D5: MACD hist > 0  AND  RSI ≤ 80                     (관대형 — 오버슈팅 허용)

[비교 기준선 — 현행 A/B 그룹]
  REF_A: 현재가 > MA200  AND  저가%B ≤ 50  AND  스퀴즈
  REF_B: 현재가 > MA200  AND  저가%B ≤ 5

[매도 조건]
  D 그룹:  목표 +15% / 손절 -15% / 30거래일 후 수익 중 / 60거래일 강제 만료
  REF A/B: 목표  +8% / 손절 -25% / 60거래일 후 수익 중 / 120거래일 강제 만료

종목: 기존 백테스트와 동일한 49개 풀
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

# ── D그룹 매도 파라미터 ────────────────────────────────────────────────────────
D_TARGET    = 0.15   # 목표 +15%
D_STOP      = 0.15   # 손절 -15%
D_HALF_DAYS = 30     # 30거래일 후 수익 중 매도
D_MAX_HOLD  = 60     # 최대 보유 60거래일

# ── REF(A/B) 매도 파라미터 ─────────────────────────────────────────────────────
REF_TARGET    = 0.08
REF_STOP      = 0.25
REF_HALF_DAYS = 60
REF_MAX_HOLD  = 120

# ── 지표 파라미터 ──────────────────────────────────────────────────────────────
BB_PERIOD      = 20
BB_STD         = 2.0
BB_AVG_PERIOD  = 60     # BB폭 기준 이동평균
BB_EXPAND_RATIO = 0.80  # BB폭 > 60일 평균 × 이 비율 → 확장 판정
HIGH_PCTB_MIN  = 95     # 고가 %B ≥ 이 값 → BB 상단 근처
SQUEEZE_RATIO  = 0.50   # 스퀴즈 기준 (REF_A용)
PCTB_LOW_A     = 50     # REF_A 저가%B 임계
PCTB_LOW_B     = 5      # REF_B 저가%B 임계
MACD_FAST      = 12
MACD_SLOW      = 26
MACD_SIGNAL    = 9

# ── 종목 ──────────────────────────────────────────────────────────────────────
KR_TICKERS = [
    "000660.KS", "005930.KS", "277810.KS", "034020.KS", "005380.KS",
    "012450.KS", "042660.KS", "042700.KQ", "096770.KS", "009150.KS",
    "000270.KS", "247540.KQ", "376900.KS", "006400.KS", "079550.KS",
]
US_TICKERS = [
    "HOOD", "AAPL", "AVGO", "AMD", "MSFT", "GOOGL", "NVDA", "TSLA",
    "AMZN", "MU", "LRCX", "ON", "SNDK", "ASTS", "AVAV", "IONQ",
    "RKLB", "PLTR", "CRWD", "APP", "SOXL", "TSLL", "TE", "ONDS",
    "BE", "PL", "VRT", "LITE", "TER", "ANET", "IREN", "HOOG",
    "SOLT", "ETHU", "NBIS", "LPTH", "CONL", "INTC",
    "CRDO", "SKYT",  # 이미지에서 확인된 종목 추가
]
ALL_TICKERS = KR_TICKERS + US_TICKERS

# ── 한글 폰트 ──────────────────────────────────────────────────────────────────
def get_kr_font():
    for c in ["AppleGothic", "NanumGothic", "Malgun Gothic", "DejaVu Sans"]:
        if c in {f.name for f in fm.fontManager.ttflist}:
            return c
    return None

KR_FONT = get_kr_font()
if KR_FONT:
    plt.rcParams["font.family"] = KR_FONT
plt.rcParams["axes.unicode_minus"] = False

# ── 지표 계산 ──────────────────────────────────────────────────────────────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # MA200
    df["ma200"] = close.rolling(200).mean()

    # BB
    ma20      = close.rolling(BB_PERIOD).mean()
    std20     = close.rolling(BB_PERIOD).std()
    bb_upper  = ma20 + BB_STD * std20
    bb_lower  = ma20 - BB_STD * std20
    bb_range  = bb_upper - bb_lower
    df["ma20"]      = ma20
    df["bb_upper"]  = bb_upper
    df["bb_lower"]  = bb_lower

    # BB폭 (% 기준)
    df["bb_width"]     = (bb_range / ma20 * 100).where(ma20 > 0)
    df["bb_width_avg"] = df["bb_width"].rolling(BB_AVG_PERIOD).mean()

    # BB 확장 여부 (D그룹용)
    df["bb_expanding"] = df["bb_width"] > df["bb_width_avg"] * BB_EXPAND_RATIO

    # 스퀴즈 (REF_A용)
    df["squeeze"] = df["bb_width"] < df["bb_width_avg"] * SQUEEZE_RATIO

    # 저가 %B
    df["pctb_low"] = np.where(bb_range > 0, (low - bb_lower) / bb_range * 100, np.nan)

    # 고가 %B (D그룹용 — close 대신 high 기준)
    df["pctb_high"] = np.where(bb_range > 0, (high - bb_lower) / bb_range * 100, np.nan)

    # RSI(14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = np.where(loss == 0, 100.0, gain / loss)
    df["rsi"] = 100 - 100 / (1 + rs)

    # CCI(14)
    tp    = (high + low + close) / 3
    tp_ma = tp.rolling(14).mean()
    tp_md = tp.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = np.where(tp_md > 0, (tp - tp_ma) / (0.015 * tp_md), 0)

    # MACD(12,26,9) histogram
    ema_fast   = close.ewm(span=MACD_FAST,   adjust=False).mean()
    ema_slow   = close.ewm(span=MACD_SLOW,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"]      = macd_line - signal_line
    df["macd_hist_prev"] = df["macd_hist"].shift(1)
    # 히스토그램 증가 여부 (오늘 > 어제)
    df["hist_rising"] = df["macd_hist"] > df["macd_hist_prev"]

    return df

# ── 데이터 다운로드 ────────────────────────────────────────────────────────────
def download_data(tickers):
    print(f"종목 데이터 다운로드 중... ({len(tickers)}개)")
    raw = yf.download(
        tickers, start=START, end=END,
        auto_adjust=True, progress=False,
        group_by="ticker"
    )
    result = {}
    for t in tickers:
        try:
            df = raw[t].copy() if len(tickers) > 1 else raw.copy()
            df.dropna(how="all", inplace=True)
            if len(df) < 250:
                print(f"  [{t}] 데이터 부족 ({len(df)}일) - 제외")
                continue
            df = calc_indicators(df)
            result[t] = df
            print(f"  [{t}] {len(df)}일 로드 완료")
        except Exception as e:
            print(f"  [{t}] 오류: {e}")
    return result

# ── 매도 조건 ─────────────────────────────────────────────────────────────────
def check_exit(close, entry_price, hold_days, target_pct, stop_pct, half_days, max_hold):
    pnl = (close - entry_price) / entry_price
    if pnl >= target_pct:
        return "목표", pnl
    if pnl <= -stop_pct:
        return "손절", pnl
    if hold_days >= half_days and pnl > 0:
        return f"{half_days}일수익", pnl
    if hold_days >= max_hold:
        return "기간만료", pnl
    return None, pnl

# ── 백테스트 엔진 ─────────────────────────────────────────────────────────────
def run_backtest(data: dict, scenario: dict) -> list:
    name       = scenario["name"]
    cond_fn    = scenario["fn"]
    target_pct = scenario["target"]
    stop_pct   = scenario["stop"]
    half_days  = scenario["half_days"]
    max_hold   = scenario["max_hold"]
    req_cols   = scenario.get("req_cols", [
        "ma200", "bb_width_avg", "bb_expanding", "squeeze",
        "pctb_low", "pctb_high", "rsi", "cci", "macd_hist", "hist_rising"
    ])

    trades = []
    for ticker, df in data.items():
        df_clean = df.dropna(subset=req_cols)
        if len(df_clean) < 10:
            continue

        in_pos      = False
        entry_price = None
        entry_date  = None
        entry_idx   = None
        idx_list    = list(df_clean.index)

        for ii, date in enumerate(idx_list):
            row = df_clean.loc[date]

            if in_pos:
                hold_days = ii - entry_idx
                reason, pnl = check_exit(
                    row["Close"], entry_price, hold_days,
                    target_pct, stop_pct, half_days, max_hold
                )
                if reason:
                    trades.append({
                        "scenario":    name,
                        "ticker":      ticker,
                        "entry_date":  entry_date,
                        "exit_date":   date,
                        "entry_price": round(float(entry_price), 4),
                        "exit_price":  round(float(row["Close"]), 4),
                        "pnl_pct":     round(pnl * 100, 2),
                        "hold_days":   hold_days,
                        "exit_reason": reason,
                    })
                    in_pos = False

            if not in_pos:
                sig = {
                    "close":        float(row["Close"]),
                    "ma200":        float(row["ma200"]),
                    "above200":     float(row["Close"]) > float(row["ma200"]),
                    "bb_expanding": bool(row["bb_expanding"]),
                    "squeeze":      bool(row["squeeze"]),
                    "pctb_low":     float(row["pctb_low"]) if not pd.isna(row["pctb_low"]) else 999,
                    "pctb_high":    float(row["pctb_high"]) if not pd.isna(row["pctb_high"]) else -999,
                    "rsi":          float(row["rsi"]),
                    "cci":          float(row["cci"]),
                    "macd_hist":    float(row["macd_hist"]),
                    "hist_rising":  bool(row["hist_rising"]),
                }
                try:
                    if cond_fn(sig):
                        in_pos      = True
                        entry_price = row["Close"]
                        entry_date  = date
                        entry_idx   = ii
                except Exception:
                    pass

        if in_pos and entry_price is not None:
            last = df_clean.iloc[-1]
            pnl  = (last["Close"] - entry_price) / entry_price
            trades.append({
                "scenario":    name,
                "ticker":      ticker,
                "entry_date":  entry_date,
                "exit_date":   df_clean.index[-1],
                "entry_price": round(float(entry_price), 4),
                "exit_price":  round(float(last["Close"]), 4),
                "pnl_pct":     round(pnl * 100, 2),
                "hold_days":   len(df_clean) - 1 - entry_idx,
                "exit_reason": "미청산",
            })
    return trades

# ── 시나리오 정의 ─────────────────────────────────────────────────────────────
# 공통 D그룹 베이스 조건
def d_base(s):
    return (
        s["above200"]
        and s["pctb_high"] >= HIGH_PCTB_MIN
        and s["bb_expanding"]
    )

SCENARIOS = [
    # ── D그룹 (목표+15% / 손절-15% / 30일수익 / 60일만료) ─────────────────────
    {
        "name":      "D1_MACD증가",
        "target":    D_TARGET, "stop": D_STOP,
        "half_days": D_HALF_DAYS, "max_hold": D_MAX_HOLD,
        "fn": lambda s: d_base(s) and s["macd_hist"] > 0 and s["hist_rising"],
    },
    {
        "name":      "D2_MACD+CCI100",
        "target":    D_TARGET, "stop": D_STOP,
        "half_days": D_HALF_DAYS, "max_hold": D_MAX_HOLD,
        "fn": lambda s: d_base(s) and s["macd_hist"] > 0 and s["cci"] > 100,
    },
    {
        "name":      "D3_MACD+RSI55-75",
        "target":    D_TARGET, "stop": D_STOP,
        "half_days": D_HALF_DAYS, "max_hold": D_MAX_HOLD,
        "fn": lambda s: d_base(s) and s["macd_hist"] > 0 and 55 <= s["rsi"] <= 75,
    },
    {
        "name":      "D4_복합고확도",
        "target":    D_TARGET, "stop": D_STOP,
        "half_days": D_HALF_DAYS, "max_hold": D_MAX_HOLD,
        "fn": lambda s: (
            d_base(s)
            and s["macd_hist"] > 0 and s["hist_rising"]
            and s["rsi"] <= 75
            and s["cci"] > 100
        ),
    },
    {
        "name":      "D5_MACD+RSI80",
        "target":    D_TARGET, "stop": D_STOP,
        "half_days": D_HALF_DAYS, "max_hold": D_MAX_HOLD,
        "fn": lambda s: d_base(s) and s["macd_hist"] > 0 and s["rsi"] <= 80,
    },
    # ── D그룹 베이스 단독 (필터 없음 — 하한선 확인용) ─────────────────────────
    {
        "name":      "D0_베이스단독",
        "target":    D_TARGET, "stop": D_STOP,
        "half_days": D_HALF_DAYS, "max_hold": D_MAX_HOLD,
        "fn": lambda s: d_base(s),
    },
    # ── 기준선 A/B (현행 전략 목표+8% / 손절-25% / 60일수익 / 120일만료) ─────
    {
        "name":      "REF_A현행전략",
        "target":    REF_TARGET, "stop": REF_STOP,
        "half_days": REF_HALF_DAYS, "max_hold": REF_MAX_HOLD,
        "req_cols":  ["ma200", "bb_width_avg", "squeeze", "pctb_low"],
        "fn": lambda s: s["above200"] and s["pctb_low"] <= PCTB_LOW_A and s["squeeze"],
    },
    {
        "name":      "REF_B현행전략",
        "target":    REF_TARGET, "stop": REF_STOP,
        "half_days": REF_HALF_DAYS, "max_hold": REF_MAX_HOLD,
        "req_cols":  ["ma200", "pctb_low"],
        "fn": lambda s: s["above200"] and s["pctb_low"] <= PCTB_LOW_B,
    },
]

# ── 결과 분석 ─────────────────────────────────────────────────────────────────
def analyze(trades: list, name: str, target_pct: float, stop_pct: float) -> dict:
    if not trades:
        return {
            "scenario": name, "trades": 0, "win_rate": 0,
            "avg_pnl": 0, "median_pnl": 0, "best": 0, "worst": 0,
            "avg_hold": 0, "total_pnl": 0,
            "target_pct": 0, "stop_pct_r": 0, "half_pct": 0, "expire_pct": 0,
            "ev": 0, "avg_win": 0, "avg_loss": 0,
            "target_cfg": target_pct * 100, "stop_cfg": stop_pct * 100,
        }
    df   = pd.DataFrame(trades)
    wins = df[df["pnl_pct"] > 0]
    loss = df[df["pnl_pct"] <= 0]
    wr   = len(wins) / len(df)
    avg_win  = wins["pnl_pct"].mean() if len(wins) > 0 else 0
    avg_loss = loss["pnl_pct"].mean() if len(loss) > 0 else 0
    ev       = round(wr * avg_win + (1 - wr) * avg_loss, 2)
    by_exit  = df["exit_reason"].value_counts(normalize=True) * 100

    # half_days key는 시나리오마다 다를 수 있어 유연하게 처리
    half_key = next((k for k in by_exit.index if "일수익" in k), None)

    return {
        "scenario":   name,
        "trades":     len(df),
        "win_rate":   round(wr * 100, 1),
        "avg_pnl":    round(df["pnl_pct"].mean(), 2),
        "median_pnl": round(df["pnl_pct"].median(), 2),
        "best":       round(df["pnl_pct"].max(), 2),
        "worst":      round(df["pnl_pct"].min(), 2),
        "avg_hold":   round(df["hold_days"].mean(), 1),
        "total_pnl":  round(df["pnl_pct"].sum(), 1),
        "avg_win":    round(avg_win, 2),
        "avg_loss":   round(avg_loss, 2),
        "ev":         ev,
        "target_pct": round(by_exit.get("목표", 0), 1),
        "stop_pct_r": round(by_exit.get("손절", 0), 1),
        "half_pct":   round(by_exit.get(half_key, 0) if half_key else 0, 1),
        "expire_pct": round(by_exit.get("기간만료", 0), 1),
        "target_cfg": target_pct * 100,
        "stop_cfg":   stop_pct * 100,
    }

# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_results(summary_df: pd.DataFrame, all_trades: list):
    d_df   = summary_df[~summary_df["scenario"].str.startswith("REF")]
    ref_df = summary_df[summary_df["scenario"].str.startswith("REF")]

    colors_d   = ["#1A237E", "#1565C0", "#1976D2", "#42A5F5", "#90CAF9", "#BBDEFB"]
    colors_ref = ["#B71C1C", "#E53935"]

    # ── 차트 1: 핵심 지표 비교 (D그룹 + REF) ─────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle(
        "D그룹 (BB 상단 돌파 모멘텀) vs A/B 현행 전략 비교\n"
        f"D그룹: 목표+{D_TARGET*100:.0f}% / 손절-{D_STOP*100:.0f}% / {D_HALF_DAYS}일수익 / {D_MAX_HOLD}일만료  │  "
        f"REF A/B: 목표+{REF_TARGET*100:.0f}% / 손절-{REF_STOP*100:.0f}% / {REF_HALF_DAYS}일수익 / {REF_MAX_HOLD}일만료",
        fontsize=12, fontweight="bold"
    )

    all_sc  = list(d_df["scenario"]) + list(ref_df["scenario"])
    all_clr = colors_d[:len(d_df)] + colors_ref[:len(ref_df)]
    comb_df = pd.concat([d_df, ref_df], ignore_index=True)

    def draw_bar(ax, df, col, title, colors, fmt="{:.1f}%", ref_line=None):
        vals = df[col].tolist()
        lbls = df["scenario"].tolist()
        bars = ax.bar(range(len(lbls)), vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xticks(range(len(lbls)))
        ax.set_xticklabels(lbls, rotation=40, ha="right", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        if ref_line is not None:
            ax.axhline(ref_line, color="red", linewidth=1.2, linestyle=":", alpha=0.8,
                       label=f"REF 기준 {ref_line:.1f}")
            ax.legend(fontsize=7)
        vrange = max(vals) - min(vals) if vals else 1
        for bar, v in zip(bars, vals):
            offset = vrange * 0.025 if v >= 0 else -vrange * 0.06
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    fmt.format(v), ha="center", va="bottom", fontsize=7.5)

    draw_bar(axes[0, 0], comb_df, "avg_pnl",  "평균 수익률 (%)",  all_clr)
    draw_bar(axes[0, 1], comb_df, "win_rate", "승률 (%)",         all_clr)
    draw_bar(axes[0, 2], comb_df, "ev",       "기대값 EV (%)",    all_clr)
    draw_bar(axes[1, 0], comb_df, "trades",   "거래 건수",         all_clr, fmt="{:.0f}건")
    draw_bar(axes[1, 1], comb_df, "avg_hold", "평균 보유일",        all_clr, fmt="{:.1f}일")
    draw_bar(axes[1, 2], comb_df, "stop_pct_r", "손절 비율 (%)",   all_clr)

    plt.tight_layout()
    out1 = "/Users/jungsoo.kim/Desktop/backtest/backtest_d_group_main.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"차트 저장: {out1}")

    # ── 차트 2: EV 분해 (승리 평균 vs 손실 평균) ─────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 7))
    fig2.suptitle("기대값(EV) 분해: 평균 수익 vs 평균 손실", fontsize=12, fontweight="bold")

    for ax, df, colors, title in zip(
        axes2,
        [d_df, ref_df],
        [colors_d, colors_ref],
        ["D그룹 조합", "REF A/B 기준선"]
    ):
        lbls    = df["scenario"].tolist()
        x       = range(len(lbls))
        avg_win = df["avg_win"].tolist()
        avg_lss = df["avg_loss"].tolist()
        ev_vals = df["ev"].tolist()
        wr_vals = df["win_rate"].tolist()

        ax.bar([i - 0.2 for i in x], avg_win, 0.35,
               label="평균 수익(승)", color="#4CAF50", alpha=0.8)
        ax.bar([i + 0.2 for i in x], avg_lss, 0.35,
               label="평균 손실(패)", color="#F44336", alpha=0.8)
        ax2_twin = ax.twinx()
        ax2_twin.plot(x, ev_vals, "o--", color="#FF6F00", linewidth=2, label="EV")
        ax2_twin.plot(x, wr_vals, "s--", color="#1565C0", linewidth=2, label="승률")
        ax2_twin.set_ylabel("EV / 승률 (%)", fontsize=9)
        ax2_twin.legend(loc="upper right", fontsize=8)

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels(lbls, rotation=35, ha="right", fontsize=8)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylabel("수익률 (%)", fontsize=9)

    plt.tight_layout()
    out2 = "/Users/jungsoo.kim/Desktop/backtest/backtest_d_group_ev.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"차트 저장: {out2}")

    # ── 차트 3: 매도 사유 비율 ─────────────────────────────────────────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(18, 7))
    fig3.suptitle("매도 사유 비율 비교", fontsize=12, fontweight="bold")

    for ax, df, colors, title in zip(
        axes3,
        [d_df, ref_df],
        [colors_d, colors_ref],
        ["D그룹 조합", "REF A/B 기준선"]
    ):
        lbls  = df["scenario"].tolist()
        x     = range(len(lbls))
        t_val = df["target_pct"].tolist()
        s_val = df["stop_pct_r"].tolist()
        h_val = df["half_pct"].tolist()
        e_val = df["expire_pct"].tolist()
        ax.bar(x, t_val,  label="목표달성",  color="#4CAF50", alpha=0.85)
        ax.bar(x, s_val,  bottom=t_val,     label="손절",     color="#F44336", alpha=0.85)
        b2 = [a + b for a, b in zip(t_val, s_val)]
        ax.bar(x, h_val,  bottom=b2,        label="시간수익", color="#2196F3", alpha=0.85)
        b3 = [a + b for a, b in zip(b2, h_val)]
        ax.bar(x, e_val,  bottom=b3,        label="기간만료", color="#FF9800", alpha=0.85)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels(lbls, rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel("비율 (%)")

    plt.tight_layout()
    out3 = "/Users/jungsoo.kim/Desktop/backtest/backtest_d_group_exit.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"차트 저장: {out3}")

    # ── 차트 4: 수익률 분포 박스플롯 ──────────────────────────────────────────
    all_df = pd.DataFrame(all_trades)
    all_sc_list = list(comb_df["scenario"])
    all_data    = [all_df[all_df["scenario"] == s]["pnl_pct"].tolist() or [0] for s in all_sc_list]

    fig4, ax4 = plt.subplots(figsize=(20, 8))
    fig4.suptitle("시나리오별 수익률 분포 (박스플롯)", fontsize=12, fontweight="bold")
    bp = ax4.boxplot(all_data, patch_artist=True, notch=False)
    for patch, c in zip(bp["boxes"], all_clr):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax4.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax4.axhline(D_TARGET * 100,   color="#4CAF50", linewidth=1, linestyle=":",
                label=f"D목표 +{D_TARGET*100:.0f}%")
    ax4.axhline(-D_STOP * 100,    color="#F44336", linewidth=1, linestyle=":",
                label=f"D손절 -{D_STOP*100:.0f}%")
    ax4.axhline(REF_TARGET * 100, color="#81C784", linewidth=1, linestyle="-.",
                label=f"REF목표 +{REF_TARGET*100:.0f}%")
    ax4.axhline(-REF_STOP * 100,  color="#EF9A9A", linewidth=1, linestyle="-.",
                label=f"REF손절 -{REF_STOP*100:.0f}%")
    ax4.set_xticks(range(1, len(all_sc_list) + 1))
    ax4.set_xticklabels(all_sc_list, rotation=40, ha="right", fontsize=9)
    ax4.legend(fontsize=9)
    ax4.set_ylabel("수익률 (%)")

    plt.tight_layout()
    out4 = "/Users/jungsoo.kim/Desktop/backtest/backtest_d_group_dist.png"
    plt.savefig(out4, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"차트 저장: {out4}")

# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("D그룹 (BB 상단 돌파 모멘텀) vs A/B 현행 전략 비교")
    print(f"기간: {START} ~ {END}")
    print(f"D그룹 매도: +{D_TARGET*100:.0f}% / -{D_STOP*100:.0f}% / {D_HALF_DAYS}일수익 / {D_MAX_HOLD}일만료")
    print(f"REF  매도: +{REF_TARGET*100:.0f}% /  -{REF_STOP*100:.0f}% / {REF_HALF_DAYS}일수익 / {REF_MAX_HOLD}일만료")
    print("=" * 80)

    data = download_data(ALL_TICKERS)
    print(f"\n유효 종목: {len(data)}개\n")

    all_trades   = []
    summary_rows = []

    for sc in SCENARIOS:
        name = sc["name"]
        print(f"[{name}] 실행 중...")
        trades = run_backtest(data, sc)
        all_trades.extend(trades)
        stats  = analyze(trades, name, sc["target"], sc["stop"])
        summary_rows.append(stats)
        print(
            f"  → 거래:{stats['trades']}건  승률:{stats['win_rate']}%  "
            f"평균수익:{stats['avg_pnl']}%  EV:{stats['ev']}%  "
            f"목표:{stats['target_pct']}%  손절:{stats['stop_pct_r']}%"
        )

    summary_df = pd.DataFrame(summary_rows)

    # CSV 저장
    summary_df.to_csv(
        "/Users/jungsoo.kim/Desktop/backtest/backtest_d_group_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(all_trades).to_csv(
        "/Users/jungsoo.kim/Desktop/backtest/backtest_d_group_trades.csv",
        index=False, encoding="utf-8-sig"
    )

    # ── 최종 요약 테이블 출력 ─────────────────────────────────────────────────
    print("\n" + "=" * 105)
    print(
        f"{'시나리오':<22} {'목표':>5} {'손절':>5} {'거래':>5} {'승률':>7} "
        f"{'평균수익':>8} {'중간값':>7} {'EV':>7} {'평균수익(승)':>11} {'평균손실(패)':>11} "
        f"{'목표%':>6} {'손절%':>6} {'평균보유':>8}"
    )
    print("-" * 105)
    for _, r in summary_df.iterrows():
        tag = "← 현행 전략" if r["scenario"].startswith("REF") else ""
        print(
            f"{r['scenario']:<22} {r['target_cfg']:>4.0f}% {r['stop_cfg']:>4.0f}%  "
            f"{r['trades']:>5}  {r['win_rate']:>6.1f}%  {r['avg_pnl']:>7.2f}%  "
            f"{r['median_pnl']:>6.2f}%  {r['ev']:>6.2f}%  "
            f"{r['avg_win']:>10.2f}%  {r['avg_loss']:>10.2f}%  "
            f"{r['target_pct']:>5.1f}%  {r['stop_pct_r']:>5.1f}%  "
            f"{r['avg_hold']:>7.1f}일  {tag}"
        )
    print("=" * 105)

    # ── 최적 D 조합 선택 (EV 기준) ───────────────────────────────────────────
    d_rows = summary_df[~summary_df["scenario"].str.startswith("REF")]
    if len(d_rows) > 0 and d_rows["trades"].max() > 0:
        best = d_rows.loc[d_rows["ev"].idxmax()]
        ref_ev = summary_df[summary_df["scenario"] == "REF_A현행전략"]["ev"].values
        ref_ev = ref_ev[0] if len(ref_ev) > 0 else 0

        print(f"\n★ EV 기준 최적 D 조합: {best['scenario']}")
        print(f"  거래:{best['trades']}건  승률:{best['win_rate']}%  "
              f"EV:{best['ev']}%  평균수익:{best['avg_pnl']}%")
        print(f"  REF_A 현행 EV: {ref_ev}%  →  "
              f"{'D그룹이 우위' if best['ev'] > ref_ev else 'REF가 우위'} "
              f"(차이 {abs(best['ev'] - ref_ev):.2f}%p)")

    plot_results(summary_df, all_trades)
    print("\n완료!")

if __name__ == "__main__":
    main()
