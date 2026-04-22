"""
backtest_c_group_exit.py
========================================
C그룹 출구 전략 최적화
  [고정 진입 조건]
    현재가 > MA200
    MACD 골든크로스 (전일 hist ≤ 0 → 당일 hist > 0)
    종가 %B > 80
    RSI > 70

  [비교할 출구 전략]
    MACD 게이트형 : 목표% 첫 도달 후 MACD 둔화전환 시 매도, 최대 5거래일 대기
      → target = 6%, 8%, 10%, 12%, 15%
    단순 목표형   : 목표% 도달 즉시 매도
      → target = 8%, 10%, 12%, 15%, 20%
    공통 : 손절 -25% / 60일수익 / 120일만료

  [비교 기준선]  REF_A (A그룹 현행), REF_B (B그룹 현행)

기간: 2015-01-01 ~ 2026-04-15 / 사용자 지정 티커
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
STOP  = 0.25
HALF  = 60
MAX   = 120
WAIT  = 5   # MACD 게이트 최대 대기일

KR_TICKERS = [
    "000660.KS","005930.KS","277810.KS","034020.KS","015760.KS",
    "005380.KS","012450.KS","042660.KS","042700.KQ","096770.KS",
    "009150.KS","000270.KS","247540.KQ","376900.KS","004020.KS",
    "329180.KS","375500.KS","086280.KS","000720.KS","353200.KS",
    "011070.KS","079550.KS",
]
US_TICKERS = [
    "HOOD","AVGO","AMD","MSFT","GOOGL","NVDA","TSLA","MU","LRCX","ON",
    "SNDK","ASTS","AVAV","IONQ","RKLB","PLTR","APP","SOXL","TSLL","TE",
    "ONDS","BE","PL","VRT","LITE","TER","ANET","IREN","HOOG","SOLT",
    "ETHU","NBIS","LPTH","CONL","GLW","FLNC","VST","ASX","CRCL","SGML",
    "AEHR","MP","PLAB","SKYT","SMTC","COHR","MPWR","CIEN","KLAC","FORM","CRDO",
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
    l     = df["Low"]
    ma20  = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bb_u  = ma20 + 2.0 * std20
    bb_l  = ma20 - 2.0 * std20
    bb_r  = bb_u - bb_l

    df["ma200"]      = c.rolling(200).mean()
    df["pctb_close"] = np.where(bb_r > 0, (c - bb_l) / bb_r * 100, np.nan)
    df["pctb_low"]   = np.where(bb_r > 0, (l - bb_l) / bb_r * 100, np.nan)

    bb_w             = (bb_r / ma20 * 100).where(ma20 > 0)
    df["bb_width"]   = bb_w
    df["bb_w_avg60"] = bb_w.rolling(60).mean()
    df["squeeze"]    = bb_w < df["bb_w_avg60"] * 0.50

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    ema_f = c.ewm(span=12, adjust=False).mean()
    ema_s = c.ewm(span=26, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=9,  adjust=False).mean()
    hist  = macd - sig
    df["hist"]     = hist
    df["hist_d1"]  = hist.shift(1)   # 전일
    df["hist_d2"]  = hist.shift(2)   # 전전일
    df["golden"]   = (df["hist_d1"] <= 0) & (df["hist"] > 0)

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


# ── C그룹 공통 진입 조건 확인 ──────────────────────────────────────────────────
C_REQ = ["ma200","pctb_close","rsi","golden","hist","hist_d1","hist_d2"]

def c_entry(row):
    try:
        return (float(row["Close"]) > float(row["ma200"])
                and bool(row["golden"])
                and float(row["pctb_close"]) > 80
                and float(row["rsi"]) > 70)
    except:
        return False


# ── 출구 전략 함수들 ───────────────────────────────────────────────────────────
def exit_macd_gate(close, entry, hold, target, arm_day, arm_idx, hist, hist_d1, hist_d2):
    """목표% 첫 도달 → MACD 둔화전환 대기, 최대 WAIT 거래일"""
    pnl = (close - entry) / entry
    new_arm = arm_day

    if pnl >= target and arm_day is None:
        new_arm = arm_idx   # 목표 도달 첫날 기록

    if arm_day is not None:
        wait = hold - arm_day
        hist_turn = (hist is not None and hist_d1 is not None and hist_d2 is not None
                     and (hist - hist_d1) < (hist_d1 - hist_d2))
        if pnl >= target and hist_turn:
            return "목표+MACD둔화", pnl, new_arm
        if wait >= WAIT:
            return "목표+5일만료", pnl, new_arm

    if pnl <= -STOP:                         return "손절",    pnl, new_arm
    if hold >= HALF and pnl > 0:             return "60일수익", pnl, new_arm
    if hold >= MAX:                          return "기간만료", pnl, new_arm
    return None, pnl, new_arm


def exit_simple(close, entry, hold, target):
    """목표% 도달 즉시 매도"""
    pnl = (close - entry) / entry
    if pnl >= target:                        return "목표",    pnl
    if pnl <= -STOP:                         return "손절",    pnl
    if hold >= HALF and pnl > 0:             return "60일수익", pnl
    if hold >= MAX:                          return "기간만료", pnl
    return None, pnl


# ── C그룹 단일 시나리오 백테스트 ──────────────────────────────────────────────
def run_c_macd_gate(data, target):
    trades = []
    for _, df in data.items():
        dfc = df.dropna(subset=C_REQ)
        if len(dfc) < 10: continue
        in_pos = False; ep = ei = None; arm = None
        for ii, date in enumerate(dfc.index):
            row = dfc.loc[date]
            if in_pos:
                h   = row["hist"];   hd1 = row["hist_d1"]; hd2 = row["hist_d2"]
                reason, pnl, arm = exit_macd_gate(
                    row["Close"], ep, ii - ei, target, arm,
                    ii - ei, h, hd1, hd2)
                if reason:
                    trades.append({"pnl_pct": round(pnl*100,2),
                                   "exit_reason": reason, "hold_days": ii - ei})
                    in_pos = False
            if not in_pos and c_entry(row):
                in_pos = True; ep = row["Close"]; ei = ii; arm = None
        if in_pos and ep:
            last = dfc.iloc[-1]
            pnl  = (last["Close"] - ep) / ep
            trades.append({"pnl_pct": round(pnl*100,2),
                           "exit_reason":"미청산","hold_days": len(dfc)-1-ei})
    return trades


def run_c_simple(data, target):
    trades = []
    for _, df in data.items():
        dfc = df.dropna(subset=C_REQ)
        if len(dfc) < 10: continue
        in_pos = False; ep = ei = None
        for ii, date in enumerate(dfc.index):
            row = dfc.loc[date]
            if in_pos:
                reason, pnl = exit_simple(row["Close"], ep, ii - ei, target)
                if reason:
                    trades.append({"pnl_pct": round(pnl*100,2),
                                   "exit_reason": reason, "hold_days": ii - ei})
                    in_pos = False
            if not in_pos and c_entry(row):
                in_pos = True; ep = row["Close"]; ei = ii
        if in_pos and ep:
            last = dfc.iloc[-1]
            pnl  = (last["Close"] - ep) / ep
            trades.append({"pnl_pct": round(pnl*100,2),
                           "exit_reason":"미청산","hold_days": len(dfc)-1-ei})
    return trades


# ── REF_A / REF_B ─────────────────────────────────────────────────────────────
def run_ref_a(data):
    trades = []
    req = ["ma200","squeeze","pctb_low","hist","hist_d1","hist_d2"]
    for _, df in data.items():
        dfc = df.dropna(subset=req)
        if len(dfc) < 10: continue
        in_pos = False; ep = ei = None; arm = None
        for ii, date in enumerate(dfc.index):
            row = dfc.loc[date]
            if in_pos:
                h = row["hist"]; hd1 = row["hist_d1"]; hd2 = row["hist_d2"]
                reason, pnl, arm = exit_macd_gate(
                    row["Close"], ep, ii - ei, 0.08, arm, ii - ei, h, hd1, hd2)
                if reason:
                    trades.append({"pnl_pct": round(pnl*100,2),
                                   "exit_reason": reason, "hold_days": ii - ei})
                    in_pos = False
            if not in_pos:
                try:
                    if (float(row["Close"]) > float(row["ma200"])
                            and bool(row["squeeze"])
                            and float(row["pctb_low"]) <= 50):
                        in_pos = True; ep = row["Close"]; ei = ii; arm = None
                except: pass
        if in_pos and ep:
            last = dfc.iloc[-1]
            pnl  = (last["Close"] - ep) / ep
            trades.append({"pnl_pct": round(pnl*100,2),
                           "exit_reason":"미청산","hold_days": len(dfc)-1-ei})
    return trades


def run_ref_b(data):
    trades = []
    req = ["ma200","pctb_low","hist","hist_d1","hist_d2"]
    for _, df in data.items():
        dfc = df.dropna(subset=req)
        if len(dfc) < 10: continue
        in_pos = False; ep = ei = None; arm = None
        for ii, date in enumerate(dfc.index):
            row = dfc.loc[date]
            if in_pos:
                h = row["hist"]; hd1 = row["hist_d1"]; hd2 = row["hist_d2"]
                reason, pnl, arm = exit_macd_gate(
                    row["Close"], ep, ii - ei, 0.08, arm, ii - ei, h, hd1, hd2)
                if reason:
                    trades.append({"pnl_pct": round(pnl*100,2),
                                   "exit_reason": reason, "hold_days": ii - ei})
                    in_pos = False
            if not in_pos:
                try:
                    if (float(row["Close"]) > float(row["ma200"])
                            and float(row["pctb_low"]) <= 5):
                        in_pos = True; ep = row["Close"]; ei = ii; arm = None
                except: pass
        if in_pos and ep:
            last = dfc.iloc[-1]
            pnl  = (last["Close"] - ep) / ep
            trades.append({"pnl_pct": round(pnl*100,2),
                           "exit_reason":"미청산","hold_days": len(dfc)-1-ei})
    return trades


# ── 통계 계산 ─────────────────────────────────────────────────────────────────
def stats(trades, label=""):
    if not trades:
        return {"label":label,"trades":0,"win_rate":0,"avg_pnl":0,"ev":0,
                "avg_win":0,"avg_loss":0,"stop_pct":0,"avg_hold":0}
    df   = pd.DataFrame(trades)
    wins = df[df["pnl_pct"] > 0]
    loss = df[df["pnl_pct"] <= 0]
    wr   = len(wins) / len(df)
    aw   = wins["pnl_pct"].mean() if len(wins) else 0
    al   = loss["pnl_pct"].mean() if len(loss) else 0
    bx   = df["exit_reason"].value_counts(normalize=True) * 100
    return {
        "label":      label,
        "trades":     len(df),
        "win_rate":   round(wr * 100, 1),
        "avg_pnl":    round(df["pnl_pct"].mean(), 2),
        "ev":         round(wr * aw + (1 - wr) * al, 2),
        "avg_win":    round(aw, 2),
        "avg_loss":   round(al, 2),
        "stop_pct":   round(bx.get("손절", 0), 1),
        "avg_hold":   round(df["hold_days"].mean(), 1),
        "exit_dist":  dict(bx.round(1)),
    }


def main():
    print("=" * 80)
    print("C그룹 출구 전략 최적화 백테스트")
    print(f"기간: {START} ~ {END}  |  종목: {len(ALL_TICKERS)}개")
    print(f"[고정 진입] MA200 위 + 골든크로스 + 종가%B>80 + RSI>70")
    print(f"[공통 출구] 손절 -{STOP*100:.0f}% / {HALF}일수익 / {MAX}일만료")
    print("=" * 80)

    data = download_data()

    # ── 시나리오 정의 ──────────────────────────────────────────────────────────
    GATE_TARGETS   = [0.06, 0.08, 0.10, 0.12, 0.15]
    SIMPLE_TARGETS = [0.08, 0.10, 0.12, 0.15, 0.20]

    rows = []

    # MACD 게이트형
    for t in GATE_TARGETS:
        label = f"MACD게이트+{int(t*100)}%"
        print(f"  {label} 실행 중...")
        s = stats(run_c_macd_gate(data, t), label)
        s["type"] = "MACD게이트"
        s["target"] = t
        rows.append(s)

    # 단순 목표형
    for t in SIMPLE_TARGETS:
        label = f"단순목표+{int(t*100)}%"
        print(f"  {label} 실행 중...")
        s = stats(run_c_simple(data, t), label)
        s["type"] = "단순목표"
        s["target"] = t
        rows.append(s)

    # 기준선
    print("  REF_A 실행 중...")
    ref_a = stats(run_ref_a(data), "REF_A (A그룹 현행)")
    ref_a["type"] = "기준선"; ref_a["target"] = 0.08
    print("  REF_B 실행 중...")
    ref_b = stats(run_ref_b(data), "REF_B (B그룹 현행)")
    ref_b["type"] = "기준선"; ref_b["target"] = 0.08

    df_all = pd.DataFrame(rows)

    # ── 콘솔 결과 출력 ────────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print(f"  {'전략':>22}  {'거래':>6}  {'승률':>7}  {'평균수익':>8}  {'EV':>7}  "
          f"{'avg승':>7}  {'avg패':>7}  {'손절%':>6}  {'평균보유':>7}")
    print("-" * 95)
    best_ev = df_all["ev"].max()
    for _, r in df_all.sort_values("ev", ascending=False).iterrows():
        marker = " ★" if r["ev"] == best_ev else ""
        print(f"  {r['label']:>22}  {r['trades']:>6}  {r['win_rate']:>6.1f}%  "
              f"{r['avg_pnl']:>7.2f}%  {r['ev']:>6.2f}%  "
              f"{r['avg_win']:>7.2f}%  {r['avg_loss']:>7.2f}%  "
              f"{r['stop_pct']:>5.1f}%  {r['avg_hold']:>6.1f}일{marker}")
    print("-" * 95)
    for ref in [ref_a, ref_b]:
        print(f"  {ref['label']:>22}  {ref['trades']:>6}  {ref['win_rate']:>6.1f}%  "
              f"{ref['avg_pnl']:>7.2f}%  {ref['ev']:>6.2f}%  "
              f"{ref['avg_win']:>7.2f}%  {ref['avg_loss']:>7.2f}%  "
              f"{ref['stop_pct']:>5.1f}%  {ref['avg_hold']:>6.1f}일 ← 기준선")
    print("=" * 95)

    # ── 시각화 ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"C그룹 출구 전략 최적화\n"
        f"[진입] MA200 위 + MACD 골든크로스 + 종가%B>80 + RSI>70  |  "
        f"[공통 출구] 손절-{STOP*100:.0f}% / {HALF}일수익 / {MAX}일만료",
        fontsize=11, fontweight="bold"
    )

    gate_df   = df_all[df_all["type"] == "MACD게이트"].sort_values("target")
    simple_df = df_all[df_all["type"] == "단순목표"].sort_values("target")

    gate_labels   = [f"+{int(t*100)}%" for t in gate_df["target"]]
    simple_labels = [f"+{int(t*100)}%" for t in simple_df["target"]]

    def draw(ax, g_vals, s_vals, g_labels, s_labels, ylabel, title, ref_a_val, ref_b_val):
        x_g = np.arange(len(g_labels))
        x_s = np.arange(len(s_labels)) + len(g_labels) + 1.5
        ax.bar(x_g, g_vals, color="#1565C0", alpha=0.8, label="MACD게이트형", width=0.7)
        ax.bar(x_s, s_vals, color="#E65100", alpha=0.8, label="단순목표형",   width=0.7)
        ax.plot(x_g, g_vals, "o--", color="#0D47A1", lw=1.2, ms=4)
        ax.plot(x_s, s_vals, "s--", color="#BF360C", lw=1.2, ms=4)
        ax.axhline(ref_a_val, color="green",  lw=1.5, ls=":", label=f"REF_A {ref_a_val:.2f}")
        ax.axhline(ref_b_val, color="purple", lw=1.2, ls=":", label=f"REF_B {ref_b_val:.2f}")
        all_x = list(x_g) + list(x_s)
        all_l = [f"게이트\n{l}" for l in g_labels] + [f"단순\n{l}" for l in s_labels]
        ax.set_xticks(all_x); ax.set_xticklabels(all_l, fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold", fontsize=10)
        ax.legend(fontsize=8, loc="best"); ax.axhline(0, color="black", lw=0.5, ls="--")

    draw(axes[0,0], gate_df["ev"].tolist(),       simple_df["ev"].tolist(),
         gate_labels, simple_labels, "EV (%)", "★ 기대값 EV", ref_a["ev"], ref_b["ev"])
    draw(axes[0,1], gate_df["win_rate"].tolist(),  simple_df["win_rate"].tolist(),
         gate_labels, simple_labels, "승률 (%)", "승률", ref_a["win_rate"], ref_b["win_rate"])
    draw(axes[1,0], gate_df["avg_pnl"].tolist(),   simple_df["avg_pnl"].tolist(),
         gate_labels, simple_labels, "평균수익 (%)", "평균 수익률", ref_a["avg_pnl"], ref_b["avg_pnl"])
    draw(axes[1,1], gate_df["stop_pct"].tolist(),  simple_df["stop_pct"].tolist(),
         gate_labels, simple_labels, "손절 비율 (%)", "손절 비율", ref_a["stop_pct"], ref_b["stop_pct"])

    plt.tight_layout()
    out = "/Users/jungsoo.kim/Desktop/backtest/backtest_c_group_exit.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n차트 저장: {out}")

    # ── 최적 조합 ─────────────────────────────────────────────────────────────
    best = df_all.loc[df_all["ev"].idxmax()]
    print("\n" + "=" * 60)
    print("★ 최적 출구 전략 (EV 기준)")
    print(f"  전략   : {best['label']}")
    print(f"  유형   : {best['type']}")
    print(f"  목표%  : +{int(best['target']*100)}%")
    print(f"  EV     : {best['ev']}%")
    print(f"  승률   : {best['win_rate']}%")
    print(f"  거래 수: {best['trades']}건")
    print(f"  손절%  : {best['stop_pct']}%")
    print(f"  REF_A EV: {ref_a['ev']}%  |  REF_B EV: {ref_b['ev']}%")
    print("=" * 60)

    # 종류별 최적 요약
    best_gate   = df_all[df_all["type"]=="MACD게이트"].loc[df_all[df_all["type"]=="MACD게이트"]["ev"].idxmax()]
    best_simple = df_all[df_all["type"]=="단순목표"].loc[df_all[df_all["type"]=="단순목표"]["ev"].idxmax()]
    print(f"\n  MACD게이트형 최적: {best_gate['label']}  EV {best_gate['ev']}%  승률 {best_gate['win_rate']}%  거래 {best_gate['trades']}건")
    print(f"  단순목표형   최적: {best_simple['label']}  EV {best_simple['ev']}%  승률 {best_simple['win_rate']}%  거래 {best_simple['trades']}건")
    print("\n완료!")


if __name__ == "__main__":
    main()
