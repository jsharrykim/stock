"""
backtest_ixic_vs_qqq_top_detection.py
======================================
나스닥 고점 감지 조건 3가지 비교:

  [A] ^IXIC  MA200×1.12 + VIX 조건  (현재 as-is)
  [B] QQQ    MA200×1.12 + VIX 조건  (인덱스만 교체)
  [C] QQQ    MA200×1.12 + VIX 조건 + 주봉 RSI≥70 + 일봉 RSI≥70 + 일봉 RSI 꺾임

평가 기준:
  - 시그널 발생 후 QQQ 기준 20/40/60 거래일 내 최대 낙폭 (MDD)
  - True Positive: 이후 60 거래일 내 QQQ -10% 이상 하락
  - False Positive: 미달
  - 쿨다운: 시그널 발생 후 20 거래일간 재발생 무시

기간: 2015-01-01 ~ 2025-12-31
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from datetime import datetime

# ─────────────────────────────────────────────
# 파라미터
# ─────────────────────────────────────────────
START = "2014-01-01"   # RSI/MA 워밍업용 1년 여유
END   = "2025-12-31"
EVAL_START = "2015-01-01"

MA_WINDOW        = 200
DISTANCE_MULTS   = [1.08, 1.09, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15]   # 8~15% 1% 단위
VIX_HIGH         = 18
VIX_SURGE_PCT    = 10
RSI_PERIOD       = 14
RSI_OVERBOUGHT   = 70

COOLDOWN_DAYS    = 20
TP_THRESHOLD_PCT = 0.07
TP_EVAL_DAY      = 30
EVAL_DAYS        = [10, 20, 30, 40]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────
def download(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """일봉 → 주봉 (Close 기준)"""
    return df["Close"].resample("W-FRI").last().dropna().to_frame("Close")


# ─────────────────────────────────────────────
# 데이터 준비
# ─────────────────────────────────────────────
print("데이터 다운로드 중...")
ixic = download("^IXIC")
qqq  = download("QQQ")
vix  = download("^VIX")

# IXIC MA200 & 이격도 (as-is 비교용, 고정 12%)
ixic["ma200"]    = ixic["Close"].rolling(MA_WINDOW).mean()
ixic["dist_ok"]  = ixic["Close"] > ixic["ma200"] * 1.12

# QQQ MA200 (이격도는 그리드에서 동적으로 계산)
qqq["ma200"]     = qqq["Close"].rolling(MA_WINDOW).mean()

# QQQ 일봉 RSI
qqq["rsi_d"]     = calc_rsi(qqq["Close"], RSI_PERIOD)
qqq["rsi_d_prev"]= qqq["rsi_d"].shift(1)

# QQQ 주봉 RSI → 일봉 인덱스에 forward-fill
qqq_weekly       = resample_weekly(qqq)
qqq_weekly["rsi_w"] = calc_rsi(qqq_weekly["Close"], RSI_PERIOD)
qqq["rsi_w"]     = qqq_weekly["rsi_w"].reindex(qqq.index, method="ffill")

# VIX 조건
vix["vix_prev"]  = vix["Close"].shift(1)
vix["vix_surge"] = ((vix["Close"] / vix["vix_prev"] - 1) * 100) >= VIX_SURGE_PCT
vix["vix_high"]  = vix["Close"] > VIX_HIGH
vix["vix_ok"]    = vix["vix_high"] | vix["vix_surge"]

# 공통 날짜 인덱스 (IXIC 기준, VIX & QQQ join)
common = ixic.index.intersection(vix.index).intersection(qqq.index)
common = common[common >= pd.Timestamp(EVAL_START)]

df = pd.DataFrame(index=common)
df["ixic_dist_ok"] = ixic.loc[common, "dist_ok"]
df["qqq_ma200"]    = qqq.loc[common, "ma200"]
df["vix_ok"]       = vix.loc[common, "vix_ok"]
df["rsi_d"]        = qqq.loc[common, "rsi_d"]
df["rsi_d_prev"]   = qqq.loc[common, "rsi_d_prev"]
df["rsi_w"]        = qqq.loc[common, "rsi_w"]
df["qqq_close"]    = qqq.loc[common, "Close"]

df = df.dropna(subset=["ixic_dist_ok", "qqq_ma200", "vix_ok"])

# ─────────────────────────────────────────────
# 시나리오별 시그널 생성 (이격도 그리드 × B/C/D)
# ─────────────────────────────────────────────
def build_scenarios(mult: float) -> dict:
    dist_ok = df["qqq_close"] > df["qqq_ma200"] * mult

    rsi_filter = (
        (df["rsi_d"] >= RSI_OVERBOUGHT) &
        (df["rsi_w"] >= RSI_OVERBOUGHT) &
        (df["rsi_d"] < df["rsi_d_prev"])
    )

    pct = int(round((mult - 1) * 100))
    return {
        f"B_{pct}%_VIX":     dist_ok & df["vix_ok"],
        f"C_{pct}%_VIX_RSI": dist_ok & df["vix_ok"] & rsi_filter,
        f"D_{pct}%_RSI":     dist_ok & rsi_filter,
    }

all_scenarios = {}
for m in DISTANCE_MULTS:
    all_scenarios.update(build_scenarios(m))


def apply_cooldown(raw_signal: pd.Series, cooldown: int) -> pd.Series:
    """시그널 쿨다운 적용"""
    result  = pd.Series(False, index=raw_signal.index)
    blocked = 0
    for date, val in raw_signal.items():
        if blocked > 0:
            blocked -= 1
            continue
        if val:
            result[date] = True
            blocked = cooldown
    return result


signals = {name: apply_cooldown(raw, COOLDOWN_DAYS) for name, raw in all_scenarios.items()}


# ─────────────────────────────────────────────
# 성과 측정
# ─────────────────────────────────────────────
qqq_close = df["qqq_close"]

def evaluate(signal_series: pd.Series) -> dict:
    dates = signal_series[signal_series].index
    rows  = []

    for sig_date in dates:
        fut = qqq_close[qqq_close.index > sig_date].head(max(EVAL_DAYS))
        if fut.empty:
            continue
        entry = qqq_close[sig_date]
        ret   = (fut / entry) - 1
        mdd   = {}
        for d in EVAL_DAYS:
            window = ret.head(d)
            mdd[d] = float(window.min()) if len(window) > 0 else np.nan

        is_tp = mdd.get(TP_EVAL_DAY, 0) <= -TP_THRESHOLD_PCT
        rows.append({
            "date":   sig_date,
            "mdd_10": mdd.get(10),
            "mdd_20": mdd.get(20),
            "mdd_30": mdd.get(30),
            "mdd_40": mdd.get(40),
            "is_tp":  is_tp,
        })

    if not rows:
        return {"n_signals": 0, "tp_rate": np.nan,
                "avg_mdd_10": np.nan, "avg_mdd_20": np.nan,
                "avg_mdd_30": np.nan, "avg_mdd_40": np.nan,
                "details": pd.DataFrame()}

    det = pd.DataFrame(rows)
    return {
        "n_signals":  len(det),
        "tp_rate":    det["is_tp"].mean() * 100,
        "avg_mdd_10": det["mdd_10"].mean() * 100,
        "avg_mdd_20": det["mdd_20"].mean() * 100,
        "avg_mdd_30": det["mdd_30"].mean() * 100,
        "avg_mdd_40": det["mdd_40"].mean() * 100,
        "details":    det,
    }


results = {name: evaluate(sig) for name, sig in signals.items()}


# ─────────────────────────────────────────────
# 실제 대형 하락 이벤트 추출
# 정의: QQQ가 직전 40 거래일 고점 대비 -10% 이상 낙폭 첫 진입 날짜
# ─────────────────────────────────────────────
CRASH_THRESHOLD  = 0.10   # -10% 이상
CRASH_PEAK_WINDOW = 40    # 직전 40일 고점 기준
CRASH_COOLDOWN   = 40     # 같은 하락 이벤트 중복 방지

rolling_peak = qqq_close.rolling(CRASH_PEAK_WINDOW).max()
drawdown     = qqq_close / rolling_peak - 1

crash_events = []
last_crash   = pd.Timestamp("2000-01-01")
for date, dd in drawdown.items():
    if dd <= -CRASH_THRESHOLD and (date - last_crash).days > CRASH_COOLDOWN * 1.4:
        crash_events.append(date)
        last_crash = date

print(f"\n실제 대형 하락 이벤트 (직전 {CRASH_PEAK_WINDOW}일 고점 대비 -{CRASH_THRESHOLD*100:.0f}% 이상)")
print(f"총 {len(crash_events)}회  |  연평균 {len(crash_events)/11:.1f}회  (2015~2025, 11년)")
print("-"*40)
for d in crash_events:
    dd_val = drawdown[d] * 100
    print(f"  {d.strftime('%Y-%m-%d')}  낙폭: {dd_val:.1f}%")


# ─────────────────────────────────────────────
# 시나리오별 Recall 계산
# 기준: 하락 이벤트 발생 전 30 거래일 내 시그널이 있었으면 커버된 것으로 간주
# ─────────────────────────────────────────────
COVER_WINDOW = 30   # 시그널이 이 기간 내에 있으면 "커버"

def calc_recall(signal_series: pd.Series) -> dict:
    sig_dates = set(signal_series[signal_series].index)
    covered   = 0
    detail    = []
    for crash_date in crash_events:
        # crash_date 이전 COVER_WINDOW 거래일 내 시그널 있는지 확인
        window_start = qqq_close.index[max(0, qqq_close.index.get_loc(crash_date) - COVER_WINDOW)]
        had_signal   = any(window_start <= s <= crash_date for s in sig_dates)
        if had_signal:
            covered += 1
        detail.append({"crash": crash_date.strftime("%Y-%m-%d"), "covered": had_signal})
    recall = covered / len(crash_events) * 100 if crash_events else 0
    return {"recall": recall, "covered": covered, "total": len(crash_events), "detail": detail}

recall_results = {name: calc_recall(sig) for name, sig in signals.items()}


# ─────────────────────────────────────────────
# 결과 출력 — 이격도 그리드 × B/C/D (TP율 + Recall 통합)
# ─────────────────────────────────────────────
print("\n" + "="*90)
print(f"  QQQ 이격도 그리드 × 조합 비교  |  {EVAL_START} ~ {END}")
print(f"  TP 기준: {TP_EVAL_DAY}일 내 -7%  |  Recall: 대형 하락 전 {COVER_WINDOW}일 내 시그널 존재 여부")
print("="*90)
print(f"{'시나리오':<30} {'시그널':>6} {'TP율':>7} {'Recall':>8} {'MDD30':>7} {'MDD40':>7}")
print("-"*90)

for mult in DISTANCE_MULTS:
    pct = int(round((mult - 1) * 100))
    print(f"\n── 이격도 +{pct}%  (MA200 × {mult:.2f}) ────────────────────────────────────────────")
    for tag in [f"B_{pct}%_VIX", f"C_{pct}%_VIX_RSI", f"D_{pct}%_RSI"]:
        r  = results.get(tag)
        rc = recall_results.get(tag)
        if r is None:
            continue
        label = {
            f"B_{pct}%_VIX":     f"  B  +{pct}%  QQQ + VIX",
            f"C_{pct}%_VIX_RSI": f"  C  +{pct}%  QQQ + VIX + RSI",
            f"D_{pct}%_RSI":     f"  D  +{pct}%  QQQ + RSI만",
        }[tag]
        if r["n_signals"] == 0:
            print(f"{label:<30} {'0':>6} {'N/A':>7} {'N/A':>8} {'N/A':>7} {'N/A':>7}")
        else:
            recall_str = f"{rc['recall']:.0f}% ({rc['covered']}/{rc['total']})"
            print(
                f"{label:<30} "
                f"{r['n_signals']:>6} "
                f"{r['tp_rate']:>6.1f}% "
                f"{recall_str:>13} "
                f"{r['avg_mdd_30']:>6.1f}% "
                f"{r['avg_mdd_40']:>6.1f}% "
            )

print("\n" + "="*90)


# ─────────────────────────────────────────────
# 차트 — TP율 히트맵 (이격도 × 조합)
# ─────────────────────────────────────────────
combos  = ["B (QQQ+VIX)", "C (QQQ+VIX+RSI)", "D (QQQ+RSI만)"]
tags_b  = ["B", "C", "D"]
pcts    = [int(round((m - 1) * 100)) for m in DISTANCE_MULTS]
suffixes= ["VIX", "VIX_RSI", "RSI"]

tp_matrix  = np.full((len(combos), len(pcts)), np.nan)
sig_matrix = np.full((len(combos), len(pcts)), 0)

for ci, (tag_prefix, suffix) in enumerate(zip(tags_b, suffixes)):
    for pi, pct in enumerate(pcts):
        key = f"{tag_prefix}_{pct}%_{suffix}"
        r   = results.get(key)
        if r and r["n_signals"] > 0:
            tp_matrix[ci, pi]  = r["tp_rate"]
            sig_matrix[ci, pi] = r["n_signals"]

fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(tp_matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

ax.set_xticks(range(len(pcts)))
ax.set_xticklabels([f"+{p}%" for p in pcts], fontsize=11)
ax.set_yticks(range(len(combos)))
ax.set_yticklabels(combos, fontsize=11)
ax.set_xlabel("QQQ MA200 이격도 기준", fontsize=11)
ax.set_title(f"TP율 히트맵  (TP={TP_EVAL_DAY}일 내 -{TP_THRESHOLD_PCT*100:.0f}%)  —  2015~2025", fontsize=12)

for ci in range(len(combos)):
    for pi in range(len(pcts)):
        val = tp_matrix[ci, pi]
        sig = sig_matrix[ci, pi]
        if np.isnan(val):
            txt = "N/A"
            color = "gray"
        else:
            txt   = f"{val:.0f}%\n({sig}회)"
            color = "black" if 20 < val < 80 else "white"
        ax.text(pi, ci, txt, ha="center", va="center", fontsize=9, color=color)

plt.colorbar(im, ax=ax, label="TP율 (%)")
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "backtest_ixic_vs_qqq_top_detection.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n차트 저장: {out_path}")
