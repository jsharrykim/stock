"""
backtest_ma200_depth.py — MA200 하방 이격 거리별 비교 백테스트
=============================================================

진입 조건 변형:
  기준: 현재가 < MA200 × (1 - depth%)
  depth = 0% (원본), 5%, 10%, 15%, 20%, 25%, 30%

나머지 조건 동일:
  VIX ≥ 25
  RSI < 40 OR CCI < -100
  진입가: 신호 발생일 다음날 시가

매도 조건 동일 (①②③④):
  ① +20% 목표 수익
  ② -25% 서킷브레이커
  ③ 60거래일 경과 & 수익 중
  ④ 120거래일 타임 익시트
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 파라미터
# ──────────────────────────────────────────────
START         = "2010-01-01"
END           = "2026-01-01"
VIX_MIN       = 25
TARGET_PCT    = 0.20
CIRCUIT_PCT   = 0.25
HALF_EXIT     = 60
MAX_HOLD      = 120
MAX_POSITIONS = 5
MAX_DAILY     = 5

DEPTHS = [0, 5, 10, 15, 20, 25, 30]   # MA200 대비 하방 이격 %

TICKERS = sorted(set([
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","COST",
    "AMD","QCOM","INTC","TXN","AMGN","INTU","AMAT","MU","LRCX","KLAC",
    "CDNS","SNPS","FTNT","PANW","MNST","ORLY","ISRG","PAYX","MELI",
    "PLTR","CPRT","NXPI","ON","CSX","ROP","ADP","ADI","BKNG",
    "MDLZ","AZN","FAST","MCHP",
]))

# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────
def dl_ohlcv(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw[["Open","High","Low","Close","Volume"]].copy()


def compute(d):
    d  = d.copy()
    c  = d["Close"]
    d["MA200"] = c.rolling(200).mean()
    delta = c.diff()
    g = delta.clip(lower=0).rolling(14).mean()
    l = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100 / (1 + g / l.replace(0, np.nan))
    tp     = (d["High"] + d["Low"] + c) / 3
    tp_ma  = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    d["CCI"] = (tp - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))
    # MA200 대비 이격률 (음수 = 아래)
    d["MA200_GAP"] = (c - d["MA200"]) / d["MA200"] * 100
    return d


# ──────────────────────────────────────────────
# 데이터 다운로드
# ──────────────────────────────────────────────
print("📥 VIX 다운로드 중...")
vix_raw = yf.download("^VIX", start=START, end=END,
                      auto_adjust=True, progress=False)
_vc = vix_raw["Close"]
if isinstance(_vc, pd.DataFrame):
    _vc = _vc.iloc[:, 0]
vix = pd.Series(_vc.values, index=_vc.index, dtype=float).dropna()
print(f"✅ VIX {len(vix)}일")

print("📥 종목 OHLCV 다운로드 중...")
stock_data = {}
for i, tk in enumerate(TICKERS, 1):
    d = dl_ohlcv(tk, START, END)
    if len(d) < 250:
        continue
    d = compute(d)
    stock_data[tk] = d
    if i % 10 == 0:
        print(f"  {i}/{len(TICKERS)} 완료")
print(f"✅ {len(stock_data)}개 종목 로드")


# ──────────────────────────────────────────────
# 신호 생성 — depth별
# ──────────────────────────────────────────────
def build_signals(depth_pct):
    """
    depth_pct: 0 → 현재가 < MA200
               5 → 현재가 < MA200 × 0.95  (5% 이상 아래)
    """
    threshold = -depth_pct   # MA200_GAP 기준 (음수)
    signals_by_date = {}

    for tk, d in stock_data.items():
        d_c    = d.dropna(subset=["MA200", "RSI", "CCI", "MA200_GAP"])
        common = d_c.index.intersection(vix.index)
        if len(common) < 50:
            continue
        vx = vix.reindex(common)

        gap  = d_c["MA200_GAP"].reindex(common)
        rsi  = d_c["RSI"].reindex(common)
        cci  = d_c["CCI"].reindex(common)

        cond = (
            (gap <= threshold) &          # MA200 대비 depth% 이상 아래
            (vx >= VIX_MIN) &
            ((rsi < 40) | (cci < -100))
        )
        for sig_day in common[cond.reindex(common).fillna(False)]:
            idx = d_c.index.get_loc(sig_day)
            if idx + 1 >= len(d_c):
                continue
            entry_day  = d_c.index[idx + 1]
            entry_open = float(d_c["Open"].iloc[idx + 1])
            if pd.isna(entry_open):
                continue
            row = d_c.loc[sig_day]
            signals_by_date.setdefault(entry_day, []).append({
                "ticker"    : tk,
                "sig_day"   : sig_day,
                "entry_day" : entry_day,
                "entry"     : entry_open,
                "rsi"       : float(row["RSI"]),
                "cci"       : float(row["CCI"]),
                "close_sig" : float(row["Close"]),
                "ma200_gap" : float(row["MA200_GAP"]),
            })

    final = []
    for entry_day, items in sorted(signals_by_date.items()):
        items.sort(key=lambda x: x["rsi"])
        final.extend(items[:MAX_DAILY])
    return final


# ──────────────────────────────────────────────
# 시뮬레이션
# ──────────────────────────────────────────────
def run_simulation(signals):
    trades        = []
    pos_exit_date = {}

    for sig in signals:
        tk        = sig["ticker"]
        entry_day = sig["entry_day"]
        entry     = sig["entry"]

        active = {t: ed for t, ed in pos_exit_date.items() if ed > entry_day}
        if len(active) >= MAX_POSITIONS or tk in active:
            continue

        d      = stock_data[tk]
        future = d.loc[d.index >= entry_day]
        if len(future) == 0:
            continue

        circuit     = entry * (1 - CIRCUIT_PCT)
        target      = entry * (1 + TARGET_PCT)
        half_exited = False
        exit_records= []

        for i, (fdt, row) in enumerate(future.iterrows()):
            lo = float(row["Low"])
            hi = float(row["High"])
            cl = float(row["Close"])

            if hi >= target:
                exit_records.append((fdt, target, 0.5 if half_exited else 1.0, "target"))
                break
            if lo <= circuit:
                exit_records.append((fdt, circuit, 0.5 if half_exited else 1.0, "circuit"))
                break
            if i + 1 == HALF_EXIT and not half_exited and (cl - entry) / entry > 0:
                exit_records.append((fdt, cl, 0.5, "half_60d"))
                half_exited = True
                continue
            if i + 1 >= MAX_HOLD:
                exit_records.append((fdt, cl, 0.5 if half_exited else 1.0, "time"))
                break

        if not exit_records:
            continue

        total_pct   = sum(r[2] for r in exit_records)
        blended_ret = sum((r[1] - entry) / entry * r[2] for r in exit_records) / total_pct
        last_exit   = exit_records[-1]
        reason      = ("+".join(r[3] for r in exit_records)
                       if len(exit_records) > 1 else exit_records[0][3])

        trades.append({
            "entry_date" : entry_day,
            "exit_date"  : last_exit[0],
            "ticker"     : tk,
            "entry"      : entry,
            "exit_price" : last_exit[1],
            "return_pct" : blended_ret * 100,
            "hold_days"  : (last_exit[0] - entry_day).days,
            "exit_reason": reason,
            "win"        : blended_ret > 0,
            "rsi_entry"  : sig["rsi"],
            "ma200_gap"  : sig["ma200_gap"],
        })
        pos_exit_date[tk] = last_exit[0]

    return pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)


# ──────────────────────────────────────────────
# 통계 계산
# ──────────────────────────────────────────────
def calc_stats(df):
    if df.empty:
        return {}
    wins   = df[df["win"]]
    losses = df[~df["win"]]
    n      = len(df)
    wr     = len(wins) / n * 100
    ar     = df["return_pct"].mean()
    aw     = wins["return_pct"].mean()   if len(wins)   else 0
    al     = losses["return_pct"].mean() if len(losses) else 0
    pf     = (wins["return_pct"].sum() / -losses["return_pct"].sum()
              if len(losses) and losses["return_pct"].sum() < 0 else np.nan)
    ev     = wr / 100 * aw + (1 - wr / 100) * al

    max_cl = cur = 0
    for r in df["return_pct"]:
        cur = cur + 1 if r < 0 else 0
        max_cl = max(max_cl, cur)

    capital = 1.0
    for _, group in df.groupby("entry_date"):
        w     = 1.0 / max(len(group), MAX_POSITIONS)
        batch = (group["return_pct"] / 100 * w).sum()
        capital *= (1 + batch)
    years = (df["exit_date"].max() - df["entry_date"].min()).days / 365.25
    cagr  = capital ** (1 / max(years, 0.01)) - 1

    return {
        "n": n, "wr": wr, "ar": ar, "aw": aw, "al": al,
        "pf": pf, "ev": ev, "max_cl": max_cl,
        "cagr": cagr * 100,
        "avg_hold"  : df["hold_days"].mean(),
        "avg_gap"   : df["ma200_gap"].mean(),
        "exit_cnt"  : df["exit_reason"].value_counts(),
    }


# ──────────────────────────────────────────────
# 전 depth 실행
# ──────────────────────────────────────────────
print("\n⚙️  depth별 시뮬레이션 시작...")
results = {}
for depth in DEPTHS:
    sigs = build_signals(depth)
    df   = run_simulation(sigs)
    stats= calc_stats(df)
    results[depth] = {"df": df, "stats": stats, "n_sig": len(sigs)}
    label = f"{depth}%" if depth > 0 else "0% (원본)"
    print(f"  MA200 -{label:>10} | 신호 {len(sigs):>5}건 → 트레이드 {len(df):>4}건 "
          f"| 승률 {stats['wr']:.1f}% | 평균 {stats['ar']:+.2f}% | CAGR {stats['cagr']:+.2f}%")


# ──────────────────────────────────────────────
# 결과 출력
# ──────────────────────────────────────────────
print("\n" + "="*95)
print("  MA200 하방 이격 거리별 백테스트 비교")
print("="*95)
print(f"  기간      : {START} ~ {END}")
print(f"  공통 조건 : VIX≥25 + (RSI<40 OR CCI<-100), 다음날 시가 진입")
print(f"  매도 조건 : +20% / -25%CB / 60일절반 / 120일")
print("="*95)

hdr_fmt = "  {:<12} {:>8} {:>8} {:>9} {:>9} {:>9} {:>9} {:>8} {:>9} {:>8}"
row_fmt = "  {:<12} {:>8} {:>8} {:>9} {:>9} {:>9} {:>9} {:>8} {:>9} {:>8}"
print(hdr_fmt.format(
    "조건", "신호수", "거래수", "승률", "평균수익", "승자평균", "패자평균", "PF", "CAGR", "평균보유"))
print("  " + "-"*91)

for depth in DEPTHS:
    s     = results[depth]["stats"]
    n_sig = results[depth]["n_sig"]
    label = f"-{depth}%" if depth > 0 else "원본(<MA200)"
    pf_s  = f"{s['pf']:.2f}" if not (isinstance(s['pf'], float) and np.isnan(s['pf'])) else "N/A"
    print(row_fmt.format(
        label,
        f"{n_sig:,}",
        f"{s['n']}",
        f"{s['wr']:.1f}%",
        f"{s['ar']:+.2f}%",
        f"{s['aw']:+.2f}%",
        f"{s['al']:+.2f}%",
        pf_s,
        f"{s['cagr']:+.2f}%",
        f"{s['avg_hold']:.0f}일",
    ))
print("="*95)

# 연도별 상세
for depth in DEPTHS:
    df    = results[depth]["df"].copy()
    label = f"-{depth}%" if depth > 0 else "원본"
    df["year"] = df["entry_date"].dt.year
    y = df.groupby("year").agg(
        trades  =("return_pct","count"),
        avg_ret =("return_pct","mean"),
        win_rate=("win", lambda x: x.mean()*100),
    )
    print(f"\n연도별 [{label}]:")
    print(y.round(2).to_string())


# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
palette = ["#2c3e50","#2980b9","#27ae60","#f39c12","#e74c3c","#8e44ad","#16a085"]

fig, axes = plt.subplots(3, 3, figsize=(22, 17))
fig.suptitle(
    f"MA200 하방 이격 거리별 진입 비교 (VIX≥25 + RSI<40/CCI<-100)\n"
    f"기간: {START} ~ {END}  |  종목: {len(stock_data)}개",
    fontsize=11, fontweight="bold"
)

# ── [0,0] 핵심 지표 — 승률
ax = axes[0, 0]
wrs   = [results[d]["stats"]["wr"]   for d in DEPTHS]
xlbls = [f"-{d}%" if d > 0 else "0%\n(원본)" for d in DEPTHS]
bars  = ax.bar(xlbls, wrs, color=palette, edgecolor="white", alpha=0.85)
ax.axhline(wrs[0], color="gray", linestyle="--", lw=1, label=f"원본 {wrs[0]:.1f}%")
for bar, v in zip(bars, wrs):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
            f"{v:.1f}%", ha="center", fontsize=8.5)
ax.set_title("승률 비교")
ax.set_ylabel("%")
ax.set_ylim(max(0, min(wrs) - 10), 100)
ax.legend(fontsize=8)

# ── [0,1] 평균 수익률
ax = axes[0, 1]
ars   = [results[d]["stats"]["ar"] for d in DEPTHS]
bars  = ax.bar(xlbls, ars, color=palette, edgecolor="white", alpha=0.85)
ax.axhline(ars[0], color="gray", linestyle="--", lw=1, label=f"원본 {ars[0]:+.2f}%")
ax.axhline(0, color="black", lw=0.8)
for bar, v in zip(bars, ars):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.2,
            f"{v:+.2f}%", ha="center", fontsize=8)
ax.set_title("평균 수익률 비교")
ax.set_ylabel("%")
ax.legend(fontsize=8)

# ── [0,2] 포트CAGR
ax = axes[0, 2]
cagrs = [results[d]["stats"]["cagr"] for d in DEPTHS]
bars  = ax.bar(xlbls, cagrs, color=palette, edgecolor="white", alpha=0.85)
ax.axhline(cagrs[0], color="gray", linestyle="--", lw=1, label=f"원본 {cagrs[0]:+.2f}%")
ax.axhline(0, color="black", lw=0.8)
for bar, v in zip(bars, cagrs):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
            f"{v:+.2f}%", ha="center", fontsize=8)
ax.set_title("포트 CAGR 비교")
ax.set_ylabel("%")
ax.legend(fontsize=8)

# ── [1,0] 거래 건수 & 신호 건수
ax = axes[1, 0]
n_sigs   = [results[d]["n_sig"] for d in DEPTHS]
n_trades = [results[d]["stats"]["n"] for d in DEPTHS]
x = np.arange(len(DEPTHS))
w = 0.35
ax.bar(x - w/2, n_sigs,   w, label="신호 건수", color="#3498db", alpha=0.7)
ax.bar(x + w/2, n_trades, w, label="거래 건수", color="#e67e22", alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(xlbls, fontsize=8)
for i, (ns, nt) in enumerate(zip(n_sigs, n_trades)):
    ax.text(i - w/2, ns + 10, str(ns), ha="center", fontsize=7)
    ax.text(i + w/2, nt + 10, str(nt), ha="center", fontsize=7)
ax.set_title("신호 건수 vs 거래 건수")
ax.legend(fontsize=8)

# ── [1,1] 승자/패자 평균
ax = axes[1, 1]
aws = [results[d]["stats"]["aw"] for d in DEPTHS]
als = [results[d]["stats"]["al"] for d in DEPTHS]
ax.plot(xlbls, aws, "o-", color="#27ae60", lw=2, label="승자 평균", markersize=7)
ax.plot(xlbls, als, "s-", color="#e74c3c", lw=2, label="패자 평균", markersize=7)
ax.axhline(0, color="black", lw=0.8, linestyle="--")
for i, (aw, al) in enumerate(zip(aws, als)):
    ax.text(i, aw + 0.5, f"{aw:+.1f}%", ha="center", fontsize=7.5, color="#27ae60")
    ax.text(i, al - 1.5, f"{al:+.1f}%", ha="center", fontsize=7.5, color="#e74c3c")
ax.set_title("승자/패자 평균 수익률")
ax.set_ylabel("%")
ax.legend(fontsize=8)

# ── [1,2] 평균 보유 일수 & Profit Factor
ax = axes[1, 2]
ax2 = ax.twinx()
holds = [results[d]["stats"]["avg_hold"] for d in DEPTHS]
pfs   = [results[d]["stats"]["pf"] if not np.isnan(results[d]["stats"]["pf"]) else 0
         for d in DEPTHS]
ax.bar(xlbls,  holds, color="#9b59b6", alpha=0.6, label="평균 보유(일)")
ax2.plot(xlbls, pfs,  "D-", color="#e67e22", lw=2, markersize=7, label="Profit Factor")
for i, (h, p) in enumerate(zip(holds, pfs)):
    ax.text(i, h + 0.5, f"{h:.0f}일", ha="center", fontsize=7.5, color="#9b59b6")
    ax2.text(i, p + 0.05, f"{p:.2f}", ha="center", fontsize=7.5, color="#e67e22")
ax.set_title("평균 보유 일수 & Profit Factor")
ax.set_ylabel("보유 일수", color="#9b59b6")
ax2.set_ylabel("Profit Factor", color="#e67e22")
lines1, lbs1 = ax.get_legend_handles_labels()
lines2, lbs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, lbs1 + lbs2, fontsize=8)

# ── [2,0] 누적 수익률 곡선 비교
ax = axes[2, 0]
for i, depth in enumerate(DEPTHS):
    df  = results[depth]["df"]
    cum = (1 + df["return_pct"] / 100).cumprod() - 1
    lbl = f"-{depth}% (CAGR {results[depth]['stats']['cagr']:+.1f}%)" if depth > 0 \
          else f"원본 (CAGR {results[depth]['stats']['cagr']:+.1f}%)"
    ax.plot(range(len(cum)), cum * 100, color=palette[i], lw=1.5, label=lbl, alpha=0.85)
ax.axhline(0, color="black", linestyle="--", lw=1)
ax.set_title("누적 수익률 비교 (Trade 순)")
ax.set_xlabel("Trade #")
ax.set_ylabel("Cum. Return %")
ax.legend(fontsize=7)

# ── [2,1] 수익률 분포 (박스플롯)
ax = axes[2, 1]
data_bp = [results[d]["df"]["return_pct"].values for d in DEPTHS]
bp = ax.boxplot(data_bp, labels=xlbls, patch_artist=True,
                medianprops={"color":"black","lw":2})
for patch, c in zip(bp["boxes"], palette):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.axhline(0, color="red", linestyle="--", lw=1)
ax.set_title("수익률 분포 (Boxplot)")
ax.set_ylabel("Return %")
ax.tick_params(axis="x", labelsize=8)

# ── [2,2] 연도별 평균 수익률 히트맵
ax = axes[2, 2]
all_years = sorted(set(
    y for d in DEPTHS
    for y in results[d]["df"]["entry_date"].dt.year.unique()
))
heatmap_data = []
for depth in DEPTHS:
    df2 = results[depth]["df"].copy()
    df2["year"] = df2["entry_date"].dt.year
    yr_avg = df2.groupby("year")["return_pct"].mean()
    row = [yr_avg.get(y, np.nan) for y in all_years]
    heatmap_data.append(row)

hm = np.array(heatmap_data, dtype=float)
im = ax.imshow(hm, aspect="auto", cmap="RdYlGn", vmin=-15, vmax=25)
ax.set_xticks(range(len(all_years)))
ax.set_xticklabels([str(y) for y in all_years], rotation=45, fontsize=7)
ax.set_yticks(range(len(DEPTHS)))
ax.set_yticklabels([f"-{d}%" if d > 0 else "원본" for d in DEPTHS], fontsize=8)
ax.set_title("연도별 평균 수익률 히트맵")
for i in range(len(DEPTHS)):
    for j in range(len(all_years)):
        v = hm[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=6.5, color="black" if abs(v) < 15 else "white")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("backtest_ma200_depth.png", dpi=150, bbox_inches="tight")
for depth in DEPTHS:
    label = f"{depth}" if depth > 0 else "0_original"
    results[depth]["df"].to_csv(f"backtest_ma200_depth_{label}pct.csv", index=False)
print("\n📊 backtest_ma200_depth.png 저장 완료")
print("📄 CSV 7개 저장 완료")
print("✅ MA200 이격 거리별 백테스트 완료")
