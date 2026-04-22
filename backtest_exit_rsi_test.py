"""
backtest_exit_rsi_test.py — ② 타임 익시트 개선 + ④ RSI/CCI 조건 강화 비교
=============================================================================

② 타임 익시트 개선 — 120일 손실 보유 문제 해결
  현재: 120거래일 무조건 청산
  변형들:
    B1: 120일 도달 시 수익일 때만 청산 (손실이면 계속 보유 → 최대 180일)
    B2: 90일로 단축 (수익/손실 무관)
    B3: 60일 절반 EXIT 제거 (단순화: 목표/CB/120일만)

④ RSI/CCI 조건 강화
  현재: RSI < 40 OR CCI < -100
  변형들:
    R1: RSI < 35 OR CCI < -150  (더 깊은 과매도)
    R2: RSI < 30 OR CCI < -150  (극단 과매도)
    R3: RSI < 40 AND CCI < -100 (OR → AND, 둘 다 충족)

원본과 함께 총 7그룹 비교
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

START         = "2010-01-01"
END           = "2026-01-01"
VIX_MIN       = 25
TARGET_PCT    = 0.20
CIRCUIT_PCT   = 0.25
MAX_POSITIONS = 5
MAX_DAILY     = 5

TICKERS = sorted(set([
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","COST",
    "AMD","QCOM","INTC","TXN","AMGN","INTU","AMAT","MU","LRCX","KLAC",
    "CDNS","SNPS","FTNT","PANW","MNST","ORLY","ISRG","PAYX","MELI",
    "PLTR","CPRT","NXPI","ON","CSX","ROP","ADP","ADI","BKNG",
    "MDLZ","AZN","FAST","MCHP",
]))

# ──────────────────────────────────────────────
# 그룹 정의
# ──────────────────────────────────────────────
GROUPS = {
    # label: (rsi_thresh, cci_thresh, rsi_cci_and, half_exit_days, max_hold, ext_hold)
    # ext_hold: 손실 시 연장 최대 보유일 (None이면 max_hold에서 무조건 청산)
    "원본":          (40, -100, False, 60,  120, None),
    "B1 (120→손익분기)": (40, -100, False, 60,  120, 180),  # 손실이면 180일까지 연장
    "B2 (90일 단축)":    (40, -100, False, 60,   90, None),
    "B3 (절반EXIT 제거)":(40, -100, False, None,120, None),  # half_exit 없음
    "R1 (RSI<35/CCI<-150)": (35, -150, False, 60, 120, None),
    "R2 (RSI<30/CCI<-150)": (30, -150, False, 60, 120, None),
    "R3 (RSI AND CCI)":     (40, -100, True,  60, 120, None),
}

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
    return d


# ──────────────────────────────────────────────
# 데이터
# ──────────────────────────────────────────────
print("📥 다운로드 중...")
vix_raw = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)
_vc = vix_raw["Close"]
if isinstance(_vc, pd.DataFrame): _vc = _vc.iloc[:, 0]
vix = pd.Series(_vc.values, index=_vc.index, dtype=float).dropna()

stock_data = {}
for i, tk in enumerate(TICKERS, 1):
    d = dl_ohlcv(tk, START, END)
    if len(d) < 250: continue
    stock_data[tk] = compute(d)
    if i % 10 == 0: print(f"  {i}/{len(TICKERS)} 완료")
print(f"✅ VIX {len(vix)}일 | 종목 {len(stock_data)}개")


# ──────────────────────────────────────────────
# 신호 생성 (RSI/CCI 조건별)
# ──────────────────────────────────────────────
def build_signals(rsi_thresh, cci_thresh, use_and, label):
    signals_by_date = {}
    for tk, d in stock_data.items():
        d_c    = d.dropna(subset=["MA200","RSI","CCI"])
        common = d_c.index.intersection(vix.index)
        if len(common) < 50: continue
        vx    = vix.reindex(common)
        close = d_c["Close"].reindex(common)
        ma200 = d_c["MA200"].reindex(common)
        rsi   = d_c["RSI"].reindex(common)
        cci   = d_c["CCI"].reindex(common)

        rsi_ok = rsi < rsi_thresh
        cci_ok = cci < cci_thresh
        osc_ok = (rsi_ok & cci_ok) if use_and else (rsi_ok | cci_ok)

        cond = (close < ma200) & (vx >= VIX_MIN) & osc_ok

        for sig_day in common[cond.reindex(common).fillna(False)]:
            idx = d_c.index.get_loc(sig_day)
            if idx + 1 >= len(d_c): continue
            entry_day  = d_c.index[idx + 1]
            entry_open = float(d_c["Open"].iloc[idx + 1])
            if pd.isna(entry_open): continue
            row = d_c.loc[sig_day]
            signals_by_date.setdefault(entry_day, []).append({
                "ticker"   : tk,
                "entry_day": entry_day,
                "entry"    : entry_open,
                "rsi"      : float(row["RSI"]),
                "cci"      : float(row["CCI"]),
            })

    final = []
    for entry_day, items in sorted(signals_by_date.items()):
        items.sort(key=lambda x: x["rsi"])
        final.extend(items[:MAX_DAILY])
    print(f"  [{label}] 신호 {len(final):,}건")
    return final


# ──────────────────────────────────────────────
# 시뮬레이션 (exit 조건별)
# ──────────────────────────────────────────────
def run_simulation(signals, half_exit_days, max_hold, ext_hold, label):
    """
    half_exit_days: None이면 절반 익시트 없음
    max_hold: 기본 최대 보유일
    ext_hold: 손실 시 연장 최대 보유일 (None이면 max_hold에서 무조건 청산)
    """
    trades        = []
    pos_exit_date = {}

    for sig in signals:
        tk        = sig["ticker"]
        entry_day = sig["entry_day"]
        entry     = sig["entry"]
        active = {t: ed for t, ed in pos_exit_date.items() if ed > entry_day}
        if len(active) >= MAX_POSITIONS or tk in active: continue
        d      = stock_data[tk]
        future = d.loc[d.index >= entry_day]
        if len(future) == 0: continue

        circuit     = entry * (1 - CIRCUIT_PCT)
        target      = entry * (1 + TARGET_PCT)
        actual_max  = ext_hold if ext_hold else max_hold
        half_exited = False
        exit_records= []

        for i, (fdt, row) in enumerate(future.iterrows()):
            lo = float(row["Low"]); hi = float(row["High"]); cl = float(row["Close"])
            ret = (cl - entry) / entry

            if hi >= target:
                exit_records.append((fdt, target, 0.5 if half_exited else 1.0, "target")); break
            if lo <= circuit:
                exit_records.append((fdt, circuit, 0.5 if half_exited else 1.0, "circuit")); break

            # 절반 익시트
            if half_exit_days and i + 1 == half_exit_days and not half_exited and ret > 0:
                exit_records.append((fdt, cl, 0.5, "half_60d")); half_exited = True; continue

            # 기본 max_hold 도달
            if i + 1 == max_hold:
                if ext_hold and ret < 0:
                    # 손실이면 연장 (ext_hold까지 계속)
                    continue
                else:
                    exit_records.append((fdt, cl, 0.5 if half_exited else 1.0, "time")); break

            # 연장 max (ext_hold)
            if ext_hold and i + 1 >= ext_hold:
                exit_records.append((fdt, cl, 0.5 if half_exited else 1.0, "time_ext")); break

        if not exit_records: continue

        total_pct   = sum(r[2] for r in exit_records)
        blended_ret = sum((r[1]-entry)/entry * r[2] for r in exit_records) / total_pct
        last_exit   = exit_records[-1]
        reason      = "+".join(r[3] for r in exit_records) if len(exit_records) > 1 else exit_records[0][3]

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
        })
        pos_exit_date[tk] = last_exit[0]

    df = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)
    print(f"  [{label}] 트레이드 {len(df)}건")
    return df


# ──────────────────────────────────────────────
# 통계
# ──────────────────────────────────────────────
def calc_stats(df):
    if df.empty: return {}
    wins = df[df["win"]]; losses = df[~df["win"]]; n = len(df)
    wr  = len(wins) / n * 100
    ar  = df["return_pct"].mean()
    aw  = wins["return_pct"].mean()   if len(wins)   else 0
    al  = losses["return_pct"].mean() if len(losses) else 0
    pf  = (wins["return_pct"].sum() / -losses["return_pct"].sum()
           if len(losses) and losses["return_pct"].sum() < 0 else np.nan)
    ev  = wr/100 * aw + (1 - wr/100) * al
    max_cl = cur = 0
    for r in df["return_pct"]:
        cur = cur+1 if r < 0 else 0; max_cl = max(max_cl, cur)
    capital = 1.0
    for _, g in df.groupby("entry_date"):
        capital *= (1 + (g["return_pct"]/100 * (1/max(len(g), MAX_POSITIONS))).sum())
    years = (df["exit_date"].max() - df["entry_date"].min()).days / 365.25
    cagr  = capital ** (1/max(years, 0.01)) - 1
    exit_detail = df.groupby("exit_reason")["return_pct"].agg(["count","mean"]).sort_values("count", ascending=False)
    return {
        "n": n, "wr": wr, "ar": ar, "aw": aw, "al": al,
        "pf": pf, "ev": ev, "max_cl": max_cl,
        "cagr"       : cagr * 100,
        "avg_hold"   : df["hold_days"].mean(),
        "med_hold"   : df["hold_days"].median(),
        "exit_cnt"   : df["exit_reason"].value_counts(),
        "exit_detail": exit_detail,
    }


# ──────────────────────────────────────────────
# 전 그룹 실행
# ──────────────────────────────────────────────
print("\n🔍 신호 생성 + 시뮬레이션...")
results = {}
for label, (rsi_t, cci_t, use_and, half_d, max_h, ext_h) in GROUPS.items():
    sigs = build_signals(rsi_t, cci_t, use_and, label)
    df   = run_simulation(sigs, half_d, max_h, ext_h, label)
    results[label] = {"df": df, "stats": calc_stats(df), "n_sig": len(sigs)}


# ──────────────────────────────────────────────
# 결과 출력
# ──────────────────────────────────────────────
group_labels = list(GROUPS.keys())

print("\n" + "="*130)
print("  ② 타임 익시트 개선 + ④ RSI/CCI 조건 강화 비교")
print("="*130)
print(f"  기간: {START}~{END}  |  공통: MA200↓ + VIX≥25, 다음날 시가")
print("="*130)

# 원본 + 타임 익시트 비교 (②)
exit_groups = ["원본", "B1 (120→손익분기)", "B2 (90일 단축)", "B3 (절반EXIT 제거)"]
rsi_groups  = ["원본", "R1 (RSI<35/CCI<-150)", "R2 (RSI<30/CCI<-150)", "R3 (RSI AND CCI)"]

def print_comparison(title, group_list):
    print(f"\n{'─'*100}")
    print(f"  {title}")
    print(f"{'─'*100}")
    col_w = 20
    fmt = f"  {{:<26}}" + f" {{:>{col_w}}}" * len(group_list)
    print(fmt.format("지표", *group_list))
    print("  " + "-"*(26 + col_w * len(group_list) + len(group_list)))

    pf_s = lambda v: f"{v:.2f}" if not (isinstance(v,float) and np.isnan(v)) else "N/A"
    stat_rows = [
        ("신호 건수",        lambda s, l: f"{results[l]['n_sig']:,}건"),
        ("총 트레이드",      lambda s, l: f"{s['n']}건"),
        ("승 률",            lambda s, l: f"{s['wr']:.1f}%"),
        ("평균 수익률",      lambda s, l: f"{s['ar']:+.2f}%"),
        ("기대값(EV)",       lambda s, l: f"{s['ev']:+.2f}%"),
        ("승자 평균",        lambda s, l: f"{s['aw']:+.2f}%"),
        ("패자 평균",        lambda s, l: f"{s['al']:+.2f}%"),
        ("Profit Factor",    lambda s, l: pf_s(s['pf'])),
        ("포트CAGR",         lambda s, l: f"{s['cagr']:+.2f}%"),
        ("최대 연속 손실",   lambda s, l: f"{s['max_cl']}건"),
        ("평균 보유 일수",   lambda s, l: f"{s['avg_hold']:.0f}일"),
        ("중간값 보유",      lambda s, l: f"{s['med_hold']:.0f}일"),
    ]
    for row_label, fn in stat_rows:
        vals = [fn(results[l]["stats"], l) for l in group_list]
        print(fmt.format(row_label, *vals))

    for lbl in group_list:
        s = results[lbl]["stats"]
        print(f"\n  [{lbl}] 청산 유형:")
        for r in s["exit_cnt"].index:
            cnt = s["exit_cnt"][r]; avg = s["exit_detail"].loc[r,"mean"]
            mark = ""
            if "time" in r:    mark = "  ← 타임 익시트"
            if "circuit" in r: mark = "  ← 손절"
            print(f"    {r:<32}: {cnt:4d}건 ({cnt/s['n']*100:.1f}%)  avg {avg:+.2f}%{mark}")

print_comparison("② 타임 익시트 조건 변형 비교", exit_groups)
print_comparison("④ RSI/CCI 과매도 조건 강화 비교", rsi_groups)

# 연도별 히트맵용
print("\n\n  연도별 평균 수익률:")
for lbl in group_labels:
    df2 = results[lbl]["df"].copy(); df2["year"] = df2["entry_date"].dt.year
    y = df2.groupby("year").agg(trades=("return_pct","count"),
                                 avg_ret=("return_pct","mean"),
                                 win_rate=("win", lambda x: x.mean()*100))
    print(f"\n  [{lbl}]"); print(y.round(1).to_string())


# ──────────────────────────────────────────────
# 시각화 — 2개 섹션 (② / ④)
# ──────────────────────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(24, 20))
fig.suptitle(
    f"② 타임 익시트 개선 + ④ RSI/CCI 조건 강화 비교  |  {START}~{END}\n"
    f"공통: MA200↓ + VIX≥25 + 다음날 시가 진입  |  매도: +20% / -25%CB",
    fontsize=10, fontweight="bold"
)

EXIT_COLORS = ["#2c3e50","#2980b9","#27ae60","#e67e22"]
RSI_COLORS  = ["#2c3e50","#8e44ad","#e74c3c","#16a085"]

clr_exit = {"target":"#27ae60","circuit":"#e74c3c","time":"#f39c12",
            "half_60d":"#3498db","half_60d+target":"#1abc9c",
            "half_60d+time":"#e67e22","time_ext":"#95a5a6"}
def get_clr(k):
    for ck, cv in clr_exit.items():
        if ck in k: return cv
    return "#bdc3c7"

def plot_section(row_offset, group_list, colors, section_title):
    # [row,0] 수익률 분포
    ax = axes[row_offset, 0]
    ax.set_title(f"{section_title} — 수익률 분포")
    for lbl, c in zip(group_list, colors):
        s = results[lbl]["stats"]
        ax.hist(results[lbl]["df"]["return_pct"], bins=35, alpha=0.5,
                color=c, edgecolor="white",
                label=f"{lbl[:18]} (avg {s['ar']:+.1f}%)")
    ax.axvline(0, color="black", lw=1.5, linestyle="--")
    ax.legend(fontsize=6.5)

    # [row,1] 핵심 지표 막대 (승률, 평균수익, CAGR)
    ax = axes[row_offset, 1]
    metrics = ["승률(%)", "평균수익(%)", "CAGR(%)"]
    x = np.arange(len(metrics)); wid = 0.2
    offsets = np.linspace(-(len(group_list)-1)/2*wid, (len(group_list)-1)/2*wid, len(group_list))
    for lbl, c, off in zip(group_list, colors, offsets):
        s = results[lbl]["stats"]
        vals = [s["wr"], s["ar"], s["cagr"]]
        bars = ax.bar(x + off, vals, wid, color=c, alpha=0.85,
                      label=lbl[:15], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.3,
                    f"{v:.0f}", ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"{section_title} — 핵심 지표")
    ax.legend(fontsize=6.5)

    # [row,2] 누적 수익률
    ax = axes[row_offset, 2]
    for lbl, c in zip(group_list, colors):
        df = results[lbl]["df"]
        cum = (1 + df["return_pct"]/100).cumprod() - 1
        s   = results[lbl]["stats"]
        ax.plot(range(len(cum)), cum*100, color=c, lw=1.8,
                label=f"{lbl[:15]} CAGR {s['cagr']:+.1f}%")
    ax.axhline(0, color="black", linestyle="--", lw=1)
    ax.set_title(f"{section_title} — 누적 수익률")
    ax.set_xlabel("Trade #"); ax.legend(fontsize=6.5)

    # [row,3] 보유 기간 박스플롯
    ax = axes[row_offset, 3]
    data_bp = [results[lbl]["df"]["hold_days"].values for lbl in group_list]
    bp = ax.boxplot(data_bp, labels=[l[:12] for l in group_list],
                    patch_artist=True, medianprops={"color":"black","lw":2})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_title(f"{section_title} — 보유 기간")
    ax.set_ylabel("Hold Days"); ax.tick_params(axis="x", labelsize=7)


plot_section(0, exit_groups, EXIT_COLORS, "② 타임 익시트")
plot_section(1, exit_groups, EXIT_COLORS, "② 타임 익시트")  # 2행은 연도별
plot_section(2, rsi_groups,  RSI_COLORS,  "④ RSI/CCI 강화")
plot_section(3, rsi_groups,  RSI_COLORS,  "④ RSI/CCI 강화")  # 4행은 연도별

# 1행, 2행 → 연도별 히트맵으로 교체
for row_offset, group_list, colors, section_title in [
    (1, exit_groups, EXIT_COLORS, "② 타임 익시트"),
    (3, rsi_groups,  RSI_COLORS,  "④ RSI/CCI 강화"),
]:
    all_years = sorted(set(
        y for lbl in group_list
        for y in results[lbl]["df"]["entry_date"].dt.year.unique()
    ))
    hm = []
    for lbl in group_list:
        df2 = results[lbl]["df"].copy(); df2["year"] = df2["entry_date"].dt.year
        yr_avg = df2.groupby("year")["return_pct"].mean()
        hm.append([yr_avg.get(y, np.nan) for y in all_years])
    hm = np.array(hm, dtype=float)

    for col_idx in range(4):
        ax = axes[row_offset, col_idx]
        if col_idx == 0:
            im = ax.imshow(hm, aspect="auto", cmap="RdYlGn", vmin=-10, vmax=25)
            ax.set_xticks(range(len(all_years)))
            ax.set_xticklabels([str(y) for y in all_years], rotation=45, fontsize=7)
            ax.set_yticks(range(len(group_list)))
            ax.set_yticklabels([l[:18] for l in group_list], fontsize=7)
            ax.set_title(f"{section_title} — 연도별 수익률 히트맵")
            for i in range(len(group_list)):
                for j in range(len(all_years)):
                    v = hm[i,j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                                fontsize=6, color="black" if abs(v) < 15 else "white")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            # 나머지 3칸: 연도별 평균 수익률 라인 비교
            if col_idx == 1:
                ax.set_title(f"{section_title} — 연도별 승률")
                for lbl, c in zip(group_list, colors):
                    df2 = results[lbl]["df"].copy(); df2["year"] = df2["entry_date"].dt.year
                    y = df2.groupby("year")["win"].mean() * 100
                    ax.plot(y.index.astype(str), y.values, "o-", color=c, lw=1.5, markersize=5,
                            label=lbl[:15])
                ax.axhline(80, color="gray", linestyle="--", lw=0.8)
                ax.set_ylabel("승률 %"); ax.legend(fontsize=6.5)
                ax.tick_params(axis="x", rotation=45, labelsize=7)
            elif col_idx == 2:
                ax.set_title(f"{section_title} — PF 비교")
                lbls_short = [l[:15] for l in group_list]
                pfs = [results[l]["stats"]["pf"] if not np.isnan(results[l]["stats"]["pf"]) else 0
                       for l in group_list]
                bars = ax.bar(lbls_short, pfs,
                              color=colors, alpha=0.85, edgecolor="white")
                for bar, v in zip(bars, pfs):
                    ax.text(bar.get_x()+bar.get_width()/2, v+0.05,
                            f"{v:.2f}", ha="center", fontsize=8)
                ax.axhline(results["원본"]["stats"]["pf"], color="gray",
                           linestyle="--", lw=1.2, label=f"원본 PF={results['원본']['stats']['pf']:.2f}")
                ax.set_title(f"{section_title} — Profit Factor")
                ax.legend(fontsize=7.5); ax.tick_params(axis="x", labelsize=7)
            else:  # col_idx == 3
                ax.set_title(f"{section_title} — 신호 건수 비교")
                lbls_short = [l[:15] for l in group_list]
                n_sigs = [results[l]["n_sig"] for l in group_list]
                n_trad = [results[l]["stats"]["n"] for l in group_list]
                x2 = np.arange(len(group_list)); wid2 = 0.35
                ax.bar(x2 - wid2/2, n_sigs, wid2, color="steelblue", alpha=0.7, label="신호")
                ax.bar(x2 + wid2/2, n_trad, wid2, color="orange",    alpha=0.7, label="거래")
                for i, (ns, nt) in enumerate(zip(n_sigs, n_trad)):
                    ax.text(i - wid2/2, ns+5, str(ns), ha="center", fontsize=7)
                    ax.text(i + wid2/2, nt+5, str(nt), ha="center", fontsize=7)
                ax.set_xticks(x2); ax.set_xticklabels(lbls_short, fontsize=7)
                ax.legend(fontsize=7.5)

plt.tight_layout()
plt.savefig("backtest_exit_rsi_test.png", dpi=150, bbox_inches="tight")
for lbl in group_labels:
    fname = lbl.replace(" ", "_").replace("(","").replace(")","").replace("/","_").replace("<","lt").replace("→","to")
    results[lbl]["df"].to_csv(f"backtest_exitrsi_{fname}.csv", index=False)
print("\n📊 backtest_exit_rsi_test.png 저장 완료")
print("📄 CSV 7개 저장 완료")
print("✅ 완료")
