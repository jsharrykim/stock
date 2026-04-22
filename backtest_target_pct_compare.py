"""
backtest_target_pct_compare.py
==============================
목표 수익률(TARGET_PCT)을 25%~50% 구간(5%p 단위)으로 바꾸며
기존 20% 포함 총 7개 그룹을 비교 백테스트.

기본 전략 (v10 동일):
  - 매수: MA200↓ + VIX≥25 + (RSI<40 OR CCI<-100)
  - 진입: 신호 발생 다음날 시가
  - 매도: TARGET_PCT 달성 / -25% CB / 60일 경과 & 수익 / 120일 time exit
  - 서킷브레이커(CIRCUIT_PCT): 25% 고정
  - 60거래일 절반 청산 / 120거래일 time exit 고정

비교 그룹:
  TARGET_GROUPS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings("ignore")

# ── 기간 & 고정 파라미터 ──────────────────────────
START        = "2010-01-01"
END          = "2026-01-01"
VIX_MIN      = 25
CIRCUIT_PCT  = 0.25
HALF_EXIT    = 60
MAX_HOLD     = 120
MAX_POSITIONS= 5
MAX_DAILY    = 5

TARGET_GROUPS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# ── 모니터링 종목 (CONTEXT.md 51개 + v10 유니버스 병합) ──
TICKERS = sorted(set([
    # CONTEXT.md 51개
    "SNPS","COST","AZN","AMGN","MDLZ","FTNT","CSGP","CDNS","ADP","FAST",
    "ADI","TXN","PAYX","BKNG","KLAC","MNST","ORLY","HOOD","CPRT","ISRG",
    "PANW","CDW","INTC","AAPL","AVGO","AMD","MSFT","GOOGL","NVDA","META",
    "TSLA","PLTR","MELI","MCHP","AMZN","SMCI","AMAT","MU","LRCX","CSX",
    "QCOM","ROP","INTU","ON","NXPI","STX","ASTS","AVAV","IONQ","SGML",
    # v10 추가
    "GOOG","NFLX","TMUS","ADBE","PEP","CSCO","MRVL","CRWD","DDOG","ZS",
    "TEAM","KDP","IDXX","REGN","VRTX","GILD","BIIB","ILMN","MRNA",
    "ODFL","PCAR","CTSH","VRSK","WDAY","PDD","DLTR","SBUX","ROST",
    "LULU","EBAY","MAR","CTAS","EA","CHTR","CMCSA","EXC","XEL","AEP",
    "MPWR","ENPH","SEDG","COIN","DOCU","ZM","OKTA","PTON",
]))

# ── 한글 폰트 설정 (macOS) ──────────────────────────
_kr_fonts = [f.name for f in fm.fontManager.ttflist
             if "Apple" in f.name or "Malgun" in f.name
             or "NanumGothic" in f.name or "Noto" in f.name]
if _kr_fonts:
    plt.rcParams["font.family"] = _kr_fonts[0]
plt.rcParams["axes.unicode_minus"] = False


# ══════════════════════════════════════════════════
# 데이터 다운로드 & 지표 계산
# ══════════════════════════════════════════════════
def dl_ohlcv(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw[["Open","High","Low","Close","Volume"]].copy()


def compute(d):
    d = d.copy()
    c = d["Close"]
    d["MA200"] = c.rolling(200).mean()
    delta = c.diff()
    g  = delta.clip(lower=0).rolling(14).mean()
    lo = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100 / (1 + g / lo.replace(0, np.nan))
    tp    = (d["High"] + d["Low"] + c) / 3
    tp_ma = tp.rolling(20).mean()
    tp_mad= tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    d["CCI"] = (tp - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))
    return d


# ══════════════════════════════════════════════════
# 시장 데이터 & 종목 데이터 로드
# ══════════════════════════════════════════════════
print("📥 VIX 다운로드...")
vix_raw = yf.download("^VIX", start=START, end=END,
                      auto_adjust=True, progress=False)
_vc = vix_raw["Close"]
if isinstance(_vc, pd.DataFrame):
    _vc = _vc.iloc[:, 0]
vix = pd.Series(_vc.values, index=_vc.index, dtype=float).dropna()
print(f"  ✅ VIX {len(vix)}일")

print("📥 종목 데이터 다운로드...")
stock_data = {}
for i, tk in enumerate(TICKERS, 1):
    d = dl_ohlcv(tk, START, END)
    if len(d) < 250:
        continue
    d = compute(d)
    stock_data[tk] = d
    if i % 20 == 0:
        print(f"  {i}/{len(TICKERS)} 완료")
print(f"  ✅ {len(stock_data)}개 종목 로드")


# ══════════════════════════════════════════════════
# 신호 생성 (TARGET_PCT 무관 — 공통)
# ══════════════════════════════════════════════════
print("🔍 신호 생성...")
signals_by_date = {}

for tk, d in stock_data.items():
    d_c   = d.dropna(subset=["MA200","RSI","CCI"])
    close = d_c["Close"]
    rsi   = d_c["RSI"]
    cci   = d_c["CCI"]

    common = d_c.index.intersection(vix.index)
    if len(common) < 50:
        continue
    vx = vix.reindex(common)

    cond = (
        (close < d_c["MA200"]) &
        (vx >= VIX_MIN) &
        ((rsi < 40) | (cci < -100))
    )
    sig_days = d_c.index[cond.reindex(d_c.index).fillna(False)]

    for sig_day in sig_days:
        idx = d_c.index.get_loc(sig_day)
        if idx + 1 >= len(d_c):
            continue
        entry_day  = d_c.index[idx + 1]
        entry_open = float(d_c["Open"].iloc[idx + 1])
        if pd.isna(entry_open):
            continue
        row = d_c.loc[sig_day]
        if entry_day not in signals_by_date:
            signals_by_date[entry_day] = []
        signals_by_date[entry_day].append({
            "ticker"   : tk,
            "sig_day"  : sig_day,
            "entry_day": entry_day,
            "entry"    : entry_open,
            "rsi"      : float(row["RSI"]),
            "cci"      : float(row["CCI"]),
            "close_sig": float(row["Close"]),
        })

final_signals = []
for entry_day, items in sorted(signals_by_date.items()):
    items.sort(key=lambda x: x["rsi"])
    for item in items[:MAX_DAILY]:
        final_signals.append(item)

print(f"  ✅ 원시 신호: {len(final_signals):,}건")


# ══════════════════════════════════════════════════
# 시뮬레이션 함수
# ══════════════════════════════════════════════════
def run_simulation(signals, target_pct):
    trades        = []
    pos_exit_date = {}
    circuit       = None  # 루프 내부에서 entry 기준 계산

    for sig in signals:
        tk        = sig["ticker"]
        entry_day = sig["entry_day"]
        entry     = sig["entry"]

        active = {t: ed for t, ed in pos_exit_date.items() if ed > entry_day}
        if len(active) >= MAX_POSITIONS:
            continue
        if tk in active:
            continue

        d      = stock_data[tk]
        future = d.loc[d.index >= entry_day]
        if len(future) == 0:
            continue

        cb_price     = entry * (1 - CIRCUIT_PCT)
        tgt_price    = entry * (1 + target_pct)
        half_exited  = False
        exit_records = []

        for i, (fdt, row) in enumerate(future.iterrows()):
            lo = float(row["Low"])
            hi = float(row["High"])
            cl = float(row["Close"])

            # 목표가 도달
            if hi >= tgt_price:
                exit_records.append((fdt, tgt_price, 0.5 if half_exited else 1.0, "target"))
                break
            # 서킷브레이커
            if lo <= cb_price:
                exit_records.append((fdt, cb_price, 0.5 if half_exited else 1.0, "circuit"))
                break
            # 60거래일 경과 & 수익 중 → 절반 청산
            if i + 1 == HALF_EXIT and not half_exited and (cl - entry) / entry > 0:
                exit_records.append((fdt, cl, 0.5, "half_60d"))
                half_exited = True
                continue
            # 120거래일 time exit
            if i + 1 >= MAX_HOLD:
                exit_records.append((fdt, cl, 0.5 if half_exited else 1.0, "time"))
                break

        if not exit_records:
            continue

        total_pct   = sum(r[2] for r in exit_records)
        weighted    = sum((r[1] - entry) / entry * r[2] for r in exit_records)
        blended_ret = weighted / total_pct if total_pct > 0 else 0
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
            "half_exit"  : half_exited,
            "win"        : blended_ret > 0,
            "rsi_entry"  : sig["rsi"],
            "cci_entry"  : sig["cci"],
            "gap_pct"    : (entry - sig["close_sig"]) / sig["close_sig"] * 100,
        })
        pos_exit_date[tk] = last_exit[0]

    return pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)


# ══════════════════════════════════════════════════
# 전체 그룹 실행 & 결과 수집
# ══════════════════════════════════════════════════
print("\n⚙️  그룹별 시뮬레이션 실행 중...")
results = {}

for tgt in TARGET_GROUPS:
    lbl = f"+{int(tgt*100)}%"
    print(f"  {lbl} 처리 중...", end="")
    df = run_simulation(final_signals, tgt)
    if df.empty:
        print(" (결과 없음)")
        continue

    n       = len(df)
    wins    = df[df["win"]]
    losses  = df[~df["win"]]
    wr      = len(wins) / n * 100
    avg_ret = df["return_pct"].mean()
    avg_w   = wins["return_pct"].mean()   if len(wins)   else 0
    avg_l   = losses["return_pct"].mean() if len(losses) else 0
    pf      = (wins["return_pct"].sum() / -losses["return_pct"].sum()
               if len(losses) and losses["return_pct"].sum() < 0 else np.nan)
    ev      = wr/100 * avg_w + (1 - wr/100) * avg_l

    max_cl = cur = 0
    for r in df["return_pct"]:
        cur = cur + 1 if r < 0 else 0
        max_cl = max(max_cl, cur)

    # CAGR
    cap = 1.0
    for _, grp in df.groupby("entry_date"):
        w   = 1.0 / max(len(grp), MAX_POSITIONS)
        cap *= (1 + (grp["return_pct"] / 100 * w).sum())
    yrs  = (df["exit_date"].max() - df["entry_date"].min()).days / 365.25
    cagr = cap ** (1 / yrs) - 1 if yrs > 0 else 0

    avg_hold   = df["hold_days"].mean()
    exit_dist  = df["exit_reason"].value_counts(normalize=True) * 100
    target_pct_share = exit_dist.get("target", 0) + exit_dist.get("half_60d+target", 0)
    circuit_pct_share= exit_dist.get("circuit", 0) + exit_dist.get("half_60d+circuit", 0)

    results[lbl] = {
        "target_pct"  : tgt,
        "n"           : n,
        "win_rate"    : wr,
        "avg_ret"     : avg_ret,
        "avg_win"     : avg_w,
        "avg_loss"    : avg_l,
        "pf"          : pf,
        "ev"          : ev,
        "cagr"        : cagr * 100,
        "max_consec_loss": max_cl,
        "avg_hold_days": avg_hold,
        "exit_dist"   : dict(df["exit_reason"].value_counts()),
        "target_hit_pct": target_pct_share,
        "circuit_hit_pct": circuit_pct_share,
        "df"          : df,
    }
    print(f" ✅  {n}건, 승률 {wr:.1f}%, 평균 {avg_ret:+.2f}%, CAGR {cagr*100:+.1f}%")


# ══════════════════════════════════════════════════
# 콘솔 출력 — 요약 테이블
# ══════════════════════════════════════════════════
print("\n" + "="*100)
print("  목표 수익률별 백테스트 비교 결과 (2010-2026, v10 전략)")
print("  매수: MA200↓ + VIX≥25 + RSI<40 OR CCI<-100 | 진입: 다음날 시가 | CB: -25%")
print("="*100)

hdr = f"  {'그룹':<7} {'건수':>5} {'승률':>7} {'평균수익':>9} {'기대값':>8} {'승평균':>9} {'패평균':>9} {'PF':>6} {'CAGR':>8} {'연속손':>6} {'평균보유':>8} {'목표도달%':>9} {'CB손절%':>8}"
print(hdr)
print("  " + "-"*95)

for lbl, r in results.items():
    pf_str = f"{r['pf']:.2f}" if not np.isnan(r['pf']) else "  N/A"
    print(
        f"  {lbl:<7} {r['n']:>5} {r['win_rate']:>6.1f}% {r['avg_ret']:>+8.2f}% "
        f"{r['ev']:>+7.2f}% {r['avg_win']:>+8.2f}% {r['avg_loss']:>+8.2f}% "
        f"{pf_str:>6} {r['cagr']:>+7.1f}% {r['max_consec_loss']:>5}건 "
        f"{r['avg_hold_days']:>7.0f}일 "
        f"{r['target_hit_pct']:>8.1f}% {r['circuit_hit_pct']:>7.1f}%"
    )

print("="*100)

# 연도별 상세
print("\n[연도별 승률 & 평균수익률]")
years_all = sorted(set().union(*[set(r["df"]["entry_date"].dt.year.unique()) for r in results.values()]))

hdr2 = f"  {'연도':<6}" + "".join(f" {lbl:>14}" for lbl in results.keys())
print(hdr2)
print("  " + "-"*90)
for yr in years_all:
    row_str = f"  {yr:<6}"
    for lbl, r in results.items():
        sub = r["df"][r["df"]["entry_date"].dt.year == yr]
        if len(sub) == 0:
            row_str += f" {'  -':>14}"
        else:
            wr  = sub["win"].mean() * 100
            avg = sub["return_pct"].mean()
            row_str += f" {wr:.0f}%/{avg:+.1f}%({len(sub)})"
    print(row_str)

# 청산 유형 분포
print("\n[청산 유형 분포]")
for lbl, r in results.items():
    ec = r["exit_dist"]
    total = sum(ec.values())
    parts = [f"{k}: {v}건({v/total*100:.0f}%)" for k, v in sorted(ec.items(), key=lambda x: -x[1])]
    print(f"  {lbl}: {' | '.join(parts)}")


# ══════════════════════════════════════════════════
# 최적 조건 도출 (복합 점수)
# ══════════════════════════════════════════════════
print("\n" + "="*70)
print("  최적 조건 분석")
print("="*70)

score_data = []
for lbl, r in results.items():
    # 복합 점수: 승률(30%) + CAGR(30%) + 기대값(25%) + PF(15%)
    pf_val = r["pf"] if not np.isnan(r["pf"]) else 0
    score = (
        r["win_rate"]   * 0.30 +
        r["cagr"]       * 0.30 +
        r["ev"]         * 0.25 +
        min(pf_val, 5)  * 0.15 * 10  # PF 정규화 (5점 만점 → ×10)
    )
    score_data.append((lbl, score, r))

score_data.sort(key=lambda x: -x[1])

print(f"\n  복합 점수 순위 (승률 30% + CAGR 30% + 기대값 25% + PF 15%):")
print(f"  {'순위':<5} {'그룹':<8} {'점수':>8} {'승률':>7} {'CAGR':>8} {'기대값':>8} {'PF':>6} {'평균보유':>8}")
print("  " + "-"*65)
for rank, (lbl, score, r) in enumerate(score_data, 1):
    pf_str = f"{r['pf']:.2f}" if not np.isnan(r['pf']) else " N/A"
    print(f"  {rank:<5} {lbl:<8} {score:>8.2f} {r['win_rate']:>6.1f}% {r['cagr']:>+7.1f}% {r['ev']:>+7.2f}% {pf_str:>6} {r['avg_hold_days']:>7.0f}일")

best_lbl, best_score, best_r = score_data[0]
print(f"\n  ★ 복합 점수 1위: {best_lbl}")
print(f"    - 승률: {best_r['win_rate']:.1f}%")
print(f"    - 평균 수익률: {best_r['avg_ret']:+.2f}%")
print(f"    - 기대값(EV): {best_r['ev']:+.2f}%")
print(f"    - CAGR: {best_r['cagr']:+.1f}%")
print(f"    - Profit Factor: {best_r['pf']:.2f}")
print(f"    - 평균 보유 기간: {best_r['avg_hold_days']:.0f}일")
print(f"    - 목표 도달 비율: {best_r['target_hit_pct']:.1f}%")


# ══════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════
labels = list(results.keys())
wr_vals   = [results[l]["win_rate"]    for l in labels]
ret_vals  = [results[l]["avg_ret"]     for l in labels]
cagr_vals = [results[l]["cagr"]        for l in labels]
ev_vals   = [results[l]["ev"]          for l in labels]
pf_vals   = [results[l]["pf"] if not np.isnan(results[l]["pf"]) else 0 for l in labels]
hold_vals = [results[l]["avg_hold_days"] for l in labels]
n_vals    = [results[l]["n"]           for l in labels]
tgt_hit   = [results[l]["target_hit_pct"] for l in labels]
cb_hit    = [results[l]["circuit_hit_pct"] for l in labels]

fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.suptitle(
    "목표 수익률 그룹별 백테스트 비교 (2010-2026)\n"
    "전략: MA200↓ + VIX≥25 + RSI<40 OR CCI<-100 | CB: -25% | 60일 절반 / 120일 time exit",
    fontsize=12, fontweight="bold"
)

colors = ["#e74c3c","#e67e22","#f1c40f","#2ecc71","#1abc9c","#3498db","#9b59b6"]

# 1. 승률
ax = axes[0, 0]
bars = ax.bar(labels, wr_vals, color=colors, edgecolor="white", linewidth=1.2)
ax.set_title("승률 (%)", fontweight="bold")
ax.set_ylim(0, 100)
for bar, v in zip(bars, wr_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{v:.1f}%", ha="center", fontsize=9)
ax.axhline(70, color="gray", linestyle="--", lw=1, alpha=0.5, label="70% 기준선")
ax.legend(fontsize=8)

# 2. 평균 수익률
ax = axes[0, 1]
bar_c = ["#2ecc71" if v >= 0 else "#e74c3c" for v in ret_vals]
bars = ax.bar(labels, ret_vals, color=bar_c, edgecolor="white", linewidth=1.2)
ax.set_title("평균 수익률 (%)", fontweight="bold")
ax.axhline(0, color="black", lw=0.8)
for bar, v in zip(bars, ret_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.2 if v >= 0 else -0.6),
            f"{v:+.2f}%", ha="center", fontsize=9)

# 3. CAGR
ax = axes[0, 2]
bar_c = ["#2ecc71" if v >= 0 else "#e74c3c" for v in cagr_vals]
bars = ax.bar(labels, cagr_vals, color=bar_c, edgecolor="white", linewidth=1.2)
ax.set_title("포트폴리오 CAGR (%)", fontweight="bold")
ax.axhline(0, color="black", lw=0.8)
for bar, v in zip(bars, cagr_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.3 if v >= 0 else -0.8),
            f"{v:+.1f}%", ha="center", fontsize=9)

# 4. 기대값(EV)
ax = axes[1, 0]
bar_c = ["#2ecc71" if v >= 0 else "#e74c3c" for v in ev_vals]
bars = ax.bar(labels, ev_vals, color=bar_c, edgecolor="white", linewidth=1.2)
ax.set_title("기대값 EV (%)", fontweight="bold")
ax.axhline(0, color="black", lw=0.8)
for bar, v in zip(bars, ev_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.1 if v >= 0 else -0.3),
            f"{v:+.2f}%", ha="center", fontsize=9)

# 5. Profit Factor
ax = axes[1, 1]
bars = ax.bar(labels, pf_vals, color=colors, edgecolor="white", linewidth=1.2)
ax.set_title("Profit Factor", fontweight="bold")
ax.axhline(1.5, color="gray", linestyle="--", lw=1, alpha=0.5, label="1.5 기준선")
for bar, v in zip(bars, pf_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{v:.2f}", ha="center", fontsize=9)
ax.legend(fontsize=8)

# 6. 평균 보유 기간
ax = axes[1, 2]
bars = ax.bar(labels, hold_vals, color=colors, edgecolor="white", linewidth=1.2)
ax.set_title("평균 보유 기간 (일)", fontweight="bold")
for bar, v in zip(bars, hold_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{v:.0f}일", ha="center", fontsize=9)

# 7. 청산 유형 분포 (목표 vs CB)
ax = axes[2, 0]
x  = np.arange(len(labels))
w  = 0.35
ax.bar(x - w/2, tgt_hit, width=w, label="목표 도달", color="#2ecc71", edgecolor="white")
ax.bar(x + w/2, cb_hit,  width=w, label="CB 손절",   color="#e74c3c", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_title("목표 도달 vs CB 손절 비율 (%)", fontweight="bold")
ax.legend(fontsize=9)

# 8. 누적 수익 곡선 비교
ax = axes[2, 1]
for lbl, color in zip(labels, colors):
    df_r = results[lbl]["df"]
    cum  = (1 + df_r["return_pct"] / 100).cumprod() - 1
    ax.plot(range(len(cum)), cum * 100, label=lbl, color=color, lw=1.5, alpha=0.85)
ax.axhline(0, color="black", lw=0.8, linestyle="--")
ax.set_title("누적 수익률 곡선", fontweight="bold")
ax.set_xlabel("Trade #")
ax.set_ylabel("누적 수익 %")
ax.legend(fontsize=8, loc="upper left")

# 9. 복합 점수 (레이더 대신 가로 막대)
ax = axes[2, 2]
score_labels = [s[0] for s in score_data]
score_values = [s[1] for s in score_data]
bar_colors   = [colors[labels.index(l)] for l in score_labels]
bars = ax.barh(score_labels, score_values, color=bar_colors, edgecolor="white", linewidth=1.2)
ax.set_title("복합 점수 순위\n(승률30%+CAGR30%+EV25%+PF15%)", fontweight="bold")
ax.set_xlabel("점수")
for bar, v in zip(bars, score_values):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f"{v:.2f}", va="center", fontsize=9)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("backtest_target_compare.png", dpi=150, bbox_inches="tight")
print("\n📊 backtest_target_compare.png 저장 완료")

# CSV 저장
for lbl, r in results.items():
    fname = f"backtest_target_{lbl.replace('+','').replace('%','pct')}_trades.csv"
    r["df"].to_csv(fname, index=False)
print("📄 그룹별 CSV 저장 완료")

# ══════════════════════════════════════════════════
# 최종 권고안 출력
# ══════════════════════════════════════════════════
print("\n" + "="*70)
print("  ★ 최종 권고안")
print("="*70)

# 상위 3개 그룹
top3 = score_data[:3]
print(f"\n  📊 복합 점수 상위 3개 그룹:")
for rank, (lbl, score, r) in enumerate(top3, 1):
    print(f"\n  [{rank}위] TARGET_PCT = {lbl}  (복합점수: {score:.2f})")
    print(f"    승률: {r['win_rate']:.1f}% | 평균수익: {r['avg_ret']:+.2f}% | CAGR: {r['cagr']:+.1f}%")
    print(f"    기대값: {r['ev']:+.2f}% | PF: {r['pf']:.2f} | 평균보유: {r['avg_hold_days']:.0f}일")
    print(f"    최대연속손실: {r['max_consec_loss']}건 | 목표도달율: {r['target_hit_pct']:.1f}%")

# 전략 특성별 권장
print("\n  📌 전략 성향별 권장:")

# 최고 승률
best_wr  = max(results.items(), key=lambda x: x[1]["win_rate"])
print(f"  - 최고 승률: {best_wr[0]} ({best_wr[1]['win_rate']:.1f}%) — 안정형 선호 시")

# 최고 CAGR
best_cagr = max(results.items(), key=lambda x: x[1]["cagr"])
print(f"  - 최고 CAGR: {best_cagr[0]} ({best_cagr[1]['cagr']:+.1f}%) — 복리 성장 극대화")

# 최고 기대값
best_ev = max(results.items(), key=lambda x: x[1]["ev"])
print(f"  - 최고 EV  : {best_ev[0]} ({best_ev[1]['ev']:+.2f}%) — 기대수익 극대화")

# 가장 짧은 보유 기간
best_hold = min(results.items(), key=lambda x: x[1]["avg_hold_days"])
print(f"  - 최단 보유: {best_hold[0]} ({best_hold[1]['avg_hold_days']:.0f}일) — 회전율 우선 시")

print("\n  ※ 위 결과는 2010-2026 과거 데이터 기반 시뮬레이션입니다.")
print("     미래 성과를 보장하지 않으며, 실전 적용 시 슬리피지·세금 고려 필요.")
print("="*70)
print("\n✅ 백테스트 완료")
