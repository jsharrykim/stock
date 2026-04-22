"""
backtest_compare_universes.py
==============================
v10 전략으로 3개 종목군 비교 백테스트
  - 나스닥 100
  - S&P 500
  - 다우존스 30
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

START = "2010-01-01"
END   = "2026-01-01"

VIX_MIN      = 25
TARGET_PCT   = 0.20
CIRCUIT_PCT  = 0.25
HALF_EXIT    = 60
MAX_HOLD     = 120
MAX_POSITIONS= 5
MAX_DAILY    = 5

# ──────────────────────────────────────────────
# 종목 유니버스
# ──────────────────────────────────────────────
UNIVERSES = {
    "나스닥100": sorted(set([
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
        "NFLX","TMUS","AMD","ADBE","PEP","CSCO","QCOM","INTC","TXN","AMGN",
        "INTU","AMAT","MU","LRCX","KLAC","CDNS","SNPS","MRVL","FTNT","PANW",
        "CRWD","DDOG","ZS","TEAM","MNST","KDP","MDLZ","ORLY","AZN",
        "ISRG","IDXX","REGN","VRTX","GILD","BIIB","ILMN","MRNA",
        "ODFL","FAST","PCAR","CTSH","PAYX","VRSK","WDAY","MELI","PDD",
        "DLTR","SBUX","ROST","LULU","EBAY","MAR","CTAS","EA",
        "CHTR","CMCSA","EXC","XEL","AEP","MKL",
        "PLTR","CPRT","NXPI","MPWR","ENPH","SEDG","ON","COIN",
        "DOCU","ZM","OKTA","PTON","JD","BIDU","NTES","FANG",
    ])),

    "S&P500": sorted(set([
        # 대형 기술/통신
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","ORCL","CRM",
        "ACN","ADBE","NOW","TXN","QCOM","INTC","AMD","MU","AMAT","LRCX",
        "KLAC","ADI","MCHP","CDNS","SNPS","ANSS","KEYS","ENPH","FSLR",
        # 헬스케어
        "LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","DHR","AMGN","ISRG",
        "VRTX","REGN","GILD","MDT","SYK","BSX","ZBH","HOLX","IDXX","DXCM",
        # 금융
        "BRK-B","JPM","BAC","WFC","GS","MS","BLK","SCHW","AXP","COF",
        "USB","PNC","TFC","FITB","KEY","CFG","HBAN","RF","MTB","ZION",
        # 소비재
        "AMZN","HD","MCD","SBUX","NKE","TJX","ROST","LULU","CMG","YUM",
        "DPZ","QSR","DKNG","BKNG","HLT","MAR","H","RCL","CCL","NCLH",
        # 산업재
        "CAT","DE","HON","GE","MMM","UPS","FDX","RTX","BA","LMT",
        "NOC","GD","HII","LHX","LDOS","CACI","SAIC",
        # 에너지
        "XOM","CVX","COP","EOG","SLB","HAL","BKR","MPC","PSX","VLO",
        # 필수소비재
        "WMT","PG","KO","PEP","MDLZ","KHC","GIS","CPB","MKC","SJM",
        "CAG","HRL","TSN","K","CLX","CHD","CL","COTY","EL",
        # 유틸리티/부동산
        "NEE","DUK","SO","D","AEP","XEL","ES","EIX","PPL","CNP",
        "PLD","AMT","CCI","EQIX","SPG","O","VICI","IRM","DLR",
        # 통신
        "T","VZ","TMUS","CMCSA","CHTR","DISH","LUMN",
    ])),

    "다우30": [
        "AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW",
        "GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD","MMM",
        "MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WBA","WMT",
    ],
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
    d = d.copy()
    c = d["Close"]
    d["MA200"] = c.rolling(200).mean()
    delta = c.diff()
    g = delta.clip(lower=0).rolling(14).mean()
    l = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100 / (1 + g / l.replace(0, np.nan))
    tp    = (d["High"] + d["Low"] + c) / 3
    tp_ma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    d["CCI"] = (tp - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))
    return d


def run_backtest(tickers, vix, label):
    print(f"\n{'='*55}")
    print(f"  [{label}] 백테스트 시작 ({len(tickers)}개 종목)")
    print(f"{'='*55}")

    # 종목 데이터 로드
    stock_data = {}
    for i, tk in enumerate(tickers, 1):
        d = dl_ohlcv(tk, START, END)
        if len(d) < 250:
            continue
        d = compute(d)
        stock_data[tk] = d
    print(f"  ✅ {len(stock_data)}개 종목 로드")

    # 신호 생성
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

    # 시뮬레이션
    trades        = []
    pos_exit_date = {}

    for sig in final_signals:
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

        circuit      = entry * (1 - CIRCUIT_PCT)
        target       = entry * (1 + TARGET_PCT)
        half_exited  = False
        exit_records = []

        for i, (fdt, row) in enumerate(future.iterrows()):
            lo = float(row["Low"])
            hi = float(row["High"])
            cl = float(row["Close"])
            ret = (cl - entry) / entry

            if hi >= target:
                exit_records.append((fdt, target, 0.5 if half_exited else 1.0, "target"))
                break
            if lo <= circuit:
                exit_records.append((fdt, circuit, 0.5 if half_exited else 1.0, "circuit"))
                break
            if i + 1 == HALF_EXIT and not half_exited and ret > 0:
                exit_records.append((fdt, cl, 0.5, "half_60d"))
                half_exited = True
                continue
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
            "win"        : blended_ret > 0,
        })
        pos_exit_date[tk] = last_exit[0]

    if not trades:
        print(f"  ⚠️  트레이드 없음")
        return None

    df   = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)
    wins = df[df["win"]]
    loss = df[~df["win"]]
    n    = len(df)

    win_rate = len(wins) / n * 100
    avg_ret  = df["return_pct"].mean()
    avg_win  = wins["return_pct"].mean() if len(wins) else 0
    avg_loss = loss["return_pct"].mean() if len(loss) else 0
    pf       = (wins["return_pct"].sum() / -loss["return_pct"].sum()
                if len(loss) and loss["return_pct"].sum() < 0 else np.nan)
    ev       = win_rate/100 * avg_win + (1 - win_rate/100) * avg_loss

    # CAGR
    capital = 1.0
    for ed, group in df.groupby("entry_date"):
        w     = 1.0 / max(len(group), MAX_POSITIONS)
        batch = (group["return_pct"] / 100 * w).sum()
        capital *= (1 + batch)
    years = (df["exit_date"].max() - df["entry_date"].min()).days / 365.25
    cagr  = capital ** (1 / years) - 1

    # 1억 → N년 후
    init    = 100_000_000
    val_5y  = init * ((1 + cagr) ** 5)
    val_10y = init * ((1 + cagr) ** 10)
    val_15y = init * ((1 + cagr) ** 15)

    result = {
        "label"   : label,
        "n"       : n,
        "win_rate": win_rate,
        "avg_ret" : avg_ret,
        "avg_win" : avg_win,
        "avg_loss": avg_loss,
        "pf"      : pf,
        "ev"      : ev,
        "cagr"    : cagr * 100,
        "val_5y"  : val_5y,
        "val_10y" : val_10y,
        "val_15y" : val_15y,
        "df"      : df,
    }

    print(f"  트레이드: {n}건 | 승률: {win_rate:.1f}% | "
          f"평균수익: {avg_ret:+.2f}% | CAGR: {cagr*100:+.1f}%")
    return result


# ──────────────────────────────────────────────
# VIX 다운로드
# ──────────────────────────────────────────────
print("📥 VIX 데이터 다운로드 중...")
vix_raw = yf.download("^VIX", start=START, end=END,
                      auto_adjust=True, progress=False)
_vc = vix_raw["Close"]
if isinstance(_vc, pd.DataFrame): _vc = _vc.iloc[:, 0]
vix = pd.Series(_vc.values, index=_vc.index, dtype=float).dropna()
print(f"✅ VIX {len(vix)}일")

# ──────────────────────────────────────────────
# 3개 유니버스 백테스트
# ──────────────────────────────────────────────
results = {}
for label, tickers in UNIVERSES.items():
    r = run_backtest(tickers, vix, label)
    if r:
        results[label] = r

# ──────────────────────────────────────────────
# 비교 결과 출력
# ──────────────────────────────────────────────
print("\n\n" + "="*70)
print("  종목군별 v10 전략 백테스트 비교 결과")
print("  기간: 2010-01-01 ~ 2026-01-01")
print("  조건: MA200↓ + VIX≥25 + (RSI<40 OR CCI<-100)")
print("  청산: +20% / -25% CB / 60일절반(수익중) / 120일")
print("="*70)

fmt_h = "  {:<12} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8}"
fmt_r = "  {:<12} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8}"
print(fmt_h.format("종목군", "트레이드", "승률", "평균수익", "PF", "기대값", "CAGR"))
print("  " + "-"*66)
for label, r in results.items():
    pf_str = f"{r['pf']:.2f}" if not np.isnan(r['pf']) else "N/A"
    print(fmt_r.format(
        label,
        f"{r['n']}건",
        f"{r['win_rate']:.1f}%",
        f"{r['avg_ret']:+.2f}%",
        pf_str,
        f"{r['ev']:+.2f}%",
        f"{r['cagr']:+.1f}%",
    ))

print("\n" + "="*70)
print("  1억 투자 시 N년 후 자산 (세전, 복리 가정)")
print("="*70)
fmt_m = "  {:<12} {:>16} {:>16} {:>16}"
print(fmt_m.format("종목군", "5년 후", "10년 후", "15년 후"))
print("  " + "-"*62)
for label, r in results.items():
    def fmt_won(v):
        if v >= 1e8:
            return f"{v/1e8:.1f}억"
        return f"{v/1e4:.0f}만"
    print(fmt_m.format(
        label,
        fmt_won(r['val_5y']),
        fmt_won(r['val_10y']),
        fmt_won(r['val_15y']),
    ))

print("\n" + "="*70)
print("  종목군별 상세")
print("="*70)
for label, r in results.items():
    print(f"\n  [{label}]")
    print(f"    트레이드   : {r['n']}건")
    print(f"    승률       : {r['win_rate']:.1f}%")
    print(f"    평균 수익  : {r['avg_ret']:+.2f}%")
    print(f"    승자 평균  : {r['avg_win']:+.2f}%")
    print(f"    패자 평균  : {r['avg_loss']:+.2f}%")
    pf_str = f"{r['pf']:.2f}" if not np.isnan(r['pf']) else "N/A"
    print(f"    PF         : {pf_str}")
    print(f"    기대값     : {r['ev']:+.2f}%")
    print(f"    CAGR       : {r['cagr']:+.1f}%")
    print(f"    1억→5년    : {r['val_5y']/1e8:.2f}억")
    print(f"    1억→10년   : {r['val_10y']/1e8:.2f}억")
    print(f"    1억→15년   : {r['val_15y']/1e8:.2f}억")

    # 연도별
    df = r["df"]
    df["year"] = df["entry_date"].dt.year
    yearly = df.groupby("year").agg(
        trades   =("return_pct","count"),
        avg_ret  =("return_pct","mean"),
        win_rate =("win", lambda x: x.mean()*100),
    ).round(1)
    print(f"    연도별:\n{yearly.to_string()}")

# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    "v10 전략 종목군별 비교 (MA200↓ + VIX≥25 + RSI<40 OR CCI<-100)\n"
    "나스닥100 vs S&P500 vs 다우30",
    fontsize=12, fontweight="bold"
)

colors = {"나스닥100": "#2ecc71", "S&P500": "#3498db", "다우30": "#e74c3c"}

# 1. CAGR 막대
ax = axes[0, 0]
labels = list(results.keys())
cagrs  = [results[l]["cagr"] for l in labels]
bars   = ax.bar(labels, cagrs, color=[colors[l] for l in labels], edgecolor="white")
ax.axhline(0, color="black", lw=0.8)
for bar, val in zip(bars, cagrs):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
            f"{val:+.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_title("CAGR 비교")
ax.set_ylabel("%")

# 2. 승률 막대
ax = axes[0, 1]
wrs  = [results[l]["win_rate"] for l in labels]
bars = ax.bar(labels, wrs, color=[colors[l] for l in labels], edgecolor="white")
ax.axhline(50, color="red", lw=1, linestyle="--", label="50%")
for bar, val in zip(bars, wrs):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
            f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_title("승률 비교")
ax.set_ylabel("%")
ax.legend()

# 3. 1억 → 10년 후
ax = axes[0, 2]
val10s = [results[l]["val_10y"] / 1e8 for l in labels]
bars   = ax.bar(labels, val10s, color=[colors[l] for l in labels], edgecolor="white")
for bar, val in zip(bars, val10s):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
            f"{val:.1f}억", ha="center", fontsize=11, fontweight="bold")
ax.set_title("1억 투자 → 10년 후 (억원)")
ax.set_ylabel("억원")

# 4. 수익률 분포 (겹쳐서)
ax = axes[1, 0]
for label, r in results.items():
    ax.hist(r["df"]["return_pct"], bins=40, alpha=0.5,
            label=label, color=colors[label])
ax.axvline(0, color="black", lw=1.5, linestyle="--")
ax.set_title("수익률 분포 비교")
ax.legend()

# 5. 누적 수익률 곡선
ax = axes[1, 1]
for label, r in results.items():
    cum = (1 + r["df"]["return_pct"] / 100).cumprod() - 1
    ax.plot(range(len(cum)), cum * 100,
            label=f"{label} (CAGR {r['cagr']:+.1f}%)",
            color=colors[label], lw=1.5)
ax.axhline(0, color="black", lw=0.8, linestyle="--")
ax.set_title("누적 수익률 비교")
ax.set_xlabel("Trade #")
ax.set_ylabel("%")
ax.legend(fontsize=8)

# 6. 1억 → 5/10/15년 비교
ax = axes[1, 2]
x     = np.arange(3)
width = 0.25
for j, label in enumerate(labels):
    vals = [
        results[label]["val_5y"]  / 1e8,
        results[label]["val_10y"] / 1e8,
        results[label]["val_15y"] / 1e8,
    ]
    bars = ax.bar(x + j * width, vals, width,
                  label=label, color=colors[label], edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                f"{val:.0f}", ha="center", fontsize=7)
ax.set_xticks(x + width)
ax.set_xticklabels(["5년 후", "10년 후", "15년 후"])
ax.set_title("1억 투자 → N년 후 자산 (억원)")
ax.set_ylabel("억원")
ax.legend()

plt.tight_layout()
plt.savefig("backtest_compare_universes.png", dpi=150, bbox_inches="tight")
print("\n\n📊 backtest_compare_universes.png 저장 완료")
print("✅ 비교 백테스트 완료")
