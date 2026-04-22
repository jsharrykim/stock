"""
backtest_vix_v10.py  — 다음날 시가 진입 기준 (현실적 백테스트)
================================================================
변경 사항:
  - 진입가: 당일 종가 → 다음날 시가(Open)
  - 신호: 전일 종가 기준 조건 확인 → 당일 시가로 진입
  - 진입 조건: MA200↓ + VIX≥25 + (RSI<40 OR CCI<-100)
  - 청산: +20% / -25% CB / 60일절반 / 120일 (v9 동일)

실전 운용 방식:
  - 장중 2시간마다 모니터링
  - 조건 충족 시 다음 모니터링 시점(또는 즉시) 현재가로 진입
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

TICKERS = sorted(set([
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
    d = d.copy()
    c = d["Close"]
    d["MA200"] = c.rolling(200).mean()
    # RSI(14)
    delta = c.diff()
    g = delta.clip(lower=0).rolling(14).mean()
    l = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100 / (1 + g / l.replace(0, np.nan))
    # CCI(20)
    tp     = (d["High"] + d["Low"] + c) / 3
    tp_ma  = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    d["CCI"] = (tp - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))
    return d


# ──────────────────────────────────────────────
# 시장 데이터
# ──────────────────────────────────────────────
print("📥 시장 데이터 다운로드 중...")
vix_raw = yf.download("^VIX", start=START, end=END,
                      auto_adjust=True, progress=False)
_vc = vix_raw["Close"]
if isinstance(_vc, pd.DataFrame): _vc = _vc.iloc[:, 0]
vix = pd.Series(_vc.values, index=_vc.index, dtype=float).dropna()
print(f"✅ VIX {len(vix)}일")

# ──────────────────────────────────────────────
# 종목 데이터
# ──────────────────────────────────────────────
print("📥 종목 데이터 다운로드 중...")
stock_data = {}
for i, tk in enumerate(TICKERS, 1):
    d = dl_ohlcv(tk, START, END)
    if len(d) < 250:
        continue
    d = compute(d)
    stock_data[tk] = d
    if i % 20 == 0:
        print(f"  {i}/{len(TICKERS)} 완료")
print(f"✅ {len(stock_data)}개 종목 로드")

# ──────────────────────────────────────────────
# 신호 생성 — 전일 종가 기준 조건 확인
# ──────────────────────────────────────────────
print("🔍 신호 생성 중...")
signals_by_date = {}   # 진입 날짜(다음날) → 신호 목록

for tk, d in stock_data.items():
    d_c = d.dropna(subset=["MA200", "RSI", "CCI"])
    close = d_c["Close"]
    rsi   = d_c["RSI"]
    cci   = d_c["CCI"]

    common = d_c.index.intersection(vix.index)
    if len(common) < 50:
        continue
    vx = vix.reindex(common)

    # 조건: 전일 종가 기준
    cond = (
        (close < d_c["MA200"]) &
        (vx >= VIX_MIN) &
        ((rsi < 40) | (cci < -100))
    )

    sig_days = d_c.index[cond.reindex(d_c.index).fillna(False)]

    for sig_day in sig_days:
        # ★ 진입은 다음날 시가
        idx = d_c.index.get_loc(sig_day)
        if idx + 1 >= len(d_c):
            continue
        entry_day = d_c.index[idx + 1]
        entry_open = float(d_c["Open"].iloc[idx + 1])

        if pd.isna(entry_open):
            continue

        row = d_c.loc[sig_day]
        if entry_day not in signals_by_date:
            signals_by_date[entry_day] = []
        signals_by_date[entry_day].append({
            "ticker"   : tk,
            "sig_day"  : sig_day,        # 신호 발생일 (전일)
            "entry_day": entry_day,      # 실제 진입일 (당일 시가)
            "entry"    : entry_open,     # ★ 다음날 시가
            "rsi"      : float(row["RSI"]),
            "cci"      : float(row["CCI"]),
            "close_sig": float(row["Close"]),  # 신호일 종가 (참고용)
        })

# RSI 낮은 순 → 상위 MAX_DAILY
final_signals = []
for entry_day, items in sorted(signals_by_date.items()):
    items.sort(key=lambda x: x["rsi"])
    for item in items[:MAX_DAILY]:
        final_signals.append(item)

print(f"✅ 원시 신호: {len(final_signals):,}건 ({len(signals_by_date)}거래일)")

# ──────────────────────────────────────────────
# 시뮬레이션 — 진입 다음날부터 가격 추적
# ──────────────────────────────────────────────
print("⚙️  시뮬레이션 중...")
trades        = []
pos_exit_date = {}

for sig in final_signals:
    tk         = sig["ticker"]
    entry_day  = sig["entry_day"]
    entry      = sig["entry"]       # 다음날 시가

    active = {t: ed for t, ed in pos_exit_date.items() if ed > entry_day}
    if len(active) >= MAX_POSITIONS:
        continue
    if tk in active:
        continue

    d = stock_data[tk]
    # 진입일 이후 데이터 (당일 포함 — 시가로 들어갔으므로 당일 고/저도 유효)
    future = d.loc[d.index >= entry_day]
    if len(future) == 0:
        continue

    circuit = entry * (1 - CIRCUIT_PCT)
    target  = entry * (1 + TARGET_PCT)
    half_exited  = False
    exit_records = []

    for i, (fdt, row) in enumerate(future.iterrows()):
        lo  = float(row["Low"])
        hi  = float(row["High"])
        cl  = float(row["Close"])

        # 진입일 당일은 시가로 들어갔으므로
        # 시가 이후 움직임만 반영 (시가보다 낮은 Low는 진입 후 발생)
        if i == 0:
            # 당일: 시가(entry) 기준으로 target/circuit 체크
            # High는 entry 이후 가능, Low도 entry 이후 가능
            pass  # 그대로 처리 (보수적)

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

    gap_pct = (entry - sig["close_sig"]) / sig["close_sig"] * 100  # 갭 크기

    trades.append({
        "sig_date"   : sig["sig_day"],
        "entry_date" : entry_day,
        "exit_date"  : last_exit[0],
        "ticker"     : tk,
        "sig_close"  : sig["close_sig"],
        "entry"      : entry,
        "exit_price" : last_exit[1],
        "gap_pct"    : gap_pct,
        "return_pct" : blended_ret * 100,
        "hold_days"  : (last_exit[0] - entry_day).days,
        "exit_reason": reason,
        "half_exit"  : half_exited,
        "win"        : blended_ret > 0,
        "rsi_entry"  : sig["rsi"],
        "cci_entry"  : sig["cci"],
    })
    pos_exit_date[tk] = last_exit[0]

print(f"✅ 완료: {len(trades)}건")

# ──────────────────────────────────────────────
# 분석
# ──────────────────────────────────────────────
df = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)
wins   = df[df["win"]]
losses = df[~df["win"]]
n      = len(df)

win_rate = len(wins) / n * 100
avg_ret  = df["return_pct"].mean()
avg_win  = wins["return_pct"].mean()   if len(wins)   else 0
avg_loss = losses["return_pct"].mean() if len(losses) else 0
pf       = (wins["return_pct"].sum() / -losses["return_pct"].sum()
            if len(losses) and losses["return_pct"].sum() < 0 else np.nan)
ev       = win_rate/100 * avg_win + (1 - win_rate/100) * avg_loss

max_cl = cur = 0
for r in df["return_pct"]:
    cur = cur + 1 if r < 0 else 0
    max_cl = max(max_cl, cur)

exit_cnt = df["exit_reason"].value_counts()

# 갭 통계
avg_gap   = df["gap_pct"].mean()
pos_gap   = (df["gap_pct"] > 0).mean() * 100
neg_gap   = (df["gap_pct"] < 0).mean() * 100

# 포트폴리오 CAGR
capital = 1.0
for ed, group in df.groupby("entry_date"):
    w       = 1.0 / max(len(group), MAX_POSITIONS)
    batch   = (group["return_pct"] / 100 * w).sum()
    capital *= (1 + batch)
years = (df["exit_date"].max() - df["entry_date"].min()).days / 365.25
cagr  = capital ** (1 / years) - 1

print("\n" + "="*66)
print("  v10 다음날 시가 진입 — 백테스트 결과")
print("="*66)
print(f"  기간           : {START} ~ {END}")
print(f"  진입 기준      : 전일 종가 조건 확인 → 다음날 시가 진입")
print(f"  진입 조건      : MA200↓ + VIX≥25 + (RSI<40 OR CCI<-100)")
print(f"  청산 규칙      : +20% / -25% CB / 60일절반 / 120일")
print(f"  총 트레이드    : {n}건")
print(f"  활성 연도      : {df['entry_date'].dt.year.nunique()}개년")
print(f"  승 률          : {win_rate:.1f}%")
print(f"  평균 수익률    : {avg_ret:+.2f}%")
print(f"  기대값         : {ev:+.2f}%")
print(f"  승자 평균      : {avg_win:+.2f}%")
print(f"  패자 평균      : {avg_loss:+.2f}%")
print(f"  Profit Factor  : {pf:.2f}")
print(f"  포트CAGR       : {cagr*100:+.2f}%")
print(f"  최대 연속 손실 : {max_cl}건")
print(f"  평균 보유 일수 : {df['hold_days'].mean():.0f}일")
print(f"  청산 유형:")
for r, c in exit_cnt.items():
    print(f"    {r:<25}: {c:4d}건 ({c/n*100:.1f}%)")
print(f"\n  ── 갭 분석 (신호 종가 → 다음날 시가) ──")
print(f"  평균 갭        : {avg_gap:+.2f}%")
print(f"  갭 상승 비율   : {pos_gap:.1f}%  (불리: 비싸게 진입)")
print(f"  갭 하락 비율   : {neg_gap:.1f}%  (유리: 싸게 진입)")
print(f"  갭 절대값 평균 : {df['gap_pct'].abs().mean():.2f}%")
print("="*66)

# 연도별
df["year"] = df["entry_date"].dt.year
yearly = df.groupby("year").agg(
    trades   =("return_pct","count"),
    avg_ret  =("return_pct","mean"),
    win_rate =("win", lambda x: x.mean()*100),
    total_ret=("return_pct","sum"),
    avg_gap  =("gap_pct","mean"),
)
print("\n연도별 성과:")
print(yearly.round(2).to_string())

# ── v9 vs v10 비교 ─────────────────────────────────
print("\n" + "="*60)
print("  v9 (종가 진입) vs v10 (다음날 시가 진입) 비교")
print("="*60)
v9 = {"n":148, "wr":80.4, "ar":9.91, "pf":3.47, "cagr":20.4}
v10= {"n":n,   "wr":win_rate, "ar":avg_ret,
      "pf":pf if not np.isnan(pf) else 0, "cagr":cagr*100}

rows = [
    ("총 트레이드",    f"{v9['n']}건",          f"{v10['n']}건"),
    ("승률",          f"{v9['wr']:.1f}%",      f"{v10['wr']:.1f}%"),
    ("평균 수익률",    f"{v9['ar']:+.2f}%",     f"{v10['ar']:+.2f}%"),
    ("Profit Factor", f"{v9['pf']:.2f}",       f"{v10['pf']:.2f}"),
    ("포트CAGR",      f"{v9['cagr']:+.1f}%",   f"{v10['cagr']:+.1f}%"),
    ("갭 평균",       "0.00% (종가=진입)",       f"{avg_gap:+.2f}%"),
]
fmt = "  {:<18} {:>22} {:>18}"
print(fmt.format("지표", "v9 종가진입", "v10 다음날시가"))
print("  " + "-"*58)
for label, a, b in rows:
    print(fmt.format(label, a, b))
print("="*60)

print(f"\n▶ 수익 상위 10:")
print(df.nlargest(10, "return_pct")[
    ["entry_date","ticker","sig_close","entry","gap_pct",
     "return_pct","exit_reason","hold_days"]
].to_string(index=False))
print(f"\n▶ 손실 하위 10:")
print(df.nsmallest(10, "return_pct")[
    ["entry_date","ticker","sig_close","entry","gap_pct",
     "return_pct","exit_reason","hold_days"]
].to_string(index=False))

# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    f"v10 다음날 시가 진입 (MA200↓ + VIX≥25 + RSI<40 or CCI<-100)\n"
    f"총 {n}건 | 승률 {win_rate:.1f}% | 평균 {avg_ret:+.2f}% | "
    f"PF {pf:.2f} | CAGR {cagr*100:+.1f}% | 평균갭 {avg_gap:+.2f}%",
    fontsize=11, fontweight="bold"
)

ax = axes[0, 0]
ax.hist(df["return_pct"], bins=40, color="steelblue",
        edgecolor="white", alpha=0.85)
ax.axvline(0,       color="red",    linestyle="--", lw=1.5)
ax.axvline(avg_ret, color="orange", linestyle="-",  lw=1.5,
           label=f"평균 {avg_ret:+.2f}%")
ax.set_title("수익률 분포")
ax.legend()

ax = axes[0, 1]
clr = {"target":"#2ecc71","circuit":"#e74c3c",
       "time":"#f39c12","half_60d":"#3498db"}
top_ec = exit_cnt.head(6)
ax.pie(top_ec.values,
       labels=[f"{k[:15]}\n{v}건({v/n*100:.0f}%)" for k,v in top_ec.items()],
       colors=[clr.get(k.split("+")[0], "#95a5a6") for k in top_ec.index],
       startangle=140, textprops={"fontsize": 7.5})
ax.set_title("청산 유형")

ax = axes[0, 2]
ax.hist(df["gap_pct"], bins=40, color="purple",
        edgecolor="white", alpha=0.8)
ax.axvline(0,       color="black", linestyle="--", lw=1.5)
ax.axvline(avg_gap, color="orange", linestyle="-", lw=1.5,
           label=f"평균갭 {avg_gap:+.2f}%")
ax.set_title("진입 갭 분포 (다음날시가-전날종가)")
ax.set_xlabel("Gap %")
ax.legend()

ax = axes[1, 0]
cum = (1 + df["return_pct"] / 100).cumprod() - 1
ax.plot(range(len(cum)), cum * 100, color="navy", lw=1.5)
ax.axhline(0, color="red", linestyle="--", lw=1)
ax.fill_between(range(len(cum)), cum*100, 0,
                where=(cum >= 0), alpha=0.2, color="green")
ax.fill_between(range(len(cum)), cum*100, 0,
                where=(cum < 0),  alpha=0.2, color="red")
ax.set_title(f"누적 수익률 (CAGR {cagr*100:+.1f}%)")
ax.set_xlabel("Trade #")

ax = axes[1, 1]
bar_c = ["#2ecc71" if v >= 0 else "#e74c3c" for v in yearly["avg_ret"]]
ax.bar(yearly.index.astype(str), yearly["avg_ret"],
       color=bar_c, edgecolor="white")
ax.axhline(0, color="black", lw=0.8)
for x, (yr, row) in enumerate(yearly.iterrows()):
    ax.text(x, row["avg_ret"] + (0.3 if row["avg_ret"] >= 0 else -0.7),
            f'{int(row["trades"])}건', ha="center", fontsize=8)
ax.set_title("연도별 평균 수익률")
ax.tick_params(axis="x", rotation=45)

ax = axes[1, 2]
ax.scatter(df["gap_pct"], df["return_pct"],
           c=df["return_pct"], cmap="RdYlGn",
           alpha=0.4, s=20, vmin=-25, vmax=20)
ax.axhline(0, color="black", lw=0.8, linestyle="--")
ax.axvline(0, color="gray",  lw=0.8, linestyle=":")
ax.set_title("갭 크기 vs 수익률")
ax.set_xlabel("Gap % (다음날시가 - 전날종가)")
ax.set_ylabel("Return %")

plt.tight_layout()
plt.savefig("backtest_v10_results.png", dpi=150, bbox_inches="tight")
df.to_csv("backtest_v10_trades.csv", index=False)
print("\n📊 backtest_v10_results.png 저장 완료")
print("📄 backtest_v10_trades.csv 저장 완료")
print("✅ v10 백테스트 완료")
