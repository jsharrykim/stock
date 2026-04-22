"""
backtest_kr_v10.py — 한국 주식 v10 전략 백테스트
================================================================
전략 (미국 v10과 동일):
  - 진입 조건: MA200↓ + VIX≥25(미국 VIX 그대로) + (RSI<40 OR CCI<-100)
  - 진입가: 신호 다음날 시가
  - 청산: +20% / -25% CB / 60거래일 수익 중 / 120거래일 타임아웃

유니버스:
  - 코스피 전체 (yfinance .KS)
  - 코스닥 시총 Top100 (yfinance .KQ)

데이터:
  - yfinance (무료, 수정주가 기준)
  - VIX: Yahoo Finance ^VIX (미국 VIX 그대로 사용)
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
START        = "2015-01-01"   # 한국 주식은 10년치 확보
END          = "2026-03-01"
VIX_MIN      = 25
TARGET_PCT   = 0.20
CIRCUIT_PCT  = 0.25
HALF_EXIT    = 60    # 거래일 기준
MAX_HOLD     = 120
MAX_POSITIONS = 5
MAX_DAILY    = 5

# ──────────────────────────────────────────────
# 코스피 전체 + 코스닥 Top100 종목 코드
# (yfinance: 코스피 .KS, 코스닥 .KQ)
# pykrx로 평일에 동적 수집하거나, 아래 주요 종목 목록 사용
# ──────────────────────────────────────────────
KOSPI_TICKERS = [
    # 코스피 시총 상위 100 (2025 기준)
    "005930","000660","035420","005380","000270",
    "068270","105560","055550","012330","028260",
    "066570","003550","051910","034730","017670",
    "015760","032830","086790","018260","009150",
    "011170","316140","003670","010950","011790",
    "096770","302440","033780","024110","000810",
    "090430","010130","035720","006400","011200",
    "004020","008770","034020","003490","030200",
    "023530","047050","259960","000100","002790",
    "097950","018880","004370","009830","011780",
    "139480","071050","010060","161390","267250",
    "004170","001040","180640","036570","002380",
    "009540","000720","042660","085310","005490",
    "007070","016360","008560","006360","021240",
    "025840","052690","000080","011300","005940",
    "004490","000990","005830","326030","005850",
    "002550","001800","026960","002310","009410",
    "004800","012750","006280","001270","011070",
    "010140","005600","000480","014680","003230",
    "019170","006650","028050","004000","000020",
]

KOSDAQ_TOP100 = [
    # 코스닥 시총 상위 100 (2025 기준)
    "247540","373220","196170","086520","091990",
    "357780","145020","036030","263750","078600",
    "041510","054040","000250","900140","035900",
    "058970","950130","311690","290650","052260",
    "238490","039030","108320","112040","263720",
    "293490","241560","131370","214150","065060",
    "240810","039200","122870","211050","259960",
    "095340","066900","064760","236810","032640",
    "214430","031860","054620","060280","030520",
    "145720","049630","003230","043360","067160",
    "204840","263800","054490","225570","023160",
    "035080","085370","002860","083790","032500",
    "078020","036810","140940","048260","319400",
    "019570","048870","033160","064260","014620",
    "053350","093320","094970","039440","041960",
    "048470","051500","077970","214180","025980",
    "036810","126600","237690","011280","041020",
    "071200","049270","039290","191420","000440",
    "033230","025980","057500","140410","060900",
    "073490","159580","237820","038870","099440",
]

def make_yf_tickers(codes, suffix):
    return [c + suffix for c in codes]

KOSPI_YF  = make_yf_tickers(KOSPI_TICKERS,  ".KS")
KOSDAQ_YF = make_yf_tickers(KOSDAQ_TOP100, ".KQ")
ALL_TICKERS = KOSPI_YF + KOSDAQ_YF

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
    needed = [c for c in ["Open","High","Low","Close","Volume"] if c in raw.columns]
    if len(needed) < 4:
        return pd.DataFrame()
    return raw[needed].copy()


def compute(d):
    d = d.copy()
    c = d["Close"]
    d["MA200"] = c.rolling(200).mean()
    # RSI(14) — Wilder's EMA
    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta.clip(upper=0))
    avg_gain = gain.ewm(com=13, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(com=13, min_periods=14, adjust=False).mean()
    rs        = avg_gain / avg_loss.replace(0, np.nan)
    d["RSI"]  = 100 - 100 / (1 + rs)
    # CCI(20)
    tp     = (d["High"] + d["Low"] + c) / 3
    tp_ma  = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    d["CCI"] = (tp - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))
    return d


# ──────────────────────────────────────────────
# VIX 다운로드
# ──────────────────────────────────────────────
print("📥 VIX 다운로드 중...")
vix_raw = yf.download("^VIX", start=START, end=END,
                      auto_adjust=True, progress=False)
_vc = vix_raw["Close"]
if isinstance(_vc, pd.DataFrame): _vc = _vc.iloc[:, 0]
vix = pd.Series(_vc.values, index=_vc.index, dtype=float).dropna()
print(f"✅ VIX {len(vix)}일")

# ──────────────────────────────────────────────
# 종목 데이터 다운로드
# ──────────────────────────────────────────────
print(f"📥 한국 종목 데이터 다운로드 중... (총 {len(ALL_TICKERS)}개)")
stock_data = {}
fail_list  = []

for i, tk in enumerate(ALL_TICKERS, 1):
    d = dl_ohlcv(tk, START, END)
    if len(d) < 250:
        fail_list.append(tk)
        continue
    d = compute(d)
    stock_data[tk] = d
    if i % 30 == 0:
        print(f"  {i}/{len(ALL_TICKERS)} 완료 ({len(stock_data)}개 유효)")

print(f"✅ {len(stock_data)}개 종목 로드 (실패 {len(fail_list)}개)")
if fail_list:
    print(f"  실패 종목: {fail_list[:10]}{'...' if len(fail_list)>10 else ''}")

# ──────────────────────────────────────────────
# 신호 생성
# ──────────────────────────────────────────────
print("🔍 신호 생성 중...")
signals_by_date = {}

for tk, d in stock_data.items():
    d_c = d.dropna(subset=["MA200", "RSI", "CCI"])
    if len(d_c) < 50:
        continue
    close = d_c["Close"]

    common = d_c.index.intersection(vix.index)
    if len(common) < 50:
        continue
    vx = vix.reindex(d_c.index)

    cond = (
        (close < d_c["MA200"]) &
        (vx >= VIX_MIN) &
        ((d_c["RSI"] < 40) | (d_c["CCI"] < -100))
    )

    sig_days = d_c.index[cond.reindex(d_c.index).fillna(False)]

    for sig_day in sig_days:
        idx = d_c.index.get_loc(sig_day)
        if idx + 1 >= len(d_c):
            continue
        entry_day  = d_c.index[idx + 1]
        entry_open = float(d_c["Open"].iloc[idx + 1])
        if pd.isna(entry_open) or entry_open <= 0:
            continue

        row = d_c.loc[sig_day]
        if entry_day not in signals_by_date:
            signals_by_date[entry_day] = []
        signals_by_date[entry_day].append({
            "ticker"   : tk,
            "market"   : "KOSPI" if tk.endswith(".KS") else "KOSDAQ",
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

print(f"✅ 원시 신호: {len(final_signals):,}건 ({len(signals_by_date)}거래일)")

# ──────────────────────────────────────────────
# 시뮬레이션
# ──────────────────────────────────────────────
print("⚙️  시뮬레이션 중...")
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
    weighted    = sum((r[1] - entry) / entry * r[2] for r in exit_records)
    blended_ret = weighted / total_pct if total_pct > 0 else 0
    last_exit   = exit_records[-1]
    reason      = ("+".join(r[3] for r in exit_records)
                   if len(exit_records) > 1 else exit_records[0][3])
    gap_pct = (entry - sig["close_sig"]) / sig["close_sig"] * 100

    trades.append({
        "sig_date"   : sig["sig_day"],
        "entry_date" : entry_day,
        "exit_date"  : last_exit[0],
        "ticker"     : tk,
        "market"     : sig["market"],
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

if len(trades) == 0:
    print("⚠️  트레이드가 없습니다. 종목 수신 실패 가능성이 있으니 fail_list를 확인하세요.")
    exit()

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
avg_gap  = df["gap_pct"].mean()

# 포트폴리오 CAGR
capital = 1.0
for ed, group in df.groupby("entry_date"):
    w       = 1.0 / max(len(group), MAX_POSITIONS)
    batch   = (group["return_pct"] / 100 * w).sum()
    capital *= (1 + batch)
years = (df["exit_date"].max() - df["entry_date"].min()).days / 365.25
cagr  = capital ** (1 / years) - 1

# 시장별 분석
market_stats = df.groupby("market").agg(
    trades   =("return_pct","count"),
    win_rate =("win",        lambda x: x.mean()*100),
    avg_ret  =("return_pct","mean"),
    avg_win  =("return_pct", lambda x: x[x>0].mean() if (x>0).any() else 0),
    avg_loss =("return_pct", lambda x: x[x<=0].mean() if (x<=0).any() else 0),
)

print("\n" + "="*66)
print("  한국 주식 v10 (MA200↓ + VIX≥25 + RSI<40 OR CCI<-100) 백테스트")
print("="*66)
print(f"  기간        : {START} ~ {END}")
print(f"  유니버스    : 코스피 {len(KOSPI_TICKERS)}개 + 코스닥 {len(KOSDAQ_TOP100)}개 목록")
print(f"  유효 종목   : {len(stock_data)}개")
print(f"  진입 기준   : 신호 다음날 시가")
print(f"  총 트레이드 : {n}건")
print(f"  승 률       : {win_rate:.1f}%")
print(f"  평균 수익률 : {avg_ret:+.2f}%")
print(f"  기대값      : {ev:+.2f}%")
print(f"  승자 평균   : {avg_win:+.2f}%")
print(f"  패자 평균   : {avg_loss:+.2f}%")
print(f"  Profit Factor: {pf:.2f}" if not np.isnan(pf) else "  Profit Factor: N/A")
print(f"  포트CAGR    : {cagr*100:+.2f}%")
print(f"  최대연속손실: {max_cl}건")
print(f"  평균보유일수: {df['hold_days'].mean():.0f}일")
print(f"  청산 유형:")
for r, c in exit_cnt.items():
    print(f"    {r:<25}: {c:4d}건 ({c/n*100:.1f}%)")

print(f"\n  ── 시장별 성과 ──")
print(market_stats.round(2).to_string())

print(f"\n  ── 미국 v10 vs 한국 v10 비교 ──")
us = {"wr": 75.7, "ar": 7.82, "cagr": 12.2}  # CONTEXT.md 그룹A 기준
rows = [
    ("승률",      f"{us['wr']:.1f}%",      f"{win_rate:.1f}%"),
    ("평균수익률", f"{us['ar']:+.2f}%",     f"{avg_ret:+.2f}%"),
    ("포트CAGR",  f"{us['cagr']:+.1f}%",   f"{cagr*100:+.1f}%"),
]
fmt2 = "  {:<14} {:>18} {:>18}"
print(fmt2.format("지표", "미국 v10 (그룹A)", "한국 v10"))
print("  " + "-"*52)
for label, a, b in rows:
    print(fmt2.format(label, a, b))

# 연도별
df["year"] = df["entry_date"].dt.year
yearly = df.groupby("year").agg(
    trades   =("return_pct","count"),
    avg_ret  =("return_pct","mean"),
    win_rate =("win",        lambda x: x.mean()*100),
)
print(f"\n  연도별 성과:")
print(yearly.round(2).to_string())

print(f"\n▶ 수익 상위 10:")
print(df.nlargest(10, "return_pct")[
    ["entry_date","ticker","market","entry","return_pct","exit_reason","hold_days"]
].to_string(index=False))

print(f"\n▶ 손실 하위 10:")
print(df.nsmallest(10, "return_pct")[
    ["entry_date","ticker","market","entry","return_pct","exit_reason","hold_days"]
].to_string(index=False))

# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    f"한국 주식 v10 (MA200↓ + VIX≥25 + RSI<40 OR CCI<-100)\n"
    f"총 {n}건 | 승률 {win_rate:.1f}% | 평균 {avg_ret:+.2f}% | "
    f"CAGR {cagr*100:+.1f}% | 기간 {START}~{END}",
    fontsize=11, fontweight="bold"
)

ax = axes[0, 0]
ax.hist(df["return_pct"], bins=40, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(0,       color="red",    linestyle="--", lw=1.5)
ax.axvline(avg_ret, color="orange", linestyle="-",  lw=1.5, label=f"평균 {avg_ret:+.2f}%")
ax.set_title("수익률 분포")
ax.legend()

ax = axes[0, 1]
clr = {"target":"#2ecc71","circuit":"#e74c3c","time":"#f39c12","half_60d":"#3498db"}
top_ec = exit_cnt.head(6)
ax.pie(top_ec.values,
       labels=[f"{k[:15]}\n{v}건({v/n*100:.0f}%)" for k,v in top_ec.items()],
       colors=[clr.get(k.split("+")[0], "#95a5a6") for k in top_ec.index],
       startangle=140, textprops={"fontsize": 7.5})
ax.set_title("청산 유형")

ax = axes[0, 2]
market_grp = df.groupby("market")["return_pct"]
for mkt, grp in market_grp:
    ax.hist(grp, bins=30, alpha=0.6, label=f"{mkt}({len(grp)}건)", edgecolor="white")
ax.axvline(0, color="red", linestyle="--", lw=1.5)
ax.set_title("시장별 수익률 분포")
ax.legend()

ax = axes[1, 0]
cum = (1 + df["return_pct"] / 100).cumprod() - 1
ax.plot(range(len(cum)), cum * 100, color="navy", lw=1.5)
ax.axhline(0, color="red", linestyle="--", lw=1)
ax.fill_between(range(len(cum)), cum*100, 0, where=(cum >= 0), alpha=0.2, color="green")
ax.fill_between(range(len(cum)), cum*100, 0, where=(cum < 0),  alpha=0.2, color="red")
ax.set_title(f"누적 수익률 (CAGR {cagr*100:+.1f}%)")
ax.set_xlabel("Trade #")

ax = axes[1, 1]
bar_c = ["#2ecc71" if v >= 0 else "#e74c3c" for v in yearly["avg_ret"]]
ax.bar(yearly.index.astype(str), yearly["avg_ret"], color=bar_c, edgecolor="white")
ax.axhline(0, color="black", lw=0.8)
for x, (yr, row) in enumerate(yearly.iterrows()):
    ax.text(x, row["avg_ret"] + (0.3 if row["avg_ret"] >= 0 else -0.8),
            f'{int(row["trades"])}건', ha="center", fontsize=8)
ax.set_title("연도별 평균 수익률")
ax.tick_params(axis="x", rotation=45)

ax = axes[1, 2]
for mkt, grp in df.groupby("market"):
    ax.scatter(grp["rsi_entry"], grp["return_pct"],
               alpha=0.35, s=15, label=mkt)
ax.axhline(0, color="black", lw=0.8, linestyle="--")
ax.axvline(40, color="gray", lw=0.8, linestyle=":")
ax.set_title("진입 RSI vs 수익률")
ax.set_xlabel("RSI at Entry")
ax.set_ylabel("Return %")
ax.legend()

plt.tight_layout()
plt.savefig("backtest_kr_v10_results.png", dpi=150, bbox_inches="tight")
df.to_csv("backtest_kr_v10_trades.csv", index=False)
print("\n📊 backtest_kr_v10_results.png 저장 완료")
print("📄 backtest_kr_v10_trades.csv 저장 완료")
print("✅ 한국 v10 백테스트 완료")
