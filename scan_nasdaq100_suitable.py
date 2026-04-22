"""
scan_nasdaq100_suitable.py
==========================
나스닥 100 종목 중 v10 전략에 적합한 종목 필터링
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

START = "2010-01-01"
END   = "2026-01-01"

VIX_MIN    = 25
TARGET_PCT = 0.20
CIRCUIT_PCT= 0.25

# 나스닥 100 전체 종목 (2024년 기준)
NASDAQ100 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","ASML","AZN","TMUS","AMD","PEP","QCOM","INTU","CSCO","TXN",
    "ISRG","AMGN","AMAT","MU","BKNG","PANW","ADI","VRTX","GILD","REGN",
    "MELI","KDP","LRCX","SNPS","CDNS","CTAS","SBUX","KLAC","PYPL","MDLZ",
    "CHTR","CEG","NXPI","ORLY","WDAY","ABNB","MAR","TEAM","PCAR","CRWD",
    "ROST","ADSK","MCHP","CSX","DXCM","IDXX","ADP","FTNT","MNST","FAST",
    "PAYX","ODFL","EA","KHC","CTSH","BIIB","ZS","GEHC","VRSK","CSGP",
    "ON","DDOG","ROP","TTD","ANSS","CPRT","MRNA","XEL","DLTR","FANG",
    "CDW","EBAY","WBD","TTWO","GFS","MDB","SIRI","ILMN","ZM","RIVN",
    "LCID","SMCI","MRVL","INTC","PLTR","HOOD","ARM","TSLL","TQQQ",
]

# 레버리지 ETF / 특수 제외
EXCLUDE = {"TSLL","TQQQ","SOXL","GOOG"}  # GOOG = GOOGL 중복

tickers = [t for t in NASDAQ100 if t not in EXCLUDE]

def dl(tk, start, end):
    raw = yf.download(tk, start=start, end=end,
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

print("📥 VIX 다운로드...")
vix_raw = yf.download("^VIX", start=START, end=END,
                      auto_adjust=True, progress=False)
_vc = vix_raw["Close"]
if isinstance(_vc, pd.DataFrame): _vc = _vc.iloc[:, 0]
vix = pd.Series(_vc.values, index=_vc.index, dtype=float).dropna()
print(f"✅ VIX {len(vix)}일\n")

results = []

for tk in tickers:
    d = dl(tk, START, END)
    if len(d) < 250:
        continue
    d  = compute(d)
    dc = d.dropna(subset=["MA200","RSI","CCI"])

    close  = dc["Close"]
    rsi    = dc["RSI"]
    cci    = dc["CCI"]
    common = dc.index.intersection(vix.index)
    if len(common) < 50:
        continue
    vx   = vix.reindex(common)
    cond = (
        (close < dc["MA200"]) &
        (vx >= VIX_MIN) &
        ((rsi < 40) | (cci < -100))
    )
    sig_idx = dc.index[cond.reindex(dc.index).fillna(False)]

    trades = []
    pos_exit = {}

    for sig_day in sig_idx:
        idx = dc.index.get_loc(sig_day)
        if idx + 1 >= len(dc):
            continue
        entry_day  = dc.index[idx + 1]
        entry_open = float(dc["Open"].iloc[idx + 1])
        if pd.isna(entry_open):
            continue
        if tk in pos_exit and pos_exit[tk] > entry_day:
            continue

        future = dc.loc[dc.index >= entry_day]
        circuit = entry_open * (1 - CIRCUIT_PCT)
        target  = entry_open * (1 + TARGET_PCT)
        half_exited = False
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
            if i + 1 == 60 and not half_exited and (cl - entry_open) / entry_open > 0:
                exit_records.append((fdt, cl, 0.5, "half_60d"))
                half_exited = True
                continue
            if i + 1 >= 120:
                exit_records.append((fdt, cl, 0.5 if half_exited else 1.0, "time"))
                break

        if not exit_records:
            continue

        total_wt = sum(r[2] for r in exit_records)
        blended  = sum((r[1] - entry_open) / entry_open * r[2] for r in exit_records)
        ret      = blended / total_wt if total_wt > 0 else 0
        pos_exit[tk] = exit_records[-1][0]
        trades.append(ret)

    if len(trades) < 3:
        continue

    n        = len(trades)
    rets     = np.array(trades)
    win_rate = (rets > 0).mean() * 100
    avg_ret  = rets.mean() * 100
    wins     = rets[rets > 0]
    losses   = rets[rets < 0]
    pf       = (wins.sum() / -losses.sum()
                if len(losses) and losses.sum() < 0 else 99)

    results.append({
        "ticker"  : tk,
        "n"       : n,
        "win_rate": round(win_rate, 1),
        "avg_ret" : round(avg_ret, 2),
        "pf"      : round(pf, 2),
    })

df = pd.DataFrame(results)
df = df.sort_values(["win_rate","avg_ret"], ascending=False).reset_index(drop=True)

# 기준: 승률 60% 이상 & 평균수익 +3% 이상 & PF 1.3 이상 & 트레이드 3건 이상
good = df[(df["win_rate"] >= 60) & (df["avg_ret"] >= 3.0) & (df["pf"] >= 1.3)]

print(f"{'='*60}")
print(f"  v10 전략 적합 종목 (나스닥 100 기준)")
print(f"  기준: 승률 ≥60% & 평균수익 ≥+3% & PF ≥1.3 & 거래 ≥3건")
print(f"{'='*60}")
print(good.to_string(index=False))
print(f"\n  총 {len(good)}개 종목")
print(f"\n티커 목록:")
print(", ".join(good["ticker"].tolist()))
