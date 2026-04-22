"""
scan_today.py
=============
오늘 (2026-03-07) 기준 나스닥 100 종목 중
v10 매수 조건 충족 종목 스캔

조건:
  1. 현재가 < MA200
  2. VIX ≥ 25
  3. RSI(14) < 40  OR  CCI(20) < -100
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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

VIX_MIN = 25
RSI_MAX = 40
CCI_MIN = -100

def compute(d):
    d = d.copy()
    c = d["Close"]
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

# ── VIX 현재값
print("📥 VIX 조회 중...")
vix_raw = yf.download("^VIX", period="5d", auto_adjust=True, progress=False)
_vc = vix_raw["Close"]
if isinstance(_vc, pd.DataFrame): _vc = _vc.iloc[:, 0]
vix_today = float(_vc.iloc[-1])
vix_date  = _vc.index[-1].strftime("%Y-%m-%d")
print(f"  VIX: {vix_today:.2f}  ({vix_date} 기준)")
print(f"  VIX ≥ {VIX_MIN} 조건: {'✅ 충족' if vix_today >= VIX_MIN else '❌ 미충족'}")

# ── 종목 스캔
print(f"\n📊 {len(TICKERS)}개 종목 스캔 중...\n")

candidates = []
skipped    = []

for i, tk in enumerate(TICKERS, 1):
    try:
        raw = yf.download(tk, period="300d", auto_adjust=True, progress=False)
        if raw.empty or len(raw) < 220:
            skipped.append(tk)
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        d   = raw[["Open","High","Low","Close","Volume"]].copy()
        d   = compute(d)
        row = d.dropna(subset=["MA200","RSI","CCI"]).iloc[-1]

        price  = float(row["Close"])
        ma200  = float(row["MA200"])
        rsi    = float(row["RSI"])
        cci    = float(row["CCI"])
        pct_vs_ma200 = (price / ma200 - 1) * 100

        cond1 = price < ma200
        cond2 = vix_today >= VIX_MIN
        cond3 = (rsi < RSI_MAX) or (cci < CCI_MIN)
        all_ok = cond1 and cond2 and cond3

        if all_ok:
            candidates.append({
                "ticker"      : tk,
                "현재가"       : round(price, 2),
                "MA200"       : round(ma200, 2),
                "MA200대비(%)" : round(pct_vs_ma200, 1),
                "RSI"         : round(rsi, 1),
                "CCI"         : round(cci, 1),
                "RSI충족"      : "✅" if rsi < RSI_MAX else "  ",
                "CCI충족"      : "✅" if cci < CCI_MIN else "  ",
            })
    except Exception as e:
        skipped.append(f"{tk}({e})")

# ── 결과 출력
print("=" * 72)
print(f"  v10 매수 조건 충족 종목 (2026-03-07 기준)")
print(f"  조건: 현재가 < MA200  +  VIX({vix_today:.1f}) ≥ 25  +  RSI<40 OR CCI<-100")
print("=" * 72)

if not candidates:
    print(f"\n  ⚠️  조건 충족 종목 없음")
    if vix_today < VIX_MIN:
        print(f"  → VIX({vix_today:.1f})가 {VIX_MIN} 미만이므로 전체 매수 차단 상태")
else:
    df = pd.DataFrame(candidates).sort_values("MA200대비(%)")
    print(f"\n  총 {len(df)}개 종목\n")
    print(f"  {'종목':<8} {'현재가':>8} {'MA200':>8} {'MA200대비':>10} "
          f"{'RSI':>6} {'CCI':>8} {'RSI<40':>7} {'CCI<-100':>9}")
    print("  " + "-" * 68)
    for _, r in df.iterrows():
        print(f"  {r['ticker']:<8} {r['현재가']:>8.2f} {r['MA200']:>8.2f} "
              f"{r['MA200대비(%)']:>9.1f}% {r['RSI']:>6.1f} {r['CCI']:>8.1f} "
              f"{r['RSI충족']:>7} {r['CCI충족']:>9}")

print("\n" + "=" * 72)
print(f"  VIX: {vix_today:.2f}  |  스캔 종목: {len(TICKERS)}개  |  충족: {len(candidates)}개  |  스킵: {len(skipped)}개")
print("=" * 72)
