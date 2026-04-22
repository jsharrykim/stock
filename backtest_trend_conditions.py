"""
backtest_trend_conditions.py
============================
v10 전략(저점 매수) 에 상승 추세 편승 조건을 OR로 추가하는 백테스트.

[기준] 원본 v10 (저점):
  - 매수: 현재가 < MA200 + VIX≥25 + (RSI<40 OR CCI<-100)

[추가 후보 조건 — 각각 OR로 추가]:
  A. 현재가 > MA200 + 50 < RSI < 70   (상승 추세 초입, 과매수 아닌 구간)
  B. MA50 > MA200 + 현재가 > MA200 + RSI > 50  (골든크로스 이후 상승 추세 유지)
  C. 현재가 > MA200 + 현재가 > 52주최고가×0.75 + RSI < 70  (눌림목 아닌 고점 근처)
  D. 20일수익률 > +5% + 현재가 > MA200  (단기 모멘텀 돌파)
  E. MA50 > MA200 + 50 < RSI < 65 + 현재가 > MA50  (건강한 상승, 과매수 아닌)
  F. MA50 > MA200 + RSI 45~65 + VIX < 20  (안정 상승장 편승)

비교 그룹:
  0. 원본 v10만 (기준)
  A~F: 원본 v10 OR 각 추세 조건
  ALL: 원본 v10 OR A~F 전부

고정 파라미터:
  TARGET_PCT : 0.20
  CIRCUIT_PCT: 0.25
  HALF_EXIT  : 60
  MAX_HOLD   : 120
  MAX_POS    : 5
  MAX_DAILY  : 5
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

# ── 기간 & 파라미터 ────────────────────────────────
START        = "2010-01-01"
END          = "2026-01-01"
TARGET_PCT   = 0.20
CIRCUIT_PCT  = 0.25
HALF_EXIT    = 60
MAX_HOLD     = 120
MAX_POSITIONS= 5
MAX_DAILY    = 5
VIX_MIN_BASE = 25   # 원본 v10 VIX 기준

# ── 종목 유니버스 ──────────────────────────────────
TICKERS = sorted(set([
    "SNPS","COST","AZN","AMGN","MDLZ","FTNT","CSGP","CDNS","ADP","FAST",
    "ADI","TXN","PAYX","BKNG","KLAC","MNST","ORLY","HOOD","CPRT","ISRG",
    "PANW","CDW","INTC","AAPL","AVGO","AMD","MSFT","GOOGL","NVDA","META",
    "TSLA","PLTR","MELI","MCHP","AMZN","SMCI","AMAT","MU","LRCX","CSX",
    "QCOM","ROP","INTU","ON","NXPI","STX","ASTS","AVAV","IONQ","SGML",
    "GOOG","NFLX","TMUS","ADBE","PEP","CSCO","MRVL","CRWD","DDOG","ZS",
    "TEAM","KDP","IDXX","REGN","VRTX","GILD","BIIB","ILMN","MRNA",
    "ODFL","PCAR","CTSH","VRSK","WDAY","PDD","DLTR","SBUX","ROST",
    "LULU","EBAY","MAR","CTAS","EA","CHTR","CMCSA","EXC","XEL","AEP",
    "MPWR","ENPH","SEDG","COIN","DOCU","ZM","OKTA","PTON",
]))

# ── 한글 폰트 ──────────────────────────────────────
_kr_fonts = [f.name for f in fm.fontManager.ttflist
             if "Apple" in f.name or "Malgun" in f.name
             or "NanumGothic" in f.name or "Noto" in f.name]
if _kr_fonts:
    plt.rcParams["font.family"] = _kr_fonts[0]
plt.rcParams["axes.unicode_minus"] = False


# ══════════════════════════════════════════════════
# 데이터 로드 & 지표 계산
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

    # 이평선
    d["MA50"]  = c.rolling(50).mean()
    d["MA200"] = c.rolling(200).mean()

    # RSI (SMA 기반)
    delta = c.diff()
    g  = delta.clip(lower=0).rolling(14).mean()
    lo = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100 / (1 + g / lo.replace(0, np.nan))

    # CCI
    tp     = (d["High"] + d["Low"] + c) / 3
    tp_ma  = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    d["CCI"] = (tp - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))

    # 20일 수익률 (모멘텀)
    d["Ret20"] = c.pct_change(20) * 100

    # 52주 최고가
    d["High52w"] = d["High"].rolling(252).max()

    return d


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
# 조건 정의 함수들
# ══════════════════════════════════════════════════
def cond_base(row, vx):
    """원본 v10: 현재가<MA200 + VIX≥25 + (RSI<40 OR CCI<-100)"""
    return (
        row["Close"] < row["MA200"] and
        vx >= VIX_MIN_BASE and
        (row["RSI"] < 40 or row["CCI"] < -100)
    )

def cond_A(row):
    """상승 추세 초입: 현재가>MA200 + 50<RSI<70"""
    return (
        row["Close"] > row["MA200"] and
        50 < row["RSI"] < 70
    )

def cond_B(row):
    """골든크로스 추세: MA50>MA200 + 현재가>MA200 + RSI>50"""
    return (
        row["MA50"] > row["MA200"] and
        row["Close"] > row["MA200"] and
        row["RSI"] > 50
    )

def cond_C(row):
    """눌림목 아닌 상승: 현재가>MA200 + 현재가>52주최고가×0.75 + RSI<70"""
    if pd.isna(row["High52w"]) or row["High52w"] == 0:
        return False
    return (
        row["Close"] > row["MA200"] and
        row["Close"] > row["High52w"] * 0.75 and
        row["RSI"] < 70
    )

def cond_D(row):
    """단기 모멘텀 돌파: 20일수익률>+5% + 현재가>MA200"""
    return (
        row["Close"] > row["MA200"] and
        row["Ret20"] > 5.0
    )

def cond_E(row):
    """건강한 상승: MA50>MA200 + 50<RSI<65 + 현재가>MA50"""
    return (
        row["MA50"] > row["MA200"] and
        50 < row["RSI"] < 65 and
        row["Close"] > row["MA50"]
    )

def cond_F(row, vx):
    """안정 상승장: MA50>MA200 + 45<RSI<65 + VIX<20"""
    return (
        row["MA50"] > row["MA200"] and
        45 < row["RSI"] < 65 and
        vx < 20
    )

# 그룹 정의: (레이블, 조건함수, 설명)
GROUPS = {
    "원본v10": {
        "label": "원본v10",
        "desc" : "현재가<MA200 + VIX≥25 + (RSI<40 OR CCI<-100)",
        "fn"   : lambda row, vx: cond_base(row, vx),
        "trend_only": False,
    },
    "+A": {
        "label": "+A (RSI추세)",
        "desc" : "원본 OR [현재가>MA200 + 50<RSI<70]",
        "fn"   : lambda row, vx: cond_base(row, vx) or cond_A(row),
        "trend_only": False,
    },
    "+B": {
        "label": "+B (골든크로스)",
        "desc" : "원본 OR [MA50>MA200 + 현재가>MA200 + RSI>50]",
        "fn"   : lambda row, vx: cond_base(row, vx) or cond_B(row),
        "trend_only": False,
    },
    "+C": {
        "label": "+C (52주고점근처)",
        "desc" : "원본 OR [현재가>MA200 + 현재가>52주최고가×75% + RSI<70]",
        "fn"   : lambda row, vx: cond_base(row, vx) or cond_C(row),
        "trend_only": False,
    },
    "+D": {
        "label": "+D (모멘텀)",
        "desc" : "원본 OR [20일수익률>+5% + 현재가>MA200]",
        "fn"   : lambda row, vx: cond_base(row, vx) or cond_D(row),
        "trend_only": False,
    },
    "+E": {
        "label": "+E (건강한상승)",
        "desc" : "원본 OR [MA50>MA200 + 50<RSI<65 + 현재가>MA50]",
        "fn"   : lambda row, vx: cond_base(row, vx) or cond_E(row),
        "trend_only": False,
    },
    "+F": {
        "label": "+F (안정상승장)",
        "desc" : "원본 OR [MA50>MA200 + 45<RSI<65 + VIX<20]",
        "fn"   : lambda row, vx: cond_base(row, vx) or cond_F(row, vx),
        "trend_only": False,
    },
    "+ALL": {
        "label": "+ALL",
        "desc" : "원본 OR A OR B OR C OR D OR E OR F",
        "fn"   : lambda row, vx: (
            cond_base(row, vx) or cond_A(row) or cond_B(row) or
            cond_C(row) or cond_D(row) or cond_E(row) or cond_F(row, vx)
        ),
        "trend_only": False,
    },
    # 추세만 단독 (참고용)
    "A단독": {
        "label": "A단독",
        "desc" : "[현재가>MA200 + 50<RSI<70] 단독",
        "fn"   : lambda row, vx: cond_A(row),
        "trend_only": True,
    },
    "E단독": {
        "label": "E단독",
        "desc" : "[MA50>MA200 + 50<RSI<65 + 현재가>MA50] 단독",
        "fn"   : lambda row, vx: cond_E(row),
        "trend_only": True,
    },
}


# ══════════════════════════════════════════════════
# 신호 생성 함수
# ══════════════════════════════════════════════════
def build_signals(condition_fn):
    signals_by_date = {}
    for tk, d in stock_data.items():
        need_cols = ["MA50","MA200","RSI","CCI","Ret20","High52w"]
        d_c = d.dropna(subset=["MA200","RSI","CCI"])

        common = d_c.index.intersection(vix.index)
        if len(common) < 50:
            continue
        vx_s = vix.reindex(d_c.index)

        for dt in d_c.index:
            row = d_c.loc[dt]
            vx  = vx_s.loc[dt] if dt in vx_s.index else np.nan
            if pd.isna(vx):
                continue

            # 필요 컬럼 NaN 체크
            if any(pd.isna(row.get(c, np.nan)) for c in ["MA200","RSI","CCI"]):
                continue

            try:
                triggered = condition_fn(row, vx)
            except Exception:
                continue

            if not triggered:
                continue

            idx = d_c.index.get_loc(dt)
            if idx + 1 >= len(d_c):
                continue
            entry_day  = d_c.index[idx + 1]
            entry_open = float(d_c["Open"].iloc[idx + 1])
            if pd.isna(entry_open):
                continue

            if entry_day not in signals_by_date:
                signals_by_date[entry_day] = []
            signals_by_date[entry_day].append({
                "ticker"   : tk,
                "sig_day"  : dt,
                "entry_day": entry_day,
                "entry"    : entry_open,
                "rsi"      : float(row["RSI"]),
                "cci"      : float(row["CCI"]),
                "close_sig": float(row["Close"]),
                "above_ma" : row["Close"] > row["MA200"],
            })

    final = []
    for entry_day, items in sorted(signals_by_date.items()):
        items.sort(key=lambda x: x["rsi"])
        for item in items[:MAX_DAILY]:
            final.append(item)
    return final


# ══════════════════════════════════════════════════
# 시뮬레이션
# ══════════════════════════════════════════════════
def run_simulation(signals):
    trades        = []
    pos_exit_date = {}

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

        cb_price    = entry * (1 - CIRCUIT_PCT)
        tgt_price   = entry * (1 + TARGET_PCT)
        half_exited = False
        exit_records= []

        for i, (fdt, row) in enumerate(future.iterrows()):
            lo = float(row["Low"])
            hi = float(row["High"])
            cl = float(row["Close"])

            if hi >= tgt_price:
                exit_records.append((fdt, tgt_price, 0.5 if half_exited else 1.0, "target"))
                break
            if lo <= cb_price:
                exit_records.append((fdt, cb_price, 0.5 if half_exited else 1.0, "circuit"))
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
            "above_ma"   : sig["above_ma"],
        })
        pos_exit_date[tk] = last_exit[0]

    return pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)


# ══════════════════════════════════════════════════
# 전체 그룹 실행
# ══════════════════════════════════════════════════
print("\n⚙️  그룹별 시뮬레이션 실행 중...")
results = {}

for key, g in GROUPS.items():
    lbl = g["label"]
    print(f"  [{lbl}] 신호 생성...", end="", flush=True)
    signals = build_signals(g["fn"])
    print(f" {len(signals):,}건 →", end="", flush=True)

    df = run_simulation(signals)
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

    cap = 1.0
    for _, grp in df.groupby("entry_date"):
        w   = 1.0 / max(len(grp), MAX_POSITIONS)
        cap *= (1 + (grp["return_pct"] / 100 * w).sum())
    yrs  = (df["exit_date"].max() - df["entry_date"].min()).days / 365.25
    cagr = cap ** (1 / yrs) - 1 if yrs > 0 else 0

    avg_hold    = df["hold_days"].mean()
    exit_dist   = df["exit_reason"].value_counts(normalize=True) * 100
    tgt_share   = exit_dist.get("target", 0) + exit_dist.get("half_60d+target", 0)
    cb_share    = exit_dist.get("circuit", 0) + exit_dist.get("half_60d+circuit", 0)

    # 추세 진입 비율 (MA200 위에서 들어간 거래)
    trend_ratio = df["above_ma"].mean() * 100 if "above_ma" in df.columns else 0

    results[key] = {
        "label"          : lbl,
        "desc"           : g["desc"],
        "trend_only"     : g["trend_only"],
        "n"              : n,
        "signal_count"   : len(signals),
        "win_rate"       : wr,
        "avg_ret"        : avg_ret,
        "avg_win"        : avg_w,
        "avg_loss"       : avg_l,
        "pf"             : pf,
        "ev"             : ev,
        "cagr"           : cagr * 100,
        "max_consec_loss": max_cl,
        "avg_hold_days"  : avg_hold,
        "exit_dist"      : dict(df["exit_reason"].value_counts()),
        "target_hit_pct" : tgt_share,
        "circuit_hit_pct": cb_share,
        "trend_ratio"    : trend_ratio,
        "df"             : df,
    }
    print(f" 거래 {n}건, 승률 {wr:.1f}%, 평균 {avg_ret:+.2f}%, CAGR {cagr*100:+.1f}%")


# ══════════════════════════════════════════════════
# 결과 출력
# ══════════════════════════════════════════════════
print("\n" + "="*120)
print("  추세 편승 조건 추가 백테스트 결과 (2010-2026)")
print("  기준: v10 원본 | 추가 조건은 각각 OR로 추가 | TARGET +20% | CB -25%")
print("="*120)

hdr = (f"  {'그룹':<14} {'신호':>6} {'거래':>5} {'승률':>7} {'평균수익':>9} {'기대값':>8} "
       f"{'승평균':>8} {'패평균':>8} {'PF':>6} {'CAGR':>8} {'연속손':>5} "
       f"{'평균보유':>7} {'목표%':>6} {'CB%':>5} {'추세진입%':>9}")
print(hdr)
print("  " + "-"*115)

base_cagr = results.get("원본v10", {}).get("cagr", 0)
base_wr   = results.get("원본v10", {}).get("win_rate", 0)

for key, r in results.items():
    pf_str    = f"{r['pf']:.2f}" if not np.isnan(r['pf']) else " N/A"
    marker    = " ◀기준" if key == "원본v10" else ""
    separator = "  ---" if r["trend_only"] else ""
    if r["trend_only"] and key == list(results.keys())[list(results.keys()).index("A단독") if "A단독" in results else 0]:
        print("  " + "-"*115 + "  (참고: 추세 단독)")
    print(
        f"  {r['label']:<14} {r['signal_count']:>6} {r['n']:>5} "
        f"{r['win_rate']:>6.1f}% {r['avg_ret']:>+8.2f}% {r['ev']:>+7.2f}% "
        f"{r['avg_win']:>+7.2f}% {r['avg_loss']:>+7.2f}% {pf_str:>6} "
        f"{r['cagr']:>+7.1f}% {r['max_consec_loss']:>4}건 "
        f"{r['avg_hold_days']:>6.0f}일 "
        f"{r['target_hit_pct']:>5.1f}% {r['circuit_hit_pct']:>4.1f}% "
        f"{r['trend_ratio']:>8.1f}%{marker}"
    )

print("="*120)

# 조건 설명
print("\n[조건 설명]")
for key, r in results.items():
    print(f"  {r['label']:<14}: {r['desc']}")

# 연도별
print("\n[연도별 승률 & 평균수익 — 주요 그룹]")
main_keys = ["원본v10", "+A", "+B", "+D", "+E", "+ALL"]
years_all = sorted(set().union(*[set(results[k]["df"]["entry_date"].dt.year.unique())
                                  for k in main_keys if k in results]))
hdr2 = f"  {'연도':<6}" + "".join(f" {results[k]['label']:>16}" for k in main_keys if k in results)
print(hdr2)
print("  " + "-"*100)
for yr in years_all:
    row_str = f"  {yr:<6}"
    for k in main_keys:
        if k not in results:
            continue
        sub = results[k]["df"][results[k]["df"]["entry_date"].dt.year == yr]
        if len(sub) == 0:
            row_str += f" {'  -':>16}"
        else:
            wr  = sub["win"].mean() * 100
            avg = sub["return_pct"].mean()
            row_str += f" {wr:.0f}%/{avg:+.1f}%({len(sub)})"
    print(row_str)

# 청산 유형
print("\n[청산 유형 분포 — 주요 그룹]")
for k in main_keys:
    if k not in results:
        continue
    r  = results[k]
    ec = r["exit_dist"]
    tt = sum(ec.values())
    parts = [f"{v}건({v/tt*100:.0f}%)" for _, v in sorted(ec.items(), key=lambda x: -x[1])[:4]]
    tops  = [kk for kk in sorted(ec.items(), key=lambda x: -x[1])[:4]]
    parts = [f"{kk}: {v}건({v/tt*100:.0f}%)" for kk, v in tops]
    print(f"  {r['label']:<14}: {' | '.join(parts)}")


# ══════════════════════════════════════════════════
# 복합 점수
# ══════════════════════════════════════════════════
print("\n" + "="*80)
print("  복합 점수 (승률30% + CAGR30% + 기대값25% + PF15%)")
print("="*80)

score_data = []
for key, r in results.items():
    pf_val = r["pf"] if not np.isnan(r["pf"]) else 0
    score  = (
        r["win_rate"]  * 0.30 +
        r["cagr"]      * 0.30 +
        r["ev"]        * 0.25 +
        min(pf_val, 5) * 0.15 * 10
    )
    score_data.append((key, score, r))
score_data.sort(key=lambda x: -x[1])

print(f"\n  {'순위':<4} {'그룹':<14} {'점수':>8} {'승률':>7} {'CAGR':>8} {'기대값':>8} {'PF':>6} {'거래수':>6} {'추세진입%':>9}")
print("  " + "-"*80)
for rank, (key, score, r) in enumerate(score_data, 1):
    pf_str = f"{r['pf']:.2f}" if not np.isnan(r['pf']) else " N/A"
    marker = " ◀기준" if key == "원본v10" else ""
    print(f"  {rank:<4} {r['label']:<14} {score:>8.2f} {r['win_rate']:>6.1f}% "
          f"{r['cagr']:>+7.1f}% {r['ev']:>+7.2f}% {pf_str:>6} {r['n']:>5}건 "
          f"{r['trend_ratio']:>8.1f}%{marker}")

best_key, best_score, best_r = score_data[0]
print(f"\n  ★ 복합 점수 1위: {best_r['label']}")
print(f"    승률: {best_r['win_rate']:.1f}% | 평균수익: {best_r['avg_ret']:+.2f}% | CAGR: {best_r['cagr']:+.1f}%")
print(f"    기대값: {best_r['ev']:+.2f}% | PF: {best_r['pf']:.2f}")
print(f"    추세 진입 비율: {best_r['trend_ratio']:.1f}% (MA200 위에서 진입한 거래 비율)")


# ══════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════
main_keys_plot = [k for k in ["원본v10","+A","+B","+C","+D","+E","+F","+ALL"] if k in results]
labels_p  = [results[k]["label"] for k in main_keys_plot]
wr_p      = [results[k]["win_rate"]      for k in main_keys_plot]
ret_p     = [results[k]["avg_ret"]       for k in main_keys_plot]
cagr_p    = [results[k]["cagr"]          for k in main_keys_plot]
ev_p      = [results[k]["ev"]            for k in main_keys_plot]
pf_p      = [results[k]["pf"] if not np.isnan(results[k]["pf"]) else 0 for k in main_keys_plot]
hold_p    = [results[k]["avg_hold_days"] for k in main_keys_plot]
n_p       = [results[k]["n"]             for k in main_keys_plot]
tgt_p     = [results[k]["target_hit_pct"]for k in main_keys_plot]
cb_p      = [results[k]["circuit_hit_pct"]for k in main_keys_plot]
trend_p   = [results[k]["trend_ratio"]   for k in main_keys_plot]
consec_p  = [results[k]["max_consec_loss"]for k in main_keys_plot]

# 색상: 원본=빨강, OR추가=파랑, ALL=초록
colors_p = []
for k in main_keys_plot:
    if k == "원본v10":   colors_p.append("#e74c3c")
    elif k == "+ALL":    colors_p.append("#2ecc71")
    else:                colors_p.append("#3498db")

fig, axes = plt.subplots(3, 3, figsize=(22, 17))
fig.suptitle(
    "추세 편승 조건 추가 백테스트 비교 (2010-2026)\n"
    "빨간색=원본v10(기준) | 파란색=OR 추가 각각 | 초록색=전체 OR\n"
    "TARGET: +20% | CB: -25% | VIX조건: 원본만 적용 (추세 조건은 VIX 무관)",
    fontsize=11, fontweight="bold"
)

def bar_label(ax, bars, vals, fmt="{:.1f}", offset_ratio=0.01):
    ylim = ax.get_ylim()
    off  = (ylim[1] - ylim[0]) * offset_ratio
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + off,
                fmt.format(v), ha="center", fontsize=7.5)

# 1. 승률
ax = axes[0, 0]
bars = ax.bar(labels_p, wr_p, color=colors_p, edgecolor="white", linewidth=1.2)
ax.set_title("승률 (%)", fontweight="bold")
ax.set_ylim(0, 105)
ax.axhline(70, color="gray", linestyle="--", lw=1, alpha=0.5)
bar_label(ax, bars, wr_p, "{:.1f}%")
ax.tick_params(axis='x', rotation=35)

# 2. 평균 수익률
ax = axes[0, 1]
bars = ax.bar(labels_p, ret_p, color=colors_p, edgecolor="white", linewidth=1.2)
ax.set_title("평균 수익률 (%)", fontweight="bold")
ax.axhline(0, color="black", lw=0.8)
bar_label(ax, bars, ret_p, "{:+.2f}%")
ax.tick_params(axis='x', rotation=35)

# 3. CAGR
ax = axes[0, 2]
bars = ax.bar(labels_p, cagr_p, color=colors_p, edgecolor="white", linewidth=1.2)
ax.set_title("CAGR (%)", fontweight="bold")
ax.axhline(0, color="black", lw=0.8)
bar_label(ax, bars, cagr_p, "{:+.1f}%")
ax.tick_params(axis='x', rotation=35)

# 4. 기대값
ax = axes[1, 0]
bars = ax.bar(labels_p, ev_p, color=colors_p, edgecolor="white", linewidth=1.2)
ax.set_title("기대값 EV (%)", fontweight="bold")
ax.axhline(0, color="black", lw=0.8)
bar_label(ax, bars, ev_p, "{:+.2f}%")
ax.tick_params(axis='x', rotation=35)

# 5. Profit Factor
ax = axes[1, 1]
bars = ax.bar(labels_p, pf_p, color=colors_p, edgecolor="white", linewidth=1.2)
ax.set_title("Profit Factor", fontweight="bold")
ax.axhline(1.5, color="gray", linestyle="--", lw=1, alpha=0.5)
bar_label(ax, bars, pf_p, "{:.2f}")
ax.tick_params(axis='x', rotation=35)

# 6. 평균 보유 기간
ax = axes[1, 2]
bars = ax.bar(labels_p, hold_p, color=colors_p, edgecolor="white", linewidth=1.2)
ax.set_title("평균 보유 기간 (일)", fontweight="bold")
bar_label(ax, bars, hold_p, "{:.0f}일")
ax.tick_params(axis='x', rotation=35)

# 7. 추세 진입 비율
ax = axes[2, 0]
bars = ax.bar(labels_p, trend_p, color=colors_p, edgecolor="white", linewidth=1.2)
ax.set_title("추세 진입 비율 (MA200 위) %", fontweight="bold")
bar_label(ax, bars, trend_p, "{:.1f}%")
ax.tick_params(axis='x', rotation=35)

# 8. 목표 도달 vs CB 손절
ax = axes[2, 1]
x = np.arange(len(labels_p))
w = 0.35
ax.bar(x - w/2, tgt_p, width=w, label="목표도달%", color="#2ecc71", edgecolor="white")
ax.bar(x + w/2, cb_p,  width=w, label="CB손절%",   color="#e74c3c", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(labels_p, rotation=35)
ax.set_title("목표 도달 vs CB 손절 비율", fontweight="bold")
ax.legend(fontsize=9)

# 9. 복합 점수 순위
ax = axes[2, 2]
s_labels_p = [results[k]["label"] for k, _, _ in score_data if k in main_keys_plot]
s_vals_p   = [s for k, s, _ in score_data if k in main_keys_plot]
s_cols_p   = ["#e74c3c" if k=="원본v10" else ("#2ecc71" if k=="+ALL" else "#3498db")
              for k, _, _ in score_data if k in main_keys_plot]
bars = ax.barh(s_labels_p, s_vals_p, color=s_cols_p, edgecolor="white", linewidth=1.2)
ax.set_title("복합 점수 순위\n(승률30%+CAGR30%+EV25%+PF15%)", fontweight="bold")
for bar, v in zip(bars, s_vals_p):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f"{v:.2f}", va="center", fontsize=8.5)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("backtest_trend_conditions.png", dpi=150, bbox_inches="tight")
print("\n📊 backtest_trend_conditions.png 저장 완료")

# CSV 저장 (주요 그룹만)
for k in main_keys_plot:
    r = results[k]
    fname = f"backtest_trend_{k.replace('+','plus').replace(' ','_')}_trades.csv"
    r["df"].to_csv(fname, index=False)
print("📄 CSV 저장 완료")

# ══════════════════════════════════════════════════
# 최종 권고
# ══════════════════════════════════════════════════
print("\n" + "="*80)
print("  ★ 최종 분석")
print("="*80)

base = results.get("원본v10")
print(f"\n  [기준: 원본 v10]")
print(f"    승률 {base['win_rate']:.1f}% | CAGR {base['cagr']:+.1f}% | PF {base['pf']:.2f} | 거래 {base['n']}건")

print(f"\n  [추세 조건 추가 효과 비교]")
print(f"  {'그룹':<14} {'승률차':>8} {'CAGR차':>8} {'PF차':>7} {'거래증가':>8} {'권고'}") 
print("  " + "-"*65)
for key in [k for k in ["+A","+B","+C","+D","+E","+F","+ALL"] if k in results]:
    r = results[key]
    dwr   = r["win_rate"] - base["win_rate"]
    dcagr = r["cagr"]     - base["cagr"]
    dpf   = r["pf"]       - base["pf"]
    dn    = r["n"]        - base["n"]
    # 판단
    if dcagr > 1 and dwr > -3 and dpf > -0.3:
        rec = "✅ 추천"
    elif dcagr > 0 and dwr > -5:
        rec = "△ 검토"
    else:
        rec = "❌ 비권고"
    print(f"  {r['label']:<14} {dwr:>+7.1f}%p {dcagr:>+7.1f}%p {dpf:>+6.2f} {dn:>+7}건  {rec}")

print("\n  ※ 2010-2026 과거 데이터 기반. 슬리피지·세금 미반영.")
print("="*80)
print("\n✅ 백테스트 완료")
