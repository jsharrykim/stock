"""
backtest_universal.py
=====================
범용 검증: Nasdaq 100 + Dow 30 종목에서 6전략이 통하는지 테스트

핵심 추가:
  1. 하락장 필터 테스트
     - 필터 없음 (현행)
     - SPY MA200 상방일 때만 B·E 진입 허용 (하락장 보호)
  2. B그룹 bull/bear 구간 분리 성과
  3. 종목 풀 범용성 검증

기간: 2015-01-01 ~ 2026-04-15
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

START = "2015-01-01"
END   = "2026-04-15"

# ── 시장 국면 정의 (SPY MA200 기준) ──────────────────────────────────────────
# Bull: SPY 종가 > SPY MA200
# Bear: SPY 종가 ≤ SPY MA200
BEAR_YEARS  = {2015, 2018, 2022}   # 실제 하락/조정장 연도 (참고용)

# ── 출구 파라미터 ─────────────────────────────────────────────────────────────
EXIT_AB = dict(target=0.08,  stop=0.25, half_days=60, max_hold=120, label="+8%/-25%")
EXIT_C  = dict(target=0.15,  stop=0.25, half_days=60, max_hold=120, label="+15%/-25%")
EXIT_D  = dict(target=0.20,  stop=0.25, half_days=60, max_hold=120, label="+20%/-25%")
EXIT_H  = dict(target=0.15,  stop=0.25, half_days=60, max_hold=120, label="+15%/-25%")
EXIT_I  = dict(target=0.15,  stop=0.25, half_days=60, max_hold=120, label="+15%/-25%")

# ── 지표 파라미터 ─────────────────────────────────────────────────────────────
BB_PERIOD = 20;  BB_STD = 2.0;  BB_AVG = 60;  SQUEEZE_RATIO = 0.50
MACD_F, MACD_S, MACD_SIG = 12, 26, 9
ADX_PERIOD = 14;  LR_WINDOW = 120

# ── 종목 목록 ─────────────────────────────────────────────────────────────────
DOW30 = [
    "AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW",
    "GS","HD","HON","IBM","JNJ","JPM","KO","MCD","MMM","MRK",
    "MSFT","NKE","PG","SHW","TRV","UNH","V","VZ","WMT","AMZN",
]

NASDAQ100 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","COST","NFLX",
    "AMD","TMUS","CSCO","ADBE","PEP","QCOM","INTU","AMGN","TXN","CMCSA",
    "HON","AMAT","BKNG","ISRG","GILD","ADI","VRTX","PANW","REGN","MU",
    "LRCX","KLAC","ASML","SNPS","CDNS","CTAS","PYPL","ORLY","FTNT","NXPI",
    "ROP","MRVL","DXCM","ROST","ABNB","WDAY","CRWD","IDXX","CPRT","PCAR",
    "TTD","FAST","CHTR","ODFL","KHC","FANG","ON","CTSH","MCHP","DLTR",
    "VRSK","TEAM","EA","ZS","ILMN","ALGN","DDOG","ANSS","PAYX","CSGP",
    "MNST","AZN","LULU","CEG","BKR","AEP","EXC","XEL","KDP","CDW",
    "ADSK","ADP","AXON","ARM","APP","SMCI","MRNA","WBD","DASH","GEHC",
    "CSX","BIIB","SGEN","RCL","MELI","FSLR","ENPH","LCID","RIVN","GRAB",
]

ALL_TICKERS = sorted(set(DOW30 + NASDAQ100))

def get_tickers_from_wikipedia():
    """Wikipedia에서 현재 Nasdaq100 + Dow30 구성 종목 시도"""
    tickers = set()
    try:
        ndx = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100", timeout=10)
        for t in ndx:
            for col in ["Ticker", "Symbol"]:
                if col in t.columns:
                    tickers.update(t[col].dropna().tolist())
                    break
        print(f"  Nasdaq100 Wikipedia: {len(tickers)}개")
    except Exception as e:
        print(f"  Wikipedia Nasdaq100 실패: {e}")
    try:
        dow = pd.read_html("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", timeout=10)
        for t in dow:
            for col in ["Ticker", "Symbol"]:
                if col in t.columns:
                    tickers.update(t[col].dropna().tolist())
                    break
        print(f"  Dow30 포함 후: {len(tickers)}개")
    except Exception as e:
        print(f"  Wikipedia Dow30 실패: {e}")
    return sorted(tickers) if len(tickers) > 50 else None


# ── 지표 계산 ─────────────────────────────────────────────────────────────────
def calc_indicators(df):
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["ma200"] = c.rolling(200).mean()
    df["ma20"]  = c.rolling(BB_PERIOD).mean()
    std20 = c.rolling(BB_PERIOD).std()
    bbu   = df["ma20"] + BB_STD * std20
    bbl   = df["ma20"] - BB_STD * std20
    bbr   = bbu - bbl
    df["bb_width"]      = (bbr / df["ma20"] * 100).where(df["ma20"] > 0)
    df["bb_width_avg"]  = df["bb_width"].rolling(BB_AVG).mean()
    df["bb_width_prev"] = df["bb_width"].shift(1)
    df["squeeze"]       = df["bb_width"] < df["bb_width_avg"] * SQUEEZE_RATIO
    df["prev_squeeze"]  = df["squeeze"].shift(1).fillna(False)
    df["pctb_close"]    = np.where(bbr > 0, (c - bbl) / bbr * 100, np.nan)
    df["pctb_low"]      = np.where(bbr > 0, (l - bbl) / bbr * 100, np.nan)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + np.where(loss == 0, 100.0, gain / loss))
    tp    = (h + l + c) / 3
    tp_ma = tp.rolling(14).mean()
    tp_md = tp.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = np.where(tp_md > 0, (tp - tp_ma) / (0.015 * tp_md), 0)
    ema_f = c.ewm(span=MACD_F, adjust=False).mean()
    ema_s = c.ewm(span=MACD_S, adjust=False).mean()
    ml    = ema_f - ema_s
    sl    = ml.ewm(span=MACD_SIG, adjust=False).mean()
    df["macd_hist"]    = ml - sl
    df["macd_prev"]    = df["macd_hist"].shift(1)
    df["golden_cross"] = (df["macd_prev"] <= 0) & (df["macd_hist"] > 0)
    p  = ADX_PERIOD
    tr = pd.concat([(h-l), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    pdm = np.where((h.diff()>0)&(h.diff()>(-l.diff())), h.diff(), 0.0)
    mdm = np.where((-l.diff()>0)&(-l.diff()>h.diff()), -l.diff(), 0.0)
    def wilder(s, n):
        s   = pd.Series(s, index=df.index, dtype=float)
        out = s.copy(); out.iloc[n-1] = s.iloc[:n].mean()
        for i in range(n, len(s)): out.iloc[i] = (out.iloc[i-1]*(n-1)+s.iloc[i])/n
        return out
    atr14 = wilder(tr.values, p)
    pdi14 = 100 * wilder(pdm, p) / atr14.replace(0, np.nan)
    mdi14 = 100 * wilder(mdm, p) / atr14.replace(0, np.nan)
    df["plus_di"]   = pdi14.values; df["minus_di"] = mdi14.values
    dis = (pdi14+mdi14).replace(0, np.nan)
    dx  = 100 * (pdi14-mdi14).abs() / dis
    df["adx"]       = wilder(dx.fillna(0).values, p).values
    df["adx_prev"]  = df["adx"].shift(1)
    df["adx_rising"]= df["adx"] > df["adx_prev"]
    va20 = v.rolling(20).mean()
    df["vol_ratio"] = (v / va20).where(va20 > 0)
    la = l.values.astype(float)
    lv = np.full(len(df), np.nan); ls = np.full(len(df), np.nan)
    w  = LR_WINDOW; x = np.arange(w, dtype=float)
    for i in range(w-1, len(df)):
        ys = la[i-w+1:i+1]
        if np.isnan(ys).any(): continue
        sl2, ic = np.polyfit(x, ys, 1)
        lv[i] = sl2*(w-1)+ic; ls[i] = sl2
    df["lr_trendline"] = lv; df["lr_slope"] = ls
    return df


# ── SPY 시장 국면 데이터 ──────────────────────────────────────────────────────
def download_spy_market_phase():
    print("SPY 시장 국면 데이터 다운로드...")
    spy = yf.download("SPY", start=START, end=END, auto_adjust=True, progress=False)
    spy_c = spy["Close"].squeeze()
    spy_c.index = pd.to_datetime(spy_c.index).tz_localize(None)
    spy_ma200 = spy_c.rolling(200).mean()
    # True = bull (SPY > MA200), False = bear
    is_bull = (spy_c > spy_ma200).rename("spy_bull")
    print(f"  SPY 불장 비율: {is_bull.mean()*100:.1f}%")
    return is_bull


# ── VIX ───────────────────────────────────────────────────────────────────────
def download_vix():
    print("VIX 다운로드...")
    vix = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)
    vix_c = vix["Close"].squeeze()
    vix_c.index = pd.to_datetime(vix_c.index).tz_localize(None)
    return vix_c


# ── 데이터 다운로드 ───────────────────────────────────────────────────────────
def download_data(tickers, vix_series, spy_bull):
    print(f"데이터 다운로드 ({len(tickers)}개)...")
    raw = yf.download(tickers, start=START, end=END,
                      auto_adjust=True, progress=False, group_by="ticker")
    result = {}
    skipped = 0
    for t in tickers:
        try:
            df = raw[t].copy() if len(tickers) > 1 else raw.copy()
            df.dropna(how="all", inplace=True)
            if len(df) < 250:
                skipped += 1; continue
            df = calc_indicators(df)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df["vix"]      = vix_series.reindex(df.index)
            df["spy_bull"] = spy_bull.reindex(df.index).fillna(True)
            result[t] = df
        except Exception as e:
            skipped += 1
    print(f"  완료: {len(result)}개 사용 / {skipped}개 제외\n")
    return result


# ── 출구 체크 ─────────────────────────────────────────────────────────────────
def check_exit(close, entry, hold, ep):
    p = (close - entry) / entry
    if p >= ep["target"]:                 return "목표달성", p
    if p <= -ep["stop"]:                  return "손절",    p
    if hold >= ep["max_hold"]:            return "만료",    p
    if hold >= ep["half_days"] and p > 0: return "중간수익", p
    return None, None


# ── sig 생성 ──────────────────────────────────────────────────────────────────
def make_sig(row):
    def fv(col, d=np.nan):
        try: v=float(row[col]); return v if not np.isnan(v) else d
        except: return d
    return {
        "above200":     float(row["Close"]) > fv("ma200", 0),
        "squeeze":      bool(row.get("squeeze", False)),
        "prev_squeeze": bool(row.get("prev_squeeze", False)),
        "pctb_close":   fv("pctb_close", 50),
        "pctb_low":     fv("pctb_low", 999),
        "rsi":          fv("rsi", 50),
        "cci":          fv("cci", 0),
        "golden_cross": bool(row.get("golden_cross", False)),
        "macd_hist":    fv("macd_hist", 0),
        "plus_di":      fv("plus_di", 20),
        "minus_di":     fv("minus_di", 20),
        "adx":          fv("adx", 15),
        "adx_rising":   bool(row.get("adx_rising", False)),
        "vol_ratio":    fv("vol_ratio", 1.0),
        "bb_width":     fv("bb_width", 5),
        "bb_width_avg": fv("bb_width_avg", 5),
        "bb_width_prev":fv("bb_width_prev", 5),
        "lr_slope":     fv("lr_slope", 0),
        "lr_trendline": fv("lr_trendline", 0),
        "low":          float(row["Low"]),
        "close":        float(row["Close"]),
        "ma200":        fv("ma200", 1),
        "vix":          fv("vix", 15),
        "spy_bull":     bool(row.get("spy_bull", True)),
    }


# ── 전략 조건 함수 ────────────────────────────────────────────────────────────
def cond_D(s):
    return (not s["above200"] and s["vix"] >= 25
            and (s["rsi"] < 40 or s["cci"] < -100)
            and s["lr_slope"] > 0 and s["lr_trendline"] > 0
            and s["low"] <= s["lr_trendline"] * 1.03)

def cond_C(s):
    return (s["above200"] and s["golden_cross"]
            and s["pctb_close"] > 80 and s["rsi"] > 70)

def cond_H(s):
    return (s["above200"] and s["prev_squeeze"]
            and s["bb_width"] > s["bb_width_prev"] * 1.05
            and s["vol_ratio"] >= 1.5 and s["pctb_close"] > 55
            and s["macd_hist"] > 0)

def cond_I(s):
    return (s["above200"] and s["plus_di"] > s["minus_di"]
            and s["adx"] > 25 and s["adx_rising"]
            and s["macd_hist"] > 0 and 30 <= s["pctb_close"] <= 75)

def cond_A(s):
    return s["above200"] and s["squeeze"] and s["pctb_low"] <= 50

def cond_B(s):
    return s["above200"] and s["pctb_low"] <= 5

# B + 하락장 필터 (SPY MA200 상방일 때만)
def cond_B_filtered(s):
    return cond_B(s) and s["spy_bull"]

# A + 하락장 필터
def cond_A_filtered(s):
    return cond_A(s) and s["spy_bull"]


# ── 포트폴리오 백테스트 ───────────────────────────────────────────────────────
def run_portfolio(data, strategies, label):
    all_trades = []
    for ticker, df in data.items():
        dfc = df.dropna(subset=["ma200"]).copy()
        if len(dfc) < 50: continue
        in_pos = False
        entry_price = entry_date = entry_idx = entry_exit = entry_strat = None
        idx_list = list(dfc.index)
        for ii, date in enumerate(idx_list):
            row = dfc.loc[date]
            if in_pos:
                hold = ii - entry_idx
                reason, pnl = check_exit(float(row["Close"]), entry_price, hold, entry_exit)
                if reason:
                    all_trades.append({
                        "label": label, "strategy": entry_strat, "ticker": ticker,
                        "entry_date": str(entry_date)[:10], "exit_date": str(date)[:10],
                        "pnl_pct": round(pnl*100, 2), "hold_days": hold,
                        "exit_reason": reason,
                        "year": pd.to_datetime(entry_date).year,
                    })
                    in_pos = False
            if not in_pos:
                sig = make_sig(row)
                for strat in strategies:
                    try:
                        if strat["fn"](sig):
                            in_pos = True; entry_price = float(row["Close"])
                            entry_date = date; entry_idx = ii
                            entry_exit = strat["exit"]; entry_strat = strat["name"]
                            break
                    except: pass
        if in_pos and entry_price is not None:
            last = dfc.iloc[-1]
            pnl  = (float(last["Close"]) - entry_price) / entry_price
            all_trades.append({
                "label": label, "strategy": entry_strat, "ticker": ticker,
                "entry_date": str(entry_date)[:10], "exit_date": str(dfc.index[-1])[:10],
                "pnl_pct": round(pnl*100, 2), "hold_days": len(dfc)-1-entry_idx,
                "exit_reason": "미청산", "year": pd.to_datetime(entry_date).year,
            })
    return all_trades


# ── 분석 함수 ─────────────────────────────────────────────────────────────────
def summarize(trades):
    if not trades: return {"trades":0,"win_rate":0,"ev":0,"avg_hold":0}
    df   = pd.DataFrame(trades)
    wins = df[df["pnl_pct"] > 0]; loss = df[df["pnl_pct"] <= 0]
    wr   = len(wins)/len(df)
    aw   = wins["pnl_pct"].mean() if len(wins) else 0
    al   = loss["pnl_pct"].mean() if len(loss) else 0
    bx   = df["exit_reason"].value_counts(normalize=True)*100
    return {
        "trades":      len(df),
        "win_rate":    round(wr*100, 1),
        "ev":          round(wr*aw+(1-wr)*al, 2),
        "avg_hold":    round(df["hold_days"].mean(), 1),
        "stop_rate":   round(bx.get("손절",0), 1),
        "target_rate": round(bx.get("목표달성",0), 1),
    }


def print_strategy_breakdown(trades, label):
    if not trades: return
    df = pd.DataFrame(trades)
    s  = summarize(trades)
    years = (pd.to_datetime(END) - pd.to_datetime(START)).days / 365.25
    print(f"\n{'─'*65}")
    print(f"  {label}")
    print(f"{'─'*65}")
    print(f"  총 거래: {s['trades']}건  연평균 {s['trades']/years:.0f}건/년  "
          f"승률 {s['win_rate']}%  EV {s['ev']}%  "
          f"목표달성 {s['target_rate']}%  손절 {s['stop_rate']}%")
    print(f"\n  [전략별]  {'전략':<22} {'거래수':>5} {'비중':>6} {'승률':>6} {'EV':>6}")
    total = len(df)
    for strat, grp in df.groupby("strategy"):
        wins = grp[grp["pnl_pct"]>0]; loss = grp[grp["pnl_pct"]<=0]
        wr = len(wins)/len(grp); aw = wins["pnl_pct"].mean() if len(wins) else 0
        al = loss["pnl_pct"].mean() if len(loss) else 0
        ev = round(wr*aw+(1-wr)*al, 2)
        pct= round(len(grp)/total*100, 1)
        print(f"  {'':2}{strat:<22} {len(grp):>5}건  {pct:>5.1f}%  {wr*100:>5.1f}%  {ev:>6.2f}%")
    return s


def print_yearly(trades, strategy_filter=None):
    if not trades: return
    df = pd.DataFrame(trades)
    if strategy_filter:
        df = df[df["strategy"].str.contains(strategy_filter)]
    if df.empty: return
    print(f"  [연도별 성과]  {'연도':<6} {'거래':>5} {'승률':>7} {'EV':>7}  시장 국면")
    for yr, grp in df.groupby("year"):
        wins = grp[grp["pnl_pct"]>0]; loss = grp[grp["pnl_pct"]<=0]
        wr = len(wins)/len(grp); aw = wins["pnl_pct"].mean() if len(wins) else 0
        al = loss["pnl_pct"].mean() if len(loss) else 0
        ev = round(wr*aw+(1-wr)*al, 2)
        phase = "🐻 하락장" if yr in BEAR_YEARS else "🐂 상승장"
        marker= " ⚠" if ev < 0 else ""
        print(f"  {yr:<6} {len(grp):>5}건  {wr*100:>6.1f}%  {ev:>6.2f}%  {phase}{marker}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("범용 검증 백테스트 (Nasdaq100 + Dow30)")
    print("=" * 65)

    # 종목 로드 (Wikipedia 시도 → 실패 시 하드코딩)
    wiki_tickers = get_tickers_from_wikipedia()
    tickers = wiki_tickers if wiki_tickers else ALL_TICKERS
    tickers = [t for t in tickers if "." not in str(t)]  # 미국 주식만
    print(f"테스트 종목: {len(tickers)}개\n")

    spy_bull = download_spy_market_phase()
    vix      = download_vix()
    data     = download_data(tickers, vix, spy_bull)

    # ── 전략 정의 ────────────────────────────────────────────────────────────
    STRAT_D  = {"name":"D그룹(공황저점)",    "fn":cond_D,          "exit":EXIT_D}
    STRAT_C  = {"name":"C그룹(과열돌파)",    "fn":cond_C,          "exit":EXIT_C}
    STRAT_H  = {"name":"H그룹(스퀴즈돌파)", "fn":cond_H,          "exit":EXIT_H}
    STRAT_I  = {"name":"I그룹(ADX추세)",    "fn":cond_I,          "exit":EXIT_I}
    STRAT_A  = {"name":"A그룹(스퀴즈저점)", "fn":cond_A,          "exit":EXIT_AB}
    STRAT_B  = {"name":"B그룹(극단저점)",   "fn":cond_B,          "exit":EXIT_AB}
    STRAT_Bf = {"name":"B그룹(극단저점+필터)","fn":cond_B_filtered,"exit":EXIT_AB}
    STRAT_Af = {"name":"A그룹(스퀴즈+필터)","fn":cond_A_filtered, "exit":EXIT_AB}

    P6       = [STRAT_D, STRAT_C, STRAT_H, STRAT_I, STRAT_A,  STRAT_B]
    P6_filt  = [STRAT_D, STRAT_C, STRAT_H, STRAT_I, STRAT_Af, STRAT_Bf]

    # ── 메인 시뮬레이션 ──────────────────────────────────────────────────────
    print("\n[1/3] 6전략 필터 없음 시뮬레이션...")
    t6     = run_portfolio(data, P6,      "6전략(필터없음)")
    print("\n[2/3] 6전략 하락장 필터(SPY MA200) 시뮬레이션...")
    t6f    = run_portfolio(data, P6_filt, "6전략(SPY필터)")

    # ── 결과 출력 ────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("▣ Nasdaq100+Dow30 기준 — 6전략 성과")
    print("=" * 65)

    s6  = print_strategy_breakdown(t6,  "필터 없음 (현행)")
    s6f = print_strategy_breakdown(t6f, "SPY MA200 필터 적용 (A·B 하락장 진입 차단)")

    # ── B그룹 연도별 비교 ────────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("▣ B그룹 연도별 비교 — 필터 전/후")
    print("=" * 65)

    print("\n  ─ 필터 없음 ─")
    print_yearly(t6, "B그룹")
    print("\n  ─ SPY MA200 필터 적용 후 ─")
    print_yearly(t6f, "B그룹")

    # ── A그룹 연도별 비교 ────────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("▣ A그룹 (스퀴즈저점) 연도별 비교 — 필터 전/후")
    print("=" * 65)

    print("\n  ─ 필터 없음 ─")
    print_yearly(t6, "A그룹")
    print("\n  ─ SPY MA200 필터 적용 후 ─")
    print_yearly(t6f, "A그룹")

    # ── 전체 비교 요약 ───────────────────────────────────────────────────────
    if s6 and s6f:
        print("\n\n" + "=" * 65)
        print("▣ 최종 비교 요약")
        print("=" * 65)
        d_ev = round(s6f["ev"] - s6["ev"], 2)
        d_n  = s6f["trades"] - s6["trades"]
        d_wr = round(s6f["win_rate"] - s6["win_rate"], 1)
        sign_e = "+" if d_ev >= 0 else ""
        sign_n = "+" if d_n >= 0 else ""
        sign_w = "+" if d_wr >= 0 else ""
        print(f"""
  필터 없음:    EV {s6['ev']}%   거래 {s6['trades']}건   승률 {s6['win_rate']}%
  SPY필터 적용: EV {s6f['ev']}%  거래 {s6f['trades']}건  승률 {s6f['win_rate']}%
  변화:         EV {sign_e}{d_ev}%p  거래 {sign_n}{d_n}건  승률 {sign_w}{d_wr}%p

  [해석]
  - EV 상승 + 거래수 소폭 감소 → 필터 적용 권고 (품질↑, 노이즈↓)
  - EV 변화 없음 + 거래수 큰 감소 → 필터 불필요 (A·B가 하락장에서도 견고)
  - EV 하락 → 필터가 좋은 신호도 차단 (필터 미적용 유지)
        """)
        print("  [Nasdaq100+Dow30 범용성 체크]")
        if s6["ev"] > 4.0 and s6["win_rate"] > 75:
            print(f"  ✅ 범용 통과: EV {s6['ev']}% > 4%, 승률 {s6['win_rate']}% > 75%")
        else:
            print(f"  ⚠️  범용성 검토 필요: EV {s6['ev']}%, 승률 {s6['win_rate']}%")

    # ── CSV ──────────────────────────────────────────────────────────────────
    pd.DataFrame(t6 + t6f).to_csv(
        "backtest_universal_trades.csv", index=False, encoding="utf-8-sig")
    print("\n저장: backtest_universal_trades.csv")


if __name__ == "__main__":
    main()
