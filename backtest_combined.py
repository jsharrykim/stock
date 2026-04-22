"""
backtest_combined.py
====================
6개 전략 포트폴리오 통합 백테스트 (우선순위 적용 시뮬레이션)

[시뮬레이션 원칙]
  - 1종목 1포지션: 포지션 보유 중엔 동일 종목 신규 진입 없음
  - 우선순위 적용: 같은 날 여러 조건 충족 시 높은 우선순위 전략이 진입
  - 우선순위: D > C > H > I > A > B

[비교 시나리오]
  Scenario-4: 기존 4전략 (D > C > A > B)
  Scenario-6: 전체 6전략 (D > C > H > I > A > B)

[추가 분석]
  - 6전략에서 각 전략 1개씩 제거했을 때 영향 (leave-one-out)
  - 핵심 파라미터 민감도 (H: vol_ratio, I: ADX, D: VIX 임계값)

기간: 2015-01-01 ~ 2026-04-15
종목: 54개 (한국15 + 미국39)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── 기간 ──────────────────────────────────────────────────────────────────────
START = "2015-01-01"
END   = "2026-04-15"

# ── 출구 파라미터 ──────────────────────────────────────────────────────────────
EXIT_AB = dict(target=0.08,  stop=0.25, half_days=60, max_hold=120, label="+8%/-25%")
EXIT_C  = dict(target=0.15,  stop=0.25, half_days=60, max_hold=120, label="+15%/-25%")
EXIT_D  = dict(target=0.20,  stop=0.25, half_days=60, max_hold=120, label="+20%/-25%")
EXIT_H  = dict(target=0.15,  stop=0.25, half_days=60, max_hold=120, label="+15%/-25%")
EXIT_I  = dict(target=0.15,  stop=0.25, half_days=60, max_hold=120, label="+15%/-25%")

# ── 지표 파라미터 ──────────────────────────────────────────────────────────────
BB_PERIOD     = 20
BB_STD        = 2.0
BB_AVG        = 60
SQUEEZE_RATIO = 0.50
MACD_F, MACD_S, MACD_SIG = 12, 26, 9
ADX_PERIOD    = 14
LR_WINDOW     = 120

# ── 종목 ──────────────────────────────────────────────────────────────────────
KR_TICKERS = [
    "000660.KS","005930.KS","277810.KS","034020.KS","015760.KS",
    "005380.KS","012450.KS","042660.KS","042700.KQ","096770.KS",
    "009150.KS","000270.KS","247540.KQ","376900.KS","004020.KS",
    "329180.KS","375500.KS","086280.KS","000720.KS","353200.KQ",
    "011070.KS","079550.KS",
]
US_TICKERS = [
    "HOOD","AVGO","AMD","MSFT","GOOGL","NVDA","TSLA",
    "MU","LRCX","ON","SNDK","ASTS","AVAV","IONQ",
    "RKLB","PLTR","APP","SOXL","TSLL","TE","ONDS",
    "BE","PL","VRT","LITE","TER","ANET","IREN","HOOG",
    "SOLT","ETHU","NBIS","LPTH","CONL","GLW","FLNC",
    "VST","ASX","CRCL","SGML","AEHR","MP","PLAB","SKYT",
    "SMTC","COHR","MPWR","CIEN","KLAC","FORM","CRDO",
]
ALL_TICKERS = KR_TICKERS + US_TICKERS


# ── 지표 계산 ──────────────────────────────────────────────────────────────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    df["ma200"] = c.rolling(200).mean()
    df["ma20"]  = c.rolling(BB_PERIOD).mean()

    std20        = c.rolling(BB_PERIOD).std()
    bb_upper     = df["ma20"] + BB_STD * std20
    bb_lower     = df["ma20"] - BB_STD * std20
    bb_range     = bb_upper - bb_lower
    df["bb_width"]      = (bb_range / df["ma20"] * 100).where(df["ma20"] > 0)
    df["bb_width_avg"]  = df["bb_width"].rolling(BB_AVG).mean()
    df["bb_width_prev"] = df["bb_width"].shift(1)
    df["squeeze"]       = df["bb_width"] < df["bb_width_avg"] * SQUEEZE_RATIO
    df["prev_squeeze"]  = df["squeeze"].shift(1).fillna(False)
    df["pctb_close"]    = np.where(bb_range > 0, (c - bb_lower) / bb_range * 100, np.nan)
    df["pctb_low"]      = np.where(bb_range > 0, (l - bb_lower) / bb_range * 100, np.nan)

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"]      = 100 - 100 / (1 + np.where(loss == 0, 100.0, gain / loss))
    df["rsi_prev"] = df["rsi"].shift(1)
    df["rsi_rising"] = df["rsi"] > df["rsi_prev"]

    tp    = (h + l + c) / 3
    tp_ma = tp.rolling(14).mean()
    tp_md = tp.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = np.where(tp_md > 0, (tp - tp_ma) / (0.015 * tp_md), 0)

    ema_f      = c.ewm(span=MACD_F, adjust=False).mean()
    ema_s      = c.ewm(span=MACD_S, adjust=False).mean()
    macd_line  = ema_f - ema_s
    sig_line   = macd_line.ewm(span=MACD_SIG, adjust=False).mean()
    df["macd_hist"]    = macd_line - sig_line
    df["macd_prev"]    = df["macd_hist"].shift(1)
    df["golden_cross"] = (df["macd_prev"] <= 0) & (df["macd_hist"] > 0)

    p = ADX_PERIOD
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    plus_dm  = np.where((h.diff() > 0) & (h.diff() > (-l.diff())), h.diff(), 0.0)
    minus_dm = np.where((-l.diff() > 0) & (-l.diff() > h.diff()), -l.diff(), 0.0)

    def wilder(s, n):
        s   = pd.Series(s, index=df.index, dtype=float)
        out = s.copy()
        out.iloc[n-1] = s.iloc[:n].mean()
        for i in range(n, len(s)):
            out.iloc[i] = (out.iloc[i-1] * (n - 1) + s.iloc[i]) / n
        return out

    atr14      = wilder(tr.values, p)
    plus_di14  = 100 * wilder(plus_dm,  p) / atr14.replace(0, np.nan)
    minus_di14 = 100 * wilder(minus_dm, p) / atr14.replace(0, np.nan)
    df["plus_di"]    = plus_di14.values
    df["minus_di"]   = minus_di14.values
    di_sum  = (plus_di14 + minus_di14).replace(0, np.nan)
    di_diff = (plus_di14 - minus_di14).abs()
    dx      = 100 * di_diff / di_sum
    df["adx"]       = wilder(dx.fillna(0).values, p).values
    df["adx_prev"]  = df["adx"].shift(1)
    df["adx_rising"]= df["adx"] > df["adx_prev"]

    vol_avg20       = v.rolling(20).mean()
    df["vol_ratio"] = (v / vol_avg20).where(vol_avg20 > 0)

    low_arr = l.values.astype(float)
    lr_val  = np.full(len(df), np.nan)
    lr_slp  = np.full(len(df), np.nan)
    win     = LR_WINDOW
    x       = np.arange(win, dtype=float)
    for i in range(win - 1, len(df)):
        y_seg = low_arr[i - win + 1 : i + 1]
        if np.isnan(y_seg).any():
            continue
        slope, intercept = np.polyfit(x, y_seg, 1)
        lr_val[i] = slope * (win - 1) + intercept
        lr_slp[i] = slope
    df["lr_trendline"] = lr_val
    df["lr_slope"]     = lr_slp

    return df


# ── VIX 다운로드 ───────────────────────────────────────────────────────────────
def download_vix() -> pd.Series:
    print("VIX 데이터 다운로드 중...")
    try:
        vix_df = yf.download("^VIX", start=START, end=END,
                             auto_adjust=True, progress=False)
        vix = vix_df["Close"].squeeze()
        vix.index = pd.to_datetime(vix.index).tz_localize(None)
        print(f"  VIX: {len(vix)}일")
        return vix
    except Exception as e:
        print(f"  VIX 오류: {e}")
        return pd.Series(dtype=float)


# ── 데이터 다운로드 ────────────────────────────────────────────────────────────
def download_data(vix_series: pd.Series) -> dict:
    print(f"데이터 다운로드 중... ({len(ALL_TICKERS)}개)")
    raw = yf.download(ALL_TICKERS, start=START, end=END,
                      auto_adjust=True, progress=False, group_by="ticker")
    result = {}
    for t in ALL_TICKERS:
        try:
            df = raw[t].copy() if len(ALL_TICKERS) > 1 else raw.copy()
            df.dropna(how="all", inplace=True)
            if len(df) < 250:
                print(f"  [{t}] 데이터 부족 — 제외")
                continue
            df = calc_indicators(df)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df["vix"] = vix_series.reindex(df.index)
            result[t] = df
        except Exception as e:
            print(f"  [{t}] 오류: {e}")
    print(f"총 {len(result)}개 종목 로드 완료\n")
    return result


# ── 출구 체크 ──────────────────────────────────────────────────────────────────
def check_exit(close: float, entry: float, hold: int, ep: dict):
    p = (close - entry) / entry
    if p >= ep["target"]:              return "목표달성", p
    if p <= -ep["stop"]:               return "손절",    p
    if hold >= ep["max_hold"]:         return "만료",    p
    if hold >= ep["half_days"] and p > 0: return "중간수익", p
    return None, None


# ── 전략 조건 함수 ─────────────────────────────────────────────────────────────
def cond_D(s, vix_thr=25):
    return (not s["above200"]
            and s["vix"] >= vix_thr
            and (s["rsi"] < 40 or s["cci"] < -100)
            and s["lr_slope"] > 0
            and s["lr_trendline"] > 0
            and s["low"] <= s["lr_trendline"] * 1.03)

def cond_C(s):
    return (s["above200"] and s["golden_cross"]
            and s["pctb_close"] > 80 and s["rsi"] > 70)

def cond_H(s, vol_thr=1.5):
    return (s["above200"]
            and s["prev_squeeze"]
            and s["bb_width"] > s["bb_width_prev"] * 1.05
            and s["vol_ratio"] >= vol_thr
            and s["pctb_close"] > 55
            and s["macd_hist"] > 0)

def cond_I(s, adx_thr=25):
    return (s["above200"]
            and s["plus_di"] > s["minus_di"]
            and s["adx"] > adx_thr
            and s["adx_rising"]
            and s["macd_hist"] > 0
            and 30 <= s["pctb_close"] <= 75)

def cond_A(s):
    return s["above200"] and s["squeeze"] and s["pctb_low"] <= 50

def cond_B(s):
    return s["above200"] and s["pctb_low"] <= 5


# ── sig 딕셔너리 생성 ──────────────────────────────────────────────────────────
def make_sig(row):
    def fv(col, default=np.nan):
        try:
            val = float(row[col])
            return val if not np.isnan(val) else default
        except:
            return default
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
    }


# ── 통합 포트폴리오 백테스트 ────────────────────────────────────────────────────
def run_portfolio(data: dict, strategies: list, label: str) -> list:
    """
    strategies: [{"name": str, "fn": callable, "exit": dict}, ...]
                우선순위 순서대로 정렬되어야 함 (앞 = 높은 우선순위)
    """
    print(f"\n[{label}] 포트폴리오 시뮬레이션 중...")
    all_trades = []

    for ticker, df in data.items():
        dfc = df.dropna(subset=["ma200"]).copy()
        if len(dfc) < 50:
            continue

        in_pos      = False
        entry_price = entry_date = entry_idx = entry_exit = entry_strat = None
        idx_list    = list(dfc.index)

        for ii, date in enumerate(idx_list):
            row = dfc.loc[date]

            # ── 출구 체크 ──────────────────────────────────────────────────────
            if in_pos:
                hold = ii - entry_idx
                reason, pnl = check_exit(float(row["Close"]), entry_price,
                                         hold, entry_exit)
                if reason:
                    all_trades.append({
                        "label":      label,
                        "strategy":   entry_strat,
                        "ticker":     ticker,
                        "entry_date": str(entry_date)[:10],
                        "exit_date":  str(date)[:10],
                        "pnl_pct":    round(pnl * 100, 2),
                        "hold_days":  hold,
                        "exit_reason":reason,
                    })
                    in_pos = False

            # ── 진입 체크 (포지션 없을 때만) ───────────────────────────────────
            if not in_pos:
                sig = make_sig(row)
                for strat in strategies:
                    try:
                        if strat["fn"](sig):
                            in_pos      = True
                            entry_price = float(row["Close"])
                            entry_date  = date
                            entry_idx   = ii
                            entry_exit  = strat["exit"]
                            entry_strat = strat["name"]
                            break
                    except Exception:
                        pass

        # 미청산 포지션
        if in_pos and entry_price is not None:
            last = dfc.iloc[-1]
            pnl  = (float(last["Close"]) - entry_price) / entry_price
            all_trades.append({
                "label":      label,
                "strategy":   entry_strat,
                "ticker":     ticker,
                "entry_date": str(entry_date)[:10],
                "exit_date":  str(dfc.index[-1])[:10],
                "pnl_pct":    round(pnl * 100, 2),
                "hold_days":  len(dfc) - 1 - entry_idx,
                "exit_reason":"미청산",
            })

    return all_trades


# ── 분석 함수 ──────────────────────────────────────────────────────────────────
def summarize(trades: list) -> dict:
    if not trades:
        return {"trades": 0, "win_rate": 0, "ev": 0, "avg_hold": 0,
                "stop_rate": 0, "target_rate": 0, "avg_win": 0, "avg_loss": 0}
    df   = pd.DataFrame(trades)
    wins = df[df["pnl_pct"] > 0]
    loss = df[df["pnl_pct"] <= 0]
    wr   = len(wins) / len(df)
    aw   = wins["pnl_pct"].mean() if len(wins) else 0
    al   = loss["pnl_pct"].mean() if len(loss) else 0
    bx   = df["exit_reason"].value_counts(normalize=True) * 100
    return {
        "trades":      len(df),
        "win_rate":    round(wr * 100, 1),
        "ev":          round(wr * aw + (1 - wr) * al, 2),
        "avg_hold":    round(df["hold_days"].mean(), 1),
        "stop_rate":   round(bx.get("손절", 0), 1),
        "target_rate": round(bx.get("목표달성", 0), 1),
        "avg_win":     round(aw, 2),
        "avg_loss":    round(al, 2),
    }


def per_strategy_breakdown(trades: list) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df  = pd.DataFrame(trades)
    rows = []
    for strat, grp in df.groupby("strategy"):
        wins = grp[grp["pnl_pct"] > 0]
        loss = grp[grp["pnl_pct"] <= 0]
        wr   = len(wins) / len(grp)
        aw   = wins["pnl_pct"].mean() if len(wins) else 0
        al   = loss["pnl_pct"].mean() if len(loss) else 0
        ev   = round(wr * aw + (1 - wr) * al, 2)
        bx   = grp["exit_reason"].value_counts(normalize=True) * 100
        rows.append({
            "전략":    strat,
            "거래수":  len(grp),
            "비중(%)": 0,
            "승률(%)": round(wr * 100, 1),
            "EV(%)":  ev,
            "평균보유":round(grp["hold_days"].mean(), 1),
            "손절(%)": round(bx.get("손절", 0), 1),
            "목표(%)": round(bx.get("목표달성", 0), 1),
        })
    result = pd.DataFrame(rows).sort_values("EV(%)", ascending=False)
    total  = result["거래수"].sum()
    result["비중(%)"] = (result["거래수"] / total * 100).round(1)
    return result


def print_summary(label: str, trades: list):
    s  = summarize(trades)
    bd = per_strategy_breakdown(trades)
    n  = s["trades"]
    years = (pd.to_datetime(END) - pd.to_datetime(START)).days / 365.25
    annualized = round(n / years, 1)

    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"{'─'*70}")
    print(f"  총 거래수: {n}건  (연평균 {annualized}건/년)")
    print(f"  승   률:  {s['win_rate']}%")
    print(f"  EV:       {s['ev']}%")
    print(f"  평균수익:  {s['avg_win']}%   평균손실: {s['avg_loss']}%")
    print(f"  목표달성:  {s['target_rate']}%   손절률: {s['stop_rate']}%")
    print(f"  평균보유:  {s['avg_hold']}일")
    if not bd.empty:
        print(f"\n  [전략별 기여]")
        print(f"  {'전략':<20} {'거래수':>5} {'비중':>6} {'승률':>6} {'EV':>6} {'손절':>6}")
        for _, r in bd.iterrows():
            print(f"  {r['전략']:<20} {int(r['거래수']):>5} {r['비중(%)']:>5.1f}% "
                  f"{r['승률(%)']:>5.1f}% {r['EV(%)']:>6.2f}% {r['손절(%)']:>5.1f}%")
    return s


def print_diff(s4: dict, s6: dict):
    d_trades = s6["trades"] - s4["trades"]
    d_ev     = round(s6["ev"] - s4["ev"], 2)
    d_wr     = round(s6["win_rate"] - s4["win_rate"], 1)
    sign_t   = "+" if d_trades >= 0 else ""
    sign_e   = "+" if d_ev >= 0 else ""
    sign_w   = "+" if d_wr >= 0 else ""
    print(f"\n  ▶ 4전략 대비 6전략 증감: "
          f"거래 {sign_t}{d_trades}건 | EV {sign_e}{d_ev}%p | 승률 {sign_w}{d_wr}%p")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("6개 전략 포트폴리오 통합 백테스트")
    print("=" * 70)

    vix = download_vix()
    data = download_data(vix)

    # ── 전략 정의 (우선순위 순서) ────────────────────────────────────────────────
    STRAT_D = {"name": "D그룹 (공황저점)",     "fn": cond_D, "exit": EXIT_D}
    STRAT_C = {"name": "C그룹 (과열돌파)",     "fn": cond_C, "exit": EXIT_C}
    STRAT_H = {"name": "H그룹 (스퀴즈돌파)",   "fn": cond_H, "exit": EXIT_H}
    STRAT_I = {"name": "I그룹 (ADX추세)",      "fn": cond_I, "exit": EXIT_I}
    STRAT_A = {"name": "A그룹 (스퀴즈저점)",   "fn": cond_A, "exit": EXIT_AB}
    STRAT_B = {"name": "B그룹 (극단저점)",     "fn": cond_B, "exit": EXIT_AB}

    PORTFOLIO_4 = [STRAT_D, STRAT_C, STRAT_A, STRAT_B]
    PORTFOLIO_6 = [STRAT_D, STRAT_C, STRAT_H, STRAT_I, STRAT_A, STRAT_B]

    # ── 메인 시뮬레이션 ──────────────────────────────────────────────────────────
    trades4 = run_portfolio(data, PORTFOLIO_4, "Scenario-4 (D>C>A>B)")
    trades6 = run_portfolio(data, PORTFOLIO_6, "Scenario-6 (D>C>H>I>A>B)")

    print("\n" + "=" * 70)
    print("▣ 메인 비교 결과")
    print("=" * 70)

    s4 = print_summary("Scenario-4: 기존 4전략 (D>C>A>B)", trades4)
    s6 = print_summary("Scenario-6: 신규 6전략 (D>C>H>I>A>B)", trades6)
    print_diff(s4, s6)

    # ── Leave-One-Out 분석 ────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("▣ Leave-One-Out: 6전략에서 각 전략 하나씩 제거")
    print("=" * 70)
    print(f"  (기준 EV: {s6['ev']}%  기준 거래수: {s6['trades']}건)")
    print()

    LOO_CASES = [
        ("D 제거 → C>H>I>A>B", [STRAT_C, STRAT_H, STRAT_I, STRAT_A, STRAT_B]),
        ("C 제거 → D>H>I>A>B", [STRAT_D, STRAT_H, STRAT_I, STRAT_A, STRAT_B]),
        ("H 제거 → D>C>I>A>B", [STRAT_D, STRAT_C, STRAT_I, STRAT_A, STRAT_B]),
        ("I 제거 → D>C>H>A>B", [STRAT_D, STRAT_C, STRAT_H, STRAT_A, STRAT_B]),
        ("A 제거 → D>C>H>I>B", [STRAT_D, STRAT_C, STRAT_H, STRAT_I, STRAT_B]),
        ("B 제거 → D>C>H>I>A", [STRAT_D, STRAT_C, STRAT_H, STRAT_I, STRAT_A]),
    ]

    loo_rows = []
    for name, strats in LOO_CASES:
        t = run_portfolio(data, strats, name)
        s = summarize(t)
        d_ev = round(s["ev"] - s6["ev"], 2)
        d_n  = s["trades"] - s6["trades"]
        sign_e = "+" if d_ev >= 0 else ""
        sign_n = "+" if d_n >= 0 else ""
        verdict = ""
        if d_ev >= 0.5:
            verdict = "← 제거 시 오히려 EV 상승! 검토 필요"
        elif d_ev <= -0.3 and d_n <= -30:
            verdict = "← 이 전략이 핵심 기여"
        loo_rows.append({
            "제거 전략": name,
            "거래수": s["trades"],
            "거래 변화": f"{sign_n}{d_n}",
            "EV": s["ev"],
            "EV 변화": f"{sign_e}{d_ev}%p",
            "판정": verdict,
        })
        print(f"  {name:<25} 거래:{s['trades']:>4}건 ({sign_n}{d_n:+d})  "
              f"EV:{s['ev']:>5.2f}% ({sign_e}{d_ev}%p)  {verdict}")

    # ── 파라미터 민감도 분석 ──────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("▣ 핵심 파라미터 민감도 분석")
    print("=" * 70)

    # H: 거래량 임계값
    print("\n  [H그룹] 거래량 임계값 (vol_ratio) 변화")
    print(f"  {'vol_thr':<10} {'거래수':>6} {'승률':>7} {'EV':>7}")
    for vol_thr in [1.2, 1.5, 2.0, 2.5]:
        strat_h_tmp = {"name": f"H(vol≥{vol_thr})", "fn": lambda s, v=vol_thr: cond_H(s, v), "exit": EXIT_H}
        portfolio_tmp = [STRAT_D, STRAT_C, strat_h_tmp, STRAT_I, STRAT_A, STRAT_B]
        t = run_portfolio(data, portfolio_tmp, f"H_vol{vol_thr}")
        bd = per_strategy_breakdown(t)
        h_row = bd[bd["전략"].str.startswith("H")]
        if not h_row.empty:
            r = h_row.iloc[0]
            marker = " ◀ 현재" if vol_thr == 1.5 else ""
            print(f"  vol≥{vol_thr:<6} {int(r['거래수']):>6}건  {r['승률(%)']:>6.1f}%  {r['EV(%)']:>6.2f}%{marker}")

    # I: ADX 임계값
    print(f"\n  [I그룹] ADX 임계값 변화")
    print(f"  {'adx_thr':<10} {'거래수':>6} {'승률':>7} {'EV':>7}")
    for adx_thr in [20, 25, 30]:
        strat_i_tmp = {"name": f"I(ADX≥{adx_thr})", "fn": lambda s, a=adx_thr: cond_I(s, a), "exit": EXIT_I}
        portfolio_tmp = [STRAT_D, STRAT_C, STRAT_H, strat_i_tmp, STRAT_A, STRAT_B]
        t = run_portfolio(data, portfolio_tmp, f"I_adx{adx_thr}")
        bd = per_strategy_breakdown(t)
        i_row = bd[bd["전략"].str.startswith("I")]
        if not i_row.empty:
            r = i_row.iloc[0]
            marker = " ◀ 현재" if adx_thr == 25 else ""
            print(f"  ADX≥{adx_thr:<6} {int(r['거래수']):>6}건  {r['승률(%)']:>6.1f}%  {r['EV(%)']:>6.2f}%{marker}")

    # D: VIX 임계값
    print(f"\n  [D그룹] VIX 임계값 변화")
    print(f"  {'vix_thr':<10} {'거래수':>6} {'승률':>7} {'EV':>7}")
    for vix_thr in [20, 25, 30]:
        strat_d_tmp = {"name": f"D(VIX≥{vix_thr})", "fn": lambda s, v=vix_thr: cond_D(s, v), "exit": EXIT_D}
        portfolio_tmp = [strat_d_tmp, STRAT_C, STRAT_H, STRAT_I, STRAT_A, STRAT_B]
        t = run_portfolio(data, portfolio_tmp, f"D_vix{vix_thr}")
        bd = per_strategy_breakdown(t)
        d_row = bd[bd["전략"].str.startswith("D")]
        if not d_row.empty:
            r = d_row.iloc[0]
            marker = " ◀ 현재" if vix_thr == 25 else ""
            print(f"  VIX≥{vix_thr:<6} {int(r['거래수']):>6}건  {r['승률(%)']:>6.1f}%  {r['EV(%)']:>6.2f}%{marker}")

    # ── CSV 저장 ────────────────────────────────────────────────────────────────
    all_trades = trades4 + trades6
    pd.DataFrame(all_trades).to_csv(
        "backtest_combined_trades.csv", index=False, encoding="utf-8-sig")
    print(f"\n\n저장 완료: backtest_combined_trades.csv")

    # ── B그룹 상세 분석 ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("▣ B그룹 (저가%B≤5) 상세 분석 — 제거 vs 유지 판단용")
    print("=" * 70)

    # B그룹 단독 백테스트
    b_trades = run_portfolio(data, [STRAT_B], "B그룹 단독")
    if b_trades:
        b_df = pd.DataFrame(b_trades)

        # 연도별 EV
        b_df["year"] = pd.to_datetime(b_df["entry_date"]).dt.year
        print("\n  [B그룹 연도별 성과]")
        print(f"  {'연도':<6} {'거래수':>5} {'승률':>7} {'EV':>7} {'평균손익':>8}")
        for yr, grp in b_df.groupby("year"):
            wins = grp[grp["pnl_pct"] > 0]
            loss = grp[grp["pnl_pct"] <= 0]
            wr   = len(wins) / len(grp) if len(grp) else 0
            aw   = wins["pnl_pct"].mean() if len(wins) else 0
            al   = loss["pnl_pct"].mean() if len(loss) else 0
            ev   = round(wr * aw + (1 - wr) * al, 2)
            print(f"  {yr:<6} {len(grp):>5}건  {wr*100:>6.1f}%  {ev:>7.2f}%  "
                  f"{grp['pnl_pct'].mean():>8.2f}%")

        # 종목별 EV (상위/하위)
        ticker_stats = []
        for t, grp in b_df.groupby("ticker"):
            if len(grp) < 2:
                continue
            wins = grp[grp["pnl_pct"] > 0]
            loss = grp[grp["pnl_pct"] <= 0]
            wr   = len(wins) / len(grp)
            aw   = wins["pnl_pct"].mean() if len(wins) else 0
            al   = loss["pnl_pct"].mean() if len(loss) else 0
            ev   = round(wr * aw + (1 - wr) * al, 2)
            ticker_stats.append({"ticker": t, "trades": len(grp), "ev": ev, "win_rate": round(wr*100,1)})

        ts_df = pd.DataFrame(ticker_stats).sort_values("ev", ascending=False)
        print(f"\n  [B그룹 종목별 EV — 상위 10개]")
        print(f"  {'종목':<12} {'거래수':>5} {'승률':>7} {'EV':>7}")
        for _, r in ts_df.head(10).iterrows():
            print(f"  {r['ticker']:<12} {int(r['trades']):>5}건  {r['win_rate']:>6.1f}%  {r['ev']:>6.2f}%")

        print(f"\n  [B그룹 종목별 EV — 하위 10개 (EV 낮은 종목)]")
        print(f"  {'종목':<12} {'거래수':>5} {'승률':>7} {'EV':>7}")
        for _, r in ts_df.tail(10).iterrows():
            print(f"  {r['ticker']:<12} {int(r['trades']):>5}건  {r['win_rate']:>6.1f}%  {r['ev']:>6.2f}%")

        # 5전략 (B 제거) vs 6전략 비교
        trades5 = run_portfolio(data, [STRAT_D, STRAT_C, STRAT_H, STRAT_I, STRAT_A], "D>C>H>I>A (B제거)")
        s5 = summarize(trades5)
        print(f"\n  [B 제거 효과 재확인 — 73개 종목 기준]")
        print(f"  5전략(B제거): 거래 {s5['trades']}건  승률 {s5['win_rate']}%  EV {s5['ev']}%")
        print(f"  6전략(B포함): 거래 {s6['trades']}건  승률 {s6['win_rate']}%  EV {s6['ev']}%")
        d_ev = round(s5['ev'] - s6['ev'], 2)
        sign = "+" if d_ev >= 0 else ""
        print(f"  EV 변화: {sign}{d_ev}%p  {'→ B 제거 시 EV 상승' if d_ev > 0 else '→ B 포함이 더 나음'}")
        print(f"\n  [판단 기준]")
        print(f"  B그룹은 단순히 EV가 낮은 게 아니라 '양의 EV'를 유지하는 전략입니다.")
        print(f"  유지 권고 조건: EV > 0 AND 연도별 일관성이 있을 것")
        b_summary = summarize(b_trades)
        if b_summary['ev'] > 2.0:
            print(f"  → B그룹 EV {b_summary['ev']}% > 2% : 유지 권고 (양의 기대값 전략)")
        else:
            print(f"  → B그룹 EV {b_summary['ev']}% ≤ 2% : 제거 검토")

    # ── 최종 요약 권고 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("▣ 최종 권고 요약")
    print("=" * 70)
    ev_diff = round(s6["ev"] - s4["ev"], 2)
    n_diff  = s6["trades"] - s4["trades"]
    print(f"""
  [전략 수 변경 영향]
    거래 수: {s4['trades']}건 → {s6['trades']}건 ({'+' if n_diff >= 0 else ''}{n_diff}건)
    EV:      {s4['ev']}%  → {s6['ev']}%  ({'+' if ev_diff >= 0 else ''}{ev_diff}%p)
    승  률:  {s4['win_rate']}%  → {s6['win_rate']}%

  [해석]
    - H/I 전략 추가가 EV를 낮추지 않으면서 거래 수를 늘리면 → 잔고 성장에 기여
    - EV가 낮아지면서 거래 수만 늘면     → H 또는 I 중 하나 제거 검토
    - Leave-One-Out 결과도 함께 참고 바람
    """)


if __name__ == "__main__":
    main()
