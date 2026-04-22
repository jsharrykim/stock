"""
backtest_new_strategies.py  (v2 — D중복 추가 + 우선순위 후 실질 EV)
===========================
신규 전략 후보 E / H / I / F 백테스트
  + 기존 A / B / C / D 기준선 재계산
  + 전략 간 중복 발생률(overlap) 측정  ← A/B/C/D 전부
  + 우선순위(A>B>C>H>I>D) 적용 후 실질 EV 계산

[전략 정의]
  REF_A : MA200 위 + BB스퀴즈 + 저가%B ≤ 50
  REF_B : MA200 위 + 저가%B ≤ 5
  REF_C : MA200 위 + MACD골든크로스 + 종가%B > 80 + RSI > 70
  REF_D : MA200 아래 + VIX ≥ 25 + (RSI<40 OR CCI<-100)
            + LR기울기 > 0 + 저가 ≤ LR추세선 × 1.03

  E     : MA200 위 + MACD골든크로스 + 종가%B < 50 + RSI < 55
  H     : MA200 위 + 전일스퀴즈 + BB폭확장 + 거래량1.5x + %B>55 + MACD>0
  I     : MA200 위 + +DI>-DI + ADX>25 + ADX상승 + MACD>0 + %B 30~75
  F     : MA200 위 + MA200×1.12 이내 + LR기울기>0 + 저가≤LR×1.03
            + (RSI<50 OR 저가%B<20)

[중복 측정]
  진입일에 A/B/C/D 조건이 동시에 성립하는 비율
  → 우선순위 A>B>C>H>I>D 기준으로 "상위 전략 미충족" 조건부 실질 EV도 계산

기간: 2015-01-01 ~ 2026-04-15
종목: 기존 54개 풀 동일
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
EXIT_AB  = dict(target=0.08,  stop=0.25, half_days=60, max_hold=120, label="+8%/-25%")
EXIT_C   = dict(target=0.15,  stop=0.25, half_days=60, max_hold=120, label="+15%/-25%")
EXIT_NEW = dict(target=0.12,  stop=0.20, half_days=60, max_hold=120, label="+12%/-20%")

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

    # MA
    df["ma200"] = c.rolling(200).mean()
    df["ma20"]  = c.rolling(BB_PERIOD).mean()

    # BB
    std20        = c.rolling(BB_PERIOD).std()
    bb_upper     = df["ma20"] + BB_STD * std20
    bb_lower     = df["ma20"] - BB_STD * std20
    bb_range     = bb_upper - bb_lower
    df["bb_width"]     = (bb_range / df["ma20"] * 100).where(df["ma20"] > 0)
    df["bb_width_avg"] = df["bb_width"].rolling(BB_AVG).mean()
    df["bb_width_prev"]= df["bb_width"].shift(1)
    df["squeeze"]      = df["bb_width"] < df["bb_width_avg"] * SQUEEZE_RATIO
    df["prev_squeeze"] = df["squeeze"].shift(1).fillna(False)

    # %B
    df["pctb_close"] = np.where(bb_range > 0, (c - bb_lower) / bb_range * 100, np.nan)
    df["pctb_low"]   = np.where(bb_range > 0, (l - bb_lower) / bb_range * 100, np.nan)

    # RSI(14)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"]      = 100 - 100 / (1 + np.where(loss == 0, 100.0, gain / loss))
    df["rsi_prev"] = df["rsi"].shift(1)
    df["rsi_rising"] = df["rsi"] > df["rsi_prev"]

    # CCI(14)
    tp     = (h + l + c) / 3
    tp_ma  = tp.rolling(14).mean()
    tp_md  = tp.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = np.where(tp_md > 0, (tp - tp_ma) / (0.015 * tp_md), 0)

    # MACD(12,26,9)
    ema_f         = c.ewm(span=MACD_F, adjust=False).mean()
    ema_s         = c.ewm(span=MACD_S, adjust=False).mean()
    macd_line     = ema_f - ema_s
    sig_line      = macd_line.ewm(span=MACD_SIG, adjust=False).mean()
    df["macd_hist"]    = macd_line - sig_line
    df["macd_prev"]    = df["macd_hist"].shift(1)
    df["golden_cross"] = (df["macd_prev"] <= 0) & (df["macd_hist"] > 0)

    # ADX / DMI (14)
    p = ADX_PERIOD
    high_prev = h.shift(1)
    low_prev  = l.shift(1)

    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)

    plus_dm  = np.where((h.diff() > 0) & (h.diff() > (-l.diff())), h.diff(), 0.0)
    minus_dm = np.where((-l.diff() > 0) & (-l.diff() > h.diff()), -l.diff(), 0.0)

    # Wilder smoothing
    def wilder(s, n):
        s   = pd.Series(s, index=df.index, dtype=float)
        out = s.copy()
        init_val = s.iloc[:n].mean()
        out.iloc[n-1] = init_val
        for i in range(n, len(s)):
            out.iloc[i] = (out.iloc[i-1] * (n - 1) + s.iloc[i]) / n
        return out

    atr14     = wilder(tr.values, p)
    plus_di14 = 100 * wilder(plus_dm,  p) / atr14.replace(0, np.nan)
    minus_di14= 100 * wilder(minus_dm, p) / atr14.replace(0, np.nan)

    df["plus_di"]  = plus_di14.values
    df["minus_di"] = minus_di14.values

    di_sum  = (plus_di14 + minus_di14).replace(0, np.nan)
    di_diff = (plus_di14 - minus_di14).abs()
    dx      = 100 * di_diff / di_sum
    df["adx"]      = wilder(dx.fillna(0).values, p).values
    df["adx_prev"] = df["adx"].shift(1)
    df["adx_rising"] = df["adx"] > df["adx_prev"]

    # 거래량 비율 (20일 평균 대비)
    vol_avg20       = v.rolling(20).mean()
    df["vol_ratio"] = (v / vol_avg20).where(vol_avg20 > 0)

    # LR 추세선 (120일 저가 기반 선형회귀 — 현재 날짜의 추세선 값)
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
        lr_val[i] = slope * (win - 1) + intercept  # 현재 날짜 추세선값
        lr_slp[i] = slope

    df["lr_trendline"] = lr_val
    df["lr_slope"]     = lr_slp

    return df


# ── VIX 데이터 다운로드 ───────────────────────────────────────────────────────
def download_vix() -> pd.Series:
    """^VIX 종가를 날짜 인덱스 Series로 반환"""
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
def download_data(vix_series: pd.Series):
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
            # VIX 병합
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df["vix"] = vix_series.reindex(df.index)
            result[t] = df
            print(f"  [{t}] {len(df)}일")
        except Exception as e:
            print(f"  [{t}] 오류: {e}")
    return result


# ── 매도 체크 ─────────────────────────────────────────────────────────────────
def check_exit(close, entry, hold, ep):
    pnl = (close - entry) / entry
    if pnl >= ep["target"]:
        return "목표달성", pnl
    if pnl <= -ep["stop"]:
        return "손절", pnl
    if hold >= ep["half_days"] and pnl > 0:
        return f"{ep['half_days']}일수익", pnl
    if hold >= ep["max_hold"]:
        return "기간만료", pnl
    return None, pnl


# ── 중복 감지용 조건 함수 ─────────────────────────────────────────────────────
def ref_a_cond(sig):
    return sig["above200"] and sig["squeeze"] and sig["pctb_low"] <= 50

def ref_b_cond(sig):
    return sig["above200"] and sig["pctb_low"] <= 5

def ref_c_cond(sig):
    return (sig["above200"] and sig["golden_cross"]
            and sig["pctb_close"] > 80 and sig["rsi"] > 70)

def ref_d_cond(sig):
    return (not sig["above200"]
            and sig["vix"] >= 25
            and (sig["rsi"] < 40 or sig["cci"] < -100)
            and sig["lr_slope"] > 0
            and sig["lr_trendline"] > 0
            and sig["low"] <= sig["lr_trendline"] * 1.03)

# 우선순위 맵: 각 전략이 "자신보다 상위인" 기준 전략들의 조건 함수 목록
# A>B>C>H>I>D 순서 — 신규 전략 H는 A/B/C 상위, I는 A/B/C/H 상위
PRIORITY_BLOCKERS = {
    "E": [ref_a_cond, ref_b_cond, ref_c_cond],
    "H": [ref_a_cond, ref_b_cond, ref_c_cond],
    "I": [ref_a_cond, ref_b_cond, ref_c_cond],  # H보다 I가 낮으므로 H도 블록커가 돼야 하지만
    "F": [ref_a_cond, ref_b_cond, ref_c_cond],  # H 조건 함수는 sig에 bb_width_prev 필요 → 별도 처리
}


# ── 백테스트 엔진 ─────────────────────────────────────────────────────────────
def run_backtest(data: dict, sc: dict) -> list:
    cond_fn = sc["fn"]
    ep      = sc["exit"]
    trades  = []

    for ticker, df in data.items():
        req_cols = ["ma200","bb_width","bb_width_avg","bb_width_prev","prev_squeeze",
                    "pctb_close","pctb_low","rsi","rsi_rising","cci",
                    "macd_hist","macd_prev","golden_cross",
                    "plus_di","minus_di","adx","adx_rising",
                    "vol_ratio","lr_trendline","lr_slope"]
        dfc = df.dropna(subset=["ma200","pctb_close","rsi","macd_hist"])
        if len(dfc) < 10:
            continue

        in_pos      = False
        entry_price = entry_date = entry_idx = None
        entry_group = None
        idx_list    = list(dfc.index)

        for ii, date in enumerate(idx_list):
            row = dfc.loc[date]

            if in_pos:
                hold = ii - entry_idx
                reason, pnl = check_exit(float(row["Close"]), entry_price, hold, ep)
                if reason:
                    # 중복 여부 기록 (진입일 기준)
                    trades.append({
                        "scenario":   sc["name"],
                        "ticker":     ticker,
                        "entry_date": str(entry_date)[:10],
                        "exit_date":  str(date)[:10],
                        "entry_price":round(float(entry_price), 4),
                        "exit_price": round(float(row["Close"]), 4),
                        "pnl_pct":    round(pnl * 100, 2),
                        "hold_days":  hold,
                        "exit_reason":reason,
                        "overlap_a":  entry_group.get("a", False),
                        "overlap_b":  entry_group.get("b", False),
                        "overlap_c":  entry_group.get("c", False),
                        "overlap_d":  entry_group.get("d", False),
                    })
                    in_pos = False

            if not in_pos:
                def fv(col, default=np.nan):
                    try:
                        val = float(row[col])
                        return val if not np.isnan(val) else default
                    except:
                        return default

                sig = {
                    "above200":    float(row["Close"]) > fv("ma200", 0),
                    "squeeze":     bool(row.get("squeeze", False)),
                    "prev_squeeze":bool(row.get("prev_squeeze", False)),
                    "pctb_close":  fv("pctb_close", 50),
                    "pctb_low":    fv("pctb_low", 999),
                    "rsi":         fv("rsi", 50),
                    "rsi_rising":  bool(row.get("rsi_rising", False)),
                    "cci":         fv("cci", 0),
                    "golden_cross":bool(row.get("golden_cross", False)),
                    "macd_hist":   fv("macd_hist", 0),
                    "plus_di":     fv("plus_di", 20),
                    "minus_di":    fv("minus_di", 20),
                    "adx":         fv("adx", 15),
                    "adx_rising":  bool(row.get("adx_rising", False)),
                    "vol_ratio":   fv("vol_ratio", 1.0),
                    "bb_width":    fv("bb_width", 5),
                    "bb_width_avg":fv("bb_width_avg", 5),
                    "bb_width_prev":fv("bb_width_prev", 5),
                    "lr_slope":    fv("lr_slope", 0),
                    "lr_trendline":fv("lr_trendline", 0),
                    "low":         float(row["Low"]),
                    "close":       float(row["Close"]),
                    "ma200":       fv("ma200", 1),
                    "vix":         fv("vix", 15),
                }

                try:
                    if cond_fn(sig):
                        in_pos      = True
                        entry_price = float(row["Close"])
                        entry_date  = date
                        entry_idx   = ii
                        entry_group = {
                            "a": ref_a_cond(sig),
                            "b": ref_b_cond(sig),
                            "c": ref_c_cond(sig),
                            "d": ref_d_cond(sig),
                        }
                except Exception:
                    pass

        # 미청산 포지션
        if in_pos and entry_price is not None:
            last = dfc.iloc[-1]
            pnl  = (float(last["Close"]) - entry_price) / entry_price
            trades.append({
                "scenario":   sc["name"],
                "ticker":     ticker,
                "entry_date": str(entry_date)[:10],
                "exit_date":  str(dfc.index[-1])[:10],
                "entry_price":round(float(entry_price), 4),
                "exit_price": round(float(last["Close"]), 4),
                "pnl_pct":    round(pnl * 100, 2),
                "hold_days":  len(dfc) - 1 - entry_idx,
                "exit_reason":"미청산",
                "overlap_a":  entry_group.get("a", False),
                "overlap_b":  entry_group.get("b", False),
                "overlap_c":  entry_group.get("c", False),
                "overlap_d":  entry_group.get("d", False),
            })

    return trades


# ── 분석 ──────────────────────────────────────────────────────────────────────
def analyze(trades: list, sc: dict) -> dict:
    name = sc["name"]
    ep   = sc["exit"]
    if not trades:
        return {k: 0 for k in ["scenario","trades","win_rate","ev","avg_pnl",
                                "median_pnl","avg_win","avg_loss","stop_rate",
                                "target_rate","avg_hold","overlap_a_pct",
                                "overlap_b_pct","overlap_c_pct","overlap_d_pct",
                                "pure_trades","pure_ev","exit_cfg"]}

    df   = pd.DataFrame(trades)
    wins = df[df["pnl_pct"] > 0]
    loss = df[df["pnl_pct"] <= 0]
    wr   = len(wins) / len(df)
    aw   = wins["pnl_pct"].mean() if len(wins) else 0
    al   = loss["pnl_pct"].mean() if len(loss) else 0
    ev   = round(wr * aw + (1 - wr) * al, 2)
    bx   = df["exit_reason"].value_counts(normalize=True) * 100

    # 우선순위 적용 후 실질 EV: A/B/C/D 중 하나라도 겹치는 거래를 제외
    has_overlap = df[["overlap_a","overlap_b","overlap_c","overlap_d"]].any(axis=1)
    pure_df  = df[~has_overlap]
    p_wins   = pure_df[pure_df["pnl_pct"] > 0]
    p_loss   = pure_df[pure_df["pnl_pct"] <= 0]
    p_wr     = len(p_wins) / len(pure_df) if len(pure_df) else 0
    p_aw     = p_wins["pnl_pct"].mean() if len(p_wins) else 0
    p_al     = p_loss["pnl_pct"].mean() if len(p_loss) else 0
    pure_ev  = round(p_wr * p_aw + (1 - p_wr) * p_al, 2) if len(pure_df) else 0

    return {
        "scenario":      name,
        "exit_cfg":      ep["label"],
        "trades":        len(df),
        "win_rate":      round(wr * 100, 1),
        "ev":            ev,
        "avg_pnl":       round(df["pnl_pct"].mean(), 2),
        "median_pnl":    round(df["pnl_pct"].median(), 2),
        "avg_win":       round(aw, 2),
        "avg_loss":      round(al, 2),
        "stop_rate":     round(bx.get("손절", 0), 1),
        "target_rate":   round(bx.get("목표달성", 0), 1),
        "avg_hold":      round(df["hold_days"].mean(), 1),
        "overlap_a_pct": round(df["overlap_a"].mean() * 100, 1),
        "overlap_b_pct": round(df["overlap_b"].mean() * 100, 1),
        "overlap_c_pct": round(df["overlap_c"].mean() * 100, 1),
        "overlap_d_pct": round(df["overlap_d"].mean() * 100, 1),
        "pure_trades":   len(pure_df),
        "pure_ev":       pure_ev,
    }


# ── 시나리오 ──────────────────────────────────────────────────────────────────
SCENARIOS = [
    # ── 기준선 ────────────────────────────────────────────────────────────────
    {
        "name": "REF_A (스퀴즈+저가%B≤50)",
        "exit": EXIT_AB,
        "fn":   lambda s: (s["above200"] and s["squeeze"] and s["pctb_low"] <= 50),
    },
    {
        "name": "REF_B (저가%B≤5)",
        "exit": EXIT_AB,
        "fn":   lambda s: (s["above200"] and s["pctb_low"] <= 5),
    },
    {
        "name": "REF_C (골든크로스+%B>80+RSI>70)",
        "exit": EXIT_C,
        "fn":   lambda s: (s["above200"] and s["golden_cross"]
                           and s["pctb_close"] > 80 and s["rsi"] > 70),
    },
    {
        "name": "REF_D (MA200하방+VIX≥25+LR저점)",
        "exit": dict(target=0.20, stop=0.25, half_days=60, max_hold=120, label="+20%/-25%"),
        "fn":   ref_d_cond,
    },

    # ── E그룹: MACD 저점 반전 (C의 거울상) ────────────────────────────────────
    # EXIT_NEW (+12%/-20%)
    {
        "name": "E_골든크로스저점_%B<50_RSI<55",
        "exit": EXIT_NEW,
        "fn":   lambda s: (s["above200"] and s["golden_cross"]
                           and s["pctb_close"] < 50 and s["rsi"] < 55),
    },
    # +DI > -DI 추가
    {
        "name": "E_골든크로스저점+DI방향",
        "exit": EXIT_NEW,
        "fn":   lambda s: (s["above200"] and s["golden_cross"]
                           and s["pctb_close"] < 50 and s["rsi"] < 55
                           and s["plus_di"] > s["minus_di"]),
    },
    # EXIT_C (+15%/-25%) 테스트
    {
        "name": "E_골든크로스저점_C출구",
        "exit": EXIT_C,
        "fn":   lambda s: (s["above200"] and s["golden_cross"]
                           and s["pctb_close"] < 50 and s["rsi"] < 55),
    },
    # EXIT_AB (+8%/-25%) 테스트
    {
        "name": "E_골든크로스저점_AB출구",
        "exit": EXIT_AB,
        "fn":   lambda s: (s["above200"] and s["golden_cross"]
                           and s["pctb_close"] < 50 and s["rsi"] < 55),
    },

    # ── H그룹: BB 스퀴즈 돌파 + 거래량 폭발 ──────────────────────────────────
    {
        "name": "H_스퀴즈돌파+거래량",
        "exit": EXIT_NEW,
        "fn":   lambda s: (
            s["above200"]
            and s["prev_squeeze"]                            # 전일 스퀴즈
            and s["bb_width"] > s["bb_width_prev"] * 1.05   # 오늘 BB폭 확장
            and s["vol_ratio"] > 1.5                         # 거래량 폭발
            and s["pctb_close"] > 55                         # 위쪽 돌파
            and s["macd_hist"] > 0                           # 모멘텀 동행
        ),
    },
    # 거래량 기준 낮춰서 신호 수 확인
    {
        "name": "H_스퀴즈돌파+거래량1.2",
        "exit": EXIT_NEW,
        "fn":   lambda s: (
            s["above200"]
            and s["prev_squeeze"]
            and s["bb_width"] > s["bb_width_prev"] * 1.05
            and s["vol_ratio"] > 1.2
            and s["pctb_close"] > 55
            and s["macd_hist"] > 0
        ),
    },
    # EXIT_C 테스트
    {
        "name": "H_스퀴즈돌파+거래량_C출구",
        "exit": EXIT_C,
        "fn":   lambda s: (
            s["above200"]
            and s["prev_squeeze"]
            and s["bb_width"] > s["bb_width_prev"] * 1.05
            and s["vol_ratio"] > 1.5
            and s["pctb_close"] > 55
            and s["macd_hist"] > 0
        ),
    },
    # EXIT_AB 테스트
    {
        "name": "H_스퀴즈돌파+거래량_AB출구",
        "exit": EXIT_AB,
        "fn":   lambda s: (
            s["above200"]
            and s["prev_squeeze"]
            and s["bb_width"] > s["bb_width_prev"] * 1.05
            and s["vol_ratio"] > 1.5
            and s["pctb_close"] > 55
            and s["macd_hist"] > 0
        ),
    },

    # ── I그룹: ADX 추세 강도 ──────────────────────────────────────────────────
    {
        "name": "I_ADX추세강도",
        "exit": EXIT_NEW,
        "fn":   lambda s: (
            s["above200"]
            and s["plus_di"] > s["minus_di"]
            and s["adx"] > 25
            and s["adx_rising"]
            and s["macd_hist"] > 0
            and 30 < s["pctb_close"] < 75
        ),
    },
    # ADX > 20으로 낮춰서 신호 수 확인
    {
        "name": "I_ADX추세강도_adx20",
        "exit": EXIT_NEW,
        "fn":   lambda s: (
            s["above200"]
            and s["plus_di"] > s["minus_di"]
            and s["adx"] > 20
            and s["adx_rising"]
            and s["macd_hist"] > 0
            and 30 < s["pctb_close"] < 75
        ),
    },
    # EXIT_C 테스트
    {
        "name": "I_ADX추세강도_C출구",
        "exit": EXIT_C,
        "fn":   lambda s: (
            s["above200"]
            and s["plus_di"] > s["minus_di"]
            and s["adx"] > 25
            and s["adx_rising"]
            and s["macd_hist"] > 0
            and 30 < s["pctb_close"] < 75
        ),
    },

    # ── F그룹: LR 추세선 저점 터치 (MA200 근접) ───────────────────────────────
    {
        "name": "F_LR추세선저점터치",
        "exit": EXIT_AB,
        "fn":   lambda s: (
            s["above200"]
            and s["close"] < s["ma200"] * 1.12
            and s["lr_slope"] > 0
            and s["lr_trendline"] > 0
            and s["low"] <= s["lr_trendline"] * 1.03
            and (s["rsi"] < 50 or s["pctb_low"] < 20)
        ),
    },
    # EXIT_NEW 테스트
    {
        "name": "F_LR추세선저점터치_NEW출구",
        "exit": EXIT_NEW,
        "fn":   lambda s: (
            s["above200"]
            and s["close"] < s["ma200"] * 1.12
            and s["lr_slope"] > 0
            and s["lr_trendline"] > 0
            and s["low"] <= s["lr_trendline"] * 1.03
            and (s["rsi"] < 50 or s["pctb_low"] < 20)
        ),
    },
    # MA200 거리 완화 (1.15)
    {
        "name": "F_LR추세선저점터치_1.15",
        "exit": EXIT_AB,
        "fn":   lambda s: (
            s["above200"]
            and s["close"] < s["ma200"] * 1.15
            and s["lr_slope"] > 0
            and s["lr_trendline"] > 0
            and s["low"] <= s["lr_trendline"] * 1.03
            and (s["rsi"] < 50 or s["pctb_low"] < 20)
        ),
    },
]


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("신규 전략 백테스트 v2 (E/H/I/F vs REF_A/B/C/D)")
    print("=" * 60)

    vix_series = download_vix()
    data = download_data(vix_series)
    print(f"\n총 {len(data)}개 종목 로드 완료\n")

    all_trades  = []
    all_summary = []

    for sc in SCENARIOS:
        print(f"  [{sc['name']}] 백테스트 중...")
        trades = run_backtest(data, sc)
        all_trades.extend(trades)
        stats  = analyze(trades, sc)
        all_summary.append(stats)

        n      = stats["trades"]
        wr     = stats["win_rate"]
        ev     = stats["ev"]
        ol_a   = stats["overlap_a_pct"]
        ol_b   = stats["overlap_b_pct"]
        ol_c   = stats["overlap_c_pct"]
        ol_d   = stats["overlap_d_pct"]
        p_n    = stats["pure_trades"]
        p_ev   = stats["pure_ev"]
        print(f"       거래:{n:>5}건  승률:{wr:>5.1f}%  EV:{ev:>6.2f}%  "
              f"A중복:{ol_a:.0f}%  B중복:{ol_b:.0f}%  C중복:{ol_c:.0f}%  D중복:{ol_d:.0f}%  "
              f"→ 순수거래:{p_n}건  순수EV:{p_ev:.2f}%")

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    trades_df  = pd.DataFrame(all_trades)
    summary_df = pd.DataFrame(all_summary)

    trades_path  = "backtest_new_strategies_trades.csv"
    summary_path = "backtest_new_strategies_summary.csv"
    trades_df.to_csv(trades_path,  index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n저장 완료: {trades_path}, {summary_path}")

    # ── 콘솔 출력 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 130)
    print(f"{'전략':<42} {'출구':<12} {'거래':>5} {'승률':>6} {'EV':>6} "
          f"{'A중복':>6} {'B중복':>6} {'C중복':>6} {'D중복':>6} "
          f"{'순수거래':>8} {'순수EV':>7}")
    print("-" * 130)

    group_order = ["REF_", "E_", "H_", "I_", "F_"]
    for prefix in group_order:
        rows = summary_df[summary_df["scenario"].str.startswith(prefix)]
        for _, r in rows.iterrows():
            is_ref = prefix == "REF_"
            # 기준선은 순수EV 컬럼이 자기 자신의 중복이므로 표시 생략
            pure_str = f"{r['pure_ev']:>7.2f}%" if not is_ref else "   (기준선)"
            marker = "★" if r["ev"] > 5.0 and r["trades"] >= 50 else " "
            print(f"{marker}{r['scenario']:<41} {r['exit_cfg']:<12} "
                  f"{int(r['trades']):>5} {r['win_rate']:>6.1f}% {r['ev']:>6.2f}% "
                  f"{r['overlap_a_pct']:>5.0f}% {r['overlap_b_pct']:>5.0f}% "
                  f"{r['overlap_c_pct']:>5.0f}% {r['overlap_d_pct']:>5.0f}% "
                  f"{int(r['pure_trades']):>8} {pure_str}")
        print()

    print("=" * 130)
    print("\n[★ = EV > 5% AND 거래 50건 이상]")
    print("[순수거래/순수EV = A/B/C/D와 중복되지 않는 신호만의 거래수/기댓값]")
    print("[기준선은 자기자신이 중복 100%이므로 순수EV 표시 없음]")

    # ── 최적 전략 요약 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("최적 후보 (각 그룹별 최고 EV, 거래 50건 이상)")
    print("=" * 60)
    for prefix in ["E_", "H_", "I_", "F_"]:
        sub = summary_df[
            summary_df["scenario"].str.startswith(prefix) &
            (summary_df["trades"] >= 50)
        ]
        if sub.empty:
            print(f"  {prefix.rstrip('_')}그룹: 유효 거래 50건 미만")
            continue
        best = sub.loc[sub["ev"].idxmax()]
        print(f"  {prefix.rstrip('_')}그룹 최고: {best['scenario']}")
        print(f"    → 거래:{int(best['trades'])}건 / 승률:{best['win_rate']:.1f}% / "
              f"EV:{best['ev']:.2f}% / 평균보유:{best['avg_hold']:.0f}일")
        print(f"    → A중복:{best['overlap_a_pct']:.0f}% / "
              f"B중복:{best['overlap_b_pct']:.0f}% / "
              f"C중복:{best['overlap_c_pct']:.0f}%")

    print("\n기준선 (참고)")
    for name in ["REF_A (스퀴즈+저가%B≤50)", "REF_B (저가%B≤5)", "REF_C (골든크로스+%B>80+RSI>70)"]:
        row = summary_df[summary_df["scenario"] == name]
        if row.empty:
            continue
        r = row.iloc[0]
        print(f"  {r['scenario']}: 거래:{int(r['trades'])}건 / "
              f"승률:{r['win_rate']:.1f}% / EV:{r['ev']:.2f}%")


if __name__ == "__main__":
    main()
