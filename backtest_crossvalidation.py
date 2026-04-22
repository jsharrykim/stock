"""
교차검증 백테스트
=================
목표: 이 스레드에서 도출한 필터 룰을 실제로 적용했을 때
      기존 전략 대비 성과가 개선되는지 검증

[기존 전략] 나스닥 이격도 필터 없음 (현재 시스템)
[신규 전략] 이격도 필터 적용:
  - IXIC > MA200 - 3%   → A, B, C 모두 정상 매수
  - IXIC -3% ~ -15%     → A/B 진입 차단, C만 허용
  - IXIC < MA200 - 15%  → A, B, C 모두 재개 (찐바닥)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

us_tickers = [
    "SNPS","COST","AZN","AMGN","FTNT","CDNS","ADP","FAST","ADI","TXN",
    "BKNG","MNST","ORLY","CPRT","ISRG","AAPL","AVGO","AMD","MSFT",
    "GOOGL","NVDA","TSLA","MCHP","AMZN","MU","LRCX","QCOM","ROP","ON",
    "PLTR","CRWD","APP","AXON","VST","SOXL","VRT","AEHR","LITE","TER","ANET"
]

def compute_rsi(data, window=14):
    delta = data.diff()
    ema_up = delta.clip(lower=0).ewm(com=window-1, adjust=False).mean()
    ema_down = (-delta.clip(upper=0)).ewm(com=window-1, adjust=False).mean()
    return 100 - (100 / (1 + ema_up / ema_down))

def compute_cci(df, window=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad)

def apply_filter(strategy, qqq_dist):
    """
    신규 필터 룰 적용
    Returns: True (진입 허용) / False (진입 차단)
    """
    if qqq_dist >= -3:
        return True            # 정상 + 휩쏘 구간 → 전략 무관 진입 허용
    elif -15 <= qqq_dist < -3:
        return strategy == 'C' # 데스존 → C그룹만 허용
    else:
        return True            # -15% 이하 찐바닥 → 전부 재개

def run_backtest(apply_new_filter=False):
    ixic = yf.download("^IXIC", start="2010-01-01", end="2026-01-01", progress=False)
    vix  = yf.download("^VIX",  start="2010-01-01", end="2026-01-01", progress=False)

    for df_ in [ixic, vix]:
        if isinstance(df_.columns, pd.MultiIndex):
            df_.columns = df_.columns.droplevel(1)

    ixic['MA200']   = ixic['Close'].rolling(200).mean()
    ixic['Distance']= (ixic['Close'] - ixic['MA200']) / ixic['MA200'] * 100

    trades = []

    for ticker in us_tickers:
        try:
            df = yf.download(ticker, start="2010-01-01", end="2026-01-01", progress=False)
            if len(df) < 250:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            df = df.join(vix['Close'].rename('VIX'),          how='left')
            df = df.join(ixic['Distance'].rename('IXIC_Dist'), how='left')

            df['MA200']          = df['Close'].rolling(200).mean()
            df['BB_Mid']         = df['Close'].rolling(20).mean()
            df['BB_Std']         = df['Close'].rolling(20).std()
            df['BB_Up']          = df['BB_Mid'] + 2 * df['BB_Std']
            df['BB_Low_Band']    = df['BB_Mid'] - 2 * df['BB_Std']
            df['BB_Width']       = (df['BB_Up'] - df['BB_Low_Band']) / df['BB_Mid']
            df['BB_Width_60Avg'] = df['BB_Width'].rolling(60).mean()
            df['PctB_Low']       = (df['Low'] - df['BB_Low_Band']) / (df['BB_Up'] - df['BB_Low_Band']) * 100
            df['RSI']            = compute_rsi(df['Close'], 14)
            df['CCI']            = compute_cci(df, 20)

            x = np.arange(120)
            df['LR_Slope'] = df['Low'].rolling(120).apply(
                lambda y: np.nan if np.isnan(y).any() else np.polyfit(x, y, 1)[0], raw=True)
            df['LR_Val'] = df['Low'].rolling(120).apply(
                lambda y: np.nan if np.isnan(y).any() else np.polyfit(x, y, 1)[0]*119 + np.polyfit(x, y, 1)[1], raw=True)

            cond_A = (df['Close'] > df['MA200']) & (df['BB_Width'] < df['BB_Width_60Avg'] * 0.5) & (df['PctB_Low'] <= 50)
            cond_B = (df['Close'] > df['MA200']) & (df['PctB_Low'] <= 5)
            cond_C = (df['Close'] < df['MA200']) & (df['VIX'] >= 25) & ((df['RSI'] < 40) | (df['CCI'] < -100)) & (df['LR_Slope'] > 0) & (df['Low'] <= df['LR_Val'] * 1.03)

            exit_A = (df['Close'] < df['MA200']) | (df['BB_Width'] >= df['BB_Width_60Avg'] * 0.5) | (df['PctB_Low'] > 50)
            exit_B = (df['Close'] < df['MA200']) | (df['PctB_Low'] > 5)
            exit_C = (df['Close'] > df['MA200']) | (df['VIX'] < 23) | ((df['RSI'] >= 40) & (df['CCI'] >= -100)) | (df['LR_Slope'] <= 0)

            in_trade    = False
            entry_price = 0
            entry_date  = None
            strategy    = ""
            qqq_dist_at = 0
            days_held   = 0

            for i in range(120, len(df)):
                if not in_trade:
                    ixic_d = df['IXIC_Dist'].iloc[i-1]
                    for s, cond in [('A', cond_A), ('B', cond_B), ('C', cond_C)]:
                        if not cond.iloc[i-1]:
                            continue
                        # 필터 적용 여부
                        if apply_new_filter and not apply_filter(s, ixic_d):
                            continue
                        strategy    = s
                        in_trade    = True
                        entry_price = df['Open'].iloc[i]
                        entry_date  = df.index[i]
                        qqq_dist_at = ixic_d
                        days_held   = 0
                        break
                else:
                    days_held += 1
                    curr = df['Close'].iloc[i]
                    ret  = (curr - entry_price) / entry_price
                    tgt  = 0.08 if strategy in ('A','B') else 0.20
                    stop = -0.25

                    exit_sig, reason = False, ""
                    if   ret >= tgt:                              exit_sig, reason = True, "Target"
                    elif ret <= stop:                             exit_sig, reason = True, "Stop"
                    elif days_held >= 120:                        exit_sig, reason = True, "Time"
                    elif days_held >= 60 and ret > 0:             exit_sig, reason = True, "TimeProfit"
                    elif strategy == 'A' and exit_A.iloc[i-1]:   exit_sig, reason = True, "CondExit"
                    elif strategy == 'B' and exit_B.iloc[i-1]:   exit_sig, reason = True, "CondExit"
                    elif strategy == 'C' and exit_C.iloc[i-1]:   exit_sig, reason = True, "CondExit"

                    if exit_sig:
                        trades.append({
                            'ticker'  : ticker,
                            'strategy': strategy,
                            'entry'   : entry_date,
                            'exit'    : df.index[i],
                            'return'  : ret * 100,
                            'ixic_d'  : qqq_dist_at,
                            'reason'  : reason,
                        })
                        in_trade = False
        except:
            pass

    return pd.DataFrame(trades)

# ─── 두 전략 모두 실행 ───────────────────────────────────────────────
print("=" * 60)
print("[1/2] 기존 전략 백테스트 (필터 없음)...")
res_old = run_backtest(apply_new_filter=False)

print("[2/2] 신규 전략 백테스트 (이격도 필터 적용)...")
res_new = run_backtest(apply_new_filter=True)

# ─── 비교 함수 ───────────────────────────────────────────────────────
def summarize(df, label):
    total   = len(df)
    wr      = (df['return'] > 0).mean() * 100
    avg_ret = df['return'].mean()
    stop_n  = (df['reason'] == 'Stop').sum()
    stop_pct= stop_n / total * 100 if total > 0 else 0
    avg_win = df[df['return'] > 0]['return'].mean() if (df['return'] > 0).any() else 0
    avg_los = df[df['return'] <= 0]['return'].mean() if (df['return'] <= 0).any() else 0
    profit_factor = abs(avg_win / avg_los) if avg_los != 0 else float('inf')
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  총 거래 횟수   : {total:>6,}건")
    print(f"  승률           : {wr:>6.2f}%")
    print(f"  평균 수익률    : {avg_ret:>+7.3f}%")
    print(f"  평균 수익(익절): {avg_win:>+7.3f}%")
    print(f"  평균 손실(손절): {avg_los:>+7.3f}%")
    print(f"  프로핏 팩터    : {profit_factor:>7.3f}")
    print(f"  -25% 손절 횟수 : {stop_n:>6}건  ({stop_pct:.1f}%)")
    return {
        'total': total, 'wr': wr, 'avg_ret': avg_ret,
        'stop_n': stop_n, 'stop_pct': stop_pct,
        'profit_factor': profit_factor
    }

print("\n" + "=" * 60)
print("          📊  교차검증 결과 비교")
print("=" * 60)

s_old = summarize(res_old, "기존 전략 (이격도 필터 없음)")
s_new = summarize(res_new, "신규 전략 (이격도 필터 적용: -3% / -15% 기준)")

print(f"\n{'─'*50}")
print("  📈  개선 효과 (신규 - 기존)")
print(f"{'─'*50}")
print(f"  거래 감소      : {s_new['total'] - s_old['total']:>+,}건  ({(s_new['total']-s_old['total'])/s_old['total']*100:+.1f}%)")
print(f"  승률 변화      : {s_new['wr'] - s_old['wr']:>+.2f}%p")
print(f"  평균 수익 변화 : {s_new['avg_ret'] - s_old['avg_ret']:>+.3f}%p")
print(f"  손절 횟수 감소 : {s_new['stop_n'] - s_old['stop_n']:>+}건  ({s_new['stop_pct'] - s_old['stop_pct']:+.1f}%p)")
print(f"  프로핏팩터 변화: {s_new['profit_factor'] - s_old['profit_factor']:>+.3f}")

# ─── 전략별 세부 비교 ────────────────────────────────────────────────
print(f"\n{'─'*50}")
print("  전략별 비교 (기존 vs 신규)")
print(f"{'─'*50}")
for s in ['A', 'B', 'C']:
    o = res_old[res_old['strategy'] == s]
    n = res_new[res_new['strategy'] == s]
    wr_o = (o['return'] > 0).mean() * 100 if len(o) else 0
    wr_n = (n['return'] > 0).mean() * 100 if len(n) else 0
    ar_o = o['return'].mean() if len(o) else 0
    ar_n = n['return'].mean() if len(n) else 0
    sn_o = (o['reason'] == 'Stop').sum()
    sn_n = (n['reason'] == 'Stop').sum()
    print(f"  [{s}그룹] 거래:{len(o):>4}→{len(n):>4}건 | 승률:{wr_o:>5.1f}→{wr_n:>5.1f}% | 평균:{ar_o:>+6.2f}→{ar_n:>+6.2f}% | 손절:{sn_o:>3}→{sn_n:>3}건")

# ─── IXIC 이격도 구간별 비교 ─────────────────────────────────────────
print(f"\n{'─'*50}")
print("  IXIC 이격도 구간별 비교 (데스존 구간 확인용)")
print(f"{'─'*50}")
bins   = [float('-inf'), -15, -3, 0, float('inf')]
blabels= ['< -15% (찐바닥)', '-15% ~ -3% (데스존)', '-3% ~ 0% (휩쏘)', '> 0% (정상)']
for b, bl in zip(zip(bins, bins[1:]), blabels):
    lo, hi = b
    o_seg = res_old[(res_old['ixic_d'] > lo) & (res_old['ixic_d'] <= hi)] if hi != float('inf') else res_old[res_old['ixic_d'] > lo]
    n_seg = res_new[(res_new['ixic_d'] > lo) & (res_new['ixic_d'] <= hi)] if hi != float('inf') else res_new[res_new['ixic_d'] > lo]
    wr_o  = (o_seg['return'] > 0).mean() * 100 if len(o_seg) else 0
    wr_n  = (n_seg['return'] > 0).mean() * 100 if len(n_seg) else 0
    ar_o  = o_seg['return'].mean() if len(o_seg) else 0
    ar_n  = n_seg['return'].mean() if len(n_seg) else 0
    print(f"  {bl}")
    print(f"    기존: {len(o_seg):>4}건 | 승률 {wr_o:>5.1f}% | 평균수익 {ar_o:>+6.2f}%")
    print(f"    신규: {len(n_seg):>4}건 | 승률 {wr_n:>5.1f}% | 평균수익 {ar_n:>+6.2f}%")
