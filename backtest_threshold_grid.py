"""
이격도 threshold 최적화 백테스트
=================================
상단 컷 (휩쏘 허용 한계): -1%, -2%, -3%, -5%, -7%
하단 재개 (찐바닥 진입): -10%, -12%, -15%, -18%, -20%, -999(사용안함)

모든 조합을 grid search → 승률/평균수익/프로핏팩터/손절횟수 비교
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
    ema_up   = delta.clip(lower=0).ewm(com=window-1, adjust=False).mean()
    ema_down = (-delta.clip(upper=0)).ewm(com=window-1, adjust=False).mean()
    return 100 - (100 / (1 + ema_up / ema_down))

def compute_cci(df, window=20):
    tp  = (df['High'] + df['Low'] + df['Close']) / 3
    sma = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad)

# ─── 데이터 사전 다운로드 ─────────────────────────────────────────────
print("데이터 다운로드 중...")
ixic = yf.download("^IXIC", start="2010-01-01", end="2026-01-01", progress=False)
vix  = yf.download("^VIX",  start="2010-01-01", end="2026-01-01", progress=False)
for df_ in [ixic, vix]:
    if isinstance(df_.columns, pd.MultiIndex):
        df_.columns = df_.columns.droplevel(1)
ixic['MA200']    = ixic['Close'].rolling(200).mean()
ixic['Distance'] = (ixic['Close'] - ixic['MA200']) / ixic['MA200'] * 100

# 종목 데이터 사전 준비
stock_data = {}
x = np.arange(120)
print("종목 지표 계산 중...")
for ticker in us_tickers:
    try:
        df = yf.download(ticker, start="2010-01-01", end="2026-01-01", progress=False)
        if len(df) < 250: continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.join(vix['Close'].rename('VIX'),           how='left')
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
        df['LR_Slope']       = df['Low'].rolling(120).apply(
            lambda y: np.nan if np.isnan(y).any() else np.polyfit(x, y, 1)[0], raw=True)
        df['LR_Val']         = df['Low'].rolling(120).apply(
            lambda y: np.nan if np.isnan(y).any() else
                      np.polyfit(x, y, 1)[0]*119 + np.polyfit(x, y, 1)[1], raw=True)

        df['cond_A'] = (df['Close'] > df['MA200']) & (df['BB_Width'] < df['BB_Width_60Avg'] * 0.5) & (df['PctB_Low'] <= 50)
        df['cond_B'] = (df['Close'] > df['MA200']) & (df['PctB_Low'] <= 5)
        df['cond_C'] = (df['Close'] < df['MA200']) & (df['VIX'] >= 25) & ((df['RSI'] < 40) | (df['CCI'] < -100)) & (df['LR_Slope'] > 0) & (df['Low'] <= df['LR_Val'] * 1.03)
        df['exit_A'] = (df['Close'] < df['MA200']) | (df['BB_Width'] >= df['BB_Width_60Avg'] * 0.5) | (df['PctB_Low'] > 50)
        df['exit_B'] = (df['Close'] < df['MA200']) | (df['PctB_Low'] > 5)
        df['exit_C'] = (df['Close'] > df['MA200']) | (df['VIX'] < 23) | ((df['RSI'] >= 40) & (df['CCI'] >= -100)) | (df['LR_Slope'] <= 0)

        stock_data[ticker] = df
    except:
        pass

print(f"{len(stock_data)}개 종목 준비 완료\n")

# ─── 단일 파라미터셋으로 백테스트 실행 ──────────────────────────────
def run_with_params(upper_cut, lower_resume):
    """
    upper_cut   : 이 값 이하로 떨어지면 A/B 차단 시작 (e.g. -3)
    lower_resume: 이 값 이하면 찐바닥이므로 다시 허용 (e.g. -15), None이면 기능 없음
    """
    all_trades = []
    for ticker, df in stock_data.items():
        in_trade = False
        entry_price = 0; entry_date = None
        strategy = ""; days_held = 0

        for i in range(120, len(df)):
            ixic_d = df['IXIC_Dist'].iloc[i-1]

            if not in_trade:
                for s in ['A','B','C']:
                    if not df[f'cond_{s}'].iloc[i-1]: continue

                    # 이격도 필터
                    if ixic_d < upper_cut:  # 정상 구간 벗어남
                        # 찐바닥 재개 조건
                        if lower_resume is not None and ixic_d <= lower_resume:
                            pass  # 허용
                        elif s == 'C':
                            pass  # C는 항상 허용
                        else:
                            continue  # A/B 차단

                    strategy = s; in_trade = True
                    entry_price = df['Open'].iloc[i]
                    entry_date  = df.index[i]
                    days_held   = 0
                    break
            else:
                days_held += 1
                ret  = (df['Close'].iloc[i] - entry_price) / entry_price
                tgt  = 0.08 if strategy in ('A','B') else 0.20

                exit_sig = False; reason = ""
                if   ret >= tgt:                                   exit_sig, reason = True, "Target"
                elif ret <= -0.25:                                 exit_sig, reason = True, "Stop"
                elif days_held >= 120:                             exit_sig, reason = True, "Time"
                elif days_held >= 60 and ret > 0:                  exit_sig, reason = True, "TimeProfit"
                elif df[f'exit_{strategy}'].iloc[i-1]:             exit_sig, reason = True, "CondExit"

                if exit_sig:
                    all_trades.append({
                        'strategy': strategy, 'return': ret*100,
                        'reason': reason, 'ixic_d': df['IXIC_Dist'].iloc[i-1]
                    })
                    in_trade = False

    if not all_trades: return None
    t = pd.DataFrame(all_trades)
    wr  = (t['return'] > 0).mean() * 100
    ar  = t['return'].mean()
    sn  = (t['reason'] == 'Stop').sum()
    aw  = t[t['return'] > 0]['return'].mean() if (t['return'] > 0).any() else 0
    al  = t[t['return'] <= 0]['return'].mean() if (t['return'] <= 0).any() else 0
    pf  = abs(aw/al) if al != 0 else 999
    return {'n': len(t), 'wr': wr, 'ar': ar, 'stop': sn, 'pf': pf}

# ─── Grid Search ─────────────────────────────────────────────────────
upper_cuts    = [0, -1, -2, -3, -5, -7]          # 상단 컷 (이 이하면 A/B 차단)
lower_resumes = [None, -10, -12, -15, -18, -20]  # 하단 재개 (찐바닥 복귀)

print("Grid Search 실행 중...\n")
results = []

# 베이스라인 (필터 없음)
base = run_with_params(upper_cut=-999, lower_resume=None)
print(f"{'기준선 (필터없음)':30s} | 거래:{base['n']:>5} | 승률:{base['wr']:>5.2f}% | 평균:{base['ar']:>+6.3f}% | 손절:{base['stop']:>3}건 | PF:{base['pf']:.3f}")

print("-"*95)
for uc in upper_cuts:
    for lr in lower_resumes:
        if lr is not None and lr >= uc:
            continue  # lower >= upper는 의미없음
        r = run_with_params(upper_cut=uc, lower_resume=lr)
        if r is None: continue
        label_lr = f"{lr}%" if lr is not None else "없음"
        label = f"상단컷:{uc:>3}% / 찐바닥재개:{label_lr:>5}"
        # 기준 대비 변화
        d_wr = r['wr'] - base['wr']
        d_ar = r['ar'] - base['ar']
        d_sn = r['stop'] - base['stop']
        results.append({**r, 'uc': uc, 'lr': lr, 'd_wr': d_wr, 'd_ar': d_ar, 'd_sn': d_sn})
        print(f"{label:35s} | 거래:{r['n']:>5} | 승률:{r['wr']:>5.2f}%({d_wr:+.2f}) | 평균:{r['ar']:>+6.3f}%({d_ar:+.3f}) | 손절:{r['stop']:>3}건({d_sn:+d}) | PF:{r['pf']:.3f}")

# ─── 최적 조합 선정 ──────────────────────────────────────────────────
df_r = pd.DataFrame(results)
print("\n" + "="*80)
print("  🏆  최적 조합 (평균수익률 기준 TOP 5)")
print("="*80)
top5_ar = df_r.nlargest(5, 'ar')
for _, row in top5_ar.iterrows():
    lr_str = f"{row['lr']}%" if row['lr'] is not None else "없음"
    print(f"  상단컷 {row['uc']:>3}% / 찐바닥재개 {lr_str:>5} | 승률:{row['wr']:>5.2f}% | 평균:{row['ar']:>+6.3f}% | PF:{row['pf']:.3f} | 손절:{row['stop']:.0f}건")

print("\n  🏆  최적 조합 (승률 기준 TOP 5)")
print("="*80)
top5_wr = df_r.nlargest(5, 'wr')
for _, row in top5_wr.iterrows():
    lr_str = f"{row['lr']}%" if row['lr'] is not None else "없음"
    print(f"  상단컷 {row['uc']:>3}% / 찐바닥재개 {lr_str:>5} | 승률:{row['wr']:>5.2f}% | 평균:{row['ar']:>+6.3f}% | PF:{row['pf']:.3f} | 손절:{row['stop']:.0f}건")

print("\n  🏆  최적 조합 (프로핏팩터 기준 TOP 5)")
print("="*80)
top5_pf = df_r.nlargest(5, 'pf')
for _, row in top5_pf.iterrows():
    lr_str = f"{row['lr']}%" if row['lr'] is not None else "없음"
    print(f"  상단컷 {row['uc']:>3}% / 찐바닥재개 {lr_str:>5} | 승률:{row['wr']:>5.2f}% | 평균:{row['ar']:>+6.3f}% | PF:{row['pf']:.3f} | 손절:{row['stop']:.0f}건")
