import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 1. Tickers (US stocks only for speed, 54 tickers)
us_tickers = ["SNPS", "COST", "AZN", "AMGN", "FTNT", "CDNS", "ADP", "FAST", "ADI", "TXN", "BKNG", "MNST", "ORLY", "HOOD", "CPRT", "ISRG", "AAPL", "AVGO", "AMD", "MSFT", "GOOGL", "NVDA", "TSLA", "MCHP", "AMZN", "MU", "LRCX", "QCOM", "ROP", "ON", "ASTS", "AVAV", "IONQ", "SGML", "RKLB", "PLTR", "CRWD", "APP", "AXON", "VST", "GEV", "SOXL", "TSLL", "TE", "ONDS", "BE", "PL", "VRT", "AEHR", "LITE", "TER", "ANET"]

def compute_rsi(data, window=14):
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=window-1, adjust=False).mean()
    ema_down = down.ewm(com=window-1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def compute_cci(df, window=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad)

def run_backtest():
    print("Downloading ^IXIC & VIX data...")
    qqq = yf.download("^IXIC", start="2010-01-01", end="2026-01-01", progress=False)
    if isinstance(qqq.columns, pd.MultiIndex):
        qqq.columns = qqq.columns.droplevel(1)
    
    vix = yf.download("^VIX", start="2010-01-01", end="2026-01-01", progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.droplevel(1)
    
    qqq['MA200'] = qqq['Close'].rolling(200).mean()
    qqq['Distance'] = (qqq['Close'] - qqq['MA200']) / qqq['MA200'] * 100
    
    trades = []

    for ticker in us_tickers:
        try:
            df = yf.download(ticker, start="2010-01-01", end="2026-01-01", progress=False)
            if len(df) < 250:
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            # Align with QQQ/VIX
            df = df.join(vix['Close'].rename('VIX'), how='left')
            df = df.join(qqq['Distance'].rename('QQQ_Dist'), how='left')
            
            df['MA200'] = df['Close'].rolling(200).mean()
            df['MA60'] = df['Close'].rolling(60).mean()
            
            # Bollinger Bands
            df['BB_Mid'] = df['Close'].rolling(20).mean()
            df['BB_Std'] = df['Close'].rolling(20).std()
            df['BB_Up'] = df['BB_Mid'] + 2 * df['BB_Std']
            df['BB_Low'] = df['BB_Mid'] - 2 * df['BB_Std']
            df['BB_Width'] = (df['BB_Up'] - df['BB_Low']) / df['BB_Mid']
            df['BB_Width_60Avg'] = df['BB_Width'].rolling(60).mean()
            df['PctB_Low'] = (df['Low'] - df['BB_Low']) / (df['BB_Up'] - df['BB_Low']) * 100
            
            # RSI & CCI
            df['RSI'] = compute_rsi(df['Close'], 14)
            df['CCI'] = compute_cci(df, 20)
            
            # LR Trendline slope and value
            x = np.arange(120)
            def lr_slope(y):
                if np.isnan(y).any(): return np.nan
                return np.polyfit(x, y, 1)[0]
            def lr_val(y):
                if np.isnan(y).any(): return np.nan
                p = np.polyfit(x, y, 1)
                return p[0]*119 + p[1]
                
            df['LR_Slope'] = df['Low'].rolling(120).apply(lr_slope, raw=True)
            df['LR_Val'] = df['Low'].rolling(120).apply(lr_val, raw=True)
            
            cond_A = (df['Close'] > df['MA200']) & (df['BB_Width'] < df['BB_Width_60Avg'] * 0.5) & (df['PctB_Low'] <= 50)
            cond_B = (df['Close'] > df['MA200']) & (df['PctB_Low'] <= 5)
            cond_C = (df['Close'] < df['MA200']) & (df['VIX'] >= 25) & ((df['RSI'] < 40) | (df['CCI'] < -100)) & (df['LR_Slope'] > 0) & (df['Low'] <= df['LR_Val'] * 1.03)
            
            exit_A = (df['Close'] < df['MA200']) | (df['BB_Width'] >= df['BB_Width_60Avg'] * 0.5) | (df['PctB_Low'] > 50)
            exit_B = (df['Close'] < df['MA200']) | (df['PctB_Low'] > 5)
            exit_C = (df['Close'] > df['MA200']) | (df['VIX'] < 23) | ((df['RSI'] >= 40) & (df['CCI'] >= -100)) | (df['LR_Slope'] <= 0)
            
            in_trade = False
            entry_price = 0
            entry_date = None
            strategy = ""
            qqq_dist_at_entry = 0
            days_held = 0
            
            for i in range(120, len(df)):
                if not in_trade:
                    A_trigger = cond_A.iloc[i-1]
                    B_trigger = cond_B.iloc[i-1]
                    C_trigger = cond_C.iloc[i-1]
                    
                    if A_trigger: strategy = 'A'; in_trade = True
                    elif B_trigger: strategy = 'B'; in_trade = True
                    elif C_trigger: strategy = 'C'; in_trade = True
                        
                    if in_trade:
                        entry_price = df['Open'].iloc[i]
                        entry_date = df.index[i]
                        qqq_dist_at_entry = df['QQQ_Dist'].iloc[i-1]
                        days_held = 0
                else:
                    days_held += 1
                    curr_close = df['Close'].iloc[i]
                    ret = (curr_close - entry_price) / entry_price
                    
                    target = 0.08 if strategy in ['A', 'B'] else 0.20
                    stop = -0.25
                    
                    exit_signal = False
                    reason = ""
                    
                    if ret >= target:
                        exit_signal, reason = True, "Target"
                    elif ret <= stop:
                        exit_signal, reason = True, "Stop"
                    elif days_held >= 120:
                        exit_signal, reason = True, "Time"
                    elif days_held >= 60 and ret > 0:
                        exit_signal, reason = True, "TimeProfit"
                    else:
                        if strategy == 'A' and exit_A.iloc[i-1]: exit_signal, reason = True, "CondExit"
                        elif strategy == 'B' and exit_B.iloc[i-1]: exit_signal, reason = True, "CondExit"
                        elif strategy == 'C' and exit_C.iloc[i-1]: exit_signal, reason = True, "CondExit"
                        
                    if exit_signal:
                        trades.append({
                            'ticker': ticker,
                            'strategy': strategy,
                            'entry_date': entry_date,
                            'exit_date': df.index[i],
                            'return': ret * 100,
                            'qqq_dist': qqq_dist_at_entry,
                            'reason': reason
                        })
                        in_trade = False
        except Exception as e:
            print(f"Error on {ticker}: {e}")

    if not trades:
        print("No trades found.")
        return
        
    res = pd.DataFrame(trades)
    
    bins = [float('-inf'), -15, -12, -9, -7, -5, -4, -3, -2, -1, 0, float('inf')]
    labels = ['< -15%', '-15% ~ -12%', '-12% ~ -9%', '-9% ~ -7%', '-7% ~ -5%', '-5% ~ -4%', '-4% ~ -3%', '-3% ~ -2%', '-2% ~ -1%', '-1% ~ 0%', '> 0%']
    res['QQQ_Dist_Bin'] = pd.cut(res['qqq_dist'], bins=bins, labels=labels)
    
    print("\n=== Overall Performance ===")
    print(f"Total Trades: {len(res)}")
    print(f"Win Rate: {(res['return'] > 0).mean()*100:.2f}%")
    print(f"Avg Return: {res['return'].mean():.2f}%")
    
    print("\n=== Performance by QQQ Distance from 200 SMA ===")
    summary = res.groupby('QQQ_Dist_Bin').agg(
        Trades=('return', 'count'),
        WinRate=('return', lambda x: (x > 0).mean() * 100 if len(x) > 0 else 0),
        AvgReturn=('return', 'mean')
    ).fillna(0)
    print(summary)
    
    print("\n=== Strategy Performance by Bins ===")
    if not res.empty:
        strat_summary = res.groupby(['strategy', 'QQQ_Dist_Bin']).agg(
            Trades=('return', 'count'),
            WinRate=('return', lambda x: (x > 0).mean() * 100 if len(x) > 0 else 0),
            AvgReturn=('return', 'mean')
        ).dropna()
        print(strat_summary[strat_summary['Trades'] > 0])
    else:
        print("No trades found.")

run_backtest()