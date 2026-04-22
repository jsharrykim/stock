import yfinance as yf
import pandas as pd
import numpy as np

# A/B/C 전략의 개념 복기용 간단한 데이터 출력
df = pd.read_csv('backtest_results_log.csv') if False else None # just keeping structure
