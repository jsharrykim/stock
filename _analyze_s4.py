import pandas as pd
import numpy as np

df = pd.read_csv("backtest_squeeze_compare_trades.csv")
upper = df[df["scenario"].str.contains("상방")].copy()

print("=== 평균 보유일 비교 ===")
for s in ["S1_기존전략_상방","S2_스퀴즈단독_상방","S3_기존AND스퀴즈_상방","S4_기존OR스퀴즈_상방"]:
    sub = upper[upper["scenario"]==s]
    print(f"{s}: 평균 {sub['hold_days'].mean():.1f}일 / 중간값 {sub['hold_days'].median():.0f}일 / 거래수 {len(sub)}")

print()
print("=== 연간 평균 거래 횟수 ===")
upper["entry_date"] = pd.to_datetime(upper["entry_date"])
upper["year"] = upper["entry_date"].dt.year
by_year = upper.groupby(["scenario","year"])["ticker"].count().reset_index()
by_year.columns = ["scenario","year","trades"]
summary = by_year.groupby("scenario")["trades"].mean().reset_index()
summary.columns = ["scenario","연평균거래수"]
for _, r in summary.iterrows():
    print(f"  {r['scenario']}: 연평균 {r['연평균거래수']:.0f}건")

print()
print("=== 수익률 분포 ===")
for s, label in [("S1_기존전략_상방","S1"), ("S2_스퀴즈단독_상방","S2"),
                 ("S3_기존AND스퀴즈_상방","S3"), ("S4_기존OR스퀴즈_상방","S4")]:
    sub = upper[upper["scenario"]==s]["pnl_pct"].astype(float)
    pct_neg = (sub < 0).mean() * 100
    pct_pos = (sub > 0).mean() * 100
    print(f"  {label}: 양수 {pct_pos:.1f}% / 음수 {pct_neg:.1f}% / 평균 {sub.mean():.2f}% / 표준편차 {sub.std():.2f}%")

print()
print("=== 연도별 평균수익 비교 (S1 vs S4) ===")
print(f"{'연도':<6}", end="")
for s in ["S1_기존전략_상방","S4_기존OR스퀴즈_상방"]:
    print(f"  {s[:6]:>10}", end="")
print()
years = sorted(upper["year"].unique())
for yr in years:
    print(f"{yr:<6}", end="")
    for s in ["S1_기존전략_상방","S4_기존OR스퀴즈_상방"]:
        sub = upper[(upper["scenario"]==s) & (upper["year"]==yr)]["pnl_pct"].astype(float)
        if len(sub) > 0:
            print(f"  {sub.mean():>10.2f}%", end="")
        else:
            print(f"  {'N/A':>10}", end="")
    print()

print()
print("=== S4에서 추가된 거래 (스퀴즈 고유 기여) 품질 추정 ===")
# S4 거래를 ticker+entry_date로 S1과 비교해서 겹치지 않는 것 찾기
s1 = upper[upper["scenario"]=="S1_기존전략_상방"][["ticker","entry_date","pnl_pct"]].copy()
s4 = upper[upper["scenario"]=="S4_기존OR스퀴즈_상방"][["ticker","entry_date","pnl_pct"]].copy()

s1["key"] = s1["ticker"] + "_" + s1["entry_date"].astype(str)
s4["key"] = s4["ticker"] + "_" + s4["entry_date"].astype(str)

extra = s4[~s4["key"].isin(s1["key"])]["pnl_pct"].astype(float)
shared = s4[s4["key"].isin(s1["key"])]["pnl_pct"].astype(float)

print(f"  S4 전체 거래: {len(s4)}건")
print(f"  S1과 공통 거래: {len(shared)}건  → 평균수익 {shared.mean():.2f}%")
print(f"  스퀴즈로 추가된 거래: {len(extra)}건  → 평균수익 {extra.mean():.2f}%")
print(f"  (추가 거래의 손절 비율: {(extra < 0).mean()*100:.1f}%)")
