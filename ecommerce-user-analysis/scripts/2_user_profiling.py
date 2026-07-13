# -*- coding: utf-8 -*-
"""
================================================================================
电商用户运营分析 — Phase 2-5: 用户画像 + 生命周期 + RFM + 留存分析
================================================================================
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 加载 Phase 1 输出
# ============================================================================
df = pd.read_csv("data/cleaned_transactions.csv", parse_dates=['InvoiceDate'])
user = pd.read_csv("data/user_metrics.csv", parse_dates=['FirstPurchase', 'LastPurchase'])
reference_date = df['InvoiceDate'].max()

print("=" * 70)
print("Phase 2-5: 用户画像 + 生命周期 + RFM + 留存分析")
print(f"数据加载: {len(df):,} 交易, {len(user):,} 用户")
print("=" * 70)

# ============================================================================
# PHASE 2: 用户基础画像分析
# ============================================================================
print("\n" + "=" * 50)
print("Phase 2: 用户基础画像分析")
print("=" * 50)

# --- 用户规模 ---
user_monthly = df.groupby('YearMonth').agg(
    MAU          = ('Customer ID', 'nunique'),
    TotalRevenue = ('Revenue', 'sum'),
    TotalOrders  = ('Invoice', 'nunique')
).reset_index()
user_monthly.columns = ['YearMonth', 'MAU', 'TotalRevenue', 'TotalOrders']

# 每月新增用户
first_purchase_by_month = user.groupby(user['FirstPurchase'].dt.to_period('M').astype(str)).size()
first_purchase_by_month = first_purchase_by_month.reset_index()
first_purchase_by_month.columns = ['YearMonth', 'NewUsers']

user_monthly = user_monthly.merge(first_purchase_by_month, on='YearMonth', how='left')
user_monthly['NewUsers'] = user_monthly['NewUsers'].fillna(0).astype(int)
user_monthly.to_csv("data/data/monthly_user_stats.csv", index=False)

print("\n--- 用户规模 ---")
print(f"用户总数:     {len(user):,}")
print(f"月均活跃用户: {user_monthly['MAU'].mean():.0f}")
print(f"月均新增用户: {user_monthly['NewUsers'].mean():.0f}")
print(f"\n月度趋势 (最近6月):")
print(user_monthly.tail(6)[['YearMonth', 'MAU', 'NewUsers', 'TotalRevenue']].to_string(index=False))

# --- 用户消费画像 ---
print("\n--- 用户消费画像 ---")
# 消费金额分布
print("\n[消费金额分布]")
rev_bins = [0, 100, 500, 1000, 3000, 5000, 10000, float('inf')]
rev_labels = ['<100', '100-500', '500-1K', '1K-3K', '3K-5K', '5K-10K', '>10K']
user['RevSegment'] = pd.cut(user['TotalRevenue'], bins=rev_bins, labels=rev_labels)
rev_dist = user['RevSegment'].value_counts().sort_index()
for seg, cnt in rev_dist.items():
    print(f"  {seg:>10s}: {cnt:>5,} 人 ({cnt/len(user)*100:5.1f}%)")

# 购买频率分布
print("\n[购买频次分布]")
freq_bins = [0, 1, 2, 5, 10, 20, 50, float('inf')]
freq_labels = ['1次', '2次', '3-5次', '6-10次', '11-20次', '21-50次', '50+次']
user['FreqSegment'] = pd.cut(user['TotalOrders'], bins=freq_bins, labels=freq_labels)
freq_dist = user['FreqSegment'].value_counts().sort_index()
for seg, cnt in freq_dist.items():
    print(f"  {seg:>10s}: {cnt:>5,} 人 ({cnt/len(user)*100:5.1f}%)")

# 地区分布
print("\n[用户地区分布 Top 10]")
country_dist = df.groupby('Country')['Customer ID'].nunique().sort_values(ascending=False)
for i, (country, cnt) in enumerate(country_dist.head(10).items()):
    print(f"  {i+1:>2}. {country:<25s}: {cnt:>5,} 人 ({cnt/len(user)*100:5.1f}%)")

# 消费金额统计
print(f"\n[消费金额统计]")
print(f"  平均消费: {user['TotalRevenue'].mean():,.2f}")
print(f"  中位消费: {user['TotalRevenue'].median():,.2f}")
print(f"  最高消费: {user['TotalRevenue'].max():,.2f}")
print(f"  消费 Top 20% 贡献: {user.nlargest(int(len(user)*0.2), 'TotalRevenue')['TotalRevenue'].sum()/user['TotalRevenue'].sum()*100:.1f}%")

# ============================================================================
# PHASE 3: 用户生命周期分析
# ============================================================================
print("\n" + "=" * 50)
print("Phase 3: 用户生命周期分析")
print("=" * 50)

# --- 新用户分析 ---
print("\n--- 新用户增长 ---")
print(f"月均新增: {user_monthly['NewUsers'].mean():.0f} 人")
print(f"新增最多月: {user_monthly.loc[user_monthly['NewUsers'].idxmax(), 'YearMonth']} ({user_monthly['NewUsers'].max()} 人)")
print(f"新增最少月: {user_monthly.loc[user_monthly['NewUsers'].idxmin(), 'YearMonth']} ({user_monthly['NewUsers'].min()} 人)")

# --- 复购分析 ---
repeat_users = user[user['TotalOrders'] > 1]
print(f"\n--- 复购分析 ---")
print(f"复购用户数:   {len(repeat_users):,} ({len(repeat_users)/len(user)*100:.1f}%)")
print(f"仅购买1次:    {len(user[user['TotalOrders']==1]):,} ({len(user[user['TotalOrders']==1])/len(user)*100:.1f}%)")
print(f"平均购买次数: {user['TotalOrders'].mean():.2f}")
print(f"中位购买次数: {user['TotalOrders'].median():.0f}")

# 平均购买间隔
multi_buyers = user[user['TotalOrders'] >= 2].copy()
if len(multi_buyers) > 0:
    multi_buyers['AvgInterval'] = multi_buyers['LifeDays'] / (multi_buyers['TotalOrders'] - 1)
    print(f"平均购买间隔 (多单用户): {multi_buyers['AvgInterval'].median():.0f} 天")

# --- 用户生命周期阶段划分 ---
print(f"\n--- 用户生命周期阶段 ---")
# 基于最近购买时间和消费行为划分
now = reference_date

def classify_lifecycle(row):
    days_since_last = (now - row['LastPurchase']).days
    if row['TotalOrders'] == 1 and days_since_last <= 90:
        return 'New'
    elif row['TotalOrders'] >= 2 and days_since_last <= 60:
        if row['TotalRevenue'] >= user['TotalRevenue'].quantile(0.7):
            return 'VIP'
        return 'Active'
    elif days_since_last > 180:
        return 'AtRisk'
    elif row['TotalOrders'] >= 2 and days_since_last <= 180:
        return 'Growing'
    else:
        return 'Normal'

user['LifeStage'] = user.apply(classify_lifecycle, axis=1)
stage_dist = user['LifeStage'].value_counts()

stage_labels = {
    'New':     '新用户 (最近90天首次购买)',
    'Active':  '活跃用户 (60天内有购买)',
    'VIP':     '高价值用户 (高频高消费)',
    'Growing': '成长用户 (持续购买中)',
    'Normal':  '普通用户',
    'AtRisk':  '流失风险用户 (180+天未购买)'
}
print(f"参考日期: {now.date()}")
for stage, cnt in stage_dist.items():
    label = stage_labels.get(stage, stage)
    print(f"  {label:<40s}: {cnt:>5,} 人 ({cnt/len(user)*100:5.1f}%)")

user[['Customer ID', 'LifeStage']].to_csv("data/data/user_lifecycle_stage.csv", index=False)

# ============================================================================
# PHASE 4: RFM 用户价值分析
# ============================================================================
print("\n" + "=" * 50)
print("Phase 4: RFM 用户价值分析")
print("=" * 50)

# --- 构建 RFM 指标 ---
rfm = user[['Customer ID', 'TotalOrders', 'TotalRevenue', 'LastPurchase']].copy()
rfm['Recency'] = (reference_date - rfm['LastPurchase']).dt.days
rfm.columns = ['CustomerID', 'Frequency', 'Monetary', 'LastPurchase', 'Recency']

print(f"\n[RFM指标描述]")
print(f"  Recency  (R): 均值={rfm['Recency'].mean():.0f}天, 中位={rfm['Recency'].median():.0f}天")
print(f"  Frequency(F): 均值={rfm['Frequency'].mean():.2f}, 中位={rfm['Frequency'].median():.0f}")
print(f"  Monetary (M): 均值={rfm['Monetary'].mean():,.2f}, 中位={rfm['Monetary'].median():,.2f}")

# --- 方法1: RFM 规则分层 ---
# 为 R, F, M 各打分 1-4
rfm['R_Score'] = pd.qcut(rfm['Recency'], q=4, labels=[4, 3, 2, 1]).astype(int)  # R越小越好
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)
rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)

rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

def classify_rfm(row):
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3 and m >= 3:
        return 'Loyal'
    elif r >= 4 and f <= 2:
        return 'Potential'
    elif r <= 2 and f >= 3 and m >= 3:
        return 'AtRisk'
    elif r <= 2 and f <= 2:
        return 'Lost'
    elif r >= 3 and f <= 2 and m <= 2:
        return 'New'
    else:
        return 'Average'

rfm['RFM_Segment'] = rfm.apply(classify_rfm, axis=1)

rfm_labels = {
    'Champions': '高价值冠军用户',
    'Loyal':     '忠诚用户',
    'Potential': '潜力用户',
    'AtRisk':    '流失风险用户',
    'Lost':      '流失用户',
    'New':       '新用户',
    'Average':   '普通用户'
}

print(f"\n[RFM 规则分层结果]")
seg_counts = rfm['RFM_Segment'].value_counts()
for seg, cnt in seg_counts.items():
    label = rfm_labels.get(seg, seg)
    rev = rfm[rfm['RFM_Segment']==seg]['Monetary'].sum()
    print(f"  {label:<20s}: {cnt:>5,} 人 ({cnt/len(rfm)*100:5.1f}%), 贡献营收: {rev:,.0f} ({rev/rfm['Monetary'].sum()*100:.1f}%)")

# --- 方法2: KMeans 聚类 ---
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print(f"\n[KMeans 聚类]")
# 标准化
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Elbow method 选择 K
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    inertias.append(km.inertia_)

# 使用 K=4 或 5
k_opt = 5
kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# 聚类画像
cluster_profile = rfm.groupby('Cluster').agg(
    Count    = ('CustomerID', 'count'),
    AvgR     = ('Recency', 'mean'),
    AvgF     = ('Frequency', 'mean'),
    AvgM     = ('Monetary', 'mean'),
    TotalM   = ('Monetary', 'sum')
).round(1)
cluster_profile['Pct'] = (cluster_profile['Count'] / len(rfm) * 100).round(1)
print(cluster_profile.to_string())

# 保存 RFM 结果
rfm.to_csv("data/data/rfm_analysis.csv", index=False)
print(f"\n[OK] rfm_analysis.csv 保存完成")

# ============================================================================
# PHASE 5: 用户留存分析 (Cohort)
# ============================================================================
print("\n" + "=" * 50)
print("Phase 5: 用户留存分析 (Cohort)")
print("=" * 50)

# 每个用户首次购买月份
user_first = df.groupby('Customer ID')['InvoiceDate'].min().reset_index()
user_first.columns = ['Customer ID', 'CohortMonth']
user_first['CohortMonth'] = user_first['CohortMonth'].dt.to_period('M').astype(str)

# 用户每月购买记录
df['OrderMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)
user_monthly_orders = df[['Customer ID', 'OrderMonth']].drop_duplicates()

# Merge cohort info
cohort_data = user_monthly_orders.merge(user_first, on='Customer ID', how='left')

# 计算 Cohort Index
def month_diff(m1, m2):
    """计算月份差"""
    y1, m1_num = map(int, m1.split('-'))
    y2, m2_num = map(int, m2.split('-'))
    return (y2 - y1) * 12 + (m2_num - m1_num)

cohort_data['PeriodIndex'] = cohort_data.apply(
    lambda r: month_diff(r['CohortMonth'], r['OrderMonth']), axis=1
)

# Cohort 留存矩阵
cohort_size = cohort_data.groupby('CohortMonth')['Customer ID'].nunique().reset_index()
cohort_size.columns = ['CohortMonth', 'CohortSize']

cohort_matrix = cohort_data.groupby(['CohortMonth', 'PeriodIndex'])['Customer ID'].nunique().reset_index()
cohort_matrix = cohort_matrix.merge(cohort_size, on='CohortMonth')
cohort_matrix['RetentionRate'] = (cohort_matrix['Customer ID'] / cohort_matrix['CohortSize'] * 100).round(1)

# Pivot
retention_pivot = cohort_matrix.pivot_table(
    index='CohortMonth', columns='PeriodIndex', values='RetentionRate'
)

print(f"\n[Cohort 留存矩阵 (留存率 %)]")
print(retention_pivot.to_string())

# 保存
retention_pivot.to_csv("data/data/cohort_retention.csv")
print(f"\n[OK] cohort_retention.csv 保存完成")

# --- 留存洞察 ---
print(f"\n[留存洞察]")
# 次月留存率趋势
month1_retention = cohort_matrix[cohort_matrix['PeriodIndex']==1][['CohortMonth', 'RetentionRate']]
if len(month1_retention) > 0:
    print(f"次月留存率: 均值 {month1_retention['RetentionRate'].mean():.1f}%, "
          f"最高 {month1_retention['RetentionRate'].max():.1f}% ({month1_retention.loc[month1_retention['RetentionRate'].idxmax(), 'CohortMonth']}), "
          f"最低 {month1_retention['RetentionRate'].min():.1f}% ({month1_retention.loc[month1_retention['RetentionRate'].idxmin(), 'CohortMonth']})")

# 3个月后留存率
month3_retention = cohort_matrix[cohort_matrix['PeriodIndex']==3][['CohortMonth', 'RetentionRate']]
if len(month3_retention) > 0:
    print(f"3月留存率:  均值 {month3_retention['RetentionRate'].mean():.1f}%, "
          f"最高 {month3_retention['RetentionRate'].max():.1f}% ({month3_retention.loc[month3_retention['RetentionRate'].idxmax(), 'CohortMonth']})")

# 流失节点
all_months = cohort_matrix.groupby('PeriodIndex')['RetentionRate'].mean().reset_index()
all_months.columns = ['Month', 'AvgRetention']
print(f"\n各月平均留存率:")
for _, row in all_months.iterrows():
    if row['Month'] <= 12:
        print(f"  M+{int(row['Month']):>2d}: {row['AvgRetention']:5.1f}%")

print("\n===== Phase 2-5 完成! =====")
