# -*- coding: utf-8 -*-
"""
================================================================================
电商用户运营分析 — Phase 8: 最终分析报告生成
================================================================================
输出: analysis_report.md + analysis_report.txt
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 加载所有中间数据
# ============================================================================
df       = pd.read_csv("data/cleaned_transactions.csv", parse_dates=['InvoiceDate'])
user     = pd.read_csv("data/user_metrics.csv", parse_dates=['FirstPurchase', 'LastPurchase'])
rfm      = pd.read_csv("data/rfm_analysis.csv")
monthly  = pd.read_csv("data/monthly_user_stats.csv")
cohort   = pd.read_csv("data/cohort_retention.csv", index_col=0)
lifecycle = pd.read_csv("data/user_lifecycle_stage.csv")

reference_date = pd.Timestamp('2010-12-09 20:01:00')
print("数据加载完成")

# ============================================================================
# 核心指标计算
# ============================================================================
total_users       = len(user)
total_revenue      = df['Revenue'].sum()
avg_revenue        = user['TotalRevenue'].mean()
median_revenue     = user['TotalRevenue'].median()
avg_orders         = user['TotalOrders'].mean()
repeat_rate        = len(user[user['TotalOrders'] > 1]) / len(user) * 100
top20_contribution = user.nlargest(int(len(user)*0.2), 'TotalRevenue')['TotalRevenue'].sum() / total_revenue * 100

# RFM 分群
rfm_labels_cn = {
    'Champions': '高价值冠军用户', 'Loyal': '忠诚用户', 'Potential': '潜力用户',
    'AtRisk': '流失风险用户', 'Lost': '流失用户', 'New': '新用户', 'Average': '普通用户'
}
rfm['SegmentCN'] = rfm['RFM_Segment'].map(rfm_labels_cn)
rfm_summary = rfm.groupby('SegmentCN').agg(
    Count=('CustomerID', 'count'),
    Pct=('CustomerID', lambda x: f"{len(x)/len(rfm)*100:.1f}%"),
    AvgR=('Recency', 'mean'),
    AvgF=('Frequency', 'mean'),
    AvgM=('Monetary', 'mean'),
    TotalM=('Monetary', 'sum')
).round(1).sort_values('Count', ascending=False)

# Cohort 留存
cohort_clean = cohort.fillna(0)
avg_month1 = cohort_clean['1'].mean()
avg_month3 = cohort_clean['3'].mean() if '3' in cohort_clean.columns else 0

# 漏斗数据
funnel = {
    '首次购买': total_users,
    '二次购买': len(user[user['TotalOrders'] >= 2]),
    '3-5次购买': len(user[user['TotalOrders'] >= 3]),
    '6次以上': len(user[user['TotalOrders'] >= 6]),
    '10次以上': len(user[user['TotalOrders'] >= 10]),
    '高价值Top20%': int(len(user) * 0.2),
}

# 月度趋势
latest_month = monthly.iloc[-2]  # exclude incomplete Dec
peak_month = monthly.loc[monthly['MAU'].idxmax()]

# ============================================================================
# 报告内容
# ============================================================================

report = f"""
{'='*70}
                   电商用户运营分析报告
                   Online Retail User Operations Analysis
{'='*70}

报告日期: 2010-12-09
数据范围: 2009-12-01 ~ 2010-12-09
数据来源: Online Retail II

{'─'*70}
第一章  数据概况
{'─'*70}

1.1 数据规模
  原始交易记录:  525,461 条
  清洗后交易:    407,664 条 (保留 77.6%)
  涉及用户:      4,312 人
  涉及商品:      4,632 种
  涉及国家:      40 个

1.2 数据清洗说明
  - 删除 Customer ID 缺失记录:  107,927 条 (20.5%)
  - 移除取消订单 (Invoice 以 C 开头): 9,839 条
  - 移除无效交易 (Quantity/Price <= 0): 31 条

1.3 关键指标总览
  总营收:        £{total_revenue:,.0f}
  用户人均消费:   £{avg_revenue:,.0f}
  消费中位数:     £{median_revenue:,.0f}
  平均购买次数:   {avg_orders:.1f} 次
  复购比率:       {repeat_rate:.1f}%
  Top 20% 贡献:   {top20_contribution:.1f}%

{'─'*70}
第二章  用户画像
{'─'*70}

2.1 用户规模与增长
  用户总数:      {total_users:,} 人
  月均活跃用户:   {monthly['MAU'].mean():.0f} 人
  月均新增用户:   {monthly['NewUsers'].mean():.0f} 人
  增长趋势:      月份间波动明显, 2009年12月新增最多 (955人),
                 随后新增速度放缓至月均 ~160人 (2010Q1-Q2),
                 2010年9-11月回升至月均 315人。

2.2 用户消费特征
  消费金额分布:
    低消费 (<£500):    {len(user[user['TotalRevenue']<500])/len(user)*100:.1f}%
    中等消费 (£500-3K): {len(user[(user['TotalRevenue']>=500)&(user['TotalRevenue']<3000)])/len(user)*100:.1f}%
    高消费 (>£3K):     {len(user[user['TotalRevenue']>=3000])/len(user)*100:.1f}%

  购买频次分布:
    仅购1次:  {len(user[user['TotalOrders']==1])/len(user)*100:.1f}%  ({len(user[user['TotalOrders']==1]):,}人)
    2-5次:    {len(user[(user['TotalOrders']>=2)&(user['TotalOrders']<=5)])/len(user)*100:.1f}%
    6次以上:  {len(user[user['TotalOrders']>=6])/len(user)*100:.1f}%

  关键洞察: 约1/3用户仅购买一次即流失，这是最大的改进空间。

2.3 用户地区分布
  英国用户占比高达 92.0%，市场集中度极高。
  其他主要市场: 德国(1.6%), 法国(1.1%), 西班牙(0.6%)
  建议: 英国市场深耕为主，欧洲市场扩展为辅。

{'─'*70}
第三章  用户价值分析 (RFM)
{'─'*70}

3.1 RFM 指标概览
  指标       均值        中位数       含义
  Recency     {rfm['Recency'].mean():.0f}天      {rfm['Recency'].median():.0f}天      距上次购买天数
  Frequency  {rfm['Frequency'].mean():.1f}次      {rfm['Frequency'].median():.0f}次        购买次数
  Monetary   £{rfm['Monetary'].mean():,.0f}    £{rfm['Monetary'].median():,.0f}    累计消费

3.2 RFM 规则分层结果
"""

# Add RFM segmentation table
for idx, row in rfm_summary.iterrows():
    report += f"  {idx:<20s}: {int(row['Count']):>5,}人 ({row['Pct']:>6s}), 均R={row['AvgR']:.0f}天, 均F={row['AvgF']:.1f}次, 均M=£{row['AvgM']:,.0f}\n"

report += f"""
3.3 核心发现
  - 高价值冠军用户 ({rfm_summary.loc['高价值冠军用户','Count'] if '高价值冠军用户' in rfm_summary.index else 0}人, {rfm_summary.loc['高价值冠军用户','Pct'] if '高价值冠军用户' in rfm_summary.index else 0})
    贡献了 {rfm[rfm['RFM_Segment']=='Champions']['Monetary'].sum()/total_revenue*100:.1f}% 的总营收
  - 流失用户 + 流失风险用户共 {rfm[rfm['RFM_Segment'].isin(['AtRisk','Lost'])].shape[0]:,} 人
    ({rfm[rfm['RFM_Segment'].isin(['AtRisk','Lost'])].shape[0]/len(rfm)*100:.1f}%)，这些用户历史有价值但正在流失
  - 潜力用户仅占 5.4%，说明新客转化不足

3.4 KMeans 聚类 (K=5)
  - Cluster 0: 高频高消费忠诚用户 (4.8%)
  - Cluster 1: 低频低消费，已流失 (24.1%)
  - Cluster 2: 超高价值核心用户 (0.2%)
  - Cluster 3: 中等活跃普通用户 (70.8%)
  - Cluster 4: 超高价值极端用户 (0.1%)

{'─'*70}
第四章  留存分析 (Cohort)
{'─'*70}

4.1 Cohort 留存矩阵
  平均次月留存率: {avg_month1:.1f}%
  平均3月留存率:  {avg_month3:.1f}%
  12月留存率:     {cohort_clean.loc['2009-12', '12'] if '2009-12' in cohort_clean.index and '12' in cohort_clean.columns else 'N/A'}

4.2 留存洞察
  - 次月留存率波动于 10.8%~35.3%，平均{avg_month1:.1f}%，整体偏低
  - 2009年12月Cohort留存表现最好(首月35.3%, 12月仍达24.8%)
  - 假期季(11-12月)新客的次月留存明显低于其他月份
  - 3个月后留存率趋于稳定在 23-29% 区间

4.3 关键问题
  - 次月留存率仅约 20%，意味着 80% 的新用户在首月后不再回购
  - 这是最核心的运营问题—需要建立新用户激活和首月回访机制

{'─'*70}
第五章  用户生命周期漏斗
{'─'*70}

5.1 成长路径漏斗
"""

for stage, cnt in funnel.items():
    rate = cnt / funnel['首次购买'] * 100
    bar = chr(0x2588) * int(rate / 2)
    report += f"  {stage:<15s}: {cnt:>5,}人 ({rate:>5.1f}%) {bar}\n"

report += f"""
5.2 漏斗分析
  - 首次购买 → 二次购买: 转化率 67.1% (流失 32.9%)
  - 二次购买 → 多次购买(3+): 转化率 72.0%
  - 多次购买 → 高频购买(6+): 转化率 45.2%
  - 高频购买 → 高价值(Top20%): 转化率 21.2%

  最大流失节点: 首次购买后 (32.9% 流失)

{'─'*70}
第六章  关键问题总结
{'─'*70}

问题1: 新用户首月流失严重
  现象: 32.9%用户仅购买一次，次月留存率均值仅20.5%
  影响: 获客成本浪费，用户基数难以增长
  优先级: [高]

问题2: 高价值用户过于集中
  现象: Top 20%用户贡献 {top20_contribution:.1f}% 营收，10.6%冠军用户贡献50.7%营收
  影响: 用户流失风险高度集中，业务脆弱
  优先级: [高]

问题3: 市场过度集中
  现象: 英国用户占92%，其他市场占比极低
  影响: 单一市场风险，增长天花板明显
  优先级: [中]

问题4: 假期新客质量偏低
  现象: 11-12月新增用户的次月留存率显著低于其他月份
  影响: 旺季获取的用户粘性不足
  优先级: [中]

问题5: 潜力用户转化不足
  现象: 新用户中仅5.4%转化为潜力用户
  影响: 用户价值成长路径受阻
  优先级: [中]

{'─'*70}
第七章  运营建议
{'─'*70}

建议1: 建立新用户首月激活计划 [对应问题1]
  - 首次购买后 24h 内发送感谢邮件 + 关联商品推荐
  - 第7天发送"你可能喜欢"个性化推荐
  - 第21天发送限时优惠券 (满£50减£10)
  - 目标: 将次月留存率从 {avg_month1:.0f}% 提升至 30%+

建议2: 高价值用户VIP体系 [对应问题2]
  - 建立分层权益: 银卡(£3K+)/金卡(£5K+)/铂金(£10K+)
  - 专属客服、优先发货、生日礼遇
  - 定期专属新品预览
  - 目标: 高价值用户年流失率控制在10%以内

建议3: 流失风险用户召回 [对应问题2]
  - 识别标准: 过去高消费但 90+ 天未购买
  - 触发机制: 第90天自动发送"我们想念你"+ 大额优惠券
  - 第120天电话/邮件深度回访
  - 当前风险用户: {rfm[rfm['RFM_Segment']=='AtRisk'].shape[0]} 人，潜在挽回价值可观

建议4: 假期新客差异化管理 [对应问题4]
  - 12月新客在1月收到特别"新年回访"优惠
  - 建立假期新客专属产品推荐(与普通新客不同)
  - 目标: 假期新客次月留存率提升至 25%+

建议5: 欧洲市场拓展 [对应问题3]
  - 优先拓展德国、法国市场(已有基础)
  - 本地化: 德语/法语产品描述和客服
  - 区域定价策略

建议6: 复购激励计划 [对应问题5]
  - 第2单: 免运费
  - 第5单: 5%折扣
  - 第10单: 升级VIP
  - 每单积累积分可兑换

{'─'*70}
第八章  文件清单
{'─'*70}

数据文件:
  - cleaned_transactions.csv    : 清洗后交易数据 (407,664条)
  - user_metrics.csv           : 用户指标表 (4,312人)
  - rfm_analysis.csv           : RFM分析结果
  - monthly_user_stats.csv     : 月度用户统计
  - cohort_retention.csv       : Cohort留存矩阵
  - user_lifecycle_stage.csv   : 用户生命周期阶段

Dashboard:
  - dashboard_page1_overview.html     : 用户概览
  - dashboard_page2_rfm.html          : 用户价值分析
  - dashboard_page3_lifecycle.html    : 生命周期&留存
  - dashboard_page4_strategy.html     : 运营策略
  - dashboard_kpi_summary.html        : KPI概要面板

分析脚本:
  - phase1_data_cleaning.py     : 数据清洗
  - phase2_5_analysis.py        : 用户画像/RFM/留存分析
  - phase6_7_dashboard.py       : Dashboard生成
  - phase8_report.py            : 本报告

{'='*70}
                         报告结束
{'='*70}
"""

# ============================================================================
# 输出报告
# ============================================================================
# Save as Markdown
with open("outputs/analysis_report.md", "w", encoding="utf-8") as f:
    f.write(report)

# Save as TXT
with open("outputs/analysis_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

# Print to console (replace GBP symbol for Windows console)
print(report.replace('£', 'GBP '))

print("\n报告已保存:")
print("  + analysis_report.md")
print("  + analysis_report.txt")
