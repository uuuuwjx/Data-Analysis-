# -*- coding: utf-8 -*-
"""
================================================================================
电商用户运营分析 — Phase 6-7: 漏斗分析 + Dashboard
================================================================================
使用 Plotly 生成交互式 HTML Dashboard
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 加载数据
# ============================================================================
df = pd.read_csv("data/cleaned_transactions.csv", parse_dates=['InvoiceDate'])
user = pd.read_csv("data/user_metrics.csv", parse_dates=['FirstPurchase', 'LastPurchase'])
rfm = pd.read_csv("data/rfm_analysis.csv")
monthly = pd.read_csv("data/monthly_user_stats.csv")
cohort = pd.read_csv("data/cohort_retention.csv", index_col=0)
lifecycle = pd.read_csv("data/user_lifecycle_stage.csv")

reference_date = df['InvoiceDate'].max()
print(f"数据加载完成: {len(df):,} 交易, {len(user):,} 用户")

# ============================================================================
# PHASE 6: 用户行为漏斗分析
# ============================================================================
print("\n" + "=" * 50)
print("Phase 6: 用户行为漏斗分析")
print("=" * 50)

# 基于交易数据构建用户生命周期漏斗
funnel_data = {
    '首次购买': len(user),
    '二次购买': len(user[user['TotalOrders'] >= 2]),
    '3-5次购买': len(user[user['TotalOrders'] >= 3]),
    '6次以上购买': len(user[user['TotalOrders'] >= 6]),
    '10次以上购买': len(user[user['TotalOrders'] >= 10]),
    '高价值 (Top20%)': len(user[user['TotalRevenue'] >= user['TotalRevenue'].quantile(0.8)]),
}

print("\n[用户生命周期漏斗]")
for stage, cnt in funnel_data.items():
    rate = cnt / funnel_data['首次购买'] * 100
    print(f"  {stage:<20s}: {cnt:>5,} 人 ({rate:5.1f}%)")

# ============================================================================
# PHASE 7: Dashboard (Plotly HTML)
# ============================================================================
print("\n" + "=" * 50)
print("Phase 7: Dashboard 生成")
print("=" * 50)

# 配色
colors = px.colors.qualitative.Set2
template = 'plotly_white'

# ============================================================
# Page 1: 用户概览
# ============================================================
print("\n[页面1] 用户概览")

fig1 = make_subplots(
    rows=2, cols=3,
    subplot_titles=(
        '月度活跃用户 (MAU) & 新增用户',
        '月度营收趋势',
        '用户地区分布 Top 10',
        '用户消费金额分布',
        '用户购买频次分布',
        '月度订单数趋势'
    ),
    specs=[
        [{"secondary_y": True}, {}, {}],
        [{}, {}, {}],
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

# 1.1 MAU & New Users
fig1.add_trace(
    go.Bar(x=monthly['YearMonth'], y=monthly['NewUsers'], name='新增用户',
           marker_color=colors[0], opacity=0.8),
    row=1, col=1, secondary_y=False
)
fig1.add_trace(
    go.Scatter(x=monthly['YearMonth'], y=monthly['MAU'], name='MAU',
               mode='lines+markers', line=dict(color=colors[1], width=2.5)),
    row=1, col=1, secondary_y=True
)

# 1.2 月度营收趋势
fig1.add_trace(
    go.Bar(x=monthly['YearMonth'], y=monthly['TotalRevenue'], name='营收',
           marker_color=colors[3], opacity=0.8),
    row=1, col=2
)

# 1.3 用户地区分布
country_top10 = df.groupby('Country')['Customer ID'].nunique().sort_values(ascending=False).head(10)
fig1.add_trace(
    go.Bar(x=country_top10.values, y=country_top10.index, orientation='h',
           marker_color=colors[4], name='用户数'),
    row=1, col=3
)

# 1.4 消费金额分布
rev_bins = ['<100', '100-500', '500-1K', '1K-3K', '3K-5K', '5K-10K', '>10K']
user['RevSegment'] = pd.cut(user['TotalRevenue'],
                            bins=[0,100,500,1000,3000,5000,10000,float('inf')],
                            labels=rev_bins)
rev_dist = user['RevSegment'].value_counts().sort_index()
fig1.add_trace(
    go.Bar(x=rev_dist.index.tolist(), y=rev_dist.values,
           marker_color=colors[5], name='用户数'),
    row=2, col=1
)

# 1.5 购买频次分布
freq_labels = ['1次', '2次', '3-5次', '6-10次', '11-20次', '21-50次', '50+次']
user['FreqSegment'] = pd.cut(user['TotalOrders'],
                             bins=[0,1,2,5,10,20,50,float('inf')],
                             labels=freq_labels)
freq_dist = user['FreqSegment'].value_counts().sort_index()
fig1.add_trace(
    go.Bar(x=freq_dist.index.tolist(), y=freq_dist.values,
           marker_color=colors[6], name='用户数'),
    row=2, col=2
)

# 1.6 月度订单数
fig1.add_trace(
    go.Scatter(x=monthly['YearMonth'], y=monthly['TotalOrders'], name='订单数',
               mode='lines+markers', line=dict(color=colors[2], width=2.5)),
    row=2, col=3
)

fig1.update_layout(
    title_text='用户运营分析 - 页面1: 用户概览',
    height=900, width=1400,
    template=template,
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.02)
)
fig1.update_xaxes(tickangle=45)
fig1.write_html("outputs/dashboard_page1_overview.html")
print("  + dashboard_page1_overview.html")

# ============================================================
# Page 2: 用户价值分析 (RFM)
# ============================================================
print("[页面2] 用户价值分析")

fig2 = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'RFM 用户分群分布',
        'RFM 各群营收贡献',
        'RFM 散点图 (Recency vs Monetary)',
        'KMeans 聚类分布 (3D 投影)'
    ),
    specs=[
        [{}, {'type': 'pie'}],
        [{}, {'type': 'xy'}],
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

rfm_labels_cn = {
    'Champions': '高价值冠军', 'Loyal': '忠诚用户', 'Potential': '潜力用户',
    'AtRisk': '流失风险', 'Lost': '流失用户', 'New': '新用户', 'Average': '普通用户'
}
rfm['SegmentCN'] = rfm['RFM_Segment'].map(rfm_labels_cn)

# 2.1 RFM 分群分布
seg_counts = rfm['SegmentCN'].value_counts()
fig2.add_trace(
    go.Bar(x=seg_counts.index.tolist(), y=seg_counts.values,
           marker_color=colors[:len(seg_counts)], name='用户数'),
    row=1, col=1
)

# 2.2 RFM 营收贡献饼图
seg_rev = rfm.groupby('SegmentCN')['Monetary'].sum().sort_values(ascending=False)
fig2.add_trace(
    go.Pie(labels=seg_rev.index.tolist(), values=seg_rev.values,
           marker_colors=px.colors.qualitative.Pastel, hole=0.4, name='营收占比'),
    row=1, col=2
)

# 2.3 Recency vs Monetary 散点图
sample_rfm = rfm.sample(min(2000, len(rfm)), random_state=42)
fig2.add_trace(
    go.Scatter(x=sample_rfm['Recency'], y=sample_rfm['Monetary'],
               mode='markers',
               marker=dict(
                   size=sample_rfm['Frequency']*2,
                   color=sample_rfm['Recency'],
                   colorscale='Viridis_r',
                   showscale=True,
                   colorbar=dict(title='Recency(天)')
               ),
               text=[f"F:{f} M:{m:.0f}" for f, m in zip(sample_rfm['Frequency'], sample_rfm['Monetary'])],
               name='用户'),
    row=2, col=1
)
fig2.update_xaxes(title_text='Recency (天)', row=2, col=1)
fig2.update_yaxes(title_text='Monetary', row=2, col=1)

# 2.4 KMeans 聚类 (用 R_Score, F_Score, M_Score 做 3D)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
rfm_3d = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

fig2.add_trace(
    go.Scatter(
        x=rfm_3d[:, 0], y=rfm_3d[:, 1],
        mode='markers',
        marker=dict(
            color=rfm['Cluster'].astype(int),
            colorscale='Viridis',
            size=5,
            showscale=True,
            colorbar=dict(title='Cluster')
        ),
        name='聚类'
    ),
    row=2, col=2
)
fig2.update_xaxes(title_text='Recency (标准化)', row=2, col=2)
fig2.update_yaxes(title_text='Frequency (标准化)', row=2, col=2)

fig2.update_layout(
    title_text='用户运营分析 - 页面2: 用户价值分析 (RFM)',
    height=900, width=1400,
    template=template,
    showlegend=True
)
fig2.write_html("outputs/dashboard_page2_rfm.html")
print("  + dashboard_page2_rfm.html")

# ============================================================
# Page 3: 用户生命周期
# ============================================================
print("[页面3] 用户生命周期")

fig3 = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        '用户生命周期阶段分布',
        'Cohort 留存热力图',
        '月度复购率趋势',
        '新增 vs 流失用户趋势'
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

# 3.1 生命周期阶段
stage_order = ['New', 'Active', 'VIP', 'Growing', 'Normal', 'AtRisk']
stage_cn = {'New':'新用户','Active':'活跃','VIP':'高价值','Growing':'成长','Normal':'普通','AtRisk':'流失风险'}
lifecycle['StageCN'] = lifecycle['LifeStage'].map(stage_cn)
stage_counts = lifecycle['StageCN'].value_counts()
ordered_counts = [stage_counts.get(stage_cn[s], 0) for s in stage_order]
ordered_names = [stage_cn[s] for s in stage_order]
fig3.add_trace(
    go.Bar(x=ordered_names, y=ordered_counts,
           marker_color=px.colors.qualitative.Set2[:6], name='用户数'),
    row=1, col=1
)

# 3.2 Cohort 热力图
cohort_clean = cohort.fillna(0)
fig3.add_trace(
    go.Heatmap(
        z=cohort_clean.values,
        x=cohort_clean.columns.tolist(),
        y=cohort_clean.index.tolist(),
        colorscale='RdYlGn',
        zmin=0, zmax=50,
        text=cohort_clean.round(1).values,
        texttemplate='%{text:.0f}',
        textfont={"size": 9},
        colorbar=dict(title='留存率%'),
        name='留存率'
    ),
    row=1, col=2
)

# 3.3 月度复购率
monthly_orders = df.groupby(['YearMonth', 'Customer ID'])['Invoice'].nunique().reset_index()
monthly_users = monthly_orders.groupby('YearMonth')['Customer ID'].count().reset_index()
monthly_users.columns = ['YearMonth', 'TotalUsers']
monthly_repeat = monthly_orders[monthly_orders['Invoice'] >= 2].groupby('YearMonth')['Customer ID'].count().reset_index()
monthly_repeat.columns = ['YearMonth', 'RepeatUsers']
repeat_rate_df = monthly_users.merge(monthly_repeat, on='YearMonth', how='left')
repeat_rate_df['RepeatRate'] = (repeat_rate_df['RepeatUsers'] / repeat_rate_df['TotalUsers'] * 100).fillna(0)

fig3.add_trace(
    go.Scatter(x=repeat_rate_df['YearMonth'], y=repeat_rate_df['RepeatRate'],
               mode='lines+markers', line=dict(color=colors[1], width=2.5),
               fill='tozeroy', fillcolor='rgba(102,194,165,0.3)', name='复购率%'),
    row=2, col=1
)
fig3.update_yaxes(title_text='复购率 (%)', row=2, col=1)

# 3.4 新增用户趋势
fig3.add_trace(
    go.Scatter(x=monthly['YearMonth'], y=monthly['NewUsers'],
               mode='lines+markers', line=dict(color=colors[0], width=2.5),
               name='新增用户'),
    row=2, col=2
)

fig3.update_layout(
    title_text='用户运营分析 - 页面3: 用户生命周期 & 留存',
    height=900, width=1400,
    template=template,
    showlegend=True
)
fig3.update_xaxes(tickangle=45)
fig3.write_html("outputs/dashboard_page3_lifecycle.html")
print("  + dashboard_page3_lifecycle.html")

# ============================================================
# Page 4: 运营策略分析
# ============================================================
print("[页面4] 运营策略")

fig4 = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        '用户漏斗: 成长路径',
        '高价值用户贡献分析',
        '流失风险用户画像',
        '按国家的客单价分析 Top 10'
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

# 4.1 用户漏斗
funnel_stages = list(funnel_data.keys())
funnel_values = list(funnel_data.values())
fig4.add_trace(
    go.Funnel(
        y=funnel_stages, x=funnel_values,
        textinfo="value+percent initial",
        marker=dict(color=px.colors.sequential.Blugrn_r[:len(funnel_stages)]),
        name='用户数'
    ),
    row=1, col=1
)

# 4.2 高价值用户贡献: Pareto
user_sorted = user.sort_values('TotalRevenue', ascending=False)
user_sorted['CumPct'] = user_sorted['TotalRevenue'].cumsum() / user_sorted['TotalRevenue'].sum() * 100
user_sorted['UserPct'] = np.arange(1, len(user_sorted)+1) / len(user_sorted) * 100

fig4.add_trace(
    go.Scatter(x=user_sorted['UserPct'], y=user_sorted['CumPct'],
               mode='lines', line=dict(color=colors[0], width=2.5),
               fill='tozeroy', fillcolor='rgba(102,194,165,0.2)', name='累计营收%'),
    row=1, col=2
)
fig4.add_trace(
    go.Scatter(x=[0, 20, 20], y=[73.5, 73.5, 0],
               mode='lines', line=dict(color='red', dash='dash', width=1),
               name='Top 20% = 73.5%'),
    row=1, col=2
)
fig4.update_xaxes(title_text='用户累计 %', row=1, col=2)
fig4.update_yaxes(title_text='营收累计 %', row=1, col=2)

# 4.3 流失风险用户画像
at_risk = rfm[rfm['RFM_Segment'].isin(['AtRisk', 'Lost'])]
safe = rfm[~rfm['RFM_Segment'].isin(['AtRisk', 'Lost'])]
risk_profile = pd.DataFrame({
    '指标': ['平均Recency(天)', '平均Frequency(次)', '平均Monetary'],
    '流失风险/已流失': [at_risk['Recency'].mean(), at_risk['Frequency'].mean(), at_risk['Monetary'].mean()],
    '正常用户': [safe['Recency'].mean(), safe['Frequency'].mean(), safe['Monetary'].mean()]
})

fig4.add_trace(
    go.Bar(x=risk_profile['指标'], y=risk_profile['流失风险/已流失'], name='流失风险/已流失',
           marker_color=colors[3]),
    row=2, col=1
)
fig4.add_trace(
    go.Bar(x=risk_profile['指标'], y=risk_profile['正常用户'], name='正常用户',
           marker_color=colors[1]),
    row=2, col=1
)

# 4.4 按国家的客单价
country_avg = df.groupby('Country').agg(
    Users = ('Customer ID', 'nunique'),
    AvgOrder = ('Revenue', 'mean')
).reset_index()
top10_country = country_avg.nlargest(10, 'Users')
fig4.add_trace(
    go.Bar(x=top10_country['Country'], y=top10_country['AvgOrder'],
           marker_color=colors[5], name='客单价'),
    row=2, col=2
)
fig4.update_yaxes(title_text='平均客单价', row=2, col=2)

fig4.update_layout(
    title_text='用户运营分析 - 页面4: 运营策略分析',
    height=900, width=1400,
    template=template,
    showlegend=True
)
fig4.update_xaxes(tickangle=45)
fig4.write_html("outputs/dashboard_page4_strategy.html")
print("  + dashboard_page4_strategy.html")

# ============================================================
# 综合 Dashboard (单页面，提取关键图表)
# ============================================================
print("[综合] 关键KPI概要")

fig_kpi = make_subplots(
    rows=2, cols=4,
    subplot_titles=(
        '总用户数', '月均活跃用户', '总营收', '平均客单价',
        '用户地区分布', '用户价值分群', 'Cohort留存热力图', '用户漏斗'
    ),
    specs=[
        [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
        [{'type': 'xy'}, {'type': 'domain'}, {'type': 'xy'}, {'type': 'xy'}],
    ],
    vertical_spacing=0.15,
    horizontal_spacing=0.05
)

# KPI indicators
fig_kpi.add_trace(go.Indicator(mode='number+delta', value=len(user),
                                title={"text": "总用户数"},
                                delta={'reference': len(user), 'relative': False},
                                number={'font': {'size': 36}}), row=1, col=1)
fig_kpi.add_trace(go.Indicator(mode='number', value=monthly['MAU'].mean(),
                                title={"text": "月均活跃用户"},
                                number={'font': {'size': 36}}), row=1, col=2)
fig_kpi.add_trace(go.Indicator(mode='number', value=df['Revenue'].sum()/1e6,
                                title={"text": "总营收 (百万)"},
                                number={'prefix': '', 'suffix': 'M', 'font': {'size': 36},
                                        'valueformat': '.2f'}), row=1, col=3)
fig_kpi.add_trace(go.Indicator(mode='number', value=df['Revenue'].mean(),
                                title={"text": "平均客单价"},
                                number={'prefix': '', 'font': {'size': 36},
                                        'valueformat': '.0f'}), row=1, col=4)

# Charts
# 地区
fig_kpi.add_trace(
    go.Bar(x=country_top10.values, y=country_top10.index, orientation='h',
           marker_color=colors[4], showlegend=False),
    row=2, col=1
)

# 价值分群
seg_counts = rfm['SegmentCN'].value_counts()
fig_kpi.add_trace(
    go.Pie(labels=seg_counts.index, values=seg_counts.values, hole=0.5,
           showlegend=False),
    row=2, col=2
)

# Cohort
fig_kpi.add_trace(
    go.Heatmap(z=cohort_clean.values, x=cohort_clean.columns.tolist(),
               y=cohort_clean.index.tolist(), colorscale='RdYlGn',
               zmin=0, zmax=50, showscale=False),
    row=2, col=3
)

# Funnel (as bar chart for subplot compat)
fig_kpi.add_trace(
    go.Bar(x=funnel_values[::-1], y=funnel_stages[::-1], orientation='h',
           marker_color=px.colors.sequential.Blugrn_r[:len(funnel_stages)][::-1],
           showlegend=False, text=funnel_values[::-1], textposition='outside'),
    row=2, col=4
)

fig_kpi.update_layout(
    title_text='<b>用户运营分析 - KPI 概要面板</b>',
    height=900, width=1600,
    template=template
)
fig_kpi.write_html("outputs/dashboard_kpi_summary.html")
print("  + dashboard_kpi_summary.html")

print("\n===== Phase 6-7 完成! =====")
print("生成文件:")
print("  - dashboard_page1_overview.html (用户概览)")
print("  - dashboard_page2_rfm.html (RFM价值分析)")
print("  - dashboard_page3_lifecycle.html (生命周期&留存)")
print("  - dashboard_page4_strategy.html (运营策略)")
print("  - dashboard_kpi_summary.html (KPI概要)")
