# -*- coding: utf-8 -*-
"""
生成全中文交互式 Dashboard — 独立图表 + CSS Grid 布局
每个图表独立渲染，清晰排列，全中文标签
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# Load data
# ====================================================================
df      = pd.read_csv("data/cleaned_transactions.csv", parse_dates=['InvoiceDate'])
user    = pd.read_csv("data/user_metrics.csv", parse_dates=['FirstPurchase', 'LastPurchase'])
rfm     = pd.read_csv("data/rfm_analysis.csv")
monthly = pd.read_csv("data/monthly_user_stats.csv")
cohort  = pd.read_csv("data/cohort_retention.csv", index_col=0)

reference_date = pd.Timestamp('2010-12-09 20:01:00')
total_revenue = df['Revenue'].sum()

# RFM Chinese labels
rfm_cn = {
    'Champions': '高价值冠军', 'Loyal': '忠诚用户', 'Potential': '潜力用户',
    'AtRisk': '流失风险', 'Lost': '已流失', 'New': '新用户', 'Average': '普通用户'
}
rfm['SegmentCN'] = rfm['RFM_Segment'].map(rfm_cn)

# Colors
C = px.colors.qualitative.Set2
TEMPLATE = 'plotly_white'
PAPER_BG = 'rgba(0,0,0,0)'
PLOT_BG  = 'rgba(0,0,0,0)'
CHART_HEIGHT = 420

print("Data loaded. Building individual charts...")

# ====================================================================
# JSON encoder
# ====================================================================
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj) or np.isinf(obj): return None
            return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, pd.Timestamp): return str(obj)
        return super().default(obj)

def fig_json(fig):
    """Return dict — serialized once in template. Sanitize NaN/Inf to null."""
    def sanitize(obj):
        if isinstance(obj, (float, np.floating)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, np.ndarray)):
            return [sanitize(v) for v in obj]
        return obj
    return sanitize(fig.to_dict())

def base_layout(fig, title=''):
    fig.update_layout(
        template=TEMPLATE, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        title=dict(text=title, font=dict(size=14, color='#2c3e50'), x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation='h', y=1.15, x=0, font=dict(size=11)),
        hovermode='closest',
        height=CHART_HEIGHT,
    )
    return fig

# ====================================================================
# PAGE 1: 用户概览 — 6 independent charts
# ====================================================================
print("  Page 1: User Overview...")

# Chart 1.1: MAU & 新增用户趋势
ch_1_1 = go.Figure()
ch_1_1.add_trace(go.Bar(x=monthly['YearMonth'], y=monthly['NewUsers'], name='新增用户',
                         marker_color=C[0], marker_line=dict(width=0)))
ch_1_1.add_trace(go.Scatter(x=monthly['YearMonth'], y=monthly['MAU'], name='月活跃用户(MAU)',
                            mode='lines+markers', line=dict(color=C[1], width=3),
                            marker=dict(size=6), yaxis='y2'))
ch_1_1.update_layout(
    yaxis=dict(title='新增用户数', gridcolor='#f0f0f0'),
    yaxis2=dict(title='MAU', overlaying='y', side='right', gridcolor='rgba(0,0,0,0)'),
    xaxis=dict(tickangle=45, gridcolor='#f0f0f0'),
    legend=dict(x=0.01, y=1.1),
)
base_layout(ch_1_1, '月度活跃用户与新增用户趋势')

# Chart 1.2: 月度营收
ch_1_2 = go.Figure()
ch_1_2.add_trace(go.Bar(x=monthly['YearMonth'], y=monthly['TotalRevenue'], name='月度营收',
                         marker_color=C[3], marker_line=dict(width=0),
                         text=[f'£{v/1000:.0f}K' for v in monthly['TotalRevenue']],
                         textposition='outside', textfont=dict(size=10)))
ch_1_2.add_trace(go.Scatter(x=monthly['YearMonth'], y=monthly['TotalRevenue'], name='趋势线',
                            mode='lines', line=dict(color='#2c3e50', width=1.5, dash='dot')))
base_layout(ch_1_2, '月度营收趋势')
ch_1_2.update_layout(xaxis=dict(tickangle=45, gridcolor='#f0f0f0'), yaxis=dict(gridcolor='#f0f0f0'))

# Chart 1.3: 用户地区分布 (Top 10)
ctop = df.groupby('Country')['Customer ID'].nunique().sort_values(ascending=False).head(10)
ch_1_3 = go.Figure()
ch_1_3.add_trace(go.Bar(x=ctop.values, y=ctop.index, orientation='h',
                         marker=dict(color=C[4], line=dict(width=0)),
                         text=ctop.values, textposition='outside', textfont=dict(size=10)))
base_layout(ch_1_3, '用户地区分布 (Top 10)')
ch_1_3.update_layout(xaxis=dict(title='用户数', gridcolor='#f0f0f0'), yaxis=dict(gridcolor='#f0f0f0'))

# Chart 1.4: 用户消费金额分布
rev_labels = ['<100', '100-500', '500-1K', '1K-3K', '3K-5K', '5K-10K', '>10K']
user['RevSeg'] = pd.cut(user['TotalRevenue'], bins=[0,100,500,1000,3000,5000,10000,float('inf')], labels=rev_labels)
rd = user['RevSeg'].value_counts().sort_index()
ch_1_4 = go.Figure()
ch_1_4.add_trace(go.Bar(x=rd.index.tolist(), y=rd.values, name='用户数',
                         marker=dict(color=C[5], line=dict(width=0)),
                         text=rd.values, textposition='outside', textfont=dict(size=10)))
base_layout(ch_1_4, '用户累计消费金额分布 (£)')
ch_1_4.update_layout(xaxis=dict(title='消费金额区间 (£)', gridcolor='#f0f0f0'), yaxis=dict(title='用户数', gridcolor='#f0f0f0'))

# Chart 1.5: 购买频次分布
freq_lbl = ['1次', '2次', '3-5次', '6-10次', '11-20次', '21-50次', '50+次']
user['FreqSeg'] = pd.cut(user['TotalOrders'], bins=[0,1,2,5,10,20,50,float('inf')], labels=freq_lbl)
fd = user['FreqSeg'].value_counts().sort_index()
ch_1_5 = go.Figure()
ch_1_5.add_trace(go.Bar(x=fd.index.tolist(), y=fd.values, name='用户数',
                         marker=dict(color=C[6], line=dict(width=0)),
                         text=fd.values, textposition='outside', textfont=dict(size=10)))
base_layout(ch_1_5, '用户购买频次分布')
ch_1_5.update_layout(xaxis=dict(title='购买次数', gridcolor='#f0f0f0'), yaxis=dict(gridcolor='#f0f0f0'))

# Chart 1.6: 月度平均客单价 & 商品数
monthly_avg = df.groupby('YearMonth').agg(平均客单价=('Revenue','mean'), 平均商品数=('Quantity','mean')).reset_index()
ch_1_6 = go.Figure()
ch_1_6.add_trace(go.Scatter(x=monthly_avg['YearMonth'], y=monthly_avg['平均客单价'], name='平均客单价(£)',
                            mode='lines+markers', line=dict(color=C[1], width=2.5), marker=dict(size=6)))
ch_1_6.add_trace(go.Scatter(x=monthly_avg['YearMonth'], y=monthly_avg['平均商品数'], name='平均商品数',
                            mode='lines+markers', line=dict(color=C[0], width=2.5), marker=dict(size=6),
                            yaxis='y2'))
ch_1_6.update_layout(
    yaxis=dict(title='客单价 (£)', gridcolor='#f0f0f0'),
    yaxis2=dict(title='商品数', overlaying='y', side='right', gridcolor='rgba(0,0,0,0)'),
    xaxis=dict(tickangle=45, gridcolor='#f0f0f0'),
)
base_layout(ch_1_6, '月度平均客单价与商品数')

page1_charts = {
    'ch_1_1': fig_json(ch_1_1), 'ch_1_2': fig_json(ch_1_2), 'ch_1_3': fig_json(ch_1_3),
    'ch_1_4': fig_json(ch_1_4), 'ch_1_5': fig_json(ch_1_5), 'ch_1_6': fig_json(ch_1_6),
}

# ====================================================================
# PAGE 2: RFM 用户价值分析 — 6 charts
# ====================================================================
print("  Page 2: RFM Analysis...")

seg_cnt = rfm['SegmentCN'].value_counts()
# keep consistent order
seg_order_cn = ['高价值冠军','忠诚用户','潜力用户','流失风险','已流失','新用户','普通用户']
seg_cnt = seg_cnt.reindex([s for s in seg_order_cn if s in seg_cnt.index])

# Chart 2.1: RFM 分群用户数
ch_2_1 = go.Figure()
ch_2_1.add_trace(go.Bar(x=seg_cnt.index.tolist(), y=seg_cnt.values, name='用户数',
                         marker=dict(color=px.colors.qualitative.Set2[:len(seg_cnt)], line=dict(width=0)),
                         text=seg_cnt.values, textposition='outside', textfont=dict(size=10)))
base_layout(ch_2_1, 'RFM 用户分群分布')
ch_2_1.update_layout(xaxis=dict(tickangle=30, gridcolor='#f0f0f0'), yaxis=dict(title='用户数', gridcolor='#f0f0f0'))

# Chart 2.2: 各分群营收贡献
seg_rev = rfm.groupby('SegmentCN')['Monetary'].sum().reindex([s for s in seg_order_cn if s in seg_cnt.index]).fillna(0)
ch_2_2 = go.Figure()
ch_2_2.add_trace(go.Pie(labels=seg_rev.index.tolist(), values=seg_rev.values, hole=0.55,
                         marker_colors=px.colors.qualitative.Pastel,
                         textinfo='label+percent', textfont=dict(size=10),
                         hovertemplate='%{label}: £%{value:,.0f} (%{percent})<extra></extra>'))
base_layout(ch_2_2, '各分群营收占比')
ch_2_2.update_layout(height=CHART_HEIGHT, margin=dict(l=10,r=10,t=40,b=10))

# Chart 2.3: Recency vs Monetary 气泡图
sample_rfm = rfm.sample(min(1200, len(rfm)), random_state=42)
ch_2_3 = go.Figure()
ch_2_3.add_trace(go.Scatter(
    x=sample_rfm['Recency'], y=sample_rfm['Monetary'], mode='markers',
    marker=dict(size=np.clip(sample_rfm['Frequency']*2.5, 4, 35),
                color=sample_rfm['Recency'], colorscale='RdYlGn_r', showscale=True,
                colorbar=dict(title='距上次购买(天)', len=0.7)),
    hovertemplate='R: %{x}天<br>M: £%{y:,.0f}<br>F: %{marker.size:.0f}次<extra></extra>',
    name='用户'))
base_layout(ch_2_3, 'Recency vs Monetary (气泡大小=购买频次)')
ch_2_3.update_layout(xaxis=dict(title='Recency (天)', gridcolor='#f0f0f0'),
                     yaxis=dict(title='Monetary (£)', gridcolor='#f0f0f0'))

# Chart 2.4: KMeans 聚类结果
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
X = StandardScaler().fit_transform(rfm[['Recency','Frequency','Monetary']])
km = KMeans(n_clusters=5, random_state=42, n_init=10)
rfm['Cluster'] = km.fit_predict(X)
cluster_profile = rfm.groupby('Cluster').agg(
    人数=('CustomerID','count'), 平均R=('Recency','mean'), 平均F=('Frequency','mean'), 平均M=('Monetary','mean')
).round(1)
cluster_profile['标签'] = ['低频新客','高价值核心','已流失','活跃中等','低频低消'][:len(cluster_profile)]

ch_2_4 = go.Figure()
ch_2_4.add_trace(go.Scatter(x=X[:,0], y=X[:,1], mode='markers',
                            marker=dict(color=rfm['Cluster'].astype(int), colorscale='Viridis',
                                        size=5, showscale=True, colorbar=dict(title='聚类', len=0.7)),
                            hovertemplate='R: %{x:.1f}<br>F: %{y:.1f}<extra></extra>',
                            name='用户'))
base_layout(ch_2_4, 'KMeans 聚类分布 (R vs F, 标准化后)')
ch_2_4.update_layout(xaxis=dict(title='Recency (标准化)', gridcolor='#f0f0f0'),
                     yaxis=dict(title='Frequency (标准化)', gridcolor='#f0f0f0'))

# Chart 2.5: R/F/M 评分分布
ch_2_5 = go.Figure()
for i, (score_name, color) in enumerate([('R_Score',C[3]), ('F_Score',C[1]), ('M_Score',C[0])]):
    label_cn = {'R_Score':'R值 (最近购买)','F_Score':'F值 (购买频次)','M_Score':'M值 (消费金额)'}[score_name]
    sc = rfm[score_name].value_counts().sort_index()
    ch_2_5.add_trace(go.Bar(x=[f'{i}分' for i in sc.index], y=sc.values, name=label_cn,
                             marker=dict(color=color, opacity=0.8, line=dict(width=0))))
base_layout(ch_2_5, 'RFM 三项评分分布 (1-4分制)')
ch_2_5.update_layout(xaxis=dict(gridcolor='#f0f0f0'), yaxis=dict(title='用户数', gridcolor='#f0f0f0'),
                     barmode='group')

# Chart 2.6: 各分群平均指标对比
seg_profile = rfm.groupby('SegmentCN').agg(avgR=('Recency','mean'), avgF=('Frequency','mean'), avgM=('Monetary','mean')).round(1)
seg_profile = seg_profile.reindex([s for s in seg_order_cn if s in seg_profile.index])
ch_2_6 = go.Figure()
ch_2_6.add_trace(go.Bar(x=seg_profile.index, y=seg_profile['avgF'], name='平均购买次数(F)',
                         marker=dict(color=C[1], line=dict(width=0)),
                         text=seg_profile['avgF'].round(1), textposition='outside', textfont=dict(size=9)))
base_layout(ch_2_6, '各分群平均购买频次对比')
ch_2_6.update_layout(xaxis=dict(tickangle=30, gridcolor='#f0f0f0'), yaxis=dict(title='平均购买次数', gridcolor='#f0f0f0'))

page2_charts = {
    'ch_2_1': fig_json(ch_2_1), 'ch_2_2': fig_json(ch_2_2), 'ch_2_3': fig_json(ch_2_3),
    'ch_2_4': fig_json(ch_2_4), 'ch_2_5': fig_json(ch_2_5), 'ch_2_6': fig_json(ch_2_6),
}

# ====================================================================
# PAGE 3: 生命周期与留存 — 6 charts
# ====================================================================
print("  Page 3: Lifecycle & Retention...")

# Chart 3.1: 生命周期阶段分布
stage_order = ['New','Active','VIP','Growing','Normal','AtRisk']
stage_cn_map = {'New':'新用户','Active':'活跃用户','VIP':'高价值用户','Growing':'成长用户','Normal':'普通用户','AtRisk':'流失风险'}
lifecycle = pd.read_csv("data/user_lifecycle_stage.csv")
lifecycle['SCN'] = lifecycle['LifeStage'].map(stage_cn_map)
sc = lifecycle['SCN'].value_counts()
ordered_vals = [sc.get(stage_cn_map[s],0) for s in stage_order]
ordered_names = [stage_cn_map[s] for s in stage_order]

ch_3_1 = go.Figure()
ch_3_1.add_trace(go.Bar(x=ordered_names, y=ordered_vals, name='用户数',
                         marker=dict(color=px.colors.qualitative.Set2[:6], line=dict(width=0)),
                         text=ordered_vals, textposition='outside', textfont=dict(size=10)))
base_layout(ch_3_1, '用户生命周期阶段分布')
ch_3_1.update_layout(xaxis=dict(tickangle=30, gridcolor='#f0f0f0'), yaxis=dict(title='用户数', gridcolor='#f0f0f0'))

# Chart 3.2: Cohort 留存热力图
cohort_clean = cohort.fillna(0)
ch_3_2 = go.Figure()
ch_3_2.add_trace(go.Heatmap(
    z=cohort_clean.values, x=cohort_clean.columns.tolist(), y=cohort_clean.index.tolist(),
    colorscale='RdYlGn', zmin=0, zmax=50,
    text=cohort_clean.round(1).values, texttemplate='%{text:.0f}%', textfont={"size":9},
    colorbar=dict(title='留存率%', len=0.8),
    hovertemplate='队列: %{y}<br>第%{x}月<br>留存率: %{z:.1f}%<extra></extra>'))
base_layout(ch_3_2, 'Cohort 用户留存热力图 (留存率 %)')
ch_3_2.update_layout(xaxis=dict(title='月份序号 (0=首月)', gridcolor='#f0f0f0'),
                     yaxis=dict(title='首次购买月份', gridcolor='#f0f0f0'))

# Chart 3.3: 新增用户趋势
ch_3_3 = go.Figure()
ch_3_3.add_trace(go.Scatter(x=monthly['YearMonth'], y=monthly['NewUsers'], name='新增用户',
                            mode='lines+markers', fill='tozeroy', fillcolor='rgba(102,194,165,0.2)',
                            line=dict(color=C[0], width=2.5), marker=dict(size=6)))
ch_3_3.add_trace(go.Bar(x=monthly['YearMonth'], y=monthly['NewUsers'], name='',
                         marker=dict(color=C[0], opacity=0.3, line=dict(width=0)), showlegend=False))
base_layout(ch_3_3, '月度新增用户趋势')
ch_3_3.update_layout(xaxis=dict(tickangle=45, gridcolor='#f0f0f0'), yaxis=dict(title='新增用户数', gridcolor='#f0f0f0'))

# Chart 3.4: 月度复购率
mo_orders = df.groupby(['YearMonth','Customer ID'])['Invoice'].nunique().reset_index()
mo_total = mo_orders.groupby('YearMonth')['Customer ID'].count()
mo_repeat = mo_orders[mo_orders['Invoice']>=2].groupby('YearMonth')['Customer ID'].count()
repeat_rate = (mo_repeat / mo_total * 100).reset_index()
repeat_rate.columns = ['YearMonth','Rate']
ch_3_4 = go.Figure()
ch_3_4.add_trace(go.Scatter(x=repeat_rate['YearMonth'], y=repeat_rate['Rate'], name='复购率',
                            mode='lines+markers', fill='tozeroy', fillcolor='rgba(251,180,174,0.25)',
                            line=dict(color=C[3], width=2.5), marker=dict(size=6)))
base_layout(ch_3_4, '月度复购率趋势 (%)')
ch_3_4.update_layout(xaxis=dict(tickangle=45, gridcolor='#f0f0f0'), yaxis=dict(title='复购率 (%)', gridcolor='#f0f0f0',
                     range=[0, max(repeat_rate['Rate'])*1.15]))

# Chart 3.5: 平均购买间隔分布
multi = user[user['TotalOrders']>=2].copy()
multi['Interval'] = multi['LifeDays'] / (multi['TotalOrders']-1)
interval_bins = [0,7,14,30,60,90,180,float('inf')]
interval_labels = ['<7天','7-14天','14-30天','30-60天','60-90天','90-180天','>180天']
multi['IntSeg'] = pd.cut(multi['Interval'], bins=interval_bins, labels=interval_labels)
idist = multi['IntSeg'].value_counts().sort_index()
ch_3_5 = go.Figure()
ch_3_5.add_trace(go.Bar(x=idist.index.tolist(), y=idist.values, name='用户数',
                         marker=dict(color=C[5], line=dict(width=0)),
                         text=idist.values, textposition='outside', textfont=dict(size=10)))
base_layout(ch_3_5, '用户平均购买间隔分布 (多单用户)')
ch_3_5.update_layout(xaxis=dict(title='平均间隔', gridcolor='#f0f0f0'), yaxis=dict(title='用户数', gridcolor='#f0f0f0'))

# Chart 3.6: 用户生命周期价值(LTV)分布
ch_3_6 = go.Figure()
ch_3_6.add_trace(go.Histogram(x=user['TotalRevenue'], nbinsx=60, name='用户数',
                              marker=dict(color=C[4], line=dict(width=1, color='white')),
                              hovertemplate='LTV £%{x:,.0f}<br>用户数: %{y}<extra></extra>'))
ch_3_6.add_vline(x=user['TotalRevenue'].median(), line_dash='dash', line_color='red',
                 annotation=dict(text=f"中位数 £{user['TotalRevenue'].median():,.0f}", font=dict(size=10, color='red')))
base_layout(ch_3_6, '用户生命周期价值 (LTV) 分布')
ch_3_6.update_layout(xaxis=dict(title='累计消费 (£)', gridcolor='#f0f0f0'), yaxis=dict(title='用户数', gridcolor='#f0f0f0'))

page3_charts = {
    'ch_3_1': fig_json(ch_3_1), 'ch_3_2': fig_json(ch_3_2), 'ch_3_3': fig_json(ch_3_3),
    'ch_3_4': fig_json(ch_3_4), 'ch_3_5': fig_json(ch_3_5), 'ch_3_6': fig_json(ch_3_6),
}

# ====================================================================
# PAGE 4: 运营策略 — 6 charts
# ====================================================================
print("  Page 4: Strategy...")

# Chart 4.1: 用户成长漏斗
funnel_stages = ['首次购买', '2次以上', '3-5次', '6次以上', '10次以上', '高价值(Top20%)']
funnel_vals = [len(user), len(user[user['TotalOrders']>=2]), len(user[user['TotalOrders']>=3]),
               len(user[user['TotalOrders']>=6]), len(user[user['TotalOrders']>=10]),
               int(len(user)*0.2)]
ch_4_1 = go.Figure()
ch_4_1.add_trace(go.Bar(x=funnel_vals[::-1], y=funnel_stages[::-1], orientation='h',
                         marker=dict(color=px.colors.sequential.Blugrn_r[2:8], line=dict(width=0)),
                         text=[f'{v:,}人 ({v/funnel_vals[0]*100:.1f}%)' for v in funnel_vals[::-1]],
                         textposition='outside', textfont=dict(size=10)))
base_layout(ch_4_1, '用户成长路径漏斗')
ch_4_1.update_layout(xaxis=dict(title='用户数', gridcolor='#f0f0f0'), yaxis=dict(gridcolor='#f0f0f0'))

# Chart 4.2: Pareto 营收集中度
usr_sorted = user.sort_values('TotalRevenue', ascending=False)
usr_sorted['CumPct'] = usr_sorted['TotalRevenue'].cumsum() / usr_sorted['TotalRevenue'].sum() * 100
usr_sorted['UserPct'] = np.arange(1, len(usr_sorted)+1) / len(usr_sorted) * 100
ch_4_2 = go.Figure()
ch_4_2.add_trace(go.Scatter(x=usr_sorted['UserPct'], y=usr_sorted['CumPct'], name='累计营收占比',
                            mode='lines', fill='tozeroy', fillcolor='rgba(102,194,165,0.15)',
                            line=dict(color=C[1], width=2.5)))
ch_4_2.add_trace(go.Scatter(x=[20,20,0], y=[0,73.5,73.5], mode='lines', name='Top20%用户=73.5%营收',
                            line=dict(color='#e74c3c', dash='dash', width=2)))
ch_4_2.add_annotation(x=25, y=68, text='Top 20% 用户<br>贡献 73.5% 营收', showarrow=False,
                      font=dict(size=11, color='#e74c3c'), bgcolor='rgba(255,255,255,0.8)')
base_layout(ch_4_2, '用户营收贡献 Pareto 曲线')
ch_4_2.update_layout(xaxis=dict(title='用户累计占比 (%)', gridcolor='#f0f0f0'),
                     yaxis=dict(title='营收累计占比 (%)', gridcolor='#f0f0f0'))

# Chart 4.3: 流失风险用户 vs 正常用户画像
atrisk = rfm[rfm['RFM_Segment'].isin(['AtRisk','Lost'])]
normal_rfm = rfm[~rfm['RFM_Segment'].isin(['AtRisk','Lost'])]
ch_4_3 = go.Figure()
metrics_cn = ['平均距上次购买(天)', '平均购买次数', '平均累计消费(£)']
ch_4_3.add_trace(go.Bar(x=metrics_cn, y=[atrisk['Recency'].mean(), atrisk['Frequency'].mean(), atrisk['Monetary'].mean()],
                         name='流失风险/已流失', marker=dict(color='#e74c3c', line=dict(width=0)),
                         text=[f'{atrisk["Recency"].mean():.0f}天', f'{atrisk["Frequency"].mean():.1f}次', f'£{atrisk["Monetary"].mean():.0f}'],
                         textposition='outside', textfont=dict(size=10)))
ch_4_3.add_trace(go.Bar(x=metrics_cn, y=[normal_rfm['Recency'].mean(), normal_rfm['Frequency'].mean(), normal_rfm['Monetary'].mean()],
                         name='正常/活跃用户', marker=dict(color=C[1], line=dict(width=0)),
                         text=[f'{normal_rfm["Recency"].mean():.0f}天', f'{normal_rfm["Frequency"].mean():.1f}次', f'£{normal_rfm["Monetary"].mean():.0f}'],
                         textposition='outside', textfont=dict(size=10)))
base_layout(ch_4_3, '流失风险用户 vs 正常用户 画像对比')
ch_4_3.update_layout(xaxis=dict(gridcolor='#f0f0f0'), yaxis=dict(title='数值', gridcolor='#f0f0f0'), barmode='group')

# Chart 4.4: 月度新增用户效率 (Rev/New User)
monthly['RevPerNew'] = monthly['TotalRevenue'] / monthly['NewUsers'].replace(0, np.nan)
ch_4_4 = go.Figure()
ch_4_4.add_trace(go.Scatter(x=monthly['YearMonth'], y=monthly['RevPerNew'], name='营收/新增用户',
                            mode='lines+markers', fill='tozeroy', line=dict(color=C[0], width=2.5), marker=dict(size=6)))
base_layout(ch_4_4, '月度新增用户效率 (总营收/新增用户数)')
ch_4_4.update_layout(xaxis=dict(tickangle=45, gridcolor='#f0f0f0'),
                     yaxis=dict(title='营收/新增 (£)', gridcolor='#f0f0f0'))

# Chart 4.5: 国家机会矩阵 (用户数 vs 客单价)
country_stats = df.groupby('Country').agg(用户数=('Customer ID','nunique'), 客单价=('Revenue','mean')).reset_index()
country_stats = country_stats[country_stats['用户数']>=5]
ch_4_5 = go.Figure()
ch_4_5.add_trace(go.Scatter(x=country_stats['用户数'], y=country_stats['客单价'],
                            mode='markers+text', text=country_stats['Country'],
                            textposition='top center', textfont=dict(size=9),
                            marker=dict(size=np.clip(country_stats['客单价']/5, 8, 50),
                                        color=country_stats['客单价'], colorscale='Blues',
                                        showscale=True, colorbar=dict(title='客单价(£)', len=0.7)),
                            hovertemplate='%{text}<br>用户: %{x}<br>客单价: £%{y:.1f}<extra></extra>'))
base_layout(ch_4_5, '国家机会矩阵 (用户规模 vs 客单价)')
ch_4_5.update_layout(xaxis=dict(title='用户数', gridcolor='#f0f0f0'),
                     yaxis=dict(title='平均客单价 (£)', gridcolor='#f0f0f0'))

# Chart 4.6: 用户半年度迁移
df['Half'] = df['InvoiceDate'].apply(lambda x: 'H1' if x <= pd.Timestamp('2010-06-01') else 'H2')
h1_users = set(df[df['Half']=='H1']['Customer ID'].unique())
h2_users = set(df[df['Half']=='H2']['Customer ID'].unique())
new_h2 = len(h2_users - h1_users)
retained = len(h1_users & h2_users)
churned_h1 = len(h1_users - h2_users)
ch_4_6 = go.Figure()
ch_4_6.add_trace(go.Pie(
    labels=['留存 (两个半年均活跃)', f'H2新用户 ({len(h2_users)-len(h1_users&h2_users)}人)', 'H1后流失'],
    values=[retained, new_h2, churned_h1], hole=0.55,
    marker_colors=[C[1], C[0], '#e74c3c'],
    textinfo='label+percent', textfont=dict(size=10)))
base_layout(ch_4_6, '用户半年度迁移 (H1 vs H2 2010)')
ch_4_6.update_layout(height=CHART_HEIGHT, margin=dict(l=10,r=10,t=40,b=10))

page4_charts = {
    'ch_4_1': fig_json(ch_4_1), 'ch_4_2': fig_json(ch_4_2), 'ch_4_3': fig_json(ch_4_3),
    'ch_4_4': fig_json(ch_4_4), 'ch_4_5': fig_json(ch_4_5), 'ch_4_6': fig_json(ch_4_6),
}

# ====================================================================
# PAGE 5: KPI 总览 — 8 KPI + 4 charts
# ====================================================================
print("  Page 5: KPI Summary...")

champions_rev_pct = rfm[rfm['RFM_Segment']=='Champions']['Monetary'].sum() / total_revenue * 100
at_risk_users = rfm[rfm['RFM_Segment'].isin(['AtRisk','Lost'])].shape[0]
repeat_pct = len(user[user['TotalOrders']>1]) / len(user) * 100
active_30d = len(user[(reference_date - user['LastPurchase']).dt.days <= 30])

# KPI gauges (individual)
ch_5_1 = go.Figure(go.Indicator(
    mode='gauge+number', value=repeat_pct, number=dict(suffix='%', font=dict(size=32)),
    title=dict(text='<b>用户复购率</b>', font=dict(size=13)),
    gauge=dict(axis=dict(range=[0,100], tickfont=dict(size=10)),
               bar=dict(color=C[1], thickness=0.25),
               steps=[{'range':[0,30],'color':'#fdebd0'},{'range':[30,60],'color':'#f5cba7'},{'range':[60,100],'color':'#d5f5e3'}],
               threshold=dict(line=dict(color='#2c3e50',width=3), thickness=0.25, value=repeat_pct))))
base_layout(ch_5_1, '')
ch_5_1.update_layout(height=280, margin=dict(l=20,r=20,t=10,b=10))

ch_5_2 = go.Figure(go.Indicator(
    mode='gauge+number', value=at_risk_users, number=dict(font=dict(size=32)),
    title=dict(text='<b>流失风险用户</b>', font=dict(size=13)),
    gauge=dict(axis=dict(range=[0,len(user)], tickfont=dict(size=10)),
               bar=dict(color='#e74c3c', thickness=0.25),
               steps=[{'range':[0,500],'color':'#d5f5e3'},{'range':[500,1500],'color':'#f5cba7'},{'range':[1500,len(user)],'color':'#fdebd0'}],
               threshold=dict(line=dict(color='#2c3e50',width=3), thickness=0.25, value=at_risk_users))))
base_layout(ch_5_2, '')
ch_5_2.update_layout(height=280, margin=dict(l=20,r=20,t=10,b=10))

ch_5_3 = go.Figure(go.Indicator(
    mode='gauge+number', value=champions_rev_pct, number=dict(suffix='%', font=dict(size=32)),
    title=dict(text='<b>冠军用户营收占比</b>', font=dict(size=13)),
    gauge=dict(axis=dict(range=[0,100], tickfont=dict(size=10)),
               bar=dict(color='#f39c12', thickness=0.25),
               steps=[{'range':[0,30],'color':'#fdebd0'},{'range':[30,70],'color':'#f5cba7'},{'range':[70,100],'color':'#fadbd8'}],
               threshold=dict(line=dict(color='#2c3e50',width=3), thickness=0.25, value=champions_rev_pct))))
base_layout(ch_5_3, '')
ch_5_3.update_layout(height=280, margin=dict(l=20,r=20,t=10,b=10))

ch_5_4 = go.Figure(go.Indicator(
    mode='gauge+number', value=active_30d, number=dict(font=dict(size=32)),
    title=dict(text='<b>30天活跃用户</b>', font=dict(size=13)),
    gauge=dict(axis=dict(range=[0,len(user)], tickfont=dict(size=10)),
               bar=dict(color=C[0], thickness=0.25),
               steps=[{'range':[0,800],'color':'#fdebd0'},{'range':[800,1500],'color':'#f5cba7'},{'range':[1500,len(user)],'color':'#d5f5e3'}],
               threshold=dict(line=dict(color='#2c3e50',width=3), thickness=0.25, value=active_30d))))
base_layout(ch_5_4, '')
ch_5_4.update_layout(height=280, margin=dict(l=20,r=20,t=10,b=10))

# Chart 5.5: 营收趋势 (bigger)
ch_5_5 = go.Figure()
ch_5_5.add_trace(go.Bar(x=monthly['YearMonth'], y=monthly['TotalRevenue'], name='月度营收',
                         marker=dict(color=C[3], line=dict(width=0)), showlegend=False))
ch_5_5.add_trace(go.Scatter(x=monthly['YearMonth'], y=monthly['TotalRevenue'].rolling(2).mean(),
                            name='2月移动平均', mode='lines', line=dict(color='#2c3e50', width=2)))
base_layout(ch_5_5, '月度营收趋势 (含2月移动平均)')
ch_5_5.update_layout(height=CHART_HEIGHT, xaxis=dict(tickangle=45, gridcolor='#f0f0f0'), yaxis=dict(gridcolor='#f0f0f0'))

# Chart 5.6: 星期×小时热力图
df['Weekday'] = df['InvoiceDate'].dt.day_name()
df['Hour'] = df['InvoiceDate'].dt.hour
heat_data = df.groupby(['Weekday','Hour'])['Revenue'].sum().reset_index()
heat_pivot = heat_data.pivot_table(values='Revenue', index='Weekday', columns='Hour', aggfunc='sum').fillna(0)
weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Sunday']
weekday_cn = ['周一','周二','周三','周四','周五','周日']
heat_pivot = heat_pivot.reindex([w for w in weekday_order if w in heat_pivot.index])
heat_pivot.index = [weekday_cn[weekday_order.index(w)] for w in heat_pivot.index]
ch_5_6 = go.Figure()
ch_5_6.add_trace(go.Heatmap(z=heat_pivot.values, x=heat_pivot.columns.tolist(), y=heat_pivot.index.tolist(),
                            colorscale='Blues', colorbar=dict(title='营收(£)', len=0.8),
                            hovertemplate='%{y} %{x}:00<br>营收: £%{z:,.0f}<extra></extra>'))
base_layout(ch_5_6, '星期×小时 营收热力图')
ch_5_6.update_layout(xaxis=dict(title='小时', gridcolor='#f0f0f0', dtick=3),
                     yaxis=dict(title='', gridcolor='#f0f0f0'))

# Chart 5.7: 分群营收条形图
ch_5_7 = go.Figure()
ch_5_7.add_trace(go.Bar(x=seg_rev.index.tolist(), y=seg_rev.values, name='营收',
                         marker=dict(color=px.colors.qualitative.Set2[:7], line=dict(width=0)),
                         text=[f'£{v/1000:.0f}K' for v in seg_rev.values],
                         textposition='outside', textfont=dict(size=9)))
base_layout(ch_5_7, '各用户分群营收贡献 (£)')
ch_5_7.update_layout(xaxis=dict(tickangle=30, gridcolor='#f0f0f0'), yaxis=dict(title='营收 (£)', gridcolor='#f0f0f0'))

# Chart 5.8: TOP 10 商品
top_prods = df.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(10)
ch_5_8 = go.Figure()
ch_5_8.add_trace(go.Bar(x=top_prods.values, y=top_prods.index, orientation='h',
                         marker=dict(color=px.colors.sequential.Blues_r[3:], line=dict(width=0)),
                         text=[f'£{v/1000:.0f}K' for v in top_prods.values],
                         textposition='outside', textfont=dict(size=9)))
base_layout(ch_5_8, 'TOP 10 畅销商品 (按营收)')
ch_5_8.update_layout(xaxis=dict(title='营收 (£)', gridcolor='#f0f0f0'), yaxis=dict(gridcolor='#f0f0f0'))

page5_charts = {
    'ch_5_1': fig_json(ch_5_1), 'ch_5_2': fig_json(ch_5_2), 'ch_5_3': fig_json(ch_5_3),
    'ch_5_4': fig_json(ch_5_4), 'ch_5_5': fig_json(ch_5_5), 'ch_5_6': fig_json(ch_5_6),
    'ch_5_7': fig_json(ch_5_7), 'ch_5_8': fig_json(ch_5_8),
}

# ====================================================================
# Build HTML with CSS Grid Layout
# ====================================================================
print("Generating HTML...")

# Merge all chart data
all_charts = {}
all_charts.update(page1_charts)
all_charts.update(page2_charts)
all_charts.update(page3_charts)
all_charts.update(page4_charts)
all_charts.update(page5_charts)

# Find max revenue month for delta
max_rev_idx = monthly['TotalRevenue'].idxmax()
nov_rev = monthly[monthly['YearMonth']=='2010-11']['TotalRevenue'].values
oct_rev = monthly[monthly['YearMonth']=='2010-10']['TotalRevenue'].values
rev_delta = ((nov_rev[0] - oct_rev[0]) / oct_rev[0] * 100) if len(nov_rev)>0 and len(oct_rev)>0 else 0

html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>电商用户运营分析看板 — Online Retail II</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  :root {{
    --bg: #f0f2f5;
    --card-bg: #ffffff;
    --primary: #1a252f;
    --accent: #2980b9;
    --accent2: #27ae60;
    --danger: #e74c3c;
    --warning: #f39c12;
    --text: #2c3e50;
    --text-light: #7f8c8d;
    --border: #dde1e7;
    --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-lg: 0 4px 12px rgba(0,0,0,0.08);
    --radius: 10px;
  }}

  * {{ margin:0; padding:0; box-sizing:border-box; }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }}

  /* ===== HEADER ===== */
  .header {{
    background: linear-gradient(135deg, #1a252f 0%, #2c3e50 100%);
    color: white;
    padding: 14px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    position: sticky;
    top: 0;
    z-index: 1000;
  }}
  .header h1 {{ font-size: 20px; font-weight: 600; letter-spacing: 0.5px; }}
  .header .sub {{ font-size: 12px; opacity: 0.75; margin-top: 2px; }}
  .header-kpis {{ display: flex; gap: 12px; }}
  .hkpi {{
    background: rgba(255,255,255,0.12);
    padding: 8px 18px;
    border-radius: 8px;
    text-align: center;
    backdrop-filter: blur(8px);
  }}
  .hkpi .v {{ font-size: 22px; font-weight: 700; }}
  .hkpi .l {{ font-size: 10px; opacity: 0.7; text-transform: uppercase; letter-spacing: 1px; }}

  /* ===== TAB NAV ===== */
  .tab-nav {{
    display: flex;
    background: #fff;
    border-bottom: 1px solid var(--border);
    padding: 0 28px;
    position: sticky;
    top: 72px;
    z-index: 999;
    box-shadow: var(--shadow);
  }}
  .tab-btn {{
    padding: 13px 24px;
    border: none;
    background: none;
    cursor: pointer;
    font-size: 13.5px;
    font-weight: 500;
    color: var(--text-light);
    border-bottom: 3px solid transparent;
    margin-bottom: -1px;
    transition: all 0.2s;
    white-space: nowrap;
    font-family: inherit;
  }}
  .tab-btn:hover {{ color: var(--accent); background: #f5f8fb; }}
  .tab-btn.active {{ color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }}

  /* ===== CONTENT ===== */
  .main-content {{ padding: 20px 28px; max-width: 1500px; margin: 0 auto; }}

  .tab-panel {{ display: none; animation: fadeIn 0.3s ease; }}
  .tab-panel.active {{ display: block; }}
  @keyframes fadeIn {{ from {{ opacity:0; transform: translateY(6px); }} to {{ opacity:1; transform: translateY(0); }} }}

  /* ===== KPI ROW ===== */
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 18px;
  }}
  .kpi-card {{
    background: var(--card-bg);
    border-radius: var(--radius);
    padding: 16px 20px;
    box-shadow: var(--shadow);
    text-align: center;
    transition: transform 0.15s, box-shadow 0.15s;
    border-top: 3px solid var(--accent);
  }}
  .kpi-card.warn {{ border-top-color: var(--warning); }}
  .kpi-card.danger {{ border-top-color: var(--danger); }}
  .kpi-card.success {{ border-top-color: var(--accent2); }}
  .kpi-card .val {{ font-size: 28px; font-weight: 700; color: var(--primary); }}
  .kpi-card .lbl {{ font-size: 11px; color: var(--text-light); text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }}
  .kpi-card .dlt {{ font-size: 11px; margin-top: 3px; font-weight: 600; }}
  .kpi-card .dlt.up {{ color: var(--accent2); }}
  .kpi-card .dlt.down {{ color: var(--danger); }}

  /* ===== INSIGHT BOX ===== */
  .insight {{
    background: var(--card-bg);
    border-radius: var(--radius);
    padding: 14px 20px;
    margin-bottom: 16px;
    box-shadow: var(--shadow);
    border-left: 4px solid var(--accent);
    font-size: 13px;
    line-height: 1.7;
  }}
  .insight.warn {{ border-left-color: var(--warning); }}
  .insight.danger {{ border-left-color: var(--danger); }}
  .insight.success {{ border-left-color: var(--accent2); }}
  .insight h3 {{ font-size: 14px; margin-bottom: 4px; color: var(--primary); }}

  /* ===== CHART GRID ===== */
  .chart-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 16px;
  }}
  .chart-grid.col2 {{ grid-template-columns: repeat(2, 1fr); }}
  .chart-grid.col4 {{ grid-template-columns: repeat(4, 1fr); }}

  .chart-box {{
    background: var(--card-bg);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    overflow: hidden;
  }}
  .chart-box .chart-inner {{
    width: 100%;
  }}
  .chart-box.span2 {{ grid-column: span 2; }}
  .chart-box.span3 {{ grid-column: span 3; }}

  /* ===== RESPONSIVE ===== */
  @media (max-width: 1200px) {{
    .chart-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .chart-grid.col2 {{ grid-template-columns: 1fr; }}
    .kpi-row {{ grid-template-columns: repeat(2, 1fr); }}
    .chart-box.span2, .chart-box.span3 {{ grid-column: span 1; }}
  }}
  @media (max-width: 768px) {{
    .chart-grid, .chart-grid.col2, .chart-grid.col4 {{ grid-template-columns: 1fr; }}
    .kpi-row {{ grid-template-columns: 1fr; }}
    .tab-btn {{ padding: 10px 14px; font-size: 12px; }}
    .header {{ flex-direction: column; gap: 10px; }}
    .header-kpis {{ gap: 6px; }}
    .hkpi {{ padding: 6px 12px; }}
    .hkpi .v {{ font-size: 16px; }}
  }}
</style>
</head>
<body>

<!-- ================================================================ -->
<!-- HEADER -->
<!-- ================================================================ -->
<div class="header">
  <div>
    <h1>电商用户运营分析看板</h1>
    <div class="sub">数据来源: Online Retail II &bull; 时间范围: 2009-12-01 ~ 2010-12-09 &bull; 清洗后交易: 407,664 条 &bull; 用户: 4,312 人</div>
  </div>
  <div class="header-kpis">
    <div class="hkpi"><div class="v">4,312</div><div class="l">总用户数</div></div>
    <div class="hkpi"><div class="v">67.1%</div><div class="l">复购率</div></div>
    <div class="hkpi"><div class="v">8,832,003</div><div class="l">总营收 (£)</div></div>
    <div class="hkpi"><div class="v">18.9%</div><div class="l">次月留存率</div></div>
  </div>
</div>

<!-- ================================================================ -->
<!-- TAB NAVIGATION -->
<!-- ================================================================ -->
<div class="tab-nav">
  <button class="tab-btn active" onclick="switchTab(0)">1. 用户概览</button>
  <button class="tab-btn" onclick="switchTab(1)">2. 用户价值 (RFM)</button>
  <button class="tab-btn" onclick="switchTab(2)">3. 生命周期与留存</button>
  <button class="tab-btn" onclick="switchTab(3)">4. 运营策略分析</button>
  <button class="tab-btn" onclick="switchTab(4)">5. KPI 总览</button>
</div>

<!-- ================================================================ -->
<!-- MAIN CONTENT -->
<!-- ================================================================ -->
<div class="main-content">

<!-- ==================== PAGE 0: 用户概览 ==================== -->
<div class="tab-panel active" id="panel-0">

  <div class="kpi-row">
    <div class="kpi-card"><div class="val">4,312</div><div class="lbl">总用户数</div><div class="dlt up">月均活跃 1,009 人</div></div>
    <div class="kpi-card"><div class="val">8,832,003</div><div class="lbl">总营收 (£)</div><div class="dlt up">H2环比增长 {rev_delta:.0f}%</div></div>
    <div class="kpi-card"><div class="val">21.67</div><div class="lbl">平均客单价 (£)</div><div class="dlt up">每单均5.2件商品</div></div>
    <div class="kpi-card warn"><div class="val">32.9%</div><div class="lbl">仅购买一次</div><div class="dlt down">1,419人首单后流失</div></div>
  </div>

  <div class="insight">
    <h3>关键洞察: 用户整体画像</h3>
    <b>英国用户占 92%</b>，市场高度集中；<b>Top 20% 用户贡献 73.5% 的营收</b>，价值分布极不均衡；
    月活用户从 2010 年中开始明显增长，11 月达到峰值 1,607 人，Q4 季节性效应显著；
    但 <b>32.9% 的用户仅购买一次就流失</b>，新用户激活是最大改进空间。
  </div>

  <div class="chart-grid">
    <div class="chart-box"><div class="chart-inner" id="ch_1_1"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_1_2"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_1_3"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_1_4"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_1_5"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_1_6"></div></div>
  </div>

  <div class="insight warn">
    <h3>注意: 季节性波动明显</h3>
    11-12月为销售旺季，2009年12月新增用户955人（全年最高）；12月数据仅含9天，进行同比分析时需注意。2010年下半年整体增长加速，建议重点关注 Q4 旺季的用户留存策略。
  </div>
</div>

<!-- ==================== PAGE 1: RFM 用户价值 ==================== -->
<div class="tab-panel" id="panel-1">

  <div class="kpi-row">
    <div class="kpi-card success"><div class="val">458</div><div class="lbl">高价值冠军用户 (10.6%)</div><div class="dlt up">贡献 50.7% 营收</div></div>
    <div class="kpi-card danger"><div class="val">1,951</div><div class="lbl">流失风险 + 已流失用户</div><div class="dlt down">占用户总量 45.2%</div></div>
    <div class="kpi-card"><div class="val">90 天</div><div class="lbl">平均 Recency</div><div class="dlt down">中位数 52 天</div></div>
    <div class="kpi-card"><div class="val">2,048</div><div class="lbl">用户平均LTV (£)</div><div class="dlt up">中位数 706</div></div>
  </div>

  <div class="insight danger">
    <h3>核心风险: 用户流失严重，营收高度集中</h3>
    <b>1,951 人 (45.2%)</b> 被标记为流失风险或已流失——这些用户历史有价值（人均消费 1,422）但已长期未购买，涉及 <b>潜在可挽回营收 190 万</b>。
    同时仅 458 位冠军用户 (10.6%) 贡献了超过一半的营收 (50.7%)，业务过度依赖极少数核心用户，一旦流失将对营收产生重大冲击。
  </div>

  <div class="chart-grid">
    <div class="chart-box"><div class="chart-inner" id="ch_2_1"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_2_2"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_2_3"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_2_4"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_2_5"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_2_6"></div></div>
  </div>

  <div class="insight success">
    <h3>机会: 忠诚用户升级为冠军用户</h3>
    当前有 <b>830 位忠诚用户</b> (19.2%)，平均购买 6 次、人均消费 2,320——他们已有良好的购买习惯。通过 VIP 体系升级激励，即使仅 15% 转化为冠军用户（约 125 人），就能显著分散营收集中风险，预计带来增量营收 50 万+。
  </div>
</div>

<!-- ==================== PAGE 2: 生命周期与留存 ==================== -->
<div class="tab-panel" id="panel-2">

  <div class="kpi-row">
    <div class="kpi-card danger"><div class="val">18.9%</div><div class="lbl">平均次月留存率</div><div class="dlt down">范围: 10.8% ~ 35.3%</div></div>
    <div class="kpi-card"><div class="val">626</div><div class="lbl">新用户 (近90天)</div><div class="dlt up">占总量 14.5%</div></div>
    <div class="kpi-card"><div class="val">52 天</div><div class="lbl">中位购买间隔</div><div class="dlt down">多单用户</div></div>
    <div class="kpi-card success"><div class="val">67.1%</div><div class="lbl">复购率</div><div class="dlt up">2,893 人购买 2 次以上</div></div>
  </div>

  <div class="insight warn">
    <h3>留存挑战: 首月是关键流失节点</h3>
    平均 <b>次月留存率仅 18.9%</b>——约 81% 的新用户次月不再回访。表现最好的队列 (2009年12月) 留存率达 35.3%，最差的 (2010年11月) 仅 10.8%。
    假期旺季获取的用户留存率系统性偏低，说明节日冲动消费用户需要差异化的再激活策略，而非一刀切的运营方案。
  </div>

  <div class="chart-grid">
    <div class="chart-box"><div class="chart-inner" id="ch_3_1"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_3_2"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_3_3"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_3_4"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_3_5"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_3_6"></div></div>
  </div>

  <div class="insight success">
    <h3>好消息: 留下来的用户忠诚度稳定</h3>
    虽然初期留存率低，但 <b>度过前 3 个月后留存率稳定在 23-29%</b>，持续到第 12 个月。
    2009年12月队列在第12月仍有 24.8% 留存——证明一旦建立购买习惯，用户忠诚度可以长期维持。
    核心策略应聚焦于 <b>"首次购买 → 第3个月"之间的激活窗口</b>。
  </div>
</div>

<!-- ==================== PAGE 3: 运营策略 ==================== -->
<div class="tab-panel" id="panel-3">

  <div class="kpi-row">
    <div class="kpi-card danger"><div class="val">32.9%</div><div class="lbl">单次购买后流失</div><div class="dlt down">1,419 人未再次购买</div></div>
    <div class="kpi-card"><div class="val">4,475,596</div><div class="lbl">冠军用户营收 (£)</div><div class="dlt up">占总营收 50.7%</div></div>
    <div class="kpi-card"><div class="val">21.8%</div><div class="lbl">高频用户 (6次+)</div><div class="dlt up">942 位核心用户</div></div>
    <div class="kpi-card success"><div class="val">2,375</div><div class="lbl">H1→H2 留存用户</div><div class="dlt up">半年留存率 55.1%</div></div>
  </div>

  <div class="insight danger">
    <h3>优先级 #1: 堵住"首单即流失"的漏洞</h3>
    <b>1,419 位用户 (32.9%) 从未进行第二次购买。</b> 这是用户增长漏斗中最大的单点流失。
    实施 "首单后 24h 感谢信 + 第7天个性化推荐 + 第21天限时优惠券" 三步激活流程，
    仅需将首单留存率提升 10 个百分点，即可挽回约 430 位用户，预计年化增量营收 <b>30 万+</b>。
  </div>

  <div class="chart-grid">
    <div class="chart-box"><div class="chart-inner" id="ch_4_1"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_4_2"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_4_3"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_4_4"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_4_5"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_4_6"></div></div>
  </div>

  <div class="insight success">
    <h3>六大运营行动方案</h3>
    <b>1. 新用户激活计划:</b> 首单后自动化 3 步触达 (24h/7天/21天)，目标次月留存率 30%+。<br>
    <b>2. VIP 分层体系:</b> 银卡 (3K+) / 金卡 (5K+) / 铂金 (10K+)，含专属客服、优先发货、生日礼遇。<br>
    <b>3. 流失召回机制:</b> 90 天未购买自动触发个性化优惠券 (514 位风险用户 = 122 万潜在挽回价值)。<br>
    <b>4. 节日新客专项:</b> 11-12 月新客在 1 月收到"新年回访"专属优惠，与普通新客差异化运营。<br>
    <b>5. 欧洲市场拓展:</b> 优先深耕德国 (67人)、法国 (47人)，提供本地化产品描述与客服。<br>
    <b>6. 积分激励计划:</b> 每单积分可兑换；第2单免运费、第5单 5% 折扣、第10单升级VIP。
  </div>
</div>

<!-- ==================== PAGE 4: KPI 总览 ==================== -->
<div class="tab-panel" id="panel-4">

  <div class="insight">
    <h3>高管摘要: 用户运营健康度全景</h3>
    业务呈现 <b>"高集中度、中等复购、低留存"</b> 的特征——Top 20% 用户贡献 73.5% 营收，67.1% 用户有复购行为，但次月留存率仅 18.9%。
    核心战略方向：<b>(1)</b> 激活新用户提升早期留存；<b>(2)</b> 分散营收集中度降低业务风险；<b>(3)</b> 拓展英国以外市场获取增量。
  </div>

  <div class="chart-grid col4">
    <div class="chart-box"><div class="chart-inner" id="ch_5_1"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_5_2"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_5_3"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_5_4"></div></div>
  </div>

  <div class="chart-grid">
    <div class="chart-box span2"><div class="chart-inner" id="ch_5_5"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_5_6"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_5_7"></div></div>
    <div class="chart-box"><div class="chart-inner" id="ch_5_8"></div></div>
  </div>

  <div class="kpi-row">
    <div class="kpi-card success"><div class="val">6</div><div class="lbl">战略行动项</div></div>
    <div class="kpi-card danger"><div class="val">1,951</div><div class="lbl">需立即干预用户</div></div>
    <div class="kpi-card warn"><div class="val">190 万</div><div class="lbl">可挽回营收池 (£)</div></div>
    <div class="kpi-card"><div class="val">30%+</div><div class="lbl">目标次月留存率</div></div>
  </div>
</div>

</div><!-- /main-content -->

<!-- ================================================================ -->
<!-- SCRIPTS -->
<!-- ================================================================ -->
<script>
// ===== TAB SWITCHING =====
function switchTab(idx) {{
  document.querySelectorAll('.tab-panel').forEach(function(p) {{ p.classList.remove('active'); }});
  document.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
  document.getElementById('panel-' + idx).classList.add('active');
  document.querySelectorAll('.tab-btn')[idx].classList.add('active');
  setTimeout(function() {{
    var plots = document.getElementById('panel-' + idx).querySelectorAll('.js-plotly-plot');
    plots.forEach(function(el) {{ Plotly.Plots.resize(el); }});
  }}, 80);
}}

// ===== RENDER ALL CHARTS =====
var chartIds = {json.dumps(list(all_charts.keys()), cls=NpEncoder)};
var chartData = {json.dumps(all_charts, cls=NpEncoder)};

chartIds.forEach(function(id) {{
  var d = chartData[id];
  var el = document.getElementById(id);
  if (el && d) {{
    Plotly.newPlot(el, d.data, d.layout, {{
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['lasso2d', 'select2d', 'sendDataToCloud'],
      displaylogo: false,
      toImageButtonOptions: {{ format: 'png', filename: id, scale: 1.5 }}
    }});
  }}
}});

console.log('Dashboard ready — 5 pages, 32 charts, all in Chinese.');
</script>

</body>
</html>
'''

# ====================================================================
# Write output
# ====================================================================
with open("../outputs/interactive_dashboard.html", "w", encoding="utf-8") as f:
    f.write(html)

import os
print(f"\nDone! outputs/interactive_dashboard.html ({os.path.getsize('outputs/interactive_dashboard.html')/1024:.0f} KB)")
print("Features: 5 tabs | 32 independent charts | CSS Grid | All Chinese | No overlap")
