# 基于 RFM 模型的电商用户价值分层与业务健康度分析

本项目基于`online_retail_II.xlsx`电商交易数据，完成从数据清洗到用户分层建模的完整分析流程，包括：

- 数据清洗（缺失值、重复值、退货订单、异常交易过滤）
- 数据探索（收入趋势、国家分布、交易统计）
- 数据预处理（截尾、对数变换、标准化）
- RFM 指标构建（Recency / Frequency / Monetary）
- K-means 聚类分层（含 K 值评估）
- 可视化与业务洞察输出

## 项目结构

```text
RFM Customer Value Analytics/
├─ online_retail_II.xlsx
├─ requirements.txt
├─ README.md
├─ src/
│  └─ rfm_pipeline.py
└─ outputs/
   ├─ figures/
   ├─ tables/
   └─ reports/
```

## 数据概览

**数据源**：`online_retail_II.xlsx`（电商交易明细表）

**时间范围**：2009年12月‑2011年12月（共25个月）

**原始规模**：1,067,371 条交易记录，清洗后保留 793,609 条（剔除缺失客户、退货订单、异常交易）

**字段说明**：

| 字段名 | 类型 | 含义 |
| ------- | ---- | ---- |
| Invoice | 字符串 | 订单编号（以 `C` 开头的为退货订单） |
| StockCode | 字符串 | 商品编码 |
| Description | 字符串 | 商品描述 |
| Quantity | 整数 | 购买数量（>0） |
| InvoiceDate | 日期时间 | 订单发生时间 |
| Price | 浮点数 | 商品单价（>0） |
| Customer ID | 整数 | 客户唯一标识 |
| Country | 字符串 | 客户所在国家 |

**衍生指标（RFM）**：

| 指标 | 计算方式 | 业务含义 |
| ---- | -------- | -------- |
| Recency (R) | 距分析截止日最近一次消费的天数 | 客户活跃程度，越小表示越活跃 |
| Frequency (F) | 客户总订单数（去重） | 客户购买频次，反映忠诚度 |
| Monetary (M) | 客户累计消费金额（Quantity × Price） | 客户价值贡献，反映消费能力 |

## 运行方式

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 执行主脚本

```bash
python src/rfm_pipeline.py
```

3. 查看输出结果

- 表格：`outputs/tables/`
- 图像：`outputs/figures/`
- 报告：`outputs/reports/business_insights.md`

## 输出说明

### 关键表格

- `data_cleaning_report.csv`：清洗前后数据量与核心统计
- `customer_rfm_raw.csv`：用户级原始 RFM 指标
- `customer_rfm_transformed.csv`：聚类输入特征
- `k_selection_metrics.csv`：各 K 值的 Inertia / Silhouette
- `customer_rfm_clustered.csv`：用户级分群结果
- `cluster_profile.csv`：分群画像（客群规模、R/F/M 均值）

### 关键图表

- `monthly_revenue_trend.png`：月度营收趋势
- `top10_country_revenue.png`：国家收入 Top10
- `rfm_distributions.png`：R/F/M 分布
- `k_selection.png`：K 值选择（肘部法 + 轮廓系数）
- `cluster_profile_heatmap.png`：分群画像热力图
- `segment_pca_scatter.png`：PCA 二维分群可视化

## 方法简述

1. **清洗规则**：去除缺失客户、退货订单（`Invoice` 以 `C` 开头）、`Quantity<=0`、`Price<=0`。
2. **RFM 定义**：
   - `Recency`：距最近一次消费天数（越小越好）
   - `Frequency`：消费频次（订单数）
   - `Monetary`：累计消费金额
3. **聚类流程**：
   - 对 R/F/M 做 1%~99% 截尾
   - `log1p` 变换
   - `StandardScaler` 标准化
   - K-means 聚类并评估最佳 K
4. **业务解释**：根据分群 R/F/M 画像自动标注客群（如 Champions、Loyal、At Risk 等）并给出运营建议。
