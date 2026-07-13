# 📊 E-Commerce User Operations Analytics

> **电商用户运营分析** — 基于 Online Retail II 数据集的完整用户行为与价值分析项目
> RFM · KMeans · Cohort · Funnel · Interactive Dashboard

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)](https://pandas.pydata.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.0+-purple.svg)](https://plotly.com)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)

---

## 📖 项目概述

本项目基于 **UCI Online Retail II** 数据集（2009.12 - 2010.12，525,461 条交易记录），按照企业级用户运营分析框架，完成从数据清洗到可视化看板的完整分析流程。

### 🎯 分析目标


| 目标               | 方法                                 |
| -------------------- | -------------------------------------- |
| 用户是谁？         | 用户画像分析（消费分布、频次、地区） |
| 用户如何消费？     | 月度趋势、客单价、复购行为           |
| 谁是高价值用户？   | **RFM 模型 + KMeans 聚类**           |
| 用户会流失吗？     | **Cohort 留存分析**                  |
| 如何提升用户价值？ | 漏斗分析 + 运营策略建议              |

---

## 📁 项目结构

```
RMF(re)/
├── README.md                          ← 项目文档
├── .gitignore
├── requirements.txt
├── skill.md                           ← 分析规范（8 阶段方法论）
│
├── data/                              ← 📦 数据层
│   ├── online_retail_II.xlsx          │  原始数据集
│   ├── cleaned_transactions.csv       │  清洗后交易（407,664 条）
│   ├── user_metrics.csv               │  用户指标表（4,312 人）
│   ├── rfm_analysis.csv               │  RFM 分群结果
│   ├── cohort_retention.csv           │  Cohort 留存矩阵
│   ├── monthly_user_stats.csv         │  月度统计
│   └── user_lifecycle_stage.csv       │  生命周期阶段
│
├── scripts/                           ← 🔧 分析脚本
│   ├── 1_data_cleaning.py             │  Phase 1: 数据清洗 & 特征工程
│   ├── 2_user_profiling.py            │  Phase 2-5: 画像 · RFM · Cohort
│   ├── 3_dashboard.py                 │  Phase 6-7: 单页看板（旧版）
│   ├── 4_report.py                    │  Phase 8: 文本报告生成
│   └── build_dashboard.py             │  🌟 主力脚本: 交互式看板
│
└── outputs/                           ← 📊 输出成果
    ├── interactive_dashboard.html     │  🌟 主看板（5 页 · 32 图表 · 全中文）
    ├── analysis_report.md             │  分析报告（Markdown）
    └── analysis_report.txt            │  分析报告（纯文本）
    
```

---

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.10
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行分析

```bash
# Step 1: 数据清洗
python scripts/1_data_cleaning.py

# Step 2: 用户画像 + RFM + Cohort
python scripts/2_user_profiling.py

# Step 3: 生成交互式看板
python scripts/build_dashboard.py

# Step 4: 生成分析报告
python scripts/4_report.py
```

### 查看结果

直接用浏览器打开：

```
outputs/interactive_dashboard.html
```

---

## 📊 核心发现

### 🎯 关键指标


| 指标         | 数值            | 评价          |
| -------------- | ----------------- | --------------- |
| 总用户       | **4,312**       | —            |
| 总营收       | **£8,832,003** | —            |
| 复购率       | **67.1%**       | ✅ 良好       |
| 次月留存率   | **18.9%**       | ⚠️ 偏低     |
| Top 20% 贡献 | **73.5%**       | ⚠️ 过度集中 |
| 仅购 1 次    | **32.9%**       | 🔴 最大流失点 |

### 🔴 五大关键问题


| # | 问题                                              | 优先级 |
| --- | --------------------------------------------------- | -------- |
| 1 | **新用户首月流失严重** — 32.9% 仅购买一次        | P0     |
| 2 | **营收高度集中** — 10.6% 冠军用户贡献 50.7% 营收 | P0     |
| 3 | 市场过度集中 — 英国占 92%                        | P2     |
| 4 | 假期新客质量偏低 — 11-12 月新客留存显著低于平均  | P1     |
| 5 | 潜力用户转化不足 — 仅 5.4% 新用户转化为潜力用户  | P2     |

### 💡 六大运营建议

1. **新用户首月激活**（24h/7天/21天 三步触达，目标留存率 30%+）
2. **VIP 分层体系**（银/金/铂金三级，年流失率 <10%）
3. **流失用户召回**（90 天自动触发优惠券，514 人 = £122 万可挽回）
4. **假期新客差异化**（1 月专属"新年回访"优惠）
5. **欧洲市场拓展**（优先德国、法国，本地化运营）
6. **复购积分激励**（第 2 单免运费、第 5 单 5% 折扣）

---

## 🖥️ 交互式看板

**单页 HTML · 5 个页面 · 32 个图表 · 全中文 · 无需服务器**


| 页面                  | 内容                                       | 图表数 |
| ----------------------- | -------------------------------------------- | -------- |
| **1. 用户概览**       | MAU 趋势、营收、地区分布、消费画像、客单价 | 6      |
| **2. 用户价值 (RFM)** | 分群分布、营收占比、气泡图、KMeans、评分   | 6      |
| **3. 生命周期与留存** | 阶段分布、Cohort 热力图、复购率、LTV       | 6      |
| **4. 运营策略**       | 用户漏斗、Pareto、流失画像、国家矩阵、迁移 | 6      |
| **5. KPI 总览**       | 4 个仪表盘 + 营收趋势 + 热力图 + TOP 商品  | 8      |

**交互功能：**

- 🔄 顶部 Tab 切换页面
- 🖱️ 单击图例筛选系列，双击隔离
- 🔍 悬停查看详细数据
- 📐 缩放/平移/下载 PNG

---

## 🔬 分析方法

### RFM 模型

- **R**ecency（最近购买时间）· **F**requency（购买频率）· **M**onetary（消费金额）
- 四分位评分 + 规则分层（7 类用户）+ KMeans 聚类（K=5）

### Cohort 留存分析

- 按月划分用户队列，计算 M+0 ~ M+12 留存率
- 识别流失节点和季节性留存模式

### 用户漏斗

- 首次购买 → 二次 → 多次 → 高频 → 高价值
- 量化每个转化节点的流失率

---

## 📄 数据来源

**Online Retail II** — UCI Machine Learning Repository[archive.ics.uci.edu/dataset/502/online+retail+ii](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

- 时间范围：2009-12-01 ~ 2010-12-09
- 原始记录：525,461 条
- 清洗后：407,664 条（77.6%）

---

## 📝 License

MIT License — 数据集归 UCI 所有，分析代码可自由使用。
