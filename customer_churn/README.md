# 电信客户流失预测

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

一个面向客户留存场景的机器学习项目。项目基于 Telco Customer Churn 数据集，完成数据清洗、特征工程、不平衡分类处理、多模型对比、阈值优化、业务洞察生成和 Streamlit 可视化展示。

## 项目亮点

- 完成端到端机器学习流程：从原始数据到模型结果、图表、业务洞察文件自动生成
- 引入多模型对比: 'Logistic Regression'`Random Forest``HistGradientBoosting``XGBoost`
- 加入`5-fold` 交叉验证，并补充 `ROC-AUC`、`PR-AUC`、`Precision`、`Recall`、`F1`
- 使用 `SMOTE` 处理类别不平衡问题，并对最佳模型做分类阈值优化
- 输出业务洞察报告、特征重要性和多种评估图表，兼顾模型效果与业务解释
- 提供 Streamlit 中文展示页面，适合直接用于 GitHub 作品集和在线演示

## 当前结果

### 最优模型

- 默认阈值最优模型：`XGBoost`
- 测试集 `ROC-AUC = 0.8386`
- 测试集 `PR-AUC = 0.6515`
- 测试集 `Precision = 0.6175`
- 测试集 `Recall = 0.5481`
- 测试集 `F1 = 0.5807`

### 阈值优化结果

- 优化目标：`F1`
- 最优阈值：`0.3212`
- 优化后 `Recall = 0.7353`
- 优化后 `F1 = 0.6229`

### 交叉验证对比

| 模型 | CV ROC-AUC | CV F1 | CV PR-AUC |
| ---- | ---------- | ----- | --------- |
| Random Forest | 0.8443 | 0.6324 | 0.6442 |
| Logistic Regression | 0.8426 | 0.6180 | 0.6553 |
| XGBoost | 0.8397 | 0.5954 | 0.6535 |
| HistGradientBoosting | 0.8326 | 0.5741 | 0.6388 |

### 测试集对比

| 模型 | Accuracy | ROC-AUC | Precision | Recall | F1 | PR-AUC |
| ---- | -------- | ------- | --------- | ------ | -- | ------ |
| XGBoost | 0.7899 | 0.8386 | 0.6175 | 0.5481 | 0.5807 | 0.6515 |
| Random Forest | 0.7743 | 0.8369 | 0.5633 | 0.6658 | 0.6103 | 0.6269 |
| Logistic Regression | 0.7871 | 0.8365 | 0.6005 | 0.5909 | 0.5957 | 0.6193 |
| HistGradientBoosting | 0.7899 | 0.8308 | 0.6154 | 0.5561 | 0.5843 | 0.6355 |

## 业务结论

基于模型结果和分群分析，当前项目得到的核心发现包括：

- 月付合同客户流失率最高，达到 `42.7%`
- 使用电子支票支付的客户流失率最高，达到 `45.3%`
- 光纤用户流失率较高，达到 `41.9%`
- 未开通技术支持的客户流失率为 `41.6%`，显著高于已开通客户的 `15.2%`
- 未开通在线安全服务的客户流失率为 `41.8%`，显著高于已开通客户的 `14.6%`

可落地的运营建议：

- 优先针对月付合同和电子支票用户开展留存活动
- 将技术支持、在线安全服务与高风险套餐进行捆绑
- 针对新客户和低 tenure 客户设计 onboarding 与合同升级激励

## 数据集说明

- 数据来源：Kaggle Telco Customer Churn
- 样本量：`7043`
- 原始字段数：`21`
- 编码后特征数：`45`
- 目标变量：`Churn`
- 整体流失率：`26.54%`

关键字段：

- `tenure`：客户在网时长
- `MonthlyCharges`：月费用
- `TotalCharges`：累计费用
- `Contract`：合同类型
- `PaymentMethod`：支付方式
- `OnlineSecurity` / `TechSupport`：增值服务开通情况

## 方法流程

1. 数据预处理
   - 将 `TotalCharges` 转换为数值并填补缺失值
   - 将 `Churn` 映射为二分类标签
   - 对类别变量进行独热编码
   - 删除不具备预测意义的 `customerID`

2. 模型训练与选择
   - 使用分层抽样划分训练集和测试集
   - 在训练阶段引入 `SMOTE` 处理类别不平衡
   - 训练逻辑回归、随机森林、梯度提升、XGBoost 进行横向比较
   - 基于交叉验证和测试集指标共同选择最优模型

3. 模型评估与解释
   - 输出混淆矩阵、ROC 曲线、PR 曲线
   - 对最佳模型进行阈值优化，提升 Recall/F1
   - 生成特征重要性表格与图表
   - 输出业务洞察 `Markdown` 和 `JSON` 文件

## 项目结构

```text
customer_churn/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   └── Telco-Customer-Churn.csv
├── outputs/
│   ├── business_insights.json
│   ├── business_insights.md
│   ├── churn_distribution.png
│   ├── feature_importances.csv
│   ├── feature_importances.png
│   ├── model_comparison.csv
│   ├── model_results.json
│   ├── logistic_regression_*.png
│   ├── random_forest_*.png
│   ├── hist_gradient_boosting_*.png
│   ├── xgboost_*.png
│   ├── xgboost_tuned_threshold_*.png
│   ├── shap_feature_importance.png
│   └── shap_summary.png
└── src/
    ├── customer_churn.py
    └── churn_project/
        ├── __init__.py
        ├── app.py
        ├── config.py
        ├── data.py
        ├── evaluation.py
        ├── explainability.py
        ├── insights.py
        ├── models.py
        └── pipeline.py
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行训练与分析

```bash
python src/customer_churn.py
```

### 3. 启动 Streamlit 展示页面

```bash
streamlit run src/churn_project/app.py
```

## 关键输出文件

- `outputs/model_results.json`：完整模型结果、阈值优化结果、特征重要性与业务洞察
- `outputs/model_comparison.csv`：多模型交叉验证对比表
- `outputs/business_insights.md`：业务洞察与运营建议
- `outputs/feature_importances.csv`：特征重要性排序
- `outputs/*.png`：ROC、PR、混淆矩阵、特征重要性等可视化结

## 后续可继续优化

- 增加 `LightGBM` 等更强模型进行对比
- 加入超参数搜索，进一步提升 `PR-AUC` 和 `Recall`
- 增加模型持久化、API 部署和在线预测功能
- 补充单元测试与实验配置管理，进一步提升工程规范性

## 许可证

本项目采用 MIT License。
