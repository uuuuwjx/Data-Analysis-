import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "outputs"

MODEL_NAME_MAP = {
    "Logistic Regression": "逻辑回归",
    "Random Forest": "随机森林",
    "HistGradientBoosting": "直方图梯度提升",
    "XGBoost": "XGBoost",
    "XGBoost (Tuned Threshold)": "XGBoost（阈值优化后）",
}

SEGMENT_NAME_MAP = {
    "Month-to-month": "月付合同",
    "One year": "一年合同",
    "Two year": "两年合同",
    "Electronic check": "电子支票",
    "Mailed check": "邮寄支票",
    "Bank transfer (automatic)": "银行自动转账",
    "Credit card (automatic)": "信用卡自动扣款",
    "Fiber optic": "光纤",
    "DSL": "DSL",
    "No": "无",
}

COLUMN_NAME_MAP = {
    "model": "模型",
    "cv_roc_auc_mean": "交叉验证 ROC-AUC 均值",
    "cv_roc_auc_std": "交叉验证 ROC-AUC 标准差",
    "cv_f1_mean": "交叉验证 F1 均值",
    "cv_f1_std": "交叉验证 F1 标准差",
    "cv_precision_mean": "交叉验证 Precision 均值",
    "cv_precision_std": "交叉验证 Precision 标准差",
    "cv_recall_mean": "交叉验证 Recall 均值",
    "cv_recall_std": "交叉验证 Recall 标准差",
    "cv_average_precision_mean": "交叉验证 PR-AUC 均值",
    "cv_average_precision_std": "交叉验证 PR-AUC 标准差",
}


def localize_model_name(name: str) -> str:
    return MODEL_NAME_MAP.get(name, name)


def localize_segment(name: str) -> str:
    return SEGMENT_NAME_MAP.get(name, name)



def render_app() -> None:
    st.set_page_config(page_title="客户流失预测看板", layout="wide")
    st.title("电信客户流失预测看板")
    st.caption("展示模型对比结果、关键指标、业务洞察和可视化图表。")

    results_path = OUTPUT_DIR / "model_results.json"
    insights_path = OUTPUT_DIR / "business_insights.json"
    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    business_md_path = OUTPUT_DIR / "business_insights.md"

    if not results_path.exists():
        st.error("请先运行 `python src/customer_churn.py` 生成输出结果。")
        st.stop()

    results = json.loads(results_path.read_text(encoding="utf-8"))
    comparison_df = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    insights = json.loads(insights_path.read_text(encoding="utf-8")) if insights_path.exists() else {}
    business_md = business_md_path.read_text(encoding="utf-8") if business_md_path.exists() else ""

    best_model_key = results["best_model"]["model_key"]
    best_result = next(row for row in results["test_results"] if row["model_key"] == best_model_key)
    tuned_result = results.get("tuned_threshold_result")

    col1, col2, col3 = st.columns(3)
    col1.metric("当前最优模型", localize_model_name(results["best_model"]["model"]))
    col2.metric("默认阈值 ROC-AUC", f"{best_result['roc_auc']:.3f}")
    col3.metric("默认阈值 PR-AUC", f"{best_result['pr_auc']:.3f}")

    if tuned_result:
        col4, col5, col6 = st.columns(3)
        col4.metric("优化后阈值", f"{tuned_result['threshold']:.3f}")
        col5.metric("优化后 Recall", f"{tuned_result['recall']:.3f}")
        col6.metric("优化后 F1", f"{tuned_result['f1']:.3f}")

    st.subheader("模型对比")
    if not comparison_df.empty:
        display_df = comparison_df.copy()
        if "model" in display_df.columns:
            display_df["model"] = display_df["model"].map(localize_model_name)
        display_df = display_df.rename(columns=COLUMN_NAME_MAP)
        st.dataframe(display_df, width="stretch")

    st.subheader("业务洞察摘要")
    if insights:
        st.write(
            {
                "整体客户流失率": f"{insights['overall_churn_rate'] * 100:.1f}%",
                "高风险合同类型": {
                    "类别": localize_segment(insights["highest_risk_contract"]["segment"]),
                    "流失率": f"{insights['highest_risk_contract']['churn_rate'] * 100:.1f}%",
                },
                "高风险支付方式": {
                    "类别": localize_segment(insights["highest_risk_payment_method"]["segment"]),
                    "流失率": f"{insights['highest_risk_payment_method']['churn_rate'] * 100:.1f}%",
                },
                "高风险网络服务": {
                    "类别": localize_segment(insights["highest_risk_internet_service"]["segment"]),
                    "流失率": f"{insights['highest_risk_internet_service']['churn_rate'] * 100:.1f}%",
                },
            }
        )

    st.subheader("业务洞察报告")
    if business_md:
        st.markdown(business_md)

    st.subheader("可视化结果")
    image_cols = st.columns(2)
    image_caption_map = {
        f"{best_model_key}_roc_curve.png": "最优模型 ROC 曲线",
        f"{best_model_key}_pr_curve.png": "最优模型 PR 曲线",
        f"{best_model_key}_tuned_threshold_pr_curve.png": "阈值优化后 PR 曲线",
        "feature_importances.png": "特征重要性",
        "shap_summary.png": "SHAP 总结图",
    }

    for idx, image_name in enumerate(
        [
            f"{best_model_key}_roc_curve.png",
            f"{best_model_key}_pr_curve.png",
            f"{best_model_key}_tuned_threshold_pr_curve.png",
            "feature_importances.png",
            "shap_summary.png",
        ]
    ):
        image_path = OUTPUT_DIR / image_name
        if image_path.exists():
            image_cols[idx % 2].image(str(image_path), caption=image_caption_map.get(image_name, image_name))


render_app()
