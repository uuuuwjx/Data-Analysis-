import json
from pathlib import Path

import pandas as pd

SEGMENT_TRANSLATIONS = {
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

FEATURE_TRANSLATIONS = {
    "Contract_Month-to-month": "合同类型_月付合同",
    "OnlineSecurity_No": "在线安全_未开通",
    "TechSupport_No": "技术支持_未开通",
    "InternetService_Fiber optic": "网络服务_光纤",
    "OnlineBackup_No": "在线备份_未开通",
    "Dependents_Yes": "家属依赖_有",
    "PaymentMethod_Electronic check": "支付方式_电子支票",
    "TechSupport_Yes": "技术支持_已开通",
    "OnlineSecurity_Yes": "在线安全_已开通",
    "OnlineBackup_Yes": "在线备份_已开通",
}


def _format_rate(value: float) -> str:
    return f"{value * 100:.1f}%"



def _translate_label(value: str) -> str:
    return SEGMENT_TRANSLATIONS.get(value, value)



def _translate_feature(value: str) -> str:
    return FEATURE_TRANSLATIONS.get(value, value)



def build_business_insights(df: pd.DataFrame, feature_importances: pd.Series) -> dict:
    working = df.copy()
    working["ChurnFlag"] = working["Churn"].map({"No": 0, "Yes": 1})

    contract_rates = working.groupby("Contract")["ChurnFlag"].mean().sort_values(ascending=False)
    payment_rates = working.groupby("PaymentMethod")["ChurnFlag"].mean().sort_values(ascending=False)
    internet_rates = working.groupby("InternetService")["ChurnFlag"].mean().sort_values(ascending=False)

    top_driver_list = [
        {"feature": feature, "importance": float(value)}
        for feature, value in feature_importances.head(10).items()
    ]

    return {
        "overall_churn_rate": float(working["ChurnFlag"].mean()),
        "highest_risk_contract": {
            "segment": contract_rates.index[0],
            "churn_rate": float(contract_rates.iloc[0]),
        },
        "highest_risk_payment_method": {
            "segment": payment_rates.index[0],
            "churn_rate": float(payment_rates.iloc[0]),
        },
        "highest_risk_internet_service": {
            "segment": internet_rates.index[0],
            "churn_rate": float(internet_rates.iloc[0]),
        },
        "tech_support_gap": {
            "without_support": float(working.loc[working["TechSupport"] == "No", "ChurnFlag"].mean()),
            "with_support": float(working.loc[working["TechSupport"] == "Yes", "ChurnFlag"].mean()),
        },
        "online_security_gap": {
            "without_security": float(working.loc[working["OnlineSecurity"] == "No", "ChurnFlag"].mean()),
            "with_security": float(working.loc[working["OnlineSecurity"] == "Yes", "ChurnFlag"].mean()),
        },
        "top_model_drivers": top_driver_list,
    }



def save_business_insights(insights: dict, output_dir: Path) -> None:
    json_path = output_dir / "business_insights.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(insights, file, indent=2, ensure_ascii=False)
    print(f"Saved: {json_path}")

    markdown_path = output_dir / "business_insights.md"
    lines = [
        "# 业务洞察",
        "",
        f"- 整体客户流失率：{_format_rate(insights['overall_churn_rate'])}",
        (
            f"- 流失风险最高的合同类型：{_translate_label(insights['highest_risk_contract']['segment'])} "
            f"({_format_rate(insights['highest_risk_contract']['churn_rate'])})"
        ),
        (
            f"- 流失风险最高的支付方式：{_translate_label(insights['highest_risk_payment_method']['segment'])} "
            f"({_format_rate(insights['highest_risk_payment_method']['churn_rate'])})"
        ),
        (
            f"- 流失风险最高的网络服务类型：{_translate_label(insights['highest_risk_internet_service']['segment'])} "
            f"({_format_rate(insights['highest_risk_internet_service']['churn_rate'])})"
        ),
        (
            f"- 未开通技术支持的客户流失率为 {_format_rate(insights['tech_support_gap']['without_support'])}，"
            f"高于已开通技术支持客户的 {_format_rate(insights['tech_support_gap']['with_support'])}。"
        ),
        (
            f"- 未开通在线安全服务的客户流失率为 {_format_rate(insights['online_security_gap']['without_security'])}，"
            f"高于已开通在线安全服务客户的 {_format_rate(insights['online_security_gap']['with_security'])}。"
        ),
        "",
        "## 运营建议",
        "",
        "- 优先针对月付合同客户和电子支票用户开展留存活动。",
        "- 将技术支持和在线安全服务与高风险套餐进行捆绑，降低流失风险。",
        "- 针对新客户设计 onboarding 和合同升级激励。",
        "",
        "## 主要模型驱动因素",
        "",
    ]
    lines.extend(
        [f"- {_translate_feature(item['feature'])}：{item['importance']:.4f}" for item in insights["top_model_drivers"]]
    )

    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {markdown_path}")
