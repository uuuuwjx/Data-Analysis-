from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="基于 RFM 模型的电商用户价值分层与业务健康度分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"


@st.cache_data
def load_data() -> dict:
    data: dict = {}
    data["clean_report"] = pd.read_csv(TABLES_DIR / "data_cleaning_report.csv")
    data["monthly_revenue"] = pd.read_csv(TABLES_DIR / "monthly_revenue_orders.csv")
    data["country_summary"] = pd.read_csv(TABLES_DIR / "country_summary.csv")
    data["rfm_raw"] = pd.read_csv(TABLES_DIR / "customer_rfm_raw.csv")
    data["rfm_clustered"] = pd.read_csv(TABLES_DIR / "customer_rfm_clustered.csv")
    data["cluster_profile"] = pd.read_csv(TABLES_DIR / "cluster_profile.csv")
    data["k_metrics"] = pd.read_csv(TABLES_DIR / "k_selection_metrics.csv")

    with open(REPORTS_DIR / "business_insights.md", "r", encoding="utf-8") as f:
        data["business_insights"] = f.read()

    data["images"] = {
        "monthly": Image.open(FIGURES_DIR / "monthly_revenue_trend.png"),
        "country": Image.open(FIGURES_DIR / "top10_country_revenue.png"),
        "rfm": Image.open(FIGURES_DIR / "rfm_distributions.png"),
        "k": Image.open(FIGURES_DIR / "k_selection.png"),
        "heatmap": Image.open(FIGURES_DIR / "cluster_profile_heatmap.png"),
        "pca": Image.open(FIGURES_DIR / "segment_pca_scatter.png"),
    }

    data["total_customers"] = int(data["rfm_raw"]["customer_id"].nunique())
    data["total_revenue"] = float(data["rfm_raw"]["monetary"].sum())
    data["avg_frequency"] = float(data["rfm_raw"]["frequency"].mean())
    data["avg_recency"] = float(data["rfm_raw"]["recency"].mean())
    data["total_months"] = int(len(data["monthly_revenue"]))
    return data


def extract_section(md_text: str, section_no: int) -> str:
    pattern = rf"^##\s*{section_no}\."
    lines = md_text.splitlines()

    start = None
    end = None
    for i, line in enumerate(lines):
        if re.match(pattern, line.strip()):
            start = i
            continue
        if start is not None and line.strip().startswith("## "):
            end = i
            break

    if start is None:
        return ""
    if end is None:
        end = len(lines)
    return "\n".join(lines[start:end]).strip()


def render_sidebar(data: dict) -> str:
    st.sidebar.title("导航")
    page = st.sidebar.radio(
        "选择页面",
        ["项目概览", "数据概览", "RFM 分析", "聚类分析", "业务洞察", "客户查询"],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("关键指标")
    st.sidebar.metric("总客户数", f"{data['total_customers']:,}")
    st.sidebar.metric("总营收", f"£{data['total_revenue']:,.0f}")
    st.sidebar.metric("平均购买频次", f"{data['avg_frequency']:.2f}")
    st.sidebar.metric("平均沉默天数", f"{data['avg_recency']:.0f}")

    st.sidebar.markdown("---")
    st.sidebar.caption("数据范围：2009-12 至 2011-12 | 方法：RFM + K-means")
    return page


def render_home(data: dict) -> None:
    st.title("基于 RFM 模型的电商用户价值分层与业务健康度分析")
    st.write("本看板展示从数据清洗、RFM 构建、K-means 分群到业务建议的完整流程。")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总客户数", f"{data['total_customers']:,}")
    c2.metric("分析月份", f"{data['total_months']}")
    c3.metric("总营收", f"£{data['total_revenue']:,.0f}")
    c4.metric("分群数", f"{int(data['cluster_profile'].shape[0])}")

    st.markdown("---")
    st.subheader("分群概览")
    profile = data["cluster_profile"].copy()
    profile["customer_pct"] = (profile["customer_pct"] * 100).round(1)
    profile["monetary"] = profile["monetary"].round(2)
    st.dataframe(
        profile[["segment", "customers", "customer_pct", "recency", "frequency", "monetary"]].rename(
            columns={
                "segment": "客群",
                "customers": "客户数",
                "customer_pct": "占比(%)",
                "recency": "平均沉默天数",
                "frequency": "平均购买频次",
                "monetary": "平均消费金额",
            }
        ),
        use_container_width=True,
    )


def render_data_overview(data: dict) -> None:
    st.title("数据概览")

    report = dict(zip(data["clean_report"]["metric"], data["clean_report"]["value"]))
    c1, c2, c3 = st.columns(3)
    c1.metric("原始记录", f"{int(report.get('raw_rows', 0)):,}")
    c1.metric("清洗后记录", f"{int(report.get('clean_rows', 0)):,}")
    c2.metric("移除记录", f"{int(report.get('rows_removed', 0)):,}")
    c2.metric("退货记录", f"{int(report.get('canceled_rows', 0)):,}")
    c3.metric("客户数", f"{int(report.get('unique_customers', 0)):,}")
    c3.metric("订单数", f"{int(report.get('unique_invoices', 0)):,}")

    st.markdown("---")
    st.subheader("月度营收趋势")
    st.image(data["images"]["monthly"], use_container_width=True)

    st.markdown("---")
    st.subheader("国家收入 Top10")
    st.image(data["images"]["country"], use_container_width=True)


def render_rfm(data: dict) -> None:
    st.title("RFM 分析")
    st.markdown(
        "- `Recency`：距离最近一次购买的天数，越小越活跃\n"
        "- `Frequency`：累计购买频次，越大越忠诚\n"
        "- `Monetary`：累计消费金额，越大价值越高"
    )

    st.subheader("RFM 分布")
    st.image(data["images"]["rfm"], use_container_width=True)

    st.subheader("RFM 描述统计")
    stats = data["rfm_raw"][["recency", "frequency", "monetary"]].describe().round(2)
    st.dataframe(stats, use_container_width=True)


def render_cluster(data: dict) -> None:
    st.title("聚类分析")

    k_metrics = data["k_metrics"]
    best_k = int(k_metrics["selected_k"].iloc[0])
    best_row = k_metrics[k_metrics["k"] == best_k].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("最终 K", f"{best_k}")
    c2.metric("Silhouette", f"{best_row['silhouette']:.3f}")
    c3.metric("Inertia", f"{best_row['inertia']:.0f}")

    st.subheader("K 值评估")
    st.image(data["images"]["k"], use_container_width=True)

    st.subheader("分群画像热力图")
    st.image(data["images"]["heatmap"], use_container_width=True)

    st.subheader("PCA 二维分群")
    st.image(data["images"]["pca"], use_container_width=True)


def render_business(data: dict) -> None:
    st.title("业务洞察与建议动作")

    sec1 = extract_section(data["business_insights"], 1)
    sec2 = extract_section(data["business_insights"], 2)
    sec3 = extract_section(data["business_insights"], 3)

    if sec1:
        with st.expander("数据清洗结果", expanded=False):
            st.markdown(sec1)
    if sec2:
        with st.expander("核心发现", expanded=True):
            st.markdown(sec2)

    st.subheader("建议动作（来自报告第 3 节）")
    if sec3:
        st.markdown(sec3)
    else:
        st.warning("未在报告中解析到“## 3.”建议动作章节，请检查 outputs/reports/business_insights.md。")


def render_query(data: dict) -> None:
    st.title("客户查询")
    df = data["rfm_clustered"].copy()

    cid = st.number_input(
        "输入客户 ID",
        min_value=int(df["customer_id"].min()),
        max_value=int(df["customer_id"].max()),
        value=int(df["customer_id"].iloc[0]),
        step=1,
    )

    result = df[df["customer_id"] == cid]
    if result.empty:
        st.warning("未找到该客户。")
        return

    r = result.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("客户 ID", f"{int(r['customer_id'])}")
    c2.metric("客群", f"{r['segment']}")
    c3.metric("Recency", f"{int(r['recency'])}")
    c4.metric("Frequency", f"{int(r['frequency'])}")
    st.metric("Monetary", f"£{r['monetary']:,.2f}")


def main() -> None:
    try:
        data = load_data()
    except Exception as e:
        st.error(f"数据加载失败：{e}")
        st.stop()

    page = render_sidebar(data)

    if page == "项目概览":
        render_home(data)
    elif page == "数据概览":
        render_data_overview(data)
    elif page == "RFM 分析":
        render_rfm(data)
    elif page == "聚类分析":
        render_cluster(data)
    elif page == "业务洞察":
        render_business(data)
    else:
        render_query(data)


if __name__ == "__main__":
    main()
