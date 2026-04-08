from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
sns.set_theme(style="whitegrid")


def ensure_dirs(base: Path) -> Dict[str, Path]:
    outputs = base / "outputs"
    dirs = {
        "figures": outputs / "figures",
        "tables": outputs / "tables",
        "reports": outputs / "reports",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def load_raw_data(file_path: Path) -> pd.DataFrame:
    excel = pd.ExcelFile(file_path)
    frames = []
    for sheet in excel.sheet_names:
        df = excel.parse(sheet)
        df["source_sheet"] = sheet
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]
    return data


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows = len(df)
    df = df.copy()

    df["invoice"] = df["invoice"].astype(str).str.strip()
    df["stockcode"] = df["stockcode"].astype(str).str.strip()
    df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    df = df.dropna(subset=["customer_id", "invoicedate", "invoice", "price", "quantity"])
    df = df.drop_duplicates()

    canceled = df[df["invoice"].str.startswith("C", na=False)].copy()
    df = df[~df["invoice"].str.startswith("C", na=False)]

    df = df[(df["quantity"] > 0) & (df["price"] > 0)]
    df["amount"] = df["quantity"] * df["price"]

    clean_report = pd.DataFrame(
        {
            "metric": [
                "raw_rows",
                "clean_rows",
                "rows_removed",
                "canceled_rows",
                "unique_customers",
                "unique_invoices",
            ],
            "value": [
                raw_rows,
                len(df),
                raw_rows - len(df),
                len(canceled),
                df["customer_id"].nunique(),
                df["invoice"].nunique(),
            ],
        }
    )
    return df, clean_report


def save_eda(df: pd.DataFrame, dirs: Dict[str, Path]) -> None:
    tables_dir = dirs["tables"]
    figures_dir = dirs["figures"]

    missing = df.isna().sum().sort_values(ascending=False)
    missing.to_csv(tables_dir / "missing_values.csv", header=["missing_count"])

    desc = df[["quantity", "price", "amount"]].describe().T
    desc.to_csv(tables_dir / "transaction_descriptive_stats.csv")

    country = (
        df.groupby("country", as_index=False)
        .agg(
            customers=("customer_id", "nunique"),
            invoices=("invoice", "nunique"),
            revenue=("amount", "sum"),
        )
        .sort_values("revenue", ascending=False)
    )
    country.to_csv(tables_dir / "country_summary.csv", index=False)

    monthly = (
        df.assign(month=df["invoicedate"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(revenue=("amount", "sum"), orders=("invoice", "nunique"))
    )
    monthly.to_csv(tables_dir / "monthly_revenue_orders.csv", index=False)

    plt.figure(figsize=(11, 5))
    plt.plot(monthly["month"], monthly["revenue"], marker="o", linewidth=2)
    plt.xticks(rotation=75)
    plt.title("Monthly Revenue Trend")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(figures_dir / "monthly_revenue_trend.png", dpi=160)
    plt.close()

    top10_country = country.head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=top10_country, x="revenue", y="country", palette="Blues_r")
    plt.title("Top 10 Countries by Revenue")
    plt.xlabel("Revenue")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig(figures_dir / "top10_country_revenue.png", dpi=160)
    plt.close()


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["invoicedate"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("customer_id", as_index=False)
        .agg(
            recency=("invoicedate", lambda x: (snapshot_date - x.max()).days),
            frequency=("invoice", "nunique"),
            monetary=("amount", "sum"),
        )
        .astype({"recency": "int64", "frequency": "int64", "monetary": "float64"})
    )
    return rfm


def preprocess_for_cluster(rfm: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    rfm_proc = rfm.copy()
    for col in ["recency", "frequency", "monetary"]:
        low, high = rfm_proc[col].quantile([0.01, 0.99])
        rfm_proc[col] = rfm_proc[col].clip(low, high)

    transformed = pd.DataFrame(
        {
            "recency": np.log1p(rfm_proc["recency"]),
            "frequency": np.log1p(rfm_proc["frequency"]),
            "monetary": np.log1p(rfm_proc["monetary"]),
        }
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(transformed)
    return transformed, X


def choose_k_and_cluster(X: np.ndarray) -> Tuple[pd.DataFrame, int, np.ndarray]:
    records = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels)
        records.append({"k": k, "inertia": model.inertia_, "silhouette": sil})

    metrics = pd.DataFrame(records)

    global_best = metrics.loc[metrics["silhouette"].idxmax()]
    business_metrics = metrics[metrics["k"] >= 3]
    business_best = business_metrics.loc[business_metrics["silhouette"].idxmax()]

    # 规则: 若 k=2 仅略优于 k>=3 的方案，则优先可运营分层
    if int(global_best["k"]) == 2 and (global_best["silhouette"] - business_best["silhouette"]) <= 0.08:
        best_k = int(business_best["k"])
    else:
        best_k = int(global_best["k"])

    final_model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    final_labels = final_model.fit_predict(X)
    return metrics, best_k, final_labels


def name_segments(profile: pd.DataFrame) -> Dict[int, str]:
    segment_names: Dict[int, str] = {}

    r35 = profile["recency"].quantile(0.35)
    r70 = profile["recency"].quantile(0.70)
    f35 = profile["frequency"].quantile(0.35)
    f65 = profile["frequency"].quantile(0.65)
    m35 = profile["monetary"].quantile(0.35)
    m65 = profile["monetary"].quantile(0.65)

    for _, row in profile.iterrows():
        c = int(row["cluster"])
        r, f, m = row["recency"], row["frequency"], row["monetary"]
        if r <= r35 and f >= f65 and m >= m65:
            segment_names[c] = "Champions"
        elif r <= profile["recency"].median() and f >= profile["frequency"].median() and m >= profile["monetary"].median():
            segment_names[c] = "Loyal Customers"
        elif r >= r70 and f <= f35:
            segment_names[c] = "At Risk"
        elif m <= m35 and f <= f35:
            segment_names[c] = "Low Value"
        else:
            segment_names[c] = "Potential Growth"

    # 防止出现名称重复导致展示混淆
    used = {}
    for k, v in list(segment_names.items()):
        if v not in used:
            used[v] = 1
            continue
        used[v] += 1
        segment_names[k] = f"{v} {used[v]}"

    return segment_names


def make_visuals(rfm: pd.DataFrame, metrics: pd.DataFrame, dirs: Dict[str, Path]) -> None:
    figures_dir = dirs["figures"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["recency", "frequency", "monetary"]):
        sns.histplot(rfm[col], bins=35, kde=True, ax=ax, color="#2f7ed8")
        ax.set_title(f"Distribution of {col.title()}")
    plt.tight_layout()
    plt.savefig(figures_dir / "rfm_distributions.png", dpi=160)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(metrics["k"], metrics["inertia"], marker="o", color="#1f77b4")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(metrics["k"], metrics["silhouette"], marker="s", color="#ff7f0e")
    ax2.set_ylabel("Silhouette", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    plt.title("K Selection: Elbow + Silhouette")
    plt.tight_layout()
    plt.savefig(figures_dir / "k_selection.png", dpi=160)
    plt.close()


def make_cluster_outputs(
    rfm: pd.DataFrame,
    X: np.ndarray,
    labels: np.ndarray,
    metrics: pd.DataFrame,
    best_k: int,
    dirs: Dict[str, Path],
) -> pd.DataFrame:
    tables_dir = dirs["tables"]
    figures_dir = dirs["figures"]

    rfm_clustered = rfm.copy()
    rfm_clustered["cluster"] = labels

    cluster_profile = (
        rfm_clustered.groupby("cluster", as_index=False)
        .agg(
            customers=("customer_id", "count"),
            recency=("recency", "mean"),
            frequency=("frequency", "mean"),
            monetary=("monetary", "mean"),
        )
        .sort_values("monetary", ascending=False)
    )

    name_map = name_segments(cluster_profile)
    rfm_clustered["segment"] = rfm_clustered["cluster"].map(name_map)
    cluster_profile["segment"] = cluster_profile["cluster"].map(name_map)
    cluster_profile["customer_pct"] = cluster_profile["customers"] / cluster_profile["customers"].sum()

    metrics_out = metrics.copy()
    metrics_out["selected_k"] = best_k

    metrics_out.to_csv(tables_dir / "k_selection_metrics.csv", index=False)
    rfm_clustered.to_csv(tables_dir / "customer_rfm_clustered.csv", index=False)
    cluster_profile.to_csv(tables_dir / "cluster_profile.csv", index=False)

    profile_for_heatmap = cluster_profile.set_index("segment")[["recency", "frequency", "monetary"]]
    profile_z = (profile_for_heatmap - profile_for_heatmap.mean()) / profile_for_heatmap.std(ddof=0)

    plt.figure(figsize=(8, 5))
    sns.heatmap(profile_z, annot=True, fmt=".2f", cmap="RdYlGn_r", center=0)
    plt.title("Cluster Profile (Z-score)")
    plt.tight_layout()
    plt.savefig(figures_dir / "cluster_profile_heatmap.png", dpi=160)
    plt.close()

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    pca_df = pd.DataFrame(coords, columns=["pc1", "pc2"])
    pca_df["segment"] = rfm_clustered["segment"].values

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=pca_df, x="pc1", y="pc2", hue="segment", alpha=0.65, s=30, palette="tab10")
    plt.title("Customer Segments in PCA Space")
    plt.tight_layout()
    plt.savefig(figures_dir / "segment_pca_scatter.png", dpi=160)
    plt.close()

    return cluster_profile


def generate_business_report(clean_report: pd.DataFrame, cluster_profile: pd.DataFrame, best_k: int, dirs: Dict[str, Path]) -> None:
    reports_dir = dirs["reports"]

    cp = cluster_profile.sort_values("monetary", ascending=False).reset_index(drop=True)
    top_seg = cp.iloc[0]
    low_seg = cp.sort_values("monetary", ascending=True).iloc[0]

    lines = [
        "# 基于 RFM 模型的电商用户价值分层与业务健康度分析报告",
        "",
        "## 1. 数据清洗结果",
    ]
    for _, row in clean_report.iterrows():
        lines.append(f"- {row['metric']}: {row['value']}")

    lines.extend(
        [
            "",
            "## 2. 用户分层核心发现",
            f"- 本次分群采用 K-means，最终选择 **K={best_k}**。",
            f"- 高价值主力分群：**{top_seg['segment']}**，人群占比约 **{top_seg['customer_pct']:.1%}**，客均消费约 **{top_seg['monetary']:.2f}**。",
            f"- 低价值分群：**{low_seg['segment']}**，客均消费约 **{low_seg['monetary']:.2f}**，建议优先做低成本唤醒与加购引导。",
            "- 业务健康度可用 Recency（活跃度）、Frequency（粘性）、Monetary（价值贡献）三维共同监控。",
            "",
            "## 3. 建议动作",
            "- Champions / Loyal Customers：会员权益、提前购、组合包提升客单与复购。",
            "- Potential Growth：优惠券门槛与个性化推荐，推动向高价值段迁移。",
            "- At Risk：针对沉默期用户做召回活动（限时折扣、关怀触达）。",
            "- Low Value：自动化触达与低成本运营，控制投放 ROI。",
            "",
            "## 4. 输出文件",
            "- `outputs/tables/customer_rfm_clustered.csv`：用户级 RFM 与分群结果",
            "- `outputs/tables/cluster_profile.csv`：分群画像",
            "- `outputs/tables/k_selection_metrics.csv`：K 值评估",
            "- `outputs/figures/*.png`：趋势、分布、聚类图",
        ]
    )

    (reports_dir / "business_insights.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    data_file = base / "online_retail_II.xlsx"

    dirs = ensure_dirs(base)
    raw = load_raw_data(data_file)
    clean, clean_report = clean_data(raw)
    save_eda(clean, dirs)

    rfm = build_rfm(clean)
    transformed, X = preprocess_for_cluster(rfm)

    metrics, best_k, labels = choose_k_and_cluster(X)

    rfm.to_csv(dirs["tables"] / "customer_rfm_raw.csv", index=False)
    transformed.to_csv(dirs["tables"] / "customer_rfm_transformed.csv", index=False)

    make_visuals(rfm, metrics, dirs)
    cluster_profile = make_cluster_outputs(rfm, X, labels, metrics, best_k, dirs)

    clean_report.to_csv(dirs["tables"] / "data_cleaning_report.csv", index=False)
    generate_business_report(clean_report, cluster_profile, best_k, dirs)

    print("Pipeline finished successfully.")
    print(f"Selected K: {best_k}")
    print("Outputs saved to:")
    print("- outputs/tables")
    print("- outputs/figures")
    print("- outputs/reports")


if __name__ == "__main__":
    main()
