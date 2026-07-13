# -*- coding: utf-8 -*-
"""
================================================================================
电商用户运营分析 — Phase 1: 数据理解与清洗
================================================================================
基于 skill.md 第一阶段规范
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 数据加载
# ============================================================================
print("=" * 70)
print("Phase 1: 数据理解与清洗")
print("=" * 70)

df = pd.read_excel("data/online_retail_II.xlsx")
df.columns = [c.strip() for c in df.columns]
print(f"\n原始数据规模: {df.shape[0]:,} 行 x {df.shape[1]} 列")

# ============================================================================
# 2. 数据概览报告
# ============================================================================
print("\n" + "-" * 50)
print("[数据概览报告]")
print("-" * 50)

print(f"数据规模:     {df.shape[0]:,} 条交易记录")
print(f"时间范围:     {df['InvoiceDate'].min()} ~ {df['InvoiceDate'].max()}")
print(f"用户数量:     {df['Customer ID'].nunique():,} (有ID)")
print(f"商品数量:     {df['StockCode'].nunique():,}")
print(f"订单数量:     {df['Invoice'].nunique():,}")
print(f"涉及国家:     {df['Country'].nunique()}")

print("\n字段类型:")
for col in df.columns:
    print(f"  {col:20s} : {df[col].dtype}")

# ============================================================================
# 3. 数据质量检查
# ============================================================================
print("\n" + "-" * 50)
print("[数据质量检查]")
print("-" * 50)

# 缺失值
print("\n缺失值统计:")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
for col in df.columns:
    if missing[col] > 0:
        print(f"  {col:20s}: {missing[col]:>8,} 条 ({missing_pct[col]}%)")

# 重点: Customer ID 缺失
print(f"\n[!] Customer ID 缺失 {df['Customer ID'].isnull().sum():,} 条 ({df['Customer ID'].isnull().mean()*100:.1f}%)")

# 异常值: 取消订单 (Invoice 以 'C' 开头)
cancel_mask = df['Invoice'].astype(str).str.startswith('C')
print(f"[!] 取消订单 (Invoice 以 C 开头): {cancel_mask.sum():,} 条 ({cancel_mask.mean()*100:.2f}%)")

# 无效交易
neg_qty = (df['Quantity'] <= 0).sum()
neg_price = (df['Price'] <= 0).sum()
print(f"[!] Quantity <= 0: {neg_qty:,} 条")
print(f"[!] Price <= 0:    {neg_price:,} 条")

# ============================================================================
# 4. 数据清洗
# ============================================================================
print("\n" + "-" * 50)
print("[数据清洗]")
print("-" * 50)

# Step 1: 删除 Customer ID 缺失的记录（用户分析需要）
df_clean = df.dropna(subset=['Customer ID']).copy()
df_clean['Customer ID'] = df_clean['Customer ID'].astype(int)
print(f"1) 删除 Customer ID 缺失: {len(df):,} -> {len(df_clean):,} 条 "
      f"(删除 {len(df)-len(df_clean):,} 条, {(1-len(df_clean)/len(df))*100:.1f}%)")

# Step 2: 标记并移除取消订单
cancel_mask = df_clean['Invoice'].astype(str).str.startswith('C')
cancel_count = cancel_mask.sum()
df_clean = df_clean[~cancel_mask].copy()
print(f"2) 移除取消订单:           -> {len(df_clean):,} 条 (删除 {cancel_count:,} 条)")

# Step 3: 移除无效交易 (Quantity <= 0 或 Price <= 0)
invalid = (df_clean['Quantity'] <= 0) | (df_clean['Price'] <= 0)
invalid_count = invalid.sum()
df_clean = df_clean[~invalid].copy()
print(f"3) 移除无效交易:           -> {len(df_clean):,} 条 (删除 {invalid_count:,} 条)")

print(f"\n[OK] 清洗后数据: {len(df_clean):,} 条 (保留 {len(df_clean)/len(df)*100:.1f}%)")

# ============================================================================
# 5. 特征构造
# ============================================================================
print("\n" + "-" * 50)
print("[特征构造]")
print("-" * 50)

# 交易金额
df_clean['Revenue'] = df_clean['Quantity'] * df_clean['Price']
print(f"  + Revenue = Quantity x Price (总营收: {df_clean['Revenue'].sum():,.2f})")

# 时间字段
df_clean['Year']   = df_clean['InvoiceDate'].dt.year
df_clean['Month']  = df_clean['InvoiceDate'].dt.month
df_clean['Week']   = df_clean['InvoiceDate'].dt.isocalendar().week.astype(int)
df_clean['Day']    = df_clean['InvoiceDate'].dt.day
df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M').astype(str)
print(f"  + 时间字段: Year, Month, Week, Day, YearMonth")

# 用户行为字段
reference_date = df_clean['InvoiceDate'].max()
print(f"\n参考日期 (分析基准): {reference_date}")

# 按订单聚合，计算每订单金额
order_rev = df_clean.groupby(['Customer ID', 'Invoice'])['Revenue'].sum().reset_index()

# 用户级别聚合
user_agg = df_clean.groupby('Customer ID').agg(
    FirstPurchase  = ('InvoiceDate', 'min'),
    LastPurchase   = ('InvoiceDate', 'max'),
    TotalOrders    = ('Invoice', 'nunique'),
    TotalQuantity  = ('Quantity', 'sum'),
    TotalRevenue   = ('Revenue', 'sum'),
).reset_index()

# 平均订单价值
avg_order = order_rev.groupby('Customer ID')['Revenue'].mean().reset_index()
avg_order.columns = ['Customer ID', 'AvgOrderValue']
user_agg = user_agg.merge(avg_order, on='Customer ID', how='left')

# 用户生命周期（天）
user_agg['LifeDays'] = (user_agg['LastPurchase'] - user_agg['FirstPurchase']).dt.days

print(f"  + 用户行为字段: FirstPurchase, LastPurchase, TotalOrders, TotalQuantity, TotalRevenue, AvgOrderValue, LifeDays")
print(f"  + 用户级数据集: {len(user_agg):,} 个用户")

# ============================================================================
# 6. 保存清洗后数据
# ============================================================================
print("\n" + "-" * 50)
print("[保存清洗后数据]")
print("-" * 50)

# 清洗后的交易数据
df_clean.to_csv("data/cleaned_transactions.csv", index=False)
print(f"  + cleaned_transactions.csv — {len(df_clean):,} 条交易记录")

# 用户指标表
user_agg.to_csv("data/user_metrics.csv", index=False)
print(f"  + user_metrics.csv — {len(user_agg):,} 个用户指标")

print("\n===== Phase 1 完成! =====")
print(f"\n数据摘要:")
print(f"  清洗后交易: {len(df_clean):,} 条")
print(f"  用户数量:   {len(user_agg):,} 人")
print(f"  时间范围:   {df_clean['InvoiceDate'].min().date()} ~ {df_clean['InvoiceDate'].max().date()}")
print(f"  总营收:     {df_clean['Revenue'].sum():,.2f}")
