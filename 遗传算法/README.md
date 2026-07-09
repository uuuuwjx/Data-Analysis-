# 遗传算法作业2（TSP）

本目录实现“作业2：遗传算法”，使用与作业1（模拟退火）一致的50城市坐标求解TSP，并完成参数对比实验。

## 目录结构

```text
遗传算法/
├── ga_tsp.py                  # 主程序：GA求解TSP + 参数对比 + 可视化
├── README.md                  # 项目说明
├── 作业2_实验报告.pdf        # 实验报告
└── results/
    ├── cities.csv             # 本实验使用的50城市坐标
    ├── summary_ga.csv         # 参数实验汇总（标签/参数/最优路径长度）
    ├── route_best_ga.png      # GA最优路径图
    ├── convergence_compare_ga.png
    ├── convergence_pop80_mut005.png
    ├── convergence_pop200_mut005.png
    ├── convergence_pop120_mut002.png
    └── convergence_pop120_mut010.png
```

## 代码说明

- `ga_tsp.py`包含以下核心模块：
- 数据模块：优先读取`../模拟退火/results/cities.csv`，确保两次作业坐标一致；若不存在则按学号规则重生。
- 编码与评价：路径采用置换编码，目标函数为闭合路径总长度。
- GA操作：锦标赛选择（k=3）、OX交叉、交换变异、精英保留、代际更新。
- 终止条件：达到最大代数或连续`patience`代最优值无改进。
- 输出模块：保存各参数组收敛曲线、最优路线图和汇总CSV。

## 运行方式

```bash
python ga_tsp.py
```

运行后仅更新`results/`中的实验数据与图像；GA与SA对比分析放在`作业2_实验报告.docx`正文，不在`results`额外生成对比表。
