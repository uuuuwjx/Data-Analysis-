# TSP 模拟退火求解器

本项目实现了使用**模拟退火算法**解决旅行商问题（TSP），并按要求对比不同降温速率（α）对算法性能的影响。

## 功能特性

- 根据学号后 5 位生成 50 座城市的随机坐标（范围 [0,1000]×[0,1000]）。
- 使用模拟退火算法搜索 TSP 最优路径，邻域操作采用随机交换两个城市。
- 对比多个降温速率（`alpha`），例如 0.85、0.92、0.99。
- 自动保存：
  - 最优路线图（城市点 + 连线）
  - 收敛曲线图（最优长度随迭代次数变化）
  - 汇总 CSV（包含 alpha、迭代次数、最优路径长度）
- 可固定随机种子以复现结果。

## 文件结构
|── tsp.py # 主程序代码
├── results/ # 运行后自动生成的结果目录
│ |── route_alpha_0_85.png
| |—— route_alpha_0_92.png
| |—— route_alpha_0_99.png
│ |── convergence_alpha_0_85.png
| |—— convergence_alpha_0_92.png
| |—— convergence_alpha_0_99.png
│ |── convergence_compare.png
│ |── summary.csv
| |__ cities.csv
|—— README.md # 本文件
|—— export_cities.py #保存 50 座城市坐标
|—— 实验报告_模拟退火.pdf