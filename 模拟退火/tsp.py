"""
作业1：模拟退火算法（TSP）

功能：
1) 使用学号生成 50 座城市坐标（坐标在 [0,1000]x[0,1000]）
2) 以模拟退火算法解决 TSP（邻域：随机交换两城市）
3) 对比 alpha = 0.85 / 0.92 / 0.99，并记录迭代次数与最优路径长度
4) 保存：
   - 最优路线图（城市点 + 连线）
   - 收敛曲线图（best_length 随迭代次数变化）
   - 汇总 CSV（alpha、迭代次数、最优路径长度）
"""

from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt


# =========================
# 填写的信息
# =========================
STUDENT_ID: Optional[int] = 125130024325

SA_RNG_SEED: Optional[int] = None


# =========================
# 作业参数
# =========================

N_CITIES = 50
COORD_LOW, COORD_HIGH = 0.0, 1000.0

ALPHAS = [0.85, 0.92, 0.99]

TMIN = 1e-3       # 最小温度
INNER_ITER = 200  # 每次降温前尝试邻域的次数（影响“迭代次数”与耗时）
STORE_EVERY = 50  # 收敛曲线采样频率


# =========================
# 数据生成
# =========================

def extract_last_k_digits(student_id: int, k: int) -> int:
    """提取学号后 k 位"""
    return student_id % (10**k)


def generate_cities(student_id: int, n: int = N_CITIES) -> np.ndarray:
    """
    按作业要求生成城市坐标。
    - 随机种子：学号最后 5 位
    - 坐标范围：1000x1000 的二维平面（这里用 [0,1000] 均匀分布）
    """
    seed = extract_last_k_digits(student_id, 5)
    rng = np.random.default_rng(seed)
    x = rng.uniform(COORD_LOW, COORD_HIGH, n)
    y = rng.uniform(COORD_LOW, COORD_HIGH, n)
    return np.column_stack((x, y))


def initial_temperature(student_id: int) -> float:
    """
    初始温度：T0 = 1000 + (学号后两位) * 20
    """
    last_two = extract_last_k_digits(student_id, 2)
    return 1000.0 + float(last_two) * 20.0


# =========================
# TSP 评价函数
# =========================

def route_length(route: List[int], cities: np.ndarray) -> float:
    """
    计算给定路径（城市访问顺序）的回路总长度：
    """
    ordered = cities[np.array(route, dtype=int)]
    # differences between city i and city i+1 (with wrap-around)
    diffs = ordered - np.roll(ordered, shift=-1, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    return float(seg_lengths.sum())


# =========================
# 模拟退火（Metropolis 接受准则）
# =========================

@dataclass
class SAResult:
    alpha: float
    best_length: float
    best_route: List[int]
    iterations: int
    curve_x: List[int]
    curve_y: List[float]


def simulated_annealing_tsp(
    cities: np.ndarray,
    t0: float,
    alpha: float,
    inner_iter: int = INNER_ITER,
    tmin: float = TMIN,
    store_every: int = STORE_EVERY,
    rng_seed: Optional[int] = None,
) -> SAResult:
    """
    SA 核心：
    - 邻域：随机交换路径中的两个城市
    - Metropolis 接受准则：
        delta < 0  -> 接受
        delta > 0  -> 以 P = exp(-delta / T) 的概率接受
    - 降温：T <- alpha * T（alpha 越大，降温越慢）
    """
    if rng_seed is not None:
        # numpy 的 legacy RNG seed 需要落在 [0, 2**32 - 1] 内；
        # 学号可能很大，因此这里做一个安全归一化，保证可复现且不会报错。
        seed32 = int(rng_seed) % (2**32 - 1)
        random.seed(seed32)
        np.random.seed(seed32)

    n = len(cities)
    # Initial permutation
    current_route = list(np.random.permutation(n).astype(int))
    current_len = route_length(current_route, cities)

    best_route = current_route[:]
    best_len = current_len

    t = float(t0)
    iterations = 0
    curve_x: List[int] = []
    curve_y: List[float] = []

    while t > tmin:
        for _ in range(inner_iter):
            iterations += 1

            # Neighbor generation: swap two positions
            i, j = random.sample(range(n), 2)
            new_route = current_route[:]
            new_route[i], new_route[j] = new_route[j], new_route[i]

            new_len = route_length(new_route, cities)
            delta = new_len - current_len

            # Metropolis acceptance
            if delta < 0.0:
                accept = True
            else:
                # If delta >= 0, accept with probability exp(-delta / T)
                # T is always > 0 since we stop when T <= tmin
                p = math.exp(-delta / t)
                accept = random.random() < p

            if accept:
                current_route = new_route
                current_len = new_len

                if current_len < best_len:
                    best_len = current_len
                    best_route = current_route[:]

            # Save convergence curve (sampled)
            if iterations % store_every == 0:
                curve_x.append(iterations)
                curve_y.append(best_len)

        t *= alpha

    # Ensure the last point exists on the curve
    if not curve_x or curve_x[-1] != iterations:
        curve_x.append(iterations)
        curve_y.append(best_len)

    return SAResult(
        alpha=alpha,
        best_length=best_len,
        best_route=best_route,
        iterations=iterations,
        curve_x=curve_x,
        curve_y=curve_y,
    )


# =========================
# 可视化
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_route(
    cities: np.ndarray,
    route: List[int],
    title: str,
    out_path: str,
) -> None:
    """Plot cities and the closed TSP tour."""
    ordered = cities[np.array(route, dtype=int)]
    closed = np.vstack([ordered, ordered[0]])

    plt.figure(figsize=(8, 6))
    plt.plot(closed[:, 0], closed[:, 1], "b-", linewidth=1.2, label="tour")
    plt.scatter(cities[:, 0], cities[:, 1], c="red", s=25, zorder=3, label="cities")

    # Label city indices (small numbers for readability)
    for idx, (x, y) in enumerate(cities):
        plt.text(x, y, str(idx), fontsize=8, alpha=0.8)

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_convergence(
    curve_x: List[int],
    curve_y: List[float],
    title: str,
    out_path: str,
) -> None:
    """Plot best length vs iteration."""
    plt.figure(figsize=(8, 6))
    plt.plot(curve_x, curve_y, "g-", linewidth=1.5)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Best Tour Length")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_convergence_compare(
    results: List[SAResult],
    out_path: str,
) -> None:
    plt.figure(figsize=(8, 6))
    for r in results:
        plt.plot(r.curve_x, r.curve_y, linewidth=1.2, label=f"alpha={r.alpha}")
    plt.title("Convergence Compare (Best Length)")
    plt.xlabel("Iteration")
    plt.ylabel("Best Tour Length")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def alpha_to_tag(alpha: float) -> str:
    # turn 0.85 -> "0_85" for filenames
    return str(alpha).replace(".", "_")


def save_summary_csv(results: List[SAResult], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "iterations", "best_length"])
        for r in results:
            writer.writerow([r.alpha, r.iterations, r.best_length])


# =========================
# 主程序
# =========================

def main() -> None:
    global STUDENT_ID, SA_RNG_SEED

    if STUDENT_ID is None:
        raw = input("请输入你的完整学号（用于生成城市与计算 T0）：").strip()
        STUDENT_ID = int(raw)

    # Generate personalized cities
    cities = generate_cities(STUDENT_ID, n=N_CITIES)
    t0 = initial_temperature(STUDENT_ID)

    # For reproducibility across alphas, we can seed SA RNG per alpha.
    # If SA_RNG_SEED is None, we derive a seed from student_id + alpha tag.
    rng_base = SA_RNG_SEED if SA_RNG_SEED is not None else STUDENT_ID

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    ensure_dir(out_dir)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Initial Temperature T0: {t0:.4f}")
    print(f"Cities generated: {len(cities)} (seed uses last 5 digits of Student ID)")

    results: List[SAResult] = []
    for alpha in ALPHAS:
        # Derive deterministic RNG seed per alpha
        derived_seed = rng_base + int(alpha * 1_000_000)

        r = simulated_annealing_tsp(
            cities=cities,
            t0=t0,
            alpha=alpha,
            inner_iter=INNER_ITER,
            tmin=TMIN,
            store_every=STORE_EVERY,
            rng_seed=derived_seed,
        )
        results.append(r)

        print(f"[alpha={alpha}] iterations={r.iterations}, best_length={r.best_length:.4f}")

        # Save plots
        route_path = os.path.join(out_dir, f"route_alpha_{alpha_to_tag(alpha)}.png")
        curve_path = os.path.join(out_dir, f"convergence_alpha_{alpha_to_tag(alpha)}.png")

        plot_route(
            cities=cities,
            route=r.best_route,
            title=f"TSP (Simulated Annealing) - alpha={alpha}",
            out_path=route_path,
        )
        plot_convergence(
            curve_x=r.curve_x,
            curve_y=r.curve_y,
            title=f"Convergence - alpha={alpha}",
            out_path=curve_path,
        )

    # Save comparison curve + summary
    compare_curve_path = os.path.join(out_dir, "convergence_compare.png")
    plot_convergence_compare(results, out_path=compare_curve_path)

    summary_csv_path = os.path.join(out_dir, "summary.csv")
    save_summary_csv(results, out_path=summary_csv_path)

    print(f"\nAll results saved to: {out_dir}")
    best = min(results, key=lambda x: x.best_length)
    print(f"Best overall: alpha={best.alpha}, best_length={best.best_length:.4f}")


if __name__ == "__main__":
    main()

