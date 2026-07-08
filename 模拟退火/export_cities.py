"""
导出个性化城市坐标数据（cities.csv）

根据 tsp.py 中的 STUDENT_ID 生成 N=50 个城市坐标，并保存到：
    results/cities.csv

字段：
    city_index, x, y
"""

from __future__ import annotations

import csv
import os

import tsp


def main() -> None:
    student_id = tsp.STUDENT_ID
    if student_id is None:
        raise ValueError("tsp.py 中的 STUDENT_ID 仍为 None，请先填写你的真实学号。")

    base_dir = os.path.dirname(tsp.__file__)
    out_dir = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    cities = tsp.generate_cities(student_id, n=tsp.N_CITIES)
    out_path = os.path.join(out_dir, "cities.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["city_index", "x", "y"])
        for i, (x, y) in enumerate(cities):
            writer.writerow([i, float(x), float(y)])

    print(f"已导出城市坐标：{out_path}")


if __name__ == "__main__":
    main()

