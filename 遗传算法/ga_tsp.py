from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Keep this consistent with Homework 1 setup.
STUDENT_ID = 125130024325
N_CITIES = 50
COORD_LOW, COORD_HIGH = 0.0, 1000.0

# Global reproducibility seed for GA internals.
GA_SEED = 20260411


def extract_last_k_digits(student_id: int, k: int) -> int:
    return student_id % (10**k)


def generate_cities(student_id: int, n: int = N_CITIES) -> np.ndarray:
    seed = extract_last_k_digits(student_id, 5)
    rng = np.random.default_rng(seed)
    x = rng.uniform(COORD_LOW, COORD_HIGH, n)
    y = rng.uniform(COORD_LOW, COORD_HIGH, n)
    return np.column_stack((x, y))


def load_cities_from_sa(repo_root: Path) -> np.ndarray:
    sa_csv = repo_root / "模拟退火" / "results" / "cities.csv"
    if sa_csv.exists():
        rows: List[Tuple[float, float]] = []
        with sa_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((float(row["x"]), float(row["y"])))
        arr = np.array(rows, dtype=float)
        if len(arr) == N_CITIES:
            return arr
    return generate_cities(STUDENT_ID, N_CITIES)


def save_cities_csv(cities: np.ndarray, out_csv: Path) -> None:
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["city_index", "x", "y"])
        for i, (x, y) in enumerate(cities):
            writer.writerow([i, float(x), float(y)])


def route_length(route: Sequence[int], cities: np.ndarray) -> float:
    ordered = cities[np.array(route, dtype=int)]
    diffs = ordered - np.roll(ordered, -1, axis=0)
    return float(np.sqrt((diffs**2).sum(axis=1)).sum())


@dataclass
class GAConfig:
    population_size: int = 120
    crossover_rate: float = 0.90
    mutation_rate: float = 0.05
    generations: int = 1200
    tournament_k: int = 3
    elite_count: int = 2
    patience: int = 250


@dataclass
class GAResult:
    label: str
    config: GAConfig
    best_route: List[int]
    best_length: float
    best_history: List[float]


def init_population(n_cities: int, pop_size: int, rng: random.Random) -> List[List[int]]:
    base = list(range(n_cities))
    population: List[List[int]] = []
    for _ in range(pop_size):
        perm = base[:]
        rng.shuffle(perm)
        population.append(perm)
    return population


def fitness_values(population: Sequence[Sequence[int]], cities: np.ndarray) -> Tuple[List[float], List[float]]:
    lengths = [route_length(ind, cities) for ind in population]
    fitness = [1.0 / (l + 1e-9) for l in lengths]
    return lengths, fitness


def tournament_select(population: Sequence[List[int]], fitness: Sequence[float], k: int, rng: random.Random) -> List[int]:
    idxs = [rng.randrange(len(population)) for _ in range(k)]
    best_idx = max(idxs, key=lambda i: fitness[i])
    return population[best_idx][:]


def ox_crossover(p1: Sequence[int], p2: Sequence[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    n = len(p1)
    a, b = sorted((rng.randrange(n), rng.randrange(n)))
    if a == b:
        b = (a + 1) % n
        if b < a:
            a, b = b, a

    def make_child(parent_a: Sequence[int], parent_b: Sequence[int]) -> List[int]:
        child = [-1] * n
        child[a:b] = parent_a[a:b]
        fill = [g for g in parent_b if g not in child]
        idx = 0
        for i in list(range(0, a)) + list(range(b, n)):
            child[i] = fill[idx]
            idx += 1
        return child

    return make_child(p1, p2), make_child(p2, p1)


def swap_mutation(route: List[int], rng: random.Random) -> None:
    i, j = rng.sample(range(len(route)), 2)
    route[i], route[j] = route[j], route[i]


def evolve_tsp(cities: np.ndarray, cfg: GAConfig, seed: int, label: str) -> GAResult:
    rng = random.Random(seed)
    n = len(cities)
    pop = init_population(n, cfg.population_size, rng)

    best_route: Optional[List[int]] = None
    best_len = math.inf
    best_history: List[float] = []
    stale = 0

    for _ in range(cfg.generations):
        lengths, fitness = fitness_values(pop, cities)
        ranked_idx = sorted(range(len(pop)), key=lambda i: lengths[i])

        if lengths[ranked_idx[0]] < best_len:
            best_len = lengths[ranked_idx[0]]
            best_route = pop[ranked_idx[0]][:]
            stale = 0
        else:
            stale += 1

        best_history.append(best_len)

        if stale >= cfg.patience:
            break

        next_pop: List[List[int]] = [pop[i][:] for i in ranked_idx[: cfg.elite_count]]

        while len(next_pop) < cfg.population_size:
            parent1 = tournament_select(pop, fitness, cfg.tournament_k, rng)
            parent2 = tournament_select(pop, fitness, cfg.tournament_k, rng)

            if rng.random() < cfg.crossover_rate:
                child1, child2 = ox_crossover(parent1, parent2, rng)
            else:
                child1, child2 = parent1[:], parent2[:]

            if rng.random() < cfg.mutation_rate:
                swap_mutation(child1, rng)
            if rng.random() < cfg.mutation_rate:
                swap_mutation(child2, rng)

            next_pop.append(child1)
            if len(next_pop) < cfg.population_size:
                next_pop.append(child2)

        pop = next_pop

    if best_route is None:
        raise RuntimeError("GA failed to produce a route")

    return GAResult(
        label=label,
        config=cfg,
        best_route=best_route,
        best_length=best_len,
        best_history=best_history,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_route(cities: np.ndarray, route: Sequence[int], title: str, out_path: Path) -> None:
    ordered = cities[np.array(route, dtype=int)]
    closed = np.vstack([ordered, ordered[0]])

    plt.figure(figsize=(8, 6))
    plt.plot(closed[:, 0], closed[:, 1], "b-", linewidth=1.1, label="tour")
    plt.scatter(cities[:, 0], cities[:, 1], c="red", s=22, zorder=3, label="cities")
    for idx, (x, y) in enumerate(cities):
        plt.text(x, y, str(idx), fontsize=7, alpha=0.8)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_convergence(result: GAResult, out_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    x = np.arange(1, len(result.best_history) + 1)
    plt.plot(x, result.best_history, "g-", linewidth=1.5)
    plt.title(f"GA Convergence - {result.label}")
    plt.xlabel("Generation")
    plt.ylabel("Best Tour Length")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_compare(results: Sequence[GAResult], out_path: Path) -> None:
    plt.figure(figsize=(9, 6))
    for r in results:
        x = np.arange(1, len(r.best_history) + 1)
        plt.plot(x, r.best_history, linewidth=1.3, label=r.label)
    plt.title("GA Parameter Comparison (Best Length)")
    plt.xlabel("Generation")
    plt.ylabel("Best Tour Length")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def load_sa_best(repo_root: Path) -> Optional[Tuple[float, str]]:
    sa_summary = repo_root / "模拟退火" / "results" / "summary.csv"
    if not sa_summary.exists():
        return None
    best_len = math.inf
    best_alpha = ""
    with sa_summary.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = float(row["best_length"])
            if val < best_len:
                best_len = val
                best_alpha = row.get("alpha", "")
    if not math.isfinite(best_len):
        return None
    return best_len, best_alpha


def save_ga_summary(results: Sequence[GAResult], out_csv: Path) -> None:
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "population_size",
                "crossover_rate",
                "mutation_rate",
                "generations_run",
                "best_length",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.label,
                    r.config.population_size,
                    r.config.crossover_rate,
                    r.config.mutation_rate,
                    len(r.best_history),
                    r.best_length,
                ]
            )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    results_dir = script_dir / "results"
    ensure_dir(results_dir)

    np.random.seed(GA_SEED)
    random.seed(GA_SEED)

    cities = load_cities_from_sa(repo_root)
    save_cities_csv(cities, results_dir / "cities.csv")

    # At least two parameter dimensions: population_size and mutation_rate.
    cases: List[Tuple[str, GAConfig, int]] = [
        ("pop80_mut005", GAConfig(population_size=80, mutation_rate=0.05), GA_SEED + 1),
        ("pop200_mut005", GAConfig(population_size=200, mutation_rate=0.05), GA_SEED + 2),
        ("pop120_mut002", GAConfig(population_size=120, mutation_rate=0.02), GA_SEED + 3),
        ("pop120_mut010", GAConfig(population_size=120, mutation_rate=0.10), GA_SEED + 4),
    ]

    all_results: List[GAResult] = []
    for label, cfg, seed in cases:
        res = evolve_tsp(cities, cfg, seed, label)
        all_results.append(res)
        plot_convergence(res, results_dir / f"convergence_{label}.png")

    ga_best = min(all_results, key=lambda x: x.best_length)
    plot_route(cities, ga_best.best_route, f"Best GA Route - {ga_best.label}", results_dir / "route_best_ga.png")
    plot_compare(all_results, results_dir / "convergence_compare_ga.png")

    save_ga_summary(all_results, results_dir / "summary_ga.csv")

    sa_best = load_sa_best(repo_root)

    print("GA homework outputs have been generated in:")
    print(results_dir)
    print(f"Best GA case: {ga_best.label}, length={ga_best.best_length:.4f}")
    if sa_best is not None:
        print(f"Best SA from homework1: alpha={sa_best[1]}, length={sa_best[0]:.4f}")


if __name__ == "__main__":
    main()
