"""Stage-one deterministic GA for CMB-DXB cargo selection."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


CARGO_PATH = Path("data/generated/cmb_dxb_cargo_consignments_100.csv")
MODEL_PATH = Path("results/model_selection/selected_fuel_model.joblib")
OUTPUT_DIR = Path("results/deterministic_ga")
FEATURES = [
    "zero_fuel_mass_kg",
    "cruise_speed_kt",
    "cruise_altitude_ft",
    "temperature_deviation_k",
    "flight_duration_min",
]


@dataclass(frozen=True)
class FlightScenario:
    """Fixed assumptions used throughout deterministic Stage One."""

    base_zero_fuel_mass_kg: float = 210_000.0
    available_cargo_payload_kg: float = 18_000.0
    available_cargo_volume_m3: float = 100.0
    handling_budget_usd: float = 2_500.0
    maximum_zero_fuel_mass_kg: float = 237_682.0
    cruise_speed_kt: float = 477.0
    cruise_altitude_ft: float = 36_000.0
    temperature_deviation_k: float = 2.0
    flight_duration_min: float = 239.5
    fuel_price_usd_per_kg: float = 0.80


@dataclass(frozen=True)
class GASettings:
    population_size: int = 150
    generations: int = 250
    crossover_probability: float = 0.90
    mutation_probability_per_gene: float = 0.01
    tournament_size: int = 3
    elite_count: int = 4
    random_seed: int = 4607
    random_baseline_samples: int = 10_000


class CargoGA:
    """Binary GA whose fitness includes an SVR fuel prediction."""

    def __init__(
        self,
        cargo: pd.DataFrame,
        model: object,
        scenario: FlightScenario,
        settings: GASettings,
    ) -> None:
        self.cargo = cargo.reset_index(drop=True)
        self.model = model
        self.scenario = scenario
        self.settings = settings
        self.rng = np.random.default_rng(settings.random_seed)
        self.weights = cargo["weight_kg"].to_numpy(float)
        self.volumes = cargo["volume_m3"].to_numpy(float)
        self.revenues = cargo["revenue_usd"].to_numpy(float)
        self.handling = cargo["handling_cost_usd"].to_numpy(float)
        self.gene_count = len(cargo)
        self.effective_weight_limit = min(
            scenario.available_cargo_payload_kg,
            scenario.maximum_zero_fuel_mass_kg
            - scenario.base_zero_fuel_mass_kg,
        )
        self.baseline_fuel_kg = float(
            self.model.predict(self._feature_frame(np.array([0.0])))[0]
        )

    def _feature_frame(self, selected_weights: np.ndarray) -> pd.DataFrame:
        rows = len(selected_weights)
        return pd.DataFrame(
            {
                "zero_fuel_mass_kg": (
                    self.scenario.base_zero_fuel_mass_kg + selected_weights
                ),
                "cruise_speed_kt": np.full(rows, self.scenario.cruise_speed_kt),
                "cruise_altitude_ft": np.full(
                    rows, self.scenario.cruise_altitude_ft
                ),
                "temperature_deviation_k": np.full(
                    rows, self.scenario.temperature_deviation_k
                ),
                "flight_duration_min": np.full(
                    rows, self.scenario.flight_duration_min
                ),
            },
            columns=FEATURES,
        )

    def totals(self, population: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "weight_kg": population @ self.weights,
            "volume_m3": population @ self.volumes,
            "revenue_usd": population @ self.revenues,
            "handling_cost_usd": population @ self.handling,
        }

    def is_feasible(self, chromosome: np.ndarray) -> bool:
        total_weight = float(chromosome @ self.weights)
        return bool(
            total_weight <= self.effective_weight_limit + 1e-9
            and float(chromosome @ self.volumes)
            <= self.scenario.available_cargo_volume_m3 + 1e-9
            and float(chromosome @ self.handling)
            <= self.scenario.handling_budget_usd + 1e-9
            and self.scenario.base_zero_fuel_mass_kg + total_weight
            <= self.scenario.maximum_zero_fuel_mass_kg + 1e-9
        )

    def repair(self, chromosome: np.ndarray) -> np.ndarray:
        """Remove the least resource-efficient items until all limits hold."""
        chromosome = chromosome.copy()
        while not self.is_feasible(chromosome):
            selected = np.flatnonzero(chromosome)
            if len(selected) == 0:
                break
            net_value = self.revenues[selected] - self.handling[selected]
            resource_pressure = (
                self.weights[selected] / self.effective_weight_limit
                + self.volumes[selected]
                / self.scenario.available_cargo_volume_m3
                + self.handling[selected]
                / self.scenario.handling_budget_usd
            )
            efficiency = net_value / np.maximum(resource_pressure, 1e-12)
            chromosome[selected[int(np.argmin(efficiency))]] = 0
        return chromosome

    def repair_population(self, population: np.ndarray) -> np.ndarray:
        return np.vstack([self.repair(row) for row in population]).astype(np.int8)

    def evaluate(self, population: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        totals = self.totals(population)
        fuel = self.model.predict(self._feature_frame(totals["weight_kg"]))
        additional_fuel = np.maximum(fuel - self.baseline_fuel_kg, 0.0)
        additional_fuel_cost = (
            additional_fuel * self.scenario.fuel_price_usd_per_kg
        )
        fitness = (
            totals["revenue_usd"]
            - totals["handling_cost_usd"]
            - additional_fuel_cost
        )
        details = {
            **totals,
            "predicted_fuel_kg": fuel,
            "additional_fuel_kg": additional_fuel,
            "additional_fuel_cost_usd": additional_fuel_cost,
        }
        return fitness, details

    def tournament_parent(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        candidates = self.rng.integers(
            0, len(population), size=self.settings.tournament_size
        )
        return population[candidates[np.argmax(fitness[candidates])]].copy()

    def create_children(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        required = self.settings.population_size - self.settings.elite_count
        children: list[np.ndarray] = []
        while len(children) < required:
            first = self.tournament_parent(population, fitness)
            second = self.tournament_parent(population, fitness)
            if self.rng.random() < self.settings.crossover_probability:
                mask = self.rng.random(self.gene_count) < 0.5
                first, second = (
                    np.where(mask, first, second),
                    np.where(mask, second, first),
                )
            for child in (first, second):
                mutation = (
                    self.rng.random(self.gene_count)
                    < self.settings.mutation_probability_per_gene
                )
                child = np.logical_xor(child, mutation).astype(np.int8)
                children.append(self.repair(child))
                if len(children) == required:
                    break
        return np.vstack(children)

    def run(self) -> tuple[np.ndarray, pd.DataFrame, dict[str, float]]:
        population = self.rng.binomial(
            1, 0.30, size=(self.settings.population_size, self.gene_count)
        ).astype(np.int8)
        population = self.repair_population(population)
        history: list[dict[str, float]] = []

        for generation in range(self.settings.generations + 1):
            fitness, _ = self.evaluate(population)
            history.append(
                {
                    "generation": generation,
                    "best_fitness_usd": float(fitness.max()),
                    "mean_fitness_usd": float(fitness.mean()),
                }
            )
            if generation == self.settings.generations:
                break
            elite_indices = np.argsort(fitness)[-self.settings.elite_count :]
            elites = population[elite_indices].copy()
            children = self.create_children(population, fitness)
            population = np.vstack([elites, children])

        final_fitness, final_details = self.evaluate(population)
        best_index = int(np.argmax(final_fitness))
        best_metrics = {
            key: float(values[best_index]) for key, values in final_details.items()
        }
        best_metrics["fitness_usd"] = float(final_fitness[best_index])
        return population[best_index], pd.DataFrame(history), best_metrics

    def greedy_baseline(self) -> tuple[np.ndarray, float]:
        resource_pressure = (
            self.weights / self.effective_weight_limit
            + self.volumes / self.scenario.available_cargo_volume_m3
            + self.handling / self.scenario.handling_budget_usd
        )
        order = np.argsort(
            -((self.revenues - self.handling) / np.maximum(resource_pressure, 1e-12))
        )
        chromosome = np.zeros(self.gene_count, dtype=np.int8)
        for index in order:
            candidate = chromosome.copy()
            candidate[index] = 1
            if self.is_feasible(candidate):
                chromosome = candidate
        fitness, _ = self.evaluate(chromosome.reshape(1, -1))
        return chromosome, float(fitness[0])

    def random_baseline(self) -> tuple[np.ndarray, float]:
        population = self.rng.binomial(
            1,
            0.30,
            size=(self.settings.random_baseline_samples, self.gene_count),
        ).astype(np.int8)
        population = self.repair_population(population)
        fitness, _ = self.evaluate(population)
        index = int(np.argmax(fitness))
        return population[index], float(fitness[index])


def validate_inputs(cargo: pd.DataFrame, scenario: FlightScenario) -> None:
    if len(cargo) != 100:
        raise ValueError("Stage One expects exactly 100 consignments")
    required = {
        "consignment_id",
        "weight_kg",
        "volume_m3",
        "revenue_usd",
        "handling_cost_usd",
    }
    if not required.issubset(cargo.columns):
        raise ValueError("Cargo input is missing a required GA column")
    if scenario.base_zero_fuel_mass_kg < 207_501.4:
        raise ValueError("Base zero-fuel mass is outside the SVR training range")
    if (
        scenario.base_zero_fuel_mass_kg + scenario.available_cargo_payload_kg
        > 237_669.95
    ):
        raise ValueError("Maximum GA mass would exceed the SVR training range")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4607)
    args = parser.parse_args()

    scenario = FlightScenario()
    settings = GASettings(random_seed=args.seed)
    cargo = pd.read_csv(CARGO_PATH)
    validate_inputs(cargo, scenario)
    model = joblib.load(MODEL_PATH)
    ga = CargoGA(cargo, model, scenario, settings)

    best, history, metrics = ga.run()
    greedy, greedy_fitness = ga.greedy_baseline()
    random_solution, random_fitness = ga.random_baseline()
    if not ga.is_feasible(best):
        raise RuntimeError("GA returned an infeasible solution")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = cargo.loc[best.astype(bool)].copy()
    selected.to_csv(OUTPUT_DIR / "selected_cargo_manifest.csv", index=False)
    history.to_csv(OUTPUT_DIR / "convergence.csv", index=False)

    metrics.update(
        {
            "selected_consignments": int(best.sum()),
            "resulting_zero_fuel_mass_kg": (
                scenario.base_zero_fuel_mass_kg + metrics["weight_kg"]
            ),
            "payload_utilisation_percent": (
                metrics["weight_kg"]
                / scenario.available_cargo_payload_kg
                * 100.0
            ),
            "volume_utilisation_percent": (
                metrics["volume_m3"]
                / scenario.available_cargo_volume_m3
                * 100.0
            ),
            "handling_budget_utilisation_percent": (
                metrics["handling_cost_usd"]
                / scenario.handling_budget_usd
                * 100.0
            ),
            "constraints_satisfied": bool(ga.is_feasible(best)),
            "greedy_baseline_fitness_usd": greedy_fitness,
            "random_baseline_fitness_usd": random_fitness,
            "ga_improvement_over_greedy_usd": (
                metrics["fitness_usd"] - greedy_fitness
            ),
            "ga_improvement_over_random_usd": (
                metrics["fitness_usd"] - random_fitness
            ),
            "greedy_selected_consignments": int(greedy.sum()),
            "random_selected_consignments": int(random_solution.sum()),
        }
    )
    summary = {
        "stage": "deterministic GA optimisation",
        "scenario": asdict(scenario),
        "ga_settings": asdict(settings),
        "baseline_fuel_without_selected_cargo_kg": ga.baseline_fuel_kg,
        "result": metrics,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
