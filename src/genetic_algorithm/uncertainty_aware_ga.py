"""Stage-two GA that optimises cargo under availability uncertainty."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from deterministic_ga import (
    CARGO_PATH,
    FEATURES,
    MODEL_PATH,
    CargoGA,
    FlightScenario,
    GASettings,
    validate_inputs,
)


OUTPUT_DIR = Path("results/uncertainty_aware_ga")
DETERMINISTIC_MANIFEST = Path(
    "results/deterministic_ga/selected_cargo_manifest.csv"
)


@dataclass(frozen=True)
class UncertaintySettings:
    training_scenarios: int = 30
    held_out_scenarios: int = 1_000
    lower_tail_fraction: float = 0.10
    expected_profit_weight: float = 0.80
    lower_tail_profit_weight: float = 0.20
    training_scenario_seed: int = 4610
    held_out_scenario_seed: int = 4611


class UncertaintyAwareCargoGA(CargoGA):
    """Cargo GA evaluated with common Bernoulli availability scenarios."""

    def __init__(
        self,
        cargo: pd.DataFrame,
        model: object,
        scenario: FlightScenario,
        settings: GASettings,
        uncertainty: UncertaintySettings,
    ) -> None:
        super().__init__(cargo, model, scenario, settings)
        self.uncertainty = uncertainty
        self.availability_probabilities = cargo[
            "availability_probability"
        ].to_numpy(float)
        self.training_availability = self.generate_availability_scenarios(
            uncertainty.training_scenarios,
            uncertainty.training_scenario_seed,
        )

    def generate_availability_scenarios(
        self, scenario_count: int, seed: int
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return (
            rng.random((scenario_count, self.gene_count))
            < self.availability_probabilities
        ).astype(np.int8)

    def scenario_profits(
        self, population: np.ndarray, availability: np.ndarray
    ) -> np.ndarray:
        """Return realised profits with shape population x scenarios."""
        realised = population[:, None, :] * availability[None, :, :]
        realised_weight = realised @ self.weights
        realised_revenue = realised @ self.revenues
        realised_handling = realised @ self.handling
        flat_weight = realised_weight.reshape(-1)
        predicted_fuel = self.model.predict(self._feature_frame(flat_weight))
        additional_fuel = np.maximum(predicted_fuel - self.baseline_fuel_kg, 0.0)
        additional_fuel_cost = (
            additional_fuel * self.scenario.fuel_price_usd_per_kg
        ).reshape(realised_weight.shape)
        return realised_revenue - realised_handling - additional_fuel_cost

    def risk_metrics(self, profits: np.ndarray) -> dict[str, np.ndarray]:
        mean_profit = profits.mean(axis=1)
        standard_deviation = profits.std(axis=1)
        tenth_percentile = np.quantile(profits, 0.10, axis=1)
        tail_count = max(
            1,
            math.ceil(
                profits.shape[1] * self.uncertainty.lower_tail_fraction
            ),
        )
        lower_tail_mean = np.partition(profits, tail_count - 1, axis=1)[
            :, :tail_count
        ].mean(axis=1)
        risk_adjusted_fitness = (
            self.uncertainty.expected_profit_weight * mean_profit
            + self.uncertainty.lower_tail_profit_weight * lower_tail_mean
        )
        return {
            "mean_scenario_profit_usd": mean_profit,
            "profit_standard_deviation_usd": standard_deviation,
            "tenth_percentile_profit_usd": tenth_percentile,
            "lower_tail_mean_profit_usd": lower_tail_mean,
            "risk_adjusted_fitness_usd": risk_adjusted_fitness,
        }

    def evaluate(
        self, population: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        planned = self.totals(population)
        profits = self.scenario_profits(population, self.training_availability)
        risk = self.risk_metrics(profits)
        details = {**planned, **risk}
        return risk["risk_adjusted_fitness_usd"], details

    def held_out_metrics(
        self, chromosome: np.ndarray, availability: np.ndarray
    ) -> dict[str, float]:
        profits = self.scenario_profits(
            chromosome.reshape(1, -1), availability
        )
        metrics = self.risk_metrics(profits)
        return {key: float(value[0]) for key, value in metrics.items()}


def chromosome_from_manifest(
    cargo: pd.DataFrame, manifest_path: Path
) -> np.ndarray:
    selected_ids = set(pd.read_csv(manifest_path)["consignment_id"])
    return cargo["consignment_id"].isin(selected_ids).astype(np.int8).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4607)
    args = parser.parse_args()

    scenario = FlightScenario()
    settings = GASettings(random_seed=args.seed, random_baseline_samples=5_000)
    uncertainty = UncertaintySettings()
    cargo = pd.read_csv(CARGO_PATH)
    validate_inputs(cargo, scenario)
    model = joblib.load(MODEL_PATH)
    ga = UncertaintyAwareCargoGA(
        cargo, model, scenario, settings, uncertainty
    )

    best, history, planned_metrics = ga.run()
    if not ga.is_feasible(best):
        raise RuntimeError("Uncertainty-aware GA returned an infeasible plan")

    deterministic = chromosome_from_manifest(cargo, DETERMINISTIC_MANIFEST)
    if not ga.is_feasible(deterministic):
        raise RuntimeError("Saved deterministic manifest is infeasible")

    held_out = ga.generate_availability_scenarios(
        uncertainty.held_out_scenarios,
        uncertainty.held_out_scenario_seed,
    )
    robust_held_out = ga.held_out_metrics(best, held_out)
    deterministic_held_out = ga.held_out_metrics(deterministic, held_out)
    greedy, _ = ga.greedy_baseline()
    random_solution, _ = ga.random_baseline()
    greedy_held_out = ga.held_out_metrics(greedy, held_out)
    random_held_out = ga.held_out_metrics(random_solution, held_out)

    planned_totals = ga.totals(best.reshape(1, -1))
    planned = {key: float(value[0]) for key, value in planned_totals.items()}
    planned.update(
        {
            "selected_consignments": int(best.sum()),
            "resulting_zero_fuel_mass_kg": (
                scenario.base_zero_fuel_mass_kg + planned["weight_kg"]
            ),
            "payload_utilisation_percent": (
                planned["weight_kg"]
                / scenario.available_cargo_payload_kg
                * 100.0
            ),
            "volume_utilisation_percent": (
                planned["volume_m3"]
                / scenario.available_cargo_volume_m3
                * 100.0
            ),
            "handling_budget_utilisation_percent": (
                planned["handling_cost_usd"]
                / scenario.handling_budget_usd
                * 100.0
            ),
            "constraints_satisfied": bool(ga.is_feasible(best)),
        }
    )

    comparison = pd.DataFrame(
        [
            {
                "method": "uncertainty-aware GA",
                "selected_consignments": int(best.sum()),
                **robust_held_out,
            },
            {
                "method": "deterministic GA",
                "selected_consignments": int(deterministic.sum()),
                **deterministic_held_out,
            },
            {
                "method": "greedy baseline",
                "selected_consignments": int(greedy.sum()),
                **greedy_held_out,
            },
            {
                "method": "risk-aware random baseline",
                "selected_consignments": int(random_solution.sum()),
                **random_held_out,
            },
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cargo.loc[best.astype(bool)].to_csv(
        OUTPUT_DIR / "selected_robust_manifest.csv", index=False
    )
    history.to_csv(OUTPUT_DIR / "convergence.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "held_out_comparison.csv", index=False)

    summary = {
        "stage": "uncertainty-aware GA optimisation",
        "scenario": asdict(scenario),
        "ga_settings": asdict(settings),
        "uncertainty_settings": asdict(uncertainty),
        "interpretation": (
            "Each selected consignment is independently available according to "
            "its generated probability. Fitness is 80% mean scenario profit plus "
            "20% mean profit in the worst 10% of training scenarios. Final methods "
            "are compared on 1,000 unseen common scenarios."
        ),
        "planned_robust_manifest": planned,
        "training_risk_metrics": {
            key: float(value) for key, value in planned_metrics.items()
        },
        "held_out_results": comparison.to_dict(orient="records"),
        "held_out_improvement_over_deterministic": {
            "mean_profit_usd": (
                robust_held_out["mean_scenario_profit_usd"]
                - deterministic_held_out["mean_scenario_profit_usd"]
            ),
            "lower_tail_mean_profit_usd": (
                robust_held_out["lower_tail_mean_profit_usd"]
                - deterministic_held_out["lower_tail_mean_profit_usd"]
            ),
            "risk_adjusted_fitness_usd": (
                robust_held_out["risk_adjusted_fitness_usd"]
                - deterministic_held_out["risk_adjusted_fitness_usd"]
            ),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
