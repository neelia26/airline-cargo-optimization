"""Focused sensitivity analysis for Stage-Two cargo uncertainty."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from deterministic_ga import (
    CARGO_PATH,
    MODEL_PATH,
    FlightScenario,
    GASettings,
    validate_inputs,
)
from uncertainty_aware_ga import (
    DETERMINISTIC_MANIFEST,
    UncertaintyAwareCargoGA,
    UncertaintySettings,
    chromosome_from_manifest,
)


OUTPUT_DIR = Path("results/uncertainty_sensitivity")

CONDITIONS = {
    "nominal": {
        "description": "Original independent availability probabilities",
        "probability_reduction": 0.0,
        "correlated_shocks": False,
        "expected_weight": 0.80,
        "tail_weight": 0.20,
    },
    "moderate_disruption": {
        "description": "Every availability probability reduced by 0.10",
        "probability_reduction": 0.10,
        "correlated_shocks": False,
        "expected_weight": 0.80,
        "tail_weight": 0.20,
    },
    "correlated_disruption": {
        "description": (
            "Nominal probabilities plus category-level cold-chain, regulated-"
            "cargo and express-connection shocks"
        ),
        "probability_reduction": 0.0,
        "correlated_shocks": True,
        "expected_weight": 0.80,
        "tail_weight": 0.20,
    },
    "risk_averse": {
        "description": (
            "Nominal independent probabilities with 60% mean and 40% lower-tail fitness"
        ),
        "probability_reduction": 0.0,
        "correlated_shocks": False,
        "expected_weight": 0.60,
        "tail_weight": 0.40,
    },
}


def availability_scenarios(
    cargo: pd.DataFrame,
    scenario_count: int,
    seed: int,
    probability_reduction: float,
    correlated_shocks: bool,
) -> np.ndarray:
    """Create transparent independent or category-correlated scenarios."""
    rng = np.random.default_rng(seed)
    probabilities = np.clip(
        cargo["availability_probability"].to_numpy(float)
        - probability_reduction,
        0.50,
        0.995,
    )
    availability = (
        rng.random((scenario_count, len(cargo))) < probabilities
    ).astype(np.int8)

    if correlated_shocks:
        cargo_type = cargo["cargo_type"].to_numpy(str)
        shock_rules = [
            (0.15, np.isin(cargo_type, ["perishable", "pharmaceutical"])),
            (0.12, cargo_type == "dangerous_goods"),
            (0.10, cargo_type == "express"),
        ]
        for shock_probability, affected in shock_rules:
            shock_occurs = rng.random(scenario_count) < shock_probability
            availability[np.ix_(shock_occurs, affected)] = 0
    return availability


def run_condition(
    name: str,
    specification: dict[str, object],
    cargo: pd.DataFrame,
    model: object,
    deterministic: np.ndarray,
    scenario: FlightScenario,
) -> tuple[dict[str, object], np.ndarray, pd.DataFrame]:
    uncertainty = UncertaintySettings(
        expected_profit_weight=float(specification["expected_weight"]),
        lower_tail_profit_weight=float(specification["tail_weight"]),
    )
    settings = GASettings(random_seed=4607, random_baseline_samples=0)
    ga = UncertaintyAwareCargoGA(
        cargo, model, scenario, settings, uncertainty
    )
    ga.training_availability = availability_scenarios(
        cargo,
        uncertainty.training_scenarios,
        uncertainty.training_scenario_seed,
        float(specification["probability_reduction"]),
        bool(specification["correlated_shocks"]),
    )
    best, history, _ = ga.run()
    held_out = availability_scenarios(
        cargo,
        uncertainty.held_out_scenarios,
        uncertainty.held_out_scenario_seed,
        float(specification["probability_reduction"]),
        bool(specification["correlated_shocks"]),
    )
    robust_metrics = ga.held_out_metrics(best, held_out)
    deterministic_metrics = ga.held_out_metrics(deterministic, held_out)
    totals = ga.totals(best.reshape(1, -1))
    deterministic_ids = set(cargo.loc[deterministic.astype(bool), "consignment_id"])
    robust_ids = set(cargo.loc[best.astype(bool), "consignment_id"])

    result: dict[str, object] = {
        "condition": name,
        "description": specification["description"],
        "selected_consignments": int(best.sum()),
        "shared_with_deterministic": len(robust_ids & deterministic_ids),
        "changed_consignments": len(robust_ids ^ deterministic_ids),
        "planned_weight_kg": float(totals["weight_kg"][0]),
        "planned_volume_m3": float(totals["volume_m3"][0]),
        "planned_handling_cost_usd": float(totals["handling_cost_usd"][0]),
        "robust_mean_profit_usd": robust_metrics["mean_scenario_profit_usd"],
        "deterministic_mean_profit_usd": deterministic_metrics[
            "mean_scenario_profit_usd"
        ],
        "mean_profit_improvement_usd": (
            robust_metrics["mean_scenario_profit_usd"]
            - deterministic_metrics["mean_scenario_profit_usd"]
        ),
        "robust_lower_tail_mean_usd": robust_metrics[
            "lower_tail_mean_profit_usd"
        ],
        "deterministic_lower_tail_mean_usd": deterministic_metrics[
            "lower_tail_mean_profit_usd"
        ],
        "lower_tail_improvement_usd": (
            robust_metrics["lower_tail_mean_profit_usd"]
            - deterministic_metrics["lower_tail_mean_profit_usd"]
        ),
        "robust_profit_std_usd": robust_metrics[
            "profit_standard_deviation_usd"
        ],
        "deterministic_profit_std_usd": deterministic_metrics[
            "profit_standard_deviation_usd"
        ],
        "robust_risk_adjusted_score_usd": robust_metrics[
            "risk_adjusted_fitness_usd"
        ],
        "deterministic_risk_adjusted_score_usd": deterministic_metrics[
            "risk_adjusted_fitness_usd"
        ],
        "risk_adjusted_improvement_usd": (
            robust_metrics["risk_adjusted_fitness_usd"]
            - deterministic_metrics["risk_adjusted_fitness_usd"]
        ),
        "constraints_satisfied": bool(ga.is_feasible(best)),
    }
    return result, best, history


def main() -> None:
    cargo = pd.read_csv(CARGO_PATH)
    scenario = FlightScenario()
    validate_inputs(cargo, scenario)
    model = joblib.load(MODEL_PATH)
    deterministic = chromosome_from_manifest(cargo, DETERMINISTIC_MANIFEST)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for name, specification in CONDITIONS.items():
        print(f"Running condition: {name}", flush=True)
        result, chromosome, history = run_condition(
            name,
            specification,
            cargo,
            model,
            deterministic,
            scenario,
        )
        results.append(result)
        cargo.loc[chromosome.astype(bool)].to_csv(
            OUTPUT_DIR / f"manifest_{name}.csv", index=False
        )
        history.to_csv(OUTPUT_DIR / f"convergence_{name}.csv", index=False)
        print(json.dumps(result, indent=2), flush=True)

    comparison = pd.DataFrame(results)
    comparison.to_csv(OUTPUT_DIR / "condition_comparison.csv", index=False)
    summary = {
        "purpose": (
            "One focused sensitivity analysis supporting the coursework requirement "
            "to simulate future risk and uncertainty"
        ),
        "fixed_flight_scenario": asdict(scenario),
        "ga_population": 150,
        "ga_generations": 250,
        "training_scenarios_per_condition": 30,
        "held_out_scenarios_per_condition": 1_000,
        "condition_definitions": CONDITIONS,
        "correlated_shock_probabilities": {
            "cold_chain_affecting_perishable_and_pharmaceutical": 0.15,
            "regulated_cargo_affecting_dangerous_goods": 0.12,
            "express_connection_affecting_express": 0.10,
        },
        "important_limitation": (
            "Disruption probabilities are transparent sensitivity assumptions, "
            "not measured airline operational frequencies."
        ),
        "results": results,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
