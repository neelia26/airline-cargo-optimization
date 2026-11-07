"""Check the relationship between take-off mass and the OpenAP fuel target."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DATASET = Path("data/generated/openap_b77w_cmb_dxb_5000.csv")
OUTPUT = Path("results/real_data_validation/mass_target_relationship.json")
RESERVE_FUEL_KG = 5_000.0
CONTINGENCY_FRACTION = 0.05


def main() -> None:
    frame = pd.read_csv(DATASET)
    loaded_fuel = frame["takeoff_mass_kg"] - frame["zero_fuel_mass_kg"]
    reconstructed_trip_fuel = (
        loaded_fuel - RESERVE_FUEL_KG
    ) / (1.0 + CONTINGENCY_FRACTION)
    reconstruction_error = (
        reconstructed_trip_fuel - frame["fuel_consumption_kg"]
    )

    result = {
        "rows": int(len(frame)),
        "calculation": (
            "loaded_fuel_kg = takeoff_mass_kg - zero_fuel_mass_kg"
        ),
        "generator_relationship": (
            "loaded_fuel_kg = fuel_consumption_kg * 1.05 + 5000"
        ),
        "reverse_calculation": (
            "reconstructed_trip_fuel_kg = (loaded_fuel_kg - 5000) / 1.05"
        ),
        "loaded_fuel_target_correlation": float(
            loaded_fuel.corr(frame["fuel_consumption_kg"])
        ),
        "mean_absolute_reconstruction_error_kg": float(
            reconstruction_error.abs().mean()
        ),
        "maximum_absolute_reconstruction_error_kg": float(
            reconstruction_error.abs().max()
        ),
        "interpretation": (
            "The target can be reconstructed because take-off mass was derived "
            "from that target by the generator. This is a consistency check, not "
            "independent validation, and take-off mass must not be an ML input."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
