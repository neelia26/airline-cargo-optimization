"""Generate reproducible synthetic cargo consignments for CMB-DXB."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEED = 4607
DEFAULT_ROWS = 100
MODELLING_LKR_PER_USD = 320.0
VOLUMETRIC_KG_PER_M3 = 1_000_000.0 / 6_000.0
DOCUMENTATION_CHARGE_LKR = 2_000.0

CATEGORIES: dict[str, dict[str, object]] = {
    "general": {
        "probability": 0.30,
        "weight_kg": (80.0, 2_200.0),
        "density_kg_m3": (120.0, 420.0),
        "rate_usd_kg": (1.20, 2.40),
        "handling_class": "general",
        "special_code": "GEN",
    },
    "electronics": {
        "probability": 0.16,
        "weight_kg": (30.0, 900.0),
        "density_kg_m3": (100.0, 300.0),
        "rate_usd_kg": (3.00, 5.50),
        "handling_class": "general",
        "special_code": "ELI",
    },
    "perishable": {
        "probability": 0.15,
        "weight_kg": (60.0, 1_500.0),
        "density_kg_m3": (140.0, 380.0),
        "rate_usd_kg": (2.20, 4.00),
        "handling_class": "special",
        "special_code": "PER-COL",
    },
    "pharmaceutical": {
        "probability": 0.12,
        "weight_kg": (20.0, 600.0),
        "density_kg_m3": (80.0, 240.0),
        "rate_usd_kg": (4.50, 7.50),
        "handling_class": "special",
        "special_code": "PIL-COL",
    },
    "fragile": {
        "probability": 0.10,
        "weight_kg": (25.0, 800.0),
        "density_kg_m3": (60.0, 200.0),
        "rate_usd_kg": (2.80, 5.00),
        "handling_class": "special",
        "special_code": "FRG",
    },
    "dangerous_goods": {
        "probability": 0.09,
        "weight_kg": (30.0, 700.0),
        "density_kg_m3": (180.0, 550.0),
        "rate_usd_kg": (3.80, 6.50),
        "handling_class": "special",
        "special_code": "DGR",
    },
    "express": {
        "probability": 0.08,
        "weight_kg": (10.0, 350.0),
        "density_kg_m3": (70.0, 220.0),
        "rate_usd_kg": (4.00, 7.00),
        "handling_class": "courier",
        "special_code": "EXP",
    },
}

PRIORITY_MULTIPLIER = {"normal": 1.00, "high": 1.18, "urgent": 1.40}
PRIORITY_DEADLINE_HOURS = {
    "normal": (48, 96),
    "high": (24, 48),
    "urgent": (8, 24),
}
PRIORITY_AVAILABILITY = {
    "normal": (0.82, 0.96),
    "high": (0.88, 0.98),
    "urgent": (0.93, 0.995),
}


def handling_charge_lkr(chargeable_weight_kg: float, handling_class: str) -> float:
    """Apply the published SriLankan Cargo charge structure as a proxy."""
    rules = {
        "general": (27.0, 2_500.0),
        "special": (60.0, 6_000.0),
        "courier": (40.0, 3_700.0),
    }
    per_kg, minimum = rules[handling_class]
    return max(per_kg * chargeable_weight_kg, minimum) + DOCUMENTATION_CHARGE_LKR


def choose_priority(rng: np.random.Generator, category: str) -> str:
    """Choose priority with more urgency for time-sensitive cargo."""
    if category in {"pharmaceutical", "perishable", "express"}:
        probabilities = (0.20, 0.45, 0.35)
    else:
        probabilities = (0.55, 0.35, 0.10)
    return str(rng.choice(["normal", "high", "urgent"], p=probabilities))


def generate_consignments(rows: int, seed: int) -> pd.DataFrame:
    """Generate one reproducible set of offered cargo consignments."""
    rng = np.random.default_rng(seed)
    category_names = list(CATEGORIES)
    probabilities = [float(CATEGORIES[name]["probability"]) for name in category_names]
    selected_categories = rng.choice(category_names, size=rows, p=probabilities)

    records: list[dict[str, object]] = []
    for index, category in enumerate(selected_categories, start=1):
        specification = CATEGORIES[str(category)]
        weight_low, weight_high = specification["weight_kg"]
        density_low, density_high = specification["density_kg_m3"]
        rate_low, rate_high = specification["rate_usd_kg"]

        # A beta distribution produces more medium consignments than extremes.
        weight_fraction = rng.beta(2.0, 3.0)
        weight_kg = float(weight_low + weight_fraction * (weight_high - weight_low))
        density = float(rng.uniform(density_low, density_high))
        volume_m3 = weight_kg / density
        volumetric_weight_kg = volume_m3 * VOLUMETRIC_KG_PER_M3
        chargeable_weight_kg = math.ceil(max(weight_kg, volumetric_weight_kg) * 2) / 2

        priority = choose_priority(rng, str(category))
        deadline_low, deadline_high = PRIORITY_DEADLINE_HOURS[priority]
        availability_low, availability_high = PRIORITY_AVAILABILITY[priority]
        deadline_hours = int(rng.integers(deadline_low, deadline_high + 1))
        availability_probability = float(
            rng.uniform(availability_low, availability_high)
        )

        rate_usd_kg = float(rng.uniform(rate_low, rate_high))
        revenue_usd = (
            chargeable_weight_kg
            * rate_usd_kg
            * PRIORITY_MULTIPLIER[priority]
        )
        handling_lkr = handling_charge_lkr(
            chargeable_weight_kg, str(specification["handling_class"])
        )
        handling_usd = handling_lkr / MODELLING_LKR_PER_USD
        typical_piece_weight = float(rng.uniform(18.0, 90.0))
        pieces = max(1, math.ceil(weight_kg / typical_piece_weight))

        records.append(
            {
                "consignment_id": f"C{index:03d}",
                "origin": "CMB",
                "destination": "DXB",
                "cargo_type": str(category),
                "special_handling_code": specification["special_code"],
                "priority_level": priority,
                "pieces": pieces,
                "weight_kg": weight_kg,
                "volume_m3": volume_m3,
                "chargeable_weight_kg": chargeable_weight_kg,
                "revenue_usd": revenue_usd,
                "handling_cost_usd": handling_usd,
                "deadline_hours": deadline_hours,
                "availability_probability": availability_probability,
            }
        )

    frame = pd.DataFrame(records)
    frame[["weight_kg", "chargeable_weight_kg"]] = frame[
        ["weight_kg", "chargeable_weight_kg"]
    ].round(1)
    frame["volume_m3"] = frame["volume_m3"].round(3)
    frame[["revenue_usd", "handling_cost_usd"]] = frame[
        ["revenue_usd", "handling_cost_usd"]
    ].round(2)
    frame["availability_probability"] = frame[
        "availability_probability"
    ].round(4)
    return frame


def validate(frame: pd.DataFrame, expected_rows: int) -> dict[str, object]:
    """Reject invalid or unhelpful generated cargo data."""
    if len(frame) != expected_rows:
        raise ValueError("Unexpected number of consignments")
    if frame["consignment_id"].duplicated().any():
        raise ValueError("Duplicate consignment identifier")
    if frame.isna().any().any():
        raise ValueError("Missing consignment value")
    positive = [
        "pieces",
        "weight_kg",
        "volume_m3",
        "chargeable_weight_kg",
        "revenue_usd",
        "handling_cost_usd",
        "deadline_hours",
    ]
    if (frame[positive] <= 0).any().any():
        raise ValueError("Non-positive consignment value")
    if (frame["chargeable_weight_kg"] < frame["weight_kg"]).any():
        raise ValueError("Chargeable weight is lower than actual weight")
    if not frame["availability_probability"].between(0, 1).all():
        raise ValueError("Invalid probability")
    if (frame["revenue_usd"] <= frame["handling_cost_usd"]).any():
        raise ValueError("A consignment cannot cover its handling cost")

    return {
        "rows": int(len(frame)),
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_ids": int(frame["consignment_id"].duplicated().sum()),
        "total_offered_weight_kg": float(frame["weight_kg"].sum()),
        "total_offered_volume_m3": float(frame["volume_m3"].sum()),
        "total_offered_revenue_usd": float(frame["revenue_usd"].sum()),
        "cargo_type_counts": frame["cargo_type"].value_counts().to_dict(),
        "priority_counts": frame["priority_level"].value_counts().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/cmb_dxb_cargo_consignments_100.csv"),
    )
    args = parser.parse_args()

    frame = generate_consignments(args.rows, args.seed)
    validation = validate(frame, args.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    metadata = {
        "dataset_type": "synthetic cargo consignments",
        "seed": args.seed,
        "route": "CMB-DXB",
        "rows": args.rows,
        "sources": {
            "iata_chargeable_weight": (
                "https://www.iata.org/en/publications/newsletters/"
                "iata-knowledge-hub/air-cargo-tariffs-and-rules-what-you-need-to-know"
            ),
            "srilankan_cargo_handling_charges": (
                "https://www.srilankancargo.com/assets/pdf/charges/import-charges.pdf"
            ),
            "manifest_structure_reference": (
                "https://www.docupipe.ai/landing/air-cargo-manifest"
            ),
            "category_and_priority_reference": (
                "https://www.kaggle.com/datasets/colabsss/"
                "air-cargo-resource-allocation-data"
            ),
        },
        "assumptions": {
            "volumetric_divisor_cm3_per_kg": 6_000,
            "modelling_exchange_rate_lkr_per_usd": MODELLING_LKR_PER_USD,
            "documentation_charge_lkr": DOCUMENTATION_CHARGE_LKR,
            "category_rules": CATEGORIES,
            "priority_revenue_multipliers": PRIORITY_MULTIPLIER,
            "note": (
                "Revenue, density, deadlines, availability probabilities and the "
                "LKR/USD conversion are transparent modelling assumptions, not "
                "observed Emirates or SriLankan Airlines transactions. Published "
                "SriLankan import terminal charges are used only as a proxy."
            ),
        },
        "validation": validation,
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
