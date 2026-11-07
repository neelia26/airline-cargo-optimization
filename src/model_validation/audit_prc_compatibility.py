"""Audit whether the PRC 2025 real-flight dataset can validate this SVR."""

from __future__ import annotations

import json
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd


RECORD = "https://zenodo.org/api/records/19184662/files"
SPLITS = ("train", "rank", "final")
AIRCRAFT = "B77W"
TARGET_ORIGIN = "VCBI"
TARGET_DESTINATION = "OMDB"
SYNTHETIC_DURATION_RANGE_MIN = (225.5, 252.5)
OUTPUT_DIR = Path("results/real_data_validation")


def download(url: str, destination: Path) -> None:
    """Download one small official table to temporary storage."""
    urllib.request.urlretrieve(url, destination)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    with tempfile.TemporaryDirectory(prefix="prc2025_") as temp_name:
        temp = Path(temp_name)
        for split in SPLITS:
            flight_path = temp / f"flightlist_{split}.parquet"
            fuel_path = temp / f"fuel_{split}.parquet"
            download(
                f"{RECORD}/flightlist_{split}.parquet/content", flight_path
            )
            download(f"{RECORD}/fuel_{split}.parquet/content", fuel_path)

            flights = pd.read_parquet(flight_path)
            flights = flights.loc[flights["aircraft_type"].eq(AIRCRAFT)].copy()
            flights["flight_duration_min"] = (
                flights["landed"] - flights["takeoff"]
            ).dt.total_seconds() / 60.0

            fuel = pd.read_parquet(fuel_path)
            fuel = fuel.loc[fuel["flight_id"].isin(flights["flight_id"])].copy()
            fuel["interval_min"] = (
                fuel["end"] - fuel["start"]
            ).dt.total_seconds() / 60.0
            coverage = fuel.groupby("flight_id").agg(
                labelled_intervals=("fuel_kg", "size"),
                labelled_fuel_kg=("fuel_kg", "sum"),
                labelled_minutes=("interval_min", "sum"),
            )

            joined = flights.merge(coverage, on="flight_id", how="left")
            joined["labelled_duration_fraction"] = (
                joined["labelled_minutes"] / joined["flight_duration_min"]
            )
            joined["split"] = split
            frames.append(joined)

    b77w = pd.concat(frames, ignore_index=True)
    b77w["exact_target_route"] = (
        b77w["origin_icao"].eq(TARGET_ORIGIN)
        & b77w["destination_icao"].eq(TARGET_DESTINATION)
    )
    lower, upper = SYNTHETIC_DURATION_RANGE_MIN
    b77w["inside_synthetic_duration_range"] = b77w[
        "flight_duration_min"
    ].between(lower, upper)
    b77w["distance_from_duration_range_min"] = (
        (lower - b77w["flight_duration_min"]).clip(lower=0)
        + (b77w["flight_duration_min"] - upper).clip(lower=0)
    )

    columns = [
        "split",
        "flight_id",
        "flight_date",
        "origin_icao",
        "destination_icao",
        "flight_duration_min",
        "labelled_intervals",
        "labelled_fuel_kg",
        "labelled_minutes",
        "labelled_duration_fraction",
        "exact_target_route",
        "inside_synthetic_duration_range",
        "distance_from_duration_range_min",
    ]
    b77w[columns].sort_values(
        ["distance_from_duration_range_min", "flight_id"]
    ).to_csv(OUTPUT_DIR / "prc_b77w_flight_summary.csv", index=False)

    closest = b77w.sort_values("distance_from_duration_range_min").iloc[0]
    summary = {
        "source": "EUROCONTROL PRC 2025 Aircraft Fuel Burn Estimation dataset",
        "doi": "10.5281/zenodo.19184662",
        "license": "CC BY 4.0",
        "aircraft_type": AIRCRAFT,
        "b77w_flights": int(len(b77w)),
        "exact_vcbi_omdb_flights": int(b77w["exact_target_route"].sum()),
        "flights_inside_synthetic_duration_range": int(
            b77w["inside_synthetic_duration_range"].sum()
        ),
        "synthetic_duration_range_min": list(SYNTHETIC_DURATION_RANGE_MIN),
        "closest_flight": {
            "flight_id": closest["flight_id"],
            "split": closest["split"],
            "origin_icao": closest["origin_icao"],
            "destination_icao": closest["destination_icao"],
            "duration_min": float(closest["flight_duration_min"]),
            "labelled_intervals": int(closest["labelled_intervals"]),
            "labelled_minutes": float(closest["labelled_minutes"]),
            "labelled_duration_fraction": float(
                closest["labelled_duration_fraction"]
            ),
        },
        "direct_accuracy_test_valid": False,
        "reasons": [
            "No CMB-DXB B77W flight is present",
            "No B77W flight falls inside the model's duration range",
            "PRC targets are discontinuous fuel intervals, while the SVR target is full-flight fuel",
            "PRC metadata does not provide zero-fuel mass or temperature deviation",
        ],
    }
    (OUTPUT_DIR / "compatibility_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
