"""Generate a reproducible OpenAP pilot dataset for the CMB-DXB B77W case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openap import FlightGenerator, FuelFlow, prop


AIRCRAFT_TYPE = "B77W"
ORIGIN = "CMB"
DESTINATION = "DXB"
ROUTE_DISTANCE_KM = 3306.0
CRUISE_DISTANCE_M = 2_754_000.0
RESERVE_FUEL_KG = 5_000.0
CONTINGENCY_FRACTION = 0.05
ASSUMED_FUEL_PRICE_PER_KG = 0.80


def stratified_uniform(
    rng: np.random.Generator, size: int, low: float, high: float
) -> np.ndarray:
    """Sample every part of a range instead of clustering values randomly."""
    positions = (np.arange(size) + rng.random(size)) / size
    rng.shuffle(positions)
    return low + positions * (high - low)


def fuel_flow_profile(
    fuel_model: FuelFlow,
    mass_profile: np.ndarray,
    speed_kt: np.ndarray,
    altitude_ft: np.ndarray,
    vertical_rate_fpm: np.ndarray,
    temperature_deviation_k: float,
) -> np.ndarray:
    """Estimate fuel flow for low-speed and airborne parts of a trajectory."""
    flow = np.empty_like(speed_kt, dtype=float)
    low_speed = speed_kt < 100.0

    if low_speed.any():
        flow[low_speed] = fuel_model.takeoff(
            tas=np.maximum(speed_kt[low_speed], 1.0),
            alt=altitude_ft[low_speed],
            throttle=0.50,
        )

    airborne = ~low_speed
    if airborne.any():
        flow[airborne] = fuel_model.enroute(
            mass=mass_profile[airborne],
            tas=speed_kt[airborne],
            alt=altitude_ft[airborne],
            vs=vertical_rate_fpm[airborne],
            dT=temperature_deviation_k,
        )

    if not np.isfinite(flow).all() or (flow <= 0).any():
        raise ValueError("OpenAP returned an invalid fuel-flow value")

    return flow


def simulate_flight(
    base_aircraft_mass_kg: float,
    cargo_mass_kg: float,
    cruise_mach: float,
    cruise_altitude_ft: float,
    temperature_deviation_k: float,
) -> dict[str, float]:
    """Simulate one flight and iteratively reconcile fuel with take-off mass."""
    zero_fuel_mass_kg = base_aircraft_mass_kg + cargo_mass_kg
    generator = FlightGenerator(AIRCRAFT_TYPE)
    trajectory = generator.complete(
        dt=30,
        range_cr=CRUISE_DISTANCE_M,
        alt_cr=cruise_altitude_ft,
        mach_cr=cruise_mach,
        random=False,
    )

    elapsed_seconds = trajectory["t"].diff().bfill().to_numpy(dtype=float)
    speed_kt = trajectory["groundspeed"].to_numpy(dtype=float)
    altitude_ft = trajectory["altitude"].to_numpy(dtype=float)
    vertical_rate_fpm = trajectory["vertical_rate"].to_numpy(dtype=float)

    fuel_model = FuelFlow(AIRCRAFT_TYPE)
    takeoff_mass_kg = zero_fuel_mass_kg + 55_000.0
    fuel_consumption_kg = 0.0

    for _ in range(10):
        mass_profile = np.maximum(
            takeoff_mass_kg - fuel_consumption_kg,
            zero_fuel_mass_kg,
        ) * np.ones(len(trajectory))

        # First pass obtains a changing mass profile along the trajectory.
        first_flow = fuel_flow_profile(
            fuel_model,
            mass_profile,
            speed_kt,
            altitude_ft,
            vertical_rate_fpm,
            temperature_deviation_k,
        )
        cumulative_fuel = np.cumsum(first_flow * elapsed_seconds)
        mass_profile = np.maximum(takeoff_mass_kg - cumulative_fuel, zero_fuel_mass_kg)

        # Second pass uses the changing mass to improve the fuel estimate.
        second_flow = fuel_flow_profile(
            fuel_model,
            mass_profile,
            speed_kt,
            altitude_ft,
            vertical_rate_fpm,
            temperature_deviation_k,
        )
        fuel_consumption_kg = float(np.sum(second_flow * elapsed_seconds))
        required_fuel_kg = (
            fuel_consumption_kg * (1.0 + CONTINGENCY_FRACTION) + RESERVE_FUEL_KG
        )
        updated_takeoff_mass_kg = zero_fuel_mass_kg + required_fuel_kg

        if abs(updated_takeoff_mass_kg - takeoff_mass_kg) < 1.0:
            takeoff_mass_kg = updated_takeoff_mass_kg
            break
        takeoff_mass_kg = updated_takeoff_mass_kg

    cruise_rows = trajectory["altitude"] >= cruise_altitude_ft - 100
    cruise_speed_kt = float(trajectory.loc[cruise_rows, "groundspeed"].median())

    return {
        "takeoff_mass_kg": takeoff_mass_kg,
        "cruise_speed_kt": cruise_speed_kt,
        "cruise_altitude_ft": float(round(cruise_altitude_ft, -2)),
        "temperature_deviation_k": temperature_deviation_k,
        "flight_duration_min": float(trajectory["t"].iloc[-1] / 60.0),
        "fuel_consumption_kg": fuel_consumption_kg,
    }


def generate_pilot(rows: int, seed: int) -> pd.DataFrame:
    """Generate the complete pilot table."""
    rng = np.random.default_rng(seed)
    base_mass = stratified_uniform(rng, rows, 205_000.0, 220_000.0)
    cargo_mass = stratified_uniform(rng, rows, 2_000.0, 18_000.0)
    cruise_mach = stratified_uniform(rng, rows, 0.80, 0.86)
    cruise_altitude = stratified_uniform(rng, rows, 33_000.0, 39_000.0)
    temperature_deviation = stratified_uniform(rng, rows, -8.0, 12.0)

    records: list[dict[str, object]] = []
    for index in range(rows):
        simulation = simulate_flight(
            base_aircraft_mass_kg=float(base_mass[index]),
            cargo_mass_kg=float(cargo_mass[index]),
            cruise_mach=float(cruise_mach[index]),
            cruise_altitude_ft=float(cruise_altitude[index]),
            temperature_deviation_k=float(temperature_deviation[index]),
        )
        fuel_cost = simulation["fuel_consumption_kg"] * ASSUMED_FUEL_PRICE_PER_KG
        records.append(
            {
                "scenario_id": index + 1,
                "aircraft_type": AIRCRAFT_TYPE,
                "origin": ORIGIN,
                "destination": DESTINATION,
                "distance_km": ROUTE_DISTANCE_KM,
                "cargo_mass_kg": float(cargo_mass[index]),
                "zero_fuel_mass_kg": float(base_mass[index] + cargo_mass[index]),
                **simulation,
                "fuel_price_per_kg": ASSUMED_FUEL_PRICE_PER_KG,
                "estimated_fuel_cost": fuel_cost,
            }
        )

    frame = pd.DataFrame.from_records(records)
    numeric_columns = frame.select_dtypes(include="number").columns
    frame[numeric_columns] = frame[numeric_columns].round(3)
    frame["scenario_id"] = frame["scenario_id"].astype(int)
    return frame


def validate(frame: pd.DataFrame) -> dict[str, object]:
    """Validate the schema and OpenAP aircraft limits before saving."""
    expected_columns = [
        "scenario_id",
        "aircraft_type",
        "origin",
        "destination",
        "distance_km",
        "cargo_mass_kg",
        "zero_fuel_mass_kg",
        "takeoff_mass_kg",
        "cruise_speed_kt",
        "cruise_altitude_ft",
        "temperature_deviation_k",
        "flight_duration_min",
        "fuel_consumption_kg",
        "fuel_price_per_kg",
        "estimated_fuel_cost",
    ]
    if frame.columns.tolist() != expected_columns:
        raise ValueError("Pilot dataset columns do not match the agreed schema")
    if frame.isna().any().any():
        raise ValueError("Pilot dataset contains missing values")
    if frame["scenario_id"].duplicated().any():
        raise ValueError("Pilot dataset contains duplicate scenario identifiers")

    aircraft = prop.aircraft(AIRCRAFT_TYPE)
    if frame["takeoff_mass_kg"].max() > aircraft["mtow"]:
        raise ValueError("A scenario exceeds the B77W maximum take-off mass")
    if frame["zero_fuel_mass_kg"].max() > aircraft["mlw"]:
        raise ValueError("A scenario has an implausible zero-fuel mass")
    if (frame["fuel_consumption_kg"] <= 0).any():
        raise ValueError("A scenario has non-positive fuel consumption")

    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_scenario_ids": int(frame["scenario_id"].duplicated().sum()),
        "openap_mtow_kg": aircraft["mtow"],
        "openap_mlw_kg": aircraft["mlw"],
        "maximum_generated_zero_fuel_mass_kg": float(
            frame["zero_fuel_mass_kg"].max()
        ),
        "minimum_generated_zero_fuel_mass_kg": float(
            frame["zero_fuel_mass_kg"].min()
        ),
        "maximum_generated_takeoff_mass_kg": float(frame["takeoff_mass_kg"].max()),
        "minimum_generated_takeoff_mass_kg": float(frame["takeoff_mass_kg"].min()),
        "minimum_fuel_consumption_kg": float(frame["fuel_consumption_kg"].min()),
        "maximum_fuel_consumption_kg": float(frame["fuel_consumption_kg"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4607)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pilot/openap_b77w_cmb_dxb_pilot.csv"),
    )
    args = parser.parse_args()

    frame = generate_pilot(args.rows, args.seed)
    validation = validate(frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    metadata = {
        "generator": "OpenAP",
        "openap_aircraft_code": AIRCRAFT_TYPE,
        "route": f"{ORIGIN}-{DESTINATION}",
        "seed": args.seed,
        "assumptions": {
            "route_distance_km": ROUTE_DISTANCE_KM,
            "base_aircraft_mass_kg": [205_000, 220_000],
            "cargo_mass_kg": [2_000, 18_000],
            "cruise_mach": [0.80, 0.86],
            "cruise_altitude_ft": [33_000, 39_000],
            "temperature_deviation_k": [-8, 12],
            "reserve_fuel_kg": RESERVE_FUEL_KG,
            "fuel_contingency_fraction": CONTINGENCY_FRACTION,
            "assumed_fuel_price_per_kg": ASSUMED_FUEL_PRICE_PER_KG,
        },
        "validation": validation,
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"csv": str(args.output), "metadata": str(metadata_path), **validation}, indent=2))


if __name__ == "__main__":
    main()
