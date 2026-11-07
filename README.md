# Airline Cargo Optimisation

Coursework project investigating machine-learning-supported cargo selection for a Boeing 777-300ER operating between Colombo and Dubai, with a genetic algorithm used to optimise the selected consignments.

## Current data-collection stage

OpenAP is used to simulate fuel consumption for a Boeing 777-300ER (`B77W`) on the Colombo (`CMB`) to Dubai (`DXB`) route. Each row represents one flight scenario. The aircraft type, route and distance remain fixed, while mass and operating conditions vary through controlled stratified sampling.

The machine-learning inputs are:

- `zero_fuel_mass_kg`
- `cruise_speed_kt`
- `cruise_altitude_ft`
- `temperature_deviation_k`
- `flight_duration_min`

The prediction target is `fuel_consumption_kg`.

Zero-fuel mass is used instead of take-off mass because it is known before fuel is predicted. Take-off mass is retained only as a supporting calculated field.

Generated records are checked against the Boeing 777-300ER maximum zero-fuel mass of 237,682 kg and the OpenAP take-off and landing limits.

## Dataset schema

| Column | Role | Description |
|---|---|---|
| `scenario_id` | Identifier | Unique flight-scenario number |
| `aircraft_type` | Metadata | OpenAP aircraft code, fixed as `B77W` |
| `origin` | Metadata | Departure airport, fixed as `CMB` |
| `destination` | Metadata | Arrival airport, fixed as `DXB` |
| `distance_km` | Metadata | Route distance, fixed as 3,306 km |
| `cargo_mass_kg` | Supporting value | Cargo mass used in the scenario |
| `zero_fuel_mass_kg` | ML input | Aircraft, passengers, baggage and cargo before usable fuel |
| `takeoff_mass_kg` | Supporting value | Zero-fuel mass plus required and reserve fuel |
| `cruise_speed_kt` | ML input | Typical speed during cruise |
| `cruise_altitude_ft` | ML input | Cruise altitude |
| `temperature_deviation_k` | ML input | Difference from standard atmospheric temperature |
| `flight_duration_min` | ML input | Simulated airborne duration |
| `fuel_consumption_kg` | ML target | OpenAP-estimated trip fuel consumption |
| `fuel_price_per_kg` | Post-prediction value | Illustrative fuel-price assumption |
| `estimated_fuel_cost` | Derived result | Fuel consumption multiplied by fuel price |

## Reproducing the datasets

Create a Python 3.12 environment and install the pinned dependencies:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Generate the 5,000-row dataset:

```bash
.venv/bin/python src/data_generation/generate_openap_pilot.py \
  --rows 5000 \
  --output data/generated/openap_b77w_cmb_dxb_5000.csv
```

The generator uses seed `4607` unless another seed is supplied. The metadata file records the assumptions and validation summary. The original 100-row pilot was removed after validation; its findings remain documented in `report/pilot_dataset_validation.md` and in Git history.

## Project structure

- `data/generated/` contains the final generated dataset.
- `src/data_generation/` contains OpenAP data-generation code.
- `src/machine_learning/` contains fuel-consumption modelling code.
- `src/genetic_algorithm/` contains cargo-selection optimisation code.
- `results/` contains generated experiment outputs.
- `report/` contains report working files.

Course materials and the assessment brief remain local and are excluded from version control.
