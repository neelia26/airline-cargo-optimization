# Airline Cargo Optimisation

Coursework project investigating machine-learning-supported cargo selection for a Boeing 777-300ER operating between Colombo and Dubai, with a genetic algorithm used to optimise the selected consignments.

## Data collection

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

## Regression model selection

Five regression algorithms were compared using the same five inputs and fuel-consumption target. The data was split into an 85% development set and a 15% held-out test set. Model selection used the lowest mean RMSE from five-fold cross-validation on the development set; the test set was used only to confirm the selected model.

| Rank | Model | CV RMSE (kg) | Test RMSE (kg) | Test MAE (kg) | Test R² |
|---:|---|---:|---:|---:|---:|
| 1 | Support Vector Regression | 29.47 | 27.62 | 21.88 | 0.99914 |
| 2 | Polynomial Regression | 32.87 | 31.70 | 25.76 | 0.99887 |
| 3 | Gradient Boosting Regression | 78.69 | 75.50 | 60.61 | 0.99359 |
| 4 | Random Forest Regression | 115.97 | 106.48 | 82.85 | 0.98726 |
| 5 | Multiple Linear Regression | 145.57 | 135.96 | 102.66 | 0.97922 |

Support Vector Regression was selected because it produced the lowest cross-validated RMSE. Its test RMSE of 27.62 kg means that its predictions were typically about 27.62 kg away from the OpenAP fuel value. The saved pipeline includes feature scaling and uses `C=1000`, `epsilon=10` and `gamma=0.1`.

Reproduce the comparison with:

```bash
.venv/bin/python src/machine_learning/select_regression_model.py
```

Detailed reasoning is recorded in `report/regression_model_selection.md`. Machine-readable metrics and the fitted pipeline are stored in `results/model_selection/`.

## Real-data compatibility audit

The public EUROCONTROL PRC 2025 operational dataset was audited for external validation. It contains 295 B777-300ER flights, but no CMB–DXB flight and no B777-300ER flight inside this model's 225.5–252.5 minute duration range. PRC also labels discontinuous fuel intervals, whereas this project predicts complete-flight fuel. Consequently, calculating a direct accuracy score would be invalid. The evidence and limitations are documented in `report/real_data_validation.md`.

The take-off-mass relationship was also audited. Subtracting zero-fuel mass from take-off mass gives loaded fuel, which the generator defined as trip fuel plus 5% contingency and a 5,000 kg reserve. Reversing that formula reconstructs the target by design; it verifies internal consistency but also confirms that take-off mass would leak the target if used as an ML input.

Reproduce the lightweight audit without retaining the downloaded source tables:

```bash
.venv/bin/python src/model_validation/audit_prc_compatibility.py
.venv/bin/python src/model_validation/audit_mass_target_relationship.py
```

## Cargo consignment generation

The genetic-algorithm input contains 100 reproducible synthetic consignments offered for the CMB–DXB flight. Weight, volume, chargeable weight, cargo category, priority, revenue, handling cost, deadline and availability probability are generated through documented rules. IATA's volumetric-weight convention and published SriLankan Cargo handling charges inform the calculations; route-specific financial and density values remain clearly labelled modelling assumptions.

The consignment metadata and methodology report include a dictionary for `GEN`, `ELI`, `PER-COL`, `PIL-COL`, `FRG`, `DGR` and `EXP`. These are simplified project handling categories rather than a complete operational IATA code set.

Generate the consignment table with:

```bash
.venv/bin/python src/data_generation/generate_cargo_consignments.py
```

The resulting CSV and metadata are stored in `data/generated/`. The complete methodology is documented in `report/cargo_consignment_generation.md`.

## Project structure

- `data/generated/` contains the final generated dataset.
- `src/data_generation/` contains OpenAP data-generation code.
- `src/machine_learning/` contains fuel-consumption modelling code.
- `src/genetic_algorithm/` contains cargo-selection optimisation code.
- `results/` contains generated experiment outputs.
- `report/` contains report working files.

Course materials and the assessment brief remain local and are excluded from version control.
