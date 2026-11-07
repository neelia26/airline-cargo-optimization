# OpenAP Pilot Dataset Validation

## Scope

The pilot contains 100 simulated Boeing 777-300ER flight scenarios for the Colombo to Dubai route. It was generated with OpenAP 2.6.2 using seed 4607.

## Structural checks

| Check | Result |
|---|---:|
| Rows | 100 |
| Columns | 15 |
| Missing values | 0 |
| Duplicate scenario identifiers | 0 |
| Reproducible from the saved seed | Yes |

## Range checks

| Variable | Minimum | Maximum | Assessment |
|---|---:|---:|---|
| Cargo mass | 2,089.723 kg | 17,873.080 kg | Within the pilot generation range |
| Base aircraft mass | 205,034.988 kg | 219,954.633 kg | Above OpenAP operating empty weight and below aircraft limits |
| Take-off mass | 258,088.425 kg | 285,987.670 kg | Below the OpenAP MTOW of 351,500 kg |
| Cruise speed | 458 kt | 497 kt | Plausible cruise-speed range |
| Cruise altitude | 33,000 ft | 39,000 ft | Plausible cruise-altitude range |
| Temperature deviation | -7.892 K | 11.989 K | Within the configured scenario range |
| Simulated flight duration | 227.5 min | 250.5 min | Plausible airborne duration for the route |
| Fuel consumption | 40,283.237 kg | 44,236.476 kg | Positive and internally consistent with OpenAP output |

## Pilot learnability check

Five-fold cross-validation was used only as a diagnostic on the 100-row pilot. It is not the final ML evaluation.

| Model | Mean R squared | Mean absolute error |
|---|---:|---:|
| Linear regression using take-off mass | 0.977 | 96.2 kg |
| Random forest using take-off mass | 0.807 | 313.0 kg |
| Linear regression using zero-fuel mass | 0.971 | 106.8 kg |

The linear relationship is strong in the controlled OpenAP scenarios. The random forest has less data to learn from in this small pilot and should be reconsidered after the full dataset is created.

## Required schema correction before scaling

`takeoff_mass_kg` includes the fuel loaded for the flight. Because fuel consumption is the target, the GA would not know the final take-off mass until after the ML model predicts the required fuel. Using take-off mass as an ML input therefore creates a circular dependency.

The recommended correction is to use `zero_fuel_mass_kg` as the mass input. It is calculated as:

```text
zero fuel mass = base aircraft mass + cargo mass
```

This value is available before fuel prediction and still gives strong pilot performance. `takeoff_mass_kg` can remain in the dataset as a supporting calculated value, but it should not be supplied to the ML model.

To keep the agreed 15-column dataset, `zero_fuel_mass_kg` can replace `base_aircraft_mass_kg`. The base mass remains an internal generation assumption and can be recovered as zero-fuel mass minus cargo mass.

## Conclusion

The pilot passes the structural, aircraft-limit, plausibility, and reproducibility checks. The full 5,000-row dataset should not be generated until the mass-input correction is accepted and applied to the pilot generator.
