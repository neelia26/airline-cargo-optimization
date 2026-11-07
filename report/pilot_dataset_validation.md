# OpenAP Dataset Validation

## Scope

The project contains a 100-row pilot and a 5,000-row final dataset of simulated Boeing 777-300ER flight scenarios for the Colombo to Dubai route. Both were generated with OpenAP 2.6.2 using seed 4607.

## Structural checks

| Check | Pilot | Final dataset |
|---|---:|---:|
| Rows | 100 | 5,000 |
| Columns | 15 | 15 |
| Missing values | 0 | 0 |
| Duplicate scenario identifiers | 0 | 0 |
| Reproducible from the saved seed | Yes | Yes |

## Range checks

| Variable | Final minimum | Final maximum | Assessment |
|---|---:|---:|---|
| Cargo mass | 2,001.814 kg | 17,999.004 kg | Within the configured generation range |
| Zero-fuel mass | 207,501.399 kg | 237,669.948 kg | Below OpenAP's maximum landing-mass reference |
| Take-off mass | 254,539.355 kg | 288,936.996 kg | Below the OpenAP MTOW of 351,500 kg |
| Cruise speed | 458 kt | 500 kt | Plausible cruise-speed range |
| Cruise altitude | 33,000 ft | 39,000 ft | Plausible cruise-altitude range |
| Temperature deviation | -7.997 K | 11.996 K | Within the configured scenario range |
| Simulated flight duration | 225.5 min | 252.5 min | Plausible airborne duration for the route |
| Fuel consumption | 39,281.426 kg | 44,693.734 kg | Positive and internally consistent with OpenAP output |

## Corrected mass input

The pilot review found that take-off mass contains the fuel that the model is intended to predict. The schema was therefore corrected before scaling. `zero_fuel_mass_kg` is now the mass input, while `takeoff_mass_kg` remains a supporting calculated value.

```text
zero fuel mass = base aircraft mass + cargo mass
```

This value is known before fuel prediction, so it avoids a circular dependency when the GA later evaluates a cargo selection.

## Full-dataset learnability diagnostic

An 85/15 train-test split with random state 4607 was used only to confirm that the generated predictors retain useful information. This is not the final model-selection experiment.

| Model | Test R squared | Test mean absolute error | Test RMSE |
|---|---:|---:|---:|
| Linear regression | 0.979 | 102.7 kg | 136.0 kg |
| Random forest | 0.987 | 84.8 kg | 108.9 kg |

## Conclusion

The corrected pilot and final dataset pass the structural, aircraft-limit, plausibility, reproducibility, and diagnostic learnability checks. Formal model selection and evaluation should use the saved final dataset in a separate, reproducible ML workflow.
