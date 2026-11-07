# Cargo Consignment Generation

## Purpose

The cargo dataset represents shipments offered for one CMB–DXB Boeing 777-300ER flight. Each row is an indivisible consignment that the genetic algorithm may either accept or reject. One hundred consignments create approximately `2^100` possible selection combinations before constraints are considered.

## Generation approach

The records are synthetic because a suitable public consignment-level airline manifest was not available. They are generated reproducibly with seed `4607`; they are not manually written by ChatGPT and are not presented as Emirates operational records.

The structure combines manifest-style fields with explicit modelling rules. Cargo category determines plausible weight, density, transport rate and handling class. Volume is calculated from weight and sampled density, preserving a physical relationship between the two. Priority influences the delivery deadline, availability probability and revenue multiplier.

IATA states that air cargo is charged using the higher of actual weight and volumetric weight, with the general volumetric calculation dividing cubic centimetres by 6,000. The generator therefore calculates chargeable weight as the greater of actual weight and `volume_m3 × 166.67`.

Published SriLankan Cargo import-terminal charges are used as a handling-cost proxy: LKR 27/kg for general cargo, LKR 60/kg for special cargo and LKR 40/kg for courier cargo, subject to the published minimums and a documentation charge. The conversion of 320 LKR/USD is a fixed modelling assumption for reproducibility, not a claimed current exchange rate. Because these are import-terminal charges rather than CMB export rates, they provide a defensible scale rather than exact operational costs.

## Dataset columns

| Column | Role |
|---|---|
| `consignment_id` | Unique shipment identifier |
| `origin`, `destination` | Fixed route metadata |
| `cargo_type` | General, electronics, perishable, pharmaceutical, fragile, dangerous goods or express |
| `special_handling_code` | Manifest-style handling category |
| `priority_level` | Normal, high or urgent |
| `pieces` | Number of packages in the consignment |
| `weight_kg` | Actual cargo mass used by the aircraft-weight constraint |
| `volume_m3` | Cargo space used by the volume constraint |
| `chargeable_weight_kg` | Higher of actual and volumetric weight, used for pricing |
| `revenue_usd` | Synthetic transport revenue available if selected |
| `handling_cost_usd` | Modelled terminal handling cost |
| `deadline_hours` | Maximum delivery time associated with priority |
| `availability_probability` | Scenario uncertainty that the booked consignment remains available |

## Validation

The generator rejects missing values, duplicate identifiers, non-positive physical or financial values, invalid probabilities, chargeable weight below actual weight and revenue that cannot cover handling cost. The metadata file records every range, probability, formula, source and assumption required to reproduce or challenge the dataset.

## Sources used

- [IATA air-cargo tariffs and volumetric-weight guidance](https://www.iata.org/en/publications/newsletters/iata-knowledge-hub/air-cargo-tariffs-and-rules-what-you-need-to-know)
- [SriLankan Cargo terminal handling charges](https://www.srilankancargo.com/assets/pdf/charges/import-charges.pdf)
- [DocuPipe manifest structure example](https://www.docupipe.ai/landing/air-cargo-manifest)
- [Kaggle Air Cargo Resource Allocation dataset](https://www.kaggle.com/datasets/colabsss/air-cargo-resource-allocation-data)
