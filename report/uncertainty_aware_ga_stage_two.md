# Stage Two: Uncertainty-Aware Genetic Algorithm

## Purpose

Stage Two extends the verified deterministic GA by recognising that a booked consignment may become unavailable before loading. Each consignment's generated `availability_probability` is used to simulate future manifests. The aircraft limits and fixed flight scenario remain identical to Stage One, allowing the two selected plans to be compared fairly.

## Scenario-based fitness

Thirty common availability scenarios are generated once with seed `4610` and reused for every chromosome. For each scenario, every selected consignment is independently marked available or unavailable using its probability. Revenue, handling cost, loaded weight and SVR-predicted additional fuel are recalculated from the cargo that is actually available.

The uncertainty-aware fitness combines normal performance with downside performance:

```text
risk-adjusted fitness = 80% mean scenario profit
                      + 20% mean profit in the worst 10% of scenarios
```

Using common scenarios prevents chromosomes from receiving different scores merely because they were evaluated against different random events. The lower-tail component discourages a manifest that has a high average but performs poorly when several valuable consignments are unavailable.

The Stage-One chromosome, repair mechanism, tournament selection, uniform crossover, mutation, elitism, population of 150 and 250-generation limit remain unchanged.

## Planned robust manifest

| Planned result | Value |
|---|---:|
| Selected consignments | 42 |
| Selected weight | 15,614.6 kg |
| Selected volume | 99.262 m³ |
| Planned revenue | $94,495.17 |
| Planned handling cost | $2,499.30 |
| Payload utilisation | 86.75% |
| Volume utilisation | 99.26% |
| Handling-budget utilisation | 99.97% |
| Resulting zero-fuel mass | 225,614.6 kg |

Every planned constraint is satisfied. The robust and deterministic manifests share 39 consignments. The robust solution adds `C009`, `C012` and `C063`, while removing `C023`, `C067` and `C096`.

## Unseen-scenario evaluation

After optimisation, all methods were evaluated on the same 1,000 previously unseen availability scenarios generated with seed `4611`. These scenarios were not used to evolve the robust manifest.

| Method | Mean profit | Profit standard deviation | 10th percentile | Worst-10% mean | Risk-adjusted score |
|---|---:|---:|---:|---:|---:|
| Uncertainty-aware GA | $84,535.26 | $4,001.53 | $78,800.90 | $76,110.69 | $82,850.35 |
| Deterministic GA | $84,281.79 | $4,036.76 | $78,786.18 | $75,889.05 | $82,603.24 |
| Greedy baseline | $81,135.53 | $3,968.12 | $75,848.67 | $72,894.18 | $79,487.26 |
| Random baseline | $72,775.90 | $3,941.23 | $67,203.17 | $64,535.09 | $71,127.73 |

Compared with the Stage-One deterministic manifest, the uncertainty-aware manifest improved unseen mean profit by $253.48, worst-10% mean profit by $221.64 and risk-adjusted score by $247.11. It also reduced profit standard deviation by $35.23. The improvement is modest rather than dramatic because many high-priority consignments already have both higher revenue multipliers and high availability probabilities.

## Interpretation and limitations

Stage Two demonstrates the Grade-A concept of optimising on simulated future scenarios under uncertainty. It does not claim that the generated probabilities are observed airline cancellation rates. Independent Bernoulli availability is a transparent prototype assumption; real consignments may have correlated disruptions caused by weather, customs, connecting flights or shipper behaviour.

The deterministic Stage-One fitness of $90,762.62 should not be compared directly with the Stage-Two scenario mean because the former assumes every selected consignment is available. The correct comparison is the common 1,000-scenario test shown above.

Further evaluation can repeat the full optimisation with multiple GA seeds, alternative scenario counts, different risk weights and correlated disruption scenarios.
