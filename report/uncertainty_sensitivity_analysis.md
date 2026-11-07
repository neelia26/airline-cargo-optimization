# Uncertainty Sensitivity Analysis

## Coursework purpose

This is one consolidated evaluation of the Stage-Two method, not four separate project components. It directly supports the Grade-A justification criterion, which asks for simulation of the future under risk and uncertainty followed by optimisation on the simulated scenario. The deterministic GA, chromosome, SVR, constraints and genetic operators remain unchanged; only the uncertainty conditions are varied.

## Conditions

| Condition | Change from the original model |
|---|---|
| Nominal | Original independent availability probabilities and 80/20 mean-to-tail fitness |
| Moderate disruption | Every availability probability reduced by 0.10 |
| Correlated disruption | Category-level cold-chain, regulated-cargo and express-connection shocks added |
| Risk averse | Original probabilities with fitness changed from 80/20 to 60/40 mean-to-tail weighting |

The correlated condition assigns a 15% scenario probability to a cold-chain disruption affecting perishable and pharmaceutical cargo, 12% to a regulated-cargo disruption affecting dangerous goods and 10% to an express-connection disruption affecting express cargo. These are transparent sensitivity assumptions, not measured airline event frequencies.

Every condition uses a population of 150, 250 generations, 30 common training scenarios and 1,000 unseen common evaluation scenarios. Each robust manifest is compared with the same Stage-One deterministic manifest under the corresponding held-out condition.

## Results

| Condition | Selected | Changed genes vs deterministic | Mean-profit difference | Worst-10% mean difference | Risk-adjusted difference |
|---|---:|---:|---:|---:|---:|
| Nominal | 42 | 6 | +$253.48 | +$221.64 | +$247.11 |
| Moderate disruption | 42 | 10 | −$651.70 | −$425.93 | −$606.55 |
| Correlated disruption | 45 | 11 | +$627.63 | +$6,338.81 | +$1,769.87 |
| Risk averse | 42 | 8 | +$71.35 | +$44.97 | +$60.80 |

All four robust manifests satisfied payload, volume, handling-budget and maximum zero-fuel-mass constraints.

## Findings

The nominal and stronger-risk-weight conditions again produced only small improvements. This confirms that the original independent availability model is mild: most probabilities are high, and simply placing more weight on the lower tail does not create substantially different decisions.

Reducing every probability by 0.10 increased variability, but the evolved manifest underperformed the deterministic manifest on the unseen scenarios. This is evidence that uniformly increasing independent cancellations does not automatically make the risk-aware solution better. It also indicates possible scenario-sample overfitting: 30 training scenarios can favour a manifest that does not generalise to a new set of 1,000 scenarios. The negative result should be reported rather than hidden.

Correlated disruption created the clearest operational value. The robust manifest improved average profit by $627.63 and worst-10% mean profit by $6,338.81. Its profit standard deviation was $11,450.32 compared with $13,793.80 for the deterministic manifest, a reduction of $2,343.48. This occurs because category-wide failures can remove several related consignments simultaneously, allowing the GA to benefit from a more diversified selection.

The central conclusion is that uncertainty-aware optimisation is most valuable when the uncertainty model reflects shared operational causes rather than only independent booking cancellations. This finding provides a stronger Grade-A justification while keeping the implemented system focused.

## Limitations

The disruption frequencies and independence assumptions are simulated, not estimated from Emirates records. Only one GA seed and one pair of scenario seeds were used in this focused experiment. The moderate-disruption result suggests that later final evaluation should include multiple GA seeds or more training scenarios before making claims about general performance. These additions belong in evaluation rather than in the core implementation.
