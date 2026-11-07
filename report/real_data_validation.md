# Real Operational Data Compatibility Check

## Dataset examined

The EUROCONTROL PRC 2025 Aircraft Fuel Burn Estimation dataset was examined as an independent operational source. It combines ACARS fuel-on-board reports, ADS-B trajectories and EUROCONTROL flight metadata from commercial flights operated between April and October 2025. The public archive is licensed under CC BY 4.0 and is available at DOI `10.5281/zenodo.19184662`.

Only the small flight-list and fuel-label tables were downloaded for the audit. The complete 4.9 GB trajectory collection was not downloaded. HTTP ranged access was used to inspect the single closest B777-300ER trajectory, avoiding a 3.3 GB training-archive download.

## Coverage findings

| Check | Finding |
|---|---:|
| Total B777-300ER (`B77W`) flights | 295 |
| B777-300ER CMB–DXB flights | 0 |
| B777-300ER flights within the OpenAP duration range of 225.5–252.5 minutes | 0 |
| Closest B777-300ER flight | Cairo–London, 275.47 minutes |
| Fuel-labelled intervals for the closest flight | 4 |
| Total labelled time for the closest flight | 71.99 minutes |
| Labelled share of its scheduled airborne duration | 26.13% |

## Why an accuracy score was not calculated

A direct RMSE or MAE would not be valid. The saved SVR predicts total fuel for a complete CMB–DXB flight from zero-fuel mass, cruise speed, cruise altitude, temperature deviation and full-flight duration. In contrast, PRC supplies fuel consumed during separate ACARS intervals. Those intervals can be discontinuous and do not necessarily cover the complete flight. Its flight metadata also does not supply zero-fuel mass or temperature deviation.

The closest B777-300ER flight is outside the duration range learned by the SVR, so using it would require extrapolation. Substituting invented values for its missing inputs and comparing a complete-flight prediction with partial interval fuel would create a numerical result, but that result would not measure model accuracy.

## Conclusion

The audit confirms that PRC is genuine operational evidence and that it contains B777-300ER records, but it cannot directly validate the present route-specific SVR. The honest conclusion is **external validation not currently possible with compatible public records**, rather than claiming an unsupported accuracy value.

For the coursework, the PRC audit remains valuable as critical evaluation. It demonstrates that the distinction between simulation accuracy and operational accuracy was investigated, identifies the exact data mismatch, and prevents the OpenAP test score from being misrepresented as real-world performance.

The SVR can remain the model selected for the simulated environment. Its reported RMSE, MAE and R² must be described as OpenAP-simulation results. A future operational validation would require complete-flight B777-300ER fuel burn together with zero-fuel mass and weather/temperature information for a route sufficiently close to CMB–DXB.

## Take-off mass consistency check

The relationship suggested between take-off mass, zero-fuel mass and the target was also tested on all 5,000 OpenAP records:

```text
loaded fuel = take-off mass − zero-fuel mass
```

This produces the total fuel loaded before departure, not the fuel consumed during the modelled trip. The generator deliberately calculates it as:

```text
loaded fuel = trip fuel consumption × 1.05 + 5,000 kg reserve
```

Therefore, trip fuel can be reconstructed by reversing the generator formula:

```text
reconstructed trip fuel = (loaded fuel − 5,000) ÷ 1.05
```

Across all records, loaded fuel and the target had a correlation effectively equal to 1.0. Reversing the formula reproduced the target with a maximum absolute difference of only 0.0014 kg, caused by CSV rounding.

This confirms that the mass columns are internally consistent. It does **not** constitute independent validation because take-off mass was created using the target itself. Supplying take-off mass to the regression model would reveal the answer indirectly and cause target leakage. Retaining zero-fuel mass as the ML input was therefore the correct decision.
