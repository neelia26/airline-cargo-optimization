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
