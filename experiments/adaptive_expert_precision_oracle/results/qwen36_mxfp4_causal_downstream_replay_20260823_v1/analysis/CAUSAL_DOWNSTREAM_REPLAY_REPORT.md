# PR #13 causal downstream replay pilot

## Outcome

The reconstruction audit passes exactly and the paired-tail impulse study covers all 40 MoE layers at the two initial PR #13 rates. This establishes a causal propagation measurement, but it does **not** yet establish an end-to-end streamed-model quality curve.

## Hard gates

- Validation groups reconstructed: 1,280.
- Group/rate reconstructions: 5,120.
- Max expert qenergy error: 1.776e-15.
- Max group qenergy error: 5.551e-17.
- Max all-Q4 reconstruction error: 4.441e-15.
- Repeated paired-tail hidden/router/logit differences: exactly zero for every layer.
- Test rows admitted or used: false.

## Isolated impulse magnitude by charged rate

| Charged bpw | Initial error energy median | Final hidden MSE median / p90 | Final route churn median / p90 | Logit KL median / p90 | Median abs delta NLL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.523478 | 2.431e-03 | 1.069e-04 / 2.065e-04 | 1.420% / 1.989% | 1.515e-03 / 2.930e-03 | 0.0059 |
| 0.998739 | 3.304e-04 | 9.423e-05 / 1.794e-04 | 1.420% / 2.017% | 1.403e-03 / 2.662e-03 | 0.0050 |

Each row summarizes 40 separate single-layer interventions; effects are not summed. Negative delta NLL values occur on this tiny cohort, so absolute delta NLL and non-negative logit KL are more useful impulse magnitudes here.

## Most logit-sensitive isolated layers

| Charged bpw | Layer | Delta NLL (95% sequence bootstrap) | Logit KL | Top-1 agreement | Final route churn |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.523478 | 1 | +0.0130 [-0.0182, +0.0387] | 4.552e-03 | 97.56% | 0.852% |
| 0.523478 | 4 | +0.0113 [-0.0091, +0.0372] | 3.637e-03 | 97.56% | 1.420% |
| 0.523478 | 6 | -0.0004 [-0.0168, +0.0254] | 3.074e-03 | 97.56% | 0.568% |
| 0.523478 | 5 | +0.0024 [-0.0370, +0.0403] | 2.952e-03 | 97.56% | 2.557% |
| 0.523478 | 7 | -0.0019 [-0.0284, +0.0139] | 2.927e-03 | 97.56% | 1.705% |
| 0.998739 | 6 | -0.0083 [-0.0225, +0.0076] | 3.826e-03 | 100.00% | 2.273% |
| 0.998739 | 7 | -0.0204 [-0.0462, -0.0053] | 2.917e-03 | 97.56% | 2.273% |
| 0.998739 | 4 | +0.0071 [-0.0040, +0.0278] | 2.909e-03 | 100.00% | 1.136% |
| 0.998739 | 3 | -0.0030 [-0.0333, +0.0199] | 2.786e-03 | 100.00% | 3.125% |
| 0.998739 | 0 | +0.0096 [-0.0094, +0.0210] | 2.649e-03 | 100.00% | 1.420% |

## Precision and trajectory controls

A fresh full GPU forward differs from the old CPU capture by RMSE 0.005481 and max absolute hidden error 0.406250; exact ordered top-8 routes match on 46.25% of stored rows. Therefore each intervention begins at the captured BF16 `xplus`, and its candidate and no-op tails run through the same resident GPU model. The no-op/candidate difference is causal within that paired trajectory. Drift from the historical CPU tail is retained in the propagation table as a separate control.

The approximately 0.999-bpw local errors are much smaller than the 0.523-bpw errors, but their final absolute hidden and logit disturbances are not proportionally smaller. In BF16, small perturbations can cross rounding and routing boundaries, so very large relative propagation coefficients at the high rate should be read together with absolute MSE, route churn and logit KL.

## Scientific boundary and next gate

- Cohort: three validation sequences and 32 captured positions per layer.
- Modes: isolated frozen PR #13 error at 0.523478 and 0.998739 charged bpw.
- Attention: complete sequence tails; errors can affect later positions.
- Not measured: simultaneous streaming of all layers, live allocation, physical wire bpw, repair-one-layer importance, or population perplexity.
- Predicted-H4 is explicitly excluded; a later predictor starts at H1.

The next quality gate is a supplementary full-sequence held-out capture and a genuine sequential streamed forward at all four PR #13 rates. Only that run can report the requested bpw-to-NLL/PPL curve, cross-layer interaction Gamma, and repair-one-layer importance.
