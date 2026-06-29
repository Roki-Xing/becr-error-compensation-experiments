# High-Dimensional Final Metrics

| Method | mean ||grad|| | mean ||grad_parallel|| | mean ||grad_perp|| | mean residual |
|---|---:|---:|---:|---:|
| proj_baseline | 9.765097e+00 | 2.795085e-01 | 9.757096e+00 | 0.000000e+00 |
| fira_raw | 1.785777e+00 | 2.795085e-01 | 1.506269e+00 | 0.000000e+00 |
| fira_clipped | 2.795085e-01 | 2.795085e-01 | 1.295155e-27 | 0.000000e+00 |
| becr | 2.795085e-01 | 2.795085e-01 | 7.020873e-32 | 8.068520e-32 |
| rho_no_lower_bound | 2.387548e+00 | 2.795085e-01 | 2.108039e+00 | 1.256995e+03 |
| coupled_unbounded | 2.387548e+00 | 2.795085e-01 | 2.108039e+00 | 1.256995e+03 |
