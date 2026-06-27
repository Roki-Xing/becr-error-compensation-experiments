# Moving Projection State Diagnostic

- `state_reset_explicit` resets state by design and therefore removes carried residual/history at refresh.
- `official_fira_carry` carries low-rank state without basis transport and uses no residual compensation.
- `projection_aware_transport` keeps low-rank history but transports it with the new basis overlap.
- `full_residual_current_projection` applies compensated decomposition `A_t = G_t + E_t` and updates residual with `Q_t - Z_eff_t`.

## Observed Result

- `full_residual_current_projection` keeps `reset_events=[]` and reaches `max_decomposition_error=0.000e+00`, `max_residual_error=0.000e+00`.
- `state_reset_explicit` records reset events at steps [2, 4], making refresh-induced resets explicit instead of silent.
- `official_fira_carry` and `projection_aware_transport` differ in final update norm (7.233e-01 vs 2.980e+00), isolating basis-coordinate mismatch from residual compensation.
