Figure 1 (2D theorem-regime diagnostics). We consider the quadratic objective
f(x,y)=0.5*(a x^2 + b y^2) under the theorem-regime conditions:
eta*a = 0.5 < 2*eps_a = 2.0, M0 = 0.25, and eta*b*M0 = 0.125 <= 1/2.

Panels: (a) ||grad f(x_t,y_t)||, (b) |y_t|, (c) raw phi_t vs clipped/applied s_t,
(d) cumulative sum(phi_t) vs sum(s_t), (e) residual norm ||e_t||, (f) objective f(x_t,y_t).

Key mechanism: raw recovery has sum(phi_t) plateauing (summable), so y_t stalls at a nonzero value;
clipped recovery / BECR yields a non-summable applied scale sum(s_t), so y_t -> 0.
