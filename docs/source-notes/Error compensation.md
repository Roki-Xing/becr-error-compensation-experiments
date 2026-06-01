## 1. One-paragraph diagnosis of the project

This project is promising, but the central danger is **mistaking norm recovery for Error Feedback**. Fira’s update does not merely project the gradient; it decomposes the full gradient into a projected part and an orthogonal residual, applies low-rank Adam to the projected part, and scales the residual by a matrix- or column-level norm ratio. That compensates **some lost gradient magnitude**, but it does **not** accumulate residual error over time and does **not** reconstruct missing Adam first- and second-moment statistics for the residual coordinates. SubTrack++ reuses essentially the same recovery-scaling idea while adding projection-aware Adam states; LDAdam goes further and explicitly compensates both gradient-compression and optimizer-state-compression errors through generalized Error Feedback; GoLore gives the key warning that SVD-selected low-rank subspaces can be dominated by anisotropic stochastic noise, causing non-convergence. So the publishable opening is not “Fira is Error Feedback,” but rather: **Fira-style recovery is a one-step magnitude compensation mechanism; it explains some empirical success, but it is theoretically weaker than residual Error Feedback and state-aware compensation.** ([OpenReview](https://openreview.net/pdf/623f06afa08d7e3bfb893ecd92b1f91f5238d10f.pdf))

------

## 2. The most promising research direction

The best primary direction is a **Branch A → positive-repair paper**:

> **Show that norm recovery alone is not Error Feedback, construct simple failures caused by scalar/column-wise adaptive-state mismatch, then design a bounded residual-compensated Fira-Adam variant with a provable SGD theorem and a carefully scoped Adam theorem or theorem-like proposition.**

The main contribution should be framed as:

> Low-rank Adam methods suffer from two distinct errors: **gradient subspace error** and **optimizer-state compression error**. Fira-style recovery partially fixes the former by scaling discarded raw-gradient components, but it does not accumulate residuals and does not correct missing adaptive moments. Adding bounded residual compensation and projection-aware state transport yields a mechanism that is both closer to Error Feedback and theoretically controllable.

The backup path is weaker but safer:

> Prove that Fira-style norm recovery is an **approximate one-step Error Feedback surrogate** under strong assumptions: good projection quality, bounded scaling factors, aligned adaptive preconditioning, and slowly changing subspaces. Then show experimentally that when these assumptions fail, explicit residual compensation improves stability.

The backup is less novel because LDAdam already proves a generalized EF result for low-dimensional Adam. The primary path is stronger because it explains **why Fira/SubTrack++ recovery works empirically but remains theoretically incomplete**.

------

## 3. Dependency graph of the four reference papers

### Conceptual dependency graph

```text
GaLore-style gradient projection
        |
        |-- Fira:
        |     full-rank raw gradient is available;
        |     low-rank Adam states are kept;
        |     discarded gradient residual is scaled by low-rank Adam norm ratio.
        |
        |-- GoLore paper:
        |     SVD subspaces can be noise-dominated;
        |     gives stochastic non-convergence counterexample;
        |     fixes by random low-rank projection.
        |
        |-- LDAdam:
        |     low-dimensional Adam states;
        |     projection-aware state transport;
        |     generalized Error Feedback for both gradient and optimizer-state compression.
        |
        |-- SubTrack++:
              Grassmannian subspace tracking;
              projection-aware Adam states;
              Fira/APOLLO-style recovery scaling.
```

### Technical comparison map

| Paper                                     | Optimizer type                                               | Compression / subspace mechanism                             | Adam, SGD, or adaptive states?                               | EF / compensation?                                         | What error is compensated?                                   | Per-step update rank                                         | Convergence proved?                                          | Main assumptions                                             | Main theorem type                                            | Key limitation                                               | Open gap for us                                              |
| ----------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Fira**                                  | GaLore-like low-rank optimizer plus full-rank residual update | SVD projection: (R_t=P_t^\top G_t); residual (G_t-P_tR_t) retained | Low-rank Adam states on (R_t)                                | Norm-based scaling + norm-growth limiter, not classical EF | Scales discarded raw-gradient residual using (\phi_t=\|\psi_t(R_t)\|/\|R_t\|), column-wise variant also used | Full-rank, because update includes (P_t\psi_t(R_t)+\phi_t(G_t-P_tR_t)) | No full convergence theorem for the optimizer                | Empirical similarity between low-rank and full-rank scaling factors; bounded-scaling-style appendix analysis | Approximation/scaling-factor analysis, not stationarity convergence | Does not accumulate residuals; residual part has no full Adam moments; limiter stabilizes norm growth but does not fix moment bias | Is norm recovery a controlled compressor? When does scalar/column scaling fail? ([OpenReview](https://openreview.net/pdf/623f06afa08d7e3bfb893ecd92b1f91f5238d10f.pdf)) |
| **SubTrack++**                            | Low-rank Adam with subspace tracking                         | Grassmannian tracking; rank-1 geodesic updates; projection-aware state transport | Adam states in low-rank coordinates                          | Recovery scaling + projection-aware optimizer              | Discarded gradient components and subspace-change state misalignment | Full-rank if recovery term is included                       | Partial; theory analyzes Grassmannian/subspace tracking under restrictive structural assumptions | Smooth/structured gradient form, bounded weights, and in its theorem the projection matrices are effectively assumed fixed for convergence | Tracking/convergence-style theorem, not a full stochastic Adam + recovery theorem | Recovery scaling inherits Fira’s limitation; theorem does not fully cover stochastic Adam recovery with changing subspaces | Can we prove the recovery term is safe only with explicit residual/state compensation? ([arXiv](https://arxiv.org/pdf/2502.01586)) |
| **GoLore / subspace optimization theory** | GaLore-style subspace optimization; theory mainly MSGD with momentum projection | SVD projection for GaLore; random Stiefel projection for GoLore | Algorithms include Adam/MSGD variants; core guarantees mainly for MSGD + momentum projection | No EF; GoLore avoids SVD noise bias by random projection   | Does not compensate error; changes projection distribution to avoid noise-dominated subspaces | Low-rank per step                                            | Yes: deterministic GaLore, large-batch GaLore, isotropic-noise GaLore, and random-projection GoLore; also non-convergence of stochastic SVD-GaLore | Lower boundedness, (L)-smoothness, unbiased stochastic gradients, bounded variance; extra isotropic-noise or large-batch assumptions for GaLore | Non-convergence counterexample and (O(1/\sqrt T))-type stationarity for GoLore | Does not analyze Fira-style full-rank residual recovery or Adam state distortion | Use its anisotropic-noise counterexample template against recovery/scaling mechanisms ([arXiv](https://arxiv.org/pdf/2410.11289)) |
| **LDAdam**                                | Low-dimensional Adam / AMSGrad-like adaptive optimizer       | Low-rank subspace (U_t), block power iteration / SVD-like projection, projection-aware state update | Adaptive optimizer states in low-dimensional coordinates     | Yes: generalized Error Feedback                            | Explicitly compensates both gradient compression error and optimizer-state compression error | Low-rank per step, but changing subspaces explore full parameter space over time | Yes                                                          | (L)-smooth lower-bounded objective, unbiased bounded stochastic gradients, bounded variance, contractive projection condition | Nonconvex (O(1/\sqrt T)) stationarity; PL-rate result        | Already covers a strong version of “Adam + state EF,” so novelty risk is high | Our algorithm must not simply rename LDAdam; it must isolate Fira-style norm recovery and show what extra residual mechanism is needed ([arXiv](https://arxiv.org/pdf/2410.16103)) |

------

## 4. A precise mathematical abstraction of the problem

We start from stochastic nonconvex optimization:

[
\min_{x\in\mathbb R^d} f(x),\qquad
g_t=\nabla f(x_t)+\xi_t,\qquad
\mathbb E[\xi_t\mid x_t]=0.
]

Let (C_t:\mathbb R^d\to\mathbb R^d) be a low-rank or compressed operator. In vector form,

[
C_t(g)=P_tP_t^\top g,\qquad P_t\in\mathbb R^{d\times r},\quad P_t^\top P_t=I_r,\quad r\ll d.
]

For matrix parameters (W_t\in\mathbb R^{m\times n}),

[
G_t=\nabla_W f(W_t)+\Xi_t.
]

A left-projection model is

[
R_t=P_t^\top G_t\in\mathbb R^{r\times n},\qquad
C_t(G_t)=P_tR_t=P_tP_t^\top G_t,
]

and the discarded residual is

[
Z_t=G_t-P_tP_t^\top G_t=G_t-P_tR_t.
]

### Model 1: Pure projected gradient / GaLore-style model

[
R_t=P_t^\top G_t,
]
[
M_t=\beta_1M_{t-1}+(1-\beta_1)R_t,
]
[
V_t=\beta_2V_{t-1}+(1-\beta_2)R_t^{\odot 2},
]
[
W_{t+1}=W_t-\eta_tP_t\frac{M_t}{\sqrt{V_t}+\varepsilon}.
]

Stored memory: (P_t), (M_t,V_t\in\mathbb R^{r\times n}).
Low-rank: optimizer states and per-step update.
Full-rank: weights and raw gradients during backprop.
Information loss: (Z_t=G_t-P_tR_t) is discarded.
Convergence measure: full gradient norm (\mathbb E|\nabla f(W_t)|_F^2), not merely projected gradient norm.

### Model 2: Classical Error Feedback compressed SGD

[
a_t=g_t+e_t,
]
[
c_t=C_t(a_t),
]
[
e_{t+1}=a_t-c_t,
]
[
x_{t+1}=x_t-\eta_t c_t.
]

Stored memory: residual (e_t\in\mathbb R^d), possibly reusable as a gradient buffer.
Low-rank/compressed: update (c_t).
Full-rank: residual buffer.
Information loss: compression error (a_t-C_t(a_t)).
Compensation: lost information is accumulated and reintroduced next step.
Convergence measure: (\frac1T\sum_t\mathbb E|\nabla f(x_t)|^2).

### Model 3: Adam with compressed gradients and compressed optimizer states

[
r_t=P_t^\top g_t,
]
[
m_t=\beta_1\mathcal T_t(m_{t-1})+(1-\beta_1)r_t,
]
[
v_t=\beta_2\mathcal S_t(v_{t-1},m_{t-1})+(1-\beta_2)r_t^{\odot 2},
]
[
x_{t+1}=x_t-\eta_tP_t\frac{m_t}{\sqrt{v_t}+\varepsilon}.
]

Here (\mathcal T_t,\mathcal S_t) are state-transport maps when (P_{t-1}\neq P_t).
Stored memory: (P_t,m_t,v_t).
Low-rank: (m_t,v_t).
Information loss: both (g_t-C_t(g_t)) and the untracked full-space moments.
Compensation needed: projection-aware state transport and possibly residual EF.
Convergence measure: full gradient norm.

### Model 4: Fira-style norm recovery

For matrix gradients with (m\le n),

[
R_t=P_t^\top G_t,
\qquad
\psi_t(R_t)=\text{low-rank Adam output on }R_t.
]

Matrix-level scaling:

[
\phi_t(R_t)=\frac{|\psi_t(R_t)|_F}{|R_t|_F}.
]

Column-wise scaling:

[
\phi_t(R_t)*i=\frac{|\psi_t(R*{t,:,i})|*2}{|R*{t,:,i}|_2}.
]

Fira-style update:

# [ W_{t+1}

W_t-\eta_tP_t\psi_t(R_t)
-\eta_t\phi_t(R_t)\odot \bigl(G_t-P_tR_t\bigr).
]

The norm-growth limiter clips the recovered residual

[
S_t=\phi_t(R_t)\odot(G_t-P_tR_t)
]

when

[
\frac{|S_t|*F}{|S*{t-1}|_F}>\gamma,
\qquad
S_t\leftarrow \frac{S_t}{|S_t|*F}\gamma|S*{t-1}|_F.
]

This is not EF: once (S_t) is applied, there is no residual state (e_{t+1}) ensuring future correction of what was under- or over-applied. It is also not full Adam: the residual part gets scalar or column-wise scaling, not element-wise first- and second-moment adaptation. ([OpenReview](https://openreview.net/pdf/623f06afa08d7e3bfb893ecd92b1f91f5238d10f.pdf))

### Model 5: LDAdam-style generalized Error Feedback

LDAdam uses an accumulator

[
A_t=g_t+\xi_t,
]

low-rank projection

[
a_t=U_t^\top A_t,
]

projection-aware state transport

[
m_{t-1/2}=U_t^\top U_{t-1}m_{t-1},
]

and then low-rank Adam/AMSGrad-style updates. Its generalized residual loading has the form

# [ \xi_{t+1}

[A_t-U_ta_t]
+
\frac{\beta_1}{1-\beta_1}
[U_{t-1}m_{t-1}-U_tm_{t-1/2}],
]

up to the paper’s practical/analytical variants. The first bracket is gradient compression error; the second bracket is optimizer-state compression/transport error. LDAdam’s convergence theorem relies on smoothness, bounded stochastic gradients, bounded variance, and a projection contraction condition. ([arXiv](https://arxiv.org/pdf/2410.16103))

------

## 5. Three candidate theorem statements

### Theorem candidate 1: Norm recovery alone is not Error Feedback

**Claim.** There exists a two-dimensional smooth convex quadratic and an Adam-like Fira-style scalar recovery rule for which the method does not converge, even though the full gradient is available.

Take

[
f(x,y)=\frac12(x^2+y^2),\qquad P=e_1,
]

and use the simplified adaptive correction

[
\psi(r)=\operatorname{sign}(r),
]

which corresponds to the (\beta_1=\beta_2=0,\varepsilon=0) normalized Adam limit. Then

[
R_t=x_t,\qquad
\phi_t=\frac{|\psi(R_t)|}{|R_t|}=\frac1{|x_t|}.
]

The Fira-style update becomes

[
x_{t+1}=x_t-\eta\operatorname{sign}(x_t),
]
[
y_{t+1}=y_t-\eta\frac{y_t}{|x_t|}.
]

If (x_0=\eta/2) and (y_0\ne0), then

[
x_1=-\eta/2,\quad x_2=\eta/2,
]

and

[
y_{t+1}=y_t-2y_t=-y_t.
]

Thus (|(x_t,y_t)|) does not converge to zero. This is not a final theorem about production Fira, because real Adam uses momentum, (\varepsilon), and a limiter. But it is a clean theorem-level warning: **scalar norm recovery can create an effective residual-coordinate stepsize controlled by the projected coordinate, not by the residual coordinate.**

### Theorem candidate 2: Bounded recovery + residual EF gives SGD convergence

Assume (f) is lower bounded and (L)-smooth, (g_t) is unbiased with bounded variance (\sigma^2), and a compressor satisfies

[
\mathbb E|C_t(u)-u|^2\le (1-\delta)|u|^2.
]

Define a recovery compressor

[
\mathcal C_t^{s}(u)=C_t(u)+s_t(I-C_t)u,
]

with bounded scale

[
s_t\in[s_{\min},2-s_{\min}]
]

for some (s_{\min}\in(0,1]). Use Error Feedback:

[
a_t=g_t+e_t,
]
[
\tilde g_t=\mathcal C_t^{s}(a_t),
]
[
e_{t+1}=a_t-\tilde g_t,
]
[
x_{t+1}=x_t-\eta_t\tilde g_t.
]

Then (\mathcal C_t^s) is itself contractive with an effective contraction depending on (\delta) and (s_{\min}), and with (\eta=\Theta(1/\sqrt T)),

[
\frac1T\sum_{t=0}^{T-1}\mathbb E|\nabla f(x_t)|^2
\le
O!\left(\frac{f(x_0)-f^\star}{\sqrt T}\right)
+
O!\left(\frac{\sigma^2}{\sqrt T}\right),
]

with constants worsening as compression becomes more aggressive. This would be our first clean positive theorem. The hardest step is bounding (|e_t|) when (s_t) is data-dependent.

### Theorem candidate 3: Projection-aware Adam with generalized residual has LDAdam-type convergence

Assume (f) is (L)-smooth and lower bounded, stochastic gradients are unbiased and bounded, variance is bounded, and (U_t) satisfies a contractive projection condition

[
|(I-U_tU_t^\top)B_t|\le q_r|B_t|,\qquad q_r<1.
]

Use projection-aware state transport and a generalized residual containing both

[
\text{gradient error: } A_t-U_tU_t^\top A_t
]

and

[
\text{state error: } U_{t-1}m_{t-1}-U_tm_{t-1/2}.
]

Then a Fira-style full-rank recovered Adam variant with bounded recovery scale and AMSGrad-type monotone denominator satisfies

[
\frac1T\sum_{t=1}^T
\mathbb E|\nabla f(x_t)|^2
\le
O!\left(\frac{C(q_r,\beta_1,\beta_2,s_{\max})}{\sqrt T}\right)
+
O!\left(\frac1T\right).
]

This theorem is risky because it is close to LDAdam. The novelty must be that the update contains an explicit **Fira-style recovered full-rank residual term**, while LDAdam’s main update remains low-dimensional at each step.

------

## 6. Three candidate counterexamples

### Counterexample 1: Pure projection loses direction completely

Let

[
f(x,y)=\frac12(x^2+y^2),
\qquad
P=e_1e_1^\top.
]

Pure projected gradient descent gives

# [ (x_{t+1},y_{t+1})

# (x_t,y_t)-\eta(x_t,0)

((1-\eta)x_t,y_t).
]

Thus

[
y_t=y_0
]

forever, and

[
|\nabla f(x_t,y_t)|^2=x_t^2+y_0^2\to y_0^2.
]

This is a real counterexample for pure projected methods with a fixed bad subspace. It is **not** a Fira counterexample, because Fira would use the residual direction (G_t-P_tR_t). Its purpose is to isolate **direction loss**.

### Counterexample 2: Fira-style scalar recovery creates denominator mismatch

Let again

[
f(x,y)=\frac12(x^2+y^2),\qquad P=e_1.
]

Use the simplified normalized Adam correction

[
\psi(r)=\operatorname{sign}(r),
]

so

[
\phi_t=\frac{1}{|x_t|}.
]

Then

[
x_{t+1}=x_t-\eta\operatorname{sign}(x_t),
]
[
y_{t+1}=y_t-\eta\frac{y_t}{|x_t|}.
]

Set (x_0=\eta/2). Then (x_t) cycles between (\eta/2) and (-\eta/2), while

[
y_{t+1}=-y_t.
]

So the method does not converge. The failure is **not direction loss**; the residual direction is correct. The failure is **adaptive denominator mismatch**: the (y)-coordinate stepsize is controlled by the projected (x)-coordinate scale.

This is the cleanest early counterexample to develop.

### Counterexample 3: Stale subspace makes recovery undefined or zero

Let

[
f(x,y)=\frac12y^2,
\qquad
\nabla f(x,y)=(0,y),
\qquad
P=e_1.
]

Then

[
R_t=P^\top g_t=0,
\qquad
g_t-P R_t=(0,y_t).
]

If the implementation uses

[
\phi_t=\frac{|\psi(R_t)|}{|R_t|+\varepsilon_s},
]

then (\psi(0)=0), hence

[
\phi_t=0.
]

The Fira-style residual update becomes zero:

[
(x_{t+1},y_{t+1})=(x_t,y_t).
]

Thus the method is stuck away from the optimum. This is a real counterexample for **fixed or stale subspaces with zero projected gradient**. It is only a warning example for periodically refreshed SVD-Fira, because a refresh using the current full gradient would choose the (y)-direction.

------

## 7. One recommended algorithmic modification

I recommend starting with:

> **Bounded Residual-Compensated Fira-Adam**, abbreviated here as **BR-Fira-Adam**.

For one matrix parameter (W_t):

**State.**
Low-rank Adam states (M_t,V_t\in\mathbb R^{r\times n}), projection (P_t\in\mathbb R^{m\times r}), and a full-size residual buffer (E_t\in\mathbb R^{m\times n}). The residual buffer can be stored in the gradient buffer, following the same practical idea LDAdam uses for EF storage. ([arXiv](https://arxiv.org/pdf/2410.16103))

**Update.**

[
A_t=G_t+E_t.
]

Project:

[
R_t=P_t^\top A_t.
]

Projection-aware Adam state update:

[
M_t=\beta_1\mathcal T_t(M_{t-1})+(1-\beta_1)R_t,
]
[
V_t=\beta_2\mathcal S_t(V_{t-1},M_{t-1})+(1-\beta_2)R_t^{\odot2}.
]

Low-rank Adam output:

[
U_t=P_t\frac{M_t}{\sqrt{V_t}+\varepsilon}.
]

Residual direction:

[
Z_t=A_t-P_tR_t.
]

Bounded Fira scale:

# [ s_{t,i}

\operatorname{clip}
\left(
\frac{\left|M_{t,:,i}/(\sqrt{V_{t,:,i}}+\varepsilon)\right|*2}
{|R*{t,:,i}|*2+\varepsilon_s},
\ s*{\min},\ s_{\max}
\right),
]

with theory-friendly choice

[
0<s_{\min}\le s_{t,i}\le s_{\max}<2.
]

Recovered update:

[
\Lambda_t=s_t\odot Z_t.
]

Weight update:

[
W_{t+1}=W_t-\eta_t(U_t+\Lambda_t).
]

Residual loading:

[
E_{t+1}=A_t-\bigl(P_tR_t+s_t\odot Z_t\bigr)
]

for the SGD theorem, and optionally

# [ E_{t+1}

A_t-\bigl(P_tR_t+s_t\odot Z_t\bigr)
+
\alpha_{\mathrm{state}}
\bigl(P_{t-1}M_{t-1}-P_t\mathcal T_t(M_{t-1})\bigr)
]

for the Adam-state compensation version.

Why this is not just Fira: it has explicit residual accumulation and bounded recovery, so under- or over-recovered components are carried forward.

Why this is not just LDAdam: it keeps the Fira-style **full-rank recovered residual update**. LDAdam’s core theoretical update is low-dimensional at each step; BR-Fira-Adam studies whether norm recovery can be made into a controlled compressor.

Why this is not just SubTrack++: SubTrack++ uses recovery scaling and projection-aware states, but does not formulate the recovery term as a bounded EF compressor with a residual-loading equation.

The first theorem should be for the SGD version. The Adam version should be developed only after the counterexamples and SGD proof are clean.

------

## 8. Four-week execution plan

### Week 1: Paper dissection + counterexample simulator

Deliverables:

1. Exact implementation of five toy optimizers:
   - projected SGD,
   - EF-SGD,
   - Fira-style recovery,
   - Fira + limiter,
   - BR-Fira.
2. 2D quadratic experiments for:
   - fixed bad subspace,
   - stale subspace,
   - scalar denominator mismatch,
   - anisotropic noise,
   - rotating gradient.
3. A two-page technical note:
   - which counterexamples are real,
   - which are only warning examples,
   - which assumptions each one violates.

Pass condition: reproduce at least one non-convergence or instability plot where Fira-style scaling fails but explicit residual compensation fixes or reduces the issue.

### Week 2: SGD theorem

Deliverables:

1. Formal definition of recovery compressor

[
\mathcal C_t^s(u)=C_t(u)+s_t(I-C_t)u.
]

1. Lemma: effective contraction of (\mathcal C_t^s).
2. Lemma: bounded EF residual.
3. Nonconvex convergence theorem:

[
\frac1T\sum_t\mathbb E|\nabla f(x_t)|^2=O(T^{-1/2}).
]

Pass condition: a proof that does not rely on Adam, diagonal preconditioning, or LLM-specific empirical scaling similarity.

### Week 3: Adam-state analysis + small neural experiments

Deliverables:

1. Decide whether the Adam theorem is:
   - full theorem,
   - theorem under AMSGrad-style denominator monotonicity,
   - or proposition with explicit state-error term.
2. Implement projection-aware state transport.
3. Run MNIST/CIFAR-10 MLP/CNN experiments comparing:
   - Adam,
   - compressed Adam,
   - Fira-style recovery,
   - EF-compressed Adam,
   - BR-Fira-Adam.

Metrics:
[
|\nabla f|,\quad
|E_t|,\quad
\angle(u_t,\nabla f_t),\quad
|Z_t|,\quad
\text{Adam }v_t\text{ bias}.
]

Pass condition: show one diagnostic where recovery scaling reduces loss versus pure projection, and residual compensation improves stability or angle alignment.

### Week 4: Paper skeleton + transformer proxy

Deliverables:

1. Full paper outline with theorem statements.
2. Tiny GPT or transformer proxy experiment:
   - small WikiText-2 or tiny Shakespeare,
   - ranks (r\in{4,8,16,32}),
   - compare loss spikes and validation loss.
3. Final positioning against LDAdam:
   - what we borrow,
   - what we do not claim,
   - where our mechanism differs.

Pass condition: a coherent “mechanism paper” story with one negative theorem, one positive theorem, and diagnostic experiments.

------

## 9. What to read first and what to skip temporarily

Read first:

1. **Fira Sections 3–4 and Appendix E/G/H.** Focus on Eq. 10, Eq. 11, the norm-growth limiter, and the appendix assumptions on scaling-factor approximation. Ignore large benchmark tables at first. ([OpenReview](https://openreview.net/pdf/623f06afa08d7e3bfb893ecd92b1f91f5238d10f.pdf))
2. **GoLore paper Sections 3–5 and Appendix B.2/B.5.** Focus on the anisotropic-noise counterexample, the non-convergence theorem, and the random-projection convergence theorem. ([arXiv](https://arxiv.org/pdf/2410.11289))
3. **LDAdam Sections 3.2–4.** Focus on projection-aware optimizer states, generalized EF, the contraction assumption, and the nonconvex theorem. ([arXiv](https://arxiv.org/pdf/2410.16103))
4. **SubTrack++ Section 2 and Algorithm 1.** Focus on how it combines Grassmannian tracking, projection-aware states, and recovery scaling. Treat its convergence section cautiously because it does not fully prove stochastic Adam + recovery scaling under changing subspaces. ([arXiv](https://arxiv.org/pdf/2502.01586))

Skip temporarily:

- Large-scale LLM benchmark tables.
- Fine-tuning dataset details.
- Systems throughput comparisons.
- Broad LoRA background.
- SubTrack++ geometry details beyond the update rule, unless we decide to work on slowly changing subspace proofs.

------

## 10. Concrete questions we need answered before going deeper

1. Which exact Fira implementation are we targeting: matrix-level scaling, column-wise scaling, or both?
2. How is (\phi_t) implemented when (|R_t|=0) or nearly zero? Is there an (\varepsilon_s), and is the scale set to zero, one, or clipped?
3. Is the norm-growth limiter applied only to the recovered residual (S_t=\phi_t(G_t-P_tR_t)), or to the full update (P_t\psi_t(R_t)+S_t)?
4. Are we allowed to store a full-size residual buffer across steps if it reuses the gradient buffer, or must optimizer-persistent memory remain strictly low-rank?
5. Is the target theorem allowed to start with SGD, or do we need the first submitted version to include Adam/AMSGrad?
6. Are we trying to improve Fira directly, SubTrack++ recovery scaling, or a more general class of low-rank Adam methods?
7. Do we want the final algorithm to produce full-rank updates every step, like Fira, or low-rank updates over changing subspaces, like LDAdam?
8. What scale of experiments is realistic in four weeks: 2D + MNIST/CIFAR only, or also a tiny transformer?
9. Should the paper be positioned as a theory paper with diagnostic experiments, or as an optimizer paper with LLM proxy experiments?
10. Are we comfortable making LDAdam the main “closest prior work,” meaning our novelty must be explicitly narrower but sharper: **Fira-style norm recovery is not generalized Error Feedback unless residual/state compensation is added**?



## 1. Diagnosis of the project

This is a promising theory-driven optimizer project, but it must be narrowed sharply. The key technical tension is that Fira-like methods do **not** simply compress gradients: they store **Adam statistics only in a low-rank subspace**, then try to recover a **full-rank update** by scaling the discarded gradient component. This is not classical Error Feedback, because no residual is accumulated and no telescoping cancellation of compression error is obvious. The strongest publishable path is therefore not “Fira is Error Feedback,” but rather: **Fira-style norm recovery is a magnitude-only compensation mechanism; it explains part of the empirical success, but it is insufficient for directional, stale-subspace, and adaptive-state errors. Adding explicit residual/state-aware compensation yields a method with provable convergence.** Fira motivates this because it decomposes (G_t) into (P_tR_t) and (G_t-P_tR_t), applies Adam only to (R_t=P_t^\top G_t), then updates with
[
W_{t+1}=W_t-\eta P_t\psi_t(R_t)-\eta \phi_t(R_t)(G_t-P_tR_t),
]
where (\phi_t) is a norm ratio; its limiter caps growth of the recovered component.

------

## 2. Most promising research direction

The best primary direction is **Branch A with a constructive fix**:

> **Show that norm recovery is not full error compensation: it can fix update magnitude but not missing residual accumulation, stale subspace, or Adam state distortion. Then propose a bounded residual-compensated Fira-Adam/AMSGrad variant and prove convergence first for SGD, then for an AMSGrad-style adaptive version.**

This is stronger than trying to prove original Fira directly. It gives a clean story: Fira is empirically effective because it reintroduces discarded full-rank gradient directions, but it is not theoretically sufficient unless scaling is bounded, subspace coverage is controlled, and optimizer-state compression is handled.

A compact evaluation of the five theorem-level questions:

| Question                                                    | Difficulty  | Novelty | Risk                                        | My recommendation                               |
| ----------------------------------------------------------- | ----------- | ------- | ------------------------------------------- | ----------------------------------------------- |
| Q1: Is Fira approximate EF?                                 | Medium      | Medium  | Risk of being semantic                      | Good as interpretation lemma, not main theorem. |
| Q2: Does Fira converge?                                     | High        | High    | Original scaling may be unbounded/undefined | Try only under bounded-scaling assumptions.     |
| Q3: Counterexample for magnitude-only recovery              | Medium      | High    | Needs not be trivial                        | Best first target.                              |
| Q4: Modified Fira with residual/projection-aware correction | Medium-high | High    | Must distinguish from LDAdam                | Best primary path.                              |
| Q5: Transfer EF to Adam with state distortion               | Very high   | High    | Very close to LDAdam                        | Good backup/extension, not first theorem.       |

The backup path is **bounded Fira-SGD/Fira-AMSGrad convergence**: assume (\phi_t\in[\phi_{\min},\phi_{\max}]), projection quality or randomized coverage, and prove first-order stationarity. This is less ambitious but likely executable.

------

## 3. Dependency graph of the four reference papers

The intellectual graph is:

[
\text{GaLore-style low-rank gradient projection}
\rightarrow
\begin{cases}
\textbf{Fira}: \text{full-rank recovery by norm scaling},\
\textbf{GoLore}: \text{random projection fixes SVD noise bias},\
\textbf{LDAdam}: \text{projection-aware Adam states + generalized EF},\
\textbf{SubTrack++}: \text{Grassmannian tracking + projection-aware optimizer + Fira-like recovery}.
\end{cases}
]

A technical comparison:

| Paper                                     | Optimizer type                                          | Compression / subspace mechanism                             | Adam, SGD, or states?                                        | EF / compensation?                                           | What error is compensated?                                   | Update rank                                                 | Convergence proved?                                          | Main assumptions                                             | Main theorem type                                     | Key limitation                                               | Open gap                                                     |
| ----------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Fira**                                  | Plug-in low-rank optimizer memory with full-rank update | GaLore-style projection (R_t=P_t^\top G_t), discarded part (G_t-P_tR_t) | Adam-like states only for (R_t)                              | No classical EF; uses norm recovery (\phi_t) and norm-growth limiter | Lost magnitude/scaling of discarded gradient component       | Full-rank update; low-rank states                           | No standard stochastic nonconvex convergence theorem in main paper | Empirical similarity of low-rank and full-rank Adam scaling; limiter stabilizes spikes | Empirical mechanism + scaling analysis                | Direction and optimizer-state errors not fully compensated; (\phi_t) may be unstable/undefined | Theory of recovery scaling and limiter                       |
| **SubTrack++**                            | Geometry-based low-rank optimizer                       | Grassmannian subspace tracking; updates subspace using residual and geodesic step | Projection-aware Adam states                                 | Recovery scaling, not explicit EF                            | Subspace drift, coordinate-state mismatch, discarded-gradient magnitude | Full-rank if recovery term included                         | Partial: theorem for Grassmannian tracking under restrictive assumptions | Assumes projection matrices remain constant in proof; special gradient form and Lipschitz continuity | Contraction of projected quantity (\|P_t\|_F\to0)     | Slowly changing subspace theory remains open; recovery scaling not fully analyzed | Theory for stochastic recovery + Adam states                 |
| **GoLore / subspace optimization theory** | Low-rank subspace optimizer                             | GaLore: SVD projection; GoLore: random Stiefel projection    | MSGD/Adam variants; theory mainly MSGD with momentum projection | No EF; uses random projection and momentum projection        | SVD subspace bias under anisotropic noise                    | Low-rank update                                             | Yes: GaLore counterexample; deterministic/large-batch/isotropic GaLore; stochastic GoLore | Smooth lower-bounded objectives, unbiased bounded-variance gradients; random projection for GoLore | Nonconvergence theorem and (O(1/\sqrt T)) GoLore rate | Does not handle Fira-style full-rank recovery or full Adam-state distortion | How recovery scaling interacts with stochastic subspace failure |
| **LDAdam**                                | Low-dimensional Adam / AMSGrad                          | Low-rank projection via block power iteration; projection-aware state transfer | Compressed first/second Adam states                          | Yes: generalized EF                                          | Both gradient projection error and optimizer-state compression error | Low-rank update each step; full-space exploration over time | Yes                                                          | Smooth lower-bounded objective, unbiased bounded gradients, bounded variance, contractive projection | Nonconvex (O(1/\sqrt T)), PL (O(\log T/T))            | Full EF buffer is memory-neutral only if stored in gradient buffer; not full-rank update per step | Combine LDAdam-style state EF with Fira full-rank recovery without losing novelty |

Fira and SubTrack++ both rely on recovery scaling: SubTrack++ explicitly says it uses scaling information from the low-rank stateful optimizer to recover discarded gradient components, and its algorithm includes
[
W_t \leftarrow W_{t-1}-\alpha \widehat G_t-\alpha \phi_t(G_t)(G_t-S_t\widetilde G_t).
]
([arXiv](https://arxiv.org/pdf/2502.01586))
LDAdam is the closest prior art for our proposed fix because it already introduces projection-aware optimizer-state updates and a generalized EF buffer for both gradient and state compression.
GoLore is the closest prior art for counterexamples: it shows GaLore can fail under anisotropic stochastic noise because SVD may chase noise near stationarity, and it fixes this with random low-rank projection. ([arXiv](https://arxiv.org/pdf/2410.11289))

------

## 4. Precise mathematical abstraction of the problem

We start with stochastic nonconvex optimization:
[
\min_{x\in\mathbb R^d} f(x),\qquad
g_t=\nabla f(x_t)+\xi_t,
]
with
[
\mathbb E[g_t\mid x_t]=\nabla f(x_t),\qquad
\mathbb E|g_t-\nabla f(x_t)|^2\le \sigma^2.
]

For vector parameters, define a rank-(r) projection
[
C_t(g)=P_tP_t^\top g,\qquad P_t\in\mathbb R^{d\times r},\quad P_t^\top P_t=I_r.
]

For matrix parameters (W_t\in\mathbb R^{m\times n}), assume (m\le n) for simplicity:
[
R_t=P_t^\top G_t\in\mathbb R^{r\times n},\qquad
C_t(G_t)=P_tR_t=P_tP_t^\top G_t,
]
and
[
G_t^\perp=G_t-C_t(G_t).
]

### Model 1: pure projected gradient

[
x_{t+1}=x_t-\eta_t C_t(g_t).
]

Stored memory: (P_t), optionally low-rank momentum.
Low-rank object: update direction.
Full-rank object: parameters (x_t).
Information loss: ((I-P_tP_t^\top)g_t).
Compensation: none.
Convergence measure: projected stationarity (|P_t^\top\nabla f(x_t)|^2), unless projection quality gives
[
|P_tP_t^\top \nabla f(x_t)|^2\ge \delta|\nabla f(x_t)|^2.
]

### Model 2: classical EF compressed SGD

[
a_t=g_t+e_t,\qquad
u_t=C_t(a_t),
]
[
e_{t+1}=a_t-u_t,\qquad
x_{t+1}=x_t-\eta_t u_t.
]

Stored memory: residual (e_t\in\mathbb R^d), or gradient-buffer equivalent.
Low-rank object: (u_t).
Full-rank object: residual and parameters.
Information loss: compression error (a_t-C_t(a_t)).
Compensation: residual accumulation.
Convergence measure:
[
\frac1T\sum_{t=0}^{T-1}\mathbb E|\nabla f(x_t)|^2.
]

Critical caveat: fixed rank projection is **not** a contractive compressor on all vectors. If (C(g)=P P^\top g) and (g\perp\operatorname{range}(P)), then (C(g)=0), so EF accumulates error forever but never releases it. EF needs either randomized coverage, changing subspaces, or a contraction assumption.

### Model 3: compressed Adam / GaLore-like Adam

[
R_t=P_t^\top g_t,
]
[
m_t=\beta_1 m_{t-1}+(1-\beta_1)R_t,
]
[
v_t=\beta_2 v_{t-1}+(1-\beta_2)R_t^{\odot 2},
]
[
z_t=\frac{m_t}{\sqrt{v_t}+\epsilon},
\qquad
x_{t+1}=x_t-\eta_t P_tz_t.
]

Stored memory: (m_t,v_t\in\mathbb R^r), (P_t).
Low-rank object: optimizer states and update.
Full-rank object: parameters.
Information loss: discarded gradient and state-coordinate mismatch when (P_t) changes.
Compensation: none unless projection-aware state transport is added.
Convergence measure: full stationarity only under strong projection assumptions, randomized projections, large batches, or isotropic noise. GoLore’s analysis shows why SVD-based projection can fail under anisotropic noise. ([arXiv](https://arxiv.org/pdf/2410.11289))

### Model 4: Fira-style norm recovery

Let (R_t=P_t^\top G_t), (G_t^\perp=G_t-P_tR_t), and let
[
Z_t=\psi_t(R_t)
]
be the low-rank Adam-corrected update. Define matrix-level or columnwise scaling
[
\phi_t(R_t)=\frac{|Z_t|}{|R_t|+\epsilon_s}.
]

# Fira-style update: [ W_{t+1}

W_t-\eta_t P_tZ_t-\eta_t\phi_t(R_t)G_t^\perp.
]

With limiter:
[
S_t=\phi_t(R_t)G_t^\perp,
]
and if
[
\frac{|S_t|}{|S_{t-1}|}>\gamma,
]
replace
[
S_t\leftarrow \frac{S_t}{|S_t|}\gamma|S_{t-1}|.
]

Stored memory: (P_t), low-rank Adam states, previous recovery norm.
Low-rank object: optimizer states.
Full-rank object: parameter update, because (G_t^\perp) is used.
Information loss: no raw orthogonal gradient loss if (G_t^\perp) is used, but there is still **adaptive-state loss**: (G_t^\perp) has no true Adam first/second moments.
Compensation: magnitude recovery, not residual compensation.
Convergence measure: full gradient norm, but proof requires bounded positive scaling and moment-alignment assumptions.

### Model 5: LDAdam-style generalized EF

A simplified vector abstraction is:

[
A_t=g_t+\xi_t,
]
choose (U_t\in\mathbb R^{d\times r}), then
[
a_t=U_t^\top A_t.
]

Projection-aware first moment:
[
m_{t-\frac12}=U_t^\top U_{t-1}m_{t-1},
]
[
m_t=\beta_1m_{t-\frac12}+(1-\beta_1)a_t.
]

Adaptive update:
[
x_{t+1}=x_t-\eta_t U_t\frac{m_t}{\sqrt{\tilde v_t}+\epsilon}.
]

# Generalized EF buffer: [ \xi_{t+1}

(A_t-U_ta_t)
+
\frac{\beta_1}{1-\beta_1}
\bigl(U_{t-1}m_{t-1}-U_tm_{t-\frac12}\bigr),
]
with additional care for second-moment consistency. LDAdam’s paper explicitly identifies this as compensation for both gradient and optimizer-state compression.

Stored memory: low-rank states plus an EF buffer, possibly reused as a gradient buffer.
Low-rank object: update and states.
Full-rank object: parameters and residual buffer.
Information loss: gradient compression and state compression.
Compensation: generalized EF.
Convergence measure:
[
\frac1T\sum_{t=1}^T\mathbb E|\nabla f(x_t)|^2,
]
with LDAdam proving an (O(1/\sqrt T))-type nonconvex rate under bounded-gradient assumptions.

------

## 5. Three candidate theorem statements

### Theorem 1: Negative theorem — norm recovery alone is not EF

**Claim.** There exists a smooth strongly convex quadratic and a stale rank-1 subspace such that Fira-style norm recovery with no positive scaling floor fails to reduce the full gradient norm.

**Setting.**
[
f(x,y)=\frac12 y^2,\qquad \nabla f(x,y)=(0,y).
]
Let (P=e_1). Then (R_t=P^\top g_t=0). If the implementation sets (\phi_t(0)=0), then
[
u_t=P\psi_t(R_t)+\phi_t(R_t)(g_t-PR_t)=0,
]
so
[
y_{t+1}=y_t,\qquad |\nabla f(x_t,y_t)|^2=y_0^2.
]

**Interpretation.** This does not refute Fira with frequent SVD on clean gradients; SVD would choose (e_2). It refutes the idea that norm recovery is a general EF mechanism. It requires subspace coverage or positive lower-bounded recovery scaling.

**Hardest reviewer issue.** A reviewer may call this too trivial. We should use it as a lemma motivating stronger counterexamples, not the main negative result.

------

### Theorem 2: Positive theorem — bounded recovery Fira-SGD converges

Define a simplified Fira-SGD update:
[
u_t=P_tP_t^\top g_t+\lambda_t(I-P_tP_t^\top)g_t,
]
[
x_{t+1}=x_t-\eta u_t,
]
where
[
0<\lambda_{\min}\le \lambda_t\le \lambda_{\max}<\infty.
]

**Assumptions.**

1. (f) is (L)-smooth and lower bounded.
2. (\mathbb E[g_t\mid x_t]=\nabla f(x_t)).
3. (\mathbb E|g_t-\nabla f(x_t)|^2\le\sigma^2).
4. (\lambda_t) is measurable w.r.t. the current stochastic gradient or is clipped independently enough to preserve a descent inner product.
5. Step size (\eta\le c/L\lambda_{\max}^2).

# **Expected theorem.** [ \frac1T\sum_{t=0}^{T-1} \mathbb E|\nabla f(x_t)|^2 \le O!\left(\frac{f(x_0)-f_\star}{\eta T}\right) + O(\eta\sigma^2). ] Choosing (\eta=O(T^{-1/2})) gives [ \frac1T\sum_{t=0}^{T-1} \mathbb E|\nabla f(x_t)|^2

O(T^{-1/2}).
]

**Why this matters.** This theorem says that Fira-style recovery can be made convergent if the discarded component is not merely scaled by an uncontrolled norm ratio but by a bounded positive scalar.

**Hardest step.** Handling the dependence of (\lambda_t) on (g_t). If (\lambda_t) is computed from noisy low-rank Adam statistics, the descent inner product can be biased.

------

### Theorem 3: Positive theorem — residual-compensated projection-aware Fira-AMSGrad

Consider an algorithm that uses:

[
A_t=G_t+E_t,
]
[
R_t=P_t^\top A_t,\qquad Q_t=A_t-P_tR_t,
]
[
Z_t=\frac{M_t}{\sqrt{\widehat V_t}+\epsilon},
]
[
\Lambda_t=\operatorname{clip}*{[\lambda*{\min},\lambda_{\max}]}
\left(
\frac{|Z_t|}{|R_t|+\epsilon_s}
\right)Q_t,
]
[
W_{t+1}=W_t-\eta_t(P_tZ_t+\Lambda_t).
]

# If the limiter truncates (\Lambda_t), store the unused raw component in a residual: [ E_{t+1}

Q_t-\Lambda_t/\phi_t,
]
with projection-aware moment transport when (P_t) changes.

**Assumptions.**

1. (f) is (L)-smooth and lower bounded.
2. Stochastic gradients are unbiased and bounded or have bounded variance plus bounded second moment.
3. The subspace mechanism is contractive in expectation:
   [
   \mathbb E|(I-P_tP_t^\top)A_t|^2
   \le
   (1-\delta)|A_t|^2.
   ]
4. Adam denominator is AMSGrad-style monotone and bounded below by (\epsilon).
5. Scaling and limiter are bounded:
   [
   0<\lambda_{\min}\le \phi_t\le \lambda_{\max}.
   ]

**Expected theorem.**
[
\frac1T\sum_{t=0}^{T-1}
\mathbb E|\nabla f(W_t)|_F^2
\le
O(T^{-1/2})
+
\text{higher-order compression terms depending on } \delta,\beta_1,\beta_2,\epsilon.
]

**Hardest step.** The residual bound must handle both (E_t) and Adam’s adaptive denominator. This is exactly where LDAdam is relevant: optimizer-state compression error is not the same as gradient compression error. LDAdam’s convergence proof uses a contractive projection condition and AMSGrad-style normalization, so our proof should borrow the proof architecture but apply it to full-rank recovery updates.

------

## 6. Three candidate counterexamples

### Counterexample A: fixed bad subspace — real counterexample to naive norm recovery

Let
[
f(x,y)=\frac12 y^2,\qquad g_t=(0,y_t).
]
Let (P=e_1), so
[
R_t=P^\top g_t=0,\qquad g_t^\perp=g_t.
]

Fira-style update:
[
u_t=P\psi_t(R_t)+\phi_t(R_t)g_t^\perp.
]

If the safe implementation convention is (\phi_t(0)=0), then
[
u_t=0,\qquad y_{t+1}=y_t.
]

Thus
[
|\nabla f(x_t,y_t)|^2=y_0^2
]
for all (t). The failure is **direction loss plus undefined scaling**. It is a real counterexample for stale or biased subspaces, but not for an ideal SVD refresh that always sees the full clean gradient.

------

### Counterexample B: anisotropic noise — kills GaLore, but may not kill Fira

Let
[
f(x,y)=\frac12x^2,\qquad \nabla f(x,y)=(x,0),
]
and stochastic gradient
[
g_t=(x_t,\sigma \zeta_t),\qquad \zeta_t\in{-1,+1}.
]

Assume a rank-1 top-magnitude/SVD-like projection. If (\sigma>|x_t|), the chosen subspace is (P_t=e_2). Then GaLore-style projected SGD gives
[
C_t(g_t)=(0,\sigma \zeta_t),
]
so
[
x_{t+1}=x_t.
]
Therefore the true gradient component (x_t) is never reduced while the projection keeps chasing noise. This is the same mechanism as GoLore’s nonconvergence result: SVD can become dominated by anisotropic noise near stationarity. ([arXiv](https://arxiv.org/pdf/2410.11289))

But Fira-style recovery with (P_t=e_2) gives
[
g_t^\perp=(x_t,0),
]
and if (\phi_t>0),
[
x_{t+1}=x_t-\eta\phi_t x_t.
]
So this is **not automatically a Fira counterexample**. It is a useful failed attempt: it shows Fira’s full-rank recovery genuinely fixes one GaLore failure mode. To break Fira, we need attack (\phi_t), the limiter, stale subspace, or Adam-state distortion.

------

### Counterexample C: Adam denominator / state mismatch — real warning for Fira-Adam

Use
[
f(x,y)=\frac12(x^2+y^2),
\qquad P=e_1.
]

Take a simplified Adam correction with (\beta_1=\beta_2=0):
[
Z_t=\frac{x_t}{|x_t|+\epsilon}.
]
Then
[
\phi_t=\frac{|Z_t|}{|x_t|+\epsilon_s}
\approx
\frac{1}{|x_t|+\epsilon}
]
when (\epsilon_s) is small.

Fira-style update becomes
[
x_{t+1}=x_t-\eta\frac{x_t}{|x_t|+\epsilon},
]
[
y_{t+1}=y_t-\eta\frac{y_t}{|x_t|+\epsilon}.
]

If (x_t) becomes small, then the orthogonal update scale becomes huge. Whenever
[
\frac{\eta}{|x_t|+\epsilon}>2,
]
the (y)-coordinate update factor
[
1-\frac{\eta}{|x_t|+\epsilon}
]
has magnitude larger than one, so (y_t) can oscillate and grow. The failure is **adaptive denominator mismatch**: a small projected coordinate can induce an excessively large recovery scale on an orthogonal coordinate whose own Adam denominator was never estimated.

Fira’s norm-growth limiter is designed precisely to suppress such spikes, but then the limiter is discarding or delaying update mass. That makes explicit residual compensation natural: if the limiter clips (\Lambda_t), store the unused component and feed it back later.

A separate but related state-mismatch toy example is alternating projections. Let (P_0=e_1), (P_1=e_2), momentum (m_t=\beta m_{t-1}+(1-\beta)R_t), and start at ((1,0)). At (t=0),
[
R_0=1,\qquad m_0=1-\beta.
]
At (t=1), (R_1=0), but naive state reuse gives
[
m_1=\beta(1-\beta).
]
The method updates in the (e_2) direction even though the true (y)-gradient is zero. Projection-aware transport would multiply by (P_1^\top P_0=0) and remove this spurious momentum. This is exactly why SubTrack++ and LDAdam use projection-aware optimizer-state updates.

------

## 7. One recommended algorithmic modification

I recommend developing **Bounded Error-Compensated Fira-AMSGrad**, abbreviated here as **BEC-Fira**.

For each matrix parameter (W_t):

**Inputs.** Rank (r), projection (P_t), low-rank states (M_t,V_t), residual buffer (E_t), recovery scale bounds (\phi_{\min},\phi_{\max}), limiter (\gamma).

1. Form compensated gradient:
   [
   A_t=G_t+E_t.
   ]
2. Choose or update subspace (P_t). For theory, use a randomized or contractive projection. For practice, use SVD/Grassmannian tracking with occasional random refresh.
3. Project:
   [
   R_t=P_t^\top A_t,\qquad Q_t=A_t-P_tR_t.
   ]
4. Projection-aware Adam-state transfer:
   [
   M_{t-\frac12}=P_t^\top P_{t-1}M_{t-1}.
   ]
   Use the LDAdam/SubTrack-style second-moment transfer, or use AMSGrad reset as a simpler first theorem.
5. Low-rank Adam update:
   [
   M_t=\beta_1M_{t-\frac12}+(1-\beta_1)R_t,
   ]
   [
   V_t=\max\Bigl(V_{t-1},\ \beta_2V_{t-\frac12}+(1-\beta_2)R_t^{\odot2}\Bigr),
   ]
   [
   Z_t=\frac{M_t}{\sqrt{V_t}+\epsilon}.
   ]
6. Bounded recovery scaling:
   [
   \phi_t
   =
   \operatorname{clip}
   \left(
   \frac{|Z_t|}{|R_t|+\epsilon_s},
   \phi_{\min},
   \phi_{\max}
   \right),
   ]
   matrixwise or columnwise.
7. Candidate recovery:
   [
   \Lambda_t=\phi_t Q_t.
   ]
8. Norm-growth limiter:
   if
   [
   |\Lambda_t|>\gamma|\Lambda_{t-1}|,
   ]
   set
   [
   \bar\Lambda_t
   =
   \frac{\gamma|\Lambda_{t-1}|}{|\Lambda_t|}
   \Lambda_t.
   ]
   Otherwise (\bar\Lambda_t=\Lambda_t).
9. Full-rank update:
   [
   W_{t+1}=W_t-\eta_t(P_tZ_t+\bar\Lambda_t).
   ]
10. Residual feedback for clipped recovery:
    [
    E_{t+1}
    =
    Q_t-\bar\Lambda_t/\phi_t.
    ]

Optional LDAdam-like state residual:
[
E_{t+1}
\leftarrow
E_{t+1}
+
\frac{\beta_1}{1-\beta_1}
\left(P_{t-1}M_{t-1}-P_tM_{t-\frac12}\right).
]

Memory cost: low-rank Adam states (O(rn)), projection (O(mr)), plus a full-size residual buffer if persistent. To preserve optimizer-state memory, store (E_t) in the gradient accumulation buffer, as LDAdam does; but this may conflict with gradient clipping and per-layer update tricks. LDAdam explicitly notes this tradeoff.

Why this is not just Fira: it adds bounded scaling, projection-aware moment transport, and explicit residual feedback for clipped or lost recovery components.

Why this is not just LDAdam: LDAdam performs low-rank adaptive updates and explores the full space over time; BEC-Fira produces a **full-rank update at every step** through (\bar\Lambda_t). The residual is used to correct recovery/limiter/subspace errors in a Fira-style full-rank update, not only to compensate low-rank update compression. This distinction is important but fragile; if we add the full LDAdam state residual, reviewers may view it as LDAdam plus Fira recovery, so the paper must frame the novelty as theory of **full-rank recovery compensation**, not as a wholly new EF mechanism.

------

## 8. Four-week execution plan

### Week 1: pin down literature and reproduce mechanisms

Deliverables:

1. Write exact equations for Fira, SubTrack++, GoLore, and LDAdam in one notation.
2. Reproduce three toy algorithms: projected SGD/Adam, Fira-style recovery, LDAdam-style EF.
3. Implement the three counterexamples above.
4. Decide whether the main theorem targets SGD, AMSGrad, or AdamW.

Minimum success criterion: one clean plot showing projected methods fail on anisotropic noise, Fira fixes that case, and Fira can still fail or spike under denominator/stale-subspace mismatch.

### Week 2: prove the simplest theorem and sharpen counterexample

Deliverables:

1. Prove bounded Fira-SGD convergence.
2. Prove a negative theorem for unbounded or zero recovery scaling.
3. Decide whether the residual theorem is for clipped recovery only or for full generalized state compensation.
4. Write a 2-page internal note: assumptions, theorem, proof skeleton, counterexample.

Minimum success criterion: a reviewer-checkable theorem with no Adam complications.

### Week 3: adaptive extension and diagnostic experiments

Deliverables:

1. Extend proof to AMSGrad-style bounded denominator.
2. Implement BEC-Fira on 2D quadratics and small neural nets.
3. Metrics: full gradient norm, projected gradient norm, residual norm, angle between update and gradient, recovery scale (\phi_t), limiter activation rate, (v_t) mismatch.
4. Baselines: SGD, Adam, projected Adam, Fira-style, EF-SGD, LDAdam-style simplified, BEC-Fira.

Minimum success criterion: diagnostic experiments match theorem/counterexample predictions.

### Week 4: paper skeleton and positioning

Deliverables:

1. Draft paper claim and abstract.
2. Prepare theorem statement and proof appendix.
3. Prepare four figures:
   - taxonomy of errors,
   - counterexample curves,
   - recovery-scale instability,
   - small-model comparison.
4. Decide whether to include a small transformer experiment or keep the first submission as theory + diagnostics.

Minimum success criterion: a coherent paper skeleton titled around **“When Does Low-Rank Adam Need Error Compensation?”**

------

## 9. What to read first and what to skip temporarily

Read first:

1. **Fira Section 4 and Appendix A/D/E/G.** Focus on norm-based scaling, columnwise scaling, limiter, and spike analysis. The key equations are Fira’s decomposition (G_t=P_tR_t+(G_t-P_tR_t)), scaling (\phi_t), update rule, and limiter.
2. **LDAdam Sections 3.2–4.** Focus on projection-aware state transfer, generalized EF, the residual equation, and the nonconvex theorem.
3. **GoLore Sections 3–5.** Focus on the anisotropic-noise counterexample, the role of SVD bias, random Stiefel projection, and momentum projection. ([arXiv](https://arxiv.org/pdf/2410.11289))
4. **SubTrack++ Section 2 and Theorem 3.2.** Focus on Grassmannian tracking, projection-aware Adam, recovery scaling, and the limitation that the proof assumes constant projections. ([arXiv](https://arxiv.org/pdf/2502.01586))
5. Classical EF-SGD and AMSGrad convergence papers after the above, not before.

Skip temporarily:

- Large LLM benchmark tables.
- LoRA/ReLoRA background except for framing.
- Detailed system throughput comparisons.
- Fine-tuning task hyperparameter tables.
- Any attempt to prove original AdamW with weight decay before proving SGD/AMSGrad.

------

## 10. Concrete questions to answer before going deeper

1. Which exact Fira variant should we analyze: matrix-level scaling or columnwise scaling?
2. What is the implementation convention when (|R_t|\approx0): set (\phi_t=0), add (\epsilon_s), reuse previous (\phi_t), or clip to a positive floor?
3. Is a full-size residual buffer acceptable if it is stored in the gradient buffer, or must the method avoid any persistent full-rank residual?
4. Do we want the first theorem for SGD, momentum SGD, AMSGrad, or AdamW?
5. What projection regime should be treated as primary: fixed/stale projection, SVD every (T) steps, Grassmannian tracking, random projection, or hybrid SVD-to-random as in GoLore?
6. Are we trying to prove something about **original Fira**, or is the intended contribution a **modified Fira-like optimizer** with explicit compensation?
7. What experimental budget is realistic: only toy + MNIST/CIFAR, or also a small GPT/WikiText-2 run?
8. Should the paper aim for a theory venue style, where toy counterexamples are acceptable, or an ML systems venue style, where a small transformer experiment is necessary?

## Phase-1 verdict

The proposed counterexample is **mathematically correct** for the idealized Fira-style scalar recovery rule with

[
\psi(r)=\operatorname{sign}(r),\qquad
\phi_t=\frac{|\psi(R_t)|}{|R_t|},
]

fixed (P=e_1), zero Adam epsilon, no clipping, and constant learning rate. It gives a clean 2-cycle and the full gradient norm does not approach zero.

However, as a theorem-level negative result against **original Fira**, it is **not yet strong enough**. The main weakness is that (\psi(r)=\operatorname{sign}(r)) with constant step size is itself a nonconvergent 1D Adam-like update near the optimum. So a skeptical reviewer can say: “this is mostly a sign-Adam constant-stepsize failure, not a Fira-specific failure.”

The counterexample is still useful as a **warm-up negative theorem**. The stronger next theorem should use positive Adam epsilon and show that scalar recovery either becomes unstable or vanishes in the orthogonal direction because the orthogonal coordinate has no own adaptive state.

------

# 1. Rigorous verification

## 1.1 Fira-style abstraction

For

[
f(x,y)=\frac12(x^2+y^2),
\qquad
\nabla f(x,y)=(x,y),
]

take fixed rank-1 projection

[
P=e_1=
\begin{pmatrix}
1\
0
\end{pmatrix}.
]

Then

[
R_t=P^\top g_t=x_t,
]

and

# [ g_t^\perp

# g_t-PP^\top g_t

(0,y_t).
]

The simplified Fira-style update is

# [ u_t

P\psi(R_t)+\phi_t g_t^\perp,
]

so

# [ u_t

\left(\psi(x_t),\phi_t y_t\right).
]

If

[
\psi(r)=\operatorname{sign}(r),
\qquad
\phi_t=\frac{|\psi(x_t)|}{|x_t|},
]

and (x_t\neq 0), then

[
\phi_t=\frac1{|x_t|}.
]

Therefore

[
x_{t+1}=x_t-\eta\operatorname{sign}(x_t),
]

[
y_{t+1}=y_t-\eta\frac{y_t}{|x_t|}.
]

So the proposed update is correctly derived.

This abstraction matches the Fira idea that the gradient is divided into projected and outside-projection parts, with Adam applied only to the projected gradient and norm-based scaling applied to the outside-projection gradient. Fira’s paper explicitly states that GaLore discards the outside-projection term because it lacks optimizer states, while Fira rescales that term using low-rank adaptive scaling factors; it also supports matrix-level and column-level scaling. ([arXiv](https://arxiv.org/html/2410.01623v3))

------

## 1.2 Is (\psi(r)=\operatorname{sign}(r)) defensible?

Yes, as an **idealized Adam limit**.

For scalar Adam with

[
\beta_1=\beta_2=0,
\qquad
\epsilon_{\mathrm{adam}}=0,
]

we have

[
m_t=r_t,
\qquad
v_t=r_t^2,
]

so

# [ \psi(r_t)=\frac{m_t}{\sqrt{v_t}+\epsilon_{\mathrm{adam}}}

# \frac{r_t}{|r_t|}

\operatorname{sign}(r_t).
]

Thus (\psi(r)=\operatorname{sign}(r)) is a defensible mathematical simplification of Adam only in the singular limit (\epsilon_{\mathrm{adam}}=0), (\beta_1=\beta_2=0). It is **not** the exact Fira implementation: the released implementation uses Adam defaults with nonzero epsilon, and its norm-based scaling denominator also adds (10^{-8}). ([GitHub](https://github.com/xichen-fy/Fira/blob/main/optimizer_torch/fira_adamw.py))

------

## 1.3 Exact 2-cycle

Assume

[
x_0=\frac{\eta}{2},
\qquad
y_0\neq 0.
]

Then

# [ x_1

# \frac{\eta}{2}-\eta

-\frac{\eta}{2},
]

and

# [ x_2

# -\frac{\eta}{2}-\eta\operatorname{sign}\left(-\frac{\eta}{2}\right)

# -\frac{\eta}{2}+\eta

\frac{\eta}{2}.
]

Thus, by induction,

[
x_t=(-1)^t\frac{\eta}{2}.
]

For (y_t),

# [ y_{t+1}

y_t-\eta\frac{y_t}{|x_t|}.
]

Since

[
|x_t|=\frac{\eta}{2},
]

we get

# [ y_{t+1}

# y_t-\eta\frac{y_t}{\eta/2}

# y_t-2y_t

-y_t.
]

Therefore

[
y_t=(-1)^t y_0.
]

Hence the full iterate satisfies

[
(x_t,y_t)=(-1)^t\left(\frac{\eta}{2},y_0\right).
]

The objective value is constant:

# [ f(x_t,y_t)

\frac12\left(\frac{\eta^2}{4}+y_0^2\right),
]

and the full gradient norm is constant:

# [ |\nabla f(x_t,y_t)|

\sqrt{\frac{\eta^2}{4}+y_0^2}.
]

So the method does **not** converge to a first-order stationary point.

------

## 1.4 Does the example survive (\epsilon>0)?

There are two epsilons to distinguish.

### Adam epsilon in (\psi)

Let

[
\psi_\epsilon(r)=\frac{r}{|r|+\epsilon_{\mathrm{adam}}},
]

and for now take

# [ \phi_t=\frac{|\psi_\epsilon(x_t)|}{|x_t|}

\frac1{|x_t|+\epsilon_{\mathrm{adam}}}.
]

Then

# [ x_{t+1}

x_t-\eta\frac{x_t}{|x_t|+\epsilon_{\mathrm{adam}}},
]

# [ y_{t+1}

y_t-\eta\frac{y_t}{|x_t|+\epsilon_{\mathrm{adam}}}.
]

A symmetric 2-cycle (x_t=\pm c) satisfies

# [ c-\eta\frac{c}{c+\epsilon_{\mathrm{adam}}}

-c.
]

For (c>0), this gives

[
2c=\eta\frac{c}{c+\epsilon_{\mathrm{adam}}},
]

hence

[
c+\epsilon_{\mathrm{adam}}=\frac{\eta}{2},
]

so

[
c=\frac{\eta}{2}-\epsilon_{\mathrm{adam}}.
]

Therefore, if

[
\eta>2\epsilon_{\mathrm{adam}},
]

then choosing

[
x_0=\frac{\eta}{2}-\epsilon_{\mathrm{adam}}
]

gives an exact 2-cycle. Moreover,

# [ |x_t|+\epsilon_{\mathrm{adam}}

\frac{\eta}{2},
]

so

# [ y_{t+1}

# y_t-\eta\frac{y_t}{\eta/2}

-y_t.
]

Thus the counterexample survives positive Adam epsilon **provided**

[
\eta>2\epsilon_{\mathrm{adam}}.
]

If

[
\eta\le 2\epsilon_{\mathrm{adam}},
]

this exact symmetric 2-cycle disappears.

### Scaling denominator epsilon

The implementation-style scaling is closer to

# [ \phi_t

# \frac{|\psi_{\epsilon_a}(x_t)|}{|x_t|+\epsilon_s}

\frac{|x_t|}
{(|x_t|+\epsilon_a)(|x_t|+\epsilon_s)}.
]

If (x_t=\pm c) with

[
c=\frac{\eta}{2}-\epsilon_a,
]

then

# [ \phi_t

# \frac{c}{(c+\epsilon_a)(c+\epsilon_s)}

\frac{2c}{\eta(c+\epsilon_s)}.
]

Then

# [ y_{t+1}

# \left( 1-\eta\phi_t \right)y_t

# \left( 1-\frac{2c}{c+\epsilon_s} \right)y_t

\frac{\epsilon_s-c}{\epsilon_s+c}y_t.
]

If (\epsilon_s=0), this is exactly (-y_t).

If (\epsilon_s>0), then

[
\left|
\frac{\epsilon_s-c}{\epsilon_s+c}
\right|
<1,
]

so (y_t\to 0). But (x_t) still cycles between (\pm c), so the full gradient norm still does not converge to zero.

Conclusion:

- (\epsilon_{\mathrm{adam}}>0) does **not** kill the counterexample if (\eta>2\epsilon_{\mathrm{adam}}).
- (\epsilon_s>0) may fix the (y)-flip, but it does not fix the (x)-cycle.
- If both epsilons and a sufficiently small learning rate are used, this exact counterexample no longer works.

------

## 1.5 Does a norm-growth limiter fix it?

No, not the original **growth-ratio** type limiter.

In the idealized counterexample,

# [ \phi_t|y_t|

# \frac{|y_0|}{\eta/2}

\frac{2|y_0|}{\eta},
]

which is constant over time. Therefore the recovered outer-gradient norm does not grow. A limiter that only restricts

[
\frac{|\text{current recovered gradient}|}
{|\text{previous recovered gradient}|}
]

is inactive after the first step.

This matches the Fira limiter’s purpose: it is designed to prevent sudden norm growth, not to guarantee descent or remove constant bad scaling. The Fira paper describes the limiter as constraining the ratio of current gradient norm to previous gradient norm, and the released implementation computes a scaling-gradient norm ratio before rescaling only if the norm grows. ([arXiv](https://arxiv.org/html/2410.01623v3))

An **absolute clipping** rule can change the conclusion. If

# [ \bar\phi_t

\min\left{\frac1{|x_t|},s_{\max}\right},
]

then in the 2-cycle

[
\bar\phi_t=\min\left{\frac2\eta,s_{\max}\right}.
]

The (y)-recursion is

# [ y_{t+1}

(1-\eta\bar\phi_t)y_t.
]

Thus:

- if (0<\eta\bar\phi_t<2), then (y_t\to 0);
- if (\eta\bar\phi_t=2), then (y_t) flips forever;
- if (\eta\bar\phi_t>2), then (y_t) diverges.

But the (x)-cycle remains unless the low-rank Adam update itself is also modified.

------

## 1.6 Does column-wise scaling fix it?

No, not in the minimal vector-as-one-column construction.

Represent the parameter as a (2\times 1) matrix,

[
W=
\begin{pmatrix}
x\
y
\end{pmatrix}.
]

There is only one column. Matrix-level scaling and column-wise scaling coincide:

# [ \phi_t

# \frac{|\psi(R_t)|}{|R_t|}

\frac{|\psi(x_t)|}{|x_t|}.
]

So the same counterexample applies.

For a multi-column matrix, column-wise scaling may avoid some cross-column contamination, but it still does not create elementwise Adam denominators for discarded coordinates. Fira itself emphasizes that it uses matrix-level or column-level scaling, whereas full Adam uses parameter-level adaptive learning rates; Fira therefore preserves the raw gradient direction within a matrix or column and is not equivalent to full Adam. ([arXiv](https://arxiv.org/html/2410.01623v3))

------

## 1.7 Which original Fira assumptions or design choices does this violate?

The counterexample violates or idealizes several aspects of original Fira:

| Aspect              | Counterexample                         | Original Fira                                                |
| ------------------- | -------------------------------------- | ------------------------------------------------------------ |
| Projection          | Fixed (P=e_1)                          | Projection matrix is initialized by SVD and reused for a switching interval. ([arXiv](https://arxiv.org/html/2410.01623v3)) |
| Adam correction     | (\beta_1=\beta_2=0,\epsilon=0)         | Adam with nonzero (\epsilon), default implementation uses (\epsilon=10^{-6}). ([GitHub](https://github.com/xichen-fy/Fira/blob/main/optimizer_torch/fira_adamw.py)) |
| Scaling denominator | Exact division by (                    | R_t                                                          |
| Limiter             | None, or growth limiter inactive       | Fira includes norm-growth limiter. ([arXiv](https://arxiv.org/html/2410.01623v3)) |
| Data regime         | Deterministic 2D quadratic             | Empirical observation from LLM training that low-rank and full-rank scaling factors are similar. ([arXiv](https://arxiv.org/html/2410.01623v3)) |
| Learning rate       | Constant                               | LLM training uses warmup and decay schedules in experiments. |
| Claim being tested  | General convergence of scalar recovery | Fira is proposed empirically as plug-and-play memory-efficient training, not with a general nonconvex convergence theorem. |

So: the example is rigorous for an idealized Fira-style scalar recovery algorithm, but not a direct disproof of practical Fira.

------

# 2. Strengthening variants

## Variant A: anisotropic quadratic

Let

[
f(x,y)=\frac12(ax^2+by^2),
\qquad a,b>0.
]

Then

[
g_t=(a x_t,b y_t),
\qquad
R_t=a x_t,
\qquad
g_t^\perp=(0,b y_t).
]

With

[
\psi(r)=\operatorname{sign}(r),
\qquad
\phi_t=\frac1{|a x_t|},
]

the update is

# [ x_{t+1}

x_t-\eta\operatorname{sign}(x_t),
]

# [ y_{t+1}

y_t-\eta\frac{b y_t}{a|x_t|}.
]

If

[
x_0=\frac\eta2,
]

then

[
x_t=(-1)^t\frac\eta2.
]

The (y)-recursion becomes

# [ y_{t+1}

\left(1-\frac{2b}{a}\right)y_t.
]

Therefore:

| Regime  | (y_t) behavior   | Full method                        |
| ------- | ---------------- | ---------------------------------- |
| (0<b<a) | (y_t\to0) if (   | 1-2b/a                             |
| (b=a)   | (y_t=(-1)^t y_0) | Fails with constant gradient norm. |
| (b>a)   | (                | y_t                                |

Verdict: **provably fails** for all (a,b>0) because (x_t) cycles; the orthogonal coordinate failure is strongest when (b\ge a).

------

## Variant B: (\psi(r)=r/(|r|+\epsilon))

Let

[
\psi_\epsilon(r)=\frac{r}{|r|+\epsilon}.
]

With no extra scaling denominator epsilon,

# [ \phi_t=\frac{|\psi_\epsilon(a x_t)|}{|a x_t|}

\frac1{|a x_t|+\epsilon}.
]

The update is

# [ x_{t+1}

x_t-\eta\frac{a x_t}{a|x_t|+\epsilon},
]

# [ y_{t+1}

y_t-\eta\frac{b y_t}{a|x_t|+\epsilon}.
]

A symmetric (x)-cycle (x_t=\pm c) exists if

[
c=\frac{\eta}{2}-\frac{\epsilon}{a}>0,
]

equivalently

[
\eta a>2\epsilon.
]

At this cycle,

[
a c+\epsilon=\frac{a\eta}{2},
]

so

# [ y_{t+1}

\left(1-\frac{2b}{a}\right)y_t.
]

Verdict:

- If (\eta a>2\epsilon): **provably fails**, same as Variant A.
- If (\eta a\le2\epsilon): the exact cycle disappears; the method may converge on this quadratic if the induced (y)-multiplier is also stable.
- This variant is closer to Adam, but if failure relies on (\eta a>2\epsilon), a reviewer can still argue that the low-rank 1D Adam component is already unstable.

A stronger epsilon-based failure is the **vanishing-scale failure**:

Let

# [ \phi_t

# \frac{|\psi_\epsilon(a x_t)|}{|a x_t|+\epsilon_s}

\frac{|a x_t|}
{(|a x_t|+\epsilon)(|a x_t|+\epsilon_s)}.
]

Assume

[
0<\eta a<\epsilon,
]

so the (x)-coordinate converges monotonically to zero. Then (x_t) decays geometrically, and

[
\phi_t
\lesssim C |x_t|.
]

Hence

[
\sum_{t=0}^\infty \phi_t<\infty.
]

For small enough (b>0),

# [ y_t

y_0\prod_{k=0}^{t-1}(1-\eta b\phi_k)
]

converges to a nonzero value because the infinite product has finite total decay. Thus

[
x_t\to0,
\qquad
y_t\to y_\infty\neq0,
]

and

[
|\nabla f(x_t,y_t)|\to b|y_\infty|>0.
]

This is a better Fira-specific negative theorem: the projected coordinate becomes well optimized, but the orthogonal coordinate stops moving because the recovery scale vanishes.

------

## Variant C: clipped scale

Let

# [ \phi_t

\operatorname{clip}
\left(
\frac1{|x_t|+\epsilon_s},
s_{\min},s_{\max}
\right).
]

For the original sign-update cycle (x_t=\pm\eta/2),

# [ \phi_t

# \bar\phi

\operatorname{clip}
\left(
\frac1{\eta/2+\epsilon_s},
s_{\min},s_{\max}
\right).
]

Then

[
y_{t+1}=(1-\eta\bar\phi)y_t.
]

Therefore:

| Condition          | Behavior             |
| ------------------ | -------------------- |
| (0<\eta\bar\phi<2) | (y_t\to0).           |
| (\eta\bar\phi=2)   | (y_t) flips forever. |
| (\eta\bar\phi>2)   | (y_t) diverges.      |
| (\bar\phi=0)       | (y_t) freezes.       |

Verdict:

- With sign (\psi), the full method still **provably fails** because (x_t) cycles.
- With stable (\psi_\epsilon), positive lower clipping and safe upper clipping probably make this 2D quadratic converge under a standard step-size condition.
- Clipping is therefore a plausible fix for this exact pathology, but it does not by itself supply Error Feedback.

------

## Variant D: momentum Adam states

Consider

[
m_t=\beta_1m_{t-1}+(1-\beta_1)x_t,
]

[
v_t=\beta_2v_{t-1}+(1-\beta_2)x_t^2,
]

[
\psi_t=\frac{m_t}{\sqrt{v_t}+\epsilon}.
]

A 2-cycle can still be constructed, at least with appropriate initial optimizer state.

Assume (x_t=\pm c). In the steady 2-cycle,

# [ m^+

\frac{1-\beta_1}{1+\beta_1}c,
\qquad
m^-=-m^+,
]

and

[
v^+=v^-=c^2.
]

Let

[
\kappa=\frac{1-\beta_1}{1+\beta_1}.
]

Then

# [ \psi^+

\frac{\kappa c}{c+\epsilon}.
]

The condition

[
c-\eta\psi^+=-c
]

gives

# [ 2c

\eta\frac{\kappa c}{c+\epsilon},
]

so

# [ c

\frac{\eta\kappa}{2}-\epsilon.
]

Thus a nonzero 2-cycle exists if

[
\eta\kappa>2\epsilon.
]

Verdict:

- With (\epsilon=0), **provably fails** for any (\beta_1<1), using a suitable periodic optimizer-state initialization.
- With standard zero initialization, the exact proof is more involved, but the dynamics are empirically likely to approach a small oscillatory regime when the effective step is too large.
- With positive epsilon, decaying learning rate, and stable Adam conditions, this variant probably converges on the simple quadratic.

This version is more faithful to Adam but less clean as a first theorem.

------

## Variant E: stale subspace (P) fixed for (K) steps

If (K=\infty), this is exactly the fixed-(P=e_1) counterexample: **provably fails**.

If (K<\infty), the conclusion depends on how (P) is refreshed.

For the sign-cycle example, if the refresh rule always picks the largest gradient coordinate and

[
|y_0|<\frac{\eta}{2},
]

then throughout the cycle

[
|x_t|=\frac{\eta}{2}>|y_t|,
]

so the top rank-1 direction remains (e_1). Therefore the counterexample can survive even under frequent top-coordinate refresh.

If (|y_t|) eventually dominates (|x_t|), then an SVD-like refresh would switch to (e_2), and the simple fixed-(P) counterexample no longer applies.

Verdict:

- Fixed or biased stale subspace: **provably fails**.
- SVD refresh with (e_1) always dominant: **provably fails**.
- SVD refresh that eventually captures (e_2): probably converges or at least escapes this specific pathology.

------

# 3. Exact failure mechanism

This is **not merely pure projection failure**.

Pure projected gradient descent with (P=e_1) gives

[
x_{t+1}=x_t-\eta x_t,
\qquad
y_{t+1}=y_t.
]

The (y)-coordinate is never updated. That is pure direction loss.

Fira-style recovery does update (y):

[
y_{t+1}=y_t-\eta\phi_t y_t.
]

So the orthogonal gradient direction is present. The failure is more subtle.

The mechanisms are:

## 3.1 Adaptive denominator mismatch

The (y)-coordinate uses a scale derived from the projected coordinate (x):

[
\phi_t=\frac1{|x_t|}.
]

Thus the orthogonal update behaves as if (y) had Adam denominator (|x_t|), not (|y_t|). When (|x_t|=\eta/2),

[
\eta\phi_t=2,
]

which is exactly the boundary that causes

[
y_{t+1}=-y_t.
]

So the (y)-coordinate fails because it receives the wrong adaptive denominator.

## 3.2 Optimizer-state compression error

There is no (m_y) or (v_y). The orthogonal coordinate does not have its own first- or second-moment statistics. The scalar (\phi_t) is borrowed from the low-rank state.

This is exactly the distinction we care about:

- gradient information (y_t) is available;
- optimizer-state information for (y_t) is missing.

## 3.3 Norm mismatch

The scale matches a norm ratio in the projected subspace, but that scalar does not necessarily produce a stable step in the orthogonal subspace.

The failure is not that the method forgets (y_t). It remembers (y_t), but scales it incorrectly.

## 3.4 Residual non-accumulation

If the recovery term is clipped or limited, the unused part is discarded. There is no residual

[
e_{t+1}=e_t+g_t-C(g_t+e_t)
]

that would force future correction.

In the exact cycle without clipping, residual non-accumulation is not the main cause. But once a limiter is introduced, lack of residual becomes central.

## 3.5 Stale subspace error

The counterexample requires (P=e_1) to remain fixed or repeatedly selected. This is a stale or biased subspace assumption.

However, it is not a pure projection counterexample, because the outside-subspace gradient is explicitly used.

------

# 4. Minimal simulator

The following code is self-contained and designed for a Jupyter notebook.

It compares:

1. full gradient descent,
2. pure projected gradient descent,
3. Fira-style scalar recovery,
4. Fira-style scalar recovery with clipping,
5. Error Feedback SGD with fixed projection,
6. bounded residual-compensated recovery.

```python
import numpy as np
import matplotlib.pyplot as plt


def simulate(
    method,
    T=60,
    eta=0.1,
    x0=None,
    y0=0.03,
    eps_adam=0.0,
    eps_scale=0.0,
    s_min=0.0,
    s_max=np.inf,
    gamma=1.01,
):
    """
    Simulates methods on f(x,y)=0.5*(x^2+y^2).

    Projection is fixed P=e1.

    Methods:
      - "gd": full gradient descent
      - "pgd": projected gradient descent
      - "fira": Fira-style scalar recovery
      - "fira_clipped": Fira-style scalar recovery with scale clipping
      - "fira_limiter": Fira-style recovery with norm-growth limiter
      - "ef_sgd": classical EF-SGD with fixed projection
      - "bounded_residual": clipped recovery with residual on the orthogonal component
    """
    tiny = 1e-15

    if x0 is None:
        x0 = eta / 2

    z = np.array([x0, y0], dtype=float)
    residual = np.zeros(2, dtype=float)
    prev_outer_norm = None

    rows = []

    def f_value(z):
        return 0.5 * np.dot(z, z)

    def grad(z):
        return z.copy()

    def angle_deg(u, g):
        nu = np.linalg.norm(u)
        ng = np.linalg.norm(g)
        if nu < tiny or ng < tiny:
            return np.nan
        c = np.dot(u, g) / (nu * ng)
        c = np.clip(c, -1.0, 1.0)
        return np.degrees(np.arccos(c))

    def adam_like_psi(r):
        if eps_adam == 0.0:
            return np.sign(r) if abs(r) > tiny else 0.0
        return r / (abs(r) + eps_adam)

    def raw_fira_phi(r, psi):
        denom = abs(r) + eps_scale
        if denom <= tiny:
            return 0.0
        return abs(psi) / denom

    for t in range(T + 1):
        g = grad(z)

        # Default diagnostics before update
        phi = np.nan
        u = np.zeros_like(z)

        if method == "gd":
            u = g.copy()
            phi = np.nan

        elif method == "pgd":
            u = np.array([g[0], 0.0])
            phi = 0.0

        elif method in ["fira", "fira_clipped", "fira_limiter"]:
            r = g[0]
            psi = adam_like_psi(r)
            phi_raw = raw_fira_phi(r, psi)

            if method == "fira_clipped":
                phi = np.clip(phi_raw, s_min, s_max)
            else:
                phi = phi_raw

            outer = np.array([0.0, phi * g[1]])

            if method == "fira_limiter":
                outer_norm = np.linalg.norm(outer)

                if prev_outer_norm is not None and prev_outer_norm > tiny:
                    # Multiplicative growth limiter:
                    # ||outer_t|| <= gamma * ||outer_{t-1}||
                    if outer_norm > gamma * prev_outer_norm:
                        outer *= (gamma * prev_outer_norm) / outer_norm
                        outer_norm = np.linalg.norm(outer)

                prev_outer_norm = outer_norm

            u = np.array([psi, 0.0]) + outer

        elif method == "ef_sgd":
            # Classical Error Feedback with fixed compressor C(a)=P P^T a.
            # Since P=e1 forever, the y-residual is accumulated but never released.
            a_vec = g + residual
            u = np.array([a_vec[0], 0.0])
            residual = a_vec - u
            phi = 0.0

        elif method == "bounded_residual":
            # Residual-compensated clipped Fira recovery on the orthogonal coordinate.
            #
            # Raw target:      phi_raw * (g_y + residual_y)
            # Applied target:  phi      * (g_y + residual_y)
            #
            # The unapplied part is stored back in residual in raw-gradient units.
            r = g[0]
            psi = adam_like_psi(r)
            phi_raw = raw_fira_phi(r, psi)
            phi = np.clip(phi_raw, s_min, s_max)

            a_y = g[1] + residual[1]
            u = np.array([psi, phi * a_y])

            if phi_raw > tiny:
                residual[1] = (1.0 - phi / phi_raw) * a_y
            else:
                residual[1] = a_y

            residual[0] = 0.0

        else:
            raise ValueError(f"Unknown method: {method}")

        rows.append(
            {
                "t": t,
                "x": z[0],
                "y": z[1],
                "f": f_value(z),
                "grad_norm": np.linalg.norm(g),
                "phi": phi,
                "angle_deg": angle_deg(u, g),
                "residual_norm": np.linalg.norm(residual),
                "u_x": u[0],
                "u_y": u[1],
            }
        )

        if t == T:
            break

        z = z - eta * u

    return rows


def run_all(T=60, eta=0.1, y0=0.03):
    x0 = eta / 2

    configs = [
        (
            "Full GD",
            dict(method="gd"),
        ),
        (
            "Projected GD",
            dict(method="pgd"),
        ),
        (
            "Fira scalar",
            dict(method="fira", eps_adam=0.0, eps_scale=0.0),
        ),
        (
            "Fira clipped",
            dict(
                method="fira_clipped",
                eps_adam=0.0,
                eps_scale=0.0,
                s_min=0.0,
                s_max=0.9 / eta,
            ),
        ),
        (
            "EF-SGD fixed P",
            dict(method="ef_sgd"),
        ),
        (
            "Bounded residual recovery",
            dict(
                method="bounded_residual",
                eps_adam=0.0,
                eps_scale=0.0,
                s_min=0.0,
                s_max=0.9 / eta,
            ),
        ),
    ]

    histories = {}

    for name, cfg in configs:
        histories[name] = simulate(
            T=T,
            eta=eta,
            x0=x0,
            y0=y0,
            **cfg,
        )

    return histories


def plot_histories(histories):
    metrics = [
        ("f", "objective f"),
        ("grad_norm", "gradient norm"),
        ("x", "x_t"),
        ("y", "y_t"),
        ("phi", "phi_t"),
        ("angle_deg", "angle(update, grad)"),
        ("residual_norm", "residual norm"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 18), sharex=True)

    for ax, (key, title) in zip(axes, metrics):
        for name, rows in histories.items():
            ts = [r["t"] for r in rows]
            vals = [r[key] for r in rows]
            ax.plot(ts, vals, label=name)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("iteration")
    axes[0].legend(loc="best")
    plt.tight_layout()
    plt.show()


# Main diagnostic run
eta = 0.1
histories = run_all(T=60, eta=eta, y0=0.03)
plot_histories(histories)


# Optional: verify the exact Fira cycle numerically
rows = histories["Fira scalar"]
print("First six Fira scalar iterates:")
for r in rows[:6]:
    print(
        f"t={r['t']:2d}, "
        f"x={r['x']:+.6f}, "
        f"y={r['y']:+.6f}, "
        f"f={r['f']:.6f}, "
        f"grad_norm={r['grad_norm']:.6f}, "
        f"phi={r['phi']:.6f}"
    )
```

Expected qualitative output:

| Method                    | Expected behavior                                            |
| ------------------------- | ------------------------------------------------------------ |
| Full GD                   | (x_t,y_t\to0), objective decays.                             |
| Projected GD              | (x_t\to0), (y_t=y_0), full gradient norm does not vanish.    |
| Fira scalar               | (x_t,y_t) both flip sign forever; objective and gradient norm constant. |
| Fira clipped              | (y_t) can converge if (s_{\max}) is safe, but (x_t) still cycles. |
| EF-SGD fixed (P)          | (x_t\to0), (y_t) unchanged, residual norm grows because (e_2) is never selected. |
| Bounded residual recovery | Orthogonal component is stabilized; low-rank sign cycle remains unless (\psi) is also stabilized. |

------

# 5. Theorem potential

## 5.1 Can this become a formal negative theorem?

Yes, but only as a theorem about an **idealized scalar-recovery method**, not as a direct theorem saying “Fira fails.”

The cleanest theorem is:

### Theorem: 2-cycle under scalar norm recovery

Let

[
f(x,y)=\frac12(x^2+y^2).
]

Consider the update

[
x_{t+1}=x_t-\eta\operatorname{sign}(x_t),
]

[
y_{t+1}=y_t-\eta\frac{y_t}{|x_t|},
]

with (\eta>0), (x_0=\eta/2), and (y_0\neq0). Then

[
x_t=(-1)^t\frac{\eta}{2},
\qquad
y_t=(-1)^t y_0.
]

Consequently,

# [ |\nabla f(x_t,y_t)|

\sqrt{\frac{\eta^2}{4}+y_0^2}
]

for all (t), and the method does not converge to a first-order stationary point.

This theorem is correct, short, and checkable.

------

## 5.2 Required assumptions

The theorem requires:

1. fixed rank-1 projection (P=e_1);
2. scalar or one-column Fira-style recovery;
3. (\psi(r)=\operatorname{sign}(r));
4. (\phi_t=1/|x_t|);
5. no Adam epsilon or (\eta>2\epsilon_{\mathrm{adam}});
6. no absolute scale clipping;
7. no subspace refresh that switches to (e_2);
8. constant learning rate.

These are strong assumptions.

------

## 5.3 Likely reviewer objections

### Objection 1: “This is just sign-Adam failing.”

This is valid. The (x)-coordinate failure is already present in the low-rank sign update.

Preemption: present this theorem as a **minimal instability certificate**, not as the main Fira-specific theorem. Then follow it with a stronger theorem where the projected coordinate converges but the orthogonal coordinate does not.

------

### Objection 2: “Original Fira uses (\epsilon>0).”

Valid. With positive epsilon and small enough learning rate, this exact cycle may disappear.

Preemption: include the epsilon analysis above. Show that the cycle persists if (\eta>2\epsilon_{\mathrm{adam}}), and then give the stronger vanishing-scale theorem for (\epsilon>0).

------

### Objection 3: “Original Fira uses a norm-growth limiter.”

Valid but not fatal. The growth limiter is inactive here because the recovered norm is constant.

Preemption: explicitly prove

# [ |\phi_t g_t^\perp|

\frac{2|y_0|}{\eta}
]

for all (t). Therefore a growth-only limiter does nothing.

------

### Objection 4: “Original Fira refreshes (P) using SVD.”

Valid. Fixed (P=e_1) is a stale-subspace assumption.

Preemption: use a diagonal matrix version with (|x_t|>|y_t|), so rank-1 top-direction refresh keeps selecting the (x)-direction. For example, choose

[
|y_0|<\frac{\eta}{2}.
]

Then throughout the cycle,

[
|x_t|=\frac{\eta}{2}>|y_t|,
]

so a top-coordinate rank-1 rule keeps (P=e_1).

------

### Objection 5: “Column-wise scaling may avoid this.”

Not for a (2\times1) parameter matrix. In that case, column-wise and scalar scaling are identical.

Preemption: state the theorem for a one-column matrix.

------

## 5.4 Is this counterexample strong enough?

As a **formal negative theorem**, yes.

As the **main theoretical contribution**, no.

It is too artificial because the sign-Adam constant-step 2-cycle is already a known type of instability. It does not yet isolate the Fira-specific issue cleanly enough.

The better theorem is the epsilon-regularized vanishing-scale theorem.

------

# 6. Recommended next theorem

The next theorem should be:

## Vanishing-scale failure under positive epsilon

Let

[
f(x,y)=\frac12(ax^2+by^2),
\qquad
a,b>0,
]

with fixed (P=e_1). Define

[
\psi(r)=\frac{r}{|r|+\epsilon_a},
]

and

# [ \phi(r)

# \frac{|\psi(r)|}{|r|+\epsilon_s}

\frac{|r|}
{(|r|+\epsilon_a)(|r|+\epsilon_s)}.
]

The update is

# [ x_{t+1}

x_t-\eta\frac{a x_t}{a|x_t|+\epsilon_a},
]

# [ y_{t+1}

y_t-\eta b\phi(a x_t)y_t.
]

Assume

[
0<\eta a<\epsilon_a,
]

and choose (b>0) small enough that

[
0\le \eta b\phi(a x_t)\le \frac12
]

for all (t). Then:

1. (x_t\to0) geometrically;
2. (\sum_t \phi(a x_t)<\infty);
3. (y_t\to y_\infty\neq0);
4. therefore

[
|\nabla f(x_t,y_t)|\to b|y_\infty|>0.
]

This is more important than the sign-cycle theorem because the projected coordinate behaves well, but the orthogonal coordinate fails because its recovery scale vanishes. That isolates the core Fira-style issue:

> The outside-subspace gradient is present, but without its own optimizer state or residual, scalar norm recovery can apply either the wrong amount or eventually no effective correction at all.

This should be the next theorem to prove.

# Theorem: positive-(\epsilon) scalar recovery can leave a nonzero orthogonal gradient

Let

[
f(x,y)=\frac12(ax^2+by^2),\qquad a,b>0,
]

so

[
\nabla f(x,y)=(ax,by).
]

Use fixed rank-one projection (P=e_1). Then

[
R_t=ax_t,\qquad g_t^\perp=(0,by_t).
]

Define

[
\psi(r)=\frac{r}{|r|+\epsilon_a},\qquad \epsilon_a>0,
]

and

# [ \phi(r)=\frac{|\psi(r)|}{|r|+\epsilon_s}

\frac{|r|}{(|r|+\epsilon_a)(|r|+\epsilon_s)},\qquad \epsilon_s>0.
]

The Fira-style scalar recovery update is

# [ x_{t+1}

x_t-\eta\frac{a x_t}{a|x_t|+\epsilon_a},
]

# [ y_{t+1}

y_t-\eta b,\phi(ax_t)y_t.
]

Equivalently,

# [ y_{t+1}

\left(1-\eta b,\phi(ax_t)\right)y_t.
]

The main failure is that as the projected gradient (ax_t\to0), the recovery scale (\phi(ax_t)\to0). Hence the orthogonal coordinate receives only a **finite total amount of update**, even though its gradient direction (by_t) is explicitly present.

------

# 1. Clean theorem statement

Define

[
\alpha:=\epsilon_a,\qquad \beta:=\epsilon_s,\qquad \lambda:=\eta a.
]

Let

[
z_0:=|x_0|>0.
]

Define

[
M_0
:=
\max_{0\le r\le a z_0}
\frac{r}{(r+\alpha)(r+\beta)}.
]

A simple global bound is

[
M_0
\le
\frac{1}{(\sqrt{\alpha}+\sqrt{\beta})^2}.
]

The exact maximum over all (r\ge0) occurs at

[
r_\star=\sqrt{\alpha\beta},
]

with value

# [ \max_{r\ge0}\frac{r}{(r+\alpha)(r+\beta)}

\frac{1}{(\sqrt{\alpha}+\sqrt{\beta})^2}.
]

------

## Theorem 1: vanishing-scale failure

Assume:

[
a,b,\eta,\epsilon_a,\epsilon_s>0,
]

[
x_0\neq0,\qquad y_0\neq0,
]

[
0<\eta a<2\epsilon_a,
]

and

[
\eta b M_0\le \frac12.
]

Then the Fira-style scalar recovery iterates satisfy:

1. (|x_t|\to0) geometrically;
2. (\sum_{t=0}^\infty \phi(ax_t)<\infty);
3. (y_t\to y_\infty) for some (y_\infty\neq0);
4. therefore

[
|\nabla f(x_t,y_t)|
\not\to0.
]

More precisely,

# [ \lim_{t\to\infty}|\nabla f(x_t,y_t)|

b|y_\infty|>0.
]

A cleaner but slightly stronger sufficient condition replacing (\eta bM_0\le1/2) is

[
\eta b
\le
\frac{(\sqrt{\epsilon_a}+\sqrt{\epsilon_s})^2}{2}.
]

------

## Simpler monotone version

If one wants (x_t) to stay positive without sign changes, assume instead

[
x_0>0,
\qquad
0<\eta a\le \epsilon_a.
]

Then

[
x_t>0
]

for all (t), and the same conclusion holds.

The theorem above is slightly stronger because it allows controlled sign changes in (x_t), as long as

[
0<\eta a<2\epsilon_a.
]

The proof uses (|x_t|), so sign changes do not matter.

------

# 2. Proof

Let

[
z_t:=|x_t|.
]

The (x)-recurrence is

# [ x_{t+1}

x_t
\left(
1-\frac{\eta a}{a|x_t|+\epsilon_a}
\right).
]

Therefore

# [ z_{t+1}

z_t
\left|
1-\frac{\eta a}{a z_t+\epsilon_a}
\right|.
]

Using (\lambda=\eta a) and (\alpha=\epsilon_a),

# [ z_{t+1}

z_t
\left|
1-\frac{\lambda}{a z_t+\alpha}
\right|.
]

------

## Lemma 1: controlled (x_t) dynamics

Assume

[
0<\lambda<2\alpha.
]

Then there exists (q\in[0,1)), depending only on (a,\lambda,\alpha,z_0), such that

[
z_{t+1}\le q z_t
]

for all (t). Consequently,

[
|x_t|=z_t\le q^t z_0,
]

so (x_t\to0) geometrically.

### Proof

Define

[
h(z)
:=
\left|
1-\frac{\lambda}{az+\alpha}
\right|,
\qquad z\in[0,z_0].
]

Because

[
0<\lambda<2\alpha,
]

we have, for every (z\ge0),

[
0<\frac{\lambda}{az+\alpha}<2.
]

Hence

[
-1
<
1-\frac{\lambda}{az+\alpha}
<
1,
]

so

[
h(z)<1
]

for all (z\in[0,z_0]).

The function (h) is continuous on the compact interval ([0,z_0]). Therefore

[
q:=\max_{0\le z\le z_0}h(z)<1.
]

Since

[
z_{t+1}=z_t h(z_t),
]

and (z_t\le z_0) by induction, we get

[
z_{t+1}\le qz_t.
]

Thus

[
z_t\le q^t z_0.
]

This proves geometric convergence of (x_t) to zero. (\square)

------

## Lemma 1': positivity under a stronger step-size condition

If

[
x_0>0
]

and

[
0<\eta a\le \epsilon_a,
]

then

[
x_t>0
]

for all (t), and

[
x_t\to0
]

geometrically.

### Proof

For (x_t>0),

# [ x_{t+1}

x_t
\left(
1-\frac{\eta a}{ax_t+\epsilon_a}
\right).
]

Since

[
ax_t+\epsilon_a>\epsilon_a\ge \eta a,
]

we have

[
0
<
1-\frac{\eta a}{ax_t+\epsilon_a}
<
1.
]

Thus (x_{t+1}>0) and (x_{t+1}<x_t).

Moreover, because (x_t\le x_0),

[
\frac{\eta a}{ax_t+\epsilon_a}
\ge
\frac{\eta a}{ax_0+\epsilon_a}.
]

Therefore

[
x_{t+1}
\le
\left(
1-\frac{\eta a}{ax_0+\epsilon_a}
\right)x_t.
]

Let

[
q
:=
1-\frac{\eta a}{ax_0+\epsilon_a}.
]

Then (0<q<1), and

[
x_t\le q^t x_0.
]

So (x_t\to0) geometrically. (\square)

------

## Lemma 2: (\phi(ax_t)) is summable

Assume (\epsilon_a,\epsilon_s>0), and suppose

[
|x_t|\le q^t z_0
]

for some (q\in[0,1)). Then

[
\sum_{t=0}^\infty \phi(ax_t)<\infty.
]

### Proof

For every (r\in\mathbb R),

# [ \phi(r)

\frac{|r|}{(|r|+\epsilon_a)(|r|+\epsilon_s)}.
]

Using (\epsilon_a,\epsilon_s>0),

[
\phi(r)
\le
\frac{|r|}{\epsilon_a\epsilon_s}.
]

Therefore,

[
\phi(ax_t)
\le
\frac{a|x_t|}{\epsilon_a\epsilon_s}
\le
\frac{a z_0}{\epsilon_a\epsilon_s}q^t.
]

Hence

# [ \sum_{t=0}^\infty \phi(ax_t) \le \frac{a z_0}{\epsilon_a\epsilon_s} \sum_{t=0}^\infty q^t

\frac{a z_0}{\epsilon_a\epsilon_s(1-q)}
<\infty.
]

(\square)

------

## Lemma 3: infinite product lemma

Let ((c_t)_{t\ge0}) satisfy

[
0\le c_t\le \frac12
]

and

[
\sum_{t=0}^\infty c_t<\infty.
]

Then

[
\prod_{t=0}^\infty (1-c_t)
]

converges to a strictly positive limit.

### Proof

For (0\le c\le1/2),

[
-\log(1-c)\le 2c.
]

Therefore,

[
0
\le
-\sum_{t=0}^\infty \log(1-c_t)
\le
2\sum_{t=0}^\infty c_t
<\infty.
]

Thus

[
\sum_{t=0}^\infty \log(1-c_t)
]

converges to a finite real number. Hence

# [ \prod_{t=0}^\infty(1-c_t)

\exp\left(
\sum_{t=0}^\infty\log(1-c_t)
\right)
]

converges to a strictly positive number. (\square)

------

## Lemma 4: (y_t) converges to a nonzero limit

Define

[
c_t:=\eta b,\phi(ax_t).
]

Under the theorem assumptions,

[
0\le c_t\le \frac12.
]

Also,

# [ \sum_{t=0}^\infty c_t

\eta b\sum_{t=0}^\infty\phi(ax_t)
<\infty.
]

Therefore,

[
\prod_{t=0}^\infty(1-c_t)
]

converges to a strictly positive limit (L>0).

Since

[
y_{t+1}=(1-c_t)y_t,
]

we have

# [ y_t

y_0\prod_{k=0}^{t-1}(1-c_k).
]

Taking (t\to\infty),

[
y_t\to y_\infty:=y_0 L.
]

Because

[
y_0\neq0
]

and

[
L>0,
]

we get

[
y_\infty\neq0.
]

(\square)

------

## Final conclusion

By Lemma 1,

[
x_t\to0.
]

By Lemma 4,

[
y_t\to y_\infty\neq0.
]

Therefore,

# [ \nabla f(x_t,y_t)

(ax_t,by_t)
\to
(0,by_\infty).
]

Since (b>0) and (y_\infty\neq0),

# [ \lim_{t\to\infty}|\nabla f(x_t,y_t)|

b|y_\infty|

> 

1. 

]

Thus the Fira-style scalar recovery method does **not** converge to a first-order stationary point, even on a deterministic two-dimensional strongly convex quadratic. (\square)

------

# 3. Why this is genuinely stronger than the sign-Adam 2-cycle

The earlier counterexample used

[
\psi(r)=\operatorname{sign}(r),
]

which produced

[
x_{t+1}=x_t-\eta\operatorname{sign}(x_t).
]

That already causes a constant-step 2-cycle in the projected coordinate. A reviewer could correctly object that the failure is mostly caused by sign-Adam with constant learning rate.

The present theorem avoids that objection.

## Does the projected coordinate (x_t) converge?

Yes.

Under

[
0<\eta a<2\epsilon_a,
]

we proved

[
|x_t|\le q^t|x_0|
]

for some (q<1). Therefore

[
x_t\to0.
]

So the projected low-rank coordinate is successfully optimized.

## Does the failure come from low-rank sign-Adam oscillation?

No.

There is no sign-Adam oscillation in this theorem. The low-rank update is

[
\psi(r)=\frac{r}{|r|+\epsilon_a},
]

with (\epsilon_a>0). For a stable step size, the projected coordinate converges geometrically to zero.

## Does the orthogonal gradient direction appear in the update?

Yes.

The (y)-update is

# [ y_{t+1}

y_t-\eta b,\phi(ax_t)y_t.
]

The true orthogonal gradient component (by_t) appears explicitly. This is not pure projection failure.

Pure projected gradient descent would give

[
y_{t+1}=y_t.
]

Here, the method does update (y_t), but the total amount of update is finite.

## Why does the orthogonal coordinate still fail?

Because the effective learning-rate multiplier for the (y)-coordinate is

[
\eta b,\phi(ax_t).
]

As (x_t\to0),

# [ \phi(ax_t)

\frac{a|x_t|}
{(a|x_t|+\epsilon_a)(a|x_t|+\epsilon_s)}
\sim
\frac{a|x_t|}{\epsilon_a\epsilon_s}.
]

Since (|x_t|) decays geometrically,

[
\sum_{t=0}^\infty \phi(ax_t)<\infty.
]

Thus

[
\sum_{t=0}^\infty \eta b,\phi(ax_t)<\infty.
]

The (y)-coordinate receives only finite total shrinkage, so it converges to a nonzero value.

## What is the precise failure mechanism?

The failure is:

[
\boxed{
\text{vanishing recovery scale caused by missing orthogonal adaptive state.}
}
]

The orthogonal coordinate (y) does not have its own Adam denominator. Its update scale is borrowed from the projected coordinate (x). Once (x) is optimized, the scalar recovery factor goes to zero, even though the orthogonal gradient (by_t) remains nonzero.

This is an **adaptive denominator mismatch** and an **optimizer-state compression error**, not merely direction loss.

------

# 4. Robustness analysis

## A. (\epsilon_s=0)

Then

# [ \phi(r)

# \frac{|r|} {(|r|+\epsilon_a)|r|}

\frac{1}{|r|+\epsilon_a}
]

for (r\neq0).

As (x_t\to0),

[
\phi(ax_t)\to \frac1{\epsilon_a}.
]

Therefore,

[
\sum_t \phi(ax_t)=\infty.
]

The vanishing-scale theorem does **not** hold.

The (y)-recursion becomes asymptotically

[
y_{t+1}
\approx
\left(
1-\frac{\eta b}{\epsilon_a}
\right)y_t.
]

So:

- if (0<\eta b/\epsilon_a<2), then (y_t\to0);
- if (\eta b/\epsilon_a=2), then (y_t) asymptotically oscillates;
- if (\eta b/\epsilon_a>2), then (y_t) diverges.

Thus positive (\epsilon_s) is essential for this particular vanishing-scale failure.

------

## B. (\epsilon_s>0)

This is the theorem setting.

As (x_t\to0),

[
\phi(ax_t)
\sim
\frac{a|x_t|}{\epsilon_a\epsilon_s}
\to0.
]

Because (x_t) decays geometrically, (\phi(ax_t)) is summable, and the orthogonal coordinate receives finite total update.

The theorem holds.

------

## C. (\eta a\ge \epsilon_a)

There are three regimes.

### Case 1: (\epsilon_a\le \eta a<2\epsilon_a)

The monotone positivity proof no longer holds, because (x_t) may change sign.

However, the stronger theorem still holds because

[
|x_t|\to0
]

geometrically whenever

[
0<\eta a<2\epsilon_a.
]

So this regime is still covered by Theorem 1.

### Case 2: (\eta a=2\epsilon_a)

Near zero,

[
\left|
1-\frac{\eta a}{a|x_t|+\epsilon_a}
\right|
\to 1.
]

The geometric contraction proof fails. In fact, the decay can become only sublinear, and

[
\sum_t \phi(ax_t)
]

may diverge. The theorem needs modification and may fail.

### Case 3: (\eta a>2\epsilon_a)

The projected coordinate (x_t) is locally unstable near zero. The method may oscillate or diverge. This becomes a different failure mode, closer to scalar Adam instability, and is not the clean vanishing-scale failure.

------

## D. (b) not sufficiently small

The small-(b) condition

[
\eta bM_0\le\frac12
]

is used only to ensure

[
0\le 1-\eta b\phi(ax_t)\le1
]

for every (t). This gives a clean monotone infinite-product proof.

It is not necessary for failure in a generic sense.

Since

[
\sum_t \eta b\phi(ax_t)<\infty
]

for every fixed (b>0), the product

[
\prod_t(1-\eta b\phi(ax_t))
]

still converges to a finite nonzero limit unless one factor is exactly zero, i.e. unless

[
\eta b\phi(ax_t)=1
]

for some finite (t).

For a fixed trajectory (x_t), the exceptional values of (b) are contained in the countable set

[
\left{
\frac{1}{\eta \phi(ax_t)}
:
\phi(ax_t)>0
\right}_{t\ge0}.
]

Thus, for generic (b), the nonzero-limit conclusion still holds. The small-(b) assumption is mainly a clean sufficient condition that avoids sign flips and exact finite-time cancellation.

------

## E. Lower-clipped scale

Let

# [ \bar\phi(r)

\max{\phi(r),s_{\min}},
\qquad s_{\min}>0.
]

Then eventually, since (\phi(ax_t)\to0),

[
\bar\phi(ax_t)=s_{\min}.
]

The (y)-update becomes asymptotically

# [ y_{t+1}

(1-\eta b s_{\min})y_t.
]

If

[
0<\eta b s_{\min}<2,
]

then

[
y_t\to0.
]

So a positive lower bound on the recovery scale repairs this particular vanishing-scale failure, provided the upper scale is also stable.

A clean repair condition is

[
0<s_{\min}\le \bar\phi(r)\le s_{\max}<\frac{2}{\eta b}.
]

Then

[
|1-\eta b\bar\phi(r)|\le \rho<1
]

for some (\rho<1), and (y_t\to0).

------

## F. Decaying learning rate (\eta_t)

The theorem needs modification, but the failure can persist.

Suppose

# [ x_{t+1}

x_t-\eta_t\frac{a x_t}{a|x_t|+\epsilon_a},
]

and assume for simplicity that (x_t>0) and

[
0<\eta_t a\le \epsilon_a.
]

Then

# [ x_t-x_{t+1}

\eta_t\frac{a x_t}{a x_t+\epsilon_a}.
]

Also,

# [ \eta_t\phi(ax_t)

# \eta_t \frac{a x_t} {(a x_t+\epsilon_a)(a x_t+\epsilon_s)}

\frac{x_t-x_{t+1}}{a x_t+\epsilon_s}
\le
\frac{x_t-x_{t+1}}{\epsilon_s}.
]

Therefore,

[
\sum_{t=0}^\infty \eta_t\phi(ax_t)
\le
\frac{x_0}{\epsilon_s}
<\infty.
]

So even with decaying learning rates, the total orthogonal update can remain finite.

If additionally

[
\sum_t \eta_t=\infty,
]

then (x_t\to0), while (y_t) can still converge to a nonzero value under the same small-(b) product condition.

Thus the failure is not purely a constant-learning-rate artifact. It extends to many decaying learning-rate schedules, but the theorem statement must be rewritten with (\eta_t).

------

## G. Periodic subspace refresh by largest gradient coordinate

The theorem as stated uses fixed (P=e_1).

If the method periodically refreshes (P) using the largest current gradient coordinate, then this exact asymptotic theorem usually fails.

Why?

The theorem implies

[
ax_t\to0,
\qquad
by_t\to by_\infty\neq0.
]

Therefore, eventually,

[
|by_t|>|ax_t|.
]

A largest-gradient-coordinate refresh would eventually switch from (P=e_1) to (P=e_2). Once (e_2) is selected, the (y)-coordinate becomes the projected coordinate and can be optimized.

So:

- fixed or stale (P=e_1): theorem holds;
- infinitely frequent exact largest-gradient refresh: theorem does not directly hold;
- long refresh interval (K): theorem gives a long-horizon failure mechanism but not an infinite-time nonconvergence theorem;
- biased or sticky subspace tracking: theorem may still apply, but this requires a separate assumption.

This is an important limitation.

------

# 5. Reviewer objections and responses

## Objection 1: fixed (P=e_1) is unrealistic

Status: partially valid.

The theorem does not claim that every refreshed-subspace Fira implementation fails. It proves that scalar norm recovery alone does not guarantee convergence without a subspace-coverage condition.

Preemption:

State the result as:

> Even when the discarded gradient direction is explicitly included, scalar norm recovery can fail under a fixed or stale subspace because its scale vanishes with the projected gradient.

Do not overclaim against projection-refresh variants.

------

## Objection 2: (b) small makes the (y)-coordinate too flat

Status: partially valid but not fatal.

The small-(b) condition is used to keep

[
0\le \eta b\phi(ax_t)\le1/2.
]

It is a proof convenience, not the core reason for failure.

For generic (b), the product still converges to a nonzero value unless

[
\eta b\phi(ax_t)=1
]

for some finite (t). The small-(b) assumption simply avoids finite-time cancellation and sign flips.

Also, first-order stationarity requires

[
by_t\to0.
]

Even if (b) is small, a nonzero limit (by_\infty\neq0) is a genuine stationarity failure. If one wants a visible gradient lower bound, choose (y_0) large enough.

------

## Objection 3: Fira refreshes projection by SVD

Status: valid for direct claims about practical Fira.

An exact full-gradient SVD refresh would eventually notice the nonzero (y)-gradient because (ax_t\to0) while (by_t\not\to0).

Preemption:

Do not claim this theorem refutes fully refreshed Fira. Claim instead that it refutes the sufficiency of scalar norm recovery under stale or biased subspaces.

A separate theorem is needed for periodic refresh or SVD-based tracking.

------

## Objection 4: Fira uses momentum Adam, not (\beta_1=\beta_2=0)

Status: partially valid.

This theorem uses the simplest Adam-like correction

[
\psi(r)=\frac{r}{|r|+\epsilon_a}.
]

That corresponds to a zero-momentum, zero-history adaptive normalization.

However, the failure mechanism is not specific to (\beta_1=\beta_2=0). The key property is:

[
\psi_t(R_t)\to0
\quad\text{as}\quad
R_t\to0.
]

With positive Adam epsilon and stable dynamics, low-rank Adam moments also decay when the projected gradient decays. Then the recovery scale can again vanish.

Preemption:

State the theorem for the clean memoryless case first. Then present momentum Adam as an extension, not as part of the initial theorem.

------

## Objection 5: Fira has a norm-growth limiter

Status: not fatal.

This theorem is a **vanishing-scale** failure, not an exploding-scale failure.

The recovery factor satisfies

[
\phi(ax_t)\to0.
]

A norm-growth limiter can only cap large recovery steps. It does not create a positive lower bound on recovery scale.

If the limiter reduces the applied recovery term further, then the (y)-coordinate receives even less total update. So a growth-only limiter cannot repair this theorem’s mechanism.

------

## Objection 6: this is a toy convex quadratic, not LLM training

Status: easy to address.

In optimization theory, failure on a deterministic strongly convex quadratic is a strong negative signal. It shows that the mechanism lacks a general convergence guarantee.

Preemption:

Frame the theorem carefully:

> This is not a claim that Fira fails in LLM training. It is a minimal counterexample showing that scalar norm recovery is not, by itself, an error-compensation mechanism with general convergence guarantees.

------

# 6. Cleanest next positive repair theorem

The mathematically cleanest repair is a **lower-bounded and upper-bounded recovery scale**.

Residual compensation is more conceptually aligned with Error Feedback, but it is technically more delicate here because the recovered term is preconditioned by (\phi_t). One has to define whether the residual is stored in raw-gradient units or preconditioned-update units. For this specific failure, lower-bounded clipping is the cleanest first positive theorem.

------

## Candidate positive theorem: clipped recovery fixes the quadratic failure

Consider the same quadratic and fixed (P=e_1). Use

# [ \bar\phi(r)

\operatorname{clip}(\phi(r),s_{\min},s_{\max})
]

with

[
0<s_{\min}\le s_{\max}<\frac{2}{\eta b}.
]

Assume

[
0<\eta a<2\epsilon_a.
]

The update is

# [ x_{t+1}

x_t-\eta\frac{a x_t}{a|x_t|+\epsilon_a},
]

# [ y_{t+1}

y_t-\eta b,\bar\phi(ax_t)y_t.
]

Then:

1. (x_t\to0) geometrically;
2. (y_t\to0) geometrically;
3. therefore

[
|\nabla f(x_t,y_t)|\to0.
]

The key reason is that

[
\bar\phi(ax_t)\ge s_{\min}>0,
]

so

[
\sum_{t=0}^\infty \eta b,\bar\phi(ax_t)=\infty.
]

Unlike the original (\phi(ax_t)), the clipped scale cannot vanish when the projected gradient vanishes.

This is the cleanest positive repair theorem to prove next.

# 1. Strict line-by-line review of the theorem

The theorem is mathematically sound under the stated assumptions. The strongest version is:

[
0<\eta a<2\epsilon_a,\qquad \epsilon_s>0,\qquad x_0\neq0,\qquad y_0\neq0,
]

and

[
\eta b M_0\le \frac12,
]

where

# [ M_0

\max_{0\le r\le a|x_0|}
\frac{r}{(r+\epsilon_a)(r+\epsilon_s)}.
]

Under these assumptions,

[
x_t\to0
]

geometrically, but

[
y_t\to y_\infty\neq0,
]

so

[
|\nabla f(x_t,y_t)|\not\to0.
]

## Review point 1: Is (0<\eta a<2\epsilon_a) sufficient for geometric convergence of (|x_t|)?

Yes.

Let

[
z_t:=|x_t|,\qquad \lambda:=\eta a,\qquad \alpha:=\epsilon_a.
]

The (x)-recursion is

# [ x_{t+1}

x_t\left(1-\frac{\lambda}{az_t+\alpha}\right),
]

hence

# [ z_{t+1}

z_t
\left|
1-\frac{\lambda}{az_t+\alpha}
\right|.
]

If

[
0<\lambda<2\alpha,
]

then for every (z\ge0),

[
0<\frac{\lambda}{az+\alpha}
\le
\frac{\lambda}{\alpha}
<2.
]

Therefore

[
\left|
1-\frac{\lambda}{az+\alpha}
\right|<1.
]

On the finite interval ([0,z_0]), this gives a uniform contraction factor (q<1). So (|x_t|) decays geometrically.

This is correct.

------

## Review point 2: Is there a hidden problem in

[
q=\max_{0\le z\le z_0}
\left|
1-\frac{\eta a}{az+\epsilon_a}
\right|<1?
]

No, provided one explicitly proves that (z_t\in[0,z_0]) for all (t).

The function

[
h(z)=\left|1-\frac{\eta a}{az+\epsilon_a}\right|
]

is continuous on the compact interval ([0,z_0]). Since (h(z)<1) pointwise on this interval, its maximum satisfies

[
q:=\max_{0\le z\le z_0}h(z)<1.
]

The subtle point is that (h(z)\to1) as (z\to\infty), so compactness matters. One cannot take (q) over ([0,\infty)). But over the invariant interval ([0,z_0]), the argument is valid.

------

## Review point 3: Does (z_t) always remain in ([0,z_0])?

Yes.

Assume (z_t\in[0,z_0]). Then

[
z_{t+1}=z_t h(z_t)\le qz_t<z_t\le z_0.
]

Thus by induction,

[
0\le z_t\le z_0
]

for all (t).

This closes the potential circularity in Lemma 1.

------

## Review point 4: Does summability depend on (\epsilon_s>0)?

Yes. This is essential.

With (\epsilon_s>0),

# [ \phi(r)

\frac{|r|}
{(|r|+\epsilon_a)(|r|+\epsilon_s)}
\le
\frac{|r|}{\epsilon_a\epsilon_s}.
]

Since (|x_t|) decays geometrically,

[
\sum_t \phi(ax_t)<\infty.
]

If (\epsilon_s=0), then for (r\neq0),

# [ \phi(r)

# \frac{|r|}{(|r|+\epsilon_a)|r|}

\frac{1}{|r|+\epsilon_a}.
]

As (r\to0),

[
\phi(r)\to\frac1{\epsilon_a},
]

so (\phi(ax_t)) does **not** go to zero. Therefore

[
\sum_t \phi(ax_t)=\infty.
]

So the vanishing-scale theorem fundamentally uses the positive denominator epsilon (\epsilon_s>0).

------

## Review point 5: Is the infinite-product lemma rigorous?

Yes, if stated as follows.

If

[
0\le c_t\le \frac12
]

and

[
\sum_{t=0}^\infty c_t<\infty,
]

then

[
\prod_{t=0}^\infty(1-c_t)
]

converges to a strictly positive number.

The proof uses

[
-\log(1-c)\le 2c,\qquad 0\le c\le\frac12.
]

Then

[
\sum_t -\log(1-c_t)<\infty,
]

so

[
\sum_t \log(1-c_t)
]

converges to a finite real number. Therefore the infinite product is positive.

This is fully rigorous.

------

## Review point 6: Is (\eta bM_0\le1/2) sufficient to prevent finite-time hitting (y_t=0)?

Yes.

Define

[
c_t:=\eta b\phi(ax_t).
]

Since

[
ax_t\in[-a|x_0|,a|x_0|],
]

we have

[
\phi(ax_t)\le M_0.
]

Thus

[
0\le c_t\le \eta bM_0\le\frac12.
]

The (y)-update is

[
y_{t+1}=(1-c_t)y_t.
]

Finite-time hitting (y_{t+1}=0) would require

[
1-c_t=0,
]

i.e.

[
c_t=1.
]

But (c_t\le1/2), so this cannot happen.

------

## Review point 7: Is the conclusion

[
|\nabla f(x_t,y_t)|\not\to0
]

completely true?

Yes.

The proof gives

[
x_t\to0
]

and

[
y_t\to y_\infty\neq0.
]

Therefore

[
\nabla f(x_t,y_t)=(ax_t,by_t)\to(0,by_\infty).
]

Since

[
b>0,\qquad y_\infty\neq0,
]

we get

# [ \lim_{t\to\infty}|\nabla f(x_t,y_t)|

b|y_\infty|>0.
]

So the full gradient norm does not vanish.

------

## Review point 8: Weaker or more elegant assumptions

The condition

[
\eta bM_0\le\frac12
]

can be weakened.

It is enough to assume that for some (\rho<1),

[
\eta bM_0\le \rho.
]

Then

[
0\le c_t\le \rho<1.
]

The product lemma still works because

[
-\log(1-c)\le \frac{c}{1-\rho},
\qquad 0\le c\le\rho<1.
]

The (\frac12) condition is cleaner and reviewer-friendly, but not minimal.

A cleaner but more conservative global condition is

[
\eta b
\le
\frac{(\sqrt{\epsilon_a}+\sqrt{\epsilon_s})^2}{2},
]

because

[
\phi(r)
\le
\frac{1}{(\sqrt{\epsilon_a}+\sqrt{\epsilon_s})^2}
]

for all (r\ge0).

------

## Review point 9: Possible mathematical objections

The theorem itself has no fatal mathematical flaw.

The main limitations are interpretive:

1. It assumes fixed (P=e_1).
2. It uses memoryless Adam, i.e. (\beta_1=\beta_2=0).
3. It depends critically on (\epsilon_s>0).
4. It assumes a scalar recovery scale.
5. It does not analyze SVD refresh or subspace tracking.
6. It proves failure of a Fira-style abstraction, not necessarily practical Fira.

These are not proof flaws, but they must be acknowledged in the theorem statement and surrounding text.

------

# 2. Appendix-ready theorem-proof version

## Theorem: positive-epsilon scalar recovery can leave a nonzero orthogonal gradient

Let

[
f(x,y)=\frac12(ax^2+by^2),
\qquad a,b>0.
]

Consider the deterministic update

# [ x_{t+1}

x_t-\eta\frac{a x_t}{a|x_t|+\epsilon_a},
]

# [ y_{t+1}

y_t-\eta b,\phi(ax_t)y_t,
]

where

[
\eta>0,\qquad \epsilon_a>0,\qquad \epsilon_s>0,
]

and

# [ \phi(r)

\frac{|r|}
{(|r|+\epsilon_a)(|r|+\epsilon_s)}.
]

Let

[
z_0:=|x_0|>0,
]

and define

[
M_0
:=
\max_{0\le r\le az_0}
\frac{r}{(r+\epsilon_a)(r+\epsilon_s)}.
]

Assume

[
y_0\neq0,
]

[
0<\eta a<2\epsilon_a,
]

and

[
\eta bM_0\le\frac12.
]

Then

[
x_t\to0
]

geometrically, but

[
y_t\to y_\infty
]

for some

[
y_\infty\neq0.
]

Consequently,

# [ \lim_{t\to\infty} |\nabla f(x_t,y_t)|

b|y_\infty|>0.
]

Therefore the method does not converge to a first-order stationary point.

------

## Proof

Let

[
z_t:=|x_t|,
\qquad
\lambda:=\eta a,
\qquad
\alpha:=\epsilon_a.
]

The (x)-update can be written as

# [ x_{t+1}

x_t
\left(
1-\frac{\lambda}{az_t+\alpha}
\right).
]

Therefore

# [ z_{t+1}

z_t
\left|
1-\frac{\lambda}{az_t+\alpha}
\right|.
]

Define

[
h(z):=
\left|
1-\frac{\lambda}{az+\alpha}
\right|.
]

Since

[
0<\lambda<2\alpha,
]

we have for every (z\ge0),

[
0<\frac{\lambda}{az+\alpha}
\le
\frac{\lambda}{\alpha}
<2.
]

Thus

[
h(z)<1
]

for all (z\ge0).

Now restrict to the compact interval ([0,z_0]). Since (h) is continuous,

[
q:=
\max_{0\le z\le z_0}h(z)
]

exists. Since (h(z)<1) pointwise on ([0,z_0]),

[
q<1.
]

We prove by induction that

[
z_t\le q^tz_0.
]

At (t=0), this is true. If (z_t\le q^tz_0), then in particular (z_t\le z_0), and hence

# [ z_{t+1}

z_th(z_t)
\le
qz_t
\le
q^{t+1}z_0.
]

Therefore

[
z_t\le q^tz_0.
]

Thus

[
x_t\to0
]

geometrically.

Next, because (\epsilon_a,\epsilon_s>0),

# [ \phi(r)

\frac{|r|}
{(|r|+\epsilon_a)(|r|+\epsilon_s)}
\le
\frac{|r|}{\epsilon_a\epsilon_s}.
]

Therefore

[
\phi(ax_t)
\le
\frac{a|x_t|}{\epsilon_a\epsilon_s}
\le
\frac{az_0}{\epsilon_a\epsilon_s}q^t.
]

Hence

[
\sum_{t=0}^{\infty}\phi(ax_t)<\infty.
]

Define

[
c_t:=\eta b\phi(ax_t).
]

By the definition of (M_0) and the fact that (|x_t|\le z_0),

[
\phi(ax_t)\le M_0.
]

Thus

[
0\le c_t\le \eta bM_0\le\frac12.
]

Also,

# [ \sum_{t=0}^{\infty}c_t

\eta b
\sum_{t=0}^{\infty}\phi(ax_t)
<\infty.
]

The (y)-recursion is

[
y_{t+1}=(1-c_t)y_t.
]

Therefore

# [ y_t

y_0\prod_{k=0}^{t-1}(1-c_k).
]

Since

[
0\le c_k\le\frac12
]

and

[
\sum_k c_k<\infty,
]

the infinite product

[
\prod_{k=0}^{\infty}(1-c_k)
]

converges to a strictly positive limit. Indeed, for (0\le c\le1/2),

[
-\log(1-c)\le2c,
]

so

[
\sum_k-\log(1-c_k)<\infty.
]

Hence

[
\sum_k\log(1-c_k)
]

conges to a finite real number, and the product is strictly positive.

Let

[
L:=
\prod_{k=0}^{\infty}(1-c_k)>0.
]

Then

[
y_t\to y_\infty:=y_0L.
]

Since

[
y_0\neq0
]

and

[
L>0,
]

we have

[
y_\infty\neq0.
]

Finally,

[
\nabla f(x_t,y_t)=(ax_t,by_t)\to(0,by_\infty).
]

Thus

# [ \lim_{t\to\infty}|\nabla f(x_t,y_t)|

b|y_\infty|>0.
]

This proves the theorem. (\square)

------

# 3. Distance between this theorem and real Fira/Adam

## 3.1 Is this a special case of Adam with (\beta_1=\beta_2=0)?

Yes.

Scalar Adam without bias correction is

[
m_t=\beta_1m_{t-1}+(1-\beta_1)r_t,
]

[
v_t=\beta_2v_{t-1}+(1-\beta_2)r_t^2,
]

[
\psi_t=\frac{m_t}{\sqrt{v_t}+\epsilon_a}.
]

If

[
\beta_1=\beta_2=0,
]

then

[
m_t=r_t,
\qquad
v_t=r_t^2,
]

so

# [ \psi_t

\frac{r_t}{|r_t|+\epsilon_a}.
]

Therefore the theorem is exactly a memoryless Adam special case.

It is not full Adam, but it is a legitimate Adam-type abstraction.

------

## 3.2 If (\beta_1,\beta_2>0) and (ax_t\to0), does (\psi_t\to0)?

Yes, assuming

[
0\le\beta_1,\beta_2<1
]

and

[
\epsilon_a>0.
]

Let

[
r_t=ax_t.
]

If

[
r_t\to0,
]

then the exponentially weighted average

[
m_t=\beta_1m_{t-1}+(1-\beta_1)r_t
]

also satisfies

[
m_t\to0.
]

Likewise,

[
v_t=\beta_2v_{t-1}+(1-\beta_2)r_t^2
]

satisfies

[
v_t\to0.
]

Since

[
\sqrt{v_t}+\epsilon_a\ge\epsilon_a>0,
]

we have

# [ |\psi_t|

\left|
\frac{m_t}{\sqrt{v_t}+\epsilon_a}
\right|
\le
\frac{|m_t|}{\epsilon_a}
\to0.
]

So positive Adam epsilon makes the Adam output vanish when the projected gradient and first moment vanish.

------

## 3.3 If (\psi_t\to0), can the Fira-style recovery scale go to zero?

Yes.

The scalar recovery scale is

# [ \phi_t

\frac{|\psi_t|}{|ax_t|+\epsilon_s}.
]

If

[
x_t\to0
]

and

[
\epsilon_s>0,
]

then

[
|ax_t|+\epsilon_s\to\epsilon_s.
]

Therefore, if

[
\psi_t\to0,
]

then

[
\phi_t\to0.
]

For the nonconvergence theorem, mere convergence (\phi_t\to0) is not enough. We need

[
\sum_t \phi_t<\infty.
]

That follows if (\psi_t) is summable, for example if (x_t) decays geometrically.

------

## 3.4 Momentum-Adam version: conditional proposition

Consider

[
r_t=ax_t,
]

[
m_t=\beta_1m_{t-1}+(1-\beta_1)r_t,
]

[
v_t=\beta_2v_{t-1}+(1-\beta_2)r_t^2,
]

[
\psi_t=\frac{m_t}{\sqrt{v_t}+\epsilon_a},
]

[
\phi_t=\frac{|\psi_t|}{|r_t|+\epsilon_s}.
]

The update is

[
x_{t+1}=x_t-\eta\psi_t,
]

[
y_{t+1}=y_t-\eta b\phi_t y_t.
]

A full proof for this coupled Adam system is more difficult, because (x_t,m_t,v_t) form a nonlinear dynamical system. But the following conditional proposition is clean.

------

## Proposition: Adam-state version under summable low-rank output

Assume

[
0\le\beta_1,\beta_2<1,
\qquad
\epsilon_a,\epsilon_s>0,
]

and suppose the projected Adam subsystem satisfies

[
x_t\to0
]

and

[
\sum_{t=0}^\infty |\psi_t|<\infty.
]

Assume also

[
\eta b\sup_t\phi_t\le\frac12,
]

and

[
y_0\neq0.
]

Then

[
y_t\to y_\infty\neq0,
]

and hence

[
|\nabla f(x_t,y_t)|\not\to0.
]

### Proof

Since

[
\epsilon_s>0,
]

we have

# [ \phi_t

\frac{|\psi_t|}{|r_t|+\epsilon_s}
\le
\frac{|\psi_t|}{\epsilon_s}.
]

Therefore

[
\sum_t\phi_t
\le
\frac1{\epsilon_s}
\sum_t|\psi_t|
<\infty.
]

Let

[
c_t:=\eta b\phi_t.
]

Then

[
0\le c_t\le\frac12
]

and

[
\sum_t c_t<\infty.
]

The same infinite-product argument gives

# [ y_t

y_0\prod_{k=0}^{t-1}(1-c_k)
\to
y_\infty\neq0.
]

Since (x_t\to0),

[
\nabla f(x_t,y_t)\to(0,by_\infty),
]

so the gradient norm does not vanish. (\square)

------

## When is (\sum_t|\psi_t|<\infty) reasonable?

If

[
|x_t|\le Cq^t
]

for some

[
0<q<1,
]

then

[
|r_t|\le aCq^t,
]

so

[
\sum_t|r_t|<\infty.
]

With zero or bounded initial momentum,

# [ m_t

\beta_1^{t+1}m_{-1}
+
(1-\beta_1)
\sum_{k=0}^{t}
\beta_1^{t-k}r_k.
]

Thus

[
\sum_t|m_t|
\le
\frac{|m_{-1}|}{1-\beta_1}
+
\sum_t|r_t|
<\infty.
]

Since

[
|\psi_t|\le\frac{|m_t|}{\epsilon_a},
]

we get

[
\sum_t|\psi_t|<\infty.
]

So the Adam-state version also fails whenever the projected Adam subsystem converges fast enough.

The hard part is proving that the projected Adam subsystem itself has geometric convergence under broad (\beta_1,\beta_2). Locally, the linearized first-moment system is stable under a step-size condition similar to

[
0<\frac{\eta a}{\epsilon_a}
<
\frac{2(1+\beta_1)}{1-\beta_1},
]

but turning this into a global theorem with (v_t), sign changes, and bias correction requires a separate Lyapunov argument.

------

## 3.5 How should the theorem be presented?

It should be presented as:

> **A counterexample for Fira-style scalar norm recovery under a stale rank-one subspace.**

Not as:

> “Fira fails.”

A reviewer will object to the second phrasing.

The precise defensible claim is:

> Even when the orthogonal gradient direction is explicitly included in the update, scalar norm recovery can fail to guarantee stationarity because the recovery scale is tied to the projected Adam state. Once the projected coordinate is optimized, the recovery scale can vanish, leaving a nonzero orthogonal gradient.

That is the theorem’s real contribution.

------

# 4. Repair directions: clipped recovery vs residual compensation

## A. Lower-bounded clipped recovery

# [ \bar\phi(r)

\operatorname{clip}(\phi(r),s_{\min},s_{\max}),
]

with

[
0<s_{\min}\le \bar\phi(r)\le s_{\max}<\frac{2}{\eta b}.
]

## B. Residual / Error Feedback recovery

Let the orthogonal gradient be

[
g_t^\perp=(0,by_t).
]

A simple residual form is

[
h_t=g_t^\perp+e_t,
]

[
u_t^\perp=\bar\phi_t h_t,
]

[
e_{t+1}=h_t-u_t^\perp.
]

This is meaningful when (\bar\phi_t) is treated as a scalar compressor/transmission factor and is bounded away from zero.

------

## Comparison

| Criterion                              | Lower-bounded clipped recovery                               | Residual-compensated recovery                                |
| -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Easier positive theorem                | Yes. Very easy on the quadratic.                             | Harder. EF needs a nonvanishing transmission/contraction condition. |
| Aligns with “Error Compensation” theme | Weakly. It is stabilization, not compensation.               | Strongly. This is the project’s conceptual direction.        |
| Closer to original Fira                | Closer. Fira already uses recovery scaling and a growth limiter. | Less close. Explicit residuals are not original Fira.        |
| Risk of overlap with LDAdam / EF       | Low. But also less novel.                                    | Higher. Must distinguish from LDAdam by focusing on full-rank recovery and stale-subspace recovery. |
| Contribution potential                 | Good as theory baseline or necessary condition.              | Better as main algorithmic contribution.                     |
| Suitable as main algorithm             | Not alone. Too simple.                                       | Yes, if combined with bounded scaling and projection-aware states. |
| Suitable as first positive theorem     | Yes.                                                         | Not first; prove after clipped theorem.                      |

The important technical point is:

> Residual compensation alone does not necessarily fix vanishing recovery if the applied scale (\phi_t) is summable.

If

[
\phi_t\to0
]

too fast, the residual may accumulate but never be transmitted. Classical Error Feedback needs a compressor with a contraction property. In this toy scalar setting, that means the applied scale should not vanish.

So the clean repair is not “residual only.” It is:

[
\boxed{
\text{bounded recovery scale} + \text{residual compensation}.
}
]

------

## Unified repair form

Use

# [ \bar\phi_t

\operatorname{clip}(\phi_t,s_{\min},s_{\max}),
]

with

[
0<s_{\min}\le s_{\max}<1
]

for the residual-compressor interpretation.

Then define

[
h_t=g_t^\perp+e_t,
]

[
u_t^\perp=\bar\phi_t h_t,
]

[
e_{t+1}=h_t-u_t^\perp.
]

The full update is

[
x_{t+1}=x_t-\eta\psi_t,
]

[
y_{t+1}=y_t-\eta u_{t,y}^\perp.
]

This simultaneously includes clipping and residual compensation.

This unified form avoids the disjointed story:

1. Clipping ensures the orthogonal channel does not vanish.
2. Residual compensation ensures under-applied orthogonal information is not discarded.

------

# 5. Recommended approach

## First positive theorem

Prove the clipped-recovery repair on the same quadratic.

Assume

[
0<\eta a<2\epsilon_a,
]

and

[
0<s_{\min}\le \bar\phi(r)\le s_{\max}<\frac{2}{\eta b}.
]

Then

[
x_t\to0
]

geometrically, and

[
y_t\to0
]

geometrically, because

[
y_{t+1}=(1-\eta b\bar\phi_t)y_t
]

with

[
|1-\eta b\bar\phi_t|\le \rho<1.
]

This is the cleanest first positive theorem.

## First algorithm name

Use a modest descriptive name first:

[
\boxed{\text{Bounded Error-Compensated Recovery, or BECR.}}
]

In the toy setting:

[
\bar\phi_t=\operatorname{clip}(\phi_t,s_{\min},s_{\max}),
]

[
h_t=g_t^\perp+e_t,
]

[
u_t^\perp=\bar\phi_t h_t,
]

[
e_{t+1}=h_t-u_t^\perp.
]

Later, if extended to Adam states, call it something like **BECR-Adam** or **BEC-Fira-Adam**.

## Minimum experiment

The minimum experiment should verify exactly this:

1. Pure projected method optimizes (x) but leaves (y) unchanged.
2. Positive-epsilon scalar recovery optimizes (x), but (\phi_t\to0), (\sum_t\phi_t<\infty), and (y_t\to y_\infty\neq0).
3. Lower-clipped recovery keeps the orthogonal scale nonzero and drives (y_t\to0).
4. Residual-compensated bounded recovery also drives (y_t\to0), and shows how residuals can be safely incorporated once the scale is lower-bounded.

------

# 6. Minimal Jupyter notebook code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Parameters satisfying theorem
# -----------------------------
a = 1.0
b = 1.0
eta = 0.5
epsilon_a = 1.0
epsilon_s = 1.0

x0 = 1.0
y0 = 1.0

T = 80

# Repair parameters
s_min = 0.20
s_max = 0.50


def psi(r):
    """Memoryless Adam-like correction."""
    return r / (abs(r) + epsilon_a)


def phi_raw(r):
    """Fira-style positive-epsilon scalar recovery scale."""
    return abs(r) / ((abs(r) + epsilon_a) * (abs(r) + epsilon_s))


def objective(x, y):
    return 0.5 * (a * x**2 + b * y**2)


def grad_norm(x, y):
    return np.sqrt((a * x)**2 + (b * y)**2)


def compute_M0(a, x0, epsilon_a, epsilon_s):
    """
    Exact maximum of r / ((r+epsilon_a)(r+epsilon_s))
    over r in [0, a |x0|].
    """
    R = a * abs(x0)

    if R == 0:
        return 0.0

    r_star = np.sqrt(epsilon_a * epsilon_s)

    def phi_of_r(r):
        return r / ((r + epsilon_a) * (r + epsilon_s))

    if R <= r_star:
        return phi_of_r(R)
    else:
        return phi_of_r(r_star)


M0 = compute_M0(a, x0, epsilon_a, epsilon_s)

cond_positive = (a > 0 and b > 0 and eta > 0 and epsilon_a > 0 and epsilon_s > 0)
cond_x = (0 < eta * a < 2 * epsilon_a)
cond_y = (eta * b * M0 <= 0.5)

print("Theorem condition check")
print("-----------------------")
print(f"a,b,eta,epsilon_a,epsilon_s > 0: {cond_positive}")
print(f"0 < eta*a < 2*epsilon_a:        {cond_x}")
print(f"eta*b*M0 <= 1/2:                {cond_y}")
print()
print(f"eta*a       = {eta*a:.6f}")
print(f"2*epsilon_a = {2*epsilon_a:.6f}")
print(f"M0          = {M0:.6f}")
print(f"eta*b*M0    = {eta*b*M0:.6f}")


def simulate(method):
    """
    Methods:
      - projected: pure projected method, no orthogonal update
      - fira: positive-epsilon scalar recovery
      - clipped: lower/upper clipped scalar recovery
      - residual: bounded residual-compensated recovery

    For all methods, the x-coordinate uses the same low-rank Adam-like correction.
    """
    x = float(x0)
    y = float(y0)

    # Orthogonal residual, stored in raw-gradient units
    e_y = 0.0

    rows = []
    cumulative_phi_used = 0.0

    for t in range(T + 1):
        r = a * x
        raw_phi = phi_raw(r)

        if method == "projected":
            phi_used = 0.0
        elif method == "fira":
            phi_used = raw_phi
        elif method in ["clipped", "residual"]:
            phi_used = np.clip(raw_phi, s_min, s_max)
        else:
            raise ValueError(f"Unknown method: {method}")

        rows.append(
            {
                "t": t,
                "method": method,
                "x": x,
                "y": y,
                "phi_raw": raw_phi,
                "phi_used": phi_used,
                "sum_phi_used": cumulative_phi_used,
                "grad_norm": grad_norm(x, y),
                "objective": objective(x, y),
                "residual_norm": abs(e_y),
            }
        )

        if t == T:
            break

        # Low-rank projected Adam-like correction
        r = a * x
        psi_t = psi(r)

        # Same x-update for all four methods
        x_next = x - eta * psi_t

        # Orthogonal gradient component
        gy = b * y

        if method == "projected":
            # No orthogonal update
            u_y = 0.0
            applied_phi = 0.0

        elif method == "fira":
            # Fira-style scalar recovery
            applied_phi = raw_phi
            u_y = applied_phi * gy

        elif method == "clipped":
            # Lower-bounded clipped recovery
            applied_phi = np.clip(raw_phi, s_min, s_max)
            u_y = applied_phi * gy

        elif method == "residual":
            # Bounded residual-compensated recovery.
            #
            # Important: residual compensation alone with raw_phi may not repair the
            # theorem's failure, because raw_phi is summable. Here we combine residual
            # feedback with a lower-bounded transmission scale.
            applied_phi = np.clip(raw_phi, s_min, s_max)

            accumulated_gradient = gy + e_y
            u_y = applied_phi * accumulated_gradient
            e_y = accumulated_gradient - u_y

        y_next = y - eta * u_y

        cumulative_phi_used += applied_phi

        x, y = x_next, y_next

    return pd.DataFrame(rows)


histories = {
    "Projected": simulate("projected"),
    "Fira positive-epsilon": simulate("fira"),
    "Lower-clipped": simulate("clipped"),
    "Residual-compensated": simulate("residual"),
}

df = pd.concat(histories.values(), ignore_index=True)


# -----------------------------
# Plots
# -----------------------------
plot_specs = [
    ("x", r"$x_t$"),
    ("y", r"$y_t$"),
    ("phi_used", r"applied recovery scale $\phi_t$"),
    ("sum_phi_used", r"cumulative applied scale $\sum_{k<t}\phi_k$"),
    ("grad_norm", r"$\|\nabla f(x_t,y_t)\|$"),
    ("objective", r"$f(x_t,y_t)$"),
]

fig, axes = plt.subplots(3, 2, figsize=(12, 12))
axes = axes.ravel()

for ax, (metric, title) in zip(axes, plot_specs):
    for name, hist in histories.items():
        ax.plot(hist["t"], hist[metric], label=name)
    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.grid(True, alpha=0.3)

axes[0].legend()
plt.tight_layout()
plt.show()


# -----------------------------
# Final summary table
# -----------------------------
summary_rows = []

for name, hist in histories.items():
    last = hist.iloc[-1]
    summary_rows.append(
        {
            "method": name,
            "final_x": last["x"],
            "final_y": last["y"],
            "final_grad_norm": last["grad_norm"],
            "sum_phi": last["sum_phi_used"],
            "final_objective": last["objective"],
            "final_residual_norm": last["residual_norm"],
        }
    )

summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))
```

Expected qualitative outcome:

| Method                | Expected behavior                                            |
| --------------------- | ------------------------------------------------------------ |
| Projected             | (x_t\to0), (y_t=y_0), gradient norm nonzero.                 |
| Fira positive-epsilon | (x_t\to0), (\phi_t\to0), (\sum_t\phi_t<\infty), (y_t\to y_\infty\neq0). |
| Lower-clipped         | (x_t\to0), applied scale stays (\ge s_{\min}), (y_t\to0).    |
| Residual-compensated  | With bounded scale, (y_t\to0), residual remains controlled.  |

The key diagnostic plot is the cumulative applied scale. For Fira-style recovery it plateaus; for lower-clipped and residual-compensated recovery it grows, which is exactly why those methods can keep reducing (y_t).Positive repair theorem: lower-bounded clipped recovery fixes the vanishing-scale failure

## 1. Formal theorem statement

Let

[
f(x,y)=\frac12(ax^2+by^2),\qquad a,b>0.
]

Consider the fixed rank-one projection (P=e_1), so the projected gradient is

[
R_t=ax_t.
]

Let

[
\psi(r)=\frac{r}{|r|+\epsilon_a},
\qquad \epsilon_a>0,
]

and let the original scalar recovery scale be

[
\phi(r)=\frac{|r|}{(|r|+\epsilon_a)(|r|+\epsilon_s)},
\qquad \epsilon_s>0.
]

Define the clipped scale

# [ \bar\phi(r)

# \operatorname{clip}(\phi(r),s_{\min},s_{\max})

\min{\max{\phi(r),s_{\min}},s_{\max}}.
]

Assume

[
0<s_{\min}\le s_{\max}<\frac{2}{\eta b}
]

and

[
0<\eta a<2\epsilon_a.
]

Consider the update

# [ x_{t+1}

x_t-\eta\frac{a x_t}{a|x_t|+\epsilon_a},
]

# [ y_{t+1}

y_t-\eta b,\bar\phi(ax_t)y_t.
]

Then for every initial point ((x_0,y_0)\in\mathbb R^2),

[
x_t\to0
]

geometrically,

[
y_t\to0
]

geometrically, and therefore

[
|\nabla f(x_t,y_t)|\to0.
]

------

# 2. Proof that (x_t\to0) geometrically

Define

[
z_t:=|x_t|,
\qquad
\lambda:=\eta a,
\qquad
\alpha:=\epsilon_a.
]

The (x)-update is

# [ x_{t+1}

x_t\left(
1-\frac{\lambda}{az_t+\alpha}
\right).
]

Hence

# [ z_{t+1}

z_t
\left|
1-\frac{\lambda}{az_t+\alpha}
\right|.
]

Assume

[
0<\lambda<2\alpha.
]

Then for every (z\ge0),

[
0<\frac{\lambda}{az+\alpha}\le \frac{\lambda}{\alpha}<2.
]

Therefore

[
\left|
1-\frac{\lambda}{az+\alpha}
\right|<1.
]

If (z_0=0), then (x_t=0) for all (t), so there is nothing to prove.

Now assume (z_0>0). Define

[
h(z)
:=
\left|
1-\frac{\lambda}{az+\alpha}
\right|.
]

The function (h) is continuous on the compact interval ([0,z_0]), and (h(z)<1) for every (z\in[0,z_0]). Therefore

[
q_x
:=
\max_{0\le z\le z_0}h(z)
<1.
]

We prove by induction that

[
z_t\le q_x^t z_0.
]

At (t=0), this is true. If (z_t\le q_x^t z_0), then in particular (z_t\le z_0), and so

# [ z_{t+1}

z_t h(z_t)
\le
q_x z_t
\le
q_x^{t+1}z_0.
]

Thus

[
|x_t|=z_t\le q_x^t |x_0|.
]

Hence

[
x_t\to0
]

geometrically.

------

# 3. Proof that (y_t\to0) geometrically

The (y)-update is

# [ y_{t+1}

\left(1-\eta b,\bar\phi(ax_t)\right)y_t.
]

By assumption,

[
s_{\min}\le \bar\phi(ax_t)\le s_{\max}
]

for all (t). Define

[
c_t:=\eta b,\bar\phi(ax_t).
]

Then

[
\eta b s_{\min}\le c_t\le \eta b s_{\max}.
]

Let

[
c_{\min}:=\eta b s_{\min}>0,
]

and

[
c_{\max}:=\eta b s_{\max}<2.
]

Therefore

[
c_t\in[c_{\min},c_{\max}]\subset(0,2).
]

Define

[
q_y
:=
\max_{c\in[c_{\min},c_{\max}]} |1-c|.
]

Because the interval ([c_{\min},c_{\max}]) is compact and contained strictly inside ((0,2)), we have

[
q_y<1.
]

Therefore

# [ |y_{t+1}|

|1-c_t|,|y_t|
\le
q_y |y_t|.
]

Iterating gives

[
|y_t|
\le
q_y^t |y_0|.
]

Hence

[
y_t\to0
]

geometrically.

A more explicit expression is

# [ q_y

\max\left{
1-\eta b s_{\min},
\eta b s_{\max}-1
\right},
]

where the second term is allowed to be negative if (\eta b s_{\max}<1). Equivalently,

# [ q_y

\max_{c\in[\eta b s_{\min},\eta b s_{\max}]} |1-c|.
]

The important fact is simply

[
0<\eta b s_{\min}
\le
\eta b\bar\phi(ax_t)
\le
\eta b s_{\max}
<2.
]

------

# 4. Proof that the full gradient norm vanishes

The full gradient is

[
\nabla f(x_t,y_t)=(ax_t,by_t).
]

Therefore

# [ |\nabla f(x_t,y_t)|^2

a^2x_t^2+b^2y_t^2.
]

Since

[
x_t\to0
]

and

[
y_t\to0,
]

we get

[
a^2x_t^2+b^2y_t^2\to0.
]

Thus

[
|\nabla f(x_t,y_t)|\to0.
]

Moreover, using the geometric bounds,

[
|x_t|\le q_x^t|x_0|,
\qquad
|y_t|\le q_y^t|y_0|,
]

we have

[
|\nabla f(x_t,y_t)|
\le
a q_x^t|x_0|+b q_y^t|y_0|.
]

So the gradient norm converges to zero at a geometric rate.

------

# 5. What mechanism of the negative theorem is fixed?

The negative theorem failed because the original recovery scale

# [ \phi(ax_t)

\frac{|ax_t|}
{(|ax_t|+\epsilon_a)(|ax_t|+\epsilon_s)}
]

vanishes as

[
x_t\to0.
]

Indeed, for small (x_t),

[
\phi(ax_t)
\sim
\frac{|ax_t|}{\epsilon_a\epsilon_s}.
]

Since (x_t\to0) geometrically, the total orthogonal update strength satisfies

[
\sum_{t=0}^{\infty}\phi(ax_t)<\infty.
]

Thus the (y)-coordinate receives only finite total shrinkage:

# [ y_t

y_0
\prod_{k=0}^{t-1}
\left(1-\eta b\phi(ax_k)\right),
]

and the product converges to a strictly positive limit.

The clipped repair forces

[
\bar\phi(ax_t)\ge s_{\min}>0.
]

Therefore

[
\sum_{t=0}^{\infty}\bar\phi(ax_t)=\infty.
]

The orthogonal coordinate now receives infinite cumulative update strength, so (y_t\to0).

In one sentence:

[
\boxed{
\text{Lower clipping fixes the vanishing-recovery-scale failure.}
}
]

------

# 6. Why (s_{\min}>0) is crucial

Without the lower bound, the recovery scale can vanish:

[
\phi(ax_t)\to0.
]

The effective (y)-learning rate is

[
\eta b\phi(ax_t).
]

If

[
\sum_t \eta b\phi(ax_t)<\infty,
]

then (y_t) may converge to a nonzero value.

The lower bound

[
\bar\phi(ax_t)\ge s_{\min}>0
]

ensures

[
\eta b\bar\phi(ax_t)\ge \eta b s_{\min}>0
]

for every iteration. Hence

[
|y_{t+1}|
\le
q_y|y_t|
]

with (q_y<1), giving geometric decay.

So (s_{\min}>0) is not a minor numerical detail. It is exactly the condition that prevents the orthogonal update channel from shutting off after the projected coordinate has been optimized.

------

# 7. Why (s_{\max}<2/(\eta b)) is a stability condition

The (y)-update is

# [ y_{t+1}

(1-\eta b\bar\phi(ax_t))y_t.
]

For scalar gradient descent on

[
\frac12 b y^2,
]

a step size (\eta \bar\phi) is stable only when

[
0<\eta b\bar\phi<2.
]

If

[
\eta b\bar\phi=2,
]

then

[
y_{t+1}=-y_t,
]

so (y_t) oscillates without decay.

If

[
\eta b\bar\phi>2,
]

then

[
|1-\eta b\bar\phi|>1,
]

so the (y)-coordinate diverges.

Thus requiring

[
s_{\max}<\frac{2}{\eta b}
]

ensures

[
\eta b\bar\phi(ax_t)<2
]

for every (t), preventing overshoot instability in the orthogonal coordinate.

------

# 8. Appendix-ready theorem-proof version

## Theorem

Let

[
f(x,y)=\frac12(ax^2+by^2)
]

with (a,b>0). Let

[
\epsilon_a>0,\qquad \epsilon_s>0,\qquad \eta>0.
]

Define

# [ \phi(r)

\frac{|r|}
{(|r|+\epsilon_a)(|r|+\epsilon_s)}
]

and

# [ \bar\phi(r)

\operatorname{clip}(\phi(r),s_{\min},s_{\max}),
]

where

[
0<s_{\min}\le s_{\max}<\frac{2}{\eta b}.
]

Consider the iteration

# [ x_{t+1}

x_t-\eta\frac{a x_t}{a|x_t|+\epsilon_a},
]

# [ y_{t+1}

y_t-\eta b,\bar\phi(ax_t)y_t.
]

Assume

[
0<\eta a<2\epsilon_a.
]

Then, for every ((x_0,y_0)\in\mathbb R^2),

[
x_t\to0,\qquad y_t\to0,
]

both geometrically. Consequently,

[
|\nabla f(x_t,y_t)|\to0.
]

## Proof

Let

[
z_t:=|x_t|,
\qquad
\lambda:=\eta a,
\qquad
\alpha:=\epsilon_a.
]

The (x)-update can be written as

# [ x_{t+1}

x_t
\left(
1-\frac{\lambda}{az_t+\alpha}
\right).
]

Thus

# [ z_{t+1}

z_t
\left|
1-\frac{\lambda}{az_t+\alpha}
\right|.
]

If (z_0=0), then (x_t=0) for all (t). Assume (z_0>0). Define

# [ h(z)

\left|
1-\frac{\lambda}{az+\alpha}
\right|.
]

Since

[
0<\lambda<2\alpha,
]

for every (z\ge0),

[
0<\frac{\lambda}{az+\alpha}\le \frac{\lambda}{\alpha}<2.
]

Hence

[
h(z)<1.
]

Because (h) is continuous on ([0,z_0]), the quantity

# [ q_x

\max_{0\le z\le z_0}h(z)
]

is well-defined and satisfies

[
q_x<1.
]

We now prove by induction that

[
z_t\le q_x^t z_0.
]

The claim is true for (t=0). If (z_t\le q_x^t z_0), then (z_t\le z_0), so

# [ z_{t+1}

z_th(z_t)
\le
q_x z_t
\le
q_x^{t+1}z_0.
]

Therefore

[
|x_t|\le q_x^t|x_0|,
]

and (x_t\to0) geometrically.

Next define

[
c_t:=\eta b,\bar\phi(ax_t).
]

Since

[
s_{\min}\le \bar\phi(ax_t)\le s_{\max},
]

we have

[
\eta b s_{\min}
\le
c_t
\le
\eta b s_{\max}.
]

Let

[
c_{\min}:=\eta b s_{\min},
\qquad
c_{\max}:=\eta b s_{\max}.
]

By assumption,

[
0<c_{\min}\le c_{\max}<2.
]

The (y)-recursion is

[
y_{t+1}=(1-c_t)y_t.
]

Define

# [ q_y

\max_{c\in[c_{\min},c_{\max}]}|1-c|.
]

Because

[
[c_{\min},c_{\max}]\subset(0,2),
]

we have

[
q_y<1.
]

Therefore

# [ |y_{t+1}|

|1-c_t||y_t|
\le
q_y|y_t|.
]

By induction,

[
|y_t|
\le
q_y^t|y_0|.
]

Hence (y_t\to0) geometrically.

Finally,

[
\nabla f(x_t,y_t)=(ax_t,by_t),
]

so

[
|\nabla f(x_t,y_t)|
\le
a|x_t|+b|y_t|
\le
a q_x^t|x_0|+b q_y^t|y_0|.
]

Since (q_x,q_y<1), the right-hand side converges to zero. Hence

[
|\nabla f(x_t,y_t)|\to0.
]

This proves the theorem. (\square)

------

# 9. Is this theorem too toy?

Yes, it is toy. It is a deterministic two-dimensional diagonal quadratic with a fixed projection. It should **not** be positioned as a general convergence theorem for Fira or Adam.

But it is still useful.

It should be positioned as a **mechanistic repair theorem** paired with the negative theorem:

1. The negative theorem shows that positive-(\epsilon) scalar recovery can fail because the recovery scale vanishes after the projected coordinate is optimized.
2. The positive theorem shows that adding a positive lower bound to the recovery scale exactly removes that failure mode.
3. The upper bound is needed because excessively large recovery scales create scalar gradient-descent instability in the orthogonal direction.

The correct framing is:

> On the minimal quadratic counterexample, scalar norm recovery fails because the orthogonal update channel vanishes. Lower-bounded clipping is the simplest repair: it prevents vanishing recovery while upper clipping prevents overshoot instability.

This theorem is best used as a **diagnostic theorem**, not as the main paper theorem. The next step should be to move from this toy clipped-scale theorem to a broader theorem involving either stochastic projections or residual compensation.