你现在最该做的不是继续问“大方向”，而是做 **两步闭环**：

1. **先把实验跑对**：你之前那组三张图没有验证 vanishing-scale theorem，因为参数不满足定理条件，而且 `ef_scaled` 不是正确 BECR。最新反馈已经明确指出：你的 (\eta a=0.1)，但 (2\epsilon_a=0.002)，所以落在 projected Adam-like 2-cycle regime，不是 theorem regime；同时 `ef_scaled` 把 residual 存在 update units，还混淆了 (\rho_t) 和 (s_t)，不能当 BECR。
2. **再审 general BECR-SGD theorem**：现在反馈已经给出 BECR-SGD 收敛 theorem 和 Fira/Adam-compatible algorithm，但这部分还需要严格审查 proof 是否可以放论文。

------

## 你现在该做什么

### 第一步：重跑最小 2D 实验

用这组参数，不要再用 (\epsilon_a=\epsilon_s=10^{-3})：

```python
a = 1.0
b = 1.0
lr = 0.5
eps_a = 1.0
eps_s = 1.0
x0 = np.array([1.0, 1.0])
```

这组满足：

[
\eta a=0.5<2\epsilon_a=2
]

[
M_0=\frac14
]

[
\eta bM_0=0.125\le \frac12
]

所以它才是真正的 **positive-(\epsilon) vanishing-scale theorem regime**。

你要跑四个方法：

| 方法               | 目的                                                   |
| ------------------ | ------------------------------------------------------ |
| ProjSGD            | 纯投影 baseline                                        |
| FiraScalar raw     | 验证 (x_t\to0)，(\phi_t\to0)，(y_t\not\to0)            |
| FiraScalar clipped | 验证 lower clipping 修复                               |
| BECRFiraToy        | 验证 bounded transmission + residual compensation 修复 |

重点画：

[
x_t,\quad y_t,\quad \phi_t,\quad \sum_t\phi_t,\quad s_t,\quad \sum_t s_t,\quad |e_t|,\quad |\nabla f|
]

你现在的目标很具体：
**让图像真正支持 negative theorem、clipped repair theorem、BECR repair theorem。**

------

## 第二步：替换错误的 `ef_scaled`

你之前的 `ef_scaled` 不要再叫 BECR。它的问题是：

[
u=e_t+\eta g_t
]

这意味着 residual 存在 update units，不是 gradient units。正确 BECR 应该是：

[
h_t=g_t^\perp+e_t
]

[
z_t=\rho_t h_t
]

[
e_{t+1}=h_t-z_t
]

[
u_t^\perp=s_t z_t
]

[
x_{t+1}=x_t-\eta(u_t^\parallel+u_t^\perp)
]

也就是说：

- (e_t)：raw-gradient units
- (\rho_t)：residual transmission factor
- (s_t)：Adam/Fira adaptive recovery scale
- (\eta)：只出现在参数更新里，不出现在 residual 更新里

------

## 你现在最该问 GPT Pro 的 prompt

直接发这个：

```text
我现在要重跑 2D theorem-regime 实验。请你基于我们当前定义，帮我检查并生成一份正确的最小 notebook。

要求：

1. 使用 theorem-regime 参数：
   a = 1.0
   b = 1.0
   lr = 0.5
   eps_a = 1.0
   eps_s = 1.0
   x0 = [1.0, 1.0]

2. 自动检查 theorem 条件：
   eta*a < 2*eps_a
   eta*b*M0 <= 1/2

3. 实现四个方法：
   - ProjSGD
   - FiraScalar raw positive-epsilon recovery
   - FiraScalar with lower/upper clipping
   - BECRFiraToy with separated rho_t and s_t

4. BECR 必须满足：
   e_t 存在 raw-gradient units；
   h_t = g_t^perp + e_t；
   z_t = rho_t h_t；
   e_{t+1} = h_t - z_t；
   u_t^perp = s_t z_t；
   参数更新中才乘 learning rate。

5. 不要把 rho_t 和 s_t 混在一起。

6. 画图：
   x_t
   y_t
   raw phi_t
   clipped s_t
   cumulative raw phi
   cumulative s_t
   residual norm
   gradient norm
   objective

7. 输出 summary table：
   final x
   final y
   final grad_norm
   final residual_norm
   sum raw phi
   sum clipped s

8. 最后解释每条曲线是否验证：
   - vanishing-scale negative theorem
   - lower-clipped repair theorem
   - BECR repair theorem

请给出可直接运行的 Python/Jupyter 代码。
```

------

## 跑完图以后，你再问这个

把图和 summary table 发给 GPT Pro，然后问：

```text
这是 theorem-regime 2D 实验结果。请你严格判断：

1. raw FiraScalar 是否验证了 positive-epsilon vanishing-scale theorem？
2. 是否出现 x_t -> 0, phi_t -> 0, cumulative phi plateau, y_t -> nonzero？
3. lower-clipped recovery 是否修复了 y_t 不收敛？
4. BECRFiraToy 是否满足 y_t -> 0 且 e_t -> 0？
5. residual without bounded transmission 是否需要作为 ablation？
6. 哪几张图可以作为论文 main figure？
7. 哪些图适合 appendix？
8. 这组 toy experiment 的文字说明应该怎么写？
```

------

## 然后再审 general theorem

等 2D 图跑对以后，再问理论：

```text
现在 2D theorem-regime 实验已经验证。请你严格审查 BECR-SGD general nonconvex convergence theorem。

重点检查：

1. C_t^BECR(v)=C_t(v)+rho_t(I-C_t)v 的 contraction proof 是否正确？
2. 对固定正交投影 C_t=P_tP_t^T，delta=1-kappa^2 是否正确？
3. residual boundedness lemma 是否严谨？
4. 在只假设 bounded variance、不假设 bounded gradient norm 时，proof 中吸收 residual 项的步骤是否成立？
5. perturbed iterate \tilde x_t=x_t-\eta e_t 的推导是否正确？
6. stochastic expectation 的条件独立性有没有隐藏问题？
7. 最终 O(1/sqrt(T)) bound 的常数和步长条件是否正确？
8. 哪些地方需要改成 appendix-ready proof？
9. 这个 theorem 和 classical EF-SGD 相比，真正新增的地方是什么？
10. 这个 theorem 和 LDAdam 相比，如何避免 novelty overlap？
```

------

## 简单说：你现在的任务顺序

| 顺序 | 你要做什么                          | 目的                |
| ---- | ----------------------------------- | ------------------- |
| 1    | 改参数重跑 2D theorem-regime        | 验证主负定理        |
| 2    | 替换错误 `ef_scaled` 为 BECRFiraToy | 验证真正 BECR       |
| 3    | 把图发回 GPT Pro                    | 判断图是否可放论文  |
| 4    | 审 general BECR-SGD theorem         | 把 toy 推成理论贡献 |
| 5    | 审 Fira/Adam-compatible pseudocode  | 准备真实算法        |
| 6    | 再做 novelty audit                  | 防止撞 LDAdam       |

你现在最该做的是 **改代码重跑**。这一步没完成前，不要继续扩展到 MNIST/CIFAR，也不要继续写 abstract。

## 总体判定

这组图**很有价值**，但它们现在验证的不是我们刚刚证明的 positive-(\epsilon) vanishing-scale theorem，也不是正确的 BECR。它们主要暴露了两个问题：

1. 当前 fixed_subspace / stale_then_switch 的参数落在 **low-rank Adam-like projected coordinate 2-cycle regime**，不是 vanishing-scale regime。
2. 当前紫色的 `ef_scaled` 不是 BECR；它把 residual 存在 update units，而且把 (\rho_t) 和 (s_t) 混在了一起，所以它的爆炸不能作为 “BECR 失败” 的证据。

------

# 1. fixed_subspace 结果在说明什么？

你的 fixed_subspace 设置是：

[
a=b=1,\qquad \eta=0.1,\qquad \epsilon_a=\epsilon_s=10^{-3}.
]

我们 positive-(\epsilon) vanishing-scale theorem 需要

[
0<\eta a<2\epsilon_a.
]

但现在

[
\eta a=0.1,\qquad 2\epsilon_a=0.002,
]

所以条件严重不满足。

因此 (x)-坐标不会几何收敛到 0，而是进入我们之前分析过的 positive-(\epsilon) sign-like 2-cycle。对

[
x_{t+1}=x_t-\eta\frac{x_t}{|x_t|+\epsilon_a},
]

对称 2-cycle 的半径是

[
c=\frac{\eta}{2}-\epsilon_a=0.05-0.001=0.049.
]

你的 summary 里，`fira` 和 `fira_clipped` 的最终 (x) 都是大约

[
x_{\mathrm{final}}\approx0.048999997,
]

这正好验证了这个 2-cycle，而不是 vanishing-scale theorem。fixed_subspace 中 `proj_sgd` 最终停在 ((0,1))，梯度范数为 1；`fira` 最终大约为 ((0.049, 5.49\times10^{-9}))，梯度范数约 (0.049)；`fira_clipped` 也是 (x\approx0.049)，(y=0)。这些数值说明 orthogonal (y)-方向被压下去了，但 projected (x)-方向没有收敛。

所以 fixed_subspace 图的正确解释是：

[
\boxed{
\text{当前 fixed_subspace 验证的是 projected Adam-like scalar update 的 2-cycle。}
}
]

不是：

[
\boxed{
\text{positive-(\epsilon) scalar recovery leaves nonzero orthogonal gradient。}
}
]

------

# 2. 为什么 fixed_subspace 中 Fira 的 (y) 没有留下 nonzero limit？

我们的 negative theorem 需要：

[
x_t\to0,
]

从而

[
\phi(ax_t)\to0,
]

并且

[
\sum_t\phi(ax_t)<\infty.
]

但你的当前实验中 (x_t) 没有趋于 0，而是停在周期半径

[
|x_t|\approx0.049.
]

于是

# [ \phi(x_t)

\frac{|x_t|}
{(|x_t|+\epsilon_a)(|x_t|+\epsilon_s)}
\approx
\frac{0.049}{0.05^2}
\approx19.6.
]

因此

[
\sum_t \phi(x_t)=\infty.
]

这会持续更新 (y)，所以 (y_t) 被压到 0 附近。换句话说，fixed_subspace 里的 Fira 没有展示 vanishing-scale failure，因为 scale 没有 vanish。

------

# 3. fira_clipped 为什么没有让梯度范数到 0？

`fira_clipped` 只 clip 了 orthogonal recovery scale：

[
u_t^\perp=\bar\phi_t g_t^\perp.
]

它没有修复 projected branch：

# [ x_{t+1}

x_t-\eta\frac{x_t}{|x_t|+\epsilon_a}.
]

所以即使 (y_t\to0)，(x_t) 仍然停在

[
|x_t|\approx0.049.
]

这正是 fixed_subspace 中 `fira_clipped` 最终梯度范数约 (0.049) 的原因。

这不是 lower-clipped recovery theorem 的失败，而是实验参数不满足 (x)-branch 稳定条件。

要验证 lower-clipped recovery theorem，需要改成例如：

[
\epsilon_a=\epsilon_s=1,\qquad \eta=0.5.
]

这时

[
\eta a=0.5<2\epsilon_a=2,
]

(x_t) 会几何收敛到 0。

------

# 4. fira_limiter 的图为什么也没有修复？

当前 limiter 是 norm-growth limiter。它限制的是 recovered residual norm 的突增：

[
|u_t^\perp| \le \gamma |u_{t-1}^\perp|.
]

它不能修复 projected branch 的 scalar Adam-like 2-cycle，也不能保证 recovery scale 有正下界。

另外，你现在的 `phi` 图里，`fira_limiter` 画的是 **raw (\phi_t)**，不是 limiter 之后的 effective applied scale。

也就是说，图中 `fira_limiter` 的 (\phi) spike 并不等于真实应用到 (g_t^\perp) 上的 effective scale。应该额外记录：

# [ \phi^{\mathrm{eff}}_t

\frac{|u_{t,\mathrm{limited}}^\perp|}
{|g_t^\perp|+\epsilon}.
]

否则 limiter 图会误导读者。

------

# 5. stale_then_switch 结果在说明什么？

stale_then_switch 里，投影先固定为 (e_1)，到第 200 步切换到 (e_2)。

`proj_sgd` 在切换后可以把两个坐标都优化掉，最终梯度范数约 (10^{-9})。这说明只要 subspace 最终覆盖缺失方向，普通 projected SGD 可以恢复。

但 `fira`、`fira_clipped`、`fira_limiter` 最终梯度范数都约 (0.049)。原因不是 stale subspace 之后仍然缺失方向，而是切换后新的 projected coordinate 也进入了同样的 Adam-like 2-cycle：

[
c=\frac{\eta}{2}-\epsilon_a=0.049.
]

所以 stale_then_switch 的正确解释是：

[
\boxed{
\text{切换 subspace 后，Fira-like projected branch 把 2-cycle 从 (x) 坐标转移到了 (y) 坐标。}
}
]

它不是 vanishing-scale theorem 的干净验证。

------

# 6. rotating_noise 结果在说明什么？

rotating_noise 是有用的 stress test，但不能作为定理验证。

参数是：

[
a=1,\qquad b=5,\qquad \eta=0.05,\qquad \epsilon_a=\epsilon_s=10^{-3}.
]

此时

# [ M_0

# \max_{r\ge0} \frac{r}{(r+\epsilon_a)(r+\epsilon_s)}

# \frac{1}{(2\sqrt{10^{-3}})^2}

1. 

]

所以 (\phi_t) 理论上最高可以接近 250。你的 (\phi) 图确实出现了接近 250 的 spike。

这说明：

[
\boxed{
\text{small denominator epsilon + rotating/noisy projection 会制造 huge recovery-scale spikes。}
}
]

从 summary 看，rotating_noise 中 `fira` 最终梯度范数约 (0.562)，`fira_clipped` 约 (0.173)，`fira_limiter` 约 (0.404)，而 `ef_scaled` 爆到约 (4.575)。

这可以支持一个实验性观察：

[
\boxed{
\text{clipping 比 raw Fira recovery 更稳定，limiter 可抑制部分 spike，但不能保证收敛。}
}
]

但不要把这个图解释成 BECR theorem。紫色 `ef_scaled` 当前不是正确 BECR。

------

# 7. 当前 `ef_scaled` 不是 BECR

我检查了 notebook 里的 `EFSgdScaled`，核心实现是：

```python
u = self.e + self.lr * g
c, phi = self._compress(u, t)
x_next = x - c
self.e = u - c
```

这有三个问题。

## 问题 1：residual 存在 update units

因为你用了

[
u=e_t+\eta g_t.
]

所以 (e_t) 是 displacement/update units，而不是 gradient units。

但我们定义 BECR 时要求：

[
e_t,g_t,h_t,z_t
]

全部在 raw-gradient units。

正确形式应是：

[
h_t=g_t^\perp+e_t,
]

[
z_t=\rho_t h_t,
]

[
e_{t+1}=h_t-z_t,
]

[
x_{t+1}=x_t-\eta(\cdots).
]

## 问题 2：`_compress(u,t)` 对 update units 使用了 Adam-like (\psi)

`_compress` 里面算：

[
\psi(r)=\frac{r}{|r|+\epsilon_a}.
]

但这里 (r) 来自 (u=e+\eta g)，不是 gradient。于是 (\epsilon_a) 的量纲也错了。

## 问题 3：(\rho_t) 和 (s_t) 被混在一起

当前 `ef_scaled` 实际上把 residual transmission、adaptive normalization、parameter update 三件事混成了一个非线性 compressor。

所以 rotating_noise 里紫色曲线爆炸不能说明 BECR 会爆炸。它说明的是：

[
\boxed{
\text{naive scaled EF with mixed units is unstable。}
}
]

这个结果反而支持我们的设计原则：必须区分

[
\rho_t=\text{raw-gradient transmission}
]

和

[
s_t=\text{adaptive recovery/preconditioning scale}.
]

------

# 8. 现在这些图可以怎么用？

可以用，但要非常谨慎。

## 可以写的结论

1. fixed_subspace 验证了 pure projected SGD 的方向缺失：`proj_sgd` 最终停在 ((0,1))，梯度范数为 1。
2. 当前 Fira-like scalar update 在 (\eta a>2\epsilon_a) 时会进入 projected-coordinate 2-cycle，最终梯度范数约 (0.049)。
3. rotating/noisy subspace 下 raw recovery scale 会产生巨大 spike，clip 明显降低不稳定性。
4. naive EF-scaled implementation 可能非常不稳定，因此 residual compensation 必须做量纲分离和 bounded transmission。

## 不能写的结论

不能写：

[
\text{“BECR fails on rotating noise.”}
]

不能写：

[
\text{“This validates the vanishing-scale theorem.”}
]

不能写：

[
\text{“lower clipping is insufficient.”}
]

当前 lower clipping 没让 grad norm 到 0 的原因是 projected branch 自身 2-cycle，不是 orthogonal clipping 失败。

------

# 9. 下一版实验应该怎么改？

## 9.1 验证 negative theorem 的参数

把 fixed_subspace 改成：

```python
lr = 0.5
eps_a = 1.0
eps_s = 1.0
a = 1.0
b = 1.0
x0 = np.array([1.0, 1.0])
```

则：

[
\eta a=0.5<2\epsilon_a=2.
]

并且

[
M_0=\max_{r\in[0,1]}\frac{r}{(r+1)^2}=\frac14,
]

所以

[
\eta bM_0=0.5\times1\times\frac14=0.125\le\frac12.
]

这正好满足 theorem 条件。

预期：

- raw Fira positive-(\epsilon)：(x_t\to0)，(\phi_t\to0)，(\sum_t\phi_t<\infty)，(y_t\to y_\infty\neq0)；
- lower-clipped recovery：(x_t\to0)，(y_t\to0)；
- BECR：(x_t\to0)，(y_t\to0)，(e_t\to0)。

------

# 10. 正确的 BECR toy implementation

请先替换当前 `EFSgdScaled`，不要把它再叫 BECR。

下面是量纲一致的 BECR-Fira toy version。

```python
class BECRFiraToy(Optim):
    def __init__(
        self,
        *,
        lr: float,
        angle_fn,
        eps_a: float = 1.0,
        eps_s: float = 1.0,
        rho: float = 0.5,
        s_clip: tuple[float, float] = (0.2, 0.5),
        limiter_gamma: float | None = None,
    ):
        self.name = "becr"
        self.lr = float(lr)
        self.angle_fn = angle_fn
        self.eps_a = float(eps_a)
        self.eps_s = float(eps_s)
        self.rho = float(rho)
        self.s_clip = s_clip
        self.limiter_gamma = limiter_gamma
        self.e = np.zeros((2,), dtype=float)  # raw-gradient units
        self._prev_u_perp_norm = None

    def _psi(self, r: float) -> float:
        return r / (abs(r) + self.eps_a)

    def step(self, x: np.ndarray, g: np.ndarray, t: int) -> tuple[np.ndarray, dict]:
        p = unit(float(self.angle_fn(t)))

        # Projection decomposition in gradient units.
        r = float(np.dot(p, g))
        g_parallel = p * r
        g_perp = g - g_parallel

        # Low-rank Adam-like branch.
        psi_r = float(self._psi(r))
        u_parallel = p * psi_r

        # Fira-style adaptive recovery/preconditioning scale.
        s_raw = abs(psi_r) / (abs(r) + self.eps_s)
        s_min, s_max = self.s_clip
        s_t = float(np.clip(s_raw, s_min, s_max))

        # BECR raw-gradient residual compensation.
        h = g_perp + self.e               # raw-gradient units
        z = self.rho * h                   # transmitted raw-gradient signal
        z_eff = z.copy()

        # Orthogonal Adam-like update direction.
        u_perp = s_t * z_eff

        # Optional norm-growth limiter on the actual orthogonal update.
        if self.limiter_gamma is not None:
            cur = float(np.linalg.norm(u_perp))
            if (
                self._prev_u_perp_norm is not None
                and self._prev_u_perp_norm > 0
                and cur > self.limiter_gamma * self._prev_u_perp_norm
            ):
                tau = self.limiter_gamma * self._prev_u_perp_norm / cur
                z_eff = tau * z
                u_perp = s_t * z_eff
                cur = float(np.linalg.norm(u_perp))

            self._prev_u_perp_norm = cur

        # Residual update must use the effective transmitted raw signal.
        self.e = h - z_eff

        u = u_parallel + u_perp
        x_next = x - self.lr * u

        gperp_norm = float(np.linalg.norm(g_perp))
        phi_eff = float(np.linalg.norm(u_perp) / (gperp_norm + 1e-12))

        return x_next, {
            "phi": s_t,
            "s_raw": float(s_raw),
            "rho": self.rho,
            "theta": float(self.angle_fn(t)),
            "u_norm": float(np.linalg.norm(u)),
            "u_perp_norm": float(np.linalg.norm(u_perp)),
            "e_norm": float(np.linalg.norm(self.e)),
            "phi_eff": phi_eff,
        }
```

这个实现满足：

[
e_t \text{ in raw-gradient units},
]

[
z_t=\rho_t(g_t^\perp+e_t),
]

[
e_{t+1}=g_t^\perp+e_t-z_t,
]

[
u_t^\perp=s_tz_t.
]

这才是我们定义的 BECR-Fira toy abstraction。

------

# 11. BECR separated stability condition

如果使用分离形式：

[
h_t=by_t+e_t,
]

[
z_t=\rho h_t,
]

[
e_{t+1}=h_t-z_t=(1-\rho)h_t,
]

[
y_{t+1}=y_t-\eta s z_t,
]

那么当 (s,\rho) 为常数时，((y_t,e_t)) 系统矩阵是

[
A=
\begin{pmatrix}
1-\eta s\rho b & -\eta s\rho\
(1-\rho)b & 1-\rho
\end{pmatrix}.
]

其 Schur 稳定的一个干净充分条件是：

[
0<\rho\le1,
\qquad
0<\eta b s_{\max}<2.
]

更精确地，对常数 (s,\rho)，稳定条件是：

[
0<\rho<\frac{4}{2+\eta b s}.
]

因此实验里建议先用：

```python
rho = 0.5
s_clip = (0.2, 0.5)
lr = 0.5
b = 1.0
```

这样

[
\eta b s_{\max}=0.5\times1\times0.5=0.25<2,
]

非常安全。

------

# 12. 建议重跑的最小实验矩阵

先只跑 deterministic fixed_subspace，别马上用 rotating_noise。

```python
def make_theorem_optimizers(*, lr, angle_fn):
    return [
        ProjSGD(lr=lr, angle_fn=angle_fn),

        FiraScalar(
            lr=lr,
            angle_fn=angle_fn,
            eps_a=1.0,
            eps_s=1.0,
            clip=None,
            limiter_gamma=None,
        ),

        FiraScalar(
            lr=lr,
            angle_fn=angle_fn,
            eps_a=1.0,
            eps_s=1.0,
            clip=(0.2, 0.5),
            limiter_gamma=None,
        ),

        BECRFiraToy(
            lr=lr,
            angle_fn=angle_fn,
            eps_a=1.0,
            eps_s=1.0,
            rho=0.5,
            s_clip=(0.2, 0.5),
            limiter_gamma=None,
        ),
    ]
```

并打印 theorem 条件：

```python
def theorem_checks(a, b, lr, eps_a, eps_s, x0):
    R = a * abs(x0[0])
    r_star = np.sqrt(eps_a * eps_s)

    def phi_r(r):
        return r / ((r + eps_a) * (r + eps_s))

    M0 = phi_r(min(R, r_star)) if R >= r_star else phi_r(R)

    print("Theorem checks")
    print("--------------")
    print("eta*a < 2 eps_a:", lr * a, "<", 2 * eps_a, "=", lr * a < 2 * eps_a)
    print("M0:", M0)
    print("eta*b*M0 <= 1/2:", lr * b * M0, "<=", 0.5, "=", lr * b * M0 <= 0.5)
```

然后记录这些指标：

- (x_t)
- (y_t)
- raw (\phi_t)
- clipped (s_t)
- (\rho_t)
- (\sum_t \phi_t)
- (\sum_t s_t)
- (|e_t|)
- (|\nabla f|)
- (f(x_t,y_t))

尤其要画：

[
\sum_t\phi_t
]

和

[
\sum_t s_t.
]

预期图像：

- raw Fira：(\sum_t\phi_t) plateau，(y_t) plateau；
- lower-clipped：(\sum_t s_t) 线性增长，(y_t\to0)；
- BECR：(\sum_t s_t) 线性增长，(y_t\to0)，(e_t\to0)。

------

# 13. 当前三组图的论文定位

这三组图现在可以放在 internal note 里，不建议直接作为 paper figure。

原因：

1. fixed_subspace 不是 theorem-regime；
2. stale_then_switch 混入了 projected-branch 2-cycle；
3. rotating_noise 是有趣 stress test，但 `ef_scaled` implementation 不是 BECR；
4. limiter 的 (\phi) 图不是 effective applied scale。

更合适的 paper figure 顺序应该是：

1. **Theorem-regime fixed subspace**：验证 vanishing-scale negative theorem。
2. **Lower-clipped repair**：验证 lower bound fixes vanishing recovery。
3. **BECR repair**：验证 residual + bounded transmission。
4. **Wrong-units naive EF ablation**：可选，用来说明为什么必须区分 (\rho_t) 和 (s_t)。

当前上传结果可以作为一个很好的 debug record：它告诉我们，下一版实验必须先把 theorem conditions 和 BECR units 对齐。