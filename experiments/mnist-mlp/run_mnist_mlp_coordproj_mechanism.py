from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen
import gzip
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
RES_DIR = BASE_DIR / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))


class MLP(nn.Module):
    def __init__(self, *, width: int = 512, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@dataclass
class CoordProjConfig:
    rank: int = 512
    proj_update_interval: int = 200
    eps_adam: float = 1e-8
    eps_scale: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    s_min: float = 0.1
    s_max: float = 10.0
    rho: float = 0.5
    rho_max: float = 1.0  # for ablation: rho_t = min(phi_raw, rho_max) (no lower bound)


class CoordProjRecoveryAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        *,
        lr: float,
        cfg: CoordProjConfig,
        mode: str,
        param_names: dict[int, str] | None = None,
    ):
        # Backward-compat alias (older logs).
        if mode == "residual_no_bound":
            mode = "coupled_unbounded"
        if mode not in {"proj_only", "fira_raw", "fira_clipped", "becr", "rho_no_lower_bound", "coupled_unbounded"}:
            raise ValueError(f"unknown mode={mode}")
        defaults = dict(lr=float(lr))
        super().__init__(params, defaults)
        self.cfg = cfg
        self.mode = mode
        self._step = 0
        self.param_names = dict(param_names) if param_names is not None else {}

    @torch.no_grad()
    def _maybe_refresh_idx(self, p: torch.Tensor, state: dict) -> None:
        cfg = self.cfg
        if p.grad is None:
            return
        # Important: ensure projection is initialized on the first optimizer step.
        do_refresh = state.get("idx", None) is None
        if cfg.proj_update_interval > 0 and self._step % cfg.proj_update_interval == 0:
            do_refresh = True
        if do_refresh:
            g = p.grad.detach().flatten()
            n = int(g.numel())
            k = int(min(cfg.rank, n))
            if k <= 0:
                state["idx"] = None
                return
            idx = torch.topk(g.abs(), k=k, largest=True, sorted=False).indices
            state["idx"] = idx
            state["m"] = torch.zeros((k,), device=g.device, dtype=torch.float32)
            state["v"] = torch.zeros((k,), device=g.device, dtype=torch.float32)
            state["t"] = 0
            if self.mode in {"becr", "rho_no_lower_bound", "coupled_unbounded"}:
                state["e"] = torch.zeros_like(g, dtype=torch.float32)

    @torch.no_grad()
    def step(self, closure=None):
        cfg = self.cfg
        self._step += 1

        tot_g2 = 0.0
        tot_gpar2 = 0.0
        tot_gperp2 = 0.0
        tot_e2 = 0.0
        phi_raw_list = []
        s_list = []
        layer_acc: dict[str, dict] = {}

        for group in self.param_groups:
            lr = float(group["lr"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                self._maybe_refresh_idx(p, state)

                pname = self.param_names.get(id(p), "")
                g = p.grad.detach().flatten().to(dtype=torch.float32)
                idx = state.get("idx", None)
                if idx is None:
                    p.add_(p.grad, alpha=-lr)
                    continue

                R = g.index_select(0, idx)
                gpar = torch.zeros_like(g)
                gpar.index_copy_(0, idx, R)
                gperp = g - gpar

                state["t"] = int(state.get("t", 0)) + 1
                t = state["t"]
                m = state["m"]
                v = state["v"]
                beta1, beta2 = float(cfg.beta1), float(cfg.beta2)
                m.mul_(beta1).add_(R, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(R, R, value=1.0 - beta2)
                m_hat = m / (1.0 - beta1**t)
                v_hat = v / (1.0 - beta2**t)
                psi = m_hat / (v_hat.sqrt().add_(float(cfg.eps_adam)))

                phi_raw = float(psi.norm() / (R.norm().add_(float(cfg.eps_scale))))
                if self.mode == "fira_raw":
                    s = phi_raw
                else:
                    s = float(max(float(cfg.s_min), min(float(cfg.s_max), phi_raw)))

                upar = torch.zeros_like(g)
                upar.index_copy_(0, idx, psi)

                if self.mode == "proj_only":
                    u = upar
                elif self.mode in {"fira_raw", "fira_clipped"}:
                    u = upar + (float(s) * gperp)
                elif self.mode == "becr":
                    e = state["e"]
                    h = gperp + e
                    z = float(cfg.rho) * h
                    state["e"] = h - z
                    u = upar + (float(s) * z)
                    tot_e2 += float(state["e"].pow(2).sum().item())
                elif self.mode == "rho_no_lower_bound":
                    e = state["e"]
                    h = gperp + e
                    rho_t = float(min(float(cfg.rho_max), float(phi_raw)))  # no lower bound, but upper bounded
                    z = rho_t * h
                    state["e"] = h - z
                    u = upar + (float(s) * z)
                    tot_e2 += float(state["e"].pow(2).sum().item())
                elif self.mode == "coupled_unbounded":
                    e = state["e"]
                    h = gperp + e
                    rho_t = float(phi_raw)
                    z = rho_t * h
                    state["e"] = h - z
                    u = upar + (float(s) * z)
                    tot_e2 += float(state["e"].pow(2).sum().item())
                else:
                    raise RuntimeError("unreachable")

                p.add_(u.view_as(p), alpha=-lr)

                tot_g2 += float(g.pow(2).sum().item())
                tot_gpar2 += float(gpar.pow(2).sum().item())
                tot_gperp2 += float(gperp.pow(2).sum().item())
                phi_raw_list.append(phi_raw)
                s_list.append(float(s))

                # Per-layer (per-parameter) aggregation for weights only (keeps logs compact).
                if pname.endswith(".weight"):
                    acc = layer_acc.get(pname)
                    if acc is None:
                        acc = {"g2": 0.0, "gpar2": 0.0, "gperp2": 0.0, "e2": 0.0, "phi_raw": [], "s": []}
                        layer_acc[pname] = acc
                    acc["g2"] += float(g.pow(2).sum().item())
                    acc["gpar2"] += float(gpar.pow(2).sum().item())
                    acc["gperp2"] += float(gperp.pow(2).sum().item())
                    if self.mode in {"becr", "rho_no_lower_bound", "coupled_unbounded"}:
                        e = state.get("e", None)
                        if e is not None:
                            acc["e2"] += float(e.pow(2).sum().item())
                    acc["phi_raw"].append(float(phi_raw))
                    acc["s"].append(float(s))

        metrics = {
            "step": int(self._step),
            "grad_norm": float(math.sqrt(max(tot_g2, 0.0))),
            "grad_par_norm": float(math.sqrt(max(tot_gpar2, 0.0))),
            "grad_perp_norm": float(math.sqrt(max(tot_gperp2, 0.0))),
            "residual_norm": (
                float(math.sqrt(max(tot_e2, 0.0)))
                if self.mode in {"becr", "rho_no_lower_bound", "coupled_unbounded"}
                else float("nan")
            ),
            "phi_raw_mean": float(np.mean(phi_raw_list)) if phi_raw_list else float("nan"),
            "phi_raw_p95": float(np.percentile(phi_raw_list, 95)) if phi_raw_list else float("nan"),
            "s_mean": float(np.mean(s_list)) if s_list else float("nan"),
            "s_p95": float(np.percentile(s_list, 95)) if s_list else float("nan"),
        }

        if layer_acc:
            per_layer = {}
            for name, acc in layer_acc.items():
                g = float(math.sqrt(max(acc["g2"], 0.0)))
                gp = float(math.sqrt(max(acc["gperp2"], 0.0)))
                gpar = float(math.sqrt(max(acc["gpar2"], 0.0)))
                e = (
                    float(math.sqrt(max(acc["e2"], 0.0)))
                    if self.mode in {"becr", "rho_no_lower_bound", "coupled_unbounded"}
                    else float("nan")
                )
                phi = np.asarray(acc["phi_raw"], dtype=float)
                ss = np.asarray(acc["s"], dtype=float)
                per_layer[name] = {
                    "grad_norm": g,
                    "grad_par_norm": gpar,
                    "grad_perp_norm": gp,
                    "g_perp_ratio": float(gp / (g + 1e-12)),
                    "residual_norm": e,
                    "phi_raw_p95": float(np.percentile(phi, 95)) if phi.size else float("nan"),
                    "s_p95": float(np.percentile(ss, 95)) if ss.size else float("nan"),
                }
            metrics["per_layer"] = per_layer
        return metrics


def get_mnist_loaders(*, batch_size: int, data_root: Path, num_workers: int = 4):
    # Prefetch archive from a mirror to avoid slow MNIST URLs.
    def _md5sum(path: Path) -> str:
        h = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _download(urls: list[str], out: Path, expected_md5: str) -> None:
        if out.exists() and _md5sum(out) == expected_md5:
            return
        if out.exists():
            out.unlink()
        tmp = out.with_suffix(out.suffix + ".partial")
        last_err = None
        for u in urls:
            try:
                with urlopen(u) as r, tmp.open("wb") as f:
                    while True:
                        chunk = r.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                tmp.replace(out)
                if _md5sum(out) != expected_md5:
                    raise RuntimeError("md5 mismatch")
                return
            except Exception as e:
                last_err = e
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
        raise RuntimeError(f"Failed to download {out.name}: {last_err!r}")

    data_root.mkdir(parents=True, exist_ok=True)
    raw_dir = data_root / "MNIST" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
        "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
        "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
    }
    base_urls = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://yann.lecun.com/exdb/mnist/",
    ]
    for fname, md5 in files.items():
        gz_path = raw_dir / fname
        _download([u + fname for u in base_urls], gz_path, md5)
        # Torchvision MNIST expects the *decompressed* raw files to exist.
        raw_path = raw_dir / Path(fname).with_suffix("").name
        if not raw_path.exists():
            with gzip.open(gz_path, "rb") as f_in, raw_path.open("wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    mean = (0.1307,)
    std = (0.3081,)
    train_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_set = torchvision.datasets.MNIST(root=str(data_root), train=True, download=False, transform=train_tf)
    test_set = torchvision.datasets.MNIST(root=str(data_root), train=False, download=False, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, test_loader


@torch.no_grad()
def eval_acc(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return float(correct) / float(total)


def run_one(*, mode: str, seed: int) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = BASE_DIR / "data"
    train_loader, test_loader = get_mnist_loaders(batch_size=256, data_root=data_root, num_workers=4)

    model = MLP(width=512).to(device)
    cfg = CoordProjConfig(rank=512, proj_update_interval=200, eps_adam=1e-8, eps_scale=1e-3, s_min=0.1, s_max=10.0, rho=0.5)

    if mode == "adamw_full":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    else:
        name_map = {id(p): n for n, p in model.named_parameters()}
        opt = CoordProjRecoveryAdam(model.parameters(), lr=1e-3, cfg=cfg, mode=mode, param_names=name_map)

    traces = []
    t0 = time.time()
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    loss_ema = None
    loss_spike_count = 0
    ema_beta = 0.95
    spike_ratio = 1.5
    spike_warmup = 50

    max_steps = 1000
    step = 0
    for epoch in range(999999):
        for x, y in train_loader:
            step += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            lval = float(loss.item())
            if math.isfinite(lval):
                if loss_ema is None:
                    loss_ema = lval
                else:
                    loss_ema = float(ema_beta * float(loss_ema) + (1.0 - ema_beta) * lval)
                if step > spike_warmup and loss_ema is not None and lval > spike_ratio * float(loss_ema):
                    loss_spike_count += 1
            else:
                loss_spike_count += 1

            if mode == "adamw_full":
                opt.step()
                metrics = {"step": step, "grad_norm": float("nan"), "grad_par_norm": float("nan"), "grad_perp_norm": float("nan"), "phi_raw_mean": float("nan"), "phi_raw_p95": float("nan"), "s_mean": float("nan"), "s_p95": float("nan"), "residual_norm": float("nan")}
            else:
                metrics = opt.step()
                metrics["step"] = step
                if step == 1:
                    # Sanity check: projection must be initialized on the first step.
                    if not math.isfinite(float(metrics.get("phi_raw_p95", float("nan")))):
                        raise RuntimeError("projection not initialized: phi_raw_p95 is not finite at step=1")

            if step % 20 == 0 or step == 1:
                per_layer = metrics.get("per_layer", None)
                trace_metrics = {k: v for k, v in metrics.items() if k not in {"step", "per_layer"}}
                traces.append(
                    {
                        "step": int(step),
                        "loss": float(loss.item()),
                        **{k: float(v) for k, v in trace_metrics.items()},
                        **({"per_layer": per_layer} if per_layer is not None else {}),
                    }
                )

            if step >= max_steps:
                break
        if step >= max_steps:
            break

    train_time_s = float(time.time() - t0)
    acc = eval_acc(model, test_loader, device)
    peak_alloc_mb = float("nan")
    peak_reserved_mb = float("nan")
    if device.type == "cuda":
        peak_alloc_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        peak_reserved_mb = float(torch.cuda.max_memory_reserved() / (1024.0 * 1024.0))

    out = {
        "mode": mode,
        "seed": int(seed),
        "max_steps": int(max_steps),
        "test_acc": float(acc),
        "train_time_s": train_time_s,
        "loss_spike_count": int(loss_spike_count),
        "loss_spike_ratio": float(spike_ratio),
        "loss_spike_warmup": int(spike_warmup),
        "peak_mem_alloc_mb": peak_alloc_mb,
        "peak_mem_reserved_mb": peak_reserved_mb,
        "traces": traces,
        "cfg": cfg.__dict__,
        "device": str(device),
    }
    (RES_DIR / f"mnist__{mode}__seed{seed}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="Random seeds to run.")
    p.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["adamw_full", "proj_only", "fira_raw", "fira_clipped", "becr", "rho_no_lower_bound", "coupled_unbounded"],
        help="Modes to run.",
    )
    args = p.parse_args()

    all_out = []
    for seed in args.seeds:
        for m in args.modes:
            all_out.append(run_one(mode=m, seed=int(seed)))
    (RES_DIR / "ALL_RUNS.json").write_text(json.dumps(all_out, indent=2), encoding="utf-8")
    print("Done. Wrote to", str(BASE_DIR), "seeds=", args.seeds, "modes=", args.modes)


if __name__ == "__main__":
    main()
