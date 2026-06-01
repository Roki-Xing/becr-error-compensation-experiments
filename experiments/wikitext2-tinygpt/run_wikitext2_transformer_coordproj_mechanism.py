from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
RES_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))


def _download(urls: list[str], out: Path) -> None:
    if out.exists() and out.stat().st_size > 0:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
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
            return
        except Exception as e:
            last_err = e
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
    raise RuntimeError(f"Failed to download {out.name}: {last_err!r}")


def load_wikitext2_raw(data_root: Path) -> dict[str, str]:
    """
    Download and load WikiText-2 raw splits as plain text.
    Uses multiple mirrors to reduce flakiness.
    """
    wt_dir = data_root / "wikitext-2-raw"
    wt_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "train": "wiki.train.raw",
        "valid": "wiki.valid.raw",
        "test": "wiki.test.raw",
    }

    base_urls = [
        "https://cosmo.zip/pub/datasets/wikitext-2-raw/",
        "https://www.cs.cornell.edu/~caiw/projects/wikitext2/wikitext-2-raw/",
    ]

    out = {}
    for split, fname in files.items():
        path = wt_dir / fname
        _download([u + fname for u in base_urls], path)
        out[split] = path.read_text(encoding="utf-8", errors="replace")
    return out


class WordTokenizer:
    def __init__(self, *, vocab_size: int):
        self.vocab_size = int(vocab_size)
        self.special = ["<pad>", "<unk>", "<eos>"]
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(self.special)}
        self.itos: list[str] = list(self.special)

    def build(self, text: str) -> None:
        toks: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            toks.extend(line.split())
            toks.append("<eos>")
        c = Counter(toks)
        keep = max(0, self.vocab_size - len(self.special))
        most = [t for (t, _) in c.most_common(keep) if t not in self.stoi]
        self.itos = list(self.special) + most
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, text: str) -> list[int]:
        unk = self.stoi["<unk>"]
        ids: list[int] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            for t in line.split():
                ids.append(self.stoi.get(t, unk))
            ids.append(self.stoi["<eos>"])
        return ids


class RandomSequenceBatcher:
    def __init__(self, token_ids: list[int], *, seq_len: int):
        if len(token_ids) < seq_len + 2:
            raise ValueError("dataset too small")
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = int(seq_len)

    def sample(self, *, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_i = int(self.data.numel() - self.seq_len - 2)
        idx = torch.randint(0, max_i, (int(batch_size),), device=device)
        x = torch.stack([self.data[i : i + self.seq_len] for i in idx.tolist()], dim=0).to(device)
        y = torch.stack([self.data[i + 1 : i + 1 + self.seq_len] for i in idx.tolist()], dim=0).to(device)
        return x, y


@dataclass
class GPTConfig:
    vocab_size: int = 10000
    seq_len: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    d_ff: int = 1024
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=int(cfg.d_model),
            num_heads=int(cfg.n_heads),
            dropout=float(cfg.dropout),
            batch_first=True,
        )
        mask = torch.triu(torch.ones((cfg.seq_len, cfg.seq_len), dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = int(x.shape[1])
        y, _ = self.attn(x, x, x, need_weights=False, attn_mask=self.causal_mask[:t, :t])
        return y


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(int(cfg.d_model))
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(int(cfg.d_model))
        self.mlp = nn.Sequential(
            nn.Linear(int(cfg.d_model), int(cfg.d_ff)),
            nn.GELU(),
            nn.Linear(int(cfg.d_ff), int(cfg.d_model)),
            nn.Dropout(float(cfg.dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(int(cfg.vocab_size), int(cfg.d_model))
        self.pos_emb = nn.Parameter(torch.zeros((1, int(cfg.seq_len), int(cfg.d_model)), dtype=torch.float32))
        self.drop = nn.Dropout(float(cfg.dropout))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(int(cfg.n_layers))])
        self.ln_f = nn.LayerNorm(int(cfg.d_model))
        self.head = nn.Linear(int(cfg.d_model), int(cfg.vocab_size), bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :t, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, int(self.cfg.vocab_size)), targets.reshape(-1))
        return logits, loss


@dataclass
class CoordProjConfig:
    rank: int = 2048
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
    """
    Mechanism-oriented optimizer:
    - coordinate projection (Top-|g| indices, stale refresh)
    - Adam moments maintained only on the projected coordinates
    - Fira-style scalar recovery scale phi_raw = ||psi||/(||R||+eps_scale)
    - orthogonal update uses either raw scale, clipped scale, or BECR residual compensation
    """

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
                state["e"] = torch.zeros((n,), device=g.device, dtype=torch.float32)

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

                pflat = p.view(-1)
                R = g.index_select(0, idx)
                g2 = float(g.pow(2).sum().item())
                r2 = float(R.pow(2).sum().item())

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

                # Metrics that don't need explicit g_perp.
                tot_g2 += g2
                tot_gpar2 += r2
                tot_gperp2 += max(0.0, g2 - r2)
                phi_raw_list.append(phi_raw)
                s_list.append(float(s))

                if self.mode == "proj_only":
                    pflat.index_add_(0, idx, psi, alpha=-lr)
                    continue

                # g_perp: clone + zero-out projected coords (saves a full g_par allocation).
                gperp = g.clone()
                gperp.index_fill_(0, idx, 0.0)

                if self.mode in {"fira_raw", "fira_clipped"}:
                    pflat.add_(gperp, alpha=-lr * float(s))
                    pflat.index_add_(0, idx, psi, alpha=-lr)
                elif self.mode == "becr":
                    e = state["e"]
                    h = gperp + e
                    z = float(cfg.rho) * h
                    e.copy_(h).sub_(z)
                    tot_e2 += float(e.pow(2).sum().item())
                    pflat.add_(z, alpha=-lr * float(s))
                    pflat.index_add_(0, idx, psi, alpha=-lr)
                elif self.mode == "rho_no_lower_bound":
                    e = state["e"]
                    h = gperp + e
                    rho_t = float(min(float(cfg.rho_max), float(phi_raw)))  # no lower bound, but upper bounded
                    z = rho_t * h
                    e.copy_(h).sub_(z)
                    tot_e2 += float(e.pow(2).sum().item())
                    pflat.add_(z, alpha=-lr * float(s))
                    pflat.index_add_(0, idx, psi, alpha=-lr)
                elif self.mode == "coupled_unbounded":
                    e = state["e"]
                    h = gperp + e
                    rho_t = float(phi_raw)
                    z = rho_t * h
                    e.copy_(h).sub_(z)
                    tot_e2 += float(e.pow(2).sum().item())
                    pflat.add_(z, alpha=-lr * float(s))
                    pflat.index_add_(0, idx, psi, alpha=-lr)
                else:
                    raise RuntimeError("unreachable")

                if pname.endswith(".weight"):
                    acc = layer_acc.get(pname)
                    if acc is None:
                        acc = {"g2": 0.0, "gpar2": 0.0, "gperp2": 0.0, "e2": 0.0, "phi_raw": [], "s": []}
                        layer_acc[pname] = acc
                    acc["g2"] += g2
                    acc["gpar2"] += r2
                    acc["gperp2"] += max(0.0, g2 - r2)
                    if self.mode in {"becr", "rho_no_lower_bound", "coupled_unbounded"}:
                        acc["e2"] += float(state["e"].pow(2).sum().item())
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


@torch.no_grad()
def eval_loss(*, model: nn.Module, batcher: RandomSequenceBatcher, device: torch.device, steps: int, batch_size: int) -> float:
    model.eval()
    losses = []
    for _ in range(int(steps)):
        x, y = batcher.sample(batch_size=int(batch_size), device=device)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def run_one(*, mode: str, seed: int, max_steps: int) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    out_path = RES_DIR / f"wikitext2__{mode}__seed{seed}.json"
    if out_path.exists() and out_path.stat().st_size > 0:
        return json.loads(out_path.read_text(encoding="utf-8"))

    print(f"[run] mode={mode} seed={seed} max_steps={int(max_steps)} device={device}")

    cache_path = DATA_DIR / "wikitext-2-raw" / "cache_vocab10000_ids.npz"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        z = np.load(str(cache_path))
        train_ids = z["train"].astype(np.int64).tolist()
        valid_ids = z["valid"].astype(np.int64).tolist()
        test_ids = z["test"].astype(np.int64).tolist()
        vocab_size = int(z["vocab_size"].reshape(-1)[0])
    else:
        splits = load_wikitext2_raw(DATA_DIR)
        tok = WordTokenizer(vocab_size=10000)
        tok.build(splits["train"])
        train_ids = tok.encode(splits["train"])
        valid_ids = tok.encode(splits["valid"])
        test_ids = tok.encode(splits["test"])
        vocab_size = int(len(tok.itos))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(cache_path),
            train=np.asarray(train_ids, dtype=np.int32),
            valid=np.asarray(valid_ids, dtype=np.int32),
            test=np.asarray(test_ids, dtype=np.int32),
            vocab_size=np.asarray([vocab_size], dtype=np.int32),
        )

    cfg_model = GPTConfig(vocab_size=int(vocab_size), seq_len=128, n_layers=4, n_heads=4, d_model=256, d_ff=1024, dropout=0.1)
    model = TinyGPT(cfg_model).to(device)
    cfg = CoordProjConfig(rank=2048, proj_update_interval=200, eps_adam=1e-8, eps_scale=1e-3, s_min=0.1, s_max=10.0, rho=0.5)

    if mode == "adamw_full":
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    else:
        name_map = {id(p): n for n, p in model.named_parameters()}
        opt = CoordProjRecoveryAdam(model.parameters(), lr=3e-4, cfg=cfg, mode=mode, param_names=name_map)

    train_batcher = RandomSequenceBatcher(train_ids, seq_len=int(cfg_model.seq_len))
    valid_batcher = RandomSequenceBatcher(valid_ids, seq_len=int(cfg_model.seq_len))
    test_batcher = RandomSequenceBatcher(test_ids, seq_len=int(cfg_model.seq_len))

    traces = []
    t0 = time.time()
    eval_every = 200
    trace_every = 50

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else None

    for step in range(1, int(max_steps) + 1):
        x, y = train_batcher.sample(batch_size=32, device=device)
        opt.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        loss.backward()

        if mode == "adamw_full":
            opt.step()
            metrics = {
                "grad_norm": float("nan"),
                "grad_par_norm": float("nan"),
                "grad_perp_norm": float("nan"),
                "phi_raw_mean": float("nan"),
                "phi_raw_p95": float("nan"),
                "s_mean": float("nan"),
                "s_p95": float("nan"),
                "residual_norm": float("nan"),
            }
        else:
            metrics = opt.step()
            if step == 1:
                # Sanity check: projection must be initialized on the first step.
                if not math.isfinite(float(metrics.get("phi_raw_p95", float("nan")))):
                    raise RuntimeError("projection not initialized: phi_raw_p95 is not finite at step=1")

        if step % trace_every == 0 or step == 1:
            val_loss = float("nan")
            if step % eval_every == 0 or step == 1:
                val_loss = eval_loss(model=model, batcher=valid_batcher, device=device, steps=30, batch_size=32)

            per_layer = metrics.get("per_layer", None)
            trace_metrics = {k: v for k, v in metrics.items() if k not in {"per_layer", "step"}}
            traces.append(
                {
                    "step": int(step),
                    "train_loss": float(loss.item()),
                    "val_loss": float(val_loss),
                    **{k: float(v) for k, v in trace_metrics.items()},
                    **({"per_layer": per_layer} if per_layer is not None else {}),
                }
            )

    train_time_s = float(time.time() - t0)
    val_loss_final = eval_loss(model=model, batcher=valid_batcher, device=device, steps=100, batch_size=32)
    test_loss_final = eval_loss(model=model, batcher=test_batcher, device=device, steps=100, batch_size=32)

    out = {
        "mode": mode,
        "seed": int(seed),
        "max_steps": int(max_steps),
        "train_time_s": train_time_s,
        "final_valid_loss": float(val_loss_final),
        "final_test_loss": float(test_loss_final),
        "final_valid_ppl": float(math.exp(min(50.0, float(val_loss_final)))),
        "final_test_ppl": float(math.exp(min(50.0, float(test_loss_final)))),
        "traces": traces,
        "cfg": cfg.__dict__,
        "model_cfg": cfg_model.__dict__,
        "vocab_size": int(vocab_size),
        "device": str(device),
    }

    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[done] mode={mode} seed={seed} valid_ppl={out['final_valid_ppl']:.2f} time={train_time_s:.1f}s")
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
    p.add_argument("--max-steps", type=int, default=1500)
    args = p.parse_args()

    all_out = []
    for seed in args.seeds:
        for m in args.modes:
            all_out.append(run_one(mode=m, seed=int(seed), max_steps=int(args.max_steps)))

    (RES_DIR / "ALL_RUNS.json").write_text(json.dumps(all_out, indent=2), encoding="utf-8")
    print("Done. Wrote to", str(BASE_DIR), "seeds=", args.seeds, "modes=", args.modes, "max_steps=", int(args.max_steps))


if __name__ == "__main__":
    main()
