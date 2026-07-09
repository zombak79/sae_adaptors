from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "EmbeddingsDataset",
    "L1Normalize",
    "L2Normalize",
    "BetaVAEConfig",
    "BetaVAE",
    "BetaVAETrainer",
]


class EmbeddingsDataset:
    """Small batch-oriented dataset for in-memory embedding matrices."""

    def __init__(
        self,
        embeddings: np.ndarray | torch.Tensor,
        *,
        batch_size: int = 128,
        shuffle: bool = True,
        seed: int = 42,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        tensor = torch.as_tensor(embeddings)

        if tensor.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {tuple(tensor.shape)}")

        if not torch.is_floating_point(tensor):
            tensor = tensor.float()

        if dtype is not None:
            tensor = tensor.to(dtype=dtype)

        self.embeddings = tensor.contiguous()
        self.n, self.dim = int(tensor.shape[0]), int(tensor.shape[1])
        self.indices = np.arange(self.n)
        self.rng = np.random.default_rng(seed)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.device = torch.device(device)

    def __len__(self) -> int:
        return int(np.ceil(self.n / self.batch_size))

    def __iter__(self):
        for batch_idx in range(len(self)):
            yield self[batch_idx]

    def __getitem__(self, batch_idx: int) -> torch.Tensor:
        start = int(batch_idx) * self.batch_size
        end = min(start + self.batch_size, self.n)
        rows = self.indices[start:end]
        batch = self.embeddings[torch.as_tensor(rows, dtype=torch.long)]
        return batch.to(self.device, non_blocking=True)

    def to(self, device: str | torch.device) -> "EmbeddingsDataset":
        device = torch.device(device)
        self.embeddings[:1].to(device)
        self.device = device
        return self

    def on_epoch_begin(self) -> None:
        pass

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)


class L1Normalize(nn.Module):
    """Apply row-wise L1 normalization."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=1.0, dim=-1)


class L2Normalize(nn.Module):
    """Apply row-wise L2 normalization."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2.0, dim=-1)


def _activation(name: Literal["relu", "gelu", "silu", "tanh", "identity"]) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"unknown activation: {name}")


def _make_mlp(
    dims: Sequence[int],
    *,
    activation: Literal["relu", "gelu", "silu", "tanh", "identity"] = "gelu",
    final_activation: nn.Module | None = None,
    bias: bool = True,
) -> nn.Sequential:
    if len(dims) < 2:
        raise ValueError("dims must contain at least input and output dimension")

    layers: list[nn.Module] = []

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(int(dims[i]), int(dims[i + 1]), bias=bias))

        is_last = i == len(dims) - 2

        if not is_last:
            layers.append(_activation(activation))
        elif final_activation is not None:
            layers.append(final_activation)

    return nn.Sequential(*layers)


@dataclass(frozen=True)
class BetaVAEConfig:
    """Configuration for :class:`BetaVAETrainer`.

    Parameters
    ----------
    latent_dim:
        Width of the latent representation.

    encoder_hidden_dims, decoder_hidden_dims:
        Optional hidden dimensions for encoder/decoder MLPs. Empty tuples
        produce simple linear projections.

    activation:
        Activation used inside encoder/decoder MLPs.

    input_normalize:
        Optional module applied to input embeddings before encoding. For
        semantic embeddings, ``L2Normalize()`` is often useful.

    latent_normalize:
        Optional module applied to returned latent means in ``encode``. Usually
        leave as ``None`` during training. If you want normalized retrieval
        vectors, normalize after training or set this carefully.

    output_normalize:
        Optional module applied to decoder output. Usually leave as ``None``.
        If evaluating reconstructions only by cosine, ``L2Normalize()`` can
        make sense.

    beta_kl:
        Weight of the KL term. For recommendation embeddings, start small:
        ``1e-4``, ``1e-3``, ``1e-2``. Standard VAE-like ``1.0`` is often too
        destructive for retrieval.

    free_bits:
        Optional minimum KL per latent dimension. This prevents the KL term from
        immediately crushing the posterior into the prior. Set to ``0.0`` to
        disable. Values like ``0.01`` can be useful.

    kl_warmup_epochs:
        Number of epochs over which beta is linearly warmed up from zero to
        ``beta_kl``. Useful because early KL pressure can cause posterior
        collapse.

    logvar_min, logvar_max:
        Clamp range for log-variance to avoid numerical instability.

    alpha_loss:
        Mixture weight for cosine reconstruction loss. Training reconstruction
        loss is
        ``alpha_loss * (1 - cosine_similarity) + (1 - alpha_loss) * mse``.

    batch_size, shuffle, seed, epochs, lr, weight_decay, decay, compile, device:
        Training parameters same as in dae.
    """

    latent_dim: int = 256
    encoder_hidden_dims: tuple[int, ...] = ()
    decoder_hidden_dims: tuple[int, ...] = ()
    activation: Literal["relu", "gelu", "silu", "tanh", "identity"] = "gelu"

    input_normalize: nn.Module | None = None
    latent_normalize: nn.Module | None = None
    output_normalize: nn.Module | None = None

    beta_kl: float = 1e-3
    free_bits: float = 0.0
    kl_warmup_epochs: int = 0
    logvar_min: float = -10.0
    logvar_max: float = 10.0

    alpha_loss: float = 0.01

    batch_size: int = 128
    shuffle: bool = True
    seed: int = 42
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    decay: bool = False
    compile: bool = False
    device: str | torch.device = "cpu"
    show_progress: bool = True


class BetaVAE(nn.Module):
    """Simple beta-VAE for dense embedding vectors."""

    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        encoder_hidden_dims: Sequence[int] = (),
        decoder_hidden_dims: Sequence[int] = (),
        activation: Literal["relu", "gelu", "silu", "tanh", "identity"] = "gelu",
        input_normalize: nn.Module | None = None,
        latent_normalize: nn.Module | None = None,
        output_normalize: nn.Module | None = None,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
        encoder_backbone: nn.Module | None = None,
        decoder: nn.Module | None = None,
    ) -> None:
        super().__init__()

        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if logvar_min >= logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max")

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        self.input_normalize = input_normalize
        self.latent_normalize = latent_normalize
        self.output_normalize = output_normalize

        if encoder_backbone is None:
            encoder_out_dim = int(encoder_hidden_dims[-1]) if len(encoder_hidden_dims) > 0 else input_dim
            self.encoder_backbone = _make_mlp(
                [input_dim, *encoder_hidden_dims],
                activation=activation,
            ) if len(encoder_hidden_dims) > 0 else nn.Identity()
        else:
            encoder_out_dim = int(encoder_hidden_dims[-1]) if len(encoder_hidden_dims) > 0 else input_dim
            self.encoder_backbone = encoder_backbone

        self.mu_head = nn.Linear(encoder_out_dim, latent_dim)
        self.logvar_head = nn.Linear(encoder_out_dim, latent_dim)

        if decoder is None:
            self.decoder = _make_mlp(
                [latent_dim, *decoder_hidden_dims, input_dim],
                activation=activation,
            )
        else:
            self.decoder = decoder

    def encode_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior parameters ``mu`` and ``logvar``."""
        if self.input_normalize is not None:
            x = self.input_normalize(x)

        h = self.encoder_backbone(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(self.logvar_min, self.logvar_max)

        return mu, logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return deterministic latent mean representation."""
        mu, _logvar = self.encode_params(x)

        if self.latent_normalize is not None:
            mu = self.latent_normalize(mu)

        return mu

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, *, sample: bool = True) -> torch.Tensor:
        """Sample latent using the reparameterization trick.

        If ``sample=False``, return ``mu``.
        """
        if not sample:
            return mu

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to input space."""
        reconstruction = self.decoder(z)

        if self.output_normalize is not None:
            reconstruction = self.output_normalize(reconstruction)

        return reconstruction

    def forward(
        self,
        x: torch.Tensor,
        *,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return reconstruction, z, mu, logvar, and stats."""
        mu, logvar = self.encode_params(x)
        z = self.reparameterize(mu, logvar, sample=sample)
        reconstruction = self.decode(z)

        target = self.input_normalize(x) if self.input_normalize is not None else x

        cosine_similarity = F.cosine_similarity(reconstruction, target, dim=-1).mean()
        reconstruction_mse = F.mse_loss(reconstruction, target)

        kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        kl = kl_per_dim.sum(dim=-1).mean()

        stats = {
            "cosine_similarity": cosine_similarity.detach(),
            "reconstruction_mse": reconstruction_mse.detach(),
            "kl": kl.detach(),
            "mu_abs_mean": mu.abs().mean().detach(),
            "mu_l2_mean": mu.pow(2).sum(dim=-1).sqrt().mean().detach(),
            "logvar_mean": logvar.mean().detach(),
            "posterior_std_mean": torch.exp(0.5 * logvar).mean().detach(),
        }

        return reconstruction, z, mu, logvar, stats


class BetaVAETrainer:
    """Efficient fit/transform wrapper around :class:`BetaVAE`.

    The trainer is intended for dense embedding matrices, such as item
    embeddings from a recommender or semantic embeddings from a text encoder.

    API
    ---
    >>> trainer = BetaVAETrainer(BetaVAEConfig(latent_dim=256, beta_kl=1e-3))
    >>> trainer.fit(embeddings)
    >>> mu = trainer.transform(embeddings)
    >>> x_hat = trainer.reconstruct(embeddings)

    ``transform`` returns deterministic latent means ``mu``.
    """

    def __init__(self, config: BetaVAEConfig | None = None) -> None:
        self.cfg = config if config is not None else BetaVAEConfig()
        self.device = torch.device(self.cfg.device)
        self.vae: BetaVAE | nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.input_dim: int | None = None
        self.history: list[dict[str, float]] = []
        self._epoch: int = 0

    @property
    def is_built(self) -> bool:
        return self.vae is not None

    def build(self, input_dim: int) -> "BetaVAETrainer":
        if self.is_built:
            if int(input_dim) != self.input_dim:
                raise ValueError(
                    f"trainer is already built for input_dim={self.input_dim}, got {input_dim}"
                )
            return self

        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if self.cfg.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.cfg.beta_kl < 0.0:
            raise ValueError("beta_kl must be >= 0")
        if self.cfg.free_bits < 0.0:
            raise ValueError("free_bits must be >= 0")
        if self.cfg.kl_warmup_epochs < 0:
            raise ValueError("kl_warmup_epochs must be >= 0")
        if not 0.0 <= self.cfg.alpha_loss <= 1.0:
            raise ValueError("alpha_loss must be in [0, 1]")

        torch.manual_seed(int(self.cfg.seed))

        self.input_dim = int(input_dim)

        model = BetaVAE(
            input_dim=self.input_dim,
            latent_dim=int(self.cfg.latent_dim),
            encoder_hidden_dims=self.cfg.encoder_hidden_dims,
            decoder_hidden_dims=self.cfg.decoder_hidden_dims,
            activation=self.cfg.activation,
            input_normalize=self.cfg.input_normalize,
            latent_normalize=self.cfg.latent_normalize,
            output_normalize=self.cfg.output_normalize,
            logvar_min=float(self.cfg.logvar_min),
            logvar_max=float(self.cfg.logvar_max),
        ).to(self.device)

        if self.cfg.compile:
            model = torch.compile(model)  # type: ignore[assignment]

        self.vae = model
        self.optimizer = torch.optim.AdamW(
            self.vae.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

        return self

    def to(self, device: str | torch.device) -> "BetaVAETrainer":
        self.device = torch.device(device)

        if self.vae is not None:
            self.vae.to(self.device)

        return self

    def _dataset(self, embeddings: np.ndarray | torch.Tensor, *, shuffle: bool) -> EmbeddingsDataset:
        return EmbeddingsDataset(
            embeddings,
            batch_size=int(self.cfg.batch_size),
            shuffle=shuffle,
            seed=int(self.cfg.seed),
            device=self.device,
        )

    def _progress(self, iterable, *, total: int | None = None):
        if not self.cfg.show_progress:
            return iterable

        try:
            from tqdm.auto import tqdm
        except Exception:
            return iterable

        return tqdm(iterable, total=total)

    def _set_lr(self, lr: float) -> None:
        if self.optimizer is None:
            raise RuntimeError("trainer must be built before setting learning rate")

        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def _current_lr(self) -> float:
        if self.optimizer is None:
            raise RuntimeError("trainer must be built before reading learning rate")

        return float(self.optimizer.param_groups[0]["lr"])

    def _current_beta(self) -> float:
        """Return KL beta after optional warmup."""
        beta = float(self.cfg.beta_kl)

        if self.cfg.kl_warmup_epochs <= 0:
            return beta

        warmup = max(1, int(self.cfg.kl_warmup_epochs))
        scale = min(1.0, float(self._epoch) / float(warmup))
        return beta * scale

    def _target(self, batch: torch.Tensor) -> torch.Tensor:
        """Return reconstruction target after optional input normalization."""
        if self.vae is None:
            raise RuntimeError("trainer must be built before preprocessing inputs")

        module = self.vae
        if hasattr(module, "_orig_mod"):
            module = module._orig_mod  # type: ignore[attr-defined]

        input_normalize = getattr(module, "input_normalize", None)

        if input_normalize is not None:
            return input_normalize(batch)

        return batch

    def _kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Return KL loss with optional free bits.

        ``free_bits`` is interpreted per latent dimension.
        """
        kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())

        if self.cfg.free_bits > 0.0:
            kl_per_dim = torch.clamp(kl_per_dim, min=float(self.cfg.free_bits))

        return kl_per_dim.sum(dim=-1).mean()

    def train_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.vae is None or self.optimizer is None:
            raise RuntimeError("trainer must be built before train_step")

        self.vae.train()
        self.optimizer.zero_grad(set_to_none=True)

        reconstruction, _z, mu, logvar, stats = self.vae(batch, sample=True)

        target = self._target(batch)

        cosine_loss = 1.0 - F.cosine_similarity(reconstruction, target, dim=-1).mean()
        mse = F.mse_loss(reconstruction, target)

        reconstruction_loss = (
            float(self.cfg.alpha_loss) * cosine_loss
            + (1.0 - float(self.cfg.alpha_loss)) * mse
        )

        kl = self._kl_loss(mu, logvar)
        beta = self._current_beta()

        loss = reconstruction_loss + beta * kl

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.detach(),
            "reconstruction_loss": reconstruction_loss.detach(),
            "cosine_loss": cosine_loss.detach(),
            "reconstruction_mse": mse.detach(),
            "kl": kl.detach(),
            "beta_kl": torch.tensor(beta, device=batch.device),
            "mu_abs_mean": mu.abs().mean().detach(),
            "mu_l2_mean": mu.pow(2).sum(dim=-1).sqrt().mean().detach(),
            "logvar_mean": logvar.mean().detach(),
            "posterior_std_mean": torch.exp(0.5 * logvar).mean().detach(),
            # forward stats, useful to compare target consistency
            "forward_cosine_similarity": stats["cosine_similarity"].detach(),
            "forward_reconstruction_mse": stats["reconstruction_mse"].detach(),
        }

    def fit(self, embeddings: np.ndarray | torch.Tensor) -> "BetaVAETrainer":
        dataset = self._dataset(embeddings, shuffle=bool(self.cfg.shuffle))
        self.build(dataset.dim)

        epochs = int(self.cfg.epochs)

        if epochs < 1:
            raise ValueError("epochs must be >= 1")

        self._set_lr(float(self.cfg.lr))

        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=0.0,
            )
            if self.cfg.decay
            else None
        )

        epoch_iter = self._progress(range(1, epochs + 1), total=epochs)

        for epoch in epoch_iter:
            self._epoch = int(epoch)

            dataset.on_epoch_begin()

            sums: dict[str, float] = {}
            n_batches = 0

            for batch in dataset:
                stats = self.train_step(batch)

                for key, value in stats.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach().cpu().item())

                n_batches += 1

            dataset.on_epoch_end()

            record = {key: value / max(1, n_batches) for key, value in sums.items()}
            record["epoch"] = float(epoch)
            record["lr"] = self._current_lr()
            self.history.append(record)

            if hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(
                    {
                        "loss": f"{record['loss']:.4f}",
                        "mse": f"{record['reconstruction_mse']:.4E}",
                        "kl": f"{record['kl']:.2f}",
                        "beta": f"{record['beta_kl']:.1E}",
                        "lr": f"{record['lr']:.2E}",
                    }
                )

            if scheduler is not None:
                scheduler.step()

        return self

    @torch.no_grad()
    def encode(self, embeddings: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return deterministic latent means ``mu``."""
        if self.vae is None:
            raise RuntimeError("trainer must be fitted or built before encode")

        dataset = self._dataset(embeddings, shuffle=False)
        self.vae.eval()

        latents: list[torch.Tensor] = []

        module = self.vae
        if hasattr(module, "_orig_mod"):
            module = module._orig_mod  # type: ignore[attr-defined]

        for batch in self._progress(dataset, total=len(dataset)):
            mu = module.encode(batch)  # type: ignore[attr-defined]
            latents.append(mu.detach().cpu())

        return torch.cat(latents, dim=0)

    @torch.no_grad()
    def encode_params(self, embeddings: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior parameters ``mu`` and ``logvar`` for embeddings."""
        if self.vae is None:
            raise RuntimeError("trainer must be fitted or built before encode_params")

        dataset = self._dataset(embeddings, shuffle=False)
        self.vae.eval()

        mus: list[torch.Tensor] = []
        logvars: list[torch.Tensor] = []

        module = self.vae
        if hasattr(module, "_orig_mod"):
            module = module._orig_mod  # type: ignore[attr-defined]

        for batch in self._progress(dataset, total=len(dataset)):
            mu, logvar = module.encode_params(batch)  # type: ignore[attr-defined]

            if getattr(module, "latent_normalize", None) is not None:
                # Keep encode_params raw. Normalization belongs to encode().
                pass

            mus.append(mu.detach().cpu())
            logvars.append(logvar.detach().cpu())

        return torch.cat(mus, dim=0), torch.cat(logvars, dim=0)

    @torch.no_grad()
    def sample_latent(
        self,
        embeddings: np.ndarray | torch.Tensor,
        *,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Return sampled latent vectors.

        If ``num_samples=1``, returns shape ``(n, latent_dim)``.
        Otherwise returns shape ``(num_samples, n, latent_dim)``.
        """
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")

        if self.vae is None:
            raise RuntimeError("trainer must be fitted or built before sample_latent")

        mu, logvar = self.encode_params(embeddings)
        std = torch.exp(0.5 * logvar)

        samples: list[torch.Tensor] = []

        for _ in range(num_samples):
            eps = torch.randn_like(std)
            samples.append(mu + eps * std)

        if num_samples == 1:
            return samples[0]

        return torch.stack(samples, dim=0)

    @torch.no_grad()
    def transform(self, embeddings: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`encode`.

        Returns deterministic latent means ``mu`` of shape
        ``(n, latent_dim)``.
        """
        return self.encode(embeddings)

    @torch.no_grad()
    def reconstruct(
        self,
        embeddings: np.ndarray | torch.Tensor,
        *,
        sample: bool = False,
    ) -> torch.Tensor:
        """Return dense reconstructions for ``embeddings``.

        By default this decodes ``mu`` instead of a sampled ``z`` because
        deterministic reconstructions are more useful for evaluation.
        """
        if self.vae is None:
            raise RuntimeError("trainer must be fitted or built before reconstruct")

        dataset = self._dataset(embeddings, shuffle=False)
        self.vae.eval()

        reconstructions: list[torch.Tensor] = []

        for batch in self._progress(dataset, total=len(dataset)):
            reconstruction, _z, _mu, _logvar, _stats = self.vae(batch, sample=sample)
            reconstructions.append(reconstruction.detach().cpu())

        return torch.cat(reconstructions, dim=0)

    @torch.no_grad()
    def decode(self, latents: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Decode latent vectors back to the original embedding space."""
        if self.vae is None:
            raise RuntimeError("trainer must be fitted or built before decode")

        z = torch.as_tensor(latents)

        if z.ndim != 2:
            raise ValueError(f"latents must be 2D, got shape {tuple(z.shape)}")

        if z.shape[1] != int(self.cfg.latent_dim):
            raise ValueError(
                f"latents second dimension must be latent_dim={self.cfg.latent_dim}, "
                f"got {z.shape[1]}"
            )

        self.vae.eval()

        module = self.vae
        if hasattr(module, "_orig_mod"):
            module = module._orig_mod  # type: ignore[attr-defined]

        outputs: list[torch.Tensor] = []
        batch_size = int(self.cfg.batch_size)

        for start in self._progress(range(0, z.shape[0], batch_size)):
            end = min(start + batch_size, z.shape[0])
            batch_z = z[start:end].to(self.device)
            reconstruction = module.decode(batch_z)  # type: ignore[attr-defined]
            outputs.append(reconstruction.detach().cpu())

        return torch.cat(outputs, dim=0)

    def fit_transform(self, embeddings: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Fit the VAE and return deterministic latent means ``mu``."""
        self.fit(embeddings)
        return self.transform(embeddings)

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        """Return a saveable trainer state dictionary."""
        if self.vae is None:
            raise RuntimeError("trainer must be built before state_dict")

        return {
            "config": self.cfg,
            "input_dim": self.input_dim,
            "model": self.vae.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "history": list(self.history),
            "epoch": self._epoch,
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
        *,
        load_optimizer: bool = True,
    ) -> "BetaVAETrainer":  # type: ignore[override]
        """Load a state dictionary produced by :meth:`state_dict`."""
        saved_config = state.get("config")
        if saved_config is not None:
            if not isinstance(saved_config, BetaVAEConfig):
                raise TypeError("state config must be a BetaVAEConfig")
            if self.is_built and self.cfg != saved_config:
                raise ValueError("cannot load a different configuration into an already built trainer")
            self.cfg = saved_config
            self.device = torch.device(self.cfg.device)

        input_dim = int(state["input_dim"])
        self.build(input_dim)

        if self.vae is None:
            raise RuntimeError("failed to build model")

        self.vae.load_state_dict(state["model"])

        if load_optimizer and state.get("optimizer") is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])

        self.history = list(state.get("history", []))
        self._epoch = int(state.get("epoch", 0))

        return self
