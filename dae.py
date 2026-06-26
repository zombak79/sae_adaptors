"""Training utilities for a denoising autoencoder over dense embeddings.

The objects in this module provide a small, sklearn-like API around a simple
denoising autoencoder:

>>> trainer = DAETrainer(DAEConfig(latent_dim=256, epochs=100))
>>> dense_domain_embeddings = trainer.fit_transform(embeddings)

The trainer intentionally optimizes for dense embedding matrices that already
fit in memory. It avoids ``torch.utils.data.DataLoader`` overhead and uses a
simple batch dataset that returns full batches directly.
"""

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
    "DAEConfig",
    "DenoisingAutoencoder",
    "DAETrainer",
]


class EmbeddingsDataset:
    """Small batch-oriented dataset for in-memory embedding matrices.

    Unlike ``torch.utils.data.Dataset``, ``__getitem__`` returns a complete
    batch, not one sample. This mirrors Keras ``PyDataset`` ergonomics and keeps
    the training loop tight for matrix-shaped embedding data.

    Parameters
    ----------
    embeddings:
        A 2D ``numpy.ndarray`` or ``torch.Tensor`` with shape ``(n, dim)``.
    batch_size:
        Number of rows returned by each batch.
    shuffle:
        Whether to shuffle row order when ``on_epoch_end`` is called.
    seed:
        Seed for the NumPy row-order generator.
    device:
        Device where returned batches should live.
    dtype:
        Optional dtype conversion for returned batches. ``None`` preserves the
        dtype from the input tensor/array as much as possible.
    """

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
        """Return the number of batches."""
        return int(np.ceil(self.n / self.batch_size))

    def __iter__(self):
        for batch_idx in range(len(self)):
            yield self[batch_idx]

    def __getitem__(self, batch_idx: int) -> torch.Tensor:
        """Return batch ``batch_idx`` as a tensor on ``self.device``."""
        start = int(batch_idx) * self.batch_size
        end = min(start + self.batch_size, self.n)
        rows = self.indices[start:end]
        batch = self.embeddings[torch.as_tensor(rows, dtype=torch.long)]
        return batch.to(self.device, non_blocking=True)

    def to(self, device: str | torch.device) -> "EmbeddingsDataset":
        """Set output device for future batches and return ``self``."""
        device = torch.device(device)
        self.embeddings[:1].to(device)
        self.device = device
        return self

    def on_epoch_begin(self) -> None:
        """Hook called by ``DAETrainer.fit`` at the beginning of an epoch."""

    def on_epoch_end(self) -> None:
        """Shuffle row order after each epoch when ``shuffle=True``."""
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
    """Create a simple MLP from a list of dimensions."""
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
class DAEConfig:
    """Configuration for :class:`DAETrainer`.

    Parameters
    ----------
    latent_dim:
        Width of the dense bottleneck representation.

    encoder_hidden_dims, decoder_hidden_dims:
        Optional hidden dimensions for encoder/decoder MLPs. Empty tuples
        produce simple linear encoder and decoder layers.

    activation:
        Activation used inside encoder/decoder MLPs.

    input_normalize:
        Optional module applied to clean inputs before corruption and before
        evaluation. For semantic embeddings, ``L2Normalize()`` is often useful.

    latent_normalize:
        Optional module applied to latent representations.

    output_normalize:
        Optional module applied to decoder output. Usually leave as ``None``.
        If you evaluate with cosine similarity only, ``L2Normalize()`` can make
        sense.

    noise_type:
        Type of denoising corruption:
        - ``"none"``: plain autoencoder.
        - ``"gaussian"``: add Gaussian noise.
        - ``"dropout"``: randomly zero input dimensions.
        - ``"mask"``: randomly zero dimensions without inverted-dropout scaling.

    noise_std:
        Standard deviation for Gaussian corruption.

    dropout_p:
        Drop probability for dropout or mask corruption.

    alpha_loss:
        Mixture weight for cosine loss. Training reconstruction loss is
        ``alpha_loss * (1 - cosine_similarity) + (1 - alpha_loss) * mse``.

    latent_l1_penalty:
        Optional L1 penalty on latent activations. Keep this at ``0.0`` for a
        dense DAE. Use SAE for serious sparse experiments.

    batch_size:
        Number of embedding rows per training batch.

    shuffle:
        Whether to shuffle training rows between epochs.

    seed:
        Random seed used for row shuffling and Torch initialization.

    epochs:
        Number of training epochs.

    lr, weight_decay:
        AdamW optimizer parameters.

    decay:
        If ``True``, use cosine learning-rate decay from ``lr`` to zero across
        the configured training epochs.

    compile:
        If ``True``, call ``torch.compile`` on the DAE when available.

    device:
        Device used for training and transforms.

    show_progress:
        Whether to show a tqdm progress bar when tqdm is installed.
    """

    latent_dim: int = 256
    encoder_hidden_dims: tuple[int, ...] = ()
    decoder_hidden_dims: tuple[int, ...] = ()
    activation: Literal["relu", "gelu", "silu", "tanh", "identity"] = "gelu"

    input_normalize: nn.Module | None = None
    latent_normalize: nn.Module | None = None
    output_normalize: nn.Module | None = None

    noise_type: Literal["none", "gaussian", "dropout", "mask"] = "gaussian"
    noise_std: float = 0.03
    dropout_p: float = 0.1

    alpha_loss: float = 0.01
    latent_l1_penalty: float = 0.0

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


class DenoisingAutoencoder(nn.Module):
    """Simple dense denoising autoencoder.

    The forward pass expects already-corrupted inputs if denoising is desired.
    Corruption is handled in the trainer so that the model itself stays clean
    and usable for deterministic inference.
    """

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
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
    ) -> None:
        super().__init__()

        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)

        self.input_normalize = input_normalize
        self.latent_normalize = latent_normalize
        self.output_normalize = output_normalize

        if encoder is None:
            self.encoder = _make_mlp(
                [input_dim, *encoder_hidden_dims, latent_dim],
                activation=activation,
            )
        else:
            self.encoder = encoder

        if decoder is None:
            self.decoder = _make_mlp(
                [latent_dim, *decoder_hidden_dims, input_dim],
                activation=activation,
            )
        else:
            self.decoder = decoder

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return dense latent representation."""
        if self.input_normalize is not None:
            x = self.input_normalize(x)

        z = self.encoder(x)

        if self.latent_normalize is not None:
            z = self.latent_normalize(z)

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to input space."""
        reconstruction = self.decoder(z)

        if self.output_normalize is not None:
            reconstruction = self.output_normalize(reconstruction)

        return reconstruction

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return reconstruction, latent representation, and stats."""
        z = self.encode(x)
        reconstruction = self.decode(z)

        cosine_similarity = F.cosine_similarity(reconstruction, x, dim=-1).mean()
        reconstruction_mse = F.mse_loss(reconstruction, x)

        stats = {
            "cosine_similarity": cosine_similarity.detach(),
            "reconstruction_mse": reconstruction_mse.detach(),
            "latent_abs_mean": z.abs().mean().detach(),
            "latent_l2_mean": z.pow(2).sum(dim=-1).sqrt().mean().detach(),
        }

        return reconstruction, z, stats


class DAETrainer:
    """Efficient fit/transform wrapper around :class:`DenoisingAutoencoder`.

    The trainer is intended for dense embedding matrices, such as item
    embeddings from a recommender or semantic embeddings from a text encoder.

    API
    ---
    >>> trainer = DAETrainer(DAEConfig(latent_dim=256, epochs=100))
    >>> trainer.fit(embeddings)
    >>> z = trainer.transform(embeddings)
    >>> x_hat = trainer.reconstruct(embeddings)

    ``transform`` returns dense latent representations.
    """

    def __init__(self, config: DAEConfig | None = None) -> None:
        self.cfg = config if config is not None else DAEConfig()
        self.device = torch.device(self.cfg.device)
        self.dae: DenoisingAutoencoder | nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.input_dim: int | None = None
        self.history: list[dict[str, float]] = []

    @property
    def is_built(self) -> bool:
        """Whether the underlying model has been initialized."""
        return self.dae is not None

    def build(self, input_dim: int) -> "DAETrainer":
        """Initialize model and optimizer for inputs of size ``input_dim``."""
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
        if not 0.0 <= self.cfg.alpha_loss <= 1.0:
            raise ValueError("alpha_loss must be in [0, 1]")
        if self.cfg.noise_std < 0.0:
            raise ValueError("noise_std must be >= 0")
        if not 0.0 <= self.cfg.dropout_p < 1.0:
            raise ValueError("dropout_p must be in [0, 1)")
        if self.cfg.latent_l1_penalty < 0.0:
            raise ValueError("latent_l1_penalty must be >= 0")

        torch.manual_seed(int(self.cfg.seed))

        self.input_dim = int(input_dim)

        model = DenoisingAutoencoder(
            input_dim=self.input_dim,
            latent_dim=int(self.cfg.latent_dim),
            encoder_hidden_dims=self.cfg.encoder_hidden_dims,
            decoder_hidden_dims=self.cfg.decoder_hidden_dims,
            activation=self.cfg.activation,
            input_normalize=self.cfg.input_normalize,
            latent_normalize=self.cfg.latent_normalize,
            output_normalize=self.cfg.output_normalize,
        ).to(self.device)

        if self.cfg.compile:
            model = torch.compile(model)  # type: ignore[assignment]

        self.dae = model
        self.optimizer = torch.optim.AdamW(
            self.dae.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

        return self

    def to(self, device: str | torch.device) -> "DAETrainer":
        """Move the underlying model to ``device`` and return ``self``."""
        self.device = torch.device(device)

        if self.dae is not None:
            self.dae.to(self.device)

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
        except Exception:  # pragma: no cover - optional dependency fallback
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

    def _clean_input(self, batch: torch.Tensor) -> torch.Tensor:
        """Apply deterministic preprocessing to the clean reconstruction target."""
        if self.dae is None:
            raise RuntimeError("trainer must be built before preprocessing inputs")

        module = self.dae
        if hasattr(module, "_orig_mod"):
            module = module._orig_mod  # type: ignore[attr-defined]

        input_normalize = getattr(module, "input_normalize", None)

        if input_normalize is not None:
            return input_normalize(batch)

        return batch

    def _corrupt(self, batch: torch.Tensor) -> torch.Tensor:
        """Return corrupted input used by the denoising objective."""
        noise_type = self.cfg.noise_type

        if noise_type == "none":
            return batch

        if noise_type == "gaussian":
            if self.cfg.noise_std == 0.0:
                return batch
            return batch + float(self.cfg.noise_std) * torch.randn_like(batch)

        if noise_type == "dropout":
            if self.cfg.dropout_p == 0.0:
                return batch
            return F.dropout(batch, p=float(self.cfg.dropout_p), training=True)

        if noise_type == "mask":
            if self.cfg.dropout_p == 0.0:
                return batch
            keep = torch.rand_like(batch) >= float(self.cfg.dropout_p)
            return batch * keep.to(dtype=batch.dtype)

        raise ValueError(f"unknown noise_type: {noise_type}")

    def train_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run one optimization step and return detached training stats."""
        if self.dae is None or self.optimizer is None:
            raise RuntimeError("trainer must be built before train_step")

        self.dae.train()
        self.optimizer.zero_grad(set_to_none=True)

        clean = self._clean_input(batch)
        corrupted = self._corrupt(clean)

        reconstruction, latent, stats = self.dae(corrupted)

        cosine_loss = 1.0 - F.cosine_similarity(reconstruction, clean, dim=-1).mean()
        mse = F.mse_loss(reconstruction, clean)

        reconstruction_loss = (
            float(self.cfg.alpha_loss) * cosine_loss
            + (1.0 - float(self.cfg.alpha_loss)) * mse
        )

        loss = reconstruction_loss

        if self.cfg.latent_l1_penalty > 0.0:
            loss = loss + float(self.cfg.latent_l1_penalty) * latent.abs().mean()

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.detach(),
            "reconstruction_loss": reconstruction_loss.detach(),
            "cosine_loss": cosine_loss.detach(),
            "reconstruction_mse": mse.detach(),
            "latent_abs_mean": latent.abs().mean().detach(),
            "latent_l2_mean": latent.pow(2).sum(dim=-1).sqrt().mean().detach(),
            # These are measured on corrupted-forward stats, but loss is against clean.
            "corrupted_cosine_similarity": stats["cosine_similarity"].detach(),
            "corrupted_reconstruction_mse": stats["reconstruction_mse"].detach(),
        }

    def fit(self, embeddings: np.ndarray | torch.Tensor) -> "DAETrainer":
        """Train the DAE on dense embeddings and return ``self``."""
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
                        "cosine": f"{record['cosine_loss']:.4f}",
                        "mse": f"{record['reconstruction_mse']:.4E}",
                        "lr": f"{record['lr']:.2E}",
                    }
                )

            if scheduler is not None:
                scheduler.step()

        return self

    @torch.no_grad()
    def encode(self, embeddings: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return dense latent representations produced by the trained DAE."""
        if self.dae is None:
            raise RuntimeError("trainer must be fitted or built before encode")

        dataset = self._dataset(embeddings, shuffle=False)
        self.dae.eval()

        latents: list[torch.Tensor] = []

        module = self.dae
        if hasattr(module, "_orig_mod"):
            module = module._orig_mod  # type: ignore[attr-defined]

        for batch in self._progress(dataset, total=len(dataset)):
            z = module.encode(batch)  # type: ignore[attr-defined]
            latents.append(z.detach().cpu())

        return torch.cat(latents, dim=0)

    @torch.no_grad()
    def transform(self, embeddings: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`encode`.

        Returns dense domain-adapted latent embeddings of shape
        ``(n, latent_dim)``.
        """
        return self.encode(embeddings)

    @torch.no_grad()
    def reconstruct(self, embeddings: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return dense reconstructions for ``embeddings``."""
        if self.dae is None:
            raise RuntimeError("trainer must be fitted or built before reconstruct")

        dataset = self._dataset(embeddings, shuffle=False)
        self.dae.eval()

        reconstructions: list[torch.Tensor] = []

        # During inference, do not corrupt.
        for batch in self._progress(dataset, total=len(dataset)):
            reconstruction, _latent, _stats = self.dae(batch)
            reconstructions.append(reconstruction.detach().cpu())

        return torch.cat(reconstructions, dim=0)

    @torch.no_grad()
    def decode(self, latents: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Decode latent vectors back to the original embedding space."""
        if self.dae is None:
            raise RuntimeError("trainer must be fitted or built before decode")

        z = torch.as_tensor(latents)

        if z.ndim != 2:
            raise ValueError(f"latents must be 2D, got shape {tuple(z.shape)}")

        if z.shape[1] != int(self.cfg.latent_dim):
            raise ValueError(
                f"latents second dimension must be latent_dim={self.cfg.latent_dim}, "
                f"got {z.shape[1]}"
            )

        self.dae.eval()

        module = self.dae
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
        """Fit the DAE and return dense latent representations."""
        self.fit(embeddings)
        return self.transform(embeddings)

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        """Return a saveable trainer state dictionary."""
        if self.dae is None:
            raise RuntimeError("trainer must be built before state_dict")

        return {
            "config": self.cfg,
            "input_dim": self.input_dim,
            "model": self.dae.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "history": list(self.history),
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
        *,
        load_optimizer: bool = True,
    ) -> "DAETrainer":  # type: ignore[override]
        """Load a state dictionary produced by :meth:`state_dict`."""
        input_dim = int(state["input_dim"])
        self.build(input_dim)

        if self.dae is None:
            raise RuntimeError("failed to build model")

        self.dae.load_state_dict(state["model"])

        if load_optimizer and state.get("optimizer") is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])

        self.history = list(state.get("history", []))
        return self