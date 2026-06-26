"""Training utilities for residual-quantized VAE semantic IDs.

This module provides a small sklearn-like API around a residual-quantized
autoencoder. It is intended for dense embedding matrices that already fit in
memory, similarly to TopKSAETrainer.

Example
-------
>>> cfg = RQVAEConfig(
...     latent_dim=256,
...     num_quantizers=4,
...     codebook_size=256,
...     epochs=100,
... )
>>> trainer = RQVAETrainer(cfg)
>>> semantic_ids = trainer.fit_transform(embeddings)  # shape: (n, 4)
>>> recon = trainer.reconstruct(embeddings)
>>> recon_from_ids = trainer.decode(semantic_ids)
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
    "RQCodes",
    "RQVAEConfig",
    "VectorQuantizer",
    "ResidualQuantizer",
    "RQVAE",
    "RQVAETrainer",
]


class EmbeddingsDataset:
    """Small batch-oriented dataset for in-memory embedding matrices.

    Unlike ``torch.utils.data.Dataset``, ``__getitem__`` returns a complete
    batch, not one sample.
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


@dataclass(frozen=True)
class RQCodes:
    """Container for semantic IDs produced by RQ-VAE.

    Attributes
    ----------
    codes:
        Long tensor of shape ``(n, num_quantizers)``.
    codebook_size:
        Number of entries per codebook.
    """

    codes: torch.Tensor
    codebook_size: int

    def __post_init__(self) -> None:
        if self.codes.ndim != 2:
            raise ValueError(f"codes must be 2D, got shape {tuple(self.codes.shape)}")
        if self.codes.dtype != torch.long:
            object.__setattr__(self, "codes", self.codes.long())

    @property
    def shape(self) -> torch.Size:
        return self.codes.shape

    @property
    def num_quantizers(self) -> int:
        return int(self.codes.shape[1])

    def numpy(self) -> np.ndarray:
        return self.codes.detach().cpu().numpy()

    def to(self, device: str | torch.device) -> "RQCodes":
        return RQCodes(self.codes.to(device), codebook_size=self.codebook_size)


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
class RQVAEConfig:
    """Configuration for :class:`RQVAETrainer`.

    Parameters
    ----------
    latent_dim:
        Dimensionality of the continuous latent vector before quantization.

    num_quantizers:
        Number of residual codebooks. This is the length of the semantic ID.

    codebook_size:
        Number of entries in each codebook. A common choice for semantic IDs is
        256 because each code fits in one byte.

    encoder_hidden_dims, decoder_hidden_dims:
        Optional MLP hidden layers. Empty tuples produce simple linear
        encoder/decoder projections.

    activation:
        Activation used inside encoder/decoder MLPs.

    distance:
        Distance used for nearest-code lookup. ``"l2"`` is the standard choice.
        ``"cosine"`` can be useful for normalized semantic embeddings.

    input_normalize:
        Optional module applied to input embeddings before the encoder. For
        semantic embeddings, ``L2Normalize()`` is often reasonable.

    latent_normalize:
        Optional module applied to encoder output before quantization.

    output_normalize:
        Optional module applied to decoder output. Usually leave as ``None``.

    commitment_beta:
        Weight for the encoder commitment loss.

    codebook_loss_weight:
        Weight for the codebook loss.

    diversity_loss_weight:
        Optional batch-level code usage balancing loss. This softly encourages
        uniform code usage and can reduce collapse. Start with ``0.0`` or a very
        small value like ``1e-3``.

    diversity_temperature:
        Softmax temperature for diversity loss.

    alpha_loss:
        Mixture weight for cosine reconstruction loss. Training reconstruction
        loss is:
        ``alpha_loss * (1 - cosine_similarity) + (1 - alpha_loss) * mse``.

    batch_size, shuffle, seed, epochs, lr, weight_decay, decay, compile, device:
        Training parameters similar to ``TopKSAEConfig``.
    """

    latent_dim: int = 256
    num_quantizers: int = 4
    codebook_size: int = 256

    encoder_hidden_dims: tuple[int, ...] = ()
    decoder_hidden_dims: tuple[int, ...] = ()
    activation: Literal["relu", "gelu", "silu", "tanh", "identity"] = "gelu"

    distance: Literal["l2", "cosine"] = "l2"

    input_normalize: nn.Module | None = None
    latent_normalize: nn.Module | None = None
    output_normalize: nn.Module | None = None

    commitment_beta: float = 0.25
    codebook_loss_weight: float = 1.0
    diversity_loss_weight: float = 0.0
    diversity_temperature: float = 1.0

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


class VectorQuantizer(nn.Module):
    """Single vector-quantization codebook."""

    def __init__(
        self,
        *,
        codebook_size: int,
        dim: int,
        distance: Literal["l2", "cosine"] = "l2",
        diversity_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        if codebook_size < 2:
            raise ValueError("codebook_size must be >= 2")
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if diversity_temperature <= 0:
            raise ValueError("diversity_temperature must be > 0")

        self.codebook_size = int(codebook_size)
        self.dim = int(dim)
        self.distance = distance
        self.diversity_temperature = float(diversity_temperature)

        self.embedding = nn.Embedding(self.codebook_size, self.dim)

        # Conservative initialization. RQ stacks several quantizers, so too-large
        # code vectors can destabilize the early residuals.
        nn.init.uniform_(
            self.embedding.weight,
            -1.0 / self.codebook_size,
            1.0 / self.codebook_size,
        )

    def _distances(self, x: torch.Tensor) -> torch.Tensor:
        """Return distance-like scores of shape ``(batch, codebook_size)``."""
        weight = self.embedding.weight

        if self.distance == "l2":
            # ||x - e||^2 = ||x||^2 + ||e||^2 - 2 x.e
            x2 = x.pow(2).sum(dim=-1, keepdim=True)
            e2 = weight.pow(2).sum(dim=-1).unsqueeze(0)
            xe = x @ weight.t()
            return x2 + e2 - 2.0 * xe

        if self.distance == "cosine":
            x_norm = F.normalize(x, p=2.0, dim=-1)
            w_norm = F.normalize(weight, p=2.0, dim=-1)
            # Convert similarity to distance-like value.
            return 1.0 - x_norm @ w_norm.t()

        raise ValueError(f"unknown distance: {self.distance}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Quantize ``x``.

        Returns
        -------
        quantized:
            Selected code vectors, shape ``(batch, dim)``.
        indices:
            Selected code IDs, shape ``(batch,)``.
        stats:
            Scalar tensors with code usage information and soft probabilities.
        """
        distances = self._distances(x)
        indices = torch.argmin(distances, dim=-1)
        quantized = self.embedding(indices)

        with torch.no_grad():
            counts = torch.bincount(indices, minlength=self.codebook_size).float()
            probs = counts / counts.sum().clamp_min(1.0)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
            perplexity = entropy.exp()
            active_codes = (counts > 0).sum().float()
            dead_codes = self.codebook_size - active_codes

        soft_probs = F.softmax(-distances / self.diversity_temperature, dim=-1)
        avg_soft_probs = soft_probs.mean(dim=0)
        diversity_loss = (
            avg_soft_probs
            * (avg_soft_probs.clamp_min(1e-12) * self.codebook_size).log()
        ).sum()

        stats = {
            "perplexity": perplexity,
            "active_codes": active_codes,
            "dead_codes": dead_codes,
            "diversity_loss": diversity_loss,
        }

        return quantized, indices, stats

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices.long())


class ResidualQuantizer(nn.Module):
    """Stack of residual vector quantizers."""

    def __init__(
        self,
        *,
        num_quantizers: int,
        codebook_size: int,
        dim: int,
        distance: Literal["l2", "cosine"] = "l2",
        commitment_beta: float = 0.25,
        codebook_loss_weight: float = 1.0,
        diversity_loss_weight: float = 0.0,
        diversity_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        if num_quantizers < 1:
            raise ValueError("num_quantizers must be >= 1")
        if commitment_beta < 0:
            raise ValueError("commitment_beta must be >= 0")
        if codebook_loss_weight < 0:
            raise ValueError("codebook_loss_weight must be >= 0")
        if diversity_loss_weight < 0:
            raise ValueError("diversity_loss_weight must be >= 0")

        self.num_quantizers = int(num_quantizers)
        self.codebook_size = int(codebook_size)
        self.dim = int(dim)
        self.commitment_beta = float(commitment_beta)
        self.codebook_loss_weight = float(codebook_loss_weight)
        self.diversity_loss_weight = float(diversity_loss_weight)

        self.codebooks = nn.ModuleList(
            [
                VectorQuantizer(
                    codebook_size=codebook_size,
                    dim=dim,
                    distance=distance,
                    diversity_temperature=diversity_temperature,
                )
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Residual-quantize latent vectors.

        Returns
        -------
        z_q_st:
            Straight-through quantized latent vectors, shape ``(batch, dim)``.
        codes:
            Long tensor of shape ``(batch, num_quantizers)``.
        stats:
            Scalar losses and usage metrics.
        """
        residual = z
        quantized_sum = torch.zeros_like(z)
        all_indices: list[torch.Tensor] = []

        codebook_loss = z.new_tensor(0.0)
        diversity_loss = z.new_tensor(0.0)

        perplexities: list[torch.Tensor] = []
        active_codes: list[torch.Tensor] = []
        dead_codes: list[torch.Tensor] = []

        for codebook in self.codebooks:
            residual_before = residual
            quantized, indices, stats = codebook(residual_before)

            all_indices.append(indices)
            quantized_sum = quantized_sum + quantized

            # Codebook vectors are pulled toward the current residual.
            codebook_loss = codebook_loss + F.mse_loss(quantized, residual_before.detach())

            # Later codebooks quantize the remaining residual.
            # Detaching the selected vector keeps residual routing stable.
            residual = residual_before - quantized.detach()

            diversity_loss = diversity_loss + stats["diversity_loss"]
            perplexities.append(stats["perplexity"])
            active_codes.append(stats["active_codes"])
            dead_codes.append(stats["dead_codes"])

        codes = torch.stack(all_indices, dim=-1)

        # Encoder commitment loss: make z commit to the sum of selected codes.
        commitment_loss = F.mse_loss(z, quantized_sum.detach())

        quantizer_loss = (
            self.commitment_beta * commitment_loss
            + self.codebook_loss_weight * codebook_loss
            + self.diversity_loss_weight * diversity_loss
        )

        # Straight-through estimator.
        z_q_st = z + (quantized_sum - z).detach()

        stats = {
            "quantizer_loss": quantizer_loss,
            "commitment_loss": commitment_loss.detach(),
            "codebook_loss": codebook_loss.detach(),
            "diversity_loss": diversity_loss.detach(),
            "perplexity": torch.stack(perplexities).mean().detach(),
            "active_codes": torch.stack(active_codes).mean().detach(),
            "dead_codes": torch.stack(dead_codes).mean().detach(),
        }

        return z_q_st, codes, stats

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode semantic IDs into quantized latent vectors."""
        if codes.ndim != 2:
            raise ValueError(f"codes must be 2D, got shape {tuple(codes.shape)}")
        if codes.shape[1] != self.num_quantizers:
            raise ValueError(
                f"codes second dimension must be num_quantizers={self.num_quantizers}, "
                f"got {codes.shape[1]}"
            )

        codes = codes.long()
        z_q = None

        for level, codebook in enumerate(self.codebooks):
            part = codebook.decode(codes[:, level])
            z_q = part if z_q is None else z_q + part

        assert z_q is not None
        return z_q


class RQVAE(nn.Module):
    """Residual-quantized autoencoder for semantic IDs."""

    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int = 256,
        num_quantizers: int = 4,
        codebook_size: int = 256,
        encoder_hidden_dims: Sequence[int] = (),
        decoder_hidden_dims: Sequence[int] = (),
        activation: Literal["relu", "gelu", "silu", "tanh", "identity"] = "gelu",
        distance: Literal["l2", "cosine"] = "l2",
        input_normalize: nn.Module | None = None,
        latent_normalize: nn.Module | None = None,
        output_normalize: nn.Module | None = None,
        commitment_beta: float = 0.25,
        codebook_loss_weight: float = 1.0,
        diversity_loss_weight: float = 0.0,
        diversity_temperature: float = 1.0,
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
        self.num_quantizers = int(num_quantizers)
        self.codebook_size = int(codebook_size)

        self.input_normalize = input_normalize
        self.latent_normalize = latent_normalize
        self.output_normalize = output_normalize

        if encoder is None:
            enc_dims = [input_dim, *encoder_hidden_dims, latent_dim]
            self.encoder = _make_mlp(enc_dims, activation=activation)
        else:
            self.encoder = encoder

        if decoder is None:
            dec_dims = [latent_dim, *decoder_hidden_dims, input_dim]
            self.decoder = _make_mlp(dec_dims, activation=activation)
        else:
            self.decoder = decoder

        self.quantizer = ResidualQuantizer(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            dim=latent_dim,
            distance=distance,
            commitment_beta=commitment_beta,
            codebook_loss_weight=codebook_loss_weight,
            diversity_loss_weight=diversity_loss_weight,
            diversity_temperature=diversity_temperature,
        )

    
    
    def encode_continuous(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_normalize is not None:
            x = self.input_normalize(x)

        z = self.encoder(x)

        if self.latent_normalize is not None:
            z = self.latent_normalize(z)

        return z

    def decode_latent(self, z_q: torch.Tensor) -> torch.Tensor:
        reconstruction = self.decoder(z_q)

        if self.output_normalize is not None:
            reconstruction = self.output_normalize(reconstruction)

        return reconstruction

    def encode_codes(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode_continuous(x)
        _z_q, codes, _stats = self.quantizer(z)
        return codes

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        z_q = self.quantizer.decode(codes)
        return self.decode_latent(z_q)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return reconstruction, semantic IDs, quantized latent, stats."""
        z = self.encode_continuous(x)
        z_q, codes, q_stats = self.quantizer(z)
        reconstruction = self.decode_latent(z_q)

        cosine_similarity = F.cosine_similarity(reconstruction, x, dim=-1).mean()
        reconstruction_mse = F.mse_loss(reconstruction, x)

        stats = {
            **q_stats,
            "cosine_similarity": cosine_similarity.detach(),
            "reconstruction_mse": reconstruction_mse.detach(),
        }

        return reconstruction, codes, z_q, stats


class RQVAETrainer:
    """Efficient fit/transform wrapper around :class:`RQVAE`.

    Example
    -------
    >>> trainer = RQVAETrainer(RQVAEConfig(num_quantizers=4, codebook_size=256))
    >>> trainer.fit(embeddings)
    >>> ids = trainer.transform(embeddings)       # RQCodes, shape (n, 4)
    >>> recon = trainer.reconstruct(embeddings)
    >>> recon2 = trainer.decode(ids)
    """

    def __init__(self, config: RQVAEConfig | None = None) -> None:
        self.cfg = config if config is not None else RQVAEConfig()
        self.device = torch.device(self.cfg.device)
        self.rqvae: RQVAE | nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.input_dim: int | None = None
        self.history: list[dict[str, float]] = []

    @property
    def is_built(self) -> bool:
        return self.rqvae is not None

    def build(self, input_dim: int) -> "RQVAETrainer":
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
        if self.cfg.num_quantizers < 1:
            raise ValueError("num_quantizers must be >= 1")
        if self.cfg.codebook_size < 2:
            raise ValueError("codebook_size must be >= 2")
        if not 0.0 <= self.cfg.alpha_loss <= 1.0:
            raise ValueError("alpha_loss must be in [0, 1]")

        torch.manual_seed(int(self.cfg.seed))

        self.input_dim = int(input_dim)

        model = RQVAE(
            input_dim=self.input_dim,
            latent_dim=int(self.cfg.latent_dim),
            num_quantizers=int(self.cfg.num_quantizers),
            codebook_size=int(self.cfg.codebook_size),
            encoder_hidden_dims=self.cfg.encoder_hidden_dims,
            decoder_hidden_dims=self.cfg.decoder_hidden_dims,
            activation=self.cfg.activation,
            distance=self.cfg.distance,
            input_normalize=self.cfg.input_normalize,
            latent_normalize=self.cfg.latent_normalize,
            output_normalize=self.cfg.output_normalize,
            commitment_beta=float(self.cfg.commitment_beta),
            codebook_loss_weight=float(self.cfg.codebook_loss_weight),
            diversity_loss_weight=float(self.cfg.diversity_loss_weight),
            diversity_temperature=float(self.cfg.diversity_temperature),
        ).to(self.device)

        if self.cfg.compile:
            model = torch.compile(model)  # type: ignore[assignment]

        self.rqvae = model
        self.optimizer = torch.optim.AdamW(
            self.rqvae.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

        return self

    def to(self, device: str | torch.device) -> "RQVAETrainer":
        self.device = torch.device(device)
        if self.rqvae is not None:
            self.rqvae.to(self.device)
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

    def train_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.rqvae is None or self.optimizer is None:
            raise RuntimeError("trainer must be built before train_step")

        self.rqvae.train()
        self.optimizer.zero_grad(set_to_none=True)

        reconstruction, _codes, _z_q, stats = self.rqvae(batch)

        cosine_loss = 1.0 - F.cosine_similarity(reconstruction, batch, dim=-1).mean()
        mse = F.mse_loss(reconstruction, batch)

        reconstruction_loss = (
            float(self.cfg.alpha_loss) * cosine_loss
            + (1.0 - float(self.cfg.alpha_loss)) * mse
        )

        loss = reconstruction_loss + stats["quantizer_loss"]

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.detach(),
            "reconstruction_loss": reconstruction_loss.detach(),
            "cosine_loss": cosine_loss.detach(),
            "reconstruction_mse": mse.detach(),
            "quantizer_loss": stats["quantizer_loss"].detach(),
            "commitment_loss": stats["commitment_loss"].detach(),
            "codebook_loss": stats["codebook_loss"].detach(),
            "diversity_loss": stats["diversity_loss"].detach(),
            "perplexity": stats["perplexity"].detach(),
            "active_codes": stats["active_codes"].detach(),
            "dead_codes": stats["dead_codes"].detach(),
        }

    def fit(self, embeddings: np.ndarray | torch.Tensor) -> "RQVAETrainer":
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
                        "mse": f"{record['reconstruction_mse']:.4E}",
                        "q": f"{record['quantizer_loss']:.4f}",
                        "ppl": f"{record['perplexity']:.1f}",
                        "lr": f"{record['lr']:.2E}",
                    }
                )

            if scheduler is not None:
                scheduler.step()

        return self

    @torch.no_grad()
    def encode(self, embeddings: np.ndarray | torch.Tensor) -> RQCodes:
        """Return semantic IDs of shape ``(n, num_quantizers)``."""
        if self.rqvae is None:
            raise RuntimeError("trainer must be fitted or built before encode")

        dataset = self._dataset(embeddings, shuffle=False)
        self.rqvae.eval()

        all_codes: list[torch.Tensor] = []

        for batch in self._progress(dataset, total=len(dataset)):
            # Works for normal RQVAE. If torch.compile wraps methods awkwardly,
            # falling back to forward is safer.
            _reconstruction, codes, _z_q, _stats = self.rqvae(batch)
            all_codes.append(codes.detach().cpu())

        return RQCodes(
            torch.cat(all_codes, dim=0),
            codebook_size=int(self.cfg.codebook_size),
        )

    @torch.no_grad()
    def transform(self, embeddings: np.ndarray | torch.Tensor) -> RQCodes:
        """Alias for :meth:`encode`."""
        return self.encode(embeddings)

    @torch.no_grad()
    def reconstruct(self, embeddings: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return dense reconstructions for ``embeddings``."""
        if self.rqvae is None:
            raise RuntimeError("trainer must be fitted or built before reconstruct")

        dataset = self._dataset(embeddings, shuffle=False)
        self.rqvae.eval()

        reconstructions: list[torch.Tensor] = []

        for batch in self._progress(dataset, total=len(dataset)):
            reconstruction, _codes, _z_q, _stats = self.rqvae(batch)
            reconstructions.append(reconstruction.detach().cpu())

        return torch.cat(reconstructions, dim=0)

    @torch.no_grad()
    def decode(self, codes: RQCodes | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Decode semantic IDs back to dense reconstructed vectors."""
        if self.rqvae is None:
            raise RuntimeError("trainer must be fitted or built before decode")

        if isinstance(codes, RQCodes):
            code_tensor = codes.codes
        else:
            code_tensor = torch.as_tensor(codes)

        if code_tensor.ndim != 2:
            raise ValueError(f"codes must be 2D, got shape {tuple(code_tensor.shape)}")

        if code_tensor.shape[1] != int(self.cfg.num_quantizers):
            raise ValueError(
                f"codes second dimension must be num_quantizers={self.cfg.num_quantizers}, "
                f"got {code_tensor.shape[1]}"
            )

        self.rqvae.eval()

        outputs: list[torch.Tensor] = []
        code_tensor = code_tensor.long()

        batch_size = int(self.cfg.batch_size)

        for start in self._progress(range(0, code_tensor.shape[0], batch_size)):
            end = min(start + batch_size, code_tensor.shape[0])
            batch_codes = code_tensor[start:end].to(self.device)

            # Accessing decode_codes is fine on the raw module. For compiled
            # modules, _orig_mod is usually available.
            module = self.rqvae
            if hasattr(module, "_orig_mod"):
                module = module._orig_mod  # type: ignore[attr-defined]

            reconstruction = module.decode_codes(batch_codes)  # type: ignore[attr-defined]
            outputs.append(reconstruction.detach().cpu())

        return torch.cat(outputs, dim=0)

    def fit_transform(self, embeddings: np.ndarray | torch.Tensor) -> RQCodes:
        self.fit(embeddings)
        return self.transform(embeddings)

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        if self.rqvae is None:
            raise RuntimeError("trainer must be built before state_dict")

        return {
            "config": self.cfg,
            "input_dim": self.input_dim,
            "model": self.rqvae.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "history": list(self.history),
        }

    def load_state_dict(self, state: dict[str, Any], *, load_optimizer: bool = True) -> "RQVAETrainer":  # type: ignore[override]
        input_dim = int(state["input_dim"])
        self.build(input_dim)

        if self.rqvae is None:
            raise RuntimeError("failed to build model")

        self.rqvae.load_state_dict(state["model"])

        if load_optimizer and state.get("optimizer") is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])

        self.history = list(state.get("history", []))
        return self

    @torch.no_grad()
    def encode_latent(
        self,
        embeddings: np.ndarray | torch.Tensor,
        *,
        quantized: bool = False,
    ) -> torch.Tensor:
        """Return dense latent representations.
    
        Parameters
        ----------
        embeddings:
            Input dense vectors of shape ``(n, input_dim)``.
        quantized:
            If ``False``, return continuous encoder latents ``z``.
            If ``True``, return residual-quantized latents ``z_q``.
    
        Returns
        -------
        torch.Tensor
            Dense latent tensor of shape ``(n, latent_dim)``.
        """
        if self.rqvae is None:
            raise RuntimeError("trainer must be fitted or built before encode_latent")
    
        dataset = self._dataset(embeddings, shuffle=False)
        self.rqvae.eval()
    
        latents: list[torch.Tensor] = []

        # torch.compile can wrap custom methods. This keeps things robust.
        module = self.rqvae
        if hasattr(module, "_orig_mod"):
            module = module._orig_mod  # type: ignore[attr-defined]
    
        for batch in self._progress(dataset, total=len(dataset)):
            z = module.encode_continuous(batch)  # type: ignore[attr-defined]
    
            if quantized:
                z_q, _codes, _stats = module.quantizer(z)  # type: ignore[attr-defined]
                latents.append(z_q.detach().cpu())
            else:
                latents.append(z.detach().cpu())
    
        return torch.cat(latents, dim=0)
    