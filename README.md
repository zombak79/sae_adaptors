# Embedding Adaptors for Cold-Start Recommendation

This repository accompanies an anonymous research submission. It evaluates
embedding adaptors for cold-start recommendation: item-text embeddings are
learned with SentenceTransformers and then transformed with sparse or dense
autoencoders before retrieval evaluation.

Implemented adaptors include a Top-K sparse autoencoder (SAE), denoising SAE,
denoising autoencoder (DAE), and beta-VAE.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project uses the published `compresso-pytorch` and `compresso-recsys`
packages for sparse representations, dataset preparation, checkpoint storage,
and retrieval evaluation. PyTorch device support depends on the installed
PyTorch build; configure the desired device in an experiment configuration.

## Quick sanity run

`config-office.py` is a small-scale configuration intended to check that the
end-to-end workflow runs. It uses a single Office Products dataset and a small
SentenceTransformer model.

```bash
python runsbert.py --config config-office.py
python rundae.py --config config-office.py
```

The first command builds the checkpoint if needed, creates item-text
embeddings, and evaluates the embedding baseline. The second command finds the
saved embedding stage, tunes the DAE on training items, and evaluates the
selected model on the test split.

## Experiment configuration

Experiments are controlled by Python configuration files. `config.py` is the
default configuration; copy it to create an independent experiment, set a new
`CHECKPOINT_PATH_PREFIX`, and pass the copy via `--config`.

Each configuration must define:

- `CONFIG`: dataset and split settings.
- `CHECKPOINT_PATH_PREFIX`: directory used for generated checkpoint archives.
- `DEVICE`: PyTorch/SentenceTransformers device, for example `"cuda:0"`,
  `"mps"`, or `"cpu"`.
- `SBERT_MODELS`: baseline models to evaluate. Entries may be a model name or
  a dictionary with `name` and optional `max_seq_length`.

It may also define one or more model grids: `sae_config`, `dsae_config`,
`dae_config`, and `vae_config`. Every list-valued setting in a grid is expanded
as a Cartesian product for validation-set model selection. `DEVICE` is applied
by the runners, so it should not be repeated within a grid.

Example:

```python
CHECKPOINT_PATH_PREFIX = "artifacts/my-experiment"
DEVICE = "mps"

SBERT_MODELS = [
    {"name": "sentence-transformers/all-MiniLM-L6-v2"},
]

dae_config = {
    "latent_dim": [128, 256],
    "epochs": [50],
    "lr": [1e-3],
    "decay": [False],
}
```

## Runners

Run the embedding baseline before an adaptor. Each adaptor runner operates on
every `sbert:*` stage already stored in the configured checkpoints.

| Command | Purpose |
| --- | --- |
| `python runsbert.py --config CONFIG.py` | Build checkpoints, embed item text, and evaluate each configured SentenceTransformer. |
| `python runsae.py --config CONFIG.py` | Tune and evaluate the Top-K sparse autoencoder. |
| `python rundsae.py --config CONFIG.py` | Tune and evaluate the denoising Top-K sparse autoencoder. |
| `python rundae.py --config CONFIG.py` | Tune and evaluate the denoising autoencoder. |
| `python runvae.py --config CONFIG.py` | Tune and evaluate the beta-VAE. |
| `python gather_stats.py --config CONFIG.py` | Print dataset-level checkpoint statistics. |
| `python analyze_results.py --config CONFIG.py [--output results.csv]` | Print completed stage metrics and optionally save them as CSV. |

Hyperparameter candidates are fit only on training items. Validation retrieval
selects the configuration; the selected configuration is then refit on the
training items and evaluated on the held-out test split.

## Outputs

Each dataset produces a checkpoint archive under `CHECKPOINT_PATH_PREFIX`.
Stages stored inside an archive include:

- `sbert:*`: dense item embeddings and baseline retrieval metrics.
- `sae:*` / `dsae:*`: trained sparse autoencoder state, sparse embeddings, and
  retrieval metrics.
- `dae:*` / `vae:*`: trained model state, dense adapted embeddings, and
  retrieval metrics.

The checkpoint manifest records stage metadata and metrics. Existing completed
stages are skipped, so commands can be rerun without recomputing prior work.

## Reproducing reported results

The final camera-ready artifact should provide the paper-specific configuration
files and corresponding runner commands in this section. Dataset downloads and
use are subject to the terms of their respective sources.
