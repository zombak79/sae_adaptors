# Sparse Autoencoders are Unsupervised Semantic Domain Adaptors for Recommender Systems

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

## Reported results

The table below reports NDCG@20 (N@20) and Recall@20 (R@20) across three
pretrained embedding models. Higher values are better; bold values are the
best result for a dataset, embedding model, and metric.

<details>
<summary>Full results table (13 datasets × 5 methods)</summary>

| Dataset | Method | BGE N@20 | BGE R@20 | Qwen3 N@20 | Qwen3 R@20 | MiniLM-L6 N@20 | MiniLM-L6 R@20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MovieLens 20M | Original | 0.0176 | 0.0252 | 0.0135 | 0.0292 | 0.0110 | 0.0204 |
| MovieLens 20M | DAE | 0.0408 | 0.0640 | 0.0402 | 0.0664 | 0.0347 | 0.0579 |
| MovieLens 20M | β-VAE | 0.0613 | 0.0898 | 0.0480 | 0.0745 | 0.0453 | 0.0695 |
| MovieLens 20M | SAE | 0.0559 | 0.0801 | 0.0625 | 0.0893 | 0.0488 | 0.0794 |
| MovieLens 20M | D-SAE | **0.0619** | **0.0931** | **0.0696** | **0.0959** | **0.0737** | **0.1118** |
| GoodBooks10k | Original | 0.0376 | 0.0457 | 0.0598 | 0.0721 | 0.0197 | 0.0270 |
| GoodBooks10k | DAE | 0.0855 | 0.1021 | 0.0927 | 0.1102 | 0.0736 | 0.0909 |
| GoodBooks10k | β-VAE | 0.1040 | 0.1276 | 0.1268 | 0.1511 | 0.0960 | 0.1156 |
| GoodBooks10k | SAE | 0.1386 | 0.1674 | 0.1452 | 0.1714 | 0.1002 | **0.1321** |
| GoodBooks10k | D-SAE | **0.1492** | **0.1792** | **0.1498** | **0.1767** | **0.1020** | 0.1292 |
| Amazon Automotive | Original | 0.0041 | 0.0088 | 0.0029 | 0.0062 | 0.0046 | 0.0099 |
| Amazon Automotive | DAE | 0.0083 | 0.0169 | 0.0062 | 0.0126 | 0.0085 | 0.0166 |
| Amazon Automotive | β-VAE | 0.0119 | 0.0240 | 0.0092 | 0.0181 | **0.0112** | **0.0217** |
| Amazon Automotive | SAE | 0.0115 | 0.0234 | 0.0092 | 0.0179 | 0.0109 | 0.0209 |
| Amazon Automotive | D-SAE | **0.0119** | **0.0241** | **0.0097** | **0.0186** | 0.0108 | 0.0208 |
| Amazon Baby Products | Original | 0.0040 | 0.0098 | 0.0037 | 0.0091 | 0.0038 | 0.0099 |
| Amazon Baby Products | DAE | 0.0094 | 0.0212 | 0.0081 | 0.0181 | 0.0076 | 0.0176 |
| Amazon Baby Products | β-VAE | 0.0118 | 0.0268 | 0.0097 | 0.0220 | 0.0106 | 0.0234 |
| Amazon Baby Products | SAE | **0.0136** | 0.0297 | **0.0120** | **0.0266** | **0.0123** | **0.0279** |
| Amazon Baby Products | D-SAE | 0.0132 | **0.0305** | 0.0120 | 0.0262 | 0.0119 | 0.0276 |
| Amazon Beauty and Personal Care | Original | 0.0052 | 0.0107 | 0.0025 | 0.0055 | 0.0043 | 0.0093 |
| Amazon Beauty and Personal Care | DAE | 0.0094 | 0.0189 | 0.0061 | 0.0129 | 0.0093 | 0.0188 |
| Amazon Beauty and Personal Care | β-VAE | 0.0148 | 0.0295 | 0.0107 | 0.0216 | 0.0129 | 0.0260 |
| Amazon Beauty and Personal Care | SAE | **0.0154** | **0.0312** | **0.0114** | **0.0230** | 0.0128 | 0.0255 |
| Amazon Beauty and Personal Care | D-SAE | 0.0150 | 0.0304 | 0.0113 | 0.0228 | **0.0132** | **0.0264** |
| Amazon Clothing Shoes and Jewelry | Original | 0.0015 | 0.0031 | 0.0013 | 0.0026 | 0.0014 | 0.0030 |
| Amazon Clothing Shoes and Jewelry | DAE | 0.0039 | 0.0080 | 0.0034 | 0.0070 | 0.0040 | 0.0081 |
| Amazon Clothing Shoes and Jewelry | β-VAE | **0.0066** | **0.0131** | 0.0056 | 0.0110 | **0.0058** | **0.0116** |
| Amazon Clothing Shoes and Jewelry | SAE | 0.0065 | 0.0130 | **0.0061** | **0.0119** | 0.0054 | 0.0109 |
| Amazon Clothing Shoes and Jewelry | D-SAE | 0.0063 | 0.0127 | 0.0060 | 0.0117 | 0.0054 | 0.0107 |
| Amazon Electronics | Original | 0.0020 | 0.0044 | 0.0011 | 0.0024 | 0.0018 | 0.0039 |
| Amazon Electronics | DAE | 0.0050 | 0.0106 | 0.0031 | 0.0068 | 0.0041 | 0.0085 |
| Amazon Electronics | β-VAE | 0.0073 | 0.0152 | 0.0054 | 0.0112 | **0.0057** | **0.0117** |
| Amazon Electronics | SAE | **0.0081** | **0.0168** | **0.0058** | **0.0122** | 0.0054 | 0.0111 |
| Amazon Electronics | D-SAE | 0.0079 | 0.0162 | 0.0057 | 0.0120 | 0.0056 | 0.0116 |
| Amazon Grocery and Gourmet Food | Original | 0.0052 | 0.0111 | 0.0036 | 0.0078 | 0.0052 | 0.0112 |
| Amazon Grocery and Gourmet Food | DAE | 0.0109 | 0.0221 | 0.0098 | 0.0198 | 0.0114 | 0.0231 |
| Amazon Grocery and Gourmet Food | β-VAE | 0.0162 | 0.0326 | 0.0149 | 0.0298 | 0.0157 | 0.0314 |
| Amazon Grocery and Gourmet Food | SAE | **0.0171** | **0.0354** | 0.0158 | 0.0323 | 0.0159 | 0.0325 |
| Amazon Grocery and Gourmet Food | D-SAE | 0.0170 | 0.0348 | **0.0163** | **0.0330** | **0.0161** | **0.0330** |
| Amazon Health and Household | Original | 0.0040 | 0.0079 | 0.0022 | 0.0049 | 0.0041 | 0.0083 |
| Amazon Health and Household | DAE | 0.0089 | 0.0187 | 0.0063 | 0.0131 | 0.0083 | 0.0164 |
| Amazon Health and Household | β-VAE | 0.0130 | 0.0264 | 0.0101 | 0.0199 | 0.0111 | 0.0219 |
| Amazon Health and Household | SAE | 0.0132 | 0.0276 | 0.0124 | **0.0240** | **0.0121** | **0.0233** |
| Amazon Health and Household | D-SAE | **0.0138** | **0.0280** | **0.0129** | 0.0239 | 0.0115 | 0.0226 |
| Amazon Office Products | Original | 0.0193 | 0.0419 | 0.0176 | 0.0376 | 0.0187 | 0.0398 |
| Amazon Office Products | DAE | 0.0334 | 0.0671 | 0.0277 | 0.0558 | 0.0324 | 0.0644 |
| Amazon Office Products | β-VAE | 0.0422 | 0.0824 | 0.0366 | 0.0732 | 0.0394 | 0.0793 |
| Amazon Office Products | SAE | **0.0469** | 0.0892 | 0.0390 | 0.0781 | 0.0444 | 0.0886 |
| Amazon Office Products | D-SAE | 0.0469 | **0.0902** | **0.0416** | **0.0832** | **0.0456** | **0.0904** |
| Amazon Sports and Outdoors | Original | 0.0092 | 0.0213 | 0.0082 | 0.0181 | 0.0113 | 0.0244 |
| Amazon Sports and Outdoors | DAE | 0.0188 | 0.0393 | 0.0148 | 0.0322 | 0.0192 | 0.0418 |
| Amazon Sports and Outdoors | β-VAE | 0.0242 | 0.0523 | 0.0187 | 0.0415 | 0.0226 | 0.0475 |
| Amazon Sports and Outdoors | SAE | 0.0269 | 0.0573 | 0.0221 | 0.0475 | 0.0233 | 0.0507 |
| Amazon Sports and Outdoors | D-SAE | **0.0273** | **0.0578** | **0.0224** | **0.0482** | **0.0241** | **0.0519** |
| Amazon Toys and Games | Original | 0.0136 | 0.0267 | 0.0103 | 0.0203 | 0.0132 | 0.0252 |
| Amazon Toys and Games | DAE | 0.0233 | 0.0458 | 0.0215 | 0.0406 | 0.0232 | 0.0442 |
| Amazon Toys and Games | β-VAE | 0.0292 | 0.0550 | 0.0283 | 0.0528 | 0.0285 | 0.0542 |
| Amazon Toys and Games | SAE | 0.0319 | 0.0601 | 0.0309 | 0.0570 | 0.0299 | **0.0569** |
| Amazon Toys and Games | D-SAE | **0.0326** | **0.0611** | **0.0314** | **0.0581** | **0.0305** | 0.0569 |
| Amazon Video Games | Original | 0.0272 | 0.0534 | 0.0234 | 0.0455 | 0.0217 | 0.0467 |
| Amazon Video Games | DAE | 0.0444 | 0.0849 | 0.0401 | 0.0797 | 0.0355 | 0.0729 |
| Amazon Video Games | β-VAE | 0.0469 | 0.0911 | 0.0425 | 0.0857 | 0.0380 | 0.0759 |
| Amazon Video Games | SAE | 0.0531 | **0.1026** | 0.0469 | 0.0908 | 0.0418 | 0.0832 |
| Amazon Video Games | D-SAE | **0.0542** | 0.1018 | **0.0527** | **0.1014** | **0.0426** | **0.0839** |

</details>
