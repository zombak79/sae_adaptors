"""Default experiment configuration.

Copy this file and pass the copy to a runner with ``--config`` to keep a
separate checkpoint namespace and set of experiment grids.
"""

from sae import L1Normalize


CHECKPOINT_PATH_PREFIX = "artifacts"
DEVICE = "cuda:0"

SBERT_MODELS = [
    {"name": "Qwen/Qwen3-Embedding-0.6B", "max_seq_length": 1024},
    {"name": "sentence-transformers/all-mpnet-base-v2"},
    {"name": "BAAI/bge-base-en-v1.5"},
]

AMAZON_CONFIG = {
    "--split_mode": "item_split",
    "--metadata_text_fields": "title,features,description,categories",
    "--min_entity_text_words": "20",
    "--min_user_support": "10",
    "--item_min_support": "10",
    "--min_value_to_keep": "1.0",
    "--set_all_values_to": "1.0",
    "--min_source_items": "1",
    "--min_target_items": "1",
    "--annotation_source": "none",
}

CONFIG = {
    category: AMAZON_CONFIG | {"dataset": "amazon2023", "amazon_category": category}
    for category in (
        "Toys_and_Games", "Video_Games", "Automotive", "Baby_Products",
        "Beauty_and_Personal_Care", "Clothing_Shoes_and_Jewelry",
        "Grocery_and_Gourmet_Food", "Health_and_Household", "Office_Products",
        "Sports_and_Outdoors", "Electronics",
    )
} | {
    "GoodBooks10k": {
        "dataset": "goodbooks", "--split_mode": "item_split", "--min_user_support": "5",
        "--item_min_support": "1", "--min_value_to_keep": "4.0",
        "--set_all_values_to": "1.0", "--min_entity_text_words": "20",
    },
    "MovieLens20M": {
        "dataset": "ml20m", "--split_mode": "item_split", "--min_user_support": "5",
        "--item_min_support": "1", "--min_value_to_keep": "4.0",
        "--set_all_values_to": "1.0", "--min_entity_text_words": "20",
    },
}

sae_config = {
    "epochs": [50, 100, 200], "decay": [True, False], "lr": [1e-3, 5e-3, 1e-2],
    "hidden_dim": 8192, "k": 128, "batch_size": 1024, "post_sparsify": L1Normalize(),
    "sparsify_score_mode": "abs", "sparsify_ste_alpha": 0.01,
}

dsae_config = sae_config | {"noise_type": "gaussian", "noise_std": 0.005}

dae_config = {
    "epochs": [50, 100, 200], "decay": [True, False], "lr": [1e-3, 5e-3, 1e-2],
    "latent_dim": 256,
}

vae_config = dae_config | {"beta_kl": 1e-6}
