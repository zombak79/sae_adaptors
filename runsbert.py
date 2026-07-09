"""Embed and evaluate every SentenceTransformer configured for an experiment."""

from utils import *


config = load_experiment_config()
checkpoints = build_checkpoints(config.CONFIG, config.CHECKPOINT_PATH_PREFIX)

if not hasattr(config, "SBERT_MODELS"):
    raise ValueError("config file must define SBERT_MODELS for runsbert.py")

for model_config in config.SBERT_MODELS:
    if isinstance(model_config, str):
        model_name = model_config
        max_seq_length = None
    else:
        model_name = model_config["name"]
        max_seq_length = model_config.get("max_seq_length")

    print(model_name)
    model = SentenceTransformer(model_name, device=config.DEVICE)
    if max_seq_length is not None:
        model.max_seq_length = int(max_seq_length)

    for dataset, checkpoint in checkpoints.items():
        print(dataset, end=": ")
        print(train_and_eval_sbert(checkpoint, model_name, model, device=config.DEVICE))
