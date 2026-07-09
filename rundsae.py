from utils import *

config = load_experiment_config()
checkpoints = build_checkpoints(config.CONFIG, config.CHECKPOINT_PATH_PREFIX)

grid = build_grid(config.dsae_config | {"device": config.DEVICE})
for checkpoint in checkpoints.values():
    print("*"*30)
    print(checkpoint)
    train_and_eval_sae(checkpoint["path"], grid, prefix="dsae")
