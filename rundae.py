from utils import *

config = load_experiment_config()
checkpoints = build_checkpoints(config.CONFIG, config.CHECKPOINT_PATH_PREFIX)

grid = build_grid(config.dae_config | {"device": config.DEVICE})
for checkpoint in checkpoints.values():
    print("*"*30)
    print(checkpoint)
    train_and_eval_dae(checkpoint["path"], grid)
