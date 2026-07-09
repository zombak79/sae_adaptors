from utils import *

config = load_experiment_config()
checkpoints = build_checkpoints(config.CONFIG, config.CHECKPOINT_PATH_PREFIX)

print(gather_stats(checkpoints))
