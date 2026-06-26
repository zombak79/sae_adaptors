from utils import *

checkpoints=build_checkpoints(CONFIG)

print(gather_stats(checkpoints))
