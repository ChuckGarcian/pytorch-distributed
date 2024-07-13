import os
import subprocess
import time



WORLD_RANK = int(os.environ["SLURM_PROCID"])
print("World Rank: {}".format(WORLD_RANK))
cmd = ["srun", "--ntasks=1", "python", "worker_script.py"]
subprocess.run(cmd)

exit ()