## PyTorch Distributed Applications

These are test distributed applications I have written for fun during my time at Princeton University.

### Makefile Arguments

You can run one of the demo scripts in `src` by simply calling `make`. Optional arguments include:

- `SLURM`: Specifies the SLURM script file. The default is `single.slurm`.
- `MAIN`: Specifies the Python script to run. The default is `main.py`.

For example:
  `make MAIN=dist_demo.py`

