# Slurm Parameters
NODES := 1                    # Node Count
NTASKS := 1                   # total number of tasks across all nodes
EMAIL := cg1509@princeton.edu # Output Email
NCORES := 2                   # cpu-cores per task (>1 if multi-threaded tasks)
MEM := 32G 										# memory per cpu-core (4G is default)
NGPU := 1                     # Number GPUS per node
JNAME := pytorch-dist

# Directories
OUTPUT := _output
SLURM_DIR := run2.slurm
MAIN := python3 example.py

# Colors 
BBlack='\033[1;30m'       # Black
BRed='\033[1;31m'         # Red
BGreen='\033[1;32m'       # Green
BYellow='\033[1;33m'      # Yellow
BBlue='\033[1;34m'        # Blue
BPurple='\033[1;35m'      # Purple
BCyan='\033[1;36m'        # Cyan
BWhite='\033[1;37m'       # White
NC='\033[0m' # No Color


all: clean
	@mkdir -p $(OUTPUT)
	
	$(eval SB_ARGS := --nodes=$(NODES) --ntasks=$(NTASKS) --mail-user=$(EMAIL) \
										--cpus-per-task=$(NCORES) --job-name=$(JNAME)        \
										--mem-per-cpu=$(MEM))
	$(eval SB_ARGS := --job-name=$(JNAME))
	$(eval SLURM_JOB_ID := $(shell sbatch --parsable $(SB_ARGS) $(SLURM_DIR)))
	$(eval FULL_OUTPUT := $(OUTPUT)/$(JNAME).$(SLURM_JOB_ID).out)
	@echo -e $(BBlue) Output file name: $(FULL_OUTPUT) $(NC)
	@echo -e $(BYellow) Waiting for Job to start... $(NC)
	@while [ ! -f "$(FULL_OUTPUT)" ]; do \
			sleep 1; \
	done
	@if command -v code >/dev/null 2>&1; then \
			code "$(FULL_OUTPUT)"; \
	else \
			echo -e $(BRed) Visual Studio Code not found. Output file is at: $(FULL_OUTPUT) $(NC); \
	fi
	@echo -e $(BYellow) Job Started. Opening Output file now. $(NC)
	@echo -e $(BBlue) Build Script Done! $(NC)

clean:
	@rm -rf $(OUTPUT)

.PHONY: all clean