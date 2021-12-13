#!/bin/sh
#SBATCH --account=inm71
#SBATCH --mail-user=l.frahm@fz-juelich.de
#SBATCH --mail-type=END
#SBATCH --job-name=sim_null
#SBATCH --time=2:00:00
#SBATCH --output=/p/scratch/cinm-7/pyALE_TFCE/logs/%j_null.out
#SBATCH --error=/p/scratch/cinm-7/pyALE_TFCE/logs/%j_null.err
#SBATCH --ntasks=128

module load GCCcore/.9.3.0
module load Python/3.8.5
module load Python-Neuroimaging/2020-Python-3.8.5
module load parallel


# the --exclusive to srun makes srun use distinct CPUs for each job step
# -N1 -n1 allocates a single core to each task
srun="srun --exclusive -n1"

# --delay .2 prevents overloading the controlling node
# -j is the number of tasks parallel runs so we set it to $SLURM_NTASKS
parallel="parallel --delay .2 -j $SLURM_NTASKS"

$parallel "$srun python null_cutoffs.py {1} {2} {3} > /p/project/cinm-7/pyALE_TFCE/output/null/null_{1}_{2}_{3}_{4}.out" ::: $1 ::: $2 ::: $(seq $3 $4) ::: {0..99}

for rep in $(seq $3 $4);
do
	python null_post.py $1 $2 $rep;
done
