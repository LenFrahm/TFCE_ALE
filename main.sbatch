#!/bin/bash
#SBATCH --account=inm7
#SBATCH --mail-user=l.frahm@fz-juelich.de
#SBATCH --mail-type=END
#SBATCH --job-name=main_sim   # create a name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=128           # total number of tasks
#SBATCH --cpus-per-task=1      # cpu-cores per task
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=/p/scratch/cinm-7/pyALE_TFCE/logs/%j_main.out
#SBATCH --error=/p/scratch/cinm-7/pyALE_TFCE/logs/%j_main.err

module load GCC/9.3.0
module load OpenMPI/4.1.0rc1
module load Python/3.8.5
module load Python-Neuroimaging/2020-Python-3.8.5
module load mpi4py/3.0.3-Python-3.8.5
module load h5py/2.10.0-Python-3.8.5

for num_subj in $1;
do
    for num_true in $(seq 0 10);
    do
        for rep in 0;
        do
        srun python main.py $num_subj $num_true $rep
        done
    done
done
