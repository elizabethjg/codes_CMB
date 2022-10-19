#!/bin/bash
#SBATCH --job-name=pro2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=3-00:00:00
## An email is sent when the job is canceled or finished
## Change "root" for your e-mail address
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ejgonzalez@unc.edu.ar


. /etc/profile
source $HOME/.bashrc

cd /mnt/projects/lensing/HALO_SHAPE/MICEv1.0/codes_HS_micev1.0/

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

conda activate myenv
srun python -u forGroup_qprofile.py -sample 'CMB_test' -lM_min 14.7 -lM_max 14.9 -z_min 0.4 -z_max 0.45 -ncores 56 -RIN 500 -ROUT 80000 -nbins 60 -relax True
