#!/bin/bash

# Request a gpu node
#SBATCH --account=ai4200-001-2026f
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=48G
#SBATCH --time=00-02:00:00
#SBATCH --nodes=1

spack unload
spack load py-jupyter
spack load py-torch@2.13
spack load py-python-dotenv
env_path="/work/classes/ai4200-001-2026f/bcchambers43/ai4200-p1"
. ${env_path}/bin/activate
cd /work/classes/ai4200-001-2026f/bcchambers43/AI4200-imagenet32x32-baseline-model
papermill project1_harness.ipynb harness_run.ipynb --kernel _work_classes_ai4200-001-2026f_bcchambers43_ai4200-p1_bin_python --execution-timeout -1
