vipu create partition p${SLURM_JOB_ID} --allocation c${SLURM_JOB_ID} --size 16 --reconfigurable
export IPUOF_VIPU_API_HOST=angelsfall-ctrl
export IPUOF_VIPU_API_PARTITION_ID=p${SLURM_JOB_ID}
