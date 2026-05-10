# IRC Conversation Disentanglement (PyTorch)

A BERT-based CrossEncoder with handcrafted features for IRC message linking and conversation disentanglement.


## Acknowledgements

https://github.com/jkkummerfeld/irc-disentanglement


============================
BUNYA HPC QUICK REFERENCE
============================

--- 1. CHECK IF A GPU IS FREE ---
sinfo -o "%.P %.5a %.10l %.6D %.6t %N %.C %.E %.g %.G %.m"
# States: idle=free, mix=partially used, alloc=full, drain=maintenance

# Quick GPU view:
sinfo -O Partition,NodeList,Nodes,Gres,CPUs


--- 2. ALLOCATE A COMPUTE CPU (not login node) ---
# debug QoS (1 hr max, higher priority):
salloc --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --mem=5G --job-name=CHANGE-ME --time=01:00:00 --partition=general --qos=debug --account=a_hcc srun --export=PATH,TERM,HOME,LANG --pty /bin/bash -l

# Verify you are on a compute node (NOT bunya1 or bunya2):
hostname


--- 3. CHECK JOB QUEUE / TIME LEFT ---
# Detailed view with TIME_LEFT and REASON:
squeue -o "%12i %7q %.9P %.20j %.10u %.2t %.11M %.4D %.4C %.14b %8m %16R %18p %10B %.10L" --me

# Simple view:
squeue --me


--- 4. FREE GPU NODE vs FREE GPU ---
# idle = definitely free GPU
# mix  = node has jobs but MAY have free GPUs
# SLURM will allocate based on --gres request
# Always specify GPU type:
--gres=gpu:[type]:[number]
# e.g. --gres=gpu:a100:1   or   --gres=gpu:h100:1


--- 5. CHECK STORAGE QUOTA (home + scratch) ---
rquota
# Shows usage and limits for /home and /scratch/user/username
# Scratch soft limit: 300 GB / 1M files
# Grace period: 2 weeks over limit, then write access locked


--- 6. CHECK GPU USAGE / JOB STATS ---
# Your running/pending jobs:
squeue --me

# Utilisation of a running or completed job:
module load jobstats/2024.08
jobstats JobID

# Completed job history (last 48hrs):
sacct -p -a -S now-48hours --format JobID,User,Group,State,AllocCPUS,REQMEM,TotalCPU,Elapsed,MaxRSS -u $USER

# Live GPU utilisation (must be SSH'd into compute node running your job):
/usr/bin/nvidia-smi

# QoS GPU limits:
# debug/short = 4 GPUs max
# gpu QoS     = 4 GPUs, 4 running jobs max
# sxm         = 4 H100s (approved users only)



salloc --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --mem=16G \
  --job-name=GPUInteractive --time=01:00:00 \
  --partition=gpu_cuda --qos=debug \
  --gres=gpu:1 \
  --account=a_hcc \
  srun --export=PATH,TERM,HOME,LANG --pty /bin/bash -l