# print cuda visible devices

import os
import sys

import torch

print("CUDA_VISIBLE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
print("torch.cuda.device_count(): ", torch.cuda.device_count())
print("SLURM job ID: ", os.environ["SLURM_JOB_ID"])
print("SLURM task local ID: ", os.environ["SLURM_LOCALID"])
