import argparse
import os
import datetime
from time import time

#import setGPU
import torch

from third_party.core import Smooth
from architectures import get_architecture
from datasets import get_dataset, DATASETS, get_num_classes, get_input_shape

from tqdm import trange

noise_path = '/mnt/disks/rs-cert-data/noise'

for ds in DATASETS:
    if ds == 'imagenet': continue
    input_shape = get_input_shape(ds)
    num_classes = get_num_classes(ds)
    for N in ['1e4','1e5']:
        for mode in ['gaussian','cauchy']:
            for sig in [0.25,0.5,1.0]:
                smoothed_classifier = Smooth(None, num_classes, sig, mode)

                smoothed_classifier.generate_noise(input_shape,int(float(N)))
                file_path = f'{noise_path}/{ds}/{mode}/{sig}/{N}.pt'
                print(file_path)
                if not os.path.exists(os.path.dirname(file_path)):
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                smoothed_classifier.save_noise(file_path)

