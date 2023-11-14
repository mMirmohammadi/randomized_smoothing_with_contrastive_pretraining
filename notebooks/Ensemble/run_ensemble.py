tag = ""

# %%
import os

# os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch

# torch.cuda.device_count()

# %%
# %load_ext autoreload
# %autoreload 2

print(os.getcwd())

import sys

sys.path.append("../../../")
sys.path.append("../../../SupContrast/")

from smoothing.code.datasets import get_dataset

device = "cuda"
dataset = "cifar10"

test_dataset = get_dataset(dataset, "test")
train_dataset = get_dataset(dataset, "train")

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1000, shuffle=False, num_workers=10
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=512, shuffle=True, num_workers=10
)

# %%
from smoothing.code.datasets import get_normalize_layer
import torch
from networks.resnet_big import SupConResNet
from smoothing.code.core import Smooth

import argparse

parser = argparse.ArgumentParser(
    description="This scripts trains a linear classifier on top of frozen representations and then certifies the predictions using randomized smoothing."
)

parser.add_argument(
    "--sigma",
    type=float,
    default=0.25,
    help="noise hyperparameter for the Gaussian smoothing.",
)

parser.add_argument(
    "--skip",
    type=int,
    default=10,
    help="how many test points to skip before saving a certified radius",
)

parser.add_argument(
    "--max",
    type=int,
    default=-1,
    help="maximum number of test points to certify. -1 certifies all of them",
)

parser.add_argument(
    "--N0",
    type=int,
    default=100,
    help="initial number of samples to estimate the prediction",
)

parser.add_argument(
    "--N",
    type=int,
    default=10_000,
    help="total number of samples to estimate the prediction",
)

parser.add_argument(
    "--alpha",
    type=float,
    default=0.01,
    help="failure probability",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=10_000,
    help="batch size for the linear classifier",
)

parser.add_argument(
    "--model_path",
    type=str,
    help="path to the model checkpoint",
    required=True,
)

parser.add_argument(
    "--tag",
    type=str,
    help="tag for the experiment",
    default=tag,
)

parser.add_argument(
    "--n_ensembles",
    type=int,
    help="number of models in the ensemble",
    default=1,
)

args = parser.parse_args()

print(args)

weights_file = args.model_path
base_path = os.path.dirname(weights_file) + "/"

sigma = args.sigma
tag = args.tag

import numpy as np
import torch.nn.functional as F
from smoothing.code.archs.cifar_resnet import resnet as resnet_cifar


class Ensemble(torch.nn.Module):
    def __init__(self, base_path, n_ensembles=1):
        super(Ensemble, self).__init__()

        self.classifiers = torch.nn.ModuleList()

        self.normalize_layer = get_normalize_layer(dataset).to(device)

        for i in range(7000, 7000 + n_ensembles):
            classifier = resnet_cifar(depth=110, num_classes=10).to(device)
            checkpoint = torch.load(base_path + f"checkpoint-{i:04}.pth.tar")
            # remove the first 2 chars from the keys
            renamed_state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                renamed_state_dict[k[2:]] = v

            classifier.load_state_dict(renamed_state_dict)
            classifier.eval()
            self.classifiers.append(classifier)

    def forward(self, x):
        x = self.normalize_layer(x)
        predictions = []
        for classifier in self.classifiers:
            predictions.append(classifier(x))
        return torch.stack(predictions, dim=0).mean(dim=0)


# %%
from time import time
import datetime
import os

iter_skip = args.skip
iter_max = args.max
N0 = args.N0
N = args.N
alpha = args.alpha
batch_size = args.batch_size

smoothed_classifier = Smooth(
    torch.nn.DataParallel(Ensemble(base_path, args.n_ensembles)).to(device), 10, sigma
)

# prepare output file

exp_name = "reduced_{}".format(tag)

out_file = f"{base_path}RS/{exp_name}_sigma{sigma}_skip{iter_skip}_max{iter_max}_N0{N0}_N{N}_alpha{alpha}_batch{batch_size}_time_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.txt"
os.makedirs(os.path.dirname(out_file), exist_ok=True)

print(out_file)

f = open(out_file, "w")
print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

from tqdm import tqdm

tq = tqdm(total=len(test_dataset) // iter_skip)

for i, batched_sample in enumerate(test_dataset):
    # only certify every args.skip examples, and stop after args.max examples
    if i % iter_skip != 0:
        continue
    if i == iter_max * iter_skip:
        break

    (x, label) = test_dataset[i]

    before_time = time()
    # certify the prediction of g around x
    x = x.cuda()
    prediction, radius = smoothed_classifier.certify(x, N0, N, alpha, batch_size)
    after_time = time()
    correct = int(prediction == label)

    time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
    print(
        "{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed
        ),
        file=f,
        flush=True,
    )
    tq.update(1)

f.close()
