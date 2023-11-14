tag = "EnsembleWithConsistency_LinearSVM_NoProjection_NoNoise_NoNorm_MovedToGPU"

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
    test_dataset, batch_size=1000, shuffle=False, num_workers=20
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=512, shuffle=True, num_workers=20
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

# parser.add_argument(
#     "--model_path2",
#     type=str,
#     help="path to the model checkpoint",
#     required=True,
# )

parser.add_argument(
    "--kernel",
    type=str,
    default="linear",
    help="kernel to use for smoothing",
)

parser.add_argument(
    "--C",
    type=float,
    default=1.0,
    help="regularization parameter for the SVM",
)


args = parser.parse_args()

print(args)

weights_file = args.model_path
base_path = os.path.dirname(weights_file) + "/"

sigma = args.sigma

# # use svm to classify the representations
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.svm import SVC, LinearSVC
import numpy as np

import torch.nn.functional as F


class Classifier(torch.nn.Module):
    def __init__(
        self,
        encoder_weights,
        kernel="linear",
        C=1.0,
        dataset="cifar10",
    ):
        super(Classifier, self).__init__()
        self.encoder = SupConResNet(name="resnet50", head="mlp", feat_dim=128)

        state_dict = torch.load(encoder_weights)

        renamed_state_dict = {}
        for k, v in state_dict["model"].items():
            renamed_state_dict[k.replace("module.", "")] = v

        self.encoder.load_state_dict(renamed_state_dict)
        self.encoder = self.encoder.encoder.to(device)
        # self.encoder = self.encoder.to(device)
        # print("Using the Projector !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.eval()

        self.normalizer = get_normalize_layer(dataset)
        self.normalizer.means = self.normalizer.means.to(device)
        self.normalizer.sds = self.normalizer.sds.to(device)
        self.normalizer.eval()

        # self.svm = SVC(kernel=kernel, C=C, probability=True, cache_size=2_000)
        # self.svm = SVC(kernel=kernel, C=C, probability=False, cache_size=2_000)
        self.svm = LinearSVC(C=C, max_iter=10_000)
        self.linear = torch.nn.Linear(2048, 10).to(device)

    def train_svm(self, train_loader):
        train_features = []
        train_labels = []

        with torch.no_grad():
            for i, batch in enumerate(train_loader, 0):
                (inputs, labels) = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs = inputs + torch.randn_like(inputs) * 0.25

                inputs = self.normalizer(inputs)
                inputs = self.encoder(inputs)
                inputs = F.normalize(inputs, dim=1)

                train_features.append(inputs.cpu().numpy())
                train_labels.append(labels.cpu().numpy())

        train_features = np.concatenate(train_features, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        print(train_features.shape, train_labels.shape)
        self.svm.fit(train_features, train_labels)

        # move the parameters from svm to a linear layer

        self.linear.weight.data = torch.from_numpy(self.svm.coef_).to(device).float()
        self.linear.bias.data = torch.from_numpy(self.svm.intercept_).to(device).float()

    def forward(self, x):
        with torch.no_grad():
            x = self.normalizer(x)
            x = self.encoder(x)
            # x = F.normalize(x, dim=1)

        # x = self.svm.predict_proba(x.cpu().numpy())
        # x = torch.from_numpy(x).to(device)

        # x = self.svm.decision_function(x.cpu().numpy())
        # x = torch.from_numpy(x).to(device)

        x = self.linear(x)

        x = F.softmax(x, dim=1)

        return x


classifier1 = Classifier(
    weights_file,
    kernel=args.kernel,
    C=args.C,
).to(device)

# classifier2 = Classifier(
#     args.model_path2,
#     kernel=args.kernel,
#     C=args.C,
# ).to(device)

classifier1.train_svm(train_loader)
# classifier2.train_svm(train_loader)

# # use svm to classify the representations

# test_acc1, test_acc2 = 0.0, 0.0

# with torch.no_grad():
#     for i, batch in enumerate(test_loader, 0):
#         (inputs, labels) = batch
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         predicted1 = classifier1(inputs).argmax(dim=1)
#         predicted2 = classifier2(inputs).argmax(dim=1)

#         test_acc1 += (predicted1 == labels).sum().item()
#         test_acc2 += (predicted2 == labels).sum().item()

# test_acc1 /= len(test_loader.dataset)
# test_acc2 /= len(test_loader.dataset)
# print(f"Test accuracy: {test_acc1} {test_acc2}")


class Ensemble(torch.nn.Module):
    def __init__(self, classifier1, classifier2):
        super(Ensemble, self).__init__()
        self.classifier1 = classifier1
        # self.classifier2 = classifier2
        # TODO load model from /scratch/mehrshad_mirmohammadi/Project/smoothing-consistency/logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet110/lr0.05/checkpoint.pth.tar
        from smoothing.code.archs.cifar_resnet import resnet as resnet_cifar

        self.classifier3 = resnet_cifar(depth=110, num_classes=10).cuda()
        checkpoint = torch.load(
            "/scratch/mehrshad_mirmohammadi/Project/smoothing-consistency/logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet110/lr0.05/checkpoint.pth.tar"
        )
        # remove the first 2 chars from the keys
        renamed_state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            renamed_state_dict[k[2:]] = v

        self.classifier3.load_state_dict(renamed_state_dict)

    def forward(self, x):
        x1 = self.classifier1(x)
        # x2 = self.classifier2(x)
        x3 = self.classifier3(self.classifier1.normalizer(x))
        x3 = F.softmax(x3, dim=1)
        # return torch.max(torch.max(x1, x2), x3)
        # return (x1 + x2 + x3) / 3

        return (x1 + x3) / 2


# class Ensemble(torch.nn.Module):
#     def __init__(self, classifier1, classifier2):
#         super(Ensemble, self).__init__()
#         self.classifier1 = classifier1
#         self.classifier2 = classifier2
#         # TODO load model from /scratch/mehrshad_mirmohammadi/Project/smoothing-consistency/logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet110/lr0.05/checkpoint.pth.tar
#         from smoothing.code.archs.cifar_resnet import resnet as resnet_cifar

#         self.classifier3 = resnet_cifar(depth=110, num_classes=10).cuda()
#         checkpoint = torch.load(
#             "/scratch/mehrshad_mirmohammadi/Project/smoothing-consistency/logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet110/lr0.05/checkpoint.pth.tar"
#         )
#         # remove the first 2 chars from the keys
#         renamed_state_dict = {}
#         for k, v in checkpoint["state_dict"].items():
#             renamed_state_dict[k[2:]] = v

#         self.classifier3.load_state_dict(renamed_state_dict)

#         self.classifier4 = resnet_cifar(depth=110, num_classes=10).cuda()
#         checkpoint = torch.load(
#             "/scratch/mehrshad_mirmohammadi/Project/smoothing-consistency/logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet110/lr0.01/checkpoint.pth.tar"
#         )
#         # remove the first 2 chars from the keys
#         renamed_state_dict = {}
#         for k, v in checkpoint["state_dict"].items():
#             renamed_state_dict[k[2:]] = v

#         self.classifier4.load_state_dict(renamed_state_dict)

#     def forward(self, x):
#         x3 = self.classifier3(self.classifier1.normalizer(x))
#         x3 = F.softmax(x3, dim=1)
#         x4 = self.classifier4(self.classifier1.normalizer(x))
#         x4 = F.softmax(x4, dim=1)
#         return torch.max(x3, x4)


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
    torch.nn.DataParallel(Ensemble(classifier1, None)), 10, sigma
)

# prepare output file

exp_name = "{}_reduced_svm_{}_{}".format(tag, args.kernel, args.C)

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
