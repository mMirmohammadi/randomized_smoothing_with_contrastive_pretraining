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
    test_dataset, batch_size=1000, shuffle=False, num_workers=12
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=512, shuffle=True, num_workers=12
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
    "--num_noise_vec",
    type=int,
    default=2,
    help="number of noise vectors to use for consistency loss",
)

parser.add_argument(
    "--lbd",
    type=float,
    default=20.0,
    help="lambda for consistency loss",
)

parser.add_argument(
    "--eta",
    type=float,
    default=0.5,
    help="eta for consistency loss",
)

args = parser.parse_args()

print(args)

weights_file = args.model_path
base_path = os.path.dirname(weights_file) + "/"

sigma = args.sigma


class Classifier(torch.nn.Module):
    def __init__(
        self,
        encoder_weights,
        predictor,
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

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.eval()

        self.predictor = predictor
        # self.predictor.eval()

        self.normalizer = get_normalize_layer(dataset)
        self.normalizer.means = self.normalizer.means.to(device)
        self.normalizer.sds = self.normalizer.sds.to(device)
        self.normalizer.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.normalizer(x)
            x = self.encoder(x)

        x = self.predictor(x)
        return x


predictor = torch.nn.Sequential(
    # torch.nn.Linear(2048, 2048),
    # torch.nn.ReLU(),
    # torch.nn.Linear(2048, 2048),
    # torch.nn.ReLU(),
    torch.nn.Linear(2048, 10),
).to(device)

classifier = Classifier(weights_file, predictor).to(device)

print(
    f"Number of parameters in the Encoder: {sum([p.numel() for p in classifier.encoder.parameters()]) / 1000_000}"
)
print(
    f"Number of parameters in the Predictor: {sum([p.numel() for p in classifier.predictor.parameters()]) / 1000_000}"
)
print(
    f"Number of parameters in the Classifier: {sum([p.numel() for p in classifier.parameters()]) / 1000_000}"
)
print(
    f"Number of trainable parameters in the Classifier: {sum([p.numel() for p in classifier.parameters() if p.requires_grad]) / 1000_000}"
)

# %%
from tqdm import tqdm

# train the classifier
import torch.optim as optim
import torch.nn.functional as F

import torch
import torch.nn.functional as F


def consistency_loss(logits, lbd, eta=0.5, loss="default"):
    """
    Consistency regularization for certified robustness.

    Parameters
    ----------
    logits : List[torch.Tensor]
        A list of logit batches of the same shape, where each
        is sampled from f(x + noise) with i.i.d. noises.
        len(logits) determines the number of noises, i.e., m > 1.
    lbd : float
        Hyperparameter that controls the strength of the regularization.
    eta : float (default: 0.5)
        Hyperparameter that controls the strength of the entropy term.
        Currently used only when loss='default'.
    loss : {'default', 'xent', 'kl', 'mse'} (optional)
        Which loss to minimize to obtain consistency.
        - 'default': The default form of loss.
            All the values in the paper are reproducible with this option.
            The form is equivalent to 'xent' when eta = lbd, but allows
            a larger lbd (e.g., lbd = 20) when eta is smaller (e.g., eta < 1).
        - 'xent': The cross-entropy loss.
            A special case of loss='default' when eta = lbd. One should use
            a lower lbd (e.g., lbd = 3) for better results.
        - 'kl': The KL-divergence between each predictions and their average.
        - 'mse': The mean-squared error between the first two predictions.

    """

    m = len(logits)
    softmax = [F.softmax(logit, dim=1) for logit in logits]
    avg_softmax = sum(softmax) / m

    loss_kl = [kl_div(logit, avg_softmax) for logit in logits]
    loss_kl = sum(loss_kl) / m

    if loss == "default":
        loss_ent = entropy(avg_softmax)
        consistency = lbd * loss_kl + eta * loss_ent
    elif loss == "xent":
        loss_ent = entropy(avg_softmax)
        consistency = lbd * (loss_kl + loss_ent)
    elif loss == "kl":
        consistency = lbd * loss_kl
    elif loss == "mse":
        sm1, sm2 = softmax[0], softmax[1]
        loss_mse = ((sm2 - sm1) ** 2).sum(1)
        consistency = lbd * loss_mse
    else:
        raise NotImplementedError()

    return consistency.mean()


def kl_div(input, targets):
    return F.kl_div(F.log_softmax(input, dim=1), targets, reduction="none").sum(1)


def entropy(input):
    logsoftmax = torch.log(input.clamp(min=1e-20))
    xent = (-input * logsoftmax).sum(1)
    return xent


optimizer = optim.Adam(predictor.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=15, eta_min=0.01
)

for epoch in range(15):  # loop over the dataset multiple times
    running_loss = 0.0
    running_accuracy = 0.0

    # warmup_loss_coeff = max(min(1.0, (epoch + 1) / 5), 1.0)

    classifier.train()
    tq = tqdm(enumerate(train_loader, 0), total=len(train_loader))
    for i, batch in tq:
        (inputs, labels) = batch
        inputs = inputs.to(device)

        # noise = torch.rand_like(inputs, device=device) * sigma
        # inputs += noise

        noises = [
            torch.randn_like(inputs, device=device) * sigma
            for _ in range(args.num_noise_vec)
        ]

        inputs_c = torch.cat([inputs + noise for noise in noises], dim=0)
        targets_c = labels.repeat(args.num_noise_vec)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = classifier(inputs_c)
        loss_xent = criterion(outputs, targets_c.to(device))

        logits_chunk = torch.chunk(outputs, args.num_noise_vec, dim=0)
        loss_con = consistency_loss(logits_chunk, args.lbd, args.eta)

        loss = loss_xent + loss_con
        # loss = loss * warmup_loss_coeff

        loss.backward()
        optimizer.step()

        # calc accuracy
        _, predicted = torch.max(outputs.cpu().data, 1)
        accuracy = (predicted == targets_c).float().mean().item()

        running_loss += loss.item()
        running_accuracy += accuracy

        # print statistics
        tq.set_description(
            f"Epoch {epoch} - Running Loss: {running_loss / (i + 1):.4f} - Running Accuracy: {running_accuracy / (i + 1):.4f}"
        )

    scheduler.step()

    classifier.eval()

    # accuracy on test set
    correct = 0
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader, 0):
            (inputs, labels) = batch
            inputs = inputs.to(device)

            outputs = classifier(inputs).cpu()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).cpu().sum().item()

    print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")

    # accuracy on test set
    correct = 0
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader, 0):
            (inputs, labels) = batch
            inputs = inputs.to(device)

            noise = torch.rand_like(inputs, device=device) * sigma
            inputs += noise

            outputs = classifier(inputs).cpu()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).cpu().sum().item()

    print(
        f"Accuracy of the network on the 10000 test images: {100 * correct / total}% | Using noise with sigma={sigma}"
    )


# %%
# # Save the model checkpoint
# torch.save(classifier.state_dict(), f'{super_path}one_layer.pth')

# %%
from time import time
import datetime
import os

# iter_skip = 10
# iter_max = 50000
# N0 = 100
# N = 8_000
# alpha = 0.01
# batch_size = 8_000

iter_skip = args.skip
iter_max = args.max
N0 = args.N0
N = args.N
alpha = args.alpha
batch_size = args.batch_size

smoothed_classifier = Smooth(classifier, 10, sigma)

# prepare output file

arch_name = ""

for layer in predictor.children():
    if isinstance(layer, torch.nn.Linear):
        arch_name += f"_{layer.out_features}"
    elif isinstance(layer, torch.nn.BatchNorm1d):
        arch_name += "_bn"

exp_name = "Consistency_reduced{}".format(arch_name)

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
