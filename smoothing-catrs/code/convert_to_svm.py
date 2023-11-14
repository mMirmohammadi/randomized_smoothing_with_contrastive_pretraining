import sys, os
import torch

from datasets import get_dataset, get_normalize_layer
from archs.resnet_big import SupConResNet

from sklearn.svm import SVC, LinearSVC
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F


class SVM_Wrapper(torch.nn.Module):
    def __init__(
        self,
        dataset="cifar10",
        C=1.0,
        resnet_args:dict = dict(name="resnet50", head="mlp", feat_dim=128),
    ):
        super(SVM_Wrapper, self).__init__()
        self.encoder = SupConResNet(**resnet_args).encoder
        self.normalizer = get_normalize_layer(dataset)
        self.svm = LinearSVC(C=C, max_iter=10_000)
        self.linear = torch.nn.Linear(2048, 10)
        # self.feature_normalizer = torch.nn.BatchNorm1d(2048)
        self.resnet_args = resnet_args

    def load_weights(self,weights_file):

        state_dict = torch.load(weights_file)

        renamed_state_dict = {}
        for k, v in state_dict["model"].items():
            if k.startswith('encoder.'):
                new_k = k.replace("encoder.", "").replace("module.", "")
                renamed_state_dict[new_k] = v

        self.encoder.load_state_dict(renamed_state_dict)
        # self.encoder = self.encoder.to(device)
        # print("Using the Projector !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def train_svm(self, train_loader, sigma):
        train_features = []
        train_labels = []

        with torch.no_grad():
            for i, batch in (enumerate(tqdm(train_loader), 0)):
                (inputs, labels) = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs = inputs + torch.randn_like(inputs) * sigma

                inputs = self.normalizer(inputs)
                inputs = self.encoder(inputs)
                inputs = F.normalize(inputs, dim=1)

                train_features.append(inputs.cpu().numpy())
                train_labels.append(labels.cpu().numpy())

            train_features = np.concatenate(train_features, axis=0)
            train_labels = np.concatenate(train_labels, axis=0)

            # self.feature_normalizer.running_mean.data = torch.from_numpy(
            #     train_features.mean(axis=0)
            # ).to(device)
            # self.feature_normalizer.running_var.data = torch.from_numpy(
            #     train_features.var(axis=0)
            # ).to(device)

            # self.feature_normalizer.eval()

            # train_features = (
            #     self.feature_normalizer(
            #         torch.from_numpy(train_features).to(device).float()
            #     )
            #     .detach()
            #     .cpu()
            #     .numpy()
            # )

        print(train_features.shape, train_labels.shape)
        self.svm.fit(train_features, train_labels)

        # move the parameters from svm to a linear layer

        self.linear.weight.data = torch.from_numpy(self.svm.coef_).to(device).float()
        self.linear.bias.data = torch.from_numpy(self.svm.intercept_).to(device).float()
        self.svm_trained = True

    def forward(self, x, only_linear_grad=False, use_softmax=True):
        
        if only_linear_grad:
            with torch.no_grad():
                x = self.normalizer(x)
                x = self.encoder(x)
                # x = F.normalize(x, dim=1)

        else:
            x = self.normalizer(x)
            x = self.encoder(x)
            x = F.normalize(x, dim=1)

        # x = self.svm.predict_proba(x.cpu().numpy())
        # x = torch.from_numpy(x).to(device)

        # x = self.svm.decision_function(x.cpu().numpy())
        # x = torch.from_numpy(x).to(device)

        x = self.linear(x)

        if use_softmax: x = F.softmax(x, dim=1)

        return x


if __name__ == '__main__':

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


    import argparse

    parser = argparse.ArgumentParser(
        description="This scripts trains a linear classifier on top of frozen representations and then certifies the predictions using randomized smoothing."
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

    parser.add_argument(
        "--sigma",
        type=float,
        default=0.25,
        help="noise parameter to train the SVM",
    )


    args = parser.parse_args()

    print(args)

    weights_file: str = args.model_path
    base_path = os.path.dirname(weights_file) + "/"

    from sklearnex import patch_sklearn

    patch_sklearn()



    classifier1 = SVM_Wrapper(
        C=args.C,
    ).to(device)

    classifier1.load_weights(weights_file)

    # classifier2 = SVM_Wrapper(
    #     args.model_path2,
    #     kernel=args.kernel,
    #     C=args.C,
    # ).to(device)

    classifier1.train_svm(train_loader,args.sigma)

    test_acc = 0.0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader), 0):
            (inputs, labels) = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            predicted = classifier1(inputs).argmax(dim=1)

            test_acc += (predicted == labels).sum().item()

    test_acc /= len(test_loader.dataset)
    print(f"Test accuracy: {test_acc}")
    
    test_acc = 0.0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader), 0):
            (inputs, labels) = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            predicted = classifier1(inputs+torch.randn_like(inputs)*args.sigma).argmax(dim=1)

            test_acc += (predicted == labels).sum().item()

    test_acc /= len(test_loader.dataset)
    print(f"Test+noise accuracy: {test_acc}")

    ext = weights_file.split('.')[-1]
    ext = '.' + ext if ext else ''

    svm_weights_file = weights_file.replace(ext,'_svm_n'+ext)

    torch.save({
        'model':classifier1.state_dict(),
        'arch':'svm_'+classifier1.resnet_args['name'],
    },svm_weights_file)

    classifier2 = SVM_Wrapper()
    classifier2.load_state_dict(torch.load(svm_weights_file)['model'])
    print('loaded')