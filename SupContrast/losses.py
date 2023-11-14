from __future__ import print_function

import torch
import torch.nn as nn
from typing import Optional


# A simplified version of Hierarchical Supervised Contrastive Loss
class TwoLevelV3HieSupCon(nn.Module):
    """Hierarchical Supervised Contrastive Learning
    Modifications: There is only 2 levels. There is only pull loss for the noise level.
    But we can support 3 levels two.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        alpha: tuple = (1, 1, 1),
        positive_loss: str = "linear",
        positive_loss_temperature: float = 0.07,
    ) -> None:
        super(TwoLevelV3HieSupCon, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha = self.alpha / self.alpha.sum()
        self.positive_loss = positive_loss
        self.positive_loss_temperature = positive_loss_temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def SimpleSupConLoss(
        self,
        logits: torch.Tensor,
        positives: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param logits: (N, N), similarity matrix
        :param positives: (N, N), mask of positives
        :param negatives: (N, N), mask of negatives. If None, all samples except positives and self are negatives.
        Note that the number of possitives and negatives for each sample must be the same.
        """
        N = logits.shape[0]
        if negatives is None:
            negatives = ~positives
            negatives.fill_diagonal_(False)
        # Only keep the positives and negatives
        mask = positives | negatives
        logits = logits[mask].view(N, -1)  # (N, NP + NN)
        positives = positives[mask].view(N, -1)  # (N, NP + NN)
        negatives = negatives[mask].view(N, -1)  # (N, NP + NN)

        # Calculate the cross entropy loss for each sample.
        class_probs = positives.float() / (
            positives.sum(dim=1, keepdim=True).float() + 1e-8
        )  # (N, NP + NN)
        loss = self.cross_entropy(logits, class_probs)

        return loss

    def SimpleOnlyPositiveLoss(
        self,
        logits: torch.Tensor,
        positives: torch.Tensor,
    ) -> torch.Tensor:
        N = logits.shape[0]
        logits = logits[positives].view(N, -1)  # (N, NP)

        if self.positive_loss == "linear":
            loss = -logits.mean() / self.positive_loss_temperature
        elif self.positive_loss == "exp":
            loss = torch.exp(-logits / self.positive_loss_temperature).mean()
        elif self.positive_loss == "log":
            loss = -torch.log((logits + 1) / self.positive_loss_temperature).mean()
        else:
            raise NotImplementedError()

        return loss

    def forward(
        self,
        features: torch.Tensor,  # (bs, na, nn, fd) where bs is batch size, na is number of augmentations, nn is number of noises and fd is feature dimension
        labels: torch.Tensor,  # (bs)
    ):
        bs, na, nn, fd = features.shape

        # Compute the similarity matrix
        features = features.view(bs * na * nn, fd)  # (bs * na * nn, fd)
        similarity_matrix = torch.matmul(
            features, features.T
        )  # (bs * na * nn, bs * na * nn)
        logits = similarity_matrix / self.temperature  # (bs * na * nn, bs * na * nn)

        # Calculate the noise invariant loss

        labels_noise = (
            torch.arange(bs * na).repeat_interleave(nn).to(labels.device)
        )  # (bs * na * nn)
        positives_noise = torch.eq(
            labels_noise.unsqueeze(1), labels_noise.unsqueeze(0)
        )  # (bs * na * nn, bs * na * nn)
        positives_noise.fill_diagonal_(False)
        loss_noise = self.SimpleOnlyPositiveLoss(logits, positives_noise)

        # Calculate the augmentation invariant loss

        labels_aug = (
            torch.arange(bs).repeat_interleave(na * nn).to(labels.device)
        )  # (bs * na * nn)
        positives_aug = torch.eq(labels_aug.unsqueeze(1), labels_aug.unsqueeze(0))
        positives_aug.fill_diagonal_(False)
        positives_aug[positives_noise] = False

        loss_aug = 0.0  # self.SimpleOnlyPositiveLoss(logits, positives_aug)

        # Calculate the class invariant loss

        labels_class = labels.repeat_interleave(na * nn)  # (bs * na * nn)
        positives_class = torch.eq(labels_class.unsqueeze(1), labels_class.unsqueeze(0))
        negatives_class = ~positives_class

        positives_class.fill_diagonal_(False)
        positives_class[positives_noise] = False
        positives_class[positives_aug] = False

        loss_class = self.SimpleSupConLoss(logits, positives_class, negatives_class)

        # Calculate the total loss
        loss = (
            self.alpha[0] * loss_class
            + self.alpha[1] * loss_aug
            + self.alpha[2] * loss_noise
        )

        loss *= self.temperature / self.base_temperature

        return loss


# A simplified version of Hierarchical Supervised Contrastive Loss
class SimHierSupConLoss(nn.Module):
    """Hierarchical Supervised Contrastive Learning
    In this loss we try to learn representations that are invariant to class, augmentation and noise.
    But the importance of each of these invariances can be different. We let alpha[2] be the weight
    of the noise invariance, alpha[1] be the weight of the augmentation invariance and alpha[0] be
    the weight of the class invariance.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        alpha: tuple = (1, 1, 1),
    ) -> None:
        super(SimHierSupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha = self.alpha / self.alpha.sum()
        self.cross_entropy = nn.CrossEntropyLoss()

    def SimpleSupConLoss(
        self,
        logits: torch.Tensor,
        positives: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param logits: (N, N), similarity matrix
        :param positives: (N, N), mask of positives
        :param negatives: (N, N), mask of negatives. If None, all samples except positives and self are negatives.
        Note that the number of possitives and negatives for each sample must be the same.
        """
        N = logits.shape[0]
        diagonal = torch.eye(N, dtype=torch.bool, device=logits.device)
        if negatives is None:
            negatives = ~positives
            negatives.fill_diagonal_(False)
        # Only keep the positives and negatives
        mask = positives | negatives
        logits = logits[mask].view(N, -1)  # (N, NP + NN)
        positives = positives[mask].view(N, -1)  # (N, NP + NN)
        negatives = negatives[mask].view(N, -1)  # (N, NP + NN)

        # Calculate the cross entropy loss for each sample.
        class_probs = positives.float() / (
            positives.sum(dim=1, keepdim=True).float() + 1e-8
        )  # (N, NP + NN)
        loss = self.cross_entropy(logits, class_probs)

        return loss

    def forward(
        self,
        features: torch.Tensor,  # (bs, na, nn, fd) where bs is batch size, na is number of augmentations, nn is number of noises and fd is feature dimension
        labels: torch.Tensor,  # (bs)
    ):
        bs, na, nn, fd = features.shape

        # Compute the similarity matrix
        features = features.view(bs * na * nn, fd)  # (bs * na * nn, fd)
        similarity_matrix = torch.matmul(
            features, features.T
        )  # (bs * na * nn, bs * na * nn)
        logits = similarity_matrix / self.temperature  # (bs * na * nn, bs * na * nn)

        # # Calculate the class invariant loss
        # labels = labels.repeat_interleave(na * nn)  # (bs * na * nn)
        # positives = torch.eq(
        #     labels.unsqueeze(1), labels.unsqueeze(0)
        # )  # (bs * na * nn, bs * na * nn)
        # positives.fill_diagonal_(False)
        # loss_class = self.SimpleSupConLoss(logits, positives)

        # # Calculate the augmentation invariant loss
        # labels = (
        #     torch.arange(bs).repeat_interleave(na * nn).to(labels.device)
        # )  # (bs * na * nn)
        # positives = torch.eq(
        #     labels.unsqueeze(1), labels.unsqueeze(0)
        # )  # (bs * na * nn, bs * na * nn)
        # positives.fill_diagonal_(False)
        # loss_aug = self.SimpleSupConLoss(logits, positives)

        # # Calculate the noise invariant loss
        # labels = (
        #     torch.arange(bs * na).repeat_interleave(nn).to(labels.device)
        # )  # (bs * na * nn)
        # positives = torch.eq(
        #     labels.unsqueeze(1), labels.unsqueeze(0)
        # )  # (bs * na * nn, bs * na * nn)
        # positives.fill_diagonal_(False)
        # loss_noise = self.SimpleSupConLoss(logits, positives)

        # Calculate the noise invariant loss

        labels_noise = (
            torch.arange(bs * na).repeat_interleave(nn).to(labels.device)
        )  # (bs * na * nn)
        positives_noise = torch.eq(
            labels_noise.unsqueeze(1), labels_noise.unsqueeze(0)
        )  # (bs * na * nn, bs * na * nn)
        positives_noise.fill_diagonal_(False)
        loss_noise = self.SimpleSupConLoss(logits, positives_noise)

        # Calculate the augmentation invariant loss

        labels_aug = (
            torch.arange(bs).repeat_interleave(na * nn).to(labels.device)
        )  # (bs * na * nn)
        positives_aug = torch.eq(labels_aug.unsqueeze(1), labels_aug.unsqueeze(0))
        positives_aug.fill_diagonal_(False)
        positives_aug[positives_noise] = False
        negatives_aug = ~positives_aug
        negatives_aug.fill_diagonal_(False)
        negatives_aug[positives_noise] = False

        loss_aug = 0.0  # self.SimpleSupConLoss(logits, positives_aug, negatives_aug)

        # Calculate the class invariant loss

        labels_class = labels.repeat_interleave(na * nn)  # (bs * na * nn)
        positives_class = torch.eq(labels_class.unsqueeze(1), labels_class.unsqueeze(0))
        positives_class.fill_diagonal_(False)
        positives_class[positives_noise] = False
        positives_class[positives_aug] = False
        negatives_class = ~positives_class
        negatives_class.fill_diagonal_(False)
        negatives_class[positives_noise] = False
        negatives_class[positives_aug] = False

        loss_class = self.SimpleSupConLoss(logits, positives_class, negatives_class)

        # Calculate the total loss
        loss = (
            self.alpha[0] * loss_class
            + self.alpha[1] * loss_aug
            + self.alpha[2] * loss_noise
        )

        loss *= self.temperature / self.base_temperature

        return loss


def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)


class HMLC(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        layer_penalty: Optional[callable] = None,
        loss_type: str = "hmce",
    ) -> None:
        super(HMLC, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        if not layer_penalty:
            self.layer_penalty = self.pow_2
        else:
            self.layer_penalty = layer_penalty
        self.sup_con_loss = SupConLoss(
            temperature=temperature, base_temperature=base_temperature
        )
        self.loss_type = loss_type

    def pow_2(self, value):
        return torch.pow(2, value)

    def forward(
        self,
        features: torch.Tensor,  # [bsz, n_views, dim]
        labels: torch.Tensor,  # [bsz, n_labels]
    ) -> torch.Tensor:
        device = features.device
        mask = torch.ones(labels.shape).to(device)
        cumulative_loss = torch.tensor(0.0).to(device)
        max_loss_lower_layer = torch.tensor(float("-inf"))
        for l in range(0, labels.shape[1]):  # changed from 1 to 0
            mask[:, labels.shape[1] - l :] = 0
            layer_labels = labels * mask
            mask_labels = (
                torch.stack(
                    [
                        torch.all(torch.eq(layer_labels[i], layer_labels), dim=1)
                        for i in range(layer_labels.shape[0])
                    ]
                )
                .type(torch.uint8)
                .to(device)
            )
            layer_loss = self.sup_con_loss(features, mask=mask_labels)
            if self.loss_type == "hmc":
                cumulative_loss += (
                    self.layer_penalty(torch.tensor(1 / (l)).type(torch.float))
                    * layer_loss
                )
            elif self.loss_type == "hce":
                layer_loss = torch.max(
                    max_loss_lower_layer.to(layer_loss.device), layer_loss
                )
                cumulative_loss += layer_loss
            elif self.loss_type == "hmce":
                layer_loss = torch.max(
                    max_loss_lower_layer.to(layer_loss.device), layer_loss
                )
                cumulative_loss += (
                    self.layer_penalty(torch.tensor(1 / l).type(torch.float))
                    * layer_loss
                )
            else:
                raise NotImplementedError("Unknown loss")
            _, unique_indices = unique(layer_labels, dim=0)
            max_loss_lower_layer = torch.max(
                max_loss_lower_layer.to(layer_loss.device), layer_loss
            )
            # labels = labels[unique_indices]
            # mask = mask[unique_indices]
            # features = features[unique_indices]
        return cumulative_loss / labels.shape[1]


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "all",
        base_temperature: float = 0.07,
    ) -> None:
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


#### Consistency Losses ####

from torch.nn import functional as F


def consistency_loss(logits, lbd=20.0, eta=0.5, loss="default"):
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


class ConsistencyLoss(nn.Module):
    def __init__(self, lbd=20.0, eta=0.5):
        super(ConsistencyLoss, self).__init__()
        self.lbd = lbd
        self.eta = eta
        self.criterian = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        bs, na, nn, fd = logits.shape
        logits = logits.view(bs * na * nn, fd)
        labels = labels.repeat_interleave(na * nn)
        loss_xent = self.criterian(logits, labels)

        logits_chunk = torch.chunk(
            logits, nn, dim=0
        )  # can also be na * nn depending on the interpretation of what samples shoud be close to each other
        loss_con = consistency_loss(logits_chunk, self.lbd, self.eta)

        loss = loss_xent + loss_con

        return loss
