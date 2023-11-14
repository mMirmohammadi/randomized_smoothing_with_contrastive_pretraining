"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, in_planes=64, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        self.num_layers = len(num_blocks)

        self.conv1 = nn.Conv2d(
            in_channel, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*in_planes, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*in_planes, num_blocks[2], stride=2)
        if self.num_layers >= 4: self.layer4 = self._make_layer(block, 8*in_planes, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.num_layers >= 4: out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


class WideResNet(nn.Module):
    def __init__(
        self, block, num_blocks, in_channel=3, zero_init_residual=False, widen_factor=2
    ):
        super(WideResNet, self).__init__()

        self.in_planes = widen_factor * 64

        self.conv1 = nn.Conv2d(
            in_channel,
            widen_factor * 64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(widen_factor * 64)
        self.layer1 = self._make_layer(
            block, widen_factor * 64, num_blocks[0], stride=1
        )
        self.layer2 = self._make_layer(
            block, widen_factor * 128, num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, widen_factor * 256, num_blocks[2], stride=2
        )
        self.layer4 = self._make_layer(
            block, widen_factor * 512, num_blocks[3], stride=2
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


class ThinResNet(nn.Module):
    def __init__(
        self, block, num_blocks, in_channel=3, zero_init_residual=False, widen_factor=2
    ):
        super(ThinResNet, self).__init__()

        self.in_planes = 64 // widen_factor

        self.conv1 = nn.Conv2d(
            in_channel,
            64 // widen_factor,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64 // widen_factor)
        self.layer1 = self._make_layer(
            block, 64 // widen_factor, num_blocks[0], stride=1
        )
        self.layer2 = self._make_layer(
            block, 128 // widen_factor, num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, 256 // widen_factor, num_blocks[2], stride=2
        )
        self.layer4 = self._make_layer(
            block, 512 // widen_factor, num_blocks[3], stride=2
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet20(**kwargs):
    return ResNet(BasicBlock, [3, 3, 3, ], **kwargs)

def resnet20_slim(**kwargs):
    return ResNet(BasicBlock, [3, 3, 3, ], in_planes=16, **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def wide_resnet50_2(**kwargs):
    return WideResNet(Bottleneck, [3, 4, 6, 3], widen_factor=2, **kwargs)


def thin_resnet50_2(**kwargs):
    return ThinResNet(Bottleneck, [3, 4, 6, 3], widen_factor=2, **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


from .cifar_resnet import resnet as cifar_resnet


def resnet110(**kwargs):
    return cifar_resnet(depth=110, num_classes=10)


model_dict = {
    'resnet18': [resnet18, 512],
    "resnet20_slim": [resnet20_slim, 64],
    "resnet34": [resnet34, 512],
    "resnet50": [resnet50, 2048],
    "wide_resnet50_2": [wide_resnet50_2, 2 * 2048],
    "thin_resnet50_2": [thin_resnet50_2, 2048 // 2],
    "resnet101": [resnet101, 2048],
    "resnet110": [resnet110, 64],
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""

    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x

class EncWrapper(nn.Module):
    def __init__(self, module:nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self,*args,**kwargs):
        return self.module.forward(*args,**kwargs)

class SupConResNet(nn.Module):
    """backbone + projection head"""

    def __init__(
        self,
        name="resnet50",
        head="mlp",
        feat_dim=None,
        multi_head=False,
        num_classes=10,
        wrapper=False,
    ):
        super(SupConResNet, self).__init__()
        if feat_dim is None:
            feat_dim = 128
            if name == "resnet110" or name == "resnet20_slim":
                feat_dim = 64

        model_fun, dim_in = model_dict[name]
        if wrapper:
            self.encoder = EncWrapper(model_fun())
        else:
            self.encoder = model_fun()
        # self.encoder.module = model_fun()
        if head == 'linear' or name == "resnet20_slim":
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat_projected = F.normalize(self.head(feat), dim=1)
        # if self.multi_head:
        #     return feat_projected, self.head2(feat)
        # else:
        #     return feat_projected
        return feat_projected


class SupCEResNet(nn.Module):
    """encoder + classifier"""

    def __init__(self, name="resnet50", num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


class LinearClassifier(nn.Module):
    """Linear classifier"""

    def __init__(self, name="resnet50", num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)

if __name__ == '__main__':
    net = SupConResNet('resnet20_slim')
    print(net)