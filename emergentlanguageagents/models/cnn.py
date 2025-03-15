# This code is based on https://github.com/jayelm/emergent-generalization/blob/master/code/models/backbone/vision.py
# They in turn based their computer vision code on https://github.com/facebookresearch/low-shot-shrink-hallucinate

from importlib import resources # so that we can use pretrained resnet weights included in the package!
import pickle
import torch
import torch.nn as nn
from torchvision.models import resnet18
from emergentlanguageagents import assets

class ConvBlock(nn.Module):

    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
        self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.trunk = nn.Sequential([self.C, self.BN, self.relu, self.pool])

        self.reset_parameters()

    def forward(self, x):
        out = self.trunk(x)
        return out

    def reset_parameters(self):
        self.C.reset_parameters()
        self.BN.reset_parameters()

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Conv4(nn.Module):
    """
    4-layer convolutional image encoder, returning a flattened output.
    """
    def __init__(self, d_model = 64):
        super(Conv4, self).__init__()
        trunk = []
        for i in range(4):
            indim = 3 if i == 0 else d_model
            outdim = d_model
            trunk.append(ConvBlock(indim, outdim))

        trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)

        self.final_feat_dim = 1024 # TODO: make image size more flexible. final_feat_dim here is H * W * channels of final layer where channels = 64 and h/w are original divided by 16 because we do 4 max pool operations with kernel 2

    def forward(self, x):
        out = self.trunk(x)
        return out

    def reset_parameters(self):
        for layer in self.trunk:
            if isinstance(layer, ConvBlock):
                layer.reset_parameters()


class ResNet18(nn.Module):

    def __init__(self):
        super().__init__()
        self.final_feat_dim = 512 # FIXME: It's a bit gross to have to define final_feat_dim
        self.resnet18 = resnet18()
        # Cool new way to import static files from a package:
        asset_folder = resources.files(assets)
        state_dict_file = (
            asset_folder / '2025_resnet_imagenet_1k_pretrained_state_dict.pkl'
        )
        with state_dict_file.open('rb') as f:
            resnet_state_dict = pickle.load(f)
        self.resnet18.load_state_dict(resnet_state_dict)
        self.resnet18.fc = nn.Identity()
        for param in self.resnet18.parameters():
            param.requires_grad = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
            y_pred = self.resnet18(x)
            return y_pred.argmax(dim=1)

    def reset_parameters(self):
        # Never reset parameters
        pass