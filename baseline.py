import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import pandas as pd

import torchvision
from PIL import Image


class YourName(torch.utils.data.Dataset):
    def __init__(self,
                 resolution=142,
                 root_dir="/data/png/p142/data",
                 transform=None,
                 train=True):
        """
        Args:
            resolution (int): Path to the csv file with annotations.
            root_dir   (string): Directory with all the images.
            transform  (callable, optional): Optional transform to be applied
                on a sample.
            start      (int): Starting number in file naming convention.
            stop       (int): Ending number in file naming convention.
            eps        (int): Threshold for minimum pixel-wise difference.
        """
        self.train = train
        self.root_dir = root_dir
        self.path = self.root_dir

        self.non_identical_images = self.identify_unique_images()

        self.len = len(self.non_identical_images)

        if transform == None:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((128, 128)),
                torchvision.transforms.RandomCrop(128),
                torchvision.transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        if self.train == True:
            return self.len - 14196
        else:
            return 14196

    def __getitem__(self, index):
        if self.train == True:
            i = index + 14196
        else:
            i = index

        index = self.non_identical_images[i]

        # Get path with image index
        image_path = "{}/frame_{:06d}.png".format(self.path, index)

        # Open image at path
        image = Image.open(image_path).convert('RGB')

        # Transform image
        image = self.transform(image)

        # Set adjacent first and third frame as input
        X = image

        return X

    def identify_unique_images(self):
        df = pd.read_csv("./unique-images.csv")

        return df["frame"]


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


import os
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(batch_size)
    elif distribution == 'gaussian':
        x_recon = torch.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum').div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
