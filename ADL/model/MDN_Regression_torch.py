import torch
from torch import nn
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_size,
            out_size,
            kernel_size,
            padding=padding,
            stride=stride
        )
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class MDN_Regression(nn.Module):
    def __init__(self, sizes, p, drop_out=0.0):
        super().__init__()
        self.sizes = sizes
        n_channels = 6
        self.down_1 = nn.Sequential(
            ConvBlock(n_channels, 16),
            ConvBlock(16, 16))
        self.down_2 = nn.Sequential(
            ConvBlock(16, 32),
            ConvBlock(32, 32))
        self.down_3 = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 64))
        self.down_4 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128))
        self.end_of_layer = nn.Sequential(
            nn.MaxPool2d(2, 2),
        )

        layers = []
        layers.append(self.down_1)
        layers.append(self.end_of_layer)
        layers.append(self.down_2)
        layers.append(self.end_of_layer)
        layers.append(self.down_3)
        layers.append(self.end_of_layer)
        layers.append(self.down_4)
        layers.append(self.end_of_layer)
        layers.append(nn.Flatten())
        for i in range(1, len(sizes)-1):
            if i > 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p))
            layers.append(nn.Linear(self.sizes[i-1], self.sizes[i], bias=True))
            layers.append(nn.BatchNorm1d(self.sizes[i]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p))
        self.layers = nn.Sequential(*layers)
        self.layers.apply(self.init_weights)
        self.pi = nn.Linear(self.sizes[-2], self.sizes[-1])
        self.init_weights(self.pi)
        self.mu = nn.Linear(self.sizes[-2], self.sizes[-1])
        self.init_weights(self.mu)
        self.sigma = nn.Linear(self.sizes[-2], self.sizes[-1])
        self.init_weights(self.sigma)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.n_gauss = self.sizes[-1]

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def forward(self, x, train_mode: bool = True):
        self.train(train_mode)
        x = self.layers(x)
        pi = nn.functional.gumbel_softmax(self.pi(x), tau=1, dim=-1) + 1e-10
        mu = self.relu(self.mu(x))
        sigma = self.elu(self.sigma(x)) + 1 + 1e-15
        return pi, mu, sigma

    def run_epoch(self, dataloader, optimizer=None, scheduler=None, loss_f=None, device="cpu"):
        train_mode = (optimizer is not None) and (loss_f is not None)
        true, pi, mu, sigma = [], [], [], []

        for i, data in enumerate(dataloader):

            img, y = data
            img = img.to(device)
            y = y.to(device)

            if train_mode:
                optimizer.zero_grad()
            batch_pi, batch_mu, batch_sigma = self(img, train_mode)
            if train_mode:
                loss = loss_f(y, batch_pi, batch_mu, batch_sigma)
                loss.backward()
                optimizer.step()
            true.append(y.cpu().detach().numpy())
            pi.append(batch_pi.cpu().detach().numpy())
            mu.append(batch_mu.cpu().detach().numpy())
            sigma.append(batch_sigma.cpu().detach().numpy())

        if train_mode:
            scheduler.step()
        true = np.concatenate(true)
        pi = np.vstack(pi)
        mu = np.vstack(mu)
        sigma = np.vstack(sigma)
        return true, pi, mu, sigma
