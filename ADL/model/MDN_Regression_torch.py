import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List
import matplotlib
from matplotlib import pyplot as plt
from IPython.display import clear_output
import pickle
import os


class ConvBlock(nn.Module):
    """ConvBlock."""

    def __init__(self, in_size, out_size, kernel_size=3, padding=1, stride=1):
        """__init__.

        :param in_size:
        :param out_size:
        :param kernel_size:
        :param padding:
        :param stride:
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_size, out_size, kernel_size, padding=padding, stride=stride
        )
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """forward.

        :param x:
        """
        return self.relu(self.bn(self.conv(x)))


class MDN_Regression(nn.Module):
    """MDN_Regression."""

    def __init__(
        self,
        sizes: List[int] = [512, 256, 128, 1],
        p: float = 0.0,
        drop_out: float = 0.0,
    ):
        """__init__.

        :param sizes:
        :param p:
        :param drop_out:
        """
        super().__init__()
        self.sizes = sizes
        n_channels = 6
        self.down_1 = nn.Sequential(ConvBlock(n_channels, 16), ConvBlock(16, 16))
        self.down_2 = nn.Sequential(ConvBlock(16, 32), ConvBlock(32, 32))
        self.down_3 = nn.Sequential(ConvBlock(32, 64), ConvBlock(64, 64))
        self.down_4 = nn.Sequential(ConvBlock(64, 128), ConvBlock(128, 128))
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
        for i in range(1, len(sizes) - 1):
            if i > 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p))
            layers.append(nn.Linear(self.sizes[i - 1], self.sizes[i], bias=True))
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
        """init_weights.

        :param m:
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

    def forward(self, x, train_mode: bool = True):
        """forward.

        :param x:
        :param train_mode:
        :type train_mode: bool
        """
        self.train(train_mode)
        x = self.layers(x)
        pi = nn.functional.gumbel_softmax(self.pi(x), tau=1, dim=-1) + 1e-10
        mu = self.relu(self.mu(x))
        sigma = self.elu(self.sigma(x)) + 1 + 1e-15
        return pi, mu, sigma

    def run_epoch(
        self, dataloader, optimizer=None, scheduler=None, loss_f=None, device="cpu"
    ):
        """run_epoch.

        :param dataloader:
        :param optimizer:
        :param scheduler:
        :param loss_f:
        :param device:
        """
        train_mode = (optimizer is not None) and (loss_f is not None)
        gt, pi, mu, sigma = [], [], [], []

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
            gt.append(y.cpu().detach().numpy())
            pi.append(batch_pi.cpu().detach().numpy())
            mu.append(batch_mu.cpu().detach().numpy())
            sigma.append(batch_sigma.cpu().detach().numpy())

        if train_mode:
            scheduler.step()
        gt = np.concatenate(gt)
        pi = np.vstack(pi)
        mu = np.vstack(mu)
        sigma = np.vstack(sigma)
        return gt, pi, mu, sigma


class DeepEnsemble_MDN:
    """DeepEnsemble_MDN."""

    def __init__(
        self,
        BaseModel=MDN_Regression,
        base_model_args: Dict = {},
        n_models: int = 1,
        device: str = "cpu",
        model_save_path: str = None,
        optimizer=torch.optim.Adam,
        optimizer_args={"lr": 0.0005, "weight_decay": 0.0001},
        scheduler=torch.optim.lr_scheduler.ExponentialLR,
        scheduler_args={"gamma": 0.9},
    ):
        """__init__.

        :param BaseModel:
        :param base_model_args:
        :type base_model_args: Dict
        :param n_models:
        :type n_models: int
        :param device:
        :type device: str
        :param model_save_path:
        :type model_save_path: str
        """
        self.BaseModel = BaseModel
        self.base_model_args = base_model_args
        self.n_models = n_models
        self.device = device
        self.models = []
        self.model_save_path = model_save_path
        self.epochs = 0
        self.epoch = 0
        for i in range(n_models):
            self.models.append(BaseModel(**base_model_args).to(self.device))
        optimizers = []
        schedulers = []
        for model in self.models:
            optimizers.append(optimizer(model.parameters(), **optimizer_args))
            schedulers.append(scheduler(optimizers[-1], **scheduler_args))
        self.optimizers = optimizers
        self.schedulers = schedulers

        self.metric_vals = {}
        self.loss_vals = {}
        self.loss_min = np.inf
        self.loss_max = -np.inf

    @staticmethod
    def loss(y, pi, mu, sigma):
        """loss.

        :param y:
        :param pi:
        :param mu:
        :param sigma:
        """
        if not isinstance(y, torch.Tensor):
            y, pi, mu, sigma = (
                torch.Tensor(y),
                torch.Tensor(pi),
                torch.Tensor(mu),
                torch.Tensor(sigma),
            )
        comp_prob = (
            -torch.log(sigma)
            - 0.5 * np.log(2 * np.pi)
            - 0.5 * torch.pow((y.view(-1, 1) - mu) / sigma, 2)
        )
        mix = torch.log(pi)
        res = torch.logsumexp(comp_prob + mix, dim=-1)
        return torch.mean(-res)

    def run_models_one_epoch(
        self, dataloader: DataLoader, train_mode: bool = True, mode_name: str = "train"
    ):
        """run_models_one_epoch.

        :param dataloader:
        :type dataloader: DataLoader
        :param train_mode:
        :type train_mode: bool
        :param mode_name:
        :type mode_name: str
        """
        epoch_pi, epoch_mu, epoch_sigma = [], [], []
        epoch_losses = []
        for i, model in enumerate(self.models):
            (
                epoch_true,
                pi,
                mu,
                sigma,
            ) = model.run_epoch(
                dataloader,
                self.optimizers[i] if train_mode else None,
                self.schedulers[i] if train_mode else None,
                self.loss if train_mode else None,
                device=self.device,
            )
            epoch_pi.append(pi)
            epoch_mu.append(mu)
            epoch_sigma.append(sigma)
            epoch_losses.append(self.loss(epoch_true, pi, mu, sigma).item())
        epoch_pi = np.concatenate(epoch_pi, axis=1) / len(self.models)
        epoch_mu = np.concatenate(epoch_mu, axis=1)
        epoch_sigma = np.concatenate(epoch_sigma, axis=1)
        epoch_p = (1 / (epoch_sigma * np.sqrt(2 * np.pi))) * epoch_pi
        mode = epoch_mu[np.arange(epoch_mu.shape[0]), np.argmax(epoch_p, axis=1)]
        mu = np.sum(epoch_mu * epoch_pi, axis=1)
        sigma = np.sum(epoch_sigma * epoch_pi, axis=1) + np.sum(
            (epoch_mu - mu.reshape(-1, 1)) ** 2 * epoch_pi, axis=1
        )

        self.loss_vals[mode_name].append(epoch_losses)
        for metric in self.metrics:
            self.metric_vals[mode_name][metric].append(
                self.metrics[metric](epoch_true, mode)
            )

    def show_loss(
        self,
        ax: matplotlib.axes,
        epoch: int,
        first_idx: int = 0,
        show_min: str = None,
        lower_ylim: int = 1,
    ):
        """show_loss.

        :param ax:
        :type ax: matplotlib.axes
        :param epoch:
        :type epoch: int
        :param first_idx:
        :type first_idx: int
        :param show_min:
        :type show_min: str
        :param lower_ylim:
        :type lower_ylim: int
        """
        ticks = list(range(1, epoch + 2))[first_idx:]

        y_min = self.loss_min
        y_max = self.loss_max
        for mode_name, c in zip(self.dataloaders, ["blue", "orange", "purple"]):
            losses_min = list(map(np.min, self.loss_vals[mode_name]))[first_idx:]
            losses_mean = list(map(np.mean, self.loss_vals[mode_name]))[first_idx:]
            losses_max = list(map(np.max, self.loss_vals[mode_name]))[first_idx:]
            ax.plot(ticks, losses_min, c=c, linestyle="--", alpha=0.5)
            ax.plot(ticks, losses_mean, c=c, label=mode_name)
            ax.plot(ticks, losses_max, c=c, linestyle="--", alpha=0.5)
            y_min = min(y_min, *losses_min)
            y_max = max(y_max, *losses_max)
            if show_min == mode_name:
                global_min_loss = np.argmin(losses_min)
                ax.axvline(
                    global_min_loss,
                    label=f"Loss min on {mode_name} on epoch {global_min_loss}",
                )

        y_min -= 0.3 * np.abs(y_min)
        y_min = min(y_min, 0)
        y_max += 0.3 * np.abs(y_max)
        self.loss_min = min(self, self.loss_min, y_min)
        self.loss_max = max(self, self.loss_max, y_max)
        if lower_ylim is not None:
            y_max = np.max(self.loss_vals["train"][min(lower_ylim, epoch)][0])
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(ticks[::10])
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.legend(loc=0, fontsize=12)
        ax.grid("on")

    def show_metrics(self, ax: matplotlib.axes, epoch: int, first_idx: int = 0):
        """show_metrics.

        :param ax:
        :type ax: matplotlib.axes
        :param epoch:
        :type epoch: int
        :param first_idx:
        :type first_idx: int
        """
        ticks = list(range(1, epoch + 2))[first_idx:]
        colors = ["blue", "orange", "red", "green", "black", "purple"]

        for mode_name, linestyle in zip(self.dataloaders, ["-", "--", ":"]):
            for i, metric in enumerate(self.metrics):
                ax.plot(
                    ticks,
                    self.metric_vals[mode_name][metric][first_idx:],
                    c=colors[i],
                    label=f"{mode_name} {metric}",
                    linestyle=linestyle,
                )

        t = []
        for mode_name in self.dataloaders:
            for metric in self.metrics:
                t += self.metric_vals[mode_name][metric][first_idx:]
        y_min, y_max = min(t), max(t)
        y_min -= 0.3 * np.abs(y_min)
        y_max += 0.3 * np.abs(y_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(ticks[::10])
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel("Metric", fontsize=12)
        ax.legend(loc=0, fontsize=12)
        ax.grid("on")

    def show_verbose(self):
        """show_verbose."""
        epoch = self.epoch
        clear_output(True)
        print(f"Device: {self.device}")
        print("=" * 40)
        print(f"EPOCH #{epoch + 1}/{self.epochs}:")
        cur_lr = self.schedulers[0].get_last_lr()
        print(f"Learning rate: {round(cur_lr[0], 8)}")
        print("-" * 40)

        for mode_name in self.dataloaders:
            print(
                f"{mode_name} losses: {[round(l, 5) for l in self.loss_vals[mode_name][-1]]}"
            )
            print(
                f"AVG {mode_name} loss: {round(np.mean(self.loss_vals[mode_name][-1]), 5)}"
            )
            for metric in self.metrics:
                print(
                    f"{mode_name} {metric}: {round(self.metric_vals[mode_name][metric][-1], 5)}\t",
                    end="",
                )
            print()
            print("-" * 40)

        # GRAPHICS
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
        self.show_loss(ax[0], epoch)
        self.show_metrics(ax[1], epoch)

        plt.show()

    def save_metrics(self):
        self.metric_vals.save_pickle(os.path.join(self.model_save_path, "metrics.pkl"))
        self.loss_vals.save_pickle(os.path.join(self.model_save_path, "loss.pkl"))

    def fit(
        self,
        dataloaders: Dict[str, DataLoader],
        epochs: int = 10,
        verbose: bool = True,
        metrics=[],
    ):
        """fit.

        :param dataloader:
        :type dataloader: DataLoader
        :param val_dataloader:
        :type val_dataloader: DataLoader
        :param test_dataloader:
        :type test_dataloader: DataLoader
        :param epochs:
        :type epochs: int
        :param optimizer:
        :param optimizer_args:
        :param scheduler:
        :param scheduler_args:
        :param verbose:
        :type verbose: bool
        :param metrics:
        """
        if "train" not in dataloaders:
            raise (ValueError("Need train dataloader."))
        self.dataloaders = dataloaders
        self.epochs += epochs
        self.metrics = metrics

        for mode_name in dataloaders:
            if mode_name not in self.metric_vals:
                self.metric_vals[mode_name] = {metric: [] for metric in metrics}
            if mode_name not in self.loss_vals:
                self.loss_vals[mode_name] = []

        for self.epoch in range(self.epoch, self.epochs):
            for mode_name, dataloader in dataloaders.items():
                self.run_models_one_epoch(
                    dataloader, train_mode="train" == mode_name, mode_name=mode_name
                )

            if verbose and self.epoch > 0:
                self.show_verbose()
            if self.model_save_path is not None:
                self.save_pickle(
                    os.path.join(
                        self.model_save_path, "ens_ep{}.pkl".format(self.epoch)
                    )
                )
        self.save_metrics()

    def predict(self, dataloader: DataLoader):
        """predict.

        :param dataloader:
        :type dataloader: DataLoader
        """
        epoch_pi, epoch_mu, epoch_sigma = [], [], []

        for i, model in enumerate(self.models):
            gt, pi, mu, sigma = model.run_epoch(dataloader, device=self.device)
            epoch_pi.append(pi)
            epoch_mu.append(mu)
            epoch_sigma.append(sigma)
        epoch_pi = np.concatenate(epoch_pi, axis=1) / len(self.models)
        epoch_mu = np.concatenate(epoch_mu, axis=1)
        epoch_sigma = np.concatenate(epoch_sigma, axis=1)
        epoch_p = (1 / (epoch_sigma * np.sqrt(2 * np.pi))) * epoch_pi
        mode = epoch_mu[np.arange(epoch_mu.shape[0]), np.argmax(epoch_p, axis=1)]
        mu = np.sum(epoch_mu * epoch_pi, axis=1)
        sigma = np.sum(epoch_sigma * epoch_pi, axis=1) + np.sum(
            (epoch_mu - mu.reshape(-1, 1)) ** 2 * epoch_pi, axis=1
        )

        return gt, epoch_pi, epoch_mu, epoch_sigma, mode, sigma

    def save_pickle(self, file: str):
        """save_pickle.

        :param file:
        :type file: str
        """
        with open(file, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_pickle(cls, file: str):
        """load_pickle.

        :param file:
        :type file: str
        """
        with open(file, "rb") as f:
            obj = pickle.load(f)
        return obj

    @classmethod
    def calc_err(
        cls, mu: np.ndarray, sigma: np.ndarray, mode: np.ndarray, n_models: int
    ):
        sigma_a2 = 0
        sigma_e2 = 0
        for n in range(n_models):
            epsilon = np.random.normal(0, sigma[:, n])
            f_ = mu[:, n]
            y = mu[:, n] + epsilon
            var = np.sum(epsilon)
            sigma_a2 += sigma[:, n] ** 2
            sigma_e2 += (f_ - 1 / 5 * np.sum(mu, axis=1)) ** 2
        sigma_2 = 1 / 5 * np.sum(sigma_a2) + sigma_e2
        return [np.sqrt(1 / 5 * sigma_a2), np.sqrt(1 / 5 * sigma_e2)]

    def compare_M500(
        self, ax: matplotlib.axes, dataloader: DataLoader, label: str = ""
    ):
        epoch_gt, epoch_pi, epoch_mu, epoch_sigma, mode, sigma = self.predict(
            dataloader
        )

        err_y = self.calc_err(epoch_mu, epoch_sigma, mode, self.n_models)

        ax.errorbar(
            epoch_gt,
            mode,
            yerr=np.sqrt(err_y[0] ** 2 + err_y[1] ** 2),
            xerr=0,
            fmt="ro",
            alpha=0.2,
        )
        s = ax.scatter(x=epoch_gt, y=mode, alpha=0.8)
        s.set_label(label)

        ax.legend()
        ax.set_xlabel("$M_{500}\cdot10^{14} M_\odot$ $from$ $PSZ2$")
        ax.set_ylabel("$Predicted$ $M_{500}\cdot10^{14} M_\odot$")
        ax.plot(
            np.linspace(0, 13, 100),
            np.linspace(0, 13, 100),
            "--",
            color="gray",
            alpha=0.8,
        )
        ax.set_ylim(-0.5, 14)
        ax.set_xlim(0, 14)
        ax.set_xticks(np.arange(15))
        ax.grid(which="minor", linestyle=":")
        ax.grid(which="major", linestyle="--")


def sigma_nmad(true, preds):
    diff = preds - true
    m = np.median(diff)
    return 1.48 * np.median(np.abs((diff - m) / (1 + true)))
