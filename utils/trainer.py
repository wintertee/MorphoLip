import os
import pprint
import random
from dataclasses import dataclass
from datetime import datetime
from logging import Logger

import numpy as np
import torch
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchmetrics import MeanMetric
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm

from models.mobilenetv2 import InvertedResidual, MobileNetV2
from models.resnet import resnet20

from . import utils
from .options import DatasetParams, GeneralParams, ModelParams, TrainerParams


@dataclass
class Trainer:
    dataset_params: DatasetParams
    general_params: GeneralParams
    trainer_params: TrainerParams
    model_params: ModelParams

    def __post_init__(self):
        self._init_random_seed()
        self.use_logger = not (
            self.general_params.test or self.general_params.visualize
        )
        self.logger, self.writer = self._init_logger()
        self.check_params()
        self.device, self.scaler = self._init_backend()

        (
            self.train_dataset,
            self.valid_dataset,
            self.test_dataset,
            self.num_classes,
            self.num_channels,
        ) = self._init_dataset()
        (
            self.train_dataloader,
            self.valid_dataloader,
            self.test_dataloader,
        ) = self._init_dataloader()

        self.model = self._init_model().to(self.device)
        self.logger.info(
            summary(
                self.model,
                input_size=(self.trainer_params.batch_size, 3, 32, 32),
                col_names=["kernel_size", "input_size", "output_size", "num_params"],
                verbose=0,
            )
        )

        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.criterion = self._init_criterion().to(self.device)
        self.metric = self._init_metric().to(self.device)
        self.mean_criterion = MeanMetric().to(self.device)
        self.mean_metric = MeanMetric().to(self.device)
        self.start_epoch: int = 0
        self.epoch: int = 0

    def __del__(self):
        if self.writer:
            self.writer.close()

    def check_params(self):
        return

    def _init_random_seed(self, seed=41):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _init_logger(self) -> tuple[Logger, SummaryWriter | None]:
        if self.general_params.name == "":
            self.general_params.name = datetime.now().strftime("%Y%m%d_%H%M_%f")

        self.general_params.name = os.path.join(
            "runs", self.dataset_params.dataset, self.general_params.name
        )
        if not os.path.exists(self.general_params.name):
            os.makedirs(self.general_params.name)

        logger = utils.get_logger(
            self.general_params.name,
            self.general_params.console,
            self.use_logger,
        )
        logger.info(f"{pprint.pformat(self.general_params, indent=4)}\n")
        logger.info(f"{pprint.pformat(self.dataset_params, indent=4)}\n")
        logger.info(f"{pprint.pformat(self.trainer_params, indent=4)}\n")
        logger.info(f"{pprint.pformat(self.model_params, indent=4)}\n")

        writer = (
            SummaryWriter(self.general_params.name)
            if self.general_params.tensorboard and self.use_logger
            else None
        )
        return logger, writer

    def _init_backend(self) -> torch.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = self.general_params.gpu
        if self.general_params.cuda:
            if torch.cuda.is_available():
                device = torch.device("cuda")

            else:
                raise ValueError("No GPU found!")
        else:
            device = torch.device("cpu")
        self.logger.info("Using %s backend\n", device)
        scaler = torch.cuda.amp.GradScaler(enabled=self.trainer_params.amp)
        return device, scaler

    def _init_dataset(self):
        if self.dataset_params.dataset == "CIFAR10":
            num_classes = 10
            num_channels = 3
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            trainset = CIFAR10(
                root="./data",
                train=True,
                transform=train_transform,
                download=True,
            )
            validset = CIFAR10(
                root="./data",
                train=False,
                transform=test_transform,
                download=True,
            )
            testset = validset

        elif self.dataset_params.dataset == "CIFAR100":
            num_classes = 100
            num_channels = 3
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            trainset = CIFAR100(
                root="./data",
                train=True,
                transform=train_transform,
                download=True,
            )
            validset = CIFAR100(
                root="./data",
                train=False,
                transform=test_transform,
                download=True,
            )
            testset = validset
        else:
            raise NotImplementedError

        return trainset, validset, testset, num_classes, num_channels

    def _init_dataloader(self):
        batch_size = self.trainer_params.batch_size

        if self.device.type == "cuda":
            pin_memory = True
            persistent_workers = self.general_params.num_workers > 0
        else:
            pin_memory = False
            persistent_workers = False

        if self.general_params.test:
            train_loader = None
            valid_loader = None
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.general_params.num_workers,
                persistent_workers=persistent_workers,
                pin_memory=pin_memory,
            )
        else:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.general_params.num_workers,
                persistent_workers=persistent_workers,
                pin_memory=pin_memory,
            )
            valid_loader = DataLoader(
                self.valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.general_params.num_workers,
                persistent_workers=persistent_workers,
                pin_memory=pin_memory,
            )
            test_loader = None

        return train_loader, valid_loader, test_loader

    def _init_model(self):
        if self.model_params.model == "MobileNetV2-31":
            if self.model_params.relu:
                activation_layer = nn.ReLU
            else:
                activation_layer = nn.Identity
            model = MobileNetV2(
                num_classes=self.num_classes,
                inverted_residual_setting=[
                    # t, c, n, s
                    [self.model_params.expansion, 16, 3, 1],
                    [self.model_params.expansion, 32, 3, 2],
                    [self.model_params.expansion, 64, 3, 2],
                ],
                first_layer_stride=1,
                input_channel=16,
                last_channel=64,
                block=InvertedResidual,
                morpho=self.model_params.morpho,
                morpho_init=self.model_params.morpho_init,
                activation_layer=activation_layer,
            )

        elif self.model_params.model == "ResNet20":
            model = resnet20(num_classes=self.num_classes)
        else:
            raise NotImplementedError
        return model

    def _init_optimizer(self):
        se_params = [p for name, p in self.model.named_parameters() if name == "se"]
        other_params = [p for name, p in self.model.named_parameters() if name != "se"]

        if self.trainer_params.optimizer == "SGD":
            return torch.optim.SGD(
                [
                    {
                        "params": se_params,
                        "lr": self.trainer_params.lr * self.trainer_params.lr_morpho,
                    },
                    {
                        "params": other_params,
                        "lr": self.trainer_params.lr,
                    },
                ],
                momentum=0.9,
                weight_decay=5e-4,
            )
        elif self.trainer_params.optimizer == "AdamW":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.trainer_params.lr,
            )
        else:
            raise NotImplementedError

    def _init_scheduler(self):
        if self.trainer_params.scheduler == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.trainer_params.max_epochs,
                eta_min=0.0,
            )
        else:
            raise NotImplementedError

    def _init_criterion(self):
        return nn.CrossEntropyLoss()

    def _init_metric(self):
        return torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)

    def save(self):
        checkpoint = {
            "net": self.model.state_dict(),
            "epoch": self.epoch,
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(self.general_params.name, "model.ckpt"))

    def load(self):
        path = os.path.join(self.general_params.name, "model.ckpt")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["net"])
        # self.optimizer.load_state_dict(ckpt["optimizer"])
        # self.start_epoch = ckpt["epoch"]

    def train(self) -> tuple[float, float]:
        inps: torch.Tensor
        tars: torch.Tensor

        self.model.train()

        with tqdm(
            self.train_dataloader,
            ncols=100,
            desc=f"Epoch {self.epoch}/{self.trainer_params.max_epochs}",
            unit="batch",
            leave=False,
        ) as t:
            for step, (inps, tars) in enumerate(t):
                self.optimizer.zero_grad()
                inps = inps.to(self.device)
                tars = tars.to(self.device)

                with torch.autocast(
                    device_type=self.device.type, enabled=self.trainer_params.amp
                ):
                    output = self.model(inps)
                    loss = self.criterion(output, tars)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # exit(0)

                self.mean_criterion.update(loss)
                self.mean_metric.update(self.metric(output, tars))

        loss_value = self.mean_criterion.compute().item()
        metric_value = self.mean_metric.compute().item()

        self.mean_criterion.reset()
        self.mean_metric.reset()
        return loss_value, metric_value

    def valid(self, test=False) -> float:
        inps: torch.Tensor
        tars: torch.Tensor

        self.model.eval()
        dataloader = self.test_dataloader if test else self.valid_dataloader

        with torch.no_grad():
            for step, (inps, tars) in enumerate(dataloader):
                inps = inps.to(self.device)
                tars = tars.to(self.device)

                with torch.autocast(
                    device_type=self.device.type, enabled=self.trainer_params.amp
                ):
                    output = self.model(inps)

                self.mean_metric.update(self.metric(output, tars))

        metric_value = self.mean_metric.compute().item()
        self.mean_metric.reset()
        return metric_value

    def train_model(self):
        best = 0.0
        for self.epoch in range(
            self.start_epoch + 1, self.trainer_params.max_epochs + 1
        ):
            with utils.Catchtime() as t:
                loss, train_acc = self.train()
                val_acc = self.valid()

            if self.writer:
                self.writer.add_scalar("loss", loss, self.epoch)
                self.writer.add_scalars(
                    "acc", {"train": train_acc, "val": val_acc}, self.epoch
                )

            self.logger.info(
                f"| epoch {self.epoch:3d} "
                f"| time {t.time:5.2f}s "
                f"| loss {loss:6.4f} "
                f"| train_acc {train_acc*100:5.2f} "
                f"| val_acc {val_acc*100:5.2f} "
                f"| lr {self.optimizer.param_groups[-1]['lr']:3.2e}"
            )
            self.scheduler.step()

            if val_acc > best:
                self.logger.info("| best\n")
                best = val_acc
                if self.epoch > self.trainer_params.max_epochs // 2 and self.use_logger:
                    self.save()
            else:
                self.logger.info("|\n")

    def test_model(self) -> float:
        metrics_value = self.valid(test=True)
        self.logger.info(f"test_acc {metrics_value:5.4f}\n")
        return metrics_value

    def visualize_kernel(self):
        import math

        import matplotlib.pyplot as plt

        if self.model_params.morpho == "dilation":

            def get_kernel(modules, layer_idx):
                return {"dilation": modules[0].features[layer_idx].dw_conv.se}

        else:
            raise NotImplementedError

        modules = [m for m in self.model.modules()]
        for layer_idx in range(10):
            kernel_dict = get_kernel(modules, layer_idx)
            for kernel_name, kernel in kernel_dict.items():
                kernel = kernel.detach().cpu().numpy()
                kernel_length = kernel.shape[0]  # 192
                print(kernel.shape)
                width = 8
                height = math.ceil(kernel_length / width)
                fig, axs = plt.subplots(height, width, figsize=(width, height))
                fig.suptitle(kernel_name)
                for i in range(height):
                    for j in range(width):
                        axs[i, j].imshow(
                            kernel[i * width + j], cmap="coolwarm", clim=(-0.1, 0.1)
                        )
                        axs[i, j].axis("off")
                plt.savefig(self.general_params.name + "/" + str(layer_idx) + ".png")
                plt.close()

            for kernel_name, kernel in kernel_dict.items():
                kernel = kernel.detach().cpu().numpy().flatten()
                plt.hist(kernel, alpha=0.5, bins=100, label=kernel_name)
            plt.legend(loc="upper right")
            plt.savefig(self.general_params.name + "/" + str(layer_idx) + "_dis.png")
            plt.close()

    def run(self):
        # entry point
        if self.general_params.load:
            self.load()
        if self.general_params.visualize:
            self.visualize_kernel()
            return
        if self.general_params.test is False:
            self.train_model()
        else:
            self.test_model()
