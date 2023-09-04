from dataclasses import dataclass

import simple_parsing
from simple_parsing import ArgumentParser


@dataclass
class GeneralParams:
    name: str = ""
    # Default directory to save logs and checkpoints, use current time if empty
    console: bool = True  # Whether to output logs to console
    tensorboard: bool = True  # Whether to use tensorboard
    load: bool = False  # Whether to load model
    test: bool = False  # Whether to test model

    cuda: bool = True  # Whether to use CUDA
    num_workers: int = 4  # Number of workers for dataloader
    gpu: str = "0"  # Which GPU to use
    visualize: bool = False  # Whether to visualize kernels


@dataclass
class DatasetParams:
    dataset: str = simple_parsing.field(
        default="CIFAR10", choices=["CIFAR100", "CIFAR10"]
    )
    image_size: int = 32  # Size of images


@dataclass
class TrainerParams:
    batch_size: int = 128  # Batch size, 0 for all
    optimizer: str = simple_parsing.field(
        default="SGD", choices=["SGD", "AdamW"]
    )  # Optimizer name
    lr: float = 1e-2  # Learning rate
    lr_morpho: float = 1  # Ratio of learning rate for morphological operations
    max_epochs: int = 400  # Maximum number of epochs
    scheduler: str = "CosineAnnealingLR"
    amp: bool = True  # Whether to use automatic mixed precision


@dataclass
class ModelParams:
    model: str = simple_parsing.field(
        default="MobileNetV2-31",
        choices=["MobileNetV2-31", "ResNet20"],
    )  # Feature extractor Model name
    morpho: str = simple_parsing.field(
        default="infinity",
        choices=["none", "dilation", "infinity"],
    )  # Whether to ablate morphological operations
    expansion: int = 8  # Expansion ratio for MobileNetV2
    morpho_init: str = simple_parsing.field(
        default="normal",
        choices=["normal", "uniform"],
    )
    relu: bool = False  # Whether to use ReLU


class Options:
    def __init__(self):
        self.parser = ArgumentParser()
        self.parser.add_arguments(dataclass=GeneralParams, dest="general_params")
        self.parser.add_arguments(dataclass=DatasetParams, dest="dataset_params")
        self.parser.add_arguments(dataclass=TrainerParams, dest="trainer_params")
        self.parser.add_arguments(dataclass=ModelParams, dest="model_params")

    def parse_args(self):
        return self.parser.parse_args()

    def default(self):
        return self.parser.default()


if __name__ == "__main__":
    Options().parser.print_help()
