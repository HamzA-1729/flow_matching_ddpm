"""
Configurable Dataset Factory:
Supports MNIST, Fashion-MNIST, CIFAR-10, and CelebA with automatic normalization to [-1.0, 1.0].
"""

import os
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(
    name: str = "mnist",
    img_size: int = 28,
    in_channels: int = 1,
    batch_size: int = 64,
    data_dir: str = "./data",
    num_workers: int = 2,
    train: bool = True,
) -> DataLoader:
    """
    Build a DataLoader for the requested dataset with standard [-1, 1] normalization.

    Args:
        name: Name of the dataset ('mnist', 'fashion_mnist', 'cifar10', 'celeba').
        img_size: Target square spatial resolution (e.g. 28, 32, or 64).
        in_channels: Number of image channels (1 for grayscale, 3 for RGB).
        batch_size: Batch size per iteration.
        data_dir: Path to directory for caching raw dataset files.
        num_workers: Subprocesses for data loading.
        train: Whether to load training split (True) or test split (False).

    Returns:
        dataloader: Configured PyTorch DataLoader yielding (images, labels).
    """
    os.makedirs(data_dir, exist_ok=True)
    name = name.lower()

    # Normalization mean & std for mapping [0, 1] -> [-1, 1]
    norm_mean = (0.5,) * in_channels
    norm_std = (0.5,) * in_channels

    transform_list = [
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size) if name == "celeba" else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ]

    transform = transforms.Compose(transform_list)

    if name == "mnist":
        dataset = datasets.MNIST(
            root=data_dir,
            train=train,
            download=True,
            transform=transform,
        )
    elif name == "fashion_mnist":
        dataset = datasets.FashionMNIST(
            root=data_dir,
            train=train,
            download=True,
            transform=transform,
        )
    elif name == "cifar10":
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=train,
            download=True,
            transform=transform,
        )
    elif name == "celeba":
        # CelebA: 64x64 celebrity faces (downloads from official mirror or uses local folder)
        split = "train" if train else "valid"
        dataset = datasets.CelebA(
            root=data_dir,
            split=split,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(
            f"Unsupported dataset: {name}. Supported: 'mnist', 'fashion_mnist', 'cifar10', 'celeba'"
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )

    return dataloader
