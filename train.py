# This code is an unofficial implementation of the following paper:
# "Prevalence of neural collapse during the terminal phase of deep learning training" by Papyan, V., Han, X. Y., & Donoho, D. L. (2020). Proceedings of the National Academy of Sciences, 117(40), 24652-24663.
# CIFAR-10 and CIFAR-100 datasets results are reproduced.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50
import argparse
import os


def adjust_learning_rate(optimizer, epoch, total_epochs):
    if epoch == total_epochs // 3 or epoch == 2 * total_epochs // 3:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= 0.1
    return optimizer.param_groups[0]["lr"]


def train(
    epoch, model, train_loader, optimizer, criterion, device, total_epochs
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for _, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        targets_tmp = targets.clone()
        loss = criterion(outputs, targets_tmp)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    current_lr = adjust_learning_rate(optimizer, epoch, total_epochs)

    print(
        f"Epoch {epoch} - Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}%, LR: {current_lr}"
    )


def validate(epoch, model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            targets_tmp = targets.clone()

            loss = criterion(outputs, targets_tmp)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / len(test_loader)
    val_acc = 100.0 * correct / total

    print(
        f"Epoch {epoch} - Validation Loss: {val_loss:.3f}, Validation Acc: {val_acc:.3f}%"
    )


def save_checkpoint(
    model, epoch, checkpoint_dir="checkpoints", basename="resnet18"
):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(
        checkpoint_dir, f"{basename}_{epoch:03d}.pth"
    )
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


def get_args():
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Train ResNet18 on CIFAR-10 with specified initial learning rate and number of epochs."
    )

    parser.add_argument(
        "--epochs", type=int, default=350, help="Number of epochs to train"
    )
    parser.add_argument(
        "--epoch_for_logs",
        type=int,
        default=5,
        help="Interval for saving checkpoints",
    )
    parser.add_argument(
        "--save_checkpoint",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        help="Dataset to train the model on",
    )

    args = parser.parse_args()

    # best lr from 25 experiments according to paper
    if args.dataset == "CIFAR10":
        args.init_lr = 0.1302501827396728
    elif args.dataset == "CIFAR100":
        args.init_lr = 0.18045095091164853

    return args


def main():
    args = get_args()

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations for the training and test data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    # Load CIFAR-10 dataset
    if args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    elif args.dataset == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform
        )

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2
    )

    # accoring to paper sec 2.F.
    if args.dataset == "CIFAR10":
        model = resnet18(weights=None, num_classes=10)
        checkpoints = "checkpoints_CIFAR10"
        basename = "resnet18"
    elif args.dataset == "CIFAR100":
        model = resnet50(weights=None, num_classes=100)
        checkpoints = "checkpoints_CIFAR100"
        basename = "resnet50"

    model = model.to(device)

    # Define loss function and optimizer with momentum and weight decay
    criterion = nn.CrossEntropyLoss()

    # parameters from paper sec 2.G.
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.init_lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    # Training loop
    for epoch in range(
        1, args.epochs + 1
    ):  # Training for specified number of epochs
        train(
            epoch,
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.epochs,
        )
        validate(epoch, model, test_loader, criterion, device)

        # Save checkpoints at the specified interval, save the first epoch always
        if epoch % args.epoch_for_logs == 0 or epoch == 1:
            if args.save_checkpoint:
                save_checkpoint(model, epoch, checkpoints, basename)


if __name__ == "__main__":
    main()
