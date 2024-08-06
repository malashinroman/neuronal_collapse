# This code is an unofficial implementation of the following paper:
# "Prevalence of neural collapse during the terminal phase of deep learning training" by Papyan, V., Han, X. Y., & Donoho, D. L. (2020). Proceedings of the National Academy of Sciences, 117(40), 24652-24663.
# CIFAR-10 and CIFAR-100 datasets results are reproduced.

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50
import argparse
import os
from neuronal_collapse_lib import neuronal_collapse

import matplotlib.pyplot as plt


def process_checkpoint(args, checkpoint, train_loader, test_loader, device):
    if args.dataset == "CIFAR10":
        model = resnet18(weights=None, num_classes=10)
    elif args.dataset == "CIFAR100":
        model = resnet50(weights=None, num_classes=100)

    # Load the checkpoint
    if os.path.isfile(checkpoint):
        model.load_state_dict(torch.load(checkpoint))
        print(f"Checkpoint loaded from {checkpoint}")
    else:
        print(f"Checkpoint not found at {checkpoint}")
    return neuronal_collapse(model, train_loader, test_loader, device)


def plot_estimation_results(estimation_results, folder_name, dataset):
    # Extract epochs and all the metrics
    network_name = "ResNet"
    epochs = list(estimation_results.keys())
    metrics = {
        "Top1-train": [],
        "Top1-test": [],
        "Equinormality (class-means)": [],
        "Equinormality (weights)": [],
        "Equiangularity (class-means)": [],
        "Equiangularity (weights)": [],
        "Maximal-angle equiangularity (class-means)": [],
        "Maximal-angle equiangularity (weights)": [],
        "Classifiers convergence": [],
        "Within-Class Variation": [],
        "NN Mismatch": [],
    }

    # Populate the metrics data from the estimation results
    for epoch in epochs:
        for key in metrics.keys():
            metrics[key].append(estimation_results[epoch][key])

    # Create a directory to save the plots
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Plot Top1-train and Top1-test
    plt.figure()
    plt.plot(epochs, metrics["Top1-train"], label="Train")
    plt.plot(epochs, metrics["Top1-test"], label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Fig. 1. {dataset} / {network_name}. Top-1 Accuracy")
    plt.legend()
    plt.savefig(f"{folder_name}/Figure1_Top1_Accuracy.jpg")
    plt.close()

    # Plot Equinormality (class-means) and Equinormality (weights)
    plt.figure()
    plt.plot(
        epochs,
        metrics["Equinormality (class-means)"],
        label="Mean Activations",
    )
    plt.plot(epochs, metrics["Equinormality (weights)"], label="Classifiers")
    plt.xlabel("Epoch")
    plt.ylabel("Std / Avg")
    plt.title(f"Fig. 2.  {dataset} / {network_name}. Equinormality metric")
    plt.legend()
    plt.savefig(f"{folder_name}/Figure2_Equinormality.jpg")
    plt.close()

    # Plot Equiangularity (class-means) and Equiangularity (weights)
    plt.figure()
    plt.plot(
        epochs,
        metrics["Equiangularity (class-means)"],
        label="Mean Activations",
    )
    plt.plot(
        epochs,
        metrics["Equiangularity (weights)"],
        label="Classifiers",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Std(Cos)")
    plt.title(f"Fig. 3. {dataset} / {network_name}. Equiangularity metric")
    plt.legend()
    plt.savefig(f"{folder_name}/Figure3_Equiangularity.jpg")
    plt.close()

    plt.figure()
    plt.plot(
        epochs,
        metrics["Maximal-angle equiangularity (class-means)"],
        label="Mean Activations",
    )
    plt.plot(
        epochs,
        metrics["Maximal-angle equiangularity (weights)"],
        label="Classifiers",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Avg(|Shifted Cos|)")
    plt.title(
        f"Fig. 4. {dataset} / {network_name}. Maximal-angle equiangularity"
    )
    plt.legend()
    plt.savefig(f"{folder_name}/Figure4_Maximal_Angularity.jpg")
    plt.close()

    # Plot Classifiers convergence global and Classifiers convergence
    plt.figure()
    plt.plot(
        epochs,
        metrics["Classifiers convergence"],
        color="black",
    )
    plt.xlabel("Epoch")
    plt.ylabel("|| W - M ||")
    plt.title(
        f"Fig. 5. {dataset} / {network_name} Classifier to activ. mean cnvrg."
    )
    plt.savefig(f"{folder_name}/Figure5_Weights_Convergence.jpg")
    plt.close()

    # Plot Within-Class Variation collapse
    plt.figure()
    plt.plot(
        epochs,
        metrics["Within-Class Variation"],
        color="black",
    )
    plt.xlabel("Epoch")
    plt.ylabel(r"$\mathrm{tr}(\Sigma_W \Sigma_B^{+}) / C$")
    plt.title(f"Figure 6 - {dataset} / {network_name}. Within-class variation")
    plt.yscale("log")
    plt.savefig(f"{folder_name}/Figure6_Within_Class_Variation.jpg")
    plt.close()

    # Plot NN Mismatch
    plt.figure()
    plt.plot(epochs, metrics["NN Mismatch"], color="black")
    plt.xlabel("Epoch")
    plt.ylabel("Proportion Mismatch")
    plt.title(f"Fig. 7. {dataset} / {network_name}. Nearest class-center")
    plt.savefig(f"{folder_name}/Figure7_NN_Mismatch.jpg")
    plt.close()


def get_args():
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Extract penultimate layer features and calculate mean features per class."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        help="Dataset to train the model on",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations for the training data
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
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2
    )

    nc_estimations = {}

    if args.dataset == "CIFAR10":
        checkpoints_folder = "checkpoints_CIFAR10"
    elif args.dataset == "CIFAR100":
        checkpoints_folder = "checkpoints_CIFAR100"
    checkpoints = os.listdir(checkpoints_folder)
    checkpoints = sorted(checkpoints)
    for checkpoint in checkpoints:
        epoch = int(checkpoint.split("_")[1].split(".")[0])
        estimated_nc = process_checkpoint(
            args,
            os.path.join(checkpoints_folder, checkpoint),
            train_loader,
            test_loader,
            device,
        )
        nc_estimations[epoch] = estimated_nc

    print("Saving plots...")
    plot_estimation_results(
        nc_estimations, args.dataset + "_plots", args.dataset
    )
    print("Done!")


if __name__ == "__main__":
    main()
