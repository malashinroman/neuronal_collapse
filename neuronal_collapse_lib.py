# This code is an unofficial implementation of the following paper:
# "Prevalence of neural collapse during the terminal phase of deep learning training"
# by Papyan, V., Han, X. Y., & Donoho, D. L. (2020). Proceedings of the National Academy of Sciences, 117(40), 24652-24663.

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


def get_classifeir_weights_bias(model, device=torch.device("cuda")):
    """
    Extract the weights and bias of the classifier layer of the model
    """
    if hasattr(model, "fc"):
        W = model.fc.weight.detach().to(device)  # resnet
        bias = model.fc.bias.detach().to(device)
    elif hasattr(model, "head"):
        W = model.head.weight.detach().to(device)  # vit
        bias = model.head.bias.detach().to(device)
    elif hasattr(model, "classifier"):
        W = model.classifier.weight.detach().to(device)  # mobilenet
        bias = model.classifier.bias.detach().to(device)
    else:
        raise ValueError("Model does not have a fc or head layer")
    return W.double(), bias.double()


def compute_accuracy(features, targets, W, bias=None):
    """
    Compute the accuracy of the classifier using the features (for verification purposes)
    """
    with torch.no_grad():
        # Compute class scores using W and bias
        if bias is not None:
            class_scores = features @ W.T + bias
        else:
            class_scores = features @ W.T
        _, predicted = class_scores.max(1)
        correct = predicted.eq(targets).sum().item()
        total = targets.size(0)
        accuracy = 100.0 * correct / total
    return accuracy


def get_penultimate_features(model, loader, device):
    """
    Get the penultimate features of the model, assuming the model classifer is cut off
    """
    model.eval()
    penultimate_features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward pass until the penultimate layer
            features = model(inputs)
            penultimate_features.append(features)
            labels.append(targets)

    penultimate_features = torch.cat(penultimate_features).squeeze()
    labels = torch.cat(labels)
    return penultimate_features, labels


def compute_Sigma_W(features, targets, class_means, unique_classes, device):
    """
    Efficient computation of Sigma_W
    """
    Sigma_W = torch.zeros(
        (features.shape[1], features.shape[1]),
        device=device,
        dtype=torch.float64,
    )
    for cls in unique_classes:
        class_features = features[targets == cls]
        centered_class_features = class_features - class_means[cls]
        Sigma_W += centered_class_features.T @ centered_class_features

    Sigma_W /= len(targets)
    return Sigma_W


def compute_Sigma_B(class_means, global_mean):
    class_mean_diff_all = class_means - global_mean
    Sigma_B = class_mean_diff_all.T @ class_mean_diff_all
    Sigma_B /= class_means.shape[0]
    return Sigma_B


# fails on CIFAR100 therefore the cpu version is used
# Pytorch (+gpu) fails (returns incorrect result) on CIFAR100 probably because of the size of the tensors
def within_class_metric_from_covs_gpu(Sigma_W, Sigma_B, n_classes):
    # Invert Sigma_B
    Sigma_B_pinv = torch.pinverse(Sigma_B)

    # Compute the product of Sigma_W and Sigma_B_pinv
    product = Sigma_W @ Sigma_B_pinv

    # Compute the trace of the product
    trace_value = torch.trace(product)

    # Compute the final value
    res = trace_value / float(n_classes)

    return res.item()


def within_class_metric_from_covs_cpu(Sigma_W, Sigma_B, n_classes):
    """
    Given the within-class covariance matrix Sigma_W
    and the between-class covariance matrix Sigma_B,
    compute within-class collapse metric
    """
    # Convert PyTorch tensors to NumPy arrays
    Sigma_W_np = Sigma_W.numpy()
    Sigma_B_np = Sigma_B.numpy()

    # Compute the Moore-Penrose pseudoinverse of Sigma_B
    Sigma_B_pinv = np.linalg.pinv(Sigma_B_np)

    # Compute the product of Sigma_W and Sigma_B_pinv
    product = np.dot(Sigma_W_np, Sigma_B_pinv)

    # Compute the trace of the product
    trace_value = np.trace(product)

    # Compute the final value
    res = trace_value / float(n_classes)

    return res


def estimate_withing_class_variation(
    features, targets, class_means, global_mean, device
):
    """
    Estimate the within-class variation collapse metric. Based on Figure 6 of the paper
    """
    unique_classes = torch.unique(targets)

    Sigma_W = compute_Sigma_W(
        features, targets, class_means, unique_classes, device
    )
    Sigma_B = compute_Sigma_B(class_means, global_mean)

    # The next block is not necessary for the computation of the metric,
    # Equivalence according to multivariate statisticts,
    # On gpu sometimes it fails fom time to time (for some reason)
    # but it is useful for debugging issues with gpu
    Sigma_T = ((features - global_mean).T @ (features - global_mean)) / len(
        targets
    )
    assert torch.allclose(
        Sigma_T, Sigma_W + Sigma_B
    ), "Sigma_T != Sigma_W + Sigma_B"

    res_cpu = within_class_metric_from_covs_cpu(
        Sigma_W.cpu(), Sigma_B.cpu(), len(unique_classes)
    )
    return res_cpu


def estimate_equinormity(normalized_class_vectors):
    """
    Based on Figure 2 of the paper
    """
    norms = torch.norm(normalized_class_vectors, dim=1)
    return torch.std(norms) / torch.mean(norms)


def estimate_equiangularity(normed_class_means):
    """
    Based on Figure 3 and Figure 4 of the paper
    """
    # Compute angles between class means
    cosines = normed_class_means @ normed_class_means.T
    norms = torch.norm(normed_class_means, dim=1)
    normalizer = norms.unsqueeze(1) @ norms.unsqueeze(1).T
    cosines = cosines / normalizer

    # Exclude diagonal elements
    mask = ~torch.eye(cosines.shape[0], dtype=bool)
    n_classes = cosines.shape[0]
    off_diagonal_elements = cosines[mask]

    # Compute the standard deviation of off-diagonal elements
    std_off_diagonal = torch.std(off_diagonal_elements)

    # Compute deviation from maximal angularity

    # FIXME: Why we assume negative cosine always?
    # Below and above diagonal elemants are the same
    maximal_angularity = off_diagonal_elements + 1.0 / (n_classes - 1)
    maximal_angularity = torch.mean(maximal_angularity.abs())
    return std_off_diagonal, maximal_angularity


def estimate_weight_feat_convergence_global(class_vectors, W):
    """
    Based on Figure 5 of the paper.
    They normalize classifiers and weights to be unit norm: formulas suggest global normalization of all vectors.
    As description sais they report squared frobenuis norm"""
    W_norm = W / W.norm()
    M_norm = class_vectors / class_vectors.norm()
    res = torch.sum(
        (W_norm - M_norm) ** 2
    )  # squared frobenius norm is also a norm
    return res


def estimate_NN_convergence(test_features, test_targets, class_means, W, bias):
    """
    Based on Figure 7 of the paper
    """
    predictions = test_features @ W.T + bias
    predicted_labels = predictions.argmax(dim=1)

    distances = torch.cdist(test_features, class_means)
    _, nn_predictions = distances.min(dim=1)

    test_accuracy = (predicted_labels == test_targets).float().mean()
    nn_alignment = (predicted_labels == nn_predictions).float().mean()
    prop_mismatch = 1 - nn_alignment

    return test_accuracy, prop_mismatch


def estimate_collapse_metrics(
    features_train, targets_train, features_test, targets_test, W, bias, device
):
    """
    Compute all metrics
    """
    # Use double precision for all computations
    assert features_train.dtype == torch.float64
    assert W.dtype == torch.float64
    assert bias.dtype == torch.float64
    assert features_test.dtype == torch.float64
    print("Computing class means and global mean")
    unique_classes = torch.unique(targets_train)
    class_means = torch.stack(
        [
            features_train[targets_train == cls].mean(dim=0)
            for cls in unique_classes
        ],
        dim=0,
    ).to(device)
    global_mean = features_train.mean(dim=0).to(device)

    print("Computing accuracy")
    accuracy = compute_accuracy(features_train, targets_train, W, bias)
    accuracy_test = compute_accuracy(features_test, targets_test, W, bias)
    print(f"Accuracy: {accuracy}")

    print("Computing within-class variation collapse metric")
    within_class_var = estimate_withing_class_variation(
        features_train, targets_train, class_means, global_mean, device
    )

    print("Computing equiangularity metric")
    std_class_means_cosines, maximal_angularity_feat = estimate_equiangularity(
        class_means - global_mean
    )
    std_weights_cosines, maximal_angularity_weights = estimate_equiangularity(W)

    print("Computing equinormity metric")
    equinormity = estimate_equinormity(class_means - global_mean)
    equinormity_weights = estimate_equinormity(W)

    weights_feat_convergence = estimate_weight_feat_convergence_global(
        class_means - global_mean, W
    )

    print("Computing Nearest Neighbour convergence metric")
    test_accuracy, prop_mismatch = estimate_NN_convergence(
        features_test, targets_test, class_means, W, bias
    )
    assert torch.allclose(
        test_accuracy, torch.Tensor([accuracy_test / 100.0]).to(device)
    ), "Test accuracy computation failed"

    print("Done")

    return {
        "Top1-train": accuracy,
        "Top1-test": accuracy_test,
        "Equinormality (class-means)": equinormity.item(),
        "Equinormality (weights)": equinormity_weights.item(),
        "Equiangularity (class-means)": std_class_means_cosines.item(),
        "Equiangularity (weights)": std_weights_cosines.item(),
        "Maximal-angle equiangularity (class-means)": maximal_angularity_feat.item(),
        "Maximal-angle equiangularity (weights)": maximal_angularity_weights.item(),
        "Classifiers convergence": weights_feat_convergence.item(),
        "Within-Class Variation": within_class_var,
        "NN Mismatch": prop_mismatch.item(),
    }


def neuronal_collapse(model, train_loader, test_loader, device):
    W, bias = get_classifeir_weights_bias(model, device)

    # Modify the model to output the penultimate layer features
    model = nn.Sequential(
        *list(model.children())[:-1]
    )  # Remove the final fully connected layer
    model = model.to(device)

    # Extract penultimate layer features
    penultimate_features_train, labels_train = get_penultimate_features(
        model, train_loader, device
    )
    targets_train = labels_train

    penultimate_features_test, labels_test = get_penultimate_features(
        model, test_loader, device
    )
    targets_test = labels_test

    estimated_nc = estimate_collapse_metrics(
        penultimate_features_train.double(),
        targets_train,
        penultimate_features_test.double(),
        targets_test,
        W.double(),
        bias.double(),
        device,
    )
    return estimated_nc
