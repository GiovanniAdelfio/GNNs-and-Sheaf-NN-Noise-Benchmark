"""Noise generation — transition matrices and label corruption strategies."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats


def uniform_noise(n_classes, noise_rate):
    """Build a C x C symmetric transition matrix with uniform off-diagonal noise.

    P[i,j] = noise_rate / (C - 1)  for i != j
    P[i,i] = 1 - noise_rate

    Rows are renormalized to sum to 1.

    Args:
        n_classes: Number of classes C.
        noise_rate: Total probability mass shifted away from the diagonal.

    Returns:
        numpy array of shape (C, C).
    """
    P = np.full((n_classes, n_classes), noise_rate / (n_classes - 1), dtype=np.float64)
    np.fill_diagonal(P, 1 - noise_rate)
    P[np.arange(n_classes), np.arange(n_classes)] += 1 - P.sum(axis=1)
    return P

def random_pair_noise(n_classes, noise_rate, seed):
    """Each class flips to one randomly chosen other class (seeded).

    Unlike ``pair_noise`` which uses a fixed circular pattern, here each class
    independently picks a single target class at random.

    Args:
        n_classes: Number of classes C.
        noise_rate: Flip probability.
        seed: RNG seed for reproducibility.

    Returns:
        numpy array of shape (C, C).
    """
    rng = np.random.default_rng(seed)
    P = np.eye(n_classes, dtype=np.float64) * (1 - noise_rate)
    for i in range(n_classes):
        candidates = list(range(i)) + list(range(i + 1, n_classes))
        chosen = rng.choice(candidates)
        P[i, chosen] = noise_rate
    return P

def instance_independent_noise(labels, cp, seed):
    """Apply a pre-computed transition matrix via multinomial sampling.

    For each node, draws a new label from the row of ``cp`` corresponding to
    its current label.

    Args:
        labels: 1-D numpy array of integer labels.
        cp: Transition matrix of shape (C, C), rows sum to 1.
        seed: RNG seed for reproducibility.

    Returns:
        1-D numpy array of (possibly corrupted) labels.
    """
    rs = np.random.RandomState(seed)
    noisy_labels = np.array([np.where(rs.multinomial(1, cp[label]))[0][0] for label in labels])
    return noisy_labels

def noise_operation(labels, features, n_classes, noise_type='clean', noise_rate=0, noise_seed=1, idx_train=None, debug=True):
    """Dispatcher: validate noise_type, delegate to the specific generator, log stats.

    Validates ``noise_type`` against the known set, dispatches to the
    appropriate generator function, applies the transition matrix (if
    applicable) via ``instance_independent_noise``, and optionally prints
    corruption statistics.

    Args:
        labels: 1-D tensor of original integer labels.
        features: Node feature tensor (required only for 'instance' noise).
        n_classes: Number of classes.
        noise_type: One of 'clean', 'uniform_simple', 'uniform', 'random',
            'pair', 'random_pair', 'flip', 'uniform_mix', 'deterministic',
            'instance'.
        noise_rate: Corruption rate in [0, 1].
        noise_seed: RNG seed.
        idx_train: Training indices (required for 'deterministic' noise).
        debug: If True, print noise statistics when noise_rate > 0.

    Returns:
        Tuple of (noisy_labels tensor, noisy_indices numpy array).
    """
    assert 0 <= noise_rate <= 1

    allowed_noise_types = [
        'uniform', 'random_pair', 'clean'
    ]

    if noise_type not in allowed_noise_types:
        raise ValueError(
            f"Invalid noise_type '{noise_type}'. "
            f"Please choose one of: {allowed_noise_types}"
        )

    cp = None
    noisy_labels = None

    if noise_rate == 0:
        cp = np.eye(n_classes)
        noisy_labels = labels.clone()
    else:
        if noise_type == 'clean':
            cp = np.eye(n_classes)
            noisy_labels = labels.clone()
        elif noise_type == 'uniform':
            cp = uniform_noise(n_classes, noise_rate)
        elif noise_type == 'random_pair':
            cp = random_pair_noise(n_classes, noise_rate, seed=noise_seed)

    if noisy_labels is None:
        if cp is not None:
            noisy_labels_np = instance_independent_noise(labels.cpu().numpy(), cp, seed=noise_seed)
            noisy_labels = torch.tensor(noisy_labels_np, dtype=torch.long, device=labels.device)
        else:
            noisy_labels = labels.clone()

    if debug and noise_rate > 0:
        noise_count = (noisy_labels != labels).sum().item()
        actual_noise_rate = noise_count / len(labels)
        print(f"[Noise info] Type: {noise_type}, Target rate: {noise_rate:.2f}, Actual rate: {actual_noise_rate:.2f}")
        print(f"[Noise info] Total corrupted labels: {noise_count} out of {len(labels)}")

    noisy_indices = (noisy_labels != labels).nonzero(as_tuple=True)[0].cpu().numpy()
    return noisy_labels, noisy_indices
