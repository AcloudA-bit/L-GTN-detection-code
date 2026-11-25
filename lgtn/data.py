from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import TensorDataset

NUM_CLASSES = 7
CLASS_NAMES = (
    "Normal transmission", "Eavesdropping", "Bending", "Interference",
    "Vibration", "Overlapping eavesdropping and interference",
    "Overlapping eavesdropping and vibration",
)


def make_dataset(windows, labels):
    labels = np.asarray(labels).reshape(-1)
    if (not labels.size or not np.isfinite(labels).all() or
            not np.array_equal(labels, labels.astype(np.int64)) or
            labels.min() < 1 or labels.max() > NUM_CLASSES):
        raise ValueError("Labels must be finite integers in the range 1 through 7.")
    windows = np.asarray(windows)
    if windows.dtype == object:
        rows = windows.reshape(-1)
        arrays = []
        for row in rows:
            array = np.asarray(row, dtype=np.float32)
            if array.shape == (5, 128):
                array = array.T
            if array.shape != (128, 5):
                raise ValueError(f"Unexpected window shape: {array.shape}")
            arrays.append(array)
        windows = np.stack(arrays)
    elif windows.ndim == 3 and windows.shape[1:] == (5, 128):
        windows = windows.transpose(0, 2, 1)
    if windows.shape != (len(labels), 128, 5):
        raise ValueError("Expected windows of shape (N, 128, 5) or (N, 5, 128).")
    windows = np.ascontiguousarray(windows, dtype=np.float32)
    if not np.isfinite(windows).all():
        raise ValueError("Input windows contain NaN or infinity.")
    # The source file uses one-based labels; cross-entropy uses zero-based labels.
    return TensorDataset(torch.from_numpy(windows),
                         torch.from_numpy(labels.astype(np.int64) - 1))


def load_splits(path):
    """Load MATLAB cell arrays or a portable NPZ with the documented schema."""
    path = Path(path)
    if path.suffix.lower() == ".mat":
        raw = loadmat(str(path), squeeze_me=False, struct_as_record=True)
        required = {"train", "trainlabels", "test", "testlabels"}
        candidates = [v for k, v in raw.items() if not k.startswith("__")
                      and isinstance(v, np.ndarray) and v.dtype.names
                      and required.issubset(v.dtype.names)]
        if len(candidates) != 1 or candidates[0].size != 1:
            raise ValueError("Expected one MATLAB structure with train/trainlabels/test/testlabels.")
        record = candidates[0].reshape(-1)[0]
        train = make_dataset(record["train"], record["trainlabels"])
        test = make_dataset(record["test"], record["testlabels"])
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as raw:
            train = make_dataset(raw["x_train"], raw["y_train"])
            test = make_dataset(raw["x_test"], raw["y_test"])
    else:
        raise ValueError("Dataset must be a .mat or .npz file.")
    if min(class_counts(train)) == 0:
        raise ValueError("Every class must be represented in the training split.")
    return train, test


def load_fixed_windows(path):
    """Load a fixed, externally prepared, labeled NPZ data collection."""
    with np.load(Path(path), allow_pickle=False) as raw:
        return make_dataset(raw["x"], raw["y"])


def pair_training_windows(clean, *sources):
    """Pair one or two fixed training sources with the clean windows."""
    if len(sources) not in (1, 2):
        raise ValueError("Provide one or two paired training sources.")
    for index, source in enumerate(sources, start=1):
        if len(clean) != len(source) or not torch.equal(clean.tensors[1], source.tensors[1]):
            raise ValueError(f"Paired source {index} must match the clean training length and label order.")
    return TensorDataset(clean.tensors[0], *(source.tensors[0] for source in sources), clean.tensors[1])


def class_counts(dataset):
    return torch.bincount(dataset.tensors[-1], minlength=NUM_CLASSES).tolist()


def class_weights(train):
    counts = torch.tensor(class_counts(train), dtype=torch.float32)
    if (counts == 0).any():
        raise ValueError("Cannot compute class weights for a missing class.")
    return counts.sum() / (NUM_CLASSES * counts)
