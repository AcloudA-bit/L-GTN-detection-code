import random
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or install a matching CUDA build.")
    return device


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum = 0.0
    n = 0
    num_correct = 0
    for batch in loader:
        x, *sources, y = batch
        if sources:
            source = sources[0]
            if len(sources) == 2 and torch.rand(1).item() >= 0.5:
                source = sources[1]
            # Use one source for the second half of the shuffled batch.
            split = len(y) // 2
            x = torch.cat((x[:split], source[split:]), dim=0)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(y)
        n += len(y)
        num_correct += (logits.argmax(dim=1) == y).sum().item()
    return {"loss": loss_sum / n, "accuracy_pct": 100.0 * num_correct / n}


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=7):
    model.eval()
    y_true, y_pred = [], []
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * len(y)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    truth, pred = np.asarray(y_true), np.asarray(y_pred)
    classes = list(range(num_classes))
    counts = confusion_matrix(truth, pred, labels=classes)
    return {"samples": len(y_true), "loss": loss_sum / len(y_true),
            "accuracy_pct": float(100 * np.mean(truth == pred)),
            "macro_f1_pct": float(100 * f1_score(truth, pred, average="macro",
                                                 labels=classes, zero_division=0)),
            "confusion_counts": counts.tolist()}
