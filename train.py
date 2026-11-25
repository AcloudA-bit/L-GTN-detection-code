import argparse
import csv
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from lgtn.model import LGTN
from lgtn.data import load_splits, load_fixed_windows, pair_training_windows, class_counts, class_weights
from lgtn.engine import seed_everything, resolve_device, train_epoch, evaluate


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path, help="Processed MAT or NPZ dataset")
    p.add_argument("--paired-data", type=Path, action="append", default=[],
                   help="Fixed paired training observations (NPZ); specify once or twice")
    p.add_argument("--output", type=Path, required=True, help="New run directory")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=30)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    if a.epochs < 1 or a.batch_size < 1 or a.lr <= 0 or a.weight_decay < 0:
        raise ValueError("Use positive epochs, batch size and learning rate, and nonnegative weight decay.")
    if a.num_workers < 0:
        raise ValueError("num-workers must be nonnegative.")
    if len(a.paired_data) > 2:
        raise ValueError("Use at most two --paired-data arguments.")
    if a.paired_data and (a.batch_size < 2 or a.batch_size % 2):
        raise ValueError("Mixed training requires an even batch size of at least two.")
    seed_everything(a.seed)
    device = resolve_device(a.device)
    train, selection = load_splits(a.data)
    weights = class_weights(train)
    mode = "fixed_mixed" if a.paired_data else "standard"
    training_data = train
    if a.paired_data:
        training_data = pair_training_windows(train, *(load_fixed_windows(path) for path in a.paired_data))
    train_loader = DataLoader(training_data, batch_size=a.batch_size, shuffle=True,
                              num_workers=a.num_workers)
    selection_loader = DataLoader(selection, batch_size=a.batch_size, shuffle=False,
                                  num_workers=a.num_workers)
    model = LGTN().to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    a.output.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "history.csv", "run_summary.json"):
        if (a.output / name).exists():
            raise FileExistsError(f"Refusing to overwrite an existing run: {a.output}")
    info = {"training_mode": mode, "paired_sources": len(a.paired_data), "model_config": dict(model.config),
            "parameters": model.parameter_count(), "seed": a.seed, "epochs": a.epochs,
            "batch_size": a.batch_size, "learning_rate": a.lr, "weight_decay": a.weight_decay,
            "training_samples": len(train), "selection_samples": len(selection),
            "training_class_counts": class_counts(train), "selection_class_counts": class_counts(selection),
            "class_weights": weights.tolist(), "selection_protocol": "highest clean accuracy on stored test split",
            "independent_final_test": False}
    print(json.dumps(info, indent=2))
    fields = ["epoch", "train_loss", "train_accuracy_pct", "selection_loss",
              "selection_accuracy_pct", "selection_macro_f1_pct"]
    best_accuracy = -1.0
    best_result = None
    with (a.output / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, a.epochs + 1):
            tm = train_epoch(model, train_loader, optimizer, criterion, device)
            vm = evaluate(model, selection_loader, criterion, device)
            row = dict(epoch=epoch, train_loss=tm["loss"], train_accuracy_pct=tm["accuracy_pct"],
                       selection_loss=vm["loss"], selection_accuracy_pct=vm["accuracy_pct"],
                       selection_macro_f1_pct=vm["macro_f1_pct"])
            writer.writerow(row)
            stream.flush()
            print(f"Epoch {epoch:03d}: loss={tm['loss']:.4f}, "
                  f"selection accuracy={vm['accuracy_pct']:.2f}%, F1={vm['macro_f1_pct']:.2f}%")
            if vm["accuracy_pct"] > best_accuracy:
                best_accuracy = vm["accuracy_pct"]
                best_result = {"epoch": epoch, **vm}
                torch.save({"model_state_dict": model.state_dict(), "model_config": dict(model.config),
                            "class_weights": weights, "epoch": epoch, "training_mode": mode},
                           a.output / "best.pt")
    info["best_selection_result"] = best_result
    (a.output / "run_summary.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Saved checkpoint and aggregate results to {a.output}")


if __name__ == "__main__":
    main()
