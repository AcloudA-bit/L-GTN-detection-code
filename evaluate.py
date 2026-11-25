import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from lgtn.data import load_splits, load_fixed_windows, CLASS_NAMES
from lgtn.model import LGTN
from lgtn.engine import evaluate, resolve_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--data", type=Path, help="MAT or NPZ dataset; evaluates stored test split")
    group.add_argument("--fixed-data", type=Path, help="Fixed labeled evaluation NPZ (x, y)")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.batch_size < 1:
        raise ValueError("Batch size must be positive.")
    device = resolve_device(a.device)
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    model = LGTN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    dataset = load_splits(a.data)[1] if a.data else load_fixed_windows(a.fixed_data)
    criterion = torch.nn.CrossEntropyLoss(weight=checkpoint["class_weights"].to(device))
    result = evaluate(model, DataLoader(dataset, batch_size=a.batch_size, shuffle=False), criterion, device)
    result["class_names"] = list(CLASS_NAMES)
    result["checkpoint_epoch"] = checkpoint["epoch"]
    result["data_source"] = "stored_test_split" if a.data else "fixed_labeled_windows"
    output = json.dumps(result, indent=2)
    print(output)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        if a.output.exists():
            raise FileExistsError(f"Refusing to overwrite {a.output}")
        a.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
