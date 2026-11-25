import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("demo_data"))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    main_file = a.output / "demo.npz"
    paired_file = a.output / "demo_paired.npz"
    if main_file.exists() or paired_file.exists():
        raise FileExistsError("Use a new demo directory.")
    rng = np.random.default_rng(30)
    y_train = np.repeat(np.arange(1, 8), 8)
    y_test = np.repeat(np.arange(1, 8), 3)

    def fictional_observations(labels):
        # Independent fictional measurements, not modeled optical records.
        return (rng.normal(size=(len(labels), 128, 5)) + labels[:, None, None] * 0.1).astype(np.float32)

    np.savez_compressed(main_file, x_train=fictional_observations(y_train), y_train=y_train,
                        x_test=fictional_observations(y_test), y_test=y_test)
    np.savez_compressed(paired_file, x=fictional_observations(y_train), y=y_train)
    print(f"Wrote fictional demo data to {a.output}. These are not experimental or adversarial samples.")


if __name__ == "__main__":
    main()
