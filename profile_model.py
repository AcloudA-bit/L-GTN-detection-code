import torch
from fvcore.nn import FlopCountAnalysis
from lgtn.model import LGTN


def main():
    model = LGTN().eval()
    example = torch.zeros(1, 128, 5)
    analysis = FlopCountAnalysis(model, example)
    operations = analysis.total()
    print(f"Parameters: {model.parameter_count():,} ({model.parameter_count()/1000:.2f} k)")
    print(f"fvcore operations per 5 x 128 window: {operations:,} ({operations/1e6:.2f} M)")
    print("One fused multiply-add counts as one operation.")
    print("Unsupported operations (not included by fvcore):", dict(analysis.unsupported_ops()))


if __name__ == "__main__":
    main()
