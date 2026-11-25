# L-GTN for Optical Fiber Eavesdropping Detection

L-GTN classifies seven optical event classes using joint SOP and EVM time series.

## Files

| File | Purpose |
| --- | --- |
| `lgtn/model.py` | Defines the L-GTN architecture. |
| `lgtn/data.py` | Loads processed MAT or NPZ windows and calculates class weights. |
| `lgtn/engine.py` | Runs training and computes evaluation metrics. |
| `lgtn/__init__.py` | Exposes the `LGTN` model for import. |
| `train.py` | Runs standard training or mixed training with pre-generated paired samples. |
| `evaluate.py` | Loads a checkpoint and reports accuracy, macro F1, and a confusion matrix. |
| `profile_model.py` | Reports parameter count and operations per input window. |
| `make_demo_data.py` | Creates synthetic data for testing the software. |
| `requirements.txt` | Lists the Python dependencies and their versions. |

## Environment

Tested with Python 3.8 and PyTorch 2.4.1 on CPU and CUDA 12.1.

```bash
python -m pip install -r requirements.txt
```

## Default configuration

| Setting | Value |
| --- | --- |
| Input window | 128 time steps × 5 channels |
| Number of classes | 7 |
| Model dimension | 64 |
| Encoder layers per branch | 2 |
| Feed-forward hidden dimension | 128 |
| Encoder and Conv-FFN dropout | 0.2 and 0.1 |
| Optimizer | AdamW |
| Loss | Class-weighted cross-entropy |
| Learning rate | `5e-5` |
| Weight decay | `1e-5` |
| Batch size | 16 |
| Epochs | 300 |
| Random seed | 30 |
