# ESC-50 Audio Classification

This project trains and evaluates a convolutional neural network on the [ESC-50](https://github.com/karolpiczak/ESC-50) environmental sound classification dataset.

## Contents

- `train.ipynb` and `test.ipynb`: baseline training and evaluation notebooks.
- `train-improved.ipynb` and `test-improved.ipynb`: experiment notebooks with AdamW, validation checkpointing, and early stopping.
- `train.py`, `test.py`, `train-improved.py`, and `test-improved.py`: plain-Python exports of the same notebook code for reliable viewing on GitHub.
- `outputs/`: saved checkpoints and CSV prediction results.
- `requirements.txt`: compatible Python packages for macOS.

The ESC-50 audio dataset and local virtual environments are deliberately excluded from this repository because they are large and machine-specific.

> Notebook outputs are intentionally cleared before publishing so GitHub can load the notebooks quickly. Reproducible result artifacts are retained in `outputs/`.

## Viewing the code on GitHub

GitHub sometimes fails to render Jupyter notebook previews even when the notebook file is valid. If a notebook page shows a spinner or an "Unable to render code block" message, open its matching `.py` file instead. The `.py` export contains the same executable code, without notebook cell boundaries or outputs:

- `train.ipynb` → `train.py`
- `test.ipynb` → `test.py`
- `train-improved.ipynb` → `train-improved.py`
- `test-improved.ipynb` → `test-improved.py`

## Dataset layout

Download or obtain ESC-50 separately, then unpack it so the notebooks can find:

```text
ESC-50-master/
└── ESC-50-master/
    ├── audio/
    └── meta/esc50.csv
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Use a machine-compatible Lightning trainer setting:

```python
pl.Trainer(accelerator="auto", devices=1)
```

## Evaluation protocol

- Folds 1–3: training
- Fold 4: validation
- Fold 5: final testing

The baseline completed 25 epochs and reached about 49.5% validation accuracy. The recorded improved-checkpoint test result is 48.7% (195/400 clips). This result should be interpreted as one experiment, not a final benchmark.

## Changes from the baseline

The data split, 8 kHz audio preprocessing, 50-class CNN architecture, and batch size remain unchanged. The experiment files make the following training changes.

| Area | Baseline | Experiment |
| --- | --- | --- |
| Hardware selection | `accelerator="gpu"` | `accelerator="auto"`, which works on macOS without NVIDIA CUDA |
| Optimizer | Adam with learning rate `1e-3` | AdamW with learning rate `3e-4` and weight decay `1e-4` |
| Epoch limit | 25 epochs | Up to 100 epochs |
| Checkpoint | Saves the final epoch manually | Saves the checkpoint with the highest validation accuracy |
| Early stopping | Not used | Stops after validation accuracy fails to improve for the configured patience |
| Test artifact | `test_predictions.csv` | `test_predictions_best.csv` |

### Hardware-compatible trainer

```python
trainer = pl.Trainer(accelerator="auto", devices=1)
```

### Added callback import

```python
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
```

### AdamW optimizer

```python
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
        self.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )
    return optimizer
```

### Validation checkpoint and early stopping

```python
checkpoint_callback = ModelCheckpoint(
    dirpath="outputs",
    filename="audionet-best",
    monitor="val_acc",
    mode="max",
    save_top_k=1,
)

early_stop = EarlyStopping(
    monitor="val_acc",
    mode="max",
    patience=8,
)

trainer = pl.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=100,
    callbacks=[checkpoint_callback, early_stop],
)

trainer.fit(audionet, train_loader, val_loader)
```

### Testing the selected checkpoint

`test-improved.ipynb` loads the best checkpoint from the first experiment block and writes a separately named prediction file:

```python
ckpt_path = Path("outputs/audionet-best.ckpt")
audionet = AudioNet.load_from_checkpoint(ckpt_path)

pred_path = output_dir / "test_predictions_best.csv"
predictions.to_csv(pred_path, index=False)
```

## Current experiment note

`train-improved.ipynb` currently contains a second checkpointing/training section after the first one. It creates `best-audionet.ckpt` and then saves `audionet-improved-last.ckpt`. This is retained as a record of the experiment, but a future cleanup should keep just one training section so every result maps to one clearly defined configuration.

## Attribution

The notebook workflow is based on the audio-classification material in [sscardapane/reprodl2021](https://github.com/sscardapane/reprodl2021). ESC-50 is maintained by [Karol J. Piczak](https://github.com/karolpiczak/ESC-50).
