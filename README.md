# ESC-50 Audio Classification

This project trains and evaluates a convolutional neural network on the [ESC-50](https://github.com/karolpiczak/ESC-50) environmental sound classification dataset.

## Contents

- `train.ipynb` and `test.ipynb`: baseline training and evaluation notebooks.
- `train-improved.ipynb` and `test-improved.ipynb`: experiment notebooks with AdamW, validation checkpointing, and early stopping.
- `train-optimized.ipynb` and `test-optimized.ipynb`: a separate, higher-capacity experiment with augmentation, automatic checkpoint selection, and visual reports.
- `train.py`, `test.py`, `train-improved.py`, and `test-improved.py`: plain-Python exports of the same notebook code for reliable viewing on GitHub.
- `train-optimized.py` and `test-optimized.py`: the fully reproducible optimized training and evaluation scripts.
- `outputs/`: saved checkpoints, prediction results, and charts.
- `requirements.txt`: compatible Python packages for macOS.

The ESC-50 audio dataset and local virtual environments are deliberately excluded from this repository because they are large and machine-specific.

> Notebook outputs are intentionally cleared before publishing so GitHub can load the notebooks quickly. Reproducible result artifacts are retained in `outputs/`.

## Viewing the code on GitHub

GitHub sometimes fails to render Jupyter notebook previews even when the notebook file is valid. If a notebook page shows a spinner or an "Unable to render code block" message, open its matching `.py` file instead. The `.py` export contains the same executable code, without notebook cell boundaries or outputs:

- `train.ipynb` → `train.py`
- `test.ipynb` → `test.py`
- `train-improved.ipynb` → `train-improved.py`
- `test-improved.ipynb` → `test-improved.py`
- `train-optimized.ipynb` → `train-optimized.py`
- `test-optimized.ipynb` → `test-optimized.py`

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

The baseline completed 25 epochs and reached about 49.5% validation accuracy. The recorded improved-checkpoint test result is 48.7% (195/400 clips).

The separate optimized experiment reached **67.9% best validation accuracy** and **72.0% final test accuracy (288/400 clips)**. It uses the same held-out test fold, but this is still one experimental result rather than a general benchmark.

## Optimized experiment

The optimized files are new additions; the baseline and earlier improved notebooks were not changed. The experiment uses 16 kHz log-mel spectrograms, a residual CNN, AdamW, label smoothing, learning-rate reduction, waveform gain and spectrogram masking augmentation, early stopping, and best-validation checkpoint selection.

Run it from an activated environment with:

```bash
python train-optimized.py
python test-optimized.py
```

The selected model is saved as `outputs/optimized/optimized-best.ckpt`. Evaluation writes these useful visual artifacts to `outputs/optimized/`:

- `sample_spectrogram.png`: example model input after audio preprocessing.
- `training_curves.png`: training/validation loss, accuracy, and macro-F1 across epochs.
- `confusion_matrix.png`: normalized per-class error patterns.
- `per_class_accuracy.png` and `per_class_accuracy.csv`: the strongest and weakest classes.
- `prediction_examples.png`: correctly and incorrectly classified test clips.
- `optimized_test_predictions.csv`: every test prediction, true category, and correctness flag.

| Result | Earlier improved experiment | Optimized experiment |
| --- | ---: | ---: |
| Test accuracy | 48.7% (195/400) | **72.0% (288/400)** |
| Saved model | `outputs/audionet-best.ckpt` | `outputs/optimized/optimized-best.ckpt` |
| Visual reporting | CSV predictions | Curves, confusion matrix, per-class chart, and example predictions |

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
