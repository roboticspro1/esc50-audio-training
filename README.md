# ESC-50 Environmental Sound Classification

This submission trains a deep-learning model to classify 50 types of environmental sounds from the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50).

## Final result

| Metric | Result |
| --- | ---: |
| Best validation accuracy (fold 4) | 67.9% |
| Final test accuracy (fold 5) | **72.0%** (288/400 clips) |

## Submission files

- `train-optimized.ipynb`: visual training notebook with explanation, full displayed code, training cell, spectrogram, and learning curves.
- `test-optimized.ipynb`: visual evaluation notebook with the final test result, confusion matrix, per-class accuracy, and prediction examples.
- `train.ipynb` / `test.ipynb`: preserved original baseline version.
- `train-improved.ipynb` / `test-improved.ipynb`: preserved first-improved version.
- `train-optimized.py` and `test-optimized.py`: required support scripts used by the two notebooks.
- `outputs/optimized/`: saved best checkpoint, charts, per-class metrics, training history, and test predictions.
- `requirements.txt`: compatible package versions.

The ESC-50 dataset and Python virtual environment are intentionally excluded because they are large, local machine-specific files.

The baseline and first-improved versions are retained as experiment history. The optimized notebooks and `outputs/optimized/` are the final version to present.

## Dataset layout

Download ESC-50 separately and unpack it beside the notebooks using this layout:

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

## Run the project

1. Open `train-optimized.ipynb` in Jupyter Lab and run its cells in order to train a new model and view training visuals.
2. Open `test-optimized.ipynb` and run its cells in order to evaluate the selected checkpoint and view the final results.

Training uses folds 1–3, validation uses fold 4, and final testing uses the held-out fold 5.

## Optimization approach

- 16 kHz normalized log-Mel spectrograms
- Residual convolutional neural network with dropout
- Audio-gain and SpecAugment masking during training
- AdamW optimization with weight decay
- Label smoothing and adaptive learning-rate reduction
- Best-validation checkpoint selection
- Visual analysis with learning curves, a confusion matrix, per-class accuracy, and prediction examples

## Attribution

The original assignment workflow is based on the audio-classification material from [sscardapane/reprodl2021](https://github.com/sscardapane/reprodl2021). ESC-50 is maintained by [Karol J. Piczak](https://github.com/karolpiczak/ESC-50).
