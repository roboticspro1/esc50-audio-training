# ESC-50 Audio Classification

This project trains and evaluates a convolutional neural network on the [ESC-50](https://github.com/karolpiczak/ESC-50) environmental sound classification dataset.

## Contents

- `train.ipynb` and `test.ipynb`: baseline training and evaluation notebooks.
- `train-improved.ipynb` and `test-improved.ipynb`: experiment notebooks with AdamW, validation checkpointing, and early stopping.
- `outputs/`: saved checkpoints and CSV prediction results.
- `requirements.txt`: compatible Python packages for macOS.

The ESC-50 audio dataset and local virtual environments are deliberately excluded from this repository because they are large and machine-specific.

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

## Attribution

The notebook workflow is based on the audio-classification material in [sscardapane/reprodl2021](https://github.com/sscardapane/reprodl2021). ESC-50 is maintained by [Karol J. Piczak](https://github.com/karolpiczak/ESC-50).
