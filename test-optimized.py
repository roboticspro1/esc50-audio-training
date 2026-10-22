"""Evaluate the optimized ESC-50 checkpoint and create useful visual reports."""

from pathlib import Path
import importlib.util

import matplotlib.pyplot as plt
import pandas as pd
import torch

TRAINING_SCRIPT = Path(__file__).with_name("train-optimized.py")
spec = importlib.util.spec_from_file_location("train_optimized", TRAINING_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {TRAINING_SCRIPT}")
train_optimized = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_optimized)

DATA_PATH = train_optimized.DATA_PATH
N_CLASSES = train_optimized.N_CLASSES
OUTPUT_DIR = train_optimized.OUTPUT_DIR
ESC50Dataset = train_optimized.ESC50Dataset
OptimizedAudioNet = train_optimized.OptimizedAudioNet


def main():
    checkpoint = OUTPUT_DIR / "optimized-best.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError("Run train-optimized.py before running this evaluation.")

    model = OptimizedAudioNet.load_from_checkpoint(checkpoint).eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    test_data = ESC50Dataset(folds=[5], augment=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

    predictions, targets = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            logits = model(features.to(device))
            predictions.append(logits.argmax(dim=1).cpu())
            targets.append(labels)
    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()

    label_names = (
        pd.read_csv(DATA_PATH / "meta" / "esc50.csv")[["target", "category"]]
        .drop_duplicates()
        .set_index("target")["category"]
    )
    meta = test_data.csv.copy()
    results = pd.DataFrame(
        {
            "filename": meta["filename"],
            "target": targets,
            "prediction": predictions,
        }
    )
    results["category"] = results["target"].map(label_names)
    results["predicted_category"] = results["prediction"].map(label_names)
    results["correct"] = results["target"] == results["prediction"]
    results.to_csv(OUTPUT_DIR / "optimized_test_predictions.csv", index=False)

    accuracy = results["correct"].mean()
    confusion = torch.bincount(
        torch.tensor(targets * N_CLASSES + predictions), minlength=N_CLASSES**2
    ).reshape(N_CLASSES, N_CLASSES).numpy()
    normalized = confusion / confusion.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, label="Fraction of true class")
    ax.set(
        title="Normalized confusion matrix — optimized model",
        xlabel="Predicted class index",
        ylabel="True class index",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    class_accuracy = pd.DataFrame(
        {
            "target": range(N_CLASSES),
            "category": [label_names[index] for index in range(N_CLASSES)],
            "accuracy": normalized.diagonal(),
            "samples": confusion.sum(axis=1),
        }
    ).sort_values("accuracy")
    class_accuracy.to_csv(OUTPUT_DIR / "per_class_accuracy.csv", index=False)

    display = pd.concat([class_accuracy.head(10), class_accuracy.tail(10)]).sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#c44e52" if value < 0.5 else "#55a868" for value in display["accuracy"]]
    ax.barh(display["category"], display["accuracy"], color=colors)
    ax.axvline(accuracy, color="#4c72b0", linestyle="--", label=f"Overall: {accuracy:.1%}")
    ax.set(title="10 hardest and 10 easiest ESC-50 classes", xlabel="Test accuracy", xlim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "per_class_accuracy.png", dpi=180)
    plt.close(fig)

    example_indices = list(results.index[~results["correct"]][:6]) + list(results.index[results["correct"]][:6])
    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    for ax, index in zip(axes.flat, example_indices):
        features, _ = test_data[index]
        row = results.iloc[index]
        ax.imshow(features[0], aspect="auto", origin="lower", cmap="magma")
        color = "#c44e52" if not row["correct"] else "#55a868"
        ax.set_title(f"True: {row['category']}\nPred: {row['predicted_category']}", color=color, fontsize=8)
        ax.axis("off")
    fig.suptitle("Example test spectrograms (red = incorrect, green = correct)", y=0.98)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "prediction_examples.png", dpi=180)
    plt.close(fig)

    print(f"Optimized test accuracy: {accuracy:.3f} ({results['correct'].sum()}/{len(results)})")
    print(f"Saved reports in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
