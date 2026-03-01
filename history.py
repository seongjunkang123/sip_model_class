# view classification model losses and accuracy over epochs.

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

VERSION = input("Version Number: ")
HISTORY_PATH = f"./model_performance/mod_v{VERSION}.pkl"
MOD_PERF_IMG_PATH = f"./model_performance/mod_v{VERSION}.png"

if not os.path.exists(HISTORY_PATH):
    print(f"File not found")
    exit(67)

with open(HISTORY_PATH, "rb") as f:
    perf_data = pickle.load(f)

# handle both old format (plain dict) and new format (nested dict)
if isinstance(perf_data, dict) and "history" in perf_data:
    history = perf_data["history"]
    cv_results = perf_data.get("cv_results", None)
    cv_mean_val_acc = perf_data.get("cv_mean_val_acc", None)
else:
    history = perf_data
    cv_results = None
    cv_mean_val_acc = None

# plot history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history["loss"], label="Train Loss")
if "val_loss" in history:
    axes[0].plot(history["val_loss"], label="Val Loss")
axes[0].set_title("Classification Model Loss")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(history["accuracy"], label="Train Accuracy")
if "val_accuracy" in history:
    axes[1].plot(history["val_accuracy"], label="Val Accuracy")
if cv_mean_val_acc is not None:
    axes[1].axhline(
        y=cv_mean_val_acc,
        color="r",
        linestyle="--",
        label=f"CV Val Accuracy ({cv_mean_val_acc:.3f})",
    )
axes[1].set_title("Classification Model Accuracy")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()

# print summary
print(f"Final train loss:     {history['loss'][-1]:.4f}")
print(f"Final train accuracy: {history['accuracy'][-1]:.4f}")
print(f"Max train accuracy:   {max(history['accuracy']):.4f}")
print(f"Epochs trained:       {len(history['loss'])}")

if cv_results is not None:
    print(f"\nCross-Validation Results:")
    for r in cv_results:
        print(
            f"  Fold {r['fold']}: val_acc={r['val_acc']:.4f}, train_acc={r['train_acc']:.4f}"
        )
    val_accs = [r["val_acc"] for r in cv_results]
    print(f"  Mean val accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")

if not os.path.exists(MOD_PERF_IMG_PATH):
    plt.savefig(MOD_PERF_IMG_PATH)
plt.show()
