import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model, layers
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from utils import get_version

# ── constants ────────────────────────────────────────────────────────────────
EPOCHS = 300
BATCH_SIZE = 16
N_FOLDS = 5  # for cross-validation
LEARNING_RATE = 1e-3

# get model version
v = get_version()

# ── load data ────────────────────────────────────────────────────────────────
DATASET_PATH = "../sip_data/res1/combined_synthetic_and_real.csv"
dataset = pd.read_csv(DATASET_PATH)

data_raw = dataset.iloc[:, 2:].values
label_raw = dataset["Disease"].values

# ── preprocessing: log1p + MinMaxScaler ──────────────────────────────────────
# Matches the GAN preprocessing pipeline. VOC data is heavily skewed,
# so log1p compresses the range before scaling.
data_log = np.log1p(data_raw)

minmaxscaler = MinMaxScaler()
data = minmaxscaler.fit_transform(data_log)

# encode labels to integers (0, 1, ..., n_classes-1)
label_encoder = LabelEncoder()
label = label_encoder.fit_transform(label_raw)

# ── constants for architecture ───────────────────────────────────────────────
NUM_CLASS = 3  # asthma, bronchi, copd
NUM_VOCS = data.shape[1]

print(f"Samples: {data.shape[0]}, Features: {NUM_VOCS}, Classes: {NUM_CLASS}")
print(f"Class distribution: {dict(zip(*np.unique(label_raw, return_counts=True)))}")


# ── model architecture ───────────────────────────────────────────────────────
def build_model():
    input_layer = layers.Input(shape=(NUM_VOCS,))

    x = layers.Dense(256, activation="relu")(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    output = layers.Dense(NUM_CLASS, activation="softmax")(x)

    return Model(input_layer, output, name=f"class_model_v{v}")


# ── cross-validation ─────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_results = []
best_val_acc = 0
best_fold = 0

print(f"\n{'=' * 60}")
print(f"  {N_FOLDS}-Fold Stratified Cross-Validation")
print(f"{'=' * 60}")

for fold, (train_idx, val_idx) in enumerate(skf.split(data, label)):
    print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

    X_train, X_val = data[train_idx], data[val_idx]
    y_train, y_val = label[train_idx], label[val_idx]

    # compute class weights (handles imbalanced classes)
    class_counts = np.bincount(y_train)
    total = len(y_train)
    class_weight = {
        i: total / (NUM_CLASS * count) for i, count in enumerate(class_counts)
    }

    # create tf datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=1000).batch(BATCH_SIZE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(BATCH_SIZE)

    # build fresh model for each fold
    model = build_model()

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # callbacks
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=30,
        restore_best_weights=True,
    )
    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=[early_stopping, lr_scheduler],
        verbose=0,
    )

    # evaluate
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    train_loss, train_acc = model.evaluate(train_ds, verbose=0)

    fold_results.append(
        {
            "fold": fold + 1,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epochs_trained": len(history.history["loss"]),
        }
    )

    print(f"  Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f}")
    print(f"  Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
    print(f"  Epochs: {len(history.history['loss'])}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_fold = fold + 1

# ── cross-validation summary ─────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  Cross-Validation Summary")
print(f"{'=' * 60}")

val_accs = [r["val_acc"] for r in fold_results]
train_accs = [r["train_acc"] for r in fold_results]
print(f"  Mean val accuracy:   {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
print(f"  Mean train accuracy: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
print(f"  Best fold: {best_fold} (val_acc = {best_val_acc:.4f})")
print(f"  Overfitting gap:     {np.mean(train_accs) - np.mean(val_accs):.4f}")

# ── train final model on all data ────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  Training final model on all data (v{v})")
print(f"{'=' * 60}")

final_model = build_model()

final_model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# class weights for full dataset
class_counts = np.bincount(label)
total = len(label)
class_weight = {i: total / (NUM_CLASS * count) for i, count in enumerate(class_counts)}

full_ds = tf.data.Dataset.from_tensor_slices((data, label))
full_ds = full_ds.shuffle(buffer_size=1000).batch(BATCH_SIZE)

early_stopping_final = EarlyStopping(
    monitor="accuracy",
    patience=30,
    restore_best_weights=True,
)
lr_scheduler_final = ReduceLROnPlateau(
    monitor="loss",
    factor=0.5,
    patience=10,
    min_lr=1e-6,
)

# define paths
MOD_WEIGHTS_PATH = f"./model_weights/mod_v{v}.keras"
MOD_INFO_PATH = f"./model_info/mod_v{v}.txt"
MOD_PERF_PATH = f"./model_performance/mod_v{v}.pkl"
MOD_PERF_IMG_PATH = f"./model_performance/mod_v{v}.png"

checkpoint_callback = ModelCheckpoint(
    filepath=MOD_WEIGHTS_PATH,
    save_best_only=True,
)

history = final_model.fit(
    full_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=[early_stopping_final, lr_scheduler_final, checkpoint_callback],
)

# ── save model info ──────────────────────────────────────────────────────────
with open(MOD_INFO_PATH, "w") as f:
    f.write(f"class_model_v{v} info\n\n")
    f.write(f"Architecture: Dense-only (256→128→64→32→3)\n")
    f.write(f"Preprocessing: log1p + MinMaxScaler\n")
    f.write(f"Number of Classes: {NUM_CLASS}\n")
    f.write(f"Number of VOCs: {NUM_VOCS}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Class Weights: {class_weight}\n\n")
    f.write(f"Cross-Validation Results:\n")
    f.write(f"  Mean val accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}\n")
    f.write(f"  Best fold: {best_fold} (val_acc = {best_val_acc:.4f})\n\n")
    final_model.summary(print_fn=lambda x: f.write(x + "\n"))

# ── save weights and history ─────────────────────────────────────────────────
final_model.save(MOD_WEIGHTS_PATH)

# save both CV results and training history
perf_data = {
    "history": history.history,
    "cv_results": fold_results,
    "cv_mean_val_acc": float(np.mean(val_accs)),
    "cv_std_val_acc": float(np.std(val_accs)),
}

with open(MOD_PERF_PATH, "wb") as f:
    pickle.dump(perf_data, f)

# ── plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["loss"], label="Train Loss")
axes[0].set_title("Classification Model Loss")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="Train Accuracy")
axes[1].axhline(
    y=np.mean(val_accs),
    color="r",
    linestyle="--",
    label=f"CV Val Accuracy ({np.mean(val_accs):.3f})",
)
axes[1].set_title("Classification Model Accuracy")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.savefig(MOD_PERF_IMG_PATH)
plt.show()
