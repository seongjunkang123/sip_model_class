import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ── user inputs ──────────────────────────────────────────────────────────────
class_version = input("Classification model version: ")
gen_version = input("Generator model version: ")
n_samples = int(input("Samples per condition (e.g. 1000): "))

# ── constants ────────────────────────────────────────────────────────────────
LATENT_DIM = 128
NUM_CLASSES = 3
CONDITIONS = ["asthma", "bronchi", "copd"]

# ── load generator ───────────────────────────────────────────────────────────
gen_path = f"../sip_model_gen/gen_model_weights/gen_v{gen_version}.keras"
generator = load_model(gen_path)
print(f"Loaded generator v{gen_version}")

# load generator preprocessing params
preproc_path = f"../sip_model_gen/gen_model_weights/preprocessing_v{gen_version}.json"
try:
    with open(preproc_path, "r") as f:
        preproc = json.load(f)
    use_log = preproc.get("log_transform", False)
except FileNotFoundError:
    print(f"WARNING: {preproc_path} not found, assuming no log transform.")
    use_log = False

# fit generator's scaler
real_df = pd.read_csv('../sip_data/res1/combined_data.csv')
real_data = real_df.iloc[:, 2:].values
voc_list = real_df.columns[2:]

if use_log:
    gen_scaler_data = np.log1p(real_data)
else:
    gen_scaler_data = real_data

gen_scaler = MinMaxScaler()
gen_scaler.fit(gen_scaler_data)

# ── load classifier ──────────────────────────────────────────────────────────
class_path = f"./model_weights/mod_v{class_version}.keras"
classifier = load_model(class_path)
print(f"Loaded classifier v{class_version}")

# fit classifier's scaler (log1p + MinMaxScaler on combined training data)
train_df = pd.read_csv('../sip_data/res1/combined_synthetic_and_real.csv')
train_data = train_df.iloc[:, 2:].values
class_scaler = MinMaxScaler()
class_scaler.fit(np.log1p(train_data))

# ── generate held-out test data ──────────────────────────────────────────────
print(f"\nGenerating {n_samples * NUM_CLASSES} held-out test samples...")

all_test_data = []
all_test_labels = []

for i, condition in enumerate(CONDITIONS):
    condition_onehot = tf.one_hot([i], depth=NUM_CLASSES)
    condition_onehot_batch = tf.repeat(condition_onehot, n_samples, axis=0)
    latent_vec = tf.random.normal(shape=(n_samples, LATENT_DIM))

    # generate in normalized space
    gen_output = generator.predict([latent_vec, condition_onehot_batch], verbose=0)

    # inverse transform to raw VOC values
    gen_raw = gen_scaler.inverse_transform(gen_output)
    if use_log:
        gen_raw = np.expm1(gen_raw)
    gen_raw = np.maximum(gen_raw, 0)

    all_test_data.append(gen_raw)
    all_test_labels.extend([i] * n_samples)

    print(f"  Generated {n_samples} {condition} samples")

test_data_raw = np.vstack(all_test_data)
test_labels = np.array(all_test_labels)

# ── preprocess for classifier (log1p + MinMaxScaler) ─────────────────────────
test_data = class_scaler.transform(np.log1p(test_data_raw))

# ── evaluate ─────────────────────────────────────────────────────────────────
predictions_prob = classifier.predict(test_data, verbose=0)
predictions = np.argmax(predictions_prob, axis=1)

overall_acc = np.mean(predictions == test_labels)

print(f"\n{'=' * 60}")
print(f"  Held-Out Test Results ({n_samples * NUM_CLASSES} samples)")
print(f"{'=' * 60}")
print(f"  Overall accuracy: {overall_acc:.4f} ({np.sum(predictions == test_labels)}/{len(test_labels)})")

# per-class accuracy
print(f"\n  Per-class accuracy:")
for i, condition in enumerate(CONDITIONS):
    mask = test_labels == i
    class_acc = np.mean(predictions[mask] == i)
    correct = np.sum(predictions[mask] == i)
    total = mask.sum()
    print(f"    {condition:>10}: {class_acc:.4f} ({correct}/{total})")

# confusion matrix
print(f"\n  Confusion Matrix:")
print(f"  {'':>12} {'pred_asth':>10} {'pred_bron':>10} {'pred_copd':>10}")
for true_c in range(NUM_CLASSES):
    row = []
    for pred_c in range(NUM_CLASSES):
        row.append(np.sum((test_labels == true_c) & (predictions == pred_c)))
    print(f"  {CONDITIONS[true_c]:>12} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

# confidence analysis
print(f"\n  Prediction Confidence:")
for i, condition in enumerate(CONDITIONS):
    mask = test_labels == i
    correct_mask = mask & (predictions == i)
    wrong_mask = mask & (predictions != i)

    if correct_mask.sum() > 0:
        correct_conf = predictions_prob[correct_mask, i].mean()
        print(f"    {condition:>10} correct: avg confidence = {correct_conf:.4f}")
    if wrong_mask.sum() > 0:
        wrong_conf = predictions_prob[wrong_mask].max(axis=1).mean()
        print(f"    {condition:>10} wrong:   avg confidence = {wrong_conf:.4f}")

print(f"{'=' * 60}")
