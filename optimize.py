import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

print(tf.__version__)

version = input("version: ")
MODEL_PATH = f"./model_weights/mod_v{version}.keras"
LITE_MODEL_PATH = f"./model_opt/opt_v{version}.tflite"

NUM_VOCS = 78

if not os.path.exists(MODEL_PATH):
    print("model or version number not exist")
    exit(67)

model = load_model(MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# use real data distribution for representative dataset (better quantization)
# preprocessing matches main.py: log1p + MinMaxScaler
dataset_csv = pd.read_csv("../sip_data/res1/combined_synthetic_and_real.csv")
data_raw = dataset_csv.iloc[:, 2:].values
data_log = np.log1p(data_raw)
minmaxscaler = MinMaxScaler()
data_normalized = minmaxscaler.fit_transform(data_log).astype("float32")


def representative_dataset_gen():
    for i in range(len(data_normalized)):
        yield [data_normalized[i : i + 1]]


converter.representative_dataset = representative_dataset_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter.inference_input_type = tf.float32
converter.inference_output_type = tf.uint8

tf_lite_model = converter.convert()

with open(LITE_MODEL_PATH, "wb") as f:
    f.write(tf_lite_model)

print("success!")
