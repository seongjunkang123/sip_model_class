import tensorflow as tf
from keras.models import load_model
import numpy as np, os

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

def representative_dataset_gen():
    for _ in range(100):
        data = np.random.rand(1, NUM_VOCS).astype('float32')
        yield [data]

converter.representative_dataset = representative_dataset_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tf_lite_model = converter.convert()

with open(LITE_MODEL_PATH, 'wb') as f:
    f.write(tf_lite_model)

print("success!")