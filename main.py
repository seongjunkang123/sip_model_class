import pickle
import os
import pandas as pd
import tensorflow as tf
from keras import layers, Model
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

from utils import get_version

# constants
EPOCHS = 100
BATCH_SIZE = 16

# get model version
v = get_version()

# get data from csv file
DATASET_PATH = "../sip_data/res1/combined_synthetic_and_real.csv"
dataset = pd.read_csv(DATASET_PATH)

data_raw = dataset.iloc[:, 2:].values
label_raw = dataset['Disease'].values

# encode labels to integers (0, 1, ..., n_classes-1)
label_encoder = LabelEncoder()
label = label_encoder.fit_transform(label_raw)

# scale voc profile data from 0 -> 1
minmaxscaler = MinMaxScaler()
data = minmaxscaler.fit_transform(data_raw)

# create tensorflow dataset object + shuffle
dataset_tf = tf.data.Dataset.from_tensor_slices((data, label))
dataset = dataset_tf.shuffle(buffer_size=1000).batch(BATCH_SIZE)

# constants for node counts
NUM_CLASS = 3 # asthma, bronchi, copd
NUM_VOCS = data.shape[1]

# building and structuring the model
def build_model():
    input = layers.Input(shape=(NUM_VOCS, ))

    mod = layers.Dense(128, activation='relu')(input)
    mod = layers.BatchNormalization()(mod)
    mod = layers.Dropout(0.25)(mod)

    mod = layers.Dense(256, activation='relu')(input)
    mod = layers.BatchNormalization()(mod)
    mod = layers.Dropout(0.25)(mod)

    # mod = layers.Dense(512, activation='relu')(input)
    # mod = layers.BatchNormalization()(mod)
    # mod = layers.Dropout(0.25)(mod)

    output = layers.Dense(NUM_CLASS, activation='softmax')(mod)

    return Model(input, output, name=f"class_model_v{v}")

class_model = build_model()

# define paths for classification model
MOD_WEIGHTS_PATH    = f"./model_weights/mod_v{v}.keras"
MOD_INFO_PATH       = f"./model_info/mod_v{v}.txt"
MOD_PERF_PATH       = f"./model_performance/mod_v{v}.pkl"
MOD_PERF_IMG_PATH   = f"./model_performance/mod_v{v}.png"

# write model information to a txt file
with open(MOD_INFO_PATH, 'w') as f:
    f.write(f"gen_v{v} info\n\n")

    f.write(f"Number of Classes: {NUM_CLASS}\n")
    f.write(f"Number of VOCs: {NUM_VOCS}\n")
    f.write(f"Epoch: {EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n\n")

    class_model.summary(print_fn=lambda x: f.write(x + '\n'))

# set optimizer and loss function
class_model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# callbacks
early_stopping_callback = EarlyStopping(
    patience=20,
    restore_best_weights=True
)
checkpoint_callback = ModelCheckpoint(
    filepath=MOD_WEIGHTS_PATH,
    save_best_only=True,
)

# train model
history = class_model.fit(dataset, epochs=EPOCHS, callbacks=[early_stopping_callback, checkpoint_callback])

class_model.save(MOD_WEIGHTS_PATH)

with open(MOD_PERF_PATH, 'wb') as f:
    pickle.dump(history.history, f)

plt.plot(history.history['loss'], label='Class. Model Loss')
plt.plot(history.history['accuracy'], label="Class. Model Acc")
plt.title('Classification Model Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.grid()

if not os.path.exists(MOD_PERF_IMG_PATH):
    plt.savefig(MOD_PERF_IMG_PATH)
plt.show()