# view generator and discriminator losses respect to epochs.

import pickle, matplotlib.pyplot as plt, os

VERSION = input("Version Number: ")
HISTORY_PATH = f"./model_performance/mod_v{VERSION}.pkl"
MOD_PERF_IMG_PATH   = f"./model_performance/mod_v{VERSION}.png"

if not os.path.exists(HISTORY_PATH):
    print(f"File not found")
    exit(67)

with open(HISTORY_PATH, 'rb') as f:
    history = pickle.load(f)

# plot history
plt.plot(history.history['loss'], label='Class. Model Loss')
plt.title('Classification Model Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.grid()

# save plot as png
if not os.path.exists(MOD_PERF_IMG_PATH):
    plt.savefig(MOD_PERF_IMG_PATH)
plt.show() # display plot