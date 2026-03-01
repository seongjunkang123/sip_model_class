import os, re

def get_version():
    gen_model_weights_directory = './model_weights'

    try:
        files = os.listdir(gen_model_weights_directory)
    except FileNotFoundError:
        os.makedirs(gen_model_weights_directory)
        files = []

    if not files:
        v = 1 # initial
        print(f"Version: {v}")
        return v

    # extract numbers from filenames and find the max.
    max_version = 0
    for file in files:
        numbers = re.findall(r'\d+', file)
        if numbers:
            version = int(numbers[-1])
            if version > max_version:
                max_version = version

    v = max_version + 1

    print(f"Version: {v}")
    return v