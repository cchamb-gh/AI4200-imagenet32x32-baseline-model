import os
import csv
import importlib
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

IMPORT_NAME = "residual_mlp_baseline"
DATA_DIR = "data"

# Which tasks to run. Remove some while testing to save time.
#TASKS_TO_RUN = ["mnist", "cifar10", "imagenet32"]
TASKS_TO_RUN = ["mnist", "cifar10"]
#TASKS_TO_RUN = ["imagenet32"]
#TASKS_TO_RUN = ["cifar10"]
#TASKS_TO_RUN = ["mnist"]
NUM_CLASSES = {"mnist": 10, "cifar10": 10, "imagenet32": 1000}
IMPORT_NAME = os.environ.get("HARNESS_IMPORT_NAME", IMPORT_NAME)

SEED = 0


student = importlib.import_module(IMPORT_NAME)   # your package
train_module = importlib.import_module(IMPORT_NAME + ".train")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Testing", IMPORT_NAME, "on", DEVICE)

def load_task(task):
    """Return X_train, y_train, X_test, y_test, downloading only if there is no saved copy.

    The arrays are memory-mapped: they are read from disk as they are used
    instead of all being loaded into RAM up front.
    """
    folder = Path(os.getcwd()) / DATA_DIR / task
    #folder = os.path.join(DATA_DIR, task)
    old_file = os.path.join(DATA_DIR, f"{task}.npz")   # saved by an earlier version of this notebook

    if not os.path.isdir(folder) and os.path.exists(old_file):
        # Unpack the earlier single-file copy into the folder format; no download needed.
        print(f"{task}: unpacking {old_file} (one time only; you can delete it afterwards)")
        temp_folder = folder + ".partial"
        os.makedirs(temp_folder, exist_ok=True)
        # An .npz file is a zip of .npy files, so extracting it streams them to disk
        # without loading the arrays into memory.
        with zipfile.ZipFile(old_file) as old:
            old.extractall(temp_folder)
        os.replace(temp_folder, folder)

    if not os.path.isdir(folder):                    # no saved copy yet: download it
        from datasets import load_dataset
        name, image_column, train_split, test_split = SOURCES[task]
        mode = "L" if task == "mnist" else "RGB"
        print(f"{task}: getting {name} from Hugging Face (first run only; reuses its download cache if present)")

        # Write into a temporary folder, then rename it. An interrupted run then
        # never leaves a half-written folder that looks like a finished one.
        temp_folder = folder + ".partial"
        os.makedirs(temp_folder, exist_ok=True)
        save_split(load_dataset(name, split=train_split), image_column, mode, temp_folder, "train")
        save_split(load_dataset(name, split=test_split), image_column, mode, temp_folder, "test")
        os.replace(temp_folder, folder)

    # mmap_mode="c" (copy-on-write): read from disk on demand, and any changes
    # your code makes to the arrays stay in memory and never touch the files.
    return tuple(np.load(os.path.join(folder, f"{part}.npy"), mmap_mode="c")
                 for part in ["X_train", "y_train", "X_test", "y_test"])

def get_activations(net, x):
    """Forward pass, no gradients. Returns a list: acts[0] = x (the input), acts[l] = output of the l-th ReLU (l = 1, 2, ...).
       Each acts[l] is an (n_inputs, n_units) tensor: one row per input, one column per unit in that layer."""
    acts = [x]
    h = x
    with torch.no_grad():
        for layer in net:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                acts.append(h)
    return acts


def main():
    results = []

    for task in TASKS_TO_RUN:
        print(f"\n===== {task} =====")
        X_train, y_train, X_test, y_test = load_task(task)

        # Train from scratch.
        start = time.time()

        # model = student.train(task, X_train, y_train, seed=SEED, device=DEVICE)
        model, history = student.train_and_evaluate(task, X_train, y_train, X_test, y_test, seed=SEED, device=DEVICE)
        train_minutes = (time.time() - start) / 60


        # Predict on the held-out test set.
        probs = torch.as_tensor(student.predict(model, X_test)).float().cpu()

        # Check that predict() returned one probability row per test image.
        assert probs.shape == (len(y_test), NUM_CLASSES[task]), f"predict() returned shape {tuple(probs.shape)}"
        assert torch.allclose(probs.sum(dim=1), torch.ones(len(probs)), atol=1e-3), "rows of predict() must sum to 1"

        accuracy = (probs.argmax(dim=1).numpy() == y_test).mean()
        #results.append((task, accuracy, train_minutes))
        print(f"{task}: test accuracy {accuracy:.2%}  (training took {train_minutes:.1f} min)")


        del model, X_train, y_train, X_test, y_test, probs   # free memory before the next task
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        pass

if __name__ == "__main__":
    main()