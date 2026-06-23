import math
import json
import numpy as np
from datasets import load_dataset

# -----------------------------
# Load dataset
# -----------------------------
vqa = load_dataset("MilaWang/SpatialEval", "vqa", split="test")

TASK_PREFIXES = {
    "spatialmap": "spatialmap",
    "mazenav": "mazenav",
    "spatialgrid": "spatialgrid",
}

def get_task_indices(ds, prefix):
    return [i for i, ex in enumerate(ds) if ex["id"].startswith(prefix)]

task_indices = {t: get_task_indices(vqa, p) for t, p in TASK_PREFIXES.items()}

# -----------------------------
# Split config
# -----------------------------
TRAIN_FRAC = 0.20
SEEDS = [0, 1, 2]

def make_train_test_split(task_indices_dict, seed):
    rng = np.random.RandomState(seed)

    out = {
        "seed": seed,
        "train_frac": TRAIN_FRAC,
        "train": {},
        "test": {},
    }

    train_all = []
    test_all = []

    for task, idxs in task_indices_dict.items():
        idxs = list(idxs)
        perm = rng.permutation(len(idxs))
        shuffled = [idxs[i] for i in perm]

        n_train = int(math.floor(len(shuffled) * TRAIN_FRAC))  # 1500 -> 300
        train = shuffled[:n_train]
        test = shuffled[n_train:]

        out["train"][task] = train
        out["test"][task] = test

        train_all.extend(train)
        test_all.extend(test)

    out["train"]["all"] = train_all
    out["test"]["all"] = test_all

    return out

# -----------------------------
# Build splits for 3 seeds
# -----------------------------
splits = {f"seed_{s}": make_train_test_split(task_indices, s) for s in SEEDS}

# -----------------------------
# Sanity checks
# -----------------------------
for k, sd in splits.items():
    for task in TASK_PREFIXES:
        A = set(sd["train"][task])
        B = set(sd["test"][task])
        assert len(A & B) == 0, (k, task, "overlap")

# -----------------------------
# Save
# -----------------------------
with open("", "w") as f:
    json.dump(splits, f, indent=2)

print("Saved: spatialeval_vqa_train20_test80.json")
print("Seeds:", list(splits.keys()))
