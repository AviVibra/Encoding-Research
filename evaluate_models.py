"""Training summary and test-set accuracy of all 12 models.

Writes results_all.json (accuracy of every model on all 13 test sets).
"""
import json

import torch

from train import FOLDER, POS_TYPES, SEEDS, evaluate, load_model

test_sets = torch.load("test_sets.pt")
results = {}

for pos_type in POS_TYPES:
    for seed in SEEDS:
        name = f"{pos_type}_seed{seed}"
        with open(f"{FOLDER}/{name}_log.json") as f:
            log = json.load(f)
        ckpt = torch.load(f"{FOLDER}/{name}.pt", weights_only=False, map_location="cpu")
        print(f"{name:12s} steps={ckpt['steps']:>6}  final_acc={log[-1]['acc']:.3f}  "
              f"reached={ckpt['reached_target']}")

        model = load_model(name)
        results[name] = {t: round(evaluate(model, x, y), 3) for t, (x, y) in test_sets.items()}
        print(name, results[name])

with open("results_all.json", "w") as f:
    json.dump(results, f, indent=1)
