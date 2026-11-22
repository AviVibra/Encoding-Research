"""Mean ablation of every attention head, all 12 models.

Each head is replaced by its average output over the validation set, and the model is
tested again on the five 5-digit test sets.
Writes ablation/{MODEL}_seed{N}_ablation.csv (row "none" = no head ablated).
"""
import os

import pandas as pd
import torch

from train import POS_TYPES, SEEDS, device, evaluate, load_model, val_set

SETS = ["5_random", "5_nocarry", "5_carry", "5_sum9", "5_cascade"]
OUT = "ablation"
os.makedirs(OUT, exist_ok=True)
test_sets = torch.load("test_sets.pt")


def compute_head_means(model, x):
    with torch.no_grad():
        model(x.to(device))
    for block in model.block:
        attn = block.attention_layer
        attn.head_mean = attn.last_heads_out.mean(dim=0)


def ablation_table(name):
    model = load_model(name)
    compute_head_means(model, val_set[0])
    rows = {"none": {s: evaluate(model, *test_sets[s]) for s in SETS}}
    for l, block in enumerate(model.block):
        attn = block.attention_layer
        for h in range(attn.num_heads):
            attn.head_ablate = h
            rows[f"L{l + 1}H{h}"] = {s: evaluate(model, *test_sets[s]) for s in SETS}
            attn.head_ablate = None
    return pd.DataFrame(rows).T.round(3)


if __name__ == "__main__":
    for pos_type in POS_TYPES:
        for seed in SEEDS:
            name = f"{pos_type}_seed{seed}"
            table = ablation_table(name)
            print(name)
            print(table)
            table.to_csv(f"{OUT}/{name}_ablation.csv")
