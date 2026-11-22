"""What breaks on 6-digit problems, all 12 models.

Writes:
    per_digit_6digit.csv                 accuracy of each answer digit (5- and 6-digit)
    attention_6digit_correct_old.csv     for every head: share of attention from the answer
                                         rows on the correct digits (7 and 14 tokens back)
                                         and on the digits at the old 5-digit distances
                                         (6 and 12 tokens back)
    figures/{MODEL}_seed{N}_6digit_avg.png
"""
import os

import pandas as pd
import torch

from attention_maps import FIG, get_attention, plot_attention
from train import POS_TYPES, SEEDS, device, load_model

LABELS6 = ([f"a{i}" for i in range(6)] + ["+"] +
           [f"b{i}" for i in range(6)] + ["="] +
           [f"c{i}" for i in range(6)])


def per_digit_accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        preds = model(x.to(device)).argmax(dim=-1).cpu()
    n_ans = int((y[0] != -100).sum())
    acc = (preds[:, -n_ans:] == y[:, -n_ans:]).float().mean(dim=0)
    return [round(v, 3) for v in acc.tolist()]


def correct_vs_old(avg):
    """Rows 14-19 predict the answer digits of a 6-digit problem."""
    rows = {}
    for l, W in enumerate(avg):
        for h in range(W.shape[0]):
            new = sum((W[h, r, r - 14] + W[h, r, r - 7]).item() for r in range(14, 20)) / 6
            old = sum((W[h, r, r - 12] + W[h, r, r - 6]).item() for r in range(14, 20)) / 6
            rows[f"L{l + 1}H{h}"] = {"correct": round(new, 2), "old": round(old, 2)}
    return rows


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    test_sets = torch.load("test_sets.pt")
    x6, y6 = test_sets["6_random"]
    digits, att = {}, {}
    for pos_type in POS_TYPES:
        for seed in SEEDS:
            name = f"{pos_type}_seed{seed}"
            model = load_model(name)
            for t in ["5_random", "6_random"]:
                acc = per_digit_accuracy(model, *test_sets[t])
                digits[(name, t)] = {f"digit {i + 1}": v for i, v in enumerate(acc)}
            avg = [m.mean(dim=0) for m in get_attention(model, x6)]
            plot_attention(avg, LABELS6, f"{pos_type} seed {seed} - 6-digit average of 1000 problems",
                           f"{FIG}/{name}_6digit_avg.png")
            for head, v in correct_vs_old(avg).items():
                att[(name, head)] = v
            print(name, "done")

    d = pd.DataFrame(digits).T
    d.index.names = ["model", "test set"]
    d.to_csv("per_digit_6digit.csv")
    a = pd.DataFrame(att).T
    a.index.names = ["model", "head"]
    a.to_csv("attention_6digit_correct_old.csv")
    print(d)
    print(a)
