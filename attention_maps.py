"""Attention maps and measured attention offsets for all 12 models.

Figures (folder figures/):
    PE_seed0_one.png              attention for one problem (PE seed 0)
    {MODEL}_seed{N}_avg.png       average attention of 1,000 5-digit problems
Table:
    attention_offsets.csv         for every head: share of attention from the answer rows
                                  (c1..c4) on the same column, one column right (+1),
                                  two columns right (+2), '+', '=' and other tokens
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from data import decode
from train import POS_TYPES, SEEDS, device, load_model

FIG = "figures"
os.makedirs(FIG, exist_ok=True)

LABELS5 = ["a0", "a1", "a2", "a3", "a4", "+",
           "b0", "b1", "b2", "b3", "b4", "=",
           "c0", "c1", "c2", "c3", "c4"]


def get_attention(model, x):
    with torch.no_grad():
        model(x.to(device))
    return [block.attention_layer.last_weights.cpu() for block in model.block]


def plot_attention(maps, labels, title, save_as=None):
    n_layers, n_heads = len(maps), maps[0].shape[0]
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(4 * n_heads, 4 * n_layers))
    for l in range(n_layers):
        for h in range(n_heads):
            ax = axes[l, h]
            ax.imshow(maps[l][h], cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_title(f"Layer {l + 1}, head {h}")
            ax.set_xlabel("looked at")
            ax.set_ylabel("looking")
    fig.suptitle(title)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=150)
    plt.close(fig)


def attention_offsets(avg):
    """avg: list (one per layer) of (heads, T, T) average maps of 5-digit problems.
    Rows 12-15 are the tokens that predict answer digits c1..c4 (columns 0..3)."""
    rows = {}
    for l, W in enumerate(avg):
        for h in range(W.shape[0]):
            cats = {"same": 0.0, "+1": 0.0, "+2": 0.0, "plus": 0.0, "eq": 0.0}
            for row in range(12, 16):
                w = W[h, row]
                cats["same"] += (w[row - 12] + w[row - 6]).item()
                cats["+1"] += (w[row - 11] + w[row - 5]).item()
                if row - 10 <= 4:
                    cats["+2"] += (w[row - 10] + w[row - 4]).item()
                cats["plus"] += w[5].item()
                cats["eq"] += w[11].item()
            cats = {k: round(v / 4, 2) for k, v in cats.items()}
            cats["other"] = round(1 - sum(cats.values()), 2)
            rows[f"L{l + 1}H{h}"] = cats
    return rows


if __name__ == "__main__":
    test_sets = torch.load("test_sets.pt")
    x, y = test_sets["5_random"]

    # one problem, PE seed 0
    model = load_model("PE_seed0")
    one = [m[0] for m in get_attention(model, x[:1])]
    plot_attention(one, list(decode(x[0])), "PE seed 0 - one problem", f"{FIG}/PE_seed0_one.png")

    offsets = {}
    for pos_type in POS_TYPES:
        for seed in SEEDS:
            name = f"{pos_type}_seed{seed}"
            model = load_model(name)
            avg = [m.mean(dim=0) for m in get_attention(model, x)]
            plot_attention(avg, LABELS5, f"{pos_type} seed {seed} - average of 1000 problems",
                           f"{FIG}/{name}_avg.png")
            for head, cats in attention_offsets(avg).items():
                offsets[(name, head)] = cats
            print(name, "done")

    table = pd.DataFrame(offsets).T
    table.index.names = ["model", "head"]
    table.to_csv("attention_offsets.csv")
    print(table)
