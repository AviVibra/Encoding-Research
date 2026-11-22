"""Linear carry probes, all 12 models.

Checkpoints in the residual stream:
    emb = after the embedding (input of block 1)
    A   = after layer 1 attention     B = after layer 1 feed-forward
    C   = after layer 2 attention     D = after layer 2 feed-forward
The probe reads the vectors of the tokens that predict answer digits c1..c4 and must say
whether the column to their right gives a carry.

Writes:
    probe_carry_all_seeds.csv   probe trained and tested on 5-digit problems (3,000 steps)
    probe_5to6_all_seeds.csv    probe trained on 5-digit, tested on 5- and 6-digit problems
    probe_convergence.csv       checkpoint D, 3,000 vs 10,000 probe training steps
"""
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import make_problem, to_xy
from train import POS_TYPES, SEEDS, device, load_model

CHECKPOINTS = ["emb", "A", "B", "C", "D"]


def get_vectors(model, x):
    store, hooks = {}, []
    hooks.append(model.block[0].register_forward_pre_hook(
        lambda m, inp: store.__setitem__("emb", inp[0].detach())))
    for name, layer, part in [("A", 0, 0), ("B", 0, 1), ("C", 1, 0), ("D", 1, 1)]:
        hooks.append(model.block[layer].residual_connections[part].register_forward_hook(
            lambda m, inp, out, name=name: store.__setitem__(name, out.detach())))
    with torch.no_grad():
        model(x.to(device))
    for h in hooks:
        h.remove()
    return store


def carry_labels(x, n):
    """Label for answer digit k (k = 0..n-2): does column k+1 send a carry into it?"""
    a, b = x[:, 0:n], x[:, n + 1:2 * n + 1]
    labels = torch.zeros(x.shape[0], n)
    carry = torch.zeros(x.shape[0])
    for k in range(n - 1, -1, -1):
        labels[:, k] = carry
        carry = ((a[:, k] + b[:, k] + carry) >= 10).float()
    return labels[:, :n - 1]


def probe_data(model, n_problems, seed, n_digits):
    random.seed(seed)
    x, _ = to_xy([make_problem(n_digits) for _ in range(n_problems)], n_digits)
    vecs = get_vectors(model, x)
    labels = carry_labels(x, n_digits).reshape(-1).to(device)
    start = 2 * n_digits + 2
    feats = {k: v[:, start:start + n_digits - 1, :].reshape(-1, v.shape[-1])
             for k, v in vecs.items()}
    return feats, labels


def fit_probe(X, y, epochs=3000):
    mean, std = X.mean(0), X.std(0) + 1e-6
    probe = nn.Linear(X.shape[1], 1).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=0.01)
    Xn = (X - mean) / std
    for _ in range(epochs):
        loss = F.binary_cross_entropy_with_logits(probe(Xn).squeeze(-1), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return probe, mean, std


def probe_accuracy(probe, mean, std, X, y):
    with torch.no_grad():
        preds = (probe((X - mean) / std).squeeze(-1) > 0).float()
    return round((preds == y).float().mean().item(), 3)


if __name__ == "__main__":
    carry5, carry6, conv = {}, {}, {}
    for pos_type in POS_TYPES:
        for seed in SEEDS:
            name = f"{pos_type}_seed{seed}"
            model = load_model(name)
            train_f, train_y = probe_data(model, 5000, 100, 5)
            test5_f, test5_y = probe_data(model, 1000, 200, 5)
            test6_f, test6_y = probe_data(model, 1000, 300, 6)

            row5, row6 = {}, {}
            for k in CHECKPOINTS:
                probe, m, s = fit_probe(train_f[k], train_y)
                row5[k] = probe_accuracy(probe, m, s, test5_f[k], test5_y)
                row6[k] = probe_accuracy(probe, m, s, test6_f[k], test6_y)
                if k == "D":
                    probe10, m10, s10 = fit_probe(train_f[k], train_y, epochs=10000)
                    conv[name] = {"3000 steps": row5[k],
                                  "10000 steps": probe_accuracy(probe10, m10, s10,
                                                                test5_f[k], test5_y)}
            p5, p6 = test5_y.mean().item(), test6_y.mean().item()
            row5["guess"] = round(max(p5, 1 - p5), 3)
            row6["guess"] = round(max(p6, 1 - p6), 3)
            carry5[name], carry6[name] = row5, row6
            print(name, "5-digit", row5, "| 6-digit", row6)

    pd.DataFrame(carry5).T.to_csv("probe_carry_all_seeds.csv")
    pd.DataFrame(carry6).T.to_csv("probe_5to6_all_seeds.csv")
    pd.DataFrame(conv).T.to_csv("probe_convergence.csv")
    print(pd.DataFrame(carry5).T)
    print(pd.DataFrame(carry6).T)
    print(pd.DataFrame(conv).T)
