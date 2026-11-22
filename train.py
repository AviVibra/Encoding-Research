"""Train the 12 models (4 positional encodings x seeds 0, 1, 2).

Each model trains on fresh random 5-digit problems until it gets >= 97% on the
validation set twice in a row (checked every 500 steps), or until 160,000 steps.
A resume file is saved every 5,000 steps, so a disconnected run continues where it stopped.
Models that already exist in checkpoints/ are skipped.
"""
import json
import os
import random

import torch
import torch.nn.functional as F

from data import make_batch, make_problem, to_xy, vocab_size
from model import build_transformer

config = {
    "d_model": 510, "num_heads": 3, "max_len": 64,
    "batch_size": 64, "n_steps": 160000, "lr": 0.00008,
    "weight_decay": 0.1, "cascade_share": 0.2, "n_layers": 2, "tgt_acc": 0.97,
}
POS_TYPES = ["PE", "ROPE", "ALBI", "NOPE"]
SEEDS = [0, 1, 2]
FOLDER = "checkpoints"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# validation set (separate from the test sets)
random.seed(1234)
val_set = to_xy([make_problem(5) for _ in range(1000)], 5)


def evaluate(model, x, y):
    """Share of problems where every answer digit is correct."""
    model.eval()
    with torch.no_grad():
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=-1)
        mask = y != -100
        correct = ((preds == y) | ~mask).all(dim=1)
        acc = correct.float().mean().item()
    model.train()
    return acc


def load_model(name):
    """Load a trained model from checkpoints/, e.g. load_model("ROPE_seed1")."""
    ckpt = torch.load(f"{FOLDER}/{name}.pt", weights_only=False, map_location=device)
    c = ckpt["config"]
    model = build_transformer(vocab_size, c["d_model"], c["num_heads"], c["max_len"],
                              ckpt["pos_type"], c["n_layers"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def train(pos_type, seed):
    os.makedirs(FOLDER, exist_ok=True)
    name = f"{pos_type}_seed{seed}"
    resume_path = f"{FOLDER}/{name}_resume.pt"

    random.seed(seed)
    torch.manual_seed(seed)
    transformer = build_transformer(vocab_size, config["d_model"], config["num_heads"],
                                    config["max_len"], pos_type, config["n_layers"]).to(device)
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    log, hits, start = [], 0, 0

    if os.path.exists(resume_path):
        r = torch.load(resume_path, weights_only=False)
        transformer.load_state_dict(r["model"])
        optimizer.load_state_dict(r["optimizer"])
        log, hits, start = r["log"], r["hits"], r["step"] + 1
        random.setstate(r["py_rng"])
        print(f"resuming {name} from step {start}")

    transformer.train()
    step = start
    for step in range(start, config["n_steps"]):
        x, y = make_batch(config["batch_size"], 5, config["cascade_share"])
        x, y = x.to(device), y.to(device)
        logits = transformer(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == config["n_steps"] - 1:
            acc = evaluate(transformer, *val_set)
            log.append({"step": step, "loss": loss.item(), "acc": acc})
            print(step, round(loss.item(), 4), acc)
            hits = hits + 1 if acc >= config["tgt_acc"] else 0
            if hits >= 2:
                break
        if step % 5000 == 0 and step > 0:
            torch.save({"model": transformer.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "log": log, "hits": hits,
                        "py_rng": random.getstate()}, resume_path)

    torch.save({"model": transformer.state_dict(), "config": config,
                "pos_type": pos_type, "seed": seed, "steps": step,
                "reached_target": hits >= 2}, f"{FOLDER}/{name}.pt")
    with open(f"{FOLDER}/{name}_log.json", "w") as f:
        json.dump(log, f)
    if os.path.exists(resume_path):
        os.remove(resume_path)


if __name__ == "__main__":
    print(device)
    for pos_type in POS_TYPES:
        for seed in SEEDS:
            if os.path.exists(f"{FOLDER}/{pos_type}_seed{seed}.pt"):
                print(f"skip {pos_type} seed {seed}")
                continue
            print(f"=== {pos_type} seed {seed} ===")
            train(pos_type, seed)
