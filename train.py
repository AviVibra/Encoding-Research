import os,json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_model import build_transformer
from data import vocab_size,make_batch
import random
config = {
    "d_model": 510, "num_heads": 3, "max_len": 64,
    "batch_size": 64, "n_steps": 30000, "lr": 0.00008,
    "weight_decay": 0.1, "cascade_share": 0.2,"n_layers": 2,
}


device = torch.device("cuda" if torch.cuda.is_available() else "mps")

def train(pos_type,seed):
    random.seed(seed)
    torch.manual_seed(seed)
    log = []
    transformer = build_transformer(vocab_size,config["d_model"],config["num_heads"],config["max_len"],pos_type,config["n_layers"]).to(device)
    optimizer = torch.optim.AdamW(transformer.parameters(),lr=config["lr"],weight_decay=config["weight_decay"])
    test_set = torch.load("test_sets.pt")
    transformer.train()
    for step in range(config["n_steps"]):
        x,y = make_batch(config["batch_size"],5,config["cascade_share"])
        x,y = x.to(device),y.to(device)
        logits = transformer(x)
        loss = F.cross_entropy(logits.view(-1,vocab_size),y.view(-1),ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == config["n_steps"] - 1:
            acc = evaluate(transformer, *test_set["5_random"])
            log.append({"step": step, "loss": loss.item(), "acc": acc})
            print(step, round(loss.item(), 4), acc)
    os.makedirs("checkpoints_1layer", exist_ok=True)
    name = f"{pos_type}_seed{seed}"
    torch.save({"model": transformer.state_dict(), "config": config,
                "pos_type": pos_type, "seed": seed}, f"checkpoints_1layer/{name}.pt")
    with open(f"checkpoints_1layer/{name}_log.json", "w") as f:
        json.dump(log, f)

def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=-1)
        mask = y != -100
        correct = ((preds == y) | ~mask).all(dim=1)
        acc = correct.float().mean().item()
    model.train()
    return acc

if __name__ == "__main__":
    for pos_type in ["PE", "ROPE", "ALBI", "NOPE"]:
        for seed in [0, 1, 2]:
            if pos_type == "PE" and seed == 0:
                continue
            print(f"=== {pos_type} seed {seed} ===")
            train(pos_type, seed)