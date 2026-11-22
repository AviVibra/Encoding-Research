import torch
from transformer_model import build_transformer
from data import vocab_size
from train import evaluate, device

ckpt = torch.load("checkpoints_1layer/PE_seed0.pt")
c = ckpt["config"]
model = build_transformer(vocab_size, c["d_model"], c["num_heads"], c["max_len"], ckpt["pos_type"],c['n_layers']).to(device)
model.load_state_dict(ckpt["model"])

test_sets = torch.load("test_sets.pt")
for name, (x, y) in test_sets.items():
    print(name, round(evaluate(model, x, y), 3))