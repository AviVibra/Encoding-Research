import random
import torch

tokens = "0123456789+="
stoi = {ch: i for i, ch in enumerate(tokens)}
itos = {i: ch for i, ch in enumerate(tokens)}
vocab_size = len(tokens)


def encode(x):
    return [stoi[ch] for ch in x]


def decode(x):
    return "".join(itos[int(i)] for i in x)


def pick_pair(rule):
    while True:
        da, db = random.randint(0, 9), random.randint(0, 9)
        s = da + db
        if (rule == "any"
                or (rule == "le8" and s <= 8)
                or (rule == "eq9" and s == 9)
                or (rule == "ge10" and s >= 10)):
            return da, db


def build_problem(rules):
    n = len(rules)
    a = b = 0
    for col, rule in enumerate(rules):
        da, db = pick_pair(rule)
        a += da * 10 ** col
        b += db * 10 ** col
    c = a + b
    return f"{a:0{n}d}+{b:0{n}d}={c:0{n + 1}d}"


def make_problem(n):
    return build_problem(["any"] * n)


def make_nocarry(n):
    return build_problem(["le8"] * n)


def make_carry(n):
    rules = ["le8"] * n
    rules[random.randint(0, n - 1)] = "ge10"
    return build_problem(rules)


def make_sum9(n):
    rules = ["le8"] * n
    for col in random.sample(range(n), random.randint(1, n)):
        rules[col] = "eq9"
    return build_problem(rules)


def make_cascade(n):
    rules = ["any"] * n
    col_length = random.randint(1, n - 1)
    start = random.randint(1, n - col_length)
    rules[start - 1] = "ge10"
    for col in range(start, start + col_length):
        rules[col] = "eq9"
    return build_problem(rules)


def to_xy(problems, n_digits):
    data = torch.tensor([encode(p) for p in problems])
    x = data[:, :-1]
    y = data[:, 1:].clone()
    y[:, :-(n_digits + 1)] = -100
    return x, y


def make_batch(batch_size, n_digits, cascade_share=0.0):
    problems = []
    for _ in range(batch_size):
        if random.random() < cascade_share:
            problems.append(make_cascade(n_digits))
        else:
            problems.append(make_problem(n_digits))
    return to_xy(problems, n_digits)


def make_test_sets(size=1000, seed=0):
    random.seed(seed)
    sets = {}
    five = {
        "random": make_problem,
        "nocarry": make_nocarry,
        "carry": make_carry,
        "sum9": make_sum9,
        "cascade": make_cascade,
    }
    for name, fn in five.items():
        sets[f"5_{name}"] = to_xy([fn(5) for _ in range(size)], 5)
    for n in [6, 7, 8, 9]:
        sets[f"{n}_random"] = to_xy([make_problem(n) for _ in range(size)], n)
        sets[f"{n}_cascade"] = to_xy([make_cascade(n) for _ in range(size)], n)
    return sets


if __name__ == "__main__":
    test_sets = make_test_sets()
    torch.save(test_sets, "test_sets.pt")
    for name, (x, y) in test_sets.items():
        print(name, tuple(x.shape), decode(x[0]))