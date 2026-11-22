# Do transformers learn the same addition method with every positional encoding?

Code for my research on how small transformers add 5-digit numbers with four positional
encodings (learned PE, RoPE, ALiBi, NoPE), three seeds each, and what breaks on 6-digit numbers.

## Files

| File | What it does | Output |
|---|---|---|
| `model.py` | Decoder-only transformer built from scratch, with the four positional encodings | – |
| `data.py` | Tokenizer, problem generators, fixed test sets | `test_sets.pt` |
| `train.py` | Trains all 12 models (skips models that already exist, resumes after a disconnect) | `checkpoints/` |
| `evaluate_models.py` | Training summary and accuracy on all 13 test sets | `results_all.json` |
| `attention_maps.py` | Average attention maps and measured attention offsets | `figures/*_avg.png`, `attention_offsets.csv` |
| `ablation.py` | Mean ablation of every head | `ablation/*_ablation.csv` |
| `probes.py` | Linear carry probes (5-digit, 5→6-digit, probe convergence) | `probe_*.csv` |
| `longer_numbers.py` | 6-digit per-digit accuracy, 6-digit attention maps, correct vs old attention | `per_digit_6digit.csv`, `attention_6digit_correct_old.csv`, `figures/*_6digit_avg.png` |

## How to run

```
pip install -r requirements.txt
python data.py              # 1. make the test sets
python train.py             # 2. train the 12 models (GPU recommended)
python evaluate_models.py   # 3. accuracy
python attention_maps.py    # 4. analysis
python ablation.py
python probes.py
python longer_numbers.py
```

All scripts run from this folder and every analysis script covers all 12 models
(4 encodings × seeds 0, 1, 2).

## Model names

Checkpoints are named `{ENCODING}_seed{N}`. The encoding codes are
`PE` (learned positions), `ROPE` (RoPE), `ALBI` (ALiBi) and `NOPE` (NoPE).

## Settings

d_model 510, 3 heads, 2 layers, AdamW (lr 8e-5, weight decay 0.1), batch 64, 20% cascade
problems in training, training stops at ≥97% validation accuracy twice in a row
(checked every 500 steps) or at 160,000 steps.
