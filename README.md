Do Transformers Learn the Same Addition Method With Every Positional Encoding?

📄 Paper: Positional_Encodings_Addition_Research.pdf · arXiv: (link coming soon)

I built small decoder-only transformers from scratch, trained them on 5-digit addition with four different positional encodings, and then worked out how each one adds. I asked two questions:

Do transformers learn the same method for adding numbers, no matter which positional encoding they use?
When a model fails on numbers longer than it was trained on, which part of its method breaks?
Encoding	Code name	Description
Learned PE	PE	Learned absolute position embedding added to the input
RoPE	ROPE	Rotary embedding applied to queries and keys
ALiBi	ALBI	Linear distance bias added to the attention scores
NoPE	NOPE	No positional encoding at all

Each encoding was trained with 3 seeds (12 models in total), and every analysis was run on all 12.

Key findings
With 1 layer, ALiBi and NoPE could not learn addition at all. With 2 layers, all four encodings reach ≥97% accuracy.
The encodings split into two groups:
PE and RoPE read both digits of a column with the same heads and work out the carry early, in layer 1, using dedicated carry heads.
ALiBi and NoPE first build position information in layer 1, read the two numbers with separate heads, and work out the carry late, in layer 2.
On 6-digit numbers, every model fails, but in different places:
PE and RoPE lose the carry already in layer 1. Their layer 2 heads keep looking at the old 5-digit distances.
ALiBi and NoPE keep layer 1 working and break in layer 2.
ALiBi is the only encoding with a layer 2 head that still partly finds the correct digits (34–66% of its attention), so it is the only one that keeps some per-digit accuracy on 6-digit numbers.

Conclusion: the positional encoding changes part of the addition method, and it decides where the method breaks on longer numbers.

Carry information at the end of the model (linear probe, 5-digit problems, 3 seeds)
Encoding	Probe accuracy
PE	99.4–99.6%
RoPE	99.6–99.7%
ALiBi	90.1–96.3%
NoPE	88.1–90.5%
Methods
Attention maps. These are averaged over 1,000 problems. For every head I also measured how much of its attention goes to the same column, the column one or two places to the right, +, and =.
Mean ablation. Each head is replaced by its average output, and the model is tested again on random, no-carry, carry, sum-9 and cascading-carry problems.
Linear probes. These test whether the carry can be read from the residual stream after the embedding and after each attention and feed-forward layer.
Longer numbers. On 6-digit problems I measured per-digit accuracy, drew 6-digit attention maps, and measured whether the attention goes to the correct digits or to the old 5-digit distances.
Repository structure
├── model.py               # transformer from scratch + the 4 positional encodings
├── data.py                # tokenizer, problem generators, fixed test sets
├── train.py               # trains all 12 models (resumes after a disconnect)
├── evaluate_models.py     # accuracy on all 13 test sets
├── attention_maps.py      # attention maps + measured attention offsets
├── ablation.py            # mean ablation of every head
├── probes.py              # linear carry probes (5-digit, 5→6-digit, convergence check)
├── longer_numbers.py      # 6-digit per-digit accuracy and attention
├── requirements.txt
└── Positional_Encodings_Addition_Research.pdf
How to run
bash
pip install -r requirements.txt

python data.py              # 1. create the fixed test sets  -> test_sets.pt
python train.py             # 2. train the 12 models         -> checkpoints/   (GPU recommended)
python evaluate_models.py   # 3. accuracy                    -> results_all.json
python attention_maps.py    # 4. analysis                    -> figures/, attention_offsets.csv
python ablation.py          #                                -> ablation/
python probes.py            #                                -> probe_*.csv
python longer_numbers.py    #                                -> per_digit_6digit.csv, figures/*_6digit_avg.png

train.py skips models that already exist, so you can stop and restart it at any time.

Setup
Setting	Value
Problem format	12345+67890=080235 (12-token vocabulary, loss only on the answer digits)
Model	decoder-only, 2 layers, 3 heads, d_model 510, pre-norm, GELU feed-forward
Optimizer	AdamW, lr 8e-5, weight decay 0.1, batch size 64
Training data	fresh random problems every step, 20% cascading-carry problems
Stopping	≥97% validation accuracy twice in a row, or 160,000 steps
Test sets	13 fixed sets × 1,000 problems (5-digit random / no carry / carry / sum 9 / cascade, and 6–9 digits)
Related work

This project builds on Quirke & Barez's work on understanding addition in transformers, and on the original papers for each positional encoding (RoPE, ALiBi, NoPE). The full reference list is in the paper.

Citation
bibtex
@misc{addition_positional_encodings,
  title  = {Do Transformers Learn the Same Addition Method With Every Positional Encoding?},
  author = {Abdulrhman Bakr},
  year   = {2026},
  note   = {arXiv link coming soon}
}
Author

Abdulrhman Mohammad Bakr, independent high school research project.
