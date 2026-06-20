# Fine-Tuned Code Reviewer (LoRA + TinyLlama)

A LoRA fine-tuning project that adapts TinyLlama-1.1B-Chat to give direct, no-sugarcoating 
code review feedback — built end to end locally, with an honest account of what worked, 
what didn't, and why.

## What it does

Fine-tunes a small open-source language model on a custom dataset of "brutally honest" 
code review examples — pairing flawed (or sometimes good) code snippets with direct, 
specific, jargon-aware feedback in a consistent voice.

## Pipeline

1. **Dataset** — hand-written `dataset.jsonl`, growing from 18 to 48 examples across 
   iterations, covering bugs (SQL injection, mutable defaults, bare excepts), redundant 
   code, security issues, edge cases, and a few genuinely good examples so the model 
   doesn't learn to criticize everything reflexively
2. **Base model** — TinyLlama-1.1B-Chat-v1.0, loaded via HuggingFace Transformers
3. **LoRA adapter** — frozen base model, trainable low-rank adapters on the attention 
   layers (`q_proj`, `v_proj`), tested at rank 8 and rank 16
4. **Training** — HuggingFace `Trainer`, run locally on Apple Silicon (MPS backend), 
   varying epochs (5–10) and dataset sizes across iterations
5. **Evaluation** — manual inspection of generated reviews on unseen code, checking both 
   voice/tone and factual correctness about what the code actually does

## How to run

```bash
pip install transformers datasets peft accelerate torch --break-system-packages
python3 train.py
```

## What actually happened — the honest iteration log

| Run | Examples | Epochs | LoRA r | Temp | Result |
|-----|----------|--------|--------|------|--------|
| 1 | 18 | 10 | 8 | 0.7 | Loss 2.86→1.70. Output confidently wrong about code logic. |
| 2 | 18 | 5 | 8 | 0.7 | Vague, non-specific feedback. Undertrained. |
| 3 | 33 | 10 | 8 | 0.7 | Loss dropped further (1.44) but output got *less* coherent — overfitting. |
| 4 | 48 | 5 | 8 | 0.7 | Output became topically relevant for the first time — correctly referenced actual list indices, suggested real alternatives — though still factually imprecise. Truncated at 100 tokens. |
| 5 | 48 | 5 | 8 | 0.7 (200 tokens) | Same model, full untruncated output confirmed the improvement was real. |
| 6 | 48 | 5 | 16 | 0.3 | Regressed into repetition loop ("This is a bug, not a feature" ×12). Lower temperature reduced randomness needed to escape high-probability loops. |
| 7 | 48 | 5 | 16 | 0.7 | Different errors (fabricated PEP8 claims, out-of-range indices), not a clean improvement over run 4/5. |

## Key finding

Across seven training runs, a clear and consistent pattern emerged: **the model learned 
voice and style far faster and more reliably than it learned correct reasoning about 
novel code.** By 48 examples it reliably sounds like a direct, technical code reviewer — 
but it still fabricates specifics (claiming infinite loops that don't exist, referencing 
list indices that don't exist, inventing PEP8 rules).

This isn't a failure of the process — it's a real and explainable limitation at this 
scale. LoRA fine-tuning on the attention layers (`q_proj`/`v_proj`) directly shapes how 
the model attends to and emphasizes information — which governs tone and style strongly — 
but deep factual/logical correctness is distributed across far more of the network than 
a small adapter can meaningfully override, especially on a 1.1B parameter model with 
under 50 training examples. Lowering loss further (run 3, run 6) did not reliably 
translate to better output — in both cases, lower loss coincided with *worse* coherence, 
a textbook overfitting signature.

## What I learned

This project made the gap between "the code runs and the loss number goes down" and "the 
model actually works correctly" completely concrete. Watching the same dataset produce 
better output at 5 epochs than at 10, and watching more data plus the same epoch count 
make things worse before they got better, was the clearest demonstration yet that 
optimizing a single metric (loss) without checking actual output quality is a trap. I 
also got a real, hands-on understanding of what LoRA is actually doing — not just 
conceptually, but by watching rank, temperature, and dataset size each move the result 
in directions that weren't always intuitive ahead of time. The honest result is a model 
that has learned a voice convincingly but not yet learned to reason reliably about new 
code — which is itself a useful, real finding about what it takes to get small-model 
fine-tuning right.

## Next steps

- Expand dataset further (75-100+ examples) — the clearest unexplored lever
- Try full fine-tuning (not LoRA) on a tiny model to compare ceiling
- Add an evaluation set held out from training to measure improvement objectively 
  rather than by spot-checking one example
- Experiment with `k_proj`/`o_proj` as additional LoRA targets alongside `q_proj`/`v_proj`

## Built with

- Python
- transformers
- peft (LoRA)
- datasets
- torch (MPS backend)
- TinyLlama-1.1B-Chat-v1.0