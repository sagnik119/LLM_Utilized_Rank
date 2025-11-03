# utilized-rank-llm

A modular PyTorch/Hugging Face repository to reproduce and extend **Utilized Rank** analyses and compression for LLMs.

## Key Features

- **Layer-wise utilized-rank estimation** from data-driven input/output subspaces
- **Mixed-rank (non-uniform) low-rank decompositions** preserving ≥99% spectral energy
- **Rank search via binary-search** on subspace energies with validation-drop constraint (ε)
- **Weight transformation** `W' = P_T · W · P_S` and rank-preserving factorization
- **Evaluation** on perplexity and zero-/few-shot tasks
- **Optional fine-tuning** (full or parameter-efficient)

> Supports any `transformers` model with `nn.Linear` layers (e.g., LLaMA, Mistral, GPT-2/NeoX, OPT).

## Key Concepts

- Compute input subspace `S` from `X^T X`
- Infer output subspace `T` without storing `Y` using `T^T T = W · (X^T X) · W^T`
- Build orthogonal projectors `P_S`, `P_T` from top singular vectors that capture ≥ energy thresholds `(e_S, e_T)`
- Form transformed weights `W' = P_T · W · P_S`, revealing a low utilized rank `r = rank(W') ≤ min(k_S, k_T)`
- If `r * (m + d) < m * d`, replace layer with two factors `(m×r)·(r×d)`; otherwise keep original
- Select `(k_S, k_T)` by binary search on energy to respect a validation drop `ε` (ppl or exact-match delta)

## Installation

```bash
pip install -e .
# or
pip install -r requirements.txt
```

## Quickstart

```bash
# 1) Collect activation statistics (X^T X per layer) on a text corpus
python scripts/collect_activations.py --model gpt2 --dataset wikitext --split train --samples 50000 \
  --seq-len 512 --save stats/gpt2_small_xtx.pt

# 2) Search utilized ranks with epsilon=0.1% per transformation
python scripts/search_ranks.py --model gpt2 --stats stats/gpt2_small_xtx.pt --epsilon 0.1 \
  --val-dataset wikitext --val-samples 10000 --out ranks/gpt2_small.json

# 3) Apply compression and save a rank-fixed checkpoint
python scripts/apply_compression.py --model gpt2 --ranks ranks/gpt2_small.json \
  --stats stats/gpt2_small_xtx.pt --out ckpts/gpt2_small_urank

# 4) (Optional) Fine-tune to recover any residual loss
python scripts/finetune.py --model ckpts/gpt2_small_urank --train-dataset wikitext \
  --epochs 2 --lr 5e-5 --out ckpts/gpt2_finetuned

# 5) Evaluate perplexity
python scripts/eval_perplexity.py --model ckpts/gpt2_small_urank --dataset wikitext --split test
```

## Repository Structure

```
utilized-rank-llm/
├── README.md
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── configs/              # Configuration files for different models and search parameters
├── scripts/              # Executable scripts for the full pipeline
├── src/urank/            # Core library modules
│   ├── instrumentation.py    # Layer iteration and replacement utilities
│   ├── activation_cache.py   # X^T X accumulation hooks
│   ├── subspace.py          # SVD-based subspace extraction
│   ├── transform.py         # Weight transformation and factorization
│   ├── rank_search.py       # Binary search for optimal ranks
│   ├── compression.py       # Apply compression to models
│   ├── eval/                # Evaluation utilities
│   └── utils.py             # Helper functions
├── tests/                # Unit tests
└── examples/             # Jupyter notebooks and quickstart guides
```

## Configuration

See `configs/*.yaml` to describe datasets, tokenization, ε, per-layer groups (e.g., attention Q/K/V/O; MLP up/gate/down), and which layers are eligible for replacement.

## License

See LICENSE file for details.