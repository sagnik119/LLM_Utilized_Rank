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

## Complete Pipeline

This repository provides a comprehensive pipeline for utilized rank analysis and compression. Follow these steps in order:

### Step 1: Collect Activation Statistics

Collect X^T X statistics from forward passes on a calibration dataset:

```bash
python scripts/collect_activations.py \
  --model gpt2 \
  --dataset wikitext \
  --split train \
  --samples 50000 \
  --seq-len 512 \
  --save stats/gpt2_xtx.pt
```

### Step 2: Search Optimal Ranks

Binary search for optimal ranks with validation constraint (ε=0.1%):

```bash
python scripts/search_ranks.py \
  --model gpt2 \
  --stats stats/gpt2_xtx.pt \
  --epsilon 0.1 \
  --val-dataset wikitext \
  --val-samples 2000 \
  --out ranks/gpt2_base.json
```

### Step 3: Apply Rank Allocation Policy (Optional)

Adjust ranks based on layer type (QKV vs MLP):

```bash
# List available policies
python scripts/allocate_ranks.py --list-policies

# Apply QKV-heavier policy (prioritizes attention)
python scripts/allocate_ranks.py \
  --in-ranks ranks/gpt2_base.json \
  --policy qkv_heavier \
  --out ranks/gpt2_qkv.json \
  --verbose

# Or use uniform/mlp_heavier policies
# --policy uniform
# --policy mlp_heavier
```

### Step 4: Apply Compression

Transform weights and factorize where beneficial:

```bash
python scripts/apply_compression.py \
  --model gpt2 \
  --ranks ranks/gpt2_qkv.json \
  --stats stats/gpt2_xtx.pt \
  --out ckpts/gpt2_compressed
```

### Step 5: Fine-Tune (Optional but Recommended)

Recover performance using one of three strategies:

**Option A: LoRA Fine-Tuning (Recommended)**
```bash
python scripts/finetune_llamafactory.py \
  --preset lora \
  --model ckpts/gpt2_compressed \
  --data wikitext2 \
  --out outputs/gpt2_lora \
  --epochs 2 \
  --lr 2e-4
```

**Option B: Full Fine-Tuning**
```bash
python scripts/finetune_llamafactory.py \
  --preset full \
  --model ckpts/gpt2_compressed \
  --data wikitext2 \
  --out outputs/gpt2_full \
  --epochs 2 \
  --lr 1e-5
```

**Option C: Freeze Fine-Tuning (Norms + Head only)**
```bash
python scripts/finetune_llamafactory.py \
  --preset freeze \
  --model ckpts/gpt2_compressed \
  --data wikitext2 \
  --out outputs/gpt2_freeze \
  --epochs 2 \
  --lr 5e-5
```

**Option D: Simple Fine-Tuning (HF Trainer)**
```bash
python scripts/finetune.py \
  --model ckpts/gpt2_compressed \
  --train-dataset wikitext \
  --epochs 2 \
  --lr 5e-5 \
  --out ckpts/gpt2_finetuned
```

### Step 6: Evaluate

**Option A: Comprehensive Evaluation (lm-eval-harness)**
```bash
python scripts/eval_lm_eval.py \
  --model outputs/gpt2_lora \
  --tasks wikitext arc_easy hellaswag winogrande piqa \
  --batch 8 \
  --out eval_results.json
```

**Option B: Perplexity Only**
```bash
python scripts/eval_perplexity.py \
  --model outputs/gpt2_lora \
  --dataset wikitext \
  --split test \
  --samples 5000
```

## Quick Examples

### Basic Pipeline (Uniform ranks)
```bash
# 1. Collect stats
python scripts/collect_activations.py --model gpt2 --dataset wikitext --split train \
  --samples 50000 --seq-len 512 --save stats/gpt2_xtx.pt

# 2. Search ranks
python scripts/search_ranks.py --model gpt2 --stats stats/gpt2_xtx.pt \
  --epsilon 0.1 --out ranks/gpt2.json

# 3. Compress
python scripts/apply_compression.py --model gpt2 --ranks ranks/gpt2.json \
  --stats stats/gpt2_xtx.pt --out ckpts/gpt2_compressed

# 4. LoRA fine-tune
python scripts/finetune_llamafactory.py --preset lora --model ckpts/gpt2_compressed \
  --data wikitext2 --out outputs/gpt2_lora

# 5. Evaluate
python scripts/eval_lm_eval.py --model outputs/gpt2_lora \
  --tasks wikitext arc_easy hellaswag --batch 8 --out eval.json
```

### Advanced Pipeline (With rank policies)
```bash
# 1-2. Same as above (collect & search)
python scripts/collect_activations.py --model gpt2 --dataset wikitext --split train \
  --samples 50000 --seq-len 512 --save stats/gpt2_xtx.pt
python scripts/search_ranks.py --model gpt2 --stats stats/gpt2_xtx.pt \
  --epsilon 0.1 --out ranks/gpt2_base.json

# 3. Apply QKV-heavier policy
python scripts/allocate_ranks.py --in-ranks ranks/gpt2_base.json \
  --policy qkv_heavier --out ranks/gpt2_qkv.json --verbose

# 4. Compress with policy
python scripts/apply_compression.py --model gpt2 --ranks ranks/gpt2_qkv.json \
  --stats stats/gpt2_xtx.pt --out ckpts/gpt2_qkv_compressed

# 5. Full fine-tune
python scripts/finetune_llamafactory.py --preset full \
  --model ckpts/gpt2_qkv_compressed --data wikitext2 --out outputs/gpt2_full

# 6. Comprehensive eval
python scripts/eval_lm_eval.py --model outputs/gpt2_full \
  --tasks wikitext arc_easy hellaswag winogrande piqa boolq --batch 8 --out eval_full.json
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