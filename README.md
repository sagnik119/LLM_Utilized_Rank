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

## Complete Pipeline for GPT2 Small (Small model)

This repository provides a comprehensive pipeline for utilized rank analysis and compression. Follow these steps in order:

### Step 1: Collect Activation Statistics

Collect X^T X statistics from forward passes on a calibration dataset:

*OPTION 1: Collect input stats (legacy)
```bash
python scripts/collect_activations.py \
  --model gpt2 \
  --dataset wikitext \
  --split train \
  --samples 2000 \
  --seq-len 512 \
  --save stats/gpt2_xtx.pt
```

*OPTION 2: Collect output stats (current)
python scripts/collect_output_activations.py \
  --model gpt2 \
  --dataset wikitext \
  --split train \
  --samples 1000000 \
  --max-length 512 \
  --out stats/gpt2_yty.pt

### Step 2: Search Optimal Ranks

Binary search for optimal ranks with validation constraint (ε=0.1%):

*OPTION 1: search rank based on drop of PPL (legacy)
```bash
python scripts/search_ranks.py \
  --model gpt2 \
  --stats stats/gpt2_xtx.pt \
  --epsilon 2.0 \
  --val-dataset wikitext \
  --val-samples 2000 \
  --out ranks/gpt2_base.json
```

*OPTION 2: compute energy based ranks
python scripts/search_ranks_energy.py \
  --stats stats/gpt2_yty.pt \
  --energy 0.99 \
  --out ranks/gpt2_energy.json

*OPTION 3: naive SVD based rank collection
python scripts/search_ranks_weight_svd.py \
  --model gpt2 \
  --energy 0.9495 \
  --out ranks/gpt2_weight_svd.json


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
  --ranks ranks/gpt2_energy.json \
  --stats stats/gpt2_yty.pt \
  --out ckpts/gpt2_compressed
```

<!-- OR

```bash
python scripts/apply_compression.py \
  --model gpt2 \
  --ranks ranks/gpt2_base.json \
  --stats stats/gpt2_xtx.pt \
  --out ckpts/gpt2_compressed
``` -->

*OPTION 2: Naive SVD rank based compression

python scripts/apply_compression.py \
  --model gpt2 \
  --ranks ranks/gpt2_weight_svd.json \
  --mode weight_svd \
  --out ckpts/gpt2_weight_svd

*OPTION 3: hybrid, identify layers to be compressed as per utilized rank, but compress them using naive SVD

python scripts/apply_compression.py \
  --model gpt2 \
  --ranks ranks/gpt2_energy.json \
  --stats stats/gpt2_yty.pt \
  --mode hybrid \
  --out ckpts/gpt2_hybrid


### Step 4: Eval compressed model (pre-fine tuning)

python scripts/eval_lm_eval.py \
  --model ckpts/gpt2_compressed \
  --tasks wikitext arc_easy hellaswag winogrande piqa \
  --batch 8 \
  --out results.json

### Step 4: Merge new datasets information
python scripts/setup_llamafactory_datasets.py   --data-dir /scratch/users/sagnikb/LLaMA-Factory/data   --merge

### Step 5: Fine-Tune (Optional but Recommended)

Recover performance using one of three strategies:

**Option A: LoRA Fine-Tuning (Recommended)**

# use this option for finetuning original models
```bash
python scripts/finetune_llamafactory.py \
  --preset lora \
  --model ckpts/gpt2_compressed \
  --data-dir /scratch/users/sagnikb/LLaMA-Factory/data \
  --data alpaca_en \
  --out outputs/gpt2_lora \
  --epochs 2 \
  --lr 1e-4
```

# use this option for finetuning compressed models
python scripts/finetune_compressed.py \
  --preset lora \
  --model ckpts/gpt2_compressed \
  --data tatsu-lab/alpaca \
  --out outputs/gpt2_lora \
  --epochs 2 \
  --lr 2e-4 \
  --batch-size 4 \
  --lora-r 8 \
  --lora-alpha 16
### TODO: wikitext2 is basically continued pre training, may hurt performance on tasks, use SFT dataset for fine tuning

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

python scripts/finetune_compressed.py --preset full \
  --model ckpts/gpt2_compressed \
  --data redpajama \
  --out outputs/gpt2_redpajama_fullft_50m_toks \
  --max-train-tokens 50000000 \
  --lr 2e-5 \
  --batch-size 4


### Step 5: Export compressed model for compatibility with lm eval harness

**Option A: Export base compressed model only
python scripts/export_compressed_model.py \
  --model ckpts/gpt2_compressed \
  --out exports/gpt2_compressed_eval

**Option B: Export compressed model + LoRa adapters
python scripts/export_compressed_model.py \
  --model ckpts/gpt2_compressed \
  --lora outputs/gpt2_lora \
  --out exports/gpt2_lora_merged


### Step 6: Evaluate

**Option A: Comprehensive Evaluation (lm-eval-harness)**
```bash
python scripts/eval_lm_eval.py \
  --model exports/gpt2_lora_merged \
  --tasks wikitext arc_easy hellaswag winogrande piqa \
  --batch 8 \
  --out results.json
```

**Option B: Perplexity Only**
```bash
python scripts/eval_perplexity.py \
  --model exports/gpt2_lora_merged \
  --dataset wikitext \
  --split test \
  --samples 5000
```
1. Evaluate baseline GPT-2
python scripts/eval_lm_eval.py --model gpt2 --tasks wikitext arc_easy \
  --out results_baseline.json

2. Evaluate compressed model (before fine-tuning)
python scripts/eval_lm_eval.py --model ckpts/gpt2_compressed --tasks wikitext arc_easy \
  --out results_compressed.json

3. Evaluate LoRA fine-tuned compressed model
python scripts/eval_lm_eval.py --model exports/gpt2_lora_merged --tasks wikitext arc_easy \
  --out results_finetuned.json

### Complete pipeline for Llama-2-7B (medium sized model)

python scripts/collect_output_activations.py \
  --model meta-llama/Llama-2-7b-hf \
  --dataset wikitext \
  --split train \
  --samples 1000000 \
  --max-length 512 \
  --batch-size 64 \
  --use-bf16 \
  --out stats/llama2_7b_yty.pt

python scripts/search_ranks_energy.py \
  --model meta-llama/Llama-2-7b-hf \
  --stats stats/llama2_7b_yty.pt \
  --energy 0.99 \
  --out ranks/llama2_7b_energy.json

# Apply compression with Utilized rank (full method)
python scripts/apply_compression.py \
  --model meta-llama/Llama-2-7b-hf \
  --ranks ranks/llama2_7b_energy.json \
  --stats stats/llama2_7b_yty.pt \
  --mode utilized \
  --out ckpts/llama2_7b_compressed

# OR hybrid baseline
python scripts/apply_compression.py \
  --model meta-llama/Llama-2-7b-hf \
  --ranks ranks/llama2_7b_energy.json \
  --stats stats/llama2_7b_yty.pt \
  --mode hybrid \
  --out ckpts/llama2_7b_hybrid

# OR weight SVD baseline
python scripts/search_ranks_weight_svd.py \
  --model meta-llama/Llama-2-7b-hf \
  --energy 0.9995 \
  --out ranks/llama2_7b_weight_svd.json

python scripts/apply_compression.py \
  --model meta-llama/Llama-2-7b-hf \
  --ranks ranks/llama2_7b_weight_svd.json \
  --mode weight_svd \
  --out ckpts/llama2_7b_weight_svd

# Eval compressed model (no fine tuning) - BOTTLENECK STEP
python scripts/eval_lm_eval.py \
  --model ckpts/llama2_7b_compressed \
  --tasks wikitext arc_easy hellaswag winogrande piqa \
  --batch 4 --trust-remote-code \
  --out results_llama2_7b_compressed_utilized.json

# Fine tune
python scripts/finetune_compressed.py --preset full \
  --model ckpts/llama2_7b_compressed \
  --data redpajama \
  --out outputs/llama2_7b_compressed_fullft \
  --max-train-tokens 10000000 \
  --lr 2e-5 \
  --batch-size 16 \
  --seq-length 512

# Evaluate
python scripts/eval_lm_eval.py \
  --model outputs/llama2_7b_compressed_fullft \
  --tasks wikitext arc_easy hellaswag winogrande piqa \
  --batch 4 \
  --out results_llama2_7b.json


### Complete pipeline for Llama-3-8B (medium sized model)

# 1. Collect activations (now architecture-aware!)
python scripts/collect_output_activations.py \
  --model meta-llama/Meta-Llama-3-8B \
  --dataset wikitext \
  --split train \
  --samples 1000000 \
  --max-length 512 \
  --out stats/llama3_8b_yty.pt

# 2. Search ranks (unchanged, works on any Y^T Y file)
python scripts/search_ranks_energy.py \
  --stats stats/llama3_8b_yty.pt \
  --energy 0.99 \
  --out ranks/llama3_8b_energy.json

# 3. Apply compression (unchanged, name-based lookup)
python scripts/apply_compression.py \
  --model meta-llama/Meta-Llama-3-8B \
  --ranks ranks/llama3_8b_energy.json \
  --stats stats/llama3_8b_yty.pt \
  --mode utilized \
  --out ckpts/llama3_8b_compressed

# 4. Fine-tune (unchanged, HF-based)
python scripts/finetune_compressed.py --preset full \
  --model ckpts/llama3_8b_compressed \
  --data redpajama \
  --max-train-tokens 150000000 \
  --out outputs/llama3_8b_fullft

# 5. Evaluate (unchanged, lm-eval-harness)
python scripts/eval_lm_eval.py \
  --model outputs/llama3_8b_fullft \
  --tasks wikitext arc_easy hellaswag \
  --out results_llama3.json


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

### Things needed before llama-factory installation
export CFLAGS="-std=c99"
export CFLAGS="-std=c99 -D_POSIX_C_SOURCE=200809L"
pip uninstall -y av
pip install setuptools_scm scikit-build-core cmake ninja
conda install -c conda-forge sentencepiece tiktoken
conda install -c conda-forge soxr librosa
module load gcc/14.2.0
conda install -c conda-forge scipy numpy libgcc-ng libstdcxx-ng
conda install -c conda-forge soxr-python
conda install -c conda-forge libiconv