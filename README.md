# Mirror-Peptidizer

**Mirror-Peptidizer** is a modular pipeline for designing de novo D-peptides, integrating AI-driven tools like Chroma and ProteinMPNN to provide stereoisomer-specific peptide design with enhanced precision. The pipeline is tailored for generating D-peptides that bind effectively to target proteins, offering a streamlined approach that combines binder pose generation, sequence prediction, and optional Bayesian Optimization in a cohesive workflow.

This approach leverages stereoisomer-specific modeling and sequence generation for D-peptides, aiming to improve binding accuracy and efficiency over traditional simulation and docking techniques.

```plaintext
+---------------------------+
|       Target Preparation  |
|---------------------------|
|  Invert target PDB along  |
|  X-axis to create the D-  |
|  form target.             |
+---------------------------+
             |
             v
+---------------------------+
|    Binder Pose Generation |
|---------------------------|
|  Use Chroma to generate   |
|  an L-peptide backbone    |
|  constrained to bind to   |
|  the D-form target.       |
+---------------------------+
             |
             v
+---------------------------+
|    Sequence Prediction    |
|---------------------------|
|  Apply ProteinMPNN to     |
|  predict amino acid       |
|  sequence on the L-peptide|
|  backbone.                |
+---------------------------+
             |
             v
+---------------------------+
|    Reversion for Final    |
|    D-Peptide Binder       |
|---------------------------|
|  Convert D-target and L-  |
|  peptide back to original |
|  forms, resulting in      |
|  L-target with D-peptide  |
|  binder.                  |
+---------------------------+
             |
             v  (optional)
+---------------------------+
|   Bayesian Optimization   |
|   (Tier 2)                |
|---------------------------|
|  Iteratively optimize top |
|  sequences using GP       |
|  surrogate + BO/MCMC      |
|  explorers. Supports      |
|  OneHot, physicochemical, |
|  and Boltz2 embeddings.   |
+---------------------------+
```

## How It Works

### Tier 1: Initial D-Peptide Design

1. **Target Preparation**: The pipeline starts by taking the provided receptor's PDB structure, inverting it along the X-axis to create the D-form of the target protein.
2. **Binder Pose Generation with Chroma**: Using the D-form target, Chroma generates an L-peptide binder backbone. This process is constrained by the target's structure, ensuring an optimized binding interaction.
3. **Sequence Prediction with ProteinMPNN**: The generated L-peptide backbone undergoes sequence prediction via ProteinMPNN to establish a specific amino acid sequence for the binder.
4. **Reversion for Final D-Peptide Binder**: Finally, the D-target and the L-peptide are converted back to their original forms, resulting in an L-target with a D-peptide binder.

### Tier 2: Bayesian Optimization (Optional)

After Tier 1 generates initial candidate sequences, the optional BO module iteratively optimizes them:

1. **Surrogate Model**: A Gaussian Process (Matern/RBF kernel) is trained on (sequence, fitness) pairs as a surrogate for expensive evaluations.
2. **Sequence Embedding**: Sequences are encoded via OneHot (L*20 dims), physicochemical residue properties (L*6 dims), or Boltz2Embedding (384-dim learned embeddings from a protein structure model).
3. **Exploration**: An explorer (BO_EVO or MCMC) proposes new mutations guided by an acquisition function (UCB, NEI, EI, etc.).
4. **Evaluation**: Proposed sequences are scored with ProteinMPNN and fed back into the GP for the next round.

The BO loop runs for N rounds, each proposing top-k candidates, converging toward higher-affinity sequences.

## Features

- **Stereoisomer Conversion**: Automatically transforms proteins between L and D forms.
- **AI-Driven Binder Generation**: Uses Chroma's diffusion model to create optimized peptide binder structures.
- **Protein Sequence Prediction**: Predicts sequences tailored for binding using ProteinMPNN, ensuring compatibility with generated binding poses.
- **Bayesian Optimization**: Iteratively optimizes top candidates with GP surrogate, supporting multiple acquisition functions (UCB, NEI, EI, PI, QUCB, etc.) and exploration methods (BO_EVO, MCMC).
- **Multiple Embeddings**: Supports OneHot, physicochemical residue properties, and Boltz2Embedding for sequence representation in BO.
- **Multi-Objective Scoring**: FuzzyScore combines multiple desirability functions (e.g., MPNN score, solubility) into a unified fitness metric.

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA 11+ support and at least 12 GB VRAM (24 GB recommended for larger pose batches).
- **CPU**: 8+ core CPU (Intel Xeon/i7 or AMD Ryzen/EPYC class) for preprocessing and data handling.
- **Memory**: Minimum 8 GB system RAM to accommodate intermediate tensor data and PDB processing.
- **Storage**: ~20 GB free disk space for checkpoints, temporary outputs, and generated poses.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Tier 1: Initial Design](#tier-1-initial-design)
  - [Tier 2: Bayesian Optimization](#tier-2-bayesian-optimization)
- [Project Structure](#project-structure)
- [Configuration](#configuration)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/dahuilangda/Mirror-Peptidizer.git
cd Mirror-Peptidizer
```

### 2. Install Required Packages

Set up the Python environment with `mamba` and `pip`:

```bash
mamba create -n dpep python=3.10
mamba install -c conda-forge biopython pdbfixer
mamba install -c conda-forge matplotlib seaborn
pip install generate-chroma -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install python-dotenv requests -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. Set Up Environment Variables

Copy the example `.env` file and edit it with your API keys and checkpoint path:

```bash
cp env_example .env
```

Update `.env` with:

```plaintext
CHROMA_KEY = 'your_chroma_api_key_here'
CHROMA_WEIGHTS_DIR = '/path/to/chroma_weights'
ProteinMPNN_CHECKPOINT = '/path/to/ProteinMPNN/vanilla_model_weights/v_48_020.pt'
BOLTZ2EMBEDDING_URL = 'http://your-boltz2-server:8000'
BOLTZ2EMBEDDING_TOKEN = 'your-api-token'
```

## Usage

### Tier 1: Initial Design

Run the basic pipeline (generates poses + sequences):

```bash
python run_design.py \
    --receptor data/4LWV.pdb \
    --output output_dir \
    --len_binder 11 \
    --temperature 0.1 \
    --num_poses 3 \
    --num_seqs_per_pose 10 \
    --gpu 0
```

To reject buried/non-surface Chroma poses before ProteinMPNN, enable the
geometry filter:

```bash
python run_design.py \
    --receptor data/4LWV.pdb \
    --output output_dir \
    --len_binder 11 \
    --num_poses 3 \
    --num_seqs_per_pose 10 \
    --filter_surface_poses \
    --surface_max_attempts 20
```

When this option is enabled, each requested pose must pass the surface-like
geometry criteria before sequence design. If no valid pose is found within
`--surface_max_attempts`, the run fails explicitly instead of continuing with a
bad pose. The output CSV includes the filter decision and geometry metrics.

### Tier 2: Bayesian Optimization

Add `--bo_rounds` to automatically run BO after Tier 1:

**With OneHot encoding:**

```bash
python run_design.py \
    --receptor data/4LWV.pdb \
    --output output_dir \
    --bo_rounds 8 \
    --bo_trials 10 \
    --bo_method BO \
    --bo_acquisition UCB
```

**With Boltz2Embedding:**

```bash
python run_design.py \
    --receptor data/4LWV.pdb \
    --output output_dir \
    --bo_rounds 8 \
    --bo_trials 10 \
    --bo_embedding boltz2embedding
```

Boltz2Embedding credentials are read from `.env` automatically, or can be passed via `--boltz2_url` and `--boltz2_token`.

> **Boltz2Embedding** is a protein sequence embedding service that produces 384-dim learned representations. It can be self-hosted — see [github.com/dahuilangda/Boltz2Embedding](https://github.com/dahuilangda/Boltz2Embedding) for setup instructions.

**With physicochemical encoding:**

```bash
python run_design.py \
    --receptor data/4LWV.pdb \
    --output output_dir \
    --bo_rounds 8 \
    --bo_trials 10 \
    --bo_embedding physicochemical
```

For BO strategy studies, compare embeddings as a grid dimension:

```bash
python benchmark/run_bo_study.py \
    --gpu 1 \
    --benchmark_dir benchmark \
    --tier1_dir benchmark \
    --output_dir benchmark/bo_study \
    --peptide_modes D,L \
    --seed_source proteinmpnn \
    --bo_embeddings onehot,physicochemical,boltz2embedding
```

`benchmark/run_ablation.py` and `benchmark/run_bo_study.py` are downstream benchmark-reuse runners:
they require precomputed `benchmark/<target>/D_peptide/all_results.csv` and
`benchmark/<target>/D_peptide/len_*/Poses/Binder_L_pose_*.pdb` files and will not
run Chroma. Use `benchmark/summarize_ld_benchmark.py` for a D/L CSV summary:

```bash
python benchmark/summarize_ld_benchmark.py --benchmark_dir benchmark
```

### Command-line Arguments

#### Tier 1 Arguments

| Argument | Default | Description |
|---|---|---|
| `--receptor` | (required) | Path to the receptor PDB file |
| `--output` | `output` | Output directory |
| `--len_binder` | `11` | Length of the peptide binder |
| `--temperature` | `0.1` | ProteinMPNN sampling temperature |
| `--num_poses` | `1` | Number of Chroma binding poses |
| `--num_seqs_per_pose` | `8` | Sequences per pose from ProteinMPNN |
| `--gpu` | `0` | GPU device number |

#### Tier 2 (Bayesian Optimization) Arguments

| Argument | Default | Description |
|---|---|---|
| `--bo_rounds` | `0` | BO rounds (0 = disabled) |
| `--bo_trials` | `10` | Sequences proposed per round |
| `--bo_method` | `BO` | Exploration method: `BO` or `MCMC` |
| `--bo_embedding` | `onehot` | Sequence embedding: `onehot`, `physicochemical`, or `boltz2embedding` |
| `--bo_embeddings` | unset | BO study only: comma-separated embedding grid, e.g. `onehot,physicochemical,boltz2embedding` |
| `--bo_kernel` | `Matern` | GP kernel: `Matern` or `RBF` |
| `--bo_acquisition` | `UCB` | Acquisition function: `UCB`, `LCB`, `EI`, `PI`, `TS`, `Greedy`, `NEI`, `QUCB` |
| `--bo_uf_param` | `0.2` | Acquisition function hyperparameter |
| `--bo_model_queries` | `3000` | Model queries per round |
| `--boltz2_url` | (from .env) | Boltz2Embedding API server URL |
| `--boltz2_token` | (from .env) | Boltz2Embedding API token |

## Project Structure

```
Mirror-Peptidizer/
│
├── run_design.py            # Main entry point (Tier 1 + optional Tier 2)
├── __init__.py
├── env_example              # Template for environment variables
│
├── bo/                      # Bayesian Optimization module (Tier 2)
│   ├── __init__.py          # Package exports
│   ├── acquisition.py       # Acquisition functions (UCB, NEI, EI, QUCB, etc.)
│   ├── encoders.py          # OneHotEncoder, PhysicochemicalEncoder, Boltz2Encoder
│   ├── models.py            # GPRegressor surrogate model
│   ├── explorers.py         # BO_EVO and MCMC explorers
│   ├── landscape.py         # CSV-based fitness landscape interface
│   ├── scoring.py           # FuzzyScore, SWI solubility scoring
│   └── run_bo.py            # Standalone BO CLI entry point
│
├── utils/
│   ├── __init__.py
│   ├── pdb_processing.py    # PDB I/O, L/D conversion, structure repair
│   ├── chroma_sample.py     # Binder backbone generation via Chroma
│   └── protein_mpnn.py      # Sequence prediction via ProteinMPNN
│
├── ProteinMPNN/             # Vendored ProteinMPNN model
├── examples/                # Example notebooks and scripts
└── data/                    # Example receptor PDB files
```

## Configuration

Ensure these environment variables are set in your `.env` file:

| Variable | Required | Description |
|---|---|---|
| `CHROMA_KEY` | Yes | Chroma API key for binder generation |
| `CHROMA_WEIGHTS_DIR` | Yes | Path to local Chroma model weights directory |
| `ProteinMPNN_CHECKPOINT` | Optional | ProteinMPNN checkpoint path. Defaults to `ProteinMPNN/vanilla_model_weights/v_48_020.pt`; use a soluble checkpoint only when you explicitly want the soluble-model prior. |
| `BOLTZ2EMBEDDING_URL` | For BO + Boltz2Embedding | Boltz2Embedding API server URL |
| `BOLTZ2EMBEDDING_TOKEN` | For BO + Boltz2Embedding | Boltz2Embedding API authentication token |

## References

- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) - Sequence design via protein structure
- [Chroma](https://github.com/generatebio/chroma) - Generative diffusion model for protein design
- [Boltz2Embedding](https://github.com/dahuilangda/Boltz2Embedding) - Protein sequence embedding service

## License

Licensed under the [Apache License 2.0](LICENSE).
