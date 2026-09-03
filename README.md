# Mirror-Peptidizer (v2)

**Mirror-Peptidizer** is a modular pipeline for designing de novo D-peptides that bind a given L-protein target. It combines AI-driven pose and sequence generation (Chroma, ProteinMPNN) with a structure-model-scoring Bayesian Optimization loop — every BO candidate is evaluated by **protenix2dock**, the protein–ligand structure workflow on the Protenix engine, whose interface confidence (ipSAE, peptide pLDDT, pose RMSD) is the optimization objective.

The pipeline operates in mirror space: the D-target problem is mapped onto standard L-amino-acid tooling by reflecting the target, designing an L-binder against the mirrored target, and reflecting the result back to the deliverable L-target + D-peptide pair. This stereoisomer-specific modeling aims to improve binding accuracy and efficiency over traditional simulation and docking techniques.

> **Branches.** `v2` (this branch) scores BO candidates with protenix2dock.
> The `main` branch carries the pipeline exactly as published (BO fitness =
> ProteinMPNN NLL) — see the
> [publication](#publication) below.

## Publication

This work is published in ***Research*** (Science Partner Journal):

> Bohan Ma, Zhe Wang, Yanlin Jian, Yonghong Mi, Si Chen, Honggang Hu, Xiang Li.
> **Mirror-Peptidizer: In Silico Mirror-Image Screening Enables De Novo Design of D-Peptide Binders without D-Protein Synthesis.**
> *Research* **2026**, 9, 1420. [DOI: 10.34133/research.1420](https://doi.org/10.34133/research.1420)

The manuscript reports the full computational validation (sequence-recovery controls on 3LNJ/8F10/3HTN,
mirror-geometry QC, D/L score-distribution separation, filtering + BO ablation, and new-target
generalization on TNFα/CXCR2/CXCR4) behind the headline numbers summarized in
[Validation & benchmark results](#validation--benchmark-results) below. The v2 BO objective
(protenix2dock scoring) is an engineering upgrade of the published pipeline and is not part of the
manuscript.

## Pipeline

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
|    Sequence Design        |
|---------------------------|
|  Apply ProteinMPNN to     |
|  design amino acid        |
|  sequences on the L-      |
|  peptide backbone.        |
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
|---------------------------|
|  Iteratively optimize     |
|  sequences with a GP      |
|  surrogate + BO/MCMC      |
|  explorers. Every         |
|  candidate is scored by   |
|  protenix2dock: ipSAE +   |
|  peptide pLDDT + pose     |
|  RMSD (dock mode) drive   |
|  the fitness.             |
+---------------------------+
```

## How It Works

### Tier 1: Initial D-Peptide Design

1. **Target Preparation**: The pipeline starts by taking the provided receptor's PDB structure, inverting it along the X-axis to create the D-form of the target protein.
2. **Binder Pose Generation with Chroma**: Using the D-form target, Chroma generates an L-peptide binder backbone. This process is constrained by the target's structure, ensuring an optimized binding interaction.
3. **Sequence Design with ProteinMPNN**: The generated L-peptide backbone undergoes sequence design via ProteinMPNN, producing candidate amino acid sequences for the binder.
4. **Reversion for Final D-Peptide Binder**: Finally, the D-target and the L-peptide are converted back to their original forms, resulting in an L-target with a D-peptide binder.

ProteinMPNN's NLL ranks the Tier-1 pool for *backbone compatibility only* — it is a sequence
generator here, not the optimization objective.

### Tier 2: Bayesian Optimization with Protenix2Dock Scoring

After Tier 1 generates initial candidate sequences, the optional BO module iteratively optimizes
them. Every proposed sequence is placed on the designed backbone and evaluated by **protenix2dock**
(peptide mode); the engine's interface confidence is the BO fitness:

```text
fitness = w_ipsae * ipsae_dom + w_plddt * ligand_plddt
        - w_rmsd * peptide_rmsd - w_synth * synthesis_penalty        (dock mode)

fitness = w_ipsae * ipsae_dom + w_plddt * ligand_plddt
        - w_synth * synthesis_penalty                                 (score mode)
```

| Term | Meaning |
|---|---|
| `ipsae_dom` | dominant interface ipSAE from protenix2dock's ipSAE post-processing (ptm-transformed interface PAE confidence, [0, 1], higher = more confident interface) |
| `ligand_plddt` | mean pLDDT of the designed peptide chain ([0, 1], higher = better) |
| `peptide_rmsd` | pose RMSD (Å) between the placed (Chroma) peptide and Protenix's best-sample peptide after receptor superposition; a sequence whose pose Protenix cannot reproduce drifts to a high RMSD (dock mode only) |
| `synthesis_penalty` | SPPS tractability penalty (`bo/scoring.py`: coupling/aggregation/side-reaction/purification risk), same as the published pipeline |

Two scoring flavours (`--bo_protenix_mode`):

- **dock** *(default)* — receptor-fixed peptide re-docking: Protenix's diffusion refines the placed
  peptide against the pinned receptor every step, so each evaluation both scores the sequence and
  stress-tests the Chroma pose; the pose RMSD enters the fitness. Sample and step counts default to
  the engine's peptide-mode config (8 samples / 12 steps) — tune with
  `--bo_protenix_diffusion_samples` / `--bo_protenix_sampling_steps`.
- **score** — diffusion bypassed (`--score_only`, single sample): the confidence heads evaluate the
  input coordinates directly. Fast triage pass (roughly 1–3 min per candidate on an RTX 4090 with a
  warm module cache) without the RMSD term.

By default the **final-form complex (L-target + D-peptide)** is scored (`--bo_protenix_form final`):
the large receptor chain stays in-distribution for Protenix while only the designed peptide is
mirrored — the same setting validated on the 3LNJ (L-MDM2 + D-peptide) native complex
(score mode: ipSAE 0.75 / peptide pLDDT 0.95; dock mode reproduces the crystal pose at 0.19 Å RMSD).
`--bo_protenix_form mirror` scores the Chroma-side complex (D-target + L-peptide) instead.

The remaining BO machinery is unchanged from the published pipeline:

1. **Surrogate Model**: a Gaussian Process (Matern/RBF kernel) trained on (sequence, fitness) pairs.
2. **Sequence Embedding**: OneHot (L×20), physicochemical residue properties (L×6), or Boltz2Embedding (384-dim).
3. **Exploration**: BO_EVO or MCMC propose mutations guided by an acquisition function (UCB, NEI, EI, ...).
4. **Evaluation**: proposed sequences are scored by protenix2dock and fed back into the GP for the next round.

The initial BO observations are the top `--bo_protenix_init_top` (default 10) Tier-1 sequences,
re-evaluated by protenix2dock so the GP trains on the same fitness it will optimize. The BO loop
runs for N rounds, each proposing top-k candidates, converging toward sequences Protenix itself is
confident bind the target. (The standalone `bo/run_bo.py` CLI is unchanged from the published
version and keeps the MPNN objective; `run_design.py` is the v2 entry point.)

### Module reference (step → code)

| Step | What happens | Code |
|---|---|---|
| Input preparation | read receptor chains | `utils/pdb_processing.get_pdb_chains` |
| Coordinate mirroring | reflect receptor along X (L→D); same convention as Garton et al., *PNAS* 2018 | `utils/pdb_processing.ld_convert` (`run_design.py:408`) |
| Binder backbone | Chroma samples an L-peptide pose on the D-target | `utils/chroma_sample.binder_sample` |
| (optional) Geometry filter | reject buried / non-surface poses before sequence design | `utils/pose_filtering.passes_surface_pose_filter` |
| Sequence design | ProteinMPNN redesigns the binder chain (fixed target), T = 0.1 | `utils/protein_mpnn.protein_mpnn` |
| Reversion | reflect the L-binder back to D; build D-peptide PDB | `utils/pdb_processing.ld_convert`, `seq_to_pdb` |
| BO scoring | protenix2dock peptide mode (dock/score) on the reverted complex: ipSAE, peptide pLDDT, pose RMSD | `utils/protenix2dock_client.Protenix2DockScorer` |
| BO loop | GP surrogate + acquisition optimize the protenix-scored fitness | `run_design.run_bo_optimization`, `bo/` |

## Features

- **Stereoisomer Conversion**: automatically transforms proteins between L and D forms.
- **AI-Driven Binder Generation**: Chroma's diffusion model creates optimized peptide binder structures.
- **Sequence Design**: ProteinMPNN designs binding-compatible sequences on the generated poses.
- **Protenix-Scored Bayesian Optimization**: every BO candidate is evaluated by protenix2dock —
  interface ipSAE and peptide pLDDT (plus a pose-RMSD penalty in dock mode) form the fitness —
  with GP surrogate, multiple acquisition functions (UCB, NEI, EI, PI, QUCB, ...) and explorers
  (BO_EVO, MCMC).
- **Multiple Embeddings**: OneHot, physicochemical residue properties, or Boltz2Embedding for sequence representation in BO.
- **Synthesis-Aware Penalty**: the SPPS tractability model of the published pipeline penalizes hard-to-synthesize candidates inside the fitness.

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA 11+ support and at least 12 GB VRAM (24 GB recommended) for
  Chroma + ProteinMPNN; the protenix2dock runtime additionally wants a 24 GB card (RTX 4090 class)
  — point it at a second GPU with `--bo_protenix_gpu` to run both stages in parallel.
- **CPU**: 8+ core CPU for preprocessing and data handling.
- **Memory**: 8 GB+ system RAM.
- **Storage**: ~30 GB free disk space for checkpoints, engine caches, temporary outputs and poses.
- **Docker**: the protenix2dock scoring runs in the Protenix runtime container
  (`vbio-protenix-v2-runtime:2.0.0` by default) with a V-Bio checkout providing
  `capabilities/protenix2dock`, the Protenix-v2 model weights and shared caches — every path is a
  `PROTENIX2DOCK_*` entry in `.env` (see [Configuration](#configuration)).

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Tier 1: Initial Design](#tier-1-initial-design)
  - [Tier 2: Bayesian Optimization](#tier-2-bayesian-optimization)
- [Output interpretation](#output-interpretation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Validation & benchmark results](#validation--benchmark-results)
- [Testing & validation guide (installation self-checks, recovery, mirror QC, ablation, all figures)](docs/testing_and_validation.md)
- [Installation guide](docs/installation.md)
- [Computational configuration (ProteinMPNN & BO parameters)](docs/computational_configuration.md)

## Installation

### 1. Clone the Repository

```bash
git clone -b v2 https://github.com/dahuilangda/Mirror-Peptidizer.git
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

Copy the example `.env` file and edit it with your API keys, checkpoint paths and the protenix2dock
runtime locations:

```bash
cp env_example .env
```

See [Configuration](#configuration) for every variable. The protenix2dock entries follow a standard
V-Bio deployment (runtime docker image, model weights, shared caches, optional ColabFold-compatible
MSA server); on the reference machine the receptor MSAs resolve from the shared boltz cache and no
server is needed.

### 4. Verify the installation

A quick self-check that the GPU stack and the ProteinMPNN weights load correctly:

```bash
python -c "
import torch
from utils.protein_mpnn import load_model, resolve_checkpoint_path
print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '| n GPU', torch.cuda.device_count())
print('checkpoint:', resolve_checkpoint_path())
load_model()  # loads vanilla v_48_020
print('ProteinMPNN v_48_020 loaded OK')
"
```

To also verify the protenix2dock scoring runtime (docker image, weights and caches reachable),
score a receptor-peptide complex PDB:

```bash
python -c "
from utils.protenix2dock_client import Protenix2DockScorer, summarise_metrics
scorer = Protenix2DockScorer(gpu=0)
metrics = scorer.score_peptide_complex(
    'complex.pdb', peptide_chain='B', out_dir='/tmp/p2d_check', score_only=True)
print(summarise_metrics(metrics))
"
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

Add `--bo_rounds` to automatically run BO after Tier 1. Scoring defaults to protenix2dock in
**dock mode** (receptor-fixed peptide re-docking with the pose-RMSD penalty):

```bash
python run_design.py \
    --receptor data/4LWV.pdb \
    --output output_dir \
    --len_binder 11 \
    --num_poses 1 \
    --num_seqs_per_pose 8 \
    --bo_rounds 8 \
    --bo_trials 10 \
    --bo_protenix_mode dock \
    --bo_ipsae_weight 0.6 \
    --bo_plddt_weight 0.4 \
    --bo_rmsd_weight 0.05 \
    --bo_protenix_gpu 0
```

For a fast confidence-only pass without diffusion (and without the RMSD term):

```bash
python run_design.py \
    --receptor data/4LWV.pdb \
    --output output_dir \
    --bo_rounds 8 \
    --bo_trials 10 \
    --bo_protenix_mode score
```

**With Boltz2Embedding** (sequence embedding for the GP surrogate):

```bash
python run_design.py \
    --receptor data/4LWV.pdb \
    --output output_dir \
    --bo_rounds 8 \
    --bo_trials 10 \
    --bo_embedding boltz2embedding
```

Boltz2Embedding credentials are read from `.env` automatically, or can be passed via `--boltz2_url` and `--boltz2_token`.
It is a protein sequence embedding service producing 384-dim learned representations; it can be
self-hosted — see [github.com/dahuilangda/Boltz2Embedding](https://github.com/dahuilangda/Boltz2Embedding).

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
| `--bo_ipsae_weight` | `0.6` | Weight of `ipsae_dom` in the fitness |
| `--bo_plddt_weight` | `0.4` | Weight of `ligand_plddt` in the fitness |
| `--bo_rmsd_weight` | `0.05` | Weight of the peptide pose RMSD penalty in Å (dock mode only) |
| `--bo_protenix_mode` | `dock` | protenix2dock flavour: `dock` (re-docking + RMSD) or `score` (confidence-only, no RMSD) |
| `--bo_protenix_form` | `final` | Complex scored: `final` (L-target + D-peptide) or `mirror` (D-target + L-peptide) |
| `--bo_protenix_diffusion_samples` | mode config | Diffusion samples in dock mode |
| `--bo_protenix_sampling_steps` | mode config | Diffusion steps in dock mode |
| `--bo_protenix_msa` | `auto` | MSA server usage: `auto` (shared cache first), `on`, `off` |
| `--bo_protenix_gpu` | `--gpu` | GPU for the protenix2dock runtime container |
| `--bo_protenix_seed` | `42` | Seed for the protenix2dock engine |
| `--bo_protenix_init_top` | `10` | Tier-1 seed sequences re-evaluated by protenix2dock as BO init |
| `--bo_method` | `BO` | Exploration method: `BO` or `MCMC` |
| `--bo_embedding` | `onehot` | Sequence embedding: `onehot`, `physicochemical`, or `boltz2embedding` |
| `--bo_embeddings` | unset | BO study only: comma-separated embedding grid, e.g. `onehot,physicochemical,boltz2embedding` |
| `--bo_kernel` | `Matern` | GP kernel: `Matern` or `RBF` |
| `--bo_acquisition` | `UCB` | Acquisition function: `UCB`, `LCB`, `EI`, `PI`, `TS`, `Greedy`, `NEI`, `QUCB` |
| `--bo_uf_param` | `0.2` | Acquisition function hyperparameter |
| `--bo_model_queries` | `3000` | Model queries per round |
| `--synthesis_penalty_weight` | `0.5` | Weight of the SPPS synthesis-risk penalty in the fitness |
| `--bo_init_source` | `tier1` | Initial observations: `tier1` (top MPNN sequences, protenix-scored), `random`, or `single` |
| `--bo_random_init_seqs` | `10` | Random initial sequences when `--bo_init_source random` |
| `--boltz2_url` | (from .env) | Boltz2Embedding API server URL |
| `--boltz2_token` | (from .env) | Boltz2Embedding API token |

## Output interpretation

A run produces the following layout under `--output`:

```
output_dir/
├── results.csv                 # Tier-1 candidates ranked by ProteinMPNN NLL (lower = better)
├── Poses/
│   ├── receptor_D.pdb          # D-form target (mirrored input)
│   ├── Binder_L_pose_1.pdb     # Chroma L-peptide backbones (Tier-1 input to MPNN)
│   └── Binder_D_pose_1.pdb     # reverted D-peptide backbones (BO scoring template)
├── Binders/                    # one D-peptide PDB per designed sequence
├── Images/Pose_1_amino_acid_probs.png   # MPNN per-position amino-acid probabilities
└── pose_1/BO/                  # (only with --bo_rounds > 0)
    ├── fitness.csv, bo_results.csv      # BO trajectory + final ranked sequences
    ├── Eval_PDBs/                       # evaluated complexes / final D-peptide models
    ├── Eval_PDBs/round*_<seq>_protenix/ # per-candidate protenix2dock outputs
    └── fitness_round_*.csv              # per-round intermediates
```

`results.csv` columns (Tier 1):

| Column | Meaning |
|---|---|
| `pose`, `sequence`, `score` | pose index, designed D-peptide sequence, ProteinMPNN NLL (lower = more backbone-compatible) |
| `filename` | D-peptide PDB for the sequence |
| `surface_*`, `surface_filter_pass` | geometry-filter metrics (only with `--filter_surface_poses`) |
| `synthesis_*` | SPPS tractability metrics (coupling/aggregation risk) used by the BO penalty |

`BO/bo_results.csv` columns (Tier 2, ranked by `score`):

| Column | Meaning |
|---|---|
| `Variants`, `Fitness`, `score` | sequence, BO fitness (higher = better), and `score = -Fitness` (lower = better) |
| `ipsae_dom`, `ligand_ipsae_max` | protenix2dock ipSAE interface confidence ([0, 1], higher = better) |
| `ligand_plddt` | mean pLDDT of the designed peptide chain ([0, 1], higher = better) |
| `peptide_rmsd` | pose RMSD (Å) of the engine's peptide vs the placed Chroma pose (dock mode; lower = better) |
| `iptm`, `ranking_score`, `interface_pair_count` | engine confidences reported alongside the objective terms |
| `synthesis_*` | SPPS tractability metrics; the penalty term of the fitness |

The Tier-1 ProteinMPNN **score is a sequence–backbone compatibility likelihood, not a binding
affinity**; the BO objective replaces it with Protenix's own interface confidence (ipSAE + peptide
pLDDT + pose reproducibility), which carries activity signal in cross-engine benchmarks — still a
ranking signal for experimental prioritization, not a guarantee of binding (see Discussion in the
manuscript).

## Project Structure

```
Mirror-Peptidizer/
│
├── run_design.py            # Main entry point (Tier 1 + Tier 2 BO)
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
│   ├── scoring.py           # SPPS synthesis-risk scoring, FuzzyScore
│   └── run_bo.py            # Standalone BO CLI (published MPNN objective)
│
├── utils/
│   ├── __init__.py
│   ├── pdb_processing.py    # PDB I/O, L/D conversion, structure repair
│   ├── chroma_sample.py     # Binder backbone generation via Chroma
│   ├── protein_mpnn.py      # Sequence design via ProteinMPNN
│   └── protenix2dock_client.py  # protenix2dock scoring client (dock/score modes,
│                                 ipSAE + peptide pLDDT + pose RMSD)
│
├── ProteinMPNN/             # Vendored ProteinMPNN model
├── examples/                # Example notebooks and scripts
├── docs/                    # Installation, testing & validation, computational configuration
│   ├── installation.md      # Detailed setup + 3-level self-checks
│   ├── testing_and_validation.md  # All tests, figures, tables, reproduce commands
│   ├── computational_configuration.md  # Exact ProteinMPNN & BO parameters
│   ├── figures/             # Validation figures (PNG, bundled for GitHub)
│   ├── data/                # Small result tables (CSV)
│   └── scripts/             # Bundled copies of the test scripts
├── benchmark/               # Full benchmark data, figures, recovery test (local; not in repo)
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
| `PROTENIX2DOCK_IMAGE` | For BO (Tier 2) | Protenix runtime docker image (default `vbio-protenix-v2-runtime:2.0.0`) |
| `PROTENIX2DOCK_PYTHON` | For BO (Tier 2) | Python inside the runtime image (default `/usr/local/micromamba/envs/protenix/bin/python`) |
| `PROTENIX2DOCK_VBIO_DIR` | For BO (Tier 2) | V-Bio checkout providing `capabilities/protenix2dock` (default `/data/V-Bio`) |
| `PROTENIX2DOCK_SCRIPT` | For BO (Tier 2) | protenix2dock entry script relative to the V-Bio dir |
| `PROTENIX2DOCK_MODEL_DIR` | For BO (Tier 2) | Protenix-v2 model weights directory |
| `PROTENIX2DOCK_COMMON_CACHE` | For BO (Tier 2) | Protenix shared common cache |
| `PROTENIX2DOCK_MSA_CACHE` | For BO (Tier 2) | Shared boltz MSA cache (md5-keyed `msa_<hash>.a3m`); receptor MSAs are seeded from here first |
| `PROTENIX2DOCK_MODULE_CACHE` | Optional | Writable module cache; makes repeat scoring calls ~3x faster |
| `PROTENIX2DOCK_MSA_SERVER_URL` | Optional | ColabFold-compatible MSA server; empty = shared cache only |
| `PROTENIX2DOCK_LOW_VRAM` | Optional | `1` (default) runs the engine's low-VRAM mode for 24 GB cards |

## Validation & benchmark results

The computational validation supporting the manuscript is documented in detail (with figures,
tables and reproduce commands) in **[docs/testing_and_validation.md](docs/testing_and_validation.md)**.
The bundled figures and result tables ship under `docs/figures/` and `docs/data/` so the page is
self-contained on GitHub. Headline numbers:

**Sequence-recovery validation.** Applying ProteinMPNN to the deposited MDM2 /
D-peptide complexes **3LNJ** and **8F10** after whole-complex X-axis reflection (fixed D-MDM2,
redesigned L-peptide, vanilla `v_48_020`, T = 0.1, 100 samples). A no-mirror native L-complex
control (**3HTN**) recovered **51%** of the native sequence (native score 1.37), matching the
~52% reported for ProteinMPNN on native backbones and confirming the implementation. Under
reflection, recovery was **~14% (3LNJ, mean of two complete copies)** and **21% (8F10)**;
the buried interface hot spots of the 8F10 stapled peptide (Trp, Tyr) were recovered at
**87–100%**. The depressed global recovery reflects that the reflected D-target backbone is
out-of-distribution for the L-trained model, while selective hot-spot recovery shows the
peptide backbone geometry is preserved. Script: `benchmark/recovery_test/run_recovery.py`;
report: `benchmark/recovery_test/recovery_report.md`.

**Mirror geometry QC.** φ/ψ backbones of every L-target and its D-mirror satisfy
`|φ_L + φ_D| = 0` and `|ψ_L + ψ_D| = 0` exactly across 2,397 pose pairs (6 targets) — the
reflection is a geometrically exact enantiomerization. Data: `fig_phi_psi_mirror_qc_summary.csv`.

**D- vs L-peptide score distributions.** After two-stage structural filtering, D- and
L-peptide candidate pools are highly significantly separated for all three design targets
(Kolmogorov–Smirnov p < 1 × 10⁻¹⁴⁰; e.g. MDM2 D-median 1.98 vs L-median 1.72, n = 2,007).
Data: `fig_dl_score_dist_summary.csv`.

**Ablation (filtering + Bayesian Optimization).** The geometry filter removes 350–482
buried poses per target; BO then lowers the SPPS synthesis penalty by **92–95%** relative to
Chroma+ProteinMPNN (high → low synthesis-risk class) at a modest score cost, demonstrating
the score/tractability trade-off. Data: `fig_ablation_bo_tradeoff_*.csv`.

**New-target generalization.** The same Tier-1 workflow ranks candidates on three unseen
targets — **TNFα, CXCR2, CXCR4** — producing broad, non-redundant candidate pools
(273–1,796 unique sequences; best scores 1.40–2.09; top-100 mean 1.57–2.28). Data:
`fig_new_target_score_dist_summary.csv`.

**v2 scoring oracle sanity (this branch).** On the native 3LNJ final-form complex
(L-MDM2 + D-peptide): score mode reports ipSAE `ipsae_dom` 0.75 / peptide pLDDT 0.95 /
ipTM 0.97; dock mode reproduces the crystal peptide pose at **0.19 Å** RMSD. A
sequence-scrambled control (Gly/Ala mutant on the same backbone) collapses to
`ipsae_dom` 0.05 / pLDDT 0.67 — the oracle is strongly sequence-sensitive while a
backbone-only input (side chains rebuilt by the engine) scores within 0.02 ipSAE of the
full-atom complex, matching the BO loop's input format.

### Testing & validation

> **Full guide with figures, tables and reproduce commands:** [docs/testing_and_validation.md](docs/testing_and_validation.md).
> **Detailed installation & self-checks:** [docs/installation.md](docs/installation.md).

Quick entry points:

- **Installation self-checks** (3 levels, no Chroma key needed):
  see [installation.md §6](docs/installation.md#6-verify-the-installation-3-level-self-check).
- **Protenix2dock client unit tests** (no GPU/docker needed): `python -m pytest tests/test_protenix2dock_client.py -q`
- **Full Tier-1 example** (~minutes on a 16 GB GPU): `python run_design.py --receptor data/4LWV.pdb --output output_dir --len_binder 11 --num_poses 3 --num_seqs_per_pose 10`
- **BO smoke test** (GPU + docker, synthetic pose from the 3LNJ complex): `python tests/integration/bo_v2_smoke.py`
- **Sequence-recovery validation** (~5 min on GPU): `python benchmark/recovery_test/run_recovery.py --n_samples 100 --gpu 0`
- **Mirror-axis equivalence** (pure geometry, no GPU): `python benchmark/recovery_test/mirror_axis_equivalence.py`
- **Reproduce benchmark figures**: see `benchmark/README.md` and `benchmark/figure_data/README.md` (`benchmark/scripts/plot_*.py`).

## References

- **Mirror-Peptidizer paper**: Ma B. et al. *Research* 2026, 9, 1420 — [DOI: 10.34133/research.1420](https://doi.org/10.34133/research.1420)
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) - Sequence design via protein structure
- [Chroma](https://github.com/generatebio/chroma) - Generative diffusion model for protein design
- [Boltz2Embedding](https://github.com/dahuilangda/Boltz2Embedding) - Protein sequence embedding service
- [Protenix](https://github.com/bytedance/Protenix) / protenix2dock (V-Bio `capabilities/protenix2dock`) - Protenix-engine structure workflow that scores every BO candidate (ipSAE, pLDDT, pose RMSD)

## License

Licensed under the [Apache License 2.0](LICENSE).
