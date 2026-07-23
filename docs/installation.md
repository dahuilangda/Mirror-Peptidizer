# Installation

> This page covers **installation** end-to-end. For the test/validation walkthrough see
> [testing_and_validation.md](testing_and_validation.md), and for the exact ProteinMPNN /
> Bayesian-Optimization parameters see [computational_configuration.md](computational_configuration.md).

Mirror-Peptidizer runs on Linux with an NVIDIA GPU. The full stack is **Chroma** (binder backbone
generation) + **ProteinMPNN** (sequence design, vendored) + an optional **Bayesian-Optimization**
module. Chroma and ProteinMPNN both need PyTorch with CUDA.


## 1. System requirements

| Resource | Minimum | Recommended |
|---|---|---|
| OS | Linux (Ubuntu 20.04+) | Ubuntu 22.04 |
| GPU | NVIDIA, 12 GB VRAM, CUDA 11.8+ | 24 GB VRAM |
| CPU | 8 cores | 16+ cores |
| RAM | 8 GB | 32 GB |
| Disk | ~20 GB (weights + outputs) | SSD |


## 2. Get the code

```bash
git clone https://github.com/dahuilangda/Mirror-Peptidizer.git
cd Mirror-Peptidizer
```


## 3. Create the environment

We recommend **mamba** (or conda). The reference environment used for all published results is
Python 3.9 / 3.10.

### Option A — mamba / conda (recommended)

```bash
mamba create -n dpep python=3.10 -y
mamba activate dpep

# PyTorch with CUDA (pick the line matching your CUDA version)
mamba install -c pytorch -c nvidia pytorch pytorch-cuda=12.1 -y
# or:  pip install torch --index-url https://download.pytorch.org/whl/cu121

# Structural + plotting stack
mamba install -c conda-forge biopython pdbfixer openmm matplotlib seaborn pandas scipy -y

# Chroma (binder backbone diffusion model)
pip install generate-chroma

# Misc
pip install python-dotenv requests
```

### Option B — pip + venv

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install generate-chroma biopython pdbfixer openmm matplotlib seaborn pandas scipy python-dotenv requests
```

### Reference package versions (tested)

| Package | Version | Used by |
|---|---|---|
| `torch` | 2.x (CUDA 11.8/12.1) | ProteinMPNN, Chroma |
| `generate-chroma` | latest | Tier-1 binder backbone |
| `biopython` | 1.84 | PDB I/O, φ/ψ QC, recovery test |
| `pdbfixer` / `openmm` | 1.9 / 8.1 | PDB repair + minimization |
| `matplotlib` / `seaborn` | 3.9 / 0.13 | figures |
| `pandas` / `numpy` / `scipy` | 2.2 / 1.26 / 1.13 | data handling, KS tests, GP |
| `python-dotenv` / `requests` | latest | `.env` loading, Boltz2Embedding client |


## 4. Model weights

### 4.1 ProteinMPNN (vendored, already in the repo)

The checkpoints ship with the repository under `ProteinMPNN/`:

```
ProteinMPNN/
├── vanilla_model_weights/   # v_48_020 (default), v_48_002 / 010 / 030
└── soluble_model_weights/   # v_48_010 / 020 (soluble-domain prior)
```

The manuscript documents the **vanilla `v_48_020`** checkpoint. Nothing to download.

### 4.2 Chroma weights

Download the Chroma backbone + design weights from [Generate Biomedicines](https://generatebiomedicines.com/chroma)
and place them under `chroma_weights/`, then point the env var at them (see step 5).


## 5. Environment variables

Copy the template and edit it:

```bash
cp env_example .env
```

| Variable | Required? | Purpose |
|---|---|---|
| `CHROMA_KEY` | **Yes** (Tier 1) | Chroma API key |
| `CHROMA_WEIGHTS_DIR` | **Yes** (Tier 1) | directory holding the Chroma weight files |
| `CHROMA_WEIGHTS_BACKBONE` / `CHROMA_WEIGHTS_DESIGN` | Optional | explicit backbone/design weight paths (override `CHROMA_WEIGHTS_DIR`) |
| `ProteinMPNN_CHECKPOINT` | Optional | Path to an MPNN checkpoint. **Defaults to `ProteinMPNN/vanilla_model_weights/v_48_020.pt`** — leave unset unless you intentionally want the soluble prior. |
| `BOLTZ2EMBEDDING_URL` | Only for BO + Boltz2Embedding | self-hosted Boltz2Embedding server URL |
| `BOLTZ2EMBEDDING_TOKEN` | Only for BO + Boltz2Embedding | API token for the above |

Example `.env`:

```ini
CHROMA_KEY = 'your_chroma_api_key_here'
CHROMA_WEIGHTS_DIR = '/path/to/chroma_weights'
ProteinMPNN_CHECKPOINT = 'ProteinMPNN/vanilla_model_weights/v_48_020.pt'
# BOLTZ2EMBEDDING_URL = 'http://your-boltz2-server:8000'
# BOLTZ2EMBEDDING_TOKEN = 'your-api-token'
```


## 6. Verify the installation (3-level self-check)

Run these in order. Each level adds one more dependency.

**Level 1 — structural stack (no GPU needed):**

```bash
python -c "
from Bio.PDB import PDBParser, Superimposer
import openmm, pdbfixer
import matplotlib, seaborn, pandas, scipy
print('structural + plotting stack OK')
"
```

**Level 2 — GPU + PyTorch:**

```bash
python -c "
import torch
print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(),
      '| n GPU:', torch.cuda.device_count())
"
```

**Level 3 — ProteinMPNN weights load:**

```bash
python -c "
from utils.protein_mpnn import load_model, resolve_checkpoint_path
print('checkpoint:', resolve_checkpoint_path())
load_model()
print('ProteinMPNN v_48_020 loaded OK')
"
```

**Level 4 (optional) — end-to-end smoke (needs Chroma key + GPU):**

```bash
python run_design.py --receptor data/4LWV.pdb --output output_dir \
    --len_binder 11 --num_poses 1 --num_seqs_per_pose 8 --gpu 0
```

A successful run writes `output_dir/results.csv` plus `Poses/` and `Binders/`. See
[Output interpretation in the README](../README.md#output-interpretation).


## 7. Troubleshooting

<details>
<summary><b>PyTorch / CUDA mismatch</b></summary>

`torch.cuda.is_available()` returns `False` → your torch was built for a different CUDA than your
driver. Reinstall torch from the matching wheel, e.g. for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```
</details>

<details>
<summary><b>Chroma import or weight errors</b></summary>

Confirm `generate-chroma` is installed (`python -c "from chroma import Chroma"`) and that
`CHROMA_WEIGHTS_DIR` (or the explicit `_BACKBONE` / `_DESIGN` paths) points to the unpacked
weight files. The weights must match the `generate-chroma` version.
</details>

<details>
<summary><b>Soluble vs vanilla ProteinMPNN checkpoint</b></summary>

If `ProteinMPNN_CHECKPOINT` points at the soluble weights, every run uses the soluble prior.
To reproduce the manuscript, **unset it** (or point it at `vanilla_model_weights/v_48_020.pt`).
See [computational_configuration.md §2.1](computational_configuration.md#21-model-and-weights).
</details>

<details>
<summary><b>Slow downloads behind a firewall</b></summary>

Use the Tsinghua PyPI mirror used during development:

```bash
pip install <pkg> -i https://pypi.tuna.tsinghua.edu.cn/simple
```
</details>

<details>
<summary><b>Memory errors on large pose batches</b></summary>

Reduce `--num_poses` and `--num_seqs_per_pose`, or run Tier-1 only (no `--bo_rounds`). Chroma
backbone generation is the main VRAM consumer.
</details>


## Next steps

- **Run a design:** [README → Usage](../README.md#usage)
- **Reproduce the validation figures:** [testing_and_validation.md](testing_and_validation.md)
- **Exact model/BO parameters:** [computational_configuration.md](computational_configuration.md)
