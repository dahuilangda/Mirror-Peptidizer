# Computational configuration

> See also: [installation.md](installation.md) (setup) · [testing_and_validation.md](testing_and_validation.md)
> (all tests & validation figures).

This document records the **exact parameters** used by the two learning components of
Mirror-Peptidizer — **ProteinMPNN** (sequence design) and **Bayesian Optimization**
(Tier 2) — so that every published result can be reproduced. Each entry references the
source file and line where the value is set. Defaults are what the manuscript used unless
a command-line override is noted.


## 1. Coordinate mirroring (L ↔ D conversion)

| Item | Value | Source |
|---|---|---|
| Transform | reflect X axis: `x → −x`, `y, z` unchanged | `utils/pdb_processing.py:7` (`ld_convert`) |
| Convention | same as the mirror-image PDB of Garton et al., *PNAS* 2018 (115, 1505–1510) | — |

Mirroring is a rigid, orientation-reversing operation (determinant −1); it converts an
L-protein into its D-enantiomer without changing residue identities.


## 2. ProteinMPNN sequence design

### 2.1 Model and weights

| Item | Value | Source |
|---|---|---|
| Upstream repo | `dauparas/ProteinMPNN` (vendored copy) | `ProteinMPNN/` |
| Checkpoint (default) | `ProteinMPNN/vanilla_model_weights/v_48_020.pt` | `utils/protein_mpnn.py:14-19,23-28` |
| Override env var | `ProteinMPNN_CHECKPOINT` | `utils/protein_mpnn.py:25` |
| Alternative weights | `v_48_002 / 010 / 030` (vanilla), `v_48_010 / 020` (soluble) | `ProteinMPNN/{vanilla,soluble}_model_weights/` |
| Model name meaning | 48 neighbor edges, 0.20 Å training backbone noise | checkpoint `num_edges=48`, `noise_level` |

### 2.2 Network configuration

| Hyperparameter | Value | Source |
|---|---|---|
| `node_features` / `edge_features` / `hidden_dim` | 128 | `utils/protein_mpnn.py:53,62-64` |
| Encoder layers (`num_encoder_layers`) | 3 | `utils/protein_mpnn.py:54,66` |
| Decoder layers (`num_decoder_layers`) | 3 | `utils/protein_mpnn.py:67` |
| `augment_eps` (backbone noise at inference) | 0.0 (disabled) | `utils/protein_mpnn.py:55,68` |
| `k_neighbors` | 48 (read from checkpoint) | `utils/protein_mpnn.py:69` |
| `num_letters` | 21 (20 aa + X) | `utils/protein_mpnn.py:62` |
| Alphabet | `ACDEFGHIKLMNPQRSTVWYX` | `utils/protein_mpnn.py:166` |
| Omitted residues | `X` | `utils/protein_mpnn.py:167` |
| Residue bias | none (zeros) | `utils/protein_mpnn.py:169` |

### 2.3 Input preparation

| Item | Value | Source |
|---|---|---|
| PDB parser | `protein_mpnn_utils.parse_PDB` | `utils/protein_mpnn.py:85` |
| Dataset | `StructureDatasetPDB(truncate=None, max_length=10000)` | `utils/protein_mpnn.py:86` |
| Designed chain | the binder chain (`B` by default — next letter after receptor chains) | `run_design.py:379`; `utils/protein_mpnn.py:89` |
| Fixed chains | all chains except the designed one | `utils/protein_mpnn.py:90` |
| Fixed positions | none (`fixed_positions_dict=None`) | `utils/protein_mpnn.py:79` |
| Omit / tie / PSSM / per-residue bias | none | `utils/protein_mpnn.py:80-83` |
| MPNN input in the main pipeline | the raw Chroma L-binder pose PDB (no energy minimization before MPNN) | `run_design.py:466-472` |

### 2.4 Sampling

| Parameter | Default | Source |
|---|---|---|
| `temperature` | **0.1** (Tier 1) | `run_design.py:557`; manuscript methods |
| `batch_size` (= sequences per pose) | 8 (CLI `--num_seqs_per_pose`) | `run_design.py:559` |
| `score_mode` | `designed` (score only the designed chain) | `utils/protein_mpnn.py:130,223` |
| Decoding order | model-sampled (`use_input_decoding_order=True`) | `utils/protein_mpnn.py:238-239` |
| PSSM | off (`pssm_multi=0`, all `pssm_*_flag=False`) | `utils/protein_mpnn.py:195-198` |

### 2.5 Score and recovery

- **Native / designed score** = mean negative log-likelihood over the designed-chain
  positions, `utils._scores(S, log_probs, scoring_mask)` with
  `scoring_mask = mask * chain_M * chain_M_pos`
  (`utils/protein_mpnn.py:30-44`, `223-241`). Lower is better; the manuscript ranks
  candidates by ascending score.
- **Sequence recovery** (recovery test) = fraction of designed positions whose sampled
  residue equals the native residue:
  `Σ(onehot(S)·onehot(S_sample)·scoring_mask) / Σ(scoring_mask)`
  (`utils/protein_mpnn.py:258-264`; reused verbatim in
  `benchmark/recovery_test/run_recovery.py`).

### 2.6 Sequence-recovery validation

`benchmark/recovery_test/run_recovery.py` applies the production code path to PDB **3LNJ**
and **8F10** (MDM2 / D-peptide). Per-residue D-codes are mapped to standard L codes, caps /
staple (WHL) / ligands / waters are removed, backbone atoms (N, CA, C, O) are kept, the
**whole complex** is reflected along X, the mirrored MDM2 chain is fixed and the mirrored
L-peptide chain redesigned with **vanilla `v_48_020`** at T = 0.1, 100 samples. A native
L-complex (3HTN) redesigned *without* mirroring is the positive control. Results and an
SI-ready methods paragraph are written to `benchmark/recovery_test/recovery_report.md`.


## 3. Bayesian Optimization (Tier 2)

### 3.1 Surrogate model

| Item | Value | Source |
|---|---|---|
| Model | `GPRegressor` (Gaussian process) | `bo/models.py:7` |
| Kernel | Matern (ν = 5/2) or RBF; **Matern** default | `bo/models.py:14,49-53`; `run_design.py:606` |
| Length scales | per-dimension (ARD), log-parameterized | `bo/models.py:_scaled_distance` |
| Optimized params | log-lengthscale, log-signal, log-noise | `bo/models.py:36-39` |
| Objective | maximize log marginal likelihood | `bo/models.py` docstring |
| `training_iters` | 120 | `bo/models.py:15` |
| Learning rate | 0.05 | `bo/models.py:16` |
| Output normalization | y z-score standardized (`_y_mean`, `_y_std`) | `bo/models.py:29-30` |
| Cholesky jitter | 1e-6 → 1e-2 fallback | `bo/models.py:_safe_cholesky` |

### 3.2 Sequence encoding

| Encoder | Dimension | Source |
|---|---|---|
| `OneHotEncoder` (default) | L × 20 | `bo/encoders.py:48` |
| `PhysicochemicalEncoder` | L × 6 (6 per-residue property features) | `bo/encoders.py:75` |
| `Boltz2Encoder` | 384 (pooled; needs self-hosted service) | `bo/encoders.py:100` |

### 3.3 Acquisition functions

`UCB` (default), `LCB`, `EI`, `PI`, `TS`, `NEI`, `QUCB`, `Greedy` (`bo/acquisition.py`).
Default `--bo_acquisition UCB`, `--bo_uf_param 0.2` (`run_design.py:608-612`).

### 3.4 Explorers

| Item | Value | Source |
|---|---|---|
| Explorer | `BO_EVO` (default) or `MCMC`, plus `Boltz2BO` | `run_design.py:602`; `bo/explorers.py` |
| `expmt_queries_per_round` (`--bo_trials`) | 10 | `run_design.py:600` |
| `model_queries_per_round` (`--bo_model_queries`) | 3000 | `run_design.py:613` |
| Mutation range | 1–3 substitutions (`proposal_min/max_mutations`) | `run_design.py:622-623` |
| Batch diversity | min Hamming distance 2 within a batch | `run_design.py:624` |
| Proposal preflight | 600 samples | `run_design.py:627` |

### 3.5 Initialization, fitness and stopping

| Item | Value | Source |
|---|---|---|
| `init_source` | `tier1` (pose MPNN candidates), `random`, or `single` | `run_design.py:615-617` |
| `init_seed` | `100000 + len_binder*1000 + i` | `run_design.py:535-538` |
| Fitness | `fitness = −MPNN_score − synthesis_penalty_weight · synthesis_penalty` | `run_design.py:147` |
| `synthesis_penalty_weight` | 0.5 | `run_design.py:620` |
| Proposal synthesis guard | weight 0.25, max 3.1 | `run_design.py:625-626` |
| Iterations / stopping | fixed `bo_rounds` rounds (0 = off; manuscript ablation used 5, examples use 8); no early stopping | `run_design.py:598` |

**Synthesis penalty** combines SPPS-relevant terms:
per-coupling-step risk (e.g. difficult X-Pro steps), fragment aggregation propensity, and
the longest run of consecutive difficult couplings. This is the
SWI / synthesis-risk term referenced in the manuscript fitness function.


## 4. How these map to the CLI

The defaults above are exactly what `run_design.py` uses. Representative invocations:

```bash
# Tier 1 only (Chroma pose + ProteinMPNN sequence design, vanilla v_48_020, T=0.1)
python run_design.py --receptor data/4LWV.pdb --output output_dir \
    --len_binder 11 --temperature 0.1 --num_poses 3 --num_seqs_per_pose 10 --gpu 0

# Tier 1 + Bayesian Optimization (Matern GP, UCB, one-hot, 0.5 synthesis weight)
python run_design.py --receptor data/4LWV.pdb --output output_dir \
    --bo_rounds 8 --bo_trials 10 --bo_method BO --bo_acquisition UCB \
    --bo_kernel Matern --bo_embedding onehot --synthesis_penalty_weight 0.5
```

See `README.md` for the full argument reference and `benchmark/recovery_test/` for the
sequence-recovery validation script.
