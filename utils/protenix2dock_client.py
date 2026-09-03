"""Client for scoring receptor-peptide complexes with protenix2dock.

protenix2dock is the protein-ligand structure workflow on the Protenix engine
maintained in the V-Bio repository (``capabilities/protenix2dock``). This
client shells out to the Protenix runtime docker image with the same mount and
environment contract the V-Bio backend uses, so Mirror-Peptidizer always
scores with the latest protenix2dock version without vendoring the engine.

Scoring runs in ``peptide`` mode with two flavours:

- **dock** (default): receptor-fixed peptide re-docking — the diffusion
  refines the placed peptide against the pinned receptor, so the engine both
  scores the sequence and stress-tests the Chroma pose. The pose RMSD between
  the input (Chroma) peptide and the engine's output peptide (after receptor
  superposition) is reported as ``peptide_rmsd`` in Angstrom; a sequence whose
  pose Protenix cannot reproduce drifts to a high RMSD.
- **score**: diffusion bypassed (``--score_only``); the confidence heads
  evaluate the input coordinates directly. Fast single-sample pass with no
  RMSD (``peptide_rmsd`` is None).

Both flavours report the interface metrics scoped to the declared interface
chains:

- ``ipsae_dom``         dominant interface ipSAE in [0, 1] (higher = better)
- ``ligand_ipsae_max``  best directional ligand ipSAE in [0, 1]
- ``ligand_plddt``      mean peptide-chain pLDDT in [0, 1]
- ``iptm`` / ``plddt``  engine confidences (iptm in [0, 1], plddt in [0, 100])

MSA handling: Protenix wants an MSA per receptor chain. The shared boltz MSA
cache (md5-keyed ``msa_<hash>.a3m`` files, the same keys protenix2dock
derives) is seeded into the per-run work dir so repeat scoring of the same
receptor needs no MSA server. When seeding misses, the ColabFold-compatible
MSA server configured via ``PROTENIX2DOCK_MSA_SERVER_URL`` is used and its
results are written back to the shared cache. The designed (de novo) peptide
chain deliberately runs without an MSA when no server is configured; with a
server it is searched per sequence.
"""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

DEFAULTS = {
    'PROTENIX2DOCK_IMAGE': 'vbio-protenix-v2-runtime:2.0.0',
    'PROTENIX2DOCK_PYTHON': '/usr/local/micromamba/envs/protenix/bin/python',
    'PROTENIX2DOCK_VBIO_DIR': '/data/V-Bio',
    'PROTENIX2DOCK_SCRIPT': 'capabilities/protenix2dock/protenix2dock.py',
    'PROTENIX2DOCK_MODEL_DIR': '/data/protenix/model',
    'PROTENIX2DOCK_COMMON_CACHE': '/data/protenix/common_cache',
    'PROTENIX2DOCK_MSA_CACHE': '/data/boltz_msa_cache',
    'PROTENIX2DOCK_MODULE_CACHE': '/data/protenix/module_cache',
    'PROTENIX2DOCK_MSA_SERVER_URL': '',
    'PROTENIX2DOCK_LOW_VRAM': '1',
}

_STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')

_THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

METRIC_KEYS = (
    'ipsae_dom', 'ligand_ipsae_max', 'ligand_plddt', 'iptm', 'ptm',
    'plddt', 'ranking_score', 'interface_score', 'interface_pair_count',
    'pair_iptm', 'peptide_rmsd',
)


# ---------- pose RMSD (dock mode) ----------

def _pdb_ca_coords(pdb_path):
    """{chain_letter: (n,3) ndarray of CA coords} from a PDB file."""
    import numpy as np

    coords = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith('ATOM  '):
                continue
            if line[12:16].strip() != 'CA':
                continue
            chain = line[21]
            coords.setdefault(chain, []).append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return {c: np.asarray(v, dtype=float) for c, v in coords.items()}


def _cif_ca_coords(cif_path):
    """{chain_letter: (n,3) ndarray of CA coords} from an mmCIF atom_site loop.

    Minimal whitespace-column parser — Protenix writes plain unquoted values
    for protein chains, so no full CIF machinery is needed.
    """
    import numpy as np

    lines = Path(cif_path).read_text().splitlines()
    header, rows = [], []
    in_atom_site = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('_atom_site.'):
            in_atom_site = True
            header.append(stripped)
            continue
        if in_atom_site:
            if stripped in ('loop_', '#', '') or stripped.startswith('_'):
                if rows:
                    break
                continue
            if stripped.startswith('ATOM') or stripped.startswith('HETATM'):
                rows.append(stripped.split())
            elif rows:
                break
    if not header or not rows:
        raise ValueError(f'no atom_site records parsed from {cif_path}')
    idx = {name: i for i, name in enumerate(header)}
    chain_key = ('_atom_site.auth_asym_id' if '_atom_site.auth_asym_id' in idx
                 else '_atom_site.label_asym_id')
    atom_key = ('_atom_site.label_atom_id' if '_atom_site.label_atom_id' in idx
                else '_atom_site.auth_atom_id')
    coords = {}
    for row in rows:
        if len(row) < len(header) or row[idx[atom_key]] != 'CA':
            continue
        chain = row[idx[chain_key]]
        coords.setdefault(chain, []).append([
            float(row[idx['_atom_site.Cartn_x']]),
            float(row[idx['_atom_site.Cartn_y']]),
            float(row[idx['_atom_site.Cartn_z']]),
        ])
    return {c: np.asarray(v, dtype=float) for c, v in coords.items()}


def _kabsch_transform(mobile, target):
    """Rigid transform (rotation R, translation t) mapping mobile onto target."""
    import numpy as np

    mu_m, mu_t = mobile.mean(axis=0), target.mean(axis=0)
    P, Q = mobile - mu_m, target - mu_t
    V, S, Wt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt  # row-vector convention: x_target ~ x_mobile @ R
    return R, mu_t - mu_m @ R


def peptide_pose_rmsd(input_pdb, output_cif, receptor_chains, peptide_chains):
    """Docking-style pose RMSD of the peptide between input and output.

    Superposes the output receptor CA onto the input receptor CA (Kabsch) and
    reports the peptide CA RMSD in that frame — how far the engine's peptide
    moved from the placed (Chroma) pose. Residues are matched by order.
    """
    import numpy as np

    inp = _pdb_ca_coords(input_pdb)
    out = _cif_ca_coords(output_cif)
    rec = [c for c in receptor_chains if c in inp and c in out]
    if not rec:
        raise ValueError('no common receptor chains for RMSD superposition')
    mobile = np.concatenate([out[c] for c in rec], axis=0)
    target = np.concatenate([inp[c] for c in rec], axis=0)
    R, t = _kabsch_transform(mobile, target)

    sq_sum, n_total = 0.0, 0
    for chain in peptide_chains:
        if chain not in inp or chain not in out:
            continue
        n = min(len(inp[chain]), len(out[chain]))
        if n == 0:
            continue
        moved = out[chain][:n] @ R + t
        diff = moved - inp[chain][:n]
        sq_sum += float((diff ** 2).sum())
        n_total += n
    if n_total == 0:
        raise ValueError('no peptide CA atoms matched for RMSD')
    return float(np.sqrt(sq_sum / n_total))


def _best_sample_cif(summary, sample):
    """CIF path of a given sample index from the summary structures list."""
    suffix = f'_sample_{sample}.cif'
    for entry in summary.get('structures', []):
        if str(entry.get('path', '')).endswith(suffix):
            return entry['path']
    return None


def load_protenix2dock_config(overrides=None):
    """Merge DEFAULTS <- .env <- explicit overrides (None values skipped)."""
    config = dict(DEFAULTS)
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
    except ImportError:
        pass
    for key in DEFAULTS:
        env_value = os.getenv(key)
        if env_value is not None and env_value.strip() != '':
            config[key] = env_value.strip()
    for key, value in (overrides or {}).items():
        if value is not None:
            config[key] = str(value)
    return config


def _md5(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def _normalise_sequence(seq):
    # protenix2dock resolves the shared cache key on the sequence with
    # non-standard letters collapsed to A; mirror that exactly
    return ''.join(aa if aa in _STANDARD_AA else 'A'
                   for aa in str(seq).strip().upper())


def pdb_chain_sequences(complex_pdb):
    """Polymer chain sequences from a PDB file: {chain_letter: one-letter seq}.

    Only standard-residue ATOM records are counted, in file order — the same
    residues protenix2dock's gemmi parser turns into protein chains for the
    PDB files this pipeline produces.
    """
    sequences = {}
    seen_residues = set()
    with open(complex_pdb, 'r') as f:
        for line in f:
            if not line.startswith('ATOM  '):
                continue
            chain = line[21]
            resname = line[17:20].strip().upper()
            resseq = line[22:27]
            letter = _THREE_TO_ONE.get(resname)
            if letter is None:
                letter = 'X'
            key = (chain, resseq)
            if key in seen_residues:
                continue
            seen_residues.add(key)
            sequences.setdefault(chain, []).append(letter)
    return {chain: ''.join(chars) for chain, chars in sequences.items()}


def interface_chains_string(receptor_chains, peptide_chains):
    """'A,B' / 'AB,C' group string understood by protenix2dock."""
    first = ''.join(sorted(receptor_chains))
    second = ''.join(sorted(peptide_chains))
    if not first or not second:
        raise ValueError(
            f'interface needs receptor and peptide chains, got {first!r}/{second!r}')
    return f'{first},{second}'


class Protenix2DockScorer:
    """Score receptor-peptide complexes via the protenix2dock runtime."""

    def __init__(self, gpu=0, config=None):
        self.gpu = int(gpu)
        self.config = load_protenix2dock_config(config)
        if shutil.which('docker') is None:
            raise RuntimeError(
                'docker is required to run the protenix2dock runtime image '
                f'({self.config["PROTENIX2DOCK_IMAGE"]})')

    # ---------- MSA cache seeding ----------

    def seed_receptor_msa(self, complex_pdb, peptide_chains, work_dir):
        """Seed work_dir/msa/<chain>_<hash>_msa.a3m from the shared cache.

        Returns (n_seeded, n_receptor_chains). Files land under the exact
        names resolve_msa checks first, so seeded chains never query the
        MSA server.
        """
        msa_cache = Path(self.config['PROTENIX2DOCK_MSA_CACHE'])
        msa_dir = Path(work_dir) / 'msa'
        msa_dir.mkdir(parents=True, exist_ok=True)
        peptide_set = {c for group in peptide_chains for c in group.split(',')}
        sequences = pdb_chain_sequences(complex_pdb)
        receptor_chains = [c for c in sequences if c not in peptide_set]
        seeded = 0
        for chain in receptor_chains:
            seq_hash = _md5(_normalise_sequence(sequences[chain]))
            cached = msa_cache / f'msa_{seq_hash}.a3m'
            if cached.exists():
                shutil.copyfile(cached, msa_dir / f'{chain}_{seq_hash}_msa.a3m')
                seeded += 1
        return seeded, len(receptor_chains)

    # ---------- docker invocation ----------

    def _docker_command(self, args, out_dir, work_dir, extra_env=None):
        cfg = self.config
        cmd = [
            'docker', 'run', '--rm', '--entrypoint=',
            '--gpus', f'device={self.gpu}',
            # run as the invoking user so outputs stay deletable by the BO
            # loop; HOME moves to the container's writable /tmp
            '--user', f'{os.getuid()}:{os.getgid()}',
            '--env', 'HOME=/tmp',
            '--volume', f'{cfg["PROTENIX2DOCK_VBIO_DIR"]}:/workspace/vbio:ro',
            '--volume', f'{cfg["PROTENIX2DOCK_MODEL_DIR"]}:/workspace/model:ro',
            '--volume', f'{cfg["PROTENIX2DOCK_COMMON_CACHE"]}:/cache/common:ro',
            # rw: protenix2dock writes newly fetched receptor MSAs back here
            '--volume', f'{cfg["PROTENIX2DOCK_MSA_CACHE"]}:/data/msa_cache',
            '--volume', f'{out_dir}:{out_dir}',
            '--volume', f'{work_dir}:{work_dir}',
            '--volume', '/dev/shm:/dev/shm',
            '--env', 'PYTHONPATH=/workspace/vbio/vendor/protenix-source',
            '--env', 'PROTENIX_ROOT_DIR=/cache',
        ]
        module_cache = cfg.get('PROTENIX2DOCK_MODULE_CACHE', '')
        if module_cache and os.path.isdir(module_cache):
            cmd += [
                '--volume', f'{module_cache}:/cache/module_cache',
                '--env', 'PROTENIX_MODULE_CACHE_DIR=/cache/module_cache',
            ]
        for key, value in (extra_env or {}).items():
            cmd += ['--env', f'{key}={value}']
        # the V-Bio tree is mounted at /workspace/vbio, so the script runs
        # under its container-side path
        script = f'/workspace/vbio/{cfg["PROTENIX2DOCK_SCRIPT"]}'
        cmd += [
            cfg['PROTENIX2DOCK_IMAGE'],
            cfg['PROTENIX2DOCK_PYTHON'],
            script,
            *args,
        ]
        return cmd

    # ---------- public API ----------

    def score_peptide_complex(
        self,
        complex_pdb,
        peptide_chain,
        out_dir,
        work_dir=None,
        peptide_sequence=None,
        score_only=True,
        seed=42,
        interface_chains=None,
        use_msa_server='auto',
        diffusion_samples=None,
        sampling_steps=None,
        timeout=7200,
        verbose=False,
    ):
        """Score one receptor-peptide complex PDB.

        Args:
            complex_pdb: complex with receptor chain(s) + the peptide chain
            peptide_chain: chain letter of the peptide (e.g. 'B')
            out_dir: output directory (protenix2dock_summary.json lands here)
            work_dir: scratch directory; default ``<out_dir>/_work``
            peptide_sequence: authoritative one-letter sequence for the chain
            score_only: False (default) = dock mode, receptor-fixed peptide
                re-docking with diffusion and a pose RMSD; True = score mode,
                diffusion bypassed (confidence heads on input coords, no RMSD)
            interface_chains: 'A,B' group string; default receptor,peptide
            use_msa_server: 'auto' (server only when cache seeding misses),
                'on' (always pass the server URL), 'off' (never — requires
                every receptor MSA to seed from the shared cache)
            diffusion_samples / sampling_steps: overrides in dock mode

        Returns:
            dict of interface metrics for the best sample (METRIC_KEYS +
            'sample' and 'summary_path'); nan for missing fields. In dock
            mode ``peptide_rmsd`` carries the peptide CA RMSD (A) between the
            input pose and the engine's best-sample pose.
        """
        complex_pdb = Path(complex_pdb).resolve()
        out_dir = Path(out_dir).resolve()
        work_dir = Path(work_dir).resolve() if work_dir else out_dir / '_work'
        out_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        if not complex_pdb.exists():
            raise FileNotFoundError(complex_pdb)

        peptide_chains = [c.strip() for c in str(peptide_chain).split(',') if c.strip()]
        sequences = pdb_chain_sequences(complex_pdb)
        for chain in peptide_chains:
            if chain not in sequences:
                raise ValueError(
                    f'peptide chain {chain!r} not found in {complex_pdb} '
                    f'(chains: {sorted(sequences)})')
        if interface_chains is None:
            receptor_chains = [c for c in sequences if c not in peptide_chains]
            interface_chains = interface_chains_string(receptor_chains, peptide_chains)

        # the complex is staged inside the mounted work dir so the container
        # sees it under the same absolute path
        staged = work_dir / 'complex.pdb'
        shutil.copyfile(complex_pdb, staged)

        seeded, n_receptor = self.seed_receptor_msa(complex_pdb, peptide_chains, work_dir)
        server_url = self.config.get('PROTENIX2DOCK_MSA_SERVER_URL', '')
        if use_msa_server == 'off':
            pass_server = False
        elif use_msa_server == 'on':
            pass_server = True
        else:  # auto
            pass_server = seeded < n_receptor or n_receptor == 0
        if pass_server and not server_url:
            if use_msa_server == 'on' or seeded < n_receptor:
                raise RuntimeError(
                    'receptor MSA not in the shared cache and no MSA server '
                    'configured; set PROTENIX2DOCK_MSA_SERVER_URL or pre-seed '
                    f'{self.config["PROTENIX2DOCK_MSA_CACHE"]}')

        args = [
            '--mode', 'peptide',
            '--input', str(staged),
            '--peptide_chain', ','.join(peptide_chains),
            '--output_dir', str(out_dir),
            '--work_dir', str(work_dir),
            '--msa_cache_dir', '/data/msa_cache',
            '--seed', str(seed),
            '--interface_chains', interface_chains,
        ]
        if peptide_sequence:
            args += ['--peptide_sequence', str(peptide_sequence).strip().upper()]
        if score_only:
            # confidence-only pass: a single sample suffices (the peptide-mode
            # default of 8 wastes disk and time on identical evaluations)
            args += ['--score_only', '--diffusion_samples', '1']
        else:
            # dock mode: receptor-fixed peptide re-docking with diffusion
            if diffusion_samples:
                args += ['--diffusion_samples', str(int(diffusion_samples))]
            if sampling_steps:
                args += ['--sampling_steps', str(int(sampling_steps))]
        if pass_server:
            args += ['--msa_server_url', server_url]
        if str(self.config.get('PROTENIX2DOCK_LOW_VRAM', '1')).strip() in ('1', 'true', 'yes'):
            args.append('--low_vram')

        cmd = self._docker_command(args, out_dir, work_dir)
        if verbose:
            print('[protenix2dock]', ' '.join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                'protenix2dock failed (exit '
                f'{result.returncode}):\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}')

        summary_path = out_dir / 'protenix2dock_summary.json'
        if not summary_path.exists():
            raise RuntimeError(f'protenix2dock wrote no summary at {summary_path}')
        summary = json.loads(summary_path.read_text())
        confidences = summary.get('confidences') or []
        if not confidences:
            raise RuntimeError('protenix2dock summary has no confidences')
        best = (summary.get('best_by_interface')
                or summary.get('best') or confidences[0])
        metrics = {key: best.get(key) for key in METRIC_KEYS}
        metrics['sample'] = best.get('sample')
        metrics['summary_path'] = str(summary_path)

        if not score_only:
            # dock mode: how far the engine moved the peptide from the placed
            # (Chroma) pose, in the receptor-superposed frame
            cif = _best_sample_cif(summary, best.get('sample'))
            if cif is None:
                raise RuntimeError(
                    f'no output CIF for best sample {best.get("sample")}; '
                    'cannot compute pose RMSD')
            receptor_chains = [c for c in sequences
                               if c not in peptide_chains]
            metrics['peptide_rmsd'] = peptide_pose_rmsd(
                staged, cif, receptor_chains, peptide_chains)
        return metrics


def summarise_metrics(metrics):
    """One-line human-readable digest of a metrics dict."""
    def fmt(key, scale=1.0):
        value = metrics.get(key)
        return 'n/a' if value is None else f'{float(value) * scale:.4f}'
    rmsd = metrics.get('peptide_rmsd')
    rmsd_txt = 'n/a' if rmsd is None else f'{float(rmsd):.3f}'
    return (f"ipsae_dom={fmt('ipsae_dom')} "
            f"ligand_plddt={fmt('ligand_plddt')} "
            f"ligand_ipsae_max={fmt('ligand_ipsae_max')} "
            f"iptm={fmt('iptm')} "
            f"pose_rmsd={rmsd_txt}")
