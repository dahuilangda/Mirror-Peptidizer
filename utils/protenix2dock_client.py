"""Client for scoring receptor-peptide complexes with protenix2dock.

protenix2dock is the protein-ligand structure workflow on the Protenix engine
maintained in the open-source V-Bio repository
(https://github.com/dahuilangda/V-Bio, ``capabilities/protenix2dock``). This
client shells out to the Protenix runtime docker image with the same mount and
environment contract the V-Bio backend uses, so Mirror-Peptidizer always
scores with the latest protenix2dock version without vendoring the engine.

Configuration is deliberately minimal. Everything is optional and defaults to
the standard V-Bio deployment layout; where environment variables exist they
reuse V-Bio's own names, so a working V-Bio ``deploy/docker/*.env`` can be
copied verbatim:

| Setting | Env var (V-Bio name) | Default |
|---|---|---|
| V-Bio checkout | `PROTENIX2DOCK_VBIO_DIR` | auto-discovered (sibling `../V-Bio`, `/data/V-Bio`) |
| runtime image | `PROTENIX_DOCKER_IMAGE` | `vbio-protenix-v2-runtime:2.0.0` |
| Protenix weights | `PROTENIX_MODEL_DIR` | `/data/protenix/model` |
| CCD / cluster cache | `PROTENIX_COMMON_CACHE_DIR` | `/data/protenix/common_cache` |
| module cache (speed) | `PROTENIX_MODULE_CACHE_DIR` | `/data/protenix/module_cache` |
| shared MSA cache | `BOLTZ_MSA_CACHE_DIR` | `/data/boltz_msa_cache` |
| ColabFold MSA server | `MSA_SERVER_URL` | *(empty: shared cache only)* |

Setting up the runtime from scratch is three steps (see the README):

```bash
git clone https://github.com/dahuilangda/V-Bio.git ../V-Bio
docker build -f deploy/docker/DOCKER_PROTENIX_V2_RUNTIME.Dockerfile \
    -t vbio-protenix-v2-runtime:2.0.0 .      # run inside the V-Bio checkout
# put the Protenix-v2 checkpoint + Protenix common data under /data/protenix/
python -m utils.protenix2dock_client --check  # verify everything
```

Scoring runs in ``peptide`` mode with two flavours:

- **score** (recommended for BO fitness, and the run_design default since
  2026-09-04): diffusion bypassed (``--score_only``) — the confidence heads
  evaluate the Chroma-staged complex directly, so the fitness scores the
  ACTUAL candidate pose. Validated core path (3LNJ round-trip ipTM
  0.935-0.948). Fast single-sample pass; ``peptide_rmsd`` is None.
- **dock**: receptor-pinned BLIND inpainting — the peptide denoises from
  pure noise on the full schedule, so the pose comes from the engine's own
  prior (MSA/trained docking) and the reported ``peptide_rmsd`` (input
  Chroma pose vs engine output, receptor-superposed) is an INDEPENDENT
  agreement test: low RMSD means the engine confirms the Chroma pose, high
  RMSD flags disagreement. Stress test / confirmation, not the primary
  fitness; for de novo sequences without homologs the blind pose quality
  is not guaranteed (see /data/Boltz2Score/dpeptide_test/REPORT.md).

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
MSA server configured via ``MSA_SERVER_URL`` is used and its results are
written back to the shared cache. The designed (de novo) peptide chain
deliberately runs without an MSA when no server is configured; with a server
it is searched per sequence.
"""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse, urlunparse

DEFAULT_IMAGE = 'vbio-protenix-v2-runtime:2.0.0'
DEFAULT_MODEL_DIR = '/data/protenix/model'
DEFAULT_COMMON_CACHE = '/data/protenix/common_cache'
DEFAULT_MODULE_CACHE = '/data/protenix/module_cache'
DEFAULT_MSA_CACHE = '/data/boltz_msa_cache'
DEFAULT_SCRIPT = 'capabilities/protenix2dock/protenix2dock.py'
DEFAULT_PYTHON = '/usr/local/micromamba/envs/protenix/bin/python'

# env var -> (config key, default); names mirror V-Bio's deploy env files so a
# V-Bio deployment's values work here unchanged
ENV_SETTINGS = {
    'PROTENIX_DOCKER_IMAGE': ('image', DEFAULT_IMAGE),
    'PROTENIX_MODEL_DIR': ('model_dir', DEFAULT_MODEL_DIR),
    'PROTENIX_COMMON_CACHE_DIR': ('common_cache', DEFAULT_COMMON_CACHE),
    'PROTENIX_MODULE_CACHE_DIR': ('module_cache', DEFAULT_MODULE_CACHE),
    'BOLTZ_MSA_CACHE_DIR': ('msa_cache', DEFAULT_MSA_CACHE),
    'MSA_SERVER_URL': ('msa_server_url', ''),
    'PROTENIX2DOCK_VBIO_DIR': ('vbio_dir', None),
    'PROTENIX2DOCK_PYTHON': ('python_bin', DEFAULT_PYTHON),
}


def _repo_root():
    return Path(__file__).resolve().parents[1]


def discover_vbio_dir():
    """Find a V-Bio checkout that carries capabilities/protenix2dock.

    Order: $PROTENIX2DOCK_VBIO_DIR, $VBIO_DIR, a sibling of this repo
    (``<repo>/../V-Bio``), ``/data/V-Bio``; the first directory actually
    containing the protenix2dock entry script wins.
    """
    candidates = []
    for env_name in ('PROTENIX2DOCK_VBIO_DIR', 'VBIO_DIR'):
        value = os.getenv(env_name, '').strip()
        if value:
            candidates.append(Path(value))
    candidates.append(_repo_root().parent / 'V-Bio')
    candidates.append(Path('/data/V-Bio'))
    for candidate in candidates:
        if (candidate / DEFAULT_SCRIPT).is_file():
            return str(candidate)
    return None

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
    """Resolve the client configuration.

    Precedence: explicit overrides (constructor kwargs) > environment
    (optionally loaded from the repo's .env) > V-Bio-standard defaults. The
    V-Bio checkout is auto-discovered when not configured explicitly.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(_repo_root() / '.env')
    except ImportError:
        pass
    config = {key: default for _, (key, default) in ENV_SETTINGS.items()}
    for env_name, (key, _default) in ENV_SETTINGS.items():
        value = os.getenv(env_name)
        if value is not None and value.strip() != '':
            config[key] = value.strip()
    if config.get('vbio_dir') is None:
        config['vbio_dir'] = discover_vbio_dir()
    config['script'] = DEFAULT_SCRIPT
    for key, value in (overrides or {}).items():
        if value is not None:
            config[key] = str(value)
    return config


def _md5(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def _container_msa_url(server_url):
    """Rewrite a loopback MSA server URL for use inside the runtime container.

    A URL like http://127.0.0.1:8080 (the common deployment: the MSA server
    runs on the host next to this client) points at the container itself under
    docker's default bridge network. host.docker.internal — with
    ``--add-host host.docker.internal:host-gateway`` added to the docker run —
    reaches the host instead.

    Returns (url_for_container, is_loopback).
    """
    if not server_url:
        return server_url, False
    parsed = urlparse(server_url)
    if parsed.hostname in ('127.0.0.1', 'localhost', '::1'):
        netloc = parsed.netloc.replace(parsed.hostname, 'host.docker.internal', 1)
        return urlunparse(parsed._replace(netloc=netloc)), True
    return server_url, False


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
    """Score receptor-peptide complexes via the protenix2dock runtime.

    Args mirror the environment settings (see module docstring); every kwarg
    is optional and defaults to the V-Bio-standard layout:

        vbio_dir:       V-Bio checkout (default: auto-discovered)
        image:          runtime docker image
        model_dir:      Protenix-v2 checkpoint directory
        common_cache:   Protenix CCD / cluster cache directory
        module_cache:   writable module cache (repeat-call speed-up)
        msa_cache:      shared md5-keyed MSA cache directory
        msa_server_url: ColabFold-compatible MSA server ('' = cache only)
        python_bin:     python inside the runtime image
    """

    def __init__(self, gpu=0, config=None, **overrides):
        self.gpu = int(gpu)
        self.config = load_protenix2dock_config({**(config or {}), **overrides})
        if shutil.which('docker') is None:
            raise RuntimeError(
                'docker is required to run the protenix2dock runtime image '
                f'({self.config["image"]}); install docker first')
        if not self.config.get('vbio_dir'):
            raise RuntimeError(
                'no V-Bio checkout found: clone https://github.com/dahuilangda/V-Bio '
                '(e.g. next to this repo) or set PROTENIX2DOCK_VBIO_DIR')

    # ---------- MSA cache seeding ----------

    def seed_receptor_msa(self, complex_pdb, peptide_chains, work_dir):
        """Seed work_dir/msa/<chain>_<hash>_msa.a3m from the shared cache.

        Returns (n_seeded, n_receptor_chains). Files land under the exact
        names resolve_msa checks first, so seeded chains never query the
        MSA server.
        """
        msa_cache = Path(self.config['msa_cache'])
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

    def _docker_command(self, args, out_dir, work_dir, extra_env=None,
                        add_host_gateway=False):
        cfg = self.config
        cmd = [
            'docker', 'run', '--rm', '--entrypoint=',
            '--gpus', f'device={self.gpu}',
            # run as the invoking user so outputs stay deletable by the BO
            # loop; HOME moves to the container's writable /tmp
            '--user', f'{os.getuid()}:{os.getgid()}',
            '--env', 'HOME=/tmp',
            '--volume', f'{cfg["vbio_dir"]}:/workspace/vbio:ro',
            '--volume', f'{cfg["model_dir"]}:/workspace/model:ro',
            '--volume', f'{cfg["common_cache"]}:/cache/common:ro',
            # rw: protenix2dock writes newly fetched receptor MSAs back here
            '--volume', f'{cfg["msa_cache"]}:/data/msa_cache',
            '--volume', f'{out_dir}:{out_dir}',
            '--volume', f'{work_dir}:{work_dir}',
            '--volume', '/dev/shm:/dev/shm',
            '--env', 'PYTHONPATH=/workspace/vbio/vendor/protenix-source',
            '--env', 'PROTENIX_ROOT_DIR=/cache',
        ]
        module_cache = cfg.get('module_cache', '')
        if module_cache and os.path.isdir(module_cache):
            cmd += [
                '--volume', f'{module_cache}:/cache/module_cache',
                '--env', 'PROTENIX_MODULE_CACHE_DIR=/cache/module_cache',
            ]
        if add_host_gateway:
            # lets the engine container reach an MSA server bound to the
            # host's loopback via host.docker.internal
            cmd += ['--add-host', 'host.docker.internal:host-gateway']
        for key, value in (extra_env or {}).items():
            cmd += ['--env', f'{key}={value}']
        # the V-Bio tree is mounted at /workspace/vbio, so the script runs
        # under its container-side path
        script = f'/workspace/vbio/{cfg["script"]}'
        cmd += [
            cfg['image'],
            cfg['python_bin'],
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
            use_msa_server: 'auto' (server when a receptor MSA misses the
                shared cache, and always in dock mode — the de novo peptide
                chain needs a freshly searched MSA there), 'on' (always pass
                the server URL), 'off' (never — requires every receptor MSA
                to seed from the shared cache; dock mode then needs the
                peptide MSA cached too)
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
        server_url = self.config.get('msa_server_url', '')
        if use_msa_server == 'off':
            pass_server = False
        elif use_msa_server == 'on':
            pass_server = True
        else:  # auto
            # dock mode must also fetch an MSA for the de novo peptide chain
            # (engine requirement); score mode tolerates a peptide without one
            pass_server = seeded < n_receptor or n_receptor == 0 or not score_only
        if pass_server and not server_url:
            # Official MSA semantics (2026-09-04): the MSA is OPTIONAL input.
            # Without a server the chains run on the query-only MSA (N_msa=1)
            # — the engine featurizes it natively. Loud log, not an error.
            print(
                '[protenix2dock] no MSA server configured — chains run on '
                'query-only MSAs (official N_msa=1 contract); set '
                'MSA_SERVER_URL for homolog-enriched scoring')
            pass_server = False

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
            # dock mode: receptor-pinned BLIND inpainting (2026-09-04 V-Bio
            # protocol). The peptide denoises from PURE NOISE on the full
            # schedule; the pose comes from the engine's own prior (MSA /
            # trained docking), never from a hand placement. peptide_rmsd vs
            # the Chroma pose becomes an INDEPENDENT pose agreement test.
            args += ['--blind_peptide']
            if diffusion_samples:
                args += ['--diffusion_samples', str(int(diffusion_samples))]
            if sampling_steps:
                args += ['--sampling_steps', str(int(sampling_steps))]
        loopback = False
        if pass_server:
            container_url, loopback = _container_msa_url(server_url)
            args += ['--msa_server_url', container_url]

        cmd = self._docker_command(args, out_dir, work_dir,
                                   add_host_gateway=pass_server and loopback)
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


def check_setup(gpu=0):
    """Verify every protenix2dock dependency and print a doctor report.

    Returns True when scoring should work. Purely diagnostic — never raises.
    """
    cfg = load_protenix2dock_config()
    ok = True

    def report(item, fine, detail='', fix=''):
        nonlocal ok
        ok = ok and fine
        mark = 'OK  ' if fine else 'FAIL'
        print(f'[{mark}] {item}' + (f' — {detail}' if detail else ''))
        if not fine and fix:
            print(f'       fix: {fix}')

    docker_bin = shutil.which('docker')
    report('docker', docker_bin is not None,
           docker_bin or '', 'install docker with NVIDIA Container Toolkit')

    image = cfg['image']
    image_present = False
    if docker_bin:
        probe = subprocess.run(['docker', 'image', 'inspect', image],
                               capture_output=True, text=True)
        image_present = probe.returncode == 0
    report('runtime image', image_present, image,
           'cd <V-Bio checkout> && docker build -f deploy/docker/'
           'DOCKER_PROTENIX_V2_RUNTIME.Dockerfile -t ' + image + ' .')

    vbio = cfg.get('vbio_dir')
    report('V-Bio checkout', bool(vbio), vbio or 'not found',
           'git clone https://github.com/dahuilangda/V-Bio.git (e.g. next to '
           'this repo or /data/V-Bio), or set PROTENIX2DOCK_VBIO_DIR')

    model_dir = Path(cfg['model_dir'])
    weights = sorted(model_dir.glob('*.pt')) if model_dir.is_dir() else []
    report('Protenix weights', bool(weights),
           f'{model_dir}: {", ".join(w.name for w in weights) or "empty"}',
           'place the Protenix-v2 checkpoint (protenix-v2.pt) there, or set '
           'PROTENIX_MODEL_DIR')

    common = Path(cfg['common_cache'])
    has_common = (common / 'components.cif').is_file()
    report('Protenix common cache', has_common, str(common),
           'place the Protenix CCD/cluster data (components.cif, '
           'components.cif.rdkit_mol.pkl, clusters-by-entity-40.txt) there, '
           'or set PROTENIX_COMMON_CACHE_DIR')

    module_cache = Path(str(cfg.get('module_cache') or ''))
    report('module cache (optional)', module_cache.is_dir(), str(module_cache),
           f'mkdir -p {module_cache} for ~3x faster repeat scoring '
           '(set PROTENIX_MODULE_CACHE_DIR to relocate)')

    msa_cache = Path(cfg['msa_cache'])
    n_msa = len(list(msa_cache.glob('msa_*.a3m'))) if msa_cache.is_dir() else 0
    report('shared MSA cache (optional)', True,
           f'{msa_cache}: {n_msa} cached MSAs' if msa_cache.is_dir()
           else f'{msa_cache}: missing (will be created on first fetch)',
           f'mkdir -p {msa_cache} or set BOLTZ_MSA_CACHE_DIR')

    server = cfg.get('msa_server_url', '')
    server_up = False
    if server:
        try:
            import urllib.error
            import urllib.request
            urllib.request.urlopen(f'{server.rstrip("/")}/', timeout=5)
            server_up = True
        except urllib.error.HTTPError:
            server_up = True  # any HTTP reply means the server is listening
        except Exception:
            server_up = False
    report('MSA server (optional)', True,
           f'{server}: reachable' if server_up else
           (f'{server}: unreachable' if server else 'not configured '
            '(receptor MSAs must come from the shared cache)'),
           'run V-Bio\'s ColabFold MSA server (deploy/docker/'
           'DOCKER_CAP_COLABFOLD_SERVER.compose.yml) and set MSA_SERVER_URL')

    try:
        import subprocess as _sp
        gpus = _sp.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        n_gpu = len([l for l in gpus.stdout.splitlines() if l.startswith('GPU ')])
    except Exception:
        n_gpu = 0
    report('NVIDIA GPU', n_gpu > 0,
           f'{n_gpu} device(s); scoring will use gpu {gpu}',
           'a 24 GB card (RTX 4090 class) is recommended')

    print('\n' + ('ready — protenix2dock scoring should work'
                  if ok else 'NOT ready — resolve the FAIL items above'))
    return ok


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='protenix2dock scoring client utilities',
        prog='python -m utils.protenix2dock_client')
    parser.add_argument('--check', action='store_true',
                        help='verify docker, the runtime image, the V-Bio '
                             'checkout, weights, caches and MSA settings')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if args.check:
        raise SystemExit(0 if check_setup(gpu=args.gpu) else 1)
    parser.print_help()
