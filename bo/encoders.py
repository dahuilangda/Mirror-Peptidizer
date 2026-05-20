"""Sequence encoders for Bayesian Optimization.

Supports one-hot encoding, physicochemical property encoding, and Boltz2
remote API embeddings.
"""
import hashlib
import os
import numpy as np
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


AAS = "ILVAGMFYWEDQNHCRKSTP"

# Per-residue physicochemical property vectors (6 dimensions).
#   0: Kyte-Doolittle hydrophobicity  (normalized to [-1, 1])
#   1: Side-chain volume              (Å³, normalized)
#   2: Positive charge propensity     (K, R, H)
#   3: Negative charge propensity     (D, E)
#   4: Aromaticity                    (F, W, Y)
#   5: Backbone flexibility           (G, P scored high)
_AA_PROPS = {
    'I': [ 0.85,  0.68, -0.10, -0.10,  0.00, -0.15],
    'L': [ 0.80,  0.62, -0.10, -0.10,  0.00, -0.15],
    'V': [ 0.78,  0.50, -0.10, -0.10,  0.00, -0.10],
    'A': [ 0.35,  0.15, -0.10, -0.10,  0.00, -0.05],
    'G': [-0.05, -0.25, -0.10, -0.10,  0.00,  0.85],
    'M': [ 0.50,  0.55, -0.10, -0.10,  0.00, -0.10],
    'F': [ 0.55,  0.72, -0.10, -0.10,  0.90, -0.15],
    'Y': [ 0.20,  0.78, -0.10, -0.10,  0.85, -0.10],
    'W': [ 0.25,  0.90, -0.10, -0.10,  0.95, -0.15],
    'E': [-0.65,  0.52, -0.10,  0.90,  0.00, -0.10],
    'D': [-0.60,  0.32, -0.10,  0.85,  0.00, -0.10],
    'Q': [-0.55,  0.52, -0.10, -0.10,  0.00, -0.05],
    'N': [-0.50,  0.32, -0.10, -0.10,  0.00, -0.05],
    'H': [-0.40,  0.48,  0.65, -0.10,  0.00, -0.10],
    'C': [ 0.45,  0.30, -0.10, -0.10,  0.00, -0.15],
    'R': [-0.75,  0.70,  0.95, -0.10,  0.00, -0.15],
    'K': [-0.70,  0.60,  0.90, -0.10,  0.00, -0.10],
    'S': [-0.20,  0.10, -0.10, -0.10,  0.00,  0.05],
    'T': [-0.15,  0.30, -0.10, -0.10,  0.00,  0.00],
    'P': [-0.30,  0.25, -0.10, -0.10,  0.00,  0.90],
}
_PROP_DIM = len(next(iter(_AA_PROPS.values())))


class OneHotEncoder:
    """Encode peptide sequences as one-hot vectors (L * 20 dims)."""

    def __init__(self, alphabet=AAS):
        self.alphabet = alphabet
        self.name = "OneHot"
        self.per_position_dim = len(alphabet)
        self.position_sampling = "ard"

    def encode(self, sequences):
        """Encode a list of sequences into flattened one-hot arrays.

        Args:
            sequences: list of sequence strings, all same length.

        Returns:
            np.ndarray of shape (N, L * len(alphabet)).
        """
        encodings = []
        for seq in sequences:
            one_hot = np.zeros((len(seq), len(self.alphabet)))
            for i, aa in enumerate(seq):
                one_hot[i, self.alphabet.index(aa)] = 1
            encodings.append(one_hot.flatten())
        return np.array(encodings, dtype=np.float32)


class PhysicochemicalEncoder:
    """Encode peptide sequences using amino-acid physicochemical properties.

    Each residue is represented by a 6-dim vector:
        hydrophobicity, volume, positive charge, negative charge,
        aromaticity, backbone flexibility.

    Output shape: (N, L * 6) — much lower dimension than one-hot (L * 20),
    which helps the GP generalize across similar residues and reduces
    overfitting on short peptide sequences.
    """

    def __init__(self):
        self.name = "Physicochemical"
        self.per_position_dim = _PROP_DIM
        self.position_sampling = "ard"

    def encode(self, sequences):
        encodings = []
        for seq in sequences:
            vec = np.array([_AA_PROPS[aa] for aa in seq], dtype=np.float32)
            encodings.append(vec.flatten())
        return np.array(encodings, dtype=np.float32)


class Boltz2Encoder:
    """Encode peptide sequences via Boltz2 Embedding API (384-dim pooled).

    API docs: https://github.com/dahuilangda/Boltz2Embedding

    Config:
        base_url: Boltz2 API server address
        api_token: authentication token
        timeout: max seconds to wait for job completion
        poll_interval: seconds between status checks
    """

    def __init__(
        self,
        base_url,
        api_token,
        timeout=600,
        poll_interval=5,
        cache_dir=None,
        batch_size=None,
        max_parallel_jobs=2,
    ):
        if not base_url or not api_token:
            raise ValueError("Boltz2Encoder requires base_url and api_token")
        self.base_url = base_url
        self.api_token = api_token
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.cache_dir = cache_dir
        self.batch_size = int(batch_size) if batch_size else None
        self.max_parallel_jobs = max(1, int(max_parallel_jobs))
        self.name = "Boltz2"
        self.position_sampling = "uniform"
        self._headers = {
            "X-API-Token": self.api_token,
            "Content-Type": "application/json",
        }
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, sequence):
        if not self.cache_dir:
            return None
        key = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.npy")

    def _read_cached(self, sequence):
        path = self._cache_path(sequence)
        if not path or not os.path.exists(path):
            return None
        return np.load(path).astype(np.float32)

    def _write_cached(self, sequence, embedding):
        path = self._cache_path(sequence)
        if not path:
            return
        tmp_path = f"{path}.tmp.npy"
        np.save(tmp_path, np.asarray(embedding, dtype=np.float32))
        os.replace(tmp_path, path)

    def _submit_job(self, sequences, job_name="bo_encode"):
        inputs = [
            {"id": f"seq-{i}", "type": "protein", "sequence": seq}
            for i, seq in enumerate(sequences)
        ]
        resp = requests.post(
            f"{self.base_url}/v1/embeddings/jobs",
            headers=self._headers,
            json={
                "job_name": job_name,
                "inputs": inputs,
                "include_pooled_embedding": True,
                "include_token_embedding": False,
                "include_pair_embedding": False,
            },
        )
        resp.raise_for_status()
        return resp.json()["job_id"]

    def _encode_uncached_batch(self, sequences):
        job_id = self._submit_job(sequences)
        self._poll_job(job_id)
        result = self._get_result(job_id)

        emb_map = {}
        for item in result["items"]:
            emb_map[item["id"]] = np.array(item["pooled_embedding"], dtype=np.float32)

        embeddings = []
        for i, seq in enumerate(sequences):
            embedding = emb_map[f"seq-{i}"]
            self._write_cached(seq, embedding)
            embeddings.append(embedding)
        return dict(zip(sequences, embeddings))

    def _sequence_batches(self, sequences):
        if not sequences:
            return []
        if self.batch_size is None:
            return [sequences]
        return [
            sequences[i:i + self.batch_size]
            for i in range(0, len(sequences), self.batch_size)
        ]

    def _poll_job(self, job_id):
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            resp = requests.get(
                f"{self.base_url}/v1/embeddings/jobs/{job_id}",
                headers={"X-API-Token": self.api_token},
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("ready"):
                if not data.get("successful", False):
                    meta = data.get("meta") or data.get("result") or "unknown error"
                    raise RuntimeError(f"Boltz2 job {job_id} failed: {meta}")
                return data
            time.sleep(self.poll_interval)
        raise TimeoutError(f"Boltz2 job {job_id} timed out after {self.timeout}s")

    def _get_result(self, job_id):
        resp = requests.get(
            f"{self.base_url}/v1/embeddings/jobs/{job_id}/result",
            headers={"X-API-Token": self.api_token},
        )
        resp.raise_for_status()
        return resp.json()

    def encode(self, sequences):
        """Encode sequences using Boltz2 pooled embeddings.

        Args:
            sequences: list of sequence strings.

        Returns:
            np.ndarray of shape (N, 384).
        """
        unique_sequences = list(dict.fromkeys(sequences))
        embeddings_by_seq = {}
        uncached = []
        for seq in unique_sequences:
            cached = self._read_cached(seq)
            if cached is None:
                uncached.append(seq)
            else:
                embeddings_by_seq[seq] = cached

        batches = self._sequence_batches(uncached)
        if len(batches) == 1:
            embeddings_by_seq.update(self._encode_uncached_batch(batches[0]))
        elif len(batches) > 1:
            workers = min(self.max_parallel_jobs, len(batches))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(self._encode_uncached_batch, batch)
                    for batch in batches
                ]
                for future in as_completed(futures):
                    embeddings_by_seq.update(future.result())

        return np.array(
            [embeddings_by_seq[seq] for seq in sequences],
            dtype=np.float32,
        )
