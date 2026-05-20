"""Landscape interface for the BO loop.

Reads fitness from CSV, writes proposed sequences to CSV.
Acts as the communication channel between the BO engine and external scoring.
"""
import os
from typing import Sequence

import numpy as np
import pandas as pd
from Bio import SeqIO


class EXPLandscape:
    """CSV-based fitness landscape for the BO optimization loop.

    The optimization algorithm proposes sequences by writing them to
    proposed_seqs.csv, then reads fitness values from fitness_csv.
    This decouples the algorithm from the actual measurement process.
    """

    def __init__(self, fitness_csv, wt_fasta, search_space=None, dir_path="."):
        self.name = "Measurements"
        self.dir_path = dir_path
        os.makedirs(dir_path, exist_ok=True)

        self._fitness_file = fitness_csv
        self._sequences = {}

        self._wt = str(SeqIO.read(wt_fasta, format="fasta").seq)
        if search_space is not None:
            combo = search_space.split(",")
            self._combo_protein_idxs = [int(idxs[1:]) for idxs in combo]
            self._combo_python_idxs = [idxs - 1 for idxs in self._combo_protein_idxs]
            temp_seq = [idxs[0] for idxs in combo]
            assert all(
                self._wt[self._combo_python_idxs[i]] == temp_seq[i]
                for i in range(len(temp_seq))
            )
        else:
            self._combo_protein_idxs = list(range(1, len(self._wt) + 1))
            self._combo_python_idxs = list(range(len(self._wt)))

    def _write_sequences(self, sequences):
        seqs = pd.DataFrame({"Variants": sequences})
        with open(f"{self.dir_path}/proposed_seqs.csv", "w") as f:
            seqs.to_csv(f, index=False)

    def _read_fitness(self):
        data = pd.read_csv(self._fitness_file)
        measured_seqs = set(data["Variants"])
        interset_seqs = measured_seqs.intersection(self._proposed_seqs)
        if len(interset_seqs) != len(self._proposed_seqs):
            print(
                f"There exist proposed sequences not measured or not in "
                f"fitness file: \"{self._fitness_file}\"."
            )
        self._sequences.update(zip(data["Variants"], data["Fitness"]))

    def get_fitness(self, sequences):
        """Get fitness values for proposed sequences via CSV exchange."""
        self._proposed_seqs = set(sequences)
        self._write_sequences(sequences)
        self._read_fitness()
        return np.array(
            [self._sequences.get(seq, np.nan) for seq in sequences],
            dtype=np.float32,
        )

    @property
    def wt(self):
        return self._wt

    @property
    def combo_protein_idxs(self):
        return self._combo_protein_idxs

    @property
    def combo_python_idxs(self):
        return self._combo_python_idxs
