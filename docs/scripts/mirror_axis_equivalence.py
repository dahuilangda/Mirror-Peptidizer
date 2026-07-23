"""Coordinate-mirror axis equivalence test.

Does reflecting a structure along the **Y** or **Z** axis — instead of the
**X** axis used by the production pipeline — change the results? This script
provides the *empirical* answer that complements the group-theoretic argument
in the manuscript:

* reflecting along any single Cartesian axis is an orientation-reversing
  isometry (determinant −1), so each operation converts an L-structure into the
  *same* D-enantiomer;
* the product of two such reflections is a proper rigid-body rotation
  (determinant +1), so the X-, Y- and Z-mirrored structures are mutually
  superposable;
* therefore the choice of mirror axis cannot change ProteinMPNN's input.

For a panel of L-protein / D-peptide complexes we (i) reflect each structure
along X, Y and Z independently, (ii) pairwise Kabsch-superpose the three
mirrors and report the backbone RMSD (expected ≈ 0, machine precision), and
(iii) verify the backbone dihedrals φ/ψ are identical across the three mirrors
(up to sign, as for any enantiomerization).

This is a *pure-geometry* test — it does not load ProteinMPNN or torch — so it
runs on any machine with biopython/numpy/matplotlib.

Outputs (next to this script):
    mirror_axis_equivalence.csv     per-structure pairwise RMSD + max |Δφ|/|Δψ|
    mirror_axis_equivalence_summary.csv   axis-of-reflection invariants
    ../figures/fig_mirror_axis_equivalence.{png,svg,pdf}
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.vectors import calc_dihedral
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
FIGDIR = _PROJECT_ROOT / "benchmark" / "figures"

# backbone-only, capped/staple/ligand-stripped L-receptor / D-peptide complexes
# produced by run_recovery.py. Each has a receptor chain A and a peptide chain B.
TEST_STRUCTURES = {
    "3LNJ (MDM2 / D-pep, 11aa)": _SCRIPT_DIR / "3LNJ_AB_clean.pdb",
    "8F10 (MDM2 / stapled D-pep, 15aa)": _SCRIPT_DIR / "8F10_AB_clean.pdb",
    "3HTN (native L-control, 139aa)": _SCRIPT_DIR / "3HTN_control_clean.pdb",
}

AXES = ("x", "y", "z")
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

PALETTE = {
    "blue_main": "#0F4D92",
    "orange": "#E28E2C",
    "green": "#2E8B57",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
}


def apply_nature_style(font_size: int = 7) -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.7
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.titlesize"] = font_size + 1
    plt.rcParams["axes.labelsize"] = font_size + 1
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size
    plt.rcParams["legend.fontsize"] = font_size
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["xtick.major.width"] = 0.7
    plt.rcParams["ytick.major.width"] = 0.7
    plt.rcParams["xtick.major.size"] = 2.5
    plt.rcParams["ytick.major.size"] = 2.5


def reflect_coords(coords: np.ndarray, axis: str) -> np.ndarray:
    """Reflect atomic coordinates along a single Cartesian axis (determinant −1)."""
    out = coords.astype(float).copy()
    out[:, AXIS_INDEX[axis]] *= -1.0
    return out


def _backbone_atom_coords(path: Path) -> dict:
    """Return {chain_id: ndarray(N, 3)} of N, CA, C, O backbone atoms per chain."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, path)
    per_chain: dict[str, list[tuple]] = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                for atom_name in ("N", "CA", "C", "O"):
                    if atom_name in residue:
                        per_chain.setdefault(chain.id, []).append(
                            (chain.id, residue, atom_name, residue[atom_name].get_coord())
                        )
    return per_chain


def kabsch_rmsd(fixed: np.ndarray, moved: np.ndarray) -> float:
    """Minimum backbone RMSD between two same-N coordinate sets via Kabsch.

    Uses Bio.PDB.Superimposer for the SVD alignment so the result matches the
    convention used elsewhere in the benchmark scripts.
    """
    assert fixed.shape == moved.shape
    sup = Superimposer()
    # Superimposer expects lists of Bio.PDB atoms; emulate with dummy atoms by
    # building lightweight coordinate-only vectors through numpy Kabsch instead.
    fixed_c = fixed - fixed.mean(axis=0)
    moved_c = moved - moved.mean(axis=0)
    h = moved_c.T @ fixed_c
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    s = np.eye(3)
    s[2, 2] = d
    rot = vt.T @ s @ u.T
    moved_aligned = moved_c @ rot.T
    rmsd = float(np.sqrt(np.mean(np.sum((moved_aligned - fixed_c) ** 2, axis=1))))
    return rmsd


def _chain_phi_psi(path: Path, axis: str | None) -> list[tuple[float | None, float | None]]:
    """φ/ψ (degrees) for the peptide (last) chain; reflect coords first if axis given."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, path)
    chains = [c for c in structure.get_chains()]
    chain = sorted(chains, key=lambda c: c.id)[-1]
    if axis is not None:
        for atom in structure.get_atoms():
            v = atom.get_coord().copy()
            v[AXIS_INDEX[axis]] *= -1.0
            atom.set_coord(v)
    residues = [
        r for r in chain
        if r.id[0] == " " and all(a in r for a in ("N", "CA", "C"))
    ]
    angles: list[tuple[float | None, float | None]] = []
    for idx, residue in enumerate(residues):
        phi = psi = None
        if idx > 0:
            prev = residues[idx - 1]
            phi = float(np.degrees(calc_dihedral(
                prev["C"].get_vector(), residue["N"].get_vector(),
                residue["CA"].get_vector(), residue["C"].get_vector(),
            )))
        if idx < len(residues) - 1:
            nxt = residues[idx + 1]
            psi = float(np.degrees(calc_dihedral(
                residue["N"].get_vector(), residue["CA"].get_vector(),
                residue["C"].get_vector(), nxt["N"].get_vector(),
            )))
        angles.append((phi, psi))
    return angles


def _max_abs_diff(a: list[float | None], b: list[float | None]) -> float:
    """Max absolute difference ignoring None entries (sign matters here)."""
    vals = [abs(x - y) for x, y in zip(a, b) if x is not None and y is not None]
    return float(max(vals)) if vals else float("nan")


def run() -> None:
    rows = []
    summary_rows = []
    rama_all = []  # for the φ/ψ overlay panel

    for label, path in TEST_STRUCTURES.items():
        if not path.exists():
            print(f"  [skip] {label}: {path} not found")
            continue

        per_chain = _backbone_atom_coords(path)
        if not per_chain:
            print(f"  [skip] {label}: no backbone atoms parsed")
            continue
        coords_full = np.vstack([np.vstack([np.asarray(a[3]) for a in v]) for v in per_chain.values()])

        # three independent single-axis reflections
        mirrors = {axis: reflect_coords(coords_full, axis) for axis in AXES}

        # pairwise Kabsch RMSD (should be ~0; the two reflections differ by a rotation)
        for i in range(len(AXES)):
            for j in range(i + 1, len(AXES)):
                a, b = AXES[i], AXES[j]
                rmsd = kabsch_rmsd(mirrors[a], mirrors[b])
                rows.append({"structure": label, "pair": f"{a}-{b}", "kabsch_rmsd_A": rmsd})

        # φ/ψ of the peptide chain under each reflection (vs. the unmirrored native)
        phi_psi = {axis: _chain_phi_psi(path, axis) for axis in AXES}
        native_pp = _chain_phi_psi(path, None)
        for axis in AXES:
            phi_d = [p[0] for p in phi_psi[axis]]
            psi_d = [p[1] for p in phi_psi[axis]]
            phi_l = [p[0] for p in native_pp]
            psi_l = [p[1] for p in native_pp]
            # φ/ψ must flip sign under reflection: |φ_L + φ_D| ≈ 0, |ψ_L + ψ_D| ≈ 0
            max_phi_sign = max(
                (abs(l + d) for l, d in zip(phi_l, phi_d) if l is not None and d is not None),
                default=float("nan"),
            )
            max_psi_sign = max(
                (abs(l + d) for l, d in zip(psi_l, psi_d) if l is not None and d is not None),
                default=float("nan"),
            )
            # across axes the φ/ψ sets must be identical
            other = [a for a in AXES if a != axis]
            dphi = _max_abs_diff(phi_d, [p[0] for p in phi_psi[other[0]]])
            dpsi = _max_abs_diff(psi_d, [p[1] for p in phi_psi[other[0]]])
            summary_rows.append({
                "structure": label,
                "axis": axis,
                "n_residues": len(phi_d),
                "max_abs_phi_L_plus_phi_D_deg": max_phi_sign,
                "max_abs_psi_L_plus_psi_D_deg": max_psi_sign,
                "max_abs_dphi_vs_other_axis_deg": dphi,
                "max_abs_dpsi_vs_other_axis_deg": dpsi,
            })
            for (phi, psi) in phi_psi[axis]:
                if phi is not None and psi is not None:
                    rama_all.append({"structure": label, "axis": axis, "phi": phi, "psi": psi})

        rmsd_vals = [r["kabsch_rmsd_A"] for r in rows if r["structure"] == label]
        print(f"  {label}: pairwise Kabsch RMSD "
              f"min={min(rmsd_vals):.2e} max={max(rmsd_vals):.2e} Å")

    df = pd.DataFrame(rows)
    df.to_csv(_SCRIPT_DIR / "mirror_axis_equivalence.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(
        _SCRIPT_DIR / "mirror_axis_equivalence_summary.csv", index=False
    )
    rama_df = pd.DataFrame(rama_all)
    rama_df.to_csv(_SCRIPT_DIR / "mirror_axis_equivalence_rama.csv", index=False)

    _plot(df, rama_df)


def _plot(df: pd.DataFrame, rama_df: pd.DataFrame) -> None:
    apply_nature_style()
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.2))

    # φ/ψ Ramachandran overlay under the three single-axis reflections.
    # The pairwise Kabsch RMSD between the X/Y/Z mirrors is at machine precision
    # (see mirror_axis_equivalence.csv / the table in testing_and_validation.md),
    # so it carries no visual information and is reported numerically rather than
    # plotted.
    markers = {"x": "o", "y": "s", "z": "^"}
    colors = [PALETTE["blue_main"], PALETTE["orange"], PALETTE["green"]]
    for axis in ["x", "y", "z"]:
        sub = rama_df[rama_df["axis"] == axis]
        ax.scatter(sub["phi"], sub["psi"], s=6, marker=markers[axis],
                   facecolors="none", edgecolors=colors[["x", "y", "z"].index(axis)],
                   linewidths=0.6, label=f"reflect {axis}", alpha=0.8)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_xlabel("φ (°)")
    ax.set_ylabel("ψ (°)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3,
              handletextpad=0.3, columnspacing=0.8)

    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.18, right=0.97)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIGDIR / f"fig_mirror_axis_equivalence.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    run()
