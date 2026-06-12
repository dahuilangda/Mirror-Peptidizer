"""Geometry-based filters for generated peptide poses."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable


Coord = tuple[float, float, float]


DEFAULT_SURFACE_FILTER = {
    "radial_min": 0.60,
    "outward_nearby_max": 0.25,
    "occupied_octants_max": 5,
    "min_binder_atoms_within_contact": 20,
    "nearby_distance": 8.0,
    "contact_distance": 4.5,
}


def atom_coords(
    pdb_path: str | Path,
    chains: Iterable[str] | None = None,
    exclude_chains: Iterable[str] | None = None,
) -> list[Coord]:
    """Read non-hydrogen atom coordinates from selected PDB chains."""
    selected = set(chains) if chains is not None else None
    excluded = set(exclude_chains) if exclude_chains is not None else set()
    coords: list[Coord] = []
    with Path(pdb_path).open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            chain = line[21].strip() or "_"
            if selected is not None and chain not in selected:
                continue
            if chain in excluded:
                continue
            elem = line[76:78].strip() or line[12:16].strip()[0]
            if elem.upper() == "H":
                continue
            coords.append(
                (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            )
    return coords


def _mean_coord(coords: list[Coord]) -> Coord:
    return tuple(sum(coord[i] for coord in coords) / len(coords) for i in range(3))


def _sub(a: Coord, b: Coord) -> Coord:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Coord, b: Coord) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _dist(a: Coord, b: Coord) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _unit(v: Coord) -> Coord:
    length = math.sqrt(_dot(v, v))
    if length == 0:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def surface_pose_metrics(
    pdb_path: str | Path,
    binder_chain: str = "B",
    receptor_chains: Iterable[str] | None = None,
    nearby_distance: float = DEFAULT_SURFACE_FILTER["nearby_distance"],
    contact_distance: float = DEFAULT_SURFACE_FILTER["contact_distance"],
) -> dict[str, float]:
    """Compute surface-placement metrics for a receptor-binder complex.

    If ``receptor_chains`` is omitted, all chains except ``binder_chain`` are
    treated as receptor chains.
    """
    binder = atom_coords(pdb_path, chains=[binder_chain])
    receptor = atom_coords(
        pdb_path,
        chains=receptor_chains,
        exclude_chains=None if receptor_chains is not None else [binder_chain],
    )
    if not receptor or not binder:
        raise ValueError(
            f"{pdb_path} must contain receptor atoms and binder chain {binder_chain}"
        )

    receptor_center = _mean_coord(receptor)
    binder_center = _mean_coord(binder)
    outward = _unit(_sub(binder_center, receptor_center))

    receptor_radii = sorted(_dist(atom, receptor_center) for atom in receptor)
    binder_radius = _dist(binder_center, receptor_center)
    radial_percentile = sum(r <= binder_radius for r in receptor_radii) / len(
        receptor_radii
    )

    nearby = 0
    outward_nearby = 0
    octants: set[tuple[bool, bool, bool]] = set()
    for atom in receptor:
        vector = _sub(atom, binder_center)
        if _dist(atom, binder_center) < nearby_distance:
            nearby += 1
            if _dot(vector, outward) > 0:
                outward_nearby += 1
            octants.add((vector[0] >= 0, vector[1] >= 0, vector[2] >= 0))

    nearest = [min(_dist(b_atom, r_atom) for r_atom in receptor) for b_atom in binder]
    nearest_sorted = sorted(nearest)
    return {
        "radial_percentile": radial_percentile,
        "nearby_receptor_atoms": float(nearby),
        "outward_nearby_fraction": outward_nearby / nearby if nearby else 0.0,
        "occupied_octants": float(len(octants)),
        "min_receptor_distance": min(nearest),
        "median_receptor_distance": nearest_sorted[len(nearest_sorted) // 2],
        "binder_atoms_within_contact": float(
            sum(d < contact_distance for d in nearest)
        ),
    }


def passes_surface_pose_filter(
    metrics: dict[str, float],
    radial_min: float = DEFAULT_SURFACE_FILTER["radial_min"],
    outward_nearby_max: float = DEFAULT_SURFACE_FILTER["outward_nearby_max"],
    occupied_octants_max: int = DEFAULT_SURFACE_FILTER["occupied_octants_max"],
    min_binder_atoms_within_contact: int = DEFAULT_SURFACE_FILTER[
        "min_binder_atoms_within_contact"
    ],
) -> bool:
    """Return True when a pose is surface-like by the benchmark criteria."""
    return (
        metrics["radial_percentile"] >= radial_min
        and metrics["outward_nearby_fraction"] <= outward_nearby_max
        and metrics["occupied_octants"] <= occupied_octants_max
        and metrics["binder_atoms_within_contact"] >= min_binder_atoms_within_contact
    )


def format_surface_pose_metrics(metrics: dict[str, float]) -> str:
    """Compact log string for surface-pose filter decisions."""
    return (
        f"rad={metrics['radial_percentile']:.2f} "
        f"out={metrics['outward_nearby_fraction']:.2f} "
        f"oct={metrics['occupied_octants']:.0f} "
        f"contact={metrics['binder_atoms_within_contact']:.0f} "
        f"med={metrics['median_receptor_distance']:.2f}"
    )
