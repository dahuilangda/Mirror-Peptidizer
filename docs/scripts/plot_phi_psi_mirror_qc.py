"""Plot Ramachandran-style backbone mirror QC for paired L/D peptide poses."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.vectors import calc_dihedral
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmark"
FIGDIR = BENCH / "figures"
FIGDATA = BENCH / "figure_data"

TARGET_DIRS = {
    "PDL1": BENCH / "PDL1" / "D_peptide",
    "MDM2": BENCH / "MDM2" / "D_peptide",
    "IL23R": BENCH / "IL23R" / "D_peptide",
    "TNFalpha": BENCH / "new_targets" / "designs" / "TNFalpha" / "D_peptide",
    "CXCR2": BENCH / "new_targets" / "designs" / "CXCR2" / "D_peptide",
    "CXCR4": BENCH / "new_targets" / "designs" / "CXCR4" / "D_peptide",
}

BENCHMARK_TARGETS = ["PDL1", "MDM2", "IL23R"]

PALETTE = {
    "blue_main": "#0F4D92",
    "orange": "#E28E2C",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
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


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.20,
        1.10,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _binder_chain(structure):
    chains = list(structure.get_chains())
    if not chains:
        raise ValueError("No chains found")
    return sorted(chains, key=lambda chain: chain.id)[-1]


def _chain_phi_psi(path: Path) -> list[tuple[float | None, float | None]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, path)
    chain = _binder_chain(structure)
    residues = [
        residue
        for residue in chain
        if residue.id[0] == " " and all(atom in residue for atom in ["N", "CA", "C"])
    ]
    angles: list[tuple[float | None, float | None]] = []
    for idx, residue in enumerate(residues):
        phi = None
        psi = None
        if idx > 0:
            prev = residues[idx - 1]
            phi = calc_dihedral(
                prev["C"].get_vector(),
                residue["N"].get_vector(),
                residue["CA"].get_vector(),
                residue["C"].get_vector(),
            )
        if idx < len(residues) - 1:
            nxt = residues[idx + 1]
            psi = calc_dihedral(
                residue["N"].get_vector(),
                residue["CA"].get_vector(),
                residue["C"].get_vector(),
                nxt["N"].get_vector(),
            )
        angles.append((phi, psi))
    return angles


def _paired_pose_files(base: Path) -> list[tuple[str, Path, Path]]:
    pairs = []
    for l_path in sorted(base.glob("len_*/Poses/Binder_L_pose_*.pdb")):
        d_path = l_path.with_name(l_path.name.replace("Binder_L_pose_", "Binder_D_pose_"))
        if d_path.exists():
            pose_key = f"{l_path.parents[1].name}:{l_path.stem.split('_')[-1]}"
            pairs.append((pose_key, l_path, d_path))
    return pairs


def collect_phi_psi() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rama_rows = []
    angle_rows = []
    summary_rows = []
    for target, base in TARGET_DIRS.items():
        pairs = _paired_pose_files(base)
        if not pairs:
            raise FileNotFoundError(f"No paired L/D pose files found under {base}")

        target_phi_errors: list[float] = []
        target_psi_errors: list[float] = []
        target_rama_rows = 0
        for pose_key, l_path, d_path in pairs:
            l_angles = _chain_phi_psi(l_path)
            d_angles = _chain_phi_psi(d_path)
            if len(l_angles) != len(d_angles):
                raise ValueError(f"Residue count mismatch: {l_path} vs {d_path}")

            for residue_idx, ((phi_l, psi_l), (phi_d, psi_d)) in enumerate(
                zip(l_angles, d_angles),
                start=1,
            ):
                phi_l_deg = float(np.degrees(phi_l)) if phi_l is not None else None
                psi_l_deg = float(np.degrees(psi_l)) if psi_l is not None else None
                phi_d_deg = float(np.degrees(phi_d)) if phi_d is not None else None
                psi_d_deg = float(np.degrees(psi_d)) if psi_d is not None else None

                phi_err = None
                psi_err = None
                if phi_l_deg is not None and phi_d_deg is not None:
                    phi_err = abs(phi_l_deg + phi_d_deg)
                    target_phi_errors.append(phi_err)
                    angle_rows.append(
                        {
                            "target": target,
                            "pose_key": pose_key,
                            "residue_index": residue_idx,
                            "angle": "phi",
                            "L_angle": phi_l_deg,
                            "D_angle": phi_d_deg,
                            "abs_L_plus_D": phi_err,
                        }
                    )
                if psi_l_deg is not None and psi_d_deg is not None:
                    psi_err = abs(psi_l_deg + psi_d_deg)
                    target_psi_errors.append(psi_err)
                    angle_rows.append(
                        {
                            "target": target,
                            "pose_key": pose_key,
                            "residue_index": residue_idx,
                            "angle": "psi",
                            "L_angle": psi_l_deg,
                            "D_angle": psi_d_deg,
                            "abs_L_plus_D": psi_err,
                        }
                    )

                if (
                    phi_l_deg is not None
                    and psi_l_deg is not None
                    and phi_d_deg is not None
                    and psi_d_deg is not None
                ):
                    target_rama_rows += 1
                    rama_rows.append(
                        {
                            "target": target,
                            "pose_key": pose_key,
                            "residue_index": residue_idx,
                            "phi_L": phi_l_deg,
                            "psi_L": psi_l_deg,
                            "phi_D": phi_d_deg,
                            "psi_D": psi_d_deg,
                            "phi_D_sign_inverted": -phi_d_deg,
                            "psi_D_sign_inverted": -psi_d_deg,
                            "abs_phi_L_plus_phi_D": phi_err,
                            "abs_psi_L_plus_psi_D": psi_err,
                        }
                    )

        summary_rows.append(
            {
                "target": target,
                "n_pose_pairs": len(pairs),
                "n_paired_phi": len(target_phi_errors),
                "n_paired_psi": len(target_psi_errors),
                "n_rama_coordinates": target_rama_rows,
                "max_abs_phi_L_plus_phi_D": max(target_phi_errors),
                "max_abs_psi_L_plus_psi_D": max(target_psi_errors),
            }
        )

    return pd.DataFrame(rama_rows), pd.DataFrame(angle_rows), pd.DataFrame(summary_rows)


def _mode_pose_files(target: str, mode: str) -> list[Path]:
    if mode == "L":
        return sorted((BENCH / target / "L_peptide").glob("len_*/Poses/Binder_L_pose_*.pdb"))
    if mode == "D":
        return sorted((BENCH / target / "D_peptide").glob("len_*/Poses/Binder_D_pose_*.pdb"))
    raise ValueError(f"Unknown mode: {mode}")


def collect_benchmark_ld_phi_psi() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect raw phi/psi distributions from benchmark direct-L and final-D designs."""
    rows = []
    summary_rows = []
    for target in BENCHMARK_TARGETS:
        for mode in ["L", "D"]:
            files = _mode_pose_files(target, mode)
            n_rama = 0
            for path in files:
                angles = _chain_phi_psi(path)
                pose_key = f"{path.parents[1].name}:{path.stem.split('_')[-1]}"
                for residue_idx, (phi, psi) in enumerate(angles, start=1):
                    if phi is None or psi is None:
                        continue
                    n_rama += 1
                    rows.append(
                        {
                            "target": target,
                            "mode": mode,
                            "pose_key": pose_key,
                            "residue_index": residue_idx,
                            "phi": float(np.degrees(phi)),
                            "psi": float(np.degrees(psi)),
                        }
                    )
            summary_rows.append(
                {
                    "target": target,
                    "mode": mode,
                    "n_pose_backbones": len(files),
                    "n_rama_coordinates": n_rama,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def _draw_reference_axes(ax) -> None:
    ax.axhline(0, color=PALETTE["neutral_light"], lw=0.6, zorder=0)
    ax.axvline(0, color=PALETTE["neutral_light"], lw=0.6, zorder=0)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-180, 0, 180])
    ax.set_yticks([-180, 0, 180])


def _density_contour(ax, x, y, color, label, linestyle="-", linewidth=1.3):
    hist, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=90,
        range=[[-180, 180], [-180, 180]],
    )
    z = gaussian_filter(hist.T, sigma=1.2)
    positive = z[z > 0]
    if len(positive) == 0:
        return None
    levels = np.quantile(positive, [0.70, 0.85, 0.94])
    levels = np.unique(levels)
    xc = (xedges[:-1] + xedges[1:]) / 2
    yc = (yedges[:-1] + yedges[1:]) / 2
    return ax.contour(
        xc,
        yc,
        z,
        levels=levels,
        colors=[color],
        linewidths=linewidth,
        linestyles=linestyle,
    )


def _summary_table(ax, summary: pd.DataFrame) -> None:
    ax.axis("off")
    display = summary.copy()
    display["target"] = display["target"].replace({"TNFalpha": "TNF-alpha"})
    rows = []
    for _, row in display.iterrows():
        rows.append(
            [
                row["target"],
                f"{int(row['n_pose_pairs'])}",
                f"{int(row['n_paired_phi'])}",
                f"{int(row['n_paired_psi'])}",
                f"{row['max_abs_phi_L_plus_phi_D']:.3f}",
                f"{row['max_abs_psi_L_plus_psi_D']:.3f}",
            ]
        )
    rows.append(
        [
            "All",
            f"{int(summary['n_pose_pairs'].sum())}",
            f"{int(summary['n_paired_phi'].sum())}",
            f"{int(summary['n_paired_psi'].sum())}",
            f"{summary['max_abs_phi_L_plus_phi_D'].max():.3f}",
            f"{summary['max_abs_psi_L_plus_psi_D'].max():.3f}",
        ]
    )
    table = ax.table(
        cellText=rows,
        colLabels=[
            "Target",
            "Pose\npairs",
            "Paired\nphi",
            "Paired\npsi",
            "Max phi\nerror (deg)",
            "Max psi\nerror (deg)",
        ],
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.17, 0.14, 0.15, 0.15, 0.19, 0.19],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.8)
    table.scale(1.0, 1.42)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#D7D7D7")
        cell.set_linewidth(0.55)
        if row_idx == 0:
            cell.set_facecolor(PALETTE["blue_main"])
            cell.set_text_props(color="white", weight="bold")
        elif row_idx == len(rows):
            cell.set_facecolor("#EEF2F6")
            cell.set_text_props(weight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#F7F7F7")
        else:
            cell.set_facecolor("white")
    add_panel_label(ax, "a")


def _scatter_cloud(ax, x, y, color, label, marker="o", face=True, alpha=0.20, size=5):
    facecolor = color if face else "none"
    edgecolor = "none" if face else color
    linewidth = 0.35 if not face else 0.0
    ax.scatter(
        x,
        y,
        s=size,
        marker=marker,
        alpha=alpha,
        linewidths=linewidth,
        facecolors=facecolor,
        edgecolors=edgecolor,
        color=color if face else None,
        rasterized=True,
        label=label,
    )


def _sample_points(df: pd.DataFrame, n: int = 4500) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=7)


def plot(rows: pd.DataFrame, summary: pd.DataFrame) -> None:
    apply_nature_style(font_size=7)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), sharex=True, sharey=True)
    plot_rows = _sample_points(rows, n=4500)

    ax_raw = axes[0]
    _draw_reference_axes(ax_raw)
    _scatter_cloud(
        ax_raw,
        plot_rows["phi_L"],
        plot_rows["psi_L"],
        PALETTE["blue_main"],
        "L backbone",
        alpha=0.26,
        size=5,
    )
    _scatter_cloud(
        ax_raw,
        plot_rows["phi_D"],
        plot_rows["psi_D"],
        PALETTE["orange"],
        "Final D backbone",
        alpha=0.24,
        size=5,
    )
    _density_contour(
        ax_raw,
        rows["phi_L"],
        rows["psi_L"],
        PALETTE["blue_main"],
        "L backbone",
        linewidth=1.2,
    )
    _density_contour(
        ax_raw,
        rows["phi_D"],
        rows["psi_D"],
        PALETTE["orange"],
        "Final D backbone",
        linewidth=1.2,
    )
    ax_raw.set_xlabel("φ (°)")
    ax_raw.set_ylabel("ψ (°)")
    add_panel_label(ax_raw, "a")

    ax_inv = axes[1]
    _draw_reference_axes(ax_inv)
    _scatter_cloud(
        ax_inv,
        plot_rows["phi_L"],
        plot_rows["psi_L"],
        PALETTE["blue_main"],
        "L backbone",
        alpha=0.25,
        size=5,
    )
    _scatter_cloud(
        ax_inv,
        plot_rows["phi_D_sign_inverted"],
        plot_rows["psi_D_sign_inverted"],
        PALETTE["orange"],
        "Sign-inverted D backbone",
        face=False,
        alpha=0.28,
        size=8,
    )
    _density_contour(
        ax_inv,
        rows["phi_L"],
        rows["psi_L"],
        PALETTE["blue_main"],
        "L backbone",
        linewidth=1.2,
    )
    _density_contour(
        ax_inv,
        rows["phi_D_sign_inverted"],
        rows["psi_D_sign_inverted"],
        PALETTE["neutral_dark"],
        "Sign-inverted D",
        linestyle="--",
        linewidth=1.1,
    )
    ax_inv.set_xlabel("φ (°)")
    ax_inv.set_ylabel("ψ (°)")
    add_panel_label(ax_inv, "b")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PALETTE["blue_main"],
            markersize=5,
            label="L backbone",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PALETTE["orange"],
            markersize=5,
            label="Final D backbone",
        ),
        plt.Line2D(
            [0],
            [0],
            color=PALETTE["neutral_dark"],
            lw=1.2,
            ls="--",
            label="Sign-inverted D backbone",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.92], w_pad=1.0)
    for ext in ["svg", "pdf"]:
        fig.savefig(FIGDIR / f"fig_phi_psi_mirror_qc.{ext}", bbox_inches="tight")
    fig.savefig(FIGDIR / "fig_phi_psi_mirror_qc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_ld(rows: pd.DataFrame) -> None:
    apply_nature_style(font_size=7)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), sharex=True, sharey=True)
    target_labels = {"PDL1": "PD-L1", "MDM2": "MDM2", "IL23R": "IL-23R"}

    for idx, target in enumerate(BENCHMARK_TARGETS):
        ax = axes[idx]
        _draw_reference_axes(ax)
        tdf = rows[rows["target"] == target]
        l_all = tdf[tdf["mode"] == "L"]
        d_all = tdf[tdf["mode"] == "D"]
        l_plot = _sample_points(l_all, n=1800)
        d_plot = _sample_points(d_all, n=1800)

        _scatter_cloud(
            ax,
            l_plot["phi"],
            l_plot["psi"],
            PALETTE["blue_main"],
            "Direct L-peptide",
            alpha=0.26,
            size=5,
        )
        _scatter_cloud(
            ax,
            d_plot["phi"],
            d_plot["psi"],
            PALETTE["orange"],
            "Mirror-derived D-peptide",
            alpha=0.24,
            size=5,
        )
        _density_contour(
            ax,
            l_all["phi"],
            l_all["psi"],
            PALETTE["blue_main"],
            "Direct L-peptide",
            linewidth=1.2,
        )
        _density_contour(
            ax,
            d_all["phi"],
            d_all["psi"],
            PALETTE["orange"],
            "Mirror-derived D-peptide",
            linewidth=1.2,
        )
        ax.set_xlabel(f"{target_labels[target]}  φ (°)")
        if idx == 0:
            ax.set_ylabel("ψ (°)")
        add_panel_label(ax, "abc"[idx])

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PALETTE["blue_main"],
            markersize=5,
            label="Direct L-peptide design",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PALETTE["orange"],
            markersize=5,
            label="Mirror-derived D-peptide design",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.90], w_pad=0.8)
    for ext in ["svg", "pdf"]:
        fig.savefig(FIGDIR / f"fig_phi_psi_mirror_qc.{ext}", bbox_inches="tight")
    fig.savefig(FIGDIR / "fig_phi_psi_mirror_qc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGDATA.mkdir(parents=True, exist_ok=True)
    rows, angle_rows, summary = collect_phi_psi()
    rows.to_csv(FIGDATA / "fig_phi_psi_mirror_qc_rama_coordinates.csv", index=False)
    angle_rows.to_csv(FIGDATA / "fig_phi_psi_mirror_qc_angles.csv", index=False)
    summary.to_csv(FIGDATA / "fig_phi_psi_mirror_qc_summary.csv", index=False)
    benchmark_rows, benchmark_summary = collect_benchmark_ld_phi_psi()
    benchmark_rows.to_csv(
        FIGDATA / "fig_phi_psi_benchmark_ld_rama_coordinates.csv", index=False
    )
    benchmark_summary.to_csv(
        FIGDATA / "fig_phi_psi_benchmark_ld_summary.csv", index=False
    )
    plot_benchmark_ld(benchmark_rows)
    print(summary.to_string(index=False))
    print(benchmark_summary.to_string(index=False))


if __name__ == "__main__":
    main()
