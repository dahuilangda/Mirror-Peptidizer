import os
import torch
from chroma import Chroma, Protein, conditioners
from .pdb_processing import fix_pdb


def _get_chroma_weights():
    """Resolve Chroma weight paths from .env or local chroma_weights/ directory."""
    from dotenv import load_dotenv
    # load the repo-root .env explicitly so resolution does not depend on the
    # caller's working directory (e.g. Jupyter kernels started in examples/)
    load_dotenv(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

    backbone = os.getenv('CHROMA_WEIGHTS_BACKBONE')
    design = os.getenv('CHROMA_WEIGHTS_DESIGN')
    if backbone and design:
        return backbone, design

    default_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'chroma_weights')
    weights_dir = os.getenv('CHROMA_WEIGHTS_DIR', default_dir)

    def _scan_weights(directory):
        backbone_path = None
        design_path = None
        if os.path.isdir(directory):
            for subdir in os.listdir(directory):
                full = os.path.join(directory, subdir, 'weights.pt')
                if os.path.isfile(full):
                    if os.path.getsize(full) > 60_000_000:
                        backbone_path = full
                    else:
                        design_path = full
        return backbone_path, design_path

    backbone_path, design_path = _scan_weights(weights_dir)
    if not (backbone_path and design_path) and weights_dir != default_dir:
        # configured directory holds no weights (stale/typo'd value): fall
        # back to the repo's own chroma_weights/ instead of downloading
        backbone_path, design_path = _scan_weights(default_dir)
    return backbone_path, design_path


def generate_mask(S, L_receptor, L_complex, device):
    """
    Generate amino acid mask for the protein.
    """
    mask_aa = torch.ones((1, L_complex, 20), device=device)
    allowed_aas = torch.eye(20, device=device)[S[0, :L_receptor]]
    mask_aa[0, :L_receptor, :] = allowed_aas
    return mask_aa

def create_conditioner(protein, chroma, weight=3.0, device='cuda:0'):
    """
    Create a composed conditioner for protein sampling.
    """

    conditioner_struc_R = conditioners.SubstructureConditioner(
        protein,
        backbone_model=chroma.backbone_network,
        selection='namesel receptor',
        weight=weight,
    ).to(device)

    return conditioners.ComposedConditioner([conditioner_struc_R])

CHROMA_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'chroma_weights')


def _get_local_weights():
    """Resolve local Chroma weight paths from chroma_weights/ directory."""
    backbone_path = None
    design_path = None
    for subdir in os.listdir(CHROMA_WEIGHTS_DIR):
        full = os.path.join(CHROMA_WEIGHTS_DIR, subdir, 'weights.pt')
        if os.path.isfile(full):
            size = os.path.getsize(full)
            # backbone ~74MB, design ~55MB
            if size > 60_000_000:
                backbone_path = full
            else:
                design_path = full
    return backbone_path, design_path


def binder_sample(input_pdb, len_binder, output_pdb, len_chains, device='cuda:0', weight=1.0, langevin_factor=2, sde_func='langevin'):
    """
    Generate binder for a given receptor structure.
    """
    backbone_w, design_w = _get_chroma_weights()
    if backbone_w and design_w:
        chroma = Chroma(weights_backbone=backbone_w, weights_design=design_w, device=str(device))
    else:
        chroma = Chroma(device=str(device))
    protein = Protein(input_pdb, device=device)

    # Convert protein to X, C, S representation
    X, C, S = protein.to_XCS()

    # Extend the protein with binder segment
    with torch.no_grad():
        X_new = torch.cat([X, torch.zeros(1, len_binder, 4, 3, device=device)], dim=1).clone()
        C_new = torch.cat([C, torch.full((1, len_binder, ), len_binder, device=device)], dim=1).clone()
        S_new = torch.cat([S, torch.full((1, len_binder, ), 0, device=device)], dim=1).clone()

    # Update the protein with new data
    protein = Protein(X_new, C_new, S_new, device=device)
    X, C, S = protein.to_XCS()

    # Determine lengths of receptor, binder, and complex
    L_binder = (C == len_chains + 1).sum().item()
    L_receptor = (C != len_chains + 1).sum().item()
    L_complex = L_binder + L_receptor

    # Generate mask for amino acid sequence design
    mask_aa = generate_mask(S, L_receptor, L_complex, device)

    # Define receptor residues to keep
    residues_to_keep = [i for i in range(L_receptor)]
    protein.sys.save_selection(gti=residues_to_keep, selname="receptor")

    # Create conditioner
    conditioner = create_conditioner(protein, chroma, weight, device)

    # Perform sampling to generate binder
    protein = chroma.sample(
        protein_init=protein,
        conditioner=conditioner,
        design_selection=mask_aa,
        langevin_factor=langevin_factor,
        langevin_isothermal=True,
        sde_func=sde_func,
        full_output=False,
    )

    # Save the generated binder structure
    protein.to(output_pdb)

    fix_pdb(output_pdb, output_pdb)