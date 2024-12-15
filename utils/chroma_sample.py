import torch
from chroma import Chroma, Protein, conditioners
from .pdb_processing import fix_pdb

def generate_mask(S, L_receptor, L_complex, device):
    """
    Generate amino acid mask for the protein.
    """
    mask_aa = torch.ones((1, L_complex, 20), device=device)
    allowed_aas = torch.eye(20, device=device)[S[0, :L_receptor]]
    mask_aa[0, :L_receptor, :] = allowed_aas
    return mask_aa

def create_conditioner(protein, chroma, device):
    """
    Create a composed conditioner for protein sampling.
    """

    conditioner_struc_R = conditioners.SubstructureConditioner(
        protein,
        backbone_model=chroma.backbone_network,
        selection='namesel receptor'
    ).to(device)

    return conditioners.ComposedConditioner([conditioner_struc_R])

def binder_sample(input_pdb, len_binder, output_pdb, len_chains, device='cuda:0', langevin_factor=2, sde_func='langevin'):
    """
    Generate binder for a given receptor structure.
    """
    chroma = Chroma()
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
    conditioner = create_conditioner(protein, chroma, device)

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