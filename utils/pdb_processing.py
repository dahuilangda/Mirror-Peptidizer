from Bio.PDB import PDBParser, PDBIO
from pdbfixer import PDBFixer
from openmm.app import PDBFile, Simulation, ForceField, NoCutoff, HBonds
from openmm import LangevinIntegrator, Vec3
from openmm.unit import dalton, kelvin, nanometer, picosecond, picoseconds

def ld_convert(input_pdb, output_pdb):
    parser = PDBParser()
    structure = parser.get_structure("L_protein", input_pdb)

    for atom in structure.get_atoms():
        coord = atom.get_coord()
        atom.set_coord([-coord[0], coord[1], coord[2]])

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)
    return output_pdb

def one_to_three(seq):
    '''
    Convert 1-letter amino acid code to 3-letter amino acid code
    '''
    aa_dict = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
        'X': 'UNK'
    }
    return [aa_dict.get(residue, 'UNK') for residue in seq]

BACKBONE_ATOMS = {"N", "CA", "C", "O"}


def _atom_key(atom):
    residue = atom.residue
    chain = residue.chain
    insertion_code = getattr(residue, 'insertionCode', '')
    return (chain.id, residue.id, insertion_code, atom.name)


def _positions_as_vec3_list(positions):
    return [
        pos.value_in_unit(nanometer) if hasattr(pos, 'value_in_unit') else pos
        for pos in positions
    ]


def _backbone_positions_from_pdb(pdb_file):
    pdb = PDBFile(pdb_file)
    return {
        _atom_key(atom): pos.value_in_unit(nanometer)
        for atom, pos in zip(pdb.topology.atoms(), pdb.positions)
        if atom.name in BACKBONE_ATOMS
    }


def seq_to_pdb(seq, pdb, output_pdb, design_chain='B', minimize=True, remove_hydrogens=True, fix_backbone=True):
    aa_list = one_to_three(seq)
    new_line = []
    chain_residue_num = None
    resid = -1  # Start from -1 because resid is incremented at the first new residue
    with open(pdb, 'r') as f:
        lines = f.readlines()

        # Replace the amino acid sequence of the design chain
        for line in lines:
            if len(line) > 21:
                if line.startswith('ATOM') and line[21] == design_chain:
                    atom_name = line[12:16].strip()
                    if atom_name in ['N', 'CA', 'C', 'O']:
                        if chain_residue_num != line[22:26]:
                            chain_residue_num = line[22:26]
                            resid += 1

                        if resid < len(aa_list):
                            # Replace the residue name
                            line = line[:17] + aa_list[resid].ljust(3) + line[20:]
                            new_line.append(line)
                        else:
                            print(f"Warning: More residues in chain {design_chain} than in provided sequence.")
                            break
                    else:
                        continue  # Skip side-chain atoms
                elif line.startswith('ATOM') and line[21] != design_chain:
                    # Keep other chains' atoms
                    new_line.append(line)
                elif line.startswith('TER'):
                    new_line.append(line)
                else:
                    continue
            else:
                new_line.append(line)

    with open(output_pdb, 'w') as f:
        f.writelines(new_line)

    fix_pdb(
        output_pdb,
        output_pdb,
        minimize=minimize,
        remove_hydrogens=remove_hydrogens,
        fix_backbone=fix_backbone,
    )

def get_pdb_chains(pdb_file_path):
    chains = set()
    with open(pdb_file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = line[21].strip()
                if chain_id:
                    chains.add(chain_id)
    #return list(chains)
    return sorted(list(chains))

def _remove_hydrogens(input_pdb, output_pdb):
    with open(input_pdb, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            if line[76:78].strip() != 'H':
                new_lines.append(line)

    with open(output_pdb, 'w') as f:
        f.writelines(new_lines)

def fix_pdb(
    input_pdb,
    output_pdb,
    minimize=True,
    remove_hydrogens=True,
    fix_backbone=True,
    restrain_backbone=None,
):
    if restrain_backbone is not None:
        fix_backbone = restrain_backbone

    original_backbone_positions = (
        _backbone_positions_from_pdb(input_pdb) if fix_backbone else {}
    )

    # Load the PDB file
    fixer = PDBFixer(filename=input_pdb)

    # Find and add missing residues and atoms
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens()

    backbone_positions = {}
    if fix_backbone:
        for atom, pos in zip(fixer.topology.atoms(), fixer.positions):
            if atom.name in BACKBONE_ATOMS:
                current_pos = pos.value_in_unit(nanometer)
                backbone_positions[atom.index] = original_backbone_positions.get(_atom_key(atom), current_pos)
        fixed_positions = _positions_as_vec3_list(fixer.positions)
        for atom_index, pos in backbone_positions.items():
            fixed_positions[atom_index] = pos
        fixer.positions = fixed_positions * nanometer

    if minimize:
        
        # Define the force field
        forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

        # Create the OpenMM system
        system = forcefield.createSystem(
            fixer.topology,
            constraints=None if fix_backbone else HBonds,
            nonbondedMethod=NoCutoff
        )

        if fix_backbone:
            for atom in fixer.topology.atoms():
                if atom.index in backbone_positions:
                    system.setParticleMass(atom.index, 0.0 * dalton)

        # Create an integrator
        integrator = LangevinIntegrator(
            300*kelvin,  # Temperature
            1/picosecond,  # Friction coefficient
            0.002*picoseconds  # Time step
        )

        # Set up the simulation
        simulation = Simulation(fixer.topology, system, integrator)

        # Set the initial positions
        simulation.context.setPositions(fixer.positions)

        # Minimize the energy
        simulation.minimizeEnergy(maxIterations=1000)

        # Get the minimized positions
        positions = _positions_as_vec3_list(
            simulation.context.getState(getPositions=True).getPositions()
        )
    else:
        positions = _positions_as_vec3_list(fixer.positions)

    if fix_backbone:
        for atom_index, pos in backbone_positions.items():
            positions[atom_index] = pos

    # Save the (optionally minimized) structure
    with open(output_pdb, 'w') as f:
        PDBFile.writeFile(fixer.topology, positions * nanometer, f)

    if remove_hydrogens:
        _remove_hydrogens(output_pdb, output_pdb)

if __name__ == '__main__':
    chains = get_pdb_chains('data/4LWV.pdb')
    print("Chains in the PDB file:", chains)

    seq_to_pdb(
        seq='EELARKALERI',
        pdb='data/4LWV.pdb',
        output_pdb='output_dir/Binder_D_pose_1_seq_EELARKALERI_fixed.pdb',
        design_chain='B'
    )
