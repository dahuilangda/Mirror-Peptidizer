from Bio.PDB import PDBParser, PDBIO
from pdbfixer import PDBFixer
from openmm.app import PDBFile, Simulation, ForceField, NoCutoff, HBonds
from openmm import LangevinIntegrator, CustomExternalForce, Vec3
from openmm.unit import kelvin, picosecond, picoseconds

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

def seq_to_pdb(seq, pdb, output_pdb, design_chain='B', minimize=True, remove_hydrogens=True):
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

    fix_pdb(output_pdb, output_pdb, minimize=minimize, remove_hydrogens=remove_hydrogens)

def get_pdb_chains(pdb_file_path):
    chains = set()
    with open(pdb_file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = line[21].strip()
                if chain_id:
                    chains.add(chain_id)
    return list(chains)

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

def fix_pdb(input_pdb, output_pdb, minimize=True, remove_hydrogens=True, restrain_backbone=True):
    # Load the PDB file
    fixer = PDBFixer(filename=input_pdb)

    # Find and add missing residues and atoms
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens()

    if minimize:
        
        # Define the force field
        forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

        # Create the OpenMM system
        system = forcefield.createSystem(
            fixer.topology,
            constraints=HBonds,
            nonbondedMethod=NoCutoff
        )

        if restrain_backbone:
            restraint = CustomExternalForce("0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
            restraint.addPerParticleParameter("k")
            restraint.addPerParticleParameter("x0")
            restraint.addPerParticleParameter("y0")
            restraint.addPerParticleParameter("z0")

            # 对主链原子施加强约束
            for atom, pos in zip(fixer.topology.atoms(), fixer.positions):
                if atom.name in ["N", "CA", "C", "O"]:
                    restraint.addParticle(atom.index, [1000.0, pos.x, pos.y, pos.z])

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
        positions = simulation.context.getState(getPositions=True).getPositions()
    else:
        positions = fixer.positions

    # Save the (optionally minimized) structure
    with open(output_pdb, 'w') as f:
        PDBFile.writeFile(fixer.topology, positions, f)

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