"""Tier 2: Bayesian Optimization for D-peptide sequence optimization.

Takes initial designs from Tier 1 (run_design.py) and iteratively optimizes
the top sequences using Evolutionary BO or MCMC with a GP surrogate model.

Usage:
    python -m bo.run_bo \\
        --pose_pdb output/Poses/Binder_L_pose_1.pdb \\
        --receptor_pdb data/PDL1.pdb \\
        --output bo_output \\
        --rounds 8 --trials 10 --method BO

    # With Boltz2Embedding:
    python -m bo.run_bo \\
        --pose_pdb output/Poses/Binder_L_pose_1.pdb \\
        --receptor_pdb data/PDL1.pdb \\
        --output bo_output --embedding boltz2embedding \\
        --boltz2_url http://172.17.1.248:8000 \\
        --boltz2_token woaihuadong
"""
import os
import sys
import argparse

import numpy as np
import torch
import pandas as pd
from string import ascii_uppercase, ascii_lowercase

alphabet_list = list(ascii_uppercase + ascii_lowercase)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pdb_processing import ld_convert, seq_to_pdb, get_pdb_chains
from utils.protein_mpnn import protein_mpnn, resolve_checkpoint_path, score_complex

from .encoders import OneHotEncoder, PhysicochemicalEncoder, Boltz2Encoder, AAS
from .models import GPRegressor
from .explorers import BO_EVO, MCMC
from .landscape import EXPLandscape
from .scoring import swi, swi_weights, FuzzyScore


def build_mask(binder_seq, alphabet=AAS):
    """Build default mutation mask: all positions mutable, all AAs allowed."""
    mask = np.zeros((len(alphabet), len(binder_seq)), dtype=bool)
    return mask


def parse_mut_positions(mut_positions_str, binder_seq, alphabet=AAS):
    """Parse user-specified mutable positions like '1-5,8,10-12'.

    Returns a mask where True means 'blocked' (cannot mutate to).
    """
    mask = np.ones((len(alphabet), len(binder_seq)), dtype=bool)
    positions = set()
    for part in mut_positions_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            for p in range(int(start), int(end) + 1):
                positions.add(p - 1)
        else:
            positions.add(int(part) - 1)

    # Allow all AAs at specified positions, block mutations elsewhere
    mask = np.ones((len(alphabet), len(binder_seq)), dtype=bool)
    for pos in positions:
        mask[:, pos] = False
    return mask


def get_binder_sequence(pose_pdb):
    """Extract binder sequence from the last chain of a pose PDB."""
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1

    parser = PDBParser()
    structure = parser.get_structure('pose', pose_pdb)
    chains = sorted([c.id for c in structure.get_chains()])
    binder_chain = chains[-1]

    seq = ""
    for model in structure:
        for chain in model:
            if chain.id == binder_chain:
                for residue in chain:
                    seq += seq1(residue.get_resname())
                break
        break
    return seq, binder_chain


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    # 1. Extract binder sequence from pose PDB
    binder_seq, binder_chain = get_binder_sequence(args.pose_pdb)
    receptor_chains = get_pdb_chains(args.receptor_pdb)
    design_chain = alphabet_list[len(receptor_chains)]
    print(f"Binder sequence: {binder_seq} (chain {binder_chain})")
    print(f"Design chain: {design_chain}")
    print(f"ProteinMPNN checkpoint: {resolve_checkpoint_path(args.mpnn_checkpoint)}")

    # 2. Build mutation mask
    if args.mut_positions:
        mask = parse_mut_positions(args.mut_positions, binder_seq)
        print(f"Mutable positions: {args.mut_positions}")
    else:
        mask = build_mask(binder_seq)
        print("All positions mutable")

    # 3. Setup encoder
    if args.embedding == "boltz2embedding":
        boltz2_url = args.boltz2_url
        boltz2_token = args.boltz2_token
        if not boltz2_url or not boltz2_token:
            from dotenv import load_dotenv
            load_dotenv()
            boltz2_url = boltz2_url or os.getenv('BOLTZ2EMBEDDING_URL')
            boltz2_token = boltz2_token or os.getenv('BOLTZ2EMBEDDING_TOKEN')
        if not boltz2_url or not boltz2_token:
            raise ValueError(
                "Boltz2Embedding requires BOLTZ2EMBEDDING_URL and BOLTZ2EMBEDDING_TOKEN. "
                "Set them in .env or pass via --boltz2_url and --boltz2_token. "
                "See https://github.com/dahuilangda/Boltz2Embedding for setup."
            )
        encoder = Boltz2Encoder(base_url=boltz2_url, api_token=boltz2_token)
        print(f"Using Boltz2Embedding encoder ({boltz2_url})")
    elif args.embedding == "physicochemical":
        encoder = PhysicochemicalEncoder()
        print("Using Physicochemical encoder")
    else:
        encoder = OneHotEncoder()
        print("Using OneHot encoder")

    # 4. Setup model
    model = GPRegressor(kernel=args.kernel)
    print(f"GP kernel: {args.kernel}")

    # 5. Write WT FASTA
    wt_file = os.path.join(args.output, "wt.fasta")
    with open(wt_file, 'w') as f:
        f.write(f'>wt\n{binder_seq}\n')

    # 6. Load or generate initial sequences
    fitness_file = os.path.join(args.output, "fitness.csv")

    if args.init_csv and os.path.exists(args.init_csv):
        print(f"Loading initial data from {args.init_csv}")
        df_fitness = pd.read_csv(args.init_csv)
    else:
        print("Generating initial sequences via ProteinMPNN...")
        seqs, _ = protein_mpnn(
            args.pose_pdb,
            batch_size=args.num_init_seqs,
            design_chain=design_chain,
            temperature=args.mpnn_temperature,
            checkpoint_path=args.mpnn_checkpoint,
        )

        rows = []
        for i, seq_data in enumerate(seqs):
            seq = seq_data['sequence']
            score = seq_data['score']
            swi_val = swi(seq)

            # Mutate PDB and score
            mut_pdb = os.path.join(args.output, f"init_{i}.pdb")
            seq_to_pdb(seq, args.pose_pdb, mut_pdb, design_chain=design_chain)

            rows.append({
                'Variants': seq,
                'MUT': 'WT' if seq == binder_seq else _seq_to_mut(seq, binder_seq),
                'MPNN': score,
                'SWI': swi_val,
                'Fitness': score,  # initial fitness = MPNN score
                'is_init_proposed_seq': True,
            })

        df_fitness = pd.DataFrame(rows)
        df_fitness.to_csv(fitness_file, index=False)

    # 7. Run optimization rounds
    model_queries_per_round = args.model_queries or 3000

    for r in range(args.rounds):
        print(f"\n{'='*50}")
        print(f"Round {r + 1}/{args.rounds}")
        print(f"{'='*50}")

        # Write current fitness to CSV for landscape
        df_fitness.to_csv(fitness_file, index=False)

        # Setup landscape
        landscape = EXPLandscape(
            fitness_file, wt_file,
            search_space=','.join([f'{aa}{i+1}' for i, aa in enumerate(binder_seq)]),
            dir_path=args.output,
        )

        # Build is_init_proposed_list
        if 'is_init_proposed_seq' in df_fitness.columns:
            is_init_proposed_list = df_fitness['is_init_proposed_seq'].tolist()
        else:
            is_init_proposed_list = [True] * len(df_fitness)

        # Setup explorer
        if args.method == "BO":
            explorer = BO_EVO(
                encoder=encoder,
                model=model,
                rounds=r + 1,
                expmt_queries_per_round=args.trials,
                model_queries_per_round=model_queries_per_round,
                starting_sequence=binder_seq,
                alphabet=AAS,
                log_file=os.path.join(args.output, f"round_{r+1}.csv"),
                util_func=args.acquisition,
                uf_param=args.uf_param,
            )
        else:
            explorer = MCMC(
                encoder=encoder,
                model=model,
                rounds=r + 1,
                expmt_queries_per_round=args.trials,
                model_queries_per_round=model_queries_per_round,
                starting_sequence=binder_seq,
                alphabet=AAS,
                log_file=os.path.join(args.output, f"round_{r+1}.csv"),
            )

        # Run explorer
        measured_data, metadata = explorer.run(
            landscape,
            init_seqs_file=fitness_file,
            is_init_proposed_list=is_init_proposed_list,
            verbose=True,
            mask=mask,
        )

        # Score proposed sequences
        proposed_file = os.path.join(args.output, "proposed_seqs.csv")
        if os.path.exists(proposed_file):
            df_proposed = pd.read_csv(proposed_file)
            proposed_seqs = df_proposed['Variants'].tolist()

            new_rows = []
            for seq in proposed_seqs:
                if seq in df_fitness['Variants'].values:
                    continue

                # Mutate and score
                mut_pdb = os.path.join(args.output, f"{seq}.pdb")
                mut_list = _seq_to_mut(seq, binder_seq)
                seq_to_pdb(seq, args.pose_pdb, mut_pdb, design_chain=design_chain)

                try:
                    mpnn_score = score_complex(
                        mut_pdb,
                        design_chain,
                        checkpoint_path=args.mpnn_checkpoint,
                    )
                except Exception:
                    mpnn_score = np.nan

                swi_val = swi(seq)
                new_rows.append({
                    'Variants': seq,
                    'MUT': mut_list,
                    'MPNN': mpnn_score,
                    'SWI': swi_val,
                    'Fitness': mpnn_score,
                    'is_init_proposed_seq': False,
                })

            if new_rows:
                df_new = pd.DataFrame(new_rows)
                df_fitness = pd.concat([df_fitness, df_new], ignore_index=True)

        # Cleanup explorer
        del explorer

    # 8. Final output
    df_fitness.sort_values(by='MPNN', ascending=True, inplace=True)
    df_fitness.to_csv(os.path.join(args.output, 'results.csv'), index=False)
    print(f"\nResults saved to {os.path.join(args.output, 'results.csv')}")
    print(f"Total sequences evaluated: {len(df_fitness)}")

    # Generate final D-peptide PDBs for top candidates
    binder_dir = os.path.join(args.output, 'Binders')
    os.makedirs(binder_dir, exist_ok=True)
    for i, row in df_fitness.head(args.trials).iterrows():
        seq = row['Variants']
        l_pdb = os.path.join(args.output, f"{seq}.pdb")
        d_pdb = os.path.join(binder_dir, f"D_peptide_{seq}.pdb")
        if os.path.exists(l_pdb):
            ld_convert(l_pdb, d_pdb)
    print(f"Top {args.trials} D-peptide structures saved to {binder_dir}/")


def _seq_to_mut(seq, wt_seq):
    mut_list = []
    for i, aa in enumerate(seq):
        if aa != wt_seq[i]:
            mut_list.append(f'{wt_seq[i]}{i+1}{aa}')
    return ','.join(mut_list) if mut_list else 'WT'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="D-peptide Bayesian Optimization (Tier 2)")
    parser.add_argument('--pose_pdb', type=str, required=True,
                        help='L-form pose PDB from Tier 1 (complex with D-receptor)')
    parser.add_argument('--receptor_pdb', type=str, required=True,
                        help='Original L-form receptor PDB')
    parser.add_argument('--output', type=str, default='bo_output',
                        help='Output directory')
    parser.add_argument('--rounds', type=int, default=8,
                        help='Number of BO rounds')
    parser.add_argument('--trials', type=int, default=10,
                        help='Sequences proposed per round')
    parser.add_argument('--method', type=str, default='BO', choices=['BO', 'MCMC'],
                        help='Exploration method')
    parser.add_argument('--embedding', type=str, default='onehot',
                        choices=['onehot', 'physicochemical', 'boltz2embedding'],
                        help='Sequence embedding method')
    parser.add_argument('--kernel', type=str, default='Matern',
                        choices=['Matern', 'RBF'],
                        help='GP kernel')
    parser.add_argument('--acquisition', type=str, default='UCB',
                        choices=['UCB', 'LCB', 'EI', 'PI', 'TS', 'Greedy', 'NEI', 'QUCB'],
                        help='Acquisition function (BO only)')
    parser.add_argument('--uf_param', type=float, default=0.2,
                        help='Acquisition function hyperparameter')
    parser.add_argument('--model_queries', type=int, default=3000,
                        help='Model queries per round')
    parser.add_argument('--num_init_seqs', type=int, default=32,
                        help='Initial sequences from ProteinMPNN')
    parser.add_argument('--mpnn_temperature', type=float, default=0.1,
                        help='ProteinMPNN sampling temperature')
    parser.add_argument('--mpnn_checkpoint', type=str, default=None,
                        help='ProteinMPNN checkpoint path. Defaults to env ProteinMPNN_CHECKPOINT or vanilla v_48_020.')
    parser.add_argument('--mut_positions', type=str, default=None,
                        help='Mutable positions, e.g. "1-5,8,10-12" (1-indexed)')
    parser.add_argument('--init_csv', type=str, default=None,
                        help='CSV with initial sequence data (skip MPNN init)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device number')
    parser.add_argument('--boltz2_url', type=str, default=None,
                        help='Boltz2 API server URL')
    parser.add_argument('--boltz2_token', type=str, default=None,
                        help='Boltz2 API token')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    main(args)
