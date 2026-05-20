import os
import numpy as np
import torch

import pandas as pd

from string import ascii_uppercase, ascii_lowercase
alphabet_list = list(ascii_uppercase+ascii_lowercase)

from utils.pdb_processing import ld_convert, seq_to_pdb, get_pdb_chains
from utils.chroma_sample import binder_sample
from utils.protein_mpnn import (
    protein_mpnn,
    plot_amino_acid_probs,
    resolve_checkpoint_path,
    score_complex,
)
from bo.scoring import synthesis_metrics


def run_bo_optimization(L_binder, binder_seq, design_chain, output_path, bo_cfg):
    """Run Bayesian Optimization for one generated pose.

    Args:
        L_binder: path to the L-form pose PDB
        binder_seq: wild-type binder sequence
        design_chain: chain letter for the designed binder
        output_path: base output directory
        bo_cfg: dict with BO parameters
    """
    from bo.encoders import OneHotEncoder, PhysicochemicalEncoder, Boltz2Encoder, AAS
    from bo.models import GPRegressor
    from bo.explorers import BO_EVO, MCMC, Boltz2BO, validate_bo_proposal_config
    from bo.landscape import EXPLandscape
    from bo.scoring import synthesis_metrics

    boltz2_url = bo_cfg.get('boltz2_url')
    boltz2_token = bo_cfg.get('boltz2_token')
    if not boltz2_url or not boltz2_token:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
        boltz2_url = boltz2_url or os.getenv('BOLTZ2EMBEDDING_URL')
        boltz2_token = boltz2_token or os.getenv('BOLTZ2EMBEDDING_TOKEN')

    bo_dir = os.path.join(output_path, 'BO')
    os.makedirs(bo_dir, exist_ok=True)
    keep_intermediates = bo_cfg.get('keep_intermediates', True)
    eval_pdb_dir = os.path.join(bo_dir, 'Eval_PDBs')
    if keep_intermediates:
        os.makedirs(eval_pdb_dir, exist_ok=True)

    wt_file = os.path.join(bo_dir, 'wt.fasta')
    with open(wt_file, 'w') as f:
        f.write(f'>wt\n{binder_seq}\n')

    search_space = ','.join([f'{aa}{i+1}' for i, aa in enumerate(binder_seq)])

    if bo_cfg['embedding'] == 'boltz2embedding':
        if not boltz2_url or not boltz2_token:
            raise ValueError(
                "Boltz2Embedding requires BOLTZ2EMBEDDING_URL and BOLTZ2EMBEDDING_TOKEN. "
                "Set them in .env or pass via --boltz2_url and --boltz2_token."
            )
        encoder = Boltz2Encoder(
            base_url=boltz2_url,
            api_token=boltz2_token,
            cache_dir=bo_cfg.get('boltz2_cache_dir', os.path.join(bo_dir, 'boltz2_cache')),
            batch_size=bo_cfg.get('boltz2_batch_size'),
            max_parallel_jobs=bo_cfg.get('boltz2_max_parallel_jobs', 2),
        )
    elif bo_cfg['embedding'] == 'physicochemical':
        encoder = PhysicochemicalEncoder()
    else:
        encoder = OneHotEncoder()

    model = GPRegressor(kernel=bo_cfg['kernel'])
    fitness_file = os.path.join(bo_dir, 'fitness.csv')
    synthesis_penalty_weight = bo_cfg.get('synthesis_penalty_weight', 0.0)
    proposal_cfg = {
        'min_mutations': bo_cfg.get('proposal_min_mutations', 1),
        'max_mutations': bo_cfg.get('proposal_max_mutations', min(3, len(binder_seq))),
        'batch_diversity_min_hamming': bo_cfg.get('batch_diversity_min_hamming', 2),
        'proposal_synthesis_penalty_weight': bo_cfg.get('proposal_synthesis_penalty_weight', 0.25),
        'proposal_max_synthesis_penalty': bo_cfg.get('proposal_max_synthesis_penalty', 3.1),
        'preflight_samples': bo_cfg.get('proposal_preflight_samples', 600),
    }
    if bo_cfg['method'] == 'BO':
        validate_bo_proposal_config(
            binder_seq,
            bo_cfg['trials'],
            bo_cfg.get('model_queries', 3000),
            alphabet=AAS,
            **proposal_cfg,
        )
    elif bo_cfg['method'] == 'Boltz2BO':
        pass  # Boltz2BO does its own lightweight validation

    def _score_sequence_for_bo(seq, round_idx='initial', write_pdb=False):
        metrics = synthesis_metrics(seq)
        mut_pdb = (
            os.path.join(eval_pdb_dir, f'round{round_idx}_{seq}.pdb')
            if write_pdb else os.path.join(bo_dir, f'_eval_{seq}.pdb')
        )
        mpnn_score = np.nan
        fitness = np.nan
        try:
            seq_to_pdb(seq, L_binder, mut_pdb, design_chain=design_chain)
            mpnn_score = score_complex(
                mut_pdb,
                design_chain,
                checkpoint_path=bo_cfg.get('mpnn_checkpoint'),
            )
            fitness = -mpnn_score - synthesis_penalty_weight * metrics['synthesis_penalty']
        finally:
            if not write_pdb and os.path.exists(mut_pdb):
                os.remove(mut_pdb)
        return {
            'Variants': seq,
            'Fitness': fitness,
            'MPNN_score': mpnn_score,
            **metrics,
        }

    def _random_initial_sequences(count, seed):
        rng = np.random.default_rng(seed)
        seqs = set()
        max_draws = max(1000, int(count) * 100)
        draws = 0
        while len(seqs) < int(count) and draws < max_draws:
            seq = ''.join(rng.choice(list(AAS), size=len(binder_seq)))
            if seq != binder_seq:
                seqs.add(seq)
            draws += 1
        if len(seqs) < int(count):
            raise RuntimeError(
                f'Generated only {len(seqs)} random initial sequences; requested {count}'
            )
        return sorted(seqs)

    init_source = bo_cfg.get('init_source', 'tier1')
    if init_source == 'tier1':
        tier1_df = bo_cfg.get('init_df')
        if tier1_df is None:
            tier1_csv = bo_cfg.get('init_csv')
            if not tier1_csv or not os.path.exists(tier1_csv):
                raise FileNotFoundError('BO init_source=tier1 requires init_df or init_csv')
            tier1_df = pd.read_csv(tier1_csv)
        top_seqs = (
            tier1_df
            .sort_values('score', ascending=True)
            .drop_duplicates(subset=['sequence'], keep='first')
        )
        rows = []
        for _, row in top_seqs.iterrows():
            metrics = synthesis_metrics(row['sequence'])
            mpnn_score = float(row['score'])
            fitness = -mpnn_score - synthesis_penalty_weight * metrics['synthesis_penalty']
            rows.append({
                'Variants': row['sequence'],
                'Fitness': fitness,
                'MPNN_score': mpnn_score,
                **metrics,
                'is_init_proposed_seq': True,
            })
        df_fitness = pd.DataFrame(rows)
    elif init_source == 'random':
        init_count = int(bo_cfg.get('init_count', bo_cfg.get('trials', 10)))
        init_seed = int(bo_cfg.get('init_seed', 0))
        rows = []
        for seq in _random_initial_sequences(init_count, init_seed):
            row = _score_sequence_for_bo(
                seq,
                round_idx='initial',
                write_pdb=keep_intermediates,
            )
            row['is_init_proposed_seq'] = True
            rows.append(row)
        df_fitness = pd.DataFrame(rows)
    else:
        row = _score_sequence_for_bo(
            binder_seq,
            round_idx='initial',
            write_pdb=keep_intermediates,
        )
        row['is_init_proposed_seq'] = True
        df_fitness = pd.DataFrame([row])
    df_fitness.to_csv(fitness_file, index=False)
    if keep_intermediates:
        df_fitness.to_csv(os.path.join(bo_dir, 'fitness_initial.csv'), index=False)

    for r in range(bo_cfg['rounds']):
        print(f"\n{'='*50}")
        print(f"BO Round {r+1}/{bo_cfg['rounds']}")
        print(f"{'='*50}")

        df_fitness.to_csv(fitness_file, index=False)
        if keep_intermediates:
            df_fitness.to_csv(os.path.join(bo_dir, f'fitness_before_round_{r+1}.csv'), index=False)

        landscape = EXPLandscape(
            fitness_file, wt_file,
            search_space=search_space, dir_path=bo_dir,
        )

        def _make_scorer(pose_pdb, chain):
            def scorer(sequences):
                landscape._proposed_seqs = set(sequences)
                landscape._write_sequences(sequences)
                if keep_intermediates:
                    pd.DataFrame({'Variants': sequences}).to_csv(
                        os.path.join(bo_dir, f'proposed_round_{r+1}.csv'),
                        index=False,
                    )
                rows = []
                for seq in sequences:
                    rows.append(_score_sequence_for_bo(
                        seq,
                        round_idx=r + 1,
                        write_pdb=keep_intermediates,
                    ))
                df_cur = pd.read_csv(fitness_file)
                for row in rows:
                    fit = row['Fitness']
                    if not np.isnan(fit):
                        existing = df_cur['Variants'] == row['Variants']
                        if existing.any():
                            for key, value in row.items():
                                df_cur.loc[existing, key] = value
                        else:
                            df_cur = pd.concat([df_cur, pd.DataFrame({
                                **{k: [v] for k, v in row.items()},
                                'is_init_proposed_seq': [False],
                            })], ignore_index=True)
                df_cur.to_csv(fitness_file, index=False)
                if keep_intermediates:
                    measured_df = pd.DataFrame(rows)
                    if 'Fitness' in measured_df.columns:
                        measured_df['score'] = -measured_df['Fitness']
                    measured_df.to_csv(os.path.join(bo_dir, f'measured_round_{r+1}.csv'), index=False)
                    df_cur.to_csv(os.path.join(bo_dir, f'fitness_after_round_{r+1}.csv'), index=False)
                return np.array([row['Fitness'] for row in rows], dtype=np.float32)
            return scorer

        landscape.get_fitness = _make_scorer(L_binder, design_chain)

        is_init = df_fitness['is_init_proposed_seq'].tolist() if 'is_init_proposed_seq' in df_fitness.columns else [True] * len(df_fitness)

        if bo_cfg['method'] == 'MCMC':
            explorer = MCMC(
                encoder=encoder, model=model, rounds=r+1,
                expmt_queries_per_round=bo_cfg['trials'],
                model_queries_per_round=bo_cfg.get('model_queries', 3000),
                starting_sequence=binder_seq, alphabet=AAS,
                log_file=os.path.join(bo_dir, f'round_{r+1}.csv'),
            )
        elif bo_cfg['method'] == 'Boltz2BO':
            # Auxiliary OneHot encoder/model for ARD position guidance
            aux_encoder = OneHotEncoder()
            aux_model = GPRegressor(kernel=bo_cfg['kernel'])
            # Train auxiliary GP on same data for position weights
            aux_dir = os.path.join(bo_dir, f'round_{r+1}')
            explorer = Boltz2BO(
                encoder=encoder, model=model, rounds=r+1,
                expmt_queries_per_round=bo_cfg['trials'],
                model_queries_per_round=bo_cfg.get('model_queries', 3000),
                starting_sequence=binder_seq, alphabet=AAS,
                log_file=os.path.join(bo_dir, f'round_{r+1}.csv'),
                util_func=bo_cfg['acquisition'],
                uf_param=bo_cfg.get('uf_param', 0.2),
                min_mutations=proposal_cfg['min_mutations'],
                max_mutations=proposal_cfg['max_mutations'],
                batch_diversity_min_hamming=proposal_cfg['batch_diversity_min_hamming'],
                proposal_synthesis_penalty_weight=proposal_cfg['proposal_synthesis_penalty_weight'],
                proposal_max_synthesis_penalty=proposal_cfg['proposal_max_synthesis_penalty'],
                auxiliary_encoder=aux_encoder,
                auxiliary_model=aux_model,
            )
        else:
            explorer = BO_EVO(
                encoder=encoder, model=model, rounds=r+1,
                expmt_queries_per_round=bo_cfg['trials'],
                model_queries_per_round=bo_cfg.get('model_queries', 3000),
                starting_sequence=binder_seq, alphabet=AAS,
                log_file=os.path.join(bo_dir, f'round_{r+1}.csv'),
                util_func=bo_cfg['acquisition'],
                uf_param=bo_cfg.get('uf_param', 0.2),
                **proposal_cfg,
            )

        measured_data, metadata = explorer.run(
            landscape, init_seqs_file=fitness_file,
            is_init_proposed_list=is_init, verbose=True,
        )
        df_fitness = pd.read_csv(fitness_file)
        if keep_intermediates:
            measured_data.to_csv(os.path.join(bo_dir, f'explorer_round_{r+1}.csv'), index=False)
        del explorer

    df_fitness = df_fitness.drop_duplicates(subset=['Variants'], keep='last')
    df_fitness['score'] = -df_fitness['Fitness']
    df_fitness.sort_values(by='score', ascending=True, inplace=True)
    df_fitness.to_csv(os.path.join(bo_dir, 'bo_results.csv'), index=False)
    print(f"\nBO results saved to {os.path.join(bo_dir, 'bo_results.csv')}")

    # Generate D-peptide PDBs for top BO candidates.
    D_binder = os.path.join(os.path.dirname(L_binder), 'Binder_D_pose_1.pdb')
    if os.path.exists(D_binder):
        bo_binder_dir = os.path.join(bo_dir, 'Binders')
        os.makedirs(bo_binder_dir, exist_ok=True)
        for _, row in df_fitness.head(bo_cfg['trials']).iterrows():
            seq = row['Variants']
            L_pdb = os.path.join(bo_dir, f'{seq}.pdb')
            D_pdb = os.path.join(bo_binder_dir, f'BO_{seq}.pdb')
            seq_to_pdb(seq, L_binder, L_pdb, design_chain=design_chain)
            ld_convert(L_pdb, D_pdb)

    return df_fitness


def main(receptor_pdb, output_path, len_binder, temperature, num_poses, num_seqs_per_pose, bo_cfg=None, result_file='results.csv', mpnn_checkpoint=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(f"ProteinMPNN checkpoint: {resolve_checkpoint_path(mpnn_checkpoint)}")

    df = pd.DataFrame(columns=['pose', 'sequence', 'score', 'filename'])

    receptor_chains = get_pdb_chains(receptor_pdb)
    design_chain = alphabet_list[len(receptor_chains)]

    pose_dir = os.path.join(output_path, f'Poses')
    if not os.path.exists(pose_dir):
        os.makedirs(pose_dir)

    binder_dir = os.path.join(output_path, f'Binders')
    if not os.path.exists(binder_dir):
        os.makedirs(binder_dir)

    image_dir = os.path.join(output_path, f'Images')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    D_receptor = os.path.join(pose_dir, 'receptor_D.pdb')

    # 1. Convert receptor from L to D
    ld_convert(receptor_pdb, D_receptor)

    for i in range(num_poses):

        # 1. Generate output file name for each pose
        L_binder = os.path.join(pose_dir, f'Binder_L_pose_{i+1}.pdb')
        D_binder = os.path.join(pose_dir, f'Binder_D_pose_{i+1}.pdb')

        # 2. Sample binder of L stereoisomer for each pose
        binder_sample(D_receptor, len_binder, L_binder, len_chains=len(receptor_chains), device=device)

        # 3. Run protein mpnn for each pose and generate sequences
        seqs, amino_acid_probs = protein_mpnn(
            L_binder,
            batch_size=num_seqs_per_pose,
            design_chain=design_chain,
            temperature=temperature,
            checkpoint_path=mpnn_checkpoint,
        )

        # 4. Get the length of the designed chain and Plot the amino acid probabilities heatmap
        sequence_length = amino_acid_probs.shape[0]
        plot_amino_acid_probs(amino_acid_probs, sequence_length, output_file=os.path.join(image_dir, f'Pose_{i+1}_amino_acid_probs.png'))

        # 5. Convert binder from L to D
        ld_convert(L_binder, D_binder)

        # 6. Output the generated sequences for the current pose
        print('-'*50)
        print(f"Pose {i+1}:")
        for seq in seqs:
            print(f"Seq: {seq['sequence']}, Score: {seq['score']}")
            D_binder_seq = os.path.join(binder_dir, f'Pose{i+1}_{seq["sequence"]}.pdb')
            seq_to_pdb(seq['sequence'], D_binder, D_binder_seq, design_chain=design_chain)
            metrics = synthesis_metrics(seq['sequence'])
            df = df._append({
                'pose': i+1,
                'sequence': seq['sequence'],
                'score': seq['score'],
                'filename': D_binder_seq,
                **metrics,
            }, ignore_index=True)

        print('-'*50)
        print(f"Saving results to {os.path.join(output_path, result_file)}")
        df.sort_values(by='score', ascending=True, inplace=True)
        df.to_csv(os.path.join(output_path, result_file), index=False)

        # 7. Optional Bayesian Optimization for this pose.
        if bo_cfg and bo_cfg.get('rounds', 0) > 0:
            binder_seq = seqs[0]['sequence']
            pose_init_df = pd.DataFrame([
                {
                    'pose': i + 1,
                    'sequence': seq['sequence'],
                    'score': float(seq['score']),
                }
                for seq in seqs
            ])
            pose_bo_cfg = dict(bo_cfg)
            if pose_bo_cfg.get('init_source', 'tier1') == 'tier1':
                pose_bo_cfg['init_df'] = pose_init_df
            pose_bo_cfg['init_seed'] = pose_bo_cfg.get(
                'init_seed',
                100000 + len_binder * 1000 + i + 1,
            )
            print(f"\nStarting Bayesian Optimization on pose {i+1}...")
            print(f"WT sequence: {binder_seq}")
            bo_df = run_bo_optimization(
                L_binder,
                binder_seq,
                design_chain,
                os.path.join(output_path, f'pose_{i+1}'),
                pose_bo_cfg,
            )
            print(f"BO complete. Top sequence: {bo_df.iloc[0]['Variants']} "
                  f"(score: {bo_df.iloc[0]['score']:.4f})")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Mirror-Peptidizer: D-peptide design with optional Bayesian Optimization.")
    parser.add_argument('--receptor', type=str, help='Input receptor PDB file', required=True)
    parser.add_argument('--output', type=str, help='Output directory', default='output')
    parser.add_argument('--len_binder', type=int, help='Length of binder', default=11)
    parser.add_argument('--temperature', type=float, help='Temperature for protein MPNN', default=0.1)
    parser.add_argument('--num_poses', type=int, help='Number of poses to generate', default=1)
    parser.add_argument('--num_seqs_per_pose', type=int, help='Number of sequences to generate per pose', default=8)
    parser.add_argument('--gpu', type=int, help='GPU device number', default=0)
    parser.add_argument('--mpnn_checkpoint', type=str, default=None,
                        help='ProteinMPNN checkpoint path. Defaults to env ProteinMPNN_CHECKPOINT or vanilla v_48_020.')

    # BO optional arguments
    bo_group = parser.add_argument_group('Bayesian Optimization (optional)')
    bo_group.add_argument('--bo_rounds', type=int, default=0,
                          help='Number of BO rounds (0 = disabled, default)')
    bo_group.add_argument('--bo_trials', type=int, default=10,
                          help='Sequences proposed per BO round')
    bo_group.add_argument('--bo_method', type=str, default='BO', choices=['BO', 'MCMC', 'Boltz2BO'],
                          help='BO exploration method')
    bo_group.add_argument('--bo_embedding', type=str, default='onehot', choices=['onehot', 'physicochemical', 'boltz2embedding'],
                          help='Sequence embedding for BO')
    bo_group.add_argument('--bo_kernel', type=str, default='Matern', choices=['Matern', 'RBF'],
                          help='GP kernel for BO')
    bo_group.add_argument('--bo_acquisition', type=str, default='UCB',
                          choices=['UCB', 'LCB', 'EI', 'PI', 'TS', 'Greedy', 'NEI', 'QUCB'],
                          help='Acquisition function')
    bo_group.add_argument('--bo_uf_param', type=float, default=0.2,
                          help='Acquisition function hyperparameter')
    bo_group.add_argument('--bo_model_queries', type=int, default=3000,
                          help='Model queries per BO round')
    bo_group.add_argument('--bo_init_source', type=str, default='tier1',
                          choices=['tier1', 'random', 'single'],
                          help='Initial BO observations: pose ProteinMPNN sequences, random sequences, or the first pose sequence')
    bo_group.add_argument('--bo_random_init_seqs', type=int, default=10,
                          help='Random initial sequences when --bo_init_source random')
    bo_group.add_argument('--synthesis_penalty_weight', type=float, default=0.5,
                          help='Weight for SPPS synthesis-risk penalty in BO fitness')
    bo_group.add_argument('--proposal_min_mutations', type=int, default=1)
    bo_group.add_argument('--proposal_max_mutations', type=int, default=3)
    bo_group.add_argument('--batch_diversity_min_hamming', type=int, default=2)
    bo_group.add_argument('--proposal_synthesis_penalty_weight', type=float, default=0.25)
    bo_group.add_argument('--proposal_max_synthesis_penalty', type=float, default=3.1)
    bo_group.add_argument('--proposal_preflight_samples', type=int, default=600)
    bo_group.add_argument('--boltz2_url', type=str, default=None,
                          help='Boltz2Embedding API server URL (required if --bo_embedding boltz2embedding)')
    bo_group.add_argument('--boltz2_token', type=str, default=None,
                          help='Boltz2Embedding API token (required if --bo_embedding boltz2embedding)')
    bo_group.add_argument('--boltz2_batch_size', type=int, default=None,
                          help='Optional sequences per Boltz2Embedding job; omit to submit each encode call as one job')
    bo_group.add_argument('--boltz2_max_parallel_jobs', type=int, default=2)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    bo_cfg = None
    if args.bo_rounds > 0:
        bo_cfg = {
            'rounds': args.bo_rounds,
            'trials': args.bo_trials,
            'method': args.bo_method,
            'embedding': args.bo_embedding,
            'kernel': args.bo_kernel,
            'acquisition': args.bo_acquisition,
            'uf_param': args.bo_uf_param,
            'model_queries': args.bo_model_queries,
            'boltz2_url': args.boltz2_url,
            'boltz2_token': args.boltz2_token,
            'boltz2_batch_size': args.boltz2_batch_size,
            'boltz2_max_parallel_jobs': args.boltz2_max_parallel_jobs,
            'mpnn_checkpoint': args.mpnn_checkpoint,
            'init_source': args.bo_init_source,
            'init_count': args.bo_random_init_seqs,
            'synthesis_penalty_weight': args.synthesis_penalty_weight,
            'proposal_min_mutations': args.proposal_min_mutations,
            'proposal_max_mutations': args.proposal_max_mutations,
            'batch_diversity_min_hamming': args.batch_diversity_min_hamming,
            'proposal_synthesis_penalty_weight': args.proposal_synthesis_penalty_weight,
            'proposal_max_synthesis_penalty': args.proposal_max_synthesis_penalty,
            'proposal_preflight_samples': args.proposal_preflight_samples,
            'keep_intermediates': True,
        }

    main(
        args.receptor,
        args.output,
        args.len_binder,
        args.temperature,
        args.num_poses,
        args.num_seqs_per_pose,
        bo_cfg=bo_cfg,
        mpnn_checkpoint=args.mpnn_checkpoint,
    )
