import os
import numpy as np
import torch

import pandas as pd

from string import ascii_uppercase, ascii_lowercase
alphabet_list = list(ascii_uppercase+ascii_lowercase)

from utils.pdb_processing import ld_convert, seq_to_pdb, get_pdb_chains
from utils.chroma_sample import binder_sample
from utils.pose_filtering import (
    DEFAULT_COMPLEXITY_MAX_RUN,
    DEFAULT_SURFACE_FILTER,
    format_surface_pose_metrics,
    max_homopolymer_run,
    passes_complexity_filter,
    passes_surface_pose_filter,
    surface_pose_metrics,
)
from utils.protein_mpnn import (
    protein_mpnn,
    plot_amino_acid_probs,
    resolve_checkpoint_path,
)
from bo.scoring import synthesis_metrics


SURFACE_FILTER_COLUMNS = [
    'surface_filter_pass',
    'surface_filter_attempts',
    'radial_percentile',
    'nearby_receptor_atoms',
    'outward_nearby_fraction',
    'occupied_octants',
    'min_receptor_distance',
    'median_receptor_distance',
    'binder_atoms_within_contact',
]


def _surface_filter_cfg(args):
    return {
        'radial_min': args.surface_radial_min,
        'outward_nearby_max': args.surface_outward_nearby_max,
        'occupied_octants_max': args.surface_occupied_octants_max,
        'min_binder_atoms_within_contact': args.surface_min_contact_atoms,
        'nearby_distance': args.surface_nearby_distance,
        'contact_distance': args.surface_contact_distance,
        'max_attempts': args.surface_max_attempts,
    }


def run_bo_optimization(L_binder, binder_seq, design_chain, output_path, bo_cfg):
    """Run Bayesian Optimization for one generated pose.

    Args:
        L_binder: path to the L-form pose PDB
        binder_seq: wild-type binder sequence
        design_chain: chain letter for the designed binder
        output_path: base output directory
        bo_cfg: dict with BO parameters

    The per-sequence oracle is protenix2dock (peptide mode on the complex).
    In the default **dock** flavour the engine re-docks the peptide against
    the fixed receptor and the BO fitness is ``ipsae_weight * ipsae_dom +
    plddt_weight * ligand_plddt - rmsd_weight * peptide_rmsd -
    synthesis_penalty_weight * synthesis_penalty``. The **score** flavour
    bypasses diffusion for a fast confidence-only pass without the RMSD term.
    ProteinMPNN is not part of the BO objective — it only produces the Tier-1
    seed sequences.
    """
    from bo.encoders import OneHotEncoder, PhysicochemicalEncoder, Boltz2Encoder, AAS
    from bo.models import GPRegressor
    from bo.explorers import BO_EVO, MCMC, Boltz2BO, validate_bo_proposal_config
    from bo.landscape import EXPLandscape
    from bo.scoring import synthesis_metrics
    from utils.protenix2dock_client import Protenix2DockScorer

    protenix_scorer = Protenix2DockScorer(
        gpu=bo_cfg.get('protenix_gpu', 0),
        config=None,
    )
    # final-form complex (L-target + D-peptide) by default: the large
    # receptor chain stays in-distribution for Protenix; 'mirror' scores
    # the Chroma-side complex (D-target + L-peptide) instead
    form = bo_cfg.get('protenix_form', 'final')
    template_pdb = (
        L_binder.replace('_L_', '_D_') if form == 'final' else L_binder
    )
    if not os.path.exists(template_pdb):
        raise FileNotFoundError(
            f'BO scoring needs the {"D-form" if form == "final" else "L-form"} '
            f'complex PDB {template_pdb}')
    ipsae_weight = float(bo_cfg.get('ipsae_weight', 0.6))
    plddt_weight = float(bo_cfg.get('plddt_weight', 0.4))
    rmsd_weight = float(bo_cfg.get('rmsd_weight', 0.05))
    protenix_mode = bo_cfg.get('protenix_mode', 'dock')
    if protenix_mode not in ('dock', 'score'):
        raise ValueError(
            f"bo_protenix_mode must be 'dock' or 'score', got {protenix_mode!r}")
    protenix_score_only = protenix_mode == 'score'
    protenix_use_msa = bo_cfg.get('protenix_msa', 'auto')
    protenix_seed = int(bo_cfg.get('protenix_seed', 42))
    protenix_diffusion_samples = bo_cfg.get('protenix_diffusion_samples')
    protenix_sampling_steps = bo_cfg.get('protenix_sampling_steps')
    print(
        f"BO scoring: protenix2dock (form={form}, mode={protenix_mode}, "
        f"fitness = {ipsae_weight}*ipsae_dom + {plddt_weight}*ligand_plddt "
        f"- {rmsd_weight}*peptide_rmsd"
        + (f" - {bo_cfg.get('synthesis_penalty_weight', 0.0)}*synthesis_penalty"
           if bo_cfg.get('synthesis_penalty_weight', 0) else "")
    )

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
        # protenix2dock oracle: score the mutated complex with the Protenix
        # confidence heads; ipSAE + peptide pLDDT (+ pose RMSD in dock mode)
        # drive the fitness
        metrics = synthesis_metrics(seq)
        mut_pdb = (
            os.path.join(eval_pdb_dir, f'round{round_idx}_{seq}.pdb')
            if write_pdb else os.path.join(bo_dir, f'_eval_{seq}.pdb')
        )
        seq_to_pdb(seq, template_pdb, mut_pdb, design_chain=design_chain)
        job_dir = (
            os.path.join(eval_pdb_dir, f'round{round_idx}_{seq}_protenix')
            if keep_intermediates
            else os.path.join(bo_dir, f'_protenix_{seq}')
        )
        try:
            px = protenix_scorer.score_peptide_complex(
                mut_pdb,
                peptide_chain=design_chain,
                out_dir=job_dir,
                peptide_sequence=seq,
                score_only=protenix_score_only,
                seed=protenix_seed,
                use_msa_server=protenix_use_msa,
                diffusion_samples=protenix_diffusion_samples,
                sampling_steps=protenix_sampling_steps,
            )
        finally:
            if not keep_intermediates:
                import shutil as _shutil
                _shutil.rmtree(job_dir, ignore_errors=True)
                if os.path.exists(mut_pdb):
                    os.remove(mut_pdb)
        pose_rmsd = px.get('peptide_rmsd')
        fitness = (
            ipsae_weight * float(px.get('ipsae_dom') or 0.0)
            + plddt_weight * float(px.get('ligand_plddt') or 0.0)
            - (rmsd_weight * float(pose_rmsd) if pose_rmsd is not None else 0.0)
            - synthesis_penalty_weight * metrics['synthesis_penalty']
        )
        return {
            'Variants': seq,
            'Fitness': fitness,
            'ipsae_dom': px.get('ipsae_dom'),
            'ligand_ipsae_max': px.get('ligand_ipsae_max'),
            'ligand_plddt': px.get('ligand_plddt'),
            'peptide_rmsd': pose_rmsd,
            'iptm': px.get('iptm'),
            'ranking_score': px.get('ranking_score'),
            'interface_pair_count': px.get('interface_pair_count'),
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
        # Tier-1 ranks by the MPNN NLL, which is not the BO objective: the
        # seed sequences are re-evaluated with protenix2dock so the GP trains
        # on the same fitness the BO loop will optimize
        init_top_n = int(bo_cfg.get('protenix_init_top', 10))
        rows = []
        for rank, (_, row) in enumerate(top_seqs.head(init_top_n).iterrows(), start=1):
            seq_row = _score_sequence_for_bo(
                row['sequence'],
                round_idx=f'init{rank}',
                write_pdb=keep_intermediates,
            )
            seq_row['is_init_proposed_seq'] = True
            rows.append(seq_row)
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
    D_binder = L_binder.replace('_L_', '_D_')
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


def main(
    receptor_pdb,
    output_path,
    len_binder,
    temperature,
    num_poses,
    num_seqs_per_pose,
    bo_cfg=None,
    result_file='results.csv',
    mpnn_checkpoint=None,
    filter_surface_poses=False,
    surface_filter_cfg=None,
    filter_complexity=False,
    complexity_max_run=DEFAULT_COMPLEXITY_MAX_RUN,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(f"ProteinMPNN checkpoint: {resolve_checkpoint_path(mpnn_checkpoint)}")

    df = pd.DataFrame(columns=['pose', 'sequence', 'score', 'filename'])

    receptor_chains = get_pdb_chains(receptor_pdb)
    design_chain = alphabet_list[len(receptor_chains)]
    surface_filter_cfg = surface_filter_cfg or dict(DEFAULT_SURFACE_FILTER, max_attempts=10)
    if filter_surface_poses:
        df = pd.DataFrame(columns=['pose', 'sequence', 'score', 'filename', *SURFACE_FILTER_COLUMNS])
        print(
            "Surface pose filter enabled: "
            f"radial_min={surface_filter_cfg['radial_min']}, "
            f"outward_nearby_max={surface_filter_cfg['outward_nearby_max']}, "
            f"occupied_octants_max={surface_filter_cfg['occupied_octants_max']}, "
            "min_binder_atoms_within_contact="
            f"{surface_filter_cfg['min_binder_atoms_within_contact']}, "
            f"max_attempts={surface_filter_cfg['max_attempts']}"
        )

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

        pose_filter_metrics = {}
        pose_filter_attempts = 1
        pose_filter_pass = True

        # 2. Sample binder of L stereoisomer for each pose
        if filter_surface_poses:
            pose_filter_pass = False
            for attempt in range(1, int(surface_filter_cfg['max_attempts']) + 1):
                binder_sample(
                    D_receptor,
                    len_binder,
                    L_binder,
                    len_chains=len(receptor_chains),
                    device=device,
                )
                pose_filter_metrics = surface_pose_metrics(
                    L_binder,
                    binder_chain=design_chain,
                    receptor_chains=receptor_chains,
                    nearby_distance=surface_filter_cfg['nearby_distance'],
                    contact_distance=surface_filter_cfg['contact_distance'],
                )
                pose_filter_attempts = attempt
                pose_filter_pass = passes_surface_pose_filter(
                    pose_filter_metrics,
                    radial_min=surface_filter_cfg['radial_min'],
                    outward_nearby_max=surface_filter_cfg['outward_nearby_max'],
                    occupied_octants_max=surface_filter_cfg['occupied_octants_max'],
                    min_binder_atoms_within_contact=surface_filter_cfg[
                        'min_binder_atoms_within_contact'
                    ],
                )
                status = "pass" if pose_filter_pass else "reject"
                print(
                    f"Pose {i+1} attempt {attempt}: {status} "
                    f"({format_surface_pose_metrics(pose_filter_metrics)})"
                )
                if pose_filter_pass:
                    break

            if not pose_filter_pass:
                raise RuntimeError(
                    f"Failed to generate a surface-like pose for pose {i+1} "
                    f"after {surface_filter_cfg['max_attempts']} attempts. "
                    "Increase --surface_max_attempts or relax the surface filter thresholds."
                )
        else:
            binder_sample(D_receptor, len_binder, L_binder, len_chains=len(receptor_chains), device=device)

        # 3. Run protein mpnn for each pose and generate sequences
        seqs, amino_acid_probs = protein_mpnn(
            L_binder,
            batch_size=num_seqs_per_pose,
            design_chain=design_chain,
            temperature=temperature,
            checkpoint_path=mpnn_checkpoint,
        )

        # 3b. Optional: drop sequences with a long homopolymer run.
        if filter_complexity:
            kept = []
            for seq in seqs:
                run = max_homopolymer_run(seq['sequence'])
                if passes_complexity_filter(seq['sequence'], complexity_max_run):
                    kept.append(seq)
                else:
                    print(
                        f"  [complexity] drop Pose {i+1} seq "
                        f"(run {run} > {complexity_max_run}): {seq['sequence']}"
                    )
            seqs = kept

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
                **({
                    'surface_filter_pass': pose_filter_pass,
                    'surface_filter_attempts': pose_filter_attempts,
                    **pose_filter_metrics,
                } if filter_surface_poses else {}),
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

    surface_group = parser.add_argument_group('Surface pose filtering (optional)')
    surface_group.add_argument('--filter_surface_poses', action='store_true',
                               help='Reject Chroma poses that are buried inside the receptor before ProteinMPNN.')
    surface_group.add_argument('--surface_max_attempts', type=int, default=10,
                               help='Maximum Chroma resampling attempts per requested pose when filtering is enabled.')
    surface_group.add_argument('--surface_radial_min', type=float,
                               default=DEFAULT_SURFACE_FILTER['radial_min'],
                               help='Minimum receptor radial percentile of the binder center.')
    surface_group.add_argument('--surface_outward_nearby_max', type=float,
                               default=DEFAULT_SURFACE_FILTER['outward_nearby_max'],
                               help='Maximum fraction of nearby receptor atoms beyond the binder in the outward direction.')
    surface_group.add_argument('--surface_occupied_octants_max', type=int,
                               default=DEFAULT_SURFACE_FILTER['occupied_octants_max'],
                               help='Maximum occupied 8A octants around the binder center.')
    surface_group.add_argument('--surface_min_contact_atoms', type=int,
                               default=DEFAULT_SURFACE_FILTER['min_binder_atoms_within_contact'],
                               help='Minimum binder atoms within contact distance of receptor atoms.')
    surface_group.add_argument('--surface_nearby_distance', type=float,
                               default=DEFAULT_SURFACE_FILTER['nearby_distance'],
                               help='Distance cutoff in Angstrom for local receptor enclosure metrics.')
    surface_group.add_argument('--surface_contact_distance', type=float,
                               default=DEFAULT_SURFACE_FILTER['contact_distance'],
                               help='Distance cutoff in Angstrom for binder-receptor contact atoms.')

    complexity_group = parser.add_argument_group('Sequence-complexity filtering (optional)')
    complexity_group.add_argument('--filter_complexity', action='store_true',
                                  help='Drop ProteinMPNN sequences with a long homopolymer run.')
    complexity_group.add_argument('--complexity_max_run', type=int,
                                  default=DEFAULT_COMPLEXITY_MAX_RUN,
                                  help='Max consecutive identical residues allowed before a sequence is dropped '
                                       '(default %(default)s).')

    # BO optional arguments
    bo_group = parser.add_argument_group('Bayesian Optimization (optional)')
    bo_group.add_argument('--bo_ipsae_weight', type=float, default=0.6,
                          help='Weight of ipsae_dom in the protenix BO fitness (default %(default)s)')
    bo_group.add_argument('--bo_plddt_weight', type=float, default=0.4,
                          help='Weight of ligand_plddt in the protenix BO fitness (default %(default)s)')
    bo_group.add_argument('--bo_rmsd_weight', type=float, default=0.05,
                          help='Weight of the peptide pose RMSD penalty in A (dock mode only, '
                               'default %(default)s)')
    bo_group.add_argument('--bo_protenix_form', type=str, default='final',
                          choices=['final', 'mirror'],
                          help="Complex scored by protenix2dock: 'final' (L-target + D-peptide, "
                               "default) or 'mirror' (D-target + L-peptide, the Chroma-side pose)")
    bo_group.add_argument('--bo_protenix_mode', type=str, default='dock',
                          choices=['dock', 'score'],
                          help="protenix2dock flavour: 'dock' (default) re-docks the peptide "
                               "against the fixed receptor and adds the pose RMSD to the fitness; "
                               "'score' is the fast confidence-only pass without RMSD")
    bo_group.add_argument('--bo_protenix_diffusion_samples', type=int, default=None,
                          help='Diffusion samples in dock mode (default: engine mode config)')
    bo_group.add_argument('--bo_protenix_sampling_steps', type=int, default=None,
                          help='Diffusion steps in dock mode (default: engine mode config)')
    bo_group.add_argument('--bo_protenix_msa', type=str, default='auto',
                          choices=['auto', 'on', 'off'],
                          help='MSA server usage for protenix2dock: auto (shared cache first), '
                               'on (always pass the server), off (cache hits only)')
    bo_group.add_argument('--bo_protenix_gpu', type=int, default=None,
                          help='GPU for the protenix2dock runtime container (default: same as --gpu)')
    bo_group.add_argument('--bo_protenix_seed', type=int, default=42,
                          help='Seed for the protenix2dock engine (default %(default)s)')
    bo_group.add_argument('--bo_protenix_init_top', type=int, default=10,
                          help='Tier-1 seed sequences re-evaluated with protenix2dock as BO init '
                               '(default %(default)s)')
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
            'ipsae_weight': args.bo_ipsae_weight,
            'plddt_weight': args.bo_plddt_weight,
            'rmsd_weight': args.bo_rmsd_weight,
            'protenix_form': args.bo_protenix_form,
            'protenix_mode': args.bo_protenix_mode,
            'protenix_diffusion_samples': args.bo_protenix_diffusion_samples,
            'protenix_sampling_steps': args.bo_protenix_sampling_steps,
            'protenix_msa': args.bo_protenix_msa,
            'protenix_gpu': (args.bo_protenix_gpu if args.bo_protenix_gpu is not None else args.gpu),
            'protenix_seed': args.bo_protenix_seed,
            'protenix_init_top': args.bo_protenix_init_top,
            'boltz2_url': args.boltz2_url,
            'boltz2_token': args.boltz2_token,
            'boltz2_batch_size': args.boltz2_batch_size,
            'boltz2_max_parallel_jobs': args.boltz2_max_parallel_jobs,
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
        filter_surface_poses=args.filter_surface_poses,
        surface_filter_cfg=_surface_filter_cfg(args),
        filter_complexity=args.filter_complexity,
        complexity_max_run=args.complexity_max_run,
    )
