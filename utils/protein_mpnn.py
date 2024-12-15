import os
import torch
import copy
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from ProteinMPNN.vanilla_proteinmpnn import protein_mpnn_utils as utils

from dotenv import load_dotenv
load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_native_score(model, X, S, mask, chain_M, chain_M_pos, residue_idx, chain_encoding_all, score_mode='designed'):
    randn_1 = torch.randn(mask.shape, device=X.device)
    if score_mode == 'designed':
        # Only score designed chains
        scoring_mask = mask * chain_M * chain_M_pos
    elif score_mode == 'whole':
        # Score the whole structure
        scoring_mask = mask
    else:
        raise ValueError("score_mode must be 'designed' or 'whole'.")

    log_probs = model(X, S, mask, scoring_mask, residue_idx, chain_encoding_all, randn_1)
    scores = utils._scores(S, log_probs, scoring_mask)
    native_score = scores.cpu().data.numpy()
    return native_score

def load_model():
    checkpoint_path = os.getenv("ProteinMPNN_CHECKPOINT")
    hidden_dim = 128
    num_layers = 3 
    backbone_noise = 0.00  # Set to zero to disable backbone noise

    checkpoint = torch.load(checkpoint_path, map_location=device)  # Load checkpoint

    model = utils.ProteinMPNN(
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=backbone_noise,
        k_neighbors=checkpoint['num_edges']
    )
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def prepare_inputs(input_pdb, design_chain='B'):
    fixed_positions_dict = None
    omit_AA_dict = None
    tied_positions_dict = None
    pssm_dict = None
    bias_by_res_dict = None

    pdb_dict_list = utils.parse_PDB(input_pdb)
    dataset = utils.StructureDatasetPDB(pdb_dict_list, truncate=None, verbose=True, max_length=10000)

    all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9] == 'seq_chain']
    designed_chain_list = [design_chain]
    fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
    chain_id_dict = {}
    chain_id_dict[pdb_dict_list[0]['name']] = (designed_chain_list, fixed_chain_list)

    batch_clones = [copy.deepcopy(dataset[0])]
    (
        X,
        S,
        mask,
        lengths,
        chain_M,
        chain_encoding_all,
        chain_list_list,
        visible_list_list,
        masked_list_list,
        masked_chain_length_list_list,
        chain_M_pos,
        omit_AA_mask,
        residue_idx,
        dihedral_mask,
        tied_pos_list_of_lists_list,
        pssm_coef,
        pssm_bias,
        pssm_log_odds_all,
        bias_by_res_all,
        tied_beta
    ) = utils.tied_featurize(
        batch_clones,
        device,
        chain_id_dict,
        fixed_positions_dict,
        omit_AA_dict,
        tied_positions_dict,
        pssm_dict,
        bias_by_res_dict
    )
    return X, S, mask, chain_M, chain_M_pos, residue_idx, chain_encoding_all, chain_M_pos, chain_id_dict, batch_clones, \
        chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, omit_AA_mask, \
        pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all

def protein_mpnn(input_pdb, batch_size=1, design_chain='B', temperature=0.3, score_mode='designed'):
    model = load_model()
    (
        X, S, mask, chain_M, chain_M_pos, residue_idx, chain_encoding_all,
        chain_M_pos, chain_id_dict, batch_clones,
        chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list,
        omit_AA_mask, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all
    ) = prepare_inputs(input_pdb, design_chain)

    name_ = batch_clones[0]['name']

    # Compute the native score
    native_score = compute_native_score(
        model,
        X,
        S,
        mask,
        chain_M,
        chain_M_pos,
        residue_idx,
        chain_encoding_all,
        score_mode=score_mode
    )

    with torch.no_grad():
        print('Generating sequences...')
        score_list = []
        all_probs_list = []
        all_log_probs_list = []
        S_sample_list = []
        pssm_log_odds_mask = (pssm_log_odds_all > 0.0).float()  # 1.0 for true, 0.0 for false

        # Generate sequences
        BATCH_COPIES = 1
        NUM_BATCHES = batch_size
        temperatures = [temperature]
        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        omit_AAs_list = ['X']
        omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
        bias_AAs_np = np.zeros(len(alphabet))

        seqs = []
        amino_acid_probs_list = []  # To store amino acid probabilities

        total_design_probs = None
        num_samples = 0

        for temp in temperatures:
            for j in range(NUM_BATCHES):
                randn_2 = torch.randn(chain_M.shape, device=X.device)
                sample_dict = model.sample(
                    X,
                    randn_2,
                    S,
                    chain_M,
                    chain_encoding_all,
                    residue_idx,
                    mask=mask,
                    temperature=temp,
                    omit_AAs_np=omit_AAs_np,
                    bias_AAs_np=bias_AAs_np,
                    chain_M_pos=chain_M_pos,
                    omit_AA_mask=omit_AA_mask,
                    pssm_coef=pssm_coef,
                    pssm_bias=pssm_bias,
                    pssm_multi=0,
                    pssm_log_odds_flag=False,
                    pssm_log_odds_mask=pssm_log_odds_mask,
                    pssm_bias_flag=False,
                    bias_by_res=bias_by_res_all
                )
                S_sample = sample_dict["S"]

                # Extract probability distributions
                probs = sample_dict["probs"]  # Shape: [batch_size, sequence_length, 21]

                # Convert probabilities to numpy array
                probs_np = probs.cpu().data.numpy()

                # Extract design chain probabilities
                for b_ix in range(BATCH_COPIES):
                    design_mask = (chain_M[b_ix] * chain_M_pos[b_ix]).bool().cpu().numpy()
                    design_probs = probs_np[b_ix][design_mask]

                    if total_design_probs is None:
                        total_design_probs = design_probs
                    else:
                        total_design_probs += design_probs
                    num_samples += 1

                # The rest of your code remains unchanged...

                # Define scoring_mask (batch size x sequence length)
                if score_mode == 'designed':
                    scoring_mask = mask * chain_M * chain_M_pos
                elif score_mode == 'whole':
                    scoring_mask = mask
                else:
                    raise ValueError("score_mode must be 'designed' or 'whole'.")

                log_probs = model(
                    X,
                    S_sample,
                    mask,
                    scoring_mask,
                    residue_idx,
                    chain_encoding_all,
                    randn_2,
                    use_input_decoding_order=True,
                    decoding_order=sample_dict["decoding_order"]
                )
                scores = utils._scores(S_sample, log_probs, scoring_mask)
                scores = scores.cpu().data.numpy()
                all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                all_log_probs_list.append(log_probs.cpu().data.numpy())
                S_sample_list.append(S_sample.cpu().data.numpy())
                for b_ix in range(BATCH_COPIES):
                    masked_chain_length_list = masked_chain_length_list_list[b_ix]
                    masked_list = masked_list_list[b_ix]
                    
                    # Define seq_mask per batch item
                    if score_mode == 'designed':
                        seq_mask = chain_M[b_ix]
                    elif score_mode == 'whole':
                        seq_mask = mask[b_ix]
                    else:
                        raise ValueError("score_mode must be 'designed' or 'whole'.")

                    seq_recovery_rate = torch.sum(
                        torch.sum(
                            torch.nn.functional.one_hot(S[b_ix], 21)
                            * torch.nn.functional.one_hot(S_sample[b_ix], 21),
                            axis=-1
                        ) * scoring_mask[b_ix]
                    ) / torch.sum(scoring_mask[b_ix])
                    seq = utils._S_to_seq(S_sample[b_ix], seq_mask)
                    score = scores[b_ix]
                    score_list.append(score)
                    native_seq = utils._S_to_seq(S[b_ix], seq_mask)
                    if b_ix == 0 and j == 0 and temp == temperatures[0]:
                        start = 0
                        end = 0
                        list_of_AAs = []
                        for mask_l in masked_chain_length_list:
                            end += mask_l
                            list_of_AAs.append(native_seq[start:end])
                            start = end
                        native_seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                        l0 = 0
                        for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                            l0 += mc_length
                            native_seq = native_seq[:l0] + '/' + native_seq[l0:]
                            l0 += 1
                        sorted_masked_chain_letters = np.argsort(masked_list_list[0])
                        print_masked_chains = [masked_list_list[0][i] for i in sorted_masked_chain_letters]
                        sorted_visible_chain_letters = np.argsort(visible_list_list[0])
                        print_visible_chains = [visible_list_list[0][i] for i in sorted_visible_chain_letters]
                        native_score_print = np.format_float_positional(
                            np.float32(native_score.mean()), unique=False, precision=4
                        )
                        print(
                            '>{}, score={}, fixed_chains={}, designed_chains={}, model_name={}\n{}\n'.format(
                                name_,
                                native_score_print,
                                print_visible_chains,
                                print_masked_chains,
                                'v_48_020',
                                native_seq
                            )
                        )  # Write the native sequence

                    start = 0
                    end = 0
                    list_of_AAs = []
                    for mask_l in masked_chain_length_list:
                        end += mask_l
                        list_of_AAs.append(seq[start:end])
                        start = end

                    seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                    l0 = 0
                    for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                        l0 += mc_length
                        seq = seq[:l0] + '/' + seq[l0:]
                        l0 += 1
                    score_print = np.format_float_positional(np.float32(score), unique=False, precision=4)
                    seq_rec_print = np.format_float_positional(
                        np.float32(seq_recovery_rate.detach().cpu().numpy()), unique=False, precision=4
                    )
                    seqs.append({'sequence': seq, 'score': score_print})
        # Compute average probabilities
        average_design_probs = total_design_probs / num_samples
        return seqs, average_design_probs

def score_complex(input_pdb, design_chain='B', score_mode='designed'):
    model = load_model()
    X, S, mask, chain_M, chain_M_pos, residue_idx, chain_encoding_all, *_ = prepare_inputs(input_pdb, design_chain)
    native_score = compute_native_score(
        model,
        X,
        S,
        mask,
        chain_M,
        chain_M_pos,
        residue_idx,
        chain_encoding_all,
        score_mode=score_mode
    )
    return native_score.mean()

def plot_amino_acid_probs(probs, sequence_length, output_file='amino_acid_probs_heatmap.png'):

    # probs is of shape [design_length, 21]
    # Create list of amino acids
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
    
    # Remove the 'X' column (if present)
    if probs.shape[1] == 21:
        probs = probs[:, :20]
    
    # Transpose the matrix for plotting
    probs_T = probs.T  # Shape: [20, design_length]
    
    # Create heatmap
    plt.figure(figsize=(sequence_length * 0.5, 8))
    sns.heatmap(probs_T, cmap='viridis', xticklabels=np.arange(1, sequence_length+1), yticklabels=amino_acids, vmin=0, vmax=1)
    plt.xlabel('Binder Position', fontsize=16)
    plt.ylabel('Amino Acids', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)

if __name__ == '__main__':
    input_pdb = 'path_to_your_complex.pdb'  # Replace with your PDB file path
    design_chain = 'B'  # Specify the design chain
    temperature = 0.3
    batch_size = 10  # Increase batch_size for better probability estimation
    score_mode = 'designed'  # or 'whole'

    # Generate sequences and get their scores and amino acid probabilities
    seqs, amino_acid_probs = protein_mpnn(
        input_pdb=input_pdb,
        batch_size=batch_size,
        design_chain=design_chain,
        temperature=temperature,
        score_mode=score_mode
    )

    # Print the generated sequences and their scores
    for seq_info in seqs:
        print(f"Sequence: {seq_info['sequence']}, Score: {seq_info['score']}")

    # Compute the native score of the complex
    score = score_complex(input_pdb, design_chain=design_chain, score_mode=score_mode)
    print(f"The native score of the complex is: {score}")

    # Get the length of the designed chain
    sequence_length = amino_acid_probs.shape[0]

    # Plot the amino acid probabilities heatmap
    plot_amino_acid_probs(amino_acid_probs, sequence_length)
