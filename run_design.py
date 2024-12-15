import os
import torch

import pandas as pd

from string import ascii_uppercase, ascii_lowercase
alphabet_list = list(ascii_uppercase+ascii_lowercase)

from utils.pdb_processing import ld_convert, seq_to_pdb, get_pdb_chains
from utils.chroma_sample import binder_sample
from utils.protein_mpnn import protein_mpnn, plot_amino_acid_probs

def main(receptor_pdb, output_path, len_binder, temperature, num_poses, num_seqs_per_pose, result_file='results.csv'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

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
        seqs, amino_acid_probs = protein_mpnn(L_binder, batch_size=num_seqs_per_pose, design_chain=design_chain, temperature=temperature)

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
            df = df._append({'pose': i+1, 'sequence': seq['sequence'], 'score': seq['score'], 'filename': D_binder_seq}, ignore_index=True)
    
        print('-'*50)
        print(f"Saving results to {os.path.join(output_path, result_file)}")
        df.sort_values(by='score', ascending=True, inplace=True)
        df.to_csv(os.path.join(output_path, result_file), index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Protein design pipeline.")
    parser.add_argument('--receptor', type=str, help='Input receptor PDB file', required=True)
    parser.add_argument('--output', type=str, help='Output directory', default='output')
    parser.add_argument('--len_binder', type=int, help='Length of binder', default=11)
    parser.add_argument('--temperature', type=float, help='Temperature for protein MPNN', default=0.1)
    parser.add_argument('--num_poses', type=int, help='Number of poses to generate', default=1)
    parser.add_argument('--num_seqs_per_pose', type=int, help='Number of sequences to generate per pose', default=8)
    parser.add_argument('--gpu', type=int, help='GPU device number', default=0)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    main(args.receptor, args.output, args.len_binder, args.temperature, args.num_poses, args.num_seqs_per_pose)