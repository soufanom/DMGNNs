import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs
import pandas as pd
import numpy as np
import random
import pickle
import os
from rdkit.DataStructs import ExplicitBitVect

def convert_to_explicit_bitvect(ecfp):
    """ Attempt to convert a numpy array back to an RDKit ExplicitBitVect. """
    try:
        if isinstance(ecfp, np.ndarray):
            bitvect = ExplicitBitVect(len(ecfp))
            for i, bit in enumerate(ecfp):
                if bit:
                    bitvect.SetBit(i)
            return bitvect
        return ecfp
    except Exception as e:
        print(f"Conversion to ExplicitBitVect failed: {e}")
        return None

# Function to generate ECFP (Morgan Fingerprints) for a molecule
def generate_ecfp(mol, radius=2, n_bits=1024):
    if mol is None:
        return None
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(ecfp)


# Function to generate an expanded set of chemical descriptors for a molecule
def generate_chemical_descriptors(mol):
    if mol is None:
        return None
    try:
        descriptors = np.array([
            Descriptors.MolWt(mol),                # Molecular weight
            Descriptors.MolLogP(mol),              # LogP (octanol-water partition coefficient)
            Descriptors.NumHDonors(mol),           # Number of hydrogen bond donors
            Descriptors.NumHAcceptors(mol),        # Number of hydrogen bond acceptors
            Descriptors.TPSA(mol),                 # Topological Polar Surface Area
            Descriptors.RingCount(mol),            # Number of rings
            Descriptors.FractionCSP3(mol),         # Fraction of sp3 hybridized carbons
            Descriptors.HeavyAtomCount(mol),       # Number of heavy atoms
            Descriptors.NHOHCount(mol),            # Number of NH or OH groups
            Descriptors.NOCount(mol),              # Number of nitrogen or oxygen atoms
            Descriptors.NumRotatableBonds(mol),    # Number of rotatable bonds
            Descriptors.MaxPartialCharge(mol),     # Maximum partial charge on any atom
            Descriptors.MinPartialCharge(mol),     # Minimum partial charge on any atom
            Descriptors.MaxAbsPartialCharge(mol),  # Maximum absolute partial charge on any atom
            Descriptors.MinAbsPartialCharge(mol),  # Minimum absolute partial charge on any atom
        ])
        # Clip the descriptor values to avoid overflows
        descriptors = np.clip(descriptors, -1e5, 1e5)
        return descriptors
    except Exception as e:
        print(f"Error calculating descriptors: {e}")
        return None


# Function to combine ECFP features and chemical descriptors
def generate_combined_features(mol):
    ecfp = generate_ecfp(mol)
    descriptors = generate_chemical_descriptors(mol)
    if ecfp is None or descriptors is None:
        return None
    return (ecfp, descriptors)  # Ensure this is always a tuple with exactly two items

# Function to convert SMILES or protein sequence to RDKit molecule with error handling
def to_mol(smiles_or_sequence, is_protein=False):
    try:
        if pd.isna(smiles_or_sequence):
            raise ValueError(f"Invalid input: {smiles_or_sequence} (NaN detected)")

        if is_protein:
            if not isinstance(smiles_or_sequence, str):
                raise ValueError(f"Invalid protein sequence: {smiles_or_sequence}")
            mol = Chem.MolFromSequence(smiles_or_sequence)
        else:
            mol = Chem.MolFromSmiles(smiles_or_sequence)

        if mol is None:
            raise ValueError(f"Failed to create mol object from input: {smiles_or_sequence}")

        return mol
    except Exception as e:
        print(f"Error processing input: {smiles_or_sequence}. Error: {e}")
        return None


def compute_combined_similarity(features1, features2, weight_tanimoto=0.5, weight_distance=0.5):
    if features1 is None or features2 is None:
        return None

    ecfp1, desc1 = features1
    ecfp2, desc2 = features2

    # Ensure the ECFP vectors are RDKit ExplicitBitVect objects
    try:
        ecfp1 = convert_to_explicit_bitvect(ecfp1)
        ecfp2 = convert_to_explicit_bitvect(ecfp2)

        if not isinstance(ecfp1, ExplicitBitVect) or not isinstance(ecfp2, ExplicitBitVect):
            raise TypeError("ECFP features must be RDKit ExplicitBitVect objects.")

        # Compute Tanimoto similarity for the ECFP fingerprints
        tanimoto_sim = DataStructs.TanimotoSimilarity(ecfp1, ecfp2)

        # Compute Euclidean distance for the chemical descriptors
        euclidean_dist = np.linalg.norm(desc1 - desc2)

        # Normalize the Euclidean distance to a similarity measure [0, 1]
        normalized_dist = 1 / (1 + euclidean_dist)

        # Combine the two metrics using a weighted average
        combined_similarity = weight_tanimoto * tanimoto_sim + weight_distance * normalized_dist
        return combined_similarity

    except TypeError as e:
        print(f"Error in computing similarity: {e}")
        return None

def compute_random_similarity(features_dict, subset_size=100):
    keys = list(features_dict.keys())
    similarities = {}
    for key in keys:
        random_subset = random.sample(keys, min(subset_size, len(keys)))
        for other_key in random_subset:
            if key != other_key:
                similarity = compute_combined_similarity(features_dict[key], features_dict[other_key])
                if similarity is not None:
                    similarities[(key, other_key)] = similarity
                else:
                    print(f"Skipping similarity computation between {key} and {other_key} due to an error.")
        print(f"Computed similarities for {key}")
    return similarities

# Function to generate and store features for chemicals and proteins
def generate_features(file_path, subset_size=100, features_pickle='features.pkl'):
    # Check if the features have been previously generated and saved in a pickle file
    if os.path.exists(features_pickle):
        with open(features_pickle, 'rb') as f:
            saved_features = pickle.load(f)
        drug_features = saved_features.get('drug_features', {})
        protein_features = saved_features.get('protein_features', {})
        print(f"Loaded previously generated features from {features_pickle}")
    else:
        drug_features = {}
        protein_features = {}

    data = pd.read_csv(file_path)
    unique_drugs = data['chemical'].unique()
    unique_proteins = data['protein'].unique()

    # Generate and store features for each unique drug (chemical)
    for drug in unique_drugs:
        if drug not in drug_features:  # Only generate if not already saved
            mol = to_mol(drug, is_protein=False)
            if mol is not None:
                features = generate_combined_features(mol)
                if features is not None:
                    drug_features[drug] = features
                else:
                    print(f"Skipping drug due to invalid features: {drug}")

    # Generate and store features for each unique protein
    for protein in unique_proteins:
        if protein not in protein_features:  # Only generate if not already saved
            mol = to_mol(protein, is_protein=True)
            if mol is not None:
                features = generate_combined_features(mol)
                if features is not None:
                    protein_features[protein] = features
                else:
                    print(f"Skipping protein due to invalid features: {protein}")

    # Save the generated features to a pickle file
    with open(features_pickle, 'wb') as f:
        pickle.dump({'drug_features': drug_features, 'protein_features': protein_features}, f)
    print(f"Saved generated features to {features_pickle}")

    # Compute random similarities for drugs and proteins
    drug_similarities = compute_random_similarity(drug_features, subset_size)
    protein_similarities = compute_random_similarity(protein_features, subset_size)

    return drug_similarities, protein_similarities

# Function to save similarities to files
def generate_similarity_files(drug_similarities, protein_similarities, drug_output_file, protein_output_file):
    # Save drug-drug similarities to a file
    with open(drug_output_file, 'w') as f:
        f.write('drug1,drug2,similarity_score\n')
        for (drug1, drug2), similarity in drug_similarities.items():
            f.write(f'{drug1},{drug2},{similarity}\n')

    # Save protein-protein similarities to a file
    with open(protein_output_file, 'w') as f:
        f.write('protein1,protein2,similarity_score\n')
        for (protein1, protein2), similarity in protein_similarities.items():
            f.write(f'{protein1},{protein2},{similarity}\n')


# Example usage
if __name__ == "__main__":
    input_path = "../Data/BindingDB-processed/"
    file_path = input_path+'bindingdb_ic50_data.txt'  # Replace with the path to your input file
    subset_size = 100  # Number of random molecules to compare against
    features_pickle = 'features.pkl'

    # Generate features and compute similarities
    drug_similarities, protein_similarities = generate_features(file_path, subset_size, features_pickle)

    # Save the similarities to files
    output_path = "../Data/SimilarityGraphs/"
    drug_output_file = output_path + 'drug_similarity.csv'
    protein_output_file = output_path +'protein_similarity.csv'
    generate_similarity_files(drug_similarities, protein_similarities, drug_output_file, protein_output_file)

    print("Similarity computation and file generation completed.")