import torch
from rdkit import Chem
import pandas as pd
import numpy as np


# Utility functions to featurize atoms and bonds
def atom_features(atom):
    """Generate atom features: atomic number, degree, total valence"""
    return np.array([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetTotalValence(),
    ], dtype=np.float32)


def bond_features(bond):
    """Generate bond features: bond type, is conjugated"""
    bt = bond.GetBondType()
    return np.array([
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
    ], dtype=np.float32)


# Function to convert protein sequences to SMILES and then to a molecular graph
def protein_to_smiles(protein_sequence):
    try:
        # Ensure the sequence is a string
        if not isinstance(protein_sequence, str):
            protein_sequence = str(protein_sequence)

        # Convert protein sequence to RDKit molecule
        mol = Chem.MolFromSequence(protein_sequence)
        if mol is None:
            raise ValueError(f"Failed to create mol object from protein sequence: {protein_sequence}")

        # Convert molecule to SMILES
        smiles = Chem.MolToSmiles(mol)
        return smiles

    except Exception as e:
        print(f"Error processing protein sequence: {protein_sequence}. Error: {e}")
        return None


# Function to find the maximum number of atoms and bonds for chemicals only
def find_max_atoms_bonds(smiles_list):
    max_atoms = 0
    max_bonds = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            num_atoms = len(mol.GetAtoms())
            num_bonds = len(mol.GetBonds())
            if num_atoms > max_atoms:
                max_atoms = num_atoms
            if num_bonds > max_bonds:
                max_bonds = num_bonds
    return max_atoms, max_bonds


# Function to convert SMILES to a molecular graph with dynamic padding
def smiles_to_graph(smiles, max_atoms, max_bonds):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")

    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    atom_features_list = [atom_features(atom) for atom in atoms]
    bond_features_list = [bond_features(bond) for bond in bonds]

    # Pad atom and bond features to the maximum length
    atom_features_padded = np.zeros((max_atoms, atom_features_list[0].shape[0]), dtype=np.float32)
    bond_features_padded = np.zeros((max_bonds, bond_features_list[0].shape[0]), dtype=np.float32)

    atom_features_padded[:len(atom_features_list), :] = atom_features_list
    bond_features_padded[:len(bond_features_list), :] = bond_features_list

    # For edge_index, we don't pad but instead handle it separately
    edge_index = []
    for bond in bonds:
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    return torch.tensor(atom_features_padded, dtype=torch.float), torch.tensor(edge_index,
                                                                               dtype=torch.long).t().contiguous(), torch.tensor(
        bond_features_padded, dtype=torch.float)


# Function to generate and store features for chemicals and proteins
def generate_features(file_path):
    data = pd.read_csv(file_path)
    unique_drugs = data['chemical'].unique()
    unique_proteins = data['protein'].unique()

    # Convert protein sequences to SMILES strings
    protein_smiles = [protein_to_smiles(protein) for protein in unique_proteins]
    protein_smiles = [smiles for smiles in protein_smiles if smiles is not None]  # Filter out None values

    # Combine drug and protein SMILES for finding max atoms and bonds
    all_smiles = list(unique_drugs) + protein_smiles

    # Find the maximum number of atoms and bonds in the combined dataset
    max_atoms, max_bonds = find_max_atoms_bonds(all_smiles)

    # Create dictionaries to store features
    drug_features = {}
    protein_features = {}

    # Generate features for each unique drug (chemical)
    for drug in unique_drugs:
        atom_features, edge_index, edge_attr = smiles_to_graph(drug, max_atoms, max_bonds)
        features = torch.cat([atom_features.mean(dim=0), edge_attr.mean(dim=0)], dim=0).numpy()
        drug_features[drug] = features

    # Generate features for each unique protein (using SMILES converted from sequence)
    for protein, smiles in zip(unique_proteins, protein_smiles):
        atom_features, edge_index, edge_attr = smiles_to_graph(smiles, max_atoms, max_bonds)
        features = torch.cat([atom_features.mean(dim=0), edge_attr.mean(dim=0)], dim=0).numpy()
        protein_features[protein] = features

    return drug_features, protein_features

# Example usage
file_path = '../Data-preparation/BindingDB-processed/output_data.txt'  # Replace with the path to your input file
drug_features, protein_features = generate_features(file_path)


# Function to calculate similarity between feature vectors using batch processing
def batch_compute_similarity(features, batch_size=1000):
    num_features = len(features)
    feature_matrix = np.array(list(features.values()))
    similarity_scores = []

    for i in range(0, num_features, batch_size):
        batch_features = feature_matrix[i:i + batch_size]
        sim_matrix = np.dot(batch_features, feature_matrix.T) / (
                np.linalg.norm(batch_features, axis=1, keepdims=True) *
                np.linalg.norm(feature_matrix, axis=1, keepdims=True).T
        )

        for j in range(batch_features.shape[0]):
            for k in range(i + j + 1, num_features):
                similarity_scores.append((i + j, k, sim_matrix[j, k - i]))

    return similarity_scores


# Function to generate and save similarity files without applying any threshold
def generate_similarity_files(drug_features, protein_features, drug_output_file, protein_output_file, batch_size=1000):
    # Generate drug-drug similarity file
    with open(drug_output_file, 'w') as f:
        f.write('drug1,drug2,similarity_score\n')
        drugs = list(drug_features.keys())
        drug_similarities = batch_compute_similarity(drug_features, batch_size)
        for idx1, idx2, sim in drug_similarities:
            f.write(f'{drugs[idx1]},{drugs[idx2]},{sim}\n')

    # Generate protein-protein similarity file
    with open(protein_output_file, 'w') as f:
        f.write('protein1,protein2,similarity_score\n')
        proteins = list(protein_features.keys())
        protein_similarities = batch_compute_similarity(protein_features, batch_size)
        for idx1, idx2, sim in protein_similarities:
            f.write(f'{proteins[idx1]},{proteins[idx2]},{sim}\n')

# Example usage
output_path = "../Data-preparation/SimilarityGraphs/"
drug_output_file = output_path+'drug_similarity.csv'
protein_output_file = output_path+'protein_similarity.csv'
generate_similarity_files(drug_features, protein_features, drug_output_file, protein_output_file, batch_size=1000)
