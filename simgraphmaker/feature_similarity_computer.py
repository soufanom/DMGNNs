import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs
import pandas as pd
import numpy as np
import random
import pickle
import os
from rdkit.DataStructs import ExplicitBitVect
from proteins.protfeaturegen import ProteinFeatureGenerator  # Import the ProteinFeatureGenerator class


class FeatureSimilarityComputer:
    def __init__(self):
        self.protein_feature_generator = ProteinFeatureGenerator()  # Initialize the ProteinFeatureGenerator

    def convert_to_explicit_bitvect(self, ecfp):
        """Attempt to convert a numpy array back to an RDKit ExplicitBitVect."""
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

    def generate_ecfp(self, mol, radius=2, n_bits=1024):
        """Generate ECFP (Morgan Fingerprints) for a molecule."""
        if mol is None:
            return None
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(ecfp)

    def generate_chemical_descriptors(self, mol):
        """Generate an expanded set of chemical descriptors for a molecule."""
        if mol is None:
            return None
        try:
            descriptors = np.array([
                Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol), Descriptors.TPSA(mol), Descriptors.RingCount(mol),
                Descriptors.FractionCSP3(mol), Descriptors.HeavyAtomCount(mol), Descriptors.NHOHCount(mol),
                Descriptors.NOCount(mol), Descriptors.NumRotatableBonds(mol), Descriptors.MaxPartialCharge(mol),
                Descriptors.MinPartialCharge(mol), Descriptors.MaxAbsPartialCharge(mol), Descriptors.MinAbsPartialCharge(mol),
            ])
            descriptors = np.clip(descriptors, -1e5, 1e5)
            return descriptors
        except Exception as e:
            print(f"Error calculating descriptors: {e}")
            return None

    def generate_combined_features(self, mol):
        """Combine ECFP features and chemical descriptors."""
        ecfp = self.generate_ecfp(mol)
        descriptors = self.generate_chemical_descriptors(mol)
        if ecfp is None or descriptors is None:
            return None
        return (ecfp, descriptors)

    def to_mol(self, smiles_or_sequence, is_protein=False, use_sequence_features=False):
        """Convert SMILES or protein sequence to RDKit molecule or generate protein features."""
        try:
            if pd.isna(smiles_or_sequence):
                raise ValueError(f"Invalid input: {smiles_or_sequence} (NaN detected)")

            if is_protein:
                if use_sequence_features:
                    return self.protein_feature_generator.generate_combined_features(smiles_or_sequence)
                else:
                    mol = Chem.MolFromSequence(smiles_or_sequence)
            else:
                mol = Chem.MolFromSmiles(smiles_or_sequence)

            if mol is None:
                raise ValueError(f"Failed to create mol object from input: {smiles_or_sequence}")

            return mol
        except Exception as e:
            print(f"Error processing input: {smiles_or_sequence}. Error: {e}")
            return None

    def compute_combined_similarity(self, features1, features2, weight_tanimoto=0.5, weight_distance=0.5):
        """Compute combined similarity between two feature sets."""
        if features1 is None or features2 is None:
            return None

        ecfp1, desc1 = features1
        ecfp2, desc2 = features2

        try:
            ecfp1 = self.convert_to_explicit_bitvect(ecfp1)
            ecfp2 = self.convert_to_explicit_bitvect(ecfp2)

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

    def compute_random_similarity(self, features_dict, subset_size=100):
        """Compute random similarities between feature sets."""
        keys = list(features_dict.keys())
        similarities = {}
        for key in keys:
            random_subset = random.sample(keys, min(subset_size, len(keys)))
            for other_key in random_subset:
                if key != other_key:
                    similarity = self.compute_combined_similarity(features_dict[key], features_dict[other_key])
                    if similarity is not None:
                        similarities[(key, other_key)] = similarity
                    else:
                        print(f"Skipping similarity computation between {key} and {other_key} due to an error.")
            print(f"Computed similarities for {key}")
        return similarities

    def generate_features(self, file_path, subset_size=100, features_pickle='features.pkl', use_sequence_features=False):
        """Generate and store features for chemicals and proteins."""
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

        for drug in unique_drugs:
            if drug not in drug_features:
                mol = self.to_mol(drug, is_protein=False)
                if mol is not None:
                    features = self.generate_combined_features(mol)
                    if features is not None:
                        drug_features[drug] = features
                    else:
                        print(f"Skipping drug due to invalid features: {drug}")

        for protein in unique_proteins:
            if protein not in protein_features:
                mol = self.to_mol(protein, is_protein=True, use_sequence_features=use_sequence_features)
                if mol is not None:
                    if use_sequence_features:
                        protein_features[protein] = (None, mol)
                    else:
                        features = self.generate_combined_features(mol)
                        if features is not None:
                            protein_features[protein] = features
                        else:
                            print(f"Skipping protein due to invalid features: {protein}")

        with open(features_pickle, 'wb') as f:
            pickle.dump({'drug_features': drug_features, 'protein_features': protein_features}, f)
        print(f"Saved generated features to {features_pickle}")

        drug_similarities = self.compute_random_similarity(drug_features, subset_size)
        protein_similarities = self.compute_random_similarity(protein_features, subset_size)

        return drug_similarities, protein_similarities

    def generate_similarity_files(self, drug_similarities, protein_similarities, drug_output_file, protein_output_file):
        """Save similarities to files."""
        with open(drug_output_file, 'w') as f:
            f.write('drug1,drug2,similarity_score\n')
            for (drug1, drug2), similarity in drug_similarities.items():
                f.write(f'{drug1},{drug2},{similarity}\n')

        with open(protein_output_file, 'w') as f:
            f.write('protein1,protein2,similarity_score\n')
            for (protein1, protein2), similarity in protein_similarities.items():
                f.write(f'{protein1},{protein2},{similarity}\n')
