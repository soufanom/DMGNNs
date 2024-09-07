import numpy as np
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

class ProteinFeatureGenerator:
    def __init__(self):
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")  # Standard amino acids

    def generate_aac_features(self, protein_sequence):
        """ Generate amino acid composition (AAC) features for a protein sequence. """
        length = len(protein_sequence)

        if length == 0:
            return None

        # Count occurrences of each amino acid
        aac = Counter(protein_sequence)

        # Normalize the counts to get the composition
        aac_features = np.array([aac.get(aa, 0) / length for aa in self.amino_acids])

        return aac_features

    def generate_physicochemical_features(self, protein_sequence):
        """ Generate physicochemical properties features for a protein sequence. """
        amino_acid_properties = {
            'A': [1.8, 89.09, 6.00],  # [Hydrophobicity, Molecular Weight, Isoelectric Point]
            'C': [2.5, 121.16, 5.07],
            'D': [-3.5, 133.10, 2.77],
            'E': [-3.5, 147.13, 3.22],
            'F': [2.8, 165.19, 5.48],
            'G': [-0.4, 75.07, 5.97],
            'H': [-3.2, 155.16, 7.59],
            'I': [4.5, 131.17, 6.02],
            'K': [-3.9, 146.19, 9.74],
            'L': [3.8, 131.17, 5.98],
            'M': [1.9, 149.21, 5.74],
            'N': [-3.5, 132.12, 5.41],
            'P': [-1.6, 115.13, 6.30],
            'Q': [-3.5, 146.15, 5.65],
            'R': [-4.5, 174.20, 10.76],
            'S': [-0.8, 105.09, 5.68],
            'T': [-0.7, 119.12, 5.60],
            'V': [4.2, 117.15, 5.96],
            'W': [-0.9, 204.23, 5.89],
            'Y': [-1.3, 181.19, 5.66],
        }

        # Convert the amino acid properties into a vector for the whole sequence
        property_vectors = np.array([amino_acid_properties.get(aa, [0, 0, 0]) for aa in protein_sequence])

        # Apply Min-Max scaling to scale the features between 0 and 1
        scaler = MinMaxScaler()
        scaled_properties = scaler.fit_transform(property_vectors)

        # Average the properties for the whole sequence
        avg_properties = np.mean(scaled_properties, axis=0)

        return avg_properties

    def generate_onehot_features(self, protein_sequence):
        """ Generate one-hot encoding features for a protein sequence. """
        encoder = OneHotEncoder(categories=[self.amino_acids], sparse_output=False)
        protein_array = np.array(list(protein_sequence)).reshape(-1, 1)
        one_hot_encoded = encoder.fit_transform(protein_array)
        return one_hot_encoded.flatten()

    def generate_combined_features(self, protein_sequence):
        """ Generates a combined feature vector for a protein sequence using multiple methods:
            - One-hot encoding
            - Amino acid composition (AAC)
            - Average physicochemical properties
        """
        try:
            # One-hot encoding
            one_hot_encoded = self.generate_onehot_features(protein_sequence)

            # Amino acid composition (AAC)
            aac_features = self.generate_aac_features(protein_sequence)

            # Physicochemical properties
            physicochemical_features = self.generate_physicochemical_features(protein_sequence)

            # Concatenate all features into a single combined feature vector
            combined_features = np.concatenate([one_hot_encoded, aac_features, physicochemical_features])

            return combined_features

        except Exception as e:
            print(f"Error generating combined features for protein: {e}")
            return None
