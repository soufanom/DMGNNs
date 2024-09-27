# main.py
import numpy as np
import random
from feature_similarity_computer import FeatureSimilarityComputer


def main(input_file, drug_output_file_name, protein_output_file_name, pickle_file_name):
    np.random.seed(42)
    random.seed(42)

    # Initialize the FeatureSimilarityComputer
    similarity_computer = FeatureSimilarityComputer()

    # Generate features and compute similarities
    drug_similarities, protein_similarities = similarity_computer.generate_features(
        input_file, drug_output_file_name, protein_output_file_name, subset_size=100, features_pickle=pickle_file_name, use_sequence_features=False
    )

    # # Save the similarities to files
    # similarity_computer.generate_similarity_files(drug_similarities, protein_similarities, drug_output_file_name,
    #                                               protein_output_file_name)

if __name__ == "__main__":
    # File paths
    input_path = "../Data/StitchString/"
    input_file = input_path + 'stitch-data.csv'
    output_path = "../Data/SimilarityGraphs/"
    drug_output_file_name = output_path + 'drug_similarity.csv'
    protein_output_file_name = output_path + 'protein_similarity.csv'

    pickle_file_name = "features.pkl"

    main(input_file, drug_output_file_name, protein_output_file_name, pickle_file_name)
    # advantage of similarity is weighting using two functions rebalancing features effect


