# main.py
import numpy as np
import random
from feature_similarity_computer import FeatureSimilarityComputer


def main():
    np.random.seed(42)
    random.seed(42)

    # Initialize the FeatureSimilarityComputer
    similarity_computer = FeatureSimilarityComputer()

    # File paths
    input_path = "../Data/BindingDB-processed/"
    file_path = input_path + 'bindingdb_ic50_data.txt'
    output_path = "../Data/SimilarityGraphs/"

    # Generate features and compute similarities
    drug_similarities, protein_similarities = similarity_computer.generate_features(
        file_path, subset_size=200, features_pickle='features_prot.pkl', use_sequence_features=True
    )

    # Save the similarities to files
    drug_output_file = output_path + 'drug_similarity_no_need.csv'
    protein_output_file = output_path + 'protein_similarity_prot_features.csv'
    similarity_computer.generate_similarity_files(drug_similarities, protein_similarities, drug_output_file,
                                                  protein_output_file)

if __name__ == "__main__":
    main()
    # advantage of similarity is weighting using two functions rebalancing features effect


