import pandas as pd
import numpy as np


def process_similarity_file(file_path, threshold_method='percentile', percentile=90):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Drop rows with any NaN values
    df = df.dropna()

    # Determine if the file is for drugs or proteins based on the column names
    if {'drug1', 'drug2', 'similarity_score'}.issubset(df.columns):
        print("Processing drug similarity file...")
        entity1_col = 'drug1'
        entity2_col = 'drug2'
        file_type = 'drug'
    elif {'protein1', 'protein2', 'similarity_score'}.issubset(df.columns):
        print("Processing protein similarity file...")
        entity1_col = 'protein1'
        entity2_col = 'protein2'
        file_type = 'protein'
    else:
        raise ValueError(
            "Input file must contain either 'drug1', 'drug2', 'similarity_score' columns or 'protein1', 'protein2', 'similarity_score' columns.")

    # Process each row
    for index, row in df.iterrows():
        entity1 = row[entity1_col]
        entity2 = row[entity2_col]
        similarity_score = row['similarity_score']

        # You can add any processing you need here
        #print(f"Processing similarity between {entity1} and {entity2} with score {similarity_score}")

    # Calculate a threshold for the similarity scores
    threshold = calculate_threshold(df['similarity_score'], method=threshold_method, percentile=percentile)
    print(f"Calculated threshold for similarity scores: {threshold}")

    # Calculate statistics
    above_threshold = (df['similarity_score'] > threshold).sum()
    below_threshold = (df['similarity_score'] <= threshold).sum()

    # Report statistics
    print(f"Number of values above the threshold: {above_threshold}")
    print(f"Number of values below or equal to the threshold: {below_threshold}")

    # Store the threshold in a file
    output_filename = f"{file_type}_{threshold_method}.txt"
    with open(output_filename, 'w') as f:
        f.write(f"Threshold calculated using {threshold_method} method: {threshold}\n")
        f.write(f"Number of values above the threshold: {above_threshold}\n")
        f.write(f"Number of values below or equal to the threshold: {below_threshold}\n")

    return threshold


def calculate_threshold(similarity_scores, method='percentile', percentile=90):
    if method == 'percentile':
        # Use percentile-based threshold
        threshold = np.percentile(similarity_scores, percentile)
    elif method == 'mean_std':
        # Use mean + k * std as threshold
        threshold = similarity_scores.mean() + similarity_scores.std()
    else:
        raise ValueError("Unsupported threshold method. Choose 'percentile' or 'mean_std'.")

    return threshold


# Example usage
if __name__ == "__main__":
    file_path = '../Data/SimilarityGraphs/drug_similarity.csv'  # Replace with the path to your input file
    threshold_method = 'percentile'  # Choose between 'percentile' and 'mean_std'
    percentile = 90  # Relevant if 'percentile' method is chosen

    threshold = process_similarity_file(file_path, threshold_method=threshold_method, percentile=percentile)
    print(f"Threshold determined from the file: {threshold}")
