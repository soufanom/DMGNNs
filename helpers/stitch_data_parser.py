import csv
import requests
import os

# Global dictionary to store protein sequences
protein_sequence_cache = {}


# Function to retrieve protein sequence from Ensembl using the Ensembl REST API
def get_protein_sequence(ensembl_id):
    # Check if the sequence is already in the cache
    if ensembl_id in protein_sequence_cache:
        return protein_sequence_cache[ensembl_id]

    # If not in cache, make the API call
    url = f"https://rest.ensembl.org/sequence/id/{ensembl_id}?content-type=text/plain"
    response = requests.get(url)
    if response.status_code == 200:
        sequence = response.text.strip()
        protein_sequence_cache[ensembl_id] = sequence  # Store the sequence in the cache
        return sequence
    else:
        print(f"Failed to retrieve sequence for {ensembl_id}")
        return None


# Function to read SMILES from data.smiles file
def read_smiles(file_path):
    with open(file_path, 'r') as file:
        smiles_list = [line.strip() for line in file]
    return smiles_list


# Function to parse data.csv and map drug ids to protein ids and labels
# Function to parse data.csv and map drug ids to protein ids and labels, skipping the header
def parse_csv(file_path):
    drug_protein_label_list = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row (chemical, protein, label)
        for row in reader:
            drug_protein_label_list.append((row[0], row[1], row[2]))
    return drug_protein_label_list


# Main function to process the input and generate output file
def generate_output(data_csv, smiles_file, output_file):
    # Read SMILES from file
    smiles_list = read_smiles(smiles_file)

    # Parse the data.csv file
    drug_protein_label_list = parse_csv(data_csv)

    # Get unique protein ids
    unique_protein_ids = set([item[1] for item in drug_protein_label_list])

    # Retrieve protein sequences for all unique protein ids using cache
    protein_sequences = {}
    for protein_id in unique_protein_ids:
        sequence = get_protein_sequence(protein_id)
        if sequence:
            protein_sequences[protein_id] = sequence

    # Generate output file, skip records where protein sequence is None
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['SMILEs', 'Protein Sequence', 'Label'])  # Write the header

        for i, (drug_id, protein_id, label) in enumerate(drug_protein_label_list):
            smiles = smiles_list[i]  # Corresponding SMILES for the drug_id
            protein_sequence = protein_sequences.get(protein_id)

            # Skip writing the record if protein sequence is not found
            if protein_sequence:
                writer.writerow([smiles, protein_sequence, label])

    print(f"Output file '{output_file}' generated successfully!")


# Function to process multiple folders (cv_0, cv_1, cv_2, cv_3, cv_4)
def process_folders(base_path):
    for i in range(5):
        folder = f"cv_{i}"
        folder_path = os.path.join(base_path, folder)

        # Train files
        data_csv_train = os.path.join(folder_path, 'train.csv')
        smiles_file_train = os.path.join(folder_path, 'train.smiles')
        output_file_train = os.path.join(folder_path, 'stitch-data-tr.csv')
        generate_output(data_csv_train, smiles_file_train, output_file_train)

        # Test files
        data_csv_test = os.path.join(folder_path, 'test.csv')
        smiles_file_test = os.path.join(folder_path, 'test.smiles')
        output_file_test = os.path.join(folder_path, 'stitch-data-ts.csv')
        generate_output(data_csv_test, smiles_file_test, output_file_test)


# Example usage
base_path = '../Data/StitchString'
process_folders(base_path)

