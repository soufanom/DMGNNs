import requests

# Caching dictionary for protein sequences
protein_cache = {}

# Function to retrieve a single protein sequence from UniProt
def get_protein_sequence(uniprot_id):
    # Check if the sequence is already cached
    if uniprot_id in protein_cache:
        return protein_cache[uniprot_id]

    # Fetch the sequence from UniProt if not cached
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta_data = response.text.strip().split('\n')
        sequence = ''.join(fasta_data[1:])  # Skip the header and join the sequence lines
        protein_cache[uniprot_id] = sequence  # Cache the sequence
        return sequence
    else:
        print(f"Error {response.status_code} retrieving data from UniProt for ID {uniprot_id}")
        return "-1"

# Main function to process files with caching
def process_files_with_smiles_dict(input_path, file_list, output_file, smiles_dict):
    results = []
    for file_name in file_list:
        label = 1 if file_name.endswith(".pos") else 0
        with open(input_path+file_name, 'r') as infile:
            lines = infile.readlines()

        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 4 and parts[0] == 'chem' and parts[2] == 'protein':
                pubchem_id = parts[1]
                uniprot_id = parts[3]
                smiles = smiles_dict.get(pubchem_id)
                if not smiles:
                    smiles = get_smiles_from_api(pubchem_id)
                protein_sequence = get_protein_sequence(uniprot_id)  # Retrieve or cache the sequence
                if protein_sequence != "-1" and smiles != "-1":
                    results.append((smiles, protein_sequence, label))
                print(len(results))

    with open(output_file, 'w') as outfile:
        for smiles, protein_sequence, label in results:
            outfile.write(f"{smiles},{protein_sequence},{label}\n")

# Function to retrieve SMILES for a given PubChem ID using the PubChem API
def get_smiles_from_api(pubchem_id):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pubchem_id}/property/CanonicalSMILES/JSON"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            smiles = response.json()['PropertyTable']['Properties'][0]['CanonicalSMILES']
            return smiles
        except (KeyError, IndexError):
            return "SMILES not found"
    else:
        return "-1"

# Function to build the dictionary from chem-1.txt and chem-repr-1.repr
def build_smiles_dict(chem_file, repr_file):
    smiles_dict = {}
    with open(chem_file, 'r') as chem_f, open(repr_file, 'r') as repr_f:
        chem_lines = chem_f.readlines()
        repr_lines = repr_f.readlines()

        for chem_line, repr_line in zip(chem_lines, repr_lines):
            pubchem_id = chem_line.strip()
            smiles = repr_line.strip()
            smiles_dict[pubchem_id] = smiles

    return smiles_dict

# Build the SMILES dictionary
input_path = "../Data-preparation/BindingDB/"
chem_file = input_path+'chem-1.txt'
repr_file = input_path+'chem-repr-1.repr'
smiles_dict = build_smiles_dict(chem_file, repr_file)

# List of files to process
file_list = [
    'dev-edges.neg',
    'dev-edges.pos',
    'train-edges.neg',
    'train-edges.pos',
    'test-edges.neg',
    'test-edges.pos'
]

# Specify the output file
output_path = "../Data-preparation/BindingDB-processed/"
output_file = output_path+'output_data.txt'

# Run the processing function using the SMILES dictionary
process_files_with_smiles_dict(input_path, file_list, output_file, smiles_dict)

print(f"Results have been written to {output_file}")
