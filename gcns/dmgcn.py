import torch
import numpy as np
import random
import pandas as pd
import pickle
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch import nn, optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Function to set the seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GCN(nn.Module):
    def __init__(self, num_node_features, hidden_dims=[128, 256, 128, 64, 32], dropout_rate=0.5):
        super(GCN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()  # Batch normalization layers
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout layer
        
        # First GCN layer
        self.convs.append(GCNConv(num_node_features, hidden_dims[0]))
        self.bns.append(nn.BatchNorm1d(hidden_dims[0]))

        # Hidden GCN layers
        for i in range(1, len(hidden_dims)):
            self.convs.append(GCNConv(hidden_dims[i-1], hidden_dims[i]))
            self.bns.append(nn.BatchNorm1d(hidden_dims[i]))

        # Fully connected layer
        self.fc = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x, edge_index, edge_weight=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_weight)
            x = bn(x)  # Apply batch normalization
            x = F.relu(x)
            x = self.dropout(x)  # Apply dropout

        # Get the embeddings for the nodes in the edges
        edge_embeddings = x[edge_index[0]] * x[edge_index[1]]  # Element-wise product of the embeddings of the two nodes in each edge
        out = self.fc(edge_embeddings)
        return torch.sigmoid(out)

def train_gcn(gcn, data, labels, epochs=300, learning_rate=0.01):
    optimizer = optim.Adam(gcn.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    gcn.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = gcn(data.x, data.edge_index, data.edge_attr)

        # Ensure output shape matches labels shape
        out = out.view(-1)  # Flatten output if necessary
        #print(f"Output values at epoch {epoch+1}: {out}")  # Debugging: Check output values

        # Compute loss
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    return gcn

# Shuffle and split the dataset
def shuffle_and_split_data(file_path, seed=42, test_size=0.2):
    df = pd.read_csv(file_path)
    #df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle the data
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    return train_df, test_df


# Load embeddings
def load_embeddings(embedding_file):
    with open(embedding_file, 'rb') as f:
        embeddings = pickle.load(f, encoding='latin1')
    return embeddings


# Load original features
def load_original_features(pickle_file):
    with open(pickle_file, 'rb') as f:
        features = pickle.load(f)
    return features['drug_features'], features['protein_features']

def build_graph(df, drug_features, protein_features, is_embedding=False):
    edge_index = []
    edge_weight = []
    labels = []

    node_features = {}

    for _, row in df.iterrows():
        chemical = row['chemical']
        protein = row['protein']
        label = row['label']

        # Initialize features as None
        chemical_features = None
        protein_features_combined = None

        if chemical in drug_features:
            if is_embedding:
                chemical_features = drug_features[chemical]  # Use embeddings directly
            else:
                ecfp, desc = drug_features[chemical]  # Original features need to be concatenated
                chemical_features = np.concatenate((ecfp, desc))

        if protein in protein_features:
            if is_embedding:
                protein_features_combined = protein_features[protein]  # Use embeddings directly
            else:
                ecfp, desc = protein_features[protein]  # Original features need to be concatenated
                protein_features_combined = np.concatenate((ecfp, desc))

        # Check and handle NaN values for both chemical and protein features
        if chemical_features is not None and np.isnan(chemical_features).any():
            print(f"Warning: NaN values found in features for chemical: {chemical}")
            continue
        if protein_features_combined is not None and np.isnan(protein_features_combined).any():
            print(f"Warning: NaN values found in features for protein: {protein}")
            continue

        # Add features to node_features
        if chemical_features is not None:
            node_features[chemical] = chemical_features
        if protein_features_combined is not None:
            node_features[protein] = protein_features_combined

        # Only add edge if both chemical and protein features are valid
        if chemical in node_features and protein in node_features:
            edge_index.append([chemical, protein])
            edge_weight.append(1.0)  # Assuming all edges have the same weight
            labels.append(label)

    # Check if any edges were created
    if len(edge_index) == 0:
        raise ValueError("No valid edges were created. Check the input data and feature processing.")

    # Create mapping from entity to index
    node_mapping = {entity: i for i, entity in enumerate(node_features.keys())}

    # Convert node features to tensor
    node_features_tensor = torch.tensor([node_features[node] for node in node_mapping.keys()], dtype=torch.float)

    # Convert edge indices to tensor
    edge_index_tensor = torch.tensor(
        [[node_mapping[chemical], node_mapping[protein]] for chemical, protein in edge_index],
        dtype=torch.long).t().contiguous()
    edge_weight_tensor = torch.tensor(edge_weight, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.float)

    return Data(x=node_features_tensor, edge_index=edge_index_tensor, edge_attr=edge_weight_tensor), labels_tensor


# Function to handle NaN values by imputing the mean
def handle_nan_values(matrix):
    imputer = SimpleImputer(strategy='mean')
    return imputer.fit_transform(matrix)


# Convert the feature dictionary to two separate matrices: one for ecfp and one for desc
def dict_to_matrix_separate(feature_dict):
    keys = list(feature_dict.keys())
    ecfp_matrix = []
    desc_matrix = []

    for key in keys:
        if isinstance(feature_dict[key], tuple):  # If the feature is a tuple, split into ecfp and desc
            ecfp, desc = feature_dict[key]
            ecfp_matrix.append(ecfp)
            desc_matrix.append(desc)

    return np.vstack(ecfp_matrix), np.vstack(desc_matrix), keys


# Convert the reduced matrices back to a dictionary with tuple (reduced_ecfp, reduced_desc)
def matrix_to_dict_separate(reduced_ecfp_matrix, reduced_desc_matrix, keys):
    reduced_dict = {key: (reduced_ecfp_matrix[i], reduced_desc_matrix[i]) for i, key in enumerate(keys)}
    return reduced_dict


# Apply PCA separately to ecfp and desc, handling NaN values
def apply_pca_to_dict_separate(feature_dict, n_components=128):
    ecfp_matrix, desc_matrix, keys = dict_to_matrix_separate(feature_dict)

    # Handle NaN values
    ecfp_matrix = handle_nan_values(ecfp_matrix)
    desc_matrix = handle_nan_values(desc_matrix)

    # Apply PCA separately to ecfp and desc
    pca_ecfp = PCA(n_components=n_components)

    reduced_ecfp_matrix = pca_ecfp.fit_transform(ecfp_matrix)

    # Convert the reduced matrices back into a dictionary with tuple (ecfp, desc)
    reduced_dict = matrix_to_dict_separate(reduced_ecfp_matrix, desc_matrix, keys)

    return reduced_dict

def normalize_features(features):
    scaler = MinMaxScaler()
    normalized_features = {}
    for key, value in features.items():
        if isinstance(value, tuple):  # If the feature is a tuple, normalize each part separately
            ecfp, desc = value
            # Flatten or handle the concatenation carefully
            normalized_ecfp = scaler.fit_transform(ecfp.reshape(-1, 1)).flatten() if ecfp.ndim == 1 else scaler.fit_transform(ecfp)
            normalized_desc = scaler.fit_transform(desc.reshape(-1, 1)).flatten() if desc.ndim == 1 else scaler.fit_transform(desc)
            
            # Ensure both are of the same shape before concatenation
            if normalized_ecfp.ndim == 1:
                normalized_ecfp = normalized_ecfp.reshape(1, -1)
            if normalized_desc.ndim == 1:
                normalized_desc = normalized_desc.reshape(1, -1)
            
            # Concatenate along the feature axis (usually axis=1)
            combined_normalized_features = np.concatenate((normalized_ecfp, normalized_desc), axis=1).flatten()
            normalized_features[key] = combined_normalized_features
        else:
            normalized_features[key] = scaler.fit_transform(value.reshape(-1, 1)).flatten() if value.ndim == 1 else scaler.fit_transform(value)
    return normalized_features

def evaluate_gcn(gcn, data, labels):
    gcn.eval()
    with torch.no_grad():
        # Forward pass: Get the predicted similarity scores
        out = gcn(data.x, data.edge_index, data.edge_attr)
        out = out.view(-1)  # Flatten output

        # Convert the raw output to probabilities using sigmoid
        predicted_probabilities = torch.sigmoid(out).detach().cpu().numpy()

        # Threshold at 0.5 for binary classification
        # predictions = (predicted_probabilities >= 0.5).astype(float)
        predictions = (out >= 0.5).float()

        # Convert labels to numpy
        labels_np = labels.detach().cpu().numpy()

        # Accuracy metrics
        correct = (predictions == labels_np).sum()
        accuracy = correct / labels_np.size
        precision = (predictions * labels_np).sum() / predictions.sum()
        recall = (predictions * labels_np).sum() / labels_np.sum()
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)

        # AUC calculation using roc_auc_score from sklearn
        auc = roc_auc_score(labels_np, predicted_probabilities)

        # Print evaluation metrics
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}, AUC: {auc:.4f}")

    return precision, recall, f1_score, auc


# Main script
if __name__ == "__main__":
    # File paths
    data_file = '/home/o.soufan/DMGNNs/Data/BindingDB-processed/bindingdb_ic50_data.txt'
    drug_embedding_file = 'drug_embeddings_prot.pkl'
    protein_embedding_file = 'protein_embeddings_prot.pkl'
    feature_file = '/home/o.soufan/DMGNNs/simgraphmaker/features_prot.pkl'

    # Shuffle and split the data
    train_df, test_df = shuffle_and_split_data(data_file)

    # Load embeddings and original features
    drug_embeddings = load_embeddings(drug_embedding_file)
    protein_embeddings = load_embeddings(protein_embedding_file)
    drug_features, protein_features = load_original_features(feature_file)

    # Apply PCA to reduce original features to 128 dimensions for both ecfp and desc
    drug_features = apply_pca_to_dict_separate(drug_features, n_components=113)
    protein_features = apply_pca_to_dict_separate(protein_features, n_components=105)

    # Determine the size of the node features
    embedding_feature_size = next(iter(drug_embeddings.values())).shape[0]
    original_feature_size = sum(map(lambda x: x.shape[0], next(iter(drug_features.values()))))

    # Ensure that both GCNs have the same size of node features
    assert embedding_feature_size == original_feature_size, "Node features must have the same size for both GCNs."

    # Normalize features
    drug_features_normalized = normalize_features(drug_features)
    protein_features_normalized = normalize_features(protein_features)

    # Normalize embeddings
    drug_embeddings_normalized = normalize_features(drug_embeddings)
    protein_embeddings_normalized = normalize_features(protein_embeddings)

    # Build graphs using embeddings (no concatenation)
    data_embeddings, labels_embeddings = build_graph(train_df, drug_embeddings, protein_embeddings, is_embedding=True)

    # Build graphs using original features (with concatenation)
    data_original, labels_original = build_graph(train_df, drug_features, protein_features, is_embedding=False)

    # Initialize GCN models
    seed = 42
    set_seed(seed)
    gcn_embeddings = GCN(num_node_features=data_embeddings.x.size(1))
    gcn_original = GCN(num_node_features=data_original.x.size(1))

    # Train GCN models
    print("Training GCN with embeddings...")
    set_seed(seed)
    gcn_embeddings = train_gcn(gcn_embeddings, data_embeddings, labels_embeddings)

    set_seed(seed)
    print("Training GCN with original features...")
    gcn_original = train_gcn(gcn_original, data_original, labels_original)

    # Evaluate GCN models on test data
    print("Evaluating GCN with embeddings on test data...")
    test_data_embeddings, test_labels_embeddings = build_graph(test_df, drug_embeddings, protein_embeddings,
                                                               is_embedding=True)
    precision_emb, recall_emb, f1_emb, auc_emb = evaluate_gcn(gcn_embeddings, test_data_embeddings,
                                                              test_labels_embeddings)

    print("Evaluating GCN with original features on test data...")
    test_data_original, test_labels_original = build_graph(test_df, drug_features, protein_features, is_embedding=False)
    precision_orig, recall_orig, f1_orig, auc_orig = evaluate_gcn(gcn_original, test_data_original,
                                                                  test_labels_original)

    # Build graphs using normalized original features
    data_normalized_original, labels_normalized_original = build_graph(train_df, drug_features_normalized,
                                                                       protein_features_normalized, is_embedding=True)

    # Build graphs using normalized embeddings
    data_normalized_embeddings, labels_normalized_embeddings = build_graph(train_df, drug_embeddings_normalized,
                                                                           protein_embeddings_normalized,
                                                                           is_embedding=True)

    # Initialize GCN models for normalized data
    set_seed(seed)
    gcn_norm_embedded = GCN(num_node_features=data_normalized_embeddings.x.size(1))
    gcn_norm_original = GCN(num_node_features=data_normalized_original.x.size(1))

    # Train GCN models on normalized data
    print("Training GCN with normalized embeddings...")
    set_seed(seed)
    gcn_norm_embedded = train_gcn(gcn_norm_embedded, data_normalized_embeddings, labels_normalized_embeddings)

    print("Training GCN with normalized original features...")
    set_seed(seed)
    gcn_norm_original = train_gcn(gcn_norm_original, data_normalized_original, labels_normalized_original)

    # Evaluate GCN models on test data with normalized features
    print("Evaluating GCN with normalized embeddings on test data...")
    test_data_normalized_embeddings, test_labels_normalized_embeddings = build_graph(test_df,
                                                                                     drug_embeddings_normalized,
                                                                                     protein_embeddings_normalized,
                                                                                     is_embedding=True)
    precision_norm_emb, recall_norm_emb, f1_norm_emb, auc_norm_emb = evaluate_gcn(gcn_norm_embedded,
                                                                                  test_data_normalized_embeddings,
                                                                                  test_labels_normalized_embeddings)

    print("Evaluating GCN with normalized original features on test data...")
    test_data_normalized_original, test_labels_normalized_original = build_graph(test_df, drug_features_normalized,
                                                                                 protein_features_normalized,
                                                                                 is_embedding=True)
    precision_norm_orig, recall_norm_orig, f1_norm_orig, auc_norm_orig = evaluate_gcn(gcn_norm_original,
                                                                                      test_data_normalized_original,
                                                                                      test_labels_normalized_original)

    # Print results
    print("\nResults with Embeddings:")
    print(f"Precision: {precision_emb:.4f}, Recall: {recall_emb:.4f}, F1 Score: {f1_emb:.4f}, AUC: {auc_emb:.4f}")

    print("\nResults with Original Features:")
    print(f"Precision: {precision_orig:.4f}, Recall: {recall_orig:.4f}, F1 Score: {f1_orig:.4f}, AUC: {auc_orig:.4f}")

    print("\nResults with Normalized Embeddings:")
    print(
        f"Precision: {precision_norm_emb:.4f}, Recall: {recall_norm_emb:.4f}, F1 Score: {f1_norm_emb:.4f}, AUC: {auc_norm_emb:.4f}")

    print("\nResults with Normalized Original Features:")
    print(
        f"Precision: {precision_norm_orig:.4f}, Recall: {recall_norm_orig:.4f}, F1 Score: {f1_norm_orig:.4f}, AUC: {auc_norm_orig:.4f}")
