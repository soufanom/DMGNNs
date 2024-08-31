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

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class GCN(nn.Module):
    def __init__(self, num_node_features, hidden_dim=128, num_layers=5):
        super(GCN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()  # Batch normalization layers
        
        # First GCN layer
        self.convs.append(GCNConv(num_node_features, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden GCN layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Last GCN layer
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_weight)
            x = bn(x)  # Apply batch normalization
            x = F.relu(x)

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
        print(f"Output values at epoch {epoch+1}: {out}")  # Debugging: Check output values

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
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle the data
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    return train_df, test_df


# Load embeddings
def load_embeddings(embedding_file):
    with open(embedding_file, 'rb') as f:
        embeddings = pickle.load(f)
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

# Evaluate the GCN model
def evaluate_gcn(gcn, data, labels):
    gcn.eval()
    with torch.no_grad():
        out = gcn(data.x, data.edge_index, data.edge_attr)
        out = out.view(-1).cpu().numpy()
        labels = labels.cpu().numpy()

        predictions = (out > 0.5).astype(int)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        auc = roc_auc_score(labels, out)

    return precision, recall, f1, auc


# Main script
if __name__ == "__main__":
    # File paths
    data_file = '/home/o.soufan/DMGNNs/Data/BindingDB-processed/bindingdb_ic50_data.txt'
    drug_embedding_file = 'drug_embeddings.pkl'
    protein_embedding_file = 'protein_embeddings.pkl'
    feature_file = '/home/o.soufan/DMGNNs/simgraphmaker/features.pkl'

    # Shuffle and split the data
    train_df, test_df = shuffle_and_split_data(data_file)

    # Load embeddings and original features
    drug_embeddings = load_embeddings(drug_embedding_file)
    protein_embeddings = load_embeddings(protein_embedding_file)
    drug_features, protein_features = load_original_features(feature_file)

    # Determine the size of the node features
    embedding_feature_size = next(iter(drug_embeddings.values())).shape[0]
    original_feature_size = sum(map(lambda x: x.shape[0], next(iter(drug_features.values()))))

    # Ensure that both GCNs have the same size of node features
    assert embedding_feature_size == original_feature_size, "Node features must have the same size for both GCNs."

    # Build graphs using embeddings (no concatenation)
    data_embeddings, labels_embeddings = build_graph(train_df, drug_embeddings, protein_embeddings, is_embedding=True)

    # Build graphs using original features (with concatenation)
    data_original, labels_original = build_graph(train_df, drug_features, protein_features, is_embedding=False)

    # Initialize GCN models
    gcn_embeddings = GCN(num_node_features=embedding_feature_size, hidden_dim=128, num_layers=5)
    gcn_original = GCN(num_node_features=original_feature_size, hidden_dim=128, num_layers=5)

    # Train GCN models
    print("Training GCN with embeddings...")
    gcn_embeddings = train_gcn(gcn_embeddings, data_embeddings, labels_embeddings)

    print("Training GCN with original features...")
    gcn_original = train_gcn(gcn_original, data_original, labels_original)

    # Evaluate GCN models
    print("Evaluating GCN with embeddings on test data...")
    test_data_embeddings, test_labels_embeddings = build_graph(test_df, drug_embeddings, protein_embeddings, is_embedding=True)
    precision_emb, recall_emb, f1_emb, auc_emb = evaluate_gcn(gcn_embeddings, test_data_embeddings, test_labels_embeddings)

    print("Evaluating GCN with original features on test data...")
    test_data_original, test_labels_original = build_graph(test_df, drug_features, protein_features, is_embedding=False)
    precision_orig, recall_orig, f1_orig, auc_orig = evaluate_gcn(gcn_original, test_data_original, test_labels_original)

    # Print results
    print("\nResults with Embeddings:")
    print(f"Precision: {precision_emb:.4f}, Recall: {recall_emb:.4f}, F1 Score: {f1_emb:.4f}, AUC: {auc_emb:.4f}")

    print("\nResults with Original Features:")
    print(f"Precision: {precision_orig:.4f}, Recall: {recall_orig:.4f}, F1 Score: {f1_orig:.4f}, AUC: {auc_orig:.4f}")

