import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pickle
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pickle
import pandas as pd
import numpy as np


class GCN(nn.Module):
    def __init__(self, num_node_features, hidden_dim=128, num_layers=4):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Linear layer to map from hidden_dim back to original feature size
        self.out_layer = nn.Linear(hidden_dim, num_node_features)

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = torch.relu(x)
        x = self.convs[-1](x, edge_index, edge_weight)

        # Apply the final linear layer to get back to the original feature size
        x = self.out_layer(x)
        return x


def load_features(pickle_file):
    with open(pickle_file, 'rb') as f:
        features_data = pickle.load(f)
    return features_data

def preprocess_features(features):
    clean_features = {}
    for k, v in features.items():
        if isinstance(v, tuple) and len(v) == 2:
            ecfp, desc = v

            # Concatenate ECFP and descriptor features into a single feature vector
            combined_features = np.concatenate((ecfp, desc))

            # Ensure it's a numpy array and check for NaN values
            if np.issubdtype(combined_features.dtype, np.number) and not np.isnan(combined_features).any():
                clean_features[k] = combined_features
            else:
                print(f"Skipping entity {k} due to invalid or non-numeric features.")
        else:
            print(f"Skipping entity {k} due to unexpected feature format.")

    return clean_features

def load_similarity_data(csv_file, threshold):
    df = pd.read_csv(csv_file)
    df = df.dropna()
    df = df[df['similarity_score'] > threshold]
    return df


def build_graph(data, features_dict, entity_col1, entity_col2, feature_size):
    nodes = list(features_dict.keys())
    node_mapping = {entity: i for i, entity in enumerate(nodes)}

    edge_index = []
    edge_weight = []

    for _, row in data.iterrows():
        entity1 = row[entity_col1]
        entity2 = row[entity_col2]
        similarity_score = row['similarity_score']

        if entity1 in node_mapping and entity2 in node_mapping:
            i, j = node_mapping[entity1], node_mapping[entity2]
            edge_index.append([i, j])
            edge_weight.append(similarity_score)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # Convert the list of numpy arrays to a single numpy array before tensor conversion
    node_features_np = np.array([features_dict[node] for node in nodes])
    node_features = torch.tensor(node_features_np, dtype=torch.float)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)

def train_gcn(gcn, data, epochs=100, learning_rate=0.01):
    optimizer = optim.Adam(gcn.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    gcn.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = gcn(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out, data.x)  # Autoencoder-like behavior
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    return out


def save_embeddings(embeddings, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)


def process_gcn(pickle_file, csv_file, entity_col1, entity_col2, output_file, threshold=0.5, hidden_dim=128,
                num_layers=4, feature_type='drug_features'):
    # Load and preprocess features
    features_data = load_features(pickle_file)

    if feature_type not in features_data:
        raise ValueError(
            f"Feature type '{feature_type}' not found in features file. Available types are: {list(features_data.keys())}")

    clean_features = preprocess_features(features_data[feature_type])

    # Check if clean_features is empty
    if not clean_features:
        raise ValueError("No valid features found after preprocessing. Please check your data.")

    # Load and filter similarity data
    similarity_data = load_similarity_data(csv_file, threshold)

    # Filter out rows where either entity is missing from clean_features
    similarity_data = similarity_data[
        (similarity_data[entity_col1].isin(clean_features.keys())) &
        (similarity_data[entity_col2].isin(clean_features.keys()))
        ]

    if similarity_data.empty:
        raise ValueError(
            "No valid similarity data after filtering. Ensure the entities in the similarity file match those in the features file.")

    # Determine the feature size from the first entry
    feature_size = next(iter(clean_features.values())).shape[0]

    # Build the graph
    data = build_graph(similarity_data, clean_features, entity_col1, entity_col2, feature_size)

    # Initialize GCN model
    gcn = GCN(num_node_features=feature_size, hidden_dim=hidden_dim, num_layers=num_layers)

    # Train GCN and extract embeddings
    embeddings = train_gcn(gcn, data)

    # Save the embeddings
    embeddings_dict = {node: embeddings[i].detach().numpy() for i, node in enumerate(clean_features.keys())}
    save_embeddings(embeddings_dict, output_file)

# Example usage
if __name__ == "__main__":
    # Process drug GCN
    process_gcn(
        pickle_file='/home/o.soufan/DMGNNs/simgraphmaker/features.pkl',
        csv_file='/home/o.soufan/DMGNNs/Data/SimilarityGraphs/drug_similarity.csv',
        entity_col1='drug1',
        entity_col2='drug2',
        output_file='drug_embeddings.pkl',
        threshold=0.09857327094420833, # using a percentile threshold
        feature_type='drug_features'
    )

    # Process protein GCN
    process_gcn(
        pickle_file='/home/o.soufan/DMGNNs/simgraphmaker/features.pkl',
        csv_file='/home/o.soufan/DMGNNs/Data/SimilarityGraphs/protein_similarity.csv',
        entity_col1='protein1',
        entity_col2='protein2',
        output_file='protein_embeddings.pkl',
        threshold=0.4916331202901155, # using a percentile threshold
        feature_type='protein_features'
    )

class GCN(nn.Module):
    def __init__(self, num_node_features, hidden_dim=64, num_layers=4):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Linear layer to map from hidden_dim back to original feature size
        self.out_layer = nn.Linear(hidden_dim, num_node_features)

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = torch.relu(x)
        x = self.convs[-1](x, edge_index, edge_weight)

        # Apply the final linear layer to get back to the original feature size
        x = self.out_layer(x)
        return x


def load_features(pickle_file):
    with open(pickle_file, 'rb') as f:
        features_data = pickle.load(f)
    return features_data

def preprocess_features(features):
    clean_features = {}
    for k, v in features.items():
        if isinstance(v, tuple) and len(v) == 2:
            ecfp, desc = v

            # Concatenate ECFP and descriptor features into a single feature vector
            combined_features = np.concatenate((ecfp, desc))

            # Ensure it's a numpy array and check for NaN values
            if np.issubdtype(combined_features.dtype, np.number) and not np.isnan(combined_features).any():
                clean_features[k] = combined_features
            else:
                print(f"Skipping entity {k} due to invalid or non-numeric features.")
        else:
            print(f"Skipping entity {k} due to unexpected feature format.")

    return clean_features

def load_similarity_data(csv_file, threshold):
    df = pd.read_csv(csv_file)
    df = df.dropna()
    df = df[df['similarity_score'] > threshold]
    return df


def build_graph(data, features_dict, entity_col1, entity_col2, feature_size):
    nodes = list(features_dict.keys())
    node_mapping = {entity: i for i, entity in enumerate(nodes)}

    edge_index = []
    edge_weight = []

    for _, row in data.iterrows():
        entity1 = row[entity_col1]
        entity2 = row[entity_col2]
        similarity_score = row['similarity_score']

        if entity1 in node_mapping and entity2 in node_mapping:
            i, j = node_mapping[entity1], node_mapping[entity2]
            edge_index.append([i, j])
            edge_weight.append(similarity_score)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # Convert the list of numpy arrays to a single numpy array before tensor conversion
    node_features_np = np.array([features_dict[node] for node in nodes])
    node_features = torch.tensor(node_features_np, dtype=torch.float)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)

def train_gcn(gcn, data, epochs=200, learning_rate=0.01):
    optimizer = optim.Adam(gcn.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    gcn.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = gcn(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out, data.x)  # Autoencoder-like behavior
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    return out


def save_embeddings(embeddings, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)


def process_gcn(pickle_file, csv_file, entity_col1, entity_col2, output_file, threshold=0.5, hidden_dim=64,
                num_layers=4, feature_type='drug_features'):
    # Load and preprocess features
    features_data = load_features(pickle_file)

    if feature_type not in features_data:
        raise ValueError(
            f"Feature type '{feature_type}' not found in features file. Available types are: {list(features_data.keys())}")

    clean_features = preprocess_features(features_data[feature_type])

    # Check if clean_features is empty
    if not clean_features:
        raise ValueError("No valid features found after preprocessing. Please check your data.")

    # Load and filter similarity data
    similarity_data = load_similarity_data(csv_file, threshold)

    # Filter out rows where either entity is missing from clean_features
    similarity_data = similarity_data[
        (similarity_data[entity_col1].isin(clean_features.keys())) &
        (similarity_data[entity_col2].isin(clean_features.keys()))
        ]

    if similarity_data.empty:
        raise ValueError(
            "No valid similarity data after filtering. Ensure the entities in the similarity file match those in the features file.")

    # Determine the feature size from the first entry
    feature_size = next(iter(clean_features.values())).shape[0]

    # Build the graph
    data = build_graph(similarity_data, clean_features, entity_col1, entity_col2, feature_size)

    # Initialize GCN model
    gcn = GCN(num_node_features=feature_size, hidden_dim=hidden_dim, num_layers=num_layers)

    # Train GCN and extract embeddings
    embeddings = train_gcn(gcn, data)

    # Save the embeddings
    embeddings_dict = {node: embeddings[i].detach().numpy() for i, node in enumerate(clean_features.keys())}
    save_embeddings(embeddings_dict, output_file)

# Example usage
if __name__ == "__main__":
    # Process drug GCN
    process_gcn(
        pickle_file='/home/o.soufan/DMGNNs/simgraphmaker/features.pkl',
        csv_file='/home/o.soufan/DMGNNs/Data/SimilarityGraphs/drug_similarity.csv',
        entity_col1='drug1',
        entity_col2='drug2',
        output_file='drug_embeddings.pkl',
        threshold=0.09857327094420833, # using a percentile threshold
        feature_type='drug_features'
    )

    # Process protein GCN
    process_gcn(
        pickle_file='/home/o.soufan/DMGNNs/simgraphmaker/features.pkl',
        csv_file='/home/o.soufan/DMGNNs/Data/SimilarityGraphs/protein_similarity.csv',
        entity_col1='protein1',
        entity_col2='protein2',
        output_file='protein_embeddings.pkl',
        threshold=0.4916331202901155, # using a percentile threshold
        feature_type='protein_features'
    )
