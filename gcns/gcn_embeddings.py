import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class ImprovedGCN(nn.Module):
    def __init__(self, num_node_features, hidden_dim=128, num_layers=6, dropout_rate=0.3):
        super(ImprovedGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()  # Batch normalization layers

        # First GCN layer
        self.convs.append(GATConv(num_node_features, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Hidden GCN layers with residual connections
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Last GCN layer
        self.convs.append(GATConv(hidden_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Linear layer to map residual input size (num_node_features) to hidden_dim
        self.residual_transform = nn.Linear(num_node_features, hidden_dim)

        # Final linear layer to map back to original feature size
        self.out_layer = nn.Linear(hidden_dim, num_node_features)

    def forward(self, x, edge_index, edge_weight=None):
        # Apply the transformation to the input x for residual connection
        residual = self.residual_transform(x)  # Transform to hidden_dim size

        for conv, bn in zip(self.convs[:-1], self.bns[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

            # Add residual connection (skip connection)
            x += residual
            residual = x  # Update residual for the next layer

        # Last GCN layer without activation
        x = self.convs[-1](x, edge_index, edge_weight)
        x = self.bns[-1](x)

        # Project back to the original feature size
        x = self.out_layer(x)

        # Return the embeddings of size `hidden_dim`
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
    # df2 = df.copy()
    # df.loc[df2['similarity_score'] >= threshold, 'similarity_score'] = 1
    # df.loc[df2['similarity_score'] < threshold, 'similarity_score'] = 0
    #
    # # Separate the two classes
    # df_1 = df[df['similarity_score'] == 1]
    # df_0 = df[df['similarity_score'] == 0]
    #
    # # Get the number of samples in the minority class (label 1)
    # num_minority_samples = len(df_1)
    #
    # # Randomly sample the same number of label 0
    # df_0_sampled = df_0.sample(n=num_minority_samples, random_state=42)
    #
    # # Concatenate the sampled 0s with all the 1s
    # df_balanced = pd.concat([df_1, df_0_sampled], axis=0)
    #
    # # Shuffle the dataframe
    # df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    #return df_balanced


def build_graph(data, features_dict, entity_col1, entity_col2, feature_size):
    nodes = list(features_dict.keys())
    node_mapping = {entity: i for i, entity in enumerate(nodes)}

    edge_index = []
    edge_weight = []  # This will be binary: 1 for similar pairs, 0 for dissimilar pairs

    for _, row in data.iterrows():
        entity1 = row[entity_col1]
        entity2 = row[entity_col2]
        similarity_score = row['similarity_score']

        # Here, we assume the similarity_score is already binary (1 for similar, 0 for dissimilar)
        if entity1 in node_mapping and entity2 in node_mapping:
            i, j = node_mapping[entity1], node_mapping[entity2]
            edge_index.append([i, j])
            edge_index.append([j, i])  # Ensure the graph is undirected (i -> j and j -> i)

            # Append binary similarity score as edge weight
            edge_weight.append(similarity_score)  # 1 for similar, 0 for dissimilar
            edge_weight.append(similarity_score)  # Duplicate for the reverse edge

    # Convert edge_index to a tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Convert edge weights to a tensor (binary)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # Node feature normalization (Min-Max scaling)
    node_features_np = np.array([features_dict[node] for node in nodes])
    node_scaler = MinMaxScaler()
    node_features_scaled = node_scaler.fit_transform(node_features_np)
    node_features = torch.tensor(node_features_scaled, dtype=torch.float)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)

# Define contrastive loss function
def contrastive_loss(embeddings_i, embeddings_j, label, margin=1.0):
    # Compute the Euclidean distance between the pairs
    distances = F.pairwise_distance(embeddings_i, embeddings_j)

    # Contrastive loss
    loss = 0.5 * (label * distances ** 2 + (1 - label) * F.relu(margin - distances) ** 2)
    return loss.mean()

def train_gcn(gcn, data, epochs=500, learning_rate=0.01, weight_decay=1e-5, margin=1.0, print_every=10):
    # Optimizer with weight decay for regularization
    optimizer = optim.Adam(gcn.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler (Optional but useful for long training)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # Loss function: MSE for regression on similarity scores
    criterion = nn.MSELoss()

    gcn.train()  # Set the GCN model to training mode
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass: Get node embeddings from the GCN
        out = gcn(data.x, data.edge_index, data.edge_attr)

        # Extract embeddings for the node pairs connected by edges
        embeddings_i = out[data.edge_index[0]]  # Embeddings of the first nodes in each edge
        embeddings_j = out[data.edge_index[1]]  # Embeddings of the second nodes in each edge

        # Actual similarity labels (1 for similar, 0 for dissimilar)
        actual_similarity = data.edge_attr  # These should be binary labels (1 or 0)

        # Compute the contrastive loss
        loss = contrastive_loss(embeddings_i, embeddings_j, actual_similarity, margin)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Update the learning rate (optional)
        scheduler.step()

        # Print the loss at regular intervals (every 'print_every' epochs)
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    return gcn

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
    gcn = ImprovedGCN(num_node_features=feature_size, hidden_dim=hidden_dim, num_layers=num_layers)

    # Train GCN and extract embeddings
    embeddings = train_gcn(gcn, data)

    # Pass node features through the trained GCN to get the embeddings
    gcn.eval()  # Set GCN model to evaluation mode
    with torch.no_grad():  # Disable gradient tracking for evaluation
        embeddings = gcn(data.x, data.edge_index, data.edge_attr)  # Get node embeddings

    # Save the embeddings
    embeddings_dict = {node: embeddings[i].detach().cpu().numpy() for i, node in enumerate(clean_features.keys())}
    save_embeddings(embeddings_dict, output_file)

if __name__ == "__main__":
    # Process drug GCN
    process_gcn(
        pickle_file='/home/o.soufan/DMGNNs/simgraphmaker/features.pkl',
        csv_file='/home/o.soufan/DMGNNs/Data/SimilarityGraphs/drug_similarity_3.csv',
        entity_col1='drug1',
        entity_col2='drug2',
        output_file='drug_embeddings_prot.pkl',
        threshold=0.09491138750452857, # using a percentile threshold
        feature_type='drug_features'
    )

    # Process protein GCN
    process_gcn(
        pickle_file='/home/o.soufan/DMGNNs/simgraphmaker/features_prot.pkl',
        csv_file='/home/o.soufan/DMGNNs/Data/SimilarityGraphs/protein_similarity_prot_features.csv',
        entity_col1='protein1',
        entity_col2='protein2',
        output_file='protein_embeddings_prot.pkl',
        threshold=0.7841464260052041, # using a percentile threshold
        feature_type='protein_features'
    )