

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
import numpy as np
import random
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch_geometric.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data from the local ./data directory
node_basic_features = torch.load('./data/node_basic_features.pt', map_location=device, weights_only=True)
node_advanced_features = torch.load('./data/node_advanced_features.pt', map_location=device, weights_only=True)
edge_index = torch.load('./data/edge_index.pt', map_location=device, weights_only=True)
base_edge_features = torch.load('./data/base_edge_features.pt', map_location=device, weights_only=True)
nft_multimodal_bmbedding_features = torch.load('./data/nft_multimodal_bmbedding_features.pt', map_location=device, weights_only=True)
y = torch.load('./data/y.pt', map_location=device, weights_only=True)
node_sample_prob = torch.load('./data/node_sample_prob.pt', weights_only=True)
node_sample_prob = node_sample_prob / node_sample_prob.sum()

node_features = torch.cat([node_basic_features, node_advanced_features], dim=1)
edge_features = torch.cat([base_edge_features, nft_multimodal_bmbedding_features], dim=1)

train_mask = np.zeros(y.shape[0], dtype=np.bool_)
test_mask = np.zeros(y.shape[0], dtype=np.bool_)
train_test_split_num = int(y.shape[0] * 0.75)
train_index = random.sample(range(y.shape[0]), train_test_split_num)
test_index = list(set(range(y.shape[0])) - set(train_index))
train_mask[train_index] = True
test_mask[test_index] = True
print("train node num: ", train_mask.sum())
print("test node num: ", test_mask.sum())
print("true data percentage in train data: ", y[train_mask].sum() / len(y[train_mask]))
print("true data percentage in test data: ", y[test_mask].sum() / len(y[test_mask]))

data = Data(x=node_features, y=y,
            edge_index=edge_index, edge_attr=edge_features,
            train_mask=train_mask, test_mask=test_mask).to(device)

import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.nn import MessagePassing

class ArtemisFirstLayerConv(MessagePassing):
    def __init__(self, in_node_channels, in_edge_channels, out_channels, aggr='mean'):
        super(ArtemisFirstLayerConv, self).__init__(aggr=aggr)
        self.lin = nn.Linear(in_node_channels + in_edge_channels, out_channels)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return self.lin(torch.cat([x_j, edge_attr], dim=-1))

    def update(self, aggr_out):
        return F.relu(aggr_out)

class ArtemisNet(nn.Module):
    def __init__(self, in_node_channels, in_edge_channels, hidden_channels):
        super(ArtemisNet, self).__init__()
        self.conv1 = ArtemisFirstLayerConv(in_node_channels, in_edge_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels + in_node_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index_tuple, edge_attr_tuple):
        edge_index_0, edge_index_1, edge_index_2 = edge_index_tuple
        edge_attr_0, _, _ = edge_attr_tuple

        # First layer with residual connection
        inital_embedding = x
        x = self.conv1(x, edge_index_0, edge_attr_0)
        x = F.relu(self.bn1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        # Second layer with residual connection
        x = self.conv2(x, edge_index_1)
        x = F.relu(self.bn2(x))
        x = F.dropout(x, p=0.5, training=self.training)

        # Third layer with residual connection
        x = self.conv3(x, edge_index_2)
        x = F.relu(self.bn3(x))
        x = F.dropout(x, p=0.5, training=self.training)

        # Apply MLP to the final output
        x = torch.cat([x, inital_embedding], dim=1)
        x = self.mlp(x)
        return x.squeeze()

from torch_geometric.utils import to_undirected
from torch_geometric.sampler import NeighborSampler
from typing import List, Optional, Tuple
import copy

# Compatibility import for different PyG versions
try:
    from torch_geometric.sparse import SparseTensor
except ImportError:
    from torch_sparse import SparseTensor

# Define a simple container class to replace the removed EdgeIndex
class EdgeIndex:
    def __init__(self, edge_index, e_id, size):
        self.edge_index = edge_index
        self.e_id = e_id
        self.size = size

class NeighborSamplerbyNFT(torch.utils.data.DataLoader):
    def __init__(self, edge_index: torch.Tensor, sizes: List[int],
                 node_idx: Optional[torch.Tensor] = None, 
                 edge_attr: Optional[torch.Tensor] = None,
                 prob_vector: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 return_e_id: bool = True, 
                 **kwargs):

        self.edge_index = edge_index
        self.node_idx = node_idx
        self.num_nodes = num_nodes
        self.sizes = sizes
        self.return_e_id = return_e_id
        self.edge_attr = edge_attr
        self.prob_vector = prob_vector
        self.num_nodes = edge_index.max().item() + 1

        e_id = torch.arange(edge_index.size(1), device=edge_index.device)
        self.adj_t = SparseTensor.from_edge_index(edge_index, e_id, sparse_sizes=(self.num_nodes, self.num_nodes))
        self.adj_t = self.adj_t.to(device)

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        adjs = []
        n_id = batch
        for size in self.sizes:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            row, col, e_id_sampled = adj_t.coo()
            edge_index_sampled = torch.stack([row, col], dim=0)
            size_sampled = adj_t.sparse_sizes()
            adjs.append(EdgeIndex(edge_index=edge_index_sampled, e_id=e_id_sampled, size=size_sampled))

        if len(adjs) > 1:
            return batch_size, n_id, adjs[::-1]
        else:
            return batch_size, n_id, adjs[0]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(sizes={self.sizes})'

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler
import os

log_path = "./training_log.txt"

if os.path.exists(log_path):
    os.remove(log_path)

with open(log_path, "a") as log_file:
    log_file.write("Epoch, Average Loss, Average Accuracy, Average Precision, Average Recall, Average F1 Score\n")

    for run in range(5):
        print(f"Starting run {run+1}...\n")
        model = ArtemisNet(data.x.shape[1], data.edge_attr.shape[1], 32).to(device)


        num_pos = data.y[data.train_mask].sum().item()
        num_neg = data.train_mask.sum().item() - num_pos
        class_weights = torch.tensor([1], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

        sizes = [8, 1, 1]
        edge_sampler = NeighborSamplerbyNFT(edge_index=data.edge_index, sizes=sizes, edge_attr=data.edge_attr, prob_vector=node_sample_prob)
        patience = 10
        best_loss = float('inf')
        patience_counter = 0

        train_nodes = torch.where(torch.from_numpy(data.train_mask))[0]

        labels = data.y[data.train_mask].cpu().numpy()

        class_counts = np.bincount(labels)
        weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
        sample_weights = weights[labels]

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=False)

        best_model = None
        best_f1 = 0.0

        for epoch in range(100):
            model.train()
            total_loss = 0
            total_accuracy = 0
            total_precision = 0
            total_recall = 0
            total_f1 = 0
            batch_count = 0

            for subset_nodes in DataLoader(train_nodes, batch_size=256, sampler=sampler):
                batch_size, n_id, adjs = edge_sampler.sample(subset_nodes)
                n_id = n_id.to(device)

                optimizer.zero_grad()

                edge_index_0, e_id_0, size_0 = adjs[0].edge_index, adjs[0].e_id, adjs[0].size
                edge_attr_0 = data.edge_attr[e_id_0].to(device)
                edge_index_1, _, size_1 = adjs[1].edge_index, adjs[1].e_id, adjs[1].size
                edge_index_2, _, size_2 = adjs[2].edge_index, adjs[2].e_id, adjs[2].size

                out = model(data.x[n_id], (edge_index_0.to(device), edge_index_1.to(device), edge_index_2.to(device)), (edge_attr_0, None, None))

                loss = criterion(out, data.y[n_id].float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                predictions = torch.sigmoid(out)
                pred_binary = (predictions >= 0.5).int()
                pred = pred_binary.cpu()
                y_true = data.y[n_id].int().cpu()

                total_accuracy += accuracy_score(y_true.numpy(), pred.numpy())
                total_precision += precision_score(y_true.numpy(), pred.numpy(), zero_division=1)
                total_recall += recall_score(y_true.numpy(), pred.numpy())
                total_f1 += f1_score(y_true.numpy(), pred.numpy())

                batch_count += 1

            avg_loss = total_loss / batch_count
            avg_accuracy = total_accuracy / batch_count
            avg_precision = total_precision / batch_count
            avg_recall = total_recall / batch_count
            avg_f1 = total_f1 / batch_count

            print(f"Epoch {epoch} | Average Loss: {avg_loss:.5f} | Average Accuracy: {avg_accuracy:.3f} | "
                f"Average Precision: {avg_precision:.3f} | Average Recall: {avg_recall:.3f} | "
                f"Average F1 Score: {avg_f1:.3f}")
            log_file.write(f"{epoch}, {avg_loss:.5f}, {avg_accuracy:.3f}, {avg_precision:.3f}, {avg_recall:.3f}, {avg_f1:.3f}\n")
            log_file.flush()

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter == patience:
                    print("Stopping early due to lack of improvement on the validation set.")
                    break

            model.eval()
            test_nodes = torch.where(torch.from_numpy(data.test_mask))[0]
            with torch.no_grad():
                batch_size, n_id, adjs = edge_sampler.sample(test_nodes)
                n_id = n_id.to(device)

                edge_index_0, e_id_0, size_0 = adjs[0].edge_index, adjs[0].e_id, adjs[0].size
                edge_attr_0 = data.edge_attr[e_id_0].to(device)
                edge_index_1, _, size_1 = adjs[1].edge_index, adjs[1].e_id, adjs[1].size
                edge_index_2, _, size_2 = adjs[2].edge_index, adjs[2].e_id, adjs[2].size

                out = model(data.x[n_id], (edge_index_0.to(device), edge_index_1.to(device), edge_index_2.to(device)), (edge_attr_0, None, None))

                predictions = torch.sigmoid(out)
                pred_binary = (predictions >= 0.5).int()
                pred = pred_binary.cpu()
                y_true = data.y[n_id].int().cpu()
                accuracy = accuracy_score(y_true.numpy(), pred.numpy())
                precision = precision_score(y_true.numpy(), pred.numpy(), zero_division=1)
                recall = recall_score(y_true.numpy(), pred.numpy())
                f1 = f1_score(y_true.numpy(), pred.numpy())

                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model.state_dict().copy()

        model.load_state_dict(best_model)
        model.eval()
        test_nodes = torch.where(torch.from_numpy(data.test_mask))[0]
        with torch.no_grad():
            batch_size, n_id, adjs = edge_sampler.sample(test_nodes)
            n_id = n_id.to(device)

            edge_index_0, e_id_0, size_0 = adjs[0].edge_index, adjs[0].e_id, adjs[0].size
            edge_attr_0 = data.edge_attr[e_id_0].to(device)
            edge_index_1, _, size_1 = adjs[1].edge_index, adjs[1].e_id, adjs[1].size
            edge_index_2, _, size_2 = adjs[2].edge_index, adjs[2].e_id, adjs[2].size

            out = model(data.x[n_id], (edge_index_0.to(device), edge_index_1.to(device), edge_index_2.to(device)), (edge_attr_0, None, None))

            predictions = torch.sigmoid(out)
            pred_binary = (predictions >= 0.5).int()
            pred = pred_binary.cpu()
            y_true = data.y[n_id].int().cpu()
            accuracy = accuracy_score(y_true.numpy(), pred.numpy())
            precision = precision_score(y_true.numpy(), pred.numpy(), zero_division=1)
            recall = recall_score(y_true.numpy(), pred.numpy())
            f1 = f1_score(y_true.numpy(), pred.numpy())

            print(f"Test - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")
            log_file.write(f"Test - Run {run+1}, Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}\n\n")
            log_file.flush()

        print(f"Run {run+1} completed.\n")

print("All runs completed.")
