import os

import torch
import torch.nn.functional as F
from torch import matmul

from torch_geometric.nn import GCNConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import degree
import torch_scatter

from torch.nn import Parameter, Linear

from sklearn.decomposition import PCA
import pycuda.gpuarray as gpuarray
from skcuda.linalg import PCA as cuPCA

from torch_geometric.datasets import KarateClub
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.typing import OptTensor

import numpy as np

import pycuda.autoinit

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.manual_seed(42)

torch.cuda.set_per_process_memory_fraction(0.8, 0)
torch.cuda.empty_cache()
print("GPU total memory", torch.cuda.get_device_properties(0).total_memory)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# Load Cora dataset
# dataset = Planetoid(root='data/Planetoid', name='Cora')  # , transform=NormalizeFeatures())
dataset = KarateClub()

print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]

print(f'Number of nodes: {data.num_nodes}')
print()

degrees = degree(data.edge_index[0], data.num_nodes, dtype=torch.int32)

max_degree = torch.max(degrees).item()

sort_index = np.argsort(degrees.numpy())
sort_index = np.flip(sort_index)


class PCAConv(GCNConv):
    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 add_self_loops=True,
                 normalize=True,
                 bias=True,
                 **kwargs):

        super(GCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    # def aggregate(self, inputs, index):
    #     sums = torch_scatter.scatter_add(inputs, index, dim=0)
    #     # muls = torch_scatter.scatter(inputs, index, dim=0, reduce='mul')
    #     # muls = torch_scatter.scatter_mul(inputs, index, dim=0)
    #     # maxs = torch_scatter.scatter_max(inputs, index, dim=0)[0]
    #     # mins = torch_scatter.scatter_min(inputs, index, dim=0)[0]
    #     means = torch_scatter.scatter_mean(inputs, index, dim=0)
    #     var = torch.relu(torch_scatter.scatter_mean(inputs ** 2, index, dim=0) - means ** 2)
    #
    #     # conc = torch.cat((sums, muls, maxs, mins, means, var), 1)
    #     neigh = torch.cat((sums, means, var), 1)
    #     # conc = conc.detach().cpu().numpy()
    #     # conc_gpu = gpuarray.GPUArray((data.num_nodes, dataset.num_features * 3), conc.dtype)
    #     # conc_gpu.set(conc)
    #     #
    #     # pca = cuPCA(n_components=dataset.num_features)
    #     # T_gpu = pca.fit_transform(conc_gpu)
    #     #
    #     # neigh_pca = torch.tensor(T_gpu, dtype=inputs.dtype)
    #     # neigh_pca = neigh_pca.to(device)
    #
    #     [U, S, V] = torch.pca_lowrank(neigh, q=data.num_features, center=True)
    #     V = V.to(device)
    #     neigh = neigh.to(device)
    #
    #     neigh_pca = matmul(neigh, V[:, :dataset.num_features])
    #     neigh_pca = neigh_pca.to(device)
    #     aggr = self.lin(neigh_pca)
    #
    #     return aggr

    def aggregate(self, inputs, index):
        mask = index == 0
        inputs_0 = inputs[mask]

        neigh = torch.zeros((data.num_nodes, max_degree * dataset.num_features))
        last_entry = torch.zeros(data.num_nodes, dtype=torch.int32)

        neigh.to(device)
        last_entry.to(device)

        for i in range(len(index) - 1):
            entry = index[i]
            neigh[entry, last_entry[entry]:last_entry[entry] + dataset.num_features] = inputs[i, :]
            last_entry[entry] += dataset.num_features

        # np_neigh = neigh.detach().cpu().numpy()
        # conc_gpu = gpuarray.GPUArray(neigh.shape, np_neigh.dtype)
        # conc_gpu.set(np_neigh)
        #
        # pca = cuPCA(n_components=dataset.num_features)
        # T_gpu = pca.fit_transform(conc_gpu)

        [U, S, V] = torch.pca_lowrank(neigh, q=data.num_features, center=True)
        V = V.to(device)
        neigh = neigh.to(device)

        neigh_pca = matmul(neigh, V[:, :dataset.num_features])
        neigh_pca = neigh_pca.to(device)
        aggr = self.lin(neigh_pca)

        return aggr


class PCANet(torch.nn.Module):
    def __init__(self):
        super(PCANet, self).__init__()

        self.pca1 = PCAConv(in_channels=data.num_features, out_channels=data.num_features)
        self.pca2 = PCAConv(in_channels=data.num_features, out_channels=data.num_features)
        self.pca3 = PCAConv(in_channels=data.num_features, out_channels=data.num_features)
        # self.pca2 = PCAConv(in_channels=data.num_features, out_channels=data.num_features)
        # self.pca3 = PCAConv(in_channels=data.num_features, out_channels=data.num_features)
        self.fc1 = Linear(data.num_features, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.pca1(x, edge_index)
        h = h.relu()
        h = self.pca2(h, edge_index)
        h = h.relu()
        h = self.pca3(h, edge_index)
        h = h.relu()
        out = self.fc1(h)

        return F.log_softmax(out, dim=-1)


model = PCANet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

x = torch.zeros((data.num_nodes, data.num_features + 1), data.x.dtype)
x = data.x


edge_index = data.edge_index.to(device)
y = data.y.to(device)  # .to(device)


def model_train(t_model):
    t_model.train()
    # optimizer.zero_grad()  # Clear gradients.

    out = t_model(x, edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], y[data.train_mask])  # Compute the loss solely based on the training nodes.
    # loss = criterion(out, y)
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.

    return loss


def model_test(t_model):
    t_model.eval()
    out = t_model(x, edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred == y  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.num_nodes)  # Derive ratio of correct predictions.
    return test_acc


print("Training PCA-GCN")

optimizer.zero_grad()

for epoch in range(1, 201):
    loss = model_train(model)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


test_acc = model_test(model)
print(f'Test Accuracy: {test_acc:.4f}')