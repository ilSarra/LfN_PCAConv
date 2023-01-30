import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import numpy as np

torch.manual_seed(47)

dataset = Planetoid(root='./data/', name='PubMed')

reg = False


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 24)
        self.conv4 = GCNConv(24, 16)
        self.conv5 = GCNConv(16, dataset.num_classes)

    def forward(self, sample):
        x, edge_index = sample.x, sample.edge_index

        x_1 = self.conv1(x, edge_index)
        x_2 = F.relu(x_1)
        x_3 = self.conv2(x_2, edge_index)
        x_4 = F.relu(x_3)
        x_5 = self.conv3(x_4, edge_index)
        x_6 = F.relu(x_5)
        x_7 = self.conv4(x_6, edge_index)
        x_8 = F.dropout(x_7, training=self.training)
        x_9 = self.conv5(x_8, edge_index)

        # returning the prediction together with the embedding of the last layer of the GCN

        return F.log_softmax(x_9, dim=-1), [x_2, x_5, x_8]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset[0].to(device)
avg_acc = list()
iterations = 10

for k in range(iterations):

    # Initialize model in order to reset the model parameters
    model = GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    epochs = 200

    for epoch in range(epochs):

        optimizer.zero_grad()
        out, emb_vec = model(data)

        if reg:

            reg_norm = list()
            l2_norm = list()
            reg_coeff = 2
            l2_coeff = 2

            for embeddings in emb_vec:

                nodes = embeddings.size()[0]

                # Define a vector containing the sum over all the embeddings
                add = torch.sum(embeddings, dim=0)

                # Define a copy of the embeddings matrix multiplied by the number of nodes
                mul = torch.mul(nodes, embeddings)

                # Compute the result as the abs. difference between the mul and add matrices
                # PyTorch already computes the difference considering the add vector as a matrix
                res = abs(mul - add)

                # Summing over the embeddings
                res = torch.sum(res, dim=0)
                # Summing then over the embeddings dimension
                res = torch.sum(res)
                # Normalize
                res = res / (nodes ** 2)

                # Avoid zero division
                eps = 1e-4

                # Norm
                reg_norm_partial = 1 / (res + eps)

                reg_norm.append(reg_norm_partial.item())

                # L2-Norm on embeddings
                embeddings_squared = torch.mul(embeddings, embeddings)
                embeddings_squared = torch.sum(embeddings_squared, dim=1)
                l2_norm_partial = torch.sum(embeddings_squared) / (nodes ** 2)

                l2_norm.append(l2_norm_partial.item())

            reg_norm = reg_norm[2]
            l2_norm = l2_norm[2]

            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) + reg_coeff * reg_norm + l2_coeff * l2_norm

        else:
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

    model.eval()
    pred, embedding = model(data)
    pred = pred.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    avg_acc.append(acc)
    print(f'Accuracy: {acc:.4f}')

print("Average Accuracy", np.mean(avg_acc))
