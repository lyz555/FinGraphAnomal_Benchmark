import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GAE
from tqdm import tqdm
from torch_geometric.utils import negative_sampling


class GAEModel(GAE):  # Inherit from GAE
    def __init__(self, in_channels, out_channels):
        super(GAEModel, self).__init__(GCNEncoder(in_channels, out_channels))


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return self.conv2(x, edge_index)

class GAEWrapper(torch.nn.Module):
    def __init__(self, config, feat_data, labels, edge_index, split_idx):
        super(GAEWrapper, self).__init__()
        self.device = torch.device(f'cuda:{config["cuda_id"]}' if torch.cuda.is_available() else 'cpu')

        self.model = GAEModel(feat_data.shape[1], config["emb_size"]).to(self.device)

        self.x = feat_data.to(self.device) if isinstance(feat_data, torch.Tensor) else torch.tensor(feat_data, dtype=torch.float, device=self.device)
        self.labels = labels.to(self.device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long, device=self.device)
        self.edge_index = edge_index.to(self.device) if isinstance(edge_index, torch.Tensor) else torch.tensor(edge_index, dtype=torch.long, device=self.device)
        self.split_idx = split_idx

    def forward(self):
        return self.model(self)

    def fit(self):
        assert self.edge_index.max().item() < self.x.size(0), \
            f"Invalid edge_index: max index {self.edge_index.max().item()} ≥ number of nodes {self.x.size(0)}"
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        for epoch in tqdm(range(1, 201), desc="Training GAE"):
            optimizer.zero_grad()
            z = self.model.encode(self.x, self.edge_index.to(self.device))
            pos_edge_index = self.edge_index.to(self.device)
            loss = self.model.recon_loss(z, pos_edge_index)
            loss.backward()
            optimizer.step()

        z = self.model.encoder(self.x, self.edge_index)

        pos_edge_index = self.edge_index
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index, 
            num_nodes=self.x.size(0)
        )

        auc, _ = self.model.test(z, pos_edge_index, neg_edge_index)
        return 0, 0, 0, auc, 0

