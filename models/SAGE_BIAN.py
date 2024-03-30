import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from tqdm import tqdm


class SAGE(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , device
                 , batchnorm=True):
        super(SAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.num_layers = num_layers
        self.device = device
        if self.batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for i in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            edge_index = edge_index

            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                if self.batchnorm:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def inference_all(self, data):
        x, adj_t = data.x, data.adj_t
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

    def inference(self, x_all, layer_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers * 2, ncols=80)
        pbar.set_description('Evaluating')
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in layer_loader:
                edge_index, _, size = adj
                edge_index, n_id = edge_index.cpu(), n_id.cpu()
                x = x_all[n_id].cpu()
                x_target = x[:size[1]]

                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    if self.batchnorm:
                        x = self.bns[i](x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                xs.append(x)

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        return x_all, pbar
