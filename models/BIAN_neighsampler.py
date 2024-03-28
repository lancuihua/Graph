import torch

from models import SAGE_BIAN
from models import GAT_BIAN


class BIAN(torch.nn.Module):
    def __init__(self, x_in_channels, edge_in_channels, hidden_channels, out_channels, num_classes, num_layers, dropout,
                 layer_heads=[], batchnorm=True):
        super(BIAN, self).__init__()
        self.sage = SAGE_BIAN.SAGE(x_in_channels, hidden_channels, out_channels, num_layers, dropout)
        self.w = torch.nn.Linear(in_features=2 * out_channels, out_features=num_classes)
        self.gat = GAT_BIAN.GAT(edge_in_channels, hidden_channels, out_channels, num_layers, dropout, layer_heads)

    def reset_parameters(self):
        self.w.reset_parameters()
        self.sage.reset_parameters()
        self.gat.reset_parameters()

    def forward(self, data, n_id, adjs):
        x1 = self.sage(data.x[n_id], adjs)
        x2 = self.gat(data, adjs)
        x = torch.concat([x1, x2], dim=-1)
        x = self.w(x)
        return x.log_softmax(dim=-1)

    def inference_all(self, data):
        x1 = self.sage.inference_all(data)
        x2 = self.gat(data)
        x = torch.concat([x1, x2], dim=-1)
        x = self.w(x)
        return x.log_softmax(dim=-1)

    def inference(self, data, layer_loader, device):
        x_all = data.x
        edge_all = data.edge_attr
        x1, pbar = self.sage.inference(x_all, layer_loader, device)
        x2 = self.gat.inference(edge_all, layer_loader, device, pbar)
        x = torch.concat([x1, x2], dim=-1)
        x = self.w(x)
        return x.log_softmax(dim=-1)
