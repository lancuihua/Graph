import torch

from models import SAGE_BIAN
from models import GAT_BIAN


class BIAN(torch.nn.Module):
    def __init__(self, x_in_channels, edge_in_channels, hidden_channels, out_channels, num_classes, num_layers, dropout,
                 device,
                 linears,
                 layer_heads,
                 batchnorm=True):
        super(BIAN, self).__init__()
        self.device = device
        self.sage = SAGE_BIAN.SAGE(x_in_channels, hidden_channels, out_channels, num_layers, dropout, device,batchnorm= batchnorm)
        self.gat = GAT_BIAN.GAT(edge_in_channels, hidden_channels, out_channels, num_layers, dropout, device,
                                layer_heads,batchnorm= batchnorm)

        self.Linears = torch.nn.ModuleList()
        self.Linears.append(torch.nn.Sequential(torch.nn.Linear(out_channels*2, linears[0]),
                                                torch.nn.ReLU()
                                                ))
        for i in range(len(linears)-1):
            self.Linears.append(torch.nn.Sequential(torch.nn.Linear(linears[i], linears[i+1]),
                                                    torch.nn.ReLU()
                                                    ))
        self.Linears.append(torch.nn.Sequential(torch.nn.Linear(linears[-1], num_classes)))
    def reset_parameters(self):
        self.sage.reset_parameters()
        self.gat.reset_parameters()
        for linear in self.Linears:
            linear[0].reset_parameters()

    def forward(self, data, n_id, adjs):
        x1 = self.sage(data.x[n_id], adjs)
        x2 = self.gat(data, adjs)
        x = torch.concat([x1, x2], dim=-1)
        for Linear in self.Linears:
            x = Linear(x)
        return x.log_softmax(dim=-1)

    def inference_all(self, data):
        x1 = self.sage.inference_all(data)
        x2 = self.gat(data)
        x = torch.concat([x1, x2], dim=-1)
        for Linear in self.Linears:
            x = Linear(x)
        return x.log_softmax(dim=-1)

    def inference(self, data, layer_loader):
        x_all = data.x
        edge_all = data.edge_attr
        x1, pbar = self.sage.inference(x_all, layer_loader)
        x2 = self.gat.inference(edge_all, layer_loader, pbar)
        x = torch.concat([x1, x2], dim=-1)
        for Linear in self.Linears:
            x = Linear(x)
        return x.log_softmax(dim=-1)
