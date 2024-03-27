import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor
from utils import Transform


class GAT(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , layer_heads=[]
                 , batchnorm=True):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.num_layers = num_layers

        if len(layer_heads) > 1:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=layer_heads[0], concat=True))
            if self.batchnorm:
                self.bns = torch.nn.ModuleList()
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels * layer_heads[0]))
            for i in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden_channels * layer_heads[i - 1], hidden_channels, heads=layer_heads[i], concat=True))
                if self.batchnorm:
                    self.bns.append(torch.nn.BatchNorm1d(hidden_channels * layer_heads[i - 1]))
            self.convs.append(GATConv(hidden_channels * layer_heads[num_layers - 2]
                                      , hidden_channels
                                      , heads=layer_heads[num_layers - 1]
                                      , concat=False))
        else:
            self.convs.append(GATConv(in_channels, out_channels, heads=layer_heads[0], concat=False))

        self.w = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def make_h(self,adj:SparseTensor):
        rows, cols = adj.storage.row(),adj.storage.col()
        node_edge = torch.zeros((rows.numel(), cols.numel()))
        for i,row,col in enumerate(zip(rows,cols)):
            node_edge[[row,col],i] = 1
        return node_edge
    def make_d(self,h):
        degree = 1 / torch.sum(h, dim=1)
        d = torch.diag(degree)
        return d


    def forward(self, data, n_id, adjs):
        pre_idx = adjs[0][1] # adjs[0]中边在原图的位置
        x = data.edge_attr[pre_idx]  # 边的属性

        # 转化为线图
        new_adjs = []
        for adj, e_id, _ in adjs:
            adj, edge_node_id = Transform.convert(adj)
            new_adjs.append((adj, e_id))
        # 获得节点表示
        for i, (edge_index, e_id) in enumerate(new_adjs):
            index = torch.isin(pre_idx,e_id)
            x = x[index]
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                if self.batchnorm:
                    x = self.bns[i](x)
                x = F.dropout(x, p=0.5, training=self.training)
            pre_idx = e_id
        H = self.make_h(adjs[-1][0])
        D = self.make_d(H)
        x = D @ H @ x @ self.w
        return x


    def inference_all(self, data):
        x, adj_t = data.edge_attr, data.adj_t
        new_adj, _ = Transform.convert(adj_t)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, new_adj)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, new_adj)
        H = self.make_h(adj_t)
        D = self.make_d(H)
        x = D @ H @ x @ self.w
        return x

    def inference(self, x_all, layer_loader, device,pbar):
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in layer_loader:
                # 转化为线图
                edge_index, e_id,_ = adj.to(device)
                new_adj, _ = Transform.convert(adj)
                x = x_all[e_id]
                x = self.convs[i](x, new_adj)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    if self.batchnorm:
                        x = self.bns[i](x)
                H = self.make_h(adj[0])
                D = self.make_d(H)
                x = D @ H @ x @ self.w
                xs.append(x)
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)

        return x_all