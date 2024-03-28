import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor
from utils import Transform
import pandas as pd


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

    def make_h(self, adj: SparseTensor):
        adj, e_id, size = adj
        rows, cols, _ = adj.coo()
        node_edge = torch.zeros((size[-1], e_id.numel()),dtype=torch.float32)
        for i in range(rows.numel()):
            node_edge[rows[i], i] = 1.0
        return node_edge

    def make_d(self, h):
        # 计算每行和
        degree = torch.sum(h, dim=1)

        # 标记需要处理的元素（非零且不是NaN或无穷大）
        to_process = (degree != 0) & ~torch.isinf(degree) & ~torch.isnan(degree)

        # 为非零、非NaN、非无穷大的元素取倒数
        degree[to_process] = 1 / degree[to_process]

        # 创建对角矩阵
        d = torch.diag(degree)

        return d

    def forward(self, data, adjs):
        # x_idx = pd.Series(data.edge_attr[pre_idx], index=pre_idx)
        # 转化为线图
        new_adjs = []
        adjs = Transform.balance(adjs)
        pre_e_id = adjs[0][1]
        x = data.edge_attr[pre_e_id].unsqueeze(dim=-1)
        for adj, e_id, size in adjs:
            Line_adj, edge_node_id = Transform.convert(adj, range(size[1]))
            new_adjs.append((Line_adj, e_id))
        # 获得节点表示

        for i, (edge_index, e_id) in enumerate(new_adjs):

            diff_e_id = torch.isin(pre_e_id, e_id)
            x = x[diff_e_id, :]
            row, col, _ = edge_index.coo()
            #max_attr_idx = int(max(row.max(), col.max()))
            #x = x[:max_attr_idx+1, :]

            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                if self.batchnorm:
                    x = self.bns[i](x)
                x = F.dropout(x, p=0.5, training=self.training)
            pre_e_id = e_id
        H = self.make_h(adjs[-1])
        D = self.make_d(H)
        x = self.w(D @ H @ x)
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
        x = self.w(D @ H @ x)
        return x

    def inference(self, x_all, layer_loader, device, pbar):
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in layer_loader:
                # 转化为线图
                edge_index, e_id, size = adj.to(device)
                new_adj, _ = Transform.convert(adj, range(size[1]))
                x = x_all[e_id]
                x = self.convs[i](x, new_adj)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    if self.batchnorm:
                        x = self.bns[i](x)
                H = self.make_h(adj[0])
                D = self.make_d(H)
                x = self.w( D @ H @ x)
                xs.append(x)
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)

        return x_all
