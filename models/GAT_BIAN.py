import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor
from utils import Transform
import pandas as pd
import time


class GAT(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , device
                 , layer_heads=[]
                 , batchnorm=True):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.num_layers = num_layers
        self.device = device

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

    def make_h(self, adj: tuple) -> torch.Tensor:
        """
        根据给定的邻接矩阵构造一个节点边界的张量。

        参数:
            adj (tuple): 一个包含邻接矩阵信息的元组，预期格式为 (邻接矩阵, 边ID张量, 大小信息)。

        返回:
            torch.Tensor: 一个大小为 (节点数量, 边界ID数量) 的浮点张量，表示节点和边界ID之间的关系。
        """
        # 参数验证和解包
        if not isinstance(adj, tuple) or len(adj) != 3:
            raise ValueError("参数 adj 必须是一个包含三个元素的元组。")

        adj_matrix, _, size_info = adj
        rows, cols, _ = adj_matrix.detach().coo()

        # 检查边界条件
        if rows.numel() == 0:
            raise ValueError("输入的邻接矩阵为空或不包含边信息。")

        # 初始化 node_edge 张量
        node_edge = torch.zeros((size_info[-1], rows.numel()), dtype=torch.float32)

        # 直接使用 rows 作为行索引，在对应位置标注为1.0
        node_edge[rows, torch.arange(rows.numel())] = 1.0

        return node_edge.to(self.device)

    def make_d(self, h):
        h = h.detach()
        # 计算每行和
        degree = torch.sum(h, dim=1)

        # 标记需要处理的元素（非零且不是NaN或无穷大）
        to_process = (degree != 0) & ~torch.isinf(degree) & ~torch.isnan(degree)

        # 为非零、非NaN、非无穷大的元素取倒数
        degree[to_process] = 1 / degree[to_process]

        # 创建对角矩阵
        d = torch.diag(degree)

        return d.to(self.device)

    def forward(self, data, adjs):
        # 添加输入验证
        if not data or not adjs:
            raise ValueError("Data and adjs cannot be empty.")
        if 'edge_attr' not in data:
            raise ValueError("Data must contain 'edge_attr'.")
        x = data.edge_attr
        # 补充没有的边
        adjs = Transform.balance(adjs)
        pre_e_id = adjs[0][1]
        x = x[pre_e_id]
        if x.dim() == 1:
            x = x.unsqueeze(dim=-1)
        # 转化为线图
        new_adjs = []
        for adj, e_id, size in adjs:
            Line_adj = Transform.convert_to_line_graph(adj, size=e_id.numel())
            Line_adj = Line_adj.to(self.device)
            new_adjs.append((Line_adj, e_id))
        # 获得节点表示
        for i, (edge_index, e_id) in enumerate(new_adjs):
            diff_e_id = torch.isin(pre_e_id, e_id)
            x = x[diff_e_id, :]
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                if self.batchnorm:
                    x = self.bns[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            pre_e_id = e_id
        H = self.make_h(adjs[-1])
        D = self.make_d(H)
        x = self.w(D @ H @ x)
        return x

    def inference_all(self, data):
        x, adj_t = data.edge_attr, data.adj_t
        num_edges = x.numel()
        if x.dim()  == 1:
            x = x.unsqueeze(dim=-1)
        if x.dim() != 2:
            raise ValueError('must be 2 dimension, your dimension is {}'.format(x.dim()))
        new_adj = Transform.convert_to_line_graph(adj_t, size=num_edges)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, new_adj)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, new_adj)
        H = self.make_h(
            (adj_t, torch.arange(0, data.edge_attr.numel(), dtype=torch.long), (data.y.numel(), data.y.numel())))
        D = self.make_d(H)
        x = self.w(D @ H @ x)
        return x

    def inference(self, x_all, layer_loader, pbar):
        if x_all.dim() == 1:
            x_all = x_all.unsqueeze(dim=-1)
        xs = []
        for batch_size, _, adj in layer_loader:
            # 转化为线图
            edge_index, e_id, size = adj
            edge_index, e_id = edge_index.cpu(), e_id.cpu()
            line_adj = Transform.convert_to_line_graph(adj[0], size=e_id.numel())
            line_adj = line_adj.cpu()
            x = x_all[e_id]

            for i in range(self.num_layers):
                x = self.convs[i](x, line_adj)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    if self.batchnorm:
                        x = self.bns[i](x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                pbar.update(batch_size)
            H = self.make_h(adj).cpu()
            D = self.make_d(H).cpu()
            x = self.w(D @ H @ x)
            xs.append(x)

        x_all = torch.cat(xs, dim=0)

        return x_all
