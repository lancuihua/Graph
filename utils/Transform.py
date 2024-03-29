from torch_sparse import SparseTensor
import torch


def convert(adj: SparseTensor, nodes: torch.tensor):
    if not isinstance(adj, SparseTensor):
        raise TypeError('Input type is {}; we need SparseTensor'.format(type(adj)))

        # 获取稀疏张量的行和列索引
    adj = adj.clone().detach()
    row, col,e_id = adj.coo()

    # 创建一部字典来存储每个节点关联的边的索引
    edge_to_indices = {}
    for i in range(row.size(0)):
        u, v = int(row[i]), int(col[i])
        if u not in edge_to_indices and u in nodes:
            edge_to_indices[u] = set()
        if v not in edge_to_indices and v in nodes:
            edge_to_indices[v] = set()
        if u in nodes:
            edge_to_indices[u].add(i)
        if v in nodes:
            edge_to_indices[v].add(i)

        # 初始化线图的边和值
    line_edges = []
    line_edge_id = []

    # 遍历字典中的每个节点和关联的边
    for node in edge_to_indices.keys():
        indices = edge_to_indices[node]
        for i in indices:
            u, v = row[i], col[i]
            # 遍历与当前边共享节点的其他边
            for j in indices:
                if i == j:
                    continue
                w, x = row[j], col[j]
                # 检查是否共享节点
                if (u == w) or (u == x) or (v == w) or (v == x):
                    if (i, j) not in line_edges:
                        line_edges.append((i, j))
                        line_edge_id.append(node)

                    # 转换line_edges为COO格式
    line_row, line_col = zip(*line_edges)
    line_row = torch.tensor(line_row, dtype=torch.long)
    line_col = torch.tensor(line_col, dtype=torch.long)
    line_edge_id = torch.tensor(line_edge_id, dtype=torch.long)
    size = e_id.numel()
    # print(line_row)
    # print(line_edge_id)
    # print(edge_to_indices)

    # 创建线图的SparseTensor
    line_graph = SparseTensor(row=line_row, col=line_col, value=line_edge_id,sparse_sizes=(size, size))

    return line_graph, line_edge_id


def convert_to_line_graph(adj: SparseTensor, size: int):
    # size: 原图中边的个数
    if not isinstance(adj, SparseTensor):
        raise TypeError('Input type is {}; we need SparseTensor'.format(type(adj)))

    # 获取目标节点对应的边的起点和终点
    row, col, _ = adj.detach().coo()

    a1 = row.unsqueeze(1) == row.unsqueeze(0)  # 共享起点
    a2 = col.unsqueeze(1) == col.unsqueeze(0)  # 共享重点
    a3 = (row.unsqueeze(1) == col.unsqueeze(0)) & (col.unsqueeze(1) != row.unsqueeze(0))  # 一个起点，一个终点
    # 找到所有与目标节点共享起点或终点的边
    node_relation = a1 | a2 | a3

    # 将对角线元素设为0，避免自循环
    node_relation.fill_diagonal_(0)

    # 获取线图的边索引和边对应的节点
    line_row, line_col = node_relation.nonzero().t()
    line_graph = SparseTensor(row=line_row, col=line_col, sparse_sizes=(size, size)).to_symmetric()

    return line_graph

def balance(adjs: list):
    adjs = adjs.copy()
    n = len(adjs)
    for i in range(n - 1):
        adj_1, e_id_1, _ = adjs[n - i - 1]  # 小
        adj_2, e_id_2, size = adjs[n - i - 2]  # 大
        row_1, col_1, _ = adj_1.coo()
        row_2, col_2, _ = adj_2.coo()

        diff_1_2 = ~torch.isin(e_id_1, e_id_2)
        diff_row = row_1[diff_1_2]
        diff_col = col_1[diff_1_2]
        row_2, col_2 = torch.cat([row_2, diff_row], dim=-1), torch.cat([col_2, diff_col],dim=-1)
        new_e_id = e_id_1[diff_1_2]
        e_id = torch.cat([e_id_2,new_e_id],dim=-1)
        adj = SparseTensor(row=row_2, col=col_2, value=e_id)
        adjs[n - i - 2] = (adj,e_id,size)

    return adjs


