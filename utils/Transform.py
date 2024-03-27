from torch_sparse import SparseTensor
import torch
def convert(adj: SparseTensor):
    if not isinstance(adj, SparseTensor):
        raise TypeError('Input type is {}; we need SparseTensor'.format(type(adj)))

        # 获取稀疏张量的行和列索引
    adj_sym = adj.detach().copy()
    row, col = adj_sym.storage.row(), adj_sym.storage.col()

    # 创建一部字典来存储每个节点关联的边的索引
    edge_to_indices = {}
    for i in range(row.size(0)):
        u, v = int(row[i]), int(col[i])
        if u not in edge_to_indices:
            edge_to_indices[u] = set()
        if v not in edge_to_indices:
            edge_to_indices[v] = set()
        edge_to_indices[u].add(i)
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
                    line_edges.append((i, j))
                    line_edge_id.append(node)

                    # 转换line_edges为COO格式
    line_row, line_col = zip(*line_edges)
    line_row = torch.tensor(line_row, dtype=torch.long)
    line_col = torch.tensor(line_col, dtype=torch.long)
    line_edge_id = torch.tensor(line_edge_id, dtype=torch.long)
    # print(line_row)
    # print(line_edge_id)
    # print(edge_to_indices)

    # 创建线图的SparseTensor
    line_graph = SparseTensor(row=line_row, col=line_col)
    return line_graph, line_edge_id
