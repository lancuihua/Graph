# dataset name: DGraphFin

from utils import DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from torch_geometric.loader import NeighborSampler
from models import BIAN_neighsampler
from logger import Logger
from tqdm import tqdm
import os
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import pandas as pd
import warnings

warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')

eval_metric = 'auc'
# hidden_channels, out_channels, num_layers, dropout,device
# layer_heads=[], batchnorm=True
BIAN_parameters = {'lr': 0.003
    , 'l2': 5e-7
    , 'hidden_channels': 512
    , 'out_channels': 512
    , 'num_layers': 2
    , 'dropout': 0.0
    , 'layer_heads': [4, 1]
    , 'linears': [1024, 512, 256]
    , 'batchnorm': False
                   }


def train(epoch, train_loader, model, data, train_idx, optimizer, device):
    model.train()
    data = data.to(device)
    model = model.to(device)
    pbar = tqdm(total=train_idx.size(0), ncols=80)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        n_id = n_id.to(device)
        optimizer.zero_grad()
        out = model(data, n_id, adjs)
        loss = F.nll_loss(out, data.y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        pbar.update(batch_size)

    pbar.close()
    loss = total_loss / len(train_loader)

    return loss


@torch.no_grad()
def test(layer_loader, model, data, split_idx, evaluator):
    # data.y is labels of shape (N, )
    model.eval()
    model = model.cpu()
    data = data.cpu()
    out = model.inference(data, layer_loader)
    #     out = model.inference_all(data)
    y_pred = out.exp()  # (N,num_classes)

    losses, eval_results = dict(), dict()
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        node_id = node_id.cpu()
        losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()
        eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])[eval_metric]

    return eval_results, losses, y_pred


def main():
    parser = argparse.ArgumentParser(description='minibatch_gnn_models')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='DGraphFin')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--model', type=str, default='BAIN_neighsampler')
    parser.add_argument('--use_embeddings', action='store_true')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./checkout')
    parser.add_argument('--experiment', type=str, default='2')

    args = parser.parse_args()
    print(args)

    device = f'{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(device)

    dataset = DGraphFin(root='./dataset/', name=args.dataset, transform=T.ToSparseTensor())

    nlabels = dataset.nlables
    if args.dataset == 'DGraphFin': nlabels = 2

    data = dataset[0]
    # data.adj_t = data.adj_t.to_symmetric()  # 有向图转化为无向图

    if args.dataset in ['DGraphFin']:  # 标准化
        x = data.x
        x = (x - x.mean(0)) / x.std(0)
        data.x = x
    if data.y.dim() == 2:
        data.y = data.y.squeeze(1)

    split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}

    fold = args.fold
    if split_idx['train'].dim() > 1 and split_idx['train'].shape[1] > 1:
        kfolds = True
        print('There are {} folds of splits'.format(split_idx['train'].shape[1]))
        split_idx['train'] = split_idx['train'][:, fold]
        split_idx['valid'] = split_idx['valid'][:, fold]
        split_idx['test'] = split_idx['test'][:, fold]
    else:
        kfolds = False

    data = data
    train_idx = split_idx['train']
    # 准备模型训练结果的文件夹
    result_dir = prepare_folder(args.dataset, args.model)
    print('result_dir:', result_dir)

    train_loader = NeighborSampler(data.adj_t, node_idx=train_idx, sizes=[10, 10], batch_size=1024, shuffle=True,
                                   num_workers=32)
    layer_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1], batch_size=4096, shuffle=False,
                                   num_workers=32)
    model = None
    para_dict = None
    if args.model == 'BAIN_neighsampler':
        para_dict = BIAN_parameters
        model_para = para_dict.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        # x_in_channels, edge_in_channels, num_classes,device
        model = BIAN_neighsampler.BIAN(x_in_channels=data.x.size(-1), edge_in_channels=data.edge_attr.size(-1),
                                       device=device, num_classes=nlabels, **model_para)
    if not model:
        raise ValueError('No Model')
    print(f'Model {args.model} initialized')

    evaluator = Evaluator(eval_metric)
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        import gc
        gc.collect()
        print("Total parameters: ", sum(p.numel() for p in model.parameters()))

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
        best_valid = 0
        min_valid_loss = 1e8
        best_out = None

        for epoch in range(1, args.epochs + 1):
            loss = train(epoch, train_loader, model, data, train_idx, optimizer, device)
            eval_results, losses, out = test(layer_loader, model, data, split_idx, evaluator)
            train_eval, valid_eval, test_eval = eval_results['train'], eval_results['valid'], eval_results['test']
            train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']

            #                 if valid_eval > best_valid:
            #                     best_valid = valid_result
            #                     best_out = out.cpu().exp()
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                best_out = out.cpu()

            if epoch % args.log_steps == 0:
                save_path = os.path.join(args.save_dir, args.experiment)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if epoch % 5 == 0:
                    try:
                        save_name = '{}_{}.pth'.format(epoch, valid_loss)
                        torch.save(model.state_dict(), os.path.join(save_path, save_name))
                        print("save {} to {}".format(save_name, save_path))
                    except:
                        pass

                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_eval:.3f}%, '
                      f'Valid: {100 * valid_eval:.3f}% '
                      f'Test: {100 * test_eval:.3f}%')
            logger.add_result(run, [train_eval, valid_eval, test_eval])

        logger.print_statistics(run)

    final_results = logger.print_statistics()
    print('final_results:', final_results)
    para_dict.update(final_results)
    for k, v in para_dict.items():
        if type(v) is list: para_dict.update({k: str(v)})
    pd.DataFrame(para_dict, index=[args.model]).to_csv(result_dir + '/results.csv')


if __name__ == "__main__":
    main()
