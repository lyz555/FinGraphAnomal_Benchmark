from pygod.detector import CoLA
from utils import DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from models.gae import GAEWrapper
from models import MLP, GCN, SAGE, GCGraphConvModel, pcgnn
from logger import Logger
import time
from sklearn.metrics import f1_score, accuracy_score
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import pandas as pd
from tqdm.auto import tqdm

eval_metric = 'auc'

mlp_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             }

gcn_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             }

sage_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0
              , 'batchnorm': False
              , 'l2':5e-7
             }

graphconv_parameters = {'lr':0.01
                      , 'num_layers':2
                      , 'hidden_channels':128
                      , 'dropout':0.0
                      , 'batchnorm': False
                      , 'l2':5e-7
                     }

cola_parameters = {
    'hidden_channels': 128,
    'num_layers': 2,
    'dropout': 0.0,
    'lr': 0.01,
    'l2': 5e-7,
    'batch_size': 256,
}


def train(model, data, train_idx, optimizer, no_conv=False):
    # data.y is labels of shape (N, )
    model.train()

    optimizer.zero_grad()
    if no_conv:
        out = model(data.x[train_idx])
    else:
        out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator, no_conv=False):
    # data.y is labels of shape (N, )
    model.eval()

    if no_conv:
        out = model(data.x)
    else:
        out = model(data.x, data.adj_t)

    y_pred = out.exp()  # (N,num_classes)

    losses, eval_results = dict(), dict()
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()
        eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])[eval_metric]

    return eval_results, losses, y_pred


def main():
    parser = argparse.ArgumentParser(description='gnn_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='DGraphFin')
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--use_embeddings', action='store_true')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()
    print(args)

    no_conv = False
    if args.model in ['mlp']: no_conv = True

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")

    dataset = DGraphFin(root='./dataset/', name=args.dataset, transform=T.ToSparseTensor())
    nlabels = dataset.num_classes
    if args.dataset in ['DGraphFin']: nlabels = 2
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    row, col, _ = data.adj_t.coo()
    edge_index = torch.stack([row, col], dim=0)
    data.edge_index = to_undirected(edge_index)
    if args.dataset in ['DGraphFin']:
        x = data.x
        x = (x-x.mean(0))/x.std(0)
        data.x = x
    if data.y.dim()==2:
        data.y = data.y.squeeze(1)
    split_idx = {'train':data.train_mask, 'valid':data.valid_mask, 'test':data.test_mask}

    fold = args.fold
    if split_idx['train'].dim()>1 and split_idx['train'].shape[1] >1:
        kfolds = True
        print('There are {} folds of splits'.format(split_idx['train'].shape[1]))
        split_idx['train'] = split_idx['train'][:, fold]
        split_idx['valid'] = split_idx['valid'][:, fold]
        split_idx['test'] = split_idx['test'][:, fold]
    else:
        kfolds = False

    data = data.to(device)
    train_idx = split_idx['train'].to(device)

    result_dir = prepare_folder(args.dataset, args.model)
    print('result_dir:', result_dir)

    model = None
    para_dict = {}
    metrics_for_csv = {}
    if args.model == 'gae':
        config = {
            'model': 'gae',
            'data_name': args.dataset,
            'save_dir': result_dir,
            'num_epochs': args.epochs,
            'lr': 0.01,
            'hidden_channels': 64,
            'weight_decay': 5e-4,
            'emb_size': 64,
            'cuda_id': str(args.device)
        }


        feat_data = data.x
        labels = data.y
        edge_index = data.edge_index

        model = GAEWrapper(config, feat_data, labels, edge_index, split_idx).to(device)

        start_time = time.time()
        f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = model.fit()
        end_time = time.time()
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else 0

        duration = end_time - start_time
        metrics_for_csv = config.copy()
        metrics_for_csv.update({
            'test_auc': round(auc_test, 4),
            'f1_macro': round(f1_mac_test, 4),
            'f1_binary_1': round(f1_1_test, 4),
            'f1_binary_0': round(f1_0_test, 4),
            'gmean': round(gmean_test, 4),
            'avg_epoch_time_s': round(duration / args.epochs, 2),
            'avg_run_time_s': round(duration, 2),
            'peak_gpu_mem_mb': round(peak_mem, 2)
        })

        # MODIFICATION START: Replaced CSV saving with direct print
        model_name_display = args.model.upper()
        print(f"\n--- {model_name_display} Final Results ---")
        print(f"  -> Test AUC: {metrics_for_csv.get('test_auc', 0):.4f}")
        print(f"  -> Test Accuracy: Not Calculated for this model")
        print(f"  -> Test F1-Macro: {metrics_for_csv.get('f1_macro', 0):.4f}")
        print(f"  -> Test F1-Binary(1): {metrics_for_csv.get('f1_binary_1', 0):.4f}")
        print(f"  -> Test F1-Binary(0): {metrics_for_csv.get('f1_binary_0', 0):.4f}")
        print(f"  -> G-Mean: {metrics_for_csv.get('gmean', 0):.4f}")
        print(f"  -> Average Run Time: {metrics_for_csv.get('avg_run_time_s', 0):.2f}s")
        print(f"  -> Peak GPU Memory: {metrics_for_csv.get('peak_gpu_mem_mb', 0):.2f} MB")
        print("-" * 40)
        # MODIFICATION END
        return


    if args.model in ['mlp', 'gcn', 'sage', 'graphconv', 'pcgnn' ]:
        if args.model == 'mlp':
            para_dict = mlp_parameters
            model_para = mlp_parameters.copy(); model_para.pop('lr'); model_para.pop('l2')
            model = MLP(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
        elif args.model == 'gcn':
            para_dict = gcn_parameters
            model_para = gcn_parameters.copy(); model_para.pop('lr'); model_para.pop('l2')
            model = GCN(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
        elif args.model == 'sage':
            para_dict = sage_parameters
            model_para = sage_parameters.copy(); model_para.pop('lr'); model_para.pop('l2')
            model = SAGE(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
        elif args.model == 'graphconv':
            para_dict = graphconv_parameters
            model_para = graphconv_parameters.copy(); model_para.pop('lr'); model_para.pop('l2')
            model = GCGraphConvModel(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
            no_conv = False
        elif args.model == 'pcgnn':
            print(f"Running PC-GNN model using PyG data")

            config = {
                'model': args.model,
                'data_name': args.dataset,
                'save_dir': result_dir,
                'num_epochs': args.epochs,
                'batch_size': 256,
                'valid_epochs': 10,
                'train_ratio': 0.6,
                'test_ratio': 0.5,
                'emb_size': 64,
                'multi_relation': 'GNN',
                'no_cuda': False,
                'cuda_id': str(args.device),
                'seed': 42,
                'rho': 0.5,
                'thres': 0.5,
                'lr': 0.005,
                'weight_decay': 1e-4,
                'alpha': 1.0
            }


            feat_data = data.x.cpu().numpy()  # convert to numpy
            labels = data.y.cpu().numpy()
            edge_index = data.edge_index.cpu().numpy()

            model = pcgnn(config, feat_data, labels, edge_index, split_idx)

            start_time = time.time()
            f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = model.fit()
            end_time = time.time()
            peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else 0

            duration = end_time - start_time
            metrics_for_csv = config.copy()
            metrics_for_csv.update({
                'test_auc': round(auc_test, 4),
                'f1_macro': round(f1_mac_test, 4),
                'f1_binary_1': round(f1_1_test, 4),
                'f1_binary_0': round(f1_0_test, 4),
                'gmean': round(gmean_test, 4),
                'avg_epoch_time_s': round(duration / args.epochs, 2),
                'avg_run_time_s': round(duration, 2),
                'peak_gpu_mem_mb': round(peak_mem, 2)
            })

            # MODIFICATION START: Replaced CSV saving with direct print
            model_name_display = args.model.upper()
            print(f"\n--- {model_name_display} Final Results ---")
            print(f"  -> Test AUC: {metrics_for_csv.get('test_auc', 0):.4f}")
            print(f"  -> Test Accuracy: Not Calculated for this model")
            print(f"  -> Test F1-Macro: {metrics_for_csv.get('f1_macro', 0):.4f}")
            print(f"  -> Test F1-Binary(1): {metrics_for_csv.get('f1_binary_1', 0):.4f}")
            print(f"  -> Test F1-Binary(0): {metrics_for_csv.get('f1_binary_0', 0):.4f}")
            print(f"  -> G-Mean: {metrics_for_csv.get('gmean', 0):.4f}")
            print(f"  -> Average Run Time: {metrics_for_csv.get('avg_run_time_s', 0):.2f}s")
            print(f"  -> Peak GPU Memory: {metrics_for_csv.get('peak_gpu_mem_mb', 0):.2f} MB")
            print("-" * 40)
            # MODIFICATION END
            return

        print(f'Model {args.model} initialized.')
        if device.type == 'cuda':
            print(f"  Initial GPU Memory: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
            torch.cuda.reset_peak_memory_stats(device)

        evaluator = Evaluator(eval_metric)
        logger = Logger(args.runs, args)

        metrics_for_csv = para_dict.copy()
        run_times = []
        peak_mems_per_run = []
        f1_macro_list = []
        f1_binary_1_list = []
        f1_binary_0_list = []
        accuracy_list = []


        for run in range(args.runs):
            run_start_time = time.time()
            print(f"\n--- Starting Run {run + 1}/{args.runs} for model {args.model} ---")

            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])

            min_valid_loss_this_run = float('inf')

            epoch_times_this_run = []
            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()
                loss = train(model, data, train_idx, optimizer, no_conv)
                eval_results, losses, out = test(model, data, split_idx, evaluator, no_conv)

                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                epoch_times_this_run.append(epoch_duration)

                if losses['valid'] < min_valid_loss_this_run:
                    min_valid_loss_this_run = losses['valid']

                if epoch % args.log_steps == 0:
                    print(f'Run: {run + 1:02d}, Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                          f'Train: {100*eval_results["train"]:.3f}%, Valid: {100*eval_results["valid"]:.3f}%, Test: {100*eval_results["test"]:.3f}%, '
                          f'Epoch Time: {epoch_duration:.2f}s')
                logger.add_result(run, [eval_results['train'], eval_results['valid'], eval_results['test']])

            run_end_time = time.time()
            run_duration = run_end_time - run_start_time
            run_times.append(run_duration)
            _, _, out = test(model, data, split_idx, evaluator, no_conv)
            y_pred_label = out.argmax(dim=1)
            # ==== Diagnostic: Check unique predicted labels ====
            unique_labels, counts = torch.unique(y_pred_label[split_idx['test']], return_counts=True)
            print(f"[{args.model}] Predicted test labels distribution:")
            for lbl, count in zip(unique_labels.tolist(), counts.tolist()):
                print(f"  Label {lbl}: {count} samples")

            print(f"[{args.model}] Total test samples: {len(split_idx['test'])}")

            y_true_test = data.y[split_idx['test']]
            y_pred_test = y_pred_label[split_idx['test']]

            f1_macro = f1_score(y_true_test.cpu(), y_pred_test.cpu(), average='macro', zero_division=0)
            f1_binary_1 = f1_score(y_true_test.cpu(), y_pred_test.cpu(), pos_label=1, zero_division=0)
            f1_binary_0 = f1_score(y_true_test.cpu(), y_pred_test.cpu(), pos_label=0, zero_division=0)
            accuracy = accuracy_score(y_true_test.cpu(), y_pred_test.cpu())

            f1_macro_list.append(f1_macro)
            f1_binary_1_list.append(f1_binary_1)
            f1_binary_0_list.append(f1_binary_0)
            accuracy_list.append(accuracy)
            avg_epoch_time = sum(epoch_times_this_run) / len(epoch_times_this_run) if epoch_times_this_run else 0
            print(f"Run {run + 1} finished. Total Time: {run_duration:.2f}s, Avg Epoch Time: {avg_epoch_time:.2f}s")
            if device.type == 'cuda':
                peak_mem_run = torch.cuda.max_memory_allocated(device) / 1024**2
                peak_mems_per_run.append(peak_mem_run)
                print(f"  Peak GPU Memory for this run: {peak_mem_run:.2f} MB")
                torch.cuda.reset_peak_memory_stats(device)

            logger.print_statistics(run)

        final_stats = logger.print_statistics()
        if final_stats: metrics_for_csv.update(final_stats)
        avg_total_run_time = sum(run_times) / len(run_times) if run_times else 0
        max_peak_mem_overall = max(peak_mems_per_run) if peak_mems_per_run else 0
        avg_f1_macro = sum(f1_macro_list) / len(f1_macro_list) if f1_macro_list else 0
        avg_f1_binary_1 = sum(f1_binary_1_list) / len(f1_binary_1_list) if f1_binary_1_list else 0
        avg_f1_binary_0 = sum(f1_binary_0_list) / len(f1_binary_0_list) if f1_binary_0_list else 0
        avg_accuracy = sum(accuracy_list) / len(accuracy_list) if accuracy_list else 0


        print(f"\n--- All {args.runs} runs for model {args.model} completed ---")
        print(f"Average Run Time: {avg_total_run_time:.2f}s")
        if device.type == 'cuda' and peak_mems_per_run:
            print(f"Overall Peak GPU Memory: {max_peak_mem_overall:.2f} MB")


        # MODIFICATION START: Replaced CSV saving with direct print
        model_name_display = args.model.upper()
        # Use final_stats which holds the Test AUC from logger
        test_auc_final = final_stats.get('Test', 0.0)

        print(f"\n--- {model_name_display} Final Average Results ---")
        print(f"  -> Test AUC: {test_auc_final:.4f}")
        print(f"  -> Test Accuracy: {avg_accuracy:.4f}")
        print(f"  -> Test F1-Macro: {avg_f1_macro:.4f}")
        print(f"  -> Test F1-Binary(1): {avg_f1_binary_1:.4f}")
        print(f"  -> Test F1-Binary(0): {avg_f1_binary_0:.4f}")
        print(f"  -> Average Run Time: {avg_total_run_time:.2f}s")
        if device.type == 'cuda':
            print(f"  -> Peak GPU Memory: {max_peak_mem_overall:.2f} MB")
        print("-" * 40)
        # MODIFICATION END

    elif args.model == 'cola':
        para_dict = cola_parameters
        metrics_for_csv = para_dict.copy()

        data_for_ad = data

        print(f"Data object before CoLA specific processing: {data_for_ad}")
        print(f"  num_nodes: {data_for_ad.num_nodes}")
        if data_for_ad.edge_index is not None:
            print(f"  edge_index shape before coalesce: {data_for_ad.edge_index.shape}, dtype: {data_for_ad.edge_index.dtype}")
        if data_for_ad.edge_attr is not None:
            print(f"  edge_attr shape before coalesce: {data_for_ad.edge_attr.shape}, dtype: {data_for_ad.edge_attr.dtype}")
        else:
            print(f"  edge_attr is None before coalesce")

        if data_for_ad.num_nodes is None:
            data_for_ad.num_nodes = data_for_ad.x.shape[0]

        if data_for_ad.edge_index is not None:
            data_for_ad.edge_index = data_for_ad.edge_index.to(torch.long)
            if data_for_ad.edge_attr is not None and data_for_ad.edge_index.shape[1] != data_for_ad.edge_attr.shape[0]:
                data_for_ad.edge_attr = None
            print("Coalescing data object...")
            data_for_ad = data_for_ad.coalesce()
            print(f"After coalesce: edge_index shape={data_for_ad.edge_index.shape if data_for_ad.edge_index is not None else 'None'}")
            if data_for_ad.edge_index is not None and data_for_ad.edge_index.numel() > 0:
                if data_for_ad.edge_index.max().item() >= data_for_ad.num_nodes:
                    print(f"!!! ERROR AFTER COALESCE: Max index in edge_index ({data_for_ad.edge_index.max().item()}) "
                          f">= num_nodes ({data_for_ad.num_nodes})")
                    return
        else:
            print("!!! WARNING: data_for_ad.edge_index is None. Coalesce might not do much with edges.")

        print("Validating data object before passing to CoLA...")
        try:
            data_for_ad.validate(raise_on_error=True)
            print("Data validation successful.")
        except Exception as e:
            print(f"Data validation FAILED: {e}")
            return

        final_data_for_cola = data_for_ad
        print(f"Final data for CoLA: {final_data_for_cola} on device {final_data_for_cola.x.device}")

        current_gpu_id = -1
        if device.type == 'cuda':
            current_gpu_id = device.index if device.index is not None else 0
            print(f"  Initial GPU Memory before CoLA: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
            torch.cuda.reset_peak_memory_stats(device)

        model = CoLA(
            hid_dim=para_dict['hidden_channels'],
            num_layers=para_dict['num_layers'],
            dropout=para_dict['dropout'],
            weight_decay=para_dict['l2'],
            lr=para_dict['lr'],
            epoch=args.epochs,
            batch_size=para_dict['batch_size'],
            gpu=current_gpu_id,
            verbose=True
        )

        print(f"Fitting CoLA model... (Epochs for CoLA: {args.epochs})")
        fit_start_time = time.time()
        model.fit(final_data_for_cola.to(device))
        fit_end_time = time.time()
        fit_duration = fit_end_time - fit_start_time
        avg_epoch_time_cola = fit_duration / args.epochs if args.epochs > 0 else 0
        print(f"CoLA fitting complete. Total Fit Time: {fit_duration:.2f}s, Avg Time per CoLA Epoch: {avg_epoch_time_cola:.2f}s")

        peak_gpu_mem_fit = 0
        if device.type == 'cuda':
            peak_gpu_mem_fit = torch.cuda.max_memory_allocated(device) / 1024**2
            print(f"  Peak GPU Memory during/after CoLA fit: {peak_gpu_mem_fit:.2f} MB")

        from sklearn.metrics import roc_auc_score
        scores = model.decision_score_
        y_true_cpu = final_data_for_cola.y.cpu().numpy()
        scores_cpu = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores
        test_idx_cpu = split_idx['test'].cpu().numpy()

        auc = roc_auc_score(y_true_cpu[test_idx_cpu], scores_cpu[test_idx_cpu])
        print(f"CoLA Test AUC: {auc:.4f}")
        threshold = np.percentile(scores_cpu, 90)
        y_pred_label = (scores_cpu >= threshold).astype(int)

        # Extract test labels and predictions
        y_true_test = y_true_cpu[test_idx_cpu]
        y_pred_test = y_pred_label[test_idx_cpu]

        # Compute classification metrics
        f1_macro = f1_score(y_true_test, y_pred_test, average='macro', zero_division=0)
        f1_binary_1 = f1_score(y_true_test, y_pred_test, pos_label=1, zero_division=0)
        f1_binary_0 = f1_score(y_true_test, y_pred_test, pos_label=0, zero_division=0)
        accuracy = accuracy_score(y_true_test, y_pred_test)

        # MODIFICATION START: Replaced CSV saving with direct print
        model_name_display = args.model.upper()
        print(f"\n--- {model_name_display} Final Results ---")
        print(f"  -> Test AUC: {auc:.4f}")
        print(f"  -> Test Accuracy: {accuracy:.4f}")
        print(f"  -> Test F1-Macro: {f1_macro:.4f}")
        print(f"  -> Test F1-Binary(1): {f1_binary_1:.4f}")
        print(f"  -> Test F1-Binary(0): {f1_binary_0:.4f}")
        print(f"  -> Fit Time: {fit_duration:.2f}s")
        if device.type == 'cuda':
            print(f"  -> Peak GPU Memory: {peak_gpu_mem_fit:.2f} MB")
        print("-" * 40)
        # MODIFICATION END
        return

    else:
        print(f"Error: Unknown model name '{args.model}' or model not applicable to the current training loop.")
        return

if __name__ == "__main__":
    main()