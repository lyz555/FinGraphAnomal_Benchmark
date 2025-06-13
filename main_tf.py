import numpy as np
import argparse
import time
from tfclean import Dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_dgl
from models import MLP, GCN, SAGE, GCGraphConvModel
from pygod.detector import CoLA
from models.pcgnn import pcgnn as PCGNNModel
from torch_geometric.nn import GAE as PyGGAE
from models.gae import GCNEncoder
from sklearn.linear_model import LogisticRegression
import dgl

MODEL_PARAMS = {
    'mlp': {'lr': 0.01, 'num_layers': 2, 'dropout': 0.0, 'batchnorm': False, 'l2': 5e-7, 'type': 'pyg_standard'},
    'sage': {'lr': 0.01, 'num_layers': 2, 'dropout': 0.0, 'batchnorm': False, 'l2': 5e-7, 'type': 'pyg_standard'},
    'gcn': {'lr': 0.01, 'num_layers': 2, 'dropout': 0.0, 'batchnorm': False, 'l2': 5e-7, 'type': 'pyg_standard'},
    'graphconv': {'lr': 0.01, 'num_layers': 2, 'dropout': 0.0, 'batchnorm': False, 'l2': 5e-7, 'type': 'pyg_standard'},
    'cola': {'lr': 0.01,'num_layers': 2,'dropout': 0.0,'l2': 5e-7,'batch_size': 32,'hid_dim': 128,'type': 'pygod'}
}
def sample_graph_by_nodes(dgl_graph, max_nodes, seed):
    """Sample a node-induced subgraph with up to max_nodes nodes, preserving class balance."""
    labels = dgl_graph.ndata['label'].cpu().numpy()
    total_nodes = dgl_graph.number_of_nodes()
    
    if total_nodes <= max_nodes:
        print(f"Graph has {total_nodes} nodes ≤ max_nodes={max_nodes}. No sampling needed.")
        return dgl_graph  # No need to sample

    print(f"Sampling {max_nodes} nodes from total {total_nodes} nodes...")

    from sklearn.model_selection import train_test_split
    idx_all = np.arange(total_nodes)

    # Stratified sampling to maintain class balance
    idx_sample, _ = train_test_split(
        idx_all, train_size=max_nodes, stratify=labels, random_state=seed
    )

    sampled_graph = dgl.node_subgraph(dgl_graph, idx_sample)
    print(f"Sampled subgraph: {sampled_graph.number_of_nodes()} nodes, {sampled_graph.number_of_edges()} edges")
    return sampled_graph

def run_cola_experiment(data_object, args, device, in_feats):
    print(f"\n--- Running CoLA Experiment ---")
    
    if not isinstance(data_object, Data):
        raise TypeError("CoLA requires a PyTorch Geometric Data object.")

    data_object.num_nodes = data_object.x.shape[0]
    
    if data_object.edge_index is not None:
        data_object.edge_index = data_object.edge_index.to(torch.long)
        if data_object.edge_attr is not None and data_object.edge_index.shape[1] != data_object.edge_attr.shape[0]:
            data_object.edge_attr = None
        print("Coalescing PyG data object for CoLA...")
        data_object = data_object.coalesce()
    
    try:
        data_object.validate(raise_on_error=True)
        print("PyG Data validation successful for CoLA.")
    except Exception as e:
        print(f"PyG Data validation FAILED for CoLA: {e}")
        return 0.0, 0.0

    cola_params = MODEL_PARAMS['cola']
    gpu_id_for_cola = device.index if device.type == 'cuda' and device.index is not None else (0 if device.type == 'cuda' else -1)
    
    model = CoLA(
        hid_dim=cola_params['hid_dim'], num_layers=cola_params['num_layers'],
        dropout=cola_params['dropout'], weight_decay=cola_params['l2'],
        lr=cola_params['lr'], epoch=args.epoch, batch_size=cola_params['batch_size'],
        gpu=gpu_id_for_cola, contamination=0.1
    )
    
    print(f"Fitting CoLA model (Epochs: {args.epoch}). GPU ID: {gpu_id_for_cola}")
    fit_start_time = time.time()
    model.fit(data_object)
    print(f"CoLA fitting complete. Fit Time: {time.time() - fit_start_time:.2f}s")

    # Evaluation
    decision_scores = model.decision_score_
    pred_labels = model.predict(data_object) # Get binary predictions

    labels = data_object.y
    index = list(range(len(labels)))
    if args.dataset == 'amazon':
        index = list(range(3305, len(labels)))

    labels_cpu_for_split = labels.cpu()
    _, idx_test, _, _ = train_test_split(index, labels_cpu_for_split[index], stratify=labels_cpu_for_split[index],
                                         train_size=args.train_ratio, random_state=2, shuffle=True)
    
    test_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask[torch.tensor(idx_test)] = True

    y_true_test = labels[test_mask].cpu().numpy()
    scores_test = decision_scores[test_mask.cpu().numpy()]
    preds_test = pred_labels[test_mask.cpu().numpy()]

    if len(np.unique(y_true_test)) < 2:
        print("Warning: Only one class in CoLA test set. Metrics may be ill-defined.")
        return 0.0, 0.0
    
    auc_test = roc_auc_score(y_true_test, scores_test)
    acc_test = accuracy_score(y_true_test, preds_test)
    f1_macro_test = f1_score(y_true_test, preds_test, average='macro', zero_division=0)
    f1_1_test = f1_score(y_true_test, preds_test, pos_label=1, average='binary', zero_division=0)
    f1_0_test = f1_score(y_true_test, preds_test, pos_label=0, average='binary', zero_division=0)

    print("CoLA Final Test Results:")
    print(f"  -> AUC: {auc_test:.4f}")
    print(f"  -> Accuracy: {acc_test:.4f}")
    print(f"  -> F1-Macro: {f1_macro_test:.4f}")
    print(f"  -> F1-Illicit(1): {f1_1_test:.4f}")
    print(f"  -> F1-Licit(0): {f1_0_test:.4f}")
    
    return f1_macro_test, auc_test


def run_gae_experiment(pyg_data, args, device, in_feats):
    print(f"\n--- Running GAE Experiment with Two-Stage Evaluation---")
    gae_params = MODEL_PARAMS['gae']
    
    encoder = GCNEncoder(in_channels=in_feats, out_channels=gae_params['emb_size'])
    gae_model = PyGGAE(encoder).to(device)

    optimizer = torch.optim.Adam(gae_model.parameters(), lr=gae_params['lr'], weight_decay=gae_params.get('weight_decay', 0.0))
    
    features = pyg_data.x.to(device)
    edge_index = pyg_data.edge_index.to(device)

    print(f"Fitting GAE model for {gae_params['num_epochs']} epochs to learn embeddings...")
    fit_start_time = time.time()
    for epoch in range(1, gae_params['num_epochs'] + 1):
        gae_model.train()
        optimizer.zero_grad()
        z = gae_model.encode(features, edge_index)
        loss = gae_model.recon_loss(z, edge_index)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d}, Recon Loss: {loss:.4f}")

    gae_model.eval()
    with torch.no_grad():
        final_embeddings = gae_model.encode(features, edge_index).cpu()


    print("\n--- Training downstream Logistic Regression classifier ---")
    
    labels_cpu = pyg_data.y.cpu()
    index_all_nodes = list(range(len(labels_cpu)))
    if args.dataset == 'amazon':
        index_all_nodes = list(range(3305, len(labels_cpu)))

    idx_train, idx_test, y_train, y_test = train_test_split(
        index_all_nodes, labels_cpu[index_all_nodes], stratify=labels_cpu[index_all_nodes],
        train_size=args.train_ratio, random_state=args.seed, shuffle=True)
        
    train_embs = final_embeddings[idx_train]
    test_embs = final_embeddings[idx_test]
    
    lr = LogisticRegression(solver='liblinear', random_state=args.seed, class_weight='balanced')
    lr.fit(train_embs, y_train)
    
    if len(np.unique(y_test)) < 2:
        print("Warning: Only one class in test set for GAE. Metrics may be ill-defined.")
        return 0.0, 0.0
        
    test_pred_probs = lr.predict_proba(test_embs)[:, 1]
    test_pred_binary = lr.predict(test_embs)
    
    auc_test = roc_auc_score(y_test, test_pred_probs)
    acc_test = accuracy_score(y_test, test_pred_binary)
    f1_macro_test = f1_score(y_test, test_pred_binary, average='macro', zero_division=0)
    f1_1_test = f1_score(y_test, test_pred_binary, pos_label=1, average='binary', zero_division=0)
    f1_0_test = f1_score(y_test, test_pred_binary, pos_label=0, average='binary', zero_division=0)
    
    fit_duration = time.time() - fit_start_time
    print(f"GAE training and downstream evaluation complete. Fit Time: {fit_duration:.2f}s")
    print("GAE Final Test Results:")
    print(f"  -> AUC: {auc_test:.4f}")
    print(f"  -> Accuracy: {acc_test:.4f}")
    print(f"  -> F1-Macro: {f1_macro_test:.4f}")
    print(f"  -> F1-Illicit(1): {f1_1_test:.4f}")
    print(f"  -> F1-Licit(0): {f1_0_test:.4f}")

    return f1_macro_test, auc_test


def run_pcgnn_experiment(pyg_data, args, device, in_feats):
    print(f"\n--- Running PCGNN Experiment ---")
    pcgnn_params = MODEL_PARAMS['pcgnn']

    labels_cpu = pyg_data.y.cpu()
    index_all_nodes = list(range(len(labels_cpu)))
    if args.dataset == 'amazon':
        index_all_nodes = list(range(3305, len(labels_cpu)))

    idx_train, idx_rest, _, _ = train_test_split(
        index_all_nodes, labels_cpu[index_all_nodes], stratify=labels_cpu[index_all_nodes],
        train_size=args.train_ratio, random_state=args.seed, shuffle=True)
    idx_valid, idx_test, _, _ = train_test_split(
        idx_rest, labels_cpu[idx_rest], stratify=labels_cpu[idx_rest],
        test_size=0.5, random_state=args.seed, shuffle=True) 

    split_idx_tensors = {
        'train': torch.tensor(idx_train, dtype=torch.long),
        'valid': torch.tensor(idx_valid, dtype=torch.long),
        'test': torch.tensor(idx_test, dtype=torch.long)
    }
    print(f"PCGNN splits: Train: {len(idx_train)}, Valid: {len(idx_valid)}, Test: {len(idx_test)}")

    config_pcgnn = {
        'lr': pcgnn_params['lr'], 'num_epochs': args.epoch, 'emb_size': pcgnn_params['emb_size'],
        'weight_decay': pcgnn_params['weight_decay'], 'rho': pcgnn_params['rho'],
        'alpha': pcgnn_params['alpha'], 'batch_size': pcgnn_params['batch_size'],
        'thres': pcgnn_params['thres'], 'valid_epochs': pcgnn_params['valid_epochs'],
        'gpu': 0, 'seed': args.seed, 'save_dir': f'./output_pcgnn/{args.dataset}/',
        'data_name': args.dataset, 'model': 'pcgnn'
    }
    
    pcgnn_model = PCGNNModel(config_pcgnn, pyg_data.x, pyg_data.y, pyg_data.edge_index, split_idx_tensors)
    
    print(f"Fitting PCGNN model (Epochs: {args.epoch})")
    print("IMPORTANT: For full metrics, ensure your PCGNNModel.fit() returns (epoch, auc, f1_macro, f1_1, f1_0, acc).")
    fit_start_time = time.time()
    
    # Initialize metric variables
    auc_gnn, f1_macro, f1_binary_1, f1_binary_0, overall_acc = 0.0, 0.0, 0.0, 0.0, 0.0

    returned_values_pcgnn = pcgnn_model.fit()
    
    # Check what was returned and unpack accordingly for compatibility
    if isinstance(returned_values_pcgnn, tuple) and len(returned_values_pcgnn) == 6:
        _epoch_pcgnn, auc_gnn, f1_macro, f1_binary_1, f1_binary_0, overall_acc = returned_values_pcgnn
    elif isinstance(returned_values_pcgnn, tuple) and len(returned_values_pcgnn) == 3:
        _epoch_pcgnn, pcgnn_final_auc, pcgnn_final_f1_1 = returned_values_pcgnn
        auc_gnn, f1_binary_1 = pcgnn_final_auc, pcgnn_final_f1_1 # Assign to our standard vars
        print("Warning: PCGNN fit() returned 3 values. Only AUC and F1-Illicit(1) are available.")
    else:
        # Fallback for old return signature: f1_macro, _, _, auc, _
        if isinstance(returned_values_pcgnn, tuple) and len(returned_values_pcgnn) == 5:
            f1_macro, _, _, auc_gnn, _ = returned_values_pcgnn
            print("Warning: PCGNN fit() returned 5 values. Only F1-Macro and AUC are available.")
        else:
            print("Warning: PCGNN fit() method did not return the expected tuple. Metrics will be 0.")

    fit_duration_pcgnn = time.time() - fit_start_time
    print(f"PCGNN fitting complete. Fit Time: {fit_duration_pcgnn:.2f}s")
    print(f"Final GNN Test Results:")
    print(f"  -> AUC: {auc_gnn:.4f}")
    print(f"  -> Accuracy: {overall_acc:.4f}")
    print(f"  -> F1-macro: {f1_macro:.4f}")
    print(f"  -> F1-Illicit(1): {f1_binary_1:.4f}")
    print(f"  -> F1-Licit(0): {f1_binary_0:.4f}")
    
    return f1_macro, auc_gnn


def train_standard_pyg(model, data_object, args, device, model_params_dict):
    """ Standard training loop for typical PyG models """
    features = data_object.x.to(device)
    labels = data_object.y.to(device)
    edge_index = data_object.edge_index.to(device)

    index = list(range(len(labels)))
    if args.dataset == 'amazon':
        index = list(range(3305, len(labels)))

    labels_cpu = labels.cpu()
    idx_train, idx_rest, _, _ = train_test_split(index, labels_cpu[index], stratify=labels_cpu[index],
                                                         train_size=args.train_ratio, random_state=args.seed, shuffle=True)
    idx_valid, idx_test, _, _ = train_test_split(idx_rest, labels_cpu[idx_rest], stratify=labels_cpu[idx_rest],
                                                         test_size=0.5, random_state=args.seed, shuffle=True)
    
    train_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
    val_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
    test_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)

    train_mask[torch.tensor(idx_train, device=device)] = True
    val_mask[torch.tensor(idx_valid, device=device)] = True
    test_mask[torch.tensor(idx_test, device=device)] = True
    
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())

    optimizer = torch.optim.Adam(model.parameters(), lr=model_params_dict.get('lr', 0.01), weight_decay=model_params_dict.get('l2', 0))
    
    # Dictionary to store the best test metrics
    best_val_f1 = 0.0
    final_test_metrics = {}

    # Calculate loss weight
    train_labels_for_weight = labels[train_mask]
    weight_val = 1.0
    if train_labels_for_weight.numel() > 0:
        num_pos = (train_labels_for_weight == 1).sum().item()
        num_neg = (train_labels_for_weight == 0).sum().item()
        if num_pos > 0 and num_neg > 0:
            weight_val = num_neg / num_pos
    loss_weight = torch.tensor([1., weight_val]).to(device)
    print('Loss weight: ', loss_weight)
    
    time_start = time.time()
    for e in range(args.epoch):
        model.train()
        is_mlp_type = args.model.lower() == 'mlp'
        out = model(features) if is_mlp_type else model(features, edge_index)
        loss = F.nll_loss(out[train_mask], labels[train_mask], weight=loss_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # --- Evaluation Section ---
        if (e + 1) % 10 == 0 or (e + 1) == args.epoch:
            model.eval()
            with torch.no_grad():
                out_eval = model(features) if is_mlp_type else model(features, edge_index)
                probs = out_eval.exp()

            # Find best threshold on validation set
            f1_val, thres = get_best_f1(labels[val_mask].cpu(), probs[val_mask].cpu())
            
            # Evaluate on test set using the best threshold
            labels_test_cpu = labels[test_mask].cpu().numpy()
            probs_test_positive_class_cpu = probs[test_mask].cpu().numpy()[:, 1]
            preds_test_np = (probs_test_positive_class_cpu > thres).astype(int)

            # Calculate all metrics for the test set
            current_metrics = {}
            if len(np.unique(labels_test_cpu)) < 2:
                current_metrics = {'auc': 0.0, 'acc': 0.0, 'f1_macro': 0.0, 'f1_1': 0.0, 'f1_0': 0.0}
            else:
                current_metrics['auc'] = roc_auc_score(labels_test_cpu, probs_test_positive_class_cpu)
                current_metrics['acc'] = accuracy_score(labels_test_cpu, preds_test_np)
                current_metrics['f1_macro'] = f1_score(labels_test_cpu, preds_test_np, average='macro', zero_division=0)
                current_metrics['f1_1'] = f1_score(labels_test_cpu, preds_test_np, pos_label=1, average='binary', zero_division=0)
                current_metrics['f1_0'] = f1_score(labels_test_cpu, preds_test_np, pos_label=0, average='binary', zero_division=0)

            # Update final test metrics if validation F1 improves
            if f1_val > best_val_f1:
                best_val_f1 = f1_val
                final_test_metrics = current_metrics
            
            print(f'Epoch {e+1:03d}, Loss: {loss:.4f}, Val MF1: {f1_val:.4f} | Test AUC: {current_metrics["auc"]:.4f}, Test MF1: {current_metrics["f1_macro"]:.4f}')

    print(f'Total training time: {time.time() - time_start:.2f}s')
    print('Final Best Test Results (based on best val_mf1):')
    print(f"  -> AUC: {final_test_metrics.get('auc', 0):.4f}")
    print(f"  -> Accuracy: {final_test_metrics.get('acc', 0):.4f}")
    print(f"  -> F1-Macro: {final_test_metrics.get('f1_macro', 0):.4f}")
    print(f"  -> F1-Illicit(1): {final_test_metrics.get('f1_1', 0):.4f}")
    print(f"  -> F1-Licit(0): {final_test_metrics.get('f1_0', 0):.4f}")
    
    return final_test_metrics.get('f1_macro', 0), final_test_metrics.get('auc', 0)


def get_best_f1(labels_true_cpu, probs_cpu):
    # THIS FUNCTION IS NOW FIXED
    best_f1_macro, best_threshold = 0, 0
    
    # Convert tensors to numpy arrays at the beginning
    labels_true_np = labels_true_cpu.numpy()
    target_probs_positive_class_np = probs_cpu.numpy()[:, 1] if probs_cpu.ndim == 2 and probs_cpu.shape[1] >= 2 else probs_cpu.numpy()

    for thres_candidate in np.linspace(0.05, 0.95, 19):
        # Use numpy for boolean indexing and calculations
        preds_candidate = (target_probs_positive_class_np > thres_candidate).astype(int)
        
        current_f1_macro = f1_score(labels_true_np, preds_candidate, average='macro', zero_division=0)
        if current_f1_macro > best_f1_macro:
            best_f1_macro = current_f1_macro
            best_threshold = thres_candidate
            
    return best_f1_macro, best_threshold


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN Anomaly Detection')
    parser.add_argument("--dataset", type=str, default="tfinance", help="Dataset (yelp/amazon/tfinance/tsocial)")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    # parser.add_argument("--order", type=int, default=2, help="Order C for BWGNN") <- REMOVED
    # parser.add_argument("--homo", type=int, default=1, help="1 for BWGNN(Homo), 0 for Hetero") <- REMOVED
    parser.add_argument("--epoch", type=int, default=200, help="Max epochs")
    parser.add_argument("--run", type=int, default=1, help="Number of runs")
    parser.add_argument("--model", type=str, default="mlp",
                        help="Model (mlp, sage, gcn, graphconv, cola, gae, pcgnn)") # <- UPDATED
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    MODEL_PARAMS['gae'] = {
        'lr': 0.01, 'num_epochs': args.epoch, 'emb_size': args.hid_dim,
        'weight_decay': 5e-4, 'type': 'custom_gae'
    }
    MODEL_PARAMS['pcgnn'] = {
        'lr': 0.005, 'num_epochs': args.epoch, 'emb_size': args.hid_dim,
        'weight_decay': 1e-4, 'rho': 0.5, 'alpha': 1.0, 'batch_size': 256,
        'thres': 0.5, 'valid_epochs': 10, 'type': 'custom_pcgnn'
    }

    dgl_graph = Dataset(args.dataset, homo=1).graph # Using homo=1 as default
    dgl_graph = dgl.add_self_loop(dgl_graph)
    dgl_graph = sample_graph_by_nodes(dgl_graph, max_nodes=15000, seed=42)
    in_feats = dgl_graph.ndata['feature'].shape[1]
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_type_arg = args.model.lower()
    
    # Since BWGNN is removed, all models will use the PyG data object.
    # The conversion can be done unconditionally.
    print(f"Converting DGL graph to PyG Data object for model: {model_type_arg.upper()}")
    pyg_data = from_dgl(dgl_graph)
    
    if 'feature' in dgl_graph.ndata:
        pyg_data.x = dgl_graph.ndata['feature']
    elif 'feat' in dgl_graph.ndata:
        pyg_data.x = dgl_graph.ndata['feat']
    else:
        raise ValueError("Node features ('feature' or 'feat') not found in DGL graph.")

    if 'label' in dgl_graph.ndata:
        pyg_data.y = dgl_graph.ndata['label']
    else:
        raise ValueError("Node labels ('label') not found in DGL graph.")

    pyg_data_for_experiment = pyg_data

    final_mf1s, final_aucs = [], []
    print(f"\n--- Starting {args.run} Run(s) for {args.model.upper()} on {args.dataset} ---")

    for tt in range(args.run):
        print(f"\n--- Run {tt + 1}/{args.run} ---")
        
        mf1_run, auc_run = 0.0, 0.0
        
        # The 'if' for 'bwgnn' is removed, the first condition is now for 'pyg_standard'
        if MODEL_PARAMS.get(model_type_arg, {}).get('type') == 'pyg_standard':
            params = MODEL_PARAMS[model_type_arg]
            model_class = {'mlp': MLP, 'sage': SAGE, 'gcn': GCN, 'graphconv': GCGraphConvModel}[model_type_arg]
            model = model_class(in_channels=in_feats, hidden_channels=args.hid_dim, out_channels=num_classes, 
                                num_layers=params['num_layers'], dropout=params['dropout'], batchnorm=params['batchnorm'])
            model.to(device)
            mf1_run, auc_run = train_standard_pyg(model, pyg_data_for_experiment.clone(), args, device, params)

        elif model_type_arg == 'cola':
            mf1_run, auc_run = run_cola_experiment(pyg_data_for_experiment.clone(), args, device, in_feats)
        
        elif model_type_arg == 'gae':
            mf1_run, auc_run = run_gae_experiment(pyg_data_for_experiment.clone(), args, device, in_feats)

        elif model_type_arg == 'pcgnn':
            mf1_run, auc_run = run_pcgnn_experiment(pyg_data_for_experiment.clone(), args, device, in_feats)

        else:
            # Raise error if model type is not recognized (and isn't bwgnn)
            raise ValueError(f"Model {model_type_arg} initialization or run logic failed for run {tt + 1}")

        final_mf1s.append(mf1_run)
        final_aucs.append(auc_run)
        print(f"--- Run {tt + 1} completed. MF1: {mf1_run:.4f}, AUC: {auc_run:.4f} ---")

    print("\n--- All Runs Completed ---")
    print(f'Dataset: {args.dataset}, Model: {args.model.upper()}')
    print('Macro-F1 Mean: {:.2f}% ({:.2f} std) | AUC Mean: {:.4f} ({:.4f} std)'.format(
        100 * np.mean(final_mf1s), 100 * np.std(final_mf1s),
        np.mean(final_aucs), np.std(final_aucs)
    ))