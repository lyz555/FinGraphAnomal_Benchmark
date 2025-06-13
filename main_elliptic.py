import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score 
from torch_geometric.data import Data
from torch_geometric.nn import GAE as PyGGAE
import time
import argparse

try:
    from models import MLP, GCN, SAGE, GCGraphConvModel
    from models.gae import GAEWrapper 
    from models.pcgnn import pcgnn
    from pygod.detector import CoLA
    from utils.evaluator import Evaluator 
    from models.gae import GCNEncoder

except ImportError as e:
    print(f"Error importing from models directory, utils, or pygod: {e}")
    print("Please ensure a 'models' directory with necessary model files exists and is importable,")
    print("and that 'utils.evaluator.Evaluator' is available if using the detailed GAE evaluation.")
    exit()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS_SUPERVISED = 200

# Model parameters as defined in your script
mlp_parameters = {'lr':0.01, 'num_layers':2, 'hidden_channels':128, 'dropout':0.0, 'batchnorm': False, 'l2':5e-7}
gcn_parameters = {'lr':0.01, 'num_layers':2, 'hidden_channels':128, 'dropout':0.0, 'batchnorm': False, 'l2':5e-7}
sage_parameters = {'lr':0.01, 'num_layers':2, 'hidden_channels':128, 'dropout':0.0, 'batchnorm': False, 'l2':5e-7}
graphconv_parameters = {'lr':0.01, 'num_layers':2, 'hidden_channels':128, 'dropout':0.0, 'batchnorm': False, 'l2':5e-7}
cola_user_params = {'lr': 0.01,'hidden_channels': 128,'num_layers': 2,'dropout': 0.0,  'l2': 5e-7}

cuda_device_index = -1
if DEVICE.type == 'cuda':
    cuda_device_index = DEVICE.index if DEVICE.index is not None else 0

gae_script_params = {
    'lr': 0.01,
    'emb_size': 128,
    'epochs': 200, 
    'cuda_id': cuda_device_index
}
pcgnn_script_params = {
    'lr': 0.01, 'num_layers': 2, 'hidden_channels': 128, 'dropout': 0.0,
    'batchnorm': False, 'l2': 5e-7, 'batch_size_gnn': 64, 'batch_size_pca': 256,
    'epochs_gnn': 100, 'epochs_pca': 100, 'lambda_1': 0.1, 'thres': 0.5,
    'gpu': cuda_device_index,
    'seed': 42 
}


def load_elliptic_data():
    print("Loading Elliptic dataset...")
    try:
        df_features = pd.read_csv('dataset/elliptic/elliptic_txs_features.csv', header=None)
        df_edges = pd.read_csv('dataset/elliptic/elliptic_txs_edgelist.csv', header=0)
        df_classes = pd.read_csv('dataset/elliptic/elliptic_txs_classes.csv')
    except FileNotFoundError as e_file:
        print(f"Error: Dataset file not found: {e_file}. Please check the path 'dataset/elliptic/'.")
        exit()

    if df_edges.empty:
        print("WARNING: The edge list DataFrame (df_edges) is empty after loading. Check the CSV file.")

    num_features_cols = df_features.shape[1] - 2
    feature_names = ['txId', 'timestep'] + [f'feature_{i}' for i in range(num_features_cols)]
    df_features.columns = feature_names

    try:
        feature_tx_ids = pd.to_numeric(df_features['txId'], errors='coerce').dropna().astype(int)
    except Exception as e:
        print(f"Error converting feature txId to numeric/int: {e}. Please check the txId column in features file.")
        feature_tx_ids = df_features['txId'] # Fallback

    unique_tx_ids_from_features = feature_tx_ids.unique()
    tx_id_map = {tx_id: i for i, tx_id in enumerate(unique_tx_ids_from_features)}
    num_nodes = len(tx_id_map)

    if num_nodes == 0:
        print("CRITICAL ERROR: tx_id_map is empty. No unique transaction IDs found in features file or failed conversion.")
        return Data(x=torch.empty(0,0), edge_index=torch.empty(2,0, dtype=torch.long), y=torch.empty(0, dtype=torch.long),
                    train_mask=torch.empty(0,dtype=torch.bool), val_mask=torch.empty(0,dtype=torch.bool), test_mask=torch.empty(0,dtype=torch.bool))

    temp_df_for_x = df_features[df_features['txId'].isin(unique_tx_ids_from_features)].drop_duplicates(subset=['txId']).set_index('txId')
    ordered_features_df = temp_df_for_x.reindex(unique_tx_ids_from_features)

    x_input_features = ordered_features_df.drop(columns=['timestep']).values.astype(np.float32)
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_input_features)
    x = torch.tensor(x_scaled, dtype=torch.float)

    if x.shape[0] != num_nodes:
        print(f"CRITICAL WARNING: Shape of feature matrix x ({x.shape[0]}) does not match num_nodes ({num_nodes}).")
    
        if num_nodes == 0 and x.shape[0] != 0 :
             return Data(x=torch.empty(0,0), edge_index=torch.empty(2,0, dtype=torch.long), y=torch.empty(0, dtype=torch.long),
                         train_mask=torch.empty(0,dtype=torch.bool), val_mask=torch.empty(0,dtype=torch.bool), test_mask=torch.empty(0,dtype=torch.bool))


    source_col_name = df_edges.columns[0]
    target_col_name = df_edges.columns[1]
    source_nodes_original_ids = df_edges[source_col_name]
    target_nodes_original_ids = df_edges[target_col_name]

    try:
        source_nodes_numeric = pd.to_numeric(source_nodes_original_ids, errors='coerce')
        target_nodes_numeric = pd.to_numeric(target_nodes_original_ids, errors='coerce')
    except Exception as e:
        print(f"Error converting edge txIds to numeric: {e}.")
        source_nodes_numeric = source_nodes_original_ids # Fallback
        target_nodes_numeric = target_nodes_original_ids # Fallback

    source_nodes_mapped = source_nodes_numeric.map(tx_id_map)
    target_nodes_mapped = target_nodes_numeric.map(tx_id_map)
    valid_edges_mask = source_nodes_mapped.notna() & target_nodes_mapped.notna()
    
    if valid_edges_mask.sum() == 0:
        print("CRITICAL WARNING: 0 valid edges after mapping. Graph will be empty.")
        print(f"   - Unique IDs in features used for map: {len(tx_id_map)}")
        print(f"   - Example keys from tx_id_map (if any): {list(tx_id_map.keys())[:5] if tx_id_map else 'N/A'}")
        print(f"   - Example edge source IDs (numeric, pre-map): {source_nodes_numeric.dropna().head().tolist()}")
        print(f"   - Example edge target IDs (numeric, pre-map): {target_nodes_numeric.dropna().head().tolist()}")
        source_nodes_final = np.array([], dtype=int)
        target_nodes_final = np.array([], dtype=int)
    else:
        source_nodes_final = source_nodes_mapped[valid_edges_mask].values.astype(int)
        target_nodes_final = target_nodes_mapped[valid_edges_mask].values.astype(int)
        
    edge_index = torch.tensor([source_nodes_final, target_nodes_final], dtype=torch.long)

    y_temp = torch.full((num_nodes,), -1, dtype=torch.long)
    for tx_id_class_orig, class_val_orig in zip(df_classes['txId'], df_classes['class']):
        try:
            tx_id_class_cleaned = int(pd.to_numeric(tx_id_class_orig, errors='coerce'))
            if pd.isna(tx_id_class_cleaned): continue
        except:
            continue

        if tx_id_class_cleaned in tx_id_map:
            node_idx = tx_id_map[tx_id_class_cleaned]
            if class_val_orig == '1': y_temp[node_idx] = 1 # Illicit
            elif class_val_orig == '2': y_temp[node_idx] = 0 # Licit

    node_id_to_timestep = pd.Series(index=range(num_nodes), dtype=float) 
    for tx_id_feat, ts_val in ordered_features_df['timestep'].items():
        if tx_id_feat in tx_id_map:
            node_idx = tx_id_map[tx_id_feat]
            node_id_to_timestep[node_idx] = ts_val

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    max_timestep_val = node_id_to_timestep.max() if not node_id_to_timestep.empty and node_id_to_timestep.notna().any() else 0
    
    if pd.isna(max_timestep_val): max_timestep_val = 0

    for i in range(num_nodes):
        if y_temp[i] != -1:
            ts = node_id_to_timestep.get(i)
            if ts is not None and not pd.isna(ts):
                ts_int = int(ts)
                if 1 <= ts_int <= 34: train_mask[i] = True
                elif 35 <= ts_int <= min(39, int(max_timestep_val)): val_mask[i] = True
                elif min(39, int(max_timestep_val)) < ts_int <= int(max_timestep_val): test_mask[i] = True
                            
    data_obj = Data(x=x, edge_index=edge_index, y=y_temp)
    data_obj.train_mask = train_mask
    data_obj.val_mask = val_mask
    data_obj.test_mask = test_mask
    data_obj.num_classes = 2 # Illicit (1) vs Licit (0)

    print("Dataset statistics:")
    print(f"   Number of nodes: {data_obj.num_nodes}")
    print(f"   Number of edges: {data_obj.num_edges}")
    print(f"   Number of features: {data_obj.num_node_features if data_obj.x is not None else 'N/A'}")
    print(f"   Train nodes: {data_obj.train_mask.sum().item()} (Licit: {(data_obj.y[data_obj.train_mask] == 0).sum().item()}, Illicit: {(data_obj.y[data_obj.train_mask] == 1).sum().item()})")
    print(f"   Val nodes: {data_obj.val_mask.sum().item()} (Licit: {(data_obj.y[data_obj.val_mask] == 0).sum().item()}, Illicit: {(data_obj.y[data_obj.val_mask] == 1).sum().item()})")
    print(f"   Test nodes: {data_obj.test_mask.sum().item()} (Licit: {(data_obj.y[data_obj.test_mask] == 0).sum().item()}, Illicit: {(data_obj.y[data_obj.test_mask] == 1).sum().item()})")
    
    if data_obj.num_edges == 0 and (selected_model_to_run != 'mlp' and selected_model_to_run != 'all_non_gnn_only'):
        print("\nCRITICAL DATA ISSUE: The graph has 0 edges. GNN models will likely fail or produce trivial results.")
        print("Please meticulously check your CSV files for txId consistency (type and values) between features and edgelist.")

    return data_obj


def train_supervised_epoch(model, data, optimizer, criterion, model_key):
    model.train()
    optimizer.zero_grad()
    
    if model_key == 'mlp':
        out = model(data.x)
    else:
        if data.num_edges == 0 and model_key not in ['mlp']: 
            return torch.tensor(0.0, device=DEVICE) 
        out = model(data.x, data.edge_index)
        
    if data.train_mask.sum() == 0:
        return torch.tensor(0.0, device=DEVICE)

    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    if torch.is_tensor(loss) and loss.requires_grad:
        loss.backward()
        optimizer.step()
        return loss.item()
    return 0.0


@torch.no_grad()
def test_supervised_model(model, data, model_name="mlp"):
    model.eval()
    model_name_lower = model_name.lower()

    if model_name_lower == 'mlp':
        out = model(data.x)
    else:
        if data.num_edges == 0:
            if model_name_lower in ['sage', 'gcn', 'graphconv']:
                 out = torch.log(torch.ones((data.num_nodes, data.num_classes), device=data.x.device) / data.num_classes)
            else: 
                 out = torch.zeros((data.num_nodes, data.num_classes), device=data.x.device)
        else:
            out = model(data.x, data.edge_index)

    if model_name_lower == 'mlp':
        prob_out = F.softmax(out, dim=1)
    elif model_name_lower in ['sage', 'gcn', 'graphconv']:
        prob_out = torch.exp(out) 
    else:
        raise ValueError(f"Unsupported model_name for output processing: {model_name}")

    # Dictionaries to hold all metrics for each split
    metrics = {split: {} for split in ['Train', 'Val', 'Test']}
    
    for split_name, mask in [('Train', data.train_mask), ('Val', data.val_mask), ('Test', data.test_mask)]:
        if mask.sum().item() == 0: # No nodes in this split
            metrics[split_name] = {'auc': 0.0, 'f1_macro': 0.0, 'f1_1': 0.0, 'f1_0': 0.0, 'acc': 0.0}
            continue
        
        y_true_split = data.y[mask].cpu().numpy()
        y_pred_proba_split_positive_class = prob_out[mask, 1].detach().cpu().numpy()
        y_pred_binary_split = (y_pred_proba_split_positive_class > 0.5).astype(int)

        valid_indices = y_true_split != -1
        
        if not np.any(valid_indices):
            metrics[split_name] = {'auc': 0.0, 'f1_macro': 0.0, 'f1_1': 0.0, 'f1_0': 0.0, 'acc': 0.0}
            continue

        y_true_filtered = y_true_split[valid_indices]
        y_pred_proba_filtered = y_pred_proba_split_positive_class[valid_indices]
        y_pred_binary_filtered = y_pred_binary_split[valid_indices]

        # AUC Score
        if len(np.unique(y_true_filtered)) > 1:
            metrics[split_name]['auc'] = roc_auc_score(y_true_filtered, y_pred_proba_filtered)
        else:
            metrics[split_name]['auc'] = 0.0

        # F1 and Accuracy Scores
        metrics[split_name]['f1_macro'] = f1_score(y_true_filtered, y_pred_binary_filtered, average='macro', zero_division=0)
        metrics[split_name]['f1_1'] = f1_score(y_true_filtered, y_pred_binary_filtered, pos_label=1, average='binary', zero_division=0)
        metrics[split_name]['f1_0'] = f1_score(y_true_filtered, y_pred_binary_filtered, pos_label=0, average='binary', zero_division=0)
        metrics[split_name]['acc'] = accuracy_score(y_true_filtered, y_pred_binary_filtered)

    # Unpack for returning
    train_metrics = metrics['Train']
    val_metrics = metrics['Val']
    test_metrics = metrics['Test']

    return (train_metrics['auc'], val_metrics['auc'], test_metrics['auc'],
            train_metrics['f1_macro'], val_metrics['f1_macro'], test_metrics['f1_macro'],
            train_metrics['f1_1'], val_metrics['f1_1'], test_metrics['f1_1'],
            train_metrics['f1_0'], val_metrics['f1_0'], test_metrics['f1_0'],
            train_metrics['acc'], val_metrics['acc'], test_metrics['acc'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test GNN models on the Elliptic dataset.")
    parser.add_argument(
        '--model', type=str, default='all',
        choices=['mlp', 'sage', 'gcn', 'graphconv', 'cola', 'gae', 'pcgnn', 'all'],
        help="Specify the model to test."
    )
    args = parser.parse_args()
    selected_model_to_run = args.model.lower()

    data = load_elliptic_data().to(DEVICE)

    split_idx = None
    if selected_model_to_run == 'all' or selected_model_to_run in ['gae', 'pcgnn', 'cola']:
        print("Creating split_idx for GAE/PCGNN/CoLA evaluation...")
        if data.train_mask.sum() > 0 and data.val_mask.sum() > 0 and data.test_mask.sum() > 0:
            split_idx = {
                'train': data.train_mask.nonzero(as_tuple=False).view(-1).to(DEVICE),
                'valid': data.val_mask.nonzero(as_tuple=False).view(-1).to(DEVICE),
                'test': data.test_mask.nonzero(as_tuple=False).view(-1).to(DEVICE)
            }
            print(f"   Train indices: {split_idx['train'].size(0)}")
            print(f"   Valid indices: {split_idx['valid'].size(0)}")
            print(f"   Test indices: {split_idx['test'].size(0)}")
        else:
            print("Warning: Cannot create valid split_idx due to empty masks. Some models might fail or yield 0 scores.")
            split_idx = {
                'train': torch.tensor([], dtype=torch.long, device=DEVICE), 
                'valid': torch.tensor([], dtype=torch.long, device=DEVICE), 
                'test': torch.tensor([], dtype=torch.long, device=DEVICE)
            }

    supervised_models_config = {
        "mlp": {"model_class": MLP, "params": mlp_parameters, "criterion": nn.CrossEntropyLoss()},
        "sage": {"model_class": SAGE, "params": sage_parameters, "criterion": nn.NLLLoss()},
        "gcn": {"model_class": GCN, "params": gcn_parameters, "criterion": nn.NLLLoss()},
        "graphconv": {"model_class": GCGraphConvModel, "params": graphconv_parameters, "criterion": nn.NLLLoss()}
    }

    if selected_model_to_run == 'all' or selected_model_to_run in supervised_models_config:
        print(f"\n--- Testing Standard Supervised Models (Epochs: {EPOCHS_SUPERVISED}) on {DEVICE} ---")
        for model_key, config in supervised_models_config.items():
            if selected_model_to_run == 'all' or selected_model_to_run == model_key:
                model_name_display = model_key.upper()
                print(f"\n--- Testing {model_name_display} ---")

                params_script = config["params"].copy()
                model_constructor_args = {
                    'in_channels': data.num_node_features,
                    'hidden_channels': params_script['hidden_channels'],
                    'out_channels': data.num_classes,
                    'num_layers': params_script['num_layers'],
                    'dropout': params_script['dropout']
                }
                if 'batchnorm' in params_script:
                    model_constructor_args['batchnorm'] = params_script['batchnorm']
                
                if data.num_node_features == 0:
                    print(f"Skipping {model_name_display} as num_node_features is 0.")
                    continue

                model_instance = config["model_class"](**model_constructor_args).to(DEVICE)
                optimizer = torch.optim.Adam(model_instance.parameters(), lr=params_script['lr'], weight_decay=params_script['l2'])
                criterion = config["criterion"]
                print(f"{model_name_display} Model: {model_instance}")
                
                # Variables to store metrics corresponding to the best validation AUC
                best_val_auc = 0.0
                final_test_metrics = {}

                for epoch in range(1, EPOCHS_SUPERVISED + 1):
                    loss = train_supervised_epoch(model_instance, data, optimizer, criterion, model_key=model_key)
                    if epoch % 10 == 0 or epoch == EPOCHS_SUPERVISED:
                        # Unpack all 15 performance metrics
                        metrics_tuple = test_supervised_model(model_instance, data, model_name=model_key)
                        
                        # Assign to readable names
                        _train_auc, val_auc, test_auc, \
                        _train_f1_macro, _val_f1_macro, test_f1_macro, \
                        _train_f1_1, _val_f1_1, test_f1_1, \
                        _train_f1_0, _val_f1_0, test_f1_0, \
                        _train_acc, _val_acc, test_acc = metrics_tuple

                        if val_auc > best_val_auc:
                            best_val_auc = val_auc
                            final_test_metrics = {
                                'auc': test_auc, 'f1_macro': test_f1_macro, 'f1_1': test_f1_1,
                                'f1_0': test_f1_0, 'acc': test_acc
                            }
                        
                        if epoch % 20 == 0 or epoch == EPOCHS_SUPERVISED: # Print less frequently
                             print(f'{model_name_display} Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {final_test_metrics.get("auc", 0):.4f}')
                
                print(f"{model_name_display} Final Results (based on best validation AUC):")
                print(f"  -> Best Val AUC: {best_val_auc:.4f}")
                print(f"  -> Test AUC: {final_test_metrics.get('auc', 0):.4f}")
                print(f"  -> Test ACC: {final_test_metrics.get('acc', 0):.4f}")
                print(f"  -> Test F1-Macro: {final_test_metrics.get('f1_macro', 0):.4f}")
                print(f"  -> Test F1-Illicit(1): {final_test_metrics.get('f1_1', 0):.4f}")
                print(f"  -> Test F1-Licit(0): {final_test_metrics.get('f1_0', 0):.4f}\n" + "-" * 40)


    if selected_model_to_run == 'all' or selected_model_to_run == 'cola':
        print("\n--- Testing CoLA ---")
        if data.num_edges == 0 or data.num_nodes == 0 or data.num_node_features == 0:
            print("Skipping CoLA due to 0 edges, 0 nodes, or 0 node features in the graph.")
        elif split_idx is None or split_idx['test'].numel() == 0 :
             print("Skipping CoLA due to missing or empty test indices in split_idx for evaluation.")
        else:
            contamination = 0.1
            if data.train_mask.sum().item() > 0:
                num_illicit_train = (data.y[data.train_mask] == 1).sum().item()
                num_total_train = data.train_mask.sum().item()
                if num_total_train > 0:
                    contamination = num_illicit_train / num_total_train
                    if contamination == 0:
                        print("Warning: No illicit nodes in training set for CoLA contamination, using default 0.1.")
                        contamination = 0.1 
                else:
                       print("Warning: No training data for CoLA contamination, using default 0.1.")
            else:
                print("Warning: Could not calculate contamination for CoLA from training data, using default 0.1.")
            
            print(f"CoLA Contamination (estimated from train set or default): {contamination:.4f}")

            data_for_cola = data.clone()

            cola_model = CoLA(epoch=EPOCHS_SUPERVISED, contamination=contamination, lr=cola_user_params['lr'],
                                num_layers=cola_user_params['num_layers'], hid_dim=cola_user_params['hidden_channels'],
                                dropout=cola_user_params['dropout'], weight_decay=cola_user_params['l2'],
                                verbose=0, gpu=cuda_device_index)
            
            print(f"Fitting CoLA model for {EPOCHS_SUPERVISED} epochs on specified device (GPU index: {cuda_device_index})...")
            start_time_cola = time.time()
            try:
                cola_model.fit(data_for_cola)
                fit_duration_cola = time.time() - start_time_cola
                print(f"CoLA fitting complete. Fit Time: {fit_duration_cola:.2f}s")

                decision_scores = cola_model.decision_score_
                predicted_labels_cola = cola_model.predict(data_for_cola)

                test_mask_np = data.test_mask.cpu().numpy()
                y_true_cola_test = data.y.cpu().numpy()[test_mask_np]
                
                scores_cola_test = decision_scores[test_mask_np]
                pred_labels_cola_test = predicted_labels_cola[test_mask_np]

                valid_test_indices_cola = y_true_cola_test != -1

                if np.any(valid_test_indices_cola):
                    y_true_filtered = y_true_cola_test[valid_test_indices_cola]
                    scores_filtered = scores_cola_test[valid_test_indices_cola]
                    pred_labels_filtered = pred_labels_cola_test[valid_test_indices_cola]

                    # Calculate all metrics for CoLA
                    cola_test_auc = roc_auc_score(y_true_filtered, scores_filtered) if len(np.unique(y_true_filtered)) > 1 else 0.0
                    cola_test_acc = accuracy_score(y_true_filtered, pred_labels_filtered)
                    cola_test_f1_macro = f1_score(y_true_filtered, pred_labels_filtered, average='macro', zero_division=0)
                    cola_test_f1_1 = f1_score(y_true_filtered, pred_labels_filtered, pos_label=1, average='binary', zero_division=0)
                    cola_test_f1_0 = f1_score(y_true_filtered, pred_labels_filtered, pos_label=0, average='binary', zero_division=0)
                    
                    print(f"CoLA Test Results:")
                    print(f"  -> Test AUC: {cola_test_auc:.4f}")
                    print(f"  -> Test ACC: {cola_test_acc:.4f}")
                    print(f"  -> Test F1-Macro: {cola_test_f1_macro:.4f}")
                    print(f"  -> Test F1-Illicit(1): {cola_test_f1_1:.4f}")
                    print(f"  -> Test F1-Licit(0): {cola_test_f1_0:.4f}")
                else:
                    print("CoLA: Not enough classes or no valid labeled test samples for metric calculation.")
            except Exception as e_cola_fit:
                print(f"Error during CoLA fitting or evaluation: {e_cola_fit}")
                import traceback
                traceback.print_exc()
        print("-" * 40)


    if selected_model_to_run == 'all' or selected_model_to_run == 'gae':
        print("\n--- Testing GAE ---")
        print(f"GAE Parameters: {gae_script_params}")
        if data.num_edges == 0 or data.num_nodes == 0 or data.num_node_features == 0:
            print("Skipping GAE due to 0 edges, 0 nodes, or 0 node features in the graph.")
        elif split_idx is None or split_idx['train'].numel() == 0 or split_idx['test'].numel() == 0 :
            print("Skipping GAE due to missing or empty train/test indices in split_idx.")
        else:
            try:
                start_time_gae = time.time()
                
                gae_model_actual = PyGGAE(
                    encoder=GCNEncoder(
                        in_channels=data.num_node_features,
                        out_channels=gae_script_params['emb_size']
                    )
                ).to(DEVICE)
                
                optimizer_gae = torch.optim.Adam(gae_model_actual.parameters(), lr=gae_script_params['lr'])
                gae_epochs = gae_script_params.get('epochs', 200)

                print(f"Training PyG GAE model for {gae_epochs} epochs...")
                for epoch in range(1, gae_epochs + 1):
                    gae_model_actual.train()
                    optimizer_gae.zero_grad()
                    z = gae_model_actual.encode(data.x, data.edge_index)
                    loss = gae_model_actual.recon_loss(z, data.edge_index) 
                    loss.backward()
                    optimizer_gae.step()
                    if epoch % (gae_epochs // 10 or 1) == 0 or epoch == gae_epochs:
                        print(f"GAE Training Epoch {epoch}/{gae_epochs}, Recon Loss: {loss.item():.4f}")

                gae_model_actual.eval()
                with torch.no_grad():
                    final_z = gae_model_actual.encode(data.x, data.edge_index).cpu()

                from sklearn.linear_model import LogisticRegression

                split_idx_cpu = {k: v.cpu() for k, v in split_idx.items()}
                
                train_embs = final_z[split_idx_cpu['train']]
                train_labels_all = data.y.cpu()[split_idx_cpu['train']] 
                
                test_embs = final_z[split_idx_cpu['test']]
                test_labels_all = data.y.cpu()[split_idx_cpu['test']]

                valid_train_lr_mask = train_labels_all != -1
                train_embs_lr_filtered = train_embs[valid_train_lr_mask]
                train_labels_lr_filtered = train_labels_all[valid_train_lr_mask]

                valid_test_lr_mask = test_labels_all != -1
                test_embs_lr_filtered = test_embs[valid_test_lr_mask]
                test_labels_lr_filtered_for_eval = test_labels_all[valid_test_lr_mask]
                
                if train_embs_lr_filtered.shape[0] > 0 and len(np.unique(train_labels_lr_filtered)) > 1:
                    lr = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
                    lr.fit(train_embs_lr_filtered, train_labels_lr_filtered)
                    
                    if test_embs_lr_filtered.shape[0] > 0 and len(np.unique(test_labels_lr_filtered_for_eval)) > 0 :
                        test_pred_probs_lr_positive_class = lr.predict_proba(test_embs_lr_filtered)[:, 1]
                        test_pred_binary_lr = lr.predict(test_embs_lr_filtered)
                        
                        # Calculate all metrics for GAE+LR
                        gae_test_auc = roc_auc_score(test_labels_lr_filtered_for_eval, test_pred_probs_lr_positive_class) if len(np.unique(test_labels_lr_filtered_for_eval)) > 1 else 0.0
                        gae_test_acc = accuracy_score(test_labels_lr_filtered_for_eval, test_pred_binary_lr)
                        gae_test_f1_macro = f1_score(test_labels_lr_filtered_for_eval, test_pred_binary_lr, average='macro', zero_division=0)
                        gae_test_f1_1 = f1_score(test_labels_lr_filtered_for_eval, test_pred_binary_lr, pos_label=1, average='binary', zero_division=0)
                        gae_test_f1_0 = f1_score(test_labels_lr_filtered_for_eval, test_pred_binary_lr, pos_label=0, average='binary', zero_division=0)

                        fit_duration_gae = time.time() - start_time_gae 
                        print(f"GAE training and downstream evaluation complete. Fit Time: {fit_duration_gae:.2f}s")
                        print(f"GAE+LR Test Results:")
                        print(f"  -> Test AUC: {gae_test_auc:.4f}")
                        print(f"  -> Test ACC: {gae_test_acc:.4f}")
                        print(f"  -> Test F1-Macro: {gae_test_f1_macro:.4f}")
                        print(f"  -> Test F1-Illicit(1): {gae_test_f1_1:.4f}")
                        print(f"  -> Test F1-Licit(0): {gae_test_f1_0:.4f}")
                    else:
                        print("GAE+LR: No valid labeled test samples for Logistic Regression evaluation after filtering.")
                else:
                    print("GAE+LR: Not enough training samples or classes for Logistic Regression after filtering.")

            except Exception as e_gae:
                print(f"Error during GAE testing: {e_gae}")
                import traceback
                traceback.print_exc() 
        print("-" * 40)


    if selected_model_to_run == 'all' or selected_model_to_run == 'pcgnn':
        print("\n--- Testing PCGNN ---")
        print(f"PCGNN Parameters: {pcgnn_script_params}")
        if data.num_edges == 0 or data.num_nodes == 0 or data.num_node_features == 0:
            print("Skipping PCGNN due to 0 edges, 0 nodes, or 0 node features in the graph.")
        elif split_idx is None or split_idx['train'].numel() == 0 or split_idx['test'].numel() == 0:
            print("Skipping PCGNN due to missing or empty train/test indices in split_idx.")
        else:
            try:
                pcgnn_feat_data = data.x.clone()
                pcgnn_labels = data.y.clone()
                pcgnn_edge_index = data.edge_index.clone()

                pcgnn_model_instance = pcgnn(
                    config=pcgnn_script_params,
                    feat_data=pcgnn_feat_data, 
                    labels=pcgnn_labels,
                    edge_index=pcgnn_edge_index,
                    split_idx=split_idx
                )
                print(f"Fitting PCGNN model (epochs defined in PCGNN config)...")
                print("NOTE: For full PCGNN metrics, ensure its 'fit' method returns (epoch, auc, f1_macro, f1_1, f1_0, acc).")
                start_time_pcgnn = time.time()
                
                # IMPORTANT: Assumes the pcgnn.fit() method is modified to return all metrics.
                # If it only returns (epoch, auc, f1), the line below will cause an error.
                # You must modify the 'pcgnn.py' file.
                returned_values_pcgnn = pcgnn_model_instance.fit() 
                

                # Check what was returned and unpack accordingly for compatibility
                if isinstance(returned_values_pcgnn, tuple) and len(returned_values_pcgnn) == 6:
                      _epoch_pcgnn, auc_gnn, f1_macro, f1_binary_1, f1_binary_0, overall_acc = returned_values_pcgnn
                elif isinstance(returned_values_pcgnn, tuple) and len(returned_values_pcgnn) == 3:
                    _epoch_pcgnn, pcgnn_final_auc, pcgnn_final_f1_1 = returned_values_pcgnn
                    print("Warning: PCGNN fit() returned 3 values. Only AUC and F1-Illicit(1) are available.")
                else:
                    print("Warning: PCGNN fit() method did not return the expected tuple. Metrics will be 0.")
                
                fit_duration_pcgnn = time.time() - start_time_pcgnn
                print(f"PCGNN fitting complete. Fit Time: {fit_duration_pcgnn:.2f}s")
                print(f"Final GNN Test Results:")
                print(f"  -> AUC: {auc_gnn:.4f}")
                print(f"  -> F1-macro: {f1_macro:.4f}")
                print(f"  -> F1-Illicit(1): {f1_binary_1:.4f}")
                print(f"  -> F1-Licit(0): {f1_binary_0:.4f}")
                print(f"  -> Accuracy: {overall_acc:.4f}")

            except Exception as e_pcgnn:
                print(f"Error during PCGNN testing: {e_pcgnn}")
                import traceback
                traceback.print_exc()
        print("-" * 40)

    print("\n--- Script Execution Complete ---")