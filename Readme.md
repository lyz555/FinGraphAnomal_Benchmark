Of course, here is a README for your project.

# FinGraphAnomal: A Benchmark for Anomaly Detection on Financial Graphs

This project establishes a benchmark for anomaly detection on graph-niched nodes within the financial domain. It evaluates various models on several financial datasets and provides a framework for future research in this area.

## Overview

Financial systems are often represented as graphs where nodes can be users, institutions, or transactions, and edges represent the relationships or interactions between them. Identifying anomalous nodes in these graphs is crucial for detecting fraudulent activities, money laundering, and other illicit behaviors. This benchmark provides a standardized environment to evaluate and compare different anomaly detection methods on financial graph data.

This repository contains the implementation of seven different models tested on three datasets: **DGraph**, **T-Finance**, and **Elliptic**. The framework is designed to be easily extensible for new models and datasets.

## Datasets

The benchmark utilizes three datasets, each representing different aspects of financial networks.

  * **DGraph**: A financial graph dataset where the task is to identify anomalous nodes.
  * **T-Finance**: A transaction-based graph from a financial services company.
  * **Elliptic**: A graph of Bitcoin transactions, with the task of identifying transactions associated with illicit activities.

All datasets can be downloaded from the following link: [Financial Graph Datasets](https://drive.google.com/drive/folders/16R3uQ9eLrUq3ecd7cFaPsOv5vG_D5_tf?usp=drive_link)

After downloading, place the data into a `dataset` directory in the project's root folder.

## Models

A variety of models, from simple MLPs to more complex Graph Neural Networks (GNNs), have been implemented and tested. The implemented models include:

  * **MLP (Multi-Layer Perceptron)**: A fundamental feedforward neural network.
  * **GCN (Graph Convolutional Network)**: A type of GNN that uses graph convolutions.
  * **GraphSAGE**: An inductive GNN framework that aggregates features from a node's local neighborhood.
  * **GraphConv**: A GNN model that applies graph convolutions.
  * **GAE (Graph Autoencoder)**: An unsupervised model that learns node representations by reconstructing the graph structure.
  * **CoLA (Contrastive-based Outlier-aware GNN)**: An anomaly detection model that uses contrastive learning.
  * **PC-GNN (Pick and Choose GNN)**: A GNN-based approach for imbalanced learning, suitable for fraud detection.

## Installation

To run the experiments, you need to install the required dependencies. It is recommended to use a virtual environment.

```bash
pip install -r requirement.txt --index-url https://download.pytorch.org/whl/cu121 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html -f https://data.dgl.ai/wheels/repo.html
```

## Running the Experiments

You can run the experiments for each dataset and model using the provided main scripts. The `--model` argument allows you to specify which model to test.

### DGraph Dataset

To run a model on the DGraph dataset, use the `main_dgraph.py` script:

```bash
python main_dgraph.py --model [MODEL_NAME]
```

For example, to test the **GCN** model:

```bash
python main_dgraph.py --model gcn
```

### Elliptic Dataset

To run a model on the Elliptic dataset, use the `main_elliptic.py` script:

```bash
python main_elliptic.py --model [MODEL_NAME]
```

For example, to test the **MLP** model:

```bash
python main_elliptic.py --model mlp
```

### T-Finance Dataset

To run a model on the T-Finance dataset, use the `main_tf.py` script:

```bash
python main_tf.py --model [MODEL_NAME]
```

For example, to test the **GraphSAGE** model:

```bash
python main_tf.py --model sage
```

Replace `[MODEL_NAME]` with one of the following: `mlp`, `gcn`, `sage`, `graphconv`, `gae`, `cola`, or `pcgnn`.

## Project Structure

```
.
├── dataset/                # Directory for datasets
├── model_results/          # Directory to save model results
├── models/                 # Model implementations
│   ├── __init__.py
│   ├── gae.py
│   ├── gcn.py
│   ├── graphconv.py
│   ├── mlp.py
│   ├── pcgnn.py
│   └── sage.py
├── utils/                  # Utility scripts
│   ├── __init__.py
│   ├── dgraphfin.py
│   ├── evaluator.py
│   └── utils.py
├── .gitignore
├── logger.py               # Logging utility
├── main_dgraph.py          # Main script for DGraph dataset
├── main_elliptic.py        # Main script for Elliptic dataset
├── main_tf.py              # Main script for T-Finance dataset
├── requirement.txt         # Project dependencies
└── tfclean.py              # Data cleaning script for T-Finance
```

## Citation

If you use this benchmark in your research, please consider citing this repository.

## Contributing

Contributions are welcome\! If you would like to add new models, datasets, or features, please open an issue or submit a pull request.
