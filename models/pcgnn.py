# PC-GNN Unified Module
from tqdm import tqdm
from collections import defaultdict
class pcgnn:
	def __init__(self, config, feat_data, labels, edge_index, split_idx):
		self.handler = ModelHandler(config, feat_data, labels, edge_index, split_idx)

	def fit(self):
		return self.handler.train()

# ===== File: model.py =====
import torch
import torch.nn as nn
from torch.nn import init


"""
	PC-GNN Model
	Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
	Modified from https://github.com/YingtongDou/CARE-GNN
"""


class PCALayer(nn.Module):
	"""
	One Pick-Choose-Aggregate layer
	"""

	def __init__(self, num_classes, inter1, lambda_1):
		"""
		Initialize the PC-GNN model
		:param num_classes: number of classes (2 in our paper)
		:param inter1: the inter-relation aggregator that output the final embedding
		"""
		super(PCALayer, self).__init__()
		self.inter1 = inter1
		self.xent = nn.CrossEntropyLoss()

		# the parameter to transform the final embedding
		self.weight = nn.Parameter(torch.FloatTensor(num_classes, inter1.embed_dim))
		init.xavier_uniform_(self.weight)
		self.lambda_1 = lambda_1
		self.epsilon = 0.1

	def forward(self, nodes, labels, train_flag=True):
		embeds1, label_scores = self.inter1(nodes, labels, train_flag)
		scores = self.weight.mm(embeds1)
		return scores.t(), label_scores

	def to_prob(self, nodes, labels, train_flag=True):
		gnn_logits, label_logits = self.forward(nodes, labels, train_flag)
		gnn_scores = torch.sigmoid(gnn_logits)
		label_scores = torch.sigmoid(label_logits)
		return gnn_scores, label_scores

	def loss(self, nodes, labels, train_flag=True):
		gnn_scores, label_scores = self.forward(nodes, labels, train_flag)
		# Simi loss, Eq. (7) in the paper
		label_loss = self.xent(label_scores, labels.squeeze())
		# GNN loss, Eq. (10) in the paper
		gnn_loss = self.xent(gnn_scores, labels.squeeze())
		# the loss function of PC-GNN, Eq. (11) in the paper
		final_loss = gnn_loss + self.lambda_1 * label_loss
		return final_loss


# ===== File: layers.py =====
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

from operator import itemgetter
import math

"""
	PC-GNN Layers
	Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
	Modified from https://github.com/YingtongDou/CARE-GNN
"""


class InterAgg(nn.Module):

	def __init__(self, features, feature_dim, embed_dim, 
				 train_pos, adj_lists, intraggs, inter='GNN', cuda=True):
		"""
		Initialize the inter-relation aggregator
		:param features: the input node features or embeddings for all nodes
		:param feature_dim: the input dimension
		:param embed_dim: the embed dimension
		:param train_pos: positive samples in training set
		:param adj_lists: a list of adjacency lists for each single-relation graph
		:param intraggs: the intra-relation aggregators used by each single-relation graph
		:param inter: NOT used in this version, the aggregator type: 'Att', 'Weight', 'Mean', 'GNN'
		:param cuda: whether to use GPU
		"""
		super(InterAgg, self).__init__()

		self.features = features
		self.dropout = 0.6
		self.adj_lists = adj_lists
		self.intra_agg1 = intraggs[0]
		self.intra_agg2 = intraggs[1]
		self.intra_agg3 = intraggs[2]
		self.embed_dim = embed_dim
		self.feat_dim = feature_dim
		self.inter = inter
		self.cuda = cuda
		self.intra_agg1.cuda = cuda
		self.intra_agg2.cuda = cuda
		self.intra_agg3.cuda = cuda
		self.train_pos = train_pos

		# initial filtering thresholds
		self.thresholds = [0.5, 0.5, 0.5]

		# parameter used to transform node embeddings before inter-relation aggregation
		self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim*len(intraggs)+self.feat_dim, self.embed_dim))
		init.xavier_uniform_(self.weight)

		# label predictor for similarity measure
		self.label_clf = nn.Linear(self.feat_dim, 2)

		# initialize the parameter logs
		self.weights_log = []
		self.thresholds_log = [self.thresholds]
		self.relation_score_log = []

	def forward(self, nodes, labels, train_flag=True):
		"""
		:param nodes: a list of batch node ids
		:param labels: a list of batch node labels
		:param train_flag: indicates whether in training or testing mode
		:return combined: the embeddings of a batch of input node features
		:return center_scores: the label-aware scores of batch nodes
		"""

		# extract 1-hop neighbor ids from adj lists of each single-relation graph
		to_neighs = []
		for adj_list in self.adj_lists:
			to_neighs.append([set(adj_list[int(node)]) for node in nodes])

		# find unique nodes and their neighbors used in current batch
		# find unique nodes and their neighbors used in current batch
		# Make sure we include all relevant nodes
		unique_nodes = set()
		unique_nodes.update([int(n) for n in nodes])  # center nodes

		for rel_neigh in to_neighs:
			for neigh_set in rel_neigh:
				unique_nodes.update([int(n) for n in neigh_set])  # neighbors

		unique_nodes.update([int(n) for n in self.train_pos])  # positive training nodes

		# Now compute features
		# === Use GPU-0 ===
		device = torch.device("cuda:0")

		# Ensure all IDs are integers
		unique_nodes = set(int(n) for n in nodes)
		for rel_neigh in to_neighs:
			for neigh_set in rel_neigh:
				unique_nodes.update(int(n) for n in neigh_set)
		unique_nodes.update(int(n) for n in self.train_pos)

		unique_nodes = list(unique_nodes)
		id_mapping = {node_id: idx for idx, node_id in enumerate(unique_nodes)}

		# Move to cuda:0
		unique_nodes_tensor = torch.tensor(unique_nodes, dtype=torch.long, device=device)
		train_pos_tensor = torch.tensor(self.train_pos, dtype=torch.long, device=device)

		# Lookup features on GPU
		batch_features = self.features(unique_nodes_tensor)
		pos_features = self.features(train_pos_tensor)

		# Apply label classifier
		batch_scores = self.label_clf(batch_features)
		pos_scores = self.label_clf(pos_features)

		# Map center node scores
		node_ids = [int(n) for n in nodes]
		try:
			center_idx_tensor = torch.tensor([id_mapping[n] for n in node_ids], dtype=torch.long, device=device)
		except KeyError as e:
			raise ValueError(f"Node ID {e.args[0]} not found in id_mapping.")
		center_scores = batch_scores[center_idx_tensor, :]




		# get neighbor node id list for each batch node and relation
		r1_list = [list(to_neigh) for to_neigh in to_neighs[0]]
		r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
		r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]

		# assign label-aware scores to neighbor nodes for each batch node and relation
		r1_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r1_list]
		r2_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r2_list]
		r3_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r3_list]

		# count the number of neighbors kept for aggregation for each batch node and relation
		r1_sample_num_list = [math.ceil(len(neighs) * self.thresholds[0]) for neighs in r1_list]
		r2_sample_num_list = [math.ceil(len(neighs) * self.thresholds[1]) for neighs in r2_list]
		r3_sample_num_list = [math.ceil(len(neighs) * self.thresholds[2]) for neighs in r3_list]

		# intra-aggregation steps for each relation
		# Eq. (8) in the paper
		r1_feats, r1_scores = self.intra_agg1.forward(nodes, labels, r1_list, center_scores, r1_scores, pos_scores, r1_sample_num_list, train_flag)
		r2_feats, r2_scores = self.intra_agg2.forward(nodes, labels, r2_list, center_scores, r2_scores, pos_scores, r2_sample_num_list, train_flag)
		r3_feats, r3_scores = self.intra_agg3.forward(nodes, labels, r3_list, center_scores, r3_scores, pos_scores, r3_sample_num_list, train_flag)

		# get features or embeddings for batch nodes
		device = self.features.weight.device  # use the actual device of the embedding matrix
		if isinstance(nodes, torch.Tensor):
			index = nodes.clone().detach().to(torch.long).to(device)
		else:
			index = torch.tensor(nodes, dtype=torch.long, device=device)


		self_feats = self.features(index)

		# number of nodes in a batch
		n = len(nodes)

		# concat the intra-aggregated embeddings from each relation
		# Eq. (9) in the paper
		cat_feats = torch.cat((self_feats, r1_feats, r2_feats, r3_feats), dim=1)

		combined = F.relu(cat_feats.mm(self.weight).t())

		return combined, center_scores


class IntraAgg(nn.Module):

	def __init__(self, features, feat_dim, embed_dim, train_pos, rho, cuda=False):
		"""
		Initialize the intra-relation aggregator
		:param features: the input node features or embeddings for all nodes
		:param feat_dim: the input dimension
		:param embed_dim: the embed dimension
		:param train_pos: positive samples in training set
		:param rho: the ratio of the oversample neighbors for the minority class
		:param cuda: whether to use GPU
		"""
		super(IntraAgg, self).__init__()

		self.features = features
		self.cuda = cuda
		self.feat_dim = feat_dim
		self.embed_dim = embed_dim
		self.train_pos = train_pos
		self.rho = rho
		self.weight = nn.Parameter(torch.FloatTensor(2*self.feat_dim, self.embed_dim))
		init.xavier_uniform_(self.weight)

	def forward(self, nodes, batch_labels, to_neighs_list, batch_scores, neigh_scores, pos_scores, sample_list, train_flag):
		"""
		Code partially from https://github.com/williamleif/graphsage-simple/
		:param nodes: list of nodes in a batch
		:param to_neighs_list: neighbor node id list for each batch node in one relation
		:param batch_scores: the label-aware scores of batch nodes
		:param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
		:param pos_scores: the label-aware scores 1-hop neighbors for the minority positive nodes
		:param train_flag: indicates whether in training or testing mode
		:param sample_list: the number of neighbors kept for each batch node in one relation
		:return to_feats: the aggregated embeddings of batch nodes neighbors in one relation
		:return samp_scores: the average neighbor distances for each relation after filtering
		"""

		# filer neighbors under given relation in the train mode
		if train_flag:
			samp_neighs, samp_scores = choose_step_neighs(batch_scores, batch_labels, neigh_scores, to_neighs_list, pos_scores, self.train_pos, sample_list, self.rho)
		else:
			samp_neighs, samp_scores = choose_step_test(batch_scores, neigh_scores, to_neighs_list, sample_list)
		
		# find the unique nodes among batch nodes and the filtered neighbors
		unique_nodes_list = list(set.union(*samp_neighs))
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

		# intra-relation aggregation only with sampled neighbors
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1
		if self.cuda:
			mask = mask.cuda()
		num_neigh = mask.sum(1, keepdim=True)
		mask = mask.div(num_neigh)  # mean aggregator
		if self.cuda:
			self_feats = self.features(torch.LongTensor(nodes).cuda())
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
		else:
			self_feats = self.features(torch.LongTensor(nodes))
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
		agg_feats = mask.mm(embed_matrix)  # single relation aggregator
		cat_feats = torch.cat((self_feats, agg_feats), dim=1)  # concat with last layer
		to_feats = F.relu(cat_feats.mm(self.weight))
		return to_feats, samp_scores


def choose_step_neighs(center_scores, center_labels, neigh_scores, neighs_list, minor_scores, minor_list, sample_list, sample_rate):
	"""
	Choose step for neighborhood sampling
	:param center_scores: the label-aware scores of batch nodes
	:param center_labels: the label of batch nodes
	:param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
	:param neighs_list: neighbor node id list for each batch node in one relation
	:param minor_scores: the label-aware scores for nodes of minority class in one relation
	:param minor_list: minority node id list for each batch node in one relation
	:param sample_list: the number of neighbors kept for each batch node in one relation
	:para sample_rate: the ratio of the oversample neighbors for the minority class
	"""
	samp_neighs = []
	samp_score_diff = []
	for idx, center_score in enumerate(center_scores):
		center_score = center_scores[idx][0]
		neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
		center_score_neigh = center_score.repeat(neigh_score.size()[0], 1)
		neighs_indices = neighs_list[idx]
		num_sample = sample_list[idx]

		# compute the L1-distance of batch nodes and their neighbors
		score_diff_neigh = torch.abs(center_score_neigh - neigh_score).squeeze()
		sorted_score_diff_neigh, sorted_neigh_indices = torch.sort(score_diff_neigh, dim=0, descending=False)
		selected_neigh_indices = sorted_neigh_indices.tolist()

		# top-p sampling according to distance ranking
		if len(neigh_scores[idx]) > num_sample + 1:
			selected_neighs = [neighs_indices[n] for n in selected_neigh_indices[:num_sample]]
			selected_score_diff = sorted_score_diff_neigh.tolist()[:num_sample]
		else:
			selected_neighs = neighs_indices
			selected_score_diff = score_diff_neigh.tolist()
			if isinstance(selected_score_diff, float):
				selected_score_diff = [selected_score_diff]

		if center_labels[idx] == 1:
			num_oversample = int(num_sample * sample_rate)
			center_score_minor = center_score.repeat(minor_scores.size()[0], 1)
			score_diff_minor = torch.abs(center_score_minor - minor_scores[:, 0].view(-1, 1)).squeeze()
			sorted_score_diff_minor, sorted_minor_indices = torch.sort(score_diff_minor, dim=0, descending=False)
			selected_minor_indices = sorted_minor_indices.tolist()
			selected_neighs.extend([minor_list[n] for n in selected_minor_indices[:num_oversample]])
			selected_score_diff.extend(sorted_score_diff_minor.tolist()[:num_oversample])

		samp_neighs.append(set(selected_neighs))
		samp_score_diff.append(selected_score_diff)

	return samp_neighs, samp_score_diff


def choose_step_test(center_scores, neigh_scores, neighs_list, sample_list):
	"""
	Filter neighbors according label predictor result with adaptive thresholds
	:param center_scores: the label-aware scores of batch nodes
	:param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
	:param neighs_list: neighbor node id list for each batch node in one relation
	:param sample_list: the number of neighbors kept for each batch node in one relation
	:return samp_neighs: the neighbor indices and neighbor simi scores
	:return samp_scores: the average neighbor distances for each relation after filtering
	"""

	samp_neighs = []
	samp_scores = []
	for idx, center_score in enumerate(center_scores):
		center_score = center_scores[idx][0]
		neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
		center_score = center_score.repeat(neigh_score.size()[0], 1)
		neighs_indices = neighs_list[idx]
		num_sample = sample_list[idx]

		# compute the L1-distance of batch nodes and their neighbors
		score_diff = torch.abs(center_score - neigh_score).squeeze()
		sorted_scores, sorted_indices = torch.sort(score_diff, dim=0, descending=False)
		selected_indices = sorted_indices.tolist()

		# top-p sampling according to distance ranking and thresholds
		if len(neigh_scores[idx]) > num_sample + 1:
			selected_neighs = [neighs_indices[n] for n in selected_indices[:num_sample]]
			selected_scores = sorted_scores.tolist()[:num_sample]
		else:
			selected_neighs = neighs_indices
			selected_scores = score_diff.tolist()
			if isinstance(selected_scores, float):
				selected_scores = [selected_scores]

		samp_neighs.append(set(selected_neighs))
		samp_scores.append(selected_scores)

	return samp_neighs, samp_scores


# ===== File: graphsage.py =====
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import random


"""
	GraphSAGE implementations
	Paper: Inductive Representation Learning on Large Graphs
	Source: https://github.com/williamleif/graphsage-simple/
"""


class GraphSage(nn.Module):
	"""
	Vanilla GraphSAGE Model
	Code partially from https://github.com/williamleif/graphsage-simple/
	"""
	def __init__(self, num_classes, enc):
		super(GraphSage, self).__init__()
		self.enc = enc
		self.xent = nn.CrossEntropyLoss()
		self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
		init.xavier_uniform_(self.weight)

	def forward(self, nodes):
		embeds = self.enc(nodes)
		scores = self.weight.mm(embeds)
		return scores.t()

	def to_prob(self, nodes):
		pos_scores = torch.sigmoid(self.forward(nodes))
		return pos_scores

	def loss(self, nodes, labels):
		scores = self.forward(nodes)
		return self.xent(scores, labels.squeeze())


class MeanAggregator(nn.Module):
	"""
	Aggregates a node's embeddings using mean of neighbors' embeddings
	"""

	def __init__(self, features, cuda=False, gcn=False):
		"""
		Initializes the aggregator for a specific graph.

		features -- function mapping LongTensor of node ids to FloatTensor of feature values.
		cuda -- whether to use GPU
		gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
		"""

		super(MeanAggregator, self).__init__()

		self.features = features
		self.cuda = cuda
		self.gcn = gcn

	def forward(self, nodes, to_neighs, num_sample=10):
		"""
		nodes --- list of nodes in a batch
		to_neighs --- list of sets, each set is the set of neighbors for node in batch
		num_sample --- number of neighbors to sample. No sampling if None.
		"""
		# Local pointers to functions (speed hack)
		_set = set
		if not num_sample is None:
			_sample = random.sample
			samp_neighs = [_set(_sample(to_neigh,
										num_sample,
										)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
		else:
			samp_neighs = to_neighs

		if self.gcn:
			samp_neighs = [samp_neigh.union(set([int(nodes[i])])) for i, samp_neigh in enumerate(samp_neighs)]
		unique_nodes_list = list(set.union(*samp_neighs))
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1
		if self.cuda:
			mask = mask.cuda()
		num_neigh = mask.sum(1, keepdim=True)
		mask = mask.div(num_neigh)
		if self.cuda:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
		else:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
		to_feats = mask.mm(embed_matrix)
		return to_feats


class Encoder(nn.Module):
	"""
	Vanilla GraphSAGE Encoder Module
	Encodes a node's using 'convolutional' GraphSage approach
	"""

	def __init__(self, features, feature_dim,
				 embed_dim, adj_lists, aggregator,
				 num_sample=10,
				 base_model=None, gcn=False, cuda=False,
				 feature_transform=False):
		super(Encoder, self).__init__()

		self.features = features
		self.feat_dim = feature_dim
		self.adj_lists = adj_lists
		self.aggregator = aggregator
		self.num_sample = num_sample
		if base_model != None:
			self.base_model = base_model

		self.gcn = gcn
		self.embed_dim = embed_dim
		self.cuda = cuda
		self.aggregator.cuda = cuda
		self.weight = nn.Parameter(
			torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
		init.xavier_uniform_(self.weight)

	def forward(self, nodes):
		"""
		Generates embeddings for a batch of nodes.

		nodes     -- list of nodes
		"""
		neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
											  self.num_sample)

		if isinstance(nodes, list):
			index = torch.LongTensor(nodes)
		else:
			index = nodes

		if not self.gcn:
			if self.cuda:
				self_feats = self.features(index).cuda()
			else:
				self_feats = self.features(index)
			combined = torch.cat((self_feats, neigh_feats), dim=1)
		else:
			combined = neigh_feats
		combined = F.relu(self.weight.mm(combined.t()))
		return combined



class GCN(nn.Module):
	"""
	Vanilla GCN Model
	Code partially from https://github.com/williamleif/graphsage-simple/
	"""
	def __init__(self, num_classes, enc):
		super(GCN, self).__init__()
		self.enc = enc
		self.xent = nn.CrossEntropyLoss()
		self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
		init.xavier_uniform_(self.weight)


	def forward(self, nodes):
		embeds = self.enc(nodes)
		scores = self.weight.mm(embeds)
		return scores.t()

	def to_prob(self, nodes):
		pos_scores = torch.sigmoid(self.forward(nodes))
		return pos_scores

	def loss(self, nodes, labels):
		scores = self.forward(nodes)
		return self.xent(scores, labels.squeeze())


class GCNAggregator(nn.Module):
	"""
	Aggregates a node's embeddings using normalized mean of neighbors' embeddings
	"""

	def __init__(self, features, cuda=False, gcn=False):
		"""
		Initializes the aggregator for a specific graph.

		features -- function mapping LongTensor of node ids to FloatTensor of feature values.
		cuda -- whether to use GPU
		gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
		"""

		super(GCNAggregator, self).__init__()

		self.features = features
		self.cuda = cuda
		self.gcn = gcn

	def forward(self, nodes, to_neighs):
		"""
		nodes --- list of nodes in a batch
		to_neighs --- list of sets, each set is the set of neighbors for node in batch
		"""
		# Local pointers to functions (speed hack)
		
		samp_neighs = to_neighs

		#  Add self to neighs
		samp_neighs = [samp_neigh.union(set([int(nodes[i])])) for i, samp_neigh in enumerate(samp_neighs)]
		unique_nodes_list = list(set.union(*samp_neighs))
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1.0  # Adjacency matrix for the sub-graph
		if self.cuda:
			mask = mask.cuda()
		row_normalized = mask.sum(1, keepdim=True).sqrt()
		col_normalized = mask.sum(0, keepdim=True).sqrt()
		mask = mask.div(row_normalized).div(col_normalized)
		if self.cuda:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
		else:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
		to_feats = mask.mm(embed_matrix)
		return to_feats

class GCNEncoder(nn.Module):
	"""
	GCN Encoder Module
	"""

	def __init__(self, features, feature_dim,
				 embed_dim, adj_lists, aggregator,
				 num_sample=10,
				 base_model=None, gcn=False, cuda=False,
				 feature_transform=False):
		super(GCNEncoder, self).__init__()

		self.features = features
		self.feat_dim = feature_dim
		self.adj_lists = adj_lists
		self.aggregator = aggregator
		self.num_sample = num_sample
		if base_model != None:
			self.base_model = base_model

		self.gcn = gcn
		self.embed_dim = embed_dim
		self.cuda = cuda
		self.aggregator.cuda = cuda
		self.weight = nn.Parameter(
			torch.FloatTensor(embed_dim, self.feat_dim ))
		init.xavier_uniform_(self.weight)

	def forward(self, nodes):
		"""
		Generates embeddings for a batch of nodes.
		Input:
			nodes -- list of nodes
		Output:
			embed_dim*len(nodes)
		"""
		neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes])

		if isinstance(nodes, list):
			index = torch.LongTensor(nodes)
		else:
			index = nodes

		combined = F.relu(self.weight.mm(neigh_feats.t()))
		return combined


# ===== File: utils.py =====
import pickle
import random
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import copy as cp
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix
from collections import defaultdict


"""
	Utility functions to handle data and evaluate model.
"""


def load_data(data, prefix='data/'):
	"""
	Load graph, feature, and label given dataset name
	:returns: home and single-relation graphs, feature, label
	"""

	if data == 'yelp':
		data_file = loadmat(prefix + 'YelpChi.mat')
		labels = data_file['label'].flatten()
		feat_data = data_file['features'].todense().A
		# load the preprocessed adj_lists
		with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
			homo = pickle.load(file)
		file.close()
		with open(prefix + 'yelp_rur_adjlists.pickle', 'rb') as file:
			relation1 = pickle.load(file)
		file.close()
		with open(prefix + 'yelp_rtr_adjlists.pickle', 'rb') as file:
			relation2 = pickle.load(file)
		file.close()
		with open(prefix + 'yelp_rsr_adjlists.pickle', 'rb') as file:
			relation3 = pickle.load(file)
		file.close()
	elif data == 'amazon':
		data_file = loadmat(prefix + 'Amazon.mat')
		labels = data_file['label'].flatten()
		feat_data = data_file['features'].todense().A
		# load the preprocessed adj_lists
		with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
			homo = pickle.load(file)
		file.close()
		with open(prefix + 'amz_upu_adjlists.pickle', 'rb') as file:
			relation1 = pickle.load(file)
		file.close()
		with open(prefix + 'amz_usu_adjlists.pickle', 'rb') as file:
			relation2 = pickle.load(file)
		file.close()
		with open(prefix + 'amz_uvu_adjlists.pickle', 'rb') as file:
			relation3 = pickle.load(file)

	return [homo, relation1, relation2, relation3], feat_data, labels


def normalize(mx):
	"""
		Row-normalize sparse matrix
		Code from https://github.com/williamleif/graphsage-simple/
	"""
	rowsum = np.array(mx.sum(1)) + 0.01
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx


def sparse_to_adjlist(sp_matrix, filename):
	"""
	Transfer sparse matrix to adjacency list
	:param sp_matrix: the sparse matrix
	:param filename: the filename of adjlist
	"""
	# add self loop
	homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
	# create adj_list
	adj_lists = defaultdict(set)
	edges = homo_adj.nonzero()
	for index, node in enumerate(edges[0]):
		adj_lists[node].add(edges[1][index])
		adj_lists[edges[1][index]].add(node)
	with open(filename, 'wb') as file:
		pickle.dump(adj_lists, file)
	file.close()


def pos_neg_split(nodes, labels):
	"""
	Find positive and negative nodes given a list of nodes and their labels
	:param nodes: a list or tensor of nodes
	:param labels: a list or tensor of node labels
	:returns: the split positive and negative nodes (as lists)
	"""
	if isinstance(nodes, torch.Tensor):
		nodes = nodes.cpu().numpy()
	if isinstance(labels, torch.Tensor):
		labels = labels.cpu().numpy()

	pos_nodes = []
	neg_nodes = []

	for node, label in zip(nodes, labels):
		if label == 1:
			pos_nodes.append(node)
		else:
			neg_nodes.append(node)

	return pos_nodes, neg_nodes


def pick_step(idx_train, y_train, adj_list, size):
	epsilon = 1e-6
	lf_train = (y_train.sum() - len(y_train)) * y_train + len(y_train) + epsilon
	degree_train = [max(len(adj_list[node]), 1) for node in idx_train]
	smp_prob = np.array(degree_train) / lf_train


    # Handle degenerate case where weights are zero or invalid
	if np.sum(smp_prob) <= 0 or np.any(np.isnan(smp_prob)) or np.any(np.isinf(smp_prob)):
		print("Warning: smp_prob invalid, defaulting to uniform sampling.")
		smp_prob = None
	else:
		smp_prob = smp_prob / np.sum(smp_prob) 

	return random.choices(idx_train, weights=smp_prob, k=size)



def test_sage(test_cases, labels, model, batch_size, thres=0.5):
	"""
	Test the performance of GraphSAGE
	:param test_cases: list of node indices
	:param labels: ground truth labels
	:param model: GraphSAGE model
	:param batch_size: batch size for inference
	"""

	num_batches = int(len(test_cases) / batch_size) + 1
	gnn_pred_list = []
	gnn_prob_list = []

	for i in range(num_batches):
		i_start = i * batch_size
		i_end = min((i + 1) * batch_size, len(test_cases))
		batch_nodes = test_cases[i_start:i_end]
		batch_label = labels[i_start:i_end]

		gnn_prob = model.to_prob(batch_nodes)
		gnn_prob_arr = gnn_prob.detach().cpu().numpy()[:, 1]
		gnn_pred = prob2pred(gnn_prob_arr, thres)

		gnn_pred_list.extend(gnn_pred)
		gnn_prob_list.extend(gnn_prob_arr)

	# Convert to numpy for metrics
	y_true = np.array(labels)
	y_pred = np.array(gnn_pred_list)

	auc = roc_auc_score(y_true, gnn_prob_list)
	f1_1 = f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)
	f1_0 = f1_score(y_true, y_pred, pos_label=0, average='binary', zero_division=0)
	f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
	f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
	conf = confusion_matrix(y_true, y_pred)
	tn, fp, fn, tp = conf.ravel()
	gmean = conf_gmean(conf)

	print(f"   GNN F1-binary-1: {f1_1:.4f}\tF1-binary-0: {f1_0:.4f}"
	      f"\tF1-macro: {f1_macro:.4f}\tG-Mean: {gmean:.4f}\tAUC: {auc:.4f}")
	print(f"   GNN TP: {tp}\tTN: {tn}\tFN: {fn}\tFP: {fp}")

	return f1_macro, f1_1, f1_0, auc, gmean

	

def prob2pred(y_prob, thres=0.5):
	"""
	Convert probability to predicted results according to given threshold
	:param y_prob: numpy array of probability in [0, 1]
	:param thres: binary classification threshold, default 0.5
	:returns: the predicted result with the same shape as y_prob
	"""
	y_pred = np.zeros_like(y_prob, dtype=np.int32)
	y_pred[y_prob >= thres] = 1
	y_pred[y_prob < thres] = 0
	return y_pred


def test_pcgnn(test_cases, labels, model, batch_size, thres=0.5):
	"""
	Test the performance of PC-GNN and its variants
	:param test_cases: list of test node indices
	:param labels: ground-truth labels
	:param model: trained PC-GNN model
	:param batch_size: mini-batch size for testing
	:return: macro-F1, F1 for class 1 and 0, AUC, G-Mean
	"""

	num_batches = int(len(test_cases) / batch_size) + 1

	gnn_pred_list = []
	gnn_prob_list = []
	label1_prob_list = []
	label1_true_list = []
	label1_pred_list = []

	for i in range(num_batches):
		i_start = i * batch_size
		i_end = min((i + 1) * batch_size, len(test_cases))
		batch_nodes = test_cases[i_start:i_end]
		batch_label = labels[i_start:i_end]

		gnn_prob, label_prob1 = model.to_prob(batch_nodes, batch_label, train_flag=False)

		# Collect predictions
		gnn_probs = gnn_prob.data.cpu().numpy()[:, 1]
		gnn_preds = prob2pred(gnn_probs, thres)

		label_logits = label_prob1.data.cpu().numpy()
		label_preds = label_logits.argmax(axis=1)
		label_probs = label_logits[:, 1]

		# Accumulate results
		gnn_pred_list.extend(gnn_preds)
		gnn_prob_list.extend(gnn_probs)
		label1_prob_list.extend(label_probs)
		label1_true_list.extend(batch_label)
		label1_pred_list.extend(label_preds)

	# Convert to arrays
	y_true = np.array(label1_true_list)
	y_pred = np.array(gnn_pred_list)

	# GNN evaluation
	auc_gnn = roc_auc_score(y_true, gnn_prob_list)
	ap_gnn = average_precision_score(y_true, gnn_prob_list)
	f1_binary_1 = f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)
	f1_binary_0 = f1_score(y_true, y_pred, pos_label=0, average='binary', zero_division=0)
	f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
	f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
	conf_gnn = confusion_matrix(y_true, y_pred)
	tn, fp, fn, tp = conf_gnn.ravel()
	gmean_gnn = conf_gmean(conf_gnn)
	overall_acc = accuracy_score(y_true, y_pred)

	# Label-aware classifier evaluation
	label1_f1 = f1_score(y_true, label1_pred_list, average="macro", zero_division=0)
	label1_acc = accuracy_score(y_true, label1_pred_list)
	label1_recall = recall_score(y_true, label1_pred_list, average="macro", zero_division=0)
	auc_label1 = roc_auc_score(y_true, label1_prob_list)
	ap_label1 = average_precision_score(y_true, label1_prob_list)

	# Logging
	print(f"   GNN F1-binary-1: {f1_binary_1:.4f}\tF1-binary-0: {f1_binary_0:.4f}"
		  f"\tF1-macro: {f1_macro:.4f}\tG-Mean: {gmean_gnn:.4f}\tAUC: {auc_gnn:.4f}")
	print(f"   GNN TP: {tp}\tTN: {tn}\tFN: {fn}\tFP: {fp}")
	print(f"Label1 F1: {label1_f1:.4f}\tAccuracy: {label1_acc:.4f}"
		  f"\tRecall: {label1_recall:.4f}\tAUC: {auc_label1:.4f}\tAP: {ap_label1:.4f}")

	return f1_macro, f1_binary_1, f1_binary_0, auc_gnn, gmean_gnn,overall_acc

def conf_gmean(conf):
	tn, fp, fn, tp = conf.ravel()
	return (tp*tn/((tp+fn)*(tn+fp)))**0.5

# ===== File: model_handler.py =====
import time, datetime
import os
import random
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.dgraphfin import DGraphFin  # if you still want to use that loader
from utils.utils import prepare_folder, prepare_tune_folder, save_preds_and_params
from utils.evaluator import Evaluator



class ModelHandler(object):
    def __init__(self, config, feat_data_tensor, labels_tensor, edge_index_tensor, split_idx_tensors):
        # Store config and set up device and seed
        self.config = config 
        current_seed = self.config.get('seed', 42)
        np.random.seed(current_seed)
        random.seed(current_seed)
        torch.manual_seed(current_seed)

        gpu_idx = self.config.get('gpu', -1)
        self.use_cuda = (int(gpu_idx) >= 0) and torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.manual_seed_all(current_seed) # Use manual_seed_all for all GPUs
            self.device = torch.device(f'cuda:{gpu_idx}')
        else:
            self.device = torch.device('cpu')
        
        print(f"PCGNN ModelHandler initialized on device: {self.device}")

        # Process features (normalize expects NumPy)
        if isinstance(feat_data_tensor, torch.Tensor):
            feat_data_np = feat_data_tensor.cpu().numpy()
        else: # Should not happen if called from main_elliptic.py
            feat_data_np = feat_data_tensor 
        processed_feat_data_np = normalize(feat_data_np) # normalize is from your utils
        self.feat_data_for_embedding = torch.FloatTensor(processed_feat_data_np).to(self.device)

        # Labels (keep on device for now, convert to CPU when needed for sklearn/numpy utils)
        self.labels_on_device = labels_tensor.to(self.device)

        # Adjacency lists (PCGNN's internal format)
        # edge_index_to_adjlist_internal expects CPU tensor
        adj_list = self._edge_index_to_adjlist_internal(edge_index_tensor.cpu())
        adj_lists_repeated = [adj_list for _ in range(3)] # PCGNN expects 3 relations

        # Split indices (keep on device, convert to CPU list/numpy when passed to utils)
        self.idx_train = split_idx_tensors['train'].to(self.device)
        self.idx_valid = split_idx_tensors['valid'].to(self.device)
        self.idx_test = split_idx_tensors['test'].to(self.device)

        # Get corresponding labels (still on device)
        self.y_train_device = self.labels_on_device[self.idx_train]
        self.y_valid_device = self.labels_on_device[self.idx_valid]
        self.y_test_device = self.labels_on_device[self.idx_test]
        
        # For pick_step and pos_neg_split, which use NumPy/lists
        idx_train_cpu_list = self.idx_train.cpu().tolist()
        y_train_cpu_numpy = self.y_train_device.cpu().numpy()
        train_pos, train_neg = pos_neg_split(idx_train_cpu_list, y_train_cpu_numpy)

        self.dataset = {
            'feat_data_for_embedding': self.feat_data_for_embedding,
            'labels_on_device': self.labels_on_device, # Full labels on device
            'adj_lists': adj_lists_repeated,
            'homo': adj_list, # Used by pick_step, expects dict of sets of python ints
            'idx_train': self.idx_train,   # Tensor on device
            'idx_valid': self.idx_valid,   # Tensor on device
            'idx_test': self.idx_test,     # Tensor on device
            'y_train_device': self.y_train_device, # Tensor on device
            'y_valid_device': self.y_valid_device, # Tensor on device
            'y_test_device': self.y_test_device,   # Tensor on device
            'train_pos': train_pos,        # List of ints (CPU)
            # 'train_neg': train_neg       # Not explicitly used in train loop snippet
        }

    def _edge_index_to_adjlist_internal(self, edge_index_cpu_tensor):
        adj = defaultdict(set)
        edge_list = edge_index_cpu_tensor.t().tolist() 
        for src, dst in edge_list:
            adj[src].add(dst)
            adj[dst].add(src) 
        return adj

    def train(self):
        # Create an args namespace from self.config for compatibility with PCGNN's internal code
        args = argparse.Namespace(**self.config)
        # Add defaults for any other parameters PCGNN's train method might expect if not in self.config
        args.emb_size = self.config.get('hidden_channels', 128)
        args.num_epochs = self.config.get('epochs_gnn', 100)
        args.batch_size = self.config.get('batch_size_gnn', 64)
        args.weight_decay = self.config.get('l2', 5e-7)
        args.rho = self.config.get('rho', 0.1) # Example: ensure rho is present, PCGNN uses it
        args.alpha = self.config.get('lambda_1', 0.1) # For PCALayer
        args.multi_relation = self.config.get('multi_relation', 'GNN') # Default or from config
        args.thres = self.config.get('thres', 0.5)
        args.valid_epochs = self.config.get('valid_epochs', 10) # How often to validate
        args.save_dir = self.config.get('save_dir', './pcgnn_saved_models/')
        args.data_name = self.config.get('data_name', 'elliptic') # For model saving path
        args.model = 'pcgnn' # This handler is for PCGNN
        args.cuda = self.use_cuda # Set based on self.device

        # --- Model Initialization (copied & adapted from your pcgnn.py ModelHandler.train) ---
        features_module = nn.Embedding(self.dataset['feat_data_for_embedding'].shape[0], self.dataset['feat_data_for_embedding'].shape[1])
        features_module.weight = nn.Parameter(self.dataset['feat_data_for_embedding'], requires_grad=False)
        # features_module is already on self.device because feat_data_for_embedding is

        intra1 = IntraAgg(features_module, self.dataset['feat_data_for_embedding'].shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
        intra2 = IntraAgg(features_module, self.dataset['feat_data_for_embedding'].shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
        intra3 = IntraAgg(features_module, self.dataset['feat_data_for_embedding'].shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
        
        inter1 = InterAgg(features_module, self.dataset['feat_data_for_embedding'].shape[1], args.emb_size, self.dataset['train_pos'], 
                          self.dataset['adj_lists'], [intra1, intra2, intra3], inter=args.multi_relation, cuda=args.cuda)
        
        gnn_model = PCALayer(2, inter1, args.alpha) # num_classes=2
        gnn_model.to(self.device)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), 
                                     lr=args.lr, weight_decay=args.weight_decay)
        
        timestamp = time.time()
        timestamp_str = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')
        dir_saver = os.path.join(args.save_dir, timestamp_str)
        path_saver = os.path.join(dir_saver, f'{args.data_name}_{args.model}.pkl')
        auc_best, ep_best = 0, -1
        # f1_mac_best will also be tracked if needed for return

        # For pick_step, y_train needs to be a NumPy array
        y_train_for_pick_step_np = self.dataset['y_train_device'].cpu().numpy()
        idx_train_for_pick_step_np = self.dataset['idx_train'].cpu().numpy()


        for epoch in tqdm(range(args.num_epochs), desc="PCGNN Training Epochs"):
            gnn_model.train() # Ensure model is in training mode
            # pick_step expects NumPy array for idx_train and y_train
            sampled_idx_train_np = pick_step(idx_train_for_pick_step_np, y_train_for_pick_step_np, 
                                           self.dataset['homo'], 
                                           size=len(self.dataset['train_pos']) * 2 if self.dataset['train_pos'] else args.batch_size)
            
            if not sampled_idx_train_np:
                print(f"Epoch {epoch}: sampled_idx_train is empty, skipping batch training.")
                continue

            random.shuffle(sampled_idx_train_np)
            num_batches = int(len(sampled_idx_train_np) / args.batch_size) + 1
            epoch_loss = 0.0
            epoch_time = 0

            for batch_idx in range(num_batches):
                start_time_batch = time.time()
                i_start = batch_idx * args.batch_size
                i_end = min((batch_idx + 1) * args.batch_size, len(sampled_idx_train_np))
                if i_start >= i_end: continue

                batch_nodes_np_list = sampled_idx_train_np[i_start:i_end] # Model expects list of node IDs
                
                # Get corresponding labels for this batch and move to device
                # Use original full labels tensor and index with the numpy batch, then move to device
                batch_label_tensor = self.dataset['labels_on_device'][torch.tensor(batch_nodes_np_list, dtype=torch.long)].to(self.device)

                optimizer.zero_grad()
                loss = gnn_model.loss(batch_nodes_np_list, batch_label_tensor) # Pass list of nodes and tensor of labels
                loss.backward()
                optimizer.step()
                
                epoch_time += time.time() - start_time_batch
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            # Corrected print statement from your pcgnn.py
            print(f'Epoch: {epoch}, loss: {avg_epoch_loss:.4f}, time: {epoch_time:.2f}s')


            if epoch % args.valid_epochs == 0 or epoch == args.num_epochs -1 :
                print(f"Validating PCGNN at epoch {epoch}")
                gnn_model.eval() # Set model to evaluation mode
                # **CRITICAL FIX for "can't convert cuda:0..." error**: Ensure labels passed to test_pcgnn are CPU NumPy arrays
                idx_valid_cpu_list = self.dataset['idx_valid'].cpu().tolist()
                y_valid_cpu_numpy = self.dataset['y_valid_device'].cpu().numpy()

                f1_mac_val, _, _, auc_val, _, _ = test_pcgnn(
                idx_valid_cpu_list, 
                y_valid_cpu_numpy, 
                gnn_model, 
                args.batch_size, 
                args.thres
                )
                if auc_val > auc_best:
                    auc_best = auc_val # ep_best and f1_mac_best would also be updated
                    ep_best = epoch
                    if not os.path.exists(dir_saver): os.makedirs(dir_saver)
                    print('  Saving PCGNN model ...')
                    torch.save(gnn_model.state_dict(), path_saver)
        
        if ep_best != -1: # Check if a model was saved
             print(f"Restore PCGNN model from epoch {ep_best}")
             print(f"Model path: {path_saver}")
             gnn_model.load_state_dict(torch.load(path_saver, map_location=self.device)) # Ensure map_location
        else:
            print("Warning: No best PCGNN model found during validation. Using model from last epoch for testing.")
            if not os.path.exists(dir_saver): os.makedirs(dir_saver) # Still save last model
            torch.save(gnn_model.state_dict(), path_saver)


        print("Testing PCGNN model...")
        gnn_model.eval() # Ensure model is in eval mode for final testing
        idx_test_cpu_list = self.dataset['idx_test'].cpu().tolist()
        y_test_cpu_numpy = self.dataset['y_test_device'].cpu().numpy()

        f1_mac_test, f1_bin1_test, f1_bin0_test, auc_test, _, acc_test = test_pcgnn(idx_test_cpu_list,
			y_test_cpu_numpy,
			gnn_model,
			args.batch_size,
			args.thres
		)
        return ep_best, auc_test, f1_mac_test, f1_bin1_test, f1_bin0_test, acc_test
