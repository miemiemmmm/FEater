import torch
from torch import nn
import torch.nn.functional as F 

import dgl
from dgl.nn import GraphConv
from dgl.nn.pytorch import NNConv, Set2Set

class Simple_GCN(nn.Module):
  def __init__(self, in_feats, h_feats, num_classes):
    super(Simple_GCN, self).__init__()
    self.conv1 = GraphConv(in_feats, h_feats)
    self.conv2 = GraphConv(h_feats, num_classes)

  def forward(self, g):
    node_feat = g.ndata["pos"]
    h = self.conv1(g, node_feat) 
    h = F.relu(h)
    h = self.conv2(g, h)
    # g.ndata["pos"] = h
    return dgl.max_nodes(g, "pos")

class LinearBn(nn.Module):
    def __init__(self, 
                 in_size, out_size, bias=True, do_batchnorm=True, activation=None):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size, bias=bias)
        if do_batchnorm:
            self.bn = nn.BatchNorm1d(out_size, eps=1e-5, momentum=0.1)
        else:
            self.bn = nn.Identity()
        self.activation = activation 

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x  

class VanillaMPNN(nn.Module):
    """"More or less integrating
    MPNNGNN and MPNNPredictor class from dgllifescience
    Also compare to: https://www.kaggle.com/c/champs-scalar-coupling/discussion/93972
    """
    def __init__(self, 
                 node_in_feats=32, 
                 edge_in_feats=16, 
                 node_hidden_feats=[64,128],
                 edge_hidden_feats=[32, 64, 128],
                 ntasks=1, 
                 num_step_message_passing=6,
                 do_batchnorm=True,
                 readout_type='node', # whether have graph- or node-based readout
                 bias=True):

        super(VanillaMPNN, self).__init__()

        self._num_step_message_passing = num_step_message_passing
        self._num_step_set2set = 6
        self._num_layer_set2set = 3
        self._readout_type = readout_type

        # extend list to include input/output layer sizes
        if isinstance(node_hidden_feats, int):
            node_hidden_feats = [node_in_feats, node_hidden_feats]
        else:
            node_hidden_feats = [node_in_feats] + node_hidden_feats
        if isinstance(edge_hidden_feats, int):
            edge_hidden_feats = [edge_in_feats, edge_hidden_feats, node_hidden_feats[-1]*node_hidden_feats[-1]]
        else:
            edge_hidden_feats = [edge_in_feats] + edge_hidden_feats + [node_hidden_feats[-1]*node_hidden_feats[-1]]

        n_node_layers = len(node_hidden_feats)-1
        self.preprocess = nn.ModuleList()
        for i in range(n_node_layers):
            self.preprocess.append(
                LinearBn(node_hidden_feats[i], node_hidden_feats[i+1], bias=bias, do_batchnorm=do_batchnorm, activation=None)
            )
            self.preprocess.append(nn.ReLU())
        self.preprocess = nn.Sequential(*self.preprocess) # create a sequential model from the list
        
        n_edge_layers = len(edge_hidden_feats)-1
        self.edge_function = nn.ModuleList()
        for i in range(n_edge_layers):
            self.edge_function.append(
                LinearBn(edge_hidden_feats[i], edge_hidden_feats[i+1], bias=bias, do_batchnorm=do_batchnorm, activation=None)
            )
            if i < n_edge_layers - 1:
                self.edge_function.append(nn.ReLU())
        self.edge_function = nn.Sequential(*self.edge_function) # create a sequential model from the list

        self.gnn_layer = NNConv(
            in_feats=node_hidden_feats[-1],
            out_feats=node_hidden_feats[-1],
            edge_func=self.edge_function,
            aggregator_type='mean',
            bias=True
        )

        self.gru = nn.GRU(node_hidden_feats[-1], node_hidden_feats[-1])

        if self._readout_type == 'graph':
            self.graph_readout = Set2Set(input_dim=node_hidden_feats[-1],
                                n_iters=self._num_step_set2set,
                                n_layers=self._num_layer_set2set
                                )
            self.predict = nn.Sequential(
                LinearBn(2 * node_hidden_feats[-1], node_hidden_feats[-1], 
                        bias=bias, do_batchnorm=do_batchnorm, activation=None), # 2 * -> see set2set definition
                nn.ReLU(),
                nn.Linear(node_hidden_feats[-1], ntasks)
            )
        elif self._readout_type == 'node':
            self.predict = nn.Sequential(
                LinearBn(node_hidden_feats[-1], node_hidden_feats[-1], 
                        bias=bias, do_batchnorm=do_batchnorm, activation=None),
                nn.ReLU(),
                nn.Linear(node_hidden_feats[-1], ntasks)
            )
        else:
            raise NotImplementedError(f"Readout type {self._readout_type} not implemented")

    def forward(self, graph, node_feats, edge_feats):

        node_feats = self.preprocess(node_feats)
        hidden_feats = node_feats.unsqueeze(0)

        for _ in range(self._num_step_message_passing):
            # Message:
            messages = F.relu(self.gnn_layer(graph, node_feats, edge_feats))
            node_feats, hidden_feats = self.gru(messages.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)
        
        if self._readout_type == 'graph':
            readout = self.graph_readout(graph, node_feats)
        elif self._readout_type == 'node':
            readout = node_feats 
         
        return self.predict(readout)

