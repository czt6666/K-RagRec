import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, GATConv,  SAGEConv, RGCNConv, MessagePassing


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x, edge_attr


class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
        super(GraphTransformer, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(TransformerConv(in_channels=in_channels, out_channels=hidden_channels//num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=hidden_channels//num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout,))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=out_channels//num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout,))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=adj_t, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index=adj_t, edge_attr=edge_attr)
        return x, edge_attr

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=num_heads, concat=False))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x,edge_index=edge_index, edge_attr=edge_attr)
        return x, edge_attr


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x, edge_attr

class RGCN(torch.nn.Module):
    """Relational GCN. Expects an integer ``edge_type`` tensor alongside
    ``edge_index``; ``edge_attr`` is ignored. ``num_relations`` upper-bounds
    the relation vocabulary; we clamp ids to be safe."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1, num_relations=200):
        super().__init__()
        self.num_relations = num_relations
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr, edge_type=None):
        if edge_type is None:
            raise RuntimeError("RGCN requires edge_type; pass it through encode_graphs.")
        edge_type = edge_type.clamp(max=self.num_relations - 1)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_type)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_type)
        return x, edge_attr


class CompGCNLayer(MessagePassing):
    """One CompGCN layer using the multiplicative composition op
    phi(h, r) = h * r (Vashishth et al. 2020). Edges carry a relation
    embedding; we update both node and (per-edge) relation features."""

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="mean")
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_in = nn.Linear(in_channels, out_channels)
        self.lin_rel = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = out + self.lin_self(x)
        new_edge_attr = self.lin_rel(edge_attr)
        return out, new_edge_attr

    def message(self, x_j, edge_attr):
        return self.lin_in(x_j * edge_attr)


class CompGCN(torch.nn.Module):
    """Stack of CompGCN layers; relation embeddings co-evolve with nodes."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.layers.append(CompGCNLayer(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(CompGCNLayer(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.layers.append(CompGCNLayer(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.layers:
            for m in layer.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr, edge_type=None):
        for i, layer in enumerate(self.layers[:-1]):
            x, edge_attr = layer(x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x, edge_attr = self.layers[-1](x, edge_index, edge_attr)
        return x, edge_attr


load_gnn_model = {
    'gcn': GCN,
    'gat': GAT,
    'gt': GraphTransformer,
    'graphsage': GraphSAGE,
    'rgcn': RGCN,
    'compgcn': CompGCN,
}