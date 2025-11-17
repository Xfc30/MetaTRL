from math import ceil

import torch
from torch import nn
import torch.nn.functional as F

from libcity.model.layers.gat import MetaGATLayer
# import dhg
from libcity.model.trajectory_embedding.BERT import GATLayerImp3, PositionalEmbedding


class GAT2(nn.Module):

    def __init__(self, d_model, in_feature, num_heads_per_layer, num_features_per_layer,
                 add_skip_connection=True, bias=True, dropout=0.6, load_trans_prob=True, avg_last=True):
        super().__init__()
        self.d_model = d_model
        assert len(num_heads_per_layer) == len(num_features_per_layer), f'Enter valid arch params.'

        num_features_per_layer = [in_feature] + num_features_per_layer
        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below
        if avg_last:
            assert num_features_per_layer[-1] == d_model
        else:
            assert num_features_per_layer[-1] * num_heads_per_layer[-1] == d_model
        num_of_layers = len(num_heads_per_layer) - 1

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            if i == num_of_layers - 1:
                if avg_last:
                    concat_input = False
                else:
                    concat_input = True
            else:
                concat_input = True
            layer = GATLayerImp3(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=num_features_per_layer[i + 1],
                num_of_heads=num_heads_per_layer[i + 1],
                concat=concat_input,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                load_trans_prob=load_trans_prob
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    def forward(self, node_features, edge_index_input, edge_prob_input, x, return_first_layer=False):
        """

        Args:
            node_features: (vocab_size, fea_dim)
            edge_index_input: (2, E)
            edge_prob_input: (E, 1)
            x: (B, T)
            return_first_layer: bool : return node_fea_emb in first layer gat or not
        Returns:
            (B, T, d_model)

        """
        data = (node_features, edge_index_input, edge_prob_input)
        first_layer_output = None
        # (node_fea_emb, edge_index, edge_prob) = self.gat_net(data)  # (vocab_size, num_channels[-1]), (2, E)
        for i, layer in enumerate(self.gat_net):
            data = layer(data)
            if i == 0 and return_first_layer:
                first_layer_output = data[0]  # Capture the node_fea_emb of the first layer
        node_fea_emb = data[0]
        batch_size, seq_len = x.shape
        node_fea_emb = node_fea_emb.expand((batch_size, -1, -1))  # (B, vocab_size, d_model)
        node_fea_emb = node_fea_emb.reshape(-1, self.d_model)  # (B * vocab_size, d_model)
        x = x.reshape(-1, 1).squeeze(1)  # (B * T,)
        out_node_fea_emb = node_fea_emb[x].reshape(batch_size, seq_len, self.d_model)  # (B, T, d_model)
        if return_first_layer:
            return out_node_fea_emb, first_layer_output
        else:
            return out_node_fea_emb  # (B, T, d_model)


class HyperEdgeAgg(nn.Module):
    def __init__(self, fea_dim, d_model):
        super(HyperEdgeAgg, self).__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(fea_dim, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.linear3 = nn.Linear(d_model, d_model)
        self.activation1 = nn.ELU()
        self.activation2 = nn.ELU()

    def forward(self, node_features, hyperedges):
        """
        Args:
            node_features: (num_nodes, fea_dim)
            hyperedges: (batch_size, seq_len)
        Returns:
            updated_node_features: (num_nodes, d_model)
        """
        batch_size, seq_len = hyperedges.shape

        # Extract hyperedge embeddings
        node_features_ = self.linear1(node_features)
        # node_features_ = node_features
        hyperedge_embeddings = node_features_[hyperedges]  # (batch_size, seq_len, d_model)
        hyperedge_embeddings = hyperedge_embeddings.mean(dim=1)  # (batch_size, d_model)
        hyperedge_embeddings = self.activation1(self.linear2(hyperedge_embeddings))  # (batch_size, d_model)

        # Convert hyperedges to long type for indexing
        hyperedges_flat = hyperedges.view(-1)  # Flatten hyperedges: (batch_size * seq_len,)
        hyperedge_embeddings_expanded = hyperedge_embeddings.repeat_interleave(seq_len,
                                                                               dim=0)  # (batch_size * seq_len, d_model)

        # Initialize node_features_update and counts tensor
        node_features_update = torch.zeros_like(node_features_)
        hyperedge_counts = torch.zeros(node_features_.size(0), device=node_features_.device)

        # Accumulate hyperedge embeddings into node_features_update
        node_features_update.index_add_(0, hyperedges_flat, hyperedge_embeddings_expanded)
        hyperedge_counts.index_add_(0, hyperedges_flat,
                                    torch.ones_like(hyperedges_flat, dtype=torch.float, device=node_features_.device))

        # Normalize node features
        mask = hyperedge_counts > 0
        node_features_update[mask] /= hyperedge_counts[mask].unsqueeze(-1)

        # Final updated node features
        updated_node_features = node_features_ + node_features_update
        updated_node_features = self.activation2(self.linear3(updated_node_features))
        # updated_node_features = self.activation1(self.linear1(updated_node_features))
        # Set the representation of pad_index and unk_index to zero
        node_features_update[0] = 0.0
        node_features_update[1] = 0.0
        return updated_node_features


class HyperEdgeAgg2(nn.Module):
    def __init__(self, fea_dim, d_model):
        super(HyperEdgeAgg2, self).__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(fea_dim, d_model)
        # self.linear2 = nn.Linear(d_model, d_model)
        # self.linear3 = nn.Linear(d_model, d_model)
        self.activation1 = nn.ELU()
        # self.activation2= nn.ELU()

    def forward(self, node_features, hyperedges):
        """
        Args:
            node_features: (num_nodes, fea_dim)
            hyperedges: (batch_size, seq_len)
        Returns:
            updated_node_features: (num_nodes, d_model)
        """
        batch_size, seq_len = hyperedges.shape

        # Extract hyperedge embeddings
        # node_features_ = self.linear1(node_features)
        node_features_ = node_features
        hyperedge_embeddings = node_features_[hyperedges]  # (batch_size, seq_len, d_model)
        hyperedge_embeddings = hyperedge_embeddings.mean(dim=1)  # (batch_size, d_model)
        # hyperedge_embeddings = self.activation1(self.linear2(hyperedge_embeddings))  # (batch_size, d_model)

        # Convert hyperedges to long type for indexing
        hyperedges_flat = hyperedges.view(-1)  # Flatten hyperedges: (batch_size * seq_len,)
        hyperedge_embeddings_expanded = hyperedge_embeddings.repeat_interleave(seq_len,
                                                                               dim=0)  # (batch_size * seq_len, d_model)

        # Initialize node_features_update and counts tensor
        node_features_update = torch.zeros_like(node_features_)
        hyperedge_counts = torch.zeros(node_features_.size(0), device=node_features_.device)

        # Accumulate hyperedge embeddings into node_features_update
        node_features_update.index_add_(0, hyperedges_flat, hyperedge_embeddings_expanded)
        hyperedge_counts.index_add_(0, hyperedges_flat,
                                    torch.ones_like(hyperedges_flat, dtype=torch.float, device=node_features_.device))

        # Normalize node features
        mask = hyperedge_counts > 0
        node_features_update[mask] /= hyperedge_counts[mask].unsqueeze(-1)

        # Final updated node features
        updated_node_features = node_features_ + node_features_update
        # updated_node_features = self.activation2(self.linear3(updated_node_features))
        updated_node_features = self.activation1(self.linear1(updated_node_features))
        # Set the representation of pad_index and unk_index to zero
        node_features_update[0] = 0.0
        node_features_update[1] = 0.0
        return updated_node_features


class GAT3(nn.Module):

    def __init__(self, d_model, in_feature, num_heads_per_layer, num_features_per_layer,
                 add_skip_connection=True, bias=True, dropout=0.6, load_trans_prob=True, avg_last=True):
        super().__init__()
        self.d_model = d_model
        assert len(num_heads_per_layer) == len(num_features_per_layer), f'Enter valid arch params.'

        num_features_per_layer = [d_model] + num_features_per_layer
        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below
        if avg_last:
            assert num_features_per_layer[-1] == d_model
        else:
            assert num_features_per_layer[-1] * num_heads_per_layer[-1] == d_model
        num_of_layers = len(num_heads_per_layer) - 1

        self.agg_layer = HyperEdgeAgg2(in_feature, d_model)

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            if i == num_of_layers - 1:
                if avg_last:
                    concat_input = False
                else:
                    concat_input = True
            else:
                concat_input = True
            layer = GATLayerImp3(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=num_features_per_layer[i + 1],
                num_of_heads=num_heads_per_layer[i + 1],
                concat=concat_input,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                load_trans_prob=load_trans_prob
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    def forward(self, node_features, edge_index_input, edge_prob_input, x, return_first_layer=False):
        """

        Args:
            node_features: (vocab_size, fea_dim)
            edge_index_input: (2, E)
            edge_prob_input: (E, 1)
            x: (B, T)
            return_first_layer: bool : return node_fea_emb in first layer gat or not
        Returns:
            (B, T, d_model)

        """
        new_node_features = self.agg_layer(node_features, x)
        data = (new_node_features, edge_index_input, edge_prob_input)
        first_layer_output = None
        # (node_fea_emb, edge_index, edge_prob) = self.gat_net(data)  # (vocab_size, num_channels[-1]), (2, E)
        for i, layer in enumerate(self.gat_net):
            data = layer(data)
            if i == 0 and return_first_layer:
                first_layer_output = data[0]  # Capture the node_fea_emb of the first layer
        node_fea_emb = data[0]
        batch_size, seq_len = x.shape
        node_fea_emb = node_fea_emb.expand((batch_size, -1, -1))  # (B, vocab_size, d_model)
        node_fea_emb = node_fea_emb.reshape(-1, self.d_model)  # (B * vocab_size, d_model)
        x = x.reshape(-1, 1).squeeze(1)  # (B * T,)
        out_node_fea_emb = node_fea_emb[x].reshape(batch_size, seq_len, self.d_model)  # (B, T, d_model)
        if return_first_layer:
            return out_node_fea_emb, first_layer_output
        else:
            return out_node_fea_emb  # (B, T, d_model)


class MetaGAT(nn.Module):

    def __init__(self, d_model, in_feature, num_heads_per_layer, num_features_per_layer, num_meta_features=6,
                 add_skip_connection=True, bias=True, dropout=0.6, load_trans_prob=True, avg_last=True):
        super().__init__()
        self.d_model = d_model
        assert len(num_heads_per_layer) == len(num_features_per_layer), f'Enter valid arch params.'

        num_features_per_layer = [in_feature] + num_features_per_layer
        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below
        # num_meta_features = 6
        if avg_last:
            assert num_features_per_layer[-1] == d_model
        else:
            assert num_features_per_layer[-1] * num_heads_per_layer[-1] == d_model
        num_of_layers = len(num_heads_per_layer) - 1

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            if i == num_of_layers - 1:
                if avg_last:
                    concat_input = False
                else:
                    concat_input = True
            else:
                concat_input = True
            layer = MetaGATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=num_features_per_layer[i + 1],
                num_meta_features=num_meta_features,
                num_of_heads=num_heads_per_layer[i + 1],
                concat=concat_input,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                load_trans_prob=load_trans_prob
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    def forward(self, node_features, node_meta_features, edge_index_input, edge_prob_input, x,
                return_first_layer=False):
        """

        Args:
            node_features: (vocab_size, fea_dim)
            node_meta_features: (vocab_size, meta_fea_dim)
            edge_index_input: (2, E)
            edge_prob_input: (E, 1)
            x: (B, T)
            return_first_layer: bool : return node_fea_emb in first layer gat or not
        Returns:
            (B, T, d_model)

        """
        data = (node_features, node_meta_features, edge_index_input, edge_prob_input)
        first_layer_output = None
        # (node_fea_emb, edge_index, edge_prob) = self.gat_net(data)  # (vocab_size, num_channels[-1]), (2, E)
        for i, layer in enumerate(self.gat_net):
            data = layer(data)
            if i == 0 and return_first_layer:
                first_layer_output = data[0]  # Capture the node_fea_emb of the first layer
        node_fea_emb = data[0]
        batch_size, seq_len = x.shape
        node_fea_emb = node_fea_emb.expand((batch_size, -1, -1))  # (B, vocab_size, d_model)
        node_fea_emb = node_fea_emb.reshape(-1, self.d_model)  # (B * vocab_size, d_model)
        x = x.reshape(-1, 1).squeeze(1)  # (B * T,)
        out_node_fea_emb = node_fea_emb[x].reshape(batch_size, seq_len, self.d_model)  # (B, T, d_model)
        if return_first_layer:
            return out_node_fea_emb, first_layer_output
        else:
            return out_node_fea_emb  # (B, T, d_model)


# class MYHGNNP(torch.nn.Module):
#     def __init__(self,input_dim,hidden_dim,output_dim,drop_rate=0.5):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         self.layers.append(HGNNPConv(input_dim,hidden_dim,drop_rate=drop_rate))
#         self.layers.append(HGNNPConv(hidden_dim,output_dim,drop_rate=drop_rate))
#     def forward(self,X:torch.Tensor,hg: dhg.Hypergraph):
#         for layer in self.layers:
#             X = layer(X,hg)
#         return X
# class HGNNPConv(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         bias: bool = True,
#         drop_rate: float = 0.5,
#     ):
#         super().__init__()
#         self.act = nn.ReLU(inplace=True)
#         self.drop = nn.Dropout(drop_rate)
#         self.theta = nn.Linear(in_channels, out_channels, bias=bias)
#
#     def forward(self, X: torch.Tensor, hg: dhg.Hypergraph) -> torch.Tensor:
#         X = self.theta(X)
#         Y = hg.v2e(X, aggr="mean")
#         X_ = hg.e2v(Y, aggr="mean")
#         X_ = self.drop(self.act(X_))
#         return X_

class NodeEmbedding(nn.Module):

    def __init__(self, d_model, dropout=0.1, add_time_in_day=False, add_day_in_week=False,
                 add_pe=True, node_fea_dim=10, add_gat=True,
                 gat_heads_per_layer=None, gat_features_per_layer=None, gat_dropout=0.6,
                 load_trans_prob=True, avg_last=True):
        """

        Args:
            vocab_size: total vocab size
            d_model: embedding size of token embedding
            dropout: dropout rate
        """
        super().__init__()
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        self.add_pe = add_pe
        self.add_gat = add_gat

        if self.add_gat:
            self.token_embedding = GAT2(d_model=d_model, in_feature=node_fea_dim,
                                        num_heads_per_layer=gat_heads_per_layer,
                                        num_features_per_layer=gat_features_per_layer,
                                        add_skip_connection=True, bias=True, dropout=gat_dropout,
                                        load_trans_prob=load_trans_prob, avg_last=avg_last)
        if self.add_pe:
            self.position_embedding = PositionalEmbedding(d_model=d_model)
        if self.add_time_in_day:
            self.daytime_embedding = nn.Embedding(1441, d_model, padding_idx=0)
        if self.add_day_in_week:
            self.weekday_embedding = nn.Embedding(8, d_model, padding_idx=0)

        # self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, sequence, position_ids=None, graph_dict=None, return_first_layer=False):
        """

        Args:
            sequence: (B, T, F) [loc, ts, mins, weeks, usr]
            position_ids: (B, T) or None
            graph_dict(dict): including:
                in_lap_mx: (vocab_size, lap_dim)
                out_lap_mx: (vocab_size, lap_dim)
                indegree: (vocab_size, )
                outdegree: (vocab_size, )
                return_first_layer: bool : return node_fea_emb in first layer gat or not
        Returns:
            (B, T, d_model)

        """
        node_fea_first_layer = None
        if self.add_gat:
            if return_first_layer:
                x, node_fea_first_layer = self.token_embedding(node_features=graph_dict['node_features'],
                                                               edge_index_input=graph_dict['edge_index'],
                                                               edge_prob_input=graph_dict['loc_trans_prob'],
                                                               x=sequence[:, :, 0],
                                                               return_first_layer=return_first_layer)  # (B, T, d_model)
            else:
                x = self.token_embedding(node_features=graph_dict['node_features'],
                                         edge_index_input=graph_dict['edge_index'],
                                         edge_prob_input=graph_dict['loc_trans_prob'],
                                         x=sequence[:, :, 0],
                                         return_first_layer=return_first_layer)  # (B, T, d_model)
        if self.add_pe:
            x += self.position_embedding(x, position_ids)  # (B, T, d_model)
        if self.add_time_in_day:
            x += self.daytime_embedding(sequence[:, :, 2])  # (B, T, d_model)
        if self.add_day_in_week:
            x += self.weekday_embedding(sequence[:, :, 3])  # (B, T, d_model)
        # if return_first_layer:
        #     return self.dropout(x), node_fea_first_layer
        # else:
        #     return self.dropout(x)
        # 不进行dropout
        if return_first_layer:
            return x, node_fea_first_layer
        else:
            return x


class NodeEmbedding2(nn.Module):

    def __init__(self, d_model, dropout=0.2, add_time_in_day=False, add_day_in_week=False,
                 add_pe=True, add_time_interval=True, max_time_scale_s=5000, time_interval_scale_list=[10],
                 node_fea_dim=10, add_gat=True, add_hgnn=False,
                 gat_heads_per_layer=None, gat_features_per_layer=None, gat_dropout=0.6,
                 load_trans_prob=True, avg_last=True):
        """

        Args:
            vocab_size: total vocab size
            d_model: embedding size of token embedding
            dropout: dropout rate
        """
        super().__init__()
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        self.add_time_interval = add_time_interval
        self.add_pe = add_pe
        self.add_gat = add_gat
        self.add_hgnn = add_hgnn

        if self.add_gat:
            self.token_embedding = GAT2(d_model=d_model, in_feature=node_fea_dim,
                                        num_heads_per_layer=gat_heads_per_layer,
                                        num_features_per_layer=gat_features_per_layer,
                                        add_skip_connection=True, bias=True, dropout=gat_dropout,
                                        load_trans_prob=load_trans_prob, avg_last=avg_last)
        if self.add_hgnn:
            # self.hgnn_embedding = MYHGNNP(input_dim=node_fea_dim, hidden_dim=d_model, output_dim=d_model,drop_rate=dropout)
            pass
        if self.add_pe:
            self.position_embedding = PositionalEmbedding(d_model=d_model)
        if self.add_time_in_day:
            self.daytime_embedding = nn.Embedding(1441, d_model, padding_idx=0)
        if self.add_day_in_week:
            self.weekday_embedding = nn.Embedding(8, d_model, padding_idx=0)
        if self.add_time_interval:
            if len(time_interval_scale_list) > 1:
                self.time_interval_weights = nn.Parameter(torch.randn(len(time_interval_scale_list)))
                with torch.no_grad():
                    self.time_interval_weights.copy_(torch.tensor((0.2, 0.6, 0.2)))
            else:
                self.time_interval_weights = torch.tensor([1.0])
            # 初始化权重
            self.time_interval_embedding = nn.ModuleList(
                [nn.Embedding(ceil(max_time_scale_s / scale) + 1, d_model, padding_idx=0) for scale in
                 time_interval_scale_list])
            # for scale in time_interval_scale_list:
            #     self.time_interval_embedding.append(
            #         nn.Embedding(ceil(max_time_scale_s / scale), d_model, padding_idx=0))  # 10s级时间间隙

        # self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, sequence, batch_temporal_mat_list=None, position_ids=None, graph_dict=None,
                return_first_layer=False):
        """

        Args:
            sequence: (B, T, F) [loc, ts, mins, weeks, usr]
            batch_temporal_mat: (B, T)
            position_ids: (B, T) or None
            graph_dict(dict): including:
                in_lap_mx: (vocab_size, lap_dim)
                out_lap_mx: (vocab_size, lap_dim)
                indegree: (vocab_size, )
                outdegree: (vocab_size, )
                return_first_layer: bool : return node_fea_emb in first layer gat or not
        Returns:
            (B, T, d_model)

        """
        node_fea_first_layer = None
        if self.add_gat:
            if return_first_layer:
                x, node_fea_first_layer = self.token_embedding(node_features=graph_dict['node_features'],
                                                               edge_index_input=graph_dict['edge_index'],
                                                               edge_prob_input=graph_dict['loc_trans_prob'],
                                                               x=sequence[:, :, 0],
                                                               return_first_layer=return_first_layer)  # (B, T, d_model)
            else:
                x = self.token_embedding(node_features=graph_dict['node_features'],
                                         edge_index_input=graph_dict['edge_index'],
                                         edge_prob_input=graph_dict['loc_trans_prob'],
                                         x=sequence[:, :, 0],
                                         return_first_layer=return_first_layer)  # (B, T, d_model)
        if self.add_hgnn:
            node_embedding = self.hgnn_embedding(graph_dict['node_features'], graph_dict['hyper_graph'])
            x = node_embedding[sequence[:, :, 0]]  # 对应取出
            # x = selected.mean(dim=1)
        if self.add_pe:
            x += self.position_embedding(x, position_ids)  # (B, T, d_model)
        if self.add_time_in_day:
            x += self.daytime_embedding(sequence[:, :, 2])  # (B, T, d_model)
        if self.add_day_in_week:
            x += self.weekday_embedding(sequence[:, :, 3])  # (B, T, d_model)
        if self.add_time_interval:
            # assert batch_temporal_mat is not None,"batch_temporal_mat is none when adding timeinterval embedding"
            # batch_temporal_mat = torch.div(batch_temporal_mat,60,rounding_mode='trunc') #  转为分钟
            # time_interval_emb = self.time_interval_embedding(batch_temporal_mat) # (B, T, T,d_model)
            # for i in range(time_interval_emb.shape[1]):
            #     x[:,i,:]+=time_interval_emb[:,i,i,:]
            # if torch.any(batch_temporal_mat >= 500):
            #     raise ValueError(
            #         f"Found invalid index in batch_temporal_mat. Maximum valid index is {self.embedding.num_embeddings - 1}.")
            weights = F.softmax(self.time_interval_weights, dim=0)
            time_interval_emb_list = []
            for i in range(len(self.time_interval_embedding)):
                time_interval_emb_list.append(self.time_interval_embedding[i](batch_temporal_mat_list[i]))
            sum_time_interval_emb = sum(w * emb for w, emb in zip(weights, time_interval_emb_list))
            x += sum_time_interval_emb  # (B, T, d_model)
            # x += self.time_interval_embedding(batch_temporal_mat_list[0])  # (B, T, d_model)

        # if return_first_layer:
        #     return self.dropout(x), node_fea_first_layer
        # else:
        #     return self.dropout(x)
        # 不进行dropout
        if return_first_layer:
            return x, node_fea_first_layer
        else:
            return x


class NodeEmbedding3(nn.Module):

    def __init__(self, d_model, dropout=0.2, add_time_in_day=False, add_day_in_week=False,
                 add_pe=True, add_time_interval=True, max_time_scale_s=5000, time_interval_scale_list=[10],
                 node_fea_dim=10, add_gat=True, add_hgnn=False,
                 gat_heads_per_layer=None, gat_features_per_layer=None, gat_dropout=0.6,
                 load_trans_prob=True, avg_last=True):
        """

        Args:
            vocab_size: total vocab size
            d_model: embedding size of token embedding
            dropout: dropout rate
        """
        super().__init__()
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        self.add_time_interval = add_time_interval
        self.add_pe = add_pe
        self.add_gat = add_gat
        self.add_hgnn = add_hgnn

        if self.add_gat:
            self.token_embedding = GAT2(d_model=d_model, in_feature=node_fea_dim,
                                        num_heads_per_layer=gat_heads_per_layer,
                                        num_features_per_layer=gat_features_per_layer,
                                        add_skip_connection=True, bias=True, dropout=gat_dropout,
                                        load_trans_prob=load_trans_prob, avg_last=avg_last)
        if self.add_hgnn:
            # self.hgnn_embedding = MYHGNNP(input_dim=node_fea_dim, hidden_dim=d_model, output_dim=d_model,drop_rate=dropout)
            pass
        if self.add_pe:
            self.position_embedding = PositionalEmbedding(d_model=d_model)
        if self.add_time_in_day:
            self.daytime_embedding = nn.Embedding(1441, d_model, padding_idx=0)
        if self.add_day_in_week:
            self.weekday_embedding = nn.Embedding(8, d_model, padding_idx=0)
        if self.add_time_interval:
            if len(time_interval_scale_list) > 1:
                self.time_interval_weights = nn.Parameter(torch.randn(len(time_interval_scale_list)))
                with torch.no_grad():
                    self.time_interval_weights.copy_(torch.tensor((0.6, 0.2, 0.2)))
            else:
                self.time_interval_weights = torch.tensor([1.0])
            # 初始化权重
            self.time_interval_embedding = nn.ModuleList(
                [nn.Embedding(ceil(max_time_scale_s / scale) + 1, d_model, padding_idx=0) for scale in
                 time_interval_scale_list])
            # for scale in time_interval_scale_list:
            #     self.time_interval_embedding.append(
            #         nn.Embedding(ceil(max_time_scale_s / scale), d_model, padding_idx=0))  # 10s级时间间隙

        # self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, sequence, batch_temporal_mat_list=None, position_ids=None, graph_dict=None,
                return_first_layer=False):
        """

        Args:
            sequence: (B, T, F) [loc, ts, mins, weeks, usr]
            batch_temporal_mat: (B, T)
            position_ids: (B, T) or None
            graph_dict(dict): including:
                in_lap_mx: (vocab_size, lap_dim)
                out_lap_mx: (vocab_size, lap_dim)
                indegree: (vocab_size, )
                outdegree: (vocab_size, )
                return_first_layer: bool : return node_fea_emb in first layer gat or not
        Returns:
            (B, T, d_model)

        """
        node_fea_first_layer = None
        if self.add_gat:
            if return_first_layer:
                x, node_fea_first_layer = self.token_embedding(node_features=graph_dict['node_features'],
                                                               edge_index_input=graph_dict['edge_index'],
                                                               edge_prob_input=graph_dict['loc_trans_prob'],
                                                               x=sequence[:, :, 0],
                                                               return_first_layer=return_first_layer)  # (B, T, d_model)
            else:
                x = self.token_embedding(node_features=graph_dict['node_features'],
                                         edge_index_input=graph_dict['edge_index'],
                                         edge_prob_input=graph_dict['loc_trans_prob'],
                                         x=sequence[:, :, 0],
                                         return_first_layer=return_first_layer)  # (B, T, d_model)
        if self.add_hgnn:
            node_embedding = self.hgnn_embedding(graph_dict['node_features'], graph_dict['hyper_graph'])
            x = node_embedding[sequence[:, :, 0]]  # 对应取出
            # x = selected.mean(dim=1)
        if self.add_pe:
            x += self.position_embedding(x, position_ids)  # (B, T, d_model)
        if self.add_time_in_day:
            x += self.daytime_embedding(sequence[:, :, 2])  # (B, T, d_model)
        if self.add_day_in_week:
            x += self.weekday_embedding(sequence[:, :, 3])  # (B, T, d_model)
        if self.add_time_interval:
            # assert batch_temporal_mat is not None,"batch_temporal_mat is none when adding timeinterval embedding"
            # batch_temporal_mat = torch.div(batch_temporal_mat,60,rounding_mode='trunc') #  转为分钟
            # time_interval_emb = self.time_interval_embedding(batch_temporal_mat) # (B, T, T,d_model)
            # for i in range(time_interval_emb.shape[1]):
            #     x[:,i,:]+=time_interval_emb[:,i,i,:]
            # if torch.any(batch_temporal_mat >= 500):
            #     raise ValueError(
            #         f"Found invalid index in batch_temporal_mat. Maximum valid index is {self.embedding.num_embeddings - 1}.")
            weights = F.softmax(self.time_interval_weights, dim=0)
            time_interval_emb_list = []
            for i in range(len(self.time_interval_embedding)):
                time_interval_emb_list.append(self.time_interval_embedding[i](batch_temporal_mat_list[i]))
            sum_time_interval_emb = sum(w * emb for w, emb in zip(weights, time_interval_emb_list))
            x += sum_time_interval_emb  # (B, T, d_model)
            # x += self.time_interval_embedding(batch_temporal_mat_list[0])  # (B, T, d_model)

        # if return_first_layer:
        #     return self.dropout(x), node_fea_first_layer
        # else:
        #     return self.dropout(x)
        # 不进行dropout
        if return_first_layer:
            return x, node_fea_first_layer
        else:
            return x


class NodeEmbedding4(nn.Module):

    def __init__(self, d_model, dropout=0.2, add_time_in_day=False, add_day_in_week=False,
                 add_pe=True, add_time_interval=True, max_time_scale_s=5000, time_interval_scale_list=[10],
                 node_fea_dim=10, meta_features_dim=6, add_gat=False, add_meta_gat=False, add_hgnn=False,
                 gat_heads_per_layer=None, gat_features_per_layer=None, gat_dropout=0.6,
                 load_trans_prob=True, avg_last=True):
        """

        Args:
            vocab_size: total vocab size
            d_model: embedding size of token embedding
            dropout: dropout rate
        """
        super().__init__()
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        self.add_time_interval = add_time_interval
        self.add_pe = add_pe
        self.add_gat = add_gat
        print("add gat?{}".format(add_gat))
        self.add_hgnn = add_hgnn
        self.add_meta_gat = add_meta_gat
        if self.add_gat:
            self.token_embedding = GAT2(d_model=d_model, in_feature=node_fea_dim,
                                        num_heads_per_layer=gat_heads_per_layer,
                                        num_features_per_layer=gat_features_per_layer,
                                        add_skip_connection=True, bias=True, dropout=gat_dropout,
                                        load_trans_prob=load_trans_prob, avg_last=avg_last)
        if self.add_meta_gat:
            self.token_embedding = MetaGAT(d_model=d_model, in_feature=node_fea_dim,
                                           num_heads_per_layer=gat_heads_per_layer,
                                           num_features_per_layer=gat_features_per_layer,
                                           num_meta_features=meta_features_dim,
                                           add_skip_connection=True, bias=True, dropout=gat_dropout,
                                           load_trans_prob=load_trans_prob, avg_last=avg_last)
        if self.add_hgnn:
            # self.hgnn_embedding = MYHGNNP(input_dim=node_fea_dim, hidden_dim=d_model, output_dim=d_model,drop_rate=dropout)
            pass
        if self.add_pe:
            self.position_embedding = PositionalEmbedding(d_model=d_model)
        if self.add_time_in_day:
            self.daytime_embedding = nn.Embedding(1441, d_model, padding_idx=0)
        if self.add_day_in_week:
            self.weekday_embedding = nn.Embedding(8, d_model, padding_idx=0)
        if self.add_time_interval:
            if len(time_interval_scale_list) > 1:
                self.time_interval_weights = nn.Parameter(torch.randn(len(time_interval_scale_list)))
                with torch.no_grad():
                    self.time_interval_weights.copy_(torch.tensor((0.2, 0.6, 0.2)))
            else:
                self.time_interval_weights = torch.tensor([1.0])
            # 初始化权重
            self.time_interval_embedding = nn.ModuleList(
                [nn.Embedding(ceil(max_time_scale_s / scale) + 1, d_model, padding_idx=0) for scale in
                 time_interval_scale_list])
            # for scale in time_interval_scale_list:
            #     self.time_interval_embedding.append(
            #         nn.Embedding(ceil(max_time_scale_s / scale), d_model, padding_idx=0))  # 10s级时间间隙

        # self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, sequence, batch_temporal_mat_list=None, position_ids=None, graph_dict=None,
                return_first_layer=False):
        """

        Args:
            sequence: (B, T, F) [loc, ts, mins, weeks, usr]
            batch_temporal_mat: (B, T)
            position_ids: (B, T) or None
            graph_dict(dict): including:
                in_lap_mx: (vocab_size, lap_dim)
                out_lap_mx: (vocab_size, lap_dim)
                indegree: (vocab_size, )
                outdegree: (vocab_size, )
                return_first_layer: bool : return node_fea_emb in first layer gat or not
        Returns:
            (B, T, d_model)

        """
        node_fea_first_layer = None
        if self.add_gat:
            if return_first_layer:
                x, node_fea_first_layer = self.token_embedding(node_features=graph_dict['node_features'],
                                                               edge_index_input=graph_dict['edge_index'],
                                                               edge_prob_input=graph_dict['loc_trans_prob'],
                                                               x=sequence[:, :, 0],
                                                               return_first_layer=return_first_layer)  # (B, T, d_model)
            else:
                x = self.token_embedding(node_features=graph_dict['node_features'],
                                         edge_index_input=graph_dict['edge_index'],
                                         edge_prob_input=graph_dict['loc_trans_prob'],
                                         x=sequence[:, :, 0],
                                         return_first_layer=return_first_layer)  # (B, T, d_model)
        if self.add_meta_gat:
            if return_first_layer:
                x, node_fea_first_layer = self.token_embedding(node_features=graph_dict['node_features'],
                                                               edge_index_input=graph_dict['edge_index'],
                                                               node_meta_features = graph_dict['node_struct_features'],
                                                               edge_prob_input=graph_dict['loc_trans_prob'],
                                                               x=sequence[:, :, 0],
                                                               return_first_layer=return_first_layer)  # (B, T, d_model)
            else:
                x = self.token_embedding(node_features=graph_dict['node_features'],
                                         edge_index_input=graph_dict['edge_index'],
                                         node_meta_features=graph_dict['node_struct_features'],
                                         edge_prob_input=graph_dict['loc_trans_prob'],
                                         x=sequence[:, :, 0],
                                         return_first_layer=return_first_layer)  # (B, T, d_model)
        if self.add_hgnn:
            node_embedding = self.hgnn_embedding(graph_dict['node_features'], graph_dict['hyper_graph'])
            x = node_embedding[sequence[:, :, 0]]  # 对应取出
            # x = selected.mean(dim=1)
        if self.add_pe:
            x += self.position_embedding(x, position_ids)  # (B, T, d_model)
        if self.add_time_in_day:
            x += self.daytime_embedding(sequence[:, :, 2])  # (B, T, d_model)
        if self.add_day_in_week:
            x += self.weekday_embedding(sequence[:, :, 3])  # (B, T, d_model)
        if self.add_time_interval:
            # assert batch_temporal_mat is not None,"batch_temporal_mat is none when adding timeinterval embedding"
            # batch_temporal_mat = torch.div(batch_temporal_mat,60,rounding_mode='trunc') #  转为分钟
            # time_interval_emb = self.time_interval_embedding(batch_temporal_mat) # (B, T, T,d_model)
            # for i in range(time_interval_emb.shape[1]):
            #     x[:,i,:]+=time_interval_emb[:,i,i,:]
            # if torch.any(batch_temporal_mat >= 500):
            #     raise ValueError(
            #         f"Found invalid index in batch_temporal_mat. Maximum valid index is {self.embedding.num_embeddings - 1}.")
            weights = F.softmax(self.time_interval_weights, dim=0)
            time_interval_emb_list = []
            for i in range(len(self.time_interval_embedding)):
                time_interval_emb_list.append(self.time_interval_embedding[i](batch_temporal_mat_list[i]))
            sum_time_interval_emb = sum(w * emb for w, emb in zip(weights, time_interval_emb_list))
            x += sum_time_interval_emb  # (B, T, d_model)
            # x += self.time_interval_embedding(batch_temporal_mat_list[0])  # (B, T, d_model)

        # if return_first_layer:
        #     return self.dropout(x), node_fea_first_layer
        # else:
        #     return self.dropout(x)
        # 不进行dropout
        if return_first_layer:
            return x, node_fea_first_layer
        else:
            return x


class MultiBinsIntervalEmbedding(nn.Module):
    def __init__(self, num_bins=100, d_model=768):
        super().__init__()
        self.layer1 = nn.Linear(1, num_bins)
        self.emb = nn.Embedding(num_bins, d_model)
        self.activation = nn.Softmax(dim=-1)

    def forward(self, x):
        logit = self.activation(self.layer1(x.unsqueeze(-1)))
        output = logit @ self.emb.weight
        return output
