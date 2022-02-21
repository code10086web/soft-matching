# encoding: utf-8

import torch
import numpy as np

from torch import nn
from torch_geometric.nn import MessagePassing
from torch.nn import Linear as Lin, ReLU, BatchNorm1d as BN, Sequential as Seq
from modeling.compute_edge import Compute_edge, Compute_edge_no_weight
import torch.nn.functional as F

try:
    from pykeops.torch import LazyTensor
except ImportError:
    LazyTensor = None


def __top_k__(x_s, x_t, k):  # pragma: no cover
    r"""Memory-efficient top-k correspondence computation."""
    x_s, x_t = x_s.unsqueeze(-2), x_t.unsqueeze(-3)
    if LazyTensor is not None:
        x_s, x_t = LazyTensor(x_s), LazyTensor(x_t)
        S_ij = (-x_s * x_t).sum(dim=-1)
        return S_ij.argKmin(k, dim=2, backend='auto')
    else:
        S_ij = (x_s * x_t).sum(dim=-1)
        return S_ij.topk(k, dim=2)[1]


def masked_softmax(src, mask, dim=-1):
    out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)
    return out


def to_sparse(x, mask):
    return x[mask]


def to_dense(x, mask):
    out = x.new_zeros(tuple(mask.size()) + (x.size(-1), ))
    out[mask] = x
    return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def Normalize(data):
    mx = np.max(data, axis=1)
    mn = np.min(data, axis=1)
    for i in range(data.shape[0]):
        data[i] = (data[i] - mn[i]) / (mx[i] - mn[i])
    return data


def sim_generator(feat_s, feat_t):
    feat_s, feat_t = feat_s.squeeze(0), feat_t.squeeze(0)
    m, n = feat_s.shape[0], feat_t.shape[0]
    feats = torch.cat((feat_s, feat_t), dim=0)
    feats = torch.nn.functional.normalize(feats, dim=1, p=2)
    feat_s = feats[:m]
    feat_t = feats[m:]
    distmat = torch.pow(feat_s, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(feat_t, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, feat_s, feat_t.t())
    # sim = (-distmat).exp_()
    sim = -distmat
    return sim.unsqueeze(0)


class RelConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(RelConv, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin1 = Lin(in_channels, out_channels, bias=False)
        self.lin2 = Lin(in_channels, out_channels, bias=False)
        self.root = Lin(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.root.reset_parameters()

    def forward(self, x, edge, edge_weight):
        """"""
        if edge_weight == None:
            self.flow = 'source_to_target'
            out1 = self.propagate(edge, x=self.lin1(x))
            self.flow = 'target_to_source'
            out2 = self.propagate(edge, x=self.lin2(x))
        else:
            self.flow = 'source_to_target'
            out1 = self.propagate(edge, x=self.lin1(x), norm=edge_weight[edge[0,], edge[1,]])
            self.flow = 'target_to_source'
            out2 = self.propagate(edge, x=self.lin1(x), norm=edge_weight[edge[1,], edge[0,]])
        return self.root(x) + out1 + out2

    def message(self, x_j):
        return x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class RelCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, batch_norm=False,
                 cat=True, lin=True, dropout=0.0):
        super(RelCNN, self).__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.cat = cat
        self.lin = lin
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RelConv(in_channels, out_channels))
            self.batch_norms.append(BN(out_channels))
            in_channels = out_channels

        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.final = Lin(in_channels, out_channels)
        else:
            self.out_channels = in_channels

        self.reset_parameters()

    def reset_parameters(self):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            conv.reset_parameters()
            batch_norm.reset_parameters()
        if self.lin:
            self.final.reset_parameters()

    def forward(self, x, edge, edge_weight,  *args):
        """"""
        xs = [x]

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(xs[-1], edge, edge_weight)
            x = batch_norm(F.relu(x)) if self.batch_norm else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # x = 0.3 * x + 0.7 * xs[0]
            xs.append(x)

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = self.final(x) if self.lin else x
        return x

    def __repr__(self):
        return ('{}({}, {}, num_layers={}, batch_norm={}, cat={}, lin={}, '
                'dropout={})').format(self.__class__.__name__,
                                      self.in_channels, self.out_channels,
                                      self.num_layers, self.batch_norm,
                                      self.cat, self.lin, self.dropout)


class graphc(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers):
        super(graphc, self).__init__()

        self.RelCNN = RelCNN(input_dim, output_dim, num_layers,
                             batch_norm=False, cat=True, lin=True, dropout=0.5)

    def forward(self, x, edge_index, edge_weight):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.RelCNN(x, edge_index, edge_weight)

        return x


class Graphconv(nn.Module):

    def __init__(self, num_classes, cfg):
        super(Graphconv, self).__init__()

        dim_out1 = cfg.SOLVER.graphconv1_dim_out
        dim_out2 = cfg.SOLVER.graphconv2_dim_out
        num_layers = cfg.SOLVER.num_layers

        self.k = cfg.SOLVER.edge_k
        self.graphc1 = graphc(2048, dim_out1, num_layers)
        self.graphc2 = graphc(2048, dim_out2, num_layers)


    def forward(self, x, baseline):

        global_feat, feat = baseline(x, graph_flag=True)

        edge_global = Compute_edge_no_weight(global_feat, self.k)
        global_feat = self.graphc1(global_feat, edge_global, None)

        edge = Compute_edge_no_weight(feat, self.k)
        feat = self.graphc2(feat, edge, None)

        if self.training:
            cls_score = baseline(feat, graph_classifier_flag=True)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            return feat




