# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
import random

from torch_geometric.nn import MessagePassing
from torch import nn
from .backbones.resnet import ResNet, BasicBlock
from modeling.compute_edge import Compute_edge, Compute_edge_no_weight
from torch.nn import Linear as Lin, ReLU, BatchNorm1d as BN, Sequential as Seq
from torchvision.models.resnet import Bottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a


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
        # return out1 + out2

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
            # in_channels = num_layers * out_channels
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
            xs.append(x)

        # xs.remove(xs[0])
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


class Baseline_graphconv(nn.Module):

    def __init__(self, cfg, model_b, model_g):
        super(Baseline_graphconv, self).__init__()

        self.k = cfg.SOLVER.edge_k

        self.base = model_b.base
        self.gap = model_b.gap
        self.bottleneck = model_b.bottleneck
        self.classifier = model_b.classifier

        self.graphc1 = model_g.graphc1
        self.graphc2 = model_g.graphc2

    def forward(self, x):

        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        edge_global = Compute_edge_no_weight(global_feat, self.k)
        global_feat_with_graph = self.graphc1(global_feat, edge_global, None)

        feat_with_graph = self.bottleneck(global_feat_with_graph)  # normalize for angular softmax
        edge = Compute_edge_no_weight(feat_with_graph, self.k)
        feat_with_graph = self.graphc2(feat_with_graph, edge, None)

        if self.training:
            cls_score = self.classifier(feat_with_graph)
            return cls_score, global_feat_with_graph  # global feature for triplet loss
        else:
            return feat_with_graph


class BatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h - rh)
            sy = random.randint(0, w - rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx + rh, sy:sy + rw] = 0
            x = x * mask
        return x


class Baseline_graphconv_all(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, cfg):
        super(Baseline_graphconv_all, self).__init__()

        last_stride = cfg.MODEL.LAST_STRIDE
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        model_path = cfg.MODEL.PRETRAIN_PATH
        dim_out = cfg.SOLVER.graphconv_dim_out[0]
        part_dim_out = cfg.SOLVER.graphconv_dim_out[1]
        num_layers = cfg.SOLVER.num_layers
        model_name = cfg.MODEL.NAME

        self.k = cfg.SOLVER.edge_k
        self.all_part = cfg.SOLVER.all_part
        self.edge_weight = cfg.SOLVER.edge_weight

        if model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.graphc = graphc(self.in_planes, dim_out, num_layers)
        # self.bottleneck = nn.BatchNorm1d(dim_out+self.in_planes)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        # self.reduction = nn.Sequential(
        #     nn.Linear(dim_out+self.in_planes, 1024, 1),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU()
        # )
        self.reduction = nn.Sequential(
            nn.Linear(self.in_planes, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.reduction.apply(weights_init_kaiming)
        self.classifier = nn.Linear(1024, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.res_part = Bottleneck(2048, 512)
        self.batch_crop = BatchDrop(0.33, 1.0)
        self.part_gap = nn.AdaptiveMaxPool2d((1, 1))
        self.part_graphc = graphc(self.in_planes, part_dim_out, num_layers)
        # self.part_bottleneck = nn.BatchNorm1d(dim_out+self.in_planes)
        self.part_bottleneck = nn.BatchNorm1d(self.in_planes)
        self.part_bottleneck.bias.requires_grad_(False)  # no shift
        self.part_bottleneck.apply(weights_init_kaiming)
        # self.part_classifier = nn.Linear(dim_out+self.in_planes, num_classes, bias=False)
        self.part_classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.part_classifier.apply(weights_init_classifier)


    def forward(self, x, baseline_model):

        cls_score = []
        after_feat_with_graph = []
        feat_with_graph = []

        feat = self.base(x)

        # global branch
        global_feat = self.gap(feat)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        before_feat_for_edge = baseline_model(x, graph_flag=True)
        if self.edge_weight:
            edge_before, edge_weight_before = Compute_edge(before_feat_for_edge, self.k)
            global_feat_with_graph = self.graphc(global_feat, edge_before, edge_weight_before)
        else:
            edge_before = Compute_edge_no_weight(before_feat_for_edge, self.k)
            global_feat_with_graph = self.graphc(global_feat, edge_before, None)
        # global_feat_with_graph = torch.cat((global_feat_with_graph, global_feat), 1)
        global_feat_with_graph = global_feat
        feat_with_graph.append(global_feat_with_graph)

        global_after_feat_with_graph = self.bottleneck(global_feat_with_graph)  # normalize for angular softmax
        global_after_feat_with_graph = self.reduction(global_after_feat_with_graph)
        after_feat_with_graph.append(global_after_feat_with_graph)

        if self.all_part:
            # part branch
            feat = self.res_part(feat)
            part_feat = self.batch_crop(feat)
            part_feat = self.part_gap(part_feat)
            part_feat = part_feat.view(part_feat.shape[0], -1)

            if self.edge_weight:
                part_feat_with_graph = self.part_graphc(part_feat, edge_before, edge_weight_before)
            else:
                part_feat_with_graph = self.part_graphc(part_feat, edge_before, None)
            # part_feat_with_graph = torch.cat((part_feat_with_graph, part_feat), 1)
            part_feat_with_graph = part_feat
            feat_with_graph.append(part_feat_with_graph)

            part_after_feat_with_graph = self.part_bottleneck(part_feat_with_graph)
            after_feat_with_graph.append(part_after_feat_with_graph)

        if self.training:
            cls_score.append(self.classifier(global_after_feat_with_graph))
            if self.all_part:
                cls_score.append(self.part_classifier(part_after_feat_with_graph))
            return cls_score, feat_with_graph  # global feature for triplet loss
        else:
            return torch.cat(after_feat_with_graph, 1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for k, v in param_dict.state_dict().items():
            if 'classifier' in k:
                continue
            self.state_dict()[k].copy_(param_dict.state_dict()[k])




# # encoding: utf-8
# """
# @author:  liaoxingyu
# @contact: sherlockliao01@gmail.com
# """
#
# import torch
# import torch.nn.functional as F
# import random
#
# from torch_geometric.nn import MessagePassing
# from torch import nn
# from .backbones.resnet import ResNet, BasicBlock
# from modeling.compute_edge import Compute_edge, Compute_edge_no_weight
# from torch.nn import Linear as Lin, ReLU, BatchNorm1d as BN, Sequential as Seq
# from torchvision.models.resnet import Bottleneck
# from .backbones.resnet_ibn_a import resnet50_ibn_a
#
#
# def weights_init_kaiming(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
#         nn.init.constant_(m.bias, 0.0)
#     elif classname.find('Conv') != -1:
#         nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0.0)
#     elif classname.find('BatchNorm') != -1:
#         if m.affine:
#             nn.init.constant_(m.weight, 1.0)
#             nn.init.constant_(m.bias, 0.0)
#
#
# def weights_init_classifier(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         nn.init.normal_(m.weight, std=0.001)
#         if m.bias:
#             nn.init.constant_(m.bias, 0.0)
#
# class RelConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super(RelConv, self).__init__(aggr='mean')
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#
#         self.lin1 = Lin(in_channels, out_channels, bias=False)
#         self.lin2 = Lin(in_channels, out_channels, bias=False)
#         self.root = Lin(in_channels, out_channels)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.lin1.reset_parameters()
#         self.lin2.reset_parameters()
#         self.root.reset_parameters()
#
#     def forward(self, x, edge, edge_weight):
#         """"""
#         if edge_weight == None:
#             self.flow = 'source_to_target'
#             out1 = self.propagate(edge, x=self.lin1(x))
#             self.flow = 'target_to_source'
#             out2 = self.propagate(edge, x=self.lin2(x))
#         else:
#             self.flow = 'source_to_target'
#             out1 = self.propagate(edge, x=self.lin1(x), norm=edge_weight[edge[0,], edge[1,]])
#             self.flow = 'target_to_source'
#             out2 = self.propagate(edge, x=self.lin1(x), norm=edge_weight[edge[1,], edge[0,]])
#         return self.root(x) + out1 + out2
#         # return out1 + out2
#
#     def message(self, x_j):
#         return x_j
#
#     def __repr__(self):
#         return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
#                                    self.out_channels)
#
#
# class RelCNN(nn.Module):
#     def __init__(self, in_channels, out_channels, num_layers, batch_norm=False,
#                  cat=True, lin=True, dropout=0.0):
#         super(RelCNN, self).__init__()
#
#         self.in_channels = in_channels
#         self.num_layers = num_layers
#         self.batch_norm = batch_norm
#         self.cat = cat
#         self.lin = lin
#         self.dropout = dropout
#
#         self.convs = torch.nn.ModuleList()
#         self.batch_norms = torch.nn.ModuleList()
#         for _ in range(num_layers):
#             self.convs.append(RelConv(in_channels, out_channels))
#             self.batch_norms.append(BN(out_channels))
#             in_channels = out_channels
#
#         if self.cat:
#             in_channels = self.in_channels + num_layers * out_channels
#             # in_channels = num_layers * out_channels
#         else:
#             in_channels = out_channels
#
#         if self.lin:
#             self.out_channels = out_channels
#             self.final = Lin(in_channels, out_channels)
#         else:
#             self.out_channels = in_channels
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         for conv, batch_norm in zip(self.convs, self.batch_norms):
#             conv.reset_parameters()
#             batch_norm.reset_parameters()
#         if self.lin:
#             self.final.reset_parameters()
#
#     def forward(self, x, edge, edge_weight,  *args):
#         """"""
#         xs = [x]
#
#         for conv, batch_norm in zip(self.convs, self.batch_norms):
#             x = conv(xs[-1], edge, edge_weight)
#             x = batch_norm(F.relu(x)) if self.batch_norm else F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#             xs.append(x)
#
#         # xs.remove(xs[0])
#         x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
#         x = self.final(x) if self.lin else x
#         return x
#
#     def __repr__(self):
#         return ('{}({}, {}, num_layers={}, batch_norm={}, cat={}, lin={}, '
#                 'dropout={})').format(self.__class__.__name__,
#                                       self.in_channels, self.out_channels,
#                                       self.num_layers, self.batch_norm,
#                                       self.cat, self.lin, self.dropout)
#
#
# class graphc(nn.Module):
#     def __init__(self, input_dim, output_dim, num_layers):
#         super(graphc, self).__init__()
#
#         self.RelCNN = RelCNN(input_dim, output_dim, num_layers,
#                              batch_norm=False, cat=True, lin=True, dropout=0.5)
#
#     def forward(self, x, edge_index, edge_weight):
#         """
#         :param x: input image tensor of (N, C, H, W)
#         :return: (prediction, triplet_losses, softmax_losses)
#         """
#         x = self.RelCNN(x, edge_index, edge_weight)
#
#         return x
#
#
# class Baseline_graphconv(nn.Module):
#
#     def __init__(self, cfg, model_b, model_g):
#         super(Baseline_graphconv, self).__init__()
#
#         self.k = cfg.SOLVER.edge_k
#
#         self.base = model_b.base
#         self.gap = model_b.gap
#         self.bottleneck = model_b.bottleneck
#         self.classifier = model_b.classifier
#
#         self.graphc1 = model_g.graphc1
#         self.graphc2 = model_g.graphc2
#
#     def forward(self, x):
#
#         global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
#         global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
#
#         edge_global = Compute_edge_no_weight(global_feat, self.k)
#         global_feat_with_graph = self.graphc1(global_feat, edge_global, None)
#
#         feat_with_graph = self.bottleneck(global_feat_with_graph)  # normalize for angular softmax
#         edge = Compute_edge_no_weight(feat_with_graph, self.k)
#         feat_with_graph = self.graphc2(feat_with_graph, edge, None)
#
#         if self.training:
#             cls_score = self.classifier(feat_with_graph)
#             return cls_score, global_feat_with_graph  # global feature for triplet loss
#         else:
#             return feat_with_graph
#
#
# class BatchDrop(nn.Module):
#     def __init__(self, h_ratio, w_ratio):
#         super(BatchDrop, self).__init__()
#         self.h_ratio = h_ratio
#         self.w_ratio = w_ratio
#
#     def forward(self, x):
#         if self.training:
#             h, w = x.size()[-2:]
#             rh = round(self.h_ratio * h)
#             rw = round(self.w_ratio * w)
#             sx = random.randint(0, h - rh)
#             sy = random.randint(0, w - rw)
#             mask = x.new_ones(x.size())
#             mask[:, :, sx:sx + rh, sy:sy + rw] = 0
#             x = x * mask
#         return x
#
#
# class Baseline_graphconv_all(nn.Module):
#     in_planes = 2048
#
#     def __init__(self, num_classes, cfg):
#         super(Baseline_graphconv_all, self).__init__()
#
#         last_stride = cfg.MODEL.LAST_STRIDE
#         pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         model_path = cfg.MODEL.PRETRAIN_PATH
#         dim_out = cfg.SOLVER.graphconv_dim_out[0]
#         part_dim_out = cfg.SOLVER.graphconv_dim_out[1]
#         num_layers = cfg.SOLVER.num_layers
#         model_name = cfg.MODEL.NAME
#
#         self.k = cfg.SOLVER.edge_k
#         self.all_part = cfg.SOLVER.all_part
#         self.edge_weight = cfg.SOLVER.edge_weight
#
#         if model_name == 'resnet50':
#             self.base = ResNet(last_stride=last_stride,
#                                block=Bottleneck,
#                                layers=[3, 4, 6, 3])
#         elif model_name == 'resnet50_ibn_a':
#             self.base = resnet50_ibn_a(last_stride)
#         if pretrain_choice == 'imagenet':
#             self.base.load_param(model_path)
#             print('Loading pretrained ImageNet model......')
#
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.graphc = graphc(self.in_planes, dim_out, num_layers)
#         self.bottleneck = nn.BatchNorm1d(dim_out+self.in_planes)
#         # self.bottleneck = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck.bias.requires_grad_(False)  # no shift
#         self.bottleneck.apply(weights_init_kaiming)
#         self.reduction = nn.Sequential(
#             nn.Linear(dim_out+self.in_planes, 1024, 1),
#             nn.BatchNorm1d(1024),
#             nn.ReLU()
#         )
#         # self.reduction = nn.Sequential(
#         #     nn.Linear(self.in_planes, 1024, 1),
#         #     nn.BatchNorm1d(1024),
#         #     nn.ReLU()
#         # )
#         self.reduction.apply(weights_init_kaiming)
#         self.classifier = nn.Linear(1024, num_classes, bias=False)
#         self.classifier.apply(weights_init_classifier)
#
#         self.res_part = Bottleneck(2048, 512)
#         self.batch_crop = BatchDrop(0.33, 1.0)
#         self.part_gap = nn.AdaptiveMaxPool2d((1, 1))
#         self.part_graphc = graphc(self.in_planes, part_dim_out, num_layers)
#         self.part_bottleneck = nn.BatchNorm1d(dim_out+self.in_planes)
#         # self.part_bottleneck = nn.BatchNorm1d(self.in_planes)
#         self.part_bottleneck.bias.requires_grad_(False)  # no shift
#         self.part_bottleneck.apply(weights_init_kaiming)
#         self.part_classifier = nn.Linear(dim_out+self.in_planes, num_classes, bias=False)
#         # self.part_classifier = nn.Linear(self.in_planes, num_classes, bias=False)
#         self.part_classifier.apply(weights_init_classifier)
#
#
#     def forward(self, x, baseline_model):
#
#         cls_score = []
#         after_feat_with_graph = []
#         feat_with_graph = []
#
#         feat = self.base(x)
#
#         # global branch
#         global_feat = self.gap(feat)  # (b, 2048, 1, 1)
#         global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
#
#         before_feat_for_edge = baseline_model(x, graph_flag=True)
#         if self.edge_weight:
#             edge_before, edge_weight_before = Compute_edge(before_feat_for_edge, self.k)
#             global_feat_with_graph = self.graphc(global_feat, edge_before, edge_weight_before)
#         else:
#             edge_before = Compute_edge_no_weight(before_feat_for_edge, self.k)
#             global_feat_with_graph = self.graphc(global_feat, edge_before, None)
#         global_feat_with_graph = torch.cat((global_feat_with_graph, global_feat), 1)
#         # global_feat_with_graph = global_feat
#         feat_with_graph.append(global_feat_with_graph)
#
#         global_after_feat_with_graph = self.bottleneck(global_feat_with_graph)  # normalize for angular softmax
#         global_after_feat_with_graph = self.reduction(global_after_feat_with_graph)
#         after_feat_with_graph.append(global_after_feat_with_graph)
#
#         if self.all_part:
#             # part branch
#             feat = self.res_part(feat)
#             part_feat = self.batch_crop(feat)
#             part_feat = self.part_gap(part_feat)
#             part_feat = part_feat.view(part_feat.shape[0], -1)
#
#             if self.edge_weight:
#                 part_feat_with_graph = self.part_graphc(part_feat, edge_before, edge_weight_before)
#             else:
#                 part_feat_with_graph = self.part_graphc(part_feat, edge_before, None)
#             part_feat_with_graph = torch.cat((part_feat_with_graph, part_feat), 1)
#             # part_feat_with_graph = part_feat
#             feat_with_graph.append(part_feat_with_graph)
#
#             part_after_feat_with_graph = self.part_bottleneck(part_feat_with_graph)
#             after_feat_with_graph.append(part_after_feat_with_graph)
#
#         if self.training:
#             cls_score.append(self.classifier(global_after_feat_with_graph))
#             if self.all_part:
#                 cls_score.append(self.part_classifier(part_after_feat_with_graph))
#             return cls_score, feat_with_graph  # global feature for triplet loss
#         else:
#             return torch.cat(after_feat_with_graph, 1)
#
#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         for k, v in param_dict.state_dict().items():
#             if 'classifier' in k:
#                 continue
#             self.state_dict()[k].copy_(param_dict.state_dict()[k])