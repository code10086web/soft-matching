# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
import torch

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
    print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target):
        if sampler == 'softmax':
            return F.cross_entropy(score, target)

        elif cfg.DATALOADER.SAMPLER == 'triplet':
            return triplet(feat, target)[0]

        elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                return xent(score, target) + triplet(feat, target)[0]
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    def part_loss_func(score, feat, target):
        if sampler == 'softmax':
            return F.cross_entropy(score, target)

        elif cfg.DATALOADER.SAMPLER == 'triplet':
            return triplet(feat, target)[0]

        elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                return xent(score, target) + triplet(feat, target)[0]
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))


    loss_funcs = [loss_func, part_loss_func]

    return loss_funcs


def make_loss_with_center(cfg, num_classes, center_criterion_pre=None):    # modified by gu

    feat_dim = cfg.SOLVER.graphconv_dim_out[0] + 2048
    part_feat_dim = cfg.SOLVER.graphconv_dim_out[1] + 2048
    # feat_dim = 2048
    # part_feat_dim = 2048

    if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
        part_center_criterion = CenterLoss(num_classes=num_classes, feat_dim=part_feat_dim, use_gpu=True)  # center loss
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)
        part_center_criterion = CenterLoss(num_classes=num_classes, feat_dim=part_feat_dim, use_gpu=True)
    else:
        print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if not (center_criterion_pre==None):
        center_criterion = center_criterion_pre
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    def loss_func(score, feat, target):
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            return xent(score, target) + \
                    cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            return xent(score, target) + \
                    triplet(feat, target)[0] + \
                    cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    def part_loss_func(score, feat, target):
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            return xent(score, target) + \
                   cfg.SOLVER.CENTER_LOSS_WEIGHT * part_center_criterion(feat, target)

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            return xent(score, target) + \
                   triplet(feat, target)[0] + \
                   cfg.SOLVER.CENTER_LOSS_WEIGHT * part_center_criterion(feat, target)

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    loss_funcs = [loss_func, part_loss_func]
    center_criterions = [center_criterion, part_center_criterion]
    return loss_funcs, center_criterions



