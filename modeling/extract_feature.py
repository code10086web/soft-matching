# encoding: utf-8

import logging

import torch
import torch.nn as nn
import numpy as np
from ignite.engine import Engine, Events

from utils.reid_metric import R1_mAP, R1_mAP_reranking

class _feature_update:

    def __init__(self):
        self.feats = []
        self.pids = []

    def attach(self, engine, start=Events.ITERATION_COMPLETED):
        engine.add_event_handler(start, self.update)

    def update(self, engine, *args):
        feat, pids = engine.state.output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pids))



def create_extract_feature(model, device=None):

    def _extract_feature(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids

    return Engine(_extract_feature)



def extract_feature(
        cfg,
        model,
        train_loader,
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline.extract feature")
    logger.info("Enter extract feature")
    extractor = create_extract_feature(model, device=device)
    feature_update = _feature_update()
    feature_update.attach(extractor, start=Events.ITERATION_COMPLETED)

    extractor.run(train_loader)
    feats = feature_update.feats
    feats = torch.cat(feats)

