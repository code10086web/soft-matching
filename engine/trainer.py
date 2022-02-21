# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.axislines as axislines
import numpy as np
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from utils.reid_metric import R1_mAP, R1_mAP_reranking
from layers import make_loss, make_loss_with_center
from solver import make_optimizer, make_optimizer_with_center, WarmupMultiStepLR
from modeling.baseline_graphconv import Baseline_graphconv, Baseline_graphconv_all

global ITER
ITER = 0

def create_supervised_trainer(model, optimizer, loss_fn, device=None):

    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img, engine.state.baseline_model)
        loss = loss_fn[0](score[0], feat[0], target) + loss_fn[1](score[1], feat[1], target)
        acc = ((score[0].max(1)[1] == target).float().mean() + (score[1].max(1)[1] == target).float().mean()) / 2
        loss.backward()
        optimizer.step()
        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_trainer_with_center(model, optimizer, optimizer_center, cetner_loss_weight, center_criterion,
                                          loss_fn, device=None):

    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        optimizer_center[0].zero_grad()
        optimizer_center[1].zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img, engine.state.baseline_model)
        # print(score[0].shape,score[1].shape,feat[0].shape,feat[1].shape)
        loss = loss_fn[0](score[0], feat[0], target) + loss_fn[1](score[1], feat[1], target)
        acc = ((score[0].max(1)[1] == target).float().mean() + (score[1].max(1)[1] == target).float().mean()) / 2
        # loss = loss_fn[0](score[0], feat[0], target)
        # acc = (score[0].max(1)[1] == target).float().mean()
        loss.backward()
        optimizer.step()
        for param in center_criterion[0].parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        for param in center_criterion[1].parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        optimizer_center[0].step()
        optimizer_center[1].step()

        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics, device=None):

    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data, engine.state.baseline_model)
            return feat, pids, camids

    engine_ = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine_, name)

    return engine_


def do_train_pre(cfg, model, train_loader, val_loader, num_classes, num_query, start_epoch, model_pre=None):

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    eval_period = cfg.SOLVER.all_EVAL_PERIOD
    steps = cfg.SOLVER.all_STEPS
    warmup_iters = cfg.SOLVER.all_WARMUP_ITERS

    if 'center' in cfg.SOLVER.graphconv_loss:
        loss_func, center_criterion = make_loss_with_center(cfg, num_classes)
        optimizer, optimizer_center = make_optimizer_with_center(cfg, model, center_criterion)
        trainer = create_supervised_trainer_with_center(model, optimizer, optimizer_center,
                                                        cfg.SOLVER.CENTER_LOSS_WEIGHT, center_criterion,
                                                        loss_func, device=device)
    else:
        loss_func = make_loss(cfg, num_classes)
        optimizer = make_optimizer(cfg, model)
        trainer = create_supervised_trainer(model, optimizer, loss_func, device=device)

    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, cfg, max_rank=50)}, device=device)
    scheduler = WarmupMultiStepLR(optimizer, steps, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                  warmup_iters, cfg.SOLVER.WARMUP_METHOD)


    print('Train all, the loss type is', cfg.SOLVER.graphconv_loss)
    logger = logging.getLogger("reid_baseline+graphconv.train")
    logger.info("Start training")

    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    if 'center' in cfg.SOLVER.graphconv_loss:
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                         'optimizer': optimizer,
                                                                         'center_param': center_criterion,
                                                                         'optimizer_center': optimizer_center})
    else:
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                         'optimizer': optimizer})


    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch
        engine.state.edge_weight = cfg.SOLVER.edge_weight

        model_pre.eval()
        engine.state.baseline_model = model_pre

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()


    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))

        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Loss per batch: {:.3f}, Acc per batch: {:.3f}'
                    .format(engine.state.epoch, engine.state.metrics['avg_loss'],
                            engine.state.metrics['avg_acc']))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        # if engine.state.epoch % eval_period == 0:
        if engine.state.epoch in np.array(eval_period):
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']

            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

            cfg.SOLVER.test_result_log.append([engine.state.epoch, round(mAP*100, 2), round(cmc[0]*100, 2)])

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_lossacc(engine):
        cfg.SOLVER.train_result_log.append([engine.state.epoch,
                                      '%.3f' % engine.state.metrics['avg_loss'],
                                      '%.3f' % engine.state.metrics['avg_acc'],
                                      scheduler.get_lr()[0]])

    @evaluator.on(Events.STARTED)
    def start_evaluating(engine):
        model_pre.eval()
        engine.state.baseline_model = model_pre


    return trainer

def do_train(
        cfg,
        train_loader,
        val_loader,
        num_query,
        start_epoch,
        num_classes
):

    epochs = cfg.SOLVER.all_MAX_EPOCHS
    model_pre = torch.load(cfg.MODEL.PREMODEL)
    model = Baseline_graphconv_all(num_classes, cfg)

    trainer_all = do_train_pre(cfg, model, train_loader, val_loader, num_classes, num_query,
                               start_epoch, model_pre=model_pre)
    trainer_all.run(train_loader, max_epochs=epochs)


