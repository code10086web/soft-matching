# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn

sys.path.append('.')
from util import data_manager
from config import cfg
from data import make_data_loader
from engine.inference import inference
from utils.logger import setup_logger
from modeling.baseline_graphconv import Baseline_graphconv_all


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
       "--config_file", default="configs/softmax_triplet_with_center.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    
    parser.add_argument('--root', type=str, default='data', help="root path to data directory")
    parser.add_argument('-d', '--dataset', type=str, default='msmt17',
                    choices=data_manager.get_names())
    parser.add_argument('--split-id', type=int, default=0, help="split index")
    parser.add_argument('--cuhk03-labeled', action='store_true',
                    help="whether to use labeled images, if false, detected images are used (default: False)")
    parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="whether to use classic split by Li et al. CVPR'14 (default: False)")
    
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg, args)
    # model = build_model(cfg, num_classes)
    # model.load_param(cfg.TEST.WEIGHT)
    model = Baseline_graphconv_all(num_classes, cfg)
    model.load_param(cfg.TEST.WEIGHT)
    model_pre = torch.load(cfg.MODEL.PREMODEL)
    # model_pre = torch.load('/home/Teacao/Documents/2019-CVPRW-BagofTricks1/results/{:s}/resnet50_model_120.pth'.format(cfg.DATASETS.NAMES))

    inference(cfg, model, model_pre, val_loader, num_query)


if __name__ == '__main__':
    main()
