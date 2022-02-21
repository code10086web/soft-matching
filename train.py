# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
import torch
import random
import numpy as np

from torch.backends import cudnn
from util import data_manager
from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from utils.logger import setup_logger
sys.path.append('.')

import torch
import random
import numpy as np
import os

# def set_seed(seed=1): # seed的数值可以随意设置，本人不清楚有没有推荐数值
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     #根据文档，torch.manual_seed(seed)应该已经为所有设备设置seed
#     #但是torch.cuda.manual_seed(seed)在没有gpu时也可调用，这样写没什么坏处
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     #cuDNN在使用deterministic模式时（下面两行），可能会造成性能下降（取决于model）
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False



def train(cfg, args):

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # set_seed(0)


    # prepare dataset
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg, args)

    start_epoch = 0

    do_train(
        cfg,
        train_loader,
        val_loader,
        num_query,
        start_epoch,
        num_classes
    )

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/softmax_triplet_with_center.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    
    parser.add_argument('--root', type=str, default='data', help="root path to data directory")
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
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
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline+graphconv", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    cudnn.benchmark = True
    train(cfg, args)


if __name__ == '__main__':
    main()
