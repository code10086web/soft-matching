# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import numpy as np
from random import randint

def Preprocess(num_instances, batch):

    num_single = int(num_instances / 2)
    imgs, pids = batch
    _, c, H, W = imgs.shape

    imgs = imgs.view(-1, num_instances, c, H, W)
    pids = pids.view(-1, num_instances)

    imgs_s = imgs[:, :num_single, :, :, :].reshape(-1, c, H, W)
    pids_s = pids[:, :num_single].reshape(-1)
    imgs_t = imgs[:, num_single:, :, :, :].reshape(-1, c, H, W)
    pids_t = pids[:, num_single:].reshape(-1)

    batch_s = (imgs_s, pids_s)
    batch_t = (imgs_t, pids_t)

    pids_s, pids_t = pids_s.numpy(), pids_t.numpy()
    y_s = np.arange(len(pids_s))
    y_t = np.zeros_like(y_s)
    for i in range(len(pids_s)):
        id = np.where(pids_t == pids_s[i])
        y_t[i] = id[0][randint(0, len(id))]
    y = torch.stack([torch.from_numpy(y_s), torch.from_numpy(y_t)], dim=0)


    return batch_s, batch_t, y