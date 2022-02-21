# encoding: utf-8

import torch
import random
import numpy as np

from torch import nn
from modeling.compute_edge import Compute_edge, Compute_edge_no_weight


def random_id(ids, num, batch_num):
    random.shuffle(ids)
    rem_len = num % batch_num
    if rem_len == 0:
        id = ids.reshape(-1, batch_num).tolist()
    else:
        rem = ids[:rem_len].tolist()
        id = ids[rem_len:]
        id = id.reshape(-1, batch_num).tolist()
        id.append(rem)
    return id


class Graph_test(nn.Module):
    def __init__(self, model, q_batch, g_batch, graphconv_test_epoch, k, device):
        super(Graph_test, self).__init__()
        self.model = model
        self.q_batch = q_batch
        self.g_batch = g_batch
        self.graphconv_test_epoch = graphconv_test_epoch
        self.k = k
        self.device = device


    def compute_feat(self, feat):
        self.model.eval()
        with torch.no_grad():
            edge_index = Compute_edge_no_weight(feat, self.k)
            edge_index = edge_index.to(self.device) if torch.cuda.device_count() >= 1 else edge_index
            feat = self.model(feat, edge_index)
        return feat, edge_index


    def compute_pair(self, feat, edge_index):
        self.model.eval()
        with torch.no_grad():
            S0, SL = self.model(feat, edge_index)
        return S0, SL


    def one_step(self, S_0, S_L, qf, gf, q_id_batch, g_id_batch):
        qf_batch, edge_index_q = self.compute_feat(qf[q_id_batch, :])
        gf_batch, edge_index_g = self.compute_feat(gf[g_id_batch, :])

        feat = [qf_batch, gf_batch]
        edge_index = [edge_index_q, edge_index_g]
        S0, SL = self.compute_pair(feat, edge_index)

        [np.put(S_0[q_id_batch[j], :], g_id_batch, S0[j, :].cpu()) for j in range(len(q_id_batch))]
        [np.put(S_L[q_id_batch[j], :], g_id_batch, SL[j, :].cpu()) for j in range(len(q_id_batch))]
        return S_0, S_L


    def forward(self, qf, gf):
        q_num, g_num = qf.shape[0], gf.shape[0]

        q_ids, g_ids = np.arange(q_num), np.arange(g_num)
        S_0, S_L = np.zeros([q_num, g_num]), np.zeros([q_num, g_num])
        for i in range(self.graphconv_test_epoch):
            q_id = random_id(q_ids, q_num, self.q_batch)
            g_id = random_id(g_ids, g_num, self.g_batch)

            S_0_temp, S_L_temp = np.zeros_like(S_0), np.zeros_like(S_L)
            for q_id_batch in q_id:
                for g_id_batch in g_id:
                    if i == 0:
                        S_0, S_L = self.one_step(S_0, S_L, qf, gf, q_id_batch, g_id_batch)
                    else:
                        S_0_temp, S_L_temp = self.one_step(S_0_temp, S_L_temp, qf, gf, q_id_batch, g_id_batch)

            if not i == 0:
                S_0, S_L = (S_0 + S_0_temp) / 2, (S_L + S_L_temp) / 2

            print("test phase - Epoch: [{}/{}]".format(i+1, self.graphconv_test_epoch))
        return S_0, S_L