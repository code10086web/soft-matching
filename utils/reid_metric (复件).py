# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking
from modeling.compute_edge import Compute_edge, Compute_edge_no_weight

class R1_mAP(Metric):
    def __init__(self, num_query, model, model_select, k, edge_weight_flag, device, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.model = model
        self.model_select = model_select
        self.k = k
        self.edge_weight_flag = edge_weight_flag
        self.device = device

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes' and self.model_select == 'baseline':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        if not self.model_select == 'baseline':
            self.model.eval()
            with torch.no_grad():
                if self.edge_weight_flag:
                    edge_index_q, edge_weight_q = Compute_edge(qf, self.k, k1=20, k2=6, lambda_value=0.3)
                    edge_index_q = edge_index_q.to(self.device) if torch.cuda.device_count() >= 1 else edge_index_q
                    edge_weight_q = edge_weight_q.to(self.device) if torch.cuda.device_count() >= 1 else edge_weight_q
                    qf = self.model(qf, edge_index_q, edge_weight_q)
                else:
                    edge_index_q = Compute_edge_no_weight(qf, self.k)
                    edge_index_q = edge_index_q.to(self.device) if torch.cuda.device_count() >= 1 else edge_index_q
                    qf = self.model(qf, edge_index_q)

                S0, SL = [], []
                g_pids_, g_camids_ = [], []
                for i in np.unique(g_camids):
                    fi = np.where(g_camids == i)[0]
                    gf_ = gf[fi, :]
                    if self.edge_weight_flag:
                        edge_index_g, edge_weight_g = Compute_edge(gf_, self.k, k1=20, k2=6, lambda_value=0.3)
                        edge_index_g = edge_index_g.to(self.device) if torch.cuda.device_count() >= 1 else edge_index_g
                        edge_weight_g = edge_weight_g.to(self.device) if torch.cuda.device_count() >= 1 else edge_weight_g
                        gf = self.model(gf_, edge_index_g, edge_weight_g)
                    else:
                        edge_index_g = Compute_edge_no_weight(gf_, self.k)
                        edge_index_g = edge_index_g.to(self.device) if torch.cuda.device_count() >= 1 else edge_index_g
                        gf = self.model(gf_, edge_index_g)

                    feat = [qf, gf]
                    edge_index = [edge_index_q, edge_index_g]
                    if self.edge_weight_flag:
                        edge_weight = [edge_weight_q, edge_weight_g]
                        S_0, S_L = self.model(feat, edge_index, edge_weight)
                    else:
                        S_0, S_L = self.model(feat, edge_index)
                    S0.append(S_0)
                    SL.append(S_L)
                    g_pids_.extend(g_pids[fi])
                    g_camids_.extend(g_camids[fi])

                S0, SL = torch.cat(S0, dim=1), torch.cat(SL, dim=1)
                g_pids_, g_camids_ = np.array(g_pids_), np.array(g_camids_)
                cmc[0], mAP[0] = eval_func(-S0, q_pids, g_pids_, q_camids, g_camids_)
                cmc[1], mAP[1] = eval_func(-SL, q_pids, g_pids_, q_camids, g_camids_)
        else:
            m, n = qf.shape[0], gf.shape[0]
            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(1, -2, qf, gf.t())
            distmat = distmat.cpu().numpy()
            cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP


class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP



