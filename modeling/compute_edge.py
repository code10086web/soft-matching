# encoding: utf-8

import torch
import numpy as np


def Normalize(data):
    mx = np.max(data, axis=1)
    mn = np.min(data, axis=1)
    for i in range(data.shape[0]):
        data[i] = (data[i] - mn[i]) / (mx[i] - mn[i])
    return data


def edge_pre(feat, k=20):
    all_num = feat.size(0)
    feat = torch.nn.functional.normalize(feat, dim=1, p=2)
    distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num) + \
              torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
    distmat.addmm_(1, -2, feat, feat.t())
    distmat[distmat < 0] = 0
    original_dist = distmat.cpu().detach().numpy()
    initial_rank = np.argsort(original_dist).astype(np.int32)

    k_reciprocal = []
    k_reciprocal_ = []
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        k_reciprocal_expansion_index_ = np.delete(k_reciprocal_expansion_index,
                                                  np.where(k_reciprocal_expansion_index == i))
        k_reciprocal.append(k_reciprocal_expansion_index)
        k_reciprocal_.append(k_reciprocal_expansion_index_)

    return original_dist, initial_rank, k_reciprocal, k_reciprocal_


def edge_generator(k_reciprocal, edge_weight, k):
    all_num = len(k_reciprocal)
    edge_index = []
    for i in range(all_num):
        if k < len(k_reciprocal[i]):
            weight = edge_weight[i, k_reciprocal[i]]
            value, index = torch.sort(weight, descending=True)
            k_reciprocal[i] = k_reciprocal[i][index.cpu().numpy()[:k]]
        [edge_index.append([i, j]) for j in k_reciprocal[i]]
    edge_index = torch.LongTensor(edge_index).squeeze().t()
    return edge_index


def edge_generator_no_weight(data, k):
    data = data.cpu().detach()
    data = torch.nn.functional.normalize(data, dim=1, p=2)
    m = data.shape[0]
    dist = torch.pow(data, 2).sum(dim=1, keepdim=True).expand(m, m) + \
               torch.pow(data, 2).sum(dim=1, keepdim=True).expand(m, m).t()
    dist = dist.addmm_(1, -2, data, data.t()).numpy()
    dist[dist < 0] = 0
    dist_rank = np.argsort(dist).astype(np.int32)
    dist_rank = dist_rank[:, 1:k + 1]
    edge_index = []
    for i in range(m):
        [edge_index.append([i, j]) for j in dist_rank[i, :]]
    edge_index = torch.LongTensor(edge_index).squeeze().t()
    return edge_index.cuda()


def edge_weight_generator(original_dist, initial_rank, k_reciprocal, k=6, lambda_value=0.3):
    all_num = len(k_reciprocal)
    V = np.zeros_like(original_dist).astype(np.float16)
    for i in range(all_num):
        weight = np.exp(-original_dist[i, k_reciprocal[i]])
        V[i, k_reciprocal[i]] = weight / np.sum(weight)
    if k != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k], :], axis=0)
        V = V_qe

    jaccard_sim = np.zeros_like(original_dist, dtype=np.float16)
    for i in range(all_num):
        temp_min = np.minimum(V[i, ], V)
        temp_max = np.maximum(V[i, ], V)
        jaccard_sim[i] = np.sum(temp_min, axis=1) / np.sum(temp_max, axis=1)

    original_dist = Normalize(original_dist)
    final_sim = jaccard_sim * (1 - lambda_value) + (1-original_dist) * lambda_value
    final_sim = Normalize(final_sim)
    return torch.from_numpy(final_sim)


def Compute_edge(x, k, k1=20, k2=6, lambda_value=0.3):
    original_dist, initial_rank, k_reciprocal, k_reciprocal_ = edge_pre(x, k1)
    edge_weight = edge_weight_generator(original_dist, initial_rank, k_reciprocal, k2, lambda_value)
    edge_index = edge_generator(k_reciprocal_, edge_weight, k)
    return edge_index.cuda(), edge_weight.cuda()


def Compute_edge_no_weight(x, k):
    edge_index = edge_generator_no_weight(x, k)
    return edge_index