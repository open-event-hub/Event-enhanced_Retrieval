#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch


def cos_sim_pointwise(a_emb: torch.Tensor, b_emb: torch.Tensor, is_normed: bool = False):
    """Calculate cos_sim pointwise
    sim[i] = cos_sim(a_emb[i], b_emb[i])

    :param a_emb
    :param b_emb
    :param is_normed

    :return sim: Similarity batch
    """
    if len(a_emb.shape) == 1:
        a_emb = a_emb.unsqueeze(0)

    if len(b_emb.shape) == 1:
        b_emb = b_emb.unsqueeze(0)

    if not is_normed:
        # Perform L2 regularization on vectors
        a_emb = torch.nn.functional.normalize(a_emb, p=2, dim=1)
        b_emb = torch.nn.functional.normalize(b_emb, p=2, dim=1)
    sim = torch.sum(a_emb * b_emb, 1)
    return sim


def cos_sim_listwise(a_emb: torch.Tensor, b_emb: torch.Tensor, is_normed: bool = False):
    """Calculate cos_sim listwise
    sim[i][j] = cos_sim(a[i], b[j]) for all i, j

    :param a_emb
    :param b_emb
    :param is_normed

    :return sim: Similarity matrix
    """
    if len(a_emb.shape) == 1:
        a_emb = a_emb.unsqueeze(0)
    if len(b_emb.shape) == 1:
        b_emb = b_emb.unsqueeze(0)
    if not is_normed:
        a_emb = torch.nn.functional.normalize(a_emb, p=2, dim=1)
        b_emb = torch.nn.functional.normalize(b_emb, p=2, dim=1)
    sim = torch.mm(a_emb, b_emb.transpose(0, 1))
    return sim
