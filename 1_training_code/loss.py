#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import common_def
import distance
import torch


class TrainingLoss(torch.nn.Module):

    def __init__(self, device):

        super(TrainingLoss, self).__init__()
        self.__cross_entropy_loss = torch.nn.CrossEntropyLoss()
        # Seq2seq loss function, does not calculate loss for pads
        weight_mask = [1] * common_def.BERT_VOCAB_SIZE
        weight_mask[common_def.BERT_PAD_TOKEN_ID] = 0
        self.__seq2seq_loss = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(weight_mask).to(device))
        self.__kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    def generation_loss(
        self,
        pred_logits: torch.Tensor,
        label_input_ids: torch.Tensor,
        label_attention_mask: torch.Tensor,
    ):

        label_input_ids = label_input_ids.mul(label_attention_mask)
        masked_lm_loss = self.__seq2seq_loss(pred_logits.view(-1, common_def.BERT_VOCAB_SIZE), label_input_ids.view(-1))
        masked_lm_loss = torch.sum(masked_lm_loss)
        return masked_lm_loss

    def margin_loss(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor,
        margin: float,
    ):

        scores_pos = distance.cos_sim_pointwise(query_emb, positive_emb, is_normed=True)
        scores_neg = distance.cos_sim_pointwise(query_emb, negative_emb, is_normed=True)
        zeros = torch.zeros(scores_pos.shape, dtype=torch.float).to(query_emb.device)
        loss = torch.mean(torch.max(zeros, scores_neg - scores_pos + margin))
        return loss

    def constrastive_loss(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor = None,
        scale: float = 20.0,
    ):

        if negative_emb is not None:
            positive_emb = torch.cat([positive_emb, negative_emb], dim=0)

        scores = distance.cos_sim_listwise(query_emb, positive_emb, is_normed=True) * scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
        loss = self.__cross_entropy_loss(scores, labels)
        return loss

    def kl_loss(self, target_emb: torch.Tensor, pred_emb: torch.Tensor, scale: float = 20.0):

        scores = distance.cos_sim_listwise(target_emb, pred_emb, is_normed=True) * scale
        scores = torch.nn.functional.softmax(scores, dim=1)
        labels = torch.zeros(scores.shape, dtype=torch.float, device=scores.device)
        labels = labels.fill_diagonal_(1.0)
        loss = self.__kl_loss(scores, labels)
        return loss


if __name__ == "__main__":
    pass
