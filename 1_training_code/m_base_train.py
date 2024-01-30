#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os

import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

import common_def
from args_parser import parse_args
from data_loader import Dataset
from data_loader import get_bert_input
from data_loader import load_train_data
from data_loader import TrainCollateFunc
from loss import TrainingLoss
from modeling import SentenceSimModel


def get_batch_encoder_output(batch_tokens, model, device):
    """Get model encoder output"""
    input_ids, attention_mask, token_type_ids = get_bert_input(batch_tokens, device)
    batch_encoder_emb = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,)
    return batch_encoder_emb


def do_train(args, model, train_data_loader, optimizer, loss_obj, device):

    # Convert prompt text to token_id
    if args.gpu_num > 1:
        model = torch.nn.DataParallel(model)
    model.zero_grad()
    model.train()
    for epoch in range(args.epoch_num):
        for __, (batch_query_tokens, batch_pos_tokens, batch_neg_tokens, __, __) in enumerate(
            tqdm(train_data_loader), start=1
        ):
            optimizer.zero_grad()
            query_encoder_emb = get_batch_encoder_output(batch_query_tokens, model, device)
            pos_encoder_emb = get_batch_encoder_output(batch_pos_tokens, model, device)
            neg_encoder_emb = get_batch_encoder_output(batch_neg_tokens, model, device)

            # Similarity-Calculating Contrastive Learning Loss
            cse_loss = loss_obj.constrastive_loss(
                query_encoder_emb, pos_encoder_emb, neg_encoder_emb, scale=args.cons_scale
            )
            # Similarity-Calculatingmargin loss
            margin_loss = loss_obj.margin_loss(query_encoder_emb, pos_encoder_emb, neg_encoder_emb, margin=args.margin,)
            loss = cse_loss + margin_loss
            if args.gpu_num > 1:
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            logger.info("ep: {}, loss: {}, margin: {}, cse_loss: {}".format(epoch, loss, margin_loss, cse_loss))

        output_dir = os.path.join(args.model_weight_dir, "checkpoint-{}".format(epoch))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), os.path.join(output_dir, common_def.MODEL_BASE_WEIGHTS_NAME))
        else:
            torch.save(model.state_dict(), os.path.join(output_dir, common_def.MODEL_BASE_WEIGHTS_NAME))
        logger.info("save model: {}".format(output_dir))


def main():
    args = parse_args()
    logger.add("./log/log.base_train")
    logger.info(args)
    device = common_def.get_device(args.device)
    train_batch_size = int(args.batch_size) * args.gpu_num
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)
    train_data_source = load_train_data(args.train_data_path)
    train_dataset = Dataset(train_data_source)
    train_call_func = TrainCollateFunc(tokenizer, max_len=args.max_length)
    num_workers = common_def.DATALOADER_WORKERS_NUM
    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, collate_fn=train_call_func
    )
    model = SentenceSimModel(args.pretrain_model_path, pooling_method=args.pooling_method, emb_dim=args.emb_dim).to(
        device
    )
    loss_obj = TrainingLoss(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    do_train(args, model, train_dataloader, optimizer, loss_obj, device)


if __name__ == "__main__":
    main()
