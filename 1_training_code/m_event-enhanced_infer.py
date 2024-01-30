#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import codecs

import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

import common_def
from args_parser import parse_args
from data_loader import Dataset
from data_loader import get_bert_input
from data_loader import load_test_data
from data_loader import TestCollateFunc
from modeling import SentenceSimSeq2SeqModel


def do_infer(args, model, dataloader, device, output_path):
    fout = codecs.open(output_path, "w", encoding="utf8")
    model.eval()
    with torch.no_grad():
        for __, (raw_line, batch_texta_source, batch_textb_source) in enumerate(tqdm(dataloader), start=1):
            ret_list = []
            if args.infer_task == common_def.INFER_TASK_TYPE_EMB:
                input_ids_texta, attention_mask_texta, token_type_ids_texta = get_bert_input(batch_texta_source, device)
                texta_emb, __, __ = model(
                    input_ids=input_ids_texta,
                    attention_mask=attention_mask_texta,
                    token_type_ids=token_type_ids_texta,
                    gen_label_input_ids=None,
                    gen_attention_mask=None,
                )
                ret_list = texta_emb.tolist()
            elif batch_textb_source is not None and args.infer_task == common_def.INFER_TASK_TYPE_SIM:
                input_ids_texta, attention_mask_texta, token_type_ids_texta = get_bert_input(batch_texta_source, device)
                input_ids_textb, attention_mask_textb, token_type_ids_textb = get_bert_input(batch_textb_source, device)
                texta_emb, __, __ = model(
                    input_ids=input_ids_texta,
                    attention_mask=attention_mask_texta,
                    token_type_ids=token_type_ids_texta,
                    gen_label_input_ids=None,
                    gen_attention_mask=None,
                )
                textb_emb, __, __ = model(
                    input_ids=input_ids_textb,
                    attention_mask=attention_mask_textb,
                    token_type_ids=token_type_ids_textb,
                    gen_label_input_ids=None,
                    gen_attention_mask=None,
                )
                ret_tensor = torch.nn.functional.cosine_similarity(texta_emb, textb_emb, dim=-1)
                ret_list = ret_tensor.tolist()
            else:
                logger.error("test data and task type are not match")
                raise ValueError("test data and task type are not match")
            for idx in range(len(ret_list)):
                output = ["\t".join(raw_line[idx]), str(ret_list[idx])]
                fout.write("\t".join(output) + "\n")
    fout.close()


def main():
    """主函数"""
    args = parse_args()
    logger.add("./log/log.event_link_multitask_infer")
    logger.info(args)
    device = common_def.get_device(args.device)
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)
    test_data_source = load_test_data(args.test_data_path)
    test_dataset = Dataset(test_data_source)
    test_call_func = TestCollateFunc(tokenizer, max_len=args.max_length)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=common_def.DATALOADER_WORKERS_NUM,
        collate_fn=test_call_func,
    )
    output_path = args.test_data_path + "." + args.output_postfix
    model = SentenceSimSeq2SeqModel(
        args.pretrain_model_path, emb_dim=args.emb_dim, pooling_method=args.pooling_method
    ).to(device)
    state_dict = torch.load(args.model_weight_path)
    model.load_state_dict(state_dict)
    do_infer(args, model, test_dataloader, device, output_path)


if __name__ == "__main__":
    main()
