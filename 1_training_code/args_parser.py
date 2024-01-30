#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse

import common_def


def parse_args():
    """Parse input configuration parameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="gpu or cpu")
    parser.add_argument("--emb_dim", type=int, default=64, help="embedding dim")
    parser.add_argument("--pretrain_model_path", type=str, required=True, help="pretrain model path")
    parser.add_argument("--max_length", type=int, default=50, help="max length of input sentences")
    parser.add_argument(
        "--pooling_method",
        type=str,
        choices=[common_def.POOLING_METHOD_AVG, common_def.POOLING_METHOD_CLS, common_def.POOLING_METHOD_MAX],
        default=common_def.POOLING_METHOD_AVG,
    )
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--epoch_num", type=int, default=1)
    parser.add_argument("--train_data_path", type=str, default="./data_expamle/train.txt")
    parser.add_argument("--model_weight_dir", type=str, default="./model_save")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--gen_max_length", type=int, default=30)
    parser.add_argument("--cons_scale", type=float, default=20.0)
    parser.add_argument("--kl_scale", type=float, default=20.0)
    parser.add_argument("--margin", type=float, default=1.0)

    parser.add_argument("--model_weight_path", type=str, default="./model_save/checkpoint-0/encoder_weights.bin")
    parser.add_argument("--test_data_path", type=str, default="./data_expamle/test.txt")
    parser.add_argument(
        "--infer_task",
        type=str,
        default=common_def.INFER_TASK_TYPE_EMB,
        choices=[common_def.INFER_TASK_TYPE_EMB, common_def.INFER_TASK_TYPE_SIM],
    )
    parser.add_argument("--output_postfix", type=str, default="mutitask_pred", help="add postfix to output file name")
    user_args = parser.parse_args()
    return user_args


if __name__ == "__main__":
    args = parse_args()
    print(args)
