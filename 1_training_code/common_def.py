#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch

DATALOADER_WORKERS_NUM = 12
BERT_VOCAB_SIZE: int = 21128
BERT_PAD_TOKEN_ID: int = 0
BERT_UNK_TOKEN_ID: int = 100
BERT_CLS_TOKEN_ID: int = 101
BERT_SEP_TOKEN_ID: int = 102
BERT_MASK_TOKEN_ID: int = 103
BERT_TOKEN_EMBEDDING_SIZE: int = 768

POOLING_METHOD_CLS: str = "cls"
POOLING_METHOD_AVG: str = "avg"
POOLING_METHOD_MAX: str = "max"

EVENT_PROMPT: list = ["句", "子", "的", "主", "体", "是", "[MASK]", "动", "词", "是", "[MASK]", "客", "体", "是", "[MASK]"]

MODEL_BASE_WEIGHTS_NAME: str = "encoder_weights.bin"
MODEL_MULTITASK_WEIGHTS_NAME: str = "encoder_decoder_weights.bin"

INFER_TASK_TYPE_SIM: str = "sim"
INFER_TASK_TYPE_EMB: str = "emb"

CPU_DEVICE_NAME: str = "cpu"
GPU_DEVICE_NAME: str = "cuda"


def get_device(device_name):
    """获取设备信息CPU或GPU
    :param device_name: cpu或gpu
    """
    device = None
    if device_name.lower() == CPU_DEVICE_NAME or not torch.cuda.is_available():
        device = torch.device(CPU_DEVICE_NAME)
    else:
        device = torch.device(GPU_DEVICE_NAME)
    return device


if __name__ == "__main__":
    pass
