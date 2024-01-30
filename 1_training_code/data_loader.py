#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import json
import codecs
import torch
from loguru import logger


def load_test_data(path):
    """Load test data
    Each row contains one or more columns, delimiter is Tab
    :param path: data path
    :return data_source: data list
    """
    data_source = list()
    try:
        fin = codecs.open(path, "r", encoding="utf-8")
        for line in fin:
            try:
                line = line.strip("\n\r")
                if not line:
                    continue
                items = line.split("\t")
                data_source.append(items)
            except Exception as e:
                logger.error("load_test_data read line error, msg={}".format(str(e)))
                continue
        fin.close()
    except Exception as e:
        logger.error("load_test_data file={} read error, msg={}".format(path, str(e)))
    return data_source


def load_train_data(path):
    """Load train data
    Each line has a JSON, and the data example can be found in the data directory
    which include query, pos, pos_feature, neg, neg_feature field

    :param path: data path
    :return data_source: data list
    """
    data_source = list()
    try:
        fin = codecs.open(path, "r", encoding="utf-8")
        for line in fin:
            try:
                line = line.strip("\n\r")
                if not line:
                    continue
                data = json.loads(line)
                is_pass, sample = check_train_data(data)
                if not is_pass:
                    continue
                data_source.append(sample)
            except Exception as e:
                logger.error("load_train_data read line error, {}".format(str(e)))
                continue
        fin.close()
    except Exception as e:
        logger.error("load_train_data file={} read error, msg={}".format(path, str(e)))
    return data_source


def check_train_data(data_item):
    """Training data check.

    :param data_item: item
    :return is_pass
    :return query
    :return pos_event
    :return neg_event
    :return key_info_pos
    :return key_info_neg
    """
    is_pass = False
    query = data_item.get("query", "")
    pos_event = data_item.get("pos", "")
    neg_event = data_item.get("neg", "")
    key_info_pos = data_item.get("pos_feature", {})
    key_info_neg = data_item.get("neg_feature", {})
    # SUB, OBJ, TRG
    pos_sub = key_info_pos.get("SUB", "")
    neg_sub = key_info_neg.get("SUB", "")
    if len(query) == 0 or len(pos_event) == 0 or len(neg_event) == 0 or len(pos_sub) == 0 or len(neg_sub) == 0:
        return is_pass, ("", "", "", "", "")
    key_info_pos_str = pos_sub + key_info_pos.get("TRG", "") + key_info_pos.get("OBJ", "")
    key_info_neg_str = neg_sub + key_info_neg.get("TRG", "") + key_info_neg.get("OBJ", "")
    is_pass = True
    return is_pass, (query, pos_event, neg_event, key_info_pos_str, key_info_neg_str)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        return text


class TrainCollateFunc(object):
    """Training data preprocessing"""

    def __init__(self, tokenizer, max_len=256, gen_max_len=20):
        self.__max_len = max_len
        self.__gen_max_len = gen_max_len
        self.__tokenizer = tokenizer

    def __call__(self, batch_data):
        """Training data preprocessing. Construct BERT input samples for
            Q, positive examples, negative examples, and generating tasks

        :param batch_data
        :return batch_q_tokens
        :return batch_pos_tokens
        :return batch_neg_tokens
        :return (batch_pos_gen_input_tokens, batch_pos_gen_label_tokens)
        :return (batch_neg_gen_input_tokens, batch_neg_gen_label_tokens)
        """

        batch_q_text = []
        batch_pos_text = []
        batch_neg_text = []
        batch_pos_gen_input = []
        batch_pos_gen_label = []
        batch_neg_gen_label = []
        batch_neg_gen_input = []
        for sample in batch_data:
            batch_q_text.append(sample[0].lower())
            batch_pos_text.append(sample[1].lower())
            batch_neg_text.append(sample[2].lower())
            batch_pos_gen_input.append(sample[1].lower())
            batch_pos_gen_label.append(sample[3].lower())
            batch_neg_gen_input.append(sample[2].lower())
            batch_neg_gen_label.append(sample[4].lower())
        batch_q_tokens = self.__tokenizer(
            batch_q_text, max_length=self.__max_len, truncation=True, padding="max_length", return_tensors="pt"
        )
        batch_pos_tokens = self.__tokenizer(
            batch_pos_text, max_length=self.__max_len, truncation=True, padding="max_length", return_tensors="pt"
        )
        batch_neg_tokens = self.__tokenizer(
            batch_neg_text, max_length=self.__max_len, truncation=True, padding="max_length", return_tensors="pt"
        )
        batch_pos_gen_input_tokens = self.__tokenizer(
            batch_pos_gen_input, max_length=self.__max_len, truncation=True, padding="max_length", return_tensors="pt"
        )
        batch_pos_gen_label_tokens = self.__tokenizer(
            batch_pos_gen_label,
            max_length=self.__gen_max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        batch_neg_gen_input_tokens = self.__tokenizer(
            batch_neg_gen_input, max_length=self.__max_len, truncation=True, padding="max_length", return_tensors="pt"
        )
        batch_neg_gen_label_tokens = self.__tokenizer(
            batch_neg_gen_label,
            max_length=self.__gen_max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return (
            batch_q_tokens,
            batch_pos_tokens,
            batch_neg_tokens,
            (batch_pos_gen_input_tokens, batch_pos_gen_label_tokens),
            (batch_neg_gen_input_tokens, batch_neg_gen_label_tokens),
        )


class TestCollateFunc(object):
    def __init__(self, tokenizer, max_len):

        self.__max_len = max_len
        self.__tokenizer = tokenizer

    def __call__(self, batch_text):
        """Test data preprocessing

        :param batch_text

        :return raw_line
        :return batch_texta_tokens
        :return batch_textb_tokens: The tokens of the second column of data,
            if the original data has only one column, it is None
        """
        batch_texta_text = []
        batch_textb_text = []
        raw_line = []
        for batch_item in batch_text:
            raw_line.append(batch_item)
            batch_texta_text.append(batch_item[0].lower())
            if len(batch_item) > 1:
                batch_textb_text.append(batch_item[1].lower())
        if len(batch_textb_text) > 0:
            assert len(batch_texta_text) == len(batch_textb_text)
        batch_texta_tokens = self.__tokenizer(
            batch_texta_text, max_length=self.__max_len, truncation=True, padding="max_length", return_tensors="pt"
        )
        batch_textb_tokens = None
        if len(batch_textb_text) > 0:
            batch_textb_tokens = self.__tokenizer(
                batch_textb_text, max_length=self.__max_len, truncation=True, padding="max_length", return_tensors="pt"
            )
        return raw_line, batch_texta_tokens, batch_textb_tokens


def get_bert_input(tokens, device):
    """Split the segmentation results into model inputs

    :param tokens
    :param device

    :return input_ids
    :return attention_mask
    :return token_type_ids
    """
    input_ids = tokens.get("input_ids").to(device)
    attention_mask = tokens.get("attention_mask").to(device)
    token_type_ids = tokens.get("token_type_ids").to(device)
    return input_ids, attention_mask, token_type_ids


def get_bert_input_with_prompt(tokens, prompt_tokens, device):
    """Split the segmentation results into model inputs

    :param tokens
    :param prompt_tokens
    :param device

    :return input_ids
    :return attention_mask
    :return token_type_ids
    """
    input_ids = tokens.get("input_ids").to(device)
    attention_mask = tokens.get("attention_mask").to(device)
    token_type_ids = tokens.get("token_type_ids").to(device)
    batch_size = input_ids.shape[0]
    batched_prefix_tokens = prompt_tokens.repeat(batch_size, 1)
    # template remove sep
    batched_prefix_tokens = batched_prefix_tokens[:, :-1]
    # sentence remove cls
    prompted_input_ids = input_ids[:, 1::]
    prompted_attention_mask = attention_mask[:, 1::]
    prompted_token_type_ids = token_type_ids[:, 1::]
    prompted_input_ids = torch.cat((batched_prefix_tokens, prompted_input_ids), 1)
    prompted_attention_mask = torch.cat(
        (torch.zeros(batched_prefix_tokens.shape, dtype=torch.long).to(device), prompted_attention_mask), 1
    )
    prompted_token_type_ids = torch.cat(
        (torch.zeros(batched_prefix_tokens.shape, dtype=torch.long).to(device), prompted_token_type_ids), 1
    )
    # CLS
    prompted_attention_mask[:, 0] = 1
    return prompted_input_ids, prompted_attention_mask, prompted_token_type_ids


if __name__ == "__main__":
    pass
