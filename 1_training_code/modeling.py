#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import common_def
import torch
import transformers


class LinearHead(torch.nn.Module):
    """Using a one layer linear fully connected network to reduce
        the dimensionality of the output vector of the BERT model"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.dense = torch.nn.Linear(input_dim, output_dim)

    def forward(self, input_embedding: torch.Tensor):

        out = self.dense(input_embedding)
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        return out


class SentenceSimModel(torch.nn.Module):

    def __init__(self, pretrained_model: str, pooling_method: str, emb_dim: int):
        super(SentenceSimModel, self).__init__()
        config = transformers.AutoConfig.from_pretrained(pretrained_model)
        self.__model = transformers.AutoModel.from_pretrained(pretrained_model, config=config)
        self.__encoder_linear_head = LinearHead(common_def.BERT_TOKEN_EMBEDDING_SIZE, emb_dim)
        self.pooling_method = pooling_method

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor,
    ):
        encoder_hidden = self.__model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True
        )
        encoder_pooled = hidden_state_pooling(encoder_hidden.last_hidden_state, attention_mask, self.pooling_method)
        encoder_emb = self.__encoder_linear_head(encoder_pooled)
        return encoder_emb


class SentenceSimSeq2SeqModel(torch.nn.Module):
    """A multi task learning model that includes QT semantic supervised pairwise,
        unsupervised contrastive learning, and event generation
    """

    def __init__(self, pretrained_model: str, pooling_method: str, emb_dim: int):
        super(SentenceSimSeq2SeqModel, self).__init__()
        self.__model = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(
            pretrained_model, pretrained_model
        )
        self.__model.config.pad_token_id = common_def.BERT_PAD_TOKEN_ID
        self.__model.config.encoder.bos_token_id = common_def.BERT_CLS_TOKEN_ID
        self.__model.config.encoder.eos_token_id = common_def.BERT_SEP_TOKEN_ID
        self.__model.config.decoder.bos_token_id = common_def.BERT_CLS_TOKEN_ID
        self.__model.config.decoder.eos_token_id = common_def.BERT_SEP_TOKEN_ID
        self.__model.config.decoder_start_token_id = common_def.BERT_CLS_TOKEN_ID
        self.__model.config.decoder_end_token_id = common_def.BERT_SEP_TOKEN_ID
        self.__model.config.output_hidden_states = True
        self.__model.config.encoder.output_hidden_states = True
        self.__model.config.decoder.output_hidden_states = True
        self.__encoder_linear_head = LinearHead(common_def.BERT_TOKEN_EMBEDDING_SIZE, emb_dim)
        self.__decoder_linear_head = LinearHead(common_def.BERT_TOKEN_EMBEDDING_SIZE, emb_dim)

        self.pooling_method = pooling_method

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        gen_label_input_ids: torch.Tensor,
        gen_attention_mask: torch.Tensor,
    ):

        encoder_hidden = self.__model.encoder(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
        )
        encoder_pooled = self.__hidden_state_pooling(encoder_hidden.last_hidden_state, attention_mask)
        encoder_emb = self.__encoder_linear_head(encoder_pooled)
        decoder_emb = None
        decoder_logits = None
        if gen_label_input_ids is not None and gen_attention_mask is not None:
            seq2seq_outputs = self.__model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=gen_label_input_ids,
                output_hidden_states=True,
            )
            decoder_logits = seq2seq_outputs.logits
            decoder_last_hidden_state = seq2seq_outputs.decoder_hidden_states[-1]
            decoder_pooled = hidden_state_pooling(decoder_last_hidden_state, gen_attention_mask, self.pooling_method)
            decoder_emb = self.__decoder_linear_head(decoder_pooled)
        return encoder_emb, decoder_emb, decoder_logits


def hidden_state_pooling(hidden_state: torch.Tensor, attention_mask: torch.Tensor, pooling_method: str):

    pad_attention_mask = torch.unsqueeze(attention_mask, dim=-1)
    pooled_emb = None

    if pooling_method == common_def.POOLING_METHOD_AVG:
        hidden_state_wo_mask = hidden_state * pad_attention_mask
        sequence_length = torch.sum(pad_attention_mask, dim=1)
        pooled_emb = torch.sum(hidden_state_wo_mask, dim=1) / sequence_length
    elif pooling_method == common_def.POOLING_METHOD_MAX:
        hidden_state_wo_mask = hidden_state * pad_attention_mask
        pooled_emb = torch.max(hidden_state_wo_mask, dim=1)
    else:
        pooled_emb = hidden_state[:, 0]
    return pooled_emb


if __name__ == "__main__":
    pass
