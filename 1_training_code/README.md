## Background

This is the code for "Event-enhanced Retrieval in Real-time Search". The proposed solution is an event-enhanced retrieval (EER) method that incorporates hard negative mining techniques, supervised contrastive learning, and pairwise learning to improve encoder performance. 

## Sample_data
There are 1k raw `<q,title>` pairs in folder `0_sample_data`.

## File Description
```
├── pretrain_model             Pretrain model
├── common_def.py              Constant Definition
├── data_loader.py             Training and prediction data loading and preprocessing
├── distance.py                Vector distance calculation
├── loss.py                    Definition of loss function
├── m_base_train.py            Baseline model training
├── m_base_infer.py            Baseline model infer
├── m_event-enhanced_train.py  Event-enhanced model training
├── m_event-enhanced_infer.py  Event-enhanced model infer
├── modeling.py                Model definition
├── README.md
└── requirements.txt           Dependency library
```

## Training steps

```
pip install -r requirements.txt 
```


### Event-enhanced model

Training：

```
python event_link_multitask_train.py \
--device "cuda" \
--gpu_num 8 \
--epoch_num 1 \
--model_weight_dir "./model_save/multitask_weights" \
--lr 0.00005 \
--emb_dim 64 \
--batch_size 32 \
--max_length 40 \
--gen_max_length 30 \
--train_data_path ./data_example/train_data.txt \
--pretrain_model_path ./pretrain_model \
--pooling_method avg \
--cons_scale 20 \
--kl_scale 20 \
--margin 0.1
```

Infer：

```
test_file=$1
python event_link_multitask_infer.py \
--device "cuda:0" \
--batch_size 64 \
--emb_dim 64 \
--max_length 40 \
--pretrain_model_path ./pretrain_model \
--model_weight_path model_save/multitask_weights/checkpoint-0/encoder_decoder_weights.bin \
--test_data_path ./data_example/test.txt \
--infer_task sim \
--output_postfix "multitask_pred"
```

### Base model

Training：

```
python event_link_base_train.py \
--device "cuda" \
--gpu_num 8 \
--epoch_num 1 \
--model_weight_dir "./model_save/base_weights" \
--lr 0.00005 \
--emb_dim 64 \
--batch_size 64 \
--max_length 40 \
--train_data_path ./data_example/train.txt \
--pretrain_model_path ./pretrain_model \
--pooling_method avg \
--cons_scale 20 \
--margin 0.1
```

Infer：

```
python event_link_base_infer.py \
--device "cuda:0" \
--batch_size 64 \
--emb_dim 64 \
--max_length 40 \
--pretrain_model_path ./pretrain_model \
--model_weight_path model_save/base_weights/checkpoint-0/encoder_weights.bin \
--test_data_path ./data_example/test.txt \
--infer_task sim \
--output_postfix "base_pred"
```

