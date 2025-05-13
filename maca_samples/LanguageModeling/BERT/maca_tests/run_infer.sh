#!/usr/bin/env bash

python ../inference_profile.py --bert_model bert-large-uncased --init_checkpoint /home/sw/pytorch/yfeng/bert_large_qa.pt \
    --vocab_file /home/sw/pytorch/yfeng/vocab.txt \
    --config_file /home/sw/pytorch/yfeng/LanguageModeling/BERT/bert_configs/large.json \
    --question="what food does harry like?" --context="i love apples"
