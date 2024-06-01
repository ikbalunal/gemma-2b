#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename: Utils.py
# @Author: ikbal
# @Time: 6/1/2024 11:41 AM

import torch
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

def get_config(conf_path):

    with open(conf_path) as config_file:
        config_ = json.load(config_file)

    return config_

def get_bnb_config():

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    return bnb_config

def get_tokenizer(model_id, access_token=None):

    tokenizer_ = AutoTokenizer.from_pretrained(model_id, token=access_token)

    return tokenizer_

def get_model(model_id, bnb_config, device_map="auto", access_token=None):

    model_ = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map=device_map, token=access_token)

    return model_

def get_output_dir(model, inputs, max_new_tokens, tokenizer, temperature=0.7, top_k=50, top_p=0.95):

    outputs_ = model.generate(
        **inputs,
        # max_length=inputs["input_ids"].shape[-1] + max_new_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True
    )

    return tokenizer.decode(outputs_[0], skip_special_tokens=True)
