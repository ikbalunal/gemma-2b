#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename: Service.py
# @Author: ikbal
# @Time: 6/1/2024 3:06 AM

import os
import torch
from flask import Flask, request, jsonify

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from Utils import get_config

app = Flask(__name__)

# def load_model(llm_configs,model_path=None, tokenizer_path=None):
#
#     if model_path and tokenizer_path:
#         model_id = model_path
#         tokenizer_id = tokenizer_path
#         print(f"Model Path : {model_id}")
#         print(f"Tokenizer Path : {tokenizer_id}")
#     else:
#         model_id = llm_configs['gemma_configs']['model_id']
#         tokenizer_id = llm_configs['gemma_configs']['model_id']
#         print(f"Model ID : {model_id}")
#         print(f"Tokenizer ID : {tokenizer_id}")
#
#     access_token = llm_configs['gemma_configs']['gemma_token']
#
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16
#     )
#     tokenizer_ = AutoTokenizer.from_pretrained(tokenizer_id, token=access_token)
#     model_ = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", token=access_token)
#     # model_.to('cuda')
#     return model_, tokenizer_

def load_model(llm_configs, model_path=None, tokenizer_path=None):
    if model_path and tokenizer_path:
        model_id = model_path
        tokenizer_id = tokenizer_path
        print(f"Model Path : {model_id}")
        print(f"Tokenizer Path : {tokenizer_id}")
    else:
        model_id = llm_configs['gemma_configs']['model_id']
        tokenizer_id = llm_configs['gemma_configs']['model_id']
        print(f"Model ID : {model_id}")
        print(f"Tokenizer ID : {tokenizer_id}")

    access_token = llm_configs['gemma_configs']['gemma_token']
    tokenizer_ = AutoTokenizer.from_pretrained(tokenizer_id, token=access_token)

    # Check if CUDA is available and set up quantization accordingly
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_ = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", token=access_token)
    else:
        model_ = AutoModelForCausalLM.from_pretrained(model_id, token=access_token)

    #model_.to('cpu')  # Force model to use CPU
    return model_, tokenizer_
def generate_text(input_text, max_new_tokens, temperature, top_k, top_p):
    if torch.cuda.is_available():
        inputs = tokenizer(input_text, return_tensors="pt").to('cuda')
        print("Using CUDA")
    else:
        inputs = tokenizer(input_text, return_tensors="pt").to('cpu')
        print("Using CPU")

    print("Generating text...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        do_sample=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/gemma-generate', methods=['POST'])
def generate():

    output = None
    data = request.get_json()

    text = data.get('text')
    max_new_tokens = data.get('max_new_tokens', 50)
    temperature = data.get('temperature', 0.7)
    top_k = data.get('top_k', 50)
    top_p = data.get('top_p', 0.92)

    print(f"Text: {text}", f"Max New Tokens: {max_new_tokens}", f"Temperature: {temperature}", f"Top K: {top_k}", f"Top P: {top_p}")

    output = generate_text(text, max_new_tokens, temperature, top_k, top_p)

    print(f"Output: {output}")
    return jsonify({"response": output})

if __name__ == "__main__":

    project_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    llm_config = get_config('Config/llmConfigs.json')
    service_config = get_config('Config/serviceConfigs.json')

    if llm_config['gemma_configs']['use_sft_model']:
        model_path = os.path.join(project_base_path, f"SFTTraining/Models/{llm_config['gemma_configs']['fine_tuned_model_id']}_merged")
        tokenizer_path = os.path.join(project_base_path, f"SFTTraining/{llm_config['gemma_configs']['fine_tuned_model_id']}")
        model, tokenizer = load_model(llm_config, model_path=model_path, tokenizer_path=tokenizer_path)

    else:
        model, tokenizer = load_model(llm_config)

    app.run(host=service_config['service']['gemma_service']['ip'], port=service_config['service']['gemma_service']['port'])
