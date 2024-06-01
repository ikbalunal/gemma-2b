#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename: test-service.py
# @Author: ikbal
# @Time: 6/1/2024 3:06 AM

import torch
from flask import Flask, request, jsonify

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from Utils import get_config

app = Flask(__name__)


@app.route('/gemma-generate', methods=['POST'])
def generate():

    data = request.get_json()

    model = data.get('model')
    text = data.get('text')
    max_new_tokens = data.get('max_new_tokens', 50)
    temperature = data.get('temperature', 0.7)
    top_k = data.get('top_k', 50)
    top_p = data.get('top_p', 0.92)

    output = f"Generated text: {text} with parameters: max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k}, top_p={top_p}, model={model}"

    return jsonify({"response": output})

if __name__ == "__main__":
    llm_config = get_config('Config/llmConfigs.json')
    service_config = get_config('Config/serviceConfigs.json')

    app.run(host=service_config['service']['gemma_service']['ip'], port=service_config['service']['gemma_service']['port'])
