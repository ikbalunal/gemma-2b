import torch
from flask import Flask, request, jsonify

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from LLMUtils import get_config

app = Flask(__name__)

def load_model(llm_configs):
    model_id = llm_configs['gemma_configs']['model_id']
    access_token = llm_configs['gemma_configs']['gemma_token']

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer_ = AutoTokenizer.from_pretrained(model_id, token=access_token)
    model_ = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", token=access_token)
    # model_.to('cuda')
    return model_, tokenizer_

def generate_text(input_text, max_new_tokens, temperature, top_k, top_p):
    inputs = tokenizer(input_text, return_tensors="pt").to('cuda')
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

    data = request.get_json()

    text = data.get('text')
    max_new_tokens = data.get('max_new_tokens', 50)
    temperature = data.get('temperature', 0.7)
    top_k = data.get('top_k', 50)
    top_p = data.get('top_p', 0.92)

    output = generate_text(text, max_new_tokens, temperature, top_k, top_p)

    return jsonify({"response": output})

if __name__ == "__main__":
    llm_config = get_config('Config/llmConfigs.json')
    service_config = get_config('Config/serviceConfigs.json')
    model, tokenizer = load_model(llm_config)
    app.run(host=service_config['service']['gemma_service']['ip'], port=service_config['service']['gemma_service']['port'])
