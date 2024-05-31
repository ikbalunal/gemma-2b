import os
import transformers
import torch
from datasets import load_dataset
from huggingface_hub import notebook_login


from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, GemmaTokenizer



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

# def get_output_dir(model, inputs, max_new_tokens, tokenizer):
#
#     outputs_ = model.generate(**inputs, max_new_tokens=max_new_tokens)
#
#     return tokenizer.decode(outputs_[0], skip_special_tokens=True)

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

if __name__ == "__main__":

    access_token_ = "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"

    model_id = "google/gemma-2b"
    tokenizer = get_tokenizer(model_id, access_token_)
    model = get_model(model_id, get_bnb_config(), access_token=access_token_)

    inputs = tokenizer("What is Python?", return_tensors="pt").to('cuda')
    #//// output = get_output_dir(model, inputs, 20, tokenizer)
    output = get_output_dir(model, inputs, 2048 , tokenizer, temperature=0.9, top_k=50, top_p=0.92)