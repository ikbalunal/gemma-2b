#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename: Trainer.py
# @Author: ikbal
# @Time: 6/1/2024 1:17 PM

import os
import json
import wandb
import torch

from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, BitsAndBytesConfig, TrainingArguments
from huggingface_hub import login

def get_config(conf_path):

    with open(conf_path) as config_file:
        jsonconfig_ = json.load(config_file)

    return jsonconfig_

class WandbCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            wandb.log(logs)
class GetArguments:
    def __init__(self, model_id, bnb_config, device_map, access_token):
        self.model_id = model_id
        self.bnb_config = bnb_config
        self.device_map = device_map
        self.access_token = access_token

    def get_model(self):
        pretrained_model_ = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=self.bnb_config, device_map=self.device_map, token=self.access_token)
        return pretrained_model_

    def get_tokenizer(self):
        pretrained_tokenizer_ = AutoTokenizer.from_pretrained(self.model_id, token=self.access_token)
        return pretrained_tokenizer_

    @staticmethod
    def get_bnb_config():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        return bnb_config

    @staticmethod
    def get_lora_config():
        lora_config = LoraConfig(
            r=8,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        return lora_config

class Dataset:
    def __init__(self, dataset_name, tokenizer):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer

    def get_data(self):
        data = load_dataset(self.dataset_name)
        return data.map(lambda samples: self.tokenizer(samples["question"], samples["context"]), batched=True)

    @staticmethod
    def formatting_func(example):
        text = f"Question: {example['question'][0]}\nContext: {example['context'][0]}\nAnswer: {example['answer'][0]}"
        return [text]

class SFTTrain:
    def __init__(self, model, tokenizer, data, lora_config,format_func):
        self.model = model
        self.tokenizer = tokenizer
        self.data = data
        self.lora_config = lora_config
        self.formatting_func = format_func

    def get_sft_trainer(self):
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.data["train"],
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=8,
                warmup_steps=2,
                max_steps=75,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=1,
                output_dir="Outputs",
                optim="paged_adamw_8bit",
                report_to="wandb",
            ),
            peft_config=self.lora_config,
            formatting_func=self.formatting_func,
            callbacks=[WandbCallback()]
        )

        return trainer

    def train(self):
        trainer = self.get_sft_trainer()
        trainer.train()
        return trainer

class SaveAndMerge:
    def __init__(self, base_model, trainer, fine_tuned_model_id, model_id,access_token):
        self.trainer = trainer
        self.fine_tuned_model = fine_tuned_model_id
        self.model_id = model_id
        self.access_token = access_token
        self.base_model = base_model

    def merge_model(self):

        #base_model_ = AutoModelForCausalLM.from_pretrained(
        #    self.model_id,
        #    low_cpu_mem_usage=True,
        #    return_dict=True,
        #    torch_dtype=torch.float16,
        #    device_map="auto",
        #    token=self.access_token
        #)

        # Merge the fine-tuned model with LoRA adaption along with the base model.
        fine_tuned_merged_model = PeftModel.from_pretrained(self.base_model, f"Models/{self.fine_tuned_model}_unmerged")

        return fine_tuned_merged_model.merge_and_unload()
    def save_ft_unmerged_model(self):
        self.trainer.model.save_pretrained(f"Models/{self.fine_tuned_model}_unmerged")
    def save_ft_merged_model(self):
        fine_tuned_merged_model = self.merge_model()
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        fine_tuned_merged_model.save_pretrained(f"Models/{self.fine_tuned_model}_merged", safe_serialization=True)
        tokenizer.save_pretrained(self.fine_tuned_model)
        tokenizer.padding_side = "right"


if __name__ == "__main__":

    if not os.path.exists("Outputs"):
        print("Creating Outputs directory...")
        os.makedirs("Outputs")

    if not os.path.exists("Models"):
        print("Creating Models directory...")
        os.makedirs("Models")

    print("Start training process...")
    config_ = get_config("TrainConfig/gemmaConfigs.json")
    model_id_ = config_["gemma_configs"]["model_id"]
    fine_tuned_model_id_ = config_["gemma_configs"]["fine_tuned_model_id"]
    wandb_api_key_ = config_["gemma_configs"]["wandb_api_key"]
    access_token_ = config_["gemma_configs"]["gemma_token"]

    print("Logging into huggingface...")
    login(token=access_token_)

    print("Initializing wandb...")
    wandb.login(key=wandb_api_key_)
    wandb.init(
        project="gemma_2b_sft",
        config={
            "learning_rate": 2e-4,
            "architecture": "Causal LM",
            "dataset": "sql-create-context",
            "epochs": 75,
        }
    )

    print("1- Getting the model and tokenizer...")
    bnb_config_ = GetArguments.get_bnb_config()
    lora_config_ = GetArguments.get_lora_config()

    print("1.1- Getting Model...")
    model_ = GetArguments(model_id_, bnb_config_, device_map="auto", access_token=access_token_).get_model()
    print("1.2- Getting Tokenizer...")
    tokenizer_ = GetArguments(model_id_, bnb_config_, device_map="auto", access_token= access_token_).get_tokenizer()

    print("2- Getting the dataset...")
    dataset_ = Dataset(dataset_name="b-mc2/sql-create-context", tokenizer=tokenizer_)
    data_ = dataset_.get_data()

    print("3- Training the model...")
    trainer_ = SFTTrain(model_, tokenizer_, data_, lora_config_, Dataset.formatting_func).train()

    print("4- Saving the fine-tuned model...")
    save_model = SaveAndMerge(base_model=model_,trainer=trainer_, fine_tuned_model_id=fine_tuned_model_id_, model_id=model_id_, access_token=access_token_)
    print("4.1- Saving the unmerged model...")
    save_model.save_ft_unmerged_model()
    print("4.2- Saving the merged model...")
    save_model.save_ft_merged_model()

    print("End of the training process.")