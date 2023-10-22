"""
Usage:
    train.py <config_yaml_path>

Options:
    <config_yaml_path>   The path to the configuration file. It should be a YAML-formatted file that contains various
                         configuration settings needed for the execution.
"""

import docopt
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from slugify import slugify
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from peft import prepare_model_for_kbit_training, LoraConfig
from peft import get_peft_model
import transformers
from datetime import datetime

config_path = docopt.docopt(__doc__).get("<config_yaml_path>")
config = yaml.safe_load(Path(config_path).read_text())
accelerator = Accelerator()

train_dataset = load_dataset(
    "json", data_files=config["training_dataset_jsonl_path"], split="train"
)

eval_dataset = load_dataset(
    "json", data_files=config["eval_dataset_jsonl_path"], split="train"
)

model = AutoModelForCausalLM.from_pretrained(
    config["base_model_id"],
    quantization_config=BitsAndBytesConfig(**config["bnb_config"]),
)

tokenizer = AutoTokenizer.from_pretrained(
    config["base_model_id"],
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token


def created_tokenized_prompt(input_pair):
    result = tokenizer(
        config["prompt_template"].format(**input_pair),
        truncation=True,
        max_length=config["tokenizer_max_length"],
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


tokenized_train_dataset = train_dataset.map(
    created_tokenized_prompt, num_proc=accelerator.num_processes
)
tokenized_val_dataset = eval_dataset.map(
    created_tokenized_prompt, num_proc=accelerator.num_processes
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(**config["lora_config"]))
model.to(accelerator.device)

if torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=config["output_dir"],
        bf16=True,
        ddp_find_unused_parameters=False,
        run_name=f"{slugify(config['output_dir'])}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        dataloader_num_workers=accelerator.num_processes,
        **config["training"],
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer,
                                                               mlm=False),
)

model.config.use_cache = False
trainer.train()
