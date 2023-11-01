# South Park Fanatic Mistral-7B Fine-Tuning on Multiple GPUs
[corresponding blog post here](https://int8.io/train-your-own-south-park-fanatic-ai-with-mistral-7b/)


This repository contains scripts for
- generating data with GPT-4 via `rq` 
- Mistral-7B (and other LLMs) fine-tuning on multiple GPUs with `peft` for LoRA and `bitsandbytes` for quantization

## Instructions

### Installation

1. Install the project dependencies by running the following command:

```sh
pip install -r requirements.txt
```

2. To begin training, create configuration YAML file (
   below `example_config.yaml` from this repo)

```yaml
base_model_id: "mistralai/Mistral-7B-v0.1"
training_dataset_jsonl_path: /path/to/your/train_dataset.jsonl
eval_dataset_jsonl_path: /path/to/your/eval_dataset.jsonl
prompt_template: "### question: {question}\n ### answer: {answer} </s>"
tokenizer_max_length: 256
bnb_config:
  load_in_4bit: true
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: "nf4"
lora_config:
  r: 64
  target_modules: [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
  ]
  bias: none
  lora_dropout: 0.05
  task_type: CAUSAL_LM
output_dir: /path/to/your/output/model
training:
  optim: "paged_adamw_8bit"
  warmup_steps: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  max_steps: 500
  logging_steps: 25
  logging_dir: /path/to/logging/dir,  # Directory for storing logs
  save_strategy: "steps"
  save_steps: 100
  evaluation_strategy: "steps"
  eval_steps: 100
  do_eval: true
  learning_rate: 2.5e-5
```

and run

```shell 
accelerate config 
accelerate launch train.py config.yaml # example_config.yaml is a good start  
```

### Dataset Format

Input data should be in the `jsonl` format, with each line containing an `question`
and `answer` field pair. Here are few example lines from the dataset:

```
{"question": "What is the capital of France?", "answer": "The capital of France is Paris."}
{"question": "Who wrote 'Pride and Prejudice'?", "answer": "Pride and Prejudice was written by Jane Austen."}
{"question": "When was the Declaration of Independence signed?", "answer": "The Declaration of Independence was signed on July 4, 1776."}
```

## Configuration Details

- `base_model_id`: The base model to be fine-tuned.

- `training_dataset_jsonl_path` & `eval_dataset_jsonl_path`: Path of the
  training and evaluation datasets respectively.

- `prompt_template`: How the input and output pairs should be formatted.

- `tokenizer_max_length`: The maximum length for the tokenizer.

- `bnb_config`: Configuration for the bitsandbytes library.

- `lora_config`: Configuration for the low-rank approximation.

- `output_dir`: The directory where you want the trained model to be saved.

- `training`: Configuration for the training steps, including the optimizer to
  be used, warmup steps, maximum steps, batch size, logging steps, where to save
  logs, save strategy, evaluation strategy, and learning rate.


## License

This project is licensed under the MIT License.

