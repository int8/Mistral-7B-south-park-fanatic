import yaml
from pathlib import Path
from peft import PeftModel

import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class SouthParkFanaticAI:
    def __init__(self, lora_model: str, config_path: str):
        self.lora_model = lora_model
        self.config_path = config_path
        self.config = yaml.safe_load(Path(config_path).read_text())

        self.bnb_config = BitsAndBytesConfig(
            bnb_4bit_compute_type=torch.bfloat16, **self.config["bnb_config"]
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config["base_model_id"],
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["base_model_id"], trust_remote_code=True, passing_size="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sp_model = PeftModel.from_pretrained(self.base_model, self.lora_model)
        self.sp_model.eval()

    def ask(self, question: str):
        eval_prompt = self.config["prompt_template"][:-4].format(
            question=question.lower(), answer=""
        )
        model_input = self.tokenizer(eval_prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            answer = self.tokenizer.decode(
                self.sp_model.generate(
                    **model_input,
                    max_new_tokens=150,
                    pad_token_id=2,
                    repetition_penalty=1.2,
                )[0],
                skip_special_tokens=True,
            ).strip()
            elems = re.split("answer:", answer, flags=re.IGNORECASE)
            if len(elems) < 2:
                return f"[answers not produced: raw response from model: {answer}]"
            return elems[1].strip()
