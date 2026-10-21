"""Hugging Face backend (runs in Colab). Uses PyTorch under the hood."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFChat:
    def __init__(self, name="Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens=80):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.max_new_tokens = max_new_tokens
        self.prompt_tokens = 0
        self.generated_tokens = 0

    @torch.no_grad()
    def __call__(self, prompt):
        msgs = [{"role": "user", "content": prompt}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)
        out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[0, ids.shape[1]:]
        self.prompt_tokens += ids.shape[1]
        self.generated_tokens += gen.shape[0]
        return self.tok.decode(gen, skip_special_tokens=True)
