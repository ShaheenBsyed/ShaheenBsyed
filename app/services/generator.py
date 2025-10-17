from __future__ import annotations

from typing import Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from app.config import settings


class LocalGenerator:
    def __init__(self):
        model_name = settings.generation.model_name
        device = 0 if torch.cuda.is_available() and settings.embedding.device != "cpu" else -1
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device != -1 else torch.float32)
        if device != -1:
            self.model = self.model.to("cuda")
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

    def build_prompt(self, question: str, contexts: List[str]) -> str:
        context_block = "\n\n".join([f"Context {i+1}: {c}" for i, c in enumerate(contexts)])
        system = (
            "You are a helpful assistant. Use only the provided contexts to answer.\n"
            "If the answer is not in the contexts, say you don't know.\n"
        )
        return f"{system}\n\n{context_block}\n\nQuestion: {question}\nAnswer:"

    def stream_answer(self, question: str, contexts: List[str]) -> Iterable[str]:
        prompt = self.build_prompt(question, contexts)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=settings.generation.max_tokens,
            temperature=settings.generation.temperature,
            do_sample=settings.generation.temperature > 0,
            top_p=settings.generation.top_p,
            streamer=self.streamer,
        )
        # launch generation in background thread
        import threading

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in self.streamer:
            yield new_text
        thread.join()
