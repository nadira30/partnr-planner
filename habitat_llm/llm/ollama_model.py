import requests
from typing import Optional, Dict, Any
from omegaconf import DictConfig

from habitat_llm.llm.base_llm import BaseLLM, Prompt


class OllamaModel(BaseLLM):
    def __init__(self, conf: DictConfig):
        print("\n>>> USING OLLAMA BACKEND <<<")

        self.llm_conf = conf
        self.generation_params = self.llm_conf.generation_params
        self.max_tokens = self.generation_params.max_tokens

        # Ollama server
        # self.host = getattr(self.llm_conf, "host", "localhost")
        # self.port = getattr(self.llm_conf, "port", 11434)
        self.engine = self.generation_params.engine

        self.url = "http://localhost:11434/api/generate"
        print("URL =", self.url)

    def generate(
        self,
        prompt: Prompt,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        max_length = max_length if max_length is not None else self.max_tokens
        if stop is None:
            stop = self.generation_params.stop

        # Strip special tokens from the input prompt
        # Ollama handles chat templating internally, so we need to remove any
        # special tokens that might have been added by the planner
        prompt_str = prompt if isinstance(prompt, str) else str(prompt)
        special_tokens = [
            "<|eot_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "system<|end_header_id|>",
            "user<|end_header_id|>",
            "assistant<|end_header_id|>",
        ]
        for token in special_tokens:
            prompt_str = prompt_str.replace(token, "")

        payload = {
            "model": self.engine,
            "prompt": prompt_str,
            "stream": False,
            "options": {
                "temperature": float(self.generation_params.temperature),
                "num_predict": int(max_length),
            },
        }

        resp = requests.post(self.url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        generation = data["response"]

        # Strip special tokens that Llama 3 may include
        special_tokens = [
            "<|eot_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|begin_of_text|>",
            "<|end_of_text|>",
        ]
        for token in special_tokens:
            generation = generation.replace(token, "")

        # Apply stop tokens
        if isinstance(stop, str):
            generation = generation.split(stop)[0]
        else:
            for s in stop:
                if s in generation:
                    generation = generation.split(s)[0]
                    break

        return generation.rstrip()
