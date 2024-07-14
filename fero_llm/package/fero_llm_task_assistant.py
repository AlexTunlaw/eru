import os
from pathlib import Path
import hashlib

import torch

from openai import OpenAI

# ---------------------------------------------------------------------------

class OaiCachedItem:

    # -----------------------------------------------------------------------

    def __init__(self, model, system_prompt, task_prompt):
        
        self.model = model
        self.system_prompt = system_prompt
        self.task_prompt = task_prompt
        self.cache_key = self.get_cache_key()

        self.completion = None

    # -----------------------------------------------------------------------

    def get_cache_key(self):
        key = '|'.join([self.model, self.system_prompt, self.task_prompt])
        # p(collision)=1.47e-29 (and a bit faster than sha1, etc.)
        return hashlib.md5(key.encode()).hexdigest()

    # -----------------------------------------------------------------------

    def save_in_local_cache(self, local_cache_dir):
        
        if not local_cache_dir:
            return

        file_name = Path(local_cache_dir).joinpath(self.cache_key)
        with open(file_name, "wb") as f:
            torch.save(
                (self.model, self.system_prompt, self.task_prompt, self.completion),
                f
            )

        return

    # -----------------------------------------------------------------------

    def check_against_local_cache(self, local_cache_dir):
        
        if not local_cache_dir:
            return

        file = Path(local_cache_dir).joinpath(self.cache_key)
        if file.exists():
            with open(file, "rb") as f:
                saved_model, saved_system_prompt, saved_task_prompt, self.completion = torch.load(f)
                if (
                    saved_model != self.model or
                    saved_system_prompt != self.system_prompt or
                    saved_task_prompt != self.task_prompt
                ): # (p of this happenning should be 1.47e-29)
                    self.completion = None
                
        return

# ---------------------------------------------------------------------------

class FeroLlmTaskAssistant:

    # -----------------------------------------------------------------------

    def __init__(self,
        model: str,
        system_prompt: str,
        local_cache_dir
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.local_cache_dir = local_cache_dir

        self.openai_client = OpenAI(api_key=os.getenv("OAI_API_KEY"))
        return

    # -----------------------------------------------------------------------

    def invoke(self, task_text: str, system_prompt=None) -> str: # return completion

        system_prompt = system_prompt if system_prompt is not None else self.system_prompt

        item = OaiCachedItem(
            model=self.model,
            system_prompt=system_prompt,
            task_prompt=task_text
        )

        item.check_against_local_cache(self.local_cache_dir)
        if item.completion:
            return item.completion
        
        completion = self.openai_client.chat.completions.create(
            model=item.model,
            messages=[
                { 'role': 'system','content': item.system_prompt },
                { 'role': 'user', 'content': item.task_prompt },
            ]
        ).choices[0].message.content

        item.completion = completion
        item.save_in_local_cache(self.local_cache_dir)
        return completion