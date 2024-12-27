from typing import List
import httpx
import hashlib
from pathlib import Path
import asyncio
import time

import tiktoken

from openai import APITimeoutError

import torch

# ---------------------------------------------------------------------------

class FeroLlmCachedItem:

    # -----------------------------------------------------------------------

    def __init__(self, text=None, model=""):
        
        self.text = text
        self.model = model
        self.cache_key = self.get_cache_key()
        
        self.oai_embedding = None
        self.exceptions = None

    # -----------------------------------------------------------------------

    def add_exception(self, exception):
        if self.exceptions is None:
            self.exceptions = []

        self.exceptions.append(exception)
        return

    # -----------------------------------------------------------------------

    def get_cache_key(self):
        key = "|".join([self.text, self.model]) if self.text is not None else "-"
        # p(collision)=1.47e-29 (and a bit faster than sha1, etc.)
        return hashlib.md5(key.encode()).hexdigest()

    # -----------------------------------------------------------------------

    def save_in_local_cache(self, local_cache_dir):
        
        if not local_cache_dir:
            return

        file_name = local_cache_dir.joinpath(self.cache_key)
        with open(file_name, "wb") as f:
            torch.save((self.text, self.oai_embedding), f)

        return

    # -----------------------------------------------------------------------

    def check_against_local_cache(self, local_cache_dir):
        
        if not local_cache_dir:
            return

        file = local_cache_dir.joinpath(self.cache_key)
        if file.exists():
            with open(file, "rb") as f:
                saved_text, self.oai_embedding = torch.load(f)
                if saved_text != self.text: # (p of this happenning should be 1.47e-29)
                    self.oai_embedding = None

        return

# ---------------------------------------------------------------------------
# Using v2
# Max input tokens: 8191

class FeroLlmFeaturizer:
    
    # -----------------------------------------------------------------------

    def __init__(self,
        fero_llm_client,
        model_name="text-embedding-ada-002", # 3,000 pages for $1
        embedding_dim=1536,
        local_cache_dir=None
    ):
        self.fero_llm_client = fero_llm_client

        self.model_name = model_name

        self.bpe_encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 8188 # = 8191 - 3

        self.embedding_dim = embedding_dim

        if local_cache_dir:
            self.local_cache_dir = Path(local_cache_dir)
            self.local_cache_dir.mkdir(exist_ok=True)
        else:
            self.local_cache_dir = None

    # -----------------------------------------------------------------------

    @property
    def is_async(self):
        return self.fero_llm_client.is_async

    # -----------------------------------------------------------------------

    def get_tokens(self, text) -> List[int]:
        int_tokens = self.bpe_encoding.encode(text)
        return int_tokens

    # -----------------------------------------------------------------------

    def featurize_text(self, text, force_refresh=False) -> FeroLlmCachedItem:

        return self.featurize_texts([text], force_refresh=force_refresh)[0]

    # -----------------------------------------------------------------------

    def featurize_texts(self, texts, force_refresh=False) -> List[FeroLlmCachedItem]:

        results = [FeroLlmCachedItem(text, model=self.model_name) for text in texts]
        tasks = []
        for item in results:
            c_tokens = len(self.get_tokens(item.text))
            if 1 <= c_tokens < self.max_tokens:
                if not force_refresh:
                    item.check_against_local_cache(self.local_cache_dir)
                if item.oai_embedding is None:
                    tasks.append(item)
            elif c_tokens == 0:
                item.add_exception("Too short.")
            else:
                item.add_exception("Too long.")

        if tasks:
            timeouts = [2, 3, 5, 10, 30, 60, 300]
            for i_timeout, timeout in enumerate(timeouts):
                try:
                    embeddings = self.fero_llm_client.client.embeddings.create(
                        input=[item.text for item in tasks],
                        model=self.model_name,
                        timeout=httpx.Timeout(timeout)
                    )
                    break
                except APITimeoutError as e:
                    print(f"OAI embeddings timeout ({i_timeout})")
                    if i_timeout == len(timeouts) - 1:
                        raise e
                    time.sleep(timeouts[i_timeout])
                    continue
                
            assert len(embeddings.data) == len(tasks)

            for embedding, (i_item, item) in zip(embeddings.data, enumerate(tasks)):
                assert embedding.index == i_item
                
                item.oai_embedding = torch.tensor(embedding.embedding, dtype=torch.float64)
                assert item.oai_embedding.shape == (self.embedding_dim, )
                
                item.save_in_local_cache(self.local_cache_dir)

        return results

    # -----------------------------------------------------------------------

    async def featurize_text_async(self, text, force_refresh=False) -> FeroLlmCachedItem:

        return (await self.featurize_texts_async([text], force_refresh=force_refresh))[0]

    # -----------------------------------------------------------------------
    # TODO reduce duplication with the sync version of this

    async def featurize_texts_async(self, texts, force_refresh=False) -> List[FeroLlmCachedItem]:

        results = [FeroLlmCachedItem(text, model=self.model_name) for text in texts]
        tasks = []
        for item in results:
            c_tokens = len(self.get_tokens(item.text))
            if 1 <= c_tokens < self.max_tokens:
                if not force_refresh:
                    item.check_against_local_cache(self.local_cache_dir)
                if item.oai_embedding is None:
                    tasks.append(item)
            elif c_tokens == 0:
                item.add_exception("Too short.")
            else:
                item.add_exception("Too long.")

        if tasks:
            timeouts = [2, 3, 4, 5, 10]
            for i_timeout, timeout in enumerate(timeouts):
                try:
                    embeddings = await self.fero_llm_client.client.embeddings.create(
                        input=[item.text for item in tasks],
                        model=self.model_name,
                        timeout=httpx.Timeout(timeout)
                    )
                    break
                except APITimeoutError as e:
                    print(f"OAI embeddings timeout ({i_timeout})")
                    if i_timeout == len(timeouts) - 1:
                        raise e
                    await asyncio.sleep(timeouts[i_timeout])
                    continue
                
            assert len(embeddings.data) == len(tasks)

            for embedding, (i_item, item) in zip(embeddings.data, enumerate(tasks)):
                assert embedding.index == i_item
                
                item.oai_embedding = torch.tensor(embedding.embedding, dtype=torch.float64)
                assert item.oai_embedding.shape == (self.embedding_dim, )
                
                item.save_in_local_cache(self.local_cache_dir)

        return results
