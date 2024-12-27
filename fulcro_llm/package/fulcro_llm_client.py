import os
from pathlib import Path
import hashlib
import json

import torch

from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI

import anthropic

# -----------------------------------------------------------------

get_openai_api_key = lambda: os.getenv("OAI_API_KEY")

get_responsiv_openai_api_key = lambda: os.getenv("RESPONSIV_OAI_API_KEY")

get_azure_openai_endpoint = lambda: os.getenv("AZURE_OAI_ENDPOINT")
get_azure_openai_endpoint_embeddings = lambda: os.getenv("AZURE_OAI_ENDPOINT_EMBEDDINGS")
get_azure_openai_api_key = lambda: os.getenv("AZURE_OAI_API_KEY")
get_azure_openai_api_version = lambda: os.getenv("AZURE_OAI_API_VERSION")
get_azure_openai_api_version_embeddings = lambda: os.getenv("AZURE_OAI_API_VERSION_EMBEDDINGS")

get_azure_ce_openai_endpoint = lambda: os.getenv("AZURE_CE_OAI_ENDPOINT")
get_azure_ce_openai_api_key = lambda: os.getenv("AZURE_CE_OAI_API_KEY")
get_azure_ce_openai_api_version = lambda: os.getenv("AZURE_CE_OAI_API_VERSION")

get_responsiv_anthropic_api_key = lambda: os.getenv("RESPONSIV_ANTHROPIC_API_KEY")

# -----------------------------------------------------------------

os_envs = {
    
    "openai": ["OAI_API_KEY"],
    
    "responsiv-openai": ["RESPONSIV_OAI_API_KEY"],
    
    "azure-openai": ["AZURE_OAI_ENDPOINT", "AZURE_OAI_API_KEY", "AZURE_OAI_API_VERSION"],

    "azure-openai-embeddings": ["AZURE_OAI_ENDPOINT_EMBEDDINGS", "AZURE_OAI_API_KEY", "AZURE_OAI_API_VERSION_EMBEDDINGS"],

    "azure-ce-openai": ["AZURE_CE_OAI_ENDPOINT", "AZURE_CE_OAI_API_KEY", "AZURE_CE_OAI_API_VERSION"],
}

# -----------------------------------------------------------------

async_client_constructor_lambdas = {
    
    "openai": lambda: AsyncOpenAI(api_key=get_openai_api_key()),
    
    "responsiv-openai": lambda: AsyncOpenAI(api_key=get_responsiv_openai_api_key()),
    
    "azure-openai": lambda: AsyncAzureOpenAI(
        azure_endpoint=get_azure_openai_endpoint(),
        api_key=get_azure_openai_api_key(),
        api_version=get_azure_openai_api_version()
    ),

    "azure-openai-embeddings": lambda: AsyncAzureOpenAI(
        azure_endpoint=get_azure_openai_endpoint_embeddings(),
        api_key=get_azure_openai_api_key(),
        api_version=get_azure_openai_api_version_embeddings()
    ),
    
    "azure-ce-openai": lambda: AsyncAzureOpenAI(
        azure_endpoint=get_azure_ce_openai_endpoint(),
        api_key=get_azure_ce_openai_api_key(),
        api_version=get_azure_ce_openai_api_version()
    ),

    "responsiv-anthropic": lambda: anthropic.AsyncAnthropic(api_key=get_responsiv_anthropic_api_key()),
}

# -----------------------------------------------------------------

sync_client_constructor_lambdas = {
    
    "openai": lambda: OpenAI(api_key=get_openai_api_key()),
    
    "responsiv-openai": lambda: OpenAI(api_key=get_responsiv_openai_api_key()),
    
    "azure-openai": lambda: AzureOpenAI(
        azure_endpoint=get_azure_openai_endpoint(),
        api_key=get_azure_openai_api_key(),
        api_version=get_azure_openai_api_version()
    ),

    "azure-openai-embeddings": lambda: AzureOpenAI(
        azure_endpoint=get_azure_openai_endpoint_embeddings(),
        api_key=get_azure_openai_api_key(),
        api_version=get_azure_openai_api_version_embeddings()
    ),
    
    "azure-ce-openai": lambda: AzureOpenAI(
        azure_endpoint=get_azure_ce_openai_endpoint(),
        api_key=get_azure_ce_openai_api_key(),
        api_version=get_azure_ce_openai_api_version()
    ),
}

# ---------------------------------------------------------------------------

class FeroLlmCachedItem:

    # -----------------------------------------------------------------------

    def __init__(self, payload):
        self.payload = payload
        self.cache_key = self.get_cache_key()
        self.completions = None

    # -----------------------------------------------------------------------

    def get_cache_key(self):
        key = json.dumps(self.payload)
        # p(collision)=1.47e-29 (and a bit faster than sha1, etc.)
        return hashlib.md5(key.encode()).hexdigest()

    # -----------------------------------------------------------------------

    def save_in_local_cache(self, local_cache_dir):
        
        if not local_cache_dir:
            return

        file_name = local_cache_dir.joinpath(self.cache_key)
        with open(file_name, "wb") as f:
            torch.save((self.payload, self.completions), f)

        return

    # -----------------------------------------------------------------------

    def check_against_local_cache(self, local_cache_dir):
        
        if not local_cache_dir:
            return

        file = local_cache_dir.joinpath(self.cache_key)
        if file.exists():
            with open(file, "rb") as f:
                saved_payload, self.completions = torch.load(f)
                if json.dumps(saved_payload) != json.dumps(self.payload): # (p of this happenning should be 1.47e-29)
                    self.completions = None

        return

# -----------------------------------------------------------------

class FeroCachingLlmClientBase:

    is_caching_llm_client = True

    @classmethod
    def get_os_env_params(cls, client_type: str):
        return [(env, os.getenv(env)) for env in os_envs[client_type]]

    # -------------------------------------------------------------

    def set_local_cache_dir(self, local_cache_dir):
        self.local_cache_dir = Path(local_cache_dir) if local_cache_dir is not None else None
        if self.local_cache_dir:
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------

class FeroCachingOaiClient(FeroCachingLlmClientBase):

    is_async = False

    # -------------------------------------------------------------

    def __init__(self, client_type: str, local_cache_dir=None):

        assert client_type in async_client_constructor_lambdas
        self.client = sync_client_constructor_lambdas[client_type]()
        
        super().set_local_cache_dir(local_cache_dir)
        return

    # -------------------------------------------------------------

    def create_completions(self, model, messages, temperature=None, n=None, max_tokens=None, force_key=None):

        cache_item = FeroLlmCachedItem(payload=(model, messages, temperature, n, max_tokens, force_key))
        cache_item.check_against_local_cache(local_cache_dir=self.local_cache_dir)
        if cache_item.completions is not None:
            return cache_item.completions

        result = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
        )

        cache_item.completions = [choice.message.content for choice in result.choices]
        cache_item.save_in_local_cache(local_cache_dir=self.local_cache_dir)
        return cache_item.completions

# -----------------------------------------------------------------

class FeroAsyncCachingOaiClient(FeroCachingLlmClientBase):

    is_async = True

    # -------------------------------------------------------------

    def __init__(self, client_type: str, local_cache_dir=None):

        assert client_type in async_client_constructor_lambdas
        self.client = async_client_constructor_lambdas[client_type]()
        
        super().set_local_cache_dir(local_cache_dir)
        return

    # -------------------------------------------------------------

    async def create_completions(self, model, messages, temperature=None, n=None, max_tokens=None, force_key=None, timeout=None):

        cache_item = FeroLlmCachedItem(payload=(model, messages, temperature, n, max_tokens, force_key))
        cache_item.check_against_local_cache(local_cache_dir=self.local_cache_dir)
        if cache_item.completions is not None:
            return cache_item.completions

        result = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            timeout=timeout
        )
        cache_item.completions = [choice.message.content for choice in result.choices]

        cache_item.save_in_local_cache(local_cache_dir=self.local_cache_dir)
        return cache_item.completions

# -----------------------------------------------------------------

class FeroAsyncCachingAnthropicClient(FeroCachingLlmClientBase):

    is_async = True

    # -------------------------------------------------------------

    def __init__(self, client_type: str, local_cache_dir=None):

        assert client_type in async_client_constructor_lambdas
        self.client = async_client_constructor_lambdas[client_type]()
        
        super().set_local_cache_dir(local_cache_dir)
        return

    # -------------------------------------------------------------

    async def create_completions(self, model, messages, temperature=None, n=None, max_tokens=None, force_key=None, timeout=None):

        force_key = f"anthropic-{force_key}" if force_key is not None else "anthropic"

        cache_item = FeroLlmCachedItem(payload=(model, messages, temperature, n, max_tokens, force_key))
        cache_item.check_against_local_cache(local_cache_dir=self.local_cache_dir)
        if cache_item.completions is not None:
            return cache_item.completions

        assert n is None or n == 1, "Not yet supported: n other than None"

        assert messages[0]["role"] == "system"
        system_prompt = messages[0]["content"]
        assert system_prompt
        messages = messages[1:]
        assert messages

        completion = await self.client.messages.create(
            model=model,
            max_tokens=max_tokens if max_tokens is not None else 4096, #..todo
            temperature=temperature,
            system=system_prompt,
            messages=messages,
            timeout=timeout,
        )
        cache_item.completions = [content.text for content in completion.content]

        cache_item.save_in_local_cache(local_cache_dir=self.local_cache_dir)
        return cache_item.completions

# -----------------------------------------------------------------

class FeroAsyncCachingLlmClients:

    # -------------------------------------------------------------

    @classmethod
    def make_client(cls, client_type: str, local_cache_dir: str=None, for_embeddings=False):
        
        if "-anthropic" in client_type:
            return FeroAsyncCachingAnthropicClient(
                client_type=client_type,
                local_cache_dir=local_cache_dir
            )

        if for_embeddings and "azure-" in client_type:
            client_type += "-embeddings"

        return FeroAsyncCachingOaiClient(
            client_type=client_type,
            local_cache_dir=local_cache_dir
        )