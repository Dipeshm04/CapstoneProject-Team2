# llm_selector.py
import warnings
warnings.filterwarnings('ignore')
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_community.chat_models import ChatOllama  # for local Ollama
from langchain_ollama import ChatOllama
from google.generativeai import GenerativeModel
from google.generativeai.types import GenerationConfig

# Embeddings libraries
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

# Core LCEL/Runnable imports
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, BaseMessage
from typing import Union, List, Any

# Define the input type for the prompt (either a string or a list of messages)
PromptInput = Union[str, List[BaseMessage]]

import io
import os
from cryptography.fernet import Fernet
from dotenv import dotenv_values
import google.generativeai as genai

# /////////////////////////////////////////////////////////

def get_llm(provider: str, model: str, api_key: str = None, **kwargs):
    if provider == "openai":
        return ChatOpenAI(api_key=api_key, model=model, **kwargs)
    elif provider == "google":
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, **kwargs)
    elif provider == "ollama":
        return ChatOllama(model=model, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

_ENV_LOADED = False

# /////////////////////////////////////////////////////////

def get_api_key(provider: str, env_path: str = r"C:\.env_en") -> str:
    """Retrieves an API key for the requested provider after loading the encrypted .env."""
    load_encrypted_env(env_path)

    provider_map = {
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "weather": "WEATHER_API",
        "tavily": "TAVILY_SEARCH_KEY",
    }

    key_name = provider_map.get(provider.lower())
    if not key_name:
        raise ValueError(
            f"Unsupported provider: '{provider}'. Supported providers: {list(provider_map.keys())}"
        )

    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(
            f"API key for '{provider}' ({key_name}) was not found in environment."
        )

    return api_key

# /////////////////////////////////////////////////////////

# ============================================================
# UnifiedLLM — simple ask/stream wrapper
# ============================================================
class UnifiedLLM:
    def __init__(self, provider: str, model: str, api_key: str = None,
                 use_native: bool = False, **kwargs):

        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.use_native = use_native
        self.kwargs = kwargs

        # Provider selection
        if self.provider == "openai":
            self.llm = ChatOpenAI(api_key=api_key, model=model, **kwargs)

        elif self.provider == "google":
            if use_native:
                import google.generativeai as genai
                from google.generativeai.types import GenerationConfig

                genai.configure(api_key=api_key)
                self.GenerationConfig = GenerationConfig
                self.llm = genai.GenerativeModel(model_name=model)
            else:
                self.llm = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=api_key,
                    **kwargs
                )

        elif self.provider == "ollama":
            self.llm = ChatOllama(model=model, **kwargs)

        else:
            raise ValueError(f"Unsupported provider: {provider}")