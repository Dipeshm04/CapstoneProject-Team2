LLM_PROVIDER='openai'
LLM_MODEL='gpt-4o'

import os
import requests
from IPython.display import Markdown

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.abspath(".."))
from Utill import get_llm, get_api_key, UnifiedLLM
from langchain_core.messages import HumanMessage, SystemMessage
from IPython.display import Markdown