

from langchain_ollama import ChatOllama

MODEL_NAME = "llama3.1"

llm = ChatOllama(model=MODEL_NAME, temperature=0)
