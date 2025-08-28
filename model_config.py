"""
Model configuration and initialization for LLM and embedding models.
"""

# Model Provider Configuration
# Options: "google" or "ollama"
MODEL_PROVIDER = "ollama"

# API Keys
GOOGLE_API_KEY = "AIzaSyBUZ54wD6BhRYlIlLgMA7P2zMq3vR9NuG0"

# Google Model Settings
GOOGLE_MODEL = "gemini-1.5-flash"


# Ollama Settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"  # Default model, can be changed to any supported model
OLLAMA_TIMEOUT = 300  # Timeout in seconds for Ollama API calls (increased from 60 to 300)
OLLAMA_TEST_TIMEOUT = 10  # Timeout for testing connection (shorter timeout for quick checks)

# - "llama2"
# - "llama2:7b"
# - "llama2:13b"
# - "llama2:70b"
# - "llama3.1:8b" (your current model)
# - "llama3.1:70b"
# - "mistral"
# - "codellama"
# - "neural-chat"
# - "vicuna"
# - "wizard-vicuna-uncensored"

# Embedding Model (same for both providers)
# EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"
EMBEDDING_MODEL = "ibm-granite/granite-embedding-107m-multilingual"

# Utility functions to get model config

def get_model_provider():
    return MODEL_PROVIDER

def get_llm_config():
    if MODEL_PROVIDER == "google":
        return {
            "provider": "google",
            "api_key": GOOGLE_API_KEY,
            "model": GOOGLE_MODEL
        }
    elif MODEL_PROVIDER == "ollama":
        return {
            "provider": "ollama",
            "base_url": OLLAMA_BASE_URL,
            "model": OLLAMA_MODEL,
            "timeout": OLLAMA_TIMEOUT,
            "test_timeout": OLLAMA_TEST_TIMEOUT
        }
    else:
        raise ValueError(f"Unknown model provider: {MODEL_PROVIDER}")

def get_embedding_model_name():
    return EMBEDDING_MODEL

def print_model_config():
    print("\n" + "="*60)
    print("MODEL CONFIGURATION")
    print("="*60)
    print(f"Model Provider: {MODEL_PROVIDER}")
    if MODEL_PROVIDER == "google":
        print(f"Google Model: {GOOGLE_MODEL}")
        print(f"Google API Key: {'set' if GOOGLE_API_KEY else 'not set'}")
    else:
        print(f"Ollama Base URL: {OLLAMA_BASE_URL}")
        print(f"Ollama Model: {OLLAMA_MODEL}")
        print(f"Ollama Timeout: {OLLAMA_TIMEOUT}s")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print("="*60) 

    #What part of the body was infected in Ms. Renu's diabetic foot case?
    #Her left foot