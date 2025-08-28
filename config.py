#!/usr/bin/env python3
"""
Configuration file for the RAG system.
Handles non-model settings and imports model/embedding config from model_config.py.
"""

from model_config import (
    get_model_provider,
    get_llm_config,
    get_embedding_model_name,
    print_model_config
)

# Pinecone Settings
PINECONE_API_KEY = "pcsk_3SumMY_39jUDdvLLZXUXPxWdZT5Cex1kXSB2FMoec5QCZLJfCewmfVUMnjUgt2k2nLkiHD"
PINECONE_ENVIRONMENT = "us-east-1"
PINECONE_INDEX_NAME = "ibm-index"

def get_config():
    """Get configuration for RAG system initialization (including model/embedding config)"""
    llm_cfg = get_llm_config()
    return {
        "pinecone_api_key": PINECONE_API_KEY,
        "pinecone_environment": PINECONE_ENVIRONMENT,
        "pinecone_index_name": PINECONE_INDEX_NAME,
        "model_provider": get_model_provider(),
        "llm_config": llm_cfg,
        "embedding_model_name": get_embedding_model_name(),
    }

def print_config():
    print("\n" + "="*60)
    print("CURRENT CONFIGURATION")
    print("="*60)
    print(f"Pinecone API Key: {'set' if PINECONE_API_KEY else 'not set'}")
    print(f"Pinecone Environment: {PINECONE_ENVIRONMENT}")
    print(f"Pinecone Index: {PINECONE_INDEX_NAME}")
    print_model_config()

if __name__ == "__main__":
    print_config() 