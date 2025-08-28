#!/usr/bin/env python3
"""
Demo script to showcase the metrics functionality of the RAG system.
This script demonstrates how to use the metrics features and what information is collected.
"""

from rag import RAGSystem
from config import get_config, print_config
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_metrics():
    """Demonstrate the metrics functionality"""
    try:
        print("Initializing RAG System...")
        config = get_config()
        rag_system = RAGSystem(config)
        llm_cfg = config['llm_config']
        print(f"RAG System initialized with {config['model_provider']} model provider")
        if config['model_provider'] == "google":
            print(f"Google model: {llm_cfg['model']}")
        else:
            print(f"Ollama model: {llm_cfg['model']}")
            print(f"Make sure Ollama is running at {llm_cfg['base_url']}")
        print("RAG System initialized successfully!\n")
        # Demo queries to test metrics
        demo_queries = [
            "What is machine learning?",
            "Explain artificial intelligence",
            "How does deep learning work?",
            "What are neural networks?",
            "Explain natural language processing"
        ]
        print("="*60)
        print("DEMO: QUERY METRICS COLLECTION")
        print("="*60)
        for q in demo_queries:
            print(f"\nQuery: {q}")
            try:
                response = rag_system.query(q)
                print(f"Answer: {response}")
            except Exception as e:
                print(f"Error: {e}")
        print("\n" + "="*60)
        print("METRICS SUMMARY")
        print("="*60)
        rag_system.print_metrics_summary()
        print("\nAdditional Info:")
        print("1. Query metrics include words in answer, processing time, latency, etc.")
        print("2. File metrics include processing time, file size, etc.")
        print("3. Use 'metrics' command in main.py to see real-time metrics!")
        print(f"Current LLM Model: {rag_system.llm_model_name}")
        print(f"Current Embedding Model: {rag_system.embedding_model_name}")
        print(f"Model Provider: {config['model_provider']}")
        print("\nUse 'metrics' command in main.py to see real-time metrics!")
    except Exception as e:
        logger.error(f"Error in demo: {str(e)}")
        raise

if __name__ == "__main__":
    demo_metrics() 