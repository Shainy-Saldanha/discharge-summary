from rag import RAGSystem
from config import get_config, print_config
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Get configuration
        config = get_config()
        
        # Initialize RAG system
        rag_system = RAGSystem(config)
        
        print(f"RAG System initialized successfully!")
        print(f"Model Provider: {config['model_provider']}")
        llm_cfg = config['llm_config']
        if config['model_provider'] == "google":
            print(f"Using Google model: {llm_cfg['model']}")
        else:
            print(f"Using Ollama model: {llm_cfg['model']}")
            print(f"Make sure Ollama is running at {llm_cfg['base_url']}")
        
        print("\nCommands:")
        print("- Type 'exit' to quit the program")
        print("- Type 'metrics' to see performance metrics")
        print("- Type 'stats' to see detailed statistics")
        print("- Type 'config' to see current configuration")
        print()
        
        while True:
            # Get user query
            query = input("\nEnter your question: ").strip()
            
            if query.lower() == 'exit':
                print("Goodbye!")
                break
            
            if query.lower() == 'metrics':
                rag_system.print_metrics_summary()
                continue
            
            if query.lower() == 'stats':
                stats = rag_system.get_metrics_summary()
                print("\n" + "="*60)
                print("DETAILED STATISTICS")
                print("="*60)
                
                if stats['query_stats']:
                    print("\nQUERY STATISTICS:")
                    for key, value in stats['query_stats'].items():
                        if key == 'llm_models_used' and isinstance(value, list):
                            if len(value) == 1:
                                print(f"  LLM Model: {value[0]}")
                            else:
                                print(f"  LLM Models Used: {', '.join(value)}")
                        elif key == 'primary_llm_model':
                            continue  # Skip this as it's handled above
                        elif isinstance(value, float):
                            print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
                        else:
                            print(f"  {key.replace('_', ' ').title()}: {value}")
                
                if stats['file_stats']:
                    print("\nFILE PROCESSING STATISTICS:")
                    for key, value in stats['file_stats'].items():
                        if key == 'embedding_models_used' and isinstance(value, list):
                            if len(value) == 1:
                                print(f"  Embedding Model: {value[0]}")
                            else:
                                print(f"  Embedding Models Used: {', '.join(value)}")
                        elif key == 'primary_embedding_model':
                            continue  # Skip this as it's handled above
                        elif isinstance(value, float):
                            print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
                        else:
                            print(f"  {key.replace('_', ' ').title()}: {value}")
                
                print("="*60)
                continue
            
            if query.lower() == 'config':
                print_config()
                continue
            
            if not query:
                print("Please enter a valid question.")
                continue
            
            try:
                # Ask for ground truth answer (optional)
                ground_truth_answer = input("Enter ground truth answer (optional, press Enter to skip): ").strip()
                
                # Process query
                if ground_truth_answer:
                    response = rag_system.query(query, ground_truth_answer=ground_truth_answer)
                else:
                    response = rag_system.query(query)
                
                # Display response
                print("\nAnswer:")
                print(response)
                
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                print(f"\nError: {str(e)}")
                print("Please try again with a different question.")
    
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 