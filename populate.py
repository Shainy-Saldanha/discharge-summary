import os
import tkinter as tk
from tkinter import filedialog
from rag import RAGSystem
from config import get_config, print_config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PopulateDB:
    def __init__(self, config: dict):
        self.rag_system = RAGSystem(config)
    
    def select_folder(self) -> str:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        folder_path = filedialog.askdirectory(title="Select Folder with Documents")
        return folder_path
    
    def process_files(self, folder_path: str):
        try:
            if not os.path.exists(folder_path):
                raise ValueError(f"Folder not found: {folder_path}")
            
            # Get all supported files
            files = []
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.pdf', '.pptx', '.ppt')):
                    files.append(os.path.join(folder_path, file))
            
            if not files:
                raise ValueError("No supported files found in the selected folder")
            
            logger.info(f"Found {len(files)} supported files to process")
            
            # Process each file
            for file_path in files:
                try:
                    logger.info(f"Processing file: {file_path}")
                    
                    # Extract text
                    text = self.rag_system.extract_text_from_file(file_path)
                    
                    # Preprocess text
                    processed_text = self.rag_system.preprocess_text(text)
                    
                    # Generate embeddings
                    embeddings = self.rag_system.generate_embeddings([processed_text])
                    
                    # Prepare metadata
                    metadata = [{
                        "filename": os.path.basename(file_path),
                        "filetype": os.path.splitext(file_path)[1].lower(),
                        "path": file_path
                    }]
                    
                    # Store in Pinecone
                    self.rag_system.store_in_pinecone([processed_text], embeddings, metadata)
                    
                    logger.info(f"Successfully processed and stored: {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    continue
            
            logger.info("All files processed successfully")
            
            # Display metrics summary
            print("\n" + "="*60)
            print("FILE PROCESSING COMPLETE - METRICS SUMMARY")
            print("="*60)
            self.rag_system.print_metrics_summary()
            
        except Exception as e:
            logger.error(f"Error in file processing: {str(e)}")
            raise

def main():
    try:
        # Get configuration
        config = get_config()
        
        # Initialize PopulateDB
        populator = PopulateDB(config)
        print(f"PopulateDB initialized with {config['model_provider']} model provider")
        llm_cfg = config['llm_config']
        if config['model_provider'] == "ollama":
            print(f"Using Ollama model: {llm_cfg['model']}")
            print(f"Make sure Ollama is running at {llm_cfg['base_url']}")
        
        # Select folder
        folder_path = populator.select_folder()
        if not folder_path:
            logger.info("No folder selected. Exiting...")
            return
        
        # Process files
        populator.process_files(folder_path)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

