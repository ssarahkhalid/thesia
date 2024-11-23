# src/rag/verify_model.py

from pathlib import Path
import logging
from llama_cpp import Llama
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_model_setup():
    """Verify model installation and basic functionality."""
    try:
        # Load config
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Check model file
        model_path = Path(config['paths']['models_dir']) / config['model']['llm']['filename']
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Check file size
        file_size_gb = model_path.stat().st_size / (1024 * 1024 * 1024)
        logger.info(f"Model file size: {file_size_gb:.2f} GB")
        
        # Test load model
        logger.info("Testing model loading...")
        model = Llama(
            model_path=str(model_path),
            n_ctx=config['model']['llm']['n_ctx'],
            n_gpu_layers=config['model']['llm']['n_gpu_layers']
        )
        
        # Test inference
        logger.info("Testing basic inference...")
        test_prompt = "What is the capital of France?"
        response = model.create_completion(
            test_prompt,
            max_tokens=20,
            temperature=0.7
        )
        
        logger.info("Model verification completed successfully!")
        logger.info(f"Test response: {response['choices'][0]['text']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        return False

if __name__ == "__main__":
    verify_model_setup()