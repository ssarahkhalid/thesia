# src/rag/model_downloader.py
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import yaml
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDownloader:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize model downloader with configuration."""
        self.config = self._load_config(config_path)
        self.base_path = Path(self.config['paths']['models_dir'])
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def download_gguf_model(self) -> Path:
        """Download the specified GGUF model from Hugging Face."""
        model_config = self.config['model']['llm']
        
        try:
            logger.info(f"Starting download of {model_config['filename']}")
            logger.info(f"From repository: {model_config['repo_id']}")
            
            model_path = hf_hub_download(
                repo_id=model_config['repo_id'],
                filename=model_config['filename'],
                local_dir=self.base_path,
                local_dir_use_symlinks=False
            )
            
            final_path = Path(model_path)
            logger.info(f"Model downloaded successfully to: {final_path}")
            
            # Verify file size
            file_size = final_path.stat().st_size / (1024 * 1024 * 1024)  # Convert to GB
            logger.info(f"Model file size: {file_size:.2f} GB")
            
            return final_path
            
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise

def main():
    """Main function to execute model download."""
    try:
        downloader = ModelDownloader()
        model_path = downloader.download_gguf_model()
        logger.info("Model download completed successfully")
        logger.info(f"Model location: {model_path}")
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        raise

if __name__ == "__main__":
    main()