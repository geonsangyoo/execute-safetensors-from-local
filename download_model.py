#!/usr/bin/env python3
"""
Model Downloader for Safetensors Models
This script downloads LLM models from Hugging Face Hub
and converts them to safetensors format.
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_model(
    model_name: str,
    output_dir: str,
    use_auth_token: bool = False,
) -> None:
    """
    Download a model from Hugging Face Hub.

    Args:
        model_name: Name of the model on Hugging Face Hub
        output_dir: Directory to save the model
        use_auth_token: Whether to use authentication token
    """
    try:
        logger.info(f"Downloading model: {model_name}")
        logger.info(f"Output directory: {output_dir}/{model_name}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        output_dir = f"{output_dir}/{model_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Get token from environment variable if use_auth_token is True
        token = None
        if use_auth_token:
            token = os.getenv("HUGGING_FACE_USE_AUTH_TOKEN")
            if not token:
                logger.warning(
                    "HUGGING_FACE_USE_AUTH_TOKEN not found in environment variables"
                )
                logger.info("Please set HUGGING_FACE_USE_AUTH_TOKEN in your .env file")
                logger.info("You can copy env.example to .env and add your token")

        # Download the model
        snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            token=token,
            local_dir_use_symlinks=False,
        )

        logger.info(f"Model downloaded successfully to {output_dir}")

        # Verify the download
        verify_model(output_dir)

    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        raise


def verify_model(model_dir: str) -> None:
    """
    Verify that the downloaded model is valid.

    Args:
        model_dir: Directory containing the model
    """
    try:
        logger.info("Verifying model...")

        # Check if config.json exists
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            raise ValueError("config.json not found")

        # Try to load the tokenizer (just to verify it works)
        AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        logger.info("Tokenizer loaded successfully")

        # Try to load the model config (just to verify it works)
        AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        logger.info("Model config loaded successfully")

        # Check for model files
        model_files = [
            f
            for f in os.listdir(model_dir)
            if f.endswith((".safetensors", ".bin", ".pt"))
        ]
        if not model_files:
            logger.warning("No model weight files found")
        else:
            logger.info(f"Found {len(model_files)} model weight files")

        logger.info("Model verification completed successfully")

    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        raise


def list_available_models() -> None:
    """List some popular models available for download."""
    models = [
        "LLaDA-8B-Instruct",
        "google/gemma-3-4b-it-qat-q4_0-gguf",
        "google/gemma-3n-E2B-it",
        "Qwen/Qwen2.5-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]

    print("Testing models available for download:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")


def main() -> None:
    """Main function to download models."""
    parser = argparse.ArgumentParser(
        description="Download LLM models from Hugging Face Hub"
    )
    parser.add_argument(
        "--model_name", type=str, help="Name of the model on Hugging Face Hub"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./models", help="Directory to save the model"
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Use authentication token for private models",
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List popular models available for download",
    )

    args = parser.parse_args()

    if args.list_models:
        list_available_models()
        return

    if not args.model_name:
        print("Error: --model_name is required")
        print("Use --list_models to see available models")
        sys.exit(1)

    try:
        download_model(args.model_name, args.output_dir, args.use_auth_token)
        print("\nModel downloaded successfully!")
        print("You can now run it with:")
        print(f"python llm_runner.py --model_path {args.output_dir} --interactive")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
