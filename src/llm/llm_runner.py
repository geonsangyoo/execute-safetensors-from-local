#!/usr/bin/env python3
"""
LLM Runner for Safetensors Models on macOS
This script loads and runs LLM models stored in safetensors format from local storage.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LLMRunner:
    """Main class for running LLM models with safetensors format."""

    def __init__(self, model_path: str) -> None:
        """
        Initialize the LLM Runner.

        Args:
            model_path: Path to the model directory containing safetensors files
        """
        self.model_path = Path(model_path)
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

        logger.info(f"Initializing LLM Runner with model at: {self.model_path}")

    def load_model(self, model_type: str = "auto") -> None:
        """
        Load the model and tokenizer.

        Args:
            model_type: Type of model to load ("auto", "llama", "gpt2", etc.)
        """
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )

            logger.info("Loading model...")

            # Load the model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

            logger.info("Model loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        max_new_tokens: int = 512,
    ) -> List[str]:
        """
        Generate text using the loaded model directly.

        Args:
            prompt: Input text prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to generate
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            List of generated text sequences
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")

        try:
            logger.info(f"Generating text with prompt: {prompt[:100]}...")

            # Tokenize the input prompt
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            )

            # Move inputs to the same device as the model
            if hasattr(self.model, "device"):
                device = self.model.device
            else:
                device = next(self.model.parameters()).device

            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate text using the model directly
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                    use_cache=False,  # Disable KV cache for diffusion models
                )

            # Decode the generated tokens
            generated_texts = []
            for output in outputs:
                # Decode the full sequence and remove the original prompt
                full_text = self.tokenizer.decode(output, skip_special_tokens=True)
                if full_text.startswith(prompt):
                    generated_text = full_text[len(prompt) :].strip()
                else:
                    generated_text = full_text
                generated_texts.append(generated_text)

            logger.info(f"Generated {len(generated_texts)} text sequence(s)")
            return generated_texts

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    def interactive_mode(self) -> None:
        """Run the model in interactive mode."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")

        print("\n=== Interactive LLM Mode ===")
        print("Type 'quit' to exit, 'help' for commands")
        print("=" * 30)

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() == "quit":
                    print("Goodbye!")
                    break
                elif user_input.lower() == "help":
                    print("Available commands:")
                    print("- quit: Exit the program")
                    print("- help: Show this help message")
                    print(
                        "- Any other text will be used as a prompt for text generation"
                    )
                    continue
                elif not user_input:
                    continue

                # Generate response
                responses = self.generate(
                    user_input, temperature=0.7, do_sample=True, max_new_tokens=256
                )

                if responses:
                    print(f"\nAssistant: {responses[0]}")
                else:
                    print("\nAssistant: Sorry, I couldn't generate a response.")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"Error: {e}")


def main() -> None:
    """Main function to run the LLM."""
    parser = argparse.ArgumentParser(
        description="Run LLM models with safetensors format"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory containing safetensors files",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        help="Type of model (auto, llama, gpt2, etc.)",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to generate text for (non-interactive mode)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist")
        sys.exit(1)

    try:
        # Initialize and load the model
        runner = LLMRunner(args.model_path)
        runner.load_model(args.model_type)

        if args.interactive:
            runner.interactive_mode()
        elif args.prompt:
            responses = runner.generate(
                args.prompt,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
            print(f"\nPrompt: {args.prompt}")
            if responses:
                print(f"Generated: {responses[0]}")
            else:
                print("No response generated.")
        else:
            print("No prompt provided and not in interactive mode.")
            print("Use --prompt for single generation or --interactive for chat mode.")

    except Exception as e:
        logger.error(f"Error running LLM: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
