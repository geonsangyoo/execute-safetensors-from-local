#!/usr/bin/env python3
"""
LLaDA Runner for Masked Diffusion Models
Specialized runner for LLaDA models following the official implementation.
Based on: https://github.com/ML-GSAI/LLaDA
"""

import argparse
import logging
import sys
from typing import List, Optional

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LLADARunner:
    """Specialized runner for LLaDA (Large Language Diffusion with mAsking) models."""

    def __init__(self, model_path: str) -> None:
        """
        Initialize the LLaDA Runner.

        Args:
            model_path: Path to the model directory or Hugging Face model name
        """
        self.model_path = model_path
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

        logger.info(f"Initializing LLaDA Runner with model: {self.model_path}")

    def load_model(self) -> None:
        """Load the LLaDA model and tokenizer following official guidelines."""
        try:
            logger.info("Loading LLaDA tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

            logger.info("Loading LLaDA model...")
            self.model = AutoModel.from_pretrained(
                self.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
            )

            logger.info("LLaDA model loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading LLaDA model: {e}")
            raise

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate text using LLaDA model.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to generate

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
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate text using LLaDA's generate method
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                    use_cache=False,  # LLaDA doesn't support KV cache
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

    def interactive_mode(self, args: argparse.Namespace) -> None:
        """Run the LLaDA model in interactive mode."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")

        print("\n=== LLaDA Interactive Mode ===")
        print("Type 'quit' to exit, 'help' for commands")
        print(
            "Note: LLaDA is a diffusion model, generation may take longer than "
            "autoregressive models"
        )
        print("=" * 40)

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
                    print(
                        "\nNote: LLaDA uses diffusion-based generation, which may be "
                        "slower but can produce high-quality text"
                    )
                    continue
                elif not user_input:
                    continue

                # Generate response
                responses = self.generate(
                    user_input,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                )

                if responses:
                    print(f"\nLLaDA: {responses[0]}")
                else:
                    print("\nLLaDA: Sorry, I couldn't generate a response.")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"Error: {e}")


def main() -> None:
    """Main function to run LLaDA."""
    parser = argparse.ArgumentParser(
        description="Run LLaDA (Large Language Diffusion with mAsking) models"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "Path to the LLaDA model directory or Hugging Face model name "
            "(e.g., 'GSAI-ML/LLaDA-8B-Base')"
        ),
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

    try:
        # Initialize and load the LLaDA model
        runner = LLADARunner(args.model_path)
        runner.load_model()

        if args.interactive:
            runner.interactive_mode(args)
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
            print("\nExample usage:")
            print(
                "python llada_runner.py --model_path GSAI-ML/LLaDA-8B-Base "
                "--interactive"
            )
            print(
                "python llada_runner.py --model_path GSAI-ML/LLaDA-8B-Instruct "
                "--prompt 'Hello, how are you?'"
            )

    except Exception as e:
        logger.error(f"Error running LLaDA: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
