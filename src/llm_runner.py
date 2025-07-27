"""
LLM Runner

Execute the ordinary model with the given prompt.
"""

import argparse

import torch
from transformers import pipeline


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="LLM Runner for Ordinary Language Models"
    )
    parser.add_argument(
        "--model_path_in_hugging_face",
        type=str,
        default="google/gemma-3n-e4b-it",
        help="Path to the model directory or model name from Hugging Face Hub",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to run the model on (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Question: Where is Apple Inc. headquartered?\nAnswer:",
        help="Input prompt for text generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling instead of greedy decoding",
    )

    args = parser.parse_args()

    # Set device
    device_map = args.device

    print(f"Loading model from: {args.model_path_in_hugging_face}")
    pipe = pipeline(
        "text-generation",
        model=args.model_path_in_hugging_face,
        device=device_map,
        torch_dtype=torch.float16,
    )

    # Generate an answer
    output = pipe(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
    )

    # Print result
    print(output[0]["generated_text"])


if __name__ == "__main__":
    main()
