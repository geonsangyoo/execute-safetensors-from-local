"""
LLM Runner

Execute the ordinary model with the given prompt.
Supports both single prompt execution and interactive chat mode.
"""

import argparse

import torch
from transformers import pipeline


def generate_single_response(pipe, prompt, max_new_tokens, do_sample, temperature):
    """
    Generate a single response for a given prompt.

    Args:
        pipe: The text generation pipeline
        prompt: Input prompt for text generation
        max_new_tokens: Maximum number of new tokens to generate
        do_sample: Whether to use sampling instead of greedy decoding
        temperature: Temperature for sampling (higher = more random)

    Returns:
        Generated text response
    """
    output = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )
    return output[0]["generated_text"]


def interactive_chat_mode(pipe, max_new_tokens, do_sample, temperature):
    """
    Run the model in interactive chat mode.

    Args:
        pipe: The text generation pipeline
        tokenizer: The tokenizer for processing input
        max_new_tokens: Maximum number of new tokens to generate
        do_sample: Whether to use sampling instead of greedy decoding
        temperature: Temperature for sampling
    """
    print("\n=== Interactive Chat Mode ===")
    print("Type 'quit', 'exit', or 'q' to exit the chat.")
    print("Type your message and press Enter to chat with the model.\n")

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            # Skip empty input
            if not user_input:
                continue

            # Generate response
            print("Generating response...")
            response = generate_single_response(
                pipe, user_input, max_new_tokens, do_sample, temperature
            )

            # Extract only the generated part (remove the input prompt)
            # For simple text generation, we need to handle this carefully
            if response.startswith(user_input):
                generated_part = response[len(user_input) :].strip()
            else:
                generated_part = response.strip()

            print(f"Model: {generated_part}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.\n")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="LLM Runner for Ordinary Language Models - Single Prompt or Interactive Chat Mode"
    )
    parser.add_argument(
        "--model_path_in_hugging_face",
        type=str,
        required=True,
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
        help="Input prompt for text generation (used in single prompt mode)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for sampling (higher = more random)",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling instead of greedy decoding",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive chat mode instead of single prompt mode",
    )

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.interactive:
        # In interactive mode, prompt should not be provided
        if args.prompt:
            parser.error(
                "--prompt should not be provided when using --interactive mode"
            )
    else:
        # In single prompt mode, prompt is required
        if not args.prompt:
            parser.error("--prompt is required when not using --interactive mode")

    # Set device
    device_map = args.device

    print(f"Loading model from: {args.model_path_in_hugging_face}")

    # Load the text generation pipeline
    # This creates a pipeline that can generate text based on input prompts
    pipe = pipeline(
        "text-generation",
        model=args.model_path_in_hugging_face,
        device=device_map,
        torch_dtype=torch.bfloat16,
    )

    if args.interactive:
        # Run in interactive chat mode
        # This allows continuous conversation with the model
        interactive_chat_mode(
            pipe, args.max_new_tokens, args.do_sample, args.temperature
        )
    else:
        # Run in single prompt mode
        # This generates a response for one specific prompt
        print("Generating response for single prompt...")
        response = generate_single_response(
            pipe, args.prompt, args.max_new_tokens, args.do_sample, args.temperature
        )
        print(response)


if __name__ == "__main__":
    main()
