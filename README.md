# Execute Safetensors from Local

A Python library for running LLM models including LLaDA stored in safetensors format from local storage.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up authentication (for private models):
   - Copy `env.example` to `.env`:
     ```bash
     cp env.example .env
     ```
   - Edit `.env` and replace `your_token_here` with your Hugging Face token
   - Get your token from: https://huggingface.co/settings/tokens

## Usage (LLM)

### Run Models

Run the LLM models with your local model files:

```bash
# Run with default settings
python src/llm_runner.py

# Run with custom model path
python src/llm_runner.py --model_path_in_hugging_face "./models/qwen"

# Run with custom parameters
python src/llm_runner.py --model_path_in_hugging_face "./models/qwen" --prompt "What is machine learning?" --max_new_tokens 100 --temperature 0.8
```

## Environment Variables

- `HUGGING_FACE_USE_AUTH_TOKEN`: Your Hugging Face authentication token (required for private models)

## Available Models

Some testing models you can use:
- LLaDA-8B-Instruct
- google/gemma-3-4b-it-qat-q4_0-gguf
- google/gemma-3n-E2B-it
- Qwen/Qwen2.5-3B-Instruct
- meta-llama/Llama-3.1-8B-Instruct 

## Usage (LLaDA Model)

### Running LLaDA Model

LLaDA supports running the GSAI-ML/LLaDA-8B-Instruct model in two ways:

1. Using default path (../LLaDA-8B-Instruct):
   ```bash
   # Please ensure the GSAI-ML/LLaDA-8B model is located in the root's parent folder
   python src/llada/llada_runner.py
   ```

2. Specifying a custom model path:
   ```bash
   python src/llada/llada_runner.py --model_path "/path/to/your/model"
   ```

The model will start in interactive chat mode. You can:
- Type your message and press Enter to chat
- Type 'quit', 'exit', or 'q' to end the session

Note: Make sure you have the GSAI-ML/LLaDA-8B-Instruct model available before running.
The model will be loaded in bfloat16 precision.
