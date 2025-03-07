# Hugging Face GGUF Model Browser for Ollama

A command-line tool for searching, browsing, and downloading GGUF models from Hugging Face directly into Ollama.

## Overview

This tool simplifies the process of finding and using GGUF models with Ollama by providing an interactive terminal interface to:
- Search the Hugging Face Hub for models with GGUF files
- View model details including download counts and available quantization options
- See file sizes for different quantization levels
- Manage a download queue for batch processing
- Pull models directly into Ollama with proper tagging

## Features

- **Intelligent Model Search**: Search Hugging Face for models containing GGUF files
- **Smart Filtering**: Automatically filters results to show only models with GGUF files
- **Size Information**: Displays file sizes for different quantization options
- **Quantization Detection**: Automatically detects and groups different quantization levels
- **Sharded Model Support**: Identifies and properly handles multi-part sharded models
- **Download Queue**: Add multiple models to a queue for batch downloading
- **Concurrent Downloads**: Downloads multiple models simultaneously with controlled concurrency
- **Automatic Dependency Management**: Checks for and installs required Python packages
- **HF Authentication Support**: Optional Hugging Face token support for accessing gated models

## Prerequisites

- [Ollama](https://ollama.ai/) installed and in your PATH
- Python 3.6 or higher
- Internet connection to access Hugging Face API

## Installation

1. Clone or download this repository:
   ```
   git clone https://github.com/nickk024/ollama_tools.git
   cd ollama_tools
   ```

2. Run the script directly:
   ```
   python hf_ollama_browser.py
   ```

   The script will automatically check for and install required dependencies.

3. (Optional) For accessing gated models, set your Hugging Face token as an environment variable:
   ```
   export HF_TOKEN=your_huggingface_token
   ```

## Usage

### Basic Workflow

1. **Search for Models**:
   - Enter a search query (e.g., "llama" or "mistral")
   - Specify the maximum number of results to retrieve
   - The tool will filter for models containing GGUF files

2. **Select Models**:
   - Browse the list of models sorted by popularity
   - Use spacebar to select one or more models
   - Press Enter to confirm your selection

3. **Choose Quantization Levels**:
   - For each selected model, view available GGUF files with their sizes
   - Select a specific quantization level (e.g., Q4_K, Q5_K, Q8_0)
   - Alternatively, choose "No tag" to let Ollama select automatically
   - Or enter a custom tag for special cases

4. **Manage Download Queue**:
   - View your current download queue
   - Remove items if needed
   - Start the download process

5. **Download Models**:
   - The tool will download selected models concurrently
   - Real-time progress is displayed
   - Models are automatically imported into Ollama

### Example Session

```
======= GGUF Model Browser for Ollama =======

What would you like to do?
> Search for models
  View download queue
  Remove from download queue
  Download selected models
  Exit

Enter search query for Hugging Face models: mistral
Maximum number of results (default: 50): 20

Searching for models with query: mistral
Found 20 models. Checking for GGUF files...
Filtering for GGUF models...
Progress: [20/20]
Found 12 models with GGUF files.

Select models to download (spacebar to select, enter to confirm)
> [x] mistralai/Mistral-7B-v0.1-GGUF - 1234567 downloads
  [ ] TheBloke/Mistral-7B-Instruct-v0.2-GGUF - 987654 downloads
  [ ] TheBloke/Mistral-7B-v0.1-GGUF - 876543 downloads
  ...

Getting available GGUF files for mistralai/Mistral-7B-v0.1-GGUF...
Found 6 GGUF files:
1. mistral-7b-v0.1.Q4_K_M.gguf (4.1 GB)
2. mistral-7b-v0.1.Q5_K_M.gguf (4.8 GB)
3. mistral-7b-v0.1.Q6_K.gguf (5.5 GB)
4. mistral-7b-v0.1.Q8_0.gguf (7.2 GB)
...

Available quantization levels: Q4_K_M, Q5_K_M, Q6_K, Q8_0

Select quantization tag:
> Q4_K_M (4.1 GB)
  Q5_K_M (4.8 GB)
  Q6_K (5.5 GB)
  Q8_0 (7.2 GB)
  No tag (select GGUF file automatically)
  Custom tag
```

## How It Works

The tool uses the Hugging Face API to search for models and retrieve repository information. It then:

1. Filters models to find those containing GGUF files
2. Analyzes filenames to detect different quantization levels
3. Retrieves file sizes using HTTP HEAD requests
4. Groups files by quantization level and identifies sharded models
5. Uses the `ollama pull` command to download selected models with proper tagging

### Key Components

- **ModelDownloader**: Main class handling model search, filtering, and downloading
- **Dependency Management**: Automatic checking and installation of required packages
- **Terminal UI**: Interactive menus for model selection and queue management
- **Concurrent Processing**: Parallel model filtering and controlled concurrent downloads

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for hosting the model repository
- [Ollama](https://ollama.ai/) for making local LLM deployment easy
