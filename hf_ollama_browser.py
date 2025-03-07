#!/usr/bin/env python3
import importlib.util
import subprocess
import sys
import os
import re

def check_and_install_dependencies():
    """Check if required packages are installed, and install them if they're not."""
    required_packages = ['huggingface_hub', 'simple_term_menu', 'requests']
    
    print("Checking dependencies...")
    packages_to_install = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            packages_to_install.append(package)
    
    if packages_to_install:
        print(f"Installing missing dependencies: {', '.join(packages_to_install)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages_to_install)
            print("Dependencies installed successfully!")
            
            # Re-import the modules for Python to recognize them
            for package in packages_to_install:
                if package == 'simple_term_menu':
                    globals()['TerminalMenu'] = __import__('simple_term_menu').TerminalMenu
                else:
                    __import__(package)
                    
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
            print("Please install the following packages manually:")
            for package in packages_to_install:
                print(f"  pip install {package}")
            sys.exit(1)
    else:
        print("All dependencies are already installed.")

# Check and install dependencies before importing them
check_and_install_dependencies()

# Now import the required packages
import asyncio
import concurrent.futures
from huggingface_hub import HfApi, hf_hub_url
import time
import requests

# Import TerminalMenu only if it wasn't already imported in the dependency check
if 'TerminalMenu' not in globals():
    from simple_term_menu import TerminalMenu

class ModelDownloader:
    def __init__(self):
        self.api = HfApi()
        # Add token if you have one
        self.token = os.environ.get("HF_TOKEN")  # Set this environment variable with your token
        self.download_queue = []  # List of (model_id, tag) tuples
        
    def search_models(self, query, limit=50):
        """Search for models on Hugging Face."""
        print(f"\nSearching for models with query: {query}")
        models = list(self.api.list_models(search=query, limit=limit))
        print(f"Found {len(models)} models. Checking for GGUF files...")
        return models
    
    def has_gguf_files(self, model_id):
        """Check if a model has GGUF files."""
        try:
            files = self.api.list_repo_files(model_id)
            return any(file.endswith('.gguf') for file in files)
        except Exception:
            return False
    
    def get_gguf_files(self, model_id):
        """Get all GGUF files for a model with their sizes."""
        try:
            # First try using the API to get file info
            print(f"DEBUG: Getting repo info for {model_id}")
            try:
                # Try to get repo info which might include file sizes
                repo_info = self.api.repo_info(repo_id=model_id, files_metadata=True)
                if hasattr(repo_info, 'siblings') and repo_info.siblings:
                    print(f"DEBUG: Got repo info with {len(repo_info.siblings)} siblings")
                    gguf_files = []
                    for sibling in repo_info.siblings:
                        if sibling.rfilename.endswith('.gguf'):
                            print(f"DEBUG: Found GGUF file: {sibling.rfilename}")
                            if hasattr(sibling, 'size'):
                                print(f"DEBUG: Got size from API: {sibling.size} bytes ({sibling.size/(1024*1024*1024):.2f} GB)")
                                gguf_files.append({'path': sibling.rfilename, 'size': sibling.size})
                            else:
                                print(f"DEBUG: No size info for {sibling.rfilename}")
                                gguf_files.append({'path': sibling.rfilename, 'size': None})
                    
                    if gguf_files:
                        print(f"DEBUG: Successfully got {len(gguf_files)} GGUF files with sizes from API")
                        return gguf_files
                    else:
                        print("DEBUG: No GGUF files found in repo info, falling back to list_repo_files")
                else:
                    print("DEBUG: No siblings in repo info, falling back to list_repo_files")
            except Exception as e:
                print(f"DEBUG: Error getting repo info: {str(e)}, falling back to list_repo_files")
            
            # Fallback to the original method
            files = self.api.list_repo_files(model_id)
            gguf_files = []
            
            gguf_file_count = len([f for f in files if f.endswith('.gguf')])
            print(f"DEBUG: Found {gguf_file_count} GGUF files for {model_id}")
            
            for file in files:
                if file.endswith('.gguf'):
                    # Try direct URL format first
                    try:
                        direct_url = f"https://huggingface.co/{model_id}/resolve/main/{file}"
                        print(f"DEBUG: Requesting HEAD for direct URL: {direct_url}")
                        
                        # Add token if available
                        headers = {}
                        if self.token:
                            headers["Authorization"] = f"Bearer {self.token}"
                            print("DEBUG: Using authentication token")
                        
                        response = requests.head(direct_url, headers=headers)
                        print(f"DEBUG: Response status: {response.status_code}")
                        print(f"DEBUG: Response headers: {dict(response.headers)}")
                        
                        if response.status_code == 200 and 'Content-Length' in response.headers:
                            size_bytes = int(response.headers['Content-Length'])
                            print(f"DEBUG: Got size: {size_bytes} bytes ({size_bytes/(1024*1024*1024):.2f} GB)")
                            gguf_files.append({'path': file, 'size': size_bytes})
                            continue  # Skip the next attempt if this one worked
                        else:
                            print(f"DEBUG: No Content-Length in headers or non-200 status for direct URL")
                    except Exception as e:
                        print(f"DEBUG: Exception getting size for direct URL {file}: {str(e)}")
                    
                    # If direct URL didn't work, try hf_hub_url
                    try:
                        url = hf_hub_url(model_id, file)
                        print(f"DEBUG: Requesting HEAD for hf_hub_url: {url}")
                        
                        # Add token if available
                        headers = {}
                        if self.token:
                            headers["Authorization"] = f"Bearer {self.token}"
                        
                        response = requests.head(url, headers=headers)
                        print(f"DEBUG: Response status: {response.status_code}")
                        print(f"DEBUG: Response headers: {dict(response.headers)}")
                        
                        if response.status_code == 200 and 'Content-Length' in response.headers:
                            size_bytes = int(response.headers['Content-Length'])
                            print(f"DEBUG: Got size: {size_bytes} bytes ({size_bytes/(1024*1024*1024):.2f} GB)")
                            gguf_files.append({'path': file, 'size': size_bytes})
                        else:
                            print(f"DEBUG: No Content-Length in headers or non-200 status for hf_hub_url")
                            # If we can't get the size, still include the file without size
                            gguf_files.append({'path': file, 'size': None})
                    except Exception as e:
                        print(f"DEBUG: Exception getting size for hf_hub_url {file}: {str(e)}")
                        # If we can't get the size, still include the file without size
                        gguf_files.append({'path': file, 'size': None})
            
            return gguf_files
        except Exception as e:
            print(f"Error getting files for {model_id}: {e}")
            return []
    
    def extract_quantization_levels(self, gguf_files):
        """Extract quantization levels from GGUF filenames with their sizes."""
        # Common quantization patterns like Q4_K, Q5_K, Q6_K, etc.
        quant_pattern = re.compile(r'[QF]\d+[_0-9A-Za-z]*')
        
        # Dictionary to store quantization levels and their sizes
        quant_sizes = {}
        
        for file in gguf_files:
            # Get filename without path
            filename = os.path.basename(file['path'])
            size = file['size']
            
            matches = quant_pattern.findall(filename)
            if matches:
                for match in matches:
                    if match not in quant_sizes:
                        quant_sizes[match] = size
        
        return quant_sizes
    
    def filter_gguf_models(self, models, max_workers=10):
        """Filter models that have GGUF files using parallel processing."""
        gguf_models = []
        
        print("Filtering for GGUF models...")
        count = 0
        total = len(models)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.has_gguf_files, model.id): model for model in models}
            for future in concurrent.futures.as_completed(futures):
                model = futures[future]
                count += 1
                print(f"\rProgress: [{count}/{total}]", end='', flush=True)
                if future.result():
                    gguf_models.append(model)
        
        print(f"\nFound {len(gguf_models)} models with GGUF files.")
        return gguf_models
    
    async def pull_model(self, model_id, tag=None):
        """Pull a model using ollama with real-time output."""
        model_spec = f"hf.co/{model_id}"
        if tag:
            model_spec += f":{tag}"
            
        cmd = f"ollama pull {model_spec}"
        print(f"\nRunning: {cmd}")
        
        # Use run_in_executor to directly pipe output to the terminal
        loop = asyncio.get_event_loop()
        try:
            # This approach directly shows live output
            await loop.run_in_executor(None, lambda: subprocess.run(
                cmd, 
                shell=True, 
                check=True,
                text=True
            ))
            print(f"Successfully pulled {model_spec}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to pull {model_spec}: {e}")
            return False
    
    async def pull_models(self):
        """Pull multiple models with controlled concurrency."""
        if not self.download_queue:
            print("No models selected for download.")
            return
            
        print(f"\nDownloading {len(self.download_queue)} models...")
        
        # Use a semaphore to limit concurrent downloads (to avoid overwhelming system)
        semaphore = asyncio.Semaphore(2)  # Allow 2 concurrent downloads
        
        async def pull_with_semaphore(model_id, tag):
            async with semaphore:
                return await self.pull_model(model_id, tag)
        
        tasks = [pull_with_semaphore(model_id, tag) for model_id, tag in self.download_queue]
        results = await asyncio.gather(*tasks)
        
        # Clear the download queue after downloading
        self.download_queue = []
        
        successes = results.count(True)
        failures = len(results) - successes
        
        print(f"\nDownload summary: {successes} successful, {failures} failed")

    def add_to_queue(self, model_id, tag=None):
        """Add a model to the download queue."""
        self.download_queue.append((model_id, tag))
        print(f"Added to download queue: {model_id}" + (f":{tag}" if tag else ""))
    
    def show_download_queue(self):
        """Show the current download queue."""
        if not self.download_queue:
            print("\nDownload queue is empty.")
            return
            
        print("\nCurrent download queue:")
        for i, (model_id, tag) in enumerate(self.download_queue, 1):
            model_spec = f"hf.co/{model_id}"
            if tag:
                model_spec += f":{tag}"
            print(f"{i}. {model_spec}")

    def remove_from_queue(self, indices):
        """Remove models from the download queue by indices."""
        if not indices:
            return
            
        # Sort indices in reverse order to avoid index issues when removing
        for index in sorted(indices, reverse=True):
            if 0 <= index < len(self.download_queue):
                removed = self.download_queue.pop(index)
                print(f"Removed from queue: {removed[0]}" + (f":{removed[1]}" if removed[1] else ""))


# Test function to directly test a HEAD request to a specific file
def test_head_request():
    model_id = "TheBloke/Llama-2-7B-Chat-GGUF"  # Example model with GGUF files
    file = "llama-2-7b-chat.Q4_K_M.gguf"  # Example GGUF file
    url = hf_hub_url(model_id, file)
    print(f"\nTesting HEAD request for {url}")
    try:
        # Try with and without authentication
        print("Without authentication:")
        response = requests.head(url)
        print(f"Status code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        if 'Content-Length' in response.headers:
            size_bytes = int(response.headers['Content-Length'])
            print(f"Size: {size_bytes} bytes ({size_bytes/(1024*1024*1024):.2f} GB)")
        else:
            print("No Content-Length header found")
        
        # Try with a different URL format
        print("\nTrying direct URL format:")
        direct_url = f"https://huggingface.co/{model_id}/resolve/main/{file}"
        response = requests.head(direct_url)
        print(f"Status code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        if 'Content-Length' in response.headers:
            size_bytes = int(response.headers['Content-Length'])
            print(f"Size: {size_bytes} bytes ({size_bytes/(1024*1024*1024):.2f} GB)")
        else:
            print("No Content-Length header found")
        
    except Exception as e:
        print(f"Exception: {str(e)}")

def main():
    # Run the test function first
    print("\n======= Testing HEAD Request =======")
    test_head_request()
    
    # Helper function to format file sizes
    def format_size(size_bytes):
        if size_bytes is None:
            return "unknown size"
        
        # Convert to MB
        size_mb = size_bytes / (1024 * 1024)
        
        # Format as MB or GB as appropriate
        if size_mb > 1000:
            return f"{size_mb/1024:.1f} GB"
        else:
            return f"{size_mb:.1f} MB"
    
    downloader = ModelDownloader()
    print("\n======= GGUF Model Downloader for Ollama =======")
    
    while True:
        # Main menu
        main_options = [
            "Search for models",
            "View download queue",
            "Remove from download queue",
            "Download selected models",
            "Exit"
        ]
        
        main_menu = TerminalMenu(
            main_options,
            title="\nWhat would you like to do?",
            cycle_cursor=True,
            clear_screen=True
        )
        
        main_selection = main_menu.show()
        
        if main_selection == 0:  # Search for models
            query = input("\nEnter search query for Hugging Face models: ")
            if not query:
                continue
                
            try:
                limit = int(input("Maximum number of results (default: 50): ") or "50")
            except ValueError:
                limit = 50
            
            # Search for models
            models = downloader.search_models(query, limit)
            
            if not models:
                print("No models found.")
                input("Press Enter to continue...")
                continue
            
            # Filter models with GGUF files
            gguf_models = downloader.filter_gguf_models(models)
            
            if not gguf_models:
                print("No models with GGUF files found.")
                input("Press Enter to continue...")
                continue
            
            # Sort models by downloads (most popular first)
            gguf_models.sort(key=lambda x: x.downloads, reverse=True)
            
            # Prepare model choices for selection
            model_options = [
                f"{model.id} - {model.downloads} downloads"
                for model in gguf_models
            ]
            
            # Multi-select menu for models
            model_menu = TerminalMenu(
                model_options,
                title="\nSelect models to download (spacebar to select, enter to confirm)",
                multi_select=True,
                show_multi_select_hint=True,
                cycle_cursor=True,
                clear_screen=True
            )
            
            model_selections = model_menu.show()
            
            if model_selections is None:
                continue
                
            # Get selected model IDs
            selected_models = [gguf_models[i].id for i in model_selections]
            
            # Ask for quantization tags for each selected model
            for model_id in selected_models:
                print(f"\nGetting available GGUF files for {model_id}...")
                
                # Get all GGUF files in the model repository
                gguf_files = downloader.get_gguf_files(model_id)
                
                # Display available files with sizes
                print(f"Found {len(gguf_files)} GGUF files:")
                for i, file in enumerate(gguf_files, 1):
                    size_str = format_size(file['size']) if file['size'] is not None else "unknown size"
                    print(f"{i}. {os.path.basename(file['path'])} ({size_str})")
                
                # Extract quantization levels from filenames
                quant_levels = downloader.extract_quantization_levels(gguf_files)
                
                # Prepare options for menu
                if quant_levels:
                    print(f"\nAvailable quantization levels: {', '.join(quant_levels.keys())}")
                    
                    # Create menu options with sizes
                    tag_options = ["No tag (select GGUF file automatically)"]
                    quant_options = []
                    
                    for quant, size in quant_levels.items():
                        size_str = format_size(size) if size is not None else "unknown size"
                        quant_options.append(f"{quant} ({size_str})")
                    
                    tag_options.extend(quant_options)
                    tag_options.append("Custom tag")
                else:
                    tag_options = ["No tag (select GGUF file automatically)", "Custom tag"]
                
                tag_menu = TerminalMenu(
                    tag_options,
                    title="\nSelect quantization tag:",
                    cycle_cursor=True,
                    clear_screen=False
                )
                
                tag_selection = tag_menu.show()
                
                if tag_selection == 0:  # No tag
                    downloader.add_to_queue(model_id, None)
                elif tag_selection == len(tag_options) - 1:  # Custom tag
                    custom_tag = input("Enter custom tag: ")
                    downloader.add_to_queue(model_id, custom_tag if custom_tag else None)
                else:
                    # Selected one of the extracted quantization levels
                    # Extract just the quantization level (without the size) from the selected option
                    selected_option = tag_options[tag_selection]
                    # The format is "Q4_K (2.1 GB)" - extract just the "Q4_K" part
                    quant = selected_option.split(" (")[0]
                    downloader.add_to_queue(model_id, quant)
                
        elif main_selection == 1:  # View download queue
            downloader.show_download_queue()
            input("\nPress Enter to continue...")
            
        elif main_selection == 2:  # Remove from download queue
            if not downloader.download_queue:
                print("\nDownload queue is empty.")
                input("Press Enter to continue...")
                continue
                
            # Prepare removal options
            remove_options = [
                f"{i+1}. {model_id}" + (f":{tag}" if tag else "")
                for i, (model_id, tag) in enumerate(downloader.download_queue)
            ]
            
            remove_menu = TerminalMenu(
                remove_options,
                title="\nSelect models to remove (spacebar to select, enter to confirm)",
                multi_select=True,
                show_multi_select_hint=True,
                cycle_cursor=True,
                clear_screen=True
            )
            
            remove_selections = remove_menu.show()
            
            if remove_selections is not None:
                downloader.remove_from_queue(remove_selections)
                input("Press Enter to continue...")
            
        elif main_selection == 3:  # Download selected models
            asyncio.run(downloader.pull_models())
            input("\nPress Enter to continue...")
            
        elif main_selection == 4 or main_selection is None:  # Exit
            print("\nExiting program. Goodbye!")
            break

if __name__ == "__main__":
    main()