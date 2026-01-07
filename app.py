import os
import requests
import threading
import subprocess
from PIL import Image
import os.path as osp
from pathlib import Path
from typing import Optional

import torch

# import spaces
import gradio as gr
from huggingface_hub import snapshot_download

import viz_utils

# Hugging Face model repository
HF_MODEL_REPO = "ByteDance-Seed/cryofm-v2"

# Model cache directory (will be created in the current directory)
MODEL_CACHE_DIR = osp.join(osp.dirname(__file__), "models_cache")

# Global cache for model directories
_model_dir_cache = {}

# Track download status and locks for thread safety
_download_status = {
    "cryofm2-emhancer": {"status": "pending", "error": None},
    "cryofm2-emready": {"status": "pending", "error": None},
}
_download_locks = {
    "cryofm2-emhancer": threading.Lock(),
    "cryofm2-emready": threading.Lock(),
}


def download_model(model_name: str) -> str:
    """
    Download model from Hugging Face and return the local path.
    Thread-safe version that handles concurrent downloads.
    
    Args:
        model_name: Model name (e.g., "cryofm2-emhancer", "cryofm2-emready")
    
    Returns:
        Local path to the model directory
    """
    # Get lock for this model to ensure thread safety
    lock = _download_locks.get(model_name, threading.Lock())
    
    with lock:
        # Check cache first
        if model_name in _model_dir_cache:
            cached_path = _model_dir_cache[model_name]
            if osp.exists(cached_path) and osp.exists(osp.join(cached_path, "model.safetensors")):
                return cached_path
        
        # Determine the local directory for this model
        # The repo structure is: ByteDance-Seed/cryofm-v2/cryofm2-emhancer/ or cryofm2-emready/
        local_dir = osp.join(MODEL_CACHE_DIR, model_name)
        
        # Create cache directory if it doesn't exist
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        
        # Check if already downloaded (double-check after acquiring lock)
        if osp.exists(local_dir) and osp.exists(osp.join(local_dir, "model.safetensors")):
            _model_dir_cache[model_name] = local_dir
            return local_dir
        
        # Download from Hugging Face
        # The model files are in a subdirectory of the repo
        print(f"Downloading model {model_name} from Hugging Face...")
        try:
            # Download the entire repo first, then extract the model subdirectory
            # This ensures we get the correct directory structure
            repo_cache_dir = osp.join(MODEL_CACHE_DIR, "repo")
            downloaded_path = snapshot_download(
                repo_id=HF_MODEL_REPO,
                local_dir=repo_cache_dir,
                local_dir_use_symlinks=False,  # Use actual files, not symlinks
            )
            
            # Find the model directory in the downloaded repo
            # Structure should be: repo_cache_dir/cryofm-v2/model_name/
            # or: downloaded_path/model_name/
            model_dir = None
            
            # Try different possible paths
            possible_paths = [
                osp.join(downloaded_path, model_name),
                osp.join(repo_cache_dir, model_name),
                osp.join(repo_cache_dir, "cryofm-v2", model_name),
                osp.join(MODEL_CACHE_DIR, "cryofm-v2", model_name),
            ]
            
            for path in possible_paths:
                if osp.exists(path) and osp.exists(osp.join(path, "model.safetensors")):
                    model_dir = path
                    break
            
            # If not found, search recursively
            if model_dir is None:
                for root, dirs, files in os.walk(downloaded_path):
                    if "model.safetensors" in files and "config.yaml" in files:
                        # Check if this directory name matches the model name
                        if model_name in root or osp.basename(root) == model_name:
                            model_dir = root
                            break
            
            if model_dir is None or not osp.exists(osp.join(model_dir, "model.safetensors")):
                raise FileNotFoundError(
                    f"Could not find model files for {model_name}. "
                    f"Searched in {downloaded_path}. "
                    f"Tried paths: {possible_paths}"
                )
            
            # Copy to a cleaner location for easier access
            if model_dir != local_dir:
                os.makedirs(local_dir, exist_ok=True)
                import shutil
                for file in ["model.safetensors", "config.yaml"]:
                    src = osp.join(model_dir, file)
                    dst = osp.join(local_dir, file)
                    if osp.exists(src) and not osp.exists(dst):
                        shutil.copy2(src, dst)
                model_dir = local_dir
            
            print(f"Model {model_name} downloaded successfully to {model_dir}")
            # Cache the path (still within lock)
            _model_dir_cache[model_name] = model_dir
            return model_dir
        except Exception as e:
            # If download fails, check if files exist locally
            if osp.exists(osp.join(local_dir, "model.safetensors")):
                _model_dir_cache[model_name] = local_dir
                return local_dir
            error_msg = f"Failed to download model {model_name} from Hugging Face: {str(e)}"
            print(f"Error: {error_msg}")
            # Update download status
            if model_name in _download_status:
                _download_status[model_name]["status"] = "error"
                _download_status[model_name]["error"] = str(e)
            raise RuntimeError(error_msg)


def download_example_file(url: str, file_path: str) -> bool:
    """
    Download a file from URL if it doesn't exist.
    
    Args:
        url: URL to download from
        file_path: Local file path to save to
    
    Returns:
        True if file exists or was downloaded successfully, False otherwise
    """
    # If file already exists, no need to download
    if os.path.exists(file_path):
        return True
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        print(f"Downloading {os.path.basename(file_path)} from {url}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Write file in chunks to handle large files
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Successfully downloaded {os.path.basename(file_path)}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {os.path.basename(file_path)}: {e}")
        return False


def download_examples():
    """Download example files if they don't exist"""
    examples_dir = os.path.join(os.path.dirname(__file__), "examples")
    example_file_1 = os.path.join(examples_dir, "emd_35143_half_map_1.mrc")
    example_file_2 = os.path.join(examples_dir, "emd_35143_half_map_2.mrc")
    
    example_url_1 = "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/cryofm/images/emd_35143_half_map_1.mrc"
    example_url_2 = "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/cryofm/images/emd_35143_half_map_2.mrc"
    
    print("Checking example files...")
    success_1 = download_example_file(example_url_1, example_file_1)
    success_2 = download_example_file(example_url_2, example_file_2)
    
    if success_1 and success_2:
        print("✓ All example files are ready")
    else:
        print("⚠ Some example files failed to download, but the app can still run")
    
    return success_1 and success_2


def preload_models():
    """Preload models in background threads when app starts"""
    def download_emhancer():
        try:
            print("Starting preload of cryofm2-emhancer...")
            _download_status["cryofm2-emhancer"]["status"] = "downloading"
            download_model("cryofm2-emhancer")
            _download_status["cryofm2-emhancer"]["status"] = "ready"
            print("✓ cryofm2-emhancer preloaded successfully")
        except Exception as e:
            _download_status["cryofm2-emhancer"]["status"] = "error"
            _download_status["cryofm2-emhancer"]["error"] = str(e)
            print(f"✗ Failed to preload cryofm2-emhancer: {e}")
    
    def download_emready():
        try:
            print("Starting preload of cryofm2-emready...")
            _download_status["cryofm2-emready"]["status"] = "downloading"
            download_model("cryofm2-emready")
            _download_status["cryofm2-emready"]["status"] = "ready"
            print("✓ cryofm2-emready preloaded successfully")
        except Exception as e:
            _download_status["cryofm2-emready"]["status"] = "error"
            _download_status["cryofm2-emready"]["error"] = str(e)
            print(f"✗ Failed to preload cryofm2-emready: {e}")
    
    # Start downloading both models in parallel
    thread1 = threading.Thread(target=download_emhancer, daemon=True)
    thread2 = threading.Thread(target=download_emready, daemon=True)
    thread1.start()
    thread2.start()
    print("Model preloading started in background...")


# @spaces.GPU
def fn(model_style, input_file, input_path1, input_path2, mask_path, lamb_base, num_timesteps, op):
    cmd = ["cfm", "enhance", "--bf16", ]
    
    # Auto-detect GPU count and add --num_processes parameter
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Detected {num_gpus} GPU(s)")
        cmd.extend(["--num_processes", str(num_gpus)])

    if model_style == "emhancer":
        model_name = "cryofm2-emhancer"
        cli_output_tag = "1"
    elif model_style == "emready":
        model_name = "cryofm2-emready"
        cli_output_tag = "0"
        cmd.extend(["--cfg-weight", "0.5"])
    else:
        raise ValueError(f"Invalid model style: {model_style}")
    
    # Check download status and provide user feedback
    if model_name in _download_status:
        status = _download_status[model_name]["status"]
        if status == "downloading":
            print(f"Model {model_name} is still downloading, please wait...")
        elif status == "error":
            error = _download_status[model_name]["error"]
            print(f"Warning: Previous download failed: {error}. Retrying...")
    
    # Download model from Hugging Face if not already cached
    # This will be fast if already downloaded, or will wait if still downloading
    cli_model_dir = download_model(model_name)
    cmd.extend(["--model-dir", cli_model_dir, "--output-tag", cli_output_tag])

    if input_file is not None:
        cli_input_path = input_file.name if hasattr(input_file, 'name') else str(input_file)
    else:
        raise ValueError(f"Invalid input file: {input_file}")
    cmd.extend(["-i", cli_input_path])

    if mask_path is not None:
        cli_mask_path = mask_path.name if hasattr(mask_path, 'name') else str(mask_path)
        cmd.extend(["--mask-path", cli_mask_path, "--bbox"])

    if num_timesteps is not None:
        cmd.extend(["--num-timesteps", str(num_timesteps)])
    
    if input_path1 is not None and input_path2 is not None:
        do_posterior_sampling = True
        cmd.extend(["--norm-grad", "--use-lamb-w"])
        
        cli_input_path1 = input_path1.name if hasattr(input_path1, 'name') else str(input_path1)
        cli_input_path2 = input_path2.name if hasattr(input_path2, 'name') else str(input_path2)
        cmd.extend(["-i1", cli_input_path1, "-i2", cli_input_path2])

        lamb_base = lamb_base * 1000
        cmd.extend(["--lamb-base", str(lamb_base)])

        if op in ("denoise", "non-uniform"):
            cmd.extend(["--op", op])
        else:
            raise ValueError(f"Invalid operator: {op}")
    else:
        do_posterior_sampling = False

    cmd.extend(["-o", "./output"])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__)
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {result.stderr}")

    # Get output directory (relative to script directory)
    output_dir = osp.join(os.path.dirname(__file__), "output")
    
    # Get input file stem (filename without extension)
    if hasattr(input_file, 'name'):
        input_path = input_file.name
    else:
        input_path = str(input_file)
    input_stem = Path(input_path).stem
    
    # Build output file path
    output_file = osp.join(output_dir, f"{input_stem}_external_reconstruct.mrc")
    
    # Verify output file exists
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Output file not found in {output_dir}")
    
    return output_file

CSS = """
.gradio-container { background: #ffffff !important; }

.small-slider img { max-height: 260px !important; object-fit: contain !important; }
.small-slider .image-container, 
.small-slider .wrap, 
.small-slider .canvas, 
.small-slider .image { height: 260px !important; }
"""

description = """
# CryoFM: Cryo-EM Foundation Model

> This is a demo for enhancement application of CryoFM2 on cryo-EM density maps.

**CryoFM** is a foundation model for cryo-electron microscopy (cryo-EM) density map. 
This interactive demo allows you to enhance your cryo-EM density maps using two specialized styles:

- **EMhancer**: Targeted for LocScale sharpened maps, providing enhanced resolution and clarity.
- **EMReady**: Targeted for structure-simulated maps, offering improved structural details.
"""

def create_interface():
    """Create and return the Gradio interface"""
    with gr.Blocks(title="CryoFM - Density Map Enhancement") as demo:
        gr.Markdown(description)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### EMhancer style post-processing")
            with gr.Column():
                gr.Markdown("### EMReady style post-processing")

        with gr.Row():
            with gr.Column():
                before_image = Image.open("assets/6HRA_input.png")
                after_image = Image.open("assets/6HRA_emhancer.png")

                slider = gr.ImageSlider(
                    label="EMhancer style post-processing",
                    value=(before_image, after_image),
                    elem_classes=["small-slider"]
                )

            with gr.Column():
                before_image = Image.open("assets/emd_9610_input.png")
                after_image = Image.open("assets/emd_9610_emhancer.png")

                slider = gr.ImageSlider(
                    label="EMhancer style post-processing",
                    value=(before_image, after_image),
                    elem_classes=["small-slider"]
                )
            
            with gr.Column():
                before_image = Image.open("assets/7KDT_input.png")
                after_image = Image.open("assets/7KDT_emready.png")

                slider = gr.ImageSlider(
                    label="EMReady style post-processing",
                    value=(before_image, after_image),
                    elem_classes=["small-slider"]
                )
                
            with gr.Column():
                before_image = Image.open("assets/emd_0560_input.png")
                after_image = Image.open("assets/emd_0560_emready.png")

                slider = gr.ImageSlider(
                    label="EMReady style post-processing",
                    value=(before_image, after_image),
                    elem_classes=["small-slider"]
                )
        
        # State to store loaded volumes
        in_vol_state = gr.State(None)
        in_apix_state = gr.State(None)
        out_vol_state = gr.State(None)
        out_apix_state = gr.State(None)
        out_path_state = gr.State(None)
        
        gr.Markdown("""<br><br><br><br>
        ## Play with CryoFM""")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1. Select model output style")

                model_style = gr.Radio(
                    choices=[
                        ("EMhancer (LocScale sharpened map)", "emhancer"),
                        ("EMReady (Structure simulated map)", "emready"),],
                        label="Model Style",
                        value="emhancer", )

                gr.Markdown("### 2. Upload Input Volume")
                gr.Markdown("Supported formats: `.mrc`, `.map`")

                input_file = gr.File(label="Input .mrc or .map", file_types=[".mrc", ".map"])

                gr.Markdown("### 3. (Optional) Upload additional inputs")

                with gr.Accordion("Additional Inputs", open=False):
                    input_path1 = gr.File(label="input_path1")
                    input_path2 = gr.File(label="input_path2")
                    mask_path = gr.File(label="mask_path")
                
                # Examples: two types - single input and triple input
                _examples_dir = os.path.join(os.path.dirname(__file__), "examples")
                _example_file_1 = os.path.join(_examples_dir, "emd_35143_half_map_1.mrc")
                _example_file_2 = os.path.join(_examples_dir, "emd_35143_half_map_2.mrc")
                
                # Example files should be downloaded before interface creation
                gr.Markdown("#### 📋 Examples")
                
                # Single input example (only input_file)
                gr.Markdown("**Single Input Mode** (only input_file):")
                gr.Markdown("*Note: With a single file, the model performs naive inference only. However, you can also provide additional half maps (input_path1 and input_path2) to control the inference process through posterior sampling.*")
                if os.path.exists(_example_file_1):
                    gr.Examples(
                        examples=[[_example_file_1, None, None]],
                        inputs=[input_file, input_path1, input_path2],
                        label="",
                        cache_examples=False,
                    )
                
                # Triple input example (input_file + input_path1 + input_path2)
                gr.Markdown("**Triple Input Mode** (input_file + input_path1 + input_path2):")
                gr.Markdown("*This mode enables posterior sampling with half maps for controlled inference.*")
                if os.path.exists(_example_file_1) and os.path.exists(_example_file_2):
                    gr.Examples(
                        examples=[[_example_file_1, _example_file_1, _example_file_2]],
                        inputs=[input_file, input_path1, input_path2],
                        label="",
                        cache_examples=False,
                    )
                
                gr.Markdown("### 4. (Optional) Adjust hyper-parameters")

                num_timesteps = gr.Slider(minimum=50, maximum=1000, value=200, step=50, label="Number of timesteps")
                lamb_base = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="Lambda base")
                op = gr.Radio(choices=["denoise", "non-uniform"], value="denoise", label="forward operator")
                
                with gr.Row():
                    btn = gr.Button("🚀 Run Processing", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear Output", variant="secondary")

            with gr.Column():
                gr.Markdown("### Input Preview")
                with gr.Group():
                    with gr.Accordion("Preview Settings", open=False):
                        with gr.Row():
                            iso_pct = gr.Slider(70, 99.9, value=98.5, step=0.1, label="Iso Percentile")
                            smooth_sigma = gr.Slider(0.0, 2.5, value=0.4, step=0.1, label="Smoothing σ")
                    
                    input_viz = gr.Model3D(clear_color=[0.97, 0.98, 0.99, 1.0], label="Input 3D Visualization", )
                    input_info = gr.Markdown("Metadata: -")

                gr.Markdown("### Output Preview")

                with gr.Group():
                    with gr.Accordion("Preview Settings", open=False):
                        with gr.Row():
                            out_iso_pct = gr.Slider(70, 99.9, value=98.5, step=0.1, label="Iso Percentile")
                            out_smooth_sigma = gr.Slider(0.0, 2.5, value=0.4, step=0.1, label="Smoothing σ")
                    output_viz = gr.Model3D(clear_color=[0.97, 0.98, 0.99, 1.0], label="Output 3D Visualization")
                    output_info = gr.Markdown("Metadata: -")
                

                gr.Markdown("### Output Download")
                output_file = gr.File(label="Output Download")

        # --- Events ---
        def on_upload(file_obj, iso, sigma):
            if file_obj is None:
                return None, None, "Metadata: -", None
            vol, apix, stats = viz_utils.load_mrc(file_obj.name)
            # Use default max_dim=256, outline=0.012
            glb, info = viz_utils.build_glb(vol, apix, iso, sigma, 256, 1, 0.012)
            return vol, apix, f"**Metadata:** {info}", glb

        input_file.change(
            on_upload, 
            inputs=[input_file, iso_pct, smooth_sigma], 
            outputs=[in_vol_state, in_apix_state, input_info, input_viz]
        )

        # Re-build input preview when sliders change
        preview_inputs = [in_vol_state, in_apix_state, iso_pct, smooth_sigma]
        for comp in [iso_pct, smooth_sigma]:
            comp.release(
                lambda v, a, i, s: viz_utils.build_glb(v, a, i, s, 256, 1, 0.012) if v is not None else (None, "Metadata: -"),
                inputs=preview_inputs,
                outputs=[input_viz, input_info]
            )

        def process_and_viz(style, file, p1, p2, m, lb, nt, op, iso, sigma):
            res_path = fn(style, file, p1, p2, m, lb, nt, op)
            if res_path:
                vol, apix, stats = viz_utils.load_mrc(res_path)
                # Use output-specific iso/sigma
                glb, info = viz_utils.build_glb(vol, apix, iso, sigma, 256, 1, 0.012)
                return vol, apix, glb, f"**Metadata:** {info}", res_path, res_path
            return None, None, None, "Metadata: -", None, None

        btn.click(
            process_and_viz, 
            inputs=[
                model_style, input_file, 
                input_path1, input_path2, mask_path, 
                lamb_base, num_timesteps, op,
                out_iso_pct, out_smooth_sigma
            ], 
            outputs=[out_vol_state, out_apix_state, output_viz, output_info, output_file, out_path_state]
        )

        def clear_output():
            """Clear output folder and reset all output components"""
            output_dir = osp.join(os.path.dirname(__file__), "output")
            if osp.exists(output_dir):
                for file in Path(output_dir).glob("*"):
                    if file.is_file():
                        file.unlink()
            return None, None, None, "Metadata: -", None, None

        clear_btn.click(
            clear_output,
            inputs=[],
            outputs=[out_vol_state, out_apix_state, output_viz, output_info, output_file, out_path_state]
        )

        # Re-build output preview when output sliders change
        output_preview_inputs = [out_vol_state, out_apix_state, out_iso_pct, out_smooth_sigma]
        for comp in [out_iso_pct, out_smooth_sigma]:
            comp.release(
                lambda v, a, i, s: viz_utils.build_glb(v, a, i, s, 256, 1, 0.012) if v is not None else (None, "Metadata: -"),
                inputs=output_preview_inputs,
                outputs=[output_viz, output_info]
            )
    
    return demo

if __name__ == "__main__":
    # Preload models in background before launching the app
    print("=" * 60)
    print("Starting CryoFM2 Demo")
    print("=" * 60)
    
    # Download example files first (before creating interface)
    download_examples()
    print("=" * 60)
    
    # Create interface after examples are downloaded
    demo = create_interface()
    
    # Preload models in background
    preload_models()
    print("=" * 60)
    print("Launching Gradio interface...")
    print("Note: Models are downloading in the background.")
    print("You can start using the interface, but first inference may wait for download.")
    print("=" * 60)
    
    demo.launch(
        server_name="0.0.0.0" if os.getenv("SPACE_ID") else "127.0.0.1",
        server_port=7860,
        share=False,
        css=CSS
    )
else:
    # When imported as a module, create interface without downloading examples
    # (examples should be downloaded separately or already exist)
    demo = create_interface()
