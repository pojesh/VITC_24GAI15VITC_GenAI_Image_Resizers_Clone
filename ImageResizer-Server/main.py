import io
import base64
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
import numpy as np
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from huggingface_hub import hf_hub_download
from contextlib import asynccontextmanager
import logging

import os
import sys
import warnings
import cv2
from cv2.detail import resultRoi
import datetime
import uuid
from pathlib import Path
from PIL import ImageEnhance, ImageFilter, ImageOps
import random
import torchvision.transforms as T
import gc

#sdxl
from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

# --- Compatibility and Path Fixes ---
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Error fix for torch transformer for realesrgan
def apply_realesrgan_fix():
    import types
    sys.modules['torchvision.transforms.functional_tensor'] = types.SimpleNamespace(
        rgb_to_grayscale=lambda x: x.mean(dim=1, keepdim=True)
    )

def remove_realesrgan_fix():
    if 'torchvision.transforms.functional_tensor' in sys.modules:
        del sys.modules['torchvision.transforms.functional_tensor']

# Add Real-ESRGAN to path
REALESRGAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Real-ESRGAN-master')
if REALESRGAN_DIR not in sys.path:
    sys.path.append(REALESRGAN_DIR)


# --- FastAPI App Setup ---
app = FastAPI(
    title="Galaxy Image Enhancer API",
    description="API for upscaling and outpainting images using Real-ESRGAN and SD.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["POST","GET","OPTIONS"],
    allow_headers=["*"],
)

# --- Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# --- Model Initialization ---
upsampler_x4 = None
upsampler_x2 = None
pipe = None
device = None
dtype = None

def initialize_model(model_name):

    try:
        apply_realesrgan_fix()
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan.utils import RealESRGANer
    except Exception as e:
        print("Error importing Real-ESRGAN modules:", e)
        raise

    """Initialize the Real-ESRGAN model"""
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        scale = 4
        model_path = os.path.join(REALESRGAN_DIR, 'weights', 'RealESRGAN_x4plus.pth')
        #model_path = "weights/RealESRGAN_x4plus.pth"
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        scale = 2
        model_path = os.path.join(REALESRGAN_DIR, 'weights', 'RealESRGAN_x2plus.pth')
        #model_path = "weights/RealESRGAN_x2plus.pth"
    else:
        raise ValueError(f'Model {model_name} not supported')

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'Model file {model_path} not found. Please download it first.')

    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=400,  # Use tile to avoid CUDA OOM
        tile_pad=10,
        pre_pad=0,
        half=True  # Use full precision for compatibility
    )
    
    return upsampler


@app.on_event("startup")
def load_models():
    global upsampler_x4, upsampler_x2, pipe, device, dtype
    try:
        logger.info("Loading models...")

        # Initialize CUDA if available
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32
    
        # Load SDXL models
        
        # Load ControlNet
        config_file = hf_hub_download(
            "xinsir/controlnet-union-sdxl-1.0",
            filename="config_promax.json",
        )
        
        config = ControlNetModel_Union.load_config(config_file)
        controlnet_model = ControlNetModel_Union.from_config(config)
        
        # Load the state dictionary
        model_file = hf_hub_download(
            "xinsir/controlnet-union-sdxl-1.0",
            filename="diffusion_pytorch_model_promax.safetensors",
        )
        state_dict = load_state_dict(model_file)
        
        # Extract the keys from the state_dict
        loaded_keys = list(state_dict.keys())
        
        # Load pretrained model
        result = ControlNetModel_Union._load_pretrained_model(
            controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0", loaded_keys
        )
        
        model = result[0]
        model = model.to(device=device, dtype=dtype)
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype
        ).to(device)
        
        # Load pipeline
        pipe = StableDiffusionXLFillPipeline.from_pretrained(
            "SG161222/RealVisXL_V5.0_Lightning",
            torch_dtype=dtype,
            vae=vae,
            controlnet=model,
            variant="fp16" if dtype == torch.float16 else None,
        ).to(device)
        
        pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
        
        #logger.info("Models loaded successfully!")

        # Load Real-ESRGAN models

        upsampler_x4 = initialize_model('RealESRGAN_x4plus')
        upsampler_x2 = initialize_model('RealESRGAN_x2plus')

        remove_realesrgan_fix()

        print("Models loaded successfully!")

    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please make sure the model weights are downloaded and placed in the correct directory.")
        print("You can download them from:")
        print("- RealESRGAN_x4plus.pth: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")
        print("- RealESRGAN_x2plus.pth: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth")
        upsampler_x4 = None
        upsampler_x2 = None

        pipe = None


@app.get("/", tags=["General"])
async def read_root():
    return {"message": "Welcome to the Galaxy Image Enhancer API"}

@app.get("/health", tags=["General"])
async def health_check():
    global upsampler_x4, upsampler_x2 #, sd_pipe
    return {
        'status': 'ok',
        'models_loaded': {
            'x4plus': upsampler_x4 is not None,
            'x2plus': upsampler_x2 is not None,
            'sdxl_outpaint': pipe is not None
        }
    }


# --- UPSCALER MODEL ENDPOINTS ---
@app.post("/upscale", tags=["Image Processing"])
async def upscale_image_api(
    image: UploadFile = File(...),
    scale_factor: str = Form("4"),
    outscale: float = Form(4.0)
):

    
    global upsampler_x4, upsampler_x2
    if upsampler_x4 is None or upsampler_x2 is None:
        return JSONResponse(content={'error': 'Models not loaded. Please check server logs.'}, status_code=500)

    if not image.filename:
        return JSONResponse(content={'error': 'No image selected'}, status_code=400)

    try:
        # Save the uploaded image
        img_uuid = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_FOLDER, f"{img_uuid}_input.png")
        output_path = os.path.join(RESULT_FOLDER, f"{img_uuid}_output.png")

        img = Image.open(io.BytesIO(await image.read()))
        img.save(input_path)

        img_cv = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img_cv is None:
            return JSONResponse(content={'error': 'Failed to read image'}, status_code=500)

        upsampler = upsampler_x4 if scale_factor == '4' else upsampler_x2

        # Process the image
        output, _ = upsampler.enhance(img_cv, outscale=outscale)

        # Save the output image
        cv2.imwrite(output_path, output)
        cleanup()

        # Return the result image as base64
        with open(output_path, 'rb') as f:
            img_data = f.read()
            encoded_img = base64.b64encode(img_data).decode('utf-8')

        

        return JSONResponse(content={
            'success': True,
            'image': encoded_img,
            'message': 'Image upscaled successfully'
        })
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)
    finally:
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass

# Cuda memory clean up code
def cleanup():
        """Clean up resources and free memory."""
        try:
            gc.collect()

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            print("🧹 Cleaned up resources and freed memory")
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")


# --- OUTPAINTING PIPELINE DEFINITION ---
def prepare_image_and_mask(image, width, height, overlap_percentage=10, resize_option="Full", 
                          custom_resize_percentage=50, alignment="Middle", 
                          overlap_left=True, overlap_right=True, overlap_top=True, overlap_bottom=True):
    """Prepare image and mask for outpainting"""
    target_size = (width, height)

    # Calculate the scaling factor to fit the image within the target size
    scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    
    # Resize the source image to fit within target size
    source = image.resize((new_width, new_height), Image.LANCZOS)

    # Apply resize option using percentages
    if resize_option == "Full":
        resize_percentage = 100
    elif resize_option == "50%":
        resize_percentage = 50
    elif resize_option == "33%":
        resize_percentage = 33
    elif resize_option == "25%":
        resize_percentage = 25
    else:  # Custom
        resize_percentage = custom_resize_percentage

    # Calculate new dimensions based on percentage
    resize_factor = resize_percentage / 100
    new_width = int(source.width * resize_factor)
    new_height = int(source.height * resize_factor)

    # Ensure minimum size of 64 pixels
    new_width = max(new_width, 64)
    new_height = max(new_height, 64)

    # Resize the image
    source = source.resize((new_width, new_height), Image.LANCZOS)

    # Calculate the overlap in pixels based on the percentage
    overlap_x = int(new_width * (overlap_percentage / 100))
    overlap_y = int(new_height * (overlap_percentage / 100))

    # Ensure minimum overlap of 1 pixel
    overlap_x = max(overlap_x, 1)
    overlap_y = max(overlap_y, 1)

    # Calculate margins based on alignment
    if alignment == "Middle":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Left":
        margin_x = 0
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Right":
        margin_x = target_size[0] - new_width
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Top":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = 0
    elif alignment == "Bottom":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = target_size[1] - new_height

    # Adjust margins to eliminate gaps
    margin_x = max(0, min(margin_x, target_size[0] - new_width))
    margin_y = max(0, min(margin_y, target_size[1] - new_height))

    # Create a new background image and paste the resized source image
    background = Image.new('RGB', target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    # Create the mask
    mask = Image.new('L', target_size, 255)
    mask_draw = ImageDraw.Draw(mask)

    # Calculate overlap areas
    white_gaps_patch = 2

    left_overlap = margin_x + overlap_x if overlap_left else margin_x + white_gaps_patch
    right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width - white_gaps_patch
    top_overlap = margin_y + overlap_y if overlap_top else margin_y + white_gaps_patch
    bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height - white_gaps_patch
    
    if alignment == "Left":
        left_overlap = margin_x + overlap_x if overlap_left else margin_x
    elif alignment == "Right":
        right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width
    elif alignment == "Top":
        top_overlap = margin_y + overlap_y if overlap_top else margin_y
    elif alignment == "Bottom":
        bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height

    # Draw the mask
    mask_draw.rectangle([
        (left_overlap, top_overlap),
        (right_overlap, bottom_overlap)
    ], fill=0)

    return background, mask

def can_expand(source_width, source_height, target_width, target_height, alignment):
    """Checks if the image can be expanded based on the alignment."""
    if alignment in ("Left", "Right") and source_width >= target_width:
        return False
    if alignment in ("Top", "Bottom") and source_height >= target_height:
        return False
    return True

def process_outpaint(image, width, height, num_inference_steps=8, prompt_input=""):
    """Process the outpainting with default parameters matching the original"""
    global pipe, device, dtype

    logger.info(f"Using device: {device}")
    
    # Default parameters from the original app
    overlap_percentage = 10
    resize_option = "Full"
    custom_resize_percentage = 50
    alignment = "Middle"
    overlap_left = True
    overlap_right = True
    overlap_top = True
    overlap_bottom = True
    
    # Prepare image and mask
    background, mask = prepare_image_and_mask(
        image, width, height, overlap_percentage, resize_option, 
        custom_resize_percentage, alignment, overlap_left, overlap_right, 
        overlap_top, overlap_bottom
    )
    
    # Check if expansion is possible
    if not can_expand(background.width, background.height, width, height, alignment):
        alignment = "Middle"
    
    # Create control net image
    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), mask)
    
    # Prepare prompt
    final_prompt = f"{prompt_input} , high quality, 4k" if prompt_input else "high quality, 4k"
    
    # Generate image
    with torch.autocast(device_type=device, dtype=dtype):
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(final_prompt, device, True)
        
        # Get the last image from the generator
        result_image = None
        for image in pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            image=cnet_image,
            num_inference_steps=num_inference_steps
        ):
            result_image = image
    
    # Composite the result
    result_image = result_image.convert("RGBA")
    cnet_image.paste(result_image, (0, 0), mask)
    
    return cnet_image

# --- OUTPAINTING MODEL ENDPOINTS ---

@app.post("/outpaint")
async def outpaint(
    image: UploadFile = File(...),
    target_width: str = Form(...),
    target_height: str = Form(...)
):
    """Outpaint endpoint matching the frontend API expectations"""
    try:
        # Validate dimensions
        try:
            width = int(target_width)
            height = int(target_height)
            if width <= 0 or height <= 0:
                raise ValueError("Dimensions must be positive")
        except ValueError as e:
            return JSONResponse(
                content={"success": False, "error": "Invalid target dimensions"},
                status_code=400
            )
        
        # Read and validate image
        image_data = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(image_data))
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
        except Exception as e:
            return JSONResponse(
                content={"success": False, "error": "Invalid image file"},
                status_code=400
            )
        
        # Check if target dimensions are smaller than original
        if width < pil_image.width or height < pil_image.height:
            return JSONResponse(
                content={
                    "success": False, 
                    "error": f"Target dimensions must be >= original ({pil_image.width}x{pil_image.height})"
                },
                status_code=400
            )
        
     

        # Process the image
        logger.info(f"Processing outpaint: {pil_image.width}x{pil_image.height} -> {width}x{height}")
        result_image = process_outpaint(pil_image, width, height)
        
        # Convert result to PNG bytes
        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format='PNG')
        output_buffer.seek(0)

        # Encode to base64
        base64_image = base64.b64encode(output_buffer.read()).decode('utf-8')
        
        return JSONResponse(
            content={
                "success": True,
                "image": base64_image,
                "message": f"Outpainted to {width}x{height}"
            }
        )
        
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
        return JSONResponse(
            content={"success": False, "error": "Process interrupted"},
            status_code=499
        )

    except Exception as e:
        logger.error(f"Error in outpaint: {str(e)}")
        return JSONResponse(
            content={"success": False, "error": "Internal server error"},
            status_code=500
        )

    finally:
        cleanup()
        
        pass


# Initiate server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # python -m uvicorn main:app --reload --host localhost --port 8000