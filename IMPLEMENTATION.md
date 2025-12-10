# Image Enhancer - Samsung Prism: Complete Implementation Documentation

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Module Breakdown](#4-module-breakdown)
5. [Workflow & Pipeline](#5-workflow--pipeline)
6. [Data Flow](#6-data-flow)
7. [Detailed Process Descriptions](#7-detailed-process-descriptions)
8. [API Documentation](#8-api-documentation)
9. [Frontend Implementation](#9-frontend-implementation)
10. [Deployment & Containerization](#10-deployment--containerization)
11. [Error Handling & Resource Management](#11-error-handling--resource-management)

---

## 1. Project Overview

**Image Enhancer - Samsung Prism** is a full-stack AI-powered image processing application that provides two primary functionalities:

1. **Image Upscaling**: High-fidelity image super-resolution using Real-ESRGAN models
2. **Image Outpainting**: Intelligent canvas expansion using Stable Diffusion XL (SDXL) with ControlNet Union

### Key Features
- **2x and 4x Upscaling**: Enhance image resolution while preserving quality
- **Custom Dimension Outpainting**: Expand images to any target dimensions
- **GPU-Accelerated Processing**: CUDA-enabled for optimal performance
- **Real-time Processing**: Async operations with progress feedback
- **Modern UI**: Responsive Next.js 15 interface with Framer Motion animations

### Project Structure
```
Image Enhancer - Samsung Prism/
├── ImageResizer-Server/          # Backend FastAPI service
│   ├── Real-ESRGAN-master/       # Real-ESRGAN implementation
│   │   ├── weights/              # Pre-trained model weights
│   │   ├── realesrgan/          # Core Real-ESRGAN modules
│   │   └── basicsr/             # BasicSR framework integration
│   ├── main.py                   # Main FastAPI application
│   ├── controlnet_union.py       # ControlNet Union model implementation
│   ├── pipeline_fill_sd_xl.py    # SDXL outpainting pipeline
│   ├── download_models.py        # Model download utility
│   └── requirements.txt          # Python dependencies
├── ImageResizer-Webapp/          # Frontend Next.js application
│   ├── app/                      # Next.js 15 App Router
│   │   ├── page.tsx             # Main application page
│   │   └── layout.tsx           # Root layout
│   ├── components/               # React components
│   │   ├── image-uploader.tsx   # Image upload component
│   │   ├── processing-options.tsx # Processing options selector
│   │   ├── output-display.tsx   # Result display component
│   │   └── logo.tsx             # Application logo
│   ├── lib/                      # Utility libraries
│   │   ├── api.ts               # API client functions
│   │   └── utils.ts             # Helper utilities
│   └── package.json              # Node.js dependencies
├── documents/                     # Documentation and diagrams
├── docker-compose.yml            # Docker orchestration
└── README.md                     # Project documentation
```

---

## 2. System Architecture

### High-Level Architecture

The application follows a **client-server architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Next.js 15 Frontend (Port 3000)                  │  │
│  │  - React 19 Components                                   │  │
│  │  - Tailwind CSS + Framer Motion                          │  │
│  │  - Image Upload & Preview                                │  │
│  │  - Processing Options UI                                 │  │
│  │  - Result Display & Download                             │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/REST API
                              │ (FormData + JSON)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          SERVER LAYER                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │        FastAPI Backend (Port 8000)                       │  │
│  │  - RESTful API Endpoints                                 │  │
│  │  - CORS Middleware                                       │  │
│  │  - File Upload Handling                                  │  │
│  │  - Model Management                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       AI/ML PROCESSING LAYER                    │
│  ┌───────────────────────┐  ┌──────────────────────────────┐  │
│  │   Real-ESRGAN         │  │   SDXL Outpainting           │  │
│  │   Super Resolution    │  │   Pipeline                   │  │
│  │  ┌────────────────┐   │  │  ┌────────────────────────┐ │  │
│  │  │ RRDBNet Model  │   │  │  │ ControlNet Union       │ │  │
│  │  │ (2x/4x)        │   │  │  │ (xinsir/controlnet)    │ │  │
│  │  └────────────────┘   │  │  └────────────────────────┘ │  │
│  │  ┌────────────────┐   │  │  ┌────────────────────────┐ │  │
│  │  │ RealESRGANer   │   │  │  │ RealVisXL Lightning    │ │  │
│  │  │ (Tile-based)   │   │  │  │ (Base Model)           │ │  │
│  │  └────────────────┘   │  │  └────────────────────────┘ │  │
│  └───────────────────────┘  │  ┌────────────────────────┐ │  │
│                              │  │ VAE (fp16-fix)         │ │  │
│                              │  │ (Encoder/Decoder)      │ │  │
│                              │  └────────────────────────┘ │  │
│                              │  ┌────────────────────────┐ │  │
│                              │  │ TCD Scheduler          │ │  │
│                              │  └────────────────────────┘ │  │
│                              └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      HARDWARE/RUNTIME LAYER                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  PyTorch 2.7 + CUDA 12.6                                 │  │
│  │  - GPU Memory Management                                 │  │
│  │  - CUDA Kernel Operations                                │  │
│  │  - FP16/FP32 Mixed Precision                             │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
User Action → Frontend UI → API Request → Backend Router → 
Model Processing → CUDA Operations → Result Generation → 
Base64 Encoding → JSON Response → Frontend Display → User Download
```

---

## 3. Technology Stack

### Backend Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Web Framework** | FastAPI | 0.115.12 | High-performance async web framework |
| **ASGI Server** | Uvicorn | 0.34.2 | Lightning-fast ASGI server |
| **Deep Learning Framework** | PyTorch | 2.7.0+cu126 | Neural network training and inference |
| **Computer Vision** | OpenCV | 4.11.0.86 | Image processing operations |
| **Image Library** | Pillow | 10.4.0 | Python Imaging Library |
| **Numerical Computing** | NumPy | 2.2.3 | Array operations and scientific computing |
| **Super Resolution** | BasicSR | >=1.4.2 | Basic SR framework for Real-ESRGAN |
| **Diffusion Models** | Diffusers | Latest | Hugging Face diffusion models library |
| **Transformers** | Transformers | Latest | CLIP tokenizers and text encoders |
| **Model Hub** | Hugging Face Hub | Latest | Model download and management |
| **Acceleration** | Accelerate | Latest | Distributed training and inference |
| **Multipart Forms** | Python-Multipart | 0.0.20 | File upload handling |

### Frontend Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Framework** | Next.js | 15.2.4 | React-based web framework with SSR |
| **Runtime** | React | 19.0.0 | UI component library |
| **Language** | TypeScript | 5.x | Type-safe JavaScript |
| **Styling** | Tailwind CSS | 4.x | Utility-first CSS framework |
| **Animations** | Framer Motion | 12.9.2 | Production-ready motion library |
| **Icons** | Lucide React | 0.503.0 | Beautiful icon library |
| **Utilities** | clsx + tailwind-merge | Latest | Conditional class management |

### AI/ML Models

| Model | Source | Purpose | Size |
|-------|--------|---------|------|
| **RealESRGAN_x4plus** | xinntao/Real-ESRGAN | 4x super-resolution | ~64MB |
| **RealESRGAN_x2plus** | xinntao/Real-ESRGAN | 2x super-resolution | ~64MB |
| **RealVisXL Lightning** | SG161222/RealVisXL_V5.0_Lightning | Base generative model | ~6.5GB |
| **ControlNet Union** | xinsir/controlnet-union-sdxl-1.0 | Conditional generation control | ~5GB |
| **SDXL VAE** | madebyollin/sdxl-vae-fp16-fix | Image encoding/decoding | ~334MB |

### Infrastructure

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Containerization** | Docker | Application packaging |
| **Orchestration** | Docker Compose | Multi-container management |
| **GPU Runtime** | NVIDIA Container Toolkit | GPU access in containers |
| **CUDA** | CUDA 12.6 | GPU computation framework |

---

## 4. Module Breakdown

### 4.1 Backend Modules

#### 4.1.1 Main Application Module (`main.py`)

**Purpose**: Central FastAPI application orchestrating all backend operations.

**Key Components**:

1. **FastAPI Application Setup**
   ```python
   app = FastAPI(
       title="Galaxy Image Enhancer API",
       description="API for upscaling and outpainting images",
       version="1.0.0"
   )
   ```
   - Initializes FastAPI with metadata
   - Configures automatic API documentation

2. **CORS Middleware Configuration**
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["http://localhost:3000", "*"],
       allow_credentials=True,
       allow_methods=["POST", "GET", "OPTIONS"],
       allow_headers=["*"]
   )
   ```
   - Enables cross-origin requests from frontend
   - Configures allowed HTTP methods and headers

3. **Model Initialization System**
   - Global model variables: `upsampler_x4`, `upsampler_x2`, `pipe`
   - Device detection (CUDA/CPU)
   - Lazy loading on startup event

4. **File Management**
   - Temporary upload directory: `uploads/`
   - Result directory: `results/`
   - Automatic cleanup after processing

**Functions**:

- `initialize_model(model_name)`: Loads Real-ESRGAN models
- `load_models()`: Startup event handler for model initialization
- `cleanup()`: CUDA memory cleanup and garbage collection
- `prepare_image_and_mask()`: Creates image and mask for outpainting
- `can_expand()`: Validates expansion possibility
- `process_outpaint()`: Core outpainting logic

**API Endpoints**:
- `GET /`: Welcome message
- `GET /health`: Health check with model status
- `POST /upscale`: Image upscaling endpoint
- `POST /outpaint`: Image outpainting endpoint

#### 4.1.2 Real-ESRGAN Integration

**Architecture**:

Real-ESRGAN uses a **Residual-in-Residual Dense Block (RRDB)** network architecture:

```
Input Image
    ↓
[Conv Layer + Shallow Feature Extraction]
    ↓
[RRDB Block 1] → [Residual Connection]
    ↓
[RRDB Block 2] → [Residual Connection]
    ↓
    ...
    ↓
[RRDB Block 23] → [Residual Connection]
    ↓
[Deep Feature Extraction]
    ↓
[Upsampling Layers (PixelShuffle)]
    ↓
[Conv Layer + Output]
    ↓
High-Resolution Output
```

**Key Features**:

1. **Network Configuration**:
   - **Input Channels**: 3 (RGB)
   - **Output Channels**: 3 (RGB)
   - **Feature Channels**: 64
   - **Number of Blocks**: 23 RRDB blocks
   - **Growth Channels**: 32
   - **Scale Factor**: 2x or 4x

2. **RealESRGANer Wrapper**:
   ```python
   upsampler = RealESRGANer(
       scale=scale,              # 2 or 4
       model_path=model_path,    # Path to .pth file
       model=model,              # RRDBNet instance
       tile=400,                 # Tile size for processing
       tile_pad=10,              # Padding for tiles
       pre_pad=0,                # Pre-padding
       half=True                 # FP16 precision
   )
   ```

3. **Tile-Based Processing**:
   - Large images are split into 400×400 tiles
   - Each tile processed independently
   - Overlapping regions blended seamlessly
   - Prevents CUDA out-of-memory errors

4. **Processing Pipeline**:
   ```
   Input Image (PIL/CV2)
       ↓
   Convert to CV2 format (BGR)
       ↓
   Normalize to [0, 1] range
       ↓
   Convert to PyTorch tensor
       ↓
   Move to CUDA device
       ↓
   Forward pass through RRDBNet
       ↓
   Denormalize and convert back
       ↓
   Output High-Resolution Image
   ```

#### 4.1.3 ControlNet Union Module (`controlnet_union.py`)

**Purpose**: Multi-conditional ControlNet for unified control signal processing.

**Architecture Components**:

1. **ControlNetConditioningEmbedding**:
   - Converts image-space conditions to 64×64 feature space
   - Four convolution layers: 48 → 96 → 192 → 384 channels
   - 4×4 kernels with 2×2 strides
   - ReLU activation

2. **Condition Transformer**:
   - Exchanges information between different conditions
   - Multi-head attention mechanism (8 heads)
   - 320 channels per condition type
   - Layer normalization and residual connections

3. **Control Type Embedding**:
   - Supports 6 control types (inpainting index: 6)
   - Time embedding for control type injection
   - Projects to time embedding dimension

4. **Down Blocks**:
   - CrossAttnDownBlock2D for early blocks
   - DownBlock2D for final block
   - Progressive downsampling: 320 → 640 → 1280 → 1280 channels

5. **Middle Block**:
   - UNetMidBlock2DCrossAttn
   - Attention with cross-attention to text embeddings
   - Bottleneck for latent representation

**Key Features**:

- **Union Control Type**: `[0, 0, 0, 0, 0, 0, 1, 0]` (inpainting mode)
- **Zero Initialization**: Output layers initialized to zero for stable training
- **Gradient Checkpointing**: Memory-efficient training support
- **FP16 Support**: Mixed precision for faster inference

#### 4.1.4 SDXL Fill Pipeline (`pipeline_fill_sd_xl.py`)

**Purpose**: Custom Stable Diffusion XL pipeline for outpainting.

**Pipeline Stages**:

```
Text Prompt
    ↓
[CLIP Text Encoder 1] → Prompt Embeddings (77 tokens)
    ↓
[CLIP Text Encoder 2] → Pooled Embeddings
    ↓
    ↓
Input Image + Mask
    ↓
[ControlNet Union] → Control Features
    ↓                    ↓
Random Noise    ←  [Scheduler]
    ↓                    ↓
[UNet Denoising Loop] ← Control Features
    ↓ (8 steps)
Latent Representation
    ↓
[VAE Decoder]
    ↓
Output Image (RGB)
```

**Key Components**:

1. **Text Encoding**:
   - Dual CLIP encoders for SDXL
   - Prompt embeddings: 77 × 2048 dimensions
   - Pooled embeddings for time embedding
   - Classifier-free guidance support

2. **Image Preparation**:
   - VaeImageProcessor for preprocessing
   - Normalize to [-1, 1] range
   - Resize to model dimensions (1024×1024)
   - Create control image with masked regions

3. **Latent Space Operations**:
   - Initial noise: Gaussian distribution
   - Scaled by scheduler's noise sigma
   - Latent dimensions: height/8 × width/8

4. **Denoising Loop** (8 iterations):
   ```python
   for i, t in enumerate(timesteps):
       # Expand latents for CFG
       latent_input = torch.cat([latents] * 2)
       
       # Scale by scheduler
       latent_input = scheduler.scale_model_input(latent_input, t)
       
       # Get control features
       down_samples, mid_sample = controlnet(
           latent_input, t, prompt_embeds, control_image
       )
       
       # Predict noise
       noise_pred = unet(
           latent_input, t, prompt_embeds,
           down_samples, mid_sample
       )
       
       # Remove noise
       latents = scheduler.step(noise_pred, t, latents)
   ```

5. **VAE Decoding**:
   - Decode latents to pixel space
   - FP16-fixed VAE prevents artifacts
   - Output: RGB image [0, 255]

**Optimizations**:

- **TCD Scheduler**: Fast sampling (8 steps vs 50)
- **FP16 Precision**: 2x faster inference
- **Progressive Bar**: Real-time progress updates
- **Memory Efficient**: Torch.no_grad() context

### 4.2 Frontend Modules

#### 4.2.1 Main Page Component (`app/page.tsx`)

**Purpose**: Root component orchestrating the entire UI.

**State Management**:

```typescript
const [uploadedImage, setUploadedImage] = useState<string | null>(null)
const [isProcessing, setIsProcessing] = useState(false)
const [processedImage, setProcessedImage] = useState<string | null>(null)
const [selectedUpscale, setSelectedUpscale] = useState<"2x" | "4x" | null>(null)
const [outpaintWidth, setOutpaintWidth] = useState<number | null>(null)
const [outpaintHeight, setOutpaintHeight] = useState<number | null>(null)
const [originalImageWidth, setOriginalImageWidth] = useState<number | null>(null)
const [originalImageHeight, setOriginalImageHeight] = useState<number | null>(null)
const [error, setError] = useState<string | null>(null)
```

**Event Handlers**:

1. **Image Upload Handler**:
   ```typescript
   const handleImageUpload = (imageDataUrl: string, width?: number, height?: number) => {
       setUploadedImage(imageDataUrl)
       setOriginalImageWidth(width || null)
       setOriginalImageHeight(height || null)
       // Reset processing state
       setProcessedImage(null)
       setSelectedUpscale(null)
       setError(null)
   }
   ```

2. **Process Image Handler**:
   ```typescript
   const handleProcessImage = async () => {
       setIsProcessing(true)
       setError(null)
       
       try {
           if (selectedUpscale) {
               const result = await upscaleImage(uploadedImage, {
                   scaleFactor: selectedUpscale === "2x" ? "2" : "4"
               })
               setProcessedImage(result.imageData)
           } else {
               const result = await outpaintImage(uploadedImage, {
                   width: outpaintWidth,
                   height: outpaintHeight
               })
               setProcessedImage(result.imageData)
           }
       } catch (err) {
           setError(err.message)
       } finally {
           setIsProcessing(false)
       }
   }
   ```

**UI Sections**:

1. **Header**: Logo + Title with fade-in animation
2. **Upload Section**: Drag-and-drop zone with preview
3. **Options Section**: Conditional render based on uploaded image
4. **Result Section**: Processing animation or result display

#### 4.2.2 Image Uploader Component (`components/image-uploader.tsx`)

**Features**:

1. **Drag-and-Drop Support**:
   ```typescript
   const handleDrop = (e: React.DragEvent) => {
       e.preventDefault()
       const file = e.dataTransfer.files[0]
       if (file.type.match("image.*")) {
           processFile(file)
       }
   }
   ```

2. **File Input Click**:
   - Hidden file input with `accept="image/*"`
   - Triggered by button click
   - Supports JPG, PNG, WEBP

3. **Progress Animation**:
   - Simulated upload progress bar
   - Framer Motion for smooth transitions
   - 100ms intervals, 10% increments

4. **Image Processing**:
   ```typescript
   const processFile = (file: File) => {
       const reader = new FileReader()
       reader.onload = (e) => {
           const imageDataUrl = e.target.result as string
           const img = new window.Image()
           img.onload = () => {
               onImageUpload(imageDataUrl, img.naturalWidth, img.naturalHeight)
           }
           img.src = imageDataUrl
       }
       reader.readAsDataURL(file)
   }
   ```

5. **Preview Thumbnail**:
   - 32×32 preview with remove button
   - Next.js Image component for optimization
   - Object-cover for aspect ratio preservation

#### 4.2.3 Processing Options Component (`components/processing-options.tsx`)

**Upscale Options**:

```typescript
<OptionCard
    title="Upscale 2x"
    description="Double the resolution"
    isSelected={selectedUpscale === "2x"}
    onClick={() => handleUpscaleSelect("2x")}
/>
```

**Outpaint Options**:

```typescript
<Input
    type="number"
    placeholder="Width (px)"
    value={outpaintWidth}
    onChange={handleWidthChange}
/>
<Input
    type="number"
    placeholder="Height (px)"
    value={outpaintHeight}
    onChange={handleHeightChange}
/>
```

**Mutual Exclusivity Logic**:
- Selecting upscale clears outpaint dimensions
- Entering dimensions clears upscale selection
- Visual feedback for active selection

#### 4.2.4 Output Display Component (`components/output-display.tsx`)

**States**:

1. **Processing**:
   - Dual rotating spinners (teal + blue)
   - "Processing your image..." message
   - Smooth opacity transitions

2. **Result Display**:
   - Full-size image with object-contain
   - Download button with gradient
   - Hover and tap animations

3. **Placeholder**:
   - "Your enhanced image will appear here"
   - Gray text on empty state

**Download Handler**:
```typescript
const handleDownload = () => {
    const link = document.createElement("a")
    link.href = processedImage
    link.download = "enhanced-image.jpg"
    link.click()
}
```

#### 4.2.5 API Client Module (`lib/api.ts`)

**Upscale API Function**:

```typescript
export async function upscaleImage(imageData: string, options: {
    scaleFactor: '2' | '4';
    outscale?: string;
}) {
    const imageFile = dataURLtoFile(imageData, 'image.png')
    const formData = new FormData()
    formData.append('image', imageFile)
    formData.append('scale_factor', scaleFactor)
    formData.append('outscale', outscale)
    
    const response = await fetch('http://localhost:8000/upscale', {
        method: 'POST',
        body: formData
    })
    
    const data = await response.json()
    return {
        success: data.success,
        imageData: `data:image/png;base64,${data.image}`,
        message: data.message
    }
}
```

**Outpaint API Function**:

```typescript
export async function outpaintImage(imageData: string, options: {
    width?: number;
    height?: number;
}) {
    const imageFile = dataURLtoFile(imageData, 'image.png')
    const formData = new FormData()
    formData.append('image', imageFile)
    formData.append('target_width', width.toString())
    formData.append('target_height', height.toString())
    
    const response = await fetch('http://localhost:8000/outpaint', {
        method: 'POST',
        body: formData
    })
    
    const data = await response.json()
    return {
        success: data.success,
        imageData: `data:image/png;base64,${data.image}`,
        message: data.message
    }
}
```

**Utility Functions**:

```typescript
function dataURLtoFile(dataUrl: string, filename: string): File {
    const arr = dataUrl.split(',')
    const mime = arr[0].match(/:(.*?);/)![1]
    const bstr = atob(arr[1])
    const u8arr = new Uint8Array(bstr.length)
    
    for (let i = 0; i < bstr.length; i++) {
        u8arr[i] = bstr.charCodeAt(i)
    }
    
    return new File([u8arr], filename, { type: mime })
}
```

---

## 5. Workflow & Pipeline

### 5.1 Image Upscaling Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                      UPSCALING WORKFLOW                         │
└─────────────────────────────────────────────────────────────────┘

1. USER ACTION
   └─→ User uploads image via drag-and-drop or file picker
       └─→ Frontend validates image format (JPG/PNG/WEBP)
           └─→ Image converted to data URL and dimensions extracted

2. OPTION SELECTION
   └─→ User selects 2x or 4x upscale option
       └─→ Frontend enables "Process Image" button
           └─→ Button click triggers handleProcessImage()

3. API REQUEST PREPARATION
   └─→ Data URL converted to File object
       └─→ FormData constructed:
           • image: File (binary)
           • scale_factor: "2" | "4"
           • outscale: 2.0 | 4.0
       └─→ POST request to http://localhost:8000/upscale

4. BACKEND PROCESSING
   └─→ FastAPI receives request
       └─→ Validate image file exists
           └─→ Generate UUID for temporary files
               └─→ Save uploaded image to uploads/
                   
5. MODEL SELECTION
   └─→ Select upsampler based on scale_factor:
       • scale_factor="2" → upsampler_x2
       • scale_factor="4" → upsampler_x4
   └─→ Load image with OpenCV (BGR format)

6. REAL-ESRGAN PROCESSING
   └─→ Image split into 400×400 tiles
       └─→ For each tile:
           ├─→ Convert to PyTorch tensor
           ├─→ Normalize to [0, 1]
           ├─→ Move to CUDA device
           ├─→ Forward pass through RRDBNet
           ├─→ Apply pixel shuffle upsampling
           └─→ Denormalize and convert back
       └─→ Blend overlapping regions
           └─→ Reconstruct full image

7. POST-PROCESSING
   └─→ Save output to results/
       └─→ Read output file as binary
           └─→ Encode to base64 string
               └─→ CUDA memory cleanup (gc.collect(), cuda.empty_cache())

8. RESPONSE GENERATION
   └─→ JSON response:
       {
           "success": true,
           "image": "base64_encoded_string",
           "message": "Image upscaled successfully"
       }
   └─→ Delete temporary files (uploads/, results/)

9. FRONTEND DISPLAY
   └─→ Receive JSON response
       └─→ Decode base64 to data URL
           └─→ Set processedImage state
               └─→ Display in OutputDisplay component
                   └─→ Enable download button

10. USER DOWNLOAD
    └─→ User clicks "Download Image"
        └─→ Create temporary <a> element
            └─→ Set href to data URL
                └─→ Trigger download as "enhanced-image.jpg"
                    └─→ Remove temporary element
```

**Timing Breakdown**:
- Image upload: ~100-500ms (depends on file size)
- Request transmission: ~10-50ms (local network)
- Backend processing:
  - Model selection: <1ms
  - Real-ESRGAN inference:
    - Small image (512×512): 1-2 seconds
    - Medium image (1024×1024): 3-5 seconds
    - Large image (2048×2048): 8-12 seconds
- Response transmission: ~50-200ms (depends on result size)
- Frontend display: ~100-200ms

### 5.2 Image Outpainting Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPAINTING WORKFLOW                         │
└─────────────────────────────────────────────────────────────────┘

1. USER ACTION
   └─→ User uploads image
       └─→ Original dimensions extracted and displayed
           └─→ User enters target width and height

2. VALIDATION
   └─→ Frontend validates:
       • Width and height are positive integers
       • Target dimensions ≥ original dimensions
       • Upscale option is cleared
   └─→ Enable "Process Image" button

3. API REQUEST PREPARATION
   └─→ Data URL converted to File object
       └─→ FormData constructed:
           • image: File (binary)
           • target_width: string (px)
           • target_height: string (px)
       └─→ POST request to http://localhost:8000/outpaint

4. BACKEND VALIDATION
   └─→ Parse target dimensions to integers
       └─→ Validate dimensions > 0
           └─→ Load image with PIL
               └─→ Convert to RGB if necessary
                   └─→ Validate target ≥ original

5. IMAGE & MASK PREPARATION
   └─→ Call prepare_image_and_mask():
       ├─→ Calculate scaling factor to fit target size
       ├─→ Resize image: (width×scale, height×scale)
       ├─→ Apply resize option (Full = 100%)
       ├─→ Calculate overlap (10% of dimensions)
       ├─→ Calculate margins for "Middle" alignment
       ├─→ Create white background (target size)
       ├─→ Paste resized image at center
       └─→ Create binary mask (L mode):
           • White (255): regions to generate
           • Black (0): original image regions

6. CONTROL IMAGE CREATION
   └─→ Copy background image
       └─→ Paste 0 (black) on masked regions
           └─→ Result: Visible original + black expansion areas

7. TEXT PROMPT ENCODING
   └─→ Prepare prompt: "high quality, 4k"
       └─→ Encode with CLIP Text Encoder 1:
           • Tokenize (max 77 tokens)
           • Get hidden states
           • Shape: [1, 77, 768]
       └─→ Encode with CLIP Text Encoder 2:
           • Tokenize (max 77 tokens)
           • Get pooled output
           • Shape: [1, 1280]
       └─→ Concatenate embeddings: [1, 77, 2048]
       └─→ Encode negative prompt (zeros for CFG)

8. LATENT PREPARATION
   └─→ Generate random noise:
       • Shape: [1, 4, height/8, width/8]
       • Distribution: Normal(0, 1)
       • Scale by scheduler's initial sigma
   └─→ Move to CUDA device (FP16)

9. CONTROLNET CONDITIONING
   └─→ Process control image:
       • Normalize to [0, 1]
       • Convert to tensor
       • Shape: [1, 3, height, width]
   └─→ Set union control type: [0,0,0,0,0,0,1,0]
       • Index 6 = inpainting mode

10. DENOISING LOOP (8 steps)
    └─→ For timestep t in [999, 857, 714, 571, 428, 285, 142, 0]:
        ├─→ Expand latents for CFG: [2, 4, H/8, W/8]
        ├─→ Scale by scheduler: latents * sigma_t
        ├─→ ControlNet forward pass:
        │   ├─→ Input: scaled latents + control image
        │   ├─→ Encode control image features
        │   ├─→ Inject control type embedding
        │   ├─→ Down blocks: extract multi-scale features
        │   ├─→ Mid block: bottleneck features
        │   └─→ Output: (down_samples, mid_sample)
        ├─→ UNet forward pass:
        │   ├─→ Input: scaled latents + prompt embeddings
        │   ├─→ Inject control features at each scale
        │   ├─→ Cross-attention with text embeddings
        │   ├─→ Predict noise residual
        │   └─→ Output: noise prediction [2, 4, H/8, W/8]
        ├─→ Classifier-free guidance:
        │   └─→ noise = noise_uncond + 1.5 × (noise_cond - noise_uncond)
        └─→ Scheduler step:
            └─→ latents = (latents - noise) / alpha_t

11. VAE DECODING
    └─→ Decode latents to pixel space:
        ├─→ Input: [1, 4, H/8, W/8]
        ├─→ VAE decoder upsamples 8x
        ├─→ Output: [1, 3, H, W]
        └─→ Denormalize to [0, 255]

12. FINAL COMPOSITING
    └─→ Convert result to RGBA
        └─→ Paste result onto control image using mask
            └─→ Original regions preserved
                └─→ Generated regions filled

13. RESPONSE GENERATION
    └─→ Convert result to PNG (BytesIO)
        └─→ Encode to base64
            └─→ JSON response:
                {
                    "success": true,
                    "image": "base64_encoded_string",
                    "message": "Outpainted to WxH"
                }
            └─→ CUDA memory cleanup

14. FRONTEND DISPLAY
    └─→ Receive and decode response
        └─→ Display result image
            └─→ Enable download

15. USER DOWNLOAD
    └─→ Download enhanced image
```

**Timing Breakdown**:
- Image upload: ~100-500ms
- Request transmission: ~10-50ms
- Backend processing:
  - Image preparation: ~100-300ms
  - Text encoding: ~50-100ms
  - Denoising loop (8 steps): 5-15 seconds (GPU-dependent)
  - VAE decoding: ~500ms-1s
  - Compositing: ~50-100ms
- Response transmission: ~100-500ms
- Frontend display: ~100-200ms

**Total Time**: 7-20 seconds (primarily denoising)

---

## 6. Data Flow

### 6.1 Request Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                         CLIENT SIDE                              │
└──────────────────────────────────────────────────────────────────┘

User Image File (File object)
       ↓
FileReader.readAsDataURL()
       ↓
Data URL (string)
"data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..."
       ↓
Extract dimensions via Image() constructor
       ↓
Store in state: uploadedImage, originalWidth, originalHeight
       ↓
User selects processing option
       ↓
handleProcessImage() triggered
       ↓
dataURLtoFile() conversion
       ↓
File object recreated
       ↓
FormData construction:
┌─────────────────────────────────────────────┐
│ FormData {                                  │
│   image: File (binary blob)                │
│   scale_factor: "2" | "4"  OR              │
│   target_width: "1920"                     │
│   target_height: "1080"                    │
│   outscale: "2.0" | "4.0"                  │
│ }                                           │
└─────────────────────────────────────────────┘
       ↓
fetch() with method="POST"
       ↓
HTTP Request (multipart/form-data)

═══════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────┐
│                         SERVER SIDE                              │
└──────────────────────────────────────────────────────────────────┘

FastAPI receives request
       ↓
CORS middleware validation
       ↓
Route to /upscale or /outpaint endpoint
       ↓
Parse FormData:
┌─────────────────────────────────────────────┐
│ image: UploadFile                           │
│   - filename: "image.png"                   │
│   - content_type: "image/png"               │
│   - file: SpooledTemporaryFile              │
│ scale_factor/width/height: str              │
└─────────────────────────────────────────────┘
       ↓
Read file: await image.read()
       ↓
Binary image data (bytes)
       ↓
PIL.Image.open(BytesIO(bytes))
       ↓
PIL Image object (RGB)
       ↓
Convert to numpy array (for CV2) OR keep PIL (for SDXL)
       ↓
Model processing (see workflow sections)
       ↓
Output image (numpy array or PIL)
       ↓
Convert to PNG bytes
       ↓
base64.b64encode()
       ↓
Base64 string
       ↓
JSON response:
┌─────────────────────────────────────────────┐
│ {                                           │
│   "success": true,                          │
│   "image": "iVBORw0KGg...",                 │
│   "message": "Success message"              │
│ }                                           │
└─────────────────────────────────────────────┘
       ↓
HTTP Response (application/json)

═══════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────┐
│                         CLIENT SIDE                              │
└──────────────────────────────────────────────────────────────────┘

fetch() receives response
       ↓
response.json() parsing
       ↓
Extract base64 string
       ↓
Prepend data URL prefix:
"data:image/png;base64," + base64String
       ↓
Data URL (string)
       ↓
setProcessedImage(dataURL)
       ↓
React state update triggers re-render
       ↓
<img src={processedImage} /> renders
       ↓
Browser decodes and displays image
       ↓
User clicks download
       ↓
Create <a> element with href=dataURL
       ↓
trigger click() and download
       ↓
File saved to user's system
```

### 6.2 Model Data Flow - Upscaling

```
Input Image (H × W × 3)
       ↓
cv2.imread() → numpy.ndarray [H, W, 3] (BGR, uint8)
       ↓
Normalize: img / 255.0 → [H, W, 3] (float32, [0,1])
       ↓
Transpose: (H, W, C) → (C, H, W)
       ↓
torch.from_numpy() → Tensor [3, H, W]
       ↓
Add batch dimension → [1, 3, H, W]
       ↓
Move to CUDA: .cuda() → Tensor (GPU)
       ↓
Convert to FP16: .half() → Tensor (float16)
       ↓

┌──────────────────────────────────────────────────────────────────┐
│                      RRDBNet Forward Pass                        │
└──────────────────────────────────────────────────────────────────┘

[1, 3, H, W]
       ↓
Conv2d(3, 64, 3×3) → [1, 64, H, W]  (Shallow features)
       ↓
RRDB Block 1
├─→ DenseBlock → [1, 64, H, W]
├─→ Residual add
└─→ [1, 64, H, W]
       ↓
RRDB Block 2-23 (similar structure)
       ↓
[1, 64, H, W]  (Deep features)
       ↓
Conv2d(64, 64, 3×3)
       ↓
[1, 64, H, W]
       ↓
Upsampling Block (2x or 4x):
  If 2x:
    ├─→ Conv2d(64, 256, 3×3)
    ├─→ PixelShuffle(2) → [1, 64, 2H, 2W]
    └─→ LeakyReLU
  If 4x:
    ├─→ Conv2d(64, 256, 3×3)
    ├─→ PixelShuffle(2) → [1, 64, 2H, 2W]
    ├─→ LeakyReLU
    ├─→ Conv2d(64, 256, 3×3)
    ├─→ PixelShuffle(2) → [1, 64, 4H, 4W]
    └─→ LeakyReLU
       ↓
Conv2d(64, 3, 3×3) → [1, 3, scale×H, scale×W]
       ↓

Move to CPU: .cpu()
       ↓
Convert to FP32: .float()
       ↓
Remove batch: [3, scale×H, scale×W]
       ↓
Transpose: (C, H, W) → (H, W, C)
       ↓
Denormalize: tensor × 255.0
       ↓
Clamp to [0, 255]
       ↓
Convert to uint8: .numpy().astype(np.uint8)
       ↓
Output: numpy.ndarray [scale×H, scale×W, 3] (BGR)
       ↓
cv2.imwrite() → PNG file
```

### 6.3 Model Data Flow - Outpainting

```
Input: PIL Image (W × H) + Target (W' × H')
       ↓
prepare_image_and_mask()
       ↓
┌──────────────────────────────────────────────────────────┐
│ Background: PIL Image (W' × H')                          │
│   - White canvas                                         │
│   - Original image pasted at center                      │
│                                                          │
│ Mask: PIL Image (W' × H'), mode='L'                     │
│   - 255 (white): areas to generate                      │
│   - 0 (black): original image preserved                 │
└──────────────────────────────────────────────────────────┘
       ↓
Create control image: background with masked regions black
       ↓

┌──────────────────────────────────────────────────────────────────┐
│                      Text Encoding                               │
└──────────────────────────────────────────────────────────────────┘

Prompt: "high quality, 4k"
       ↓
CLIP Tokenizer 1:
  └─→ ["<start>", "high", "quality", ",", "4k", "<end>", "<pad>", ...]
      └─→ Token IDs: [49406, 4558, 3029, 267, 306, 49407, 49407, ...]
          └─→ Tensor [1, 77]
       ↓
CLIP Text Encoder 1:
  └─→ Embedding layer: [1, 77, 768]
      └─→ Transformer blocks (12 layers)
          └─→ Hidden states: [1, 77, 768]
       ↓
CLIP Tokenizer 2:
  └─→ Similar tokenization
      └─→ Tensor [1, 77]
       ↓
CLIP Text Encoder 2:
  └─→ Embedding layer: [1, 77, 1280]
      └─→ Transformer blocks (12 layers)
          ├─→ Hidden states: [1, 77, 1280]
          └─→ Pooled output: [1, 1280]
       ↓
Concatenate: [1, 77, 768] + [1, 77, 1280] = [1, 77, 2048]
       ↓
prompt_embeds: [1, 77, 2048]
pooled_prompt_embeds: [1, 1280]

┌──────────────────────────────────────────────────────────────────┐
│                   Control Image Processing                       │
└──────────────────────────────────────────────────────────────────┘

Control Image (PIL)
       ↓
VaeImageProcessor.preprocess():
  ├─→ Convert to RGB
  ├─→ Resize to (W', H')
  ├─→ Convert to tensor: [1, 3, H', W']
  ├─→ Normalize to [0, 1]
  └─→ Don't normalize to [-1, 1] (do_normalize=False)
       ↓
control_image_tensor: [1, 3, H', W'] (float32, [0,1])
       ↓
Duplicate for CFG: [2, 3, H', W']

┌──────────────────────────────────────────────────────────────────┐
│                      Latent Initialization                       │
└──────────────────────────────────────────────────────────────────┘

Shape: [1, 4, H'/8, W'/8]
       ↓
randn_tensor() → Gaussian noise N(0, 1)
       ↓
Multiply by scheduler.init_noise_sigma (typically ~14.6)
       ↓
latents: [1, 4, H'/8, W'/8] (float16, GPU)

┌──────────────────────────────────────────────────────────────────┐
│                      Denoising Loop                              │
└──────────────────────────────────────────────────────────────────┘

For t in timesteps [999, 857, 714, 571, 428, 285, 142, 0]:
    
    latents: [1, 4, H'/8, W'/8]
           ↓
    Duplicate for CFG: [2, 4, H'/8, W'/8]
           ↓
    scheduler.scale_model_input():
      └─→ latents × (1 / sqrt(sigma_t² + 1))
           ↓
    
    ┌──────────────────────────────────────────────────────┐
    │              ControlNet Forward                      │
    └──────────────────────────────────────────────────────┘
    
    Inputs:
      - latents: [2, 4, H'/8, W'/8]
      - timestep: t
      - prompt_embeds: [2, 77, 2048]
      - control_image: [2, 3, H', W']
      - control_type: [2, 8]
    
    ControlNetConditioningEmbedding:
      control_image [2, 3, H', W']
             ↓
      Conv2d(3, 48, 3×3) + SiLU
             ↓
      Conv2d(48, 48, 3×3) + SiLU
      Conv2d(48, 96, 3×3, stride=2) + SiLU
             ↓ [2, 96, H'/2, W'/2]
      Conv2d(96, 96, 3×3) + SiLU
      Conv2d(96, 192, 3×3, stride=2) + SiLU
             ↓ [2, 192, H'/4, W'/4]
      Conv2d(192, 192, 3×3) + SiLU
      Conv2d(192, 384, 3×3, stride=2) + SiLU
             ↓ [2, 384, H'/8, W'/8]
      Conv2d(384, 320, 3×3)
             ↓
      conditioning_embeds: [2, 320, H'/8, W'/8]
    
    Time Embedding:
      timestep t
             ↓
      Sinusoidal encoding
             ↓
      Linear(320, 1280)
             ↓
      SiLU + Linear(1280, 1280)
             ↓
      time_embeds: [2, 1280]
    
    Control Type Embedding:
      control_type: [2, 8]
             ↓
      Expand each element to 256 dimensions
             ↓
      Linear(256×8, 1280)
             ↓
      control_embeds: [2, 1280]
    
    Combined time embedding:
      time_embeds + control_embeds → [2, 1280]
    
    Conv In:
      latents [2, 4, H'/8, W'/8]
             ↓
      Conv2d(4, 320, 3×3)
             ↓
      Add conditioning_embeds
             ↓
      [2, 320, H'/8, W'/8]
    
    Down Block 1 (CrossAttnDownBlock2D):
      [2, 320, H'/8, W'/8]
             ↓
      ResNet blocks (2 layers)
      Cross-attention with prompt_embeds
             ↓
      [2, 320, H'/8, W'/8]
             ↓
      Downsample: Conv2d(stride=2)
             ↓
      down_sample_0: [2, 320, H'/16, W'/16]
    
    Down Block 2 (CrossAttnDownBlock2D):
      [2, 320, H'/16, W'/16]
             ↓
      ResNet blocks + Cross-attention
             ↓
      [2, 640, H'/16, W'/16]
             ↓
      Downsample
             ↓
      down_sample_1: [2, 640, H'/32, W'/32]
    
    Down Block 3 (CrossAttnDownBlock2D):
      [2, 640, H'/32, W'/32]
             ↓
      ResNet blocks + Cross-attention
             ↓
      [2, 1280, H'/32, W'/32]
             ↓
      Downsample
             ↓
      down_sample_2: [2, 1280, H'/64, W'/64]
    
    Down Block 4 (DownBlock2D):
      [2, 1280, H'/64, W'/64]
             ↓
      ResNet blocks (no attention)
             ↓
      down_sample_3: [2, 1280, H'/64, W'/64]
    
    Mid Block (UNetMidBlock2DCrossAttn):
      [2, 1280, H'/64, W'/64]
             ↓
      ResNet + Attention + Cross-attention + ResNet
             ↓
      mid_sample: [2, 1280, H'/64, W'/64]
    
    ControlNet Down Blocks (zero-initialized Conv2d):
      down_sample_0 → [2, 320, H'/16, W'/16]
      down_sample_1 → [2, 640, H'/32, W'/32]
      down_sample_2 → [2, 1280, H'/64, W'/64]
      down_sample_3 → [2, 1280, H'/64, W'/64]
      mid_sample → [2, 1280, H'/64, W'/64]
    
    Outputs:
      - down_block_res_samples: tuple of 4 tensors
      - mid_block_res_sample: [2, 1280, H'/64, W'/64]
    
    ┌──────────────────────────────────────────────────────┐
    │                  UNet Forward                        │
    └──────────────────────────────────────────────────────┘
    
    Inputs:
      - latents: [2, 4, H'/8, W'/8]
      - timestep: t
      - prompt_embeds: [2, 77, 2048]
      - down_block_res_samples: control features
      - mid_block_res_sample: control features
    
    Similar architecture to ControlNet but with control injection:
      - Each down block adds control features
      - Mid block adds control features
      - Up blocks use skip connections from down blocks
    
    Output:
      noise_pred: [2, 4, H'/8, W'/8]
    
    ┌──────────────────────────────────────────────────────┐
    │            Classifier-Free Guidance                  │
    └──────────────────────────────────────────────────────┘
    
    noise_pred: [2, 4, H'/8, W'/8]
           ↓
    Split: noise_uncond, noise_cond = noise_pred.chunk(2)
      - noise_uncond: [1, 4, H'/8, W'/8]  (from negative prompt)
      - noise_cond: [1, 4, H'/8, W'/8]    (from positive prompt)
           ↓
    guidance_scale = 1.5
           ↓
    noise_pred = noise_uncond + 1.5 × (noise_cond - noise_uncond)
           ↓
    noise_pred: [1, 4, H'/8, W'/8]
    
    ┌──────────────────────────────────────────────────────┐
    │              Scheduler Step (TCD)                    │
    └──────────────────────────────────────────────────────┘
    
    latents_t: [1, 4, H'/8, W'/8]
    noise_pred: [1, 4, H'/8, W'/8]
    timestep: t
           ↓
    alpha_t = sqrt(1 - beta_t)
    sigma_t = sqrt(beta_t)
           ↓
    latents_{t-1} = (latents_t - sigma_t × noise_pred) / alpha_t
           ↓
    latents: [1, 4, H'/8, W'/8]  (less noisy)

End Loop

┌──────────────────────────────────────────────────────────────────┐
│                         VAE Decoding                             │
└──────────────────────────────────────────────────────────────────┘

latents: [1, 4, H'/8, W'/8]
       ↓
Scale by VAE scaling factor: latents / 0.18215
       ↓
VAE Decoder:
  [1, 4, H'/8, W'/8]
         ↓
  Conv2d(4, 128, 3×3)
         ↓
  Residual blocks + Upsampling (3 stages)
    ├─→ Upsample × 2: [1, 128, H'/4, W'/4]
    ├─→ Upsample × 2: [1, 128, H'/2, W'/2]
    └─→ Upsample × 2: [1, 128, H', W']
         ↓
  Conv2d(128, 3, 3×3)
         ↓
  [1, 3, H', W']
       ↓
Denormalize from [-1, 1] to [0, 255]:
  image = (image + 1) × 127.5
       ↓
Clamp to [0, 255]
       ↓
Convert to uint8
       ↓
Transpose: (C, H, W) → (H, W, C)
       ↓
Convert to PIL: Image.fromarray()
       ↓
result_image: PIL Image (W' × H')

┌──────────────────────────────────────────────────────────────────┐
│                        Compositing                               │
└──────────────────────────────────────────────────────────────────┘

result_image: PIL Image (W' × H')
       ↓
Convert to RGBA: result_image.convert("RGBA")
       ↓
control_image (background): PIL Image (W' × H')
       ↓
Paste result onto control_image using mask:
  control_image.paste(result_image, (0, 0), mask)
  - Where mask = 255: use result_image
  - Where mask = 0: use control_image (original)
       ↓
final_image: PIL Image (W' × H')
       ↓
Save to BytesIO as PNG
       ↓
Encode to base64
       ↓
Return in JSON response
```

---

## 7. Detailed Process Descriptions

### 7.1 Model Initialization Process

**Startup Sequence**:

```python
@app.on_event("startup")
def load_models():
    # 1. Device Detection
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
    else:
        device = "cpu"
        dtype = torch.float32
        print("⚠ CUDA not available, using CPU")
    
    # 2. Load SDXL ControlNet
    print("Loading ControlNet Union...")
    config_file = hf_hub_download(
        "xinsir/controlnet-union-sdxl-1.0",
        filename="config_promax.json"
    )
    config = ControlNetModel_Union.load_config(config_file)
    controlnet_model = ControlNetModel_Union.from_config(config)
    
    model_file = hf_hub_download(
        "xinsir/controlnet-union-sdxl-1.0",
        filename="diffusion_pytorch_model_promax.safetensors"
    )
    state_dict = load_state_dict(model_file)
    
    # Load pretrained weights
    result = ControlNetModel_Union._load_pretrained_model(
        controlnet_model, state_dict, model_file,
        "xinsir/controlnet-union-sdxl-1.0", list(state_dict.keys())
    )
    model = result[0].to(device=device, dtype=dtype)
    
    # 3. Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=dtype
    ).to(device)
    
    # 4. Load SDXL Pipeline
    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLFillPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0_Lightning",
        torch_dtype=dtype,
        vae=vae,
        controlnet=model,
        variant="fp16" if dtype == torch.float16 else None
    ).to(device)
    
    # 5. Configure Scheduler
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    
    # 6. Load Real-ESRGAN Models
    print("Loading Real-ESRGAN models...")
    apply_realesrgan_fix()  # Fix compatibility issue
    
    upsampler_x4 = initialize_model('RealESRGAN_x4plus')
    upsampler_x2 = initialize_model('RealESRGAN_x2plus')
    
    remove_realesrgan_fix()
    
    print("✓ All models loaded successfully!")
```

**Model Loading Time**:
- ControlNet Union: ~15-30 seconds (first run, cached afterward)
- VAE: ~3-5 seconds
- SDXL Pipeline: ~20-40 seconds
- Real-ESRGAN: ~2-5 seconds
- **Total**: 40-80 seconds (first run), ~5-10 seconds (subsequent)

### 7.2 Image Upscaling Process (Deep Dive)

**Tile-Based Processing**:

Real-ESRGAN uses tile-based processing to handle large images without running out of GPU memory.

```python
def enhance(img, outscale=4):
    """
    Tile-based enhancement for large images
    
    Args:
        img: Input image (H × W × 3, BGR, uint8)
        outscale: Output scale factor (2.0 or 4.0)
    
    Returns:
        output: Enhanced image (scale×H × scale×W × 3, BGR, uint8)
    """
    
    # 1. Calculate tile parameters
    tile_size = 400  # Process 400×400 tiles
    tile_pad = 10    # 10 pixel padding for blending
    
    # 2. Pre-pad image if needed
    img = np.pad(img, ((0, 0), (0, 0), (0, 0)), mode='edge')
    
    # 3. Calculate number of tiles
    h, w = img.shape[:2]
    n_tiles_h = math.ceil(h / tile_size)
    n_tiles_w = math.ceil(w / tile_size)
    
    # 4. Initialize output image
    output_h = h * scale
    output_w = w * scale
    output = np.zeros((output_h, output_w, 3), dtype=np.uint8)
    
    # 5. Process each tile
    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            # Calculate tile boundaries
            y1 = max(0, i * tile_size - tile_pad)
            y2 = min(h, (i + 1) * tile_size + tile_pad)
            x1 = max(0, j * tile_size - tile_pad)
            x2 = min(w, (j + 1) * tile_size + tile_pad)
            
            # Extract tile
            tile = img[y1:y2, x1:x2, :]
            
            # Process tile
            tile_output = process_tile(tile)
            
            # Calculate output boundaries
            out_y1 = y1 * scale
            out_y2 = y2 * scale
            out_x1 = x1 * scale
            out_x2 = x2 * scale
            
            # Blend tile into output
            output[out_y1:out_y2, out_x1:out_x2, :] = tile_output
    
    return output
```

**Single Tile Processing**:

```python
def process_tile(tile):
    """
    Process a single tile through RRDBNet
    
    Args:
        tile: numpy array (H × W × 3, BGR, uint8)
    
    Returns:
        output_tile: numpy array (scale×H × scale×W × 3, BGR, uint8)
    """
    
    # 1. Normalize
    tile = tile.astype(np.float32) / 255.0
    
    # 2. Convert BGR to RGB
    tile = tile[:, :, [2, 1, 0]]
    
    # 3. Transpose to (C, H, W)
    tile = np.transpose(tile, (2, 0, 1))
    
    # 4. Convert to PyTorch tensor
    tile_tensor = torch.from_numpy(tile).float()
    
    # 5. Add batch dimension
    tile_tensor = tile_tensor.unsqueeze(0)
    
    # 6. Move to GPU and convert to FP16
    tile_tensor = tile_tensor.cuda().half()
    
    # 7. Forward pass
    with torch.no_grad():
        output_tensor = model(tile_tensor)
    
    # 8. Move to CPU and convert to FP32
    output_tensor = output_tensor.cpu().float()
    
    # 9. Remove batch dimension
    output_tensor = output_tensor.squeeze(0)
    
    # 10. Transpose to (H, W, C)
    output_tile = output_tensor.numpy().transpose(1, 2, 0)
    
    # 11. Convert RGB to BGR
    output_tile = output_tile[:, :, [2, 1, 0]]
    
    # 12. Denormalize
    output_tile = (output_tile * 255.0).clip(0, 255)
    
    # 13. Convert to uint8
    output_tile = output_tile.astype(np.uint8)
    
    return output_tile
```

**Memory Management**:

```python
def cleanup():
    """Clean up GPU memory after processing"""
    
    # 1. Collect Python garbage
    gc.collect()
    
    # 2. Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 3. Collect CUDA IPC resources
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
    
    print("🧹 Cleaned up resources and freed memory")
```

### 7.3 Image Outpainting Process (Deep Dive)

**Mask Preparation Algorithm**:

```python
def prepare_image_and_mask(image, width, height):
    """
    Prepare image and mask for outpainting with 10% overlap
    
    Args:
        image: PIL Image, original image
        width: int, target width
        height: int, target height
    
    Returns:
        background: PIL Image (width × height), prepared background
        mask: PIL Image (width × height, L mode), binary mask
    """
    
    # 1. Calculate scaling to fit target
    scale_factor = min(width / image.width, height / image.height)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    
    # 2. Resize source image
    source = image.resize((new_width, new_height), Image.LANCZOS)
    
    # 3. Calculate overlap (10% of resized dimensions)
    overlap_x = int(new_width * 0.10)
    overlap_y = int(new_height * 0.10)
    overlap_x = max(overlap_x, 1)
    overlap_y = max(overlap_y, 1)
    
    # 4. Calculate margins for center alignment
    margin_x = (width - new_width) // 2
    margin_y = (height - new_height) // 2
    
    # 5. Create white background
    background = Image.new('RGB', (width, height), (255, 255, 255))
    
    # 6. Paste source at center
    background.paste(source, (margin_x, margin_y))
    
    # 7. Create binary mask
    mask = Image.new('L', (width, height), 255)  # All white (generate)
    mask_draw = ImageDraw.Draw(mask)
    
    # 8. Draw black rectangle for original image (with overlap)
    left = margin_x + overlap_x
    right = margin_x + new_width - overlap_x
    top = margin_y + overlap_y
    bottom = margin_y + new_height - overlap_y
    
    mask_draw.rectangle([left, top, right, bottom], fill=0)
    
    return background, mask
```

**Visualization**:

```
Target: 1920×1080, Original: 800×600 → Resized: 800×600 (fits)

┌────────────────────────────────────────────────────────────┐
│                                                            │
│     ┌──────────────────────────────────────────┐          │
│     │╔════════════════════════════════════════╗│          │
│     │║                                        ║│          │
│     │║          Original Image                ║│          │
│     │║          800×600                       ║│          │
│     │║                                        ║│          │
│     │╚════════════════════════════════════════╝│          │
│     └──────────────────────────────────────────┘          │
│                                                            │
└────────────────────────────────────────────────────────────┘
   
   ← margin_x →  ← new_width →
   
Legend:
  White area: To be generated (mask = 255)
  ╔═══╗: Original image area
  Dashed: 10% overlap region (mask = 0, but blended)
```

**Denoising Step Breakdown**:

```python
# Timestep schedule for 8 steps (TCD)
timesteps = [999, 857, 714, 571, 428, 285, 142, 0]

# Initial latents: random noise
latents = torch.randn([1, 4, height//8, width//8], device="cuda", dtype=torch.float16)
latents = latents * scheduler.init_noise_sigma  # Scale by ~14.6

for i, t in enumerate(timesteps):
    # Step 0: t=999 (mostly noise)
    # Latents: Random noise scaled by sigma
    # Goal: Start removing large-scale noise
    
    # Expand for CFG
    latent_input = torch.cat([latents] * 2)  # [2, 4, H/8, W/8]
    
    # Scale by current sigma
    latent_input = scheduler.scale_model_input(latent_input, t)
    
    # ControlNet: Extract control features from image
    down_samples, mid_sample = controlnet(
        latent_input, t, prompt_embeds, control_image, control_type
    )
    
    # UNet: Predict noise
    noise_pred = unet(
        latent_input, t, prompt_embeds,
        down_block_additional_residuals=down_samples,
        mid_block_additional_residual=mid_sample
    )
    
    # CFG: Combine conditional and unconditional predictions
    noise_uncond, noise_cond = noise_pred.chunk(2)
    noise_pred = noise_uncond + 1.5 * (noise_cond - noise_uncond)
    
    # Scheduler step: Remove predicted noise
    latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # Progress: 12.5% (1/8 steps)
```

**Timestep Effects**:

| Timestep | Noise Level | Focus | Latent State |
|----------|-------------|-------|--------------|
| 999 | Very High | Global structure, composition | Random noise |
| 857 | High | Major shapes, object placement | Rough shapes |
| 714 | Medium-High | Object boundaries, large details | Defined forms |
| 571 | Medium | Fine details, textures | Refined shapes |
| 428 | Medium-Low | Surface details, colors | Near-final details |
| 285 | Low | Final touches, smoothing | Almost done |
| 142 | Very Low | Subtle refinements | Polished |
| 0 | None | Final cleanup | Complete |

### 7.4 CUDA Memory Management

**Memory Usage Estimation**:

```
Real-ESRGAN x4 (400×400 tile):
  - Input: 400 × 400 × 3 × 4 bytes = 1.92 MB
  - Model weights: ~67 MB
  - Intermediate activations: ~500 MB (23 RRDB blocks)
  - Output: 1600 × 1600 × 3 × 4 bytes = 30.72 MB
  - Total: ~600 MB per tile

SDXL Outpainting (1920×1080):
  - ControlNet Union: ~5 GB (weights + activations)
  - UNet: ~6.5 GB (weights + activations)
  - VAE: ~334 MB
  - Latents: 1 × 4 × 135 × 240 × 2 bytes = 259 KB
  - Control image: 1 × 3 × 1080 × 1920 × 4 bytes = 24.88 MB
  - Prompt embeddings: 1 × 77 × 2048 × 2 bytes = 315 KB
  - Total: ~12-14 GB

Recommended GPU: 16 GB VRAM
```

**Cleanup Strategy**:

```python
# After each request
try:
    # 1. Delete temporary variables
    del output, result_image, latents
    
    # 2. Python garbage collection
    gc.collect()
    
    # 3. Empty CUDA cache
    torch.cuda.empty_cache()
    
    # 4. Collect CUDA IPC resources
    torch.cuda.ipc_collect()
    
except Exception as e:
    logger.error(f"Cleanup error: {e}")
```

---

## 8. API Documentation

### 8.1 Health Check Endpoint

**Endpoint**: `GET /health`

**Description**: Check API status and model availability

**Request**: None

**Response**:
```json
{
    "status": "ok",
    "models_loaded": {
        "x4plus": true,
        "x2plus": true,
        "sdxl_outpaint": true
    }
}
```

**Status Codes**:
- 200: OK

**Example**:
```bash
curl http://localhost:8000/health
```

### 8.2 Image Upscaling Endpoint

**Endpoint**: `POST /upscale`

**Description**: Upscale an image by 2x or 4x using Real-ESRGAN

**Request**:
- **Content-Type**: `multipart/form-data`
- **Fields**:
  - `image` (required): Image file (JPG, PNG, WEBP)
  - `scale_factor` (required): "2" or "4"
  - `outscale` (optional): Final output scale (default: same as scale_factor)

**Response Success**:
```json
{
    "success": true,
    "image": "iVBORw0KGgoAAAANSUhEUg...",
    "message": "Image upscaled successfully"
}
```

**Response Error**:
```json
{
    "error": "Error message description"
}
```

**Status Codes**:
- 200: Success
- 400: Bad request (no image, invalid parameters)
- 500: Server error

**Example**:
```bash
curl -X POST http://localhost:8000/upscale \
  -F "image=@input.jpg" \
  -F "scale_factor=4" \
  -F "outscale=4.0"
```

**Performance**:
- 512×512 → 2048×2048 (4x): ~3-5 seconds
- 1024×1024 → 4096×4096 (4x): ~10-15 seconds

### 8.3 Image Outpainting Endpoint

**Endpoint**: `POST /outpaint`

**Description**: Expand image to custom dimensions using SDXL

**Request**:
- **Content-Type**: `multipart/form-data`
- **Fields**:
  - `image` (required): Image file (JPG, PNG, WEBP)
  - `target_width` (required): Target width in pixels (integer)
  - `target_height` (required): Target height in pixels (integer)

**Constraints**:
- Target dimensions must be ≥ original dimensions
- Maximum recommended: 1920×1080 (hardware dependent)

**Response Success**:
```json
{
    "success": true,
    "image": "iVBORw0KGgoAAAANSUhEUg...",
    "message": "Outpainted to 1920x1080"
}
```

**Response Error**:
```json
{
    "success": false,
    "error": "Target dimensions must be >= original (1024x768)"
}
```

**Status Codes**:
- 200: Success
- 400: Bad request (invalid dimensions, missing image)
- 500: Server error

**Example**:
```bash
curl -X POST http://localhost:8000/outpaint \
  -F "image=@input.jpg" \
  -F "target_width=1920" \
  -F "target_height=1080"
```

**Performance**:
- 1024×768 → 1920×1080: ~10-15 seconds
- 800×600 → 1600×1200: ~12-18 seconds

---

## 9. Frontend Implementation

### 9.1 Component Architecture

```
app/page.tsx (Main Container)
    │
    ├─→ Logo Component
    │     └─→ SVG logo with animations
    │
    ├─→ ImageUploader Component
    │     ├─→ Drag-and-drop zone
    │     ├─→ File input (hidden)
    │     ├─→ Progress bar (Framer Motion)
    │     └─→ Preview thumbnail
    │
    ├─→ ProcessingOptions Component
    │     ├─→ Upscale options (2x, 4x)
    │     │     └─→ OptionCard × 2
    │     └─→ Outpaint options (width, height)
    │           └─→ Input × 2 (shadcn/ui)
    │
    ├─→ OutputDisplay Component
    │     ├─→ Processing animation (dual spinners)
    │     ├─→ Result image display (Next.js Image)
    │     └─→ Download button
    │
    └─→ Error Display (conditional)
```

### 9.2 State Flow Diagram

```
Initial State:
┌─────────────────────────────────────────┐
│ uploadedImage: null                     │
│ isProcessing: false                     │
│ processedImage: null                    │
│ selectedUpscale: null                   │
│ outpaintWidth: null                     │
│ outpaintHeight: null                    │
│ error: null                             │
└─────────────────────────────────────────┘
              ↓
       User uploads image
              ↓
┌─────────────────────────────────────────┐
│ uploadedImage: "data:image/png;base64"  │
│ originalImageWidth: 1024                │
│ originalImageHeight: 768                │
└─────────────────────────────────────────┘
              ↓
    User selects option (2x)
              ↓
┌─────────────────────────────────────────┐
│ selectedUpscale: "2x"                   │
└─────────────────────────────────────────┘
              ↓
    User clicks "Process Image"
              ↓
┌─────────────────────────────────────────┐
│ isProcessing: true                      │
└─────────────────────────────────────────┘
              ↓
      API call (upscaleImage)
              ↓
       Response received
              ↓
┌─────────────────────────────────────────┐
│ isProcessing: false                     │
│ processedImage: "data:image/png;base64" │
└─────────────────────────────────────────┘
              ↓
    User clicks "Download"
              ↓
        File downloaded
```

### 9.3 Animation System

**Framer Motion Variants**:

```typescript
// Fade in from top
const fadeInDown = {
    initial: { opacity: 0, y: -20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.5 }
}

// Fade in from bottom
const fadeInUp = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.5 }
}

// Scale in
const scaleIn = {
    initial: { opacity: 0, scale: 0.9 },
    animate: { opacity: 1, scale: 1 },
    transition: { duration: 0.3 }
}

// Height expand
const expandHeight = {
    initial: { opacity: 0, height: 0 },
    animate: { opacity: 1, height: "auto" },
    exit: { opacity: 0, height: 0 },
    transition: { duration: 0.5 }
}

// Button hover
const buttonHover = {
    whileHover: { scale: 1.05 },
    whileTap: { scale: 0.95 }
}
```

**Processing Animation**:

```typescript
// Dual spinner rotation
<motion.div
    className="border-t-teal-400"
    animate={{ rotate: 360 }}
    transition={{
        duration: 1,
        repeat: Infinity,
        ease: "linear"
    }}
/>
<motion.div
    className="border-l-blue-500"
    animate={{ rotate: -360 }}
    transition={{
        duration: 1.5,
        repeat: Infinity,
        ease: "linear"
    }}
/>
```

### 9.4 Responsive Design

**Breakpoints** (Tailwind CSS):
- **sm**: 640px (tablets)
- **md**: 768px (small laptops)
- **lg**: 1024px (desktops)
- **xl**: 1280px (large desktops)

**Layout Adjustments**:
```typescript
// Mobile: Stack vertically
className="flex flex-col gap-4"

// Desktop: Horizontal layout
className="md:flex-row md:gap-6"

// Responsive padding
className="px-4 md:px-6 lg:px-8"

// Responsive text
className="text-2xl md:text-3xl lg:text-4xl"
```

---

## 10. Deployment & Containerization

### 10.1 Docker Architecture

**Multi-Container Setup**:

```yaml
version: '3.8'

services:
  backend:
    image: pojesh/prism_24gai15vitc-backend:1.0
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app/backend
      - ~/.cache/huggingface:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all

  frontend:
    image: pojesh/prism_24gai15vitc-frontend:1.0
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app/frontend
    depends_on:
      - backend
```

### 10.2 Backend Dockerfile

```dockerfile
# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start server
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 10.3 Frontend Dockerfile

```dockerfile
# Use Node.js base image
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy application code
COPY . .

# Build Next.js application
RUN npm run build

# Expose port
EXPOSE 3000

# Start server
CMD ["npm", "start"]
```

### 10.4 Deployment Steps

**Local Deployment**:

```bash
# 1. Clone repository
git clone https://github.ecodesamsung.com/SRIB-PRISM/VITC_24GAI15VITC_GenAI_Image_Resizers.git
cd VITC_24GAI15VITC_GenAI_Image_Resizers

# 2. Start with Docker Compose
docker compose up

# 3. Access services
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

**Production Considerations**:

1. **Environment Variables**:
   ```bash
   # Backend
   HUGGINGFACE_TOKEN=<token>
   CUDA_VISIBLE_DEVICES=0
   MAX_BATCH_SIZE=1
   
   # Frontend
   NEXT_PUBLIC_API_URL=https://api.example.com
   ```

2. **Reverse Proxy** (Nginx):
   ```nginx
   server {
       listen 80;
       server_name example.com;
       
       location /api/ {
           proxy_pass http://localhost:8000/;
           proxy_set_header Host $host;
           client_max_body_size 50M;
       }
       
       location / {
           proxy_pass http://localhost:3000/;
           proxy_set_header Host $host;
       }
   }
   ```

3. **SSL/TLS**:
   ```bash
   certbot --nginx -d example.com
   ```

4. **Resource Limits**:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '4'
         memory: 16G
       reservations:
         memory: 8G
   ```

---

## 11. Error Handling & Resource Management

### 11.1 Backend Error Handling

**Exception Hierarchy**:

```python
try:
    # Main processing logic
    result = process_image(image)
    
except FileNotFoundError as e:
    # Model weights not found
    return JSONResponse(
        content={"error": "Model weights not found"},
        status_code=500
    )

except torch.cuda.OutOfMemoryError:
    # GPU OOM
    cleanup()
    return JSONResponse(
        content={"error": "GPU out of memory. Try smaller image or restart server."},
        status_code=507
    )

except ValueError as e:
    # Invalid input parameters
    return JSONResponse(
        content={"error": str(e)},
        status_code=400
    )

except Exception as e:
    # Unexpected error
    logger.error(f"Unexpected error: {e}")
    return JSONResponse(
        content={"error": "Internal server error"},
        status_code=500
    )

finally:
    # Always cleanup
    cleanup()
    # Delete temporary files
    if os.path.exists(input_path):
        os.remove(input_path)
    if os.path.exists(output_path):
        os.remove(output_path)
```

### 11.2 Frontend Error Handling

```typescript
try {
    setIsProcessing(true)
    setError(null)
    
    const result = await upscaleImage(uploadedImage, options)
    
    if (!result.success) {
        throw new Error(result.error)
    }
    
    setProcessedImage(result.imageData)
    
} catch (error) {
    // Network error
    if (error instanceof TypeError) {
        setError("Failed to connect to server. Please check if the backend is running.")
    }
    // API error
    else if (error instanceof Error) {
        setError(error.message)
    }
    // Unknown error
    else {
        setError("An unexpected error occurred. Please try again.")
    }
    
} finally {
    setIsProcessing(false)
}
```

### 11.3 Resource Management Best Practices

**Memory Management**:

```python
# 1. Use context managers
with torch.no_grad():
    output = model(input)

# 2. Delete large tensors
del latents, prompt_embeds
gc.collect()

# 3. Empty CUDA cache regularly
torch.cuda.empty_cache()

# 4. Use FP16 when possible
model = model.half()
input = input.half()

# 5. Limit batch size
BATCH_SIZE = 1  # Process one image at a time
```

**File Management**:

```python
# Use unique filenames
import uuid
filename = f"{uuid.uuid4()}_image.png"

# Always cleanup in finally block
try:
    process_file(filename)
finally:
    if os.path.exists(filename):
        os.remove(filename)

# Use temporary directories
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    temp_file = os.path.join(tmpdir, "temp.png")
    # Files automatically deleted when context exits
```

**Connection Management**:

```typescript
// Set request timeout
const controller = new AbortController()
const timeoutId = setTimeout(() => controller.abort(), 60000)  // 60s timeout

try {
    const response = await fetch(url, {
        signal: controller.signal
    })
} catch (error) {
    if (error.name === 'AbortError') {
        console.error('Request timed out')
    }
} finally {
    clearTimeout(timeoutId)
}
```

---

## Conclusion

This document provides a comprehensive overview of the Image Enhancer - Samsung Prism implementation, covering architecture, modules, workflows, data flows, and deployment strategies. The system leverages state-of-the-art AI models (Real-ESRGAN and SDXL) to provide high-quality image upscaling and outpainting capabilities through a modern, user-friendly web interface.

### Key Highlights

- **Dual Functionality**: Super-resolution and intelligent outpainting
- **GPU Acceleration**: CUDA-optimized for fast processing
- **Modern Stack**: FastAPI + Next.js 15 + React 19
- **Production Ready**: Docker containerization with GPU support
- **Robust Error Handling**: Comprehensive validation and cleanup
- **Scalable Architecture**: Modular design for easy extension

### Future Enhancements

- **Batch Processing**: Multiple images at once
- **Real-time Streaming**: Progressive result updates
- **Model Fine-tuning**: Custom models for specific use cases
- **API Rate Limiting**: Production-grade throttling
- **Cloud Deployment**: AWS/GCP/Azure integration
- **CDN Integration**: Faster asset delivery

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Maintainer**: Samsung Prism VITC Team
