# 1. Base Image with NVIDIA CUDA and cuDNN support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 2. Set up Environment Variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/Real-ESRGAN-master

# 3. Install System Dependencies and Python 3.10
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev git \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 && \
    rm -rf /var/lib/apt/lists/*


# 4. Set up Working Directory
WORKDIR /app

# 5. Copy requirements files
# COPY Real-ESRGAN-master/requirements.txt /app/Real-ESRGAN-master/requirements.txt
COPY requirements.txt /app/requirements.txt

# 6. Install Python Dependencies
# Install Real-ESRGAN dependencies first, then the main app dependencies
# RUN pip install --no-cache-dir -r /app/Real-ESRGAN-master/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 7. Copy the rest of the application code
COPY . /app/

# 8. Download Models
# The script prompts for input, so we pipe '3' (for all models) into it.
# RUN echo "3" | python download_models.py

# 9. Expose Port
EXPOSE 8000

# 10. Command to Run the Application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
