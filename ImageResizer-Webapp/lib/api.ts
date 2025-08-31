/**
 * API service for image processing operations
 * Integrates with the backend API for upscaling and outpainting images
 */

/**
 * Upscales an image using the backend API
 * @param imageFile - The image file or data URL to upscale
 * @param options - Configuration options for upscaling
 */
export async function upscaleImage(imageData: string, options: {
  scaleFactor: '2' | '4';
  outscale?: string;
}) {
  const {
    scaleFactor = '4',
    outscale = scaleFactor === '2' ? '2.0' : '4.0'
  } = options;

  // Convert data URL to File object
  const imageFile = dataURLtoFile(imageData, 'image.png');
  
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('scale_factor', scaleFactor);
  formData.append('outscale', outscale);
  
  try {
    const response = await fetch('http://localhost:8000/upscale', {
      method: 'POST',
      body: formData,
    });
    
    const data = await response.json();
    if (data.success) {
      return {
        success: true,
        imageData: `data:image/png;base64,${data.image}`,
        message: data.message
      };
    } else {
      return {
        success: false,
        error: data.error || 'Unknown error occurred'
      };
    }
  } catch (error) {
    console.error('Error upscaling image:', error);
    const message = error instanceof Error ? error.message : 'Failed to connect to the server';
    return {
      success: false,
      error: message,
    };
  }
}

/**
 * Outpaints an image using the backend API
 * @param imageFile - The image file or data URL to outpaint
 * @param options - Configuration options for outpainting
 */
export async function outpaintImage(imageData: string, options: {
  width?: number;
  height?: number;
}) {
  const {
    width = 1920,
    height = 1080
  } = options;

  // Convert data URL to File object
  const imageFile = dataURLtoFile(imageData, 'image.png');
  
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('target_width', width.toString());
  formData.append('target_height', height.toString());
  
  try {
    const response = await fetch('http://localhost:8000/outpaint', {
      method: 'POST',
      body: formData,
    });
    
    const data = await response.json();
    if (data.success) {
      return {
        success: true,
        imageData: `data:image/png;base64,${data.image}`,
        message: data.message
      };
    } else {
      return {
        success: false,
        error: data.error || 'Unknown error occurred'
      };
    }
  } catch (error) {
    console.error('Error outpainting image:', error);
    const message = error instanceof Error ? error.message : 'Failed to connect to the server';
    return {
      success: false,
      error: message,
    };
  }
}

/**
 * Converts a data URL to a File object
 * @param dataUrl - The data URL to convert
 * @param filename - The filename to use for the File object
 */
function dataURLtoFile(dataUrl: string, filename: string): File {
  const arr = dataUrl.split(',');
  const mime = arr[0].match(/:(.*?);/)![1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  
  return new File([u8arr], filename, { type: mime });
}