"use client"

import { useState } from "react"
import { Upload, Star, ImageIcon } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import Logo from "@/components/logo"
import ImageUploader from "@/components/image-uploader"
import ProcessingOptions from "@/components/processing-options"
import OutputDisplay from "@/components/output-display"
import { upscaleImage, outpaintImage } from "@/lib/api"

export default function Home() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processedImage, setProcessedImage] = useState<string | null>(null)
  const [selectedUpscale, setSelectedUpscale] = useState<"2x" | "4x" | null>(null)
  // Removed selectedAspectRatio, added outpaintWidth and outpaintHeight
  const [outpaintWidth, setOutpaintWidth] = useState<number | null>(null);
  const [outpaintHeight, setOutpaintHeight] = useState<number | null>(null);
  const [originalImageWidth, setOriginalImageWidth] = useState<number | null>(null);
  const [originalImageHeight, setOriginalImageHeight] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null)

  const handleImageUpload = (imageDataUrl: string, width?: number, height?: number) => {
    if (!imageDataUrl) {
      setUploadedImage(null);
      setOriginalImageWidth(null);
      setOriginalImageHeight(null);
      setProcessedImage(null)
      setSelectedUpscale(null)
      setOutpaintWidth(null)
      setOutpaintHeight(null)
      setError(null)
      return;
    }
    setUploadedImage(imageDataUrl)
    setOriginalImageWidth(width || null);
    setOriginalImageHeight(height || null);
    setProcessedImage(null)
    // Reset options on new image upload
    setSelectedUpscale(null)
    setOutpaintWidth(null)
    setOutpaintHeight(null)
    setError(null)
  }

  const handleProcessImage = async () => {
    if (!uploadedImage) return;
    if (!selectedUpscale && (outpaintWidth === null || outpaintHeight === null)) return;

    setIsProcessing(true)
    setError(null)

    try {
      if (selectedUpscale) {
        const scaleFactor = selectedUpscale === "2x" ? "2" : "4";
        const result = await upscaleImage(uploadedImage, {
          scaleFactor: scaleFactor as "2" | "4"
        });
        if (result.success) {
          setProcessedImage(result.imageData || null);
        } else {
          setError(result.error);
        }
      } else if (outpaintWidth !== null && outpaintHeight !== null) {
        // Use outpaintWidth and outpaintHeight for outpainting
        const result = await outpaintImage(uploadedImage, {
          width: outpaintWidth,
          height: outpaintHeight
        });
        if (result.success) {
          setProcessedImage(result.imageData || null);
        } else {
          setError(result.error);
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "An unexpected error occurred";
      setError(message);
    } finally {
      setIsProcessing(false);
    }
  }

  const isProcessButtonDisabled =
    !uploadedImage || (!selectedUpscale && (outpaintWidth === null || outpaintHeight === null));

  return (
    <main className="min-h-screen bg-gradient-to-b from-black to-gray-900 text-white">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="flex justify-center mb-8"
        >
          <Logo />
        </motion.div>

        <motion.h1
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.5 }}
          className="text-3xl md:text-4xl font-bold text-center mb-12"
        >
          Galaxy Image Enhancer
        </motion.h1>

        <div className="space-y-8">
          {/* Image Upload Section */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.5 }}
            className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700"
          >
            <h2 className="text-xl font-semibold mb-4 flex items-center">
              <Upload className="mr-2 text-teal-400" size={20} />
              Upload Your Image
            </h2>
            <ImageUploader onImageUpload={handleImageUpload} uploadedImage={uploadedImage} />
          </motion.section>

          {/* Processing Options Section */}
          <AnimatePresence>
            {uploadedImage && (
              <motion.section
                initial={{ opacity: 0, height: 0, y: 20 }}
                animate={{ opacity: 1, height: "auto", y: 0 }}
                exit={{ opacity: 0, height: 0, y: -20 }}
                transition={{ duration: 0.5 }}
                className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 overflow-hidden"
              >
                <h2 className="text-xl font-semibold mb-4 flex items-center">
                  <Star className="mr-2 text-teal-400" size={20} />
                  Processing Options
                </h2>
                <ProcessingOptions
                  selectedUpscale={selectedUpscale}
                  setSelectedUpscale={setSelectedUpscale}
                  outpaintWidth={outpaintWidth}
                  setOutpaintWidth={setOutpaintWidth}
                  outpaintHeight={outpaintHeight}
                  setOutpaintHeight={setOutpaintHeight}
                  originalImageWidth={originalImageWidth}
                  originalImageHeight={originalImageHeight}
                />

                <div className="mt-6 flex justify-center">
                  <motion.button
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.98 }}
                    className={`px-6 py-3 rounded-lg font-medium transition-colors duration-300 ${isProcessButtonDisabled
                        ? "bg-gray-600 cursor-not-allowed"
                        : "bg-gradient-to-r from-blue-600 to-teal-400 hover:from-blue-500 hover:to-teal-300"
                      }`}
                    disabled={isProcessButtonDisabled || undefined}
                    onClick={handleProcessImage}
                  >
                    Process Image
                  </motion.button>
                </div>
              </motion.section>
            )}
          </AnimatePresence>

          {/* Output Display Section */}
          <AnimatePresence>
            {(uploadedImage || isProcessing || processedImage) && (
              <motion.section
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.5 }}
                className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700"
              >
                <h2 className="text-xl font-semibold mb-4 flex items-center">
                  <ImageIcon className="mr-2 text-teal-400" size={20} />
                  Result
                </h2>
                <OutputDisplay isProcessing={isProcessing} processedImage={processedImage} />
                {error && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="mt-4 p-3 bg-red-900/50 border border-red-700 rounded-md text-red-200 text-center"
                  >
                    {error}
                  </motion.div>
                )}
              </motion.section>
            )}
          </AnimatePresence>
        </div>
      </div>
    </main>
  )
}
