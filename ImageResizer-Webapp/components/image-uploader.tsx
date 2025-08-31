"use client"

import type React from "react"

import Image from "next/image"
import { useState, useRef } from "react"
import { Upload, X } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"

interface ImageUploaderProps {
  onImageUpload: (imageDataUrl: string, width?: number, height?: number) => void
  uploadedImage: string | null
}

export default function ImageUploader({ onImageUpload, uploadedImage }: ImageUploaderProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0]
      if (file.type.match("image.*")) {
        processFile(file)
      }
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0])
    }
  }

  const processFile = (file: File) => {
    const reader = new FileReader()

    // Simulate upload progress
    let progress = 0
    const interval = setInterval(() => {
      progress += 10
      setUploadProgress(progress)
      if (progress >= 100) {
        clearInterval(interval)
      }
    }, 100)

    reader.onload = (e) => {
      if (e.target?.result) {
        const imageDataUrl = e.target.result as string;
        const img = new window.Image();
        img.onload = () => {
          onImageUpload(imageDataUrl, img.naturalWidth, img.naturalHeight);
          setTimeout(() => setUploadProgress(0), 500); // Keep existing progress logic
        };
        img.onerror = () => {
          // Handle potential error in loading image, maybe call onImageUpload without dimensions
          onImageUpload(imageDataUrl); // Or some error state
          setTimeout(() => setUploadProgress(0), 500);
        };
        img.src = imageDataUrl;
      }
    };

    reader.readAsDataURL(file)
  }

  const handleButtonClick = () => {
    fileInputRef.current?.click()
  }

  const handleRemoveImage = () => {
    onImageUpload("", undefined, undefined)
  }

  return (
    <div className="w-full">
      <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="image/*" className="hidden" />

      <div className="flex flex-col md:flex-row gap-6 items-center">
        <div
          className={`flex-1 w-full border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center cursor-pointer transition-colors duration-300 ${
            isDragging ? "border-teal-400 bg-teal-400/10" : "border-gray-600 hover:border-gray-400"
          }`}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={handleButtonClick}
          style={{ minHeight: "180px" }}
        >
          <Upload className="w-12 h-12 mb-3 text-gray-400" />
          <p className="text-center text-gray-300">
            Drag & drop your image here or <span className="text-teal-400">browse</span>
          </p>
          <p className="text-xs text-gray-500 mt-2">Supports JPG, PNG, WEBP</p>

          <AnimatePresence>
            {uploadProgress > 0 && uploadProgress < 100 && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="w-full mt-4"
              >
                <div className="w-full bg-gray-700 rounded-full h-2.5 mt-2">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-teal-400 h-2.5 rounded-full transition-all duration-300 ease-out"
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {uploadedImage && (
          <div className="relative">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="relative w-32 h-32 rounded-lg overflow-hidden border border-gray-700"
            >
              <Image
                src={uploadedImage || "/placeholder.svg"}
                alt="Uploaded preview"
                fill
                className="object-cover"
              />
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  handleRemoveImage()
                }}
                className="absolute top-1 right-1 bg-black/70 rounded-full p-1 hover:bg-black transition-colors"
              >
                <X size={16} className="text-white" />
              </button>
            </motion.div>
          </div>
        )}
      </div>
    </div>
  )
}
