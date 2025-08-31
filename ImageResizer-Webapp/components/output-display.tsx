"use client"
import Image from "next/image"
import { motion, AnimatePresence } from "framer-motion"
import { Download } from "lucide-react"

interface OutputDisplayProps {
  isProcessing: boolean
  processedImage: string | null
}

export default function OutputDisplay({ isProcessing, processedImage }: OutputDisplayProps) {
  const handleDownload = () => {
    if (processedImage) {
      const link = document.createElement("a")
      link.href = processedImage
      link.download = "enhanced-image.jpg"
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
  }

  return (
    <div className="flex flex-col items-center">
      <div className="w-full max-w-lg h-64 rounded-lg border border-gray-700 flex items-center justify-center overflow-hidden">
        <AnimatePresence mode="wait">
          {isProcessing ? (
            <motion.div
              key="processing"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center"
            >
              <div className="relative w-16 h-16">
                <motion.div
                  className="absolute inset-0 border-4 border-transparent border-t-teal-400 rounded-full"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
                />
                <motion.div
                  className="absolute inset-0 border-4 border-transparent border-l-blue-500 rounded-full"
                  animate={{ rotate: -360 }}
                  transition={{ duration: 1.5, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
                />
              </div>
              <p className="mt-4 text-gray-300">Processing your image...</p>
            </motion.div>
          ) : processedImage ? (
            <motion.div
              key="result"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="w-full h-full relative"
            >
              <Image
                src={processedImage || "/placeholder.svg"}
                alt="Processed result"
                fill
                className="object-contain"
              />
            </motion.div>
          ) : (
            <motion.div
              key="placeholder"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-center text-gray-500"
            >
              <p>Your enhanced image will appear here</p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {processedImage && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mt-4"
        >
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleDownload}
            className="flex items-center px-4 py-2 bg-gradient-to-r from-blue-600 to-teal-400 rounded-lg font-medium transition-colors hover:from-blue-500 hover:to-teal-300"
          >
            <Download className="mr-2 h-4 w-4" />
            Download Image
          </motion.button>
        </motion.div>
      )}
    </div>
  )
}
