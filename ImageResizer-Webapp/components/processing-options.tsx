"use client"
import { motion } from "framer-motion"
import { Input } from "@/components/ui/input"

interface ProcessingOptionsProps {
  selectedUpscale: "2x" | "4x" | null
  setSelectedUpscale: (value: "2x" | "4x" | null) => void
  outpaintWidth: number | null
  setOutpaintWidth: (value: number | null) => void
  outpaintHeight: number | null
  setOutpaintHeight: (value: number | null) => void
  originalImageWidth?: number | null;
  originalImageHeight?: number | null;
  // Removed selectedAspectRatio and setSelectedAspectRatio
}

export default function ProcessingOptions({
  selectedUpscale,
  setSelectedUpscale,
  outpaintWidth,
  setOutpaintWidth,
  outpaintHeight,
  setOutpaintHeight,
  originalImageWidth,
  originalImageHeight,
}: ProcessingOptionsProps) {
  const handleUpscaleSelect = (value: "2x" | "4x") => {
    if (selectedUpscale === value) {
      setSelectedUpscale(null)
      // When an upscale option is selected, clear outpaint dimensions
      setOutpaintWidth(null)
      setOutpaintHeight(null)
    } else {
      setSelectedUpscale(value)
      setOutpaintWidth(null)
      setOutpaintHeight(null)
    }
  }

  const handleWidthChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value ? parseInt(e.target.value, 10) : null;
    setOutpaintWidth(value);
    if (value !== null && outpaintHeight !== null) {
      setSelectedUpscale(null); // Clear upscale if dimensions are set
    }
  };

  const handleHeightChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value ? parseInt(e.target.value, 10) : null;
    setOutpaintHeight(value);
    if (value !== null && outpaintWidth !== null) {
      setSelectedUpscale(null); // Clear upscale if dimensions are set
    }
  };

  return (
    <div className="space-y-6">
      {/* Upscaling Options */}
      <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-700">
        <h3 className="text-lg font-medium mb-3">Upscale Image</h3>
        <div className="grid grid-cols-2 gap-3">
          <OptionCard
            title="Upscale 2x"
            description="Double the resolution"
            isSelected={selectedUpscale === "2x"}
            onClick={() => handleUpscaleSelect("2x")}
          />
          <OptionCard
            title="Upscale 4x"
            description="Quadruple the resolution"
            isSelected={selectedUpscale === "4x"}
            onClick={() => handleUpscaleSelect("4x")}
          />
        </div>
      </div>

      {/* Outpainting Options - Replaced Aspect Ratio Conversion */}
      <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-700">
        <h3 className="text-lg font-medium mb-3">Outpaint Image (Custom Dimensions)</h3>
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label htmlFor="outpaintWidth" className="block text-sm font-medium text-gray-300 mb-1">
              Width (px)
            </label>
            <Input
              id="outpaintWidth"
              type="number"
              placeholder={originalImageWidth ? String(originalImageWidth) : "width"}
              value={outpaintWidth === null ? "" : outpaintWidth}
              onChange={handleWidthChange}
              className="bg-gray-800 border-gray-600 text-white placeholder-gray-500 focus:ring-teal-500 focus:border-teal-500"
            />
          </div>
          <div>
            <label htmlFor="outpaintHeight" className="block text-sm font-medium text-gray-300 mb-1">
              Height (px)
            </label>
            <Input
              id="outpaintHeight"
              type="number"
              placeholder={originalImageHeight ? String(originalImageHeight) : "height"}
              value={outpaintHeight === null ? "" : outpaintHeight}
              onChange={handleHeightChange}
              className="bg-gray-800 border-gray-600 text-white placeholder-gray-500 focus:ring-teal-500 focus:border-teal-500"
            />
          </div>
        </div>
         {outpaintWidth !== null && outpaintHeight !== null && (
          <p className="text-xs text-gray-400">
            Outpainting to {outpaintWidth}px x {outpaintHeight}px. Upscaling will be disabled.
          </p>
        )}
        {selectedUpscale && (outpaintWidth !== null || outpaintHeight !== null) && (
            <p className="text-xs text-yellow-400">
                Clear width/height to enable upscaling, or select an upscale option to clear width/height.
            </p>
        )}
      </div>
    </div>
  )
}

interface OptionCardProps {
  title: string
  description: string
  isSelected: boolean
  onClick: () => void
}

function OptionCard({ title, description, isSelected, onClick }: OptionCardProps) {
  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className={`relative cursor-pointer rounded-lg p-3 border transition-all duration-200 ${isSelected ? "border-teal-400 bg-teal-400/10" : "border-gray-600 hover:border-gray-500"}`}
    >
      <div className="flex items-start">
        <div
          className={`w-4 h-4 rounded-full mr-2 mt-1 border-2 flex-shrink-0 ${isSelected ? "border-teal-400 bg-teal-400" : "border-gray-500"}`}
        >
          {isSelected && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="w-full h-full rounded-full bg-white scale-50"
            />
          )}
        </div>
        <div>
          <h4 className="font-medium">{title}</h4>
          <p className="text-xs text-gray-400">{description}</p>
        </div>
      </div>
    </motion.div>
  )
}
