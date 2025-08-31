"use client"

import { Star } from "lucide-react"
import { motion } from "framer-motion"

export default function Logo() {
  return (
    <div className="flex items-center">
      <motion.div
        className="relative w-10 h-10 mr-2"
        animate={{ rotate: 360 }}
        transition={{ duration: 20, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
      >
        <motion.div
          className="absolute inset-0 flex items-center justify-center"
          initial={{ scale: 0.8, opacity: 0.8 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY, repeatType: "reverse" }}
        >
          <Star className="text-blue-500 w-8 h-8" />
        </motion.div>
        <motion.div
          className="absolute inset-0 flex items-center justify-center rotate-45"
          initial={{ scale: 1, opacity: 1 }}
          animate={{ scale: 0.8, opacity: 0.8 }}
          transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY, repeatType: "reverse", delay: 1 }}
        >
          <Star className="text-teal-400 w-6 h-6" />
        </motion.div>
      </motion.div>
      <div className="font-bold text-xl">Galaxy Image Enhancer</div>
    </div>
  )
}
