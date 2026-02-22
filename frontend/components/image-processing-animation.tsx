'use client'

import { useEffect, useState, useRef } from 'react'

interface ImageProcessingAnimationProps {
  imageSrc: string
  isProcessing: boolean
  resultImage: string | null
  modelName: string
}

export function ImageProcessingAnimation({
  imageSrc,
  isProcessing,
  resultImage,
  modelName,
}: ImageProcessingAnimationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [phase, setPhase] = useState<'idle' | 'scanning' | 'segmenting' | 'done'>('idle')
  const [scanProgress, setScanProgress] = useState(0)
  const [segmentProgress, setSegmentProgress] = useState(0)
  const animationRef = useRef<number | null>(null)
  const phaseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (isProcessing) {
      setPhase('scanning')
      setScanProgress(0)
      setSegmentProgress(0)

      // Phase 1: Scanning (0-2s)
      phaseTimerRef.current = setTimeout(() => {
        setPhase('segmenting')
        // Phase 2: Segmenting (2-4.5s)
        phaseTimerRef.current = setTimeout(() => {
          setPhase('done')
        }, 2500)
      }, 2000)
    } else if (!isProcessing && resultImage) {
      setPhase('done')
    } else {
      setPhase('idle')
    }

    return () => {
      if (phaseTimerRef.current) clearTimeout(phaseTimerRef.current)
    }
  }, [isProcessing, resultImage])

  // Scanning animation
  useEffect(() => {
    if (phase !== 'scanning') return

    let start: number | null = null
    const duration = 2000

    const animate = (timestamp: number) => {
      if (!start) start = timestamp
      const elapsed = timestamp - start
      const progress = Math.min(elapsed / duration, 1)
      setScanProgress(progress)

      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate)
      }
    }

    animationRef.current = requestAnimationFrame(animate)
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [phase])

  // Segmenting animation
  useEffect(() => {
    if (phase !== 'segmenting') return

    let start: number | null = null
    const duration = 2500

    const animate = (timestamp: number) => {
      if (!start) start = timestamp
      const elapsed = timestamp - start
      const progress = Math.min(elapsed / duration, 1)
      setSegmentProgress(progress)

      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate)
      }
    }

    animationRef.current = requestAnimationFrame(animate)
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [phase])

  // Canvas drawing
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      canvas.width = img.width
      canvas.height = img.height

      const draw = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height)

        // Draw the original image
        ctx.drawImage(img, 0, 0)

        if (phase === 'scanning') {
          // Scanning line effect
          const scanY = scanProgress * canvas.height

          // Dark overlay below scan line
          ctx.fillStyle = 'rgba(47, 194, 217, 0.08)'
          ctx.fillRect(0, 0, canvas.width, scanY)

          // Scan line
          const gradient = ctx.createLinearGradient(0, scanY - 4, 0, scanY + 4)
          gradient.addColorStop(0, 'rgba(47, 194, 217, 0)')
          gradient.addColorStop(0.5, 'rgba(47, 194, 217, 0.9)')
          gradient.addColorStop(1, 'rgba(47, 194, 217, 0)')
          ctx.fillStyle = gradient
          ctx.fillRect(0, scanY - 4, canvas.width, 8)

          // Glow effect around scan line
          const glow = ctx.createLinearGradient(0, scanY - 30, 0, scanY + 30)
          glow.addColorStop(0, 'rgba(47, 194, 217, 0)')
          glow.addColorStop(0.5, 'rgba(47, 194, 217, 0.15)')
          glow.addColorStop(1, 'rgba(47, 194, 217, 0)')
          ctx.fillStyle = glow
          ctx.fillRect(0, scanY - 30, canvas.width, 60)

          // Grid overlay on scanned region
          ctx.strokeStyle = 'rgba(47, 194, 217, 0.12)'
          ctx.lineWidth = 0.5
          const gridSize = 20
          for (let x = 0; x < canvas.width; x += gridSize) {
            ctx.beginPath()
            ctx.moveTo(x, 0)
            ctx.lineTo(x, scanY)
            ctx.stroke()
          }
          for (let y = 0; y < scanY; y += gridSize) {
            ctx.beginPath()
            ctx.moveTo(0, y)
            ctx.lineTo(canvas.width, y)
            ctx.stroke()
          }
        }

        if (phase === 'segmenting') {
          // Full grid overlay
          ctx.strokeStyle = 'rgba(47, 194, 217, 0.08)'
          ctx.lineWidth = 0.5
          const gridSize = 20
          for (let x = 0; x < canvas.width; x += gridSize) {
            ctx.beginPath()
            ctx.moveTo(x, 0)
            ctx.lineTo(x, canvas.height)
            ctx.stroke()
          }
          for (let y = 0; y < canvas.height; y += gridSize) {
            ctx.beginPath()
            ctx.moveTo(0, y)
            ctx.lineTo(canvas.width, y)
            ctx.stroke()
          }

          // Color overlay that builds up from left to right
          const colors = [
            'rgba(45, 106, 79, 0.3)',   // Trees
            'rgba(82, 183, 136, 0.3)',   // Lush Bushes
            'rgba(212, 160, 23, 0.3)',   // Dry Grass
            'rgba(160, 120, 80, 0.3)',   // Dry Bushes
            'rgba(200, 169, 110, 0.3)',  // Landscape
            'rgba(144, 200, 224, 0.3)',  // Sky
          ]

          const revealX = segmentProgress * canvas.width
          const blockW = canvas.width / 6

          for (let i = 0; i < 6; i++) {
            const startX = i * blockW
            const endX = Math.min(startX + blockW, revealX)
            if (endX > startX) {
              // Top half or bottom half variation
              if (i < 3) {
                ctx.fillStyle = colors[i]
                ctx.fillRect(startX, canvas.height * 0.5, endX - startX, canvas.height * 0.5)
              } else {
                ctx.fillStyle = colors[i]
                ctx.fillRect(startX, 0, endX - startX, canvas.height * 0.5)
              }
            }
          }

          // Vertical reveal line
          const lineGradient = ctx.createLinearGradient(revealX - 3, 0, revealX + 3, 0)
          lineGradient.addColorStop(0, 'rgba(136, 231, 178, 0)')
          lineGradient.addColorStop(0.5, 'rgba(136, 231, 178, 0.8)')
          lineGradient.addColorStop(1, 'rgba(136, 231, 178, 0)')
          ctx.fillStyle = lineGradient
          ctx.fillRect(revealX - 3, 0, 6, canvas.height)

          // Pulse dots at key points
          const dotCount = Math.floor(segmentProgress * 8)
          for (let d = 0; d < dotCount; d++) {
            const dx = (d / 8) * canvas.width + blockW / 2
            const dy = canvas.height * (d % 2 === 0 ? 0.3 : 0.7)
            const pulse = 1 + 0.3 * Math.sin(Date.now() / 200 + d)
            ctx.beginPath()
            ctx.arc(dx, dy, 4 * pulse, 0, Math.PI * 2)
            ctx.fillStyle = 'rgba(47, 194, 217, 0.7)'
            ctx.fill()
          }
        }

        if (phase !== 'done') {
          requestAnimationFrame(draw)
        }
      }

      draw()
    }
    img.src = imageSrc
  }, [imageSrc, phase, scanProgress, segmentProgress])

  if (phase === 'done' && resultImage) {
    return (
      <div className="w-full h-full relative">
        <img
          src={resultImage}
          alt="Segmentation Mask"
          className="w-full h-full object-cover animate-in fade-in duration-500"
        />
      </div>
    )
  }

  if (phase === 'idle' && !isProcessing) {
    return null
  }

  return (
    <div className="w-full h-full relative overflow-hidden">
      <canvas
        ref={canvasRef}
        className="w-full h-full object-cover"
      />
      {/* Overlay info */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-4">
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-8 h-8">
            <div className="absolute inset-0 rounded-full border-2 border-primary/30" />
            <div
              className="absolute inset-0 rounded-full border-2 border-primary border-t-transparent animate-spin"
              style={{ animationDuration: '1s' }}
            />
          </div>
          <div>
            <p className="text-xs font-medium text-white">
              {phase === 'scanning'
                ? 'Analyzing image features...'
                : 'Generating segmentation map...'}
            </p>
            <p className="text-[10px] text-gray-400">
              {modelName} {phase === 'scanning' ? `${Math.round(scanProgress * 100)}%` : `${Math.round(segmentProgress * 100)}%`}
            </p>
          </div>
        </div>
        {/* Progress bar */}
        <div className="mt-2 h-1 bg-white/10 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-100"
            style={{
              width: `${phase === 'scanning' ? scanProgress * 50 : 50 + segmentProgress * 50}%`,
              background: 'linear-gradient(90deg, #88E7B2, #2FC2D9)',
            }}
          />
        </div>
      </div>
    </div>
  )
}
