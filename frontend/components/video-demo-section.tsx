'use client'

import { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Spinner } from '@/components/ui/spinner'

export function VideoDemoSection() {
  const [videoSrc, setVideoSrc] = useState<string | null>(
    'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/image_segmentation_demo_video_final-VK4IUYW3pjeneHqktr9ky2iUbBsZ2w.mp4'
  )
  const [uploading, setUploading] = useState(false)
  const [model, setModel] = useState('segformer')
  const [fpsLimit, setFpsLimit] = useState(5)
  const [overlayAlpha, setOverlayAlpha] = useState(0.6)
  const [metadata, setMetadata] = useState<any | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  // Cleanup blob URLs on unmount
  useEffect(() => {
    return () => {
      if (videoSrc && videoSrc.startsWith('blob:')) {
        URL.revokeObjectURL(videoSrc)
      }
    }
  }, [videoSrc])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const file = fileInputRef.current?.files?.[0]
    if (!file) return

    const formData = new FormData()
    formData.append('video', file)
    formData.append('model', model)
    formData.append('fps_limit', String(fpsLimit))
    formData.append('overlay_alpha', String(overlayAlpha))

    try {
      setUploading(true)
      setMetadata(null)

      const res = await fetch('/api/predict-video', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const errorText = await res.text()
        console.error('Video backend error', errorText)
        return
      }

      const data = await res.json()
      if (data?.video) {
        // Clean up previous blob URL
        if (videoSrc && videoSrc.startsWith('blob:')) {
          URL.revokeObjectURL(videoSrc)
        }

        // Convert base64 data URI → Blob → Object URL
        // (server returns data:video/mp4;base64,...)
        const base64 = data.video.split(',')[1]
        const byteChars = atob(base64)
        const byteNums = new Uint8Array(byteChars.length)
        for (let i = 0; i < byteChars.length; i++) {
          byteNums[i] = byteChars.charCodeAt(i)
        }
        const blob = new Blob([byteNums], { type: 'video/mp4' })
        const blobUrl = URL.createObjectURL(blob)

        setVideoSrc(blobUrl)
        setMetadata(data.metadata ?? null)
      }
    } catch (err) {
      console.error('Video upload failed', err)
    } finally {
      setUploading(false)
    }
  }

  return (
    <section className="py-24 bg-zinc-900">
      <div className="container mx-auto px-6 max-w-6xl">
        <div className="text-center mb-12">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Live Video Segmentation
          </h2>
          <p className="text-lg text-gray-400 max-w-3xl mx-auto">
            Upload a short desert video and see SegFormer segment every frame with colored overlays.
          </p>
        </div>

        <div className="grid lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)] gap-8 items-start">
          <div className="space-y-4">
            <div className="relative rounded-xl overflow-hidden border-2 border-zinc-800 shadow-2xl bg-black">
              {videoSrc ? (
                <video
                  key={videoSrc}
                  src={videoSrc}
                  controls
                  className="w-full h-auto"
                  preload="metadata"
                >
                  Your browser does not support the video tag.
                </video>
              ) : (
                <div className="aspect-video flex items-center justify-center text-gray-500 text-sm">
                  Segmented video preview will appear here
                </div>
              )}
            </div>

            {metadata && (
              <div className="grid sm:grid-cols-4 gap-4 text-center">
                <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
                  <div className="text-2xl font-bold text-[#2FC2D9] mb-1">
                    {metadata.processingTime ?? 0}s
                  </div>
                  <div className="text-xs text-gray-400">Processing Time</div>
                </div>
                <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
                  <div className="text-2xl font-bold text-[#88E7B2] mb-1">
                    {metadata.framesProcessed ?? 0}
                  </div>
                  <div className="text-xs text-gray-400">Frames Segmented</div>
                </div>
                <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
                  <div className="text-2xl font-bold text-[#2764E7] mb-1">
                    {Math.round(metadata.originalFps ?? 0)} fps
                  </div>
                  <div className="text-xs text-gray-400">Original FPS</div>
                </div>
                <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
                  <div className="text-2xl font-bold text-[#F5A623] mb-1">
                    {metadata.resolution
                      ? `${metadata.resolution.width}×${metadata.resolution.height}`
                      : '—'}
                  </div>
                  <div className="text-xs text-gray-400">Resolution</div>
                </div>
              </div>
            )}
          </div>

          <form
            onSubmit={handleSubmit}
            className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 space-y-6"
          >
            <div className="space-y-2">
              <Label htmlFor="video" className="text-gray-200">
                Upload video
              </Label>
              <Input
                id="video"
                type="file"
                accept="video/*"
                ref={fileInputRef}
                className="cursor-pointer bg-zinc-950 border-zinc-800 text-gray-100"
              />
              <p className="text-xs text-gray-500">
                MP4/AVI/MOV, ideally &lt; 30s to keep processing fast.
              </p>
            </div>

            <div className="space-y-2">
              <Label className="text-gray-200">Model</Label>
              <Select value={model} onValueChange={setModel}>
                <SelectTrigger className="bg-zinc-950 border-zinc-800 text-gray-100">
                  <SelectValue placeholder="Choose model" />
                </SelectTrigger>
                <SelectContent className="bg-zinc-900 border-zinc-800">
                  <SelectItem value="segformer">SegFormer Desert (50 epochs)</SelectItem>
                  <SelectItem value="segformer-b2">SegFormer-B2</SelectItem>
                  <SelectItem value="segformer-b1">SegFormer-B1</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs text-gray-400">
                  <span>Max FPS processed</span>
                  <span className="text-gray-200 font-medium">{fpsLimit} fps</span>
                </div>
                <Slider
                  min={1}
                  max={10}
                  step={1}
                  value={[fpsLimit]}
                  onValueChange={([v]) => setFpsLimit(v)}
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs text-gray-400">
                  <span>Overlay strength</span>
                  <span className="text-gray-200 font-medium">
                    {Math.round(overlayAlpha * 100)}%
                  </span>
                </div>
                <Slider
                  min={0.2}
                  max={0.9}
                  step={0.05}
                  value={[overlayAlpha]}
                  onValueChange={([v]) => setOverlayAlpha(v)}
                />
              </div>
            </div>

            <Button
              type="submit"
              className="w-full bg-[#2FC2D9] hover:bg-[#29a8be] text-black font-semibold"
              disabled={uploading}
            >
              {uploading ? (
                <span className="inline-flex items-center gap-2">
                  <Spinner className="h-4 w-4" />
                  Processing video…
                </span>
              ) : (
                'Run video segmentation'
              )}
            </Button>

            <p className="text-xs text-gray-500">
              We process a subset of frames to keep latency low. For best results, use short clips with
              clear desert terrain.
            </p>
          </form>
        </div>
      </div>
    </section>
  )
}

