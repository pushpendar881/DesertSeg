export function VideoDemoSection() {
  return (
    <section className="py-24 bg-zinc-900">
      <div className="container mx-auto px-6 max-w-6xl">
        <div className="text-center mb-12">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Live Segmentation Demo
          </h2>
          <p className="text-lg text-gray-400 max-w-3xl mx-auto">
            Watch our model perform real-time semantic segmentation on desert terrain
          </p>
        </div>

        <div className="relative rounded-xl overflow-hidden border-2 border-zinc-800 shadow-2xl">
          <video
            src="https://hebbkx1anhila5yf.public.blob.vercel-storage.com/image_segmentation_demo_video_final-VK4IUYW3pjeneHqktr9ky2iUbBsZ2w.mp4"
            controls
            className="w-full h-auto"
            preload="metadata"
          >
            Your browser does not support the video tag.
          </video>
        </div>

        <div className="mt-8 grid sm:grid-cols-3 gap-4 text-center">
          <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
            <div className="text-2xl font-bold text-[#2FC2D9] mb-1">Real-Time</div>
            <div className="text-sm text-gray-400">Processing Speed</div>
          </div>
          <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
            <div className="text-2xl font-bold text-[#88E7B2] mb-1">10 Classes</div>
            <div className="text-sm text-gray-400">Terrain Types</div>
          </div>
          <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
            <div className="text-2xl font-bold text-[#2764E7] mb-1">82.5%</div>
            <div className="text-sm text-gray-400">mIoU Accuracy</div>
          </div>
        </div>
      </div>
    </section>
  )
}
