'use client'

export function Footer() {
  const scrollToSection = (id: string) => {
    const element = document.getElementById(id)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' })
    }
  }

  return (
    <footer className="border-t border-zinc-800 bg-zinc-950 py-12">
      <div className="container mx-auto px-6 max-w-7xl">
        <div className="flex flex-col md:flex-row justify-between items-center gap-6">
          <div>
            <h3 className="text-lg font-semibold text-white mb-2">DesertSeg</h3>
            <p className="text-sm text-gray-400">
              Built for Duality AI × OSEN Hackathon at Technex
            </p>
          </div>

          <div className="flex gap-6">
            <button
              onClick={() => scrollToSection('dashboard')}
              className="text-sm text-gray-400 hover:text-accent transition-colors"
            >
              Dashboard
            </button>
            <button
              onClick={() => scrollToSection('how-it-works')}
              className="text-sm text-gray-400 hover:text-accent transition-colors"
            >
              How It Works
            </button>
            <button
              onClick={() => scrollToSection('demo')}
              className="text-sm text-gray-400 hover:text-accent transition-colors"
            >
              Demo
            </button>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t border-zinc-800 text-center">
          <p className="text-sm text-gray-400">
            © {new Date().getFullYear()} DesertSeg Research Team. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  )
}
