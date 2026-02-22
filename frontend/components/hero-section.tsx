'use client'

import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

export function HeroSection() {
  const scrollToSection = (id: string) => {
    const element = document.getElementById(id)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' })
    }
  }

  return (
    <section id="hero" className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 -z-10">
        <svg className="w-full h-full opacity-20" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="currentColor" strokeWidth="0.5" className="text-muted-foreground"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>
      </div>

      <div className="container mx-auto px-6 py-24 text-center max-w-5xl">
        <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold text-white mb-6 leading-tight">
          DesertSeg
        </h1>
        <p className="text-2xl md:text-3xl text-gray-300 mb-8 font-medium">
          Semantic Scene Segmentation for Autonomous UGV Navigation
        </p>
        <p className="text-lg md:text-xl text-gray-400 mb-12 max-w-3xl mx-auto">
          Trained on digital twin–generated desert data. Evaluated on unseen terrain.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
          <Button
            size="lg"
            onClick={() => scrollToSection('dashboard')}
            className="text-base px-8 gradient-primary hover:gradient-primary-hover text-white font-semibold border-0 shadow-lg hover:shadow-xl transition-all"
          >
            Explore Dashboard
          </Button>
          <Button
            size="lg"
            variant="outline"
            onClick={() => scrollToSection('how-it-works')}
            className="text-base px-8 border-2 border-accent hover:bg-accent hover:text-white transition-all"
          >
            How It Works
          </Button>
        </div>

        <div className="flex flex-wrap gap-3 justify-center">
          <Badge variant="secondary" className="text-sm px-4 py-2">
            10 classes
          </Badge>
          <Badge variant="secondary" className="text-sm px-4 py-2">
            2 models compared
          </Badge>
          <Badge variant="secondary" className="text-sm px-4 py-2">
            mIoU evaluated
          </Badge>
        </div>
      </div>
    </section>
  )
}
