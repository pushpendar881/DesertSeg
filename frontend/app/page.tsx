import { HeroSection } from '@/components/hero-section'
import { DashboardSection } from '@/components/dashboard-section'
import { HowItWorksSection } from '@/components/how-it-works-section'
import { DemoSection } from '@/components/demo-section'
import { VideoDemoSection } from '@/components/video-demo-section'
import { Footer } from '@/components/footer'
import { Navigation } from '@/components/navigation'

export default function Home() {
  return (
    <main className="min-h-screen bg-background">
      <Navigation />
      <HeroSection />
      <DashboardSection />
      <HowItWorksSection />
      <DemoSection />
      <VideoDemoSection />
      <Footer />
    </main>
  )
}
