'use client'

import { FloatingDock } from '@/components/ui/floating-dock'
import {
  IconChartBar,
  IconPresentationAnalytics,
  IconPhotoScan,
  IconBrain,
} from '@tabler/icons-react'

export function Navigation() {
  const links = [
    {
      title: 'Dashboard',
      icon: (
        <IconChartBar className="h-full w-full text-neutral-500 dark:text-neutral-300" />
      ),
      href: '#dashboard',
    },
    {
      title: 'How It Works',
      icon: (
        <IconPresentationAnalytics className="h-full w-full text-neutral-500 dark:text-neutral-300" />
      ),
      href: '#how-it-works',
    },
    {
      title: 'Demo',
      icon: (
        <IconPhotoScan className="h-full w-full text-neutral-500 dark:text-neutral-300" />
      ),
      href: '#demo',
    },
    {
      title: 'Architecture Explainer',
      icon: (
        <IconBrain className="h-full w-full text-neutral-500 dark:text-neutral-300" />
      ),
      href: '/explainer',
    },
  ]

  return (
    <div className="fixed left-4 top-1/2 -translate-y-1/2 z-50 flex flex-col items-center">
      <FloatingDock items={links} />
    </div>
  )
}
