'use client'

import { useEffect, useState, useRef, useCallback } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceDot,
} from 'recharts'
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from '@/components/ui/chart'

interface DataPoint {
  [key: string]: number
}

interface AnimatedLineChartProps {
  data: DataPoint[]
  xKey: string
  yKey: string
  xLabel: string
  yLabel: string
  color?: string
  height?: number
  animationDuration?: number
  yDomain?: [number | string, number | string]
  xTickFormatter?: (value: number) => string
  yTickFormatter?: (value: number) => string
  chartConfig?: Record<string, { label: string; color: string }>
}

export function AnimatedLineChart({
  data,
  xKey,
  yKey,
  xLabel,
  yLabel,
  color = 'hsl(var(--chart-1))',
  height = 250,
  animationDuration = 2500,
  yDomain,
  xTickFormatter,
  yTickFormatter,
  chartConfig,
}: AnimatedLineChartProps) {
  const [visibleCount, setVisibleCount] = useState(0)
  const [hasAnimated, setHasAnimated] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const animationRef = useRef<number | null>(null)

  const startAnimation = useCallback(() => {
    if (hasAnimated) return
    setHasAnimated(true)
    setVisibleCount(0)

    const totalPoints = data.length
    const startTime = performance.now()

    const animate = (now: number) => {
      const elapsed = now - startTime
      // Use easeOutCubic for smooth deceleration
      const rawProgress = Math.min(elapsed / animationDuration, 1)
      const eased = 1 - Math.pow(1 - rawProgress, 3)
      const count = Math.min(Math.ceil(eased * totalPoints), totalPoints)

      setVisibleCount(count)

      if (rawProgress < 1) {
        animationRef.current = requestAnimationFrame(animate)
      }
    }

    animationRef.current = requestAnimationFrame(animate)
  }, [data.length, animationDuration, hasAnimated])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !hasAnimated) {
            startAnimation()
          }
        })
      },
      { threshold: 0.2 }
    )

    observer.observe(el)
    return () => {
      observer.disconnect()
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [startAnimation, hasAnimated])

  const visibleData = data.slice(0, visibleCount)

  const defaultConfig = {
    [yKey]: {
      label: yLabel,
      color: color,
    },
  }

  // Show a trailing dot at the last visible point
  const lastPoint = visibleData.length > 0 ? visibleData[visibleData.length - 1] : null

  return (
    <div ref={containerRef} style={{ height }}>
      <ChartContainer config={chartConfig || defaultConfig} className="h-full w-full">
        <LineChart
          data={visibleData}
          margin={{ left: 5, right: 15, top: 10, bottom: 25 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.5} />
          <XAxis
            dataKey={xKey}
            tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
            tickFormatter={xTickFormatter}
            label={{
              value: xLabel,
              position: 'insideBottom',
              offset: -15,
              style: { fontSize: 10, fill: 'hsl(var(--muted-foreground))' },
            }}
          />
          <YAxis
            domain={yDomain || ['auto', 'auto']}
            tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
            tickFormatter={yTickFormatter}
            width={55}
          />
          <ChartTooltip
            content={<ChartTooltipContent />}
          />
          <Line
            type="monotone"
            dataKey={yKey}
            stroke={color}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          {lastPoint && visibleCount < data.length && (
            <ReferenceDot
              x={lastPoint[xKey]}
              y={lastPoint[yKey]}
              r={4}
              fill={color}
              stroke="hsl(var(--background))"
              strokeWidth={2}
            />
          )}
        </LineChart>
      </ChartContainer>
    </div>
  )
}
