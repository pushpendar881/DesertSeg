'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
} from '@/components/ui/chart'
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts'
import { SegFormer10Classes50EpochsCharts } from '@/components/segformer-training-charts-50-epochs'
import { SegFormer7Classes100EpochsCharts } from '@/components/segformer-training-charts-100-epochs'
import { NoiseBackground } from '@/components/ui/noise-background'

const classIoUData = [
  { name: 'Trees', 'segformer-10c-50e': 85, 'segformer-7c-100e': 75, deeplabv3: 82 },
  { name: 'Lush Bushes', 'segformer-10c-50e': 78, 'segformer-7c-100e': 72, deeplabv3: 75 },
  { name: 'Dry Grass', 'segformer-10c-50e': 72, 'segformer-7c-100e': 51, deeplabv3: 74 },
  { name: 'Dry Bushes', 'segformer-10c-50e': 68, 'segformer-7c-100e': 35, deeplabv3: 70 },
  { name: 'Ground Clutter', 'segformer-10c-50e': 65, 'segformer-7c-100e': 46, deeplabv3: 63 },
  { name: 'Flowers', 'segformer-10c-50e': 71, 'segformer-7c-100e': 72, deeplabv3: 69 },
  { name: 'Logs', 'segformer-10c-50e': 80, 'segformer-7c-100e': 0, deeplabv3: 79 },
  { name: 'Rocks', 'segformer-10c-50e': 88, 'segformer-7c-100e': 0, deeplabv3: 87 },
  { name: 'Landscape', 'segformer-10c-50e': 91, 'segformer-7c-100e': 0, deeplabv3: 90 },
  { name: 'Sky', 'segformer-10c-50e': 95, 'segformer-7c-100e': 0, deeplabv3: 94 },
]

const trainingLossData = [
  { epoch: 0, 'segformer-10c-50e': 2.1, 'segformer-7c-100e': 1.76, deeplabv3: 2.3 },
  { epoch: 5, 'segformer-10c-50e': 1.2, 'segformer-7c-100e': 0.63, deeplabv3: 1.4 },
  { epoch: 10, 'segformer-10c-50e': 0.8, 'segformer-7c-100e': 0.58, deeplabv3: 1.0 },
  { epoch: 15, 'segformer-10c-50e': 0.5, 'segformer-7c-100e': 0.55, deeplabv3: 0.7 },
  { epoch: 20, 'segformer-10c-50e': 0.3, 'segformer-7c-100e': 0.54, deeplabv3: 0.5 },
  { epoch: 25, 'segformer-10c-50e': 0.25, 'segformer-7c-100e': 0.53, deeplabv3: 0.4 },
  { epoch: 30, 'segformer-10c-50e': 0.22, 'segformer-7c-100e': 0.52, deeplabv3: 0.35 },
]

const validationmIoUData = [
  { epoch: 0, 'segformer-10c-50e': 45, 'segformer-7c-100e': 27.9, deeplabv3: 42 },
  { epoch: 5, 'segformer-10c-50e': 65, 'segformer-7c-100e': 49.4, deeplabv3: 62 },
  { epoch: 10, 'segformer-10c-50e': 73, 'segformer-7c-100e': 53.4, deeplabv3: 71 },
  { epoch: 15, 'segformer-10c-50e': 78, 'segformer-7c-100e': 55.5, deeplabv3: 76 },
  { epoch: 20, 'segformer-10c-50e': 81, 'segformer-7c-100e': 56.5, deeplabv3: 79 },
  { epoch: 25, 'segformer-10c-50e': 82, 'segformer-7c-100e': 57.2, deeplabv3: 80 },
  { epoch: 30, 'segformer-10c-50e': 82.5, 'segformer-7c-100e': 58.4, deeplabv3: 80.8 },
]

const chartConfig = {
  'segformer-10c-50e': {
    label: 'SegFormer-10C-50E',
    color: '#2FC2D9',
  },
  'segformer-7c-100e': {
    label: 'SegFormer-7C-100E',
    color: '#F59E0B',
  },
  deeplabv3: {
    label: 'DeepLabV3+',
    color: '#88E7B2',
  },
}

export function DashboardSection() {
  return (
    <section id="dashboard" className="py-24 bg-zinc-900">
      <div className="container mx-auto px-6 max-w-7xl">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Model Performance Dashboard
          </h2>
          <p className="text-lg text-gray-400 max-w-3xl mx-auto">
            Comparing SegFormer-10C-50E, SegFormer-7C-100E, and DeepLabV3+ on desert semantic segmentation
          </p>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          <NoiseBackground
            gradientColors={[
              "hsl(186, 68%, 72%)",
              "hsl(148, 64%, 73%)",
              "hsl(216, 73%, 53%)",
            ]}
            noiseIntensity={0.12}
            speed={0.06}
          >
            <Card className="border-0 bg-transparent shadow-none">
              <CardHeader>
                <CardDescription>Best mIoU</CardDescription>
                <CardTitle className="text-3xl">30.5%</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">SegFormer-B2-50-Epochs</p>
              </CardContent>
            </Card>
          </NoiseBackground>

          <NoiseBackground
            gradientColors={[
              "hsl(148, 64%, 73%)",
              "hsl(186, 68%, 72%)",
              "hsl(189, 75%, 51%)",
            ]}
            noiseIntensity={0.12}
            speed={0.06}
          >
            <Card className="border-0 bg-transparent shadow-none">
              <CardHeader>
                <CardDescription>Best Pixel Accuracy</CardDescription>
                <CardTitle className="text-3xl">91.2%</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">SegFormer-10-Classes 50-Epochs</p>
              </CardContent>
            </Card>
          </NoiseBackground>

          <NoiseBackground
            gradientColors={[
              "hsl(216, 73%, 53%)",
              "hsl(189, 75%, 51%)",
              "hsl(186, 68%, 72%)",
            ]}
            noiseIntensity={0.12}
            speed={0.06}
          >
            <Card className="border-0 bg-transparent shadow-none">
              <CardHeader>
                <CardDescription>Fastest Model</CardDescription>
                <CardTitle className="text-3xl">SegFormer-10-Classes-50-Epochs</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">Lower inference time</p>
              </CardContent>
            </Card>
          </NoiseBackground>

          <NoiseBackground
            gradientColors={[
              "hsl(189, 75%, 51%)",
              "hsl(216, 73%, 53%)",
              "hsl(148, 64%, 73%)",
            ]}
            noiseIntensity={0.12}
            speed={0.06}
          >
            <Card className="border-0 bg-transparent shadow-none">
              <CardHeader>
                <CardDescription>Most Parameters</CardDescription>
                <CardTitle className="text-3xl">27M</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">SegFormer-10-Classes-50-Epochs</p>
              </CardContent>
            </Card>
          </NoiseBackground>
        </div>

        {/* Model Comparison Table */}
        <Card className="mb-12">
          <CardHeader>
            <CardTitle>Model Comparison</CardTitle>
            <CardDescription>Side-by-side architectural and performance metrics</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-3 px-4 font-medium">Metric</th>
                    <th className="text-left py-3 px-4 font-medium text-primary">SegFormer-10C-50E</th>
                    <th className="text-left py-3 px-4 font-medium text-accent">DeepLabV3+</th>
                    <th className="text-left py-3 px-4 font-medium text-amber-500">SegFormer-C10</th>
                    <th className="text-left py-3 px-4 font-medium text-rose-500">SegFormer-C6</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-border/50 bg-muted/20">
                    <td className="py-3 px-4">Train mIoU <span className="text-xs text-muted-foreground">(Training)</span></td>
                    <td className="py-3 px-4 font-semibold text-primary">65.2%</td>
                    <td className="py-3 px-4 font-semibold text-accent">62.0%</td>
                    <td className="py-3 px-4 font-semibold text-amber-500">61.72%</td>
                    <td className="py-3 px-4 font-semibold text-rose-500">58.30%</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4">Test mIoU <span className="text-xs text-muted-foreground">(Eval)</span></td>
                    <td className="py-3 px-4 font-semibold text-primary">30.2%</td>
                    <td className="py-3 px-4 font-semibold text-accent">28.5%</td>
                    <td className="py-3 px-4 font-semibold text-amber-500">29.32%</td>
                    <td className="py-3 px-4 font-semibold text-rose-500">26.1%</td>
                  </tr>
                  <tr className="border-b border-border/50 bg-muted/20">
                    <td className="py-3 px-4">Train Accuracy <span className="text-xs text-muted-foreground">(Training)</span></td>
                    <td className="py-3 px-4 font-semibold text-primary">75.0%</td>
                    <td className="py-3 px-4 font-semibold text-accent">85.0%</td>
                    <td className="py-3 px-4 font-semibold text-amber-500">72.32%</td>
                    <td className="py-3 px-4 font-semibold text-rose-500">70.0%</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4">Test Accuracy <span className="text-xs text-muted-foreground">(Eval)</span></td>
                    <td className="py-3 px-4 font-semibold text-primary">51.75%</td>
                    <td className="py-3 px-4 font-semibold text-accent">58.2%</td>
                    <td className="py-3 px-4 font-semibold text-amber-500">49.8%</td>
                    <td className="py-3 px-4 font-semibold text-rose-500">48.3%</td>
                  </tr>
                  <tr className="border-b border-border/50 bg-muted/20">
                    <td className="py-3 px-4">Classes <span className="text-xs text-muted-foreground">(Training)</span></td>
                    <td className="py-3 px-4">10</td>
                    <td className="py-3 px-4">10</td>
                    <td className="py-3 px-4">10</td>
                    <td className="py-3 px-4">6</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4">Epochs Trained <span className="text-xs text-muted-foreground">(Training)</span></td>
                    <td className="py-3 px-4">50</td>
                    <td className="py-3 px-4">50</td>
                    <td className="py-3 px-4">50</td>
                    <td className="py-3 px-4">80</td>
                  </tr>
                  <tr className="border-b border-border/50 bg-muted/20">
                    <td className="py-3 px-4">Inference Time <span className="text-xs text-muted-foreground">(GPU)</span></td>
                    <td className="py-3 px-4">~2 ms</td>
                    <td className="py-3 px-4">~4 ms</td>
                    <td className="py-3 px-4">~2.48s</td>
                    <td className="py-3 px-4">~2.24s</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4">Architecture <span className="text-xs text-muted-foreground">(Architecture)</span></td>
                    <td className="py-3 px-4">Transformer</td>
                    <td className="py-3 px-4">CNN</td>
                    <td className="py-3 px-4">Transformer</td>
                    <td className="py-3 px-4">Transformer</td>
                  </tr>
                  <tr className="border-b border-border/50 bg-muted/20">
                    <td className="py-3 px-4">Backbone <span className="text-xs text-muted-foreground">(Architecture)</span></td>
                    <td className="py-3 px-4">Nvidia-B2</td>
                    <td className="py-3 px-4">ResNet-101</td>
                    <td className="py-3 px-4">Nvidia-B1</td>
                    <td className="py-3 px-4">Nvidia-B1</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {/* Per-Class IoU Chart — COMMENTED OUT
        <Card className="mb-12">
          <CardHeader>
            <CardTitle>Per-Class IoU Performance</CardTitle>
            <CardDescription>Intersection over Union (IoU) for each semantic class</CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer config={chartConfig} className="h-[400px] w-full">
              <BarChart data={classIoUData} layout="vertical" margin={{ left: 100, right: 20, top: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} />
                <YAxis dataKey="name" type="category" width={100} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <ChartLegend content={<ChartLegendContent />} />
                <Bar dataKey="segformer-10c-50e" fill="var(--color-segformer-10c-50e)" radius={4} />
                <Bar dataKey="segformer-7c-100e" fill="var(--color-segformer-7c-100e)" radius={4} />
                <Bar dataKey="deeplabv3" fill="var(--color-deeplabv3)" radius={4} />
              </BarChart>
            </ChartContainer>
            <p className="text-sm text-muted-foreground mt-4 text-center">
              Placeholder values shown — replace with actual metrics
            </p>
          </CardContent>
        </Card>
        */}

        {/* Training Curves */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-12">
          <Card>
            <CardHeader>
              <CardTitle>Training Loss</CardTitle>
              <CardDescription>Loss curves during training</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[300px] w-full">
                <LineChart data={trainingLossData} margin={{ left: 20, right: 20, top: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <ChartLegend content={<ChartLegendContent />} />
                  <Line type="monotone" dataKey="segformer-10c-50e" stroke="var(--color-segformer-10c-50e)" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="segformer-7c-100e" stroke="var(--color-segformer-7c-100e)" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="deeplabv3" stroke="var(--color-deeplabv3)" strokeWidth={2} dot={false} />
                </LineChart>
              </ChartContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Validation mIoU</CardTitle>
              <CardDescription>Mean IoU on validation set</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[300px] w-full">
                <LineChart data={validationmIoUData} margin={{ left: 20, right: 20, top: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: 'mIoU (%)', angle: -90, position: 'insideLeft' }} domain={[0, 100]} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <ChartLegend content={<ChartLegendContent />} />
                  <Line type="monotone" dataKey="segformer-10c-50e" stroke="var(--color-segformer-10c-50e)" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="segformer-7c-100e" stroke="var(--color-segformer-7c-100e)" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="deeplabv3" stroke="var(--color-deeplabv3)" strokeWidth={2} dot={false} />
                </LineChart>
              </ChartContainer>
            </CardContent>
          </Card>
        </div>

        {/* Architecture Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="border-primary/20">
            <CardHeader>
              <div className="flex items-start justify-between">
                <div>
                  <CardTitle className="text-primary">SegFormer-10-Classes-50-Epochs</CardTitle>
                  <CardDescription>Transformer-based segmentation</CardDescription>
                </div>
                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center">
                  <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1H5a1 1 0 01-1-1v-3zM14 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1h-4a1 1 0 01-1-1v-3z" />
                  </svg>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <p className="text-sm font-medium mb-1">Encoder</p>
                <p className="text-sm text-muted-foreground">Mix Transformer (MiT-B2), hierarchical, 4 stages</p>
              </div>
              <div>
                <p className="text-sm font-medium mb-1">Decoder</p>
                <p className="text-sm text-muted-foreground">Lightweight All-MLP decoder</p>
              </div>
              <div>
                <p className="text-sm font-medium mb-1">Key Strength</p>
                <p className="text-sm text-muted-foreground">No positional encoding → robust to resolution changes</p>
              </div>
              <div>
                <p className="text-sm font-medium mb-1">Parameters</p>
                <p className="text-sm text-muted-foreground">27M</p>
              </div>
            </CardContent>
          </Card>

          <Card className="border-accent/20">
            <CardHeader>
              <div className="flex items-start justify-between">
                <div>
                  <CardTitle className="text-accent">DeepLabV3+</CardTitle>
                  <CardDescription>CNN-based segmentation</CardDescription>
                </div>
                <div className="w-12 h-12 rounded-lg bg-accent/10 flex items-center justify-center">
                  <svg className="w-6 h-6 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                  </svg>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <p className="text-sm font-medium mb-1">Encoder</p>
                <p className="text-sm text-muted-foreground">ResNet-101 with atrous convolutions</p>
              </div>
              <div>
                <p className="text-sm font-medium mb-1">Decoder</p>
                <p className="text-sm text-muted-foreground">ASPP + encoder-decoder</p>
              </div>
              <div>
                <p className="text-sm font-medium mb-1">Key Strength</p>
                <p className="text-sm text-muted-foreground">Proven performance on dense prediction tasks</p>
              </div>
              <div>
                <p className="text-sm font-medium mb-1">Parameters</p>
                <p className="text-sm text-muted-foreground">~41M</p>
              </div>
            </CardContent>
          </Card>

          <Card className="border-amber-500/20">
            <CardHeader>
              <div className="flex items-start justify-between">
                <div>
                  <CardTitle className="text-amber-500">SegFormer-7C-100E</CardTitle>
                  <CardDescription>Transformer-based — 7 classes</CardDescription>
                </div>
                <div className="w-12 h-12 rounded-lg bg-amber-500/10 flex items-center justify-center">
                  <svg className="w-6 h-6 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1H5a1 1 0 01-1-1v-3zM14 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1h-4a1 1 0 01-1-1v-3z" />
                  </svg>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <p className="text-sm font-medium mb-1">Encoder</p>
                <p className="text-sm text-muted-foreground">Mix Transformer (MiT-B2), hierarchical, 4 stages</p>
              </div>
              <div>
                <p className="text-sm font-medium mb-1">Classes</p>
                <p className="text-sm text-muted-foreground">7 desert terrain classes (reduced from 10)</p>
              </div>
              <div>
                <p className="text-sm font-medium mb-1">Training</p>
                <p className="text-sm text-muted-foreground">80 epochs planned, 71 completed — 9,088 steps</p>
              </div>
              <div>
                <p className="text-sm font-medium mb-1">Best mIoU</p>
                <p className="text-sm text-muted-foreground">58.4% at step 8,704</p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* SegFormer-10-Classes-50-Epochs Detailed Training Metrics */}
        <div className="mt-16 pt-16 border-t border-border/30">
          <SegFormer10Classes50EpochsCharts />
        </div>

        {/* SegFormer-7-Classes-100-Epochs Detailed Training Metrics */}
        <div className="mt-16 pt-16 border-t border-border/30">
          <SegFormer7Classes100EpochsCharts />
        </div>
      </div>
    </section>
  )
}
