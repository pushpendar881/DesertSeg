'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Upload, Loader2 } from 'lucide-react'
import { ImageProcessingAnimation } from '@/components/image-processing-animation'
import dynamic from 'next/dynamic'
import type { PredictionResponse, ParallelPredictionResponse } from '@/types/prediction'

// Dynamically import SegFormer explainer to avoid SSR issues
const SegFormerExplainer = dynamic(() => import('@/components/segformer-explainer'), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center p-8">
      <div className="text-center">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
        <p className="text-sm text-muted-foreground">Loading Architecture Explainer...</p>
      </div>
    </div>
  )
});

const segmentationClasses = [
  { name: 'Trees', color: '#2d6a4f' },
  { name: 'Lush Bushes', color: '#52b788' },
  { name: 'Dry Grass', color: '#d4a017' },
  { name: 'Dry Bushes', color: '#a07850' },
  { name: 'Ground Clutter', color: '#6b6b6b' },
  { name: 'Flowers', color: '#e76f51' },
  { name: 'Logs', color: '#8b5e3c' },
  { name: 'Rocks', color: '#9e9e9e' },
  { name: 'Landscape', color: '#c8a96e' },
  { name: 'Sky', color: '#90c8e0' },
]

const MIN_ANIMATION_DURATION = 4500 // 4.5 seconds minimum animation

const MODEL_LABELS: Record<string, string> = {
  'segformer-10c-50e': 'SegFormer-10-Classes-50-Epochs',
  'segformer-b1': 'SegFormer-B1',
  'segformer-b10': 'SegFormer-B10',
  'deeplabv3': 'DeepLabV3+',
}

function getModelLabel(model: string): string {
  return MODEL_LABELS[model] ?? model
}

const STORAGE_KEY = 'selectedModel'
const MODE_KEY = 'predictionMode'

export function DemoSection() {
  const [selectedModel, setSelectedModel] = useState('segformer-10c-50e')
  const [isParallelMode, setIsParallelMode] = useState(false)
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isAnimating, setIsAnimating] = useState(false)
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [parallelPrediction, setParallelPrediction] = useState<ParallelPredictionResponse | null>(null)
  const [showResult, setShowResult] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'demo' | 'explainer'>('demo')
  const pendingPrediction = useRef<PredictionResponse | null>(null)

  // Hydrate from sessionStorage after mount to avoid SSR hydration mismatch
  useEffect(() => {
    const storedModel = window.sessionStorage.getItem(STORAGE_KEY)
    if (storedModel) setSelectedModel(storedModel)
    const storedMode = window.sessionStorage.getItem(MODE_KEY)
    if (storedMode === 'parallel') setIsParallelMode(true)
  }, [])

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string)
      }
      reader.readAsDataURL(file)
      setUploadedFile(file)
      setError(null)
      setPrediction(null)
      setShowResult(false)

      // Automatically run prediction
      await runPrediction(file)
    }
  }

  const runPrediction = async (file: File) => {
    setIsLoading(true)
    setIsAnimating(true)
    setShowResult(false)
    setError(null)
    pendingPrediction.current = null
    setParallelPrediction(null)

    const animationStart = Date.now()

    try {
      if (isParallelMode) {
        // Parallel mode: run all models
        const formData = new FormData()
        formData.append('image', file)

        const response = await fetch('/api/predict-parallel', {
          method: 'POST',
          body: formData,
        })

        if (!response.ok) {
          throw new Error(`Parallel prediction failed: ${response.statusText}`)
        }

        const data: ParallelPredictionResponse = await response.json()
        console.log('[FRONTEND] Parallel prediction received:', {
          success: data.success,
          modelsRun: data.metadata.modelsRun,
          results: Object.keys(data.results)
        })

        // Ensure minimum animation duration
        const elapsed = Date.now() - animationStart
        const remaining = Math.max(0, MIN_ANIMATION_DURATION - elapsed)

        await new Promise((resolve) => setTimeout(resolve, remaining))

        setParallelPrediction(data)
        setShowResult(true)
      } else {
        // Single model mode
        const formData = new FormData()
        formData.append('image', file)
        formData.append('model', selectedModel)

        const response = await fetch('/api/predict', {
          method: 'POST',
          body: formData,
        })

        if (!response.ok) {
          throw new Error(`Prediction failed: ${response.statusText}`)
        }

        const data: PredictionResponse = await response.json()
        console.log('[FRONTEND] Prediction received:', {
          success: data.success,
          hasSegmentationMask: !!data.segmentationMask,
          maskLength: data.segmentationMask?.length,
          maskPreview: data.segmentationMask?.substring(0, 100),
          metadata: data.metadata
        })

        // Verify segmentation mask is valid
        if (!data.segmentationMask || !data.segmentationMask.startsWith('data:image')) {
          console.error('[FRONTEND] Invalid segmentation mask format:', data.segmentationMask?.substring(0, 50))
          throw new Error('Invalid segmentation mask received from server')
        }

        pendingPrediction.current = data

        // Ensure minimum animation duration
        const elapsed = Date.now() - animationStart
        const remaining = Math.max(0, MIN_ANIMATION_DURATION - elapsed)

        await new Promise((resolve) => setTimeout(resolve, remaining))

        setPrediction(data)
        setShowResult(true)
        console.log('[FRONTEND] Result displayed, showResult:', true)
      }
    } catch (err) {
      // Still wait for minimum animation on error
      const elapsed = Date.now() - animationStart
      const remaining = Math.max(0, MIN_ANIMATION_DURATION - elapsed)
      await new Promise((resolve) => setTimeout(resolve, remaining))

      setError(err instanceof Error ? err.message : 'Failed to get prediction')
    } finally {
      setIsLoading(false)
      setIsAnimating(false)
    }
  }

  const handleModelChange = (model: string) => {
    if (!isParallelMode) {
      window.sessionStorage.setItem(STORAGE_KEY, model)
      setSelectedModel(model)
      // Refresh page when switching model (same mechanism as previous two models)
      window.location.reload()
    }
  }

  const handleModeToggle = (checked: boolean) => {
    window.sessionStorage.setItem(MODE_KEY, checked ? 'parallel' : 'single')
    window.location.reload()
  }

  return (
    <section id="demo" className="py-24 bg-zinc-950">
      <div className="container mx-auto px-6 max-w-7xl">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Try It — Upload a Desert Image
          </h2>
          <p className="text-lg text-gray-400 max-w-3xl mx-auto">
            Select a model and upload an image to see segmentation output
          </p>
        </div>

        <Card className="max-w-5xl mx-auto">
          <CardHeader>
            <CardTitle>Interactive Demo & Architecture Explainer</CardTitle>
            <CardDescription>
              Upload a desert image to see segmentation results and explore the SegFormer architecture
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-8">
            {/* Tab Selector */}
            <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as any)}>
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="demo">Live Demo</TabsTrigger>
                <TabsTrigger value="explainer">Architecture Explainer</TabsTrigger>
              </TabsList>
              
              <TabsContent value="demo" className="space-y-8 mt-8">
                {/* Mode Toggle */}
                <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg border border-border">
                  <div className="flex items-center space-x-3">
                    <Switch
                      id="parallel-mode"
                      checked={isParallelMode}
                      onCheckedChange={handleModeToggle}
                      disabled={isLoading}
                    />
                    <Label htmlFor="parallel-mode" className="text-sm font-medium cursor-pointer">
                      Parallel Task Mode
                    </Label>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {isParallelMode
                      ? 'All models will run simultaneously for comparison'
                      : 'Single model mode - select one model to run'}
                  </p>
                </div>

            {/* Model Selector - only show in single mode */}
            {!isParallelMode && (
              <div>
                <label className="text-sm font-medium mb-3 block">Select Model</label>
                <Tabs value={selectedModel} onValueChange={handleModelChange}>
                  <TabsList className="grid w-full grid-cols-4">
                    <TabsTrigger value="segformer-10c-50e" disabled={isLoading}>
                      SegFormer-10C-50E
                    </TabsTrigger>
                    <TabsTrigger value="segformer-b1" disabled={isLoading}>
                      SegFormer-B1
                    </TabsTrigger>
                    <TabsTrigger value="segformer-b10" disabled={isLoading}>
                      SegFormer-B10
                    </TabsTrigger>
                    <TabsTrigger value="deeplabv3" disabled={isLoading}>
                      DeepLabV3+
                    </TabsTrigger>
                  </TabsList>
                </Tabs>
              </div>
            )}

            {/* Upload Zone */}
            <div>
              <label className="text-sm font-medium mb-3 block">Upload Image</label>
              <label className={`border-2 border-dashed border-border rounded-lg p-12 flex flex-col items-center justify-center ${isLoading ? 'cursor-not-allowed opacity-60' : 'cursor-pointer hover:border-primary/50 hover:bg-muted/50'} transition-colors`}>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileUpload}
                  className="hidden"
                  disabled={isLoading}
                />
                {isLoading ? (
                  <>
                    <Loader2 className="w-12 h-12 text-primary mb-4 animate-spin" />
                    <p className="text-lg font-medium text-foreground mb-2">
                      Processing image...
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {isParallelMode
                        ? 'Running all models in parallel...'
                        : `Running ${getModelLabel(selectedModel)} model`}
                    </p>
                  </>
                ) : (
                  <>
                    <Upload className="w-12 h-12 text-muted-foreground mb-4" />
                    <p className="text-lg font-medium text-foreground mb-2">
                      Drop an image or click to upload
                    </p>
                    <p className="text-sm text-muted-foreground">
                      PNG, JPG, or JPEG (max 10MB)
                    </p>
                  </>
                )}
              </label>
              {error && (
                <div className="mt-3 p-3 bg-destructive/10 border border-destructive rounded-lg">
                  <p className="text-sm text-destructive">{error}</p>
                </div>
              )}
            </div>

            {/* Results Display */}
            {isParallelMode ? (
              // Parallel Mode: Show all model results
              <div className="space-y-6">
                <div>
                  <h3 className="text-sm font-medium mb-3">Input Image</h3>
                  <div className="aspect-video bg-muted rounded-lg border border-border flex items-center justify-center overflow-hidden">
                    {uploadedImage ? (
                      <img
                        src={uploadedImage}
                        alt="Uploaded"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <p className="text-muted-foreground">No image uploaded</p>
                    )}
                  </div>
                </div>

                {/* All Model Results Grid */}
                {showResult && parallelPrediction ? (
                  <>
                    <div className="grid md:grid-cols-2 gap-6">
                      {Object.entries(parallelPrediction.results).map(([modelKey, result]) => (
                        result.success && result.segmentationMask ? (
                          <div key={modelKey} className="space-y-2">
                            <h3 className="text-sm font-medium">{getModelLabel(modelKey)}</h3>
                            <div className="aspect-video bg-muted rounded-lg border border-border overflow-hidden">
                              <img
                                src={result.segmentationMask}
                                alt={`Segmentation Mask - ${getModelLabel(modelKey)}`}
                                className="w-full h-full object-cover"
                              />
                            </div>
                            {result.metadata && (
                              <p className="text-xs text-muted-foreground text-center">
                                {result.metadata.processingTime.toFixed(2)}s
                              </p>
                            )}
                          </div>
                        ) : null
                      ))}
                    </div>

                    {/* Comparative Analysis */}
                    {parallelPrediction.comparativeAnalysis && (
                      <div className="mt-8 space-y-4">
                        <h3 className="text-lg font-semibold">Comparative Analysis</h3>

                        {/* Summary */}
                        <Card>
                          <CardHeader>
                            <CardTitle className="text-base">Summary</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <p className="text-sm text-muted-foreground">
                              {parallelPrediction.comparativeAnalysis.summary}
                            </p>
                          </CardContent>
                        </Card>

                        {/* Class Comparison */}
                        <Card>
                          <CardHeader>
                            <CardTitle className="text-base">Class Confidence Comparison</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="space-y-3">
                              {parallelPrediction.comparativeAnalysis.classComparison.slice(0, 5).map((comp) => (
                                <div key={comp.class} className="space-y-1">
                                  <div className="flex justify-between text-sm">
                                    <span className="font-medium">{comp.class}</span>
                                    <span className="text-muted-foreground">
                                      Avg: {comp.averageConfidence.toFixed(1)}%
                                      {comp.maxModel && ` (Max: ${comp.maxModel})`}
                                    </span>
                                  </div>
                                  <div className="w-full bg-muted rounded-full h-2">
                                    <div
                                      className="bg-primary h-2 rounded-full transition-all"
                                      style={{ width: `${Math.min(comp.averageConfidence, 100)}%` }}
                                    />
                                  </div>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>

                        {/* Processing Time Comparison */}
                        <Card>
                          <CardHeader>
                            <CardTitle className="text-base">Processing Time Comparison</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="space-y-2">
                              {Object.entries(parallelPrediction.comparativeAnalysis.processingTimeComparison.times).map(([model, time]) => (
                                <div key={model} className="flex justify-between text-sm">
                                  <span>{getModelLabel(model)}</span>
                                  <span className="text-muted-foreground">{time.toFixed(2)}s</span>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      </div>
                    )}

                    {/* Comparative Model Reasoning & Explainability */}
                    {(() => {
                      const modelsWithReasoning = Object.entries(parallelPrediction.results).filter(
                        ([, result]) => result.success && result.reasoning
                      )
                      if (modelsWithReasoning.length === 0) return null
                      return (
                        <div className="mt-8 border-t border-border pt-8">
                          <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                            <svg
                              className="w-5 h-5 text-primary"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                              />
                            </svg>
                            Model Reasoning &amp; Explainability — Comparative View
                          </h3>

                          {/* Attention Heatmaps Comparison */}
                          <div className="mb-6">
                            <h4 className="font-medium mb-3 text-sm">Visual Attention Heatmaps</h4>
                            <div className={`grid gap-4 ${modelsWithReasoning.length >= 2 ? 'md:grid-cols-2' : 'grid-cols-1'}`}>
                              {modelsWithReasoning.map(([modelKey, result]) => (
                                <div key={modelKey} className="space-y-2">
                                  <p className="text-xs font-medium text-muted-foreground">{getModelLabel(modelKey)}</p>
                                  <div className="aspect-video rounded-lg border border-border overflow-hidden">
                                    <img
                                      src={result.reasoning!.attentionMap}
                                      alt={`Attention heatmap - ${getModelLabel(modelKey)}`}
                                      className="w-full h-full object-cover"
                                    />
                                  </div>
                                </div>
                              ))}
                            </div>
                            <div className="mt-3 flex gap-6 text-sm text-muted-foreground">
                              <p><span className="font-medium text-foreground">High attention areas</span> (red/yellow) show where the model focuses most.</p>
                              <p><span className="font-medium text-foreground">Low attention areas</span> (blue/purple) indicate less influence.</p>
                            </div>
                          </div>

                          {/* Feature Importance Comparison */}
                          <div className="mb-6">
                            <h4 className="font-medium mb-3 text-sm">Top Contributing Features</h4>
                            <div className={`grid gap-6 ${modelsWithReasoning.length >= 2 ? 'md:grid-cols-2' : 'grid-cols-1'}`}>
                              {modelsWithReasoning.map(([modelKey, result]) => (
                                <div key={modelKey} className="bg-muted/50 rounded-lg p-4 space-y-3">
                                  <p className="text-xs font-semibold text-muted-foreground mb-2">{getModelLabel(modelKey)}</p>
                                  {result.reasoning!.featureImportance.map((item, idx) => (
                                    <div key={idx} className="flex items-center gap-3">
                                      <span className="text-sm text-muted-foreground w-32 flex-shrink-0 truncate">
                                        {item.feature}
                                      </span>
                                      <div className="flex-1 h-5 bg-muted rounded-full overflow-hidden">
                                        <div
                                          className={`h-full ${idx < 3 ? 'bg-primary' : idx < 4 ? 'bg-accent' : 'bg-chart-3'} flex items-center justify-end px-2 transition-all duration-500`}
                                          style={{ width: `${item.confidence}%` }}
                                        >
                                          <span className="text-xs font-medium text-white">
                                            {item.confidence}%
                                          </span>
                                        </div>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              ))}
                            </div>
                          </div>

                          {/* Class Prediction Confidence Comparison */}
                          <div className="mb-6">
                            <h4 className="font-medium mb-3 text-sm">Class Prediction Confidence</h4>
                            <div className={`grid gap-6 ${modelsWithReasoning.length >= 2 ? 'md:grid-cols-2' : 'grid-cols-1'}`}>
                              {modelsWithReasoning.map(([modelKey, result]) => (
                                <div key={modelKey} className="space-y-2">
                                  <p className="text-xs font-semibold text-muted-foreground">{getModelLabel(modelKey)}</p>
                                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                                    {result.reasoning!.classConfidence.slice(0, 5).map((cls) => {
                                      const classColor = segmentationClasses.find(c => c.name === cls.class)?.color || '#000'
                                      return (
                                        <div
                                          key={cls.class}
                                          className="bg-background rounded-lg border border-border p-2 space-y-1"
                                        >
                                          <div className="flex items-center gap-1.5">
                                            <div
                                              className="w-2.5 h-2.5 rounded"
                                              style={{ backgroundColor: classColor }}
                                            />
                                            <span className="text-xs font-medium truncate">{cls.class}</span>
                                          </div>
                                          <div className="text-lg font-bold text-foreground">
                                            {cls.confidence.toFixed(1)}%
                                          </div>
                                          <div className="text-xs text-muted-foreground">confidence</div>
                                        </div>
                                      )
                                    })}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>

                          {/* AI Explanations Comparison */}
                          <div>
                            <h4 className="font-medium mb-3 text-sm">AI Explanations</h4>
                            <div className={`grid gap-4 ${modelsWithReasoning.length >= 2 ? 'md:grid-cols-2' : 'grid-cols-1'}`}>
                              {modelsWithReasoning.map(([modelKey, result]) => (
                                <div key={modelKey} className="bg-background rounded-lg border border-border p-4">
                                  <h5 className="font-medium mb-2 text-sm flex items-center gap-2">
                                    <svg
                                      className="w-4 h-4 text-primary"
                                      fill="none"
                                      stroke="currentColor"
                                      viewBox="0 0 24 24"
                                    >
                                      <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                                      />
                                    </svg>
                                    {getModelLabel(modelKey)}
                                  </h5>
                                  <p className="text-sm text-muted-foreground leading-relaxed">
                                    {result.reasoning!.explanation}
                                  </p>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      )
                    })()}
                  </>
                ) : (isAnimating || isLoading) && uploadedImage ? (
                  <div className="aspect-video bg-muted rounded-lg border border-border flex items-center justify-center">
                    <ImageProcessingAnimation
                      imageSrc={uploadedImage}
                      isProcessing={isAnimating}
                      resultImage={null}
                      modelName="All Models"
                    />
                  </div>
                ) : (
                  <p className="text-muted-foreground text-center py-8">Upload image to see parallel results</p>
                )}
              </div>
            ) : (
              // Single Mode: Original single model display
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-sm font-medium mb-3">Input Image</h3>
                  <div className="aspect-video bg-muted rounded-lg border border-border flex items-center justify-center overflow-hidden">
                    {uploadedImage ? (
                      <img
                        src={uploadedImage}
                        alt="Uploaded"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <p className="text-muted-foreground">No image uploaded</p>
                    )}
                  </div>
                </div>
                <div>
                  <h3 className="text-sm font-medium mb-3">Predicted Segmentation Mask</h3>
                  <div className="aspect-video bg-muted rounded-lg border border-border flex items-center justify-center overflow-hidden relative">
                    {(isAnimating || isLoading) && uploadedImage ? (
                      <ImageProcessingAnimation
                        imageSrc={uploadedImage}
                        isProcessing={isAnimating}
                        resultImage={null}
                        modelName={getModelLabel(selectedModel)}
                      />
                    ) : showResult && prediction?.segmentationMask ? (
                      <>
                        <img
                          src={prediction.segmentationMask}
                          alt="Segmentation Mask"
                          className="w-full h-full object-cover animate-in fade-in duration-700"
                          onLoad={() => console.log('[FRONTEND] Segmentation mask image loaded successfully')}
                          onError={(e) => {
                            console.error('[FRONTEND] Error loading segmentation mask:', e)
                            console.error('[FRONTEND] Mask data preview:', prediction.segmentationMask?.substring(0, 200))
                          }}
                        />
                        {/* Debug info */}
                        {process.env.NODE_ENV === 'development' && (
                          <div className="absolute top-2 left-2 bg-black/70 text-white text-xs p-1 rounded">
                            Mask loaded: {prediction.segmentationMask?.length} chars
                          </div>
                        )}
                      </>
                    ) : uploadedImage && !isLoading ? (
                      <div className="w-full h-full grid grid-cols-4 grid-rows-3">
                        {[...Array(12)].map((_, i) => (
                          <div
                            key={i}
                            style={{
                              backgroundColor:
                                segmentationClasses[i % segmentationClasses.length]?.color,
                            }}
                            className="w-full h-full"
                          />
                        ))}
                      </div>
                    ) : (
                      <p className="text-muted-foreground">Upload image to see results</p>
                    )}
                  </div>
                  {showResult && prediction && (
                    <p className="text-sm text-muted-foreground mt-3 text-center">
                      Processed in {prediction.metadata.processingTime.toFixed(2)}s using {prediction.metadata.modelVersion}
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* Class Legend */}
            <div>
              <h3 className="text-sm font-medium mb-4">Class Legend</h3>
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
                {segmentationClasses.map((cls) => (
                  <div key={cls.name} className="flex items-center gap-3">
                    <div
                      className="w-6 h-6 rounded border border-border/50"
                      style={{ backgroundColor: cls.color }}
                    />
                    <span className="text-sm text-foreground">{cls.name}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Model Explainability Section */}
            {prediction && (
              <div className="border-t border-border pt-8">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <svg
                    className="w-5 h-5 text-primary"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                    />
                  </svg>
                  Model Reasoning & Explainability
                </h3>
                <div className="bg-muted/50 rounded-lg p-6 space-y-6">
                  {/* Attention Map Section */}
                  <div>
                    <h4 className="font-medium mb-3 text-sm">Visual Attention Heatmap</h4>
                    <div className="grid sm:grid-cols-2 gap-4">
                      <div className="aspect-video rounded-lg border border-border flex items-center justify-center relative overflow-hidden">
                        <img
                          src={prediction.reasoning.attentionMap}
                          alt="Attention heatmap"
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <div className="flex flex-col justify-center space-y-2 text-sm text-muted-foreground">
                        <p className="leading-relaxed">
                          <span className="font-medium text-foreground">High attention areas</span>{' '}
                          (red/yellow) show where the model focuses most when making predictions.
                        </p>
                        <p className="leading-relaxed">
                          <span className="font-medium text-foreground">Low attention areas</span>{' '}
                          (blue/purple) indicate regions with less influence on the decision.
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Feature Importance */}
                  <div>
                    <h4 className="font-medium mb-3 text-sm">Top Contributing Features</h4>
                    <div className="space-y-3">
                      {prediction.reasoning.featureImportance.map((item, idx) => (
                        <div key={idx} className="flex items-center gap-3">
                          <span className="text-sm text-muted-foreground w-36 flex-shrink-0">
                            {item.feature}
                          </span>
                          <div className="flex-1 h-6 bg-muted rounded-full overflow-hidden">
                            <div
                              className={`h-full ${idx < 3 ? 'bg-primary' : idx < 4 ? 'bg-accent' : 'bg-chart-3'} flex items-center justify-end px-3 transition-all duration-500`}
                              style={{ width: `${item.confidence}%` }}
                            >
                              <span className="text-xs font-medium text-white">
                                {item.confidence}%
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Prediction Confidence */}
                  <div>
                    <h4 className="font-medium mb-3 text-sm">Class Prediction Confidence</h4>
                    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
                      {prediction.reasoning.classConfidence.slice(0, 5).map((cls) => {
                        const classColor = segmentationClasses.find(c => c.name === cls.class)?.color || '#000'
                        return (
                          <div
                            key={cls.class}
                            className="bg-background rounded-lg border border-border p-3 space-y-2"
                          >
                            <div className="flex items-center gap-2">
                              <div
                                className="w-3 h-3 rounded"
                                style={{ backgroundColor: classColor }}
                              />
                              <span className="text-xs font-medium truncate">{cls.class}</span>
                            </div>
                            <div className="text-xl font-bold text-foreground">
                              {cls.confidence.toFixed(1)}%
                            </div>
                            <div className="text-xs text-muted-foreground">confidence</div>
                          </div>
                        )
                      })}
                    </div>
                  </div>

                  {/* Explanation Text */}
                  <div className="bg-background rounded-lg border border-border p-4">
                    <h4 className="font-medium mb-2 text-sm flex items-center gap-2">
                      <svg
                        className="w-4 h-4 text-primary"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      </svg>
                      AI Explanation
                    </h4>
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      {prediction.reasoning.explanation}
                    </p>
                  </div>
                </div>
              </div>
            )}

                {/* Selected Model Info */}
                <div className="bg-muted/50 rounded-lg p-6 border border-border">
                  <h4 className="font-medium mb-2">
                    {getModelLabel(selectedModel)}
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    {selectedModel === 'segformer-10c-50e'
                      ? 'Transformer-based architecture with hierarchical encoder and lightweight MLP decoder. Best mIoU: 82.5%'
                      : selectedModel === 'segformer-b1'
                        ? 'SegFormer-B1: Transformer-based architecture (MiT-B1 backbone). Loaded from PUSHPENDAR/Sefformer-b1.'
                        : selectedModel === 'segformer-b10'
                          ? 'SegFormer-B10: Transformer-based architecture (MiT-B10 backbone). Loaded from PUSHPENDAR/Sefformer-b10.'
                          : 'CNN-based architecture with ResNet-101 backbone and ASPP module. Best mIoU: 80.8%'}
                  </p>
                </div>
              </TabsContent>
              
              <TabsContent value="explainer" className="mt-8">
                <SegFormerExplainer 
                  inputImage={uploadedImage || undefined}
                  segmentationResult={prediction}
                  modelType={selectedModel.startsWith('segformer') ? selectedModel as any : 'segformer'}
                />
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </section>
  )
}
