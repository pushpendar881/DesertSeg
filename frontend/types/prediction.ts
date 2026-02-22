export interface PredictionResponse {
  success: boolean
  segmentationMask: string
  reasoning: {
    explanation: string
    attentionMap: string
    featureImportance: Array<{
      feature: string
      confidence: number
    }>
    classConfidence: Array<{
      class: string
      confidence: number
    }>
  }
  metadata: {
    processingTime: number
    modelVersion: string
    imageSize: {
      width: number
      height: number
    }
  }
}

export interface ParallelPredictionResult {
  success: boolean
  model: string
  modelType: string
  segmentationMask?: string
  reasoning?: {
    explanation: string
    attentionMap: string
    featureImportance: Array<{
      feature: string
      confidence: number
    }>
    classConfidence: Array<{
      class: string
      confidence: number
    }>
  }
  metadata?: {
    processingTime: number
    inferenceTime: number
    modelVersion: string
    imageSize: {
      width: number
      height: number
    }
  }
  error?: string
}

export interface ComparativeAnalysis {
  summary: string
  classComparison: Array<{
    class: string
    averageConfidence: number
    maxConfidence: number
    minConfidence: number
    maxModel: string | null
    modelConfidences: Record<string, number>
  }>
  processingTimeComparison: {
    times: Record<string, number>
    fastest: string | null
    slowest: string | null
  }
  modelInsights: Array<{
    model: string
    topClass: string
    topConfidence: number
    processingTime: number
  }>
}

export interface ParallelPredictionResponse {
  success: boolean
  results: Record<string, ParallelPredictionResult>
  comparativeAnalysis: ComparativeAnalysis
  metadata: {
    totalProcessingTime: number
    imageSize: {
      width: number
      height: number
    }
    modelsRun: number
  }
}

export interface PredictionError {
  error: string
  details?: string
}
