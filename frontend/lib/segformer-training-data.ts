// Processed SegFormer training data from trainer_state JSON
// This file contains pre-processed, typed data arrays extracted from the raw training log.

import trainerState from '@/public/segformer-trainer-state.json'

// ── Class ID → Human-readable name mapping ──────────────────────────
const CLASS_NAMES: Record<string, string> = {
  '100': 'Trees',
  '200': 'Lush Bushes',
  '300': 'Dry Grass',
  '500': 'Dry Bushes',
  '550': 'Ground Clutter',
  '600': 'Flowers',
  '700': 'Logs',
  '800': 'Rocks',
  '7100': 'Landscape',
  '10000': 'Sky',
}

const CLASS_IDS = ['100', '200', '300', '500', '550', '600', '700', '800', '7100', '10000']

// ── Type definitions ────────────────────────────────────────────────
export interface TrainingEntry {
  step: number
  epoch: number
  loss: number
  gradNorm: number
  learningRate: number
}

export interface EvaluationEntry {
  step: number
  epoch: number
  evalLoss: number
  meanIoU: number
  meanAccuracy: number
  overallAccuracy: number
  perClassIoU: Record<string, number>
  perClassAccuracy: number[]
}

export interface PerClassIoURow {
  epoch: number
  Trees: number
  'Lush Bushes': number
  'Dry Grass': number
  'Dry Bushes': number
  'Ground Clutter': number
  Flowers: number
  Logs: number
  Rocks: number
  Landscape: number
  Sky: number
  [key: string]: number
}

export interface FinalClassMetric {
  name: string
  iou: number
  accuracy: number
}

// ── Data extraction ─────────────────────────────────────────────────
const logHistory = (trainerState as any).log_history as any[]

// Separate training steps from evaluation entries
export const trainingData: TrainingEntry[] = logHistory
  .filter((entry: any) => 'loss' in entry && !('eval_loss' in entry))
  .map((entry: any) => ({
    step: entry.step,
    epoch: Math.round(entry.epoch * 100) / 100,
    loss: Math.round(entry.loss * 10000) / 10000,
    gradNorm: Math.round(entry.grad_norm * 1000) / 1000,
    learningRate: entry.learning_rate,
  }))

export const evaluationData: EvaluationEntry[] = logHistory
  .filter((entry: any) => 'eval_loss' in entry)
  .map((entry: any) => ({
    step: entry.step,
    epoch: Math.round(entry.epoch),
    evalLoss: Math.round(entry.eval_loss * 10000) / 10000,
    meanIoU: Math.round(entry.eval_mean_iou * 10000) / 100,
    meanAccuracy: Math.round(entry.eval_mean_accuracy * 10000) / 100,
    overallAccuracy: Math.round(entry.eval_overall_accuracy * 10000) / 100,
    perClassIoU: Object.fromEntries(
      CLASS_IDS.map((id) => [
        CLASS_NAMES[id],
        Math.round((entry[`eval_iou_class_${id}`] ?? 0) * 10000) / 100,
      ])
    ),
    perClassAccuracy: (entry.eval_per_category_accuracy ?? []).map(
      (v: number) => Math.round(v * 10000) / 100
    ),
  }))

// Per-class IoU over epochs for multi-line chart
export const perClassIoUData: PerClassIoURow[] = evaluationData.map((entry) => ({
  epoch: entry.epoch,
  ...entry.perClassIoU,
} as PerClassIoURow))

// Eval metrics for line charts (epoch-level)
export const evalMetricsData = evaluationData.map((entry) => ({
  epoch: entry.epoch,
  'Eval Loss': entry.evalLoss,
  'Mean IoU': entry.meanIoU,
  'Mean Accuracy': entry.meanAccuracy,
  'Overall Accuracy': entry.overallAccuracy,
}))

// Final epoch per-class breakdown for bar chart
const lastEval = evaluationData[evaluationData.length - 1]
export const finalPerClassData: FinalClassMetric[] = CLASS_IDS.map((id, idx) => ({
  name: CLASS_NAMES[id],
  iou: lastEval.perClassIoU[CLASS_NAMES[id]],
  accuracy: lastEval.perClassAccuracy[idx],
}))

// Training loss sampled (every 5th point for cleaner visualization)
export const trainingLossSampled = trainingData.filter((_, i) => i % 3 === 0)

// Learning rate schedule
export const learningRateData = trainingData.map((entry) => ({
  step: entry.step,
  epoch: entry.epoch,
  learningRate: entry.learningRate * 1e6, // scale to μ for readability
}))

// Gradient norm data
export const gradNormData = trainingData.filter((_, i) => i % 3 === 0).map((entry) => ({
  step: entry.step,
  epoch: entry.epoch,
  gradNorm: entry.gradNorm,
}))

// ── Summary statistics ──────────────────────────────────────────────
const bestEvalByIoU = evaluationData.reduce((best, entry) =>
  entry.meanIoU > best.meanIoU ? entry : best
, evaluationData[0])

export const summaryStats = {
  bestMeanIoU: bestEvalByIoU.meanIoU,
  bestMeanIoUEpoch: bestEvalByIoU.epoch,
  bestOverallAccuracy: Math.max(...evaluationData.map((e) => e.overallAccuracy)),
  finalEvalLoss: lastEval.evalLoss,
  totalEpochs: (trainerState as any).num_train_epochs as number,
  totalSteps: (trainerState as any).global_step as number,
  bestGlobalStep: (trainerState as any).best_global_step as number,
  finalMeanIoU: lastEval.meanIoU,
  finalMeanAccuracy: lastEval.meanAccuracy,
  finalOverallAccuracy: lastEval.overallAccuracy,
}

// ── Class names export ──────────────────────────────────────────────
export const classNames = Object.values(CLASS_NAMES)

// ── Chart color palette ─────────────────────────────────────────────
export const CLASS_COLORS: Record<string, string> = {
  Trees: '#22c55e',
  'Lush Bushes': '#10b981',
  'Dry Grass': '#f59e0b',
  'Dry Bushes': '#ef4444',
  'Ground Clutter': '#8b5cf6',
  Flowers: '#ec4899',
  Logs: '#f97316',
  Rocks: '#6366f1',
  Landscape: '#06b6d4',
  Sky: '#3b82f6',
}
