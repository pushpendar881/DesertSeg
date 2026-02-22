'use client'

import { useState } from 'react'
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    AreaChart,
    Area,
    BarChart,
    Bar,
    ResponsiveContainer,
    Tooltip,
    Legend,
} from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'

// =============================================
// Import trainer state JSON directly
// =============================================
import trainerState from '@/lib/segformer-7c-100e-trainer-state.json'

interface LogEntry {
    epoch: number
    step: number
    loss?: number
    grad_norm?: number | string
    learning_rate?: number
    eval_loss?: number
    eval_mean_iou?: number
    eval_mean_accuracy?: number
    eval_overall_accuracy?: number
    eval_runtime?: number
    eval_per_category_iou?: number[]
    eval_per_category_accuracy?: number[]
    [key: string]: any
}

const logHistory: LogEntry[] = (trainerState as any).log_history || []

const trainLogs = logHistory.filter((e) => e.loss !== undefined && e.eval_loss === undefined)
const evalLogs = logHistory.filter((e) => e.eval_mean_iou !== undefined)

// Downsample training logs for performance
function downsample<T>(arr: T[], maxPoints: number): T[] {
    if (arr.length <= maxPoints) return arr
    const step = arr.length / maxPoints
    const result: T[] = []
    for (let i = 0; i < maxPoints; i++) {
        result.push(arr[Math.floor(i * step)])
    }
    if (result[result.length - 1] !== arr[arr.length - 1]) {
        result.push(arr[arr.length - 1])
    }
    return result
}

const sampledTrainLogs = downsample(trainLogs, 120)

// ── TRAINING DATA ──
const trainLossData = sampledTrainLogs.map((e) => ({
    step: e.step,
    loss: +(e.loss!.toFixed(4)),
}))
const trainLRData = sampledTrainLogs.map((e) => ({
    step: e.step,
    lr: e.learning_rate!,
}))
const trainGradNormData = sampledTrainLogs
    .filter((e) => typeof e.grad_norm === 'number' && isFinite(e.grad_norm as number))
    .map((e) => ({
        step: e.step,
        gradNorm: +((e.grad_norm as number).toFixed(4)),
    }))

// ── EVAL DATA ──
const evalLossData = evalLogs.map((e) => ({
    epoch: Math.round(e.epoch),
    evalLoss: +(e.eval_loss!.toFixed(4)),
}))
const evalMeanIoUData = evalLogs.map((e) => ({
    epoch: Math.round(e.epoch),
    meanIoU: +((e.eval_mean_iou! * 100).toFixed(2)),
}))
const evalAccData = evalLogs.map((e) => ({
    epoch: Math.round(e.epoch),
    'Overall Accuracy': +((e.eval_overall_accuracy! * 100).toFixed(2)),
    'Mean Accuracy': +((e.eval_mean_accuracy! * 100).toFixed(2)),
}))

// ── PER-CLASS IoU ──
// Class IDs: 200, 300, 500, 550, 800, 7100
const CLASS_NAMES = [
    'Class 200', 'Class 300', 'Class 500', 'Class 550', 'Class 800', 'Class 7100',
]
const CLASS_COLORS: Record<string, string> = {
    'Class 200': '#22c55e',
    'Class 300': '#3b82f6',
    'Class 500': '#f59e0b',
    'Class 550': '#ef4444',
    'Class 800': '#8b5cf6',
    'Class 7100': '#06b6d4',
}

const perClassOverEpochs = evalLogs.map((e) => {
    const row: any = { epoch: Math.round(e.epoch) }
    if (e.eval_per_category_iou) {
        e.eval_per_category_iou.forEach((iou, i) => {
            if (CLASS_NAMES[i]) row[CLASS_NAMES[i]] = +((iou * 100).toFixed(1))
        })
    }
    return row
})

// Last eval for bar chart
const lastEval = evalLogs[evalLogs.length - 1]
const perCategoryBarData = lastEval?.eval_per_category_iou
    ? lastEval.eval_per_category_iou.map((iou, i) => ({
        name: CLASS_NAMES[i] || `Class ${i}`,
        'IoU (%)': +((iou * 100).toFixed(1)),
        'Accuracy (%)': lastEval.eval_per_category_accuracy
            ? +((lastEval.eval_per_category_accuracy[i] * 100).toFixed(1))
            : 0,
    }))
    : []

// Summary stats
const bestMeanIoU = evalLogs.reduce((best, e) => Math.max(best, (e.eval_mean_iou || 0) * 100), 0)
const bestOverallAcc = evalLogs.reduce((best, e) => Math.max(best, (e.eval_overall_accuracy || 0) * 100), 0)
const totalEpochs = (trainerState as any).num_train_epochs || 0
const totalSteps = (trainerState as any).global_step || 0
const bestStep = (trainerState as any).best_global_step || 0
const finalMeanIoU = lastEval ? +((lastEval.eval_mean_iou! * 100).toFixed(2)) : 0

// Step formatter
const stepFmt = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : `${v}`)

// Custom tooltip
function CustomTooltip({ active, payload, label }: any) {
    if (!active || !payload?.length) return null
    return (
        <div className="rounded-lg bg-zinc-800 border border-zinc-700 px-3 py-2 shadow-xl">
            <p className="text-[11px] text-gray-400 mb-1">{label}</p>
            {payload.map((p: any, i: number) => (
                <p key={i} className="text-xs font-medium" style={{ color: p.color || p.stroke }}>
                    {p.name}: <span className="font-bold">{typeof p.value === 'number' ? p.value.toLocaleString() : p.value}</span>
                </p>
            ))}
        </div>
    )
}

// =============================================
// Main Component
// =============================================
export function SegFormer7Classes100EpochsCharts() {
    const [activeTab, setActiveTab] = useState('training')

    if (logHistory.length === 0) {
        return (
            <div className="py-12 text-center">
                <h3 className="text-2xl font-bold text-white mb-2">SegFormer-7-Classes-100-Epochs Training Analytics</h3>
                <p className="text-yellow-500">Training data unavailable.</p>
            </div>
        )
    }

    return (
        <div className="mt-16">
            {/* Header */}
            <div className="text-center mb-8">
                <div className="inline-block rounded-full bg-emerald-500/15 border border-emerald-500/25 px-4 py-1.5 text-xs font-medium text-emerald-400 mb-4">
                    Real Training Data
                </div>
                <h3 className="text-2xl md:text-3xl font-bold text-white mb-2">
                    SegFormer-7-Classes-100-Epochs Training Analytics
                </h3>
                <p className="text-sm text-gray-400">
                    Detailed metrics from {totalEpochs} epochs ({totalSteps.toLocaleString()} steps) — 7-class desert segmentation
                </p>
            </div>

            {/* Summary stat cards */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-10">
                <StatCard label="Best mIoU" value={`${bestMeanIoU.toFixed(2)}%`} color="text-emerald-400" />
                <StatCard label="Final mIoU" value={`${finalMeanIoU}%`} color="text-blue-400" />
                <StatCard label="Best Accuracy" value={`${bestOverallAcc.toFixed(2)}%`} color="text-green-400" />
                <StatCard label="Total Steps" value={totalSteps.toLocaleString()} color="text-purple-400" />
                <StatCard label="Epochs" value={`${totalEpochs}`} color="text-amber-400" />
                <StatCard label="Best Step" value={bestStep.toLocaleString()} color="text-rose-400" />
            </div>

            {/* Tabs */}
            <Tabs value={activeTab} onValueChange={setActiveTab}>
                <TabsList className="grid w-full max-w-xl mx-auto grid-cols-3 mb-8">
                    <TabsTrigger value="training">Training</TabsTrigger>
                    <TabsTrigger value="evaluation">Evaluation</TabsTrigger>
                    <TabsTrigger value="per-class">Per-Class IoU</TabsTrigger>
                    {/* <TabsTrigger value="final">Final Results</TabsTrigger> */}
                </TabsList>

                {/* ========= TRAINING TAB ========= */}
                <TabsContent value="training">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Training Loss */}
                        <Card className="border-zinc-800 bg-zinc-900/60">
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm text-white">Training Loss</CardTitle>
                                <CardDescription className="text-xs">
                                    {trainLossData[0]?.loss} → {trainLossData[trainLossData.length - 1]?.loss}
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <ResponsiveContainer width="100%" height={260}>
                                    <AreaChart data={trainLossData}>
                                        <defs>
                                            <linearGradient id="loss7cGrad" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="0%" stopColor="#f43f5e" stopOpacity={0.3} />
                                                <stop offset="100%" stopColor="#f43f5e" stopOpacity={0.02} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                        <XAxis dataKey="step" tick={{ fontSize: 10, fill: '#a1a1aa' }} tickFormatter={stepFmt} stroke="#3f3f46" />
                                        <YAxis tick={{ fontSize: 10, fill: '#a1a1aa' }} stroke="#3f3f46" />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Area type="monotone" dataKey="loss" stroke="#f43f5e" strokeWidth={2} fill="url(#loss7cGrad)"
                                            animationDuration={2000} animationBegin={200} />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </CardContent>
                        </Card>

                        {/* Learning Rate */}
                        <Card className="border-zinc-800 bg-zinc-900/60">
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm text-white">Learning Rate Schedule</CardTitle>
                                <CardDescription className="text-xs">Warmup → linear decay</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <ResponsiveContainer width="100%" height={260}>
                                    <AreaChart data={trainLRData}>
                                        <defs>
                                            <linearGradient id="lr7cGrad" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="0%" stopColor="#10b981" stopOpacity={0.3} />
                                                <stop offset="100%" stopColor="#10b981" stopOpacity={0.02} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                        <XAxis dataKey="step" tick={{ fontSize: 10, fill: '#a1a1aa' }} tickFormatter={stepFmt} stroke="#3f3f46" />
                                        <YAxis tick={{ fontSize: 10, fill: '#a1a1aa' }} tickFormatter={(v: number) => v.toExponential(1)} stroke="#3f3f46" />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Area type="monotone" dataKey="lr" stroke="#10b981" strokeWidth={2} fill="url(#lr7cGrad)"
                                            animationDuration={2000} animationBegin={400} />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </CardContent>
                        </Card>

                        {/* Gradient Norm */}
                        <Card className="border-zinc-800 bg-zinc-900/60 lg:col-span-2">
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm text-white">Gradient Norm</CardTitle>
                                <CardDescription className="text-xs">Gradient magnitude over training (Infinity values filtered)</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <ResponsiveContainer width="100%" height={260}>
                                    <LineChart data={trainGradNormData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                        <XAxis dataKey="step" tick={{ fontSize: 10, fill: '#a1a1aa' }} tickFormatter={stepFmt} stroke="#3f3f46" />
                                        <YAxis tick={{ fontSize: 10, fill: '#a1a1aa' }} stroke="#3f3f46" />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Line type="monotone" dataKey="gradNorm" stroke="#a855f7" strokeWidth={2} dot={false}
                                            animationDuration={2500} animationBegin={200} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                {/* ========= EVALUATION TAB ========= */}
                <TabsContent value="evaluation">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Eval Loss */}
                        <Card className="border-zinc-800 bg-zinc-900/60">
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm text-white">Evaluation Loss</CardTitle>
                                <CardDescription className="text-xs">Validation loss per epoch</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <ResponsiveContainer width="100%" height={260}>
                                    <AreaChart data={evalLossData}>
                                        <defs>
                                            <linearGradient id="evalLoss7cGrad" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="0%" stopColor="#f43f5e" stopOpacity={0.3} />
                                                <stop offset="100%" stopColor="#f43f5e" stopOpacity={0.02} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                        <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: '#a1a1aa' }} stroke="#3f3f46" />
                                        <YAxis tick={{ fontSize: 10, fill: '#a1a1aa' }} stroke="#3f3f46" />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Area type="monotone" dataKey="evalLoss" stroke="#f43f5e" strokeWidth={2} fill="url(#evalLoss7cGrad)"
                                            animationDuration={2000} animationBegin={200} />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </CardContent>
                        </Card>

                        {/* Mean IoU */}
                        <Card className="border-zinc-800 bg-zinc-900/60">
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm text-white">Mean IoU</CardTitle>
                                <CardDescription className="text-xs">
                                    {evalMeanIoUData[0]?.meanIoU}% → {evalMeanIoUData[evalMeanIoUData.length - 1]?.meanIoU}%
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <ResponsiveContainer width="100%" height={260}>
                                    <AreaChart data={evalMeanIoUData}>
                                        <defs>
                                            <linearGradient id="miou7cGrad" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="0%" stopColor="#22c55e" stopOpacity={0.3} />
                                                <stop offset="100%" stopColor="#22c55e" stopOpacity={0.02} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                        <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: '#a1a1aa' }} stroke="#3f3f46" />
                                        <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#a1a1aa' }} stroke="#3f3f46" />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Area type="monotone" dataKey="meanIoU" stroke="#22c55e" strokeWidth={2} fill="url(#miou7cGrad)"
                                            name="Mean IoU (%)" animationDuration={2000} animationBegin={200} />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </CardContent>
                        </Card>

                        {/* Accuracy Curves */}
                        <Card className="border-zinc-800 bg-zinc-900/60 lg:col-span-2">
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm text-white">Accuracy Curves</CardTitle>
                                <CardDescription className="text-xs">Overall Accuracy vs Mean Accuracy over epochs</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <ResponsiveContainer width="100%" height={300}>
                                    <LineChart data={evalAccData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                        <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: '#a1a1aa' }} stroke="#3f3f46" />
                                        <YAxis domain={[20, 100]} tick={{ fontSize: 10, fill: '#a1a1aa' }} stroke="#3f3f46" />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Legend wrapperStyle={{ fontSize: 12 }} />
                                        <Line type="monotone" dataKey="Overall Accuracy" stroke="#f59e0b" strokeWidth={2.5} dot={false}
                                            animationDuration={2000} animationBegin={200} />
                                        <Line type="monotone" dataKey="Mean Accuracy" stroke="#3b82f6" strokeWidth={2.5} dot={false}
                                            animationDuration={2000} animationBegin={500} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                {/* ========= PER-CLASS IoU TAB ========= */}
                <TabsContent value="per-class">
                    <div className="space-y-6">
                        <Card className="border-zinc-800 bg-zinc-900/60">
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm text-white">Per-Class IoU Over Epochs</CardTitle>
                                <CardDescription className="text-xs">IoU progression for all 6 desert terrain classes (7-class model)</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <ResponsiveContainer width="100%" height={460}>
                                    <LineChart data={perClassOverEpochs}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                        <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: '#a1a1aa' }} stroke="#3f3f46" />
                                        <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#a1a1aa' }} stroke="#3f3f46" />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Legend wrapperStyle={{ fontSize: 11 }} />
                                        {CLASS_NAMES.map((name, i) => (
                                            <Line
                                                key={name} type="monotone" dataKey={name}
                                                stroke={CLASS_COLORS[name]} strokeWidth={2} dot={false}
                                                animationDuration={2000} animationBegin={i * 100}
                                            />
                                        ))}
                                    </LineChart>
                                </ResponsiveContainer>
                            </CardContent>
                        </Card>

                        {/* Per-class summary cards */}
                        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                            {perCategoryBarData.map((item) => {
                                const iou = item['IoU (%)']
                                const isStrong = iou > 60
                                const isWeak = iou < 40
                                return (
                                    <div key={item.name} className="flex items-center gap-3 p-3 rounded-xl bg-zinc-900 border border-zinc-800">
                                        <div className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: CLASS_COLORS[item.name] || '#666' }} />
                                        <div className="flex-1 min-w-0">
                                            <p className="text-xs font-medium text-white truncate">{item.name}</p>
                                            <p className="text-[10px] text-gray-500">Final IoU</p>
                                        </div>
                                        <p className={`text-sm font-bold ${isStrong ? 'text-green-400' : isWeak ? 'text-red-400' : 'text-amber-400'}`}>
                                            {iou}%
                                        </p>
                                    </div>
                                )
                            })}
                        </div>
                    </div>
                </TabsContent>

                {/* ========= FINAL RESULTS TAB — COMMENTED OUT =========
                <TabsContent value="final">
                    <div className="space-y-6">
                        <Card className="border-zinc-800 bg-zinc-900/60">
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm text-white">Final Epoch — Per-Class IoU & Accuracy</CardTitle>
                                <CardDescription className="text-xs">Epoch {Math.round(lastEval?.epoch || 0)} evaluation results</CardDescription>
                            </CardHeader>
                            <CardContent>
                                ...
                            </CardContent>
                        </Card>
                        <Card className="border-zinc-800 bg-zinc-900/60">
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm text-white">Detailed Results Table</CardTitle>
                                <CardDescription className="text-xs">Final evaluation metrics per class</CardDescription>
                            </CardHeader>
                            <CardContent>
                                ...
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>
                */}
            </Tabs>
        </div>
    )
}

// =============================================
// Stat Card
// =============================================
function StatCard({ label, value, color }: { label: string; value: string; color: string }) {
    return (
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-3">
            <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-0.5">{label}</p>
            <p className={`text-lg font-bold ${color}`}>{value}</p>
        </div>
    )
}
