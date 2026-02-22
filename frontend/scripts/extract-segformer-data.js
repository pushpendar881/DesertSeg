const { readFileSync, writeFileSync, mkdirSync } = require('fs');
const { join } = require('path');

const raw = JSON.parse(readFileSync('/vercel/share/v0-project/scripts/segformer-trainer-state.json', 'utf8'));

const logHistory = raw.log_history;

// Separate training logs (have "loss") from eval logs (have "eval_mean_iou")
const trainLogs = logHistory.filter(e => e.loss !== undefined);
const evalLogs = logHistory.filter(e => e.eval_mean_iou !== undefined);

// Extract training data
const trainData = {
  loss: trainLogs.map(e => ({ step: e.step, epoch: Math.round(e.epoch * 100) / 100, value: e.loss })),
  learningRate: trainLogs.map(e => ({ step: e.step, epoch: Math.round(e.epoch * 100) / 100, value: e.learning_rate })),
  gradNorm: trainLogs.map(e => ({ step: e.step, epoch: Math.round(e.epoch * 100) / 100, value: Math.round(e.grad_norm * 1000) / 1000 })),
};

// Extract eval data
const evalData = {
  overallAccuracy: evalLogs.map(e => ({ step: e.step, epoch: Math.round(e.epoch * 100) / 100, value: Math.round(e.eval_overall_accuracy * 10000) / 10000 })),
  meanIoU: evalLogs.map(e => ({ step: e.step, epoch: Math.round(e.epoch * 100) / 100, value: Math.round(e.eval_mean_iou * 10000) / 10000 })),
  meanAccuracy: evalLogs.map(e => ({ step: e.step, epoch: Math.round(e.epoch * 100) / 100, value: Math.round(e.eval_mean_accuracy * 10000) / 10000 })),
  evalLoss: evalLogs.map(e => ({ step: e.step, epoch: Math.round(e.epoch * 100) / 100, value: Math.round(e.eval_loss * 10000) / 10000 })),
  stepsPerSecond: evalLogs.map(e => ({ step: e.step, epoch: Math.round(e.epoch * 100) / 100, value: e.eval_steps_per_second })),
  samplesPerSecond: evalLogs.map(e => ({ step: e.step, epoch: Math.round(e.epoch * 100) / 100, value: e.eval_samples_per_second })),
  runtime: evalLogs.map(e => ({ step: e.step, epoch: Math.round(e.epoch * 100) / 100, value: Math.round(e.eval_runtime * 100) / 100 })),
};

// Per-class IoU from LAST eval entry
const lastEval = evalLogs[evalLogs.length - 1];
const classNames = ['Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'];
const classIoU = lastEval.eval_per_category_iou.map((v, i) => ({
  name: classNames[i] || `Class ${i}`,
  iou: Math.round(v * 10000) / 100,
}));

// Best metrics
const bestMeanIoU = Math.max(...evalLogs.map(e => e.eval_mean_iou));
const bestAccuracy = Math.max(...evalLogs.map(e => e.eval_overall_accuracy));
const lowestLoss = Math.min(...trainLogs.map(e => e.loss));

console.log("=== SegFormer Training State Summary ===");
console.log("Total training steps:", raw.global_step);
console.log("Total epochs:", raw.epoch);
console.log("Best mIoU:", (bestMeanIoU * 100).toFixed(2) + "%");
console.log("Best Overall Accuracy:", (bestAccuracy * 100).toFixed(2) + "%");
console.log("Lowest Training Loss:", lowestLoss.toFixed(4));
console.log("Train log entries:", trainLogs.length);
console.log("Eval log entries:", evalLogs.length);
console.log("\nFinal Per-Class IoU:");
classIoU.forEach(c => console.log("  " + c.name + ": " + c.iou + "%"));

// Down-sample training data for charts (every 2nd entry to keep manageable)
const sampleEvery = (arr, n) => arr.filter((_, i) => i % n === 0 || i === arr.length - 1);

const output = {
  metadata: {
    totalSteps: raw.global_step,
    totalEpochs: raw.epoch,
    bestMeanIoU: Math.round(bestMeanIoU * 10000) / 10000,
    bestOverallAccuracy: Math.round(bestAccuracy * 10000) / 10000,
    lowestTrainLoss: Math.round(lowestLoss * 10000) / 10000,
    bestCheckpoint: raw.best_model_checkpoint,
    trainBatchSize: raw.train_batch_size,
  },
  train: {
    loss: sampleEvery(trainData.loss, 2),
    learningRate: sampleEvery(trainData.learningRate, 2),
    gradNorm: sampleEvery(trainData.gradNorm, 2),
  },
  eval: evalData,
  classIoU,
};

mkdirSync('/vercel/share/v0-project/lib', { recursive: true });
writeFileSync('/vercel/share/v0-project/lib/segformer-training-data.json', JSON.stringify(output, null, 2));
console.log("\nData written to lib/segformer-training-data.json");
