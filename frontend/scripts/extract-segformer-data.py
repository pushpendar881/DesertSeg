import json
import os

# Read the raw trainer state
with open('/vercel/share/v0-project/scripts/segformer-trainer-state.json', 'r') as f:
    raw = json.load(f)

log_history = raw['log_history']

# Separate training logs (have "loss") from eval logs (have "eval_mean_iou")
train_logs = [e for e in log_history if 'loss' in e]
eval_logs = [e for e in log_history if 'eval_mean_iou' in e]

# Extract training data
train_loss = [{"step": e["step"], "epoch": round(e["epoch"], 2), "value": e["loss"]} for e in train_logs]
train_lr = [{"step": e["step"], "epoch": round(e["epoch"], 2), "value": e["learning_rate"]} for e in train_logs]
train_grad = [{"step": e["step"], "epoch": round(e["epoch"], 2), "value": round(e["grad_norm"], 3)} for e in train_logs]

# Extract eval data
eval_overall_acc = [{"step": e["step"], "epoch": round(e["epoch"], 2), "value": round(e["eval_overall_accuracy"], 4)} for e in eval_logs]
eval_mean_iou = [{"step": e["step"], "epoch": round(e["epoch"], 2), "value": round(e["eval_mean_iou"], 4)} for e in eval_logs]
eval_mean_acc = [{"step": e["step"], "epoch": round(e["epoch"], 2), "value": round(e["eval_mean_accuracy"], 4)} for e in eval_logs]
eval_loss = [{"step": e["step"], "epoch": round(e["epoch"], 2), "value": round(e["eval_loss"], 4)} for e in eval_logs]
eval_steps_ps = [{"step": e["step"], "epoch": round(e["epoch"], 2), "value": e["eval_steps_per_second"]} for e in eval_logs]
eval_samples_ps = [{"step": e["step"], "epoch": round(e["epoch"], 2), "value": e["eval_samples_per_second"]} for e in eval_logs]
eval_runtime = [{"step": e["step"], "epoch": round(e["epoch"], 2), "value": round(e["eval_runtime"], 2)} for e in eval_logs]

# Per-class IoU from LAST eval entry
last_eval = eval_logs[-1]
class_names = ['Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky']
class_iou = [{"name": class_names[i] if i < len(class_names) else f"Class {i}", "iou": round(v * 100, 2)} for i, v in enumerate(last_eval["eval_per_category_iou"])]

# Best metrics
best_miou = max(e["eval_mean_iou"] for e in eval_logs)
best_acc = max(e["eval_overall_accuracy"] for e in eval_logs)
lowest_loss = min(e["loss"] for e in train_logs)

print("=== SegFormer Training State Summary ===")
print(f"Total training steps: {raw['global_step']}")
print(f"Total epochs: {raw['epoch']}")
print(f"Best mIoU: {best_miou * 100:.2f}%")
print(f"Best Overall Accuracy: {best_acc * 100:.2f}%")
print(f"Lowest Training Loss: {lowest_loss:.4f}")
print(f"Train log entries: {len(train_logs)}")
print(f"Eval log entries: {len(eval_logs)}")
print("\nFinal Per-Class IoU:")
for c in class_iou:
    print(f"  {c['name']}: {c['iou']}%")

# Down-sample training data (every 2nd entry)
def sample_every(arr, n):
    return [v for i, v in enumerate(arr) if i % n == 0 or i == len(arr) - 1]

output = {
    "metadata": {
        "totalSteps": raw["global_step"],
        "totalEpochs": raw["epoch"],
        "bestMeanIoU": round(best_miou, 4),
        "bestOverallAccuracy": round(best_acc, 4),
        "lowestTrainLoss": round(lowest_loss, 4),
        "bestCheckpoint": raw["best_model_checkpoint"],
        "trainBatchSize": raw["train_batch_size"],
    },
    "train": {
        "loss": sample_every(train_loss, 2),
        "learningRate": sample_every(train_lr, 2),
        "gradNorm": sample_every(train_grad, 2),
    },
    "eval": {
        "overallAccuracy": eval_overall_acc,
        "meanIoU": eval_mean_iou,
        "meanAccuracy": eval_mean_acc,
        "evalLoss": eval_loss,
        "stepsPerSecond": eval_steps_ps,
        "samplesPerSecond": eval_samples_ps,
        "runtime": eval_runtime,
    },
    "classIoU": class_iou,
}

os.makedirs('/vercel/share/v0-project/lib', exist_ok=True)
with open('/vercel/share/v0-project/lib/segformer-training-data.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\nData written to lib/segformer-training-data.json")
