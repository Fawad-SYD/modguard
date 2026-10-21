"""Evaluation metrics, written without dependencies so they are easy to test."""
from statistics import mean


def confusion(y_true, y_pred, positive="spam"):
    tp = sum(t == positive and p == positive for t, p in zip(y_true, y_pred))
    fp = sum(t != positive and p == positive for t, p in zip(y_true, y_pred))
    fn = sum(t == positive and p != positive for t, p in zip(y_true, y_pred))
    tn = sum(t != positive and p != positive for t, p in zip(y_true, y_pred))
    return tp, fp, fn, tn


def scores(y_true, y_pred, positive="spam"):
    tp, fp, fn, tn = confusion(y_true, y_pred, positive)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def latency_summary(latencies_s):
    if not latencies_s:
        return {"mean_ms": 0.0, "p95_ms": 0.0}
    s = sorted(latencies_s)
    p95 = s[min(len(s) - 1, int(0.95 * len(s)))]
    return {"mean_ms": mean(s) * 1000, "p95_ms": p95 * 1000}
