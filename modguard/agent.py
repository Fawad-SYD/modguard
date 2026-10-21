"""A small moderation agent: classify -> verify when unsure -> escalate to human review.

The model is any callable `llm(prompt) -> str`, so the agent can be tested with a stub
and run with a real Hugging Face model (see llm_hf.py).
"""
import json
import re
import time
from dataclasses import dataclass, field

POLICY = (
    "Policy: label a message 'spam' if it is unsolicited promotion, a scam, a prize or "
    "payment lure, or mass-marketing. Otherwise label it 'ham' (normal message)."
)

CLASSIFY_PROMPT = (
    POLICY + "\nMessage: {text}\n"
    'Answer with JSON only: {{"label": "spam" or "ham", "confidence": 0-1, "reason": "short"}}'
)
VERIFY_PROMPT = (
    POLICY + "\nA first review said '{first}'. Re-check carefully and look for evidence "
    "against that label.\nMessage: {text}\n"
    'Answer with JSON only: {{"label": "spam" or "ham", "confidence": 0-1, "reason": "short"}}'
)


def parse_decision(raw):
    """Extract a decision dict from model output. Returns None if unusable."""
    m = re.search(r"\{.*?\}", raw, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    label = str(d.get("label", "")).strip().lower()
    if label not in ("spam", "ham"):
        return None
    try:
        conf = float(d.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return {"label": label, "confidence": max(0.0, min(1.0, conf)), "reason": str(d.get("reason", ""))}


@dataclass
class Result:
    label: str
    steps: int
    needs_review: bool
    latency_s: float
    trace: list = field(default_factory=list)


class ModerationAgent:
    def __init__(self, llm, confidence_threshold=0.7, max_steps=2):
        self.llm = llm
        self.threshold = confidence_threshold
        self.max_steps = max_steps

    def moderate(self, text):
        start = time.perf_counter()
        trace, decision = [], None
        for step in range(self.max_steps):
            if decision is None:
                prompt = CLASSIFY_PROMPT.format(text=text)
            else:
                prompt = VERIFY_PROMPT.format(text=text, first=decision["label"])
            new = parse_decision(self.llm(prompt))
            trace.append({"step": step + 1, "decision": new})
            if new is None:
                continue  # malformed output: retry within the step budget
            decision = new
            if decision["confidence"] >= self.threshold:
                break
        latency = time.perf_counter() - start
        if decision is None:
            return Result("ham", len(trace), True, latency, trace)  # fail open, send to a human
        needs_review = decision["confidence"] < self.threshold
        return Result(decision["label"], len(trace), needs_review, latency, trace)
