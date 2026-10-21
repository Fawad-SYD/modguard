"""Run all systems on the same test set and write results.md."""
import json
import sys

from .agent import ModerationAgent
from .metrics import latency_summary, scores


def eval_agent(llm, texts, labels, **kw):
    agent = ModerationAgent(llm, **kw)
    results = [agent.moderate(t) for t in texts]
    preds = [r.label for r in results]
    review = sum(r.needs_review for r in results) / len(results)
    avg_steps = sum(r.steps for r in results) / len(results)
    return preds, [r.latency_s for r in results], {"review_rate": review, "avg_steps": avg_steps}


def row(name, y_true, preds, lat, extra=None):
    s, l = scores(y_true, preds), latency_summary(lat)
    return {"system": name, **{k: round(v, 3) for k, v in s.items()},
            "mean_ms": round(l["mean_ms"], 1), "p95_ms": round(l["p95_ms"], 1), **(extra or {})}


def to_markdown(rows):
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def main(path="results.md"):
    from .baseline import run_baseline
    from .data import load_sms
    from .finetune import run_finetune
    from .llm_hf import HFChat

    train, test = load_sms()
    y = test[1]
    rows = []
    p, l = run_baseline(train, test)
    rows.append(row("TF-IDF + LogReg (scikit-learn)", y, p, l))
    p, l = run_finetune(train, test)
    rows.append(row("DistilBERT fine-tuned (PyTorch)", y, p, l))
    llm = HFChat()
    p, l, ex = eval_agent(llm, test[0], y)
    ex["llm_tokens"] = llm.prompt_tokens + llm.generated_tokens
    rows.append(row("LLM agent (Qwen2.5-1.5B, classify+verify)", y, p, l, ex))
    md = to_markdown(rows)
    open(path, "w").write(md + "\n")
    print(md)
    json.dump(rows, open("results.json", "w"), indent=2)


if __name__ == "__main__":
    main(*sys.argv[1:])
