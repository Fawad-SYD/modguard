# modguard: LLM moderation agent vs. classic ML baselines

Detects spam in short messages three ways and compares them on the same held-out set
(UCI SMS Spam Collection, 300 test messages):

1. **TF-IDF + logistic regression** (scikit-learn) baseline
2. **DistilBERT fine-tuned** with PyTorch
3. **LLM agent** (Qwen2.5-1.5B-Instruct via Hugging Face Transformers): classify with a written
   policy, **verify** when confidence is low, and **escalate to human review** if still unsure.
   Malformed model output is retried; if every attempt fails the message goes to review.

Reported: accuracy, precision, recall, F1, mean and p95 latency, review rate, average agent
steps and token usage. See `results.md`.

## Run (Google Colab, GPU runtime)
```
!git clone <this repo> && cd modguard
!pip install -r requirements.txt
!python -m modguard.evaluate
```
Tests (no model needed): `pytest -q tests`

## Results
(paste results.md here after running)
