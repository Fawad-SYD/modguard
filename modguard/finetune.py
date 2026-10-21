"""Fine-tune DistilBERT with PyTorch as a supervised comparison point."""
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["ham", "spam"]


def run_finetune(train, test, name="distilbert-base-uncased", epochs=1, n_train=2000, batch_size=16, lr=3e-5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(device)
    xtr, ytr = train[0][:n_train], train[1][:n_train]
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    data = list(zip(xtr, [LABELS.index(y) for y in ytr]))
    model.train()
    for _ in range(epochs):
        loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        for texts, labels in loader:
            enc = tok(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            loss = model(**enc, labels=labels.to(device)).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
    model.eval()
    preds, lat = [], []
    with torch.no_grad():
        for x in test[0]:
            t = time.perf_counter()
            enc = tok(x, truncation=True, max_length=128, return_tensors="pt").to(device)
            preds.append(LABELS[model(**enc).logits.argmax(-1).item()])
            lat.append(time.perf_counter() - t)
    return preds, lat
