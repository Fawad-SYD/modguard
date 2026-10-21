"""Classic baseline: TF-IDF + logistic regression (scikit-learn)."""
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def run_baseline(train, test):
    (xtr, ytr), (xte, yte) = train, test
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=1000))
    model.fit(xtr, ytr)
    lat, preds = [], []
    for x in xte:
        t = time.perf_counter()
        preds.append(model.predict([x])[0])
        lat.append(time.perf_counter() - t)
    return preds, lat
