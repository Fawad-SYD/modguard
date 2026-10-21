"""Load the UCI SMS Spam Collection (labels: 'spam' / 'ham')."""
import io
import random
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def load_sms(seed=0, test_size=300):
    raw = urllib.request.urlopen(URL).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        lines = z.read("SMSSpamCollection").decode("utf-8", "ignore").splitlines()
    rows = [l.split("\t", 1) for l in lines if "\t" in l]
    random.Random(seed).shuffle(rows)
    test, train = rows[:test_size], rows[test_size:]
    return ([t for _, t in train], [y for y, _ in train]), ([t for _, t in test], [y for y, _ in test])
