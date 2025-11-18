import sklearn.datasets as datasets


d_train = datasets.fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

d_test = datasets.fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
import string
import re
from stop_words import get_stop_words

stop_words = set(get_stop_words('en'))
translator = str.maketrans('', '', string.punctuation)

data = []

for i in range(len(d_train.data)):
    text = d_train.data[i].replace('\n', ' ')
    text = text.translate(translator)
    text = text.split()
    text = [word for word in text if not re.search(r'\d', word)]
    text = [word for word in text if word.lower() not in stop_words]
    text = ' '.join(text).lower()
    data.append([d_train.target[i], text])

# ...existing code...
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ensure we have a keyword_set in lowercase
try:
    keyword_set
except NameError:
    try:
        keyword_set = [k.lower() for k in keywords]
    except NameError:
        keyword_set = [k.lower() for k in keyword_set]

# Build X_test_raw exactly like X_raw (same keys/order)
X_test_raw = []
y_test = []
for i, doc in enumerate(d_test.data):
    text = doc.replace('\n', ' ')
    text = text.translate(translator)  # uses translator defined earlier
    words = text.split()
    words = [w for w in words if not re.search(r'\d', w)]
    words = [w.lower() for w in words if w.lower() not in stop_words]
    cnt = Counter(words)
    row = {kw: cnt.get(kw, 0) for kw in keyword_set}
    X_test_raw.append(row)
    y_test.append(d_test.target[i])

# Vectorize (use same vectorizer)
X_test = vectorizer.transform(X_test_raw)