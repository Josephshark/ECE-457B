# Best Performing Model for Google Colab
# Logistic Regression (One-vs-Rest) with 769 keywords
# Test Accuracy: 47.2%

# Install required packages (run this first in Colab)
# !pip install scikit-learn stop-words

import sklearn.datasets as datasets
import string
import re
from stop_words import get_stop_words
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading data...")
# Load 20 newsgroups dataset (train and test)
d_train = datasets.fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
d_test = datasets.fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

print("Preprocessing training data...")
# Preprocessing setup
stop_words = set(get_stop_words('en'))
translator = str.maketrans('', '', string.punctuation)

# Process training data
data = []
for i in range(len(d_train.data)):
    text = d_train.data[i].replace('\n', ' ')
    text = text.translate(translator)
    text = text.split()
    text = [word for word in text if not re.search(r'\d', word)]
    text = [word for word in text if word.lower() not in stop_words]
    text = ' '.join(text).lower()
    data.append([d_train.target[i], text])

# Convert to word counts
for i in range(len(data)):
    words = data[i][1].split()
    word_counts = Counter(words)
    data[i][1] = list(word_counts.items())

# IMPORTANT: Upload keywords769.txt to Colab first!
# In Colab, use the file upload feature or mount Google Drive
print("Loading keywords...")
# For Colab: You'll need to upload keywords769.txt or use:
# from google.colab import files
# uploaded = files.upload()  # Then select keywords769.txt

with open('keywords769.txt', 'r', encoding='utf-8') as f:
    keywords = [line.strip() for line in f if line.strip()]
keyword_set = set(keywords)

print(f"Using {len(keyword_set)} keywords")

# Extract features from training data
y = []
samples = []
for i in range(len(data)):
    y.append(data[i][0])
    samples.append(data[i][1])

# Build feature vectors
X_raw = []
for sample in samples:
    d = dict(sample)
    d_lower = {k.lower(): v for k, v in d.items()}
    row = {kw: d_lower.get(kw, 0) for kw in keyword_set}
    X_raw.append(row)

# Vectorize features
print("Vectorizing features...")
vectorizer = DictVectorizer(sparse=True)
X = vectorizer.fit_transform(X_raw)

# Train the model
print("Training Logistic Regression (One-vs-Rest)...")
model = LogisticRegression(C=1000, multi_class='ovr', solver='lbfgs', max_iter=1000)
model.fit(X, y)

# Process test data
print("Processing test data...")
X_test_raw = []
y_test = []
for i, doc in enumerate(d_test.data):
    text = doc.replace('\n', ' ')
    text = text.translate(translator)
    words = text.split()
    words = [w for w in words if not re.search(r'\d', w)]
    words = [w.lower() for w in words if w.lower() not in stop_words]
    cnt = Counter(words)
    row = {kw: cnt.get(kw, 0) for kw in keyword_set}
    X_test_raw.append(row)
    y_test.append(d_test.target[i])

X_test = vectorizer.transform(X_test_raw)

# Make predictions
print("Making predictions...")
y_pred_train = model.predict(X)
y_pred_test = model.predict(X_test)

# Print results
print("\n" + "="*60)
print("RESULTS - Best Performing Model")
print("="*60)
print(f"Train accuracy: {accuracy_score(y, y_pred_train):.4f}")
print(f"Test accuracy:  {accuracy_score(y_test, y_pred_test):.4f}")
print("\nClassification report (test):")
print(classification_report(y_test, y_pred_test))
print("\nConfusion matrix (test):")
print(confusion_matrix(y_test, y_pred_test))

# Category names for reference
print("\n" + "="*60)
print("Category names:")
for i, name in enumerate(d_test.target_names):
    print(f"{i}: {name}")
