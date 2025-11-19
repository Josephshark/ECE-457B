# Colab-ready: improved sklearn pipeline (Option A)
# Copy/paste into a fresh Colab cell and run

# --- installs (Colab often already has these; harmless to run) ---

# --- imports ---
import re, string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from joblib import dump

# --- config ---
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_JOBS = -1           # set to -1 to use all CPUs (Colab has multiple cores)
VERBOSE = 2

# --- load dataset ---
print("Loading 20 Newsgroups dataset...")
d_train = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'))
d_test  = fetch_20newsgroups(subset='test',  remove=('headers','footers','quotes'))
print(f"Train size: {len(d_train.data)}, Test size: {len(d_test.data)}")

# --- preprocessing ---
translator = str.maketrans('', '', string.punctuation)
stop_words = set(stopwords.words('english'))

def preprocess(doc):
    # minimal cleaning: remove newlines, punctuation, lower, collapse spaces, remove digits & short tokens
    txt = doc.replace('\n', ' ').translate(translator).lower()
    txt = re.sub(r'\s+', ' ', txt).strip()
    # optionally remove tokens with digits and very short tokens
    tokens = [w for w in txt.split() if not re.search(r'\d', w) and len(w) > 2]
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

print("Preprocessing train texts (this can take a moment)...")
raw_texts = [preprocess(t) for t in d_train.data]
raw_texts_test = [preprocess(t) for t in d_test.data]
labels = list(d_train.target)
labels_test = list(d_test.target)

# --- train/validation split (stratified) ---
X_train, X_val, y_train, y_val = train_test_split(
    raw_texts, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE
)
print(f"Train split: {len(X_train)} examples, Val split: {len(X_val)} examples")

# --- vectorizers: word ngrams + char ngrams ---
word_tfidf = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1,2),   # unigrams + bigrams
    max_features=30000,
    min_df=3,
    max_df=0.9
)
char_tfidf = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3,5),   # char-level ngrams
    max_features=20000,
    min_df=5
)

vectorizer = FeatureUnion([
    ('word', word_tfidf),
    ('char', char_tfidf)
])

# --- pipeline: vectorizer + classifier ---
pipeline = Pipeline([
    ('vec', vectorizer),
    ('clf', LogisticRegression(
        solver='saga',        # works well with large sparse features and supports l1/l2
        penalty='l2',
        max_iter=2000,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ))
])

# --- grid to search (keeps size manageable on Colab) ---
param_grid = {
    'vec__word__ngram_range': [(1,1), (1,2)],
    'vec__word__max_features': [20000, 30000],
    'vec__char__ngram_range': [(3,5), (4,6)],
    'clf__C': [0.5, 1.0, 2.0]
}

print("Starting GridSearchCV (this may take several minutes)...")
grid = GridSearchCV(pipeline, param_grid=param_grid, cv=CV_FOLDS, n_jobs=N_JOBS, verbose=VERBOSE)
grid.fit(X_train, y_train)

print("\n=== Grid search results ===")
print("Best CV score:", grid.best_score_)
print("Best params:", grid.best_params_)

# --- evaluate on validation split ---
y_val_pred = grid.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"\nValidation accuracy: {val_acc:.4f}")
print(classification_report(y_val, y_val_pred, digits=4))

# --- retrain best estimator on train+val (for final test evaluation) ---
best = grid.best_estimator_
X_train_val = X_train + X_val
y_train_val = y_train + y_val
print("Retraining best estimator on train+val (this may take a bit)...")
best.fit(X_train_val, y_train_val)

# --- evaluate on official test set ---
y_test_pred = best.predict(raw_texts_test)
test_acc = accuracy_score(labels_test, y_test_pred)
print(f"\nOfficial test accuracy: {test_acc:.4f}")
print(classification_report(labels_test, y_test_pred, digits=4))

# --- confusion matrix (test) ---
cm = confusion_matrix(labels_test, y_test_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, cmap='Blues', annot=False)
plt.title('Confusion Matrix (test set)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# --- save the best model ---
dump(best, "best_tfidf_logreg_joblib.joblib")
print("Saved best model to best_tfidf_logreg_joblib.joblib")
