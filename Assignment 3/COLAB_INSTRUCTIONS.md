# Running the Best Model on Google Colab

## Overview
This is the **highest performing model** from your Assignment 3:
- **Model**: Logistic Regression (One-vs-Rest)
- **Features**: 769 keywords
- **Test Accuracy**: 47.2%

## Steps to Run on Colab

### 1. Upload the keywords file
First, upload `keywords769.txt` to Colab:

```python
# Option A: Upload file directly
from google.colab import files
uploaded = files.upload()  # Select keywords769.txt when prompted
```

OR

```python
# Option B: Mount Google Drive (if you saved it there)
from google.colab import drive
drive.mount('/content/drive')
# Then update the file path in the code to point to your Drive location
```

### 2. Install required packages
Run this in a Colab cell:

```python
!pip install scikit-learn stop-words
```

### 3. Run the model
Copy the entire `colab_best_model.py` content into a Colab cell and run it.

OR just run:

```python
# After uploading colab_best_model.py to Colab
!python colab_best_model.py
```

## What the model does

1. **Loads data**: Fetches the 20 newsgroups dataset
2. **Preprocesses**: Removes punctuation, numbers, stopwords
3. **Extracts features**: Uses 769 keywords to build feature vectors
4. **Trains**: Logistic Regression with One-vs-Rest strategy
5. **Evaluates**: Reports accuracy, classification report, and confusion matrix

## Expected Output

```
Train accuracy: 0.7854
Test accuracy:  0.4716

Classification report and confusion matrix will follow...
```

## Model Performance Summary

This model correctly classifies about **47% of newsgroup posts** into 20 categories. 

Best performing categories:
- Category 10 (talk.politics.mideast): 69% recall
- Category 6 (misc.forsale): 66% recall
- Category 11 (talk.politics.misc): 57% recall

Challenging categories:
- Category 19 (talk.religion.misc): 15% recall
- Category 18 (talk.politics.guns): 21% recall
- Category 0 (alt.atheism): 31% recall
