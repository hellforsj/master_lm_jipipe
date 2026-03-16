from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
import joblib
import numpy as np
import pandas as pd

train_df = pd.read_csv("data/fine_tuning/node_search/train.csv",sep=";")
val_df = pd.read_csv("data/fine_tuning/node_search/val.csv", sep=";")

train_texts = train_df["text"].tolist()
train_labels = train_df["label"].tolist()

val_texts = val_df["text"].tolist()
val_labels = val_df["label"].tolist()

# =====================================
# 4. Load embedding model
# =====================================
name="all-mpnet-base-v2"
#name="all-MiniLM-L6-v2"
model = SentenceTransformer(name)

# =====================================
# 5. Encode text into embeddings
# =====================================
print("Encoding training data...")
train_embeddings = model.encode(train_texts, batch_size=64, show_progress_bar=True)

print("Encoding validation data...")
val_embeddings = model.encode(val_texts, batch_size=64, show_progress_bar=True)

# =====================================
# 6. Train classifier
# =====================================
clf = LinearSVC(
    max_iter=2000,
    C=1.0,
    multi_class='ovr'
)
clf.fit(train_embeddings, train_labels)

# =====================================
# 7. Evaluate on validation set
# =====================================
val_preds = clf.predict(val_embeddings)
val_probs = clf.predict_proba(val_embeddings)

acc = accuracy_score(val_labels, val_preds)
f1 = f1_score(val_labels, val_preds, average="macro")

print(f"\nValidation Accuracy: {acc:.4f}")
print(f"Validation Macro F1: {f1:.4f}")

for k in [1, 3, 5]:
    print(f"Top-{k} Accuracy: {top_k_accuracy_score(val_labels, val_probs, k=k):.4f}")

# =====================================
# 8. Save model for reuse
# =====================================
joblib.dump((model, clf), "data/models/text_classification/text_phrase_classifier_"+name+"_LinearSVC.joblib")